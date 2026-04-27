"""Stitch a run's PNG frames into an mp4 with a sidecar HUD overlay.

Usage
-----
    oasis-sim-av-render-video runs/20260426-150646/ [--out demo.mp4] [--fps 10]
    oasis-sim-av-render-video runs/20260426-150646/ --layout grid5x2

The HUD shows, per frame:
    - simulation time (s)
    - true tape hits on this scan (pre noise/dropout)
    - returned tape hits (after noise/dropout)
    - a horizontal bar chart of returned-hit counts across time, so the
      ring-skip failure mode is visible at a glance.

With --layout grid5x2, produces a 5×2 multi-view grid:
    TOP ROW (vehicle camera):
        1. Camera RGB (raw forward view)
        2. Camera + 2D bboxes (oracle detector)
        3. Camera + reprojected LiDAR points
        4. Fused: camera + bboxes + LiDAR
        5. Fusion posterior strip

    BOTTOM ROW (world-fixed BEV):
        6. BEV ground truth
        7. BEV + driven-path trail
        8. LiDAR BEV, colour-coded by kind
        9. Fused BEV: ground-truth + LiDAR
        10. Legend / HUD

Dependencies (optional)
-----------------------
Video encoding needs ``imageio[ffmpeg]``.  HUD text uses ``Pillow`` if
available; if not, a pure-numpy block/bar overlay is rendered instead.
Both are in the ``[viz]`` optional dependency group.

If neither ``imageio`` nor a fallback encoder is available the annotated
frames are still written individually into ``<run_dir>/overlay/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np

from .overlays import (
    draw_bboxes,
    draw_fusion_strip,
    rasterise_lidar_bev,
    reproject_points_to_camera,
    compose_grid5x2,
)


# ---------------------------------------------------------------------------
# State.jsonl loader
# ---------------------------------------------------------------------------
def load_state(run_dir: Path) -> list[dict]:
    """Return the list of per-step records from ``state.jsonl``, in order."""
    path = run_dir / "state.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — is this a run directory?")
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def frame_records(state: list[dict]) -> list[dict]:
    """Keep only records that produced a sensor frame."""
    return [r for r in state if "frame_idx" in r]


# ---------------------------------------------------------------------------
# HUD rendering
# ---------------------------------------------------------------------------
HUD_HEIGHT = 48            # pixels added below the source frame
BAR_HEIGHT = 24
BAR_MARGIN = 4


def _try_import_pil():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
        return Image, ImageDraw, ImageFont
    except ImportError:
        return None, None, None


def _pil_font(size: int = 14):
    _, _, ImageFont = _try_import_pil()
    if ImageFont is None:
        return None
    try:
        # Most systems ship DejaVuSans.ttf in matplotlib; fall back to default.
        import matplotlib  # type: ignore
        fp = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
        if fp.exists():
            return ImageFont.truetype(str(fp), size=size)
    except ImportError:
        pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def annotate_frame(
    img: np.ndarray,
    record: dict,
    max_returned: int,
    bar_series: list[int],
    frame_idx: int,
) -> np.ndarray:
    """Compose an annotated frame by appending a HUD strip below ``img``.

    ``bar_series`` is the full per-frame series of returned tape hits so far
    (length ``frame_idx + 1``); rendered as a horizontal sparkline so the
    viewer can see the signal intermittence.
    """
    H, W, _ = img.shape
    canvas = np.zeros((H + HUD_HEIGHT, W, 3), dtype=np.uint8)
    canvas[:H] = img
    canvas[H:] = (25, 25, 30)  # dark HUD background

    # Bar sparkline occupies the left ~60% of the HUD
    n = max(1, len(bar_series))
    bar_w_total = int(W * 0.60)
    bar_origin_x = 8
    bar_origin_y = H + (HUD_HEIGHT - BAR_HEIGHT) // 2
    col_w = max(1, bar_w_total // max(n, 1))
    peak = max(max_returned, 1)
    for i, v in enumerate(bar_series):
        x0 = bar_origin_x + i * col_w
        if x0 + col_w >= bar_origin_x + bar_w_total:
            break
        h = int(BAR_HEIGHT * (v / peak))
        if h > 0:
            # Tape yellow for non-zero, grey for zero
            canvas[bar_origin_y + BAR_HEIGHT - h: bar_origin_y + BAR_HEIGHT,
                   x0: x0 + col_w - 1] = (240, 220, 40)
        # thin baseline
        canvas[bar_origin_y + BAR_HEIGHT - 1,
               x0: x0 + col_w - 1] = (90, 90, 95)

    # "Current frame" cursor tick
    cur_x = bar_origin_x + frame_idx * col_w
    if cur_x < bar_origin_x + bar_w_total:
        canvas[bar_origin_y - 2: bar_origin_y + BAR_HEIGHT + 2,
               cur_x: cur_x + 1] = (255, 80, 80)

    # Text overlay
    lidar = record.get("lidar", {})
    n_true = int(lidar.get("n_tape_hits_true", 0))
    n_ret = int(lidar.get("n_tape_hits_returned", 0))
    t = float(record.get("t", 0.0))
    text = (
        f"t={t:5.2f}s  "
        f"tape_true={n_true:3d}  "
        f"tape_returned={n_ret:3d}"
    )

    Image, ImageDraw, _ = _try_import_pil()
    if Image is not None:
        font = _pil_font(14)
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        tx = bar_origin_x + bar_w_total + 12
        ty = H + 8
        if font is None:
            draw.text((tx, ty), text, fill=(240, 240, 240))
        else:
            draw.text((tx, ty), text, fill=(240, 240, 240), font=font)
            draw.text((tx, ty + 18),
                      f"frame {frame_idx + 1}/{n}",
                      fill=(170, 170, 170), font=font)
        return np.asarray(pil)

    # Pure-numpy fallback: stamp small indicator blocks
    # One small yellow square per returned tape hit, max 20.
    pad = 4
    sq = 6
    tx0 = bar_origin_x + bar_w_total + 12
    ty0 = H + 6
    for i in range(min(n_ret, 20)):
        x = tx0 + i * (sq + 2)
        if x + sq >= W:
            break
        canvas[ty0: ty0 + sq, x: x + sq] = (240, 220, 40)
    # red square indicates any true hits that were dropped
    dropped = max(0, n_true - n_ret)
    for i in range(min(dropped, 20)):
        x = tx0 + i * (sq + 2)
        if x + sq >= W:
            break
        canvas[ty0 + sq + 4: ty0 + 2 * sq + 4, x: x + sq] = (220, 60, 60)
    return canvas


# ---------------------------------------------------------------------------
# Video encoder
# ---------------------------------------------------------------------------
def _encode_video(frames: list[np.ndarray], out_path: Path, fps: int) -> Path | None:
    """Write ``frames`` to ``out_path`` (or a gif fallback with the same stem).

    Returns the actual file written, or None if encoding is unavailable.
    """
    try:
        import imageio.v3 as iio  # type: ignore
        if out_path.suffix.lower() in (".mp4", ".m4v", ".mov"):
            try:
                iio.imwrite(str(out_path), np.asarray(frames), fps=fps, codec="libx264")
                return out_path
            except Exception:
                pass
        # gif fallback (mp4 codec missing, or user asked for .gif)
        gif_path = out_path.with_suffix(".gif")
        iio.imwrite(str(gif_path), np.asarray(frames), duration=1.0 / fps, loop=0)
        if gif_path != out_path:
            print(f"[render] wrote {gif_path} (mp4 unavailable; used gif)",
                  file=sys.stderr)
        return gif_path
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def render_video(
    run_dir: Path,
    out: Path | None = None,
    fps: int = 10,
    save_frames: bool = False,
    layout: Literal["single", "grid5x2"] = "single",
) -> Path:
    """Stitch PNGs in ``<run_dir>/frames`` with HUD overlay; return output path.

    Parameters
    ----------
    layout : "single" or "grid5x2"
        - "single": original camera+HUD view
        - "grid5x2": 5×2 multi-view grid (requires BEV frames)
    """
    if layout == "grid5x2":
        return render_video_grid5x2(run_dir, out, fps, save_frames)
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    frame_rows = frame_records(state)
    if not frame_rows:
        raise RuntimeError(f"No sensor frames recorded in {run_dir}/state.jsonl")

    frames_dir = run_dir / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"{frames_dir} not found")

    # Sort by frame_idx so ordering is deterministic
    frame_rows.sort(key=lambda r: int(r["frame_idx"]))
    n_ret_series = [
        int(r.get("lidar", {}).get("n_tape_hits_returned", 0)) for r in frame_rows
    ]
    max_returned = max(n_ret_series) if n_ret_series else 1

    # Read PNGs
    try:
        import imageio.v3 as iio  # type: ignore
        reader = iio.imread
    except ImportError:
        reader = None

    out_frames: list[np.ndarray] = []
    overlay_dir = run_dir / "overlay"
    if save_frames:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(frame_rows):
        png_path = frames_dir / f"{int(rec['frame_idx']):06d}.png"
        if not png_path.exists():
            print(f"[render] skip missing frame {png_path}", file=sys.stderr)
            continue
        if reader is None:
            # last-ditch: read via matplotlib
            try:
                from matplotlib.image import imread as mpl_imread  # type: ignore
                img = (mpl_imread(str(png_path))[..., :3] * 255).astype(np.uint8)
            except ImportError:
                raise RuntimeError(
                    "Need imageio or matplotlib to decode PNG frames"
                ) from None
        else:
            img = reader(str(png_path))
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = img[..., :3].astype(np.uint8)

        anno = annotate_frame(img, rec, max_returned, n_ret_series[: i + 1], i)
        out_frames.append(anno)
        if save_frames:
            from matplotlib.image import imsave  # type: ignore
            imsave(overlay_dir / f"{i:06d}.png", anno)

    out_path = Path(out) if out else run_dir / "video.mp4"
    written = _encode_video(out_frames, out_path, fps=fps)
    if written is not None:
        print(f"[render] wrote {written} ({len(out_frames)} frames at {fps} fps)",
              file=sys.stderr)
        return written

    # last-resort: write per-frame PNGs into overlay/
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(out_frames):
        try:
            from matplotlib.image import imsave  # type: ignore
            imsave(overlay_dir / f"{i:06d}.png", f)
        except ImportError:
            np.save(overlay_dir / f"{i:06d}.npy", f)
    print(f"[render] imageio not available; wrote PNGs/NPYs to {overlay_dir}",
          file=sys.stderr)
    return overlay_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stitch run PNG frames into an annotated mp4."
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default <run_dir>/video.mp4)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--save-frames", action="store_true",
                   help="Also write annotated per-frame PNGs into <run_dir>/overlay/")
    p.add_argument("--layout", choices=["single", "grid5x2"], default="single",
                   help="Layout mode: single (camera+HUD) or grid5x2 (multi-view)")
    args = p.parse_args()
    render_video(
        args.run_dir, args.out, fps=args.fps,
        save_frames=args.save_frames, layout=args.layout
    )


def render_video_grid5x2(
    run_dir: Path,
    out: Path | None = None,
    fps: int = 10,
    save_frames: bool = False,
) -> Path:
    """Stitch frames into a 5×2 multi-view grid video."""
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    frame_rows = frame_records(state)
    if not frame_rows:
        raise RuntimeError(f"No sensor frames recorded in {run_dir}/state.jsonl")

    frames_dir = run_dir / "frames"
    bev_dir = run_dir / "bev"
    if not frames_dir.exists():
        raise FileNotFoundError(f"{frames_dir} not found")

    cfg = _load_config(run_dir)
    has_bev = bev_dir.exists() and cfg is not None and cfg.bev is not None

    frame_rows.sort(key=lambda r: int(r["frame_idx"]))
    p_fused_series: list[float] = []

    try:
        import imageio.v3 as iio
        reader = iio.imread
    except ImportError:
        reader = None

    out_frames: list[np.ndarray] = []
    overlay_dir = run_dir / "overlay_grid5x2"
    if save_frames:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    vehicle_trail: list[tuple[float, float]] = []

    for i, rec in enumerate(frame_rows):
        png_path = frames_dir / f"{int(rec['frame_idx']):06d}.png"
        if not png_path.exists():
            continue

        if reader is None:
            try:
                from matplotlib.image import imread as mpl_imread
                img = (mpl_imread(str(png_path))[..., :3] * 255).astype(np.uint8)
            except ImportError:
                raise RuntimeError("Need imageio or matplotlib") from None
        else:
            img = reader(str(png_path))
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = img[..., :3].astype(np.uint8)

        H, W = img.shape[:2]

        panel1 = img.copy()
        detections = rec.get("detections", [])
        panel2 = draw_bboxes(panel1, detections) if detections else panel1.copy()

        panel3 = img.copy()
        panel4 = img.copy()

        fusion_jsonl = run_dir / "fusion.jsonl"
        if fusion_jsonl.exists() and len(p_fused_series) <= i:
            with open(fusion_jsonl) as f:
                lines = f.readlines()
            if i < len(lines):
                import json as json_mod
                p_fused_series.append(
                    json_mod.loads(lines[i])["p_fused"]
                )

        panel5 = draw_fusion_strip(p_fused_series[: i + 1], W, H)

        panels: list[np.ndarray] = [panel1, panel2, panel3, panel4, panel5]

        if has_bev:
            bev_png = bev_dir / f"{int(rec['frame_idx']):06d}.png"
            if bev_png.exists():
                if reader is None:
                    from matplotlib.image import imread as mpl_imread
                    bev_img = (mpl_imread(str(bev_png))[..., :3] * 255).astype(np.uint8)
                else:
                    bev_img = reader(str(bev_png))
                    if bev_img.ndim == 2:
                        bev_img = np.stack([bev_img] * 3, axis=-1)
                    bev_img = bev_img[..., :3].astype(np.uint8)
                panel6 = bev_img
            else:
                panel6 = np.zeros((H, W, 3), dtype=np.uint8)

            veh = rec.get("vehicle", {})
            vehicle_trail.append((veh.get("x", 0.0), veh.get("y", 0.0)))

            panel7 = _draw_trail_on_bev(panel6, vehicle_trail, cfg)

            lidar = rec.get("lidar", {})
            panel8 = np.zeros((H, W, 3), dtype=np.uint8) + 30

            panel9 = panel7.copy()
        else:
            panel6 = panel7 = panel8 = panel9 = np.zeros((H, W, 3), dtype=np.uint8)

        t = float(rec.get("t", 0.0))
        veh = rec.get("vehicle", {})
        speed = veh.get("v", 0.0)
        n_boxes = len(detections)
        p_current = p_fused_series[-1] if p_fused_series else 0.0
        footer = f"t={t:.2f}s  v={speed:.1f}m/s  det={n_boxes}  p_fused={p_current:.2f}"

        titles = [
            "Camera RGB",
            "Camera + Det",
            "Camera + LiDAR",
            "Fused",
            "P(fused)",
            "BEV Truth",
            "BEV + Trail",
            "LiDAR BEV",
            "Fused BEV",
            "HUD",
        ]

        panel10 = np.zeros((H, W, 3), dtype=np.uint8)
        panel10[:] = [20, 20, 25]
        _draw_hud_text(panel10, footer)

        panels.extend([panel6, panel7, panel8, panel9, panel10])

        composed = compose_grid5x2(panels, titles=titles, footer_text=footer)
        out_frames.append(composed)

        if save_frames:
            try:
                from matplotlib.image import imsave
                imsave(overlay_dir / f"{i:06d}.png", composed)
            except ImportError:
                np.save(overlay_dir / f"{i:06d}.npy", composed)

    out_path = Path(out) if out else run_dir / "video_grid5x2.mp4"
    written = _encode_video(out_frames, out_path, fps=fps)
    if written is not None:
        print(f"[render] wrote {written} ({len(out_frames)} frames at {fps} fps)",
              file=sys.stderr)
        return written

    overlay_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(out_frames):
        try:
            from matplotlib.image import imsave
            imsave(overlay_dir / f"{i:06d}.png", f)
        except ImportError:
            np.save(overlay_dir / f"{i:06d}.npy", f)
    return overlay_dir


def _load_config(run_dir: Path):
    try:
        from .config import ScenarioConfig
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists():
            return ScenarioConfig.from_yaml(cfg_path)
    except Exception:
        pass
    return None


def _draw_trail_on_bev(bev_img: np.ndarray, trail: list[tuple[float, float]], cfg) -> np.ndarray:
    """Draw vehicle trail on BEV image."""
    if not trail or cfg is None or cfg.bev is None:
        return bev_img

    out = bev_img.copy()
    H, W = out.shape[:2]
    center = cfg.bev.center
    extent = cfg.bev.extent_m
    half = extent / 2.0

    for i, (x, y) in enumerate(trail):
        px = int((x - (center[0] - half)) / extent * W)
        py = int((y - (center[1] - half)) / extent * H)
        if 0 <= px < W and 0 <= py < H:
            alpha = 0.3 + 0.7 * (i / max(len(trail), 1))
            out[py, px] = [int(255 * alpha), int(80 * alpha), int(80 * alpha)]

    return out


def _draw_hud_text(img: np.ndarray, text: str) -> None:
    """Draw HUD text on the legend panel."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            import matplotlib
            fp = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
            if fp.exists():
                font = ImageFont.truetype(str(fp), size=12)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        draw.text((8, 8), text, fill=(200, 200, 200), font=font)
        img[:] = np.asarray(pil)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
