"""Stitch a run's PNG frames into an mp4 with a sidecar HUD overlay.

Usage
-----
    oasis-sim-av-render-video runs/20260426-150646/ [--out demo.mp4] [--fps 10]

The HUD shows, per frame:
    - simulation time (s)
    - true tape hits on this scan (pre noise/dropout)
    - returned tape hits (after noise/dropout)
    - a horizontal bar chart of returned-hit counts across time, so the
      ring-skip failure mode is visible at a glance.

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

import numpy as np


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
) -> Path:
    """Stitch PNGs in ``<run_dir>/frames`` with HUD overlay; return output path."""
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
    args = p.parse_args()
    render_video(args.run_dir, args.out, fps=args.fps, save_frames=args.save_frames)


if __name__ == "__main__":
    main()
