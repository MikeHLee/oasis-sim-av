"""Minimal 1D complementary filter that fuses LiDAR range + camera yellow-detection.

The MVP simulator produces two parallel sensor streams off the same scene:
LiDAR (`n_tape_hits_returned` per scan) and a rendered RGB image (PNGs).
A downstream detection stack would combine them; the purpose of this module
is to show — with the simplest possible fusion you can actually defend —
that the combined posterior *also* stays below a sensible detection
threshold for our baseline police-tape scenario.  Which is to say: the
failure is in the geometry, not in the filter.

Model
-----
For each sensor frame::

    p_lidar = clip(n_tape_returns / LIDAR_PEAK, 0, 1)
    p_camera = clip(n_yellow_pixels / CAM_PEAK, 0, 1)

Then a first-order low-pass over the weighted-sum measurement::

    z_t  = w_l * p_lidar + w_c * p_camera
    p_t  = (1 - alpha * dt) * p_{t-1}  +  alpha * dt * z_t

``dt`` is the inter-frame interval (sensor cadence, not the sim dt).  The
`w_l + w_c = 1` weighting is deliberately symmetric in the default config;
scenarios can tilt it to model a more-trusted sensor.

Writes ``<run_dir>/fusion.jsonl`` and, when matplotlib is available,
``<run_dir>/fusion.png``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Yellow detection heuristic (no classifier, just a colour box)
# ---------------------------------------------------------------------------
# Tape colour as rendered by camera._shade_float is world.tape_color()
# ~ (0.95, 0.85, 0.15) scaled by shade ∈ [AMBIENT, 1].  We pick a window that
# includes dimmer tape shade (shade ~ 0.22) → (210, 187, 33) worst case.
YELLOW_R_MIN = 120
YELLOW_G_MIN = 90
YELLOW_B_MAX = 110


def yellow_pixel_count(
    img: np.ndarray,
    crop_hfrac: tuple[float, float] = (0.3, 1.0),
    crop_wfrac: tuple[float, float] = (0.15, 0.85),
) -> int:
    """Count tape-coloured pixels in the central crop of ``img``.

    The crop is deliberately narrow-in-x wide-in-y: tape spans horizontally
    across the scene in our baseline and sits in the upper-middle of the
    view.  Adjust per-scenario via CLI if needed.
    """
    if img.ndim != 3 or img.shape[2] < 3:
        return 0
    H, W, _ = img.shape
    y0 = int(H * crop_hfrac[0])
    y1 = int(H * crop_hfrac[1])
    x0 = int(W * crop_wfrac[0])
    x1 = int(W * crop_wfrac[1])
    patch = img[y0:y1, x0:x1, :3]
    R = patch[..., 0].astype(np.int16)
    G = patch[..., 1].astype(np.int16)
    B = patch[..., 2].astype(np.int16)
    mask = (
        (R >= YELLOW_R_MIN)
        & (G >= YELLOW_G_MIN)
        & (B <= YELLOW_B_MAX)
        & (R >= G)          # tape has R >= G (yellow, not green)
        & ((R - B) >= 60)   # chromaticity, rules out sky/grey
    )
    return int(mask.sum())


# ---------------------------------------------------------------------------
# Complementary filter
# ---------------------------------------------------------------------------
@dataclass
class FusionConfig:
    """Tunable knobs for the 1D fusion."""

    lidar_peak: float = 6.0        # returns/scan at "definitely detected"
    camera_peak: float = 40.0      # yellow pixels at "definitely detected"
    w_lidar: float = 0.5
    w_camera: float = 0.5
    alpha: float = 6.0             # low-pass cutoff-ish, 1/s
    detect_threshold: float = 0.5


@dataclass
class FusionRecord:
    t: float
    p_lidar: float
    p_camera: float
    z: float
    p_fused: float
    detected: bool

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "p_lidar": self.p_lidar,
            "p_camera": self.p_camera,
            "z": self.z,
            "p_fused": self.p_fused,
            "detected": self.detected,
        }


class ComplementaryFilter:
    """First-order low-pass over a weighted-sum measurement."""

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.state: float = 0.0

    def update(self, p_lidar: float, p_camera: float, dt: float) -> float:
        c = self.cfg
        z = c.w_lidar * p_lidar + c.w_camera * p_camera
        z = max(0.0, min(1.0, z))
        # Zero-order hold smoothing.  At dt * alpha >= 1 the filter tracks
        # the measurement exactly (no lag) — common edge case at low fps.
        k = min(1.0, max(0.0, c.alpha * dt))
        self.state = (1.0 - k) * self.state + k * z
        return self.state


# ---------------------------------------------------------------------------
# Offline pipeline: run_dir -> fusion.jsonl / fusion.png
# ---------------------------------------------------------------------------
def _read_png(path: Path) -> np.ndarray | None:
    try:
        import imageio.v3 as iio  # type: ignore
        img = iio.imread(str(path))
    except ImportError:
        try:
            from matplotlib.image import imread  # type: ignore
            arr = imread(str(path))
            img = (arr * 255).astype(np.uint8) if arr.dtype != np.uint8 else arr
        except ImportError:
            return None
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[..., :3].astype(np.uint8)


def _frame_records(run_dir: Path) -> list[dict]:
    path = run_dir / "state.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — is this a run directory?")
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "frame_idx" in rec:
                rows.append(rec)
    rows.sort(key=lambda r: int(r["frame_idx"]))
    return rows


def run_fusion(
    run_dir: Path,
    cfg: FusionConfig | None = None,
    save_png: bool = True,
) -> tuple[list[FusionRecord], Path]:
    """Run the offline filter over all frames of ``run_dir``.

    Returns ``(records, jsonl_path)``.
    """
    run_dir = Path(run_dir)
    cfg = cfg or FusionConfig()
    frame_rows = _frame_records(run_dir)
    if not frame_rows:
        raise RuntimeError(f"No sensor frames in {run_dir}/state.jsonl")

    flt = ComplementaryFilter(cfg)

    records: list[FusionRecord] = []
    prev_t = frame_rows[0]["t"]
    frames_dir = run_dir / "frames"

    for rec in frame_rows:
        fi = int(rec["frame_idx"])
        t = float(rec["t"])
        dt = max(1e-6, t - prev_t) if records else 0.0
        prev_t = t

        n_ret = int(rec.get("lidar", {}).get("n_tape_hits_returned", 0))
        p_l = min(1.0, n_ret / max(1e-9, cfg.lidar_peak))

        img = _read_png(frames_dir / f"{fi:06d}.png")
        n_yellow = yellow_pixel_count(img) if img is not None else 0
        p_c = min(1.0, n_yellow / max(1e-9, cfg.camera_peak))

        p_f = flt.update(p_l, p_c, dt if records else 1.0 / max(1.0, cfg.alpha))
        z = cfg.w_lidar * p_l + cfg.w_camera * p_c
        records.append(
            FusionRecord(
                t=t, p_lidar=p_l, p_camera=p_c, z=z,
                p_fused=p_f, detected=p_f >= cfg.detect_threshold,
            )
        )

    jsonl_path = run_dir / "fusion.jsonl"
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    if save_png:
        _plot_fusion(records, run_dir / "fusion.png", cfg)

    return records, jsonl_path


def _plot_fusion(records: list[FusionRecord], out_path: Path, cfg: FusionConfig) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("[fusion] matplotlib not installed; skipping fusion.png", file=sys.stderr)
        return
    t = np.array([r.t for r in records])
    pl = np.array([r.p_lidar for r in records])
    pc = np.array([r.p_camera for r in records])
    pf = np.array([r.p_fused for r in records])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, pl, label="P(tape|lidar)", lw=1.3, alpha=0.75)
    ax.plot(t, pc, label="P(tape|camera)", lw=1.3, alpha=0.75)
    ax.plot(t, pf, label="P(tape|fused)", lw=2.0, color="black")
    ax.axhline(
        cfg.detect_threshold, color="red", lw=1.0, ls="--",
        label=f"detect threshold = {cfg.detect_threshold:.2f}",
    )
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("detection probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Sensor fusion (complementary filter): thin tape stays sub-threshold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def summary_stats(records: Iterable[FusionRecord]) -> dict:
    recs = list(records)
    if not recs:
        return {"n": 0, "max_p_fused": 0.0, "frac_detected": 0.0}
    pf = np.array([r.p_fused for r in recs])
    det = np.array([r.detected for r in recs])
    return {
        "n": len(recs),
        "max_p_fused": float(pf.max()),
        "mean_p_fused": float(pf.mean()),
        "frac_detected": float(det.mean()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Offline 1D complementary fusion of LiDAR + camera."
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--lidar-peak", type=float, default=FusionConfig.lidar_peak)
    p.add_argument("--camera-peak", type=float, default=FusionConfig.camera_peak)
    p.add_argument("--w-lidar", type=float, default=FusionConfig.w_lidar)
    p.add_argument("--w-camera", type=float, default=FusionConfig.w_camera)
    p.add_argument("--alpha", type=float, default=FusionConfig.alpha)
    p.add_argument("--threshold", type=float, default=FusionConfig.detect_threshold)
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args()

    cfg = FusionConfig(
        lidar_peak=args.lidar_peak,
        camera_peak=args.camera_peak,
        w_lidar=args.w_lidar,
        w_camera=args.w_camera,
        alpha=args.alpha,
        detect_threshold=args.threshold,
    )
    records, jsonl_path = run_fusion(args.run_dir, cfg, save_png=not args.no_png)
    stats = summary_stats(records)
    print(
        f"[fusion] {jsonl_path}\n"
        f"         n={stats['n']}  max_p_fused={stats['max_p_fused']:.3f}  "
        f"frac_detected={stats['frac_detected']:.3f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
