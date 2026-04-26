"""Optional visualisation helpers.

Matplotlib-based so we can ship zero-GPU-dependency previews.  Everything here
is import-on-demand — ``run.py`` can complete a full simulation without
matplotlib installed, as long as ``--no-viz`` is passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def quick_plot(run_dir: str, frame: int = 0) -> None:
    """Open one frame + its LiDAR scan side by side.

    Expects ``run_dir`` to contain ``frames/NNNNNN.png`` and
    ``lidar/NNNNNN.ply``.
    """
    import matplotlib.pyplot as plt  # lazy

    run = Path(run_dir)
    frames = sorted((run / "frames").glob("*.png"))
    lidars = sorted((run / "lidar").glob("*.ply"))
    if not frames and not lidars:
        raise SystemExit(f"No frames or lidar scans found in {run_dir}")
    idx = min(frame, len(frames) - 1)

    fig = plt.figure(figsize=(12, 5))
    if frames:
        ax1 = fig.add_subplot(1, 2, 1)
        img = _read_png(frames[idx])
        ax1.imshow(img)
        ax1.set_title(f"Camera frame {idx}")
        ax1.axis("off")
    if lidars:
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        pts, cols = _read_ply(lidars[min(frame, len(lidars) - 1)])
        if pts.size:
            # Flip colours from 0-255 uint8 to 0-1 float for matplotlib
            ax2.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=cols / 255.0,
                s=2,
                depthshade=False,
            )
        ax2.set_title(f"LiDAR scan {idx}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
    plt.tight_layout()
    plt.show()


def detection_plot(run_dir: str) -> None:
    """Plot tape-hit-count per frame from ``state.jsonl``.

    This is the key MVP output: shows how many of the tape triangles the
    LiDAR actually returns, vs what a perfect-visibility scan would return.
    Demonstrates the failure mode quantitatively.
    """
    import matplotlib.pyplot as plt  # lazy

    run = Path(run_dir)
    js = run / "state.jsonl"
    if not js.exists():
        raise SystemExit(f"No state.jsonl in {run_dir}")

    t, tape_hits, noisy_tape_hits = [], [], []
    with open(js) as f:
        for line in f:
            rec = json.loads(line)
            if "lidar" not in rec:
                continue
            t.append(rec["t"])
            tape_hits.append(rec["lidar"]["n_tape_hits_true"])
            noisy_tape_hits.append(rec["lidar"]["n_tape_hits_returned"])
    if not t:
        raise SystemExit("No LiDAR records in state.jsonl")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, tape_hits, label="Geometric tape hits (pre-noise)", lw=1.5)
    ax.plot(t, noisy_tape_hits, label="Tape returns after noise+dropout", lw=1.5)
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("tape LiDAR returns / scan")
    ax.set_title("Thin fluttering tape: LiDAR detection count vs. time")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# PNG / PLY readers (no external deps beyond numpy + imageio-if-present)
# ---------------------------------------------------------------------------
def _read_png(path: Path) -> np.ndarray:
    try:
        import imageio.v3 as iio  # lazy
        return iio.imread(str(path))
    except ImportError:
        from matplotlib.image import imread  # type: ignore
        arr = imread(str(path))
        if arr.dtype != np.uint8:
            arr = (arr * 255).astype(np.uint8)
        return arr


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse the ascii PLY written by ``lidar.py``."""
    with open(path) as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        body = f.read().split("\n")
    n = 0
    for h in header:
        if h.startswith("element vertex"):
            n = int(h.split()[-1])
            break
    pts = np.zeros((n, 3))
    cols = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        parts = body[i].split()
        if len(parts) < 6:
            continue
        pts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        cols[i] = [int(parts[3]), int(parts[4]), int(parts[5])]
    return pts, cols


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="oasis-sim-av visualisation")
    p.add_argument("run_dir", help="Path to a run directory")
    p.add_argument("--frame", type=int, default=0, help="Frame index (0-based)")
    p.add_argument(
        "--detection",
        action="store_true",
        help="Plot tape-return-count time series instead of a single frame",
    )
    args = p.parse_args()
    if args.detection:
        detection_plot(args.run_dir)
    else:
        quick_plot(args.run_dir, frame=args.frame)


if __name__ == "__main__":
    main()
