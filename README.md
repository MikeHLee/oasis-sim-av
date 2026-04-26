# oasis-sim-av

Lightweight Python simulator for autonomous-driving **sensor-fusion edge cases**.

Built to reproduce and study a specific failure mode: **thin, fluttering objects
(e.g. police/crime-scene tape) slipping through the gaps in LiDAR scans and
washing out in camera frames** due to motion blur and resolution limits, especially
in noisy environmental conditions (rain, dust).

## What it models

| Module            | Physics / math                                                                  |
|-------------------|---------------------------------------------------------------------------------|
| `world.py`        | Axis-aligned-bounding-box (AABB) city + ground plane + road polygons            |
| `vehicle.py`      | Kinematic bicycle model, inputs: forward velocity `v` and steering angle `δ`    |
| `cloth.py`        | 3D mass-spring grid for the tape, symplectic Euler integration, wind forcing    |
| `geometry.py`     | Vectorized ray-AABB slab test and Möller-Trumbore ray-triangle with ε guard     |
| `lidar.py`        | Spherical LiDAR sweep (FoV / angular resolution / range), Gaussian + dropout    |
| `camera.py`       | Pinhole eye-tracer, shaded RGB per pixel, temporal motion-blur by substepping   |
| `fusion.py`       | 1D complementary filter: weighted LiDAR + camera detections → fused posterior   |
| `render_video.py` | Stitches PNG frames into an annotated mp4 with per-frame tape-hit HUD overlay   |
| `noise.py`        | Gaussian / uniform / drift noise injectors (ported from oasis-firmware sim)     |
| `run.py`          | Fixed-dt orchestrator: controller → vehicle → cloth → LiDAR → camera → persist  |
| `viz.py`          | Optional matplotlib visualisation of point cloud + image                        |

## Install

```bash
pip install -e .[viz,dev]
```

## Run the baseline scenario

```bash
oasis-sim-av scenarios/police_tape_rain.yaml
```

Artifacts land in `runs/<timestamp>/`:

```
runs/20260426-142300/
├── frames/            000000.png  000001.png  ...
├── lidar/             000000.ply  000001.ply  ...
├── state.jsonl        per-step vehicle + cloth-energy + scan summary
└── config.yaml        resolved config, pinned for repro
```

Quick viz of the first frame:

```bash
python -m oasis_sim_av.viz runs/<timestamp> --frame 0
```

## Annotated video + fusion filter

After a run you can:

```bash
# Stitch PNG frames into an annotated mp4 with a per-frame tape-hit HUD:
oasis-sim-av-render-video runs/<timestamp>/ --fps 10
#   -> runs/<timestamp>/video.mp4  (or video.gif if ffmpeg is unavailable)

# Run the 1D complementary fusion filter (LiDAR + camera yellow detection):
oasis-sim-av-fuse runs/<timestamp>/
#   -> runs/<timestamp>/fusion.jsonl  + fusion.png
```

The baseline `police_tape_rain.yaml` is tuned so the fused posterior
`P(tape)` stays below the detection threshold across the full run even
though both raw sensors produce *some* signal — exactly the "each sensor
almost-sees it, the filter still can't" breakdown the brief targets.

## Tests

```bash
pytest -v
```

Unit tests cover: ray-AABB on axis / corner / grazing / parallel, Möller-Trumbore
correctness + ε-parallel guard, cloth rest-state + bounded energy, bicycle radius
`R = L / tan(δ)`, LiDAR σ statistics, and a 0.5 s smoke run.

## Design notes

- **Why vectorized numpy and not raw Python loops?** A 320×240 camera frame needs
  76,800 primary rays. A scalar Python ray tracer runs minutes per frame; the
  brief's "lightweight" requirement is understood as "no native extensions, no
  GPU" — numpy is still pure-pip and zero-compile. See `geometry.py` for
  fully-vectorized slab test and batched Möller-Trumbore.
- **Why symplectic Euler for cloth?** Brief requirement and it keeps long-running
  energy drift bounded under damping. See `cloth.py::MassSpringCloth.step`.
- **Why AABB-only buildings?** Brief requirement; keeps intersection math closed-
  form and cheap (6 comparisons per box per ray batch).
- **Why the specific LiDAR resolution in the default scenario?** Tuned so the tape
  width (5 cm) is narrower than the vertical ring spacing at 15+ m range, so the
  failure mode is physically inevitable and measurable. See
  `scenarios/police_tape_rain.yaml`.
- **Why `.ply` over `.las`?** Smaller footprint for ~10k-point scans; writable
  from stdlib. Add `laspy` later if you need ASPRS-compliant outputs.

## Repository context

This is a sibling module to `oasis-firmware` (firmware codegen + behavioural
signal-bus sim) and `swarm-city` (agent orchestration CLI). It deliberately
does **not** extend those — the domains are disjoint. See `./SIM_RATIONALE.md`
for the audit that motivated a from-scratch build.

See `.swarm/context.md` for agent-coordination notes and `.swarm/queue.md` for
open work items.
