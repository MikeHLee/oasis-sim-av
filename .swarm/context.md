# oasis-sim-av context

## What this division is
Lightweight Python simulator for autonomous-driving **sensor-fusion edge cases**.
Reproduces LiDAR + camera failure on thin, fluttering objects (police tape)
in noisy environments.

## Why it exists
Pre-build audit of `oasis-firmware/simulation/` and `swarm-city/` concluded
that neither has any geometry, ray tracing, point-cloud, vehicle dynamics,
cloth, or sensor modelling. `oasis-firmware/simulation/` is a scalar-signal
IoT behavioural sim at 100 ms tick granularity (see
`oasis-firmware/simulation/behavioral/runtime.py:69-236`). `swarm-city/` is a
markdown-file agent orchestration CLI. This module is therefore built from
scratch as a clean sibling — the only code reused is the ~12-line Gaussian
noise injector pattern from `runtime.py:227-238`, now vectorised in
`src/oasis_sim_av/noise.py`.

## Scope
- 3D AABB city + ground + road polygons (`world.py`)
- Kinematic bicycle vehicle with step/ramp/sine/impulse controllers (`vehicle.py`)
- Mass-spring cloth with symplectic Euler + wind forcing (`cloth.py`)
- Vectorised ray-AABB + Möller-Trumbore ray-triangle with epsilon guard (`geometry.py`)
- Spherical LiDAR with configurable FoV/resolution + Gaussian + dropout (`lidar.py`)
- Eye-tracing pinhole camera with temporal motion-blur substepping (`camera.py`)
- YAML-driven deterministic scenarios, run artefacts in `runs/<stamp>/`

## Out of scope (for the MVP)
- Kalman / EKF / UKF filters — the demonstration produces the sensor streams
  that a downstream fusion stack would ingest, but no filter is bundled.
- GPU rendering
- Bicycle dynamic model (tyre forces) — kinematic is sufficient for the edge case
- Full CARLA / LGSVL-style scenario language

## Key cross-repo dependencies
None. This module has zero import-time dependencies on any other oasis-x repo.

## People / roles
Single-author module as of initial commit. PRs should be labelled `oasis-sim-av`.

## Recent session hits
- 2026-04-26 initial scaffold; ~2k LoC, passes unit tests, baseline scenario
  `scenarios/police_tape_rain.yaml` runs end-to-end.
