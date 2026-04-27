"""Main orchestrator: steps physics + sensors + persistence at fixed dt.

Usage
-----
    oasis-sim-av scenarios/police_tape_rain.yaml [--out runs] [--no-viz]

A "run" is a timestamped subdirectory containing:

    runs/<stamp>/
        config.yaml       # resolved config, pinned for repro
        state.jsonl       # per-step record (vehicle pose, cloth KE, scan summary)
        frames/NNNNNN.png # camera frames (at the sensor cadence)
        lidar/NNNNNN.ply  # LiDAR scans (at the sensor cadence)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

from .camera import PinholeCamera
from .cloth import MassSpringCloth
from .config import ScenarioConfig
from .lidar import SimulatedLiDAR
from .vehicle import KinematicBicycle, make_controller
from .world import World
from .bev import BEVRenderer
from .detect import OracleDetector, OracleDetectorConfig
from .rain import RainField


# ---------------------------------------------------------------------------
def run(cfg: ScenarioConfig, out_root: Path, log: bool = True) -> Path:
    """Run one simulation and return the run directory."""
    run_dir = out_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    (run_dir / "lidar").mkdir(parents=True, exist_ok=True)
    if cfg.bev is not None:
        (run_dir / "bev").mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")

    rng = np.random.default_rng(cfg.seed)

    world = World.from_config(cfg.world)
    cloth = MassSpringCloth.from_config(cfg.world.tape, rng)
    vehicle = KinematicBicycle.from_config(cfg.vehicle)
    controller = make_controller(cfg.vehicle.controller)
    lidar = SimulatedLiDAR.from_config(cfg.lidar, rng)
    camera = PinholeCamera.from_config(cfg.camera)
    bev_renderer = BEVRenderer.from_config(cfg.bev) if cfg.bev else None

    detector = OracleDetector(
        OracleDetectorConfig(),
        cfg.camera,
        rain_dropout_prob=cfg.lidar.rain_dropout_prob,
        rng=rng,
    )

    rain_field: RainField | None = None
    if cfg.rain_clutter is not None and cfg.rain_clutter.enabled:
        rain_field = RainField.from_config(
            cfg.rain_clutter, world.ground_z, rng
        )

    # Settle cloth under gravity for a moment so it sags naturally before t=0
    for _ in range(50):
        cloth.step(cfg.dt)

    n_steps = int(cfg.duration_s / cfg.dt)
    sensor_stride = max(1, int(cfg.output.frame_every))
    substeps = max(1, int(cfg.camera.motion_blur_samples))

    state_file = open(run_dir / "state.jsonl", "w") if cfg.output.save_jsonl else None
    t0 = time.time()
    frame_idx = 0

    try:
        for step_i in range(n_steps):
            t = step_i * cfg.dt

            # Controller -> vehicle
            v, delta = controller(t, vehicle.state)
            vehicle.step(v, delta, cfg.dt)

            # Cloth physics
            cloth.step(cfg.dt)

            # Rain field advection
            if rain_field is not None:
                rain_field.step(cfg.dt)

            # Sensors fire every `sensor_stride` sim steps
            fire_sensors = (step_i % sensor_stride == 0)

            record: dict = {
                "step": step_i,
                "t": t,
                "vehicle": {
                    "x": float(vehicle.state[0]),
                    "y": float(vehicle.state[1]),
                    "theta": float(vehicle.state[2]),
                    "v": float(vehicle.v),
                    "delta": float(vehicle.delta),
                },
                "cloth_ke": cloth.kinetic_energy(),
            }

            if fire_sensors:
                # Capture cloth triangle snapshots across the exposure window
                # for temporally blurred camera.  Substeps are tiny extra
                # physics ticks (dt_expose = exposure_s / substeps).
                cloth_snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
                if substeps > 1 and cfg.camera.exposure_s > 0.0:
                    # Advance cloth separately without mutating the main
                    # simulation state: work on a deep-copy of position/vel.
                    P = cloth.positions.copy()
                    V = cloth.velocities.copy()
                    dt_expose = cfg.camera.exposure_s / substeps
                    for _ in range(substeps):
                        _substep_cloth_in_place(cloth, dt_expose)
                        cloth_snapshots.append(cloth.triangles())
                    # Restore state so main-loop integration is unaffected
                    cloth.positions = P
                    cloth.velocities = V
                else:
                    cloth_snapshots.append(cloth.triangles())

                # Vehicle motion during exposure: assume constant v, delta,
                # so vehicle origin at frac f is what Euler gives at dt*f.
                def veh_origin(frac: float) -> np.ndarray:
                    x, y, th = vehicle.state
                    ddt = cfg.camera.exposure_s * frac
                    return np.array([
                        x + vehicle.v * np.cos(th) * ddt,
                        y + vehicle.v * np.sin(th) * ddt,
                        0.0,
                    ])

                def veh_R(frac: float) -> np.ndarray:
                    # Ignore yaw change over the short exposure (first-order)
                    return vehicle.body_to_world()

                # LiDAR: single snapshot of scene (at start of exposure)
                v0_base, v1_base, v2_base = cloth_snapshots[0]
                scan = lidar.scan(
                    vehicle.pose_xyz(),
                    vehicle.body_to_world(),
                    world.boxes_min, world.boxes_max,
                    v0_base, v1_base, v2_base,
                    ground_z=world.ground_z,
                )

                # Camera: average over cloth snapshots + vehicle substeps
                img = camera.render_with_motion_blur(
                    veh_origin, veh_R, world, cloth_snapshots
                )

                if cfg.output.save_ply:
                    lidar.write_ply(scan, str(run_dir / "lidar" / f"{frame_idx:06d}.ply"))
                if cfg.output.save_png:
                    _write_png(run_dir / "frames" / f"{frame_idx:06d}.png", img)

                if bev_renderer is not None and cfg.output.save_png:
                    v0_tri, v1_tri, v2_tri = cloth.triangles()
                    bev_img = bev_renderer.render(
                        world, v0_tri, v1_tri, v2_tri, vehicle.state
                    )
                    _write_png(run_dir / "bev" / f"{frame_idx:06d}.png", bev_img)

                detections = detector.detect_to_dict(
                    cloth, vehicle, vehicle.pose_xyz(), vehicle.body_to_world()
                )
                record["detections"] = detections

                record["lidar"] = {
                    "n_rays": scan.n_rays,
                    "n_returns": int(scan.points.shape[0]),
                    "n_tape_hits_true": scan.n_tape_hits,
                    "n_tape_hits_returned": int(np.sum(scan.kind == 2)),
                    "n_building_hits": int(np.sum(scan.kind == 1)),
                    "n_ground_hits": int(np.sum(scan.kind == 0)),
                }
                record["frame_idx"] = frame_idx
                frame_idx += 1

            if state_file is not None:
                state_file.write(json.dumps(record) + "\n")
                state_file.flush()

            if log and step_i % max(1, n_steps // 20) == 0:
                print(
                    f"[sim] step {step_i}/{n_steps} t={t:.2f}s "
                    f"veh=({vehicle.state[0]:.1f},{vehicle.state[1]:.1f},"
                    f"{np.rad2deg(vehicle.state[2]):+.1f}deg) "
                    f"ke={cloth.kinetic_energy():.3f}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if state_file is not None:
            state_file.close()

    if log:
        dt_real = time.time() - t0
        print(
            f"[sim] done. {n_steps} steps, {frame_idx} frames/scans in {dt_real:.1f}s "
            f"-> {run_dir}",
            file=sys.stderr,
        )
    return run_dir


# ---------------------------------------------------------------------------
def _substep_cloth_in_place(cloth: MassSpringCloth, dt: float) -> None:
    """Advance cloth one tiny substep; used inside the exposure window.

    The cloth step itself performs auto-substepping for stability, so we just
    ask for ``dt`` directly.
    """
    cloth.step(dt)


def _write_png(path: Path, img: np.ndarray) -> None:
    """Write a PNG.  Prefers imageio if available, falls back to matplotlib."""
    try:
        import imageio.v3 as iio  # lazy
        iio.imwrite(str(path), img)
        return
    except ImportError:
        pass
    try:
        from matplotlib.image import imsave  # type: ignore
        imsave(str(path), img)
        return
    except ImportError:
        pass
    # Last resort: raw .npy sidecar
    np.save(path.with_suffix(".npy"), img)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="oasis-sim-av: run a simulation scenario")
    p.add_argument("scenario", nargs="?", default=None, help="Path to a YAML scenario")
    p.add_argument("--out", default=None, help="Run output root (default: scenario.output.dir)")
    p.add_argument("--duration", type=float, default=None, help="Override duration_s")
    p.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    p.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    args = p.parse_args()

    if args.scenario:
        cfg = ScenarioConfig.from_yaml(args.scenario)
    else:
        cfg = ScenarioConfig()

    if args.duration is not None:
        cfg.duration_s = float(args.duration)
    if args.seed is not None:
        cfg.seed = int(args.seed)

    out_root = Path(args.out) if args.out else Path(cfg.output.dir)
    out_root.mkdir(parents=True, exist_ok=True)

    run(cfg, out_root, log=not args.quiet)


if __name__ == "__main__":
    main()
