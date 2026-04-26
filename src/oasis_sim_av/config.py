"""Scenario and sub-configuration dataclasses with YAML loader."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import yaml


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BuildingConfig:
    aabb: list[float]  # [x_min, y_min, z_min, x_max, y_max, z_max]


@dataclass
class TapeConfig:
    anchor_a: list[float] = field(default_factory=lambda: [-10.0, -5.0, 3.0])
    anchor_b: list[float] = field(default_factory=lambda: [10.0, -5.0, 3.0])
    length: float = 20.2   # slightly longer than distance -> hangs
    width: float = 0.05    # 5 cm
    n_length: int = 40     # particles along the length
    n_width: int = 2       # particles across the width
    mass_per_particle: float = 0.01
    spring_k: float = 500.0
    spring_damping: float = 0.8
    global_damping: float = 0.02
    wind_bias: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.2])
    wind_noise_std: float = 2.0


@dataclass
class WorldConfig:
    ground_z: float = 0.0
    buildings: list[BuildingConfig] = field(default_factory=list)
    roads: list[list[list[float]]] = field(default_factory=list)  # list of polygons (2D)
    tape: TapeConfig = field(default_factory=TapeConfig)


@dataclass
class VehicleControllerConfig:
    type: Literal[
        "constant", "step", "ramp", "sine", "impulse_steer", "bezier_pursuit"
    ] = "impulse_steer"
    base_v: float = 10.0
    base_delta: float = 0.0
    step_time: float = 0.0
    step_value: float = 0.0
    ramp_rate: float = 0.0
    sine_amp: float = 0.0
    sine_hz: float = 0.0
    impulse_time: float = 3.0
    impulse_delta: float = 0.2
    impulse_duration: float = 0.3
    # bezier_pursuit: tracks a cubic Bezier centerline via pure-pursuit
    bezier_control_points: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0], [20.0, 0.0], [40.0, 10.0], [60.0, 10.0]]
    )
    bezier_lookahead_m: float = 5.0
    bezier_max_delta_rad: float = 0.5


@dataclass
class VehicleConfig:
    wheelbase: float = 2.7
    initial_x: float = -30.0
    initial_y: float = 0.0
    initial_theta: float = 0.0
    controller: VehicleControllerConfig = field(default_factory=VehicleControllerConfig)


@dataclass
class LiDARConfig:
    offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.5])
    elevation_fov_deg: list[float] = field(default_factory=lambda: [-15.0, 15.0])
    elevation_rings: int = 32
    azimuth_fov_deg: list[float] = field(default_factory=lambda: [-60.0, 60.0])
    azimuth_rays: int = 512
    range_m: float = 80.0
    range_noise_std_m: float = 0.03
    rain_dropout_prob: float = 0.05


@dataclass
class CameraConfig:
    offset: list[float] = field(default_factory=lambda: [1.5, 0.0, 1.2])
    forward: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    up: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    fov_h_deg: float = 60.0
    width: int = 320
    height: int = 240
    motion_blur_samples: int = 4
    exposure_s: float = 0.01
    shadow_rays: bool = False   # SIM-004: cast secondary rays for hard shadows


@dataclass
class OutputConfig:
    dir: str = "runs"
    frame_every: int = 10   # simulate sensors every Nth sim step
    save_png: bool = True
    save_ply: bool = True
    save_jsonl: bool = True


@dataclass
class ScenarioConfig:
    seed: int = 42
    duration_s: float = 10.0
    dt: float = 0.01
    world: WorldConfig = field(default_factory=WorldConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    lidar: LiDARConfig = field(default_factory=LiDARConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -----------------------------------------------------------------------
    # IO
    # -----------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> ScenarioConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioConfig:
        w = data.get("world", {}) or {}
        tape = w.get("tape", {}) or {}
        buildings = [BuildingConfig(**b) for b in (w.get("buildings", []) or [])]
        world = WorldConfig(
            ground_z=w.get("ground_z", 0.0),
            buildings=buildings,
            roads=w.get("roads", []) or [],
            tape=TapeConfig(**tape) if tape else TapeConfig(),
        )
        v = data.get("vehicle", {}) or {}
        ctrl = v.get("controller", {}) or {}
        vehicle = VehicleConfig(
            wheelbase=v.get("wheelbase", 2.7),
            initial_x=v.get("initial_x", v.get("initial", {}).get("x", -30.0)),
            initial_y=v.get("initial_y", v.get("initial", {}).get("y", 0.0)),
            initial_theta=v.get(
                "initial_theta", v.get("initial", {}).get("theta", 0.0)
            ),
            controller=VehicleControllerConfig(**ctrl) if ctrl else VehicleControllerConfig(),
        )
        lidar = LiDARConfig(**(data.get("lidar", {}) or {}))
        cam = CameraConfig(**(data.get("camera", {}) or {}))
        out = OutputConfig(**(data.get("output", {}) or {}))
        return cls(
            seed=data.get("seed", 42),
            duration_s=data.get("duration_s", 10.0),
            dt=data.get("dt", 0.01),
            world=world,
            vehicle=vehicle,
            lidar=lidar,
            camera=cam,
            output=out,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
