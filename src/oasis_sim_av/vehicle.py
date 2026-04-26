"""Kinematic bicycle model and benchmark-input controllers.

State: ``(x, y, theta)`` — 2-D position and heading (rad).  Vertical z is
tracked as a constant equal to the wheelbase centre height for rendering.

Inputs: forward velocity ``v`` (m/s) and steering angle ``delta`` (rad).

Equations (bicycle, rear-axle reference):

    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = (v / L) * tan(delta)

Controllers implemented (brief requirement): ``step``, ``ramp``, ``sine``,
``impulse_steer``, plus ``constant`` as a default.  Any controller returns a
``(v, delta)`` tuple for a given sim time ``t``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .config import VehicleConfig, VehicleControllerConfig


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
class Controller(Protocol):
    def __call__(self, t: float, state: np.ndarray) -> tuple[float, float]: ...


def make_controller(cfg: VehicleControllerConfig) -> Controller:
    """Factory for the benchmark controllers."""
    kind = cfg.type
    if kind == "constant":
        base_v, base_delta = cfg.base_v, cfg.base_delta

        def ctrl_constant(t: float, state: np.ndarray) -> tuple[float, float]:
            return base_v, base_delta

        return ctrl_constant

    if kind == "step":
        base_v, base_delta = cfg.base_v, cfg.base_delta
        step_t, step_val = cfg.step_time, cfg.step_value

        def ctrl_step(t: float, state: np.ndarray) -> tuple[float, float]:
            delta = base_delta if t < step_t else base_delta + step_val
            return base_v, delta

        return ctrl_step

    if kind == "ramp":
        base_v, base_delta = cfg.base_v, cfg.base_delta
        rate = cfg.ramp_rate

        def ctrl_ramp(t: float, state: np.ndarray) -> tuple[float, float]:
            return base_v, base_delta + rate * t

        return ctrl_ramp

    if kind == "sine":
        base_v, base_delta = cfg.base_v, cfg.base_delta
        amp, hz = cfg.sine_amp, cfg.sine_hz

        def ctrl_sine(t: float, state: np.ndarray) -> tuple[float, float]:
            return base_v, base_delta + amp * np.sin(2.0 * np.pi * hz * t)

        return ctrl_sine

    if kind == "impulse_steer":
        base_v, base_delta = cfg.base_v, cfg.base_delta
        t0 = cfg.impulse_time
        dur = cfg.impulse_duration
        amp = cfg.impulse_delta

        def ctrl_impulse(t: float, state: np.ndarray) -> tuple[float, float]:
            in_window = (t >= t0) and (t < t0 + dur)
            delta = base_delta + (amp if in_window else 0.0)
            return base_v, delta

        return ctrl_impulse

    raise ValueError(f"Unknown controller type: {kind!r}")


# ---------------------------------------------------------------------------
# Kinematic bicycle
# ---------------------------------------------------------------------------
@dataclass
class KinematicBicycle:
    """Kinematic bicycle with explicit-Euler update.

    Parameters
    ----------
    wheelbase
        Distance between front and rear axles (metres).
    state
        ``(x, y, theta)`` as a length-3 numpy array.
    """

    wheelbase: float
    state: np.ndarray  # shape (3,) [x, y, theta]
    v: float = 0.0
    delta: float = 0.0

    @classmethod
    def from_config(cls, cfg: VehicleConfig) -> KinematicBicycle:
        return cls(
            wheelbase=cfg.wheelbase,
            state=np.array([cfg.initial_x, cfg.initial_y, cfg.initial_theta], dtype=np.float64),
        )

    def step(self, v: float, delta: float, dt: float) -> None:
        """Advance one timestep with commanded ``(v, delta)``."""
        self.v = v
        self.delta = delta
        x, y, theta = self.state
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += (v / self.wheelbase) * np.tan(delta) * dt
        # Wrap heading to (-pi, pi] for numerical stability
        theta = (theta + np.pi) % (2.0 * np.pi) - np.pi
        self.state = np.array([x, y, theta])

    # ------------------------------------------------------------------
    # Frame queries used by the sensor rigs
    # ------------------------------------------------------------------
    def pose_xyz(self, height: float = 0.0) -> np.ndarray:
        x, y, _ = self.state
        return np.array([x, y, height])

    def body_to_world(self) -> np.ndarray:
        """3x3 rotation matrix from vehicle body frame (x forward, y left, z up)
        to world frame.
        """
        _, _, theta = self.state
        c, s = np.cos(theta), np.sin(theta)
        return np.array(
            [
                [c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def sensor_origin_world(self, offset_body: np.ndarray) -> np.ndarray:
        """World-space position of a sensor mounted at ``offset_body``."""
        R = self.body_to_world()
        base = np.array([self.state[0], self.state[1], 0.0])
        return base + R @ np.asarray(offset_body, dtype=np.float64)
