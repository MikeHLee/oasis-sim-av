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

    if kind == "bezier_pursuit":
        base_v = cfg.base_v
        pts = np.asarray(cfg.bezier_control_points, dtype=np.float64)
        if pts.shape[0] < 2 or pts.shape[1] != 2:
            raise ValueError(
                "bezier_pursuit needs >=2 2D control points in "
                "cfg.bezier_control_points"
            )
        Ld = max(0.1, float(cfg.bezier_lookahead_m))
        max_delta = float(cfg.bezier_max_delta_rad)
        # Pre-sample a dense polyline for fast nearest-projection + arc-length
        samples, arc_len = _bezier_sample(pts, n=400)

        def ctrl_bezier(t: float, state: np.ndarray) -> tuple[float, float]:
            x, y, theta = state
            # Find closest sample on the path, then walk ahead by lookahead
            # distance along cumulative arc length.
            dvec = samples - np.array([x, y])
            d2 = np.einsum("ij,ij->i", dvec, dvec)
            i0 = int(np.argmin(d2))
            target_s = arc_len[i0] + Ld
            # Clip to path end; this makes pursuit stop turning once we pass it
            target_s = min(target_s, arc_len[-1])
            # Binary search for the sample whose arc-length >= target_s
            i_target = int(np.searchsorted(arc_len, target_s))
            i_target = min(i_target, samples.shape[0] - 1)
            tgt = samples[i_target]
            # Pure-pursuit geometry: heading-to-target, relative to body
            dx = tgt[0] - x
            dy = tgt[1] - y
            alpha = np.arctan2(dy, dx) - theta
            # Wrap to (-pi, pi] so steering direction is well-defined
            alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
            # delta = atan(2 L sin(alpha) / Ld)  (requires wheelbase -> pass via state?)
            # We don't have L here; infer a reasonable L from state is not possible,
            # so use a unit-free approximation: delta ≈ alpha scaled and clipped.
            # This gives correct sign, tracks well at low speeds, and respects
            # bezier_max_delta_rad. For exact kinematics use the bicycle follower
            # at the orchestrator layer (SIM-003 v2).
            delta = float(np.clip(alpha, -max_delta, max_delta))
            return base_v, delta

        return ctrl_bezier

    raise ValueError(f"Unknown controller type: {kind!r}")


# ---------------------------------------------------------------------------
# Bezier helpers
# ---------------------------------------------------------------------------
def _bezier_sample(control_points: np.ndarray, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Sample a Bezier curve at ``n`` evenly-parameterised ``u`` values.

    Uses de Casteljau's algorithm — degree is ``len(control_points) - 1``, so
    this natively handles linear (2 pts), quadratic (3), cubic (4), and beyond.

    Returns
    -------
    samples : (n, 2) array of xy points along the curve.
    arc_len : (n,) cumulative arc-length along the polyline from samples[0].
    """
    P = np.asarray(control_points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("control_points must be (K, 2)")
    u = np.linspace(0.0, 1.0, n)
    samples = np.empty((n, 2), dtype=np.float64)
    for i, t in enumerate(u):
        Q = P.copy()
        # de Casteljau
        while Q.shape[0] > 1:
            Q = (1.0 - t) * Q[:-1] + t * Q[1:]
        samples[i] = Q[0]
    seg = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    return samples, arc


def bezier_point(control_points: np.ndarray, u: float) -> np.ndarray:
    """Evaluate the Bezier curve at parameter ``u ∈ [0, 1]``."""
    P = np.asarray(control_points, dtype=np.float64)
    t = float(np.clip(u, 0.0, 1.0))
    Q = P.copy()
    while Q.shape[0] > 1:
        Q = (1.0 - t) * Q[:-1] + t * Q[1:]
    return Q[0]


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
