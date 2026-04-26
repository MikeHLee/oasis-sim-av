"""Tests for SIM-003 (Bezier pursuit) and SIM-004 (shadow rays)."""
from __future__ import annotations

import numpy as np
import pytest

from oasis_sim_av.camera import _shadow_mask, PinholeCamera
from oasis_sim_av.config import CameraConfig, VehicleControllerConfig
from oasis_sim_av.vehicle import (
    KinematicBicycle,
    _bezier_sample,
    bezier_point,
    make_controller,
)
from oasis_sim_av.world import World
from oasis_sim_av.config import WorldConfig, BuildingConfig


# ---------------------------------------------------------------------------
# Bezier sampling
# ---------------------------------------------------------------------------
def test_bezier_endpoints_are_control_endpoints() -> None:
    P = np.array([[0.0, 0.0], [3.0, 2.0], [7.0, 0.0], [10.0, 5.0]])
    s, _ = _bezier_sample(P, n=50)
    np.testing.assert_allclose(s[0], P[0])
    np.testing.assert_allclose(s[-1], P[-1])
    # bezier_point(u=0) / (u=1)
    np.testing.assert_allclose(bezier_point(P, 0.0), P[0])
    np.testing.assert_allclose(bezier_point(P, 1.0), P[-1])


def test_bezier_linear_is_straight_line() -> None:
    P = np.array([[0.0, 0.0], [10.0, 0.0]])
    s, arc = _bezier_sample(P, n=20)
    # All samples should lie on y = 0 between x = 0 and x = 10
    assert np.allclose(s[:, 1], 0.0)
    assert np.all((s[:, 0] >= 0.0) & (s[:, 0] <= 10.0))
    # Arc length to end = 10
    assert arc[-1] == pytest.approx(10.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Bezier-pursuit controller behaviour
# ---------------------------------------------------------------------------
def test_bezier_pursuit_steers_toward_curve() -> None:
    # S-curve: forward then sharp right
    cfg = VehicleControllerConfig(
        type="bezier_pursuit",
        base_v=5.0,
        bezier_control_points=[[0.0, 0.0], [10.0, 0.0], [20.0, 10.0], [30.0, 10.0]],
        bezier_lookahead_m=4.0,
        bezier_max_delta_rad=0.6,
    )
    ctrl = make_controller(cfg)
    # At start the path already bends upward (cubic pulls y up early), so the
    # controller commands a small but nonzero positive delta.
    v, d = ctrl(0.0, np.array([0.0, 0.0, 0.0]))
    assert v == 5.0
    assert 0.0 < d < 0.25, f"expected modest positive delta at start, got {d}"

    # Mid-way (at x=15, y=0, heading=0) path is climbing up to y=10 -> should
    # command a strong positive steer (turn left).
    v, d = ctrl(0.0, np.array([15.0, 0.0, 0.0]))
    assert d > 0.2, f"expected positive steering to track up-right curve, got {d}"


def test_bezier_pursuit_runs_on_bicycle() -> None:
    """Closed-loop smoke: bicycle with Bezier controller tracks ~the path."""
    cfg = VehicleControllerConfig(
        type="bezier_pursuit",
        base_v=4.0,
        bezier_control_points=[[0.0, 0.0], [10.0, 0.0], [20.0, 5.0], [30.0, 5.0]],
        bezier_lookahead_m=3.0,
    )
    ctrl = make_controller(cfg)
    bike = KinematicBicycle(
        wheelbase=2.0, state=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    dt = 0.02
    for step in range(int(8.0 / dt)):
        v, d = ctrl(step * dt, bike.state)
        bike.step(v, d, dt)
    # After 8s at 4 m/s we should have covered ~32m total path length, so the
    # bicycle should be near the path end point (30, 5) within a few metres.
    dx = bike.state[0] - 30.0
    dy = bike.state[1] - 5.0
    err = (dx * dx + dy * dy) ** 0.5
    assert err < 6.0, f"bicycle strayed too far from target endpoint: err={err:.1f}"


def test_bezier_pursuit_respects_max_delta() -> None:
    cfg = VehicleControllerConfig(
        type="bezier_pursuit",
        base_v=1.0,
        bezier_control_points=[[0.0, 0.0], [1.0, 0.0], [1.0, 10.0]],
        bezier_lookahead_m=1.0,
        bezier_max_delta_rad=0.25,
    )
    ctrl = make_controller(cfg)
    # Vehicle pointed perpendicular to the path -> large alpha, should clip
    _, d = ctrl(0.0, np.array([0.0, 0.0, -np.pi / 2]))
    assert abs(d) <= 0.25 + 1e-9


# ---------------------------------------------------------------------------
# Shadow rays
# ---------------------------------------------------------------------------
def _tiny_world() -> World:
    # Tall thin pole at x in [-0.5, 0.5], y in [-5, 5].  Light direction
    # (-0.4, -0.3, 1)/norm shadows ground at (+x, +y) relative to the pole.
    cfg = WorldConfig(
        buildings=[BuildingConfig(aabb=[-0.5, -5.0, 0.0, 0.5, 5.0, 20.0])],
    )
    return World.from_config(cfg)


def test_shadow_mask_returns_bool_array() -> None:
    world = _tiny_world()
    # A ground point that *should* be in shadow: (3, 2, 0) — walking toward
    # light from there passes through the pole at around x=0.
    origin = np.array([[3.0, 2.0, 0.0]])
    # Camera "primary ray" doesn't matter for _shadow_mask; it only uses t
    # and kind.  Fake a ground hit by giving a downward ray.
    dir_ = np.array([[0.0, 0.0, -1.0]])
    # Shift the origin up so hit_point = (3, 2, 0) after t=1.
    origin[0, 2] = 1.0
    t = np.array([1.0])
    kind = np.array([0], dtype=np.int8)  # ground
    mask = _shadow_mask(
        origin, dir_, t, kind,
        world.boxes_min, world.boxes_max,
        np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)),
    )
    assert mask.shape == (1,)
    assert mask.dtype == bool
    assert mask[0], "(3, 2, 0) should be shadowed by the pole"


def test_shadow_mask_unshadowed_open_sky() -> None:
    world = _tiny_world()
    # Ground at (3, 2, 0) on the open side (away from pole along light dir):
    # walking toward light from (-3, -2, 0) we go further -x, -y — away
    # from the pole, so no occlusion.
    origin = np.array([[-3.0, -2.0, 1.0]])
    dir_ = np.array([[0.0, 0.0, -1.0]])
    t = np.array([1.0])
    kind = np.array([0], dtype=np.int8)
    mask = _shadow_mask(
        origin, dir_, t, kind,
        world.boxes_min, world.boxes_max,
        np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)),
    )
    assert not mask[0], "(-3, -2, 0) is on the lit side, must not be shadowed"


def test_shadow_mask_skips_sky_rays() -> None:
    # Ray that missed everything (kind == -1) should not be shadowed
    origins = np.array([[0.0, 0.0, 0.0]])
    dirs = np.array([[0.0, 0.0, 1.0]])
    t = np.array([np.inf])
    kind = np.array([-1], dtype=np.int8)
    mask = _shadow_mask(
        origins, dirs, t, kind,
        np.zeros((0, 3)), np.zeros((0, 3)),
        np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)),
    )
    assert mask.shape == (1,)
    assert not mask.any()


def test_camera_shadow_flag_changes_output() -> None:
    """Same scene, shadow_rays on vs off -> different image."""
    cfg_off = CameraConfig(
        offset=[0.0, 0.0, 0.0],
        forward=[0.0, 0.0, -1.0],       # look straight down
        up=[1.0, 0.0, 0.0],
        fov_h_deg=40.0,
        width=24, height=18,
        motion_blur_samples=1, shadow_rays=False,
    )
    cfg_on = CameraConfig(
        offset=[0.0, 0.0, 0.0],
        forward=[0.0, 0.0, -1.0],
        up=[1.0, 0.0, 0.0],
        fov_h_deg=40.0,
        width=24, height=18,
        motion_blur_samples=1, shadow_rays=True,
    )
    cam_off = PinholeCamera.from_config(cfg_off)
    cam_on = PinholeCamera.from_config(cfg_on)
    assert cam_off.shadow_rays is False
    assert cam_on.shadow_rays is True

    world = _tiny_world()
    tri0 = tri1 = tri2 = np.zeros((0, 3))
    # Overhead camera at (3, 2, 10) looking straight down: sees ground patch
    # around (3, 2) which is shadowed by the pole at x=0.
    origin = np.array([3.0, 2.0, 10.0])
    R = np.eye(3)
    img_off = cam_off.render(origin, R, world, tri0, tri1, tri2)
    img_on = cam_on.render(origin, R, world, tri0, tri1, tri2)
    assert img_off.shape == img_on.shape
    assert not np.array_equal(img_off, img_on), (
        "enabling shadow_rays must change at least some pixels"
    )
    # Shadowed pixels in `on` should be darker than the same pixels in `off`
    # (ambient-only vs ambient+Lambert).
    diff = img_off.astype(int) - img_on.astype(int)
    assert diff.sum() > 0, "shadow_on image should be overall darker"
