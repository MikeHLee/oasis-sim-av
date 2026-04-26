"""Kinematic bicycle + controller tests."""
from __future__ import annotations

import numpy as np

from oasis_sim_av.config import VehicleConfig, VehicleControllerConfig
from oasis_sim_av.vehicle import KinematicBicycle, make_controller


def test_straight_line_constant_velocity():
    cfg = VehicleConfig(wheelbase=2.5, initial_x=0.0, initial_y=0.0, initial_theta=0.0)
    v = KinematicBicycle.from_config(cfg)
    dt = 0.01
    for _ in range(100):  # 1 second
        v.step(v=10.0, delta=0.0, dt=dt)
    # After 1s @ 10 m/s -> x = 10, y unchanged, theta unchanged
    assert np.isclose(v.state[0], 10.0, atol=1e-6)
    assert np.isclose(v.state[1], 0.0, atol=1e-6)
    assert np.isclose(v.state[2], 0.0, atol=1e-6)


def test_constant_steering_circle_radius():
    """R = L / tan(delta) is the steady-state turning radius.

    Drive a full quarter circle and check that the displacement is consistent
    with the expected radius to within a few percent (Euler integration bias).
    """
    L = 2.5
    delta = np.deg2rad(10.0)
    R_expected = L / np.tan(delta)
    cfg = VehicleConfig(
        wheelbase=L, initial_x=0.0, initial_y=0.0, initial_theta=0.0
    )
    veh = KinematicBicycle.from_config(cfg)
    dt = 0.001
    v = 5.0
    # Quarter turn: distance = pi/2 * R, time = distance / v
    T = (np.pi / 2.0) * R_expected / v
    n = int(T / dt)
    for _ in range(n):
        veh.step(v=v, delta=delta, dt=dt)
    # After a quarter turn starting at (0,0,0), position should be near (R, R, pi/2)
    assert np.isclose(veh.state[0], R_expected, rtol=0.02)
    assert np.isclose(veh.state[1], R_expected, rtol=0.02)
    assert np.isclose(veh.state[2], np.pi / 2.0, rtol=0.02)


def test_step_controller():
    ctrl = make_controller(
        VehicleControllerConfig(
            type="step", base_v=5.0, base_delta=0.0, step_time=1.0, step_value=0.1
        )
    )
    v0, d0 = ctrl(0.5, np.zeros(3))
    v1, d1 = ctrl(2.0, np.zeros(3))
    assert d0 == 0.0
    assert d1 == 0.1
    assert v0 == v1 == 5.0


def test_ramp_controller():
    ctrl = make_controller(
        VehicleControllerConfig(type="ramp", base_v=5.0, base_delta=0.0, ramp_rate=0.05)
    )
    _, d0 = ctrl(0.0, np.zeros(3))
    _, d1 = ctrl(10.0, np.zeros(3))
    assert np.isclose(d0, 0.0)
    assert np.isclose(d1, 0.5)


def test_sine_controller():
    ctrl = make_controller(
        VehicleControllerConfig(type="sine", base_v=5.0, base_delta=0.0, sine_amp=0.2, sine_hz=1.0)
    )
    # At t=0.25 s the sine should hit +amp
    _, d = ctrl(0.25, np.zeros(3))
    assert np.isclose(d, 0.2, atol=1e-6)


def test_impulse_steer_controller():
    ctrl = make_controller(
        VehicleControllerConfig(
            type="impulse_steer",
            base_v=5.0,
            base_delta=0.0,
            impulse_time=3.0,
            impulse_delta=0.3,
            impulse_duration=0.2,
        )
    )
    _, d_before = ctrl(2.9, np.zeros(3))
    _, d_in = ctrl(3.1, np.zeros(3))
    _, d_after = ctrl(3.3, np.zeros(3))
    assert d_before == 0.0
    assert d_in == 0.3
    assert d_after == 0.0


def test_sensor_origin_world():
    cfg = VehicleConfig(wheelbase=2.5, initial_x=10.0, initial_y=0.0, initial_theta=np.pi / 2)
    v = KinematicBicycle.from_config(cfg)
    # Sensor 1 m forward of the body origin; heading is +y so world offset should be +y
    origin = v.sensor_origin_world(np.array([1.0, 0.0, 1.5]))
    assert np.isclose(origin[0], 10.0, atol=1e-6)
    assert np.isclose(origin[1], 1.0, atol=1e-6)
    assert np.isclose(origin[2], 1.5, atol=1e-6)
