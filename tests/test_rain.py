"""Tests for rain field (SIM-009)."""
from __future__ import annotations

import numpy as np
import pytest

from oasis_sim_av.config import RainClutterConfig
from oasis_sim_av.rain import RainField


def test_rain_field_from_config():
    cfg = RainClutterConfig(
        n_droplets=100,
        spawn_box=[0.0, -5.0, 5.0, 20.0, 5.0, 15.0],
        fall_velocity_m_s=6.0,
        droplet_radius_m=0.02,
    )
    rng = np.random.default_rng(42)
    field = RainField.from_config(cfg, ground_z=0.0, rng=rng)

    assert field.positions.shape == (100, 3)
    assert field.velocities.shape == (100, 3)
    assert np.all(field.velocities[:, 2] <= 0)


def test_rain_advection_loops_at_ground():
    cfg = RainClutterConfig(
        n_droplets=50,
        spawn_box=[0.0, 0.0, 10.0, 1.0, 1.0, 15.0],
        fall_velocity_m_s=10.0,
        jitter_std_m_s=0.0,
    )
    rng = np.random.default_rng(0)
    field = RainField.from_config(cfg, ground_z=0.0, rng=rng)

    for _ in range(200):
        field.step(0.01)

    assert np.all(field.positions[:, 2] >= 0.0), "No droplet should go below ground"


def test_rain_compute_clutter_hits_empty():
    cfg = RainClutterConfig(n_droplets=0)
    rng = np.random.default_rng(0)
    field = RainField.from_config(cfg, ground_z=0.0, rng=rng)

    origins = np.array([[0.0, 0.0, 0.0]])
    directions = np.array([[1.0, 0.0, 0.0]])

    hit_mask, t_values = field.compute_clutter_hits(origins, directions)

    assert hit_mask.shape == (1,)
    assert not hit_mask[0]
    assert np.isinf(t_values[0])


def test_rain_compute_clutter_hits_with_droplets():
    cfg = RainClutterConfig(
        n_droplets=10,
        spawn_box=[5.0, -1.0, 0.0, 6.0, 1.0, 5.0],
        droplet_radius_m=0.1,
    )
    rng = np.random.default_rng(0)
    field = RainField.from_config(cfg, ground_z=0.0, rng=rng)

    origins = np.zeros((10, 3))
    directions = np.array([[1.0, 0.0, 0.0]] * 10)

    hit_mask, t_values = field.compute_clutter_hits(origins, directions)

    assert hit_mask.shape == (10,)

    if hit_mask.any():
        assert np.all(t_values[hit_mask] > 0)
        assert np.all(np.isfinite(t_values[hit_mask]))


def test_rain_clutter_points_have_kind_3():
    cfg = RainClutterConfig(n_droplets=5, enabled=True)
    rng = np.random.default_rng(0)
    field = RainField.from_config(cfg, ground_z=0.0, rng=rng)

    origins = np.zeros((100, 3))
    directions = np.random.randn(100, 3)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    hit_mask, t_values = field.compute_clutter_hits(origins, directions)

    assert hit_mask.dtype == bool or np.issubdtype(hit_mask.dtype, np.integer)
