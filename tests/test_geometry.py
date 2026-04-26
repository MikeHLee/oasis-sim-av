"""Tests for ray-AABB + ray-triangle intersection (the sensor-fusion core)."""
from __future__ import annotations

import numpy as np
import pytest

from oasis_sim_av.geometry import (
    EPS,
    ray_aabb_batch,
    ray_aabb_many,
    ray_ground,
    ray_triangle_batch,
    nearest_hit,
)


# ---------------------------------------------------------------------------
# ray_aabb_batch
# ---------------------------------------------------------------------------
def test_aabb_axis_hit():
    origins = np.array([[0.0, 0.0, 0.0]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    t = ray_aabb_batch(origins, dirs, np.array([5, -1, -1]), np.array([6, 1, 1]))
    np.testing.assert_allclose(t, [5.0], atol=1e-6)


def test_aabb_miss_above():
    origins = np.array([[0.0, 0.0, 10.0]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    t = ray_aabb_batch(origins, dirs, np.array([5, -1, -1]), np.array([6, 1, 1]))
    assert np.isinf(t[0])


def test_aabb_behind_origin():
    # Ray starts past the box and points away
    origins = np.array([[10.0, 0.0, 0.0]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    t = ray_aabb_batch(origins, dirs, np.array([5, -1, -1]), np.array([6, 1, 1]))
    assert np.isinf(t[0])


def test_aabb_grazing_corner():
    """Ray passing through a corner should report a hit (or be infinitesimally far)."""
    origins = np.array([[0.0, 0.0, 0.0]])
    # Aim at the (1,1,1) corner of the box
    dirs = np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3.0)
    t = ray_aabb_batch(origins, dirs, np.array([1, 1, 1]), np.array([2, 2, 2]))
    assert np.isfinite(t[0])


def test_aabb_vectorised_many_rays():
    # 100 rays fanning out, half should hit the target box
    rng = np.random.default_rng(0)
    origins = np.zeros((100, 3))
    az = np.linspace(-np.pi, np.pi, 100)
    dirs = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=-1)
    t = ray_aabb_batch(origins, dirs, np.array([-1, -1, -0.5]), np.array([1, 1, 0.5]))
    # Rays pointing directly along axes should all hit (origin is inside the box)
    # Origin is inside box so every ray hits.
    assert np.all(np.isfinite(t))


def test_aabb_many_nearest_selection():
    origins = np.array([[0.0, 0.0, 0.0]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    boxes_min = np.array([[10, -1, -1], [3, -1, -1]])
    boxes_max = np.array([[11, 1, 1],  [4, 1, 1]])
    t, idx = ray_aabb_many(origins, dirs, boxes_min, boxes_max)
    assert idx[0] == 1  # the nearer box at x=3
    np.testing.assert_allclose(t[0], 3.0, atol=1e-6)


# ---------------------------------------------------------------------------
# ray_triangle_batch (Moller-Trumbore with epsilon guard)
# ---------------------------------------------------------------------------
def test_triangle_direct_hit():
    origins = np.array([[0.0, 0.0, 0.0]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    v0 = np.array([[5.0, -1.0, -1.0]])
    v1 = np.array([[5.0,  1.0, -1.0]])
    v2 = np.array([[5.0,  0.0,  1.0]])
    t, idx = ray_triangle_batch(origins, dirs, v0, v1, v2)
    np.testing.assert_allclose(t, [5.0], atol=1e-6)
    assert idx[0] == 0


def test_triangle_miss_outside():
    origins = np.array([[0.0, 0.0, 0.0]])
    dirs = np.array([[1.0, 2.0, 0.0]]) / np.sqrt(5.0)
    v0 = np.array([[5.0, -1.0, -1.0]])
    v1 = np.array([[5.0,  1.0, -1.0]])
    v2 = np.array([[5.0,  0.0,  1.0]])
    t, idx = ray_triangle_batch(origins, dirs, v0, v1, v2)
    assert np.isinf(t[0])
    assert idx[0] == -1


def test_triangle_parallel_ray_epsilon_guard():
    """Ray perfectly parallel to the triangle plane must be rejected.

    Brief: "ensure an epsilon distance check to handle rays that are perfectly
    parallel to the cloth planes."
    """
    origins = np.array([[0.0, 0.0, 0.0]])
    # Triangle lies in z = 5 plane (normal is +z)
    v0 = np.array([[-1.0, -1.0, 5.0]])
    v1 = np.array([[ 1.0, -1.0, 5.0]])
    v2 = np.array([[ 0.0,  1.0, 5.0]])
    # Ray perfectly parallel (direction in xy plane only)
    dirs = np.array([[1.0, 0.0, 0.0]])
    t, idx = ray_triangle_batch(origins, dirs, v0, v1, v2)
    assert np.isinf(t[0])
    assert idx[0] == -1


def test_triangle_vectorized_nearest():
    origins = np.array([[0.0, 0.0, 0.0]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    v0 = np.array([[10.0, -1.0, -1.0], [3.0, -1.0, -1.0]])
    v1 = np.array([[10.0,  1.0, -1.0], [3.0,  1.0, -1.0]])
    v2 = np.array([[10.0,  0.0,  1.0], [3.0,  0.0,  1.0]])
    t, idx = ray_triangle_batch(origins, dirs, v0, v1, v2)
    np.testing.assert_allclose(t[0], 3.0, atol=1e-6)
    assert idx[0] == 1


# ---------------------------------------------------------------------------
# Ground plane
# ---------------------------------------------------------------------------
def test_ray_ground_hits_below():
    origins = np.array([[0.0, 0.0, 10.0]])
    dirs = np.array([[0.0, 0.0, -1.0]])
    t = ray_ground(origins, dirs, z0=0.0)
    np.testing.assert_allclose(t, [10.0], atol=1e-6)


def test_ray_ground_miss_upward():
    origins = np.array([[0.0, 0.0, 10.0]])
    dirs = np.array([[0.0, 0.0, 1.0]])
    t = ray_ground(origins, dirs, z0=0.0)
    assert np.isinf(t[0])


# ---------------------------------------------------------------------------
# nearest_hit composite
# ---------------------------------------------------------------------------
def test_nearest_hit_ground_beats_infinity():
    origins = np.array([[0.0, 0.0, 5.0]])
    dirs = np.array([[0.0, 0.0, -1.0]])
    empty = np.zeros((0, 3))
    t, kind = nearest_hit(origins, dirs, empty, empty, empty, empty, empty, ground_z=0.0)
    np.testing.assert_allclose(t, [5.0], atol=1e-6)
    assert kind[0] == 0  # ground


def test_nearest_hit_building_in_front_of_ground():
    origins = np.array([[0.0, 0.0, 1.0]])
    # Ray aims mostly forward with slight downward
    d = np.array([1.0, 0.0, -0.05])
    d = d / np.linalg.norm(d)
    dirs = d[None, :]
    boxes_min = np.array([[3.0, -1.0, 0.0]])
    boxes_max = np.array([[4.0,  1.0, 2.0]])
    empty = np.zeros((0, 3))
    t, kind = nearest_hit(origins, dirs, boxes_min, boxes_max, empty, empty, empty, ground_z=0.0)
    assert kind[0] == 1  # building should win over the far ground hit
