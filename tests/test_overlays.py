"""Tests for overlay helpers (SIM-007)."""
from __future__ import annotations

import numpy as np
import pytest

from oasis_sim_av.overlays import (
    reproject_points_to_camera,
    rasterise_lidar_bev,
    draw_bboxes,
    draw_fusion_strip,
    compose_grid5x2,
)


def test_reproject_empty_points():
    points = np.zeros((0, 3))
    u, v, mask = reproject_points_to_camera(
        points,
        np.array([0.0, 0.0, 1.5]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        60.0, 320, 240,
        np.array([0.0, 0.0, 0.0]),
        np.eye(3),
    )
    assert len(u) == 0
    assert len(v) == 0
    assert len(mask) == 0


def test_reproject_points_in_front():
    points = np.array([[10.0, 0.0, 1.5]])
    u, v, mask = reproject_points_to_camera(
        points,
        np.array([0.0, 0.0, 1.5]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        60.0, 320, 240,
        np.array([0.0, 0.0, 0.0]),
        np.eye(3),
    )
    assert len(u) == 1
    assert mask[0], "Point in front of camera should be visible"


def test_reproject_points_behind():
    points = np.array([[-10.0, 0.0, 1.5]])
    u, v, mask = reproject_points_to_camera(
        points,
        np.array([0.0, 0.0, 1.5]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        60.0, 320, 240,
        np.array([0.0, 0.0, 0.0]),
        np.eye(3),
    )
    assert len(u) == 1
    assert not mask[0], "Point behind camera should not be visible"


def test_rasterise_lidar_bev_empty():
    points = np.zeros((0, 3))
    kinds = np.zeros((0,), dtype=np.int8)
    img = rasterise_lidar_bev(points, kinds, np.array([0.0, 0.0]), 40.0, 64)
    assert img.shape == (64, 64, 3)
    assert np.all(img == 0)


def test_rasterise_lidar_bev_with_points():
    points = np.array([[5.0, 0.0, 1.0], [-5.0, 0.0, 1.0]])
    kinds = np.array([0, 1], dtype=np.int8)
    img = rasterise_lidar_bev(points, kinds, np.array([0.0, 0.0]), 40.0, 64)
    assert img.shape == (64, 64, 3)

    non_zero = np.sum(img > 0)
    assert non_zero > 0, "Should have non-zero pixels for LiDAR points"


def test_draw_bboxes_empty():
    img = np.zeros((240, 320, 3), dtype=np.uint8) + 100
    out = draw_bboxes(img, [])
    assert np.array_equal(out, img)


def test_draw_bboxes_single():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    bboxes = [{"bbox": [100, 100, 200, 150], "score": 0.9}]
    out = draw_bboxes(img, bboxes)

    red_in_bbox = np.sum(
        (out[100:150, 100:200, 0] > 200) &
        (out[100:150, 100:200, 1] < 100) &
        (out[100:150, 100:200, 2] < 100)
    )
    assert red_in_bbox > 0, "Should draw red bbox outline"


def test_draw_fusion_strip_empty():
    strip = draw_fusion_strip([], 320, 48)
    assert strip.shape == (48, 320, 3)


def test_draw_fusion_strip_with_data():
    p_series = [0.3, 0.5, 0.7, 0.4]
    strip = draw_fusion_strip(p_series, 320, 48)
    assert strip.shape == (48, 320, 3)

    non_zero = np.sum(strip > 0)
    assert non_zero > 0


def test_compose_grid5x2():
    panels = [np.zeros((100, 120, 3), dtype=np.uint8) + (i * 20) for i in range(10)]

    composed = compose_grid5x2(panels)

    assert composed.ndim == 3
    assert composed.shape[2] == 3

    total_w = 5 * 120
    raw_h = 2 * (100 + 18) + 24
    expected_h = ((raw_h + 15) // 16) * 16
    assert composed.shape[1] == total_w
    assert composed.shape[0] == expected_h
    assert composed.shape[0] % 16 == 0, "Height must be divisible by 16 for macro-block alignment"


def test_compose_grid5x2_wrong_count():
    panels = [np.zeros((100, 120, 3), dtype=np.uint8) for _ in range(5)]
    with pytest.raises(ValueError, match="Expected 10 panels"):
        compose_grid5x2(panels)
