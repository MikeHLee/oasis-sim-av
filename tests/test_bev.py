"""Tests for BEV renderer (SIM-007)."""
from __future__ import annotations

import numpy as np
import pytest

from oasis_sim_av.bev import BEVRenderer
from oasis_sim_av.config import BEVConfig, WorldConfig, BuildingConfig
from oasis_sim_av.world import World


def test_bev_config_from_dict():
    bev = BEVConfig(center=[10.0, 5.0], extent_m=60.0, size_px=128)
    assert bev.center == [10.0, 5.0]
    assert bev.extent_m == 60.0
    assert bev.size_px == 128


def test_bev_renderer_from_config():
    cfg = BEVConfig(center=[0.0, 0.0], extent_m=40.0, size_px=64)
    renderer = BEVRenderer.from_config(cfg)
    assert renderer.size_px == 64
    assert renderer.extent_m == 40.0


def test_bev_renders_nonzero_scene():
    cfg = BEVConfig(center=[0.0, 0.0], extent_m=20.0, size_px=64)
    renderer = BEVRenderer.from_config(cfg)

    world_cfg = WorldConfig(
        ground_z=0.0,
        buildings=[BuildingConfig(aabb=[-5.0, -5.0, 0.0, 5.0, 5.0, 10.0])],
    )
    world = World.from_config(world_cfg)

    tri_v0 = np.array([[0.0, 0.0, 3.0]])
    tri_v1 = np.array([[2.0, 0.0, 3.0]])
    tri_v2 = np.array([[1.0, 1.0, 3.0]])

    img = renderer.render(world, tri_v0, tri_v1, tri_v2)

    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8

    non_zero = np.sum(img > 0)
    assert non_zero > 0, "BEV image should have non-zero pixels"


def test_bev_renders_building():
    cfg = BEVConfig(center=[0.0, 0.0], extent_m=20.0, size_px=64)
    renderer = BEVRenderer.from_config(cfg)

    world_cfg = WorldConfig(
        ground_z=0.0,
        buildings=[BuildingConfig(aabb=[-3.0, -3.0, 0.0, 3.0, 3.0, 8.0])],
    )
    world = World.from_config(world_cfg)

    empty_tris = np.zeros((0, 3))
    img = renderer.render(world, empty_tris, empty_tris, empty_tris)

    assert img.shape == (64, 64, 3)

    building_color = (np.array([0.55, 0.55, 0.58]) * 255).astype(np.uint8)
    has_building = np.any(np.all(img == building_color, axis=2))
    assert has_building, "BEV should show building in rendered image"


def test_bev_vehicle_marker():
    cfg = BEVConfig(center=[0.0, 0.0], extent_m=20.0, size_px=64, show_vehicle_marker=True)
    renderer = BEVRenderer.from_config(cfg)

    world_cfg = WorldConfig(ground_z=0.0)
    world = World.from_config(world_cfg)

    empty_tris = np.zeros((0, 3))
    vehicle_state = np.array([0.0, 0.0, 0.0])

    img = renderer.render(world, empty_tris, empty_tris, empty_tris, vehicle_state)

    red_pixels = np.sum((img[:, :, 0] > 200) & (img[:, :, 1] < 100) & (img[:, :, 2] < 100))
    assert red_pixels > 0, "Vehicle marker should appear as red pixels"
