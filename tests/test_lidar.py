"""LiDAR ray-casting and noise statistics tests."""
from __future__ import annotations

import numpy as np

from oasis_sim_av.config import LiDARConfig
from oasis_sim_av.lidar import SimulatedLiDAR


def _empty_tris():
    return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))


def test_lidar_scans_single_box_returns():
    rng = np.random.default_rng(0)
    cfg = LiDARConfig(
        offset=[0, 0, 0],
        elevation_fov_deg=[-5, 5],
        elevation_rings=5,
        azimuth_fov_deg=[-10, 10],
        azimuth_rays=20,
        range_m=50.0,
        range_noise_std_m=0.0,
        rain_dropout_prob=0.0,
    )
    lidar = SimulatedLiDAR.from_config(cfg, rng)
    # One big wall at +x = 20m
    boxes_min = np.array([[20.0, -10.0, -5.0]])
    boxes_max = np.array([[21.0,  10.0,  5.0]])
    v0, v1, v2 = _empty_tris()
    vehicle_origin = np.array([0.0, 0.0, 0.0])
    body_to_world = np.eye(3)
    scan = lidar.scan(vehicle_origin, body_to_world, boxes_min, boxes_max, v0, v1, v2, ground_z=None)
    assert scan.points.shape[0] > 0
    assert np.allclose(scan.points[:, 0], 20.0, atol=1e-6)
    assert int(np.sum(scan.kind == 1)) == scan.points.shape[0]


def test_lidar_range_noise_statistics():
    """Known-range return with sigma noise should match sigma within 30%."""
    rng = np.random.default_rng(42)
    cfg = LiDARConfig(
        offset=[0, 0, 0],
        elevation_fov_deg=[0, 0],
        elevation_rings=1,
        azimuth_fov_deg=[0, 0],
        azimuth_rays=1,
        range_m=100.0,
        range_noise_std_m=0.05,
        rain_dropout_prob=0.0,
    )
    lidar = SimulatedLiDAR.from_config(cfg, rng)
    boxes_min = np.array([[10.0, -5.0, -5.0]])
    boxes_max = np.array([[11.0,  5.0,  5.0]])
    v0, v1, v2 = _empty_tris()
    R = np.eye(3)
    origin = np.zeros(3)
    dists = []
    for _ in range(500):
        scan = lidar.scan(origin, R, boxes_min, boxes_max, v0, v1, v2, ground_z=None)
        if scan.ranges.size:
            dists.append(float(scan.ranges[0]))
    dists = np.asarray(dists)
    assert len(dists) > 400  # most scans should hit
    mean = np.mean(dists)
    std = np.std(dists)
    assert abs(mean - 10.0) < 0.02           # bias should be near zero
    assert 0.03 < std < 0.08                  # within 30% of configured sigma


def test_lidar_rain_dropout_statistics():
    rng = np.random.default_rng(0)
    cfg = LiDARConfig(
        offset=[0, 0, 0],
        elevation_fov_deg=[-5, 5],
        elevation_rings=4,
        azimuth_fov_deg=[-10, 10],
        azimuth_rays=10,
        range_m=50.0,
        range_noise_std_m=0.0,
        rain_dropout_prob=0.5,
    )
    lidar = SimulatedLiDAR.from_config(cfg, rng)
    boxes_min = np.array([[10.0, -20.0, -10.0]])
    boxes_max = np.array([[11.0,  20.0,  10.0]])
    v0, v1, v2 = _empty_tris()
    R = np.eye(3)
    origin = np.zeros(3)
    returns = []
    for _ in range(50):
        scan = lidar.scan(origin, R, boxes_min, boxes_max, v0, v1, v2, ground_z=None)
        returns.append(scan.points.shape[0])
    total = sum(returns)
    max_possible = 4 * 10 * 50
    ratio = total / max_possible
    # With prob=0.5 dropout, we should keep roughly half
    assert 0.35 < ratio < 0.65


def test_lidar_tape_can_slip_between_rings():
    """With a sparse vertical scan, a thin ribbon at range can produce zero hits.

    This is the MVP's core demonstration: coarse angular resolution allows a
    narrow object to fall between scan rings.
    """
    rng = np.random.default_rng(7)
    cfg = LiDARConfig(
        offset=[0, 0, 0],
        elevation_fov_deg=[-2, 2],
        elevation_rings=4,        # 4 rings across 4deg -> ~1.33 deg spacing
        azimuth_fov_deg=[-5, 5],
        azimuth_rays=60,
        range_m=100.0,
        range_noise_std_m=0.0,
        rain_dropout_prob=0.0,
    )
    lidar = SimulatedLiDAR.from_config(cfg, rng)
    # A ribbon 5 cm tall at 50 m range, between two rings:
    # 1.33 deg at 50 m = 1.16 m ring spacing -> 5 cm ribbon can slip through
    # Build a thin horizontal triangle strip at z = 0.0 (between -0.33deg and +0.33deg rings)
    v0 = np.array([[50.0, -2.0, 0.0]])
    v1 = np.array([[50.0,  2.0, 0.0]])
    v2 = np.array([[50.0,  0.0, 0.025]])  # 2.5 cm half-height
    boxes_min = np.zeros((0, 3))
    boxes_max = np.zeros((0, 3))
    R = np.eye(3)
    scan = lidar.scan(np.zeros(3), R, boxes_min, boxes_max, v0, v1, v2, ground_z=None)
    # We expect very few hits (not guaranteed zero, but << 60)
    assert int(np.sum(scan.kind == 2)) < 10


def test_lidar_writes_ply(tmp_path):
    rng = np.random.default_rng(0)
    cfg = LiDARConfig(
        elevation_fov_deg=[-2, 2], elevation_rings=2,
        azimuth_fov_deg=[-2, 2], azimuth_rays=3,
        range_m=50.0, range_noise_std_m=0.0, rain_dropout_prob=0.0,
    )
    lidar = SimulatedLiDAR.from_config(cfg, rng)
    boxes_min = np.array([[10.0, -10.0, -10.0]])
    boxes_max = np.array([[11.0, 10.0, 10.0]])
    v0, v1, v2 = _empty_tris()
    scan = lidar.scan(np.zeros(3), np.eye(3), boxes_min, boxes_max, v0, v1, v2, ground_z=None)
    out = tmp_path / "scan.ply"
    lidar.write_ply(scan, str(out))
    text = out.read_text()
    assert text.startswith("ply")
    assert "element vertex" in text
