"""Tests for the 1D complementary fusion filter."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oasis_sim_av.config import ScenarioConfig
from oasis_sim_av.fusion import (
    ComplementaryFilter,
    FusionConfig,
    run_fusion,
    summary_stats,
    yellow_pixel_count,
)
from oasis_sim_av.run import run


# ---------------------------------------------------------------------------
# Unit: yellow_pixel_count heuristic
# ---------------------------------------------------------------------------
def test_yellow_pixel_count_matches_tape_pixels() -> None:
    # Blank grey image with a 6x20 tape-yellow stripe across the middle
    img = np.full((60, 100, 3), 120, dtype=np.uint8)
    img[30:36, 40:60, :] = (240, 218, 38)
    # Tight crop that includes the stripe
    n = yellow_pixel_count(img, crop_hfrac=(0.0, 1.0), crop_wfrac=(0.0, 1.0))
    assert n == 6 * 20, f"expected 120 yellow pixels, got {n}"


def test_yellow_pixel_count_ignores_sky_and_grey() -> None:
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:20, :, :] = (150, 180, 220)  # sky blue -> not yellow
    img[20:, :, :] = (100, 100, 100)  # grey    -> not yellow
    assert yellow_pixel_count(img, crop_hfrac=(0.0, 1.0), crop_wfrac=(0.0, 1.0)) == 0


# ---------------------------------------------------------------------------
# Unit: ComplementaryFilter
# ---------------------------------------------------------------------------
def test_complementary_filter_tracks_sustained_input() -> None:
    cfg = FusionConfig(alpha=10.0, w_lidar=0.5, w_camera=0.5)
    flt = ComplementaryFilter(cfg)
    # Drive with p_l = 1.0, p_c = 1.0 for 1 second at 100 Hz
    y = 0.0
    for _ in range(100):
        y = flt.update(1.0, 1.0, 0.01)
    assert y > 0.99, f"filter should saturate near 1.0, got {y:.3f}"


def test_complementary_filter_low_input_stays_low() -> None:
    cfg = FusionConfig(alpha=4.0)
    flt = ComplementaryFilter(cfg)
    y_max = 0.0
    for _ in range(200):
        y = flt.update(0.05, 0.05, 0.01)
        y_max = max(y_max, y)
    assert y_max < cfg.detect_threshold, (
        f"sustained p=0.05 should never cross detection, peaked at {y_max:.3f}"
    )


def test_weighted_fusion_respects_bias() -> None:
    # Trust LiDAR 100%, camera 0%: fused should track LiDAR exactly
    cfg = FusionConfig(w_lidar=1.0, w_camera=0.0, alpha=1000.0)  # alpha*dt>=1 -> no lag
    flt = ComplementaryFilter(cfg)
    y = flt.update(0.8, 0.0, 0.1)
    assert y == pytest.approx(0.8, abs=1e-3)


# ---------------------------------------------------------------------------
# Integration: run the sim, then fuse
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_run(tmp_path: Path) -> Path:
    cfg = ScenarioConfig()
    cfg.duration_s = 0.3
    cfg.dt = 0.01
    cfg.output.frame_every = 5
    cfg.camera.width = 48
    cfg.camera.height = 32
    cfg.camera.motion_blur_samples = 1
    cfg.lidar.elevation_rings = 8
    cfg.lidar.azimuth_rays = 32
    cfg.world.buildings = []
    return run(cfg, tmp_path / "runs", log=False)


def test_run_fusion_produces_jsonl_and_records(tiny_run: Path) -> None:
    records, jpath = run_fusion(tiny_run, save_png=False)
    assert jpath.exists()
    assert len(records) >= 5
    # jsonl roundtrip
    with open(jpath) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == len(records)
    for line in lines:
        assert set(line) >= {"t", "p_lidar", "p_camera", "p_fused", "detected"}
        assert 0.0 <= line["p_fused"] <= 1.0


def test_baseline_tape_stays_below_threshold(tiny_run: Path) -> None:
    """The whole point of the demo: fused probability stays sub-threshold."""
    cfg = FusionConfig(detect_threshold=0.5, alpha=6.0)
    records, _ = run_fusion(tiny_run, cfg, save_png=False)
    stats = summary_stats(records)
    # With a tiny 32x32 camera and 32-ray azimuth sweep, the sensor signal is
    # extremely sparse -> fused posterior should not reach detection.
    assert stats["max_p_fused"] < cfg.detect_threshold, stats
    assert stats["frac_detected"] == 0.0, stats
