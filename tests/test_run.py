"""Smoke test: run a brief simulation end-to-end."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from oasis_sim_av.config import ScenarioConfig
from oasis_sim_av.run import run


@pytest.fixture
def short_cfg() -> ScenarioConfig:
    cfg = ScenarioConfig()
    cfg.duration_s = 0.3
    cfg.dt = 0.01
    cfg.output.frame_every = 5   # 6 frames over 0.3 s
    # Small render so the test is fast
    cfg.camera.width = 48
    cfg.camera.height = 32
    cfg.camera.motion_blur_samples = 2
    cfg.lidar.elevation_rings = 8
    cfg.lidar.azimuth_rays = 32
    # Give it a building to hit
    cfg.world.buildings = []
    # Keep the default tape between default anchors
    return cfg


def test_smoke_end_to_end(tmp_path: Path, short_cfg: ScenarioConfig) -> None:
    out = tmp_path / "runs"
    run_dir = run(short_cfg, out, log=False)
    assert run_dir.exists()
    # Output directories populated
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "state.jsonl").exists()
    frames = list((run_dir / "frames").glob("*.png"))
    ply = list((run_dir / "lidar").glob("*.ply"))
    assert len(frames) >= 5
    assert len(ply) >= 5

    # state.jsonl contains at least one lidar record
    with open(run_dir / "state.jsonl") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert records
    lidar_records = [r for r in records if "lidar" in r]
    assert lidar_records


def test_run_with_no_buildings_still_produces_output(tmp_path: Path) -> None:
    cfg = ScenarioConfig()
    cfg.duration_s = 0.05
    cfg.dt = 0.01
    cfg.output.frame_every = 1
    cfg.camera.width = 16
    cfg.camera.height = 12
    cfg.camera.motion_blur_samples = 1
    cfg.lidar.elevation_rings = 4
    cfg.lidar.azimuth_rays = 8
    cfg.world.buildings = []
    run_dir = run(cfg, tmp_path / "runs", log=False)
    # Still get at least one frame
    assert len(list((run_dir / "frames").glob("*.png"))) >= 1
