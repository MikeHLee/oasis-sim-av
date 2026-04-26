"""Tests for the render_video HUD overlay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oasis_sim_av.config import ScenarioConfig
from oasis_sim_av.render_video import (
    annotate_frame,
    frame_records,
    load_state,
    render_video,
)
from oasis_sim_av.run import run


# ---------------------------------------------------------------------------
# Unit tests (no simulation needed)
# ---------------------------------------------------------------------------
def test_annotate_frame_appends_hud_strip() -> None:
    img = np.full((40, 80, 3), 100, dtype=np.uint8)
    rec = {"t": 0.5, "frame_idx": 2,
           "lidar": {"n_tape_hits_true": 3, "n_tape_hits_returned": 2}}
    out = annotate_frame(img, rec, max_returned=5, bar_series=[0, 1, 2], frame_idx=2)
    assert out.shape[0] > img.shape[0], "HUD strip should extend the frame height"
    assert out.shape[1] == img.shape[1]
    # Original frame is preserved at the top
    np.testing.assert_array_equal(out[: img.shape[0]], img)
    # HUD region is not all-black — we stamped at least bar columns
    hud = out[img.shape[0]:]
    assert hud.any()


def test_annotate_frame_handles_zero_hits() -> None:
    img = np.full((30, 60, 3), 50, dtype=np.uint8)
    rec = {"t": 0.0, "frame_idx": 0,
           "lidar": {"n_tape_hits_true": 0, "n_tape_hits_returned": 0}}
    out = annotate_frame(img, rec, max_returned=1, bar_series=[0], frame_idx=0)
    assert out.shape == (30 + 48, 60, 3)


# ---------------------------------------------------------------------------
# Integration: run sim, render video
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_run(tmp_path: Path) -> Path:
    cfg = ScenarioConfig()
    cfg.duration_s = 0.2
    cfg.dt = 0.01
    cfg.output.frame_every = 5
    cfg.camera.width = 48
    cfg.camera.height = 32
    cfg.camera.motion_blur_samples = 1
    cfg.lidar.elevation_rings = 4
    cfg.lidar.azimuth_rays = 16
    cfg.world.buildings = []
    return run(cfg, tmp_path / "runs", log=False)


def test_load_state_and_frame_records_round_trip(tiny_run: Path) -> None:
    state = load_state(tiny_run)
    assert state, "state.jsonl must not be empty"
    rows = frame_records(state)
    assert rows
    for r in rows:
        assert "frame_idx" in r and "lidar" in r


def test_render_video_writes_output(tiny_run: Path) -> None:
    # --save-frames gives us a deterministic artefact regardless of whether
    # imageio/ffmpeg is installed.
    out = render_video(tiny_run, out=tiny_run / "video.mp4", fps=5, save_frames=True)
    assert out.exists()
    overlay = list((tiny_run / "overlay").glob("*"))
    assert overlay, "expected annotated frames in overlay/"
