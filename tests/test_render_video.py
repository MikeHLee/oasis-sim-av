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


# ---------------------------------------------------------------------------
# Integration: grid5x2 renderer (SIM-010 — panels 3, 4, 8, 9 must be populated)
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_run_with_bev(tmp_path: Path) -> Path:
    """Tiny run that also produces BEV images + lidar npz sidecars."""
    from oasis_sim_av.config import BEVConfig

    cfg = ScenarioConfig()
    cfg.duration_s = 0.2
    cfg.dt = 0.01
    cfg.output.frame_every = 5
    cfg.camera.width = 48
    cfg.camera.height = 32
    cfg.camera.motion_blur_samples = 1
    cfg.lidar.elevation_rings = 4
    cfg.lidar.azimuth_rays = 16
    # A building gives the LiDAR something to return kind=1 points for,
    # which in turn gives panel 8 (LiDAR BEV) non-empty pixels.
    from oasis_sim_av.config import BuildingConfig
    cfg.world.buildings = [
        BuildingConfig(aabb=[5.0, -4.0, 0.0, 15.0, -2.0, 5.0])
    ]
    cfg.bev = BEVConfig(center=[5.0, 0.0], extent_m=30.0, size_px=48)
    return run(cfg, tmp_path / "runs", log=False)


def test_grid5x2_panels_are_not_stubs(tiny_run_with_bev: Path) -> None:
    """Regression guard: panels 3, 4, 8, 9 must contain non-trivial content.

    Before SIM-010 these were placeholders — panel 3/4 were copies of the
    raw camera image, panel 8 was a solid grey square, panel 9 was a copy
    of panel 7. This test would have caught all four stubs.
    """
    from oasis_sim_av.render_video import render_video_grid5x2

    out = render_video_grid5x2(
        tiny_run_with_bev,
        out=tiny_run_with_bev / "video_grid5x2.mp4",
        fps=5,
        save_frames=True,
    )
    assert out.exists()

    # Find a composed per-frame artefact (PNG or NPY).
    overlay_dir = tiny_run_with_bev / "overlay_grid5x2"
    frame_files = sorted(overlay_dir.glob("*.png"))
    if not frame_files:
        # Fallback to .npy if matplotlib is missing.
        frame_files = sorted(overlay_dir.glob("*.npy"))
    assert frame_files, "grid5x2 must produce per-frame artefacts"

    # Load the last (most settled) composed frame.
    last = frame_files[-1]
    if last.suffix == ".png":
        try:
            import imageio.v3 as iio
            composed = iio.imread(str(last))[..., :3].astype(np.uint8)
        except ImportError:
            from matplotlib.image import imread as mpl_imread
            composed = (mpl_imread(str(last))[..., :3] * 255).astype(np.uint8)
    else:
        composed = np.load(last).astype(np.uint8)

    # The composed image is 5 columns × 2 rows. Carve out each panel and
    # check that the wiring produces non-placeholder content.
    H, W = composed.shape[:2]
    title_h = 18
    footer_h = 24
    row_h = (H - footer_h) // 2
    col_w = W // 5
    panel_h = row_h - title_h

    def panel(row: int, col: int) -> np.ndarray:
        y0 = row * row_h + title_h
        x0 = col * col_w
        return composed[y0 : y0 + panel_h, x0 : x0 + col_w]

    p1 = panel(0, 0)  # camera raw
    p3 = panel(0, 2)  # camera + LiDAR
    p4 = panel(0, 3)  # fused
    p7 = panel(1, 1)  # BEV + trail
    p8 = panel(1, 2)  # LiDAR BEV
    p9 = panel(1, 3)  # fused BEV

    # Panel 3 differs from panel 1: overlaid LiDAR points should inject
    # at least a handful of coloured pixels somewhere in the crop.
    delta_cam = np.abs(p3.astype(np.int16) - p1.astype(np.int16)).sum()
    assert delta_cam > 0, (
        "panel 3 (camera + LiDAR) is identical to panel 1 — LiDAR reprojection "
        "was not wired up"
    )

    # Panel 4 differs from panel 3: bboxes may or may not fire depending on
    # RNG, but panel 4 must at minimum contain the LiDAR overlay (delta
    # vs panel 1 is non-zero).
    delta_fused = np.abs(p4.astype(np.int16) - p1.astype(np.int16)).sum()
    assert delta_fused > 0, "panel 4 (fused) has no LiDAR overlay"

    # Panel 8 (LiDAR BEV) — must not be a solid grey square. The old stub
    # filled it with 30-valued pixels; a real rasterisation produces at
    # least a few kind-coloured points (or all zeros if no hits in
    # viewport, which is still distinguishable from the old stub).
    uniques = len(np.unique(p8.reshape(-1, 3), axis=0))
    assert uniques > 1, (
        "panel 8 (LiDAR BEV) is a single flat colour — rasterise_lidar_bev was "
        "not called"
    )

    # Panel 9 (fused BEV) must differ from panel 7 (trail-only). Since the
    # LiDAR rasterisation in panel 8 produced at least one coloured pixel,
    # the overlay should propagate to panel 9.
    delta_bev = np.abs(p9.astype(np.int16) - p7.astype(np.int16)).sum()
    assert delta_bev > 0, (
        "panel 9 (fused BEV) is identical to panel 7 — LiDAR overlay on BEV "
        "truth was not wired up"
    )


def test_grid5x2_with_rain_clutter_writes_viz_scan(tmp_path: Path) -> None:
    """With rain_clutter enabled, lidar_viz/NNNNNN.npz contains kind=3 points.

    Guards memory.md Decision 4: clean .ply stays rain-free, viz scan
    carries the droplets. The grid renderer prefers the viz scan.
    """
    from oasis_sim_av.config import BEVConfig, BuildingConfig, RainClutterConfig

    cfg = ScenarioConfig()
    cfg.duration_s = 0.1
    cfg.dt = 0.01
    cfg.output.frame_every = 5
    cfg.camera.width = 32
    cfg.camera.height = 24
    cfg.camera.motion_blur_samples = 1
    cfg.lidar.elevation_rings = 4
    cfg.lidar.azimuth_rays = 16
    cfg.world.buildings = [
        BuildingConfig(aabb=[5.0, -4.0, 0.0, 15.0, -2.0, 5.0])
    ]
    cfg.bev = BEVConfig(center=[5.0, 0.0], extent_m=30.0, size_px=32)
    cfg.rain_clutter = RainClutterConfig(
        enabled=True, n_droplets=30,
        spawn_box=[0.0, -5.0, 0.0, 15.0, 5.0, 3.0],
    )

    run_dir = run(cfg, tmp_path / "runs", log=False)

    viz_files = list((run_dir / "lidar_viz").glob("*.npz"))
    assert viz_files, "lidar_viz/ should be populated when rain_clutter enabled"

    # First viz scan must contain kind=3 (rain) points.
    data = np.load(viz_files[0])
    kinds = data["kind"]
    assert (kinds == 3).sum() > 0, (
        "rain-augmented scan must contain kind=3 points (decision 4 cyan)"
    )

    # And the clean .ply file for the same frame must NOT contain rain
    # points. The .ply writer clips kind to [0, 2] so even if we
    # regressed the scan_clean path we would spot it as colour-clash.
    ply_files = sorted((run_dir / "lidar").glob("*.ply"))
    assert ply_files
    with open(ply_files[0]) as f:
        txt = f.read()
    # Rain cyan is (0, 200, 220) per overlays.KIND_COLORS; it must not
    # appear in the clean PLY colour table.
    assert "0 200 220" not in txt, (
        "clean .ply leaked rain points — Decision 4 violated"
    )
