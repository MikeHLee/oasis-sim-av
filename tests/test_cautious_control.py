"""Tests for SIM-011: in-loop fusion + cautious_pursuit controller + abstention."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oasis_sim_av.config import (
    BuildingConfig,
    ScenarioConfig,
    VehicleControllerConfig,
)
from oasis_sim_av.run import run
from oasis_sim_av.vehicle import make_controller


# ---------------------------------------------------------------------------
# Unit: controller callables accept the new `percept` kwarg
# ---------------------------------------------------------------------------
def test_controllers_accept_percept_kwarg() -> None:
    """Every controller type must accept (t, state, percept=...)."""
    for kind in ("constant", "step", "ramp", "sine", "impulse_steer"):
        cfg = VehicleControllerConfig(type=kind, base_v=5.0)
        ctrl = make_controller(cfg)
        v, d = ctrl(0.5, np.array([0.0, 0.0, 0.0]), percept=None)
        assert v == pytest.approx(5.0)
        # Also with a real percept dict
        v2, d2 = ctrl(0.5, np.array([0.0, 0.0, 0.0]),
                      percept={"p_fused": 0.9})
        assert v2 == pytest.approx(5.0)


def test_non_cautious_controller_ignores_percept() -> None:
    """Without cautious=True the velocity must not be modulated."""
    cfg = VehicleControllerConfig(type="constant", base_v=10.0, cautious=False)
    ctrl = make_controller(cfg)
    v_high, _ = ctrl(0.0, np.zeros(3), percept={"p_fused": 1.0})
    v_low, _ = ctrl(0.0, np.zeros(3), percept={"p_fused": 0.0})
    assert v_high == v_low == 10.0


# ---------------------------------------------------------------------------
# Unit: cautious wrapper modulates velocity by p_fused
# ---------------------------------------------------------------------------
def test_cautious_full_confidence_passes_through() -> None:
    cfg = VehicleControllerConfig(
        type="constant",
        base_v=10.0,
        cautious=True,
        cautious_p_threshold=0.5,
        cautious_min_v_frac=0.1,
    )
    ctrl = make_controller(cfg)
    v, _ = ctrl(0.0, np.zeros(3), percept={"p_fused": 0.9})
    assert v == pytest.approx(10.0), "high-confidence must not be scaled down"


def test_cautious_low_confidence_slows_vehicle() -> None:
    cfg = VehicleControllerConfig(
        type="constant",
        base_v=10.0,
        cautious=True,
        cautious_p_threshold=0.5,
        cautious_min_v_frac=0.1,
    )
    ctrl = make_controller(cfg)
    v, _ = ctrl(0.0, np.zeros(3), percept={"p_fused": 0.0})
    assert v == pytest.approx(1.0), "zero confidence -> min_v_frac * base_v"


def test_cautious_half_threshold_half_speed() -> None:
    cfg = VehicleControllerConfig(
        type="constant",
        base_v=10.0,
        cautious=True,
        cautious_p_threshold=0.4,
        cautious_min_v_frac=0.0,
    )
    ctrl = make_controller(cfg)
    v, _ = ctrl(0.0, np.zeros(3), percept={"p_fused": 0.2})
    # scale = clip(0.2 / 0.4, 0, 1) = 0.5
    assert v == pytest.approx(5.0)


def test_cautious_no_percept_passes_through() -> None:
    """Before any sensor fires, percept is None — don't stall the vehicle."""
    cfg = VehicleControllerConfig(
        type="constant", base_v=10.0, cautious=True,
        cautious_p_threshold=0.5, cautious_min_v_frac=0.1,
    )
    ctrl = make_controller(cfg)
    v, _ = ctrl(0.0, np.zeros(3), percept=None)
    assert v == pytest.approx(10.0)


def test_cautious_steering_is_not_modulated() -> None:
    """Cautious mode only scales velocity, steering must pass through."""
    cfg = VehicleControllerConfig(
        type="constant",
        base_v=10.0,
        base_delta=0.3,
        cautious=True,
    )
    ctrl = make_controller(cfg)
    _, d_high = ctrl(0.0, np.zeros(3), percept={"p_fused": 1.0})
    _, d_low = ctrl(0.0, np.zeros(3), percept={"p_fused": 0.0})
    assert d_high == d_low == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Integration: in-loop fusion populates state.jsonl["fusion"] each frame
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
    cfg.lidar.elevation_rings = 4
    cfg.lidar.azimuth_rays = 16
    cfg.world.buildings = [BuildingConfig(aabb=[5.0, -4.0, 0.0, 15.0, -2.0, 5.0])]
    return run(cfg, tmp_path / "runs", log=False)


def test_state_jsonl_has_fusion_per_frame(tiny_run: Path) -> None:
    """Every sensor-fire record must now carry a `fusion` block."""
    with open(tiny_run / "state.jsonl") as f:
        records = [json.loads(line) for line in f if line.strip()]
    frame_rows = [r for r in records if "frame_idx" in r]
    assert frame_rows, "no sensor frames — fixture is too short"
    for r in frame_rows:
        assert "fusion" in r, f"frame {r['frame_idx']} missing fusion block"
        f = r["fusion"]
        assert 0.0 <= f["p_fused"] <= 1.0
        assert 0.0 <= f["p_lidar"] <= 1.0
        assert 0.0 <= f["p_camera"] <= 1.0
        assert isinstance(f["detected"], bool)


def test_abstain_jsonl_exists_and_is_valid(tiny_run: Path) -> None:
    """abstain.jsonl must exist (may be empty if p_fused always stays high)."""
    path = tiny_run / "abstain.jsonl"
    assert path.exists(), "abstain.jsonl must be created even when empty"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            assert "frame_idx" in entry
            assert "p_fused" in entry
            assert entry["p_fused"] < 0.2  # default abstain_p_threshold


# ---------------------------------------------------------------------------
# Integration: cautious mode slows the vehicle in a low-confidence scenario
# ---------------------------------------------------------------------------
def _run_comparing(tmp_path: Path, cautious: bool) -> list[dict]:
    cfg = ScenarioConfig()
    cfg.duration_s = 1.0
    cfg.dt = 0.01
    cfg.output.frame_every = 5
    cfg.camera.width = 48
    cfg.camera.height = 32
    cfg.camera.motion_blur_samples = 1
    cfg.lidar.elevation_rings = 4
    cfg.lidar.azimuth_rays = 16
    cfg.world.buildings = []
    # Very low confidence scenario: no buildings, no sensor data hits tape
    # hard enough to push p_fused above threshold — so cautious slows down.
    cfg.vehicle.controller = VehicleControllerConfig(
        type="constant",
        base_v=10.0,
        cautious=cautious,
        cautious_p_threshold=0.5,
        cautious_min_v_frac=0.2,
        abstain_p_threshold=0.2,
    )
    out = run(cfg, tmp_path / ("runs_c" if cautious else "runs_n"), log=False)
    with open(out / "state.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_cautious_moves_less_than_aggressive(tmp_path: Path) -> None:
    """A cautious vehicle in a low-confidence scene must travel less far.

    This is the end-to-end "does low-confidence actually slow us down?" test.
    """
    agg = _run_comparing(tmp_path, cautious=False)
    cau = _run_comparing(tmp_path, cautious=True)

    agg_x = agg[-1]["vehicle"]["x"]
    cau_x = cau[-1]["vehicle"]["x"]

    # Both start at the same x=-30 and move in +x.
    assert agg_x > cau_x, (
        f"cautious vehicle must travel less far than aggressive "
        f"(cautious x={cau_x:.2f}, aggressive x={agg_x:.2f})"
    )


# ---------------------------------------------------------------------------
# Back-compat: default configs still work
# ---------------------------------------------------------------------------
def test_default_scenario_still_runs(tmp_path: Path) -> None:
    """A run with no controller changes must behave identically to pre-SIM-011."""
    cfg = ScenarioConfig()
    cfg.duration_s = 0.1
    cfg.dt = 0.01
    cfg.output.frame_every = 5
    cfg.camera.width = 16
    cfg.camera.height = 12
    cfg.camera.motion_blur_samples = 1
    cfg.lidar.elevation_rings = 4
    cfg.lidar.azimuth_rays = 8
    cfg.world.buildings = []
    out = run(cfg, tmp_path / "runs", log=False)
    assert (out / "state.jsonl").exists()
    assert (out / "abstain.jsonl").exists()
