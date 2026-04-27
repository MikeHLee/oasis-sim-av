"""Tests for oracle detector (SIM-008)."""
from __future__ import annotations

import numpy as np
import pytest

from oasis_sim_av.config import CameraConfig
from oasis_sim_av.cloth import MassSpringCloth, TapeConfig
from oasis_sim_av.detect import OracleDetector, OracleDetectorConfig
from oasis_sim_av.vehicle import KinematicBicycle


def make_test_detector(
    rain_dropout: float = 0.05,
    seed: int = 42,
) -> tuple[OracleDetector, CameraConfig]:
    cam_cfg = CameraConfig(
        offset=[1.5, 0.0, 1.2],
        forward=[1.0, 0.0, 0.0],
        up=[0.0, 0.0, 1.0],
        fov_h_deg=60.0,
        width=320,
        height=240,
    )
    det_cfg = OracleDetectorConfig()
    rng = np.random.default_rng(seed)
    return OracleDetector(det_cfg, cam_cfg, rain_dropout, rng), cam_cfg


def make_test_cloth(x: float = 20.0, seed: int = 42) -> MassSpringCloth:
    tape_cfg = TapeConfig(
        anchor_a=[x - 5.0, -2.0, 3.0],
        anchor_b=[x + 5.0, -2.0, 3.0],
        length=10.5,
        n_length=20,
        n_width=2,
    )
    rng = np.random.default_rng(seed)
    cloth = MassSpringCloth.from_config(tape_cfg, rng)
    for _ in range(30):
        cloth.step(0.01)
    return cloth


def test_oracle_detects_tape_in_baseline():
    detector, cam_cfg = make_test_detector(rain_dropout=0.05)
    cloth = make_test_cloth(x=20.0)
    vehicle = KinematicBicycle(
        wheelbase=2.7, state=np.array([-10.0, 0.0, 0.0])
    )

    veh_origin = vehicle.pose_xyz()
    veh_R = vehicle.body_to_world()

    bboxes = detector.detect(cloth, vehicle, veh_origin, veh_R)

    assert len(bboxes) > 0, "Should detect tape in good conditions"
    assert bboxes[0].cls == "tape"


def test_oracle_bbox_inside_image_bounds():
    detector, cam_cfg = make_test_detector()
    cloth = make_test_cloth(x=20.0)
    vehicle = KinematicBicycle(wheelbase=2.7, state=np.array([-10.0, 0.0, 0.0]))

    bboxes = detector.detect(cloth, vehicle, vehicle.pose_xyz(), vehicle.body_to_world())

    if bboxes:
        bb = bboxes[0]
        assert 0 <= bb.xmin < cam_cfg.width
        assert 0 <= bb.xmax < cam_cfg.width
        assert 0 <= bb.ymin < cam_cfg.height
        assert 0 <= bb.ymax < cam_cfg.height
        assert bb.xmin < bb.xmax
        assert bb.ymin < bb.ymax


def test_oracle_score_degrades_with_conditions():
    detector_low_rain, _ = make_test_detector(rain_dropout=0.01, seed=100)
    detector_high_rain, _ = make_test_detector(rain_dropout=0.30, seed=100)

    cloth = make_test_cloth(x=35.0)
    vehicle = KinematicBicycle(wheelbase=2.7, state=np.array([0.0, 0.0, 0.0]))

    veh_origin = vehicle.pose_xyz()
    veh_R = vehicle.body_to_world()

    bboxes_low = detector_low_rain.detect(cloth, vehicle, veh_origin, veh_R)
    bboxes_high = detector_high_rain.detect(cloth, vehicle, veh_origin, veh_R)

    if bboxes_low and bboxes_high:
        assert bboxes_high[0].score <= bboxes_low[0].score or len(bboxes_high) <= len(bboxes_low)


def test_oracle_detect_to_dict():
    detector, _ = make_test_detector()
    cloth = make_test_cloth(x=20.0)
    vehicle = KinematicBicycle(wheelbase=2.7, state=np.array([-10.0, 0.0, 0.0]))

    detections = detector.detect_to_dict(cloth, vehicle, vehicle.pose_xyz(), vehicle.body_to_world())

    assert isinstance(detections, list)
    if detections:
        d = detections[0]
        assert "class" in d
        assert "bbox" in d
        assert "score" in d
        assert len(d["bbox"]) == 4
        assert 0.0 <= d["score"] <= 1.0


def test_oracle_misses_hidden_tape():
    detector, _ = make_test_detector()
    cloth = make_test_cloth(x=-50.0)
    vehicle = KinematicBicycle(wheelbase=2.7, state=np.array([0.0, 0.0, 0.0]))

    veh_origin = vehicle.pose_xyz()
    veh_R = vehicle.body_to_world()

    bboxes = detector.detect(cloth, vehicle, veh_origin, veh_R)

    assert len(bboxes) == 0, "Should not detect tape behind vehicle"
