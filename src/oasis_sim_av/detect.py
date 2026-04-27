"""Oracle-projection detector with condition-dependent noise.

NOT a learned detector. This is a deliberately simple stand-in that projects
the known tape bounding region into camera space, then applies noise that
degrades with the same physical conditions that would degrade a real camera:
range, motion blur, rain intensity, and cloth flutter.

This decouples "detector quality" from "camera visibility" so the multi-view
demo still tells the correct story about sensor fusion failure modes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cloth import MassSpringCloth
from .config import CameraConfig
from .overlays import reproject_points_to_camera
from .vehicle import KinematicBicycle
from .world import World


@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    score: float
    cls: str


@dataclass
class OracleDetectorConfig:
    base_score: float = 0.9
    min_area_px: int = 4
    min_corners_visible: int = 1


class OracleDetector:
    """Condition-modulated oracle detector for police tape.

    Projects the cloth's bounding region into camera image space and applies
    noise that depends on range, rain, and cloth velocity.
    """

    def __init__(
        self,
        cfg: OracleDetectorConfig,
        camera_cfg: CameraConfig,
        rain_dropout_prob: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.cfg = cfg
        self.camera_cfg = camera_cfg
        self.rain_dropout_prob = rain_dropout_prob
        self.rng = rng or np.random.default_rng()

    def detect(
        self,
        cloth: MassSpringCloth,
        vehicle: KinematicBicycle,
        veh_origin: np.ndarray,
        veh_R: np.ndarray,
    ) -> list[BBox]:
        """Detect tape in the current frame.

        Returns a list of bounding boxes (usually 0 or 1 for single-tape scenes).
        """
        positions = cloth.positions.reshape(-1, 3)

        corners_3d = np.array([
            [positions[:, 0].min(), positions[:, 1].min(), positions[:, 2].min()],
            [positions[:, 0].max(), positions[:, 1].min(), positions[:, 2].min()],
            [positions[:, 0].min(), positions[:, 1].max(), positions[:, 2].min()],
            [positions[:, 0].max(), positions[:, 1].max(), positions[:, 2].min()],
            [positions[:, 0].min(), positions[:, 1].min(), positions[:, 2].max()],
            [positions[:, 0].max(), positions[:, 1].min(), positions[:, 2].max()],
            [positions[:, 0].min(), positions[:, 1].max(), positions[:, 2].max()],
            [positions[:, 0].max(), positions[:, 1].max(), positions[:, 2].max()],
        ])

        u, v, mask = reproject_points_to_camera(
            corners_3d,
            np.asarray(self.camera_cfg.offset, dtype=np.float64),
            np.asarray(self.camera_cfg.forward, dtype=np.float64),
            np.asarray(self.camera_cfg.up, dtype=np.float64),
            self.camera_cfg.fov_h_deg,
            self.camera_cfg.width,
            self.camera_cfg.height,
            veh_origin,
            veh_R,
        )

        visible_idx = np.where(mask)[0]
        if len(visible_idx) < self.cfg.min_corners_visible:
            return []

        u_vis = u[mask]
        v_vis = v[mask]

        xmin = max(0, int(u_vis.min()))
        xmax = min(self.camera_cfg.width - 1, int(u_vis.max()))
        ymin = max(0, int(v_vis.min()))
        ymax = min(self.camera_cfg.height - 1, int(v_vis.max()))

        area = (xmax - xmin) * (ymax - ymin)
        if area < self.cfg.min_area_px:
            return []

        center_3d = positions.mean(axis=0)
        range_m = np.linalg.norm(center_3d - veh_origin)

        cloth_rms_velocity = np.sqrt(np.mean(np.sum(cloth.velocities.reshape(-1, 3) ** 2, axis=1)))

        score = self._compute_score(range_m, cloth_rms_velocity)

        if self.rng.random() > score:
            return []

        xmin, ymin, xmax, ymax = self._apply_jitter(
            xmin, ymin, xmax, ymax, range_m, area
        )

        return [BBox(xmin, ymin, xmax, ymax, score, "tape")]

    def _compute_score(self, range_m: float, cloth_rms_velocity: float) -> float:
        """Compute detection score based on physical conditions."""
        score = self.cfg.base_score

        range_norm = min(1.0, range_m / 50.0)
        score -= 0.15 * self.rain_dropout_prob
        score -= 0.05 * min(1.0, cloth_rms_velocity / 5.0)
        score -= 0.1 * range_norm

        return max(0.1, min(1.0, score))

    def _apply_jitter(
        self,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
        range_m: float,
        area: float,
    ) -> tuple[int, int, int, int]:
        """Apply condition-dependent jitter to bbox coordinates."""
        bbox_width = xmax - xmin

        xy_jitter_std = max(1.0, bbox_width * 0.05 + range_m * 0.02)
        size_jitter_std = max(2.0, bbox_width * 0.1)

        dx = int(self.rng.normal(0, xy_jitter_std))
        dy = int(self.rng.normal(0, xy_jitter_std))
        dw = int(self.rng.normal(0, size_jitter_std))
        dh = int(self.rng.normal(0, size_jitter_std))

        xmin = max(0, xmin + dx - dw // 2)
        xmax = min(self.camera_cfg.width - 1, xmax + dx + dw // 2)
        ymin = max(0, ymin + dy - dh // 2)
        ymax = min(self.camera_cfg.height - 1, ymax + dy + dh // 2)

        return xmin, ymin, xmax, ymax

    def detect_to_dict(
        self,
        cloth: MassSpringCloth,
        vehicle: KinematicBicycle,
        veh_origin: np.ndarray,
        veh_R: np.ndarray,
    ) -> list[dict]:
        """Detect and return bboxes as serializable dicts."""
        bboxes = self.detect(cloth, vehicle, veh_origin, veh_R)
        return [
            {
                "class": bb.cls,
                "bbox": [bb.xmin, bb.ymin, bb.xmax, bb.ymax],
                "score": round(bb.score, 3),
            }
            for bb in bboxes
        ]
