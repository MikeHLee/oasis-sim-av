"""Static world geometry: buildings (AABBs), ground plane, road polygons.

Deliberately primitive: per the brief, the city is built only from 3D rectangles
(buildings) and 2D squares (lanes, sidewalks).  That keeps ray-scene
intersection closed-form and cheap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import WorldConfig


@dataclass
class World:
    """Container for the static part of the scene."""

    boxes_min: np.ndarray  # (M, 3)
    boxes_max: np.ndarray  # (M, 3)
    ground_z: float
    roads: list[np.ndarray]  # list of (K, 2) arrays defining road polygons

    @classmethod
    def from_config(cls, cfg: WorldConfig) -> World:
        if cfg.buildings:
            arr = np.asarray([b.aabb for b in cfg.buildings], dtype=np.float64)
            boxes_min = arr[:, :3].copy()
            boxes_max = arr[:, 3:].copy()
        else:
            boxes_min = np.zeros((0, 3))
            boxes_max = np.zeros((0, 3))
        roads = [np.asarray(p, dtype=np.float64) for p in cfg.roads]
        return cls(
            boxes_min=boxes_min,
            boxes_max=boxes_max,
            ground_z=cfg.ground_z,
            roads=roads,
        )

    # ------------------------------------------------------------------
    # Shading helpers used by the camera
    # ------------------------------------------------------------------
    @staticmethod
    def building_color() -> np.ndarray:
        return np.array([0.55, 0.55, 0.58])

    @staticmethod
    def road_color() -> np.ndarray:
        return np.array([0.15, 0.15, 0.16])

    @staticmethod
    def ground_color() -> np.ndarray:
        return np.array([0.35, 0.38, 0.32])

    @staticmethod
    def tape_color() -> np.ndarray:
        # Crime-scene yellow
        return np.array([0.95, 0.85, 0.15])

    @staticmethod
    def sky_color(ray_dirs: np.ndarray) -> np.ndarray:
        """Simple gradient sky from horizon (pale) to zenith (blue)."""
        z = np.clip(ray_dirs[:, 2], 0.0, 1.0)
        horizon = np.array([0.75, 0.80, 0.85])
        zenith = np.array([0.20, 0.45, 0.80])
        return horizon[None, :] * (1.0 - z[:, None]) + zenith[None, :] * z[:, None]

    def point_on_road(self, xy: np.ndarray) -> np.ndarray:
        """Boolean mask for points that lie inside any road polygon (in xy plane).

        Uses the even-odd rule; the MVP's roads are convex so this is exact.
        """
        if not self.roads:
            return np.zeros(len(xy), dtype=bool)
        out = np.zeros(len(xy), dtype=bool)
        for poly in self.roads:
            out |= _point_in_poly(xy, poly)
        return out


def _point_in_poly(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized even-odd point-in-polygon test."""
    n = len(poly)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= cond
        j = i
    return inside
