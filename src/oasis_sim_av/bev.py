"""Bird's-eye-view (BEV) orthographic renderer.

Produces a top-down view of the scene from a fixed world position. Used in
the multi-view demo grid to provide context for LiDAR and detections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BEVConfig
from .geometry import nearest_hit
from .world import World


@dataclass
class BEVRenderer:
    """World-fixed orthographic top-down renderer."""

    center: np.ndarray
    extent_m: float
    size_px: int
    show_vehicle_marker: bool
    show_road: bool

    @classmethod
    def from_config(cls, cfg: BEVConfig) -> BEVRenderer:
        return cls(
            center=np.asarray(cfg.center, dtype=np.float64),
            extent_m=float(cfg.extent_m),
            size_px=int(cfg.size_px),
            show_vehicle_marker=bool(cfg.show_vehicle_marker),
            show_road=bool(cfg.show_road),
        )

    def render(
        self,
        world: World,
        tri_v0: np.ndarray,
        tri_v1: np.ndarray,
        tri_v2: np.ndarray,
        vehicle_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render a top-down orthographic view.

        Returns (H, W, 3) uint8 image.
        """
        H = W = self.size_px
        half = self.extent_m / 2.0
        cx, cy = self.center

        xs = np.linspace(cx - half, cx + half, W)
        ys = np.linspace(cy - half, cy + half, H)

        X, Y = np.meshgrid(xs, ys)

        origins = np.stack([
            X.ravel(),
            Y.ravel(),
            np.full(H * W, 100.0),
        ], axis=1)

        directions = np.broadcast_to(
            np.array([0.0, 0.0, -1.0]), (H * W, 3)
        ).copy()

        t, kind = nearest_hit(
            origins, directions,
            world.boxes_min, world.boxes_max,
            tri_v0, tri_v1, tri_v2,
            ground_z=world.ground_z,
        )

        img = np.zeros((H * W, 3), dtype=np.float64)

        sky = kind == -1
        img[sky] = [0.55, 0.65, 0.75]

        ground = kind == 0
        if ground.any():
            hit_pts = origins[ground] + directions[ground] * t[ground, None]
            base = np.where(
                world.point_on_road(hit_pts[:, :2])[:, None],
                world.road_color(),
                world.ground_color(),
            )
            img[ground] = base

        building = kind == 1
        if building.any():
            img[building] = world.building_color()

        tape = kind == 2
        if tape.any():
            img[tape] = world.tape_color()

        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = img.reshape(H, W, 3)

        if self.show_vehicle_marker and vehicle_state is not None:
            x, y, th = vehicle_state[:3]
            px = int((x - (cx - half)) / self.extent_m * W)
            py = int((y - (cy - half)) / self.extent_m * H)
            if 0 <= px < W and 0 <= py < H:
                self._draw_vehicle_marker(img, px, py, th)

        return img

    def _draw_vehicle_marker(
        self, img: np.ndarray, px: int, py: int, theta: float
    ) -> None:
        """Draw a simple vehicle marker (arrow) at the given pixel."""
        H, W = img.shape[:2]
        length = 8
        width = 4

        dx = np.cos(theta)
        dy = -np.sin(theta)

        for d in range(-length // 2, length // 2 + 1):
            for w in range(-width // 2, width // 2 + 1):
                ix = int(px + d * dx + w * dy)
                iy = int(py + d * (-dy) + w * dx)
                if 0 <= ix < W and 0 <= iy < H:
                    img[iy, ix] = [255, 80, 80]
