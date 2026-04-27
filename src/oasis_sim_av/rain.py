"""Visual-only rain clutter field for LiDAR.

Advected droplet field that produces rain-like returns in LiDAR scans.
These points are ONLY used for visualization (BEV, fused panels) and are
NEVER written to .ply files or fed to fusion.py — this preserves the
baseline fusion numbers and test expectations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RainClutterConfig
from .geometry import ray_aabb_batch


@dataclass
class RainField:
    """Advected droplet field for visual-only LiDAR rain clutter."""

    positions: np.ndarray
    velocities: np.ndarray
    spawn_box: np.ndarray
    fall_velocity_m_s: float
    jitter_std_m_s: float
    droplet_radius_m: float
    ground_z: float
    rng: np.random.Generator

    @classmethod
    def from_config(
        cls,
        cfg: RainClutterConfig,
        ground_z: float,
        rng: np.random.Generator,
    ) -> RainField:
        spawn_box = np.asarray(cfg.spawn_box, dtype=np.float64)

        n = cfg.n_droplets
        positions = np.zeros((n, 3), dtype=np.float64)
        positions[:, 0] = rng.uniform(spawn_box[0], spawn_box[3], n)
        positions[:, 1] = rng.uniform(spawn_box[1], spawn_box[4], n)
        positions[:, 2] = rng.uniform(spawn_box[2], spawn_box[5], n)

        velocities = np.zeros((n, 3), dtype=np.float64)
        velocities[:, 2] = -cfg.fall_velocity_m_s

        return cls(
            positions=positions,
            velocities=velocities,
            spawn_box=spawn_box,
            fall_velocity_m_s=cfg.fall_velocity_m_s,
            jitter_std_m_s=cfg.jitter_std_m_s,
            droplet_radius_m=cfg.droplet_radius_m,
            ground_z=ground_z,
            rng=rng,
        )

    def step(self, dt: float) -> None:
        """Advect droplets downward and recycle those below ground."""
        self.positions += self.velocities * dt

        if self.jitter_std_m_s > 0.0:
            jitter = self.rng.normal(0, self.jitter_std_m_s, self.positions.shape)
            self.positions += jitter * dt

        below = self.positions[:, 2] < self.ground_z
        if below.any():
            n_below = below.sum()
            self.positions[below, 0] = self.rng.uniform(
                self.spawn_box[0], self.spawn_box[3], n_below
            )
            self.positions[below, 1] = self.rng.uniform(
                self.spawn_box[1], self.spawn_box[4], n_below
            )
            self.positions[below, 2] = self.spawn_box[5]

    def compute_clutter_hits(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute which rays hit droplets.

        Parameters
        ----------
        origins : (N, 3) array
            Ray origins (LiDAR sensor position, one per ray).
        directions : (N, 3) array
            Ray directions.

        Returns
        -------
        hit_mask : (N,) bool array
            True for rays that hit a droplet.
        t_values : (N,) float array
            Hit distances (inf for misses).
        """
        n_rays = origins.shape[0]
        n_drops = self.positions.shape[0]

        if n_drops == 0:
            return np.zeros(n_rays, dtype=bool), np.full(n_rays, np.inf)

        r = self.droplet_radius_m
        droplet_boxes_min = self.positions - r
        droplet_boxes_max = self.positions + r

        t_all = np.full((n_rays, n_drops), np.inf, dtype=np.float64)

        for j in range(n_drops):
            t_all[:, j] = ray_aabb_batch(
                origins, directions,
                droplet_boxes_min[j], droplet_boxes_max[j]
            )

        t_nearest = t_all.min(axis=1)
        hit_mask = np.isfinite(t_nearest) & (t_nearest > 0)

        return hit_mask, t_nearest
