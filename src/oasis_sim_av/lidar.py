"""Simulated LiDAR.

Ray-casts a spherical-sector sweep of primary rays against the scene and
returns a 3D point cloud plus metadata (ring index, azimuth index, return
kind — ground / building / tape / miss).  Applies Gaussian range noise
(representing rain / dust / ranging jitter) and a rain-dropout mask.

The default scenario is parameterised so the tape width (~5 cm) is smaller
than the vertical ring spacing at typical stand-off ranges — this reproduces
the real-world failure mode where a thin, fluttering ribbon slips *between*
LiDAR scan rings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LiDARConfig
from .geometry import nearest_hit
from .noise import apply_noise, dropout_mask


@dataclass
class LiDARScan:
    """Output of a single sweep."""

    points: np.ndarray       # (K, 3) world-space hits
    ranges: np.ndarray       # (K,) range in metres
    kind: np.ndarray         # (K,) 0=ground, 1=building, 2=tape
    ring: np.ndarray         # (K,) elevation ring index
    az: np.ndarray           # (K,) azimuth ray index
    origin: np.ndarray       # (3,) sensor origin this sweep
    n_rays: int              # total rays cast including misses
    n_tape_hits: int         # true tape returns (pre-noise-dropout)


@dataclass
class SimulatedLiDAR:
    """Parameterised spherical-sweep LiDAR with Gaussian noise + dropout."""

    offset: np.ndarray
    elevation_fov_deg: tuple[float, float]
    elevation_rings: int
    azimuth_fov_deg: tuple[float, float]
    azimuth_rays: int
    range_m: float
    range_noise_std_m: float
    rain_dropout_prob: float
    rng: np.random.Generator

    _ray_dirs_body: np.ndarray  # (N, 3) precomputed unit directions in body frame

    @classmethod
    def from_config(cls, cfg: LiDARConfig, rng: np.random.Generator) -> SimulatedLiDAR:
        az = np.deg2rad(
            np.linspace(cfg.azimuth_fov_deg[0], cfg.azimuth_fov_deg[1], cfg.azimuth_rays)
        )
        el = np.deg2rad(
            np.linspace(
                cfg.elevation_fov_deg[0], cfg.elevation_fov_deg[1], cfg.elevation_rings
            )
        )
        # Meshgrid: rings x azimuths
        EL, AZ = np.meshgrid(el, az, indexing="ij")
        # Body frame: x forward, y left, z up
        cx = np.cos(EL) * np.cos(AZ)
        cy = np.cos(EL) * np.sin(AZ)
        cz = np.sin(EL)
        dirs = np.stack([cx, cy, cz], axis=-1).reshape(-1, 3)
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        return cls(
            offset=np.asarray(cfg.offset, dtype=np.float64),
            elevation_fov_deg=tuple(cfg.elevation_fov_deg),
            elevation_rings=int(cfg.elevation_rings),
            azimuth_fov_deg=tuple(cfg.azimuth_fov_deg),
            azimuth_rays=int(cfg.azimuth_rays),
            range_m=float(cfg.range_m),
            range_noise_std_m=float(cfg.range_noise_std_m),
            rain_dropout_prob=float(cfg.rain_dropout_prob),
            rng=rng,
            _ray_dirs_body=dirs,
        )

    # ------------------------------------------------------------------
    def scan(
        self,
        vehicle_origin: np.ndarray,
        body_to_world: np.ndarray,
        boxes_min: np.ndarray,
        boxes_max: np.ndarray,
        tri_v0: np.ndarray,
        tri_v1: np.ndarray,
        tri_v2: np.ndarray,
        ground_z: float = 0.0,
    ) -> LiDARScan:
        """Perform one full sweep.  All geometry arguments are in world coords."""
        origin = vehicle_origin + body_to_world @ self.offset
        # Rotate ray directions into world frame
        dirs_world = self._ray_dirs_body @ body_to_world.T
        origins = np.broadcast_to(origin, dirs_world.shape).copy()

        t, kind = nearest_hit(
            origins, dirs_world,
            boxes_min, boxes_max,
            tri_v0, tri_v1, tri_v2,
            ground_z=ground_z,
        )

        # Clip to max range (treat further as misses)
        in_range = np.isfinite(t) & (t <= self.range_m)
        t_clean = np.where(in_range, t, np.inf)

        # Count tape hits before noise-based dropouts (for diagnostics)
        n_tape_hits_true = int(np.sum((kind == 2) & in_range))

        # Apply Gaussian range noise + rain dropout to the returning rays
        keep = dropout_mask(in_range.shape, self.rain_dropout_prob, self.rng) & in_range
        t_out = np.where(keep, t_clean, np.inf)
        if self.range_noise_std_m > 0.0:
            noisy = apply_noise(
                t_out, "gaussian", sigma=self.range_noise_std_m, rng=self.rng
            )
            t_out = np.where(keep, noisy, np.inf)

        hit_idx = np.where(np.isfinite(t_out))[0]
        points = origins[hit_idx] + dirs_world[hit_idx] * t_out[hit_idx, None]

        # Reconstruct (ring, az) indices from the flat layout
        ring_idx = hit_idx // self.azimuth_rays
        az_idx = hit_idx % self.azimuth_rays

        return LiDARScan(
            points=points,
            ranges=t_out[hit_idx],
            kind=kind[hit_idx].astype(np.int8),
            ring=ring_idx.astype(np.int32),
            az=az_idx.astype(np.int32),
            origin=origin,
            n_rays=int(dirs_world.shape[0]),
            n_tape_hits=n_tape_hits_true,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def write_npz(
        path: str,
        points: np.ndarray,
        kind: np.ndarray,
        ranges: np.ndarray | None = None,
        origin: np.ndarray | None = None,
    ) -> None:
        """Persist scan arrays to a compressed .npz sidecar.

        Used by the multi-view renderer, which needs lossless access to the
        per-point ``kind`` field (the ASCII .ply writer below reserves colour,
        not kind, so reverse-mapping is lossy once rain points at kind=3 are
        added — their colour can clash with degraded-yellow tape returns).
        """
        payload: dict[str, np.ndarray] = {
            "points": np.asarray(points, dtype=np.float32),
            "kind": np.asarray(kind, dtype=np.int8),
        }
        if ranges is not None:
            payload["ranges"] = np.asarray(ranges, dtype=np.float32)
        if origin is not None:
            payload["origin"] = np.asarray(origin, dtype=np.float32)
        np.savez_compressed(path, **payload)

    # ------------------------------------------------------------------
    def write_ply(self, scan: LiDARScan, path: str) -> None:
        """Write an ASCII .ply file.  Colour-encodes ``kind``."""
        color_lut = np.array(
            [
                [80, 95, 75],      # 0 ground
                [140, 140, 150],   # 1 building
                [240, 220, 40],    # 2 tape
            ],
            dtype=np.uint8,
        )
        pts = scan.points
        k = np.clip(scan.kind, 0, 2)
        col = color_lut[k]
        n = pts.shape[0]
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(n):
                f.write(
                    f"{pts[i,0]:.4f} {pts[i,1]:.4f} {pts[i,2]:.4f} "
                    f"{int(col[i,0])} {int(col[i,1])} {int(col[i,2])}\n"
                )
