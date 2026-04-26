"""Pinhole eye-tracing camera.

Renders an RGB frame by casting one primary ray per pixel (plus optional
temporal sub-samples for motion blur) from the camera origin through the
image plane into the scene, then shading the nearest hit with a simple
Lambert + ambient + sky model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CameraConfig
from .geometry import nearest_hit
from .world import World


LIGHT_DIR = np.array([-0.4, -0.3, 1.0])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
LIGHT_COLOR = np.array([1.0, 0.97, 0.9])
AMBIENT = 0.22


@dataclass
class PinholeCamera:
    """Axis-aligned pinhole camera anchored to the vehicle."""

    offset: np.ndarray          # body-frame offset (3,)
    forward: np.ndarray         # body-frame forward (3,) unit
    up: np.ndarray              # body-frame up (3,) unit
    fov_h_deg: float
    width: int
    height: int
    motion_blur_samples: int
    exposure_s: float
    shadow_rays: bool = False

    @classmethod
    def from_config(cls, cfg: CameraConfig) -> PinholeCamera:
        f = np.asarray(cfg.forward, dtype=np.float64)
        u = np.asarray(cfg.up, dtype=np.float64)
        f = f / (np.linalg.norm(f) + 1e-12)
        u = u / (np.linalg.norm(u) + 1e-12)
        return cls(
            offset=np.asarray(cfg.offset, dtype=np.float64),
            forward=f,
            up=u,
            fov_h_deg=float(cfg.fov_h_deg),
            width=int(cfg.width),
            height=int(cfg.height),
            motion_blur_samples=max(1, int(cfg.motion_blur_samples)),
            exposure_s=float(cfg.exposure_s),
            shadow_rays=bool(getattr(cfg, "shadow_rays", False)),
        )

    # ------------------------------------------------------------------
    def _primary_rays(
        self,
        vehicle_origin: np.ndarray,
        body_to_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(origins, directions)`` for one ray per pixel, row-major."""
        W, H = self.width, self.height
        fx = 1.0 / np.tan(np.deg2rad(self.fov_h_deg) * 0.5)
        aspect = W / H
        # Image plane NDC: x in [-1, 1], y in [-1/aspect, 1/aspect] scaled by fx
        xs = np.linspace(-1.0, 1.0, W) / fx
        ys = np.linspace(1.0, -1.0, H) / (fx * aspect)
        X, Y = np.meshgrid(xs, ys)
        # Body-frame camera: forward = +x, right = -y_body (so world-left = y>0)
        # Build orthonormal basis in body frame
        fwd = self.forward
        up = self.up
        right = np.cross(fwd, up)
        right = right / (np.linalg.norm(right) + 1e-12)
        up = np.cross(right, fwd)
        dirs_body = (
            fwd[None, None, :]
            + X[..., None] * right[None, None, :]
            + Y[..., None] * up[None, None, :]
        )
        dirs_body = dirs_body.reshape(-1, 3)
        dirs_body = dirs_body / np.linalg.norm(dirs_body, axis=1, keepdims=True)
        # Rotate into world
        dirs_world = dirs_body @ body_to_world.T
        origin = vehicle_origin + body_to_world @ self.offset
        origins = np.broadcast_to(origin, dirs_world.shape).copy()
        return origins, dirs_world

    # ------------------------------------------------------------------
    def render(
        self,
        vehicle_origin: np.ndarray,
        body_to_world: np.ndarray,
        world: World,
        tri_v0: np.ndarray,
        tri_v1: np.ndarray,
        tri_v2: np.ndarray,
    ) -> np.ndarray:
        """Render one frame (no motion blur).  Returns ``(H, W, 3)`` uint8."""
        origins, dirs = self._primary_rays(vehicle_origin, body_to_world)
        t, kind = nearest_hit(
            origins, dirs,
            world.boxes_min, world.boxes_max,
            tri_v0, tri_v1, tri_v2,
            ground_z=world.ground_z,
        )
        shadow_mask = None
        if self.shadow_rays:
            shadow_mask = _shadow_mask(
                origins, dirs, t, kind,
                world.boxes_min, world.boxes_max,
                tri_v0, tri_v1, tri_v2,
                ground_z=world.ground_z,
            )
        return _shade(
            origins, dirs, t, kind, world, shadow_mask=shadow_mask,
        ).reshape(self.height, self.width, 3)

    # ------------------------------------------------------------------
    def render_with_motion_blur(
        self,
        vehicle_origin_fn,  # callable t -> origin(3,)
        body_to_world_fn,   # callable t -> 3x3
        world: World,
        cloth_snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Render one frame by averaging over ``len(cloth_snapshots)`` sub-samples.

        Each sub-sample uses a different cloth triangle snapshot (taken at
        evenly spaced times within the exposure window).  ``vehicle_origin_fn``
        and ``body_to_world_fn`` receive a float in ``[0, 1]`` denoting
        fractional position within the exposure window.
        """
        N = len(cloth_snapshots)
        if N == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        acc = np.zeros((self.height * self.width, 3), dtype=np.float64)
        for k in range(N):
            frac = (k + 0.5) / N
            o = vehicle_origin_fn(frac)
            R = body_to_world_fn(frac)
            origins, dirs = self._primary_rays(o, R)
            v0, v1, v2 = cloth_snapshots[k]
            t, kind = nearest_hit(
                origins, dirs,
                world.boxes_min, world.boxes_max,
                v0, v1, v2,
                ground_z=world.ground_z,
            )
            shadow_mask = None
            if self.shadow_rays:
                shadow_mask = _shadow_mask(
                    origins, dirs, t, kind,
                    world.boxes_min, world.boxes_max,
                    v0, v1, v2,
                    ground_z=world.ground_z,
                )
            img = _shade_float(
                origins, dirs, t, kind, world, shadow_mask=shadow_mask,
            )
            acc += img
        acc /= N
        img8 = np.clip(acc * 255.0, 0.0, 255.0).astype(np.uint8)
        return img8.reshape(self.height, self.width, 3)


# ---------------------------------------------------------------------------
# Shading
# ---------------------------------------------------------------------------
def _shade_float(
    origins: np.ndarray,
    dirs: np.ndarray,
    t: np.ndarray,
    kind: np.ndarray,
    world: World,
    shadow_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return HxW*3 float image in [0, 1].  If ``shadow_mask`` is provided,
    it is a boolean array the same length as ``origins`` with ``True`` at
    hits that are occluded from the directional light — those pixels drop
    to ambient-only shading.
    """
    n = origins.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)

    # Sky for misses
    miss = kind == -1
    if miss.any():
        out[miss] = world.sky_color(dirs[miss])

    in_shadow = shadow_mask if shadow_mask is not None else np.zeros(n, dtype=bool)

    # Buildings
    building = kind == 1
    if building.any():
        normal = _box_ish_normal(dirs[building])
        out[building] = _lambert(
            world.building_color(), dirs[building], normal=normal,
            in_shadow=in_shadow[building],
        )

    # Ground
    ground = kind == 0
    if ground.any():
        hit_points = origins[ground] + dirs[ground] * t[ground, None]
        base = np.where(
            world.point_on_road(hit_points[:, :2])[:, None],
            world.road_color(),
            world.ground_color(),
        )
        normal = np.tile(np.array([0.0, 0.0, 1.0]), (ground.sum(), 1))
        out[ground] = _lambert_many(
            base, dirs[ground], normal, in_shadow=in_shadow[ground],
        )

    # Tape (cloth)
    tape = kind == 2
    if tape.any():
        d = dirs[tape]
        ndotl = np.clip(np.abs(np.sum(d * LIGHT_DIR[None, :], axis=1)), 0.0, 1.0)
        shade = AMBIENT + (1.0 - AMBIENT) * ndotl
        # Tape is translucent, so we don't zero-out shadowed tape; instead
        # clamp to the darker end of its shade range.
        shade = np.where(in_shadow[tape], AMBIENT, shade)
        out[tape] = world.tape_color()[None, :] * shade[:, None]

    return out


def _shade(origins, dirs, t, kind, world, shadow_mask=None) -> np.ndarray:
    img = _shade_float(origins, dirs, t, kind, world, shadow_mask=shadow_mask)
    return np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)


def _lambert(
    base_color: np.ndarray,
    ray_dirs: np.ndarray,
    normal: np.ndarray,
    in_shadow: np.ndarray | None = None,
) -> np.ndarray:
    ndotl = np.clip(np.sum(normal * LIGHT_DIR[None, :], axis=1), 0.0, 1.0)
    shade = AMBIENT + (1.0 - AMBIENT) * ndotl
    if in_shadow is not None:
        shade = np.where(in_shadow, AMBIENT, shade)
    return base_color[None, :] * shade[:, None]


def _lambert_many(
    base_colors: np.ndarray,
    ray_dirs: np.ndarray,
    normal: np.ndarray,
    in_shadow: np.ndarray | None = None,
) -> np.ndarray:
    ndotl = np.clip(np.sum(normal * LIGHT_DIR[None, :], axis=1), 0.0, 1.0)
    shade = AMBIENT + (1.0 - AMBIENT) * ndotl
    if in_shadow is not None:
        shade = np.where(in_shadow, AMBIENT, shade)
    return base_colors * shade[:, None]


def _shadow_mask(
    origins: np.ndarray,
    dirs: np.ndarray,
    t: np.ndarray,
    kind: np.ndarray,
    boxes_min: np.ndarray,
    boxes_max: np.ndarray,
    tri_v0: np.ndarray,
    tri_v1: np.ndarray,
    tri_v2: np.ndarray,
    ground_z: float = 0.0,
) -> np.ndarray:
    """Cast one secondary ray per hit toward the directional light, return a
    boolean mask (True = pixel is occluded by some other geometry).

    Only rays that hit something (``kind != -1``) get a shadow test; sky-ray
    entries are always False.
    """
    n = origins.shape[0]
    mask = np.zeros(n, dtype=bool)
    hits = kind != -1
    if not hits.any():
        return mask
    idx = np.where(hits)[0]
    hit_points = origins[idx] + dirs[idx] * t[idx, None]
    # Nudge origin a tiny bit toward the light to avoid self-intersection
    eps = 1e-3
    shadow_origins = hit_points + eps * LIGHT_DIR[None, :]
    shadow_dirs = np.broadcast_to(LIGHT_DIR, shadow_origins.shape).copy()
    t_shadow, k_shadow = nearest_hit(
        shadow_origins, shadow_dirs,
        boxes_min, boxes_max,
        tri_v0, tri_v1, tri_v2,
        ground_z=ground_z,
    )
    # "Light" is at infinity along LIGHT_DIR; anything between the surface
    # and infinity shadows us.  We exclude ground_z self-shadowing by
    # requiring the occluder kind to be building or tape.
    occluded = np.isfinite(t_shadow) & (t_shadow > 0.0) & (
        (k_shadow == 1) | (k_shadow == 2)
    )
    mask[idx] = occluded
    return mask


def _box_ish_normal(ray_dirs: np.ndarray) -> np.ndarray:
    """Cheap approximation: assume building faces are roughly perpendicular to
    the viewing direction, using the axis closest to the ray.  Good enough for
    MVP renders where faces are axis aligned anyway.
    """
    ad = np.abs(ray_dirs)
    axis = np.argmax(ad, axis=1)
    normal = np.zeros_like(ray_dirs)
    # Face's normal opposes the ray direction along the dominant axis
    for a in range(3):
        m = axis == a
        if m.any():
            s = -np.sign(ray_dirs[m, a])
            normal[m, a] = s
    return normal
