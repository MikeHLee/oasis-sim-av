"""Vectorized ray intersection routines.

All functions operate on batches of rays at once using NumPy.  The cost model
is: one batched call per scene primitive (box / triangle set).  For the MVP
this is more than sufficient: a 320x240 camera frame against ~10 buildings and
~80 tape triangles evaluates in well under a second on a laptop CPU.

Conventions
-----------
* ``origins``: ``(N, 3)`` float64 array of ray start points.
* ``directions``: ``(N, 3)`` float64 array of **unit-length** ray directions.
* ``t``: ray parameter.  Returned arrays use ``+inf`` to denote "no hit".
"""
from __future__ import annotations

import numpy as np

EPS = 1e-8  # parallel-ray epsilon for ray-triangle
_INF = np.float64(np.inf)


# ----------------------------------------------------------------------------
# Ray - axis-aligned bounding box (slab test), vectorized
# ----------------------------------------------------------------------------
def ray_aabb_batch(
    origins: np.ndarray,
    directions: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> np.ndarray:
    """Intersect ``N`` rays against a single AABB.

    Parameters
    ----------
    origins, directions
        ``(N, 3)`` float arrays; ``directions`` should be unit length for ``t``
        to be in world units, but any non-zero direction works.
    box_min, box_max
        ``(3,)`` arrays of the box corners (inclusive).

    Returns
    -------
    t : ``(N,)`` float array.  ``+inf`` for no hit or a hit strictly behind
        the ray origin.
    """
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    box_min = np.asarray(box_min, dtype=np.float64)
    box_max = np.asarray(box_max, dtype=np.float64)

    # Guard against zero components in directions so the divide is defined; the
    # slab test still returns correct results for rays parallel to an axis.
    safe_dir = np.where(np.abs(directions) < EPS, EPS, directions)
    inv = 1.0 / safe_dir

    t1 = (box_min - origins) * inv
    t2 = (box_max - origins) * inv
    t_small = np.minimum(t1, t2)
    t_large = np.maximum(t1, t2)

    t_enter = np.max(t_small, axis=1)
    t_exit = np.min(t_large, axis=1)

    # Miss or behind-origin
    hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit >= 0.0)
    # Distance to enter. If origin is inside the box, t_enter is negative;
    # in that case the first visible surface is the exit plane.
    t = np.where(t_enter >= 0.0, t_enter, t_exit)
    t = np.where(hit, t, _INF)
    return t


def ray_aabb_many(
    origins: np.ndarray,
    directions: np.ndarray,
    boxes_min: np.ndarray,
    boxes_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect ``N`` rays against ``M`` AABBs.

    Returns
    -------
    t_nearest : ``(N,)`` nearest ``t`` across all boxes, ``+inf`` on miss.
    box_index : ``(N,)`` int index of the hit box (``-1`` on miss).
    """
    n = origins.shape[0]
    m = boxes_min.shape[0]
    if m == 0:
        return np.full(n, _INF), np.full(n, -1, dtype=np.int64)
    t_all = np.full((n, m), _INF)
    for j in range(m):
        t_all[:, j] = ray_aabb_batch(origins, directions, boxes_min[j], boxes_max[j])
    idx = np.argmin(t_all, axis=1)
    t = t_all[np.arange(n), idx]
    idx = np.where(np.isfinite(t), idx, -1)
    return t, idx


# ----------------------------------------------------------------------------
# Ray - triangle (Möller-Trumbore), vectorized
# ----------------------------------------------------------------------------
def ray_triangle_batch(
    origins: np.ndarray,
    directions: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    *,
    backface_cull: bool = False,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect ``N`` rays against ``M`` triangles.

    Returns
    -------
    t_nearest : ``(N,)`` nearest ``t``, ``+inf`` on miss.
    tri_index : ``(N,)`` int index of the hit triangle (``-1`` on miss).

    Notes
    -----
    This is the standard Möller-Trumbore algorithm.  The ``eps`` guard on the
    determinant implements the brief's "epsilon distance check to handle rays
    that are perfectly parallel to the cloth planes" requirement.
    """
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    n = origins.shape[0]
    m = v0.shape[0]
    if m == 0:
        return np.full(n, _INF), np.full(n, -1, dtype=np.int64)

    e1 = v1 - v0  # (M, 3)
    e2 = v2 - v0  # (M, 3)

    # Broadcast: rays on axis 0, triangles on axis 1
    D = directions[:, None, :]            # (N, 1, 3)
    O = origins[:, None, :]                # (N, 1, 3)
    E1 = e1[None, :, :]                    # (1, M, 3)
    E2 = e2[None, :, :]                    # (1, M, 3)
    V0 = v0[None, :, :]                    # (1, M, 3)

    pvec = np.cross(D, E2)                 # (N, M, 3)
    det = np.sum(E1 * pvec, axis=2)        # (N, M)

    if backface_cull:
        valid_det = det > eps
    else:
        valid_det = np.abs(det) > eps

    inv_det = np.where(valid_det, 1.0 / np.where(valid_det, det, 1.0), 0.0)
    tvec = O - V0                          # (N, M, 3)
    u = np.sum(tvec * pvec, axis=2) * inv_det

    qvec = np.cross(tvec, E1)              # (N, M, 3)
    v = np.sum(D * qvec, axis=2) * inv_det

    t = np.sum(E2 * qvec, axis=2) * inv_det

    hit = valid_det & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t > eps)
    t = np.where(hit, t, _INF)

    idx = np.argmin(t, axis=1)
    t_near = t[np.arange(n), idx]
    idx = np.where(np.isfinite(t_near), idx, -1)
    return t_near, idx


# ----------------------------------------------------------------------------
# Ray - ground plane (z = z0)
# ----------------------------------------------------------------------------
def ray_ground(
    origins: np.ndarray,
    directions: np.ndarray,
    z0: float = 0.0,
) -> np.ndarray:
    """Return ``t`` for ``N`` rays intersecting the horizontal plane ``z = z0``.

    ``+inf`` for rays that point upward from above (or downward from below).
    """
    dz = directions[:, 2]
    t = (z0 - origins[:, 2]) / np.where(np.abs(dz) < EPS, EPS, dz)
    hit = (t > 0.0) & (np.abs(dz) > EPS)
    return np.where(hit, t, _INF)


# ----------------------------------------------------------------------------
# Composite: nearest intersection with all scene primitives
# ----------------------------------------------------------------------------
def nearest_hit(
    origins: np.ndarray,
    directions: np.ndarray,
    boxes_min: np.ndarray,
    boxes_max: np.ndarray,
    tri_v0: np.ndarray,
    tri_v1: np.ndarray,
    tri_v2: np.ndarray,
    ground_z: float | None = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest ``t`` across boxes / triangles / ground plane.

    Returns
    -------
    t : ``(N,)``
    kind : ``(N,)`` int array.  ``0`` = ground, ``1`` = building (AABB),
        ``2`` = tape triangle, ``-1`` = miss.
    """
    n = origins.shape[0]
    t_best = np.full(n, _INF)
    kind = np.full(n, -1, dtype=np.int8)

    if ground_z is not None:
        t_g = ray_ground(origins, directions, ground_z)
        better = t_g < t_best
        t_best = np.where(better, t_g, t_best)
        kind = np.where(better, 0, kind)

    if boxes_min is not None and boxes_min.size > 0:
        t_b, _ = ray_aabb_many(origins, directions, boxes_min, boxes_max)
        better = t_b < t_best
        t_best = np.where(better, t_b, t_best)
        kind = np.where(better, 1, kind)

    if tri_v0 is not None and tri_v0.size > 0:
        t_t, _ = ray_triangle_batch(origins, directions, tri_v0, tri_v1, tri_v2)
        better = t_t < t_best
        t_best = np.where(better, t_t, t_best)
        kind = np.where(better, 2, kind)

    return t_best, kind
