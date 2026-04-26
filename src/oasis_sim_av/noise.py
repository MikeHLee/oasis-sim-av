"""Noise injectors.

Ported and generalised from ``oasis-firmware/simulation/behavioral/runtime.py:227-238``
(the only reusable piece identified in the pre-build audit).  The original was a
scalar helper; here we vectorise it so it can be applied to whole arrays of
LiDAR returns or pixel radiance values in one call.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

NoiseKind = Literal["gaussian", "uniform", "drift", "none"]


def apply_noise(
    values: np.ndarray,
    kind: NoiseKind,
    *,
    sigma: float = 0.0,
    low: float = 0.0,
    high: float = 0.0,
    drift_rate: float = 0.0,
    elapsed_s: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply an in-place-safe noise perturbation to ``values``.

    Parameters
    ----------
    values
        Array of any shape; returned array has the same shape.
    kind
        One of ``"gaussian"``, ``"uniform"``, ``"drift"``, ``"none"``.
    sigma
        Standard deviation for ``"gaussian"``.
    low, high
        Bounds for ``"uniform"`` (added, not replaced).
    drift_rate
        Units-per-second constant drift for ``"drift"``.
    elapsed_s
        Seconds elapsed — multiplied by ``drift_rate`` for the drift term.
    rng
        A ``numpy.random.Generator``. A fresh default-seeded one is used if None.
    """
    if kind == "none":
        return values
    rng = rng if rng is not None else np.random.default_rng()
    if kind == "gaussian":
        return values + rng.normal(0.0, sigma, size=values.shape)
    if kind == "uniform":
        return values + rng.uniform(low, high, size=values.shape)
    if kind == "drift":
        return values + drift_rate * elapsed_s
    raise ValueError(f"Unknown noise kind: {kind!r}")


def dropout_mask(shape: tuple[int, ...], prob: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask where True means *keep* the sample.

    Used to model rain/dust knocking out a fraction of LiDAR returns.
    """
    if prob <= 0.0:
        return np.ones(shape, dtype=bool)
    if prob >= 1.0:
        return np.zeros(shape, dtype=bool)
    return rng.random(size=shape) >= prob
