"""Mass-spring cloth tests."""
from __future__ import annotations

import numpy as np

from oasis_sim_av.cloth import MassSpringCloth
from oasis_sim_av.config import TapeConfig


def _default_tape(**kw) -> TapeConfig:
    d = dict(
        anchor_a=[-5.0, 0.0, 2.0],
        anchor_b=[ 5.0, 0.0, 2.0],
        length=10.1,
        width=0.05,
        n_length=8,
        n_width=2,
        mass_per_particle=0.01,
        spring_k=500.0,
        spring_damping=1.0,
        global_damping=0.05,
        wind_bias=[0.0, 0.0, 0.0],
        wind_noise_std=0.0,
    )
    d.update(kw)
    return TapeConfig(**d)


def test_endpoints_remain_fixed():
    rng = np.random.default_rng(0)
    cloth = MassSpringCloth.from_config(_default_tape(), rng)
    anchor0 = cloth.positions[0].copy()
    anchor1 = cloth.positions[-1].copy()
    for _ in range(200):
        cloth.step(0.005)
    # End rows should not move
    np.testing.assert_allclose(cloth.positions[0], anchor0, atol=1e-9)
    np.testing.assert_allclose(cloth.positions[-1], anchor1, atol=1e-9)


def test_bounded_energy_with_damping():
    """Under gravity + damping + no wind, kinetic energy should NOT blow up.

    We record KE over 1 s and assert it's bounded by a generous constant.
    """
    rng = np.random.default_rng(0)
    cloth = MassSpringCloth.from_config(
        _default_tape(spring_damping=2.0, global_damping=0.2),
        rng,
    )
    max_ke = 0.0
    for _ in range(400):
        cloth.step(0.0025)
        max_ke = max(max_ke, cloth.kinetic_energy())
    assert np.isfinite(max_ke)
    assert max_ke < 1.0  # very loose bound; if unbounded this will blow past


def test_wind_injects_kinetic_energy():
    rng = np.random.default_rng(1)
    noisy = MassSpringCloth.from_config(
        _default_tape(wind_bias=[0, 2, 0], wind_noise_std=1.5, global_damping=0.02),
        rng,
    )
    quiet = MassSpringCloth.from_config(
        _default_tape(wind_bias=[0, 0, 0], wind_noise_std=0.0, global_damping=0.02),
        rng,
    )
    for _ in range(200):
        noisy.step(0.005)
        quiet.step(0.005)
    assert noisy.kinetic_energy() > quiet.kinetic_energy()


def test_triangle_emission_consistent():
    rng = np.random.default_rng(0)
    cloth = MassSpringCloth.from_config(_default_tape(n_length=5, n_width=2), rng)
    v0, v1, v2 = cloth.triangles()
    # (n_length-1) * (n_width-1) * 2 = 4 * 1 * 2 = 8 triangles
    assert v0.shape == (8, 3)
    assert v1.shape == (8, 3)
    assert v2.shape == (8, 3)
    # All triangles should have finite vertices and positive area
    e1 = v1 - v0
    e2 = v2 - v0
    area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
    assert np.all(area2 > 1e-9)
