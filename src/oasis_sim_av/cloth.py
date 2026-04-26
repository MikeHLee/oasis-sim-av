"""Mass-spring cloth for the fluttering police tape.

Brief requirements implemented
------------------------------
* 3D grid of mass particles connected by springs
* Symplectic Euler time integration (position-then-velocity update with the
  force evaluated at the new velocity -> energy-bounded under damping)
* Configurable particle mass, spring constant, damping, external wind force
* Structural springs between 4-neighbours plus shear (diagonal) springs for
  bend stiffness, so the tape holds a recognisable ribbon shape while
  flickering under wind.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TapeConfig
from .noise import apply_noise

GRAVITY = np.array([0.0, 0.0, -9.81])


@dataclass
class MassSpringCloth:
    """A rectangular grid of point masses connected by springs.

    The grid has ``n_rows`` particles along the length (anchor_a -> anchor_b
    direction) and ``n_cols`` particles across the width.  The two end rows are
    fixed to the anchor points (Dirichlet boundary); the rest are free.
    """

    # Particle state
    positions: np.ndarray      # (R, C, 3)
    velocities: np.ndarray     # (R, C, 3)
    fixed: np.ndarray          # (R, C) bool
    mass: float

    # Springs: list of (i0, j0, i1, j1, rest_length)
    springs: np.ndarray        # (S, 5) float -> stored as float for numpy math
    spring_k: float
    spring_damping: float
    global_damping: float

    # Wind
    wind_bias: np.ndarray      # (3,)
    wind_noise_std: float

    rng: np.random.Generator

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: TapeConfig, rng: np.random.Generator) -> MassSpringCloth:
        a = np.asarray(cfg.anchor_a, dtype=np.float64)
        b = np.asarray(cfg.anchor_b, dtype=np.float64)
        along = b - a
        L_straight = float(np.linalg.norm(along))
        along_unit = along / (L_straight + 1e-9)
        # A perpendicular in the horizontal plane for the "width" axis
        world_up = np.array([0.0, 0.0, 1.0])
        perp = np.cross(along_unit, world_up)
        if np.linalg.norm(perp) < 1e-6:
            perp = np.array([0.0, 1.0, 0.0])
        perp = perp / np.linalg.norm(perp)

        R = int(cfg.n_length)
        C = int(cfg.n_width)
        # Positions: distribute along the straight line first; physics will
        # sag it into the natural droop during the first few timesteps.
        s = np.linspace(0.0, 1.0, R)
        w = np.linspace(-cfg.width * 0.5, cfg.width * 0.5, C) if C > 1 else np.zeros(1)
        positions = np.empty((R, C, 3))
        for i in range(R):
            base = a + along * s[i]
            # Encode the slack length by drooping the middle; this gives the
            # integrator a realistic starting shape instead of a taut line.
            sag = cfg.length - L_straight
            droop = -0.5 * sag * 4.0 * s[i] * (1.0 - s[i])  # parabola
            base = base + np.array([0.0, 0.0, droop])
            for j in range(C):
                positions[i, j] = base + perp * w[j]
        velocities = np.zeros_like(positions)

        fixed = np.zeros((R, C), dtype=bool)
        fixed[0, :] = True
        fixed[-1, :] = True

        # Build spring list: structural (right + down) + shear (diagonals)
        springs: list[tuple[int, int, int, int, float]] = []
        for i in range(R):
            for j in range(C):
                for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < R and 0 <= nj < C:
                        rest = float(
                            np.linalg.norm(positions[i, j] - positions[ni, nj])
                        )
                        springs.append((i, j, ni, nj, rest))
        springs_arr = np.asarray(springs, dtype=np.float64)

        return cls(
            positions=positions,
            velocities=velocities,
            fixed=fixed,
            mass=float(cfg.mass_per_particle),
            springs=springs_arr,
            spring_k=float(cfg.spring_k),
            spring_damping=float(cfg.spring_damping),
            global_damping=float(cfg.global_damping),
            wind_bias=np.asarray(cfg.wind_bias, dtype=np.float64),
            wind_noise_std=float(cfg.wind_noise_std),
            rng=rng,
        )

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def step(self, dt: float, max_substep: float | None = None) -> None:
        """Advance by ``dt`` using symplectic Euler, auto-substepping for stability.

        The explicit mass-spring system with ``N`` neighbour springs has a
        stability bound ``omega * dt < 2`` where ``omega ≈ sqrt(N * k / m)``.
        To handle stiff configurations transparently, we split ``dt`` into as
        many substeps as needed to stay under that bound with a safety factor.
        """
        if max_substep is None:
            # Conservative: assume each particle has ~8 spring bonds worst case
            omega_max = float(np.sqrt(8.0 * self.spring_k / max(self.mass, 1e-9)))
            max_substep = 0.5 / max(omega_max, 1e-6)  # factor-of-4 margin vs ω·dt<2
        n_sub = max(1, int(np.ceil(dt / max_substep)))
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            self._step_once(sub_dt)

    def _step_once(self, dt: float) -> None:
        """One symplectic-Euler step at the given ``dt``."""
        R, C, _ = self.positions.shape
        F = np.tile(GRAVITY * self.mass, (R, C, 1))

        # Wind: persistent bias + Gaussian per-particle jitter
        wind = np.tile(self.wind_bias, (R, C, 1))
        wind = apply_noise(wind, "gaussian", sigma=self.wind_noise_std, rng=self.rng)
        F += wind * self.mass

        # Springs
        if self.springs.size:
            i0 = self.springs[:, 0].astype(np.int64)
            j0 = self.springs[:, 1].astype(np.int64)
            i1 = self.springs[:, 2].astype(np.int64)
            j1 = self.springs[:, 3].astype(np.int64)
            rest = self.springs[:, 4]

            p0 = self.positions[i0, j0]   # (S, 3)
            p1 = self.positions[i1, j1]
            v0 = self.velocities[i0, j0]
            v1 = self.velocities[i1, j1]

            d = p0 - p1                   # (S, 3)
            L = np.linalg.norm(d, axis=1) + 1e-12
            dir_ = d / L[:, None]
            stretch = L - rest
            f_spring = -self.spring_k * stretch[:, None] * dir_
            # Spring damping along the spring axis
            rel_v = v0 - v1
            vdot = np.sum(rel_v * dir_, axis=1)
            f_damp = -self.spring_damping * vdot[:, None] * dir_
            f_total = f_spring + f_damp

            # Scatter-add to F
            np.add.at(F, (i0, j0), f_total)
            np.add.at(F, (i1, j1), -f_total)

        # Global velocity damping (models air drag)
        F -= self.global_damping * self.velocities * self.mass

        # Apply fixed (Dirichlet) mask
        free = ~self.fixed
        a = np.where(free[..., None], F / max(self.mass, 1e-9), 0.0)

        # Symplectic Euler
        self.velocities[free] = self.velocities[free] + a[free] * dt
        # Safety: clamp runaway velocities (single-step max speed cap)
        vmag = np.linalg.norm(self.velocities, axis=-1, keepdims=True)
        vmax = 100.0  # m/s; well above any physical tape motion
        over = vmag > vmax
        if np.any(over):
            scale = np.where(over, vmax / np.maximum(vmag, 1e-9), 1.0)
            self.velocities = self.velocities * scale
        self.positions[free] = self.positions[free] + self.velocities[free] * dt

    # ------------------------------------------------------------------
    # Geometry output for the renderer
    # ------------------------------------------------------------------
    def triangles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return triangle vertex arrays ``(v0, v1, v2)`` each shape ``(T, 3)``.

        Each grid quad is split into two triangles with consistent winding.
        """
        R, C, _ = self.positions.shape
        if R < 2 or C < 2:
            # Single-row / single-col tape: synthesize degenerate zero-width
            # quads so the ray tracer has triangles to hit.  We expand across
            # width using a small offset in world-z.
            P = self.positions
            if C == 1:
                # Duplicate column with tiny z offset to form quads
                off = np.zeros_like(P)
                off[..., 2] = 1e-4
                P2 = P + off
                P = np.concatenate([P, P2], axis=1)
                R, C, _ = P.shape
            else:
                return (
                    np.zeros((0, 3)),
                    np.zeros((0, 3)),
                    np.zeros((0, 3)),
                )
        else:
            P = self.positions

        v0 = []
        v1 = []
        v2 = []
        for i in range(R - 1):
            for j in range(C - 1):
                p00 = P[i, j]
                p01 = P[i, j + 1]
                p10 = P[i + 1, j]
                p11 = P[i + 1, j + 1]
                # Triangle A: p00, p10, p11
                v0.append(p00); v1.append(p10); v2.append(p11)
                # Triangle B: p00, p11, p01
                v0.append(p00); v1.append(p11); v2.append(p01)
        return (
            np.asarray(v0, dtype=np.float64),
            np.asarray(v1, dtype=np.float64),
            np.asarray(v2, dtype=np.float64),
        )

    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * float(np.sum(self.velocities**2))
