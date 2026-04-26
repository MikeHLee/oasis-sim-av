"""oasis-sim-av: lightweight AV sensor-fusion edge-case simulator.

Top-level imports re-export the principal classes so downstream users can
simply do ``from oasis_sim_av import KinematicBicycle, MassSpringCloth, ...``.
"""
from __future__ import annotations

from .camera import PinholeCamera
from .cloth import MassSpringCloth
from .config import ScenarioConfig
from .geometry import ray_aabb_batch, ray_triangle_batch
from .lidar import SimulatedLiDAR
from .noise import apply_noise
from .vehicle import KinematicBicycle, make_controller
from .world import World

__all__ = [
    "KinematicBicycle",
    "MassSpringCloth",
    "PinholeCamera",
    "ScenarioConfig",
    "SimulatedLiDAR",
    "World",
    "apply_noise",
    "make_controller",
    "ray_aabb_batch",
    "ray_triangle_batch",
]

__version__ = "0.1.0"
