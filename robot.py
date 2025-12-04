from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from environment import (
    WorldConfig,
    Site,
    clip_to_world,
    sense_sites,
)


@dataclass
class RobotConfig:
    """Robot physical + sensing/comm parameters."""
    radius: float = 0.1

    # sensing and communication
    sensor_range: float = 1.5     # for detecting victims / false positives
    comm_range: float = 3.0       # for consensus graph

    # motion
    v_search: float = 0.8         # speed during phase 1
    v_nav: float = 1.0            # speed during phase 2
    max_omega: float = 3.0        # max turning rate (rad/s)


@dataclass
class Robot:
    id: int
    x: float
    y: float
    theta: float

    mode: str = "phase1"
    assigned_victim: Optional[int] = None
    slot_angle: Optional[float] = None   # <--- ADD THIS LINE

    info_state: float = 0.0
    info_history: List[float] = field(default_factory=list)

    belief_victims: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=float)
    )
    has_seen_victims: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )

    pos_history_phase1: List[np.ndarray] = field(default_factory=list)
    pos_history_phase2: List[np.ndarray] = field(default_factory=list)
    belief_history_phase1: list = field(default_factory=list)
    belief_var_history_phase1: list = field(default_factory=list)
    fp_confusion_events_phase1: list = field(default_factory=list)
    meas_noise_scatter_phase1: list = field(default_factory=list)
    detection_time_phase1: dict = field(default_factory=dict)

    def pose(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    # ---------------- Motion ---------------- #

    def step_unicycle(self,
                      v: float,
                      omega: float,
                      dt: float,
                      world_cfg: WorldConfig) -> None:
        """
        Simple unicycle update in continuous world, with clipping
        back into the box. Logging is handled by the phase code.
        """
        max_omega = np.inf  # could use RobotConfig.max_omega if passed in
        omega = float(np.clip(omega, -max_omega, max_omega))

        # integrate
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt

        # wrap heading to [-pi, pi]
        self.theta = (self.theta + np.pi) % (2.0 * np.pi) - np.pi

        # keep inside world
        p = clip_to_world(self.pose(), world_cfg)
        self.x, self.y = float(p[0]), float(p[1])

    # ---------------- Sensing ---------------- #

    def sense_environment(self,
                          victims: List[Site],
                          false_sites: List[Site],
                          r_cfg: RobotConfig) -> List[Site]:
        """
        Return list of all sites within sensor range.
        Uses environment.sense_sites under the hood.
        """
        hits = sense_sites(self.pose(), victims, false_sites, r_cfg.sensor_range)
        return hits


# ------------- Helper to create initial swarm ------------- #

def init_robots(n_robots: int,
                world_cfg: WorldConfig,
                r_cfg: RobotConfig,
                rng_seed: Optional[int] = None) -> List[Robot]:
    """
    Spawn robots along the left side of the world with random y and heading.
    Belief arrays are initialized later (in phase1_markov) once we know n_victims.
    """
    rng = np.random.default_rng(rng_seed)
    robots: List[Robot] = []

    for i in range(n_robots):
        # small offset from left wall
        x0 = 0.5
        y0 = rng.uniform(1.0, world_cfg.height - 1.0)

        # heading roughly pointing into the world (towards +x)
        theta0 = rng.uniform(-np.pi / 4, np.pi / 4)

        r = Robot(
            id=i,
            x=x0,
            y=y0,
            theta=theta0,
            mode="phase1",
        )
        robots.append(r)

    return robots
