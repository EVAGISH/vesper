"""Forklift driver: velocity commands for K vehicles that behave like vehicles.

Pure torch over a WorldMap, no physics of its own: the env reads each hull's
actual pose and speed out of PhysX, asks for a command, and writes the
velocity back. That keeps the driver CPU-testable (integrate the commands
kinematically on the synthetic map) and keeps PhysX in charge of contacts,
slopes and the ride over terrain.

Behaviour, per vehicle:
  * a role sets the cruise speed and the spawn layer: cruising on the roads,
    crawling under canopy, parked along a facade;
  * a waypoint drawn from the roads (or drivable ground) is the goal; a new
    one is drawn on arrival, on a timeout, or when nothing drivable is ahead;
  * a fan of probes ahead scores headings: drivable and inside the arena
    first, toward the goal second, on a road third; the hull leaves its
    current heading only when that heading is blocked or clearly worse;
  * the hull cannot do what a forklift cannot: speed ramps from rest, the
    turn rate is capped by a lateral-acceleration budget, speed drops into a
    turn, and a hull that has pushed against something for two seconds turns
    away and tries again.
"""
from __future__ import annotations

import math

import torch

# (name, cruise m/s, spawn layer, drives)
ROLES = [
    ("cruise", 4.5, "road", True),
    ("crawl", 1.2, "concealed", True),
    ("parked", 0.0, "parking", False),
]
ROLE_INDEX = {r[0]: i for i, r in enumerate(ROLES)}

ACCEL = 1.5            # m/s^2 speed ramp from rest
LAT_ACCEL = 2.5        # m/s^2 lateral budget -> turn-rate limit at speed
TURN_MAX = 1.0         # rad/s at walking pace
STUCK_S = 2.0          # seconds below 30% of commanded speed before turning away
PROBE_AHEAD = 14.0     # m
GOAL_RADIUS = 8.0      # m: arrived
GOAL_TIMEOUT_S = 45.0
YAW_SERVO_MAX = 3.6    # rad/s the hull's yaw servo can ask for
ROAD_WEIGHT = 0.8      # a road follower's preference for road over lawn, vs 0.25 for the goal bearing


class ForkliftDriver:
    def __init__(self, world, k: int, arena_half: float, device="cpu", generator=None):
        self.world, self.k, self.half = world, int(k), float(arena_half)
        self.device, self.gen = device, generator
        z = lambda *s, **kw: torch.zeros(*s, device=device, **kw)      # noqa: E731
        self.role = z(self.k, dtype=torch.long)
        self.cruise = z(self.k)
        self.speed_cmd = z(self.k)
        self.heading = z(self.k)
        self.turn_rate = z(self.k)
        self.stuck_s = z(self.k)
        self.goal = z(self.k, 2)
        self.goal_age = z(self.k)
        self.on_road = z(self.k, dtype=torch.bool)
        self.probe = torch.tensor([0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 1.8, -1.8, 3.14159], device=device)

    # ---------------------------------------------------------------- placement
    def assign_roles(self, ids, roles=None):
        """Give the vehicles in `ids` a role each (random from ROLES unless given)."""
        ids = torch.as_tensor(ids, device=self.device)
        m = len(ids)
        if roles is None:
            roles = torch.randint(0, len(ROLES), (m,), device=self.device, generator=self.gen)
        cruise = torch.tensor([r[1] for r in ROLES], device=self.device)
        drives = torch.tensor([r[3] for r in ROLES], device=self.device)
        self.role[ids] = roles
        self.cruise[ids] = cruise[roles] * (0.7 + 0.6 * torch.rand(m, device=self.device, generator=self.gen))
        self.on_road[ids] = drives[roles]
        self.speed_cmd[ids] = 0.0
        self.stuck_s[ids] = 0.0
        self.turn_rate[ids] = 0.0

    def place(self, ids):
        """(xy [m,2], heading [m]) spawn poses for `ids` by role layer, with fallback
        to open drivable ground when the layer is empty inside the arena."""
        ids = torch.as_tensor(ids, device=self.device)
        m = len(ids)
        w, g = self.world, self.gen
        xy, _ = w.sample_cells_xy(w.drivable, m, g, half=self.half)
        heading = torch.rand(m, device=self.device, generator=g) * (2 * math.pi)
        for li, (_, _, layer, _) in enumerate(ROLES):
            want = self.role[ids] == li
            if not want.any():
                continue
            mask = getattr(w, layer)
            cand, ok = w.sample_cells_xy(mask, m, g, half=self.half)
            use = want & ok
            xy = torch.where(use.unsqueeze(1), cand, xy)
            if layer in ("road", "parking"):
                field = w.road_yaw if layer == "road" else w.park_yaw
                along = w.yaw_at(field, cand[:, 0], cand[:, 1])
                flip = (torch.rand(m, device=self.device, generator=g) < 0.5).float() * math.pi
                heading = torch.where(use, along + flip, heading)
        self.heading[ids] = heading
        self.goal_age[ids] = 1e6                     # draw a goal on the first step
        return xy, heading

    def _new_goals(self, ids):
        if len(ids) == 0:
            return
        w = self.world
        road = self.on_road[ids]
        xy_r, ok_r = w.sample_cells_xy(w.road, len(ids), self.gen, half=self.half)
        xy_d, _ = w.sample_cells_xy(w.drivable, len(ids), self.gen, half=self.half)
        self.goal[ids] = torch.where((road & ok_r).unsqueeze(1), xy_r, xy_d)
        self.goal_age[ids] = 0.0

    # ---------------------------------------------------------------- driving
    def command(self, pos_xy, yaw, speed, dt: float):
        """One control step. pos_xy [K,2], yaw [K] (hull nose), speed [K] actual
        ground speed -> (v_xy [K,2] world, yaw_rate [K])."""
        K, dev = self.k, self.device
        moving = self.cruise > 0.05

        # --- goals
        to_goal = self.goal - pos_xy
        dist = to_goal.norm(dim=1)
        self.goal_age = self.goal_age + dt
        need = moving & ((dist < GOAL_RADIUS) | (self.goal_age > GOAL_TIMEOUT_S))
        self._new_goals(torch.nonzero(need).flatten())
        to_goal = self.goal - pos_xy
        bearing = torch.atan2(to_goal[:, 1], to_goal[:, 0])

        # --- speed ramp and stuck detection on the actual hull speed
        self.speed_cmd = torch.minimum(self.speed_cmd + ACCEL * dt, self.cruise)
        want_move = self.speed_cmd > 0.5
        stuck_now = want_move & (speed < 0.3 * self.speed_cmd)
        self.stuck_s = torch.where(stuck_now, self.stuck_s + dt, torch.zeros_like(self.stuck_s))
        stuck = self.stuck_s > STUCK_S
        if stuck.any():
            sign = torch.where(torch.rand(K, device=dev, generator=self.gen) < 0.5, -1.0, 1.0)
            self.heading = torch.where(stuck, self.heading + sign * (math.pi / 2), self.heading)
            self.speed_cmd = torch.where(stuck, torch.zeros_like(self.speed_cmd), self.speed_cmd)
            self.stuck_s = torch.where(stuck, torch.zeros_like(self.stuck_s), self.stuck_s)
            self._new_goals(torch.nonzero(stuck).flatten())

        # --- wander, bounded by what the hull can steer at this speed
        turn_max = torch.minimum(torch.full_like(self.speed_cmd, TURN_MAX),
                                 LAT_ACCEL / self.speed_cmd.clamp(min=0.5))
        noise = torch.randn(K, device=dev, generator=self.gen)
        self.turn_rate = (self.turn_rate * 0.995 + noise * 0.05).clamp(-1.0, 1.0) * turn_max
        # where the hull would like to go: the goal bearing, with some wander
        desired = bearing + 0.3 * self.turn_rate

        # --- probe fan: drivable and inside first, toward the goal second, road third
        probe_h = self.heading.unsqueeze(1) + self.probe.view(1, -1)                 # [K,P]
        px = pos_xy[:, 0:1] + PROBE_AHEAD * torch.cos(probe_h)
        py = pos_xy[:, 1:2] + PROBE_AHEAD * torch.sin(probe_h)
        r, c = self.world.nearest_cell(px, py)
        ok = self.world.drivable[r, c]
        road = self.world.road[r, c]
        inside = (px.abs() < self.half) & (py.abs() < self.half)
        score = ok * inside.float() - 0.03 * self.probe.abs().view(1, -1)
        score = score + 0.25 * torch.cos(probe_h - bearing.unsqueeze(1))
        score = score + ROAD_WEIGHT * road * self.on_road.unsqueeze(1).float()
        best = score.argmax(dim=1)
        best_score = score.gather(1, best.unsqueeze(1)).squeeze(1)
        straight = score[:, 0]
        blocked = ok[:, 0] * inside[:, 0].float() < 0.5
        leave = blocked | (best_score > straight + 0.35)
        chosen = probe_h.gather(1, best.unsqueeze(1)).squeeze(1)
        # a road follower that is on a road follows it: the best probe, every
        # step, rather than the goal bearing that would cut across the lawn
        following = self.on_road & (road[:, 0] > 0.5)
        desired = torch.where(leave | following, chosen, desired)
        lost = (ok * inside.float()).amax(dim=1) < 0.5
        home = torch.atan2(-pos_xy[:, 1], -pos_xy[:, 0])
        desired = torch.where(lost, home, desired)
        self.turn_rate = torch.where(leave | lost, torch.zeros_like(self.turn_rate), self.turn_rate)

        # --- steer toward it no faster than the hull can, slowing for the turn:
        # a reversal is a stop and a three-point turn, not a flip of the velocity
        err = torch.atan2(torch.sin(desired - self.heading), torch.cos(desired - self.heading))
        self.heading = self.heading + err.clamp(-turn_max * dt, turn_max * dt)
        self.heading = torch.atan2(torch.sin(self.heading), torch.cos(self.heading))
        turn_factor = (1.0 - err.abs() / math.pi).clamp(min=0.15)

        # --- the command: slow into turns, servo the nose onto the heading
        sp = self.speed_cmd * turn_factor * moving.float()
        v_xy = torch.stack([sp * torch.cos(self.heading), sp * torch.sin(self.heading)], dim=1)
        yaw_err = torch.atan2(torch.sin(self.heading - yaw), torch.cos(self.heading - yaw))
        yaw_rate = (yaw_err / dt).clamp(-YAW_SERVO_MAX, YAW_SERVO_MAX) * moving.float()
        return v_xy, yaw_rate
