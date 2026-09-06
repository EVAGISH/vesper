"""Ground-vehicle roles and driving, shared by the Isaac and native search envs.

The role table and hull limits used to live in vesper.lab.search_env, which
imports isaaclab at module top and so cannot be loaded outside the container.
They are here so the native env (vesper.native) and the Isaac env agree on what
a tank is without either importing the other's runtime.

`steer` is the pure-torch steering tick ported from SearchEnv._drive_vehicles:
speed ramp, stuck recovery, bounded wander, and the probe fan that keeps a hull
on drivable ground (and on the road, for the roles that follow one). It mutates
the caller's state tensors and returns the ground speed to drive at this step.
SearchEnv still carries its own copy of this logic around PhysX velocity
writes; converging it onto this function is a droplet-side change (it needs
check_search.py run under Isaac to verify) and is deliberately not done here.

Pure torch, no Isaac. Frame: world metres, x east, y north, z up.
"""
import math

import torch

# (name, ground speed m/s, optical contrast, spawn layer, follows roads)
#   open        a tank driving on the roads, in plain sight
#   camouflaged same behaviour, painted to blend into the ground (geometric mode only:
#               the rendered tank wears whatever paint the asset ships with)
#   concealed   crawling under tree canopy, plain paint but hard to see through leaves
#   parked      shut down a few metres off a building wall, nose along it
ROLES = [
    ("open", 4.5, 1.00, "road", True),
    ("camouflaged", 3.5, 0.32, "road", True),
    ("concealed", 1.2, 0.90, "concealed", False),
    ("parked", 0.0, 0.75, "parking", False),
]
VEHICLE_SEMANTIC = "vehicle"
VEHICLE_PATH_RX = r"env_(\d+)/Vehicle_(\d+)"

# vehicle driving: what the hull can do
VEH_ACCEL = 1.5            # m/s^2 speed ramp from rest
VEH_LAT_ACCEL = 2.5        # m/s^2 lateral budget -> turn-rate limit at speed
VEH_TURN_MAX = 1.0         # rad/s at walking pace
VEH_STUCK_S = 2.0          # seconds below 30% of commanded speed before it turns away

# probe fan used to steer vehicles away from ground they cannot drive on
PROBE = (0.0, 0.5, -0.5, 1.0, -1.0, 1.8, -1.8, 3.14159)


def steer(state, world, arena_half: float, actual_speed: torch.Tensor, dt: float,
          gen=None, probe: torch.Tensor | None = None):
    """One steering tick for every hull. All tensors [N,K].

    `state` is any object carrying veh_heading, veh_turn_rate, veh_speed,
    veh_speed_cmd, veh_stuck_s and veh_on_road (the search envs do); they are
    updated in place. `actual_speed` is the hull's measured ground speed, for
    stuck detection -- a kinematic caller passes its own commanded speed and the
    stuck branch simply never fires (nothing to be stuck on). Returns the
    ground speed [N,K] to drive at this step, already slowed into turns.
    """
    dev = state.veh_heading.device
    N, K = state.veh_heading.shape
    if probe is None:
        probe = torch.tensor(PROBE, device=dev)
    pos_x, pos_y = state.veh_pos[..., 0], state.veh_pos[..., 1]

    # --- speed: ramp toward the role's cruise, stuck detection on the actual hull speed
    state.veh_speed_cmd = torch.minimum(state.veh_speed_cmd + VEH_ACCEL * dt, state.veh_speed)
    want_move = state.veh_speed_cmd > 0.5
    slow = actual_speed < 0.3 * state.veh_speed_cmd
    state.veh_stuck_s = torch.where(want_move & slow, state.veh_stuck_s + dt,
                                    torch.zeros_like(state.veh_stuck_s))
    stuck = state.veh_stuck_s > VEH_STUCK_S
    if stuck.any():
        sign = torch.where(torch.rand(N, K, device=dev, generator=gen) < 0.5, -1.0, 1.0)
        state.veh_heading = torch.where(stuck, state.veh_heading + sign * (math.pi / 2),
                                        state.veh_heading)
        state.veh_speed_cmd = torch.where(stuck, torch.zeros_like(state.veh_speed_cmd),
                                          state.veh_speed_cmd)
        state.veh_stuck_s = torch.where(stuck, torch.zeros_like(state.veh_stuck_s),
                                        state.veh_stuck_s)

    # --- wander, bounded by what the hull can steer at this speed
    turn_max = torch.minimum(torch.full_like(state.veh_speed_cmd, VEH_TURN_MAX),
                             VEH_LAT_ACCEL / state.veh_speed_cmd.clamp(min=0.5))
    noise = torch.randn(N, K, device=dev, generator=gen)
    state.veh_turn_rate = (state.veh_turn_rate * 0.995 + noise * 0.05).clamp(-1.0, 1.0) * turn_max
    state.veh_heading = state.veh_heading + state.veh_turn_rate * dt

    # --- look ahead along a fan of candidate headings and pick drivable ground
    probe_h = state.veh_heading.unsqueeze(2) + probe.view(1, 1, -1)              # [N,K,P]
    ahead = 14.0
    px = pos_x.unsqueeze(2) + ahead * torch.cos(probe_h)
    py = pos_y.unsqueeze(2) + ahead * torch.sin(probe_h)
    r, col = world.nearest_cell(px, py)
    ok = world.drivable[r, col]                                                  # [N,K,P]
    road = world.road[r, col]
    inside = (px.abs() < arena_half) & (py.abs() < arena_half)
    score = ok * inside.float() - 0.03 * probe.abs().view(1, 1, -1)              # prefer straight on
    score = score + 0.6 * road * state.veh_on_road.unsqueeze(2).float()          # and road, if that is the job
    best = score.argmax(dim=2)
    best_score = score.gather(2, best.unsqueeze(2)).squeeze(2)
    straight = score[..., 0]
    # leave the current heading only when it is blocked, or when a road-
    # follower's own heading is off the road and a probe finds one
    blocked = straight < 0.5
    leave = blocked | (state.veh_on_road & (best_score > straight + 0.3))
    chosen = probe_h.gather(2, best.unsqueeze(2)).squeeze(2)
    state.veh_heading = torch.where(leave, chosen, state.veh_heading)
    # nothing drivable in any direction: turn back toward the arena centre
    home = torch.atan2(-pos_y, -pos_x)
    lost = score.amax(dim=2) < 0.5
    state.veh_heading = torch.where(lost, home, state.veh_heading)
    state.veh_turn_rate = torch.where(leave | lost, torch.zeros_like(state.veh_turn_rate),
                                      state.veh_turn_rate)

    # slow into a turn: at the full steering rate the hull drops to half
    # speed, which is what keeps a tank from drifting off a bend
    return state.veh_speed_cmd * (1.0 - 0.5 * (state.veh_turn_rate.abs() / VEH_TURN_MAX).clamp(max=1.0))
