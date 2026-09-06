"""Score a search policy on the native sim and say HOW it searches, not just
how often it wins -- the Mac-side counterpart of scripts/eval_search.py.

    .venv/bin/python scripts/eval_search_native.py --policy runs/<id>/search.pt \
        --map assets/kramatorsk/kramatorsk_map.npz --num_envs 256 --episodes 512

Beyond the classic funnel (swept / found / cleared / how each episode ended) it
measures the two failure modes of project memory circling-root-cause:

  detection spread   when in the episode first sightings happen, bucketed by
                     episode quarter. A policy that only finds things in its
                     opening sweep piles everything into Q1.
  orbit fraction     share of 10 s windows a drone spent going nowhere: net
                     displacement under 30 m. The scan-orbit (r~9 m, 5.3 s
                     period) lives squarely in these windows.

--fleet evals the live-demo shape instead: N drones sharing one vehicle set
(n_groups=1), fleet-level distinct kills with the warm session's wreck pinning
(a struck tank stays struck when its striker's episode auto-resets), and
expended airframes counted. That is the number the operator sees: kills /3.
"""
import argparse
import json

import torch

from vesper.lab.ppo import load_policy
from vesper.native import NativeSearchEnv, NativeSearchEnvCfg

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--map", default=None)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--targets", type=int, default=3)
parser.add_argument("--arena", type=float, default=300.0)
parser.add_argument("--episode_s", type=float, default=90.0)
parser.add_argument("--reach_radius", type=float, default=None)
parser.add_argument("--pos_id_range", type=float, default=None,
                    help="close-range omni positive ID (m); 0 restores the cone-only sensor")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--device", default="cpu")
parser.add_argument("--no_expend", action="store_true",
                    help="historical semantics: a striking drone keeps flying")
parser.add_argument("--fleet", action="store_true",
                    help="demo shape: all drones hunt one shared vehicle set; "
                         "score fleet-level distinct kills over --episodes sorties")
parser.add_argument("--out", default=None)
args = parser.parse_args()

cfg = NativeSearchEnvCfg()
cfg.num_envs = args.num_envs
cfg.n_targets = args.targets
cfg.episode_length_s = args.episode_s
cfg.search = {"arena_half": args.arena}
if args.reach_radius is not None:
    cfg.search["reach_radius"] = args.reach_radius
if args.pos_id_range is not None:
    cfg.search["pos_id_range"] = args.pos_id_range
if args.no_expend:
    cfg.search["expend_on_reach"] = False
cfg.n_groups = 1 if args.fleet else 0
if args.map:
    cfg.world_map = args.map
env = NativeSearchEnv(cfg, device=args.device, seed=args.seed)

ck = torch.load(args.policy, map_location=env.device)
if ck["obs_dim"] != env.num_obs:
    raise SystemExit(f"policy expects {ck['obs_dim']}-wide obs, env gives {env.num_obs}")
ac, norm = load_policy(ck, env.device)


@torch.no_grad()
def policy(o):
    return ac.actor(norm(o))


N, K = env.num_envs, env.k
dev = env.device
W = int(10.0 / env._dt)                      # 10 s orbit-detection window

obs = env.ppo_reset()
prev_known = env.task.known.clone()
pos0_win = env.flight_state()[0][:, :2].clone()   # window-start position
win_step = torch.zeros(N, dtype=torch.long, device=dev)

quarters = torch.zeros(4, dtype=torch.long)  # first sightings per episode quarter
orbit_win = still_win = 0                    # windows: net-disp < 30 m / total
radial_max = torch.zeros(N, device=dev)      # farthest each drone got from centre this episode
reach_radii = []                             # per-drone episode max radius, on reset
outer_serviced = []                          # fleet: was the outermost target killed?
ep = {"found": [], "cleared": [], "coverage": [], "len_s": [],
      "crash": 0, "oob": 0, "flip": 0, "expend": 0, "timeout": 0, "all": 0}
fleet = {"kills": [], "found": [], "expended": []}   # per completed sortie
wrecked = torch.zeros(K, dtype=torch.bool, device=dev)
expended = torch.zeros(N, dtype=torch.bool, device=dev)
sortie_found = torch.zeros(K, dtype=torch.bool, device=dev)
sortie_tr = env.target_pos[0, :, :2].norm(dim=1).clone()   # this sortie's target radii
grp0 = env.group == 0

done_eps = 0
step = 0
while done_eps < args.episodes:
    obs, rew, done, info = env.ppo_step(policy(obs))
    step += 1
    st = info["strike"]

    # detection spread: bucket each env's NEW first-sightings by episode quarter
    newly = env.task.known & ~prev_known
    if bool(newly.any()):
        q = (env.episode_length_buf.float() / env.max_episode_length * 4).long().clamp(0, 3)
        for qi in range(4):
            quarters[qi] += int(newly[q == qi].sum())
    prev_known = env.task.known.clone()

    # orbit windows: every W steps, how far did each still-running drone get?
    win_step += 1
    roll = win_step >= W
    if bool(roll.any()):
        p = env.flight_state()[0][:, :2]
        disp = (p[roll] - pos0_win[roll]).norm(dim=1)
        orbit_win += int((disp < 30.0).sum())
        still_win += int(roll.sum())
        pos0_win[roll] = p[roll]
        win_step[roll] = 0
    # coverage reach: how far from the arena centre each drone flies
    radial_max = torch.maximum(radial_max, env.flight_state()[0][:, :2].norm(dim=1))
    # a reset drone starts a fresh window
    if bool(done.any()):
        pos0_win[done] = env.flight_state()[0][done, :2]
        win_step[done] = 0
        reach_radii += radial_max[done].tolist()
        radial_max[done] = 0.0

    if args.fleet:
        expended |= st.any(dim=1)
        wrecked |= st[grp0].any(dim=0)
        # the sortie ends the step the lead's episode does (owner_done resets
        # the whole group), so the final step's knowledge lives in `st` and the
        # running accumulators, not in the freshly-wiped task tensors
        if bool(done[0]):                     # the lead ended -> sortie over
            fleet["kills"].append(int(wrecked.sum()))
            fleet["found"].append(int((sortie_found | wrecked).sum()))
            fleet["expended"].append(int(expended.sum()))
            # outer-ring acceptance: the target farthest from the AO centre --
            # was it serviced (killed) this sortie? sortie_tr held the radii
            # before this step's auto-reset swapped in a new target set
            outer_serviced.append(float(wrecked[int(sortie_tr.argmax())]))
            wrecked[:] = False
            expended[:] = False
            sortie_found[:] = False
            prev_known = env.task.known.clone()
            sortie_tr = env.target_pos[0, :, :2].norm(dim=1).clone()   # new sortie
            done_eps += 1
        else:
            sortie_found |= env.task.known[grp0].any(dim=0)
            # warm-session wreck pinning: the striker's auto-reset must not
            # revive the tank for the rest of this sortie
            if bool(wrecked.any()):
                env.task.reached[grp0] |= wrecked
                env.task.known[grp0] |= wrecked
    else:
        if bool(done.any()):
            idx = done.nonzero().flatten()
            ep["found"] += info["found"][idx].tolist()
            ep["cleared"] += info["cleared"][idx].tolist()
            ep["coverage"] += info["coverage"][idx].tolist()
            ep["crash"] += int(info["crash"][idx].sum())
            ep["oob"] += int(info["oob"][idx].sum())
            ep["flip"] += int(info["flip"][idx].sum())
            ep["all"] += int(info["intercept"][idx].sum())
            expends = st.any(dim=1)[idx] & ~info["intercept"][idx]
            ep["expend"] += int(expends.sum())
            done_eps += len(idx)


def mean(v):
    return round(sum(v) / max(len(v), 1), 3)


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return round(s[min(len(s) - 1, int(p * len(s)))], 1)


qt = quarters.float()
res = {
    "policy": args.policy, "map": cfg.world_map, "episodes": done_eps,
    "episode_s": args.episode_s, "reach_radius": env.tcfg.reach_radius,
    "expend_on_reach": env.tcfg.expend_on_reach, "seed": args.seed,
    "detect_quarters": [round(float(x / qt.sum().clamp(min=1)), 3) for x in qt],
    "detections_total": int(qt.sum()),
    "orbit_frac": round(orbit_win / max(still_win, 1), 3),
    "drone_radius_p50_m": pct(reach_radii, 0.5),
    "drone_radius_p95_m": pct(reach_radii, 0.95),
    "drone_radius_max_m": round(max(reach_radii), 1) if reach_radii else 0.0,
}
if args.fleet:
    res |= {"mode": "fleet", "drones": N,
            "kills_per_sortie": mean(fleet["kills"]),
            "found_per_sortie": mean(fleet["found"]),
            "expended_per_sortie": mean(fleet["expended"]),
            "sorties_3_for_3": mean([1.0 if k >= K else 0.0 for k in fleet["kills"]]),
            "outer_target_serviced": mean(outer_serviced)}
else:
    n = done_eps
    res |= {"mode": "solo", "found": mean(ep["found"]), "cleared": mean(ep["cleared"]),
            "coverage": mean(ep["coverage"]),
            "end_crash": round(ep["crash"] / n, 3), "end_oob": round(ep["oob"] / n, 3),
            "end_flip": round(ep["flip"] / n, 3), "end_expend": round(ep["expend"] / n, 3),
            "end_all_cleared": round(ep["all"] / n, 3)}
print(json.dumps(res, indent=1))
if args.out:
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
