"""Train the search-and-reach teacher on the native sim -- no Isaac, no droplet.

Same task, same PPO, same checkpoints as scripts/train_search.py; the flight
runs on vesper.native.NativeSearchEnv (our dynamics + SE(3) + the ground raster)
so it trains on a Mac (--device mps) or any CUDA box. The vision student still
needs Isaac's renderer; this lane is the state-based teacher only.

    .venv/bin/python scripts/train_search_native.py --num_envs 4096 --iters 1500 --device mps

Writes runs/<id>/search.pt (best policy + obs-norm), search_last.pt, curve.jsonl.
"""
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from vesper.capture import RunCapture
from vesper.lab.ppo import PPO, PPOCfg
from vesper.native import NativeSearchEnv, NativeSearchEnvCfg

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iters", type=int, default=1500)
parser.add_argument("--horizon", type=int, default=48)
parser.add_argument("--targets", type=int, default=3)
parser.add_argument("--arena", type=float, default=300.0, help="search box half-extent (m)")
parser.add_argument("--arena_start", type=float, default=None,
                    help="curriculum: begin here and grow linearly to --arena over "
                         "--arena_iters, so the first reaches are found in a small box")
parser.add_argument("--arena_iters", type=int, default=600)
parser.add_argument("--episode_s", type=float, default=90.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.997,
                    help="search needs a long horizon; 0.997 is ~13 s at 25 Hz")
parser.add_argument("--map", default=None, help="world map npz (default the Cornell site)")
parser.add_argument("--resume", default=None, help="checkpoint to warm-start from")
parser.add_argument("--hidden", default="256,256,128",
                    help="MLP widths, comma-separated; 1024,1024,512 is ~3.4M parameters")
parser.add_argument("--groups", type=int, default=0,
                    help="vehicle sets shared by groups of envs (0 = one per env)")
parser.add_argument("--obs", choices=["privileged", "policy"], default="privileged",
                    help="which observation PPO trains on: the state-based teacher's, or the honest proprio vector")
parser.add_argument("--device", default=None, help="cpu | mps | cuda (default: best available)")
# --- reward overrides (None = keep the SearchCfg default) ---
parser.add_argument("--w_proximity", type=float, default=None,
                    help="dense per-step reward for loitering near a known target. Lowering it "
                         "stops the policy farming 'stay near' income and forces it to actually reach")
parser.add_argument("--w_time", type=float, default=None,
                    help="per-step time penalty -- the 'staying alive is bad' finish pressure")
parser.add_argument("--reach_radius", type=float, default=None,
                    help="3D range that counts as neutralizing a target (m); larger = easier reaches")
parser.add_argument("--snapshot_every", type=int, default=200,
                    help="save snap_<iter>.pt every N iters and render a short progression "
                         "clip per snapshot after training (0 = off)")
parser.add_argument("--tag", default="search-train-native")
args = parser.parse_args()

if args.device is None:
    args.device = ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")

cfg = NativeSearchEnvCfg()
cfg.num_envs = args.num_envs
cfg.n_targets = args.targets
cfg.episode_length_s = args.episode_s
cfg.search = {"arena_half": args.arena}
for _k in ("w_proximity", "w_time", "reach_radius"):   # reward overrides, if given
    _v = getattr(args, _k)
    if _v is not None:
        cfg.search[_k] = _v
cfg.n_groups = args.groups
cfg.ppo_key = args.obs
if args.map:
    cfg.world_map = args.map

env = NativeSearchEnv(cfg, device=args.device, seed=args.seed)


class Adapter:
    """PPO <-> env (gym-style) bridge, same as train_search.py."""

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_obs = env.num_obs
        self.num_actions = env.num_actions

    def reset(self):
        return self.env.ppo_reset()

    def step(self, a):
        return self.env.ppo_step(a)


track = ("found", "cleared", "coverage", "oob", "flip", "crash")
hidden = tuple(int(h) for h in args.hidden.split(","))
ppo = PPO(Adapter(env), PPOCfg(horizon=args.horizon, lr=args.lr, gamma=args.gamma, track=track),
          hidden=hidden, device=env.device, seed=args.seed)
print(f"policy: {sum(p.numel() for p in ppo.ac.parameters())/1e6:.2f}M parameters, hidden {hidden}", flush=True)
if args.resume:
    ck = torch.load(args.resume, map_location=env.device)
    ppo.ac.load_state_dict(ck["ac"]); ppo.norm.load_state_dict(ck["norm"])
    print(f"resumed from {args.resume}", flush=True)

cap = RunCapture(args.tag)
cap.note(num_envs=args.num_envs, iters=args.iters, targets=args.targets, arena=args.arena,
         episode_s=args.episode_s, gamma=args.gamma, world=cfg.world_map, seed=args.seed,
         hidden=list(hidden), groups=env.G, obs=args.obs, device=args.device, native=True)
curve = open(cap.dir / "curve.jsonl", "w")
print(f"train (native, {args.device}): {args.num_envs} envs x {args.iters} iters, "
      f"{args.targets} targets in a {2*args.arena:.0f} m box, obs {env.num_obs} -> {cap.dir}", flush=True)

t0 = time.time()
best = -1e9


def log(row):
    global best
    curve.write(json.dumps(row) + "\n"); curve.flush()
    steps = (row["iter"] + 1) * args.horizon * args.num_envs
    sps = steps / (time.time() - t0 + 1e-9)
    print(f"it {row['iter']:5d} | ret {row['ep_return']:8.1f} | found {row['found']:.2f} "
          f"| cleared {row['cleared']:.2f} | swept {row['coverage']:.2f} "
          f"| all {row['intercept_rate']:.2f} | t {row['time_to_intercept']:5.1f}s "
          f"| box {2*env.tcfg.arena_half:.0f}m "
          f"| oob {row['oob']:.2f} flip {row['flip']:.2f} crash {row['crash']:.2f} "
          f"| eps {row['episodes']:5d} | {sps/1e3:.0f}k step/s", flush=True)
    # "cleared" is the score that matters: the fraction of vehicles reached
    score = row["cleared"]
    if score == score and score >= best:
        best = score
        ppo.save(cap.dir / "search.pt")
    # progression snapshots (log fires every 10 iters, so any multiple of 10 works)
    if args.snapshot_every and row["iter"] and row["iter"] % args.snapshot_every == 0:
        ppo.save(cap.dir / f"snap_{row['iter']:04d}.pt")


if args.snapshot_every:
    ppo.save(cap.dir / "snap_0000.pt")     # the untrained policy -- the "before" clip

if args.arena_start:
    def grow(row):
        f = min(1.0, (row["iter"] + 1) / max(1, args.arena_iters))
        env.task.set_arena(args.arena_start + f * (args.arena - args.arena_start))
        log(row)

    print(f"curriculum: arena {args.arena_start:.0f} -> {args.arena:.0f} m over "
          f"{args.arena_iters} iters", flush=True)
    env.task.set_arena(args.arena_start)
    hist = ppo.learn(args.iters, log_every=10, on_log=grow)
else:
    hist = ppo.learn(args.iters, log_every=10, on_log=log)
ppo.save(cap.dir / "search_last.pt")

# after-action record: one deterministic episode -> trajectory.parquet (Runs
# plot) + replay.json (tactical video). Every run carries a visual record.
replay_ok = False
try:
    from vesper.native.replay import log_episode
    world_name = Path(cfg.world_map).stem.replace("_map", "")
    # demo-tuned reach radius (same lever as the live warm session): training may
    # use a tight 10 m sphere, but the recorded eval should reliably end in a kill.
    n = log_episode(env, ppo, cap.dir, world_name, reach_radius=25.0)
    replay_ok = (cap.dir / "replay.json").exists()
    print(f"logged eval episode: {n} frames -> trajectory.parquet + replay.json", flush=True)
except Exception as e:                                       # noqa: BLE001
    print(f"[warn] replay logging failed: {e}", flush=True)
    traceback.print_exc()

# auto-render the tactical mp4 so the video attaches without a manual step.
# Best-effort: a render failure must never fail the training run.
if replay_ok:
    try:
        subprocess.run([sys.executable, "scripts/render_replay.py", str(cap.dir)],
                       cwd=ROOT, check=True)
        print(f"rendered tactical video -> {cap.dir / 'tactical.mp4'}", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"[warn] tactical render failed: {e}", flush=True)
        traceback.print_exc()

# progression reel: a short eval clip per snapshot, so the Runs tab shows the
# policy going from flailing to hunting. Same renderer, higher stride, short
# episodes; each snapshot rolls in a temp dir so the run's own trajectory.parquet
# and replay.json stay the final policy's. Best-effort, like the main render.
if args.snapshot_every:
    import tempfile

    from vesper.lab.ppo import load_policy
    from vesper.native.replay import log_episode

    world_name = Path(cfg.world_map).stem.replace("_map", "")
    snaps = sorted(cap.dir.glob("snap_*.pt")) + [cap.dir / "search_last.pt"]
    for ck_path in snaps:
        tag = "final" if ck_path.name == "search_last.pt" else ck_path.stem.split("_")[1]
        caption = ("FINAL POLICY" if tag == "final"
                   else f"ITER {tag} / {args.iters:04d}")
        try:
            ac, norm = load_policy(torch.load(ck_path, map_location=env.device), env.device)

            class _Snap:
                @torch.no_grad()
                def action(self, obs, deterministic=True):
                    return ac.actor(norm(obs))

            with tempfile.TemporaryDirectory() as td:
                nfr = log_episode(env, _Snap(), td, world_name,
                                  max_steps=int(45.0 / env._dt), reach_radius=25.0)
                subprocess.run(
                    [sys.executable, "scripts/render_replay.py", td, "--stride", "5",
                     "--caption", caption,
                     "--out", str(cap.dir / f"progression_{tag}.mp4")],
                    cwd=ROOT, check=True)
            print(f"progression clip {tag}: {nfr} frames -> progression_{tag}.mp4", flush=True)
        except Exception as e:                               # noqa: BLE001
            print(f"[warn] progression clip {tag} failed: {e}", flush=True)
            traceback.print_exc()

# manifest.json names + dates the run for the Runs tab (headless training never
# added frames, so finish() just writes the manifest -- no ffmpeg, no cleanup).
cap.finish()

curve.close()
wall = time.time() - t0
final = [h for h in hist if h["episodes"] > 0][-20:] or [{}]


def avg(key):
    v = [h[key] for h in final if h.get(key) == h.get(key)]
    return round(sum(v) / len(v), 3) if v else float("nan")


summary = {"iters": args.iters, "wall_s": round(wall), "envs": args.num_envs,
           "mean_return": avg("ep_return"), "found": avg("found"), "cleared": avg("cleared"),
           "coverage": avg("coverage"), "all_cleared_rate": avg("intercept_rate"),
           "time_to_clear_s": avg("time_to_intercept"), "best_cleared": round(best, 3),
           "oob": avg("oob"), "flip": avg("flip"), "crash": avg("crash"),
           "policy": str(cap.dir / "search.pt")}
print("DONE " + json.dumps(summary), flush=True)
(cap.dir / "summary.json").write_text(json.dumps(summary, indent=1))
