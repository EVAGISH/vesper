"""Train the end-to-end vision policy: camera in, touch a forklift, fast.

    /isaac-sim/python.sh scripts/train_chase.py --num_envs 128 --iters 3000 \
        --headless --enable_cameras

The actor sees the body-fixed camera (RGB + depth) and the airframe's own
instruments, and nothing else -- no GPS, no map, no target list. The critic
sees the truth. Recurrent PPO (vesper.lab.recurrent_ppo) keeps the GRU's
sequences intact.

--teacher trains the state-based policy instead: the same task and reward, the
privileged vector as the observation, the flat trainer, thousands of
environments and no rendering. That is the fast way to check the task is
learnable at all, and the source of the actions a vision student can be
distilled from.

Writes runs/<id>/chase.pt (best), chase_last.pt, curve.jsonl, summary.json.
"""
import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=3000)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--targets", type=int, default=6)
parser.add_argument("--arena", type=float, default=300.0)
parser.add_argument("--arena_start", type=float, default=None,
                    help="curriculum: begin here and grow to --arena over --arena_iters, so the "
                         "first touches are found in a small box")
parser.add_argument("--arena_iters", type=int, default=800)
parser.add_argument("--episode_s", type=float, default=60.0)
parser.add_argument("--res", type=int, default=96)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--teacher", action="store_true",
                    help="train the privileged state policy with the flat trainer (no rendering)")
parser.add_argument("--hidden", default="256,256,128", help="teacher MLP widths")
parser.add_argument("--world", default=None)
parser.add_argument("--map", default=None)
parser.add_argument("--zones", default=None)
parser.add_argument("--vehicle", default=None)
parser.add_argument("--resume", default=None)
parser.add_argument("--tag", default="chase-train")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if not args.teacher:
    args.enable_cameras = True
app = AppLauncher(args).app

import carb  # noqa: E402

if not args.teacher:
    _s = carb.settings.get_settings()
    _s.set("/rtx-transient/resourcemanager/enableTextureStreaming", False)
    _s.set("/rtx/post/aa/op", 2)
    _s.set("/rtx/post/dlss/execMode", 0)
    _s.set("/rtx/post/motionblur/enabled", False)

import torch  # noqa: E402

from vesper.capture import RunCapture  # noqa: E402
from vesper.lab.chase_env import ChaseEnv, ChaseEnvCfg  # noqa: E402
from vesper.lab.ppo import PPO, PPOCfg  # noqa: E402
from vesper.lab.recurrent_ppo import RPPOCfg, RecurrentPPO  # noqa: E402

cfg = ChaseEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 0.0
cfg.n_targets = args.targets
cfg.episode_length_s = args.episode_s
cfg.chase = {"arena_half": args.arena}
cfg.vehicle_model = args.vehicle
cfg.camera = not args.teacher
cfg.cam_res = args.res
cfg.ppo_key = "privileged"
if args.world:
    cfg.world_usd = args.world
if args.map:
    cfg.world_map = args.map
if args.zones:
    cfg.zones = args.zones

env = ChaseEnv(cfg, seed=args.seed)
track = ("crash", "oob", "flip", "seen")


class Adapter:
    """flat PPO <-> ChaseEnv, for the privileged teacher."""

    def __init__(self, env):
        self.env, self.num_envs, self.device = env, env.num_envs, env.device
        self.num_obs, self.num_actions = env.num_obs, env.num_actions

    def reset(self):
        return self.env.ppo_reset()

    def step(self, a):
        return self.env.ppo_step(a)


if args.teacher:
    hidden = tuple(int(h) for h in args.hidden.split(","))
    trainer = PPO(Adapter(env), PPOCfg(horizon=args.horizon, lr=args.lr, gamma=args.gamma, track=track),
                  hidden=hidden, device=env.device, seed=args.seed)
    net = trainer.ac
    rate_key = "intercept_rate"
else:
    trainer = RecurrentPPO(env, RPPOCfg(horizon=args.horizon, lr=args.lr, gamma=args.gamma, track=track),
                           device=env.device, seed=args.seed, res=args.res)
    net = trainer.ac
    rate_key = "touch_rate"
if args.resume:
    ck = torch.load(args.resume, map_location=env.device)
    net.load_state_dict(ck["ac"])
    trainer.norm.load_state_dict(ck["norm"])
    print(f"resumed from {args.resume}", flush=True)

n_par = sum(p.numel() for p in net.parameters())
on_board = net.n_params(deployed=True) if hasattr(net, "n_params") else n_par
print(f"policy: {n_par/1e6:.2f}M parameters"
      + (f" ({on_board/1e6:.2f}M on the airframe, {net.macs_per_frame()/1e6:.0f}M MACs/frame)"
         if not args.teacher else ""), flush=True)

cap = RunCapture(args.tag)
cap.note(num_envs=args.num_envs, iters=args.iters, targets=args.targets, arena=args.arena,
         episode_s=args.episode_s, gamma=args.gamma, world=cfg.world_usd, seed=args.seed,
         mode="teacher" if args.teacher else "vision", res=args.res, params=n_par,
         zones=env.zones_path)
curve = open(cap.dir / "curve.jsonl", "w")
print(f"train: {args.num_envs} envs x {args.iters} iters, {args.targets} forklifts in a "
      f"{2*args.arena:.0f} m box -> {cap.dir}", flush=True)

t0 = time.time()
best = -1e9


def log(row):
    global best
    curve.write(json.dumps(row) + "\n"); curve.flush()
    steps = (row["iter"] + 1) * args.horizon * args.num_envs
    sps = steps / (time.time() - t0 + 1e-9)
    rate = row.get(rate_key, float("nan"))
    print(f"it {row['iter']:5d} | ret {row['ep_return']:8.1f} | touch {rate:.2f} "
          f"| t {row.get('time_to_touch', float('nan')):5.1f}s | seen {row.get('seen', float('nan')):.2f} "
          f"| box {2*env.task.cfg.arena_half:.0f}m "
          f"| crash {row.get('crash', float('nan')):.2f} oob {row.get('oob', float('nan')):.2f} "
          f"| eps {row['episodes']:5d} | {sps/1e3:.1f}k step/s", flush=True)
    score = rate
    if score == score and score >= best:
        best = score
        trainer.save(cap.dir / "chase.pt")


if args.arena_start:
    def grow(row):
        f = min(1.0, (row["iter"] + 1) / max(1, args.arena_iters))
        env.task.cfg.arena_half = args.arena_start + f * (args.arena - args.arena_start)
        log(row)

    print(f"curriculum: arena {args.arena_start:.0f} -> {args.arena:.0f} m over {args.arena_iters} iters",
          flush=True)
    env.task.cfg.arena_half = args.arena_start
    hist = trainer.learn(args.iters, log_every=10, on_log=grow)
else:
    hist = trainer.learn(args.iters, log_every=10, on_log=log)

trainer.save(cap.dir / "chase_last.pt")
curve.close()
final = [h for h in hist if h["episodes"] > 0][-20:] or [{}]


def avg(key):
    v = [h[key] for h in final if h.get(key) == h.get(key)]
    return round(sum(v) / len(v), 3) if v else float("nan")


summary = {"iters": args.iters, "wall_s": round(time.time() - t0), "envs": args.num_envs,
           "mode": "teacher" if args.teacher else "vision", "params": n_par,
           "mean_return": avg("ep_return"), "touch_rate": avg(rate_key),
           "time_to_touch_s": avg("time_to_touch"), "seen": avg("seen"),
           "crash": avg("crash"), "oob": avg("oob"), "best_touch_rate": round(best, 3),
           "policy": str(cap.dir / "chase.pt")}
print("DONE " + json.dumps(summary), flush=True)
(cap.dir / "summary.json").write_text(json.dumps(summary, indent=1))
env.close()
app.close()
