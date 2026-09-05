"""Train the pursuit policy: a drone learns to run down and reach a moving tank
vehicle in minimum time, on the stable plane-terrain throughput lane.

    /isaac-sim/python.sh scripts/train_pursuit.py --num_envs 4096 --iters 600 --headless

Writes runs/<id>/pursuit.pt (policy+obs-norm) and runs/<id>/curve.jsonl.
Self-contained PPO (vesper.lab.ppo) -- no rsl_rl/skrl dependency.
"""
import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iters", type=int, default=600)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--target_speed", type=float, default=4.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--vehicle", default=None, help="tank | path to a custom USD")
parser.add_argument("--tag", default="pursuit-train")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app


from vesper.capture import RunCapture  # noqa: E402
from vesper.lab.ppo import PPO, PPOCfg  # noqa: E402
from vesper.lab.pursuit_env import PursuitEnv, PursuitEnvCfg  # noqa: E402


class Adapter:
    """PPO <-> PursuitEnv (gym-style) bridge."""
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


cfg = PursuitEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.pursuit = {"target_speed": args.target_speed}
cfg.vehicle_model = args.vehicle
cfg.scene.env_spacing = 110.0   # one drone + one vehicle per env, well separated
env = PursuitEnv(cfg, seed=args.seed)
adapter = Adapter(env)

ppo = PPO(adapter, PPOCfg(horizon=args.horizon, lr=args.lr), device=env.device, seed=args.seed)
cap = RunCapture(args.tag)
curve = open(cap.dir / "curve.jsonl", "w")
print(f"train: {args.num_envs} envs x {args.iters} iters, target_speed={args.target_speed} m/s -> {cap.dir}",
      flush=True)

t0 = time.time()
best = -1.0


def log(row):
    global best
    curve.write(json.dumps(row) + "\n"); curve.flush()
    steps = (row["iter"] + 1) * args.horizon * args.num_envs
    sps = steps / (time.time() - t0 + 1e-9)
    print(f"it {row['iter']:4d} | return {row['ep_return']:7.2f} | intercept {row['intercept_rate']:.2f} "
          f"| t_int {row['time_to_intercept']:5.2f}s | eps {row['episodes']:5d} | {sps/1e3:.0f}k step/s",
          flush=True)
    hr = row["intercept_rate"]
    if hr == hr and hr >= best:  # not NaN and best-so-far
        best = hr
        ppo.save(cap.dir / "pursuit.pt")


hist = ppo.learn(args.iters, log_every=10, on_log=log)
ppo.save(cap.dir / "pursuit_last.pt")
curve.close()
wall = time.time() - t0
final = [h for h in hist if h["episodes"] > 0][-20:] or [{"ep_return": float("nan"), "intercept_rate": float("nan")}]
summary = {
    "iters": args.iters,
    "wall_s": round(wall),
    "mean_return": round(sum(h["ep_return"] for h in final) / len(final), 2),
    "intercept_rate": round(sum(h["intercept_rate"] for h in final) / len(final), 3),
    "best_intercept": round(best, 3),
    "policy": str(cap.dir / "pursuit.pt"),
}
print("DONE " + json.dumps(summary), flush=True)
env.close()
app.close()
