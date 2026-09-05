"""Log one deterministic episode of a trained policy for after-action review.

Every run should carry a visual record. This rolls a policy out for one episode
on env 0 of a NativeSearchEnv and writes two artifacts into the run dir:

  trajectory.parquet   drone-0 pose over time -- the Runs tab's top-down plot
                       already reads this (vesper.record.trajectory schema).
  replay.json          every drone + every target + found/reached + agl over
                       time -- the source the tactical replay renderer turns
                       into the impressive per-run video.

Pure torch + the native env, so it runs wherever training ran (Mac included);
no Isaac. Photoreal replay is a separate, heavier track that consumes the same
replay.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from vesper.record.trajectory import TrajectoryWriter


POOL = 128        # envs buffered as hero candidates (memory stays a few tens of MB)
SWARM = 16        # drones written per replay frame (hero + the first few others)


def log_episode(env, policy, run_dir, world_name: str, max_steps: int | None = None,
                reach_radius: float | None = None):
    """Roll `policy` (anything with .action(obs, deterministic=True) or an
    ActorCritic) for one episode, writing trajectory.parquet + replay.json
    into run_dir. Returns the number of frames logged.

    The episode is logged from the perspective of a "hero" env: the first of
    the buffered pool to actually neutralize a target (falling back to the one
    that detects the most, then env 0). Every env rolls anyway, so this makes
    the recorded eval show a kill whenever *any* candidate scores one, instead
    of gambling on env 0's seed.

    reach_radius temporarily widens the kill sphere for this eval rollout only
    (the same demo lever the live warm session uses), so the recorded episode
    reliably ends in a neutralization even when training used a tight radius.
    The training config is restored before returning.
    """
    run_dir = Path(run_dir)
    dt = env._dt
    steps = int(max_steps or env.max_episode_length)
    P = min(env.num_envs, POOL)

    act_fn = _action_fn(policy)
    buf = []                                  # per-step cpu snapshots of the pool

    old_reach = env.tcfg.reach_radius
    old_len = env.max_episode_length
    if reach_radius is not None:
        env.tcfg.reach_radius = float(reach_radius)
    if steps > env.max_episode_length:
        env.max_episode_length = steps

    killed = torch.zeros(P, dtype=torch.bool)     # reached during its first episode
    ended = torch.zeros(P, dtype=torch.bool)      # first episode is over
    try:
        obs = env.ppo_reset()
        for k in range(steps):
            with torch.no_grad():
                act = act_fn(obs)
            obs, _, done, info = env.ppo_step(act)
            pos, _, quat, _ = env.flight_state()
            buf.append({
                "pos": pos[:P].cpu().clone(), "quat": quat[:P].cpu().clone(),
                "tp": env.target_pos[:P].cpu().clone(),
                "known": env.task.known[:P].cpu().clone(),
                "reached": env.task.reached[:P].cpu().clone(),
                "agl": info["agl"][:P].cpu().clone(), "done": done[:P].cpu().clone(),
            })
            killed |= buf[-1]["reached"].any(dim=1) & ~ended
            ended |= buf[-1]["done"].bool()
            # stop once a killer's episode is complete, or every candidate's
            # first (kill-less) episode has run its course
            if bool((killed & ended).any()) or bool(ended.all()):
                break
    finally:
        env.tcfg.reach_radius = old_reach
        env.max_episode_length = old_len

    hero, end = _pick_hero(buf, P)
    tw = TrajectoryWriter(run_dir)
    order = [hero] + [e for e in range(P) if e != hero][: SWARM - 1]
    frames = []
    for k in range(end):
        b = buf[k]
        t = k * dt
        tw.append(t, b["pos"][hero].numpy(), b["quat"][hero].numpy())
        tp, known, reached = b["tp"][hero], b["known"][hero], b["reached"][hero]
        frames.append({
            "t": round(t, 2),
            "d": [[round(float(x), 1), round(float(y), 1), round(float(z), 1)]
                  for x, y, z in b["pos"][order].tolist()],
            "hdg": [round(float(h), 3) for h in _yaws(b["quat"][order]).tolist()],
            "tg": [[round(float(tp[i, 0]), 1), round(float(tp[i, 1]), 1),
                    int(bool(known[i])), int(bool(reached[i]))] for i in range(env.k)],
            "agl": round(float(b["agl"][hero]), 1),
        })
    tw.close()

    (run_dir / "replay.json").write_text(json.dumps({
        "world": world_name, "half_m": float(env.world.half_m), "dt": dt,
        "targets": int(env.k), "frames": frames,
    }, separators=(",", ":")))
    return len(frames)


def _pick_hero(buf, P):
    """(hero_env, end_step): the pool env whose first episode we replay.

    Best kill count at episode end, tiebroken by earliest first kill; if nobody
    killed, the env that detected the most; else env 0. end_step is the step
    its first episode terminated (frames at/after it are post-auto-reset).
    """
    S = len(buf)
    done = torch.stack([b["done"].bool() for b in buf])            # [S,P]
    reach = torch.stack([b["reached"] for b in buf])               # [S,P,K]
    first_done = torch.full((P,), S, dtype=torch.long)
    for e in range(P):
        nz = done[:, e].nonzero()
        if len(nz):
            first_done[e] = int(nz[0])
    best = None                                                    # (kills, -first_kill, e)
    for e in range(P):
        fd = int(first_done[e])
        if fd == 0:
            continue
        kills = int(reach[fd - 1, e].sum())
        if kills == 0:
            continue
        fk = int(reach[:fd, e].any(dim=1).nonzero()[0])
        cand = (kills, -fk, -e)
        if best is None or cand > best:
            best = cand
    if best is not None:
        e = -best[2]
        return e, int(first_done[e])
    finds = [int(buf[int(first_done[e]) - 1]["known"][e].sum()) if first_done[e] > 0 else -1
             for e in range(P)]
    e = int(torch.tensor(finds).argmax())
    return e, max(1, int(first_done[e]))


def _action_fn(policy):
    if hasattr(policy, "action"):          # a PPO trainer
        return lambda obs: policy.action(obs, deterministic=True)
    if hasattr(policy, "act"):             # (ActorCritic, norm) or bare ActorCritic
        return lambda obs: policy.act(obs)[0]
    raise TypeError("policy needs .action() or .act()")


def _yaws(quat):
    """World yaw [N] from wxyz quats, for the replay's per-drone heading."""
    w, x, y, z = quat.unbind(dim=1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).cpu()
