"""Local CPU smoke test for tree collisions, blast hits, and tank driving.

This intentionally avoids Isaac and the GPU droplet. It exercises the pure task
and driver code used by ChaseEnv, then writes inspectable traces and an animated
explosion under runs/<timestamp>-strike-smoke/.

    .venv/bin/python scripts/smoke_strike.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from vesper.capture.explosion import explosion_frame
from vesper.lab.chase_task import ChaseCfg, ChaseTask
from vesper.lab.vehicles import TankDriver
from vesper.worlds.heightmap import WorldMap


DT = 0.04
HALF = 50.0
TREE_RADIUS = 4.5
TREE_HEIGHT = 13.0
TANK_HALF_WIDTH = 1.675


def build_world(path: Path) -> WorldMap:
    n, cell = 201, 0.5
    axis = np.linspace(-HALF, HALF, n)
    x, y = np.meshgrid(axis, axis)
    ground = np.zeros((n, n), np.float32)
    tree = x * x + y * y <= TREE_RADIUS * TREE_RADIUS
    tree_z = np.where(tree, TREE_HEIGHT, 0.0).astype(np.float32)
    canopy = np.where(x * x + y * y <= 7.0**2, 15.0, 0.0).astype(np.float32)
    drivable = (~tree).astype(np.uint8)
    np.savez(path, half_m=np.float32(HALF), cell=np.float32(cell),
             ground_z=ground, obstacle_z=ground, tree_z=tree_z,
             canopy_z=canopy, canopy_d=(canopy > 0).astype(np.float32) * 0.8,
             drivable=drivable, concealed=np.zeros_like(drivable),
             road=np.zeros_like(drivable), road_yaw=ground,
             parking=np.zeros_like(drivable), park_yaw=ground)
    return WorldMap(path)


def state():
    return torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1, 3)


def run_tree_case(world: WorldMap):
    cfg = ChaseCfg(n_targets=1, arena_half=45.0)
    task = ChaseTask(world, cfg, 1, DT, 200)
    vel, quat, ang = state()
    target = torch.tensor([[[12.0, 0.0, 1.2]]])
    path, result = [], None
    for step, x in enumerate(torch.linspace(-14.0, 1.0, 31)):
        drone = torch.tensor([[float(x), 0.0, 3.0]])
        _, reward, term, info = task.step(
            drone, vel, quat, ang, target, torch.tensor([step]),
            detonated=torch.zeros(1, dtype=torch.bool))
        path.append([float(x), 0.0])
        if bool(term[0]):
            result = {"step": step, "x_m": round(float(x), 2),
                      "reward": round(float(reward[0]), 3),
                      "crash": bool(info["crash"][0]), "hit": bool(info["hit"][0])}
            break
    passed = result is not None and result["crash"] and not result["hit"]
    return {"name": "tree_collision_before_detonation", "pass": passed,
            "result": result, "path": path}


def detonation_case(world: WorldMap, drone_x: float, expected_hit: bool):
    cfg = ChaseCfg(n_targets=1, arena_half=45.0, blast_radius=4.0)
    task = ChaseTask(world, cfg, 1, DT, 200)
    vel, quat, ang = state()
    drone = torch.tensor([[drone_x, 15.0, 3.0]])
    target = torch.tensor([[[12.0, 15.0, 1.2]]])
    distance = float((target[0, 0] - drone[0]).norm())
    _, reward, term, info = task.step(
        drone, vel, quat, ang, target, torch.tensor([25]),
        detonated=torch.ones(1, dtype=torch.bool), crashed=torch.zeros(1, dtype=torch.bool),
        seen_px=torch.tensor([[10]]))
    hit = bool(info["hit"][0])
    passed = bool(term[0]) and hit == expected_hit and bool(info["miss"][0]) != expected_hit
    return {"name": "in_radius_hit" if expected_hit else "out_of_radius_miss", "pass": passed,
            "distance_m": round(distance, 3), "blast_radius_m": cfg.blast_radius,
            "hit": hit, "miss": bool(info["miss"][0]), "reward": round(float(reward[0]), 3),
            "drone": [drone_x, 15.0], "tank": [12.0, 15.0]}


def run_tank_case(world: WorldMap):
    driver = TankDriver(world, 1, arena_half=45.0, generator=torch.Generator().manual_seed(4))
    driver.cruise[0] = 4.0
    driver.on_road[0] = False
    driver.heading[0] = 0.0
    driver.goal[0] = torch.tensor([35.0, 0.0])
    driver.goal_age[0] = 0.0
    pos = torch.tensor([[-35.0, 0.0]])
    yaw = torch.zeros(1)
    speed = torch.zeros(1)
    path = [pos[0].tolist()]
    max_yaw_rate = 0.0
    for _ in range(int(24.0 / DT)):
        command, yaw_rate = driver.command(pos, yaw, speed, DT)
        pos = pos + command * DT
        yaw = yaw + yaw_rate * DT
        speed = command.norm(dim=1)
        max_yaw_rate = max(max_yaw_rate, abs(float(yaw_rate[0])))
        path.append(pos[0].tolist())
    points = torch.tensor(path)
    drivable_fraction = float(world.is_drivable(points[:, 0], points[:, 1]).float().mean())
    travelled = float((points[1:] - points[:-1]).norm(dim=1).sum())
    centre_clearance = float(torch.sqrt((points * points).sum(dim=1)).min()) - TREE_RADIUS
    hull_clearance = centre_clearance - TANK_HALF_WIDTH
    passed = travelled > 55.0 and drivable_fraction > 0.99 and hull_clearance > 0.0
    return {"name": "tank_avoids_tree_and_keeps_driving", "pass": passed,
            "travelled_m": round(travelled, 2), "drivable_fraction": round(drivable_fraction, 4),
            "minimum_centre_clearance_m": round(centre_clearance, 2),
            "minimum_hull_clearance_m": round(hull_clearance, 2),
            "max_yaw_rate_rad_s": round(max_yaw_rate, 3), "path": path}


def world_to_px(x: float, y: float, box):
    left, top, right, bottom = box
    return (left + (x + HALF) / (2 * HALF) * (right - left),
            bottom - (y + HALF) / (2 * HALF) * (bottom - top))


def draw_trace(out: Path, tree_case, hit_case, miss_case, tank_case):
    image = Image.new("RGB", (1200, 440), (20, 24, 22))
    draw = ImageDraw.Draw(image, "RGBA")
    panels = [(20, 50, 380, 410), (420, 50, 780, 410), (820, 50, 1180, 410)]
    titles = ("TREE CONTACT", "BLAST RADIUS", "TANK MOTION")
    for box, title in zip(panels, titles):
        draw.rectangle(box, fill=(31, 38, 34, 255), outline=(105, 122, 111, 255), width=2)
        draw.text((box[0], 20), title, fill=(235, 240, 234, 255))

    # Case 1: the tree collider stops the drone path before the tank.
    box = panels[0]
    tc = world_to_px(0, 0, box)
    rr = TREE_RADIUS / (2 * HALF) * (box[2] - box[0])
    draw.ellipse((tc[0] - rr, tc[1] - rr, tc[0] + rr, tc[1] + rr),
                 fill=(56, 120, 67, 220), outline=(105, 215, 121, 255), width=2)
    pts = [world_to_px(x, y, box) for x, y in tree_case["path"]]
    draw.line(pts, fill=(123, 211, 255, 255), width=4)
    draw.ellipse((pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6),
                 fill=(240, 91, 91, 255))
    tank = world_to_px(12, 0, box)
    draw.rectangle((tank[0] - 10, tank[1] - 6, tank[0] + 10, tank[1] + 6), fill=(188, 171, 90, 255))
    draw.text((box[0] + 10, box[3] - 24),
              f"PASS  crash at x={tree_case['result']['x_m']:.1f} m; no hit",
              fill=(124, 232, 151, 255))

    # Cases 2/3: exact 3D distances against the configured radius.
    box = panels[1]
    tank = world_to_px(*hit_case["tank"], box)
    blast_px = hit_case["blast_radius_m"] / (2 * HALF) * (box[2] - box[0])
    for case, color in ((hit_case, (124, 232, 151, 255)), (miss_case, (240, 91, 91, 255))):
        drone = world_to_px(*case["drone"], box)
        draw.ellipse((drone[0] - blast_px, drone[1] - blast_px,
                      drone[0] + blast_px, drone[1] + blast_px), outline=color, width=3)
        draw.ellipse((drone[0] - 4, drone[1] - 4, drone[0] + 4, drone[1] + 4), fill=color)
    draw.rectangle((tank[0] - 10, tank[1] - 6, tank[0] + 10, tank[1] + 6), fill=(188, 171, 90, 255))
    draw.text((box[0] + 10, box[3] - 42),
              f"PASS  {hit_case['distance_m']:.2f} m -> hit (+{hit_case['reward']:.1f})",
              fill=(124, 232, 151, 255))
    draw.text((box[0] + 10, box[3] - 24),
              f"PASS  {miss_case['distance_m']:.2f} m -> miss ({miss_case['reward']:.1f})",
              fill=(240, 150, 130, 255))

    # Case 4: integrated tank path bends around the same blocked tree cells.
    box = panels[2]
    tc = world_to_px(0, 0, box)
    draw.ellipse((tc[0] - rr, tc[1] - rr, tc[0] + rr, tc[1] + rr),
                 fill=(56, 120, 67, 220), outline=(105, 215, 121, 255), width=2)
    pts = [world_to_px(x, y, box) for x, y in tank_case["path"]]
    draw.line(pts, fill=(245, 204, 92, 255), width=4)
    draw.text((box[0] + 10, box[3] - 42),
              f"PASS  {tank_case['travelled_m']:.1f} m driven; {tank_case['drivable_fraction']*100:.1f}% drivable",
              fill=(124, 232, 151, 255))
    draw.text((box[0] + 10, box[3] - 24),
              f"minimum hull clearance {tank_case['minimum_hull_clearance_m']:.1f} m",
              fill=(220, 225, 218, 255))
    image.save(out)


def draw_explosion_gif(out: Path):
    bg = np.zeros((360, 640, 3), dtype=np.uint8)
    bg[:] = (35, 42, 36)
    base = Image.fromarray(bg)
    d = ImageDraw.Draw(base)
    d.rectangle((355, 172, 455, 212), fill=(74, 91, 56), outline=(170, 181, 118), width=3)
    d.rectangle((385, 150, 430, 177), fill=(74, 91, 56))
    d.line((430, 160, 505, 160), fill=(145, 151, 122), width=5)
    frames = [Image.fromarray(explosion_frame(np.asarray(base), (350, 185), i / 17, 78))
              for i in range(18)]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=45, loop=0, disposal=2)
    return {"frames": len(frames),
            "peak_changed_pixels": max(int((np.asarray(f) != np.asarray(base)).any(axis=2).sum())
                                       for f in frames)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-strike-smoke"
    out.mkdir(parents=True, exist_ok=True)
    world = build_world(out / "synthetic_world.npz")

    tree = run_tree_case(world)
    hit = detonation_case(world, 8.7, True)
    miss = detonation_case(world, 7.9, False)
    tank = run_tank_case(world)
    draw_trace(out / "smoke-traces.png", tree, hit, miss, tank)
    animation = draw_explosion_gif(out / "explosion.gif")
    cases = [tree, hit, miss, tank]
    report = {"pass": all(case["pass"] for case in cases), "cases": cases,
              "explosion_animation": animation,
              "artifacts": {"trace": "smoke-traces.png", "animation": "explosion.gif"}}
    # Paths are useful in the visual artifact but needlessly inflate the report.
    report["cases"][0].pop("path", None)
    report["cases"][3].pop("path", None)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"run": str(out), **{c["name"]: c["pass"] for c in cases},
                      "pass": report["pass"]}, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
