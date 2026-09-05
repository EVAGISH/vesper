"""Synthetic drone camera: raymarched frames from the world rasters, no Isaac.

The native sim publishes video by rendering what it *knows* -- the same rasters
the task flies against (ground_z, obstacle_z, canopy_z/d from WorldMap) draped
with the site's ground orthophoto, plus the vehicles and drones as world-space
boxes. One raymarch per pixel against the height field, pure torch, so it runs
wherever the sim runs and the downlink shows the truth of *this* sim rather
than a prettier lie: a tank under canopy is behind leaves here exactly when the
geometric sensor says it is. RTX-grade footage stays the Isaac session's job.

Budget: ~65k rays x 96 samples for a 256 px FPV tile -- tens of ms on a laptop
CPU, faster on MPS/CUDA; the warm session renders every few control steps for
an ~8 fps MJPEG downlink, same cadence the Isaac session published.

Frame conventions match vesper.lab.frames: world x east / y north / z up,
body x forward / y left / z up, camera pitched cam_pitch_deg below the nose.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from vesper.control.se3 import quat_to_rot

SKY_TOP = (96, 148, 201)
SKY_HORIZON = (196, 214, 228)
BUILDING = (158, 152, 146)
CANOPY = (52, 84, 46)
TANK_BODY = (72, 78, 58)


class RasterCamera:
    def __init__(self, world, ortho_path: str | None = None, res=(256, 256),
                 fov_half_deg=55.0, max_range=900.0, samples=96, device="cpu"):
        self.world = world
        self.W, self.H = int(res[0]), int(res[1])
        self.tan_fov = math.tan(math.radians(fov_half_deg))
        self.max_range = float(max_range)
        self.samples = int(samples)
        self.device = device

        # what a ray can hit (trees are visual: the crown is a surface here even
        # when the map gave them no colliders, so the forest is not invisible)
        self.surf = torch.maximum(world.solid_z, world.canopy_z)
        self.is_building = (world.obstacle_z > world.ground_z + 0.5)
        self.is_canopy = (world.canopy_z > world.ground_z + 0.5) & ~self.is_building

        if ortho_path:
            img = Image.open(ortho_path).convert("RGB")
            self.tex = torch.as_tensor(np.asarray(img, np.float32) / 255.0,
                                       device=device)                  # [Ht,Wt,3], row 0 = north
        else:
            self.tex = None

        # pixel grid -> camera-frame offsets (u right, v down), made once
        u = (torch.arange(self.W, device=device) + 0.5) / self.W * 2 - 1
        v = (torch.arange(self.H, device=device) + 0.5) / self.H * 2 - 1
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        self._uu = (uu * self.tan_fov).reshape(-1)                      # [P]
        self._vv = (vv * self.tan_fov).reshape(-1)
        # march stations, denser close in (where the ground detail is)
        t = torch.linspace(0.0, 1.0, self.samples, device=device)
        self._t = 2.0 + (t ** 1.7) * self.max_range                     # [S]

    # ------------------------------------------------------------- sampling
    def _tex_at(self, x, y):
        """Bilinear ortho lookup: nearest sampling of the 0.5 m/px source
        aliases into moire on roofs and reads as noise; bilinear is the fix."""
        h = self.world.half_m
        Ht, Wt = self.tex.shape[0], self.tex.shape[1]
        c = ((x + h) / (2 * h) * (Wt - 1)).clamp(0, Wt - 1.001)
        r = ((h - y) / (2 * h) * (Ht - 1)).clamp(0, Ht - 1.001)
        c0, r0 = c.floor().long(), r.floor().long()
        c1, r1 = c0 + 1, r0 + 1
        fc, fr = (c - c0).unsqueeze(-1), (r - r0).unsqueeze(-1)
        return (self.tex[r0, c0] * (1 - fc) * (1 - fr) + self.tex[r0, c1] * fc * (1 - fr)
                + self.tex[r1, c0] * (1 - fc) * fr + self.tex[r1, c1] * fc * fr)

    def _grid(self, field, x, y):
        r, c = self.world.nearest_cell(x, y)
        return field[r, c]

    # ------------------------------------------------------------- raymarch
    def render(self, cam_pos, quat, pitch_deg: float, targets=None, drones=None):
        """One frame. cam_pos [3], quat [4] wxyz (the airframe's), pitch below nose.

        targets [K,3] world (env-0 truth: video shows what a camera would see,
        not what the policy believes); drones [D,3] draws the rest of the
        fleet. Returns uint8 [H,W,3].
        """
        dev = self.device
        pos = torch.as_tensor(cam_pos, dtype=torch.float32, device=dev)
        q = torch.as_tensor(quat, dtype=torch.float32, device=dev).view(1, 4)
        R = quat_to_rot(q)[0]                                           # body -> world
        p = math.radians(pitch_deg)
        f_b = torch.tensor([math.cos(p), 0.0, -math.sin(p)], device=dev)
        r_b = torch.tensor([0.0, -1.0, 0.0], device=dev)
        u_b = torch.linalg.cross(r_b, f_b)
        f_w, r_w, u_w = R @ f_b, R @ r_b, R @ u_b                       # camera axes, world
        dirs = (f_w.view(1, 3) + self._uu.view(-1, 1) * r_w.view(1, 3)
                - self._vv.view(-1, 1) * u_w.view(1, 3))
        dirs = torch.nn.functional.normalize(dirs, dim=1)               # [P,3]

        P, S = dirs.shape[0], self.samples
        px = pos[0] + dirs[:, 0:1] * self._t.view(1, S)                 # [P,S]
        py = pos[1] + dirs[:, 1:2] * self._t.view(1, S)
        pz = pos[2] + dirs[:, 2:3] * self._t.view(1, S)
        below = pz < self.world.sample(self.surf, px, py)               # [P,S]
        hit_any = below.any(dim=1)
        first = below.float().argmax(dim=1).clamp(min=1)                # [P]
        t_hi = self._t[first]
        t_lo = self._t[first - 1]
        # bisection passes tighten silhouettes without more stations
        for _ in range(4):
            t_mid = 0.5 * (t_lo + t_hi)
            mx = pos[0] + dirs[:, 0] * t_mid
            my = pos[1] + dirs[:, 1] * t_mid
            mz = pos[2] + dirs[:, 2] * t_mid
            mid_below = mz < self.world.sample(self.surf, mx, my)
            t_hi = torch.where(mid_below, t_mid, t_hi)
            t_lo = torch.where(mid_below, t_lo, t_mid)
        t_hit = torch.where(hit_any, t_hi, torch.full_like(t_hi, self.max_range))
        hx = pos[0] + dirs[:, 0] * t_hit
        hy = pos[1] + dirs[:, 1] * t_hit
        hz = pos[2] + dirs[:, 2] * t_hit

        # ----------------------------------------------------------- shade
        sky_t = (dirs[:, 2].clamp(0, 0.6) / 0.6).view(-1, 1)
        sky = (torch.tensor(SKY_HORIZON, device=dev, dtype=torch.float32) / 255 * (1 - sky_t)
               + torch.tensor(SKY_TOP, device=dev, dtype=torch.float32) / 255 * sky_t)
        if self.tex is not None:
            ground_col = self._tex_at(hx, hy)
        else:
            g = ((hz - hz.min()) / (hz.max() - hz.min() + 1e-6)).view(-1, 1)
            ground_col = torch.tensor([0.35, 0.4, 0.3], device=dev) * (0.6 + 0.4 * g)
        # directional light: surface normal from the height-field gradient,
        # sun from the south-west and high -- real relief without shadow rays
        eps = self.world.cell
        gx = (self.world.sample(self.surf, hx + eps, hy)
              - self.world.sample(self.surf, hx - eps, hy)) / (2 * eps)
        gy = (self.world.sample(self.surf, hx, hy + eps)
              - self.world.sample(self.surf, hx, hy - eps)) / (2 * eps)
        inv = torch.rsqrt(gx * gx + gy * gy + 1.0)
        lambert = ((-gx * -0.45 - gy * -0.35 + 0.82) * inv).clamp(0.35, 1.0)
        shade = (0.55 + 0.55 * lambert).view(-1, 1)
        ground_col = ground_col * shade

        building = (self._grid(self.is_building, hx, hy) & hit_any).view(-1, 1)
        canopy = (self._grid(self.is_canopy, hx, hy) & hit_any).view(-1, 1)
        # roofs keep the ortho's own roof pixels; only facades go flat gray
        wall = building & (hz < self._grid(self.world.obstacle_z, hx, hy) - 0.7).view(-1, 1)
        bcol = torch.tensor(BUILDING, device=dev, dtype=torch.float32) / 255 * 0.8
        bshade = shade.clamp(0.8, 1.1)
        dens = self._grid(self.world.canopy_d, hx, hy).view(-1, 1)
        ccol = (torch.tensor(CANOPY, device=dev, dtype=torch.float32) / 255 * (0.65 + 0.5 * dens)
                + ground_col * 0.15)

        col = torch.where(wall, bcol * bshade, ground_col)
        col = torch.where(canopy, ccol, col)
        # a whisper of haze for depth; the old ^1.5 curve washed half the frame
        haze = 0.45 * (t_hit / self.max_range).clamp(0, 1).view(-1, 1) ** 2.2
        col = col * (1 - haze) + sky * haze
        col = torch.where(hit_any.view(-1, 1), col, sky)

        img = col.view(self.H, self.W, 3)
        depth = t_hit.view(self.H, self.W)

        # ------------------------------------------------- world-space marks
        axes = (f_w, r_w, u_w)
        if targets is not None and len(targets):
            body = torch.tensor(TANK_BODY, device=dev, dtype=torch.float32) / 255
            for t in targets:
                self._box(img, depth, pos, axes,
                          torch.as_tensor(t, dtype=torch.float32, device=dev),
                          3.6, 1.8, body, min_px=4)
        if drones is not None and len(drones):
            white = torch.tensor([0.95, 0.95, 0.98], device=dev)
            for d in drones:
                self._box(img, depth, pos, axes,
                          torch.as_tensor(d, dtype=torch.float32, device=dev),
                          0.5, 0.25, white, min_px=2)
        return (img.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    def _box(self, img, depth, cam_pos, axes, p_w, half_w, half_h, color, min_px=1):
        """Occlusion-tested screen-space sprite for a world object (tank, drone).

        True perspective scale, but never smaller than `min_px` half-extent: a
        7 m hull at 300 m is 2 px and unfindable, and the point of the downlink
        is that the operator can find it. Two-tone (lit deck over dark hull)
        with a dark rim so it reads as a vehicle against any ground.
        """
        f_w, r_w, u_w = axes
        rel = p_w - cam_pos
        zc = float(rel @ f_w)
        if zc < 1.5 or zc > self.max_range:
            return
        xc = float(rel @ r_w)
        yc = float(rel @ u_w)
        cx = int((xc / zc / self.tan_fov + 1) * 0.5 * self.W)
        cy = int((1 - yc / zc / self.tan_fov) * 0.5 * self.H)
        pw = max(min_px, int(half_w / zc / self.tan_fov * self.W * 0.5))
        ph = max(min_px, int(half_h / zc / self.tan_fov * self.H * 0.5))
        x0, x1 = max(0, cx - pw), min(self.W, cx + pw + 1)
        y0, y1 = max(0, cy - ph), min(self.H, cy + ph + 1)
        if x0 >= x1 or y0 >= y1:
            return
        patch = depth[y0:y1, x0:x1]
        vis = patch > (zc - 3.0)                       # terrain in front occludes
        if not bool(vis.any()):
            return
        h = y1 - y0
        deck = torch.zeros(h, x1 - x0, 3, device=img.device)
        deck[:] = color * 0.7                          # hull sides, dark
        deck[: max(1, int(h * 0.45))] = color * 1.35   # sun-lit deck on top
        deck[0, :] = color * 0.25                      # dark rim
        deck[-1, :] = color * 0.25
        deck[:, 0] = color * 0.25
        deck[:, -1] = color * 0.25
        img[y0:y1, x0:x1][vis] = deck.clamp(0, 1)[vis]
        depth[y0:y1, x0:x1][vis] = zc
