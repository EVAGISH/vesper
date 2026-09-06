"""Frames, actions and sensor helpers shared by every drone task.

Pure torch. Frame: world metres, x east, y north, z up. Body: x forward,
y left, z up. Quaternions are wxyz.

  proprio        what the airframe measures with no GPS and no map
  setpoint       body-frame guidance action -> world setpoint for the SE3 loop
  camera_axis    where the body-fixed, forward-pitched camera looks
  sensor_pose    the same, as a pose for a render camera
  seg_lookup /   instance-segmentation ids -> per-target pixel counts
  seg_counts
"""
from __future__ import annotations

import math
import re

import torch

from vesper.control.se3 import quat_to_rot


PROPRIO_DIM = 11


def tilt_from_quat(quat):
    """Angle [N] between body +z and world +z, from a wxyz quaternion."""
    w, x, y, z = quat.unbind(dim=1)
    return torch.arccos((1 - 2 * (x * x + y * y)).clamp(-1.0, 1.0))


def yaw_from_quat(quat):
    w, x, y, z = quat.unbind(dim=1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def setpoint(drone_pos_w, action, yaw, look_ahead: float):
    """Body-frame guidance action [N,3] in [-1,1] -> world setpoint for the SE3 loop.

    Action axes are forward / left / up in the airframe's yaw frame, so the
    policy never expresses anything in world coordinates. The inner loop's PD
    turns the offset into a velocity, which makes this a velocity command in
    all but name.
    """
    off = torch.tanh(action) * look_ahead
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx = off[:, 0] * c - off[:, 1] * s
    dy = off[:, 0] * s + off[:, 1] * c
    return drone_pos_w + torch.stack([dx, dy, off[:, 2]], dim=1)


def camera_axis(quat, pitch_rad: float):
    """World-frame unit vector [N,3] the body-fixed camera looks along."""
    R = quat_to_rot(quat)
    d = torch.tensor([math.cos(pitch_rad), 0.0, -math.sin(pitch_rad)],
                     device=quat.device, dtype=quat.dtype)
    return R @ d


def quat_mul(a, b):
    """Hamilton product of wxyz quaternions, [N,4] x [N,4]."""
    aw, ax, ay, az = a.unbind(dim=1)
    bw, bx, by, bz = b.unbind(dim=1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=1)


def sensor_pose(drone_pos, quat, pitch_deg: float, offset=(0.0, 0.0, 0.0)):
    """World pose (pos [N,3], quat wxyz [N,4]) of the body-fixed camera.

    Isaac's "world" camera convention looks along +x with +z up, so the camera
    frame is the body frame pitched down about body y. A render camera placed
    here films exactly what the policy's tiled camera sees; the flight scripts
    use it so the video and the observation are the same lens.
    """
    p = math.radians(pitch_deg) / 2.0
    q_pitch = torch.tensor([math.cos(p), 0.0, math.sin(p), 0.0], device=quat.device,
                           dtype=quat.dtype).expand_as(quat)
    R = quat_to_rot(quat)
    off = torch.tensor(offset, device=quat.device, dtype=quat.dtype)
    return drone_pos + R @ off, quat_mul(quat, q_pitch)


def seg_lookup(id_to_labels: dict, pattern: str, n_slots: int, groups: int | None = None):
    """Instance-segmentation id -> target index, from the renderer's idToLabels.

    `pattern` is a regex with two groups, (env, slot), matched against each prim
    path; the table maps an id to env * n_slots + slot and everything else (sky,
    ground, trees, the drone itself) to -1. Ids that name an env beyond `groups`
    are dropped too: those vehicles are dormant stand-ins, not targets.
    Keys arrive as ints or strings depending on the annotator; both are handled.
    """
    rx = re.compile(pattern)
    ids, vals = [], []
    for k, label in id_to_labels.items():
        m = rx.search(str(label))
        if not m:
            continue
        e, s = int(m.group(1)), int(m.group(2))
        if groups is not None and e >= groups:
            continue
        ids.append(int(k)); vals.append(e * n_slots + s)
    size = (max(ids) + 1) if ids else 1
    table = torch.full((size,), -1, dtype=torch.long)
    if ids:
        table[torch.tensor(ids)] = torch.tensor(vals)
    return table


def seg_counts(seg, table, group_of_env, n_slots: int):
    """Pixels of each of the env's own targets in its frame.

    seg [N,H,W] (or [N,H,W,1]) integer instance ids; table from seg_lookup;
    group_of_env [N] the env whose vehicles are this env's targets.
    Returns [N, n_slots] int64 pixel counts. A vehicle belonging to another
    group is scenery: it is in the frame, but it is not a sighting.
    """
    n = seg.shape[0]
    ids = seg.reshape(n, -1).long()
    ids = ids.clamp(min=0, max=table.numel() - 1)
    hit = table.to(seg.device)[ids]                                       # [N,P]
    grp = torch.div(hit, n_slots, rounding_mode="floor")
    slot = hit % n_slots
    ok = (hit >= 0) & (grp == group_of_env.view(n, 1))
    out = torch.zeros(n, n_slots, dtype=torch.long, device=seg.device)
    out.scatter_add_(1, slot.clamp(min=0), ok.long())
    return out




def proprio(drone_vel, quat, ang_vel_b, agl, time_frac):
    """[N, 11] -- what the airframe measures with no GPS and no map.

    body-frame velocity / 15   (3)   optical-flow / VIO class estimate
    gravity in the body frame  (3)   roll and pitch as the IMU knows them;
                                     deliberately no yaw: there is no compass
    body rates                 (3)
    height above ground / 100  (1)   downward rangefinder
    episode clock              (1)
    """
    R = quat_to_rot(quat)
    v_b = (R.transpose(1, 2) @ drone_vel.unsqueeze(2)).squeeze(2)
    g_b = R[:, 2, :]                       # world +z expressed in body axes
    return torch.cat([v_b / 15.0, g_b, ang_vel_b,
                      (agl / 100.0).unsqueeze(1), time_frac.unsqueeze(1)], dim=1)
