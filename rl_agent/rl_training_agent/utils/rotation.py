"""以 Isaac Gym 的 xyzw 四元数计算逐控制步机体坐标旋转，避免欧拉俯仰回绕。"""
from __future__ import annotations

import math

import numpy as np

MAX_ROTATION_STEP = math.pi / 2


def quaternion_increment_numpy(previous, current):
    """返回前一机体坐标中的最短旋转向量与有效标志，拒绝零四元数和欠采样。"""
    previous, current = np.asarray(previous, dtype=float), np.asarray(current, dtype=float)
    if previous.ndim < 1 or previous.shape != current.shape or previous.shape[-1] != 4:
        raise ValueError("quaternions must have matching (..., 4) xyzw shapes")
    a_norm = np.linalg.norm(previous, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(current, axis=-1, keepdims=True)
    valid = (np.isfinite(previous).all(axis=-1) & np.isfinite(current).all(axis=-1) &
             (a_norm[..., 0] > 1e-8) & (b_norm[..., 0] > 1e-8))
    a = np.nan_to_num(previous) / np.maximum(a_norm, 1e-8)
    b = np.nan_to_num(current) / np.maximum(b_norm, 1e-8)
    vector = a[..., 3:] * b[..., :3] - b[..., 3:] * a[..., :3] - np.cross(a[..., :3], b[..., :3])
    scalar = (a * b).sum(axis=-1, keepdims=True)
    vector = vector * np.where(scalar < 0, -1., 1.)
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(length, np.abs(scalar))
    rotation = vector * (angle / np.maximum(length, 1e-8))
    valid &= np.isfinite(rotation).all(axis=-1) & (angle[..., 0] <= MAX_ROTATION_STEP + 1e-6)
    return np.where(valid[..., None], rotation, 0.), valid


def quaternion_increment_torch(previous, current):
    """批量计算与 NumPy 验收同语义的 xyzw 增量，不把张量复制到 CPU。"""
    import torch
    a_norm = torch.linalg.vector_norm(previous, dim=-1, keepdim=True)
    b_norm = torch.linalg.vector_norm(current, dim=-1, keepdim=True)
    valid = (torch.isfinite(previous).all(dim=-1) & torch.isfinite(current).all(dim=-1) &
             (a_norm[..., 0] > 1e-8) & (b_norm[..., 0] > 1e-8))
    a = torch.nan_to_num(previous) / torch.clamp(a_norm, min=1e-8)
    b = torch.nan_to_num(current) / torch.clamp(b_norm, min=1e-8)
    vector = a[..., 3:] * b[..., :3] - b[..., 3:] * a[..., :3] - torch.cross(a[..., :3], b[..., :3], dim=-1)
    scalar = (a * b).sum(dim=-1, keepdim=True)
    vector = vector * torch.where(scalar < 0, -torch.ones_like(scalar), torch.ones_like(scalar))
    length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, scalar.abs())
    rotation = vector * angle / torch.clamp(length, min=1e-8)
    valid &= torch.isfinite(rotation).all(dim=-1) & (angle[..., 0] <= MAX_ROTATION_STEP + 1e-6)
    return torch.where(valid[..., None], rotation, torch.zeros_like(rotation)), valid
