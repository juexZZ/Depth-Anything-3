from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import os

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"rmse requires same shape, got {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    x, y, z = qx, qy, qz
    w = qw
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def parse_colmap_images_txt(path: str) -> Dict[str, np.ndarray]:
    """
    Parse COLMAP images.txt and return {basename -> camera_center(3,)}.
    """
    centers: Dict[str, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Image line has: IMAGE_ID, QW QX QY QZ, TX TY TZ, CAMERA_ID, NAME
            if len(parts) < 10:
                continue
            try:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
            except ValueError:
                continue
            name = parts[9]
            R = _quat_to_rot(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=np.float64)
            center = -R.T @ t
            centers[os.path.basename(name)] = center
            # Skip the next line of 2D points if present (handled by loop)
    return centers


@dataclass
class Sim3:
    s: float
    R: np.ndarray
    t: np.ndarray

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.s * self.R
        T[:3, 3] = self.t
        return T


def umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """
    Umeyama alignment: find Sim(3) s, R, t such that dst ~= s * R * src + t.
    """
    X = np.asarray(src, dtype=np.float64)
    Y = np.asarray(dst, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape != Y.shape:
        raise ValueError(f"umeyama_sim3 expects (N,3) arrays, got {X.shape} and {Y.shape}")

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    cov = (Yc.T @ Xc) / X.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = U @ Vt

    if with_scale:
        var_x = np.mean(np.sum(Xc**2, axis=1))
        s = float(np.trace(np.diag(S)) / max(var_x, 1e-12))
    else:
        s = 1.0

    t = mu_y - s * (R @ mu_x)
    return Sim3(s=s, R=R.astype(np.float64), t=t.astype(np.float64))
