import argparse
import glob
import os

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


def to_4x4(ext: np.ndarray) -> np.ndarray:
    if ext.shape[-2:] == (4, 4):
        return ext
    out = np.eye(4, dtype=ext.dtype)
    out[:3, :4] = ext
    return out


def compute_relative_pose(E_current: np.ndarray, E_target: np.ndarray) -> np.ndarray:
    """Pose of `target` camera expressed in `current` camera frame.

    Extrinsics are W2C (X_cam = E @ X_world). Returns a 4x4 such that
    points in target-cam coords map to current-cam coords via:
        X_current = T_current_target @ X_target
    """
    E_cur = to_4x4(E_current)
    E_tgt = to_4x4(E_target)
    return E_cur @ np.linalg.inv(E_tgt)


def to_planar_pose(T_current_target: np.ndarray):
    """Project a 4x4 SE(3) into a ground-plane SE(2) pose (x, z, yaw).

    Assumes OpenCV camera convention: +X right, +Y down, +Z forward.
    Ground plane is X-Z; "up" axis is Y. Yaw = rotation about Y.

    Returns:
        dx     : forward translation in current-cam frame (meters / model units)
        dy     : lateral translation (right is +)
        yaw    : yaw angle in radians (target heading wrt current heading)
    """
    R = T_current_target[:3, :3]
    t = T_current_target[:3, 3]

    # Forward axis of target in current-cam frame is R @ [0,0,1] = R[:,2].
    # Yaw is the angle this forward vector makes in the X-Z plane.
    fwd = R[:, 2]
    yaw = np.arctan2(fwd[0], fwd[2])  # angle from +Z toward +X

    # Map camera-frame translation (X right, Z forward) to a top-down
    # convention where forward is +x and right is +y (REP-103-ish but on the
    # ground plane viewed from above):
    forward = t[2]
    right = t[0]
    return forward, right, yaw


def find_pair(image_dir: str):
    images = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
    )
    cur = [p for p in images if "current" in os.path.basename(p).lower()]
    tgt = [p for p in images if "target" in os.path.basename(p).lower()]
    if cur and tgt:
        return cur[0], tgt[0]
    if len(images) >= 2:
        return images[0], images[1]
    raise FileNotFoundError(f"Need at least 2 images in {image_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="./test_images")
    parser.add_argument("--model_name", default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--output", default=None, help="Optional .npz to save the relative pose")
    args = parser.parse_args()

    cur_path, tgt_path = find_pair(args.image_dir)
    print(f"current: {cur_path}")
    print(f"target : {tgt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.model_name).to(device=device)
    model.eval()

    with torch.no_grad():
        # Order matters: index 0 = current, index 1 = target.
        prediction = model.inference([cur_path, tgt_path])

    E_cur = prediction.extrinsics[0]
    E_tgt = prediction.extrinsics[1]
    T_cur_tgt = compute_relative_pose(E_cur, E_tgt)

    R = T_cur_tgt[:3, :3]
    t = T_cur_tgt[:3, 3]
    # Rotation angle from trace
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))

    np.set_printoptions(precision=4, suppress=True)
    print("\nT_current_target (target pose in current-camera frame):")
    print(T_cur_tgt)
    print(f"\nrotation angle: {angle_deg:.3f} deg")
    print(f"translation   : {t}  (norm={np.linalg.norm(t):.4f})")

    forward, right, yaw = to_planar_pose(T_cur_tgt)
    print("\n--- 2D planar (top-down) pose ---")
    print(f"forward (+Z): {forward:+.4f}")
    print(f"right   (+X): {right:+.4f}")
    print(f"yaw         : {np.degrees(yaw):+.3f} deg  ({yaw:+.4f} rad)")
    print(f"distance    : {np.hypot(forward, right):.4f}")
    print(f"bearing     : {np.degrees(np.arctan2(right, forward)):+.3f} deg "
          "(0=straight ahead, +=right)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        np.savez(
            args.output,
            T_current_target=T_cur_tgt,
            E_current=to_4x4(E_cur),
            E_target=to_4x4(E_tgt),
            planar_forward=forward,
            planar_right=right,
            planar_yaw_rad=yaw,
            current_image=cur_path,
            target_image=tgt_path,
        )
        print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
