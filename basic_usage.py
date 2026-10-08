import argparse
import glob
import os

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from da3_streaming.loop_utils.alignment_torch import depth_to_point_cloud_optimized_torch


def infer_dataset_name(image_dir: str) -> str:
    norm_dir = os.path.normpath(image_dir)
    base = os.path.basename(norm_dir)
    # If the input points to an image subdir (common: ".../<dataset>/images"),
    # use the parent folder as dataset name.
    if base.lower() in {"images", "image", "imgs"}:
        parent = os.path.basename(os.path.dirname(norm_dir))
        if parent:
            return parent
    return base


def save_per_frame_outputs(prediction, output_dir, start_index=0):
    os.makedirs(output_dir, exist_ok=True)
    for local_idx in range(prediction.depth.shape[0]):
        frame_idx = start_index + local_idx
        frame_path = os.path.join(output_dir, f"frame_{frame_idx}.npz")
        np.savez_compressed(
            frame_path,
            image=prediction.processed_images[local_idx],
            depth=prediction.depth[local_idx],
            conf=prediction.conf[local_idx],
            intrinsics=prediction.intrinsics[local_idx],
            extrinsics=prediction.extrinsics[local_idx],
        )


def save_confident_pointcloud_batch_local(
    points, colors, confs, output_path, conf_threshold, sample_ratio=1.0
):
    points = points.reshape(-1, 3).astype(np.float32, copy=False)
    colors = colors.reshape(-1, 3).astype(np.uint8, copy=False)
    confs = confs.reshape(-1).astype(np.float32, copy=False)

    conf_mask = (confs >= conf_threshold) & (confs > 1e-5)
    points = points[conf_mask]
    colors = colors[conf_mask]
    confs = confs[conf_mask]

    if points.shape[0] == 0:
        raise ValueError("No points left after confidence filtering.")

    if sample_ratio < 1.0:
        sample_ratio = float(np.clip(sample_ratio, 0.0, 1.0))
        sample_size = max(1, int(points.shape[0] * sample_ratio))
        idx = np.random.choice(points.shape[0], sample_size, replace=False)
        points = points[idx]
        colors = colors[idx]
        confs = confs[idx]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def save_combined_ply(prediction, output_dir, conf_threshold_coef=1.0, sample_ratio=1.0):
    os.makedirs(output_dir, exist_ok=True)
    points = depth_to_point_cloud_optimized_torch(
        prediction.depth, prediction.intrinsics, prediction.extrinsics
    )
    confs = prediction.conf
    conf_threshold = float(np.mean(confs)) * conf_threshold_coef
    ply_path = os.path.join(output_dir, "combined_pcd.ply")
    save_confident_pointcloud_batch_local(
        points=points,
        colors=prediction.processed_images,
        confs=confs,
        output_path=ply_path,
        conf_threshold=conf_threshold,
        sample_ratio=sample_ratio,
    )
    return ply_path


def run_inference(image_dir, output_root, model_name, conf_threshold_coef, sample_ratio):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_name).to(device=device)
    model.eval()

    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    dataset_name = infer_dataset_name(image_dir)
    output_base = os.path.join(output_root, dataset_name)
    output_dir = os.path.join(output_base, "results_output")
    pcd_dir = os.path.join(output_base, "pcd")

    with torch.no_grad():
        prediction = model.inference(image_paths)

    save_per_frame_outputs(prediction, output_dir)
    ply_path = save_combined_ply(
        prediction, pcd_dir, conf_threshold_coef=conf_threshold_coef, sample_ratio=sample_ratio
    )
    print(f"Saved {len(image_paths)} frames to {output_dir}")
    print(f"Saved combined point cloud to {ply_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Anything 3 per-frame output")
    parser.add_argument("--image_dir", type=str, required=True, help="Folder of images")
    parser.add_argument(
        "--output_root", type=str, default="outputs", help="Root output folder"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--conf_threshold_coef",
        type=float,
        default=1.0,
        help="Confidence threshold multiplier for PLY export",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Sampling ratio for PLY export (0 < r <= 1)",
    )
    args = parser.parse_args()

    run_inference(
        args.image_dir,
        args.output_root,
        args.model_name,
        args.conf_threshold_coef,
        args.sample_ratio,
    )