import argparse
import glob
import os

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


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


def run_inference(image_dir, output_root, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_name).to(device=device)
    model.eval()

    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    dataset_name = os.path.basename(os.path.normpath(image_dir))
    output_dir = os.path.join(output_root, dataset_name, "results_output")

    with torch.no_grad():
        prediction = model.inference(image_paths)

    save_per_frame_outputs(prediction, output_dir)
    print(f"Saved {len(image_paths)} frames to {output_dir}")


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
    args = parser.parse_args()

    run_inference(args.image_dir, args.output_root, args.model_name)