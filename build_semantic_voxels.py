import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import torch

# Ensure da3_streaming is on sys.path so loop_utils resolves when importing sim3utils.
_DA3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "da3_streaming")
if _DA3_DIR not in sys.path:
    sys.path.insert(0, _DA3_DIR)

from depth_anything_3.api import DepthAnything3
# from da3_streaming.da3_streaming import depth_to_point_cloud_vectorized
from semantic_voxels.semantic_voxel import SemanticVoxel, SemanticVoxelMap
from loop_utils.sim3utils import save_confident_pointcloud_batch

def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    depth: [N, H, W] numpy array or torch tensor
    intrinsics: [N, 3, 3] numpy array or torch tensor
    extrinsics: [N, 3, 4] (w2c) numpy array or torch tensor
    Returns: point_cloud_world: [N, H, W, 3] same type as input
    """
    input_is_numpy = False
    if isinstance(depth, np.ndarray):
        input_is_numpy = True

        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)

        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intrinsics_tensor = intrinsics_tensor.to(device)
            extrinsics_tensor = extrinsics_tensor.to(device)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    # main logic

    N, H, W = depth_tensor.shape

    device = depth_tensor.device

    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)  # [N, 3, 3]
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()

    return point_cloud_world


def _resolve_feature_path(
    image_path: str, feature_dir: str, feature_suffix: str, feature_ext: str
) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    feature_ext = feature_ext if feature_ext.startswith(".") else f".{feature_ext}"
    return os.path.join(feature_dir, f"{stem}{feature_suffix}{feature_ext}")


def _load_feature_map(
    feature_path: str, feature_key: Optional[str], image_size: Tuple[int, int]
) -> np.ndarray:
    with np.load(feature_path) as data:
        feat = None
        if feature_key:
            if feature_key not in data:
                raise KeyError(f"Feature key '{feature_key}' not found in {feature_path}")
            feat = data[feature_key]
        else:
            for key in ("feat", "feature", "features", "embedding", "embeddings"):
                if key in data:
                    feat = data[key]
                    break
            if feat is None:
                if len(data.files) == 1:
                    feat = data[data.files[0]]
                else:
                    raise KeyError(
                        f"Could not infer feature array key in {feature_path}; keys={data.files}"
                    )
    if feat.ndim != 3:
        raise ValueError(f"Feature map must be (H,W,d); got shape {feat.shape}")
    if feat.shape[0] != image_size[0] or feat.shape[1] != image_size[1]:
        import cv2

        feat = cv2.resize(
            feat.astype(np.float32),
            (image_size[1], image_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return feat.astype(np.float32)


def build_semantic_voxel_map(
    images: List[str],
    predictions,
    feature_dir: str,
    feature_key: Optional[str],
    feature_suffix: str,
    feature_ext: str,
    voxel_size: float,
    conf_threshold_coef: float,
    save_pcd_path: Optional[str] = None,
    pcd_sample_ratio: float = 0.015,
) -> SemanticVoxelMap:
    depth = predictions.depth
    conf = predictions.conf
    intrinsics = predictions.intrinsics
    extrinsics = predictions.extrinsics

    if depth.ndim == 2:
        depth = depth[None, ...]
    if conf.ndim == 2:
        conf = conf[None, ...]
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None, ...]
    if extrinsics.ndim == 2:
        extrinsics = extrinsics[None, ...]

    points_world = depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics)
    points_world = np.asarray(points_world, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    colors = getattr(predictions, "processed_images", None)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
    print("points_world shape: ", points_world.shape)
    print("conf shape: ", conf.shape)

    conf_threshold = float(np.mean(conf)) * conf_threshold_coef
    frame_name_maps: Dict[str, Dict[str, str]] = {"0": {}}

    voxel_sums: Dict[Tuple[int, int, int], np.ndarray] = {}
    voxel_counts: Dict[Tuple[int, int, int], int] = {}
    voxel_contribs: Dict[Tuple[int, int, int], set] = {}

    image_h, image_w = int(points_world.shape[1]), int(points_world.shape[2])

    for idx, image_path in tqdm(enumerate(images), desc="Building semantic voxel map"):
        frame_id = str(idx)
        frame_name_maps["0"][frame_id] = os.path.basename(image_path)

        mask = conf[idx] >= conf_threshold
        if not np.any(mask):
            continue

        feature_path = _resolve_feature_path(
            image_path, feature_dir, feature_suffix, feature_ext
        )
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Missing feature map: {feature_path}")

        feat = _load_feature_map(feature_path, feature_key, (image_h, image_w))
        pts = points_world[idx]

        pts_sel = pts[mask]
        sem_sel = feat[mask].astype(np.float32)
        if pts_sel.shape[0] == 0:
            continue

        finite_mask = np.isfinite(pts_sel).all(axis=1) & np.isfinite(sem_sel).all(axis=1)
        if not np.all(finite_mask):
            pts_sel = pts_sel[finite_mask]
            sem_sel = sem_sel[finite_mask]
        if pts_sel.shape[0] == 0:
            continue

        # Robust bbox filter (percentiles)
        lo = np.percentile(pts_sel, 0.5, axis=0)
        hi = np.percentile(pts_sel, 99.5, axis=0)
        bbox_mask = (pts_sel >= lo).all(axis=1) & (pts_sel <= hi).all(axis=1)
        if not np.all(bbox_mask):
            pts_sel = pts_sel[bbox_mask]
            sem_sel = sem_sel[bbox_mask]
        if pts_sel.shape[0] == 0:
            continue

        # outlier filtering: points that fall into very sparse coarse cells are likely outliers.
        coarse_cell = float(voxel_size) * 3.0
        min_points_per_cell = 10
        if coarse_cell > 0.0 and pts_sel.shape[0] > 0:
            coarse_coords = np.floor(pts_sel / coarse_cell).astype(np.int64)
            _, inv, counts = np.unique(
                coarse_coords, axis=0, return_inverse=True, return_counts=True
            )
            dense_mask = counts[inv] >= min_points_per_cell
            if not np.all(dense_mask):
                pts_sel = pts_sel[dense_mask]
                sem_sel = sem_sel[dense_mask]
        if pts_sel.shape[0] == 0:
            continue

        voxel_coords = np.floor(pts_sel / voxel_size).astype(np.int64)
        unique_coords, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
        num_vox = unique_coords.shape[0]
        d = sem_sel.shape[1]
        feat_sum = np.zeros((num_vox, d), dtype=np.float32)
        counts = np.zeros((num_vox,), dtype=np.int64)
        np.add.at(feat_sum, inverse, sem_sel)
        np.add.at(counts, inverse, 1)

        for v_i, coord in enumerate(unique_coords):
            key = (int(coord[0]), int(coord[1]), int(coord[2]))
            if key not in voxel_sums:
                voxel_sums[key] = feat_sum[v_i].copy()
                voxel_counts[key] = int(counts[v_i])
                voxel_contribs[key] = set()
            else:
                voxel_sums[key] += feat_sum[v_i]
                voxel_counts[key] += int(counts[v_i])
            voxel_contribs[key].add((0, frame_id))

    if save_pcd_path is not None:
        os.makedirs(os.path.dirname(save_pcd_path) or ".", exist_ok=True)
        save_confident_pointcloud_batch(
            points=points_world,
            colors=colors if colors is not None else None,
            confs=conf,
            output_path=save_pcd_path,
            conf_threshold=conf_threshold,
            sample_ratio=pcd_sample_ratio,
            batch_size=1000000,
        )

    if len(voxel_sums) == 0:
        vox = SemanticVoxel(
            voxel_size=float(voxel_size),
            centers_world=np.zeros((0, 3), dtype=np.float32),
            features=np.zeros((0, 0), dtype=np.float32),
            contributors=[],
        )
        return SemanticVoxelMap(vox, frame_name_maps=frame_name_maps)

    coords_list = np.array(list(voxel_sums.keys()), dtype=np.int64)
    centers_world = ((coords_list.astype(np.float32) + 0.5) * voxel_size).astype(
        np.float32
    )
    features = np.stack(
        [voxel_sums[k] / max(voxel_counts[k], 1) for k in voxel_sums.keys()], axis=0
    ).astype(np.float32)
    contributors = [sorted(list(voxel_contribs[k])) for k in voxel_sums.keys()]

    vox = SemanticVoxel(
        voxel_size=float(voxel_size),
        centers_world=centers_world,
        features=features,
        contributors=contributors,
    )
    return SemanticVoxelMap(vox, frame_name_maps=frame_name_maps)


def main():
    parser = argparse.ArgumentParser(
        description="Run DepthAnything3 once and build/query a semantic voxel map."
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Image directory.")
    parser.add_argument(
        "--feature_dir", type=str, required=True, help="Feature npz directory."
    )
    parser.add_argument("--feature_key", type=str, default="", help="Key in npz.")
    parser.add_argument("--feature_suffix", type=str, default="", help="Feature suffix.")
    parser.add_argument("--feature_ext", type=str, default=".npz", help="Feature extension.")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="Voxel size.")
    parser.add_argument(
        "--conf_threshold_coef", type=float, default=0.75, help="Conf threshold coef."
    )
    parser.add_argument(
        "--pcd_path",
        type=str,
        default=None,
        help="Optional output PLY path for a colored point cloud.",
    )
    parser.add_argument(
        "--pcd_sample_ratio",
        type=float,
        default=0.015,
        help="Sample ratio for PLY output.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_semantic_voxels",
        help="Output directory.",
    )
    # Query is handled by query_semantic_voxels.py
    args = parser.parse_args()

    images = sorted(
        glob.glob(os.path.join(args.image_dir, "*.png"))
        + glob.glob(os.path.join(args.image_dir, "*.jpg"))
    )
    if len(images) == 0:
        raise ValueError(f"No images found in {args.image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)
    model.eval()

    with torch.no_grad():
        predictions = model.inference(images)
        predictions.conf -= 1.0

    voxel_map = build_semantic_voxel_map(
        images=images,
        predictions=predictions,
        feature_dir=args.feature_dir,
        feature_key=args.feature_key or None,
        feature_suffix=args.feature_suffix,
        feature_ext=args.feature_ext,
        voxel_size=args.voxel_size,
        conf_threshold_coef=args.conf_threshold_coef,
        save_pcd_path=args.pcd_path,
        pcd_sample_ratio=args.pcd_sample_ratio,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    voxel_map.save_to_directory(args.output_dir)
    print(f"Saved semantic voxel map to {args.output_dir}")

if __name__ == "__main__":
    main()
