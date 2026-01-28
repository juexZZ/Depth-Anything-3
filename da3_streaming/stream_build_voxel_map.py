"""
Stream build voxel map.
Build the global voxel map sequentially by reading frames in temporal order.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from tqdm import tqdm

from semantic_voxels.semantic_voxel import SemanticVoxel, SemanticVoxelMap
import torch
import yaml
import argparse

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


class BaseStreamReader:
    """
    Stream reader reads the presaved per-frame information and manages filenames/features.
    """

    def __init__(self, scene_res_dir: str, feature_dir: str, image_dir: str):
        self.scene_res_dir = scene_res_dir
        self.feature_dir = feature_dir
        self.image_dir = image_dir
        self.num_frames = 0

    def read_frame_geometry(
        self, frame_id: int
    ) -> Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Return (points_world [H,W,3], conf [H,W], depth [H,W], intrinsics [3,3], extrinsics [3,4])."""
        raise NotImplementedError

    def read_frame_feature(
        self, frame_id: int, image_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        From frame_id, find frame name, load the feature map of that frame,
        return HxWxD, resized to match the image size if needed.
        """
        raise NotImplementedError

    def read_filename(self, frame_id: int) -> str:
        """From frame_id, return the filename."""
        raise NotImplementedError

    def get_frame_ids(self) -> List[int]:
        """Return the ordered list of frame ids to process."""
        return list(range(self.num_frames))


class DA3StreamReader(BaseStreamReader):
    """
    Read DA3 per-frame results from outputs/results_output/frame_<index>.npz.
    """

    def __init__(
        self,
        scene_res_dir: str,
        feature_dir: str,
        image_dir: str,
        feature_key: Optional[str] = "embedding",
        feature_suffix: str = "",
        feature_ext: str = ".npz",
        image_size: Optional[Tuple[int, int]] = None,
        image_list_path: Optional[str] = None,
    ):
        super().__init__(scene_res_dir, feature_dir, image_dir)
        self.feature_key = feature_key
        self.feature_suffix = feature_suffix
        self.feature_ext = feature_ext if feature_ext.startswith(".") else f".{feature_ext}"
        self.image_size = image_size

        self.result_output_dir = os.path.join(scene_res_dir, "results_output")
        if not os.path.isdir(self.result_output_dir):
            raise FileNotFoundError(f"results_output not found: {self.result_output_dir}")

        if image_dir and os.path.isdir(image_dir):
            self._image_list = sorted(
                [
                    os.path.join(image_dir, name)
                    for name in os.listdir(image_dir)
                    if os.path.isfile(os.path.join(image_dir, name))
                ]
            )
        else:
            self._image_list = []

        available_ids = self._available_frame_ids()
        if image_list_path:
            with open(image_list_path, "r") as f:
                requested = [line.strip() for line in f if line.strip()]
            basename_to_idx: Dict[str, int] = {}
            for idx, path in enumerate(self._image_list):
                basename_to_idx[os.path.basename(path)] = idx
            frame_ids: List[int] = []
            for name in requested:
                base = os.path.basename(name)
                if base not in basename_to_idx:
                    continue
                idx = basename_to_idx[base]
                if idx in available_ids:
                    frame_ids.append(idx)
        else:
            frame_ids = sorted(i for i in available_ids if i < len(self._image_list))

        self._frame_ids = frame_ids
        self._frame_id_to_image = {
            idx: self._image_list[idx] for idx in self._frame_ids
        }
        self.num_frames = len(self._frame_ids)

    def _available_frame_ids(self) -> List[int]:
        indices: List[int] = []
        for name in os.listdir(self.result_output_dir):
            idx = int(name[len("frame_") : -len(".npz")])
            indices.append(idx)
        return sorted(set(indices))

    def _resolve_feature_path(self, image_path: str) -> str:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        feature_dir = self.feature_dir or self.image_dir
        return os.path.join(feature_dir, f"{stem}{self.feature_suffix}{self.feature_ext}")

    def _load_feature_map(
        self, feature_path: str, image_size: Tuple[int, int]
    ) -> np.ndarray:
        with np.load(feature_path) as data:
            feat = None
            if self.feature_key:
                feat = data[self.feature_key]
            else:
                raise KeyError(
                    f"Could not infer feature array key in {feature_path}; keys={data.files}, feature_key={self.feature_key}"
                )
        if feat.ndim != 3:
            raise ValueError(f"Feature map must be (H,W,d); got shape {feat.shape}")
        if feat.shape[0] != image_size[0] or feat.shape[1] != image_size[1]:
            feat = feat.astype(np.float32)
            feat = cv2.resize(feat, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
        return feat.astype(np.float32)

    def read_frame_geometry(
        self, frame_id: int
    ) -> Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        npz_path = os.path.join(self.result_output_dir, f"frame_{frame_id}.npz")
        if not os.path.exists(npz_path):
            return None
        data = np.load(npz_path)
        depth = data["depth"].astype(np.float32)
        conf = data["conf"].astype(np.float32)
        intrinsics = data["intrinsics"].astype(np.float32)
        extrinsics = data["extrinsics"].astype(np.float32)
        points_world = depth_to_point_cloud_vectorized(
            depth[None, ...], intrinsics[None, ...], extrinsics[None, ...]
        )[0]
        return points_world, conf, depth, intrinsics, extrinsics

    def read_frame_feature(
        self, frame_id: int, image_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        image_path = self._frame_id_to_image[frame_id]
        feature_path = self._resolve_feature_path(image_path)
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Missing feature map: {feature_path}")
        return self._load_feature_map(feature_path, image_size=image_size)

    def read_filename(self, frame_id: int) -> str:
        return os.path.basename(self._frame_id_to_image[frame_id])

    def get_frame_ids(self) -> List[int]:
        return list(self._frame_ids)

    def adjust_voxel_size(self, voxel_size: float, max_voxels: int) -> float:
        """Adjust voxel size to fit under max_voxels."""
        # read the ply, compute the number of voxels, and adjust the voxel size accordingly
        ply_path = os.path.join(self.scene_res_dir, "pcd/combined_pcd.ply")
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"result.ply not found: {ply_path}")
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        print(f"current voxel size: {voxel_size}, number of points: {points.shape[0]}")
        extents = points.max(axis=0) - points.min(axis=0)
        num_voxels = np.ceil(extents / voxel_size).astype(np.int64)
        num_voxels = int(num_voxels.prod())
        print(f"got number of voxels: {num_voxels}")
        if num_voxels > max_voxels:
            voxel_size = voxel_size * (num_voxels / max_voxels) ** (1.0 / 3.0)
        print(f"adjusted voxel size: {voxel_size}", f"number of voxels: {num_voxels}")
        return voxel_size


def voxel_subtract_free_space(
    voxel_coords_list: List[np.ndarray],
    feat_sum_map: Dict[Tuple[int, int, int], np.ndarray],
    count_map: Dict[Tuple[int, int, int], int],
    contributors_map: Dict[Tuple[int, int, int], List[Tuple[int, str]]],
    contrib_set_map: Dict[Tuple[int, int, int], set],
    coord_to_index: Dict[Tuple[int, int, int], int],
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    voxel_size: float,
    depth_percentile: float = 99.5,
) -> Tuple[
    List[np.ndarray],
    Dict[Tuple[int, int, int], np.ndarray],
    Dict[Tuple[int, int, int], int],
    Dict[Tuple[int, int, int], List[Tuple[int, str]]],
    Dict[Tuple[int, int, int], set],
    Dict[Tuple[int, int, int], int],
]:
    if len(voxel_coords_list) == 0:
        return (
            voxel_coords_list,
            feat_sum_map,
            count_map,
            contributors_map,
            contrib_set_map,
            coord_to_index,
        )

    depth_valid = depth[np.isfinite(depth)]
    if depth_valid.size == 0:
        return (
            voxel_coords_list,
            feat_sum_map,
            count_map,
            contributors_map,
            contrib_set_map,
            coord_to_index,
        )

    depth_max = float(np.percentile(depth_valid, depth_percentile))
    depth_clip = np.minimum(depth, depth_max)

    centers_world = (
        (np.asarray(voxel_coords_list, dtype=np.float32) + 0.5) * float(voxel_size)
    ).astype(np.float32)

    R = extrinsics[:, :3].astype(np.float32)
    t = extrinsics[:, 3].astype(np.float32)
    cam = (R @ centers_world.T + t[:, None]).T
    z = cam[:, 2]
    valid = z > 0
    if not np.any(valid):
        return (
            voxel_coords_list,
            feat_sum_map,
            count_map,
            contributors_map,
            contrib_set_map,
            coord_to_index,
        )
    # make z==0 case safer
    z = np.where(z == 0.0, 1e-6, z)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    u = cam[:, 0] / z * fx + cx
    v = cam[:, 1] / z * fy + cy

    u_int = np.floor(u).astype(np.int64)
    v_int = np.floor(v).astype(np.int64)
    h, w = depth.shape
    inside = (
        valid & (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)
    )
    if not np.any(inside):
        return (
            voxel_coords_list,
            feat_sum_map,
            count_map,
            contributors_map,
            contrib_set_map,
            coord_to_index,
        )

    depth_at = depth_clip[v_int[inside], u_int[inside]]
    depth_at = np.where(np.isfinite(depth_at), depth_at, 0.0)
    remove_mask = np.zeros(len(voxel_coords_list), dtype=bool)
    remove_idx = np.where(inside)[0]
    remove_mask[remove_idx] = (z[inside] < depth_at) & (depth_at > 0.0)

    if not np.any(remove_mask):
        return (
            voxel_coords_list,
            feat_sum_map,
            count_map,
            contributors_map,
            contrib_set_map,
            coord_to_index,
        )

    remove_coords = [
        tuple(map(int, voxel_coords_list[i])) for i in np.flatnonzero(remove_mask)
    ]

    def _remove_coord(coord: Tuple[int, int, int]) -> None:
        idx = coord_to_index.get(coord, None)
        if idx is None:
            return
        last_idx = len(voxel_coords_list) - 1
        last_coord = tuple(map(int, voxel_coords_list[last_idx]))
        if idx != last_idx:
            voxel_coords_list[idx] = voxel_coords_list[last_idx]
            coord_to_index[last_coord] = idx
        voxel_coords_list.pop()
        coord_to_index.pop(coord, None)
        feat_sum_map.pop(coord, None)
        count_map.pop(coord, None)
        contributors_map.pop(coord, None)
        contrib_set_map.pop(coord, None)

    for coord in remove_coords:
        _remove_coord(coord)
    return (
        voxel_coords_list,
        feat_sum_map,
        count_map,
        contributors_map,
        contrib_set_map,
        coord_to_index,
    )


def build_streaming_semantic_voxel_map(
    reader: BaseStreamReader,
    voxel_size: float,
    output_dir: str,
    pixel_subsample_rate: float = 1.0,
    conf_threshold_coef: Optional[float] = None,
    seed: int = 42,
    max_frames: Optional[int] = None,
    subtract_free_space: bool = True,
) -> SemanticVoxelMap:
    """
    Sequentially read frames and build a global semantic voxel map.

    - For each frame, unproject depth to world points and voxelize.
    - Average feature vectors per voxel across all contributing pixels.
    - Keep a list of contributing frame names in temporal reverse order (latest first).
    """
    rng = np.random.default_rng(seed)
    submap_id = 0
    frame_name_maps: Dict[str, Dict[str, str]] = {str(submap_id): {}}

    coord_to_index: Dict[Tuple[int, int, int], int] = {}
    voxel_coords_list: List[np.ndarray] = []
    feat_sum_map: Dict[Tuple[int, int, int], np.ndarray] = {}
    count_map: Dict[Tuple[int, int, int], int] = {}
    contributors_map: Dict[Tuple[int, int, int], List[Tuple[int, str]]] = {}
    contrib_set_map: Dict[Tuple[int, int, int], set] = {}
    frame_extrinsics: List[Tuple[int, str, np.ndarray]] = []

    frame_ids = reader.get_frame_ids()

    for frame_id in tqdm(frame_ids, desc="Streaming voxelization"):
        geom = reader.read_frame_geometry(frame_id)
        if geom is None:
            print(f"No geometry found for frame {frame_id}, skipping")
            continue
        points_world, conf, depth, intrinsics, extrinsics = geom
        image_size = points_world.shape[:2]
        feat = reader.read_frame_feature(frame_id, image_size=image_size)
        frame_name = reader.read_filename(frame_id)
        frame_name_maps[str(submap_id)][str(frame_id)] = frame_name
        frame_extrinsics.append((frame_id, frame_name, extrinsics.astype(np.float32)))
        
        if subtract_free_space:
            # print(f"subtracting free space for frame {frame_id}")
            (
                voxel_coords_list,
                feat_sum_map,
                count_map,
                contributors_map,
                contrib_set_map,
                coord_to_index,
            ) = voxel_subtract_free_space(
                voxel_coords_list=voxel_coords_list,
                feat_sum_map=feat_sum_map,
                count_map=count_map,
                contributors_map=contributors_map,
                contrib_set_map=contrib_set_map,
                coord_to_index=coord_to_index,
                depth=depth.astype(np.float32),
                intrinsics=intrinsics.astype(np.float32),
                extrinsics=extrinsics.astype(np.float32),
                voxel_size=float(voxel_size),
                depth_percentile=99.5,
            )

        mask = np.isfinite(points_world).all(axis=2)
        if conf_threshold_coef is not None:
            conf_thresh = float(np.mean(conf)) * conf_threshold_coef
            mask = mask & (conf >= conf_thresh)
        if not mask.any():
            continue

        pts_flat = points_world[mask]
        feat_flat = feat[mask].astype(np.float32)
        if pts_flat.shape[0] == 0:
            continue

        if pixel_subsample_rate < 1.0:
            keep = rng.random(pts_flat.shape[0]) < float(pixel_subsample_rate)
            pts_flat = pts_flat[keep]
            feat_flat = feat_flat[keep]
        if pts_flat.shape[0] == 0:
            continue

        voxel_coords = np.floor(pts_flat / float(voxel_size)).astype(np.int64)
        unique_coords, inv = np.unique(voxel_coords, axis=0, return_inverse=True)
        num_unique = unique_coords.shape[0]
        if num_unique == 0:
            continue
        # print(f"updating features for frame {frame_id}")
        dim = feat_flat.shape[1]
        sums = np.zeros((num_unique, dim), dtype=np.float32)
        cnts = np.zeros((num_unique,), dtype=np.int64)
        np.add.at(sums, inv, feat_flat)
        np.add.at(cnts, inv, 1)
        # add the voxel coordinates, feature sums, counts, and contributors to the maps
        for i in range(num_unique):
            coord = unique_coords[i]
            key = (int(coord[0]), int(coord[1]), int(coord[2]))
            if key not in coord_to_index:
                coord_to_index[key] = len(voxel_coords_list)
                voxel_coords_list.append(coord)
                feat_sum_map[key] = sums[i]
                count_map[key] = int(cnts[i])
                contributors_map[key] = []
                contrib_set_map[key] = set()
            else:
                feat_sum_map[key] = feat_sum_map[key] + sums[i]
                count_map[key] = count_map[key] + int(cnts[i])

            if frame_id not in contrib_set_map[key]:
                contributors_map[key].append((submap_id, str(frame_id)))
                contrib_set_map[key].add(frame_id)

    if len(voxel_coords_list) == 0:
        vox = SemanticVoxel(
            voxel_size=float(voxel_size),
            centers_world=np.zeros((0, 3), dtype=np.float32),
            features=np.zeros((0, 0), dtype=np.float32),
            contributors=[],
        )
        voxel_map = SemanticVoxelMap(vox, frame_name_maps=frame_name_maps)
        voxel_map.save_to_directory(output_dir)
        return voxel_map
    print(f"aggregating features for the final output")
    centers_world = (
        (np.asarray(voxel_coords_list, dtype=np.float32) + 0.5) * float(voxel_size)
    ).astype(np.float32)
    feat_sums_arr = torch.stack(
        [
            torch.from_numpy(feat_sum_map[tuple(map(int, coord))].astype(np.float32, copy=False))
            for coord in voxel_coords_list
        ],
        dim=0,
    )
    counts_arr = torch.tensor(
        [count_map[tuple(map(int, coord))] for coord in voxel_coords_list],
        dtype=torch.float32,
    )
    features = (feat_sums_arr / torch.clamp(counts_arr[:, None], min=1)).cpu().numpy()

    # Reverse contributor lists so latest frames come first.
    contributors = [
        list(reversed(contributors_map[tuple(map(int, coord))]))
        for coord in voxel_coords_list
    ]
    print(f"creating semantic voxel map")
    vox = SemanticVoxel(
        voxel_size=float(voxel_size),
        centers_world=centers_world,
        features=features.astype(np.float32),
        contributors=contributors,
    )
    voxel_map = SemanticVoxelMap(vox, frame_name_maps=frame_name_maps)
    os.makedirs(output_dir, exist_ok=True)
    extrinsics_path = os.path.join(output_dir, "camera_extrinsics.txt")
    with open(extrinsics_path, "w") as f:
        for frame_id, frame_name, w2c in frame_extrinsics:
            flat = w2c.reshape(-1)
            f.write(f"{frame_id} {frame_name} " + " ".join([str(x) for x in flat]) + "\n")
    print(f"Saved camera extrinsics to {extrinsics_path}")
    print(f"saving semantic voxel map to {output_dir}")
    voxel_map.save_to_directory(output_dir)
    return voxel_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build semantic voxel map by streaming per-frame results."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml (e.g., configs/base_config.yaml)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to input images (used for filename mapping)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Scene output directory containing results_output/",
    )
    parser.add_argument(
        "--result_subdir",
        type=str,
        default="",
        help="Subdirectory of output_dir to save the results",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="",
        help="Directory containing per-frame feature maps (defaults to image_dir if empty)",
    )
    parser.add_argument(
        "--image_list",
        type=str,
        default=None,
        help="Optional text file with one image filename/path per line to process",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to process",
    )
    parser.add_argument(
        "--subtract_free_space",
        action="store_true",
        default=False,
        help="Subtract free space from the voxel map",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    sem_cfg = config.get("Semantic_Voxels", {})
    if not sem_cfg or not sem_cfg.get("enable", False):
        print("[stream] Semantic_Voxels is disabled in config.")
        return

    voxel_size = float(sem_cfg.get("voxel_size", 0.1))
    image_size = sem_cfg.get("image_size", None)
    feature_dir = args.feature_dir
    if feature_dir is None:
        feature_dir = sem_cfg.get("feature_dir")
    feature_suffix = sem_cfg.get("feature_suffix", "")
    feature_ext = sem_cfg.get("feature_ext", ".npz")
    conf_threshold_coef = float(sem_cfg.get("conf_threshold_coef", 0.75))
    base_output_dir = (
        os.path.join(args.output_dir, args.result_subdir)
        if args.result_subdir
        else args.output_dir
    )
    output_dir = os.path.join(
        base_output_dir, sem_cfg.get("output_dir", "semantic_voxels")
    )
    pixel_subsample_rate = sem_cfg.get("pixel_subsample_rate", 1.0)
    
    subtract_free_space = args.subtract_free_space
    
    reader = DA3StreamReader(
        scene_res_dir=args.output_dir,
        feature_dir=feature_dir,
        image_dir=args.image_dir,
        feature_key="embedding",
        feature_suffix=feature_suffix,
        feature_ext=feature_ext,
        image_size=tuple(image_size) if image_size is not None else None,
        image_list_path=args.image_list,
    )
    
    max_voxels = sem_cfg.get("max_voxels", 0)
    if max_voxels > 0:
        voxel_size = reader.adjust_voxel_size(voxel_size=voxel_size, max_voxels=max_voxels)
        

    build_streaming_semantic_voxel_map(
        reader=reader,
        voxel_size=voxel_size,
        output_dir=output_dir,
        pixel_subsample_rate=pixel_subsample_rate,
        conf_threshold_coef=conf_threshold_coef,
        max_frames=args.max_frames,
        subtract_free_space=subtract_free_space,
    )


if __name__ == "__main__":
    main()
