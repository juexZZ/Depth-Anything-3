# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)

import argparse
import gc
import glob
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from loop_utils.alignment_torch import (
    apply_sim3_direct_torch,
    depth_to_point_cloud_optimized_torch,
)
from loop_utils.config_utils import load_config
from loop_utils.loop_detector import LoopDetector
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    compute_sim3_ab,
    merge_ply_files,
    precompute_scale_chunks_with_depth,
    process_loop_list,
    save_confident_pointcloud_batch,
    warmup_numba,
    weighted_align_point_maps,
)
from safetensors.torch import load_file

from depth_anything_3.api import DepthAnything3
from semantic_voxels.map import GraphMap
from semantic_voxels.semantic_voxel import SemanticVoxel, SemanticVoxelMap
from semantic_voxels.submap import SemanticSubmap
import cv2
from tqdm import tqdm

matplotlib.use("Agg")


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


def remove_duplicates(data_list):
    """
    data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {}
    result = []

    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])

        if key not in seen.keys():
            seen[key] = True
            result.append(item)

    return result


class DA3_Streaming:
    def __init__(self, image_dir, save_dir, config):
        self.config = config

        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap = self.config["Model"]["overlap"]
        self.overlap_s = 0
        self.overlap_e = self.overlap - self.overlap_s
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        )

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        self.result_aligned_dir = os.path.join(save_dir, "_tmp_results_aligned")
        self.result_loop_dir = os.path.join(save_dir, "_tmp_results_loop")
        self.result_output_dir = os.path.join(save_dir, "results_output")
        self.pcd_dir = os.path.join(save_dir, "pcd")
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)

        self.all_camera_poses = []
        self.all_camera_intrinsics = []

        self.delete_temp_files = self.config["Model"]["delete_temp_files"]

        print("Loading model...")

        with open(self.config["Weights"]["DA3_CONFIG"]) as f:
            config = json.load(f)
        self.model = DepthAnything3(**config)
        weight = load_file(self.config["Weights"]["DA3"])
        self.model.load_state_dict(weight, strict=False)

        self.model.eval()
        self.model = self.model.to(self.device)

        self.skyseg_session = None

        self.chunk_indices = None  # [(begin_idx, end_idx), ...]

        self.loop_list = []  # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = []  # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = []  # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config["Model"]["loop_enable"]

        if self.loop_enable:
            loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
            self.loop_detector = LoopDetector(
                image_dir=image_dir, output=loop_info_save_path, config=self.config
            )
            self.loop_detector.load_model()

        print("init done.")

    def get_loop_pairs(self):
        self.loop_detector.run()
        loop_list = self.loop_detector.get_loop_list()
        return loop_list

    def save_depth_conf_result(self, predictions, chunk_idx, s, R, T):
        if not self.config["Model"]["save_depth_conf_result"]:
            return
        os.makedirs(self.result_output_dir, exist_ok=True)

        chunk_start, chunk_end = self.chunk_indices[chunk_idx]

        if chunk_idx == 0:
            save_indices = list(range(0, chunk_end - chunk_start - self.overlap_e))
        elif chunk_idx == len(self.chunk_indices) - 1:
            save_indices = list(range(self.overlap_s, chunk_end - chunk_start))
        else:
            save_indices = list(range(self.overlap_s, chunk_end - chunk_start - self.overlap_e))

        print("[save_depth_conf_result] save_indices:")

        for local_idx in save_indices:
            global_idx = chunk_start + local_idx
            print(f"{global_idx}, ", end="")

            image = predictions.processed_images[local_idx]  # [H, W, 3] uint8
            depth = predictions.depth[local_idx]  # [H, W] float32
            conf = predictions.conf[local_idx]  # [H, W] float32
            intrinsics = predictions.intrinsics[local_idx]  # [3, 3] float32

            filename = f"frame_{global_idx}.npz"
            filepath = os.path.join(self.result_output_dir, filename)

            if self.config["Model"]["save_debug_info"]:
                np.savez_compressed(
                    filepath,
                    image=image,
                    depth=depth,
                    conf=conf,
                    intrinsics=intrinsics,
                    extrinsics=predictions.extrinsics[local_idx],
                    s=s,
                    R=R,
                    T=T,
                )
            else:
                np.savez_compressed(
                    filepath, image=image, depth=depth, conf=conf, intrinsics=intrinsics
                )
        print("")

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        # images = load_and_preprocess_images(chunk_image_paths).to(self.device)
        print(f"Loaded {len(chunk_image_paths)} images")

        ref_view_strategy = self.config["Model"][
            "ref_view_strategy" if not is_loop else "ref_view_strategy_loop"
        ]

        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images = chunk_image_paths
                # images: ['xxx.png', 'xxx.png', ...]

                predictions = self.model.inference(images, ref_view_strategy=ref_view_strategy)

                predictions.depth = np.squeeze(predictions.depth)
                predictions.conf -= 1.0

                print(predictions.processed_images.shape)  # [N, H, W, 3] uint8
                print(predictions.depth.shape)  # [N, H, W] float32
                print(predictions.conf.shape)  # [N, H, W] float32
                print(predictions.extrinsics.shape)  # [N, 3, 4] float32 (w2c)
                print(predictions.intrinsics.shape)  # [N, 3, 3] float32
        torch.cuda.empty_cache()

        # Save predictions to disk instead of keeping in memory
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            extrinsics = predictions.extrinsics
            intrinsics = predictions.intrinsics
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        np.save(save_path, predictions)

        return predictions

    def get_chunk_indices(self):
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                chunk_indices.append((start_idx, end_idx))
        return chunk_indices, num_chunks

    def align_2pcds(
        self,
        point_map1,
        conf1,
        point_map2,
        conf2,
        chunk1_depth,
        chunk2_depth,
        chunk1_depth_conf,
        chunk2_depth_conf,
    ):

        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

        scale_factor = None
        if self.config["Model"]["align_method"] == "scale+se3":
            scale_factor_return, quality_score, method_used = precompute_scale_chunks_with_depth(
                chunk1_depth,
                chunk1_depth_conf,
                chunk2_depth,
                chunk2_depth_conf,
                method=self.config["Model"]["scale_compute_method"],
            )
            print(
                f"[Depth Scale Precompute] scale: {scale_factor_return}, \
                    quality_score: {quality_score}, method_used: {method_used}"
            )
            scale_factor = scale_factor_return

        s, R, t = weighted_align_point_maps(
            point_map1,
            conf1,
            point_map2,
            conf2,
            conf_threshold=conf_threshold,
            config=self.config,
            precompute_scale=scale_factor,
        )
        print("Estimated Scale:", s)
        print("Estimated Rotation:\n", R)
        print("Estimated Translation:", t)

        return s, R, t

    def get_loop_sim3_from_loop_predict(self, loop_predict_list):
        loop_sim3_list = []
        for item in loop_predict_list:
            chunk_idx_a = item[0][0]
            chunk_idx_b = item[0][2]
            chunk_a_range = item[0][1]
            chunk_b_range = item[0][3]

            point_map_loop_org = depth_to_point_cloud_vectorized(
                item[1].depth, item[1].intrinsics, item[1].extrinsics
            )

            chunk_a_s = 0
            chunk_a_e = chunk_a_len = chunk_a_range[1] - chunk_a_range[0]
            chunk_b_s = -chunk_b_range[1] + chunk_b_range[0]
            chunk_b_e = point_map_loop_org.shape[0]
            chunk_b_len = chunk_b_range[1] - chunk_b_range[0]

            chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
            chunk_a_rela_end = chunk_a_rela_begin + chunk_a_len
            chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
            chunk_b_rela_end = chunk_b_rela_begin + chunk_b_len

            print("chunk_a align")

            point_map_loop_a = point_map_loop_org[chunk_a_s:chunk_a_e]
            conf_loop = item[1].conf[chunk_a_s:chunk_a_e]
            print(self.chunk_indices[chunk_idx_a])
            print(chunk_a_range)
            print(chunk_a_rela_begin, chunk_a_rela_end)
            chunk_data_a = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"),
                allow_pickle=True,
            ).item()

            point_map_a = depth_to_point_cloud_vectorized(
                chunk_data_a.depth, chunk_data_a.intrinsics, chunk_data_a.extrinsics
            )
            point_map_a = point_map_a[chunk_a_rela_begin:chunk_a_rela_end]
            conf_a = chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_a_depth = np.squeeze(chunk_data_a.depth[chunk_a_rela_begin:chunk_a_rela_end])
                chunk_a_depth_conf = np.squeeze(
                    chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]
                )
                chunk_a_loop_depth = np.squeeze(item[1].depth[chunk_a_s:chunk_a_e])
                chunk_a_loop_depth_conf = np.squeeze(item[1].conf[chunk_a_s:chunk_a_e])
            else:
                chunk_a_depth = None
                chunk_a_loop_depth = None
                chunk_a_depth_conf = None
                chunk_a_loop_depth_conf = None

            s_a, R_a, t_a = self.align_2pcds(
                point_map_a,
                conf_a,
                point_map_loop_a,
                conf_loop,
                chunk_a_depth,
                chunk_a_loop_depth,
                chunk_a_depth_conf,
                chunk_a_loop_depth_conf,
            )

            print("chunk_b align")

            point_map_loop_b = point_map_loop_org[chunk_b_s:chunk_b_e]
            conf_loop = item[1].conf[chunk_b_s:chunk_b_e]
            print(self.chunk_indices[chunk_idx_b])
            print(chunk_b_range)
            print(chunk_b_rela_begin, chunk_b_rela_end)
            chunk_data_b = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                allow_pickle=True,
            ).item()

            point_map_b = depth_to_point_cloud_vectorized(
                chunk_data_b.depth, chunk_data_b.intrinsics, chunk_data_b.extrinsics
            )
            point_map_b = point_map_b[chunk_b_rela_begin:chunk_b_rela_end]
            conf_b = chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_b_depth = np.squeeze(chunk_data_b.depth[chunk_b_rela_begin:chunk_b_rela_end])
                chunk_b_depth_conf = np.squeeze(
                    chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]
                )
                chunk_b_loop_depth = np.squeeze(item[1].depth[chunk_b_s:chunk_b_e])
                chunk_b_loop_depth_conf = np.squeeze(item[1].conf[chunk_b_s:chunk_b_e])
            else:
                chunk_b_depth = None
                chunk_b_loop_depth = None
                chunk_b_depth_conf = None
                chunk_b_loop_depth_conf = None

            s_b, R_b, t_b = self.align_2pcds(
                point_map_b,
                conf_b,
                point_map_loop_b,
                conf_loop,
                chunk_b_depth,
                chunk_b_loop_depth,
                chunk_b_depth_conf,
                chunk_b_loop_depth_conf,
            )

            print("a -> b SIM 3")
            s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
            print("Estimated Scale:", s_ab)
            print("Estimated Rotation:\n", R_ab)
            print("Estimated Translation:", t_ab)

            loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        return loop_sim3_list

    def plot_loop_closure(
        self, input_abs_poses, optimized_abs_poses, save_name="sim3_opt_result.png"
    ):
        def extract_xyz(pose_tensor):
            poses = pose_tensor.cpu().numpy()
            return poses[:, 0], poses[:, 1], poses[:, 2]

        x0, _, y0 = extract_xyz(input_abs_poses)
        x1, _, y1 = extract_xyz(optimized_abs_poses)

        # Visual in png format
        plt.figure(figsize=(8, 6))
        plt.plot(x0, y0, "o--", alpha=0.45, label="Before Optimization")
        plt.plot(x1, y1, "o-", label="After Optimization")
        for i, j, _ in self.loop_sim3_list:
            plt.plot(
                [x0[i], x0[j]],
                [y0[i], y0[j]],
                "r--",
                alpha=0.25,
                label="Loop (Before)" if i == 5 else "",
            )
            plt.plot(
                [x1[i], x1[j]],
                [y1[i], y1[j]],
                "g-",
                alpha=0.25,
                label="Loop (After)" if i == 5 else "",
            )
        plt.gca().set_aspect("equal")
        plt.title("Sim3 Loop Closure Optimization")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) \
                    must be less than chunk size ({self.chunk_size})"
            )

        self.chunk_indices, num_chunks = self.get_chunk_indices()

        print(
            f"Processing {len(self.img_list)} images in {num_chunks} \
                chunks of size {self.chunk_size} with {self.overlap} overlap"
        )

        pre_predictions = None
        for chunk_idx in range(len(self.chunk_indices)):
            print(f"[Progress]: {chunk_idx}/{len(self.chunk_indices)}")
            cur_predictions = self.process_single_chunk(
                self.chunk_indices[chunk_idx], chunk_idx=chunk_idx
            )
            torch.cuda.empty_cache()

            if chunk_idx > 0:
                print(
                    f"Aligning {chunk_idx-1} and {chunk_idx} (Total {len(self.chunk_indices)-1})"
                )
                chunk_data1 = pre_predictions
                chunk_data2 = cur_predictions

                point_map1 = depth_to_point_cloud_vectorized(
                    chunk_data1.depth, chunk_data1.intrinsics, chunk_data1.extrinsics
                )
                point_map2 = depth_to_point_cloud_vectorized(
                    chunk_data2.depth, chunk_data2.intrinsics, chunk_data2.extrinsics
                )

                point_map1 = point_map1[-self.overlap :]
                point_map2 = point_map2[: self.overlap]
                conf1 = chunk_data1.conf[-self.overlap :]
                conf2 = chunk_data2.conf[: self.overlap]

                if self.config["Model"]["align_method"] == "scale+se3":
                    chunk1_depth = np.squeeze(chunk_data1.depth[-self.overlap :])
                    chunk2_depth = np.squeeze(chunk_data2.depth[: self.overlap])
                    chunk1_depth_conf = np.squeeze(chunk_data1.conf[-self.overlap :])
                    chunk2_depth_conf = np.squeeze(chunk_data2.conf[: self.overlap])
                else:
                    chunk1_depth = None
                    chunk2_depth = None
                    chunk1_depth_conf = None
                    chunk2_depth_conf = None

                s, R, t = self.align_2pcds(
                    point_map1,
                    conf1,
                    point_map2,
                    conf2,
                    chunk1_depth,
                    chunk2_depth,
                    chunk1_depth_conf,
                    chunk2_depth_conf,
                )
                self.sim3_list.append((s, R, t))

            pre_predictions = cur_predictions

        if self.loop_enable:
            self.loop_list = self.get_loop_pairs()
            del self.loop_detector  # Save GPU Memory

            torch.cuda.empty_cache()

            print("Loop SIM(3) estimating...")
            loop_results = process_loop_list(
                self.chunk_indices,
                self.loop_list,
                half_window=int(self.config["Model"]["loop_chunk_size"] / 2),
            )
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            # return e.g. (31, (1574, 1594), 2, (129, 149))
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(
                    item[1], range_2=item[3], is_loop=True
                )

                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)

            self.loop_sim3_list = self.get_loop_sim3_from_loop_predict(self.loop_predict_list)

            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                self.sim3_list
            )  # just for plot
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                self.sim3_list
            )  # just for plot

            self.plot_loop_closure(
                input_abs_poses, optimized_abs_poses, save_name="sim3_opt_result.png"
            )

        print("Apply alignment")
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f"Applying {chunk_idx+1} -> {chunk_idx} (Total {len(self.chunk_indices)-1})")
            s, R, t = self.sim3_list[chunk_idx]

            chunk_data = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True,
            ).item()

            aligned_chunk_data = {}

            aligned_chunk_data["world_points"] = depth_to_point_cloud_optimized_torch(
                chunk_data.depth, chunk_data.intrinsics, chunk_data.extrinsics
            )
            aligned_chunk_data["world_points"] = apply_sim3_direct_torch(
                aligned_chunk_data["world_points"], s, R, t
            )

            aligned_chunk_data["conf"] = chunk_data.conf
            aligned_chunk_data["images"] = chunk_data.processed_images

            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
            np.save(aligned_path, aligned_chunk_data)

            if chunk_idx == 0:
                chunk_data_first = np.load(
                    os.path.join(self.result_unaligned_dir, "chunk_0.npy"), allow_pickle=True
                ).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
                points_first = depth_to_point_cloud_vectorized(
                    chunk_data_first.depth,
                    chunk_data_first.intrinsics,
                    chunk_data_first.extrinsics,
                )
                colors_first = chunk_data_first.processed_images
                confs_first = chunk_data_first.conf
                ply_path_first = os.path.join(self.pcd_dir, "0_pcd.ply")
                save_confident_pointcloud_batch(
                    points=points_first,  # shape: (H, W, 3)
                    colors=colors_first,  # shape: (H, W, 3)
                    confs=confs_first,  # shape: (H, W)
                    output_path=ply_path_first,
                    conf_threshold=np.mean(confs_first)
                    * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                    sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
                )
                if self.config["Model"]["save_depth_conf_result"]:
                    predictions = chunk_data_first
                    self.save_depth_conf_result(predictions, 0, 1, np.eye(3), np.array([0, 0, 0]))

            points = aligned_chunk_data["world_points"].reshape(-1, 3)
            colors = (aligned_chunk_data["images"].reshape(-1, 3)).astype(np.uint8)
            confs = aligned_chunk_data["conf"].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f"{chunk_idx+1}_pcd.ply")
            save_confident_pointcloud_batch(
                points=points,  # shape: (H, W, 3)
                colors=colors,  # shape: (H, W, 3)
                confs=confs,  # shape: (H, W)
                output_path=ply_path,
                conf_threshold=np.mean(confs)
                * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )

            if self.config["Model"]["save_depth_conf_result"]:
                predictions = chunk_data
                predictions.depth *= s
                self.save_depth_conf_result(predictions, chunk_idx + 1, s, R, t)

        self.save_camera_poses()
        self.build_semantic_voxel_map()

        print("Done.")

    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg"))
            + glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        # print(self.img_list)
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        semantic_cfg = self.config.get("Semantic_Voxels", {})
        only_from_temp = bool(semantic_cfg.get("only_from_temp", False))
        if only_from_temp:
            if not self._has_aligned_temp_results():
                raise FileNotFoundError(
                    f"No aligned temp results found in {self.result_aligned_dir}."
                )
            self.chunk_indices, _ = self.get_chunk_indices()
            print("[semantic] using aligned temp results; skipping point cloud build")
            self.build_semantic_voxel_map()
            return

        self.process_long_sequence()

    def save_camera_poses(self):
        """
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        """
        chunk_colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],  # Dark Red
            [0, 128, 0],  # Dark Green
            [0, 0, 128],  # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]

        for i, idx in enumerate(
            range(first_chunk_range[0], first_chunk_range[1] - self.overlap_e)
        ):
            w2c = np.eye(4)
            w2c[:3, :] = first_chunk_extrinsics[i]
            c2w = np.linalg.inv(w2c)
            all_poses[idx] = c2w
            all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[
                chunk_idx - 1
            ]  # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            chunk_range_end = (
                chunk_range[1] - self.overlap_e
                if chunk_idx < len(self.all_camera_poses) - 1
                else chunk_range[1]
            )

            for i, idx in enumerate(range(chunk_range[0] + self.overlap_s, chunk_range_end)):
                w2c = np.eye(4)
                w2c[:3, :] = chunk_extrinsics[i + self.overlap_s]
                c2w = np.linalg.inv(w2c)

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!
                transformed_c2w[:3, :3] /= s  # Normalize rotation

                all_poses[idx] = transformed_c2w
                all_intrinsics[idx] = chunk_intrinsics[i + self.overlap_s]

        poses_path = os.path.join(self.output_dir, "camera_poses.txt")
        with open(poses_path, "w") as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(" ".join([str(x) for x in flat_pose]) + "\n")

        print(f"Camera poses saved to {poses_path}")

        intrinsics_path = os.path.join(self.output_dir, "intrinsic.txt")
        with open(intrinsics_path, "w") as f:
            for intrinsic in all_intrinsics:
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
                f.write(f"{fx} {fy} {cx} {cy}\n")

        print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, "camera_poses.ply")
        with open(ply_path, "w") as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_poses)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(
                    f"{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n"
                )

        print(f"Camera poses visualization saved to {ply_path}")

    def _resolve_feature_path(
        self, image_path: str, feature_dir: str, feature_suffix: str, feature_ext: str
    ) -> str:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        feature_ext = feature_ext if feature_ext.startswith(".") else f".{feature_ext}"
        return os.path.join(feature_dir, f"{stem}{feature_suffix}{feature_ext}")

    def _load_feature_map(self, feature_path: str, feature_key: Optional[str], image_size: tuple[int, int]) -> np.ndarray:
        with np.load(feature_path) as data:
            feat = None
            if feature_key:
                if feature_key not in data:
                    raise KeyError(f"Feature key '{feature_key}' not found in {feature_path}")
                feat = data[feature_key]
            else:
                # Try common keys, then fallback to the only array if unambiguous.
                for key in ("feat", "feature", "features", "embedding", "embeddings"):
                    if key in data:
                        # print(f"Using feature key: {key} in {feature_path}")
                        feat = data[key]
                        break
                if feat is None:
                    raise KeyError(f"Could not infer feature array key in {feature_path}; keys={data.files}")
        if feat.ndim != 3:
            raise ValueError(f"Feature map must be (H,W,d); got shape {feat.shape}")
        # resize the feature map to the image size
        if feat.shape[0] != image_size[0] or feat.shape[1] != image_size[1]:
            feat = feat.astype(np.float32)
            feat = cv2.resize(feat, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
        return feat.astype(np.float32)

    def build_semantic_voxel_map(self) -> None:
        cfg = self.config.get("Semantic_Voxels", {})
        if not cfg or not cfg.get("enable", False):
            return

        voxel_size = float(cfg.get("voxel_size", 0.1))
        max_voxels = int(cfg.get("max_voxels", 0))
        image_size = cfg.get("image_size", [504, 504])
        stride = int(cfg.get("stride", 1))
        conf_threshold_coef = float(
            cfg.get("conf_threshold_coef", self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"])
        )
        feature_dir = cfg.get("feature_dir") or self.img_dir
        feature_key = cfg.get("feature_key", None)
        feature_suffix = cfg.get("feature_suffix", "")
        feature_ext = cfg.get("feature_ext", ".npz")
        output_dir = os.path.join(self.output_dir, cfg.get("output_dir", "semantic_voxels"))
        stream_submaps = bool(cfg.get("stream_submaps", False))
        cache_submaps = bool(cfg.get("cache_submaps", False))
        submap_cache_dir = cfg.get("submap_cache_dir", "")
        if cache_submaps:
            if not submap_cache_dir:
                submap_cache_dir = os.path.join(output_dir, "submaps")
            os.makedirs(submap_cache_dir, exist_ok=True)
        skip_missing_features = bool(cfg.get("skip_missing_features", True))
        use_torch = bool(cfg.get("use_torch", True))
        ignore_loop_closure_frames = bool(cfg.get("ignore_loop_closure_frames", False))
        deduplicate_contributors = bool(cfg.get("deduplicate_contributors", True))

        if not os.path.isdir(feature_dir):
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

        def _filtered_points_for_bounds(
            world_points: np.ndarray, conf: np.ndarray, conf_coef: float, base_voxel: float
        ) -> Optional[np.ndarray]:
            if world_points.ndim == 3:
                world_points = world_points[None, ...]
            if conf.ndim == 2:
                conf = conf[None, ...]

            conf_threshold = float(np.mean(conf)) * conf_coef
            mask = conf >= conf_threshold
            pts_flat = world_points[mask]
            if pts_flat.shape[0] == 0:
                return None

            finite_mask = np.isfinite(pts_flat).all(axis=1)
            if not np.all(finite_mask):
                pts_flat = pts_flat[finite_mask]
            if pts_flat.shape[0] == 0:
                return None

            lo = np.percentile(pts_flat, 0.5, axis=0)
            hi = np.percentile(pts_flat, 99.5, axis=0)
            bbox_mask = (pts_flat >= lo).all(axis=1) & (pts_flat <= hi).all(axis=1)
            if not np.all(bbox_mask):
                pts_flat = pts_flat[bbox_mask]
            if pts_flat.shape[0] == 0:
                return None

            coarse_cell = float(base_voxel) * 3.0
            min_points_per_cell = 10
            if coarse_cell > 0.0 and pts_flat.shape[0] > 0:
                coarse_coords = np.floor(pts_flat / coarse_cell).astype(np.int64)
                _, inv, counts_c = np.unique(
                    coarse_coords, axis=0, return_inverse=True, return_counts=True
                )
                dense_mask = counts_c[inv] >= min_points_per_cell
                if not np.all(dense_mask):
                    pts_flat = pts_flat[dense_mask]
            if pts_flat.shape[0] == 0:
                return None

            return pts_flat

        if max_voxels > 0:
            bounds_min = None
            bounds_max = None
            for chunk_idx, chunk_range in enumerate(self.chunk_indices):
                aligned_path = os.path.join(
                    self.result_aligned_dir, f"chunk_{chunk_idx}.npy"
                )
                if not os.path.exists(aligned_path):
                    continue

                chunk_data = np.load(aligned_path, allow_pickle=True).item()
                if isinstance(chunk_data, dict):
                    world_points = chunk_data.get("world_points", None)
                    conf = chunk_data.get("conf", None)
                else:
                    depth = chunk_data.depth
                    intrinsics = chunk_data.intrinsics
                    extrinsics = chunk_data.extrinsics
                    if depth.ndim == 2:
                        depth = depth[None, ...]
                    if intrinsics.ndim == 2:
                        intrinsics = intrinsics[None, ...]
                    if extrinsics.ndim == 2:
                        extrinsics = extrinsics[None, ...]
                    world_points = depth_to_point_cloud_vectorized(
                        depth, intrinsics, extrinsics
                    )
                    conf = chunk_data.conf

                if world_points is None or conf is None:
                    continue

                pts_flat = _filtered_points_for_bounds(
                    np.asarray(world_points, dtype=np.float32),
                    np.asarray(conf, dtype=np.float32),
                    conf_threshold_coef,
                    voxel_size,
                )
                if pts_flat is None or pts_flat.shape[0] == 0:
                    continue

                lo = np.min(pts_flat, axis=0)
                hi = np.max(pts_flat, axis=0)
                if bounds_min is None:
                    bounds_min = lo
                    bounds_max = hi
                else:
                    bounds_min = np.minimum(bounds_min, lo)
                    bounds_max = np.maximum(bounds_max, hi)

            if bounds_min is not None and bounds_max is not None:
                extents = bounds_max - bounds_min
                extents = np.maximum(extents, 1e-6)
                volume = float(np.prod(extents))
                required_size = float((volume / float(max_voxels)) ** (1.0 / 3.0))
                if required_size > voxel_size:
                    print(
                        "[semantic] increasing voxel_size "
                        f"from {voxel_size:.6f} to {required_size:.6f} "
                        f"to keep voxels under {max_voxels}"
                    )
                    voxel_size = required_size
                else:
                    print(f"[semantic] voxel_size {voxel_size:.6f} is already sufficient for {max_voxels} voxels")
            else:
                print("[semantic] max_voxels set, but no valid points to estimate bounds")

        if stream_submaps:
            frame_name_maps = {}
            coord_to_index = {}
            voxel_coords_list = []
            voxel_contribs = []
            frame_to_voxels = {}

            # First pass: build voxel set + contributors + inverse frame->voxels map (no features).
            for chunk_idx, chunk_range in enumerate(self.chunk_indices):
                aligned_path = os.path.join(
                    self.result_aligned_dir, f"chunk_{chunk_idx}.npy"
                )
                if not os.path.exists(aligned_path):
                    print(f"[semantic] missing aligned chunk {aligned_path}, skipping")
                    continue

                chunk_data = np.load(aligned_path, allow_pickle=True).item()
                if isinstance(chunk_data, dict):
                    world_points = chunk_data.get("world_points", None)
                    conf = chunk_data.get("conf", None)
                else:
                    depth = chunk_data.depth
                    intrinsics = chunk_data.intrinsics
                    extrinsics = chunk_data.extrinsics
                    if depth.ndim == 2:
                        depth = depth[None, ...]
                    if intrinsics.ndim == 2:
                        intrinsics = intrinsics[None, ...]
                    if extrinsics.ndim == 2:
                        extrinsics = extrinsics[None, ...]
                    world_points = depth_to_point_cloud_vectorized(
                        depth, intrinsics, extrinsics
                    )
                    conf = chunk_data.conf

                if world_points is None or conf is None:
                    print(
                        f"[semantic] chunk {chunk_idx} missing world_points/conf, skipping"
                    )
                    continue
                world_points = np.asarray(world_points, dtype=np.float32)
                conf = np.asarray(conf, dtype=np.float32)
                print(f"world_points shape: {world_points.shape} for chunk {chunk_idx}")
                print(f"conf shape: {conf.shape} for chunk {chunk_idx}")
                if world_points.ndim == 3:
                    world_points = world_points[None, ...]
                if conf.ndim == 2:
                    conf = conf[None, ...]

                frame_ids = []
                frame_id_to_name = {}
                frame_paths = self.img_list[chunk_range[0] : chunk_range[1]]
                for local_idx, (frame_id, image_path) in enumerate(
                    zip(range(chunk_range[0], chunk_range[1]), frame_paths)
                ):
                    frame_ids.append(int(frame_id))
                    frame_id_to_name[str(frame_id)] = os.path.basename(image_path)

                if len(frame_ids) == 0:
                    continue

                conf_threshold = float(np.mean(conf)) * conf_threshold_coef
                mask = conf >= conf_threshold
                pts_flat = world_points[mask]
                if pts_flat.shape[0] == 0:
                    continue

                frame_idx_grid = np.broadcast_to(
                    np.arange(len(frame_ids), dtype=np.int32)[:, None, None],
                    mask.shape,
                )
                frame_idx_flat = frame_idx_grid[mask].astype(np.int32)

                finite_mask = np.isfinite(pts_flat).all(axis=1)
                if not np.all(finite_mask):
                    pts_flat = pts_flat[finite_mask]
                    frame_idx_flat = frame_idx_flat[finite_mask]
                if pts_flat.shape[0] == 0:
                    continue

                lo = np.percentile(pts_flat, 0.5, axis=0)
                hi = np.percentile(pts_flat, 99.5, axis=0)
                bbox_mask = (pts_flat >= lo).all(axis=1) & (pts_flat <= hi).all(axis=1)
                if not np.all(bbox_mask):
                    pts_flat = pts_flat[bbox_mask]
                    frame_idx_flat = frame_idx_flat[bbox_mask]
                if pts_flat.shape[0] == 0:
                    continue

                # outlier filtering: points that fall into very sparse coarse cells are likely outliers.
                coarse_cell = float(voxel_size) * 3.0
                min_points_per_cell = 10
                if coarse_cell > 0.0 and pts_flat.shape[0] > 0:
                    coarse_coords = np.floor(pts_flat / coarse_cell).astype(np.int64)
                    _, inv, counts = np.unique(
                        coarse_coords, axis=0, return_inverse=True, return_counts=True
                    )
                    dense_mask = counts[inv] >= min_points_per_cell
                    if not np.all(dense_mask):
                        pts_flat = pts_flat[dense_mask]
                        frame_idx_flat = frame_idx_flat[dense_mask]
                if pts_flat.shape[0] == 0:
                    continue

                keep_mask_full = np.zeros_like(mask, dtype=bool)
                mask_idx = np.flatnonzero(mask.reshape(-1))
                keep_idx = mask_idx[finite_mask][bbox_mask]
                if coarse_cell > 0.0 and pts_flat.shape[0] > 0:
                    keep_idx = keep_idx[dense_mask]
                keep_mask_full.reshape(-1)[keep_idx] = True

                frame_name_maps[str(chunk_idx)] = dict(frame_id_to_name)
                print("building voxel set + contributors + inverse frame->voxels map (no features)...")
                for local_idx, (frame_id, image_path) in enumerate(
                    zip(range(chunk_range[0], chunk_range[1]), frame_paths)
                ):
                    frame_keep = keep_mask_full[local_idx]
                    if not frame_keep.any():
                        continue
                    pts = np.asarray(world_points[local_idx], dtype=np.float32)
                    pts_sel = pts[frame_keep]
                    if pts_sel.shape[0] == 0:
                        continue

                    voxel_coords = np.floor(pts_sel / voxel_size).astype(np.int64)
                    unique_coords = np.unique(voxel_coords, axis=0)

                    fid_str = str(frame_id)
                    frame_name = frame_id_to_name.get(fid_str, fid_str)
                    if frame_name not in frame_to_voxels:
                        frame_to_voxels[frame_name] = set()

                    for coord in unique_coords:
                        key = (int(coord[0]), int(coord[1]), int(coord[2]))
                        if key not in coord_to_index:
                            coord_to_index[key] = len(voxel_coords_list)
                            voxel_coords_list.append(coord)
                            voxel_contribs.append(set())
                        v_i = coord_to_index[key]
                        voxel_contribs[v_i].add((int(chunk_idx), fid_str))
                        frame_to_voxels[frame_name].add(int(v_i))

                if cache_submaps:
                    submap_path = os.path.join(submap_cache_dir, f"submap_{chunk_idx}.npz")
                    np.savez_compressed(
                        submap_path,
                        pointclouds=world_points.astype(np.float32),
                        conf=conf,
                        conf_threshold=np.float32(conf_threshold),
                        frame_ids=np.asarray(frame_ids, dtype=np.int64),
                        frame_id_to_name=np.array(frame_id_to_name, dtype=object),
                        H_world_map=np.eye(4, dtype=np.float64),
                        last_non_loop_frame_index=np.array(-1, dtype=np.int64),
                        submap_id=np.int64(chunk_idx),
                    )

            if len(coord_to_index) == 0:
                vox = SemanticVoxel(
                    voxel_size=float(voxel_size),
                    centers_world=np.zeros((0, 3), dtype=np.float32),
                    features=np.zeros((0, 0), dtype=np.float32),
                    contributors=[],
                )
                voxel_map = SemanticVoxelMap(vox, frame_name_maps=frame_name_maps)
                voxel_map.save_to_directory(output_dir)
                print(f"[semantic] saved semantic voxel map to {output_dir}")
                return

            # Second pass: load frame embeddings and accumulate voxel features.
            print("Second pass: loading frame embeddings and accumulating voxel features...")
            num_vox = len(voxel_coords_list)
            feat_sums = None
            counts = np.zeros((num_vox,), dtype=np.int64)

            for chunk_idx, chunk_range in enumerate(self.chunk_indices):
                aligned_path = os.path.join(
                    self.result_aligned_dir, f"chunk_{chunk_idx}.npy"
                )
                if not os.path.exists(aligned_path):
                    continue

                chunk_data = np.load(aligned_path, allow_pickle=True).item()
                if isinstance(chunk_data, dict):
                    world_points = chunk_data.get("world_points", None)
                    conf = chunk_data.get("conf", None)
                else:
                    depth = chunk_data.depth
                    intrinsics = chunk_data.intrinsics
                    extrinsics = chunk_data.extrinsics
                    if depth.ndim == 2:
                        depth = depth[None, ...]
                    if intrinsics.ndim == 2:
                        intrinsics = intrinsics[None, ...]
                    if extrinsics.ndim == 2:
                        extrinsics = extrinsics[None, ...]
                    world_points = depth_to_point_cloud_vectorized(
                        depth, intrinsics, extrinsics
                    )
                    conf = chunk_data.conf

                if world_points is None or conf is None:
                    continue
                world_points = np.asarray(world_points, dtype=np.float32)
                conf = np.asarray(conf, dtype=np.float32)
                if world_points.ndim == 3:
                    world_points = world_points[None, ...]
                if conf.ndim == 2:
                    conf = conf[None, ...]

                frame_ids = []
                frame_id_to_name = {}
                frame_paths = self.img_list[chunk_range[0] : chunk_range[1]]
                for local_idx, (frame_id, image_path) in tqdm(enumerate(
                    zip(range(chunk_range[0], chunk_range[1]), frame_paths)
                ), desc=f"Loading frame embeddings and accumulating voxel features for chunk {chunk_idx} out of {len(self.chunk_indices)}"):
                    frame_ids.append(int(frame_id))
                    frame_id_to_name[str(frame_id)] = os.path.basename(image_path)

                if len(frame_ids) == 0:
                    continue

                conf_threshold = float(np.mean(conf)) * conf_threshold_coef
                mask = conf >= conf_threshold
                pts_flat = world_points[mask]
                if pts_flat.shape[0] == 0:
                    continue

                frame_idx_grid = np.broadcast_to(
                    np.arange(len(frame_ids), dtype=np.int32)[:, None, None],
                    mask.shape,
                )
                frame_idx_flat = frame_idx_grid[mask].astype(np.int32)

                finite_mask = np.isfinite(pts_flat).all(axis=1)
                if not np.all(finite_mask):
                    pts_flat = pts_flat[finite_mask]
                    frame_idx_flat = frame_idx_flat[finite_mask]
                if pts_flat.shape[0] == 0:
                    continue

                lo = np.percentile(pts_flat, 0.5, axis=0)
                hi = np.percentile(pts_flat, 99.5, axis=0)
                bbox_mask = (pts_flat >= lo).all(axis=1) & (pts_flat <= hi).all(axis=1)
                if not np.all(bbox_mask):
                    pts_flat = pts_flat[bbox_mask]
                    frame_idx_flat = frame_idx_flat[bbox_mask]
                if pts_flat.shape[0] == 0:
                    continue

                coarse_cell = float(voxel_size) * 3.0
                min_points_per_cell = 10
                if coarse_cell > 0.0 and pts_flat.shape[0] > 0:
                    coarse_coords = np.floor(pts_flat / coarse_cell).astype(np.int64)
                    _, inv, counts_c = np.unique(
                        coarse_coords, axis=0, return_inverse=True, return_counts=True
                    )
                    dense_mask = counts_c[inv] >= min_points_per_cell
                    if not np.all(dense_mask):
                        pts_flat = pts_flat[dense_mask]
                        frame_idx_flat = frame_idx_flat[dense_mask]
                if pts_flat.shape[0] == 0:
                    continue

                keep_mask_full = np.zeros_like(mask, dtype=bool)
                mask_idx = np.flatnonzero(mask.reshape(-1))
                keep_idx = mask_idx[finite_mask][bbox_mask]
                if coarse_cell > 0.0 and pts_flat.shape[0] > 0:
                    keep_idx = keep_idx[dense_mask]
                keep_mask_full.reshape(-1)[keep_idx] = True

                for local_idx, (frame_id, image_path) in tqdm(
                    enumerate(zip(range(chunk_range[0], chunk_range[1]), frame_paths)), 
                    desc=f"Accumulating voxel features for chunk {chunk_idx} out of {len(self.chunk_indices)}"):
                    frame_keep = keep_mask_full[local_idx]
                    if not frame_keep.any():
                        continue

                    feature_path = self._resolve_feature_path(
                        image_path, feature_dir, feature_suffix, feature_ext
                    )
                    if not os.path.exists(feature_path):
                        msg = f"[semantic] missing feature map {feature_path}"
                        if skip_missing_features:
                            print(msg + ", skipping frame")
                            continue
                        raise FileNotFoundError(msg)

                    feat = self._load_feature_map(feature_path, feature_key, image_size)
                    pts = np.asarray(world_points[local_idx], dtype=np.float32)
                    if feat.shape[0] != pts.shape[0] or feat.shape[1] != pts.shape[1]:
                        raise ValueError(
                            f"Feature map shape {feat.shape[:2]} doesn't match points "
                            f"shape {pts.shape[:2]} for {image_path}"
                        )

                    pts_sel = pts[frame_keep]
                    sem_sel = feat[frame_keep].astype(np.float32)
                    if pts_sel.shape[0] == 0:
                        continue
                    sem_finite = np.isfinite(sem_sel).all(axis=1)
                    if not np.all(sem_finite):
                        pts_sel = pts_sel[sem_finite]
                        sem_sel = sem_sel[sem_finite]
                    if pts_sel.shape[0] == 0:
                        continue

                    voxel_coords = np.floor(pts_sel / voxel_size).astype(np.int64)
                    voxel_indices = np.array(
                        [coord_to_index[tuple(c)] for c in voxel_coords], dtype=np.int64
                    )

                    if feat_sums is None:
                        feat_sums = np.zeros((num_vox, sem_sel.shape[1]), dtype=np.float32)

                    np.add.at(feat_sums, voxel_indices, sem_sel)
                    np.add.at(counts, voxel_indices, 1)

            if feat_sums is None:
                feat_sums = np.zeros((num_vox, 0), dtype=np.float32)

            centers_world = (
                (np.asarray(voxel_coords_list, dtype=np.float32) + 0.5) * voxel_size
            ).astype(np.float32)
            features = feat_sums / np.maximum(counts[:, None], 1)
            contributors = []
            for s in voxel_contribs:
                contrib_list = (
                    sorted(list(s)) if deduplicate_contributors else list(s)
                )
                contributors.append(contrib_list)
            print(f"[semantic] buildingsemantic voxel map to {output_dir}")
            vox = SemanticVoxel(
                voxel_size=float(voxel_size),
                centers_world=centers_world,
                features=features.astype(np.float32),
                contributors=contributors,
            )
            voxel_map = SemanticVoxelMap(vox, frame_name_maps=frame_name_maps)
            print(f"[semantic] saving semantic voxel map to {output_dir}")
            voxel_map.save_to_directory(output_dir)
            # Also persist inverse mapping frame -> voxel indices for later querying.
            frame_to_voxels_path = os.path.join(output_dir, "frame_to_voxels.json")
            with open(frame_to_voxels_path, "w") as f:
                json.dump(
                    {k: sorted(list(v)) for k, v in frame_to_voxels.items()}, f, indent=2
                )
            print(f"[semantic] saved semantic voxel map to {output_dir}")
            return

        else:
            print("[Let's do streaming semantic voxel map]")

    def _has_aligned_temp_results(self) -> bool:
        if not os.path.isdir(self.result_aligned_dir):
            return False
        return any(
            name.startswith("chunk_") and name.endswith(".npy")
            for name in os.listdir(self.result_aligned_dir)
        )

    def close(self):
        """
        Clean up temporary files and calculate reclaimed disk space.

        This method deletes all temporary files generated during processing from three directories:
        - Unaligned results
        - Aligned results
        - Loop results

        ~50 GiB for 4500-frame KITTI 00,
        ~35 GiB for 2700-frame KITTI 05,
        or ~5 GiB for 300-frame short seq.
        """
        if not self.delete_temp_files:
            return

        total_space = 0

        print(f"Deleting the temp files under {self.result_unaligned_dir}")
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f"Deleting the temp files under {self.result_aligned_dir}")
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f"Deleting the temp files under {self.result_loop_dir}")
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print("Deleting temp files done.")

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, os.path.basename(src_path))

        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path

    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DA3-Streaming")
    parser.add_argument("--image_dir", type=str, required=True, help="Image path")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./configs/base_config.yaml",
        help="Image path",
    )
    parser.add_argument("--output_dir", type=str, required=False, default=None, help="Output path")
    args = parser.parse_args()

    config = load_config(args.config)

    image_dir = args.image_dir
    path = image_dir.split("/")

    if args.output_dir is not None:
        save_dir = args.output_dir
    else:
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        exp_dir = "./exps"
        save_dir = os.path.join(exp_dir, image_dir.replace("/", "_"), current_datetime)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"The exp will be saved under dir: {save_dir}")
        copy_file(args.config, save_dir)

    if config["Model"]["align_lib"] == "numba":
        warmup_numba()

    da3_streaming = DA3_Streaming(image_dir, save_dir, config)
    da3_streaming.run()
    da3_streaming.close()

    del da3_streaming
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, "pcd/combined_pcd.ply")
    input_dir = os.path.join(save_dir, "pcd")
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print("DA3-Streaming done.")
    sys.exit()
