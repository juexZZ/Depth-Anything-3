import os
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from .semantic_voxel import SemanticVoxel, SemanticVoxelMap
from tqdm import tqdm
from .alignment import parse_colmap_images_txt, rmse, umeyama_sim3

class GraphMap:
    def __init__(self):
        self.submaps = dict()
    
    def get_num_submaps(self):
        return len(self.submaps)

    def add_submap(self, submap):
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
    
    def get_largest_key(self):
        if len(self.submaps) == 0:
            return -1
        return max(self.submaps.keys())
    
    def get_submap(self, id):
        return self.submaps[id]

    def get_latest_submap(self):
        return self.get_submap(self.get_largest_key())
    
    def retrieve_best_score_frame(self, query_vector, current_submap_id, ignore_last_submap=True):
        overall_best_score = 1000
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image, overall: the best submap, and the best frame index in that submap
        for submap_key in self.submaps.keys():
            if submap_key == current_submap_id:
                # skip the current submap
                continue

            if ignore_last_submap and (submap_key == current_submap_id-1):
                # skip the last submap
                continue

            else:
                submap = self.submaps[submap_key]
                submap_embeddings = submap.get_all_retrieval_vectors()
                scores = []
                for embedding in submap_embeddings:
                    # score is defined as the L2 distance between the embedding and the query vector, the smaller the better
                    score = torch.linalg.norm(embedding-query_vector)
                    scores.append(score.item())
                
                best_score_id = np.argmin(scores)
                best_score = scores[best_score_id]

                if best_score < overall_best_score:
                    overall_best_score = best_score
                    overall_best_submap_id = submap_key
                    overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index

    def get_frames_from_loops(self, loops):
        frames = []
        for detected_loop in loops:
            frames.append(self.submaps[detected_loop.detected_submap_id].get_frame_at_index(detected_loop.detected_submap_frame))
        
        return frames
    
    def update_submap_homographies(self, graph):
        for submap_key in self.submaps.keys():
            submap = self.submaps[submap_key]
            submap.set_reference_homography(graph.get_homography(submap_key).matrix())
    
    def get_submaps(self):
        return self.submaps.values()

    def ordered_submaps_by_key(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]

    def write_poses_to_file(self, file_name):
        with open(file_name, "w") as f:
            for submap in self.ordered_submaps_by_key():
                poses = submap.get_all_poses_world(ignore_loop_closure_frames=True)
                frame_ids = submap.get_frame_ids()
                assert len(poses) == len(frame_ids), "Number of provided poses and number of frame ids do not match"
                for frame_id, pose in zip(frame_ids, poses):
                    x, y, z = pose[0:3, 3]
                    rotation_matrix = pose[0:3, 0:3]
                    quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                    output = np.array([float(frame_id), x, y, z, *quaternion])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")

    def save_framewise_pointclouds(self, file_name):
        os.makedirs(file_name, exist_ok=True)
        for submap in self.ordered_submaps_by_key():
            pointclouds, frame_ids, conf_masks = submap.get_points_list_in_world_frame(ignore_loop_closure_frames=True)
            for frame_id, pointcloud, conf_masks in zip(frame_ids, pointclouds, conf_masks):
                # save pcd as numpy array
                np.savez(f"{file_name}/{frame_id}.npz", pointcloud=pointcloud, mask=conf_masks)
                

    def write_points_to_file(self, file_name):
        pcd_all = []
        colors_all = []
        for submap in self.ordered_submaps_by_key():
            pcd = submap.get_points_in_world_frame()
            pcd = pcd.reshape(-1, 3)
            pcd_all.append(pcd)
            colors_all.append(submap.get_points_colors())
        pcd_all = np.concatenate(pcd_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        if colors_all.max() > 1.0:
            colors_all = colors_all / 255.0
        pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_all))
        pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
        o3d.io.write_point_cloud(file_name, pcd_all)

    def build_semantic_voxel_map(
        self,
        voxel_size: float,
        stride: int = 1,
        ignore_loop_closure_frames: bool = True,
        deduplicate_contributors: bool = True,
        use_torch: bool = True,
    ) -> SemanticVoxelMap:
        """
        Build a *global* semantic voxel map by:
        1) Gathering all (point_world, feature) observations from every submap (already aligned to world frame).
        2) Voxelizing once in the global frame, averaging features for points that land in the same voxel.

        This handles overlaps between submaps naturally (same voxel coordinate => averaged).
        """
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be > 0")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        all_pts_world = []
        all_feats = []
        all_submap_ids = []
        all_frame_ids = []
        frame_name_maps = {}

        for submap in self.ordered_submaps_by_key():
            print(f"Building semantic voxel map for submap {submap.get_id()}...")
            if getattr(submap, "semantic_embeddings", None) is None:
                continue
            if submap.pointclouds is None or submap.conf is None or submap.conf_threshold is None:
                continue
            if submap.H_world_map is None:
                continue

            end_idx = submap.pointclouds.shape[0]
            if ignore_loop_closure_frames and (submap.last_non_loop_frame_index is not None):
                end_idx = min(end_idx, submap.last_non_loop_frame_index + 1)

            pts = submap.pointclouds[:end_idx]
            sem = submap.semantic_embeddings[:end_idx]
            conf = submap.conf[:end_idx]

            if stride > 1:
                pts = pts[:, ::stride, ::stride, :]
                sem = sem[:, ::stride, ::stride, :]
                conf = conf[:, ::stride, ::stride]

            mask = conf >= submap.conf_threshold  # (S,H,W)
            pts_flat = pts[mask]  # (N,3) in submap-local "VGGT world"
            sem_flat = sem[mask]  # (N,d)
            if pts_flat.shape[0] == 0:
                continue

            # Per-point frame provenance
            frame_idx_grid = np.broadcast_to(
                np.arange(end_idx, dtype=np.int32)[:, None, None],
                mask.shape,
            )
            frame_idx_flat = frame_idx_grid[mask].astype(np.int32)

            # Transform points into global/world frame using the submap reference transform.
            pts_h = np.concatenate([pts_flat, np.ones((pts_flat.shape[0], 1), dtype=pts_flat.dtype)], axis=1)  # (N,4)
            pts_w_h = (submap.H_world_map @ pts_h.T).T
            pts_world = (pts_w_h[:, :3] / pts_w_h[:, 3:]).astype(np.float32)

            sid = int(submap.get_id())
            
            # Build per-point frame-id strings (aligned with pts_world/sem_flat order)
            frame_ids = submap.frame_ids
            frame_id_strs = np.array([str(frame_ids[int(i)]) for i in frame_idx_flat], dtype=object)

            # -------------------------
            # Per-submap filtering (FAST)
            # -------------------------
            print(f" submap {sid} before outlier filtering, pts_world shape {pts_world.shape}")
            # 1) Remove non-finite points/features
            finite_mask = np.isfinite(pts_world).all(axis=1) & np.isfinite(sem_flat).all(axis=1)
            if not np.all(finite_mask):
                pts_world = pts_world[finite_mask]
                sem_flat = sem_flat[finite_mask]
                frame_id_strs = frame_id_strs[finite_mask]

            if pts_world.shape[0] == 0:
                continue

            # 2) Robust bbox filter (percentiles) to drop extreme tails quickly
            lo = np.percentile(pts_world, 0.5, axis=0)
            hi = np.percentile(pts_world, 99.5, axis=0)
            bbox_mask = (pts_world >= lo).all(axis=1) & (pts_world <= hi).all(axis=1)
            if not np.all(bbox_mask):
                pts_world = pts_world[bbox_mask]
                sem_flat = sem_flat[bbox_mask]
                frame_id_strs = frame_id_strs[bbox_mask]

            if pts_world.shape[0] == 0:
                continue

            # 3) Cheap "isolation" outlier filter via coarse occupancy grid:
            #    Points that fall into very sparse coarse cells are likely outliers.
            #    This is much faster than global Open3D radius filtering.
            coarse_cell = float(voxel_size) * 3.0
            min_points_per_cell = 10
            if coarse_cell > 0.0 and pts_world.shape[0] > 0:
                coarse_coords = np.floor(pts_world / coarse_cell).astype(np.int64)
                _, inv, counts = np.unique(coarse_coords, axis=0, return_inverse=True, return_counts=True)
                dense_mask = counts[inv] >= min_points_per_cell
                if not np.all(dense_mask):
                    pts_world = pts_world[dense_mask]
                    sem_flat = sem_flat[dense_mask]
                    frame_id_strs = frame_id_strs[dense_mask]
                    
            print(f" submap {sid} after outlier filtering, pts_world shape {pts_world.shape}")

            if pts_world.shape[0] == 0:
                continue

            # Append filtered observations
            all_pts_world.append(pts_world)
            all_feats.append(sem_flat.astype(np.float32))
            all_submap_ids.append(np.full((pts_world.shape[0],), sid, dtype=np.int32))
            all_frame_ids.append(frame_id_strs)

            # Collect mapping frame_id -> filename for this submap (used for later lookup).
            # Stored as: { "<submap_id>": { "<frame_id>": "<filename>" } }
            print(f"store submap {sid} frame name mappings as dict <submap_id> -> dict(<frame_id>:<filename>)")
            if getattr(submap, "frame_id_to_name", None) is not None:
                frame_name_maps[str(sid)] = dict(submap.frame_id_to_name)

        if len(all_pts_world) == 0:
            vox = SemanticVoxel(
                voxel_size=float(voxel_size),
                centers_world=np.zeros((0, 3), dtype=np.float32),
                features=np.zeros((0, 0), dtype=np.float32),
                contributors=[],
            )
            return SemanticVoxelMap(vox, frame_name_maps=frame_name_maps)

        pts_world_all = np.concatenate(all_pts_world, axis=0)
        feats_all = np.concatenate(all_feats, axis=0)
        submap_ids_all = np.concatenate(all_submap_ids, axis=0)
        frame_ids_all = np.concatenate(all_frame_ids, axis=0)
        # print sizes
        print(f"[After per-submap filtering] gathered global stats: pts_world_all shape {pts_world_all.shape}, feats_all shape {feats_all.shape}, submap_ids_all shape {submap_ids_all.shape}, frame_ids_all shape {frame_ids_all.shape}")

        # Global voxelization
        print(f"Global voxelization with voxel size {voxel_size}...")
        inverse = None
        unique_coords = None
        centers_world = None
        feat_avg = None

        if use_torch and torch.cuda.is_available():
            # GPU path: voxelize + average features on CUDA.
            device = torch.device("cuda")
            pts_t = torch.from_numpy(pts_world_all).to(device=device, dtype=torch.float32)  # (N,3)
            feats_t = torch.from_numpy(feats_all)   # (N,d)

            voxel_coords_t = torch.floor(pts_t / float(voxel_size)).to(torch.int64)  # (N,3)
            unique_coords_t, inverse_t = torch.unique(voxel_coords_t, dim=0, return_inverse=True)
            num_vox = int(unique_coords_t.shape[0])
            print(f"number of final voxels: {num_vox}")
            d = int(feats_t.shape[1])
            feat_sum_t = torch.zeros((num_vox, d), device=device, dtype=torch.float32)
            # feat_sum_t.index_add_(0, inverse_t, feats_t)
            # gradually bring the features to cuda and add them to the feat_sum_t
            for i in tqdm(range(0, feats_t.shape[0], 1000), desc="Adding features to feat_sum_t by chunk 1000"):
                feats_t_chunk = feats_t[i:i+1000].to(device=device, dtype=torch.float32)
                feat_sum_t.index_add_(0, inverse_t[i:i+1000], feats_t_chunk)
            counts_t = torch.bincount(inverse_t, minlength=num_vox).to(torch.float32)  # (num_vox,)
            feat_avg_t = feat_sum_t / counts_t[:, None].clamp_min(1.0)

            centers_world_t = (unique_coords_t.to(torch.float32) + 0.5) * float(voxel_size)

            # Back to CPU numpy for the SemanticVoxel container / visualization stack.
            unique_coords = unique_coords_t.detach().cpu().numpy()
            inverse = inverse_t.detach().cpu().numpy()
            centers_world = centers_world_t.detach().cpu().numpy().astype(np.float32)
            feat_avg = feat_avg_t.detach().cpu().numpy().astype(np.float32)
        else:
            # CPU numpy path (fallback)
            voxel_coords = np.floor(pts_world_all / voxel_size).astype(np.int64)  # (N,3)
            unique_coords, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
            num_vox = unique_coords.shape[0]
            print(f"number of final voxels: {num_vox}")
            d = feats_all.shape[1]
            feat_sum = np.zeros((num_vox, d), dtype=np.float32)
            counts = np.zeros((num_vox,), dtype=np.int64)
            np.add.at(feat_sum, inverse, feats_all)
            np.add.at(counts, inverse, 1)
            feat_avg = feat_sum / counts[:, None]

            centers_world = ((unique_coords.astype(np.float32) + 0.5) * voxel_size).astype(np.float32)

        # Contributors per voxel
        if deduplicate_contributors:
            contrib_sets = [set() for _ in range(num_vox)]
            for p_i, v_i in enumerate(inverse.tolist()):
                contrib_sets[v_i].add((int(submap_ids_all[p_i]), str(frame_ids_all[p_i])))
            contributors = [sorted(list(s)) for s in contrib_sets]
        else:
            contributors = [[] for _ in range(num_vox)]
            for p_i, v_i in enumerate(inverse.tolist()):
                contributors[v_i].append((int(submap_ids_all[p_i]), str(frame_ids_all[p_i])))

        vox = SemanticVoxel(
            voxel_size=float(voxel_size),
            centers_world=centers_world,
            features=feat_avg,
            contributors=contributors,
        )
        return SemanticVoxelMap(vox, frame_name_maps=frame_name_maps)

    def apply_similarity_transform(self, T_world_from_pred: np.ndarray) -> None:
        """
        Apply a global similarity/affine transform to the entire map by left-multiplying each submap's
        reference transform:
            H_world_map := T_world_from_pred @ H_world_map
        """
        T = np.asarray(T_world_from_pred, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T_world_from_pred must be 4x4, got {T.shape}")
        for submap in self.ordered_submaps_by_key():
            H = submap.get_reference_homography()
            if H is None:
                continue
            submap.set_reference_homography((T @ H).astype(np.float64))

    def align_scale_to_colmap(
        self,
        colmap_images_txt: str,
        with_scale: bool = True,
        ignore_loop_closure_frames: bool = True,
    ) -> np.ndarray:
        """
        Compute a global Sim(3) alignment between predicted camera centers and COLMAP GT camera centers,
        then apply it to all submaps via `apply_similarity_transform`.

        Returns:
            T_world_from_pred (4x4): maps predicted-world points into COLMAP/world coordinates.
        """
        gt_centers_by_name = parse_colmap_images_txt(colmap_images_txt)  # basename -> (3,)

        pred_pts = []
        gt_pts = []
        matched = 0

        for submap in self.ordered_submaps_by_key():
            # predicted cam2world poses in the current predicted world frame
            poses = submap.get_all_poses_world(ignore_loop_closure_frames=ignore_loop_closure_frames)
            if poses is None:
                continue
            # basenames, aligned to poses
            names = getattr(submap, "frame_names", None)
            if names is None:
                # fallback to frame_id->name mapping
                id_to_name = submap.frame_id_to_name
                frame_ids = submap.get_frame_ids()
                names = [id_to_name[str(fid)] for fid in frame_ids]

            if len(names) != poses.shape[0]:
                # can't align if we can't match per-frame
                print(f"can't align submap {submap.get_id()} because it has {len(names)} frames but {poses.shape[0]} poses")
                continue

            for name, pose in zip(names, poses):
                basename = str(name).split("/")[-1]
                if basename not in gt_centers_by_name:
                    continue
                pred_center = pose[:3, 3].astype(np.float64)
                gt_center = gt_centers_by_name[basename].astype(np.float64)
                pred_pts.append(pred_center)
                gt_pts.append(gt_center)
                matched += 1

        if matched < 3:
            raise RuntimeError(f"Need >=3 matched frames for alignment; got {matched}.")

        pred_pts_np = np.stack(pred_pts, axis=0)
        gt_pts_np = np.stack(gt_pts, axis=0)

        before = rmse(pred_pts_np, gt_pts_np)
        sim3 = umeyama_sim3(pred_pts_np, gt_pts_np, with_scale=with_scale)
        T = sim3.as_matrix()
        after = rmse(((sim3.s * (sim3.R @ pred_pts_np.T)).T + sim3.t[None, :]), gt_pts_np)

        print(f"[align] matched frames: {matched}")
        print(f"[align] RMSE before: {before:.4f}  after: {after:.4f}")
        print(f"[align] scale: {sim3.s:.6f}")

        self.apply_similarity_transform(T)
        return T