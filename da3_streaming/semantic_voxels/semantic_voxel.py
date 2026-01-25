from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json
import os

import numpy as np
import torch

@dataclass
class SemanticVoxel:
    """
    Dense semantic voxel map produced by aggregating per-pixel features onto 3D points.

    - `centers_world`: (N, 3) voxel centers in world frame.
    - `features`:      (N, d) averaged semantic feature per voxel.
    - `contributors`:  length-N list; each entry is a list of (submap_id, frame_idx) pairs
                       that contributed points/features to that voxel.
    """

    voxel_size: float
    centers_world: np.ndarray
    features: np.ndarray
    contributors: List[List[Tuple[int, str]]]
    

class SemanticVoxelMap:
    """Semantic voxel map that holds the semantic voxel data and handles saving, loading, querying and visualization."""
    def __init__(self, voxels: SemanticVoxel, frame_name_maps: Dict[str, Dict[str, str]]):
        self.voxels = voxels
        self.voxel_size = float(voxels.voxel_size)
        # submap_id(str) -> {frame_id(str) -> filename(str)}
        self.frame_name_maps = frame_name_maps

        # Build integer voxel coords (so queries don't rely on float equality).
        self._voxel_coords = self._centers_to_voxel_coords(self.voxels.centers_world, self.voxel_size)
        self._coord_to_index: Dict[Tuple[int, int, int], int] = {
            (int(c[0]), int(c[1]), int(c[2])): i for i, c in enumerate(self._voxel_coords)
        }

    def get_voxels(self) -> SemanticVoxel:
        return self.voxels

    def get_voxel_size(self) -> float:
        return self.voxel_size

    def get_centers_world(self) -> np.ndarray:
        return self.voxels.centers_world

    def get_features(self) -> np.ndarray:
        return self.voxels.features

    def get_contributors(self) -> List[List[Tuple[int, str]]]:
        return self.voxels.contributors

    def resolve_contributor(self, submap_id: int, frame_id: str) -> Optional[str]:
        """Return original filename for a (submap_id, frame_id) contributor, if known."""
        submap_map = self.frame_name_maps.get(str(submap_id))
        if submap_map is None:
            return None
        return submap_map.get(str(frame_id))

    @staticmethod
    def _centers_to_voxel_coords(centers_world: np.ndarray, voxel_size: float) -> np.ndarray:
        # centers are computed as (coord + 0.5) * voxel_size; invert that to integer coords.
        # centers_world = ((unique_coords.astype(np.float32) + 0.5) * voxel_size).astype(np.float32)
        return np.floor(centers_world / voxel_size - 0.5).astype(np.int64)

    @staticmethod
    def _position_to_voxel_coord(position_world: np.ndarray, voxel_size: float) -> Tuple[int, int, int]:
        p = np.asarray(position_world, dtype=np.float32).reshape(3)
        c = np.floor(p / voxel_size).astype(np.int64)
        return int(c[0]), int(c[1]), int(c[2])

    def get_index_at_position(self, position_world: np.ndarray) -> Optional[int]:
        """
        Return voxel index for a world position by computing its voxel coordinate.
        Returns None if no voxel exists at that coordinate.
        """
        key = self._position_to_voxel_coord(position_world, self.voxel_size)
        return self._coord_to_index.get(key, None)

    def get_features_at_position(self, position_world: np.ndarray) -> Optional[np.ndarray]:
        idx = self.get_index_at_position(position_world)
        if idx is None:
            return None
        return self.voxels.features[idx]
    
    def get_voxel_coord_at_index(self, index: int) -> Optional[Tuple[int, int, int]]:
        return self._voxel_coords[index]

    def get_contributors_at_position(self, position_world: np.ndarray) -> Optional[List[Tuple[int, str]]]:
        idx = self.get_index_at_position(position_world)
        if idx is None:
            return None
        return self.voxels.contributors[idx]
    
    def query_with_embedding(self, qe: np.ndarray, top_k: int = 1):
        """
        Query the voxelmap with a embedding vector (from a image or a text query), 
        return the most similar voxel inde, coordinates and the similarity score
        """
        feats = self.voxels.features
        if isinstance(feats, np.ndarray):
            feats_t = torch.from_numpy(feats).float()
        else:
            feats_t = feats.float()
        qe_t = torch.from_numpy(qe).float()
        if qe_t.ndim == 1:
            qe_t = qe_t[None, :]
        # cosine similarity assumes embeddings are normalized; otherwise this is just dot-product similarity
        similarities = torch.matmul(feats_t, qe_t.T).squeeze(-1)
        topk_indices = torch.topk(similarities, top_k).indices
        topk_indices = topk_indices.tolist()
        topk_coords = self._voxel_coords[topk_indices]
        topk_similarities = similarities[topk_indices].tolist()
        return topk_indices, topk_coords, topk_similarities
    
    def get_latest_frame_at_voxel(self, voxel_index: int) -> Optional[str]:
        """
        Return the latest frame at a voxel index.
        """
        voxel_contributors = self.voxels.contributors[voxel_index]
        # rank revers by submap id and then frame id
        voxel_contributors.sort(key=lambda x: (x[0], x[1]), reverse=True)
        submap_id, frame_id = voxel_contributors[0]
        return self.resolve_contributor(submap_id, frame_id), submap_id, frame_id

    def save_to_directory(self, directory_path: str) -> None:
        """
        Save semantic voxel map to a directory:
        - `voxels.npz`: voxel_size, centers_world, features, contributors
        - `frame_names.json`: {submap_id: {frame_id: filename}}
        - `voxel_frame_names.json`: {voxel_index: [frame_name, ...]}

        Note: contributors are stored as an object array; loading requires allow_pickle=True.
        """
        os.makedirs(directory_path, exist_ok=True)
        npz_path = os.path.join(directory_path, "semantic_voxels.npz")
        json_path = os.path.join(directory_path, "frame_names.json")
        voxel_frames_path = os.path.join(directory_path, "voxel_frame_names.json")
        contrib_arr = np.array(self.voxels.contributors, dtype=object)
        np.savez_compressed(
            npz_path,
            voxel_size=np.float32(self.voxel_size),
            centers_world=self.voxels.centers_world.astype(np.float32),
            features=self.voxels.features.astype(np.float32),
            contributors=contrib_arr,
        )
        with open(json_path, "w") as f:
            json.dump(self.frame_name_maps, f, indent=2)
        with open(voxel_frames_path, "w") as f:
            json.dump(self.get_voxel_frame_names(), f, indent=2)

    @staticmethod
    def load_from_directory(directory_path: str) -> "SemanticVoxelMap":
        npz_path = os.path.join(directory_path, "semantic_voxels.npz")
        json_path = os.path.join(directory_path, "frame_names.json")
        data = np.load(npz_path, allow_pickle=True)
        frame_name_maps: Dict[str, Dict[str, str]] = {}
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                frame_name_maps = json.load(f)
        vox = SemanticVoxel(
            voxel_size=float(data["voxel_size"]),
            centers_world=data["centers_world"],
            features=data["features"],
            contributors=list(data["contributors"].tolist()),
        )
        return SemanticVoxelMap(vox, frame_name_maps=frame_name_maps)

    def get_voxel_frame_names(self) -> Dict[str, List[str]]:
        """
        Build a {voxel_index: [frame_name,...]} dict for external querying.
        """
        voxel_to_frames: Dict[str, List[str]] = {}
        for idx, contributors in enumerate(self.voxels.contributors):
            names: List[str] = []
            for submap_id, frame_id in contributors:
                name = self.resolve_contributor(submap_id, frame_id)
                if name is None:
                    name = f"{submap_id}:{frame_id}"
                names.append(name)
            voxel_to_frames[str(idx)] = names
        return voxel_to_frames

    @staticmethod
    def _features_to_rgb(features: np.ndarray, max_points_for_pca: int = 20000) -> np.ndarray:
        """
        Convert (N,d) features into (N,3) RGB colors in [0,1].
        - If d==3: min-max normalize each channel.
        - If d<3: pad/replicate.
        - If d>3: PCA to 3 dims then min-max normalize.
        """
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"features must be (N,d), got shape={x.shape}")
        n, d = x.shape
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if d == 3:
            y = x
        elif d == 1:
            y = np.repeat(x, 3, axis=1)
        elif d == 2:
            y = np.concatenate([x, np.zeros((n, 1), dtype=np.float32)], axis=1)
        else:
            # PCA via SVD on a subsample if large.
            if n > max_points_for_pca:
                idx = np.random.choice(n, size=max_points_for_pca, replace=False)
                x_fit = x[idx]
            else:
                x_fit = x
            x_fit = x_fit - x_fit.mean(axis=0, keepdims=True)
            # Vt: (d,d); take first 3 principal directions
            _, _, vt = np.linalg.svd(x_fit, full_matrices=False)
            comps = vt[:3].T  # (d,3)
            y = (x - x.mean(axis=0, keepdims=True)) @ comps  # (N,3)

        # Robust-ish min-max normalization per channel
        y_min = y.min(axis=0, keepdims=True)
        y_ptp = y.ptp(axis=0, keepdims=True) + 1e-8
        rgb = (y - y_min) / y_ptp
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)

    def visualize(
        self,
        port: int = 8081,
        name: str = "semantic_voxels",
        point_size: float = 0.01,
        color_mode: str = "pca",
        render_mode: str = "points",
        max_voxels: Optional[int] = 20000,
        query_voxel_indices: Optional[List[int]] = None,
        base_color: Tuple[float, float, float] = (0.75, 0.75, 0.75),
        highlight_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        wireframe: bool = False,
        opacity: Optional[float] = None,
        keep_alive: bool = True,
    ):
        """
        Visualize the semantic voxel map with viser.

        Args:
            port: viser server port.
            name: scene node name.
            point_size: point size in viser.
            color_mode:
                - "pca": PCA-reduce features to RGB (default, works for any d>=1)
                - "first3": use first 3 feature dims as RGB (requires d>=3)
                - "ones": constant color
                - "query": color all voxels with `base_color` and highlight voxels in `query_voxel_indices`
            render_mode:
                - "points": render voxel centers as a point cloud (fast, default)
                - "cubes": render each voxel as a cube using `scene.add_box` (can be slow for many voxels)
            max_voxels: if not None, subsample to at most this many voxels for visualization.
            query_voxel_indices: list of voxel indices (into the full voxelmap) to highlight when color_mode="query".
            base_color: RGB in [0,1] for non-highlight voxels when color_mode="query".
            highlight_color: RGB in [0,1] for highlighted voxels when color_mode="query".
            wireframe: if render_mode="cubes", draw cubes as wireframes.
            opacity: if render_mode="cubes", set cube opacity (None lets viser default).
            keep_alive: block and keep the server alive until user presses Enter.

        Returns:
            (server, handle)
        """
        import viser  # local import to keep this module importable without viser installed

        points = self.voxels.centers_world.astype(np.float32)
        feats = self.voxels.features.astype(np.float32)
        orig_indices = np.arange(points.shape[0], dtype=np.int64)

        # Optional subsample for performance
        if max_voxels is not None and points.shape[0] > max_voxels:
            print(f"Subsampling to {max_voxels} voxels out of {points.shape[0]} for visualization...")
            idx = np.random.choice(points.shape[0], size=max_voxels, replace=False)
            points = points[idx]
            feats = feats[idx]
            orig_indices = orig_indices[idx]

        if color_mode == "query":
            colors = np.tile(np.asarray(base_color, dtype=np.float32)[None, :], (points.shape[0], 1))
            if query_voxel_indices is not None and len(query_voxel_indices) > 0:
                qset = set(int(i) for i in query_voxel_indices)
                mask = np.array([int(i) in qset for i in orig_indices.tolist()], dtype=bool)
                colors[mask] = np.asarray(highlight_color, dtype=np.float32)[None, :]
        elif color_mode == "ones":
            colors = np.ones((points.shape[0], 3), dtype=np.float32)
        elif color_mode == "first3":
            if feats.shape[1] < 3:
                raise ValueError(f"color_mode='first3' requires d>=3, got d={feats.shape[1]}")
            colors = self._features_to_rgb(feats[:, :3])
        elif color_mode == "pca":
            colors = self._features_to_rgb(feats)
        else:
            raise ValueError(f"Unknown color_mode={color_mode}")

        server = viser.ViserServer(host="0.0.0.0", port=port)
        handle = None
        if render_mode == "points":
            handle = server.scene.add_point_cloud(
                name=name,
                points=points,
                colors=colors,
                point_size=point_size,
                point_shape="circle",
            )
        elif render_mode == "cubes":
            # Add one cube per voxel. This can be expensive if there are many voxels.
            dims = (float(self.voxel_size), float(self.voxel_size), float(self.voxel_size))
            for i in range(points.shape[0]):
                c = colors[i]
                server.scene.add_box(
                    name=f"{name}/voxel_{i}",
                    position=(float(points[i, 0]), float(points[i, 1]), float(points[i, 2])),
                    dimensions=dims,
                    color=(float(c[0]), float(c[1]), float(c[2])),
                    wireframe=wireframe, # not available in vider 0.2.23
                    opacity=opacity,
                )
        else:
            raise ValueError(f"Unknown render_mode={render_mode}")

        if keep_alive:
            print(f"Viser server running on port {port}. Press Enter to exit...")
            try:
                input()
            except KeyboardInterrupt:
                pass

        return server, handle
