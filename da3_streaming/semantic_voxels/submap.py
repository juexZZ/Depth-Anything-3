from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class SemanticSubmap:
    def __init__(
        self,
        submap_id: int,
        pointclouds: np.ndarray,
        semantic_embeddings: np.ndarray,
        conf: np.ndarray,
        conf_threshold: float,
        frame_ids: np.ndarray,
        frame_id_to_name: Dict[str, str],
        H_world_map: Optional[np.ndarray] = None,
        last_non_loop_frame_index: Optional[int] = None,
    ) -> None:
        self._id = int(submap_id)
        self.pointclouds = pointclouds
        self.semantic_embeddings = semantic_embeddings
        self.conf = conf
        self.conf_threshold = float(conf_threshold)
        self.frame_ids = np.asarray(frame_ids, dtype=np.int64)
        self.frame_id_to_name = dict(frame_id_to_name)
        self.H_world_map = (
            np.asarray(H_world_map, dtype=np.float64) if H_world_map is not None else np.eye(4)
        )
        self.last_non_loop_frame_index = last_non_loop_frame_index

    def get_id(self) -> int:
        return self._id
