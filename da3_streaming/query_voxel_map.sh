#!/bin/bash

DATA_DIR=/local_data/jz4725/metacam/data_v3/8thfloor_add/
SCENE_RES_DIR=outputs/8thfloor_add
python semantic_voxels/query_voxelmap.py \
    --voxel_map_dir $SCENE_RES_DIR/no_subtract/semantic_voxels/ \
    --output_dir $SCENE_RES_DIR/ \
    --image_dir $DATA_DIR/images/ \
    --vis_pcd \
    --pcd_path $SCENE_RES_DIR/pcd/combined_pcd.ply \
    --query_prompt "carrot"