#!/bin/bash

#! Stream semantic voxel map build from results_output
#! semantic embedding is specified in the config file

# # first, run da3 without getting the voxel map
# CUDA_VISIBLE_DEVICES=2 python da3_streaming.py \
#     --image_dir /local_data/jz4725/metacam/data_v3/8thfloor_move/images \
#     --config ./configs/base_config.yaml \
#     --output_dir outputs/8thfloor_move

# CUDA_VISIBLE_DEVICES=2 python da3_streaming.py \
#     --image_dir /local_data/jz4725/metacam/data_v3/8thfloor_remove/images \
#     --config ./configs/base_config.yaml \
#     --output_dir outputs/8thfloor_remove

# CUDA_VISIBLE_DEVICES=0 python stream_build_voxel_map.py \
#     --image_dir /local_data/jz4725/metacam/data_v3/8thfloor_move/images \
#     --config ./configs/stream_voxel.yaml \
#     --output_dir outputs/8thfloor_move \
#     --result_subdir no_subtract \
#     --feature_dir /local_data/jz4725/metacam/data_v3/8thfloor_move/semantic_emb

CUDA_VISIBLE_DEVICES=0 python stream_build_voxel_map.py \
    --image_dir /local_data/jz4725/metacam/data_v3/8thfloor_move/images \
    --config ./configs/stream_voxel.yaml \
    --output_dir outputs/8thfloor_move \
    --result_subdir subtract \
    --feature_dir /local_data/jz4725/metacam/data_v3/8thfloor_move/semantic_emb \
    --subtract_free_space

## Example alternate scenes:
# CUDA_VISIBLE_DEVICES=3 python stream_build_voxel_map.py \
#     --image_dir /local_data/jz4725/metacam/data_v3/8thfloor_remove/images \
#     --config ./configs/base_config.yaml \
#     --output_dir outputs/8thfloor_remove \
#     --feature_dir /local_data/jz4725/metacam/data_v3/8thfloor_remove/semantic_emb
