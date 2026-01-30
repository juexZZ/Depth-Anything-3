#!/bin/bash

# first, run da3 without getting the voxel map
CUDA_VISIBLE_DEVICES=4 python da3_streaming.py \
    --image_dir /local_data/jz4725/metacam/data_v3/2ndfloor/image_trav_1 \
    --config ./configs/base_config.yaml \
    --output_dir outputs/2ndfloor_trav1

CUDA_VISIBLE_DEVICES=4 python da3_streaming.py \
    --image_dir /local_data/jz4725/metacam/data_v3/12thfloor/image_trav_1 \
    --config ./configs/base_config.yaml \
    --output_dir outputs/12thfloor_trav1

CUDA_VISIBLE_DEVICES=4 python da3_streaming.py \
    --image_dir /local_data/jz4725/metacam/data_v3/6metro_add/image_trav_1 \
    --config ./configs/base_config.yaml \
    --output_dir outputs/6metro_add_trav1