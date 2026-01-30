#!/bin/bash

# first, run da3 without getting the voxel map
CUDA_VISIBLE_DEVICES=4 python da3_streaming.py \
    --image_dir /local_data/jz4725/metacam/data_v3/2ndfloor/images \
    --config ./configs/base_config.yaml \
    --output_dir outputs/2ndfloor

CUDA_VISIBLE_DEVICES=4 python da3_streaming.py \
    --image_dir /local_data/jz4725/metacam/data_v3/12thfloor/images \
    --config ./configs/base_config.yaml \
    --output_dir outputs/12thfloor