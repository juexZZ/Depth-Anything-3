#!/bin/bash

#! sem embedding is specidied in the config file
# 8thfloor_add
CUDA_VISIBLE_DEVICES=3 python da3_streaming.py \
    --image_dir /local_data/jz4725/metacam/data_v3/8thfloor_add/images \
    --config ./configs/base_config.yaml \
    --output_dir outputs/8thfloor_add_15M


# CUDA_VISIBLE_DEVICES=3 python da3_streaming.py \
#     --image_dir /local_data/jz4725/metacam/data_v3/8thfloor_remove/images \
#     --config ./configs/base_config.yaml \
#     --output_dir outputs/8thfloor_remove
    
# # traverse 12
# CUDA_VISIBLE_DEVICES=3 python da3_streaming.py \
#     --image_dir /local_data/jz4725/metacam/data_v3/8thfloor_small_5times/image_trav_12 \
#     --config ./configs/base_config.yaml \
#     --output_dir outputs/8thfloor_5times_trav_12

# # traverse 123
# CUDA_VISIBLE_DEVICES=3 python da3_streaming.py \
#     --image_dir /local_data/jz4725/metacam/data_v3/8thfloor_small_5times/image_trav_123 \
#     --config ./configs/base_config.yaml \
#     --output_dir outputs/8thfloor_5times_trav_123

# # traverse 1234
# CUDA_VISIBLE_DEVICES=3 python da3_streaming.py \
#     --image_dir /local_data/jz4725/metacam/data_v3/8thfloor_small_5times/image_trav_1234 \
#     --config ./configs/base_config.yaml \
#     --output_dir outputs/8thfloor_5times_trav_1234

# # traverse 12345
# CUDA_VISIBLE_DEVICES=3 python da3_streaming.py \
#     --image_dir /local_data/jz4725/metacam/data_v3/8thfloor_small_5times/images \
#     --config ./configs/base_config.yaml \
#     --output_dir outputs/8thfloor_5times_full