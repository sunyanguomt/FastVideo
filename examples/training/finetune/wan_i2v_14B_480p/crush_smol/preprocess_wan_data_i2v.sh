#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="/data/caizhi/wan2.1_i2v_diffusers"
MODEL_TYPE="wan"
DATA_MERGE_PATH="/data/caizhi/FastVideo_wan2.2/FastVideo/examples/training/finetune/wan_i2v_14B_480p/crush_smol/data/crush-smol/merge.txt"
DATASET_PATH="/data/caizhi/FastVideo_wan2.2/FastVideo/examples/training/finetune/wan_i2v_14B_480p/crush_smol/data/crush-smol/"
OUTPUT_DIR="/data/caizhi/FastVideo_wan2.2/FastVideo/examples/training/finetune/wan_i2v_14B_480p/crush_smol/data/crush-smol_processed_i2v_not_new"

torchrun --master_port 29502 --nproc_per_node=$GPU_NUM \
    /data/caizhi/FastVideo_wan2.2/FastVideo/fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 8 \
    --seed 42 \
    --max_height 480 \
    --max_width 832 \
    --num_frames 77 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --train_fps 16 \
    --samples_per_file 8 \
    --flush_frequency 8 \
    --video_length_tolerance_range 5 \
    --preprocess_task "i2v" 
