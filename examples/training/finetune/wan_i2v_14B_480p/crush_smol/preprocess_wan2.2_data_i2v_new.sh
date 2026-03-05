#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="/data/caizhi/wan2.2_i2v_diffusers/"
DATASET_PATH="/data/caizhi/FastVideo_wan2.2/FastVideo/examples/training/finetune/wan_i2v_14B_480p/crush_smol/data/crush-smol/"
OUTPUT_DIR="/data/caizhi/FastVideo_wan2.2/FastVideo/examples/training/finetune/wan_i2v_14B_480p/crush_smol/data/wan2.2_crush-smol_processed_i2v/"

torchrun --master_port 29501 --nproc_per_node=$GPU_NUM \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path $MODEL_PATH \
    --mode preprocess \
    --workload_type i2v \
    --preprocess.dataset_type merged \
    --preprocess.dataset_path $DATASET_PATH \
    --preprocess.dataset_output_dir $OUTPUT_DIR \
    --preprocess.preprocess_video_batch_size 2 \
    --preprocess.dataloader_num_workers 0 \
    --preprocess.max_height 480 \
    --preprocess.max_width 832 \
    --preprocess.num_frames 77 \
    --preprocess.train_fps 16 \
    --preprocess.samples_per_file 8 \
    --preprocess.flush_frequency 8 \
    --preprocess.video_length_tolerance_range 5
