#!/usr/bin/env bash


python3 run/train.py \
    --model_name_or_path /T5-base \
    --do_train \
    --seed=88 \
    --save_total_limit=1 \
    --train_file Dataset_train/Thai_1_step/jsonl_data/train.json \
    --validation_file Dataset_train/Thai_1_step/jsonl_data/val.json \
    --output_dir Trained_models/T5_1_step \
    --per_device_train_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate