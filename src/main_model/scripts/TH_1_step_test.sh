#!/bin/bash


# TH 1 step
# Base directory containing the 16 directories
BASE_DIR="/Dataset_train/TH_1_step/TH_S/"

MODELS_DIR="Trained_models/T5_1_step/"
OUTPUT_BASE_DIR="/pred_results/TH/TH_1_step"

for DIR in "$BASE_DIR"/*/; do
    # Extract the directory name (without the full path)
    DIR_NAME=$(basename "$DIR")

    OUTPUT_DIR="$OUTPUT_BASE_DIR/$DIR_NAME"

    python3 run/test.py \
        -t "$DIR" \
        -m "$MODELS_DIR" \
        -s "$OUTPUT_DIR" \
        -d "$DIR_NAME" \
        --model_name T5 \
        -b 16
done