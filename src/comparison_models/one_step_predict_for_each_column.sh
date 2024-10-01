#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Base directory containing the CSV files
base_dir="/home/eutaboo/Downloads/southeastAsia"

# List of directories (assuming each directory has a `final_combined_data.csv`)
directories=("TH")


# List of models to iterate through
models=("DLinear" "FEDformer" "FiLM" "Informer" "iTransformer" "Koopa" "LightTS" "Pyraformer" "Reformer" "TSMixer" "Transformer")

# Iterate through each directory
for dir in "${directories[@]}"; do
  csv_file="$base_dir/$dir/combined_data_without_6.csv"

  # Get the list of columns (excluding the first column which is typically time)
  columns=$(head -1 $csv_file | tr ',' '\n' | tail -n +2 | tr -d '\r')

  # Iterate through each column
  for column in $columns; do

    # Iterate through each model
    for model_name in "${models[@]}"; do
      python -u /home/eutaboo/PycharmProjects/Time-Series-Library/Time-Series-Library/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$base_dir/$dir/" \
        --data_path "combined_data_without_6.csv" \
        --model_id "${dir}_${column}_24_5" \
        --model $model_name \
        --data custom \
        --features S \
        --seq_len 24 \
        --label_len 12 \
        --pred_len 5 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --des 'Exp' \
        --itr 1 \
        --target "$column"
    done

  done

done

