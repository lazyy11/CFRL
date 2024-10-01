import pandas as pd
import numpy as np
import os

def generate_prompts(df, window_len, pre_len, output_path, k):
    with open(output_path, 'w') as file:
        for i in range(len(df) - window_len - pre_len):
            start_date = df.index[i].strftime('%m/%d/%Y %I:%M:%S %p')
            end_date = df.index[i + window_len - 1].strftime('%m/%d/%Y %I:%M:%S %p')
            true_values = ', '.join(
                f"{df.iloc[j, k]:.15g}" for j in range(i, i + window_len)
            )
            next_hours = ', '.join(
                df.index[i + window_len + j].strftime('%m/%d/%Y %I:%M:%S %p') for j in range(pre_len)
            )
            file.write(f"From {start_date} to {end_date}, the observation of {df.columns[k]} for the past is {true_values} on each hour. What is prediction on {next_hours}?\n")

def generate_labels(df, window_len, pre_len, output_path, k):
    with open(output_path, 'w') as file:
        for i in range(len(df) - window_len - pre_len):
            labels = ', '.join(
                f"{df.iloc[i + window_len + j, k]:.15g}" for j in range(pre_len)
            )
            file.write(f"The prediction is {labels}.\n")

base_path = '/home/eutaboo/Downloads/southeastAsia/'
files_name = ['TH']
for f in files_name:
    input_path = base_path + f + '/final_combined_data.csv'
    df_daily = pd.read_csv(input_path, parse_dates=[0], index_col=0)

    total_rows = len(df_daily)
    train_size = int(total_rows * 0.7)
    val_size = int(total_rows * 0.1)
    test_size = total_rows - train_size - val_size

    # df_train = df_daily.iloc[:train_size]
    # df_val = df_daily.iloc[train_size:train_size + val_size]
    df_test = df_daily.iloc[train_size + val_size:]

    window_len = 24
    pre_len = 1

    output_folder = f'/home/eutaboo/PycharmProjects/PromptCast/LMP/Thailand/Dataset_test/TH_5_steps/{f}_S/'
    os.makedirs(output_folder, exist_ok=True)

    for k in range(len(df_test.columns)):
        columns = df_test.columns
        path = output_folder + columns[k]
        os.makedirs(path, exist_ok=True)  # Create directory if it does not exist
        generate_prompts(df_test, window_len, pre_len, os.path.join(path, 'test_x_prompt.txt'), k)

    for k in range(len(df_test.columns)):
        columns = df_test.columns
        path = output_folder + columns[k]
        os.makedirs(path, exist_ok=True)  # Create directory if it does not exist
        generate_labels(df_test, window_len, pre_len, os.path.join(path, 'test_y_prompt.txt'), k)

    print("done!")
