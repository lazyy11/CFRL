import pandas as pd
import numpy as np
import os

def generate_prompts(df, window_len, pre_len, output_path):
    with open(output_path, 'w') as file:
        num_clients = len(df.columns)
        for k in range(num_clients):
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

def generate_labels(df, window_len, pre_len, output_path):
    with open(output_path, 'w') as file:
        num_clients = len(df.columns)
        for k in range(num_clients):
            for i in range(len(df) - window_len - pre_len):
                # next_day = df.index[i + window_len].strftime('%m/%d/%Y %I:%M:%S %p')
                labels = ', '.join(
                    f"{df.iloc[i + window_len + j, k]:.15g}" for j in range(pre_len)
                )
                file.write(f"The prediction is {labels}.\n")

input_path = '/home/eutaboo/Downloads/southeastAsia/TH/final_combined_data.csv'
df_daily = pd.read_csv(input_path, parse_dates=[0], index_col=0)

total_rows = len(df_daily)
train_size = int(total_rows * 0.7)
val_size = int(total_rows * 0.1)
test_size = total_rows - train_size - val_size

df_train = df_daily.iloc[:train_size]
df_val = df_daily.iloc[train_size:train_size + val_size]
# df_test = df_daily.iloc[train_size + val_size:]

window_len = 24
pre_len = 1

output_folder = '/home/eutaboo/PycharmProjects/PromptCast/LMP/Thailand/Dataset_train/Thai_5_steps'
os.makedirs(output_folder, exist_ok=True)  # Create directory if it does not exist

generate_prompts(df_train, window_len, pre_len, os.path.join(output_folder, 'train_x_prompt.txt'))
generate_labels(df_train, window_len, pre_len, os.path.join(output_folder, 'train_y_prompt.txt'))

generate_prompts(df_val, window_len, pre_len, os.path.join(output_folder, 'val_x_prompt.txt'))
generate_labels(df_val, window_len, pre_len, os.path.join(output_folder, 'val_y_prompt.txt'))

# generate_prompts(df_test, window_len, pre_len, os.path.join(output_folder, 'test_x_prompt.txt'))
# generate_labels(df_test, window_len, pre_len, os.path.join(output_folder, 'test_y_prompt.txt'))

print("done!")
