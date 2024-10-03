import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re

def np_evaluate(gt_output, pred_output):
    r2 = r2_score(gt_output, pred_output)
    mae =MAE(gt_output, pred_output)
    rmse = RMSE(gt_output, pred_output)

    return rmse, mae, r2

def MAE(gt_output, pred_output):
    mask = gt_output != 0

    gt_filtered = gt_output[mask]
    pred_filtered = pred_output[mask]

    n = len(gt_filtered)
    if n == 0:
        return np.nan, np.nan

    mae = np.sum(np.abs(gt_filtered - pred_filtered)) / n
    return mae

def mape_smape(gt_output, pred_output):
    mask = gt_output != 0

    gt_filtered = gt_output[mask]
    pred_filtered = pred_output[mask]

    n = len(gt_filtered)
    if n == 0:
        return np.nan, np.nan

    mape = np.sum(np.abs((gt_filtered - pred_filtered) / gt_filtered)) / n * 100
    smape = np.sum(np.abs(gt_filtered - pred_filtered) / ((np.abs(gt_filtered) + np.abs(pred_filtered)) / 2)) / n * 100

    return mape, smape


def RMSE(gt_output, pred_output):
    mask = gt_output != 0
    if np.any(mask):
        mse = np.mean(np.square(gt_output[mask] - pred_output[mask]))
    else:
        mse = np.nan
    return np.sqrt(mse)


def metric_with_missing_rate(gt_text, predicted_text, dataset):
    output_data = []
    gt_data = []
    missing_count = 0

    number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    if len(gt_text) != len(predicted_text):
        raise ValueError(f"Length mismatch: gt_text has {len(gt_text)} lines, but predicted_text has {len(predicted_text)} lines.")

    for i in range(len(gt_text)):
        predicted_line = predicted_text[i]
        gt_line = gt_text[i]

        try:
            predicted_values = np.array([float(val) for val in number_pattern.findall(predicted_line)])
            gt_values = np.array([float(val) for val in number_pattern.findall(gt_line)])

            if len(predicted_values) != len(gt_values):
                raise ValueError(f"Mismatch in number of values at line {i}: predicted has {len(predicted_values)}, which is {predicted_values}, but gt has {len(gt_values)}, which is {gt_values}")

            output_data.extend(predicted_values)
            gt_data.extend(gt_values)
        except Exception as e:
            print(f"Error processing line {i}: {e}")
            missing_count += 1

    print(f"Length of output_data: {len(output_data)}, Length of gt_data: {len(gt_data)}")

    output = np.reshape(output_data, [len(output_data), 1])
    gt_output = np.reshape(gt_data, [len(gt_data), 1])
    print("output: ", output, "\n", "gt_output: ", gt_output, "\n")
    rmse, mae, r2 = np_evaluate(gt_output, output)
    mape, smape = mape_smape(gt_output, output)
    missing_rate = missing_count / len(gt_text)

    return rmse, mae, mape, smape, r2, missing_rate