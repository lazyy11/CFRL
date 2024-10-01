import numpy as np
from sklearn.metrics import r2_score

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

    for i in range(len(gt_text)):
        predicted_line = predicted_text[i]
        gt_line = gt_text[i]
        try:
            out = float(predicted_line.split(" ")[3].rstrip('.\n'))
            gt_out = float(gt_line.split(" ")[3].rstrip('.\n'))
            output_data.append(out)
            gt_data.append(gt_out)
        except Exception:
            missing_count += 1

    output = np.reshape(output_data, [len(output_data), 1])
    gt_output = np.reshape(gt_data, [len(gt_data), 1])

    rmse, mae, r2 = np_evaluate(gt_output, output)
    mape, smape = mape_smape(gt_output, output)
    missing_rate = missing_count / len(gt_text)

    return rmse, mae, smape, r2, missing_rate