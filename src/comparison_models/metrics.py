import numpy as np
from sklearn.metrics import r2_score

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    mask = true != 0

    gt_filtered = true[mask]
    pred_filtered = pred[mask]

    n = len(gt_filtered)
    if n == 0:
        return np.nan, np.nan

    mae = np.sum(np.abs(gt_filtered - pred_filtered)) / n
    return mae

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    mask = true != 0
    if np.any(mask):
        mse = np.mean(np.square(true[mask] - pred[mask]))
    else:
        mse = np.nan
    return np.sqrt(mse)

def MAPE(pred, true):
    mask = (true != 0)

    gt_filtered = true[mask]
    pred_filtered = pred[mask]

    n = len(gt_filtered)
    if n == 0:
        return np.nan, np.nan

    mape = np.sum(np.abs((gt_filtered - pred_filtered) / gt_filtered)) / n * 100
    return mape

def SMAPE(pred, true):
    mask = true != 0

    gt_filtered = true[mask]
    pred_filtered = pred[mask]

    n = len(gt_filtered)
    if n == 0:
        return np.nan, np.nan

    smape = np.sum(np.abs(gt_filtered - pred_filtered) / ((np.abs(gt_filtered) + np.abs(pred_filtered)) / 2)) / n * 100

    return smape
def MSPE(pred, true):
    mask = (true != 0)
    if np.any(mask):
        mspe = np.mean(np.square((true[mask] - pred[mask]) / true[mask])) * 100
    else:
        mspe = np.nan
    return mspe

def R2(pred, true):
    r2 = r2_score(true, pred)
    return r2

def metric(pred, true):
    # Flatten the arrays to 1-D
    true_values_flat = true.flatten()
    pred_values_flat = pred.flatten()

    mae = MAE(pred_values_flat, true_values_flat)
    rmse = RMSE(pred_values_flat, true_values_flat)
    mape = MAPE(pred_values_flat, true_values_flat)
    smape = SMAPE(pred_values_flat, true_values_flat)
    r2 = R2(pred_values_flat, true_values_flat)

    return mae, rmse, smape, r2
