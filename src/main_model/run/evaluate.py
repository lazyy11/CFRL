import os
import csv
from metrics import metric_with_missing_rate

def read_values(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    return lines


if __name__ == "__main__":
    truths_file_path = '/LMP/Thailand/Dataset_test/Thai_S/'
    predictions_file_path = '/home/eutaboo/PycharmProjects/PromptCast/LMP/Thailand/Thailand_Predictions_S'

    truths_dirs = set(os.listdir(truths_file_path))
    predictions_dirs = set(os.listdir(predictions_file_path))

    common_dirs = truths_dirs.intersection(predictions_dirs)

    results_data = []

    for dir_name in common_dirs:
        truth_dir = os.path.join(truths_file_path, dir_name, 'test_y_prompt.txt')
        prediction_dir = os.path.join(predictions_file_path, dir_name, 'predicted.txt')
        truths_from_file = read_values(truth_dir)
        predictions = read_values(prediction_dir)

        if len(predictions) != len(truths_from_file):
            raise ValueError(f"The number of predictions and truths must be the same for directory '{dir_name}'.")

        rmse, mae, smape, r2, missing_rate = metric_with_missing_rate(predictions, truths_from_file, dir_name)
        parts = dir_name.split('_')
        if len(parts) > 2:
            dir_name = parts[0] + '_' + parts[2]
        else:
            dir_name = parts[0]
        print(f"Results for {dir_name}:")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        # print(f"MAPE: {mape}%")
        print(f"missing_rate: {missing_rate}%\n")

        results_dir = f'/home/eutaboo/PycharmProjects/PromptCast/LMP/Thailand/run/results/TH/results_of_{dir_name}/'
        os.makedirs(results_dir, exist_ok=True)

        results_file_path = os.path.join(results_dir, 'evaluation.txt')
        with open(results_file_path, 'w') as file:
            file.write(f"Results for {dir_name}:\n")
            file.write(f"MAE: {mae}\n")
            file.write(f"RMSE: {rmse}\n")
            # file.write(f"MAPE: {mape}%\n")
            file.write(f"SMAPE: {smape}%\n")
            file.write(f"missing_rate: {missing_rate}%\n")

        results_data.append({
            'Variable': dir_name,
            'MAE': mae,
            'RMSE': rmse,
            # 'MAPE': mape,
            'SMAPE': smape,
            'R^2': r2,
            'Missing Rate (%)': missing_rate
        })

    results_data = sorted(results_data, key=lambda x: x['Variable'])

    csv_file_path = '/home/eutaboo/PycharmProjects/PromptCast/LMP/Thailand/run/results/TH/summary_results.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Variable', 'MAE', 'RMSE', 'MAPE', 'SMAPE', 'R^2', 'Missing Rate (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results_data)

    print(f"Summary results written to {csv_file_path}")