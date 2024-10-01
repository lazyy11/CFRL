# CFRL model for weather factor forecasting

## Introduction
This repository contains the code for **A Prompt-based Learning Paradigm for Time Series Forecasting of Weather Factors**.

## Folder Structure

- **data/**: Contains sample data used for training, validation, and testing, including the prompt-based data formats. Due to the large size of the original files, only the first 100 lines from each file have been included in this sample dataset for demonstration purposes.
  - `predicted_results/`: Contains predicted results for the models.
  - `train_x_prompt.txt`, `train_y_prompt.txt`: Training data in prompt format.
  - `val_x_prompt.txt`, `val_y_prompt.txt`: Validation data in prompt format.
  - `test_x_prompt.txt`, `test_y_prompt.txt`: Testing data in prompt format.

- **src/**: Contains source code for the models and preprocessing scripts.
  - **comparison_models/**: See `comparison_models/README.md` for details.
  - **main_model/**: The main model code for the prompt-based learning approach.
    - **run/**: Scripts for running the models and evaluations.
      - `train.py`: Main training script.
      - `test.py`: Testing script.
      - `evaluate.py`: Evaluation script.
      - `metrics.py`: Contains the evaluation metrics.
      - `csv_to_txt.py`, `csv_to_txt_single_column.py`: Scripts to convert CSV data to text format.
      - `txt_to_json.py`: Script to convert text data to JSON format for further training.

  - **scripts/**: Contains shell scripts for running experiments.
     - `TH_1_step_train.sh`: Script for training the model with a prediction step length of 1.
     - `TH_1_step_test.sh`: Script for testing the model with a prediction step length of 1.

  - **T5-base/**: Folder containing files related to the T5-base model used in the project.
    - `README.md`: Specific details on how the T5 model is used in the project.

## How to Use

1. **Install Dependencies**:  
   Install all dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data for Fine-tuning**:  
   To prepare the text files required to fine-tune the model:
   - First, run the script to convert the CSV files to text:
     ```bash
     python run/csv_to_txt.py
     ```
   - Then, run the script to convert the text files to JSON format:
     ```bash
     python run/txt_to_json.py
     ```

3. **Training**:  
   Use the shell script `TH_1_step_train.sh` to start training the model. Example usage:
   ```bash
   bash ./scripts/TH_1_step_train.sh
   ```

4. **Testing Each Weather Factor Individually**:  
   To test each weather factor individually, prepare the test dataset:
   - Run the script to convert individual weather factors for testing:
     ```bash
     python run/csv_to_txt_single_column.py
     ```

5. **Testing the Model**:  
   After training, use the following script to test the model and get the predicted results on the test dataset:
   ```bash
   bash ./scripts/TH_1_step_test.sh
   ```

6. **Evaluation**:  
   After testing, use the `evaluate.py` script to generate performance metrics for the predictions:
   ```bash
   python run/evaluate.py
   ```
