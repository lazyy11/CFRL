import argparse
import os
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_file', type=str, default='SG')
    parser.add_argument('-m', '--model_path', type=str, default='results')
    parser.add_argument('--model_name', type=str, default='bart')
    parser.add_argument('-s', '--save_path', type=str, default='results')
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-d', '--dataset_name', type=str, default='SG')
    return parser.parse_args()

def compute_max_lengths(data_dir, tokenizer):
    max_input_len = 0
    max_output_len = 0
    for filename in os.listdir(data_dir):
        if filename.endswith("x_prompt.txt"):
            with open(os.path.join(data_dir, filename), 'r') as f:
                inputs = f.readlines()
            for line in inputs:
                input_ids = tokenizer.encode(line.strip(), add_special_tokens=True, return_tensors='pt')
                max_input_len = max(max_input_len, input_ids.size(1))
        elif filename.endswith("y_prompt.txt"):
            with open(os.path.join(data_dir, filename), 'r') as f:
                outputs = f.readlines()
            for line in outputs:
                output_ids = tokenizer.encode(line.strip(), add_special_tokens=True, return_tensors='pt')
                max_output_len = max(max_output_len, output_ids.size(1))
    print(f"Max input length: {max_input_len}, Max output length: {max_output_len}")
    return max_input_len, max_output_len

def tokenize_texts(text_list, tokenizer, max_length):
    all_token_ids = []
    for text in text_list:
        encoded = tokenizer.encode_plus(
            text.strip(),
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        all_token_ids.append(encoded['input_ids'])
    return torch.cat(all_token_ids, dim=0)

if __name__ == "__main__":
    arguments = parse_arguments()
    os.makedirs(arguments.save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = arguments.model_name.lower()
    if model_name == 't5':
        model = T5ForConditionalGeneration.from_pretrained(arguments.model_path)
        tokenizer = AutoTokenizer.from_pretrained(arguments.model_path)
    else:
        raise ValueError(f"Unsupported model name: {arguments.model_name}")
    model.to(device)

    test_input_path = os.path.join(arguments.test_file, "test_x_prompt.txt")
    test_output_path = os.path.join(arguments.test_file, "test_y_prompt.txt")
    with open(test_input_path, 'r') as f_in, open(test_output_path, 'r') as f_out:
        input_lines = f_in.readlines()
        gt_lines = f_out.readlines()

    max_len_input, max_len_output = compute_max_lengths(arguments.test_file, tokenizer)

    predictions = []
    total_batches = (len(input_lines) + arguments.batch_size - 1) // arguments.batch_size
    for batch_idx in tqdm(range(total_batches)):
        batch_start = batch_idx * arguments.batch_size
        batch_end = min((batch_idx + 1) * arguments.batch_size, len(input_lines))
        batch_inputs = input_lines[batch_start:batch_end]
        input_ids = tokenize_texts(batch_inputs, tokenizer, max_len_input).to(device)
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=max_len_output,
            num_beams=4,
            early_stopping=True
        )
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(decoded_preds)

    output_prediction_file = os.path.join(arguments.save_path, "predicted.txt")
    with open(output_prediction_file, 'w') as f_pred:
        for pred in predictions:
            f_pred.write(pred + "\n")

