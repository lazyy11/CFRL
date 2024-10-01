import os
import jsonlines


def clean(x):
    x = x.replace(",", "")
    return x


def to_jsonl(src_file, dst_file):
    if not os.path.exists(dst_file):
        os.makedirs(dst_file)

    val_in_list = []
    val_out_list = []
    train_in_list = []
    train_out_list = []
    test_in_list = []
    test_out_list = []

    for f in os.listdir(src_file):
        if "val_x_prompt" in f:
            val_in_list = open(os.path.join(src_file, f)).readlines()
            val_in_list = [line.strip() for line in val_in_list]
        elif "val_y_prompt" in f:
            val_out_list = open(os.path.join(src_file, f)).readlines()
            val_out_list = [line.strip() for line in val_out_list]
        elif "train_x_prompt" in f:
            train_in_list = open(os.path.join(src_file, f)).readlines()
            train_in_list = [line.strip() for line in train_in_list]
        elif "train_y_prompt" in f:
            train_out_list = open(os.path.join(src_file, f)).readlines()
            train_out_list = [line.strip() for line in train_out_list]
        elif "test_x_prompt" in f:
            test_in_list = open(os.path.join(src_file, f)).readlines()
            test_in_list = [line.strip() for line in test_in_list]
        elif "test_y_prompt" in f:
            test_out_list = open(os.path.join(src_file, f)).readlines()
            test_out_list = [line.strip() for line in test_out_list]

    val_items = [{"text": val_in_list[i], "summary": val_out_list[i]} for i in range(len(val_in_list))]
    train_items = [{"text": train_in_list[i], "summary": train_out_list[i]} for i in range(len(train_in_list))]
    test_items = [{"text": test_in_list[i], "summary": test_out_list[i]} for i in range(len(test_in_list))]

    with jsonlines.open(os.path.join(dst_file, "val.json"), 'w') as writer:
        writer.write_all(val_items)

    with jsonlines.open(os.path.join(dst_file, "train.json"), 'w') as writer:
        writer.write_all(train_items)

    with jsonlines.open(os.path.join(dst_file, "test.json"), 'w') as writer:
        writer.write_all(test_items)


if __name__ == "__main__":
    split_data_folder = '/home/eutaboo/PycharmProjects/PromptCast/LMP/Thailand/Dataset_train/Thai_3_steps'
    output_jsonl_folder = '/home/eutaboo/PycharmProjects/PromptCast/LMP/Thailand/Dataset_train/Thai_3_steps/jsonl_data'
    to_jsonl(split_data_folder, output_jsonl_folder)
    print('done!')



