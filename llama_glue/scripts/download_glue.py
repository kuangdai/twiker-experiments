from datasets import load_dataset
import os

tasks = [
    "cola", "sst2", "mrpc", "stsb", "qqp",
    "mnli", "qnli", "rte", "wnli", "ax"
]

root_dir = "../datasets"  # 适配你llama_glue结构
os.makedirs(root_dir, exist_ok=True)

for task in tasks:
    print(f"Downloading {task}...")
    dataset = load_dataset("nyu-mll/glue", task)

    save_path = os.path.join(root_dir, task)
    os.makedirs(save_path, exist_ok=True)

    for split in dataset.keys():
        dataset[split].to_json(os.path.join(save_path, f"{split}.jsonl"), orient="records", lines=True)

print("All GLUE tasks downloaded.")
