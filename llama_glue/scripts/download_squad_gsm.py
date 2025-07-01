from datasets import load_dataset
import os

root_dir = "../datasets"  # 保持一致
os.makedirs(root_dir, exist_ok=True)

# SQuAD v1.1
print("Downloading SQuAD v1.1...")
squad = load_dataset("squad")

squad_path = os.path.join(root_dir, "squad")
os.makedirs(squad_path, exist_ok=True)

for split in squad.keys():
    squad[split].to_json(os.path.join(squad_path, f"{split}.jsonl"), orient="records", lines=True)

# GSM8K
print("Downloading GSM8K...")
gsm8k = load_dataset("gsm8k", "main")

gsm_path = os.path.join(root_dir, "gsm8k")
os.makedirs(gsm_path, exist_ok=True)

for split in gsm8k.keys():
    gsm8k[split].to_json(os.path.join(gsm_path, f"{split}.jsonl"), orient="records", lines=True)

print("SQuAD and GSM8K downloaded.")
