import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments
from datasets import load_dataset
from evaluate import load as load_metric
import numpy as np
import sys
from peft import LoraConfig, get_peft_model

from transformers import Trainer
import torch.nn as nn
from transformers import TrainerCallback


class EvalLoggerCallback(TrainerCallback):
    def __init__(self, save_dir, task, config_name):
        self.save_dir = save_dir
        self.task = task
        self.config_name = config_name

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(
            self.save_dir,
            f"{self.task}_{self.config_name}_epoch{int(state.epoch)}.txt"
        )

        with open(save_path, "w") as f:
            f.write(f"🌟 Evaluation results for {self.task} | Config: {self.config_name} | Epoch: {state.epoch}\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        print(f"✅ Eval results saved to {save_path}")


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if args.task == "stsb":
            loss = nn.MSELoss()(logits.view(-1), labels.view(-1))
        else:
            # 统一用CrossEntropyLoss，适用于二分类和多分类
            loss = nn.CrossEntropyLoss()(logits, labels)

        return (loss, outputs) if return_outputs else loss


# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, type=str)
parser.add_argument("--epochs", required=True, type=int)
parser.add_argument("--twiker-config", required=True, type=str)
parser.add_argument("--max-steps", default=-1, type=int)
args = parser.parse_args()

print("✅ 参数解析完成")
sys.stdout.flush()

# 内置路径
DATA_ROOT = "../datasets"
MODEL_ROOT = "../models/llama-8b"
OUTPUT_ROOT = f"../outputs/{args.task}/twiker_results"

print("✅ 目录设置完成")
sys.stdout.flush()

# 读取TWiKer配置
print(f"✅ 正在读取配置文件: {args.twiker_config}")
with open(args.twiker_config, "r") as f:
    twiker_cfg = json.load(f)

# 加载配置与模型
print("✅ 加载Tokenizer与配置")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT)
tokenizer.pad_token = tokenizer.eos_token
config = AutoConfig.from_pretrained(MODEL_ROOT)
for k, v in twiker_cfg.items():
    setattr(config, k, v)
config.pad_token_id = tokenizer.pad_token_id
if args.task == "stsb":
    config.num_labels = 1
elif args.task == "mnli":
    config.num_labels = 3
else:
    config.num_labels = 2

print("✅ 加载模型")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ROOT, config=config)

# 动态确定保留梯度模块
additional_modules = []
if getattr(config, "twiker_activated", False):
    additional_modules.append("twiker_model.embedding")

# 配置LoRA
print(f"✅ 保留梯度模块 (不加LoRA): {additional_modules}")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    modules_to_save=additional_modules,
    bias="none",
    task_type="SEQ_CLS"
)

# 注入
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✅ 可优化参数模块 (统计总元素个数):")
for name, module in model.named_modules():
    param_list = list(module.parameters(recurse=False))
    if not param_list:
        continue

    total_elements = sum(p.numel() for p in param_list)
    trainable_elements = sum(p.numel() for p in param_list if p.requires_grad)

    if trainable_elements > 0 and "modules_to_save" in name:
        print(f"{name} | 总参数元素: {total_elements} | 可训练元素: {trainable_elements}")

# 数据加载
print("✅ 加载数据集")
dataset = load_dataset("json", data_files={
    "train": os.path.join(DATA_ROOT, args.task, "train.jsonl"),
    "validation": os.path.join(DATA_ROOT, args.task, "validation.jsonl")
})

# 预处理
print("✅ 开始数据预处理")


def preprocess(examples):
    if args.task in ["sst2", "cola"]:
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

    elif args.task in ["rte", "stsb", "wnli", "mrpc"]:
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length",
                         max_length=128)

    elif args.task == "mnli":
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length",
                         max_length=128)

    elif args.task == "qnli":
        return tokenizer(examples["question"], examples["sentence"], truncation=True, padding="max_length",
                         max_length=128)

    elif args.task == "qqp":
        return tokenizer(examples["question1"], examples["question2"], truncation=True, padding="max_length",
                         max_length=128)

    else:
        raise NotImplementedError(f"Task {args.task} not supported")


dataset = dataset.map(preprocess, batched=True)
print("✅ 数据预处理完成")
sys.stdout.flush()

# 评估指标
print("✅ 加载评估指标")
glue_metric = load_metric("glue_metrics/glue/glue.py", args.task)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if args.task == "stsb":
        preds = np.squeeze(preds)
    else:
        preds = np.argmax(preds, axis=1)
    results = glue_metric.compute(predictions=preds, references=labels)
    return results


# 训练参数
print("✅ 配置训练参数")
training_args = TrainingArguments(
    output_dir=OUTPUT_ROOT,
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=args.epochs,
    save_strategy="no",
    report_to=[],
    fp16=True,
    gradient_accumulation_steps=4,
    max_steps=args.max_steps,
    disable_tqdm=True,
)

# Trainer 初始化
# 获取配置名，去掉路径和后缀
config_basename = os.path.splitext(os.path.basename(args.twiker_config))[0]

# 回调实例
eval_logger = EvalLoggerCallback(
    save_dir="../results/eval_epochs",
    task=args.task,
    config_name=config_basename
)

# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[eval_logger],
)

# 训练
print("✅ 正式开始训练")
sys.stdout.flush()
trainer.train()

# 评估
print("✅ 训练完成，开始评估")
final_metrics = trainer.evaluate()

# 输出与保存结果
print("✅ Final Evaluation Results:")
result_dir = f"../results/{args.task}_{os.path.splitext(os.path.basename(args.twiker_config))[0]}"
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, "metrics.txt")

with open(result_path, "w") as f:
    for k, v in final_metrics.items():
        if isinstance(v, float):
            line = f"{k}: {v:.4f}"
            print(line)
            f.write(line + "\n")

print(f"✅ 结果已保存到: {result_path}")
