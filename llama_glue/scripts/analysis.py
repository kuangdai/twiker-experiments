import os

# 结果路径
result_root = "../results"
tasks = ["wnli", "rte", "mrpc", "stsb", "cola", "sst2", "qnli", "qqp", "mnli"]
configs = ["twiker_off", "twiker_small", "twiker_large"]
config_names = {"twiker_off": "Off", "twiker_small": "Small", "twiker_large": "Large"}

# 缩写规则
metric_rename = {
    "loss": "Loss",
    "matthews_correlation": "MC",
    "accuracy": "Acc",
    "pearson": "PC",
    "spearmanr": "SC",
    "f1": "F1",
}


# 保证Loss排第一
def metric_sort_key(name):
    if name.lower() == "loss":
        return "0"
    return name.lower()


# 读取结果
task_metrics = {}

for task in tasks:
    task_metrics[task] = {}
    for config in configs:
        result_file = os.path.join(result_root, f"{task}_{config}", "metrics.txt")
        if not os.path.exists(result_file):
            continue
        with open(result_file) as f:
            for line in f:
                if line.startswith("eval_"):
                    key, val = line.strip().split(":")
                    key = key.replace("eval_", "").strip()
                    val = float(val.strip())
                    task_metrics[task].setdefault(key, {})[config] = val

# 生成 LaTeX 表格
lines = []
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\begin{tabular}{llccc}")
lines.append(r"\toprule")
lines.append(r"Task & Metric & \textbf{Off} & \textbf{Small} & \textbf{Large} \\")
lines.append(r"\midrule")

# 逐任务输出
for task in tasks:
    all_metrics = sorted(task_metrics.get(task, {}).keys(), key=metric_sort_key)
    if not all_metrics:
        continue

    for idx, metric in enumerate(all_metrics):
        if metric not in metric_rename:
            continue

        short_name = metric_rename[metric]
        row = []

        # 第一行显示Task名
        row.append(task if idx == 0 else "")
        row.append(short_name)

        # 收集各配置指标
        values = []
        for config in configs:
            val = task_metrics[task].get(metric, {}).get(config, "NA")
            values.append(val)

        # 确定理想值
        float_vals = [v for v in values if isinstance(v, float)]
        if float_vals:
            if short_name == "Loss":
                best = min(float_vals)
            else:
                best = max(float_vals)
        else:
            best = None

        # 填充每列
        for v in values:
            if isinstance(v, float):
                formatted = f"{v:.3f}"
                if v == best:
                    formatted = r"\textbf{" + formatted + r"}"
                row.append(formatted)
            else:
                row.append("NA")

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\midrule")

lines.append(r"\end{tabular}")
lines.append(
    r"\caption{GLUE Benchmark with LLaMA-3-8B and \texttt{TWiKer}. "
    r"\textbf{Acc} = Accuracy, \textbf{MC} = Matthews Correlation, "
    r"\textbf{PC} = Pearson Correlation, \textbf{SC} = Spearman Correlation.}"
)
lines.append(r"\end{table}")

# 打印输出
print("\n".join(lines))
