#!/bin/bash

# 所有GLUE子任务
tasks=("sst2" "cola" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte" "wnli")

# TWiKer配置路径列表
configs=("../configs/twiker_large.json")

# 双层循环
for task in "${tasks[@]}"; do
  for config in "${configs[@]}"; do
    echo "🟢 开始任务: $task 配置: $config"
    python train_eval_glue.py --task $task --epochs 1 --twiker-config $config --max-step 1
    echo "✅ 任务完成: $task 配置: $config"
    echo ""
    echo ""
    echo ""
    echo ""
  done
done
