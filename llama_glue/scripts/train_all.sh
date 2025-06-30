#!/bin/bash

# 按任务消耗时间从短到长排序
tasks=("wnli" "rte" "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli")

# TWiKer配置路径列表，可添加多个
configs=(
  "../configs/twiker_large.json"
  "../configs/twiker_small.json"
  "../configs/twiker_off.json"
)

# 双层循环
for task in "${tasks[@]}"; do
  for config in "${configs[@]}"; do
    echo "🟢 开始正式任务: $task 配置: $config"
    python train_eval_glue.py --task $task --epochs 3 --twiker-config $config
    echo "✅ 任务完成: $task 配置: $config"
    echo ""
    echo ""
    echo ""
    echo ""
  done
done
