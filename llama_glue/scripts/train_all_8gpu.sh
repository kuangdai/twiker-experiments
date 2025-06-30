#!/bin/bash

# 卡0：小任务组合（每个任务3个config），最小的wnli优先
CUDA_VISIBLE_DEVICES=0 bash -c "
python train_eval_glue.py --task wnli --epochs 3 --twiker-config ../configs/twiker_large.json &&
python train_eval_glue.py --task wnli --epochs 3 --twiker-config ../configs/twiker_small.json &&
python train_eval_glue.py --task wnli --epochs 3 --twiker-config ../configs/twiker_off.json &&

python train_eval_glue.py --task rte --epochs 3 --twiker-config ../configs/twiker_large.json &&
python train_eval_glue.py --task rte --epochs 3 --twiker-config ../configs/twiker_small.json &&
python train_eval_glue.py --task rte --epochs 3 --twiker-config ../configs/twiker_off.json &&

python train_eval_glue.py --task mrpc --epochs 3 --twiker-config ../configs/twiker_large.json &&
python train_eval_glue.py --task mrpc --epochs 3 --twiker-config ../configs/twiker_small.json &&
python train_eval_glue.py --task mrpc --epochs 3 --twiker-config ../configs/twiker_off.json &&

python train_eval_glue.py --task stsb --epochs 3 --twiker-config ../configs/twiker_large.json &&
python train_eval_glue.py --task stsb --epochs 3 --twiker-config ../configs/twiker_small.json &&
python train_eval_glue.py --task stsb --epochs 3 --twiker-config ../configs/twiker_off.json &&

python train_eval_glue.py --task cola --epochs 3 --twiker-config ../configs/twiker_large.json &&
python train_eval_glue.py --task cola --epochs 3 --twiker-config ../configs/twiker_small.json &&
python train_eval_glue.py --task cola --epochs 3 --twiker-config ../configs/twiker_off.json &&

python train_eval_glue.py --task sst2 --epochs 3 --twiker-config ../configs/twiker_large.json &&
python train_eval_glue.py --task sst2 --epochs 3 --twiker-config ../configs/twiker_small.json &&
python train_eval_glue.py --task sst2 --epochs 3 --twiker-config ../configs/twiker_off.json
" > log_card0.txt 2>&1 &


# 卡1：qnli 三个配置
CUDA_VISIBLE_DEVICES=1 bash -c "
python train_eval_glue.py --task qnli --epochs 3 --twiker-config ../configs/twiker_large.json &&
python train_eval_glue.py --task qnli --epochs 3 --twiker-config ../configs/twiker_small.json &&
python train_eval_glue.py --task qnli --epochs 3 --twiker-config ../configs/twiker_off.json
" > log_card1.txt 2>&1 &


# 卡2-4：mnli 三个配置并行
CUDA_VISIBLE_DEVICES=2 python train_eval_glue.py --task mnli --epochs 3 --twiker-config ../configs/twiker_large.json > log_card2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train_eval_glue.py --task mnli --epochs 3 --twiker-config ../configs/twiker_small.json > log_card3.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 python train_eval_glue.py --task mnli --epochs 3 --twiker-config ../configs/twiker_off.json > log_card4.txt 2>&1 &

# 卡5-7：qqp 三个配置并行
CUDA_VISIBLE_DEVICES=5 python train_eval_glue.py --task qqp --epochs 3 --twiker-config ../configs/twiker_large.json > log_card5.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 python train_eval_glue.py --task qqp --epochs 3 --twiker-config ../configs/twiker_small.json > log_card6.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 python train_eval_glue.py --task qqp --epochs 3 --twiker-config ../configs/twiker_off.json > log_card7.txt 2>&1 &
