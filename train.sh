#!/bin/bash
#SBATCH --job-name=deep_speed
#SBATCH --output=Results/deep_speed_output.txt
#SBATCH --error=Results/deep_speed_error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=y.jin@ufl.edu
#SBATCH --account=vemuri
#SBATCH --qos=vemuri
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4  # DeepSpeed通常需要多个进程，取决于你的GPU数量
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4  # 假设你使用4个GPU
#SBATCH --time=96:00:00

# 准备输出文件
FILE="Results/deep_speed_output.txt"
echo "Date      = $(date)" > $FILE
echo "host      = $(hostname -s)" >> $FILE
echo "Directory = $(pwd)" >> $FILE
echo >> $FILE

# 记录开始时间
T1=$(date +%s)

# 激活你的conda环境
ml conda
conda activate /blue/amolstad/y.jin/anaconda3/envs/MyNlp

# 运行DeepSpeed命令
# 注意，这里使用srun来启动DeepSpeed，以确保它与SLURM集成
srun deepspeed --include localhost:0,1,2,3 train_sft.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path /blue/amolstad/y.jin/NLPCV/zero_nlp/internlm-sft/internlm-7b \
    --use_lora true \
    --cache_dir /blue/amolstad/y.jin/NLPCV/zero_nlp/internlm-sft/internlm-7b \
    --use_deepspeed true \
    --data_path /blue/amolstad/y.jin/NLPCV/zero_nlp/internlm-sft/sft_data \
    --bf16 true \
    --fp16 false \
    --output_dir /blue/amolstad/y.jin/NLPCV/zero_nlp/internlm-sft/Results \
    --num_train_epochs 2 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 4e-4 \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048 >> $FILE

# 记录结束时间并计算经过时间
echo >> $FILE
T2=$(date +%s)
ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED" >> $FILE