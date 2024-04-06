#!/bin/bash
#SBATCH --job-name=deep_speed
#SBATCH --output=Results/output.txt
#SBATCH --error=Results/error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=y.jin@ufl.edu
#SBATCH --account=vemuri
#SBATCH --qos=vemuri
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=96:00:00

FILE="Results/output.txt"
echo "Date      = $(date)" > $FILE
echo "host      = $(hostname -s)" >> $FILE
echo "Directory = $(pwd)" >> $FILE
echo >> $FILE

T1=$(date +%s)

ml conda
conda activate /blue/amolstad/y.jin/anaconda3/envs/MyNlp

srun deepspeed --include localhost:0,1,2,3 train_sft.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path /blue/amolstad/y.jin/sft-internlm/internlm-7b \
    --use_lora true \
    --cache_dir /blue/amolstad/y.jin/sft-internlm/internlm-7b \
    --use_deepspeed true \
    --data_path /blue/amolstad/y.jin/sft-internlm/sft_data \
    --bf16 true \
    --fp16 false \
    --output_dir /blue/amolstad/y.jin/sft-internlm/Results \
    --num_train_epochs 2 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 4e-6 \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048 >> $FILE

echo >> $FILE
T2=$(date +%s)
ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED" >> $FILE