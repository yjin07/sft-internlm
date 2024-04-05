# --nnodes 1 --nproc_per_node 4 --master_port 25641

deepspeed --include localhost:0,1,2,3 train_sft.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path /blue/amolstad/y.jin/NLPCV/zero_nlp/internlm-sft/internlm-7b \
    --use_lora true \
    --cache_dir /blue/amolstad/y.jin/NLPCV/zero_nlp/internlm-sft/internlm-7b \
    --use_deepspeed true \
    --data_path /blue/amolstad/y.jin/NLPCV/zero_nlp/internlm-sft/sft_data \
    --bf16 true \
    --fp16 false \
    --output_dir /blue/amolstad/y.jin/NLPCV/zero_nlp/internlm-sft/Results \
    --num_train_epochs 5 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 4e-4 \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048

# --save_steps 1000 \
