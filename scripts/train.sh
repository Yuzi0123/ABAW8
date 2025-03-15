CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=1 \
MAX_PIXELS=1003520 \
swift sft \
    --train_type lora \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --lora_rank 8 \
    --lora_alpha 32 \
    --num_train_epochs 20 \
    --save_steps 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --dataset data/RAF-DB/compound_original_train.jsonl \
    --val_dataset data/RAF-DB/compound_original_test.jsonl \