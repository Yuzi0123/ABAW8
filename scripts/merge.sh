CUDA_VISIBLE_DEVICES=1 \
swift export \
    --ckpt_dir output/Qwen2.5-72B-Instruct/v18-20250314-172200/checkpoint-1600 \
    --merge_lora true \