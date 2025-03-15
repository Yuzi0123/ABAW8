CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=1 \
MAX_PIXELS=1003520 \
swift infer \
    --model output/Qwen2.5-72B-Instruct/v18-20250314-172200/checkpoint-1600-merged \
    --val_dataset data/RAF-DB/compound_original_test.jsonl \