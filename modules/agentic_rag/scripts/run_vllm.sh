CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B \
    --served-model-name qwen3-8b \
    --port 7300 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    --enforce-eager