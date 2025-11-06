#!/bin/bash
# SGLang server with LoRA support for benchmarking
# Uses Qwen/Qwen3-4B with 8 demo LoRA adapters

MODEL="Qwen/Qwen3-4B"
TP_SIZE=1
LORA_ADAPTER="ahmadmuf/product-price-predictor-2025-05-17_07.41.06"

echo "Starting SGLang server with LoRA support"
echo "Model: ${MODEL}"
echo "LoRA Adapter: ${LORA_ADAPTER} (same adapter for all 8 slots)"

python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp-size "${TP_SIZE}" \
  --enable-lora \
  --max-loras-per-batch 8 \
  --lora-backend csgmv \
  --lora-paths \
    lora_finance="${LORA_ADAPTER}" \
    lora_medical="${LORA_ADAPTER}" \
    lora_legal="${LORA_ADAPTER}" \
    lora_coding="${LORA_ADAPTER}" \
    lora_creative="${LORA_ADAPTER}" \
    lora_technical="${LORA_ADAPTER}" \
    lora_customer="${LORA_ADAPTER}" \
    lora_research="${LORA_ADAPTER}" \
  --port 8000
