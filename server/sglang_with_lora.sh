#!/bin/bash
#8 different LoRAs possible, we use the same adapter for sake of simplicity
python3 -m sglang.launch_server \
  --model-path /nfs/scratch-aa/qwen/Qwen3-4B \
  --tp-size 2 \
  --enable-lora \
  --max-loras-per-batch 8 \
  --lora-backend csgmv \
  --lora-paths \
    lora_finance=ahmadmuf/product-price-predictor-2025-05-17_07.41.06 \
    lora_medical=ahmadmuf/product-price-predictor-2025-05-17_07.41.06 \
    lora_legal=ahmadmuf/product-price-predictor-2025-05-17_07.41.06 \
    lora_coding=ahmadmuf/product-price-predictor-2025-05-17_07.41.06 \
    lora_creative=ahmadmuf/product-price-predictor-2025-05-17_07.41.06 \
    lora_technical=ahmadmuf/product-price-predictor-2025-05-17_07.41.06 \
    lora_customer=ahmadmuf/product-price-predictor-2025-05-17_07.41.06 \
    lora_research=ahmadmuf/product-price-predictor-2025-05-17_07.41.06 \
  --port 8000