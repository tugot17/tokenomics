#!/bin/bash

MODEL="LiquidAI/LFM2.5-1.2B-Instruct"
TP_SIZE=1

echo "Starting SGLang server with model: ${MODEL}"
echo "Tensor Parallel Size: ${TP_SIZE}"

python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --tp-size "${TP_SIZE}" \
    --port 8000
