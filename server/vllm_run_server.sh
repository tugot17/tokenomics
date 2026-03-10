#!/bin/bash

MODEL="LiquidAI/LFM2.5-1.2B-Instruct"
TP_SIZE=1

echo "Starting vLLM server with model: ${MODEL}"
echo "Tensor Parallel Size: ${TP_SIZE}"

vllm serve "${MODEL}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --port 8000
