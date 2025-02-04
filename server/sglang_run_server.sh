#!/bin/bash
# sglang_run_server.sh (Modified)
#
# This script launches the sglang server using configuration values that mirror the vLLM server.
# It uses a hardcoded model tag and model path, the same tensor parallelism and device settings,
# and similar speculative decoding options.
#
# Note: sglang does not support pipeline parallelism, so that parameter is omitted.

# ----------------------------------
# Hardcoded model tag and model path.
# ----------------------------------
MODEL_TAG="distill-llama-8b"
MODEL_PATH="/nfs/checkpoint-tuning/deepseek/DeepSeek-R1-Distill-Llama-8B"

# ----------------------------------
# Speculative model settings.
# ----------------------------------
USE_SPECULATIVE_MODEL=false  # Change to true to enable speculative decoding.
SPECULATIVE_MODEL_PATH="/nfs/checkpoint-tuning/deepseek/DeepSeek-R1-Distill-Llama-8B"
NUM_SPECULATIVE_TOKENS=5

# ----------------------------------
# Parallelism and device options.
# ----------------------------------
TENSOR_PARALLEL_SIZE=8
# Note: Pipeline parallelism is not supported in sglang.
DEVICE="cuda"

echo "Starting sglang server with model tag: ${MODEL_TAG}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "Device: ${DEVICE}"

# ----------------------------------
# Build the command array.
# ----------------------------------
# The following command uses:
#   --model-path: to point to the model directory,
#   --served-model-name: to register the model with a tag (mirroring vLLM usage),
#   --tp: to set the tensor parallelism,
#   --device: to choose the hardware.
CMD=(python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${MODEL_TAG}" \
    --tp "${TENSOR_PARALLEL_SIZE}" \
    --device "${DEVICE}")

# Append speculative decoding options if enabled.
if [[ "${USE_SPECULATIVE_MODEL}" == true ]]; then
    echo "Speculative decoding enabled:"
    echo "  Speculative Model Path: ${SPECULATIVE_MODEL_PATH}"
    echo "  Number of Speculative Tokens: ${NUM_SPECULATIVE_TOKENS}"
    CMD+=(--speculative-draft-model-path "${SPECULATIVE_MODEL_PATH}" \
          --speculative-num-steps "${NUM_SPECULATIVE_TOKENS}" \
          --speculative-num-draft-tokens "${NUM_SPECULATIVE_TOKENS}")
fi

# Print the command for verification.
echo "Running command: ${CMD[*]}"

# Execute the command.
exec "${CMD[@]}"
