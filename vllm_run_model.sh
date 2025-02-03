#!/bin/bash
# run_model.sh
# This script launches the vLLM server using a hardcoded model tag.
# The model tag "distill-llama-8b" is mapped internally to its full model path.
# The --served-model-name flag is used to expose the model using its tag.
#
# The speculative model usage is controlled by a hardcoded boolean variable.
# When USE_SPECULATIVE_MODEL is true, the script appends the speculative model options:
#   --speculative-model <speculative_model_path>
#   --num-speculative-tokens 5

# Hardcoded model tag and its internal mapping.
MODEL_TAG="distill-llama-8b"
MODEL_PATH="/nfs/checkpoint-tuning/deepseek/DeepSeek-R1-Distill-Llama-8B"

# Hardcoded speculative model path and settings.
USE_SPECULATIVE_MODEL=false  # Change to true to enable speculative model.
SPECULATIVE_MODEL_PATH="/nfs/checkpoint-tuning/deepseek/DeepSeek-R1-Distill-Llama-8B"
NUM_SPECULATIVE_TOKENS=5

# Other parameters.
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=1
DEVICE="cuda"


echo "Starting vLLM server with model tag: ${MODEL_TAG}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "Pipeline Parallel Size: ${PIPELINE_PARALLEL_SIZE}"
echo "Device: ${DEVICE}"

# Build the base command.
# The full model path (mapped from the hardcoded tag) is passed as a positional argument,
# and the --served-model-name flag is used to register the model with the tag.
CMD=(vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_TAG}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
    --device "${DEVICE}")

# Append speculative model options if enabled.
if [[ "${USE_SPECULATIVE_MODEL}" == true ]]; then
    echo "Speculative model enabled. Using speculative model: ${SPECULATIVE_MODEL_PATH} with ${NUM_SPECULATIVE_TOKENS} speculative tokens."
    CMD+=(--speculative-model "${SPECULATIVE_MODEL_PATH}" --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}")
fi

# Print and execute the command.
echo "Running command: ${CMD[*]}"
exec "${CMD[@]}"
