#!/bin/bash
# Usage: ./run_benchmark.sh
MODEL="${MODEL:-LiquidAI/LFM2-24B-A2B-Preview}"
API_BASE="${API_BASE:-http://localhost:30000/v1}"
IN_TOKENS="${IN_TOKENS:-8192}"
OUT_TOKENS="${OUT_TOKENS:-1024}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16,32,64,128}"
NUM_RUNS="${NUM_RUNS:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"

MODEL_NAME=$(basename "$MODEL")
RESULTS_FILE="${RESULTS_FILE:-${MODEL_NAME}-long.json}"

python3 completion_benchmark.py \
  --model "$MODEL" \
  --scenario "N(${IN_TOKENS},0)/(${OUT_TOKENS},0)" \
  --dataset-config examples/dataset_configs/aime_simple.json \
  --api-base "$API_BASE" \
  --batch-sizes "$BATCH_SIZES" \
  --num-runs "$NUM_RUNS" \
  --warmup-runs "$WARMUP_RUNS" \
  --results-file "$RESULTS_FILE"
