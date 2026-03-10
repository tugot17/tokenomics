#!/bin/bash
# Usage: ./run_benchmark.sh
MODEL="${MODEL:-LiquidAI/LFM2-24B-A2B}"
API_BASE="${API_BASE:-http://tus1-p13-g57:30000/v1}"
IN_TOKENS="${IN_TOKENS:-1024}"
OUT_TOKENS="${OUT_TOKENS:-512}"
NUM_RUNS="${NUM_RUNS:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"

MODEL_NAME=$(basename "$MODEL")
RESULTS_DIR="${RESULTS_DIR:-${MODEL_NAME}-MI325X-TP1/}"

tokenomics completion \
  --model "$MODEL" \
  --scenario "N(${IN_TOKENS},0)/(${OUT_TOKENS},0)" \
  --dataset-config examples/dataset_configs/aime_simple.json \
  --api-base "$API_BASE" \
  --max-concurrency 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192 \
  --num-runs "$NUM_RUNS" \
  --description "${MODEL_NAME} burst mode MI325X TP1" \
  --warmup-runs "$WARMUP_RUNS" \
  --results-dir "$RESULTS_DIR"
