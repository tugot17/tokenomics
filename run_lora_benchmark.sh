#!/bin/bash

MODEL="Qwen/Qwen3-4B"
TOKENIZER="Qwen/Qwen3-4B"
API_BASE="http://localhost:8000/v1"
DATASET_CONFIG="examples/dataset_configs/aime_simple.json"
SCENARIO="N(3000,50)/(50,0)"
BATCH_SIZES="1,2,4,8,16,32,64,128"
NUM_RUNS=3
RESULTS_BASE="${1:-lora_results}"

echo "Test 1: Baseline (no LoRA)"
tokenomics completion \
    --model "$MODEL" \
    --scenario "$SCENARIO" \
    --dataset-config "$DATASET_CONFIG" \
    --api-base "$API_BASE" \
    --batch-sizes "$BATCH_SIZES" \
    --num-runs "$NUM_RUNS" \
    --tokenizer "$TOKENIZER" \
    --description "Baseline: No LoRA" \
    --results-dir "$RESULTS_BASE/baseline"

echo ""
echo "Test 2: Uniform 4 LoRAs"
tokenomics completion \
    --model "$MODEL" \
    --scenario "$SCENARIO" \
    --dataset-config "$DATASET_CONFIG" \
    --api-base "$API_BASE" \
    --batch-sizes "$BATCH_SIZES" \
    --num-runs "$NUM_RUNS" \
    --tokenizer "$TOKENIZER" \
    --lora-strategy uniform \
    --lora-names lora_finance,lora_medical,lora_legal,lora_coding \
    --description "Uniform: 4 LoRAs" \
    --results-dir "$RESULTS_BASE/uniform_4"

echo ""
echo "Test 3: All unique 8 LoRAs"
tokenomics completion \
    --model "$MODEL" \
    --scenario "$SCENARIO" \
    --dataset-config "$DATASET_CONFIG" \
    --api-base "$API_BASE" \
    --batch-sizes "$BATCH_SIZES" \
    --num-runs "$NUM_RUNS" \
    --tokenizer "$TOKENIZER" \
    --lora-strategy all-unique \
    --lora-names lora_finance,lora_medical,lora_legal,lora_coding,lora_creative,lora_technical,lora_customer,lora_research \
    --description "Stress test: 8 unique LoRAs" \
    --results-dir "$RESULTS_BASE/all_unique_8"

echo ""
echo "Done. Results in $RESULTS_BASE/"
