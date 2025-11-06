#!/bin/bash

MODEL="Qwen3-4B"
TOKENIZER="Qwen/Qwen3-4B"
API_BASE="http://localhost:8000/v1"
DATASET_CONFIG="examples/dataset_configs/aime_simple.json"
SCENARIO="N(3000,50)/(50,0)"
BATCH_SIZES="1,2,4,8,16,32,64,128"
NUM_RUNS=3
RESULTS_DIR="${1:-lora_results}"

mkdir -p "$RESULTS_DIR"

echo "Test 1: Baseline (no LoRA)"
python completion_advanced_benchmark.py \
    --model "$MODEL" \
    --scenario "$SCENARIO" \
    --dataset-config "$DATASET_CONFIG" \
    --api-base "$API_BASE" \
    --batch-sizes "$BATCH_SIZES" \
    --num-runs "$NUM_RUNS" \
    --tokenizer "$TOKENIZER" \
    --description "Baseline: No LoRA" \
    --results-file "$RESULTS_DIR/baseline.json"

echo ""
echo "Test 2: Uniform 4 LoRAs"
python completion_advanced_benchmark.py \
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
    --results-file "$RESULTS_DIR/uniform_4.json"

echo ""
echo "Test 3: All unique 8 LoRAs"
python completion_advanced_benchmark.py \
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
    --results-file "$RESULTS_DIR/all_unique_8.json"

echo ""
echo "Done. Results in $RESULTS_DIR/"
