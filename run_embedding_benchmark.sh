uv run python embedding_benchmark.py \
  --model Qwen/Qwen3-Embedding-4B \
  --sequence_lengths "200" \
  --batch_sizes "1024,2048" \
  --description "Qwen3-Embedding-4B running on 4090" \