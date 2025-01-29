from time import perf_counter_ns
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np
from typing import List, Dict, Any
import torch
import statistics


def warmup_gpu_cache(model_path: str, tensor_parallel_size: int = 8) -> None:
    """Warm up GPU L2 cache similar to Triton's do_bench approach."""
    # Create a dummy input of similar size to actual workload
    dummy_llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
    )
    dummy_params = SamplingParams(temperature=0.5, max_tokens=200)
    dummy_conversation = [[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"}
    ]]
    
    # Run multiple warmup iterations
    for _ in range(3):
        dummy_llm.chat(dummy_conversation, sampling_params=dummy_params)
    
    # Clear memory
    del dummy_llm
    torch.cuda.empty_cache()

def create_sample_conversations(dataset_name: str, num_samples: int, seed: int = 42) -> List[List[Dict[str, str]]]:
    """Create conversation samples from the dataset."""
    ds = load_dataset(dataset_name)
    sampled_dataset = ds['train'].shuffle(seed=seed).select(range(num_samples))
    
    return [
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": question["Question"]}
        ] for question in sampled_dataset
    ]

def run_single_benchmark(
    llm: LLM,
    conversations: List[List[Dict[str, str]]],
    sampling_params: SamplingParams
) -> Dict[str, float]:
    """Run a single benchmark iteration and return metrics."""
    # Clear cache before the run (not included in timing)
    # clear_cuda_cache()
    
    # Start timing
    start = perf_counter_ns()
    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False)
    end = perf_counter_ns()
    
    # Calculate metrics
    total_time_seconds = (end - start) / 1e9
    lengths = [
        {
            "input_length": len(output.prompt_token_ids),
            "output_length": len(output.outputs[0].token_ids)
        }
        for output in outputs
    ]
    
    total_output_tokens = sum(length["output_length"] for length in lengths)
    tokens_per_second = total_output_tokens / total_time_seconds
    tokens_per_second_per_request = tokens_per_second / len(conversations)
        
    return {
        "total_output_tokens": total_output_tokens,
        "total_time_seconds": total_time_seconds,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_per_request": tokens_per_second_per_request
    }


# Initialize the model from local path with tensor parallelism
model_path = "/nfs/checkpoint-tuning/deepseek/DeepSeek-R1-Distill-Llama-8B"
llm = LLM(
    model=model_path,
    tensor_parallel_size=8,
)

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.5, max_tokens=2000)

def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print("-" * 80)
        print("=" * 80)


k = 8
conversations = create_sample_conversations("gneubig/aime-1983-2024", k)

stats = run_single_benchmark(llm, conversations, sampling_params)

print("\nPerformance Metrics:")
print("-" * 80)
print(f"Total output tokens: {stats['total_output_tokens']}")
print(f"Total time (seconds): {stats['total_time_seconds']:.2f}")
print(f"Tokens per second: {stats['tokens_per_second']:.2f}")
print(f"Tokens per sercond per request: {stats['tokens_per_second_per_request']:.2f}")
print("-" * 80)