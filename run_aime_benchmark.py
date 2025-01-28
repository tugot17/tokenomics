from time import perf_counter_ns
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Initialize the model from local path with tensor parallelism
model_path = "/nfs/checkpoint-tuning/deepseek/DeepSeek-R1-Distill-Llama-70B"
llm = LLM(
    model=model_path,
    tensor_parallel_size=8,  # For TP2
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

# Load the dataset
ds = load_dataset("gneubig/aime-1983-2024")
k = 1
sampled_dataset = ds['train'].shuffle(seed=42).select(range(k))

# Create conversations using list comprehension
conversations = [
    [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": question["Question"]
        }
    ] for question in sampled_dataset
]
start = perf_counter_ns()
# Generate responses
outputs = llm.chat(
    conversations,
    sampling_params=sampling_params,
    use_tqdm=True
)
end = perf_counter_ns()

# Function to capture input and output lengths
def capture_lengths(outputs):
    lengths = []
    for output in outputs:
        input_length = len(output.prompt_token_ids)
        output_length = len(output.outputs[0].token_ids)
        lengths.append({
            "input_length": input_length,
            "output_length": output_length
        })
    return lengths

# Capture and print the lengths
lengths = capture_lengths(outputs)

# Calculate total output tokens per second and output tokens per request
total_output_tokens = sum(length["output_length"] for length in lengths)
total_time_seconds = (end - start) / 1e9  # Convert nanoseconds to seconds
total_tokens_per_second = total_output_tokens / total_time_seconds
avere_tokens_per_second_per_request = total_output_tokens / k / total_time_seconds

print("\nPerformance Metrics:")
print("-" * 80)
print(f"Total output tokens: {total_output_tokens}")
print(f"Total time (seconds): {total_time_seconds:.2f}")
print(f"Tokens per second: {total_tokens_per_second:.2f}")
print(f"Tokens per sercond per request: {avere_tokens_per_second_per_request:.2f}")
print("-" * 80)