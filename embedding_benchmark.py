import argparse
import concurrent.futures
import time
import statistics
import json
import random
from datetime import datetime
from openai import OpenAI
from typing import List, Tuple, Dict, Any


class TextGenerator:
    """Generate random text of varying lengths for embedding benchmarks."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
        # Sample vocabulary for generating random text
        self.words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "and", "runs", "through", "forest", "green", "mountain", "river",
            "flows", "gently", "under", "bright", "sun", "shines", "down",
            "upon", "beautiful", "landscape", "where", "birds", "sing",
            "songs", "of", "joy", "happiness", "peace", "tranquility",
            "nature", "provides", "sanctuary", "for", "all", "living",
            "creatures", "who", "seek", "refuge", "from", "busy", "world",
            "technology", "advances", "rapidly", "changing", "how", "we",
            "live", "work", "communicate", "with", "each", "other",
            "artificial", "intelligence", "machine", "learning", "deep",
            "neural", "networks", "computer", "science", "data", "analysis",
            "algorithms", "processing", "information", "knowledge", "wisdom",
            "understanding", "human", "experience", "consciousness",
            "philosophy", "science", "mathematics", "physics", "chemistry",
            "biology", "history", "culture", "society", "economics",
            "politics", "government", "education"
        ]
        
        self.sentence_starters = [
            "The researchers discovered that",
            "According to recent studies,",
            "In the field of artificial intelligence,",
            "Scientists have observed that",
            "The latest developments in technology show",
            "Experts believe that",
            "Recent experiments demonstrate",
            "Analysis of the data reveals",
            "The study concluded that",
            "Observations indicate that"
        ]
    
    def generate_text(self, target_length) -> str:
        """Generate random text of specified target length.
        
        Args:
            target_length: Either a string category ('short', 'medium', 'long')
                          or an integer specifying exact number of words
        """
        if isinstance(target_length, int):
            return self._generate_exact_words(target_length)
        elif target_length == "short":
            return self._generate_short_text()
        elif target_length == "medium":
            return self._generate_medium_text()
        elif target_length == "long":
            return self._generate_long_text()
        else:
            raise ValueError(f"Unknown target length: {target_length}")
    
    def _generate_exact_words(self, num_words: int) -> str:
        """Generate text with exactly the specified number of words."""
        if num_words <= 0:
            return ""
        
        words = random.choices(self.words, k=num_words)
        text = " ".join(words).capitalize()
        
        # Add appropriate punctuation based on length
        if num_words <= 20:
            return text + "."
        elif num_words <= 50:
            # Add a sentence starter for medium length
            starter = random.choice(self.sentence_starters)
            remaining_words = max(0, num_words - len(starter.split()))
            if remaining_words > 0:
                additional_words = random.choices(self.words, k=remaining_words)
                return starter + " " + " ".join(additional_words) + "."
            else:
                return starter + "."
        else:
            # For longer texts, create multiple sentences
            sentences = []
            words_remaining = num_words
            
            while words_remaining > 0:
                if words_remaining <= 15:
                    # Final short sentence
                    sentence_words = random.choices(self.words, k=words_remaining)
                    sentences.append(" ".join(sentence_words).capitalize() + ".")
                    words_remaining = 0
                else:
                    # Create a sentence with 10-25 words
                    sentence_length = min(random.randint(10, 25), words_remaining)
                    if len(sentences) == 0:
                        # First sentence gets a starter
                        starter = random.choice(self.sentence_starters)
                        starter_words = len(starter.split())
                        remaining = max(0, sentence_length - starter_words)
                        if remaining > 0:
                            additional = random.choices(self.words, k=remaining)
                            sentence = starter + " " + " ".join(additional) + "."
                        else:
                            sentence = starter + "."
                        words_remaining -= sentence_length
                    else:
                        # Subsequent sentences
                        sentence_words = random.choices(self.words, k=sentence_length)
                        sentence = " ".join(sentence_words).capitalize() + "."
                        words_remaining -= sentence_length
                    
                    sentences.append(sentence)
            
            return " ".join(sentences)

    def _generate_short_text(self) -> str:
        """Generate 5-15 word phrases."""
        num_words = random.randint(5, 15)
        words = random.choices(self.words, k=num_words)
        return " ".join(words).capitalize() + "."
    
    def _generate_medium_text(self) -> str:
        """Generate 20-50 word sentences."""
        starter = random.choice(self.sentence_starters)
        num_words = random.randint(15, 40)
        words = random.choices(self.words, k=num_words)
        return starter + " " + " ".join(words) + "."
    
    def _generate_long_text(self) -> str:
        """Generate 100-300 word paragraphs."""
        num_sentences = random.randint(4, 8)
        sentences = []
        
        for _ in range(num_sentences):
            starter = random.choice(self.sentence_starters)
            num_words = random.randint(15, 40)
            words = random.choices(self.words, k=num_words)
            sentence = starter + " " + " ".join(words) + "."
            sentences.append(sentence)
        
        return " ".join(sentences)
    
    def generate_batch(self, batch_size: int, length_distribution="mixed") -> List[str]:
        """Generate a batch of texts with specified length distribution.
        
        Args:
            batch_size: Number of texts to generate
            length_distribution: Either:
                - String category: 'short', 'medium', 'long', 'mixed'  
                - Integer: exact number of words for all texts
                - List of integers: specific word counts for mixed lengths
        """
        texts = []
        
        if isinstance(length_distribution, int):
            # All texts with the same exact word count
            for _ in range(batch_size):
                texts.append(self.generate_text(length_distribution))
        elif isinstance(length_distribution, list):
            # Mix of specific word counts
            for _ in range(batch_size):
                word_count = random.choice(length_distribution)
                texts.append(self.generate_text(word_count))
        elif length_distribution == "mixed":
            # Mix of different lengths
            lengths = ["short", "medium", "long"]
            for _ in range(batch_size):
                length = random.choice(lengths)
                texts.append(self.generate_text(length))
        elif length_distribution in ["short", "medium", "long"]:
            # All texts of the same category
            for _ in range(batch_size):
                texts.append(self.generate_text(length_distribution))
        else:
            raise ValueError(f"Unknown length distribution: {length_distribution}")
        
        return texts


def call_tokenize_api(client: OpenAI, model: str, text: str) -> int:
    """Call the tokenize endpoint to get token count for a text."""
    try:
        # Note: This is a placeholder - the actual tokenize endpoint might be different
        # The user provided a curl example but we need to adapt it for the OpenAI client
        response = client.post(
            "/tokenize",
            json={
                "model": model,
                "prompt": text,
                "add_special_tokens": True,
                "return_token_strs": False
            }
        )
        return len(response.json().get("tokens", []))
    except Exception as e:
        # Fallback: estimate tokens (rough approximation)
        return len(text.split()) * 1.3  # Approximate tokens per word

def call_server_embeddings(client: OpenAI, model: str, texts: List[str], max_retries: int = 3) -> Tuple[int, int, float, float, float, float]:
    """Call the embeddings API and return metrics with retry logic."""
    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            
            # Set timeout for the request (60 seconds for large batches)
            response = client.embeddings.create(
                input=texts,
                model=model,
                timeout=60.0
            )
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            
            # Calculate token counts (approximate)
            total_tokens = sum(len(text.split()) * 1.3 for text in texts)  # Rough approximation
            embedding_count = len(response.data)
            
            embeddings_per_second = embedding_count / elapsed if elapsed > 0 else 0
            tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0
            
            # Success - return results
            return int(total_tokens), embedding_count, embeddings_per_second, tokens_per_second, start_time, end_time
            
        except Exception as e:
            print(f"Error during embeddings API call (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed. Skipping this batch.")
                return 0, 0, 0, 0, 0, 0

def run_benchmark(client: OpenAI, model: str, text_batches: List[List[str]]) -> Dict[str, Any]:
    """Run embedding benchmark with concurrent requests."""
    batch_start_time = time.perf_counter()
    results = []
    
    print(f"Running {len(text_batches)} concurrent requests")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(text_batches)) as executor:
        futures = [
            executor.submit(call_server_embeddings, client, model, batch)
            for batch in text_batches
        ]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                total_tokens, embedding_count, eps, tps, start_time, end_time = future.result()
                results.append((total_tokens, embedding_count, eps, tps, start_time, end_time))
                if (i + 1) % 50 == 0 or i == len(futures) - 1:
                    print(f"Completed {i+1}/{len(futures)} requests")
            except Exception as e:
                print(f"Request {i+1} failed: {e}")
                results.append((0, 0, 0, 0, 0, 0))
    
    batch_end_time = time.perf_counter()
    
    # Calculate relative end times
    relative_end_times = [end_time - batch_start_time for _, _, _, _, _, end_time in results]
    
    # Extract metrics
    token_counts = [r[0] for r in results]
    embedding_counts = [r[1] for r in results]
    eps_list = [r[2] for r in results]  # embeddings per second
    tps_list = [r[3] for r in results]  # tokens per second
    
    metrics = {
        "tokens": {
            "total_per_batch": token_counts,
            "embeddings_per_batch": embedding_counts
        },
        "timings": {
            "batch_total_seconds": batch_end_time - batch_start_time,
            "fastest_seconds": min(relative_end_times) if relative_end_times else 0,
            "slowest_seconds": max(relative_end_times) if relative_end_times else 0,
            "spread_seconds": max(relative_end_times) - min(relative_end_times) if relative_end_times else 0
        },
        "throughput": {
            "batch_embeddings_per_second": sum(embedding_counts) / (batch_end_time - batch_start_time) if batch_end_time > batch_start_time else 0,
            "batch_tokens_per_second": sum(token_counts) / (batch_end_time - batch_start_time) if batch_end_time > batch_start_time else 0,
            "request_embeddings_per_second": eps_list,
            "request_tokens_per_second": tps_list
        }
    }
    
    return metrics

def calculate_stats(run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics across multiple runs."""
    def compute_stats(metrics_list: List[Dict], key_path: List[str]) -> Dict[str, float]:
        if isinstance(metrics_list[0][key_path[0]][key_path[1]], list):
            # For lists of values
            values = [item for m in metrics_list for item in m[key_path[0]][key_path[1]]]
        else:
            # For single values
            values = [get_nested_value(m, key_path) for m in metrics_list]
        
        return {
            "mean": statistics.mean(values) if values else 0,
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0
        }
    
    def get_nested_value(d: Dict, path: List[str]) -> Any:
        for key in path:
            d = d[key]
        return d
    
    stats = {}
    for category in ["tokens", "timings", "throughput"]:
        stats[category] = {}
        for metric in run_metrics[0][category]:
            stats[category][metric] = compute_stats(run_metrics, [category, metric])
    
    return stats

def save_results(results: Dict[str, Any], filename: str):
    """Save benchmark results to JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Embedding Benchmark for vLLM/OpenAI Compatible Server"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Embedding model to use (e.g., 'sentence-transformers/all-MiniLM-L6-v2')")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1",
                        help="Base URL of the server API")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8,16",
                        help="Comma-separated batch sizes to test (e.g., '1,2,4,8,16')")
    parser.add_argument("--sequence_lengths", type=str, default="short,medium,long,mixed",
                        help="Comma-separated sequence lengths: categories (short,medium,long,mixed) or word counts (10,25,50)")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs per configuration")
    parser.add_argument("--warmup_runs", type=int, default=2,
                        help="Number of warmup runs before each configuration")
    # Note: Concurrent/sequential options removed - embedding benchmarks test single large batches
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for text generation")
    parser.add_argument("--description", type=str, default="",
                        help="Optional description for the experiment")
    parser.add_argument("--results_file", type=str, default="embedding_benchmark_results.json",
                        help="Path to JSON file for saving results")

    parser.add_argument("--save_progress", action="store_true", default=False,
                        help="Save results after each configuration")
    
    args = parser.parse_args()
    
    # Parse arguments
    batch_sizes = [int(bs.strip()) for bs in args.batch_sizes.split(",") if bs.strip()]
    
    # Parse sequence lengths - can be categories or word counts
    sequence_lengths = []
    for sl in args.sequence_lengths.split(","):
        sl = sl.strip()
        if sl.isdigit():
            sequence_lengths.append(int(sl))
        else:
            sequence_lengths.append(sl)
    
    # Embedding benchmarks test single large batches
    
    # Initialize client with better timeout and retry settings
    client = OpenAI(
        api_key="dummy", 
        base_url=args.api_base,
        timeout=120.0,  # 2 minute timeout
        max_retries=2
    )
    text_gen = TextGenerator(seed=args.seed)
    
    # Initialize results structure
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "api_base": args.api_base,
            "batch_sizes": batch_sizes,
            "sequence_lengths": sequence_lengths,
            "num_runs": args.num_runs,
            "warmup_runs": args.warmup_runs,
            "concurrent_requests": True,
            "seed": args.seed,
            "description": args.description
        },
        "results": {}
    }
    
    print(f"Starting embedding benchmark for model: {args.model}")
    print(f"Testing concurrent requests: {batch_sizes}")
    
    # Run benchmarks for each configuration
    for batch_size in batch_sizes:
        for seq_length in sequence_lengths:
            seq_display = str(seq_length)
            config_key = f"batch_{batch_size}_seq_{seq_display}"
            print(f"\n=== Benchmarking {config_key} ===")
            
            # Warmup runs
            print(f"Performing {args.warmup_runs} warmup runs...", end="", flush=True)
            for _ in range(args.warmup_runs):
                warmup_texts = [text_gen.generate_batch(1, seq_length)]
                call_server_embeddings(client, args.model, warmup_texts[0])
            print(" done")
            
            # Actual benchmark runs
            run_metrics = []
            for run in range(1, args.num_runs + 1):
                print(f"Run {run}/{args.num_runs} ... ", end="", flush=True)
                
                # Generate separate batches for concurrent testing
                # Each batch has 1 text, we send batch_size concurrent requests
                text_batches = [
                    text_gen.generate_batch(1, seq_length)
                    for _ in range(batch_size)
                ]
                
                metrics = run_benchmark(client, args.model, text_batches)
                run_metrics.append(metrics)
                
                # Print run summary
                mean_embeddings = statistics.mean(metrics['tokens']['embeddings_per_batch'])
                mean_tokens = statistics.mean(metrics['tokens']['total_per_batch'])
                batch_eps = metrics['throughput']['batch_embeddings_per_second']
                
                print(f"done")
                print(f"  Embeddings: {mean_embeddings:.1f}/batch")
                print(f"  Tokens: {mean_tokens:.1f}/batch")
                print(f"  Batch time: {metrics['timings']['batch_total_seconds']:.3f}s")
                print(f"  Batch EPS: {batch_eps:.2f}")
            
            # Calculate statistics across runs
            stats = calculate_stats(run_metrics)
            results["results"][config_key] = stats
            
            # Print summary statistics
            print(f"\nSummary for {config_key}:")
            print(f"  Batch Embeddings/sec: {stats['throughput']['batch_embeddings_per_second']['mean']:.2f} ± {stats['throughput']['batch_embeddings_per_second']['std']:.2f}")
            print(f"  Batch Tokens/sec: {stats['throughput']['batch_tokens_per_second']['mean']:.2f} ± {stats['throughput']['batch_tokens_per_second']['std']:.2f}")
            print(f"  Batch Time: {stats['timings']['batch_total_seconds']['mean']:.3f} ± {stats['timings']['batch_total_seconds']['std']:.3f}s")
    
    # Save results
    save_results(results, args.results_file)
    print(f"\nBenchmark results saved to {args.results_file}")
    
    # Print overall summary
    print(f"\n=== Overall Summary ===")
    print(f"Tested {len(batch_sizes)} batch sizes × {len(sequence_lengths)} sequence lengths")
    print(f"Total configurations: {len(results['results'])}")
    print(f"Results saved to: {args.results_file}")

if __name__ == "__main__":
    main() 