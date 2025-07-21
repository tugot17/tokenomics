"""
Text sampler that combines statistical scenarios with dataset content.

This module provides the core sampling functionality that generates
realistic requests with controlled token distributions.
"""

import random
import warnings
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer

from .scenarios import Scenario
from .dataset import DatasetLoader


class UserRequest:
    """Represents a user request with prompt and generation parameters."""
    
    def __init__(self, 
                 prompt: str, 
                 max_tokens: int,
                 actual_input_tokens: int,
                 target_input_tokens: int,
                 target_output_tokens: int):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.actual_input_tokens = actual_input_tokens
        self.target_input_tokens = target_input_tokens
        self.target_output_tokens = target_output_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary format for API calls."""
        return {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "actual_input_tokens": self.actual_input_tokens,
            "target_input_tokens": self.target_input_tokens,
            "target_output_tokens": self.target_output_tokens
        }


class TextSampler:
    """
    Text sampler that combines scenario-based parameter generation 
    with dataset-driven content sampling.
    """
    
    def __init__(self, 
                 tokenizer_name: str,
                 dataset_loader: DatasetLoader):
        """
        Initialize the text sampler.
        
        Args:
            tokenizer_name: Name or path of the tokenizer to use
            dataset_loader: Loaded dataset for content sampling
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset_loader = dataset_loader
        
        # Ensure we have a padding token for token counting
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def sample(self, scenario: Scenario) -> UserRequest:
        """
        Sample a user request based on the scenario and dataset.
        
        Args:
            scenario: Statistical scenario for token count sampling
            
        Returns:
            UserRequest with prompt and generation parameters
        """
        # Sample target token counts from scenario
        target_input_tokens, target_output_tokens = scenario.sample()
        
        # Generate prompt with approximately target_input_tokens
        prompt = self._sample_text(target_input_tokens)
        
        # Count actual tokens in the generated prompt
        actual_input_tokens = self._count_tokens(prompt)
        
        # Note: actual_input_tokens may differ from target_input_tokens due to dataset characteristics
        
        return UserRequest(
            prompt=prompt,
            max_tokens=target_output_tokens,
            actual_input_tokens=actual_input_tokens,
            target_input_tokens=target_input_tokens,
            target_output_tokens=target_output_tokens
        )
    
    def _sample_text(self, target_tokens: int) -> str:
        """
        Sample text from dataset to approximately match target token count.
        
        Args:
            target_tokens: Target number of input tokens
            
        Returns:
            Concatenated text approximately matching target token count
        """
        if len(self.dataset_loader) == 0:
            raise ValueError("Dataset is empty")
        
        # For very large token counts, we need a smarter strategy
        all_texts = self.dataset_loader.get_all_texts()
        
        # Start with a random text from the dataset
        texts = []
        current_tokens = 0
        cycles = 0
        
        # Keep adding texts until we reach approximately the target
        while current_tokens < target_tokens:
            # Get a random text sample
            sample_text = random.choice(all_texts)
            texts.append(sample_text)
            
            # Count tokens in the concatenated text so far
            combined_text = " ".join(texts)
            current_tokens = self._count_tokens(combined_text)
            
            # Safety check to avoid infinite loops
            # Allow more samples for very large target token counts
            # Calculate based on average tokens per text in the dataset
            avg_tokens_per_text = target_tokens / len(all_texts) if len(all_texts) > 0 else 100
            max_samples = max(100, int(target_tokens / max(10, avg_tokens_per_text * 0.5)))
            
            if len(texts) > max_samples:
                warnings.warn(f"Exceeded {max_samples} text samples while trying to reach {target_tokens} tokens. Got {current_tokens} tokens from {len(texts)} samples.")
                break
            
            # Track cycles through the dataset
            if len(texts) % len(all_texts) == 0:
                cycles += 1
                if cycles > 10:  # Prevent excessive cycling
                    warnings.warn(f"Cycled through dataset {cycles} times trying to reach {target_tokens} tokens. Got {current_tokens} tokens.")
                    break
        
        return " ".join(texts)
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception as e:
            warnings.warn(f"Failed to tokenize text: {e}")
            # Fallback to rough word count * 1.3 (typical token-to-word ratio)
            return int(len(text.split()) * 1.3)
    


class BatchSampler:
    """
    Batch sampler that generates multiple requests based on scenarios.
    """
    
    def __init__(self, text_sampler: TextSampler):
        self.text_sampler = text_sampler
    
    def sample_batch(self, scenario: Scenario, batch_size: int) -> List[UserRequest]:
        """
        Sample a batch of requests based on the scenario.
        
        Args:
            scenario: Statistical scenario for token count sampling
            batch_size: Number of requests to generate
            
        Returns:
            List of UserRequest objects
        """
        requests = []
        for _ in range(batch_size):
            request = self.text_sampler.sample(scenario)
            requests.append(request)
        
        return requests
    
    def sample_mixed_batch(self, scenarios: List[Tuple[Scenario, float]], batch_size: int) -> List[UserRequest]:
        """
        Sample a batch with mixed scenarios based on weights.
        
        Args:
            scenarios: List of (scenario, weight) tuples
            batch_size: Number of requests to generate
            
        Returns:
            List of UserRequest objects
        """
        if not scenarios:
            raise ValueError("At least one scenario must be provided")
        
        # Normalize weights
        total_weight = sum(weight for _, weight in scenarios)
        normalized_scenarios = [(scenario, weight / total_weight) for scenario, weight in scenarios]
        
        requests = []
        for _ in range(batch_size):
            # Choose scenario based on weights
            rand_val = random.random()
            cumulative_weight = 0
            
            for scenario, weight in normalized_scenarios:
                cumulative_weight += weight
                if rand_val <= cumulative_weight:
                    request = self.text_sampler.sample(scenario)
                    requests.append(request)
                    break
        
        return requests