"""
Text sampler that combines statistical scenarios with dataset content.
"""

import random
import warnings
import hashlib
import contextlib
import numpy as np
from typing import List, NamedTuple
from tokenizers import Tokenizer

from .scenarios import Scenario
from .dataset import DatasetLoader


class UserRequest(NamedTuple):
    prompt: str
    max_tokens: int
    actual_input_tokens: int
    target_input_tokens: int
    target_output_tokens: int


class TextSampler:
    """Combines scenario-based parameter generation with dataset-driven content sampling."""

    def __init__(self, tokenizer_name: str, dataset_loader: DatasetLoader):
        self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        self.dataset_loader = dataset_loader

        import time
        print(f"Pre-tokenizing {len(dataset_loader)} dataset texts...")
        t0 = time.time()
        self.text_token_cache = [
            (text, self._count_tokens(text))
            for text in dataset_loader.get_all_texts()
        ]
        total_tokens = sum(count for _, count in self.text_token_cache)
        print(f"Pre-tokenization complete. {len(self.text_token_cache)} texts, "
              f"{total_tokens:,} tokens in {time.time() - t0:.2f}s")

    def sample(self, scenario: Scenario) -> UserRequest:
        """Sample a single user request based on the scenario and dataset."""
        target_input, target_output = scenario.sample()
        prompt, actual_input = self._sample_text_with_count(target_input)
        return UserRequest(prompt, target_output, actual_input, target_input, target_output)

    def sample_batch(self, scenario: Scenario, batch_size: int) -> List[UserRequest]:
        """Sample a batch of requests."""
        return [self.sample(scenario) for _ in range(batch_size)]

    def _sample_text_with_count(self, target_tokens: int) -> tuple[str, int]:
        """Sample and concatenate texts until reaching approximately target_tokens."""
        if not self.text_token_cache:
            raise ValueError("Dataset is empty")

        texts = []
        current_tokens = 0
        max_iters = max(100, 10 * len(self.text_token_cache))

        while current_tokens < target_tokens and len(texts) < max_iters:
            text, tokens = random.choice(self.text_token_cache)
            texts.append(text)
            current_tokens += tokens

        return " ".join(texts), current_tokens

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=True).ids)
        except Exception:
            return int(len(text.split()) * 1.3)


class DatasetReplaySampler:
    """Sends each dataset row verbatim as one request, walking the dataset in order.

    The prompt *is* the dataset row
    (no scenario-driven length padding, no concatenation of random snippets), and
    generation runs to natural EOS capped by ``max_tokens``. Rows are returned
    deterministically in dataset order, so the prompt set is identical across every
    concurrency level and run -- which is what makes throughput comparable example
    by example. It implements the same ``sample_batch`` / ``tokenizer`` surface as
    :class:`TextSampler`, so it is a drop-in replacement in the benchmark loop.
    """

    def __init__(self, tokenizer_name: str, dataset_loader: DatasetLoader, max_tokens: int):
        self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        self.dataset_loader = dataset_loader
        self.max_tokens = max_tokens
        self.rows = dataset_loader.get_all_texts()
        if not self.rows:
            raise ValueError("Dataset is empty")

    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=True).ids)
        except Exception:
            return int(len(text.split()) * 1.3)

    def sample_batch(self, scenario: Scenario, batch_size: int) -> List[UserRequest]:
        """Return the first ``batch_size`` rows verbatim, in dataset order.

        ``scenario`` is accepted for interface compatibility with
        :class:`TextSampler` but ignored -- the row content defines the prompt.
        If ``batch_size`` exceeds the dataset size it is capped (we walk the
        dataset once, never cycle). ``target_output_tokens`` carries max_tokens
        so the benchmark loop's ``min(args.max_tokens, target_output_tokens)``
        resolves to max_tokens; generation still stops at EOS (unless
        --ignore-eos).
        """
        out = []
        for i in range(min(batch_size, len(self.rows))):
            text = self.rows[i]
            n_tok = self._count_tokens(text)
            out.append(UserRequest(text, self.max_tokens, n_tok, n_tok, self.max_tokens))
        return out


@contextlib.contextmanager
def use_seed(seed: int):
    state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    # NumPy's RandomState requires a 32-bit unsigned seed
    np.random.seed(int(seed % (2**32)))
    try:
        yield
    finally:
        random.setstate(state)
        np.random.set_state(np_state)


def derive_seed(base_seed: int, batch_size: int, run_idx: int) -> int:
    s = f"{base_seed}|{batch_size}|{run_idx}".encode()
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")