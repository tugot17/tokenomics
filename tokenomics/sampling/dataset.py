"""
Dataset integration system for content sampling.

This module provides flexible dataset loading and content sampling
to work with the scenario-based parameter generation.
"""

import json
import csv
import itertools
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset


def _coerce_prompt(value: Any) -> str:
    """Reduce a prompt-column value to a single string.

    Handles plain strings, multi-turn list columns (e.g. MT-Bench ``prompt`` is
    a list of turn strings -> first turn), and list-of-dict columns (e.g.
    Arena-Hard ``turns`` is ``[{"content": ...}]`` -> first turn's content).
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("content") or value.get("value") or value.get("text") or ""
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return first.get("content") or first.get("value") or first.get("text") or ""
        return str(first)
    return str(value) if value is not None else ""


class DatasetConfig:
    """Configuration for dataset loading."""
    
    def __init__(self, config_dict: Dict[str, Any], base_dir: Optional[Path] = None):
        self.config = config_dict
        self.base_dir = base_dir or Path.cwd()
        self.source_type = config_dict.get("source", {}).get("type", "file")
        self.source_path = config_dict.get("source", {}).get("path", "")
        self.prompt_column = config_dict.get("prompt_column", "text")
        # Optional image column for multimodal (VL) replay datasets.
        self.image_column = config_dict.get("image_column")

    @classmethod
    def from_file(cls, config_path: str) -> "DatasetConfig":
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict, base_dir=config_path.parent)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DatasetConfig":
        """Create configuration from dictionary."""
        return cls(config_dict)


class DatasetLoader:
    """Loads and manages dataset content for sampling."""
    
    def __init__(self, config: DatasetConfig, limit: Optional[int] = None):
        self.config = config
        # Stop after `limit` rows instead of materializing the whole dataset —
        # avoids eagerly base64-encoding every image of a large vision dataset
        # when the run only replays the first N rows. None = load everything.
        self.limit = limit
        self.data: List[str] = []
        # Per-row image data URIs, aligned with self.data (empty for text datasets).
        self.images: List[List[str]] = []
        self._load_data()

    def _reached_limit(self) -> bool:
        return self.limit is not None and len(self.data) >= self.limit
    
    def _load_data(self) -> None:
        """Load data based on configuration."""
        if self.config.source_type == "file":
            self._load_from_file()
        elif self.config.source_type == "huggingface":
            self._load_from_huggingface()
        elif self.config.source_type == "aime":
            self._load_aime_dataset()
        else:
            raise ValueError(f"Unknown source type: {self.config.source_type}")
    
    def _load_from_file(self) -> None:
        """Load data from local file."""
        file_path = self.config.base_dir / self.config.source_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        if file_path.suffix == '.csv':
            self._load_from_csv(file_path)
        elif file_path.suffix == '.json':
            self._load_from_json(file_path)
        elif file_path.suffix == '.txt':
            self._load_from_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_from_csv(self, file_path: Path) -> None:
        """Load data from CSV file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.config.prompt_column in row:
                    self.data.append(row[self.config.prompt_column])
                    if self._reached_limit():
                        break

    def _load_from_json(self, file_path: Path) -> None:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and self.config.prompt_column in item:
                    self.data.append(item[self.config.prompt_column])
                elif isinstance(item, str):
                    self.data.append(item)
                if self._reached_limit():
                    break

    def _load_from_text(self, file_path: Path) -> None:
        """Load data from text file (one line per sample)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = (line.strip() for line in f if line.strip())
            if self.limit is not None:
                self.data = list(itertools.islice(lines, self.limit))
            else:
                self.data = list(lines)
    
    def _load_from_huggingface(self) -> None:
        """Load data from HuggingFace dataset."""
        hf_config = self.config.config.get("source", {})
        dataset_name = hf_config.get("path", "")
        kwargs = hf_config.get("huggingface_kwargs", {})
        
        if not dataset_name:
            raise ValueError("HuggingFace dataset path is required")
        
        try:
            dataset = load_dataset(dataset_name, **kwargs)

            # load_dataset returns a bare Dataset when `split` is given, or a
            # DatasetDict (split name -> Dataset) otherwise. Handle both.
            if hasattr(dataset, "keys"):
                if "train" in dataset:
                    data_split = dataset["train"]
                else:
                    data_split = dataset[list(dataset.keys())[0]]
            else:
                data_split = dataset

            img_col = self.config.image_column
            if img_col:
                from .images import image_value_to_uris
            for item in data_split:
                if self.config.prompt_column in item:
                    text = _coerce_prompt(item[self.config.prompt_column])
                    if not text and not img_col:
                        continue
                    self.data.append(text)
                    if img_col:
                        self.images.append(image_value_to_uris(item.get(img_col)))
                    if self._reached_limit():
                        break

        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace dataset: {e}")
    
    def _load_aime_dataset(self) -> None:
        """Load AIME dataset (for compatibility with existing code)."""
        try:
            dataset = load_dataset("gneubig/aime-1983-2024")
            for item in dataset["train"]:
                # Create a system + user prompt like in the original
                prompt = f"You are an assistant that helps students solve challenging problems.\n\n{item['Question']}"
                self.data.append(prompt)
                if self._reached_limit():
                    break
        except Exception as e:
            raise RuntimeError(f"Failed to load AIME dataset: {e}")
    
    def get_sample_texts(self, count: int = 1) -> List[str]:
        """Get random sample texts from the dataset."""
        if not self.data:
            raise ValueError("No data loaded")
        
        return random.choices(self.data, k=count)
    
    def get_all_texts(self) -> List[str]:
        """Get all texts from the dataset."""
        return self.data.copy()

    def get_all_images(self) -> List[List[str]]:
        """Per-row image ``data:`` URIs, aligned with get_all_texts (empty if none)."""
        return self.images.copy()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)