"""
Sampling system for tokenomics benchmarking.

This module provides a GenAI-Bench inspired sampling mechanism that separates
statistical parameter generation from content sampling.
"""

from .scenarios import Scenario, NormalDistribution, DeterministicDistribution, UniformDistribution
from .sampler import TextSampler, BatchSampler, use_seed, derive_seed
from .dataset import DatasetConfig, DatasetLoader

__all__ = [
    'Scenario',
    'NormalDistribution', 
    'DeterministicDistribution',
    'UniformDistribution',
    'TextSampler',
    'BatchSampler',
    'use_seed',
    'derive_seed',
    'DatasetConfig',
    'DatasetLoader'
]