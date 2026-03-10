"""
Scenario definitions for statistical parameter generation.

Inspired by GenAI-Bench's scenario system, this module provides
different distribution types for generating request parameters.
"""

import re
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Type, Tuple, Union


class DistributionType(Enum):
    """Distribution types for scenario sampling."""
    NORMAL = "N"
    DETERMINISTIC = "D" 
    UNIFORM = "U"


class Scenario(ABC):
    """
    Abstract base class for different statistical scenarios.
    
    Each scenario defines how to sample input and output token counts
    based on different statistical distributions.
    """
    
    _registry: Dict[str, Type["Scenario"]] = {}
    distribution_type: DistributionType
    validation_pattern: str
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.distribution_type.value] = cls
    
    @abstractmethod
    def sample(self) -> Tuple[int, int]:
        """
        Sample input and output token counts.
        
        Returns:
            Tuple of (num_input_tokens, num_output_tokens)
        """
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Convert scenario back to string representation."""
        pass
    
    @classmethod
    @abstractmethod
    def parse(cls, params_str: str) -> "Scenario":
        """Parse scenario parameters from string."""
        pass
    
    @classmethod
    def from_string(cls, scenario_str: str) -> "Scenario":
        """
        Factory method to create scenario from string.
        
        Args:
            scenario_str: String like "N(480,240)/(300,150)" or "D(100,100)"
            
        Returns:
            Scenario instance
        """
        cls.validate(scenario_str)
        type_identifier = scenario_str[0]
        scenario_class = cls._registry.get(type_identifier)
        
        if scenario_class is None:
            raise ValueError(f"Unknown scenario type: {type_identifier}")
            
        return scenario_class.parse(scenario_str[1:])
    
    @classmethod
    def validate(cls, scenario_str: str) -> bool:
        """Validate scenario string format."""
        if not scenario_str:
            raise ValueError("Scenario string cannot be empty")
            
        type_identifier = scenario_str[0]
        if type_identifier not in cls._registry:
            raise ValueError(f"Unknown scenario type: {type_identifier}")
            
        scenario_class = cls._registry[type_identifier]
        pattern = scenario_class.validation_pattern
        
        if not re.match(pattern, scenario_str):
            raise ValueError(f"Invalid scenario format: {scenario_str}")
            
        return True


class NormalDistribution(Scenario):
    """Normal distribution scenario: N(mean_input,std_input)/(mean_output,std_output)"""
    
    distribution_type = DistributionType.NORMAL
    validation_pattern = r"^N\(\d+,\d+\)/\(\d+,\d+\)$"
    
    def __init__(self, mean_input: int, std_input: int, mean_output: int, std_output: int):
        self.mean_input = mean_input
        self.std_input = std_input
        self.mean_output = mean_output
        self.std_output = std_output
    
    def sample(self) -> Tuple[int, int]:
        """Sample from normal distributions with minimum constraints."""
        input_tokens = max(1, int(np.random.normal(self.mean_input, self.std_input)))
        output_tokens = max(1, int(np.random.normal(self.mean_output, self.std_output)))
        return input_tokens, output_tokens
    
    def to_string(self) -> str:
        return f"N({self.mean_input},{self.std_input})/({self.mean_output},{self.std_output})"
    
    @classmethod
    def parse(cls, params_str: str) -> "NormalDistribution":
        """Parse N(mean_input,std_input)/(mean_output,std_output) format."""
        # Remove parentheses and split by )/(
        params_str = params_str.strip("()")
        input_part, output_part = params_str.split(")/(")
        
        # Parse input parameters
        mean_input, std_input = map(int, input_part.split(","))
        
        # Parse output parameters  
        mean_output, std_output = map(int, output_part.split(","))
        
        return cls(mean_input, std_input, mean_output, std_output)


class DeterministicDistribution(Scenario):
    """Deterministic scenario: D(input_tokens,output_tokens)"""
    
    distribution_type = DistributionType.DETERMINISTIC
    validation_pattern = r"^D\(\d+,\d+\)$"
    
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
    
    def sample(self) -> Tuple[int, int]:
        """Return fixed token counts."""
        return self.input_tokens, self.output_tokens
    
    def to_string(self) -> str:
        return f"D({self.input_tokens},{self.output_tokens})"
    
    @classmethod
    def parse(cls, params_str: str) -> "DeterministicDistribution":
        """Parse D(input_tokens,output_tokens) format."""
        params_str = params_str.strip("()")
        input_tokens, output_tokens = map(int, params_str.split(","))
        return cls(input_tokens, output_tokens)


class UniformDistribution(Scenario):
    """Uniform distribution scenario: U(min_input,max_input)/(min_output,max_output)"""
    
    distribution_type = DistributionType.UNIFORM
    validation_pattern = r"^U\(\d+,\d+\)/\(\d+,\d+\)$"
    
    def __init__(self, min_input: int, max_input: int, min_output: int, max_output: int):
        self.min_input = min_input
        self.max_input = max_input
        self.min_output = min_output
        self.max_output = max_output
    
    def sample(self) -> Tuple[int, int]:
        """Sample from uniform distributions."""
        input_tokens = np.random.randint(self.min_input, self.max_input + 1)
        output_tokens = np.random.randint(self.min_output, self.max_output + 1)
        return input_tokens, output_tokens
    
    def to_string(self) -> str:
        return f"U({self.min_input},{self.max_input})/({self.min_output},{self.max_output})"
    
    @classmethod
    def parse(cls, params_str: str) -> "UniformDistribution":
        """Parse U(min_input,max_input)/(min_output,max_output) format."""
        params_str = params_str.strip("()")
        input_part, output_part = params_str.split(")/(")
        
        min_input, max_input = map(int, input_part.split(","))
        min_output, max_output = map(int, output_part.split(","))
        
        return cls(min_input, max_input, min_output, max_output)