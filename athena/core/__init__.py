"""
Core components for Project Athena Multimodal Ethics Framework

This module contains the fundamental building blocks of the ethics framework:
- MultimodalEthicsEngine: Central coordination and processing engine
- EthicsConfig: Configuration management and validation
- EthicsEvaluator: Core evaluation logic and scoring algorithms

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

from .ethics_engine import MultimodalEthicsEngine
from .config import EthicsConfig
from .evaluator import EthicsEvaluator

__all__ = [
    "MultimodalEthicsEngine",
    "EthicsConfig", 
    "EthicsEvaluator",
]