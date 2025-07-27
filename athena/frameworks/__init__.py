"""
Ethical Frameworks Integration for Project Athena

Advanced ethical reasoning systems including RLHF, Constitutional AI,
and formal ethical frameworks with Meta AI integration.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

from .rlhf_integration import RLHFIntegration
from .constitutional_ai import ConstitutionalAI
from .ethical_reasoning import EthicalReasoning
from .utilitarian_framework import UtilitarianFramework
from .deontological_framework import DeontologicalFramework
from .virtue_ethics_framework import VirtueEthicsFramework

__all__ = [
    "RLHFIntegration",
    "ConstitutionalAI",
    "EthicalReasoning",
    "UtilitarianFramework",
    "DeontologicalFramework", 
    "VirtueEthicsFramework",
]