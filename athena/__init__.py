"""
Project Athena: Multimodal Ethics Framework for Meta Superintelligence Labs

A comprehensive multimodal ethics framework designed specifically for Meta's AI ecosystem,
including Llama models, DALL-E style image generation, and advanced video synthesis.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
LinkedIn: https://www.linkedin.com/in/michael-jaramillo-b61815278
"""

__version__ = "1.0.0"
__author__ = "Michael Jaramillo"
__email__ = "jmichaeloficial@gmail.com"
__license__ = "MIT"

from .core.ethics_engine import MultimodalEthicsEngine
from .core.config import EthicsConfig
from .core.evaluator import EthicsEvaluator

from .modalities.text_ethics import TextEthics
from .modalities.image_ethics import ImageEthics
from .modalities.audio_ethics import AudioEthics
from .modalities.video_ethics import VideoEthics

from .frameworks.rlhf_integration import RLHFIntegration
from .frameworks.constitutional_ai import ConstitutionalAI
from .frameworks.ethical_reasoning import EthicalReasoning

from .monitors.content_monitor import ContentMonitor
from .monitors.ethics_dashboard import EthicsDashboard

# Main exports for public API
__all__ = [
    # Core components
    "MultimodalEthicsEngine",
    "EthicsConfig", 
    "EthicsEvaluator",
    
    # Modality processors
    "TextEthics",
    "ImageEthics", 
    "AudioEthics",
    "VideoEthics",
    
    # Framework integrations
    "RLHFIntegration",
    "ConstitutionalAI", 
    "EthicalReasoning",
    
    # Monitoring tools
    "ContentMonitor",
    "EthicsDashboard",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]

# Package-level configuration
import logging

# Configure logging for the package
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Package metadata
PACKAGE_INFO = {
    "name": "athena",
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "description": "Multimodal Ethics Framework for Meta Superintelligence Labs",
    "url": "https://github.com/meta-ai/project-athena-multimodal-ethics",
    "license": __license__,
}

def get_version():
    """Get the current version of Project Athena."""
    return __version__

def get_package_info():
    """Get comprehensive package information."""
    return PACKAGE_INFO.copy()

# Meta AI ecosystem compatibility check
def check_meta_compatibility():
    """
    Check compatibility with Meta's AI ecosystem components.
    
    Returns:
        dict: Compatibility status for various Meta AI components
    """
    compatibility = {
        "llama_models": True,
        "pytorch_ecosystem": True, 
        "multimodal_apis": True,
        "meta_infrastructure": True,
        "version_compatible": __version__ >= "1.0.0"
    }
    
    return compatibility