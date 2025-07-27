"""
Modality-specific ethics processors for Project Athena

Specialized analyzers for text, image, audio, and video content
with Meta AI ecosystem integration and advanced ethical frameworks.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

from .text_ethics import TextEthics
from .image_ethics import ImageEthics
from .audio_ethics import AudioEthics
from .video_ethics import VideoEthics

__all__ = [
    "TextEthics",
    "ImageEthics", 
    "AudioEthics",
    "VideoEthics",
]