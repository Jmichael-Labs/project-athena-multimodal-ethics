"""
Data Processing Utilities for Project Athena

Comprehensive data processing, validation, and preprocessing utilities
for multimodal content handling with Meta AI integration.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json
import pickle
import gzip
import base64
from pathlib import Path
import hashlib
import mimetypes

# Data processing imports
try:
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageOps
    import cv2
    import librosa
    import soundfile as sf
    from moviepy.editor import VideoFileClip
    import torch
    import torchvision.transforms as transforms
except ImportError as e:
    logging.warning(f"Some data processing dependencies not available: {e}")

logger = logging.getLogger(__name__)

@dataclass
class DataValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

@dataclass
class ProcessingResult:
    """Result of data processing operation."""
    success: bool
    data: Any
    processing_time: float
    metadata: Dict[str, Any]
    errors: List[str] = None

class DataValidator:
    """
    Comprehensive data validation for multimodal content.
    
    Validates content format, size, quality, and safety
    before processing through ethics evaluation.
    """
    
    def __init__(self, config):
        """Initialize data validator."""
        self.config = config
        
        # Validation limits
        self.limits = {
            "max_text_length": 50000,
            "max_image_size": (4096, 4096),
            "max_file_size": 100 * 1024 * 1024,  # 100MB
            "min_image_size": (32, 32),
            "max_audio_duration": 3600,  # 1 hour
            "max_video_duration": 7200,  # 2 hours
            "supported_text_encodings": ["utf-8", "ascii", "latin-1"],
            "supported_image_formats": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"],
            "supported_audio_formats": [".mp3", ".wav", ".flac", ".ogg", ".m4a"],
            "supported_video_formats": [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        }
        
        logger.info("Data Validator initialized")
    
    async def validate_content(self, content: Any, content_type: str) -> DataValidationResult:
        """
        Validate content based on type.
        
        Args:
            content: Content to validate
            content_type: Type of content (text, image, audio, video)
        
        Returns:
            DataValidationResult with validation status
        """
        try:
            if content_type == "text":
                return await self._validate_text(content)
            elif content_type == "image":
                return await self._validate_image(content)
            elif content_type == "audio":
                return await self._validate_audio(content)
            elif content_type == "video":
                return await self._validate_video(content)
            else:
                return DataValidationResult(
                    is_valid=False,
                    errors=[f"Unsupported content type: {content_type}"],
                    warnings=[],
                    metadata={}
                )
        
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return DataValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                metadata={}
            )
    
    async def _validate_text(self, text: Union[str, bytes]) -> DataValidationResult:
        """Validate text content."""
        errors = []
        warnings = []
        metadata = {}
        
        # Convert bytes to string if needed
        if isinstance(text, bytes):
            try:
                text = text.decode('utf-8')
            except UnicodeDecodeError:
                errors.append("Invalid UTF-8 encoding")
                return DataValidationResult(False, errors, warnings, metadata)
        
        if not isinstance(text, str):
            errors.append("Text content must be string or bytes")
            return DataValidationResult(False, errors, warnings, metadata)
        
        # Length validation
        text_length = len(text)
        metadata["length"] = text_length
        
        if text_length == 0:
            errors.append("Empty text content")
        elif text_length > self.limits["max_text_length"]:
            errors.append(f"Text too long: {text_length} > {self.limits['max_text_length']}")
        
        # Character validation
        non_printable_count = sum(1 for c in text if not c.isprintable() and c not in ['\n', '\r', '\t'])
        if non_printable_count > 0:
            warnings.append(f"Contains {non_printable_count} non-printable characters")
        
        # Language detection (basic)
        try:
            ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text) if text else 0
            metadata["ascii_ratio"] = ascii_ratio
            
            if ascii_ratio < 0.7:
                warnings.append("Text contains significant non-ASCII characters")
        except:
            pass
        
        # Check for common issues
        if text.count('\x00') > 0:
            errors.append("Text contains null bytes")
        
        if len(text.strip()) == 0:
            warnings.append("Text is only whitespace")
        
        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    async def _validate_image(self, image: Union[str, np.ndarray, Image.Image]) -> DataValidationResult:
        """Validate image content."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Load image if it's a file path
            if isinstance(image, str):
                if not Path(image).exists():
                    errors.append(f"Image file not found: {image}")
                    return DataValidationResult(False, errors, warnings, metadata)
                
                # Check file extension
                file_ext = Path(image).suffix.lower()
                if file_ext not in self.limits["supported_image_formats"]:
                    errors.append(f"Unsupported image format: {file_ext}")
                
                # Check file size
                file_size = Path(image).stat().st_size
                metadata["file_size"] = file_size
                
                if file_size > self.limits["max_file_size"]:
                    errors.append(f"Image file too large: {file_size} bytes")
                
                # Load image
                try:
                    pil_image = Image.open(image)
                except Exception as e:
                    errors.append(f"Cannot open image: {e}")
                    return DataValidationResult(False, errors, warnings, metadata)
            
            elif isinstance(image, np.ndarray):
                try:
                    pil_image = Image.fromarray(image)
                except Exception as e:
                    errors.append(f"Cannot convert array to image: {e}")
                    return DataValidationResult(False, errors, warnings, metadata)
            
            elif isinstance(image, Image.Image):
                pil_image = image
            
            else:
                errors.append(f"Unsupported image type: {type(image)}")
                return DataValidationResult(False, errors, warnings, metadata)
            
            # Validate image properties
            width, height = pil_image.size
            metadata.update({
                "width": width,
                "height": height,
                "mode": pil_image.mode,
                "format": pil_image.format
            })
            
            # Size validation
            if width < self.limits["min_image_size"][0] or height < self.limits["min_image_size"][1]:
                errors.append(f"Image too small: {width}x{height}")
            
            if width > self.limits["max_image_size"][0] or height > self.limits["max_image_size"][1]:
                warnings.append(f"Large image size: {width}x{height}")
            
            # Format validation
            if pil_image.mode not in ["RGB", "RGBA", "L", "P"]:
                warnings.append(f"Unusual color mode: {pil_image.mode}")
            
            # Check for corruption
            try:
                pil_image.verify()
            except Exception as e:
                errors.append(f"Image appears corrupted: {e}")
            
        except Exception as e:
            errors.append(f"Image validation error: {e}")
        
        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    async def _validate_audio(self, audio: Union[str, np.ndarray]) -> DataValidationResult:
        """Validate audio content."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Load audio if it's a file path
            if isinstance(audio, str):
                if not Path(audio).exists():
                    errors.append(f"Audio file not found: {audio}")
                    return DataValidationResult(False, errors, warnings, metadata)
                
                # Check file extension
                file_ext = Path(audio).suffix.lower()
                if file_ext not in self.limits["supported_audio_formats"]:
                    errors.append(f"Unsupported audio format: {file_ext}")
                
                # Check file size
                file_size = Path(audio).stat().st_size
                metadata["file_size"] = file_size
                
                if file_size > self.limits["max_file_size"]:
                    errors.append(f"Audio file too large: {file_size} bytes")
                
                # Load audio
                try:
                    audio_data, sample_rate = librosa.load(audio, sr=None)
                    metadata.update({
                        "sample_rate": sample_rate,
                        "duration": len(audio_data) / sample_rate,
                        "channels": 1 if audio_data.ndim == 1 else audio_data.shape[0]
                    })
                except Exception as e:
                    errors.append(f"Cannot load audio: {e}")
                    return DataValidationResult(False, errors, warnings, metadata)
            
            elif isinstance(audio, np.ndarray):
                audio_data = audio
                metadata.update({
                    "sample_rate": 22050,  # Default assumption
                    "duration": len(audio_data) / 22050,
                    "channels": 1 if audio_data.ndim == 1 else audio_data.shape[0]
                })
            
            else:
                errors.append(f"Unsupported audio type: {type(audio)}")
                return DataValidationResult(False, errors, warnings, metadata)
            
            # Duration validation
            duration = metadata.get("duration", 0)
            if duration > self.limits["max_audio_duration"]:
                errors.append(f"Audio too long: {duration:.1f}s")
            
            if duration == 0:
                errors.append("Audio has zero duration")
            
            # Quality checks
            if np.max(np.abs(audio_data)) == 0:
                errors.append("Audio contains only silence")
            
            # Check for clipping
            clipping_ratio = np.sum(np.abs(audio_data) > 0.99) / len(audio_data)
            if clipping_ratio > 0.01:  # More than 1% clipped
                warnings.append(f"Audio may be clipped: {clipping_ratio:.1%}")
            
        except Exception as e:
            errors.append(f"Audio validation error: {e}")
        
        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    async def _validate_video(self, video: Union[str, np.ndarray]) -> DataValidationResult:
        """Validate video content."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Currently only support file paths for video
            if isinstance(video, str):
                if not Path(video).exists():
                    errors.append(f"Video file not found: {video}")
                    return DataValidationResult(False, errors, warnings, metadata)
                
                # Check file extension
                file_ext = Path(video).suffix.lower()
                if file_ext not in self.limits["supported_video_formats"]:
                    errors.append(f"Unsupported video format: {file_ext}")
                
                # Check file size
                file_size = Path(video).stat().st_size
                metadata["file_size"] = file_size
                
                if file_size > self.limits["max_file_size"]:
                    warnings.append(f"Large video file: {file_size} bytes")
                
                # Load video metadata
                try:
                    video_clip = VideoFileClip(video)
                    metadata.update({
                        "duration": video_clip.duration,
                        "fps": video_clip.fps,
                        "size": video_clip.size,
                        "has_audio": video_clip.audio is not None
                    })
                    video_clip.close()
                except Exception as e:
                    errors.append(f"Cannot load video: {e}")
                    return DataValidationResult(False, errors, warnings, metadata)
            
            else:
                errors.append(f"Unsupported video type: {type(video)}")
                return DataValidationResult(False, errors, warnings, metadata)
            
            # Duration validation
            duration = metadata.get("duration", 0)
            if duration > self.limits["max_video_duration"]:
                errors.append(f"Video too long: {duration:.1f}s")
            
            if duration == 0:
                errors.append("Video has zero duration")
            
            # Quality checks
            fps = metadata.get("fps", 0)
            if fps < 1:
                errors.append("Invalid frame rate")
            elif fps < 10:
                warnings.append(f"Low frame rate: {fps}")
            
            size = metadata.get("size", [0, 0])
            if size[0] < 64 or size[1] < 64:
                errors.append(f"Video resolution too low: {size[0]}x{size[1]}")
        
        except Exception as e:
            errors.append(f"Video validation error: {e}")
        
        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

class ContentPreprocessor:
    """
    Content preprocessing for multimodal ethics evaluation.
    
    Handles normalization, formatting, and optimization of content
    before ethics analysis.
    """
    
    def __init__(self, config):
        """Initialize content preprocessor."""
        self.config = config
        
        # Preprocessing settings
        self.text_settings = {
            "max_length": 10000,
            "normalize_unicode": True,
            "remove_control_chars": True,
            "preserve_formatting": False
        }
        
        self.image_settings = {
            "target_size": (224, 224),
            "normalize": True,
            "convert_mode": "RGB",
            "quality_threshold": 0.5
        }
        
        self.audio_settings = {
            "target_sr": 16000,
            "normalize_volume": True,
            "remove_silence": True,
            "max_duration": 300  # 5 minutes
        }
        
        self.video_settings = {
            "target_fps": 1.0,  # Sample rate for frame extraction
            "max_frames": 30,
            "extract_audio": True,
            "quality_filter": True
        }
        
        logger.info("Content Preprocessor initialized")
    
    async def preprocess_content(
        self, 
        content: Any, 
        content_type: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Preprocess content for ethics evaluation.
        
        Args:
            content: Content to preprocess
            content_type: Type of content
            settings: Optional preprocessing settings
        
        Returns:
            ProcessingResult with processed content
        """
        start_time = datetime.now()
        
        try:
            if content_type == "text":
                result = await self._preprocess_text(content, settings)
            elif content_type == "image":
                result = await self._preprocess_image(content, settings)
            elif content_type == "audio":
                result = await self._preprocess_audio(content, settings)
            elif content_type == "video":
                result = await self._preprocess_video(content, settings)
            else:
                return ProcessingResult(
                    success=False,
                    data=None,
                    processing_time=0.0,
                    metadata={},
                    errors=[f"Unsupported content type: {content_type}"]
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            return result
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Preprocessing error: {e}")
            
            return ProcessingResult(
                success=False,
                data=None,
                processing_time=processing_time,
                metadata={},
                errors=[f"Preprocessing failed: {str(e)}"]
            )
    
    async def _preprocess_text(
        self, 
        text: Union[str, bytes], 
        settings: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Preprocess text content."""
        
        settings = {**self.text_settings, **(settings or {})}
        errors = []
        metadata = {}
        
        # Convert to string if bytes
        if isinstance(text, bytes):
            try:
                text = text.decode('utf-8')
            except UnicodeDecodeError as e:
                return ProcessingResult(
                    success=False,
                    data=None,
                    processing_time=0.0,
                    metadata={},
                    errors=[f"Encoding error: {e}"]
                )
        
        original_length = len(text)
        metadata["original_length"] = original_length
        
        # Normalize unicode
        if settings["normalize_unicode"]:
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters
        if settings["remove_control_chars"]:
            text = ''.join(char for char in text if not unicodedata.category(char).startswith('C'))
        
        # Truncate if too long
        if len(text) > settings["max_length"]:
            text = text[:settings["max_length"]]
            metadata["truncated"] = True
        
        # Clean whitespace
        if not settings["preserve_formatting"]:
            # Normalize whitespace
            text = ' '.join(text.split())
        
        metadata.update({
            "final_length": len(text),
            "character_count": len(text),
            "word_count": len(text.split()),
            "line_count": text.count('\n') + 1
        })
        
        return ProcessingResult(
            success=True,
            data=text,
            processing_time=0.0,  # Will be set by caller
            metadata=metadata,
            errors=errors
        )
    
    async def _preprocess_image(
        self, 
        image: Union[str, np.ndarray, Image.Image], 
        settings: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Preprocess image content."""
        
        settings = {**self.image_settings, **(settings or {})}
        errors = []
        metadata = {}
        
        # Load image
        if isinstance(image, str):
            try:
                pil_image = Image.open(image)
                metadata["source"] = "file"
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    data=None,
                    processing_time=0.0,
                    metadata={},
                    errors=[f"Cannot load image: {e}"]
                )
        elif isinstance(image, np.ndarray):
            try:
                pil_image = Image.fromarray(image)
                metadata["source"] = "array"
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    data=None,
                    processing_time=0.0,
                    metadata={},
                    errors=[f"Cannot convert array: {e}"]
                )
        elif isinstance(image, Image.Image):
            pil_image = image
            metadata["source"] = "pil"
        else:
            return ProcessingResult(
                success=False,
                data=None,
                processing_time=0.0,
                metadata={},
                errors=[f"Unsupported image type: {type(image)}"]
            )
        
        original_size = pil_image.size
        metadata["original_size"] = original_size
        metadata["original_mode"] = pil_image.mode
        
        # Convert color mode
        if pil_image.mode != settings["convert_mode"]:
            pil_image = pil_image.convert(settings["convert_mode"])
        
        # Resize image
        target_size = settings["target_size"]
        if original_size != target_size:
            pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            metadata["resized"] = True
        
        # Convert to array for processing
        image_array = np.array(pil_image)
        
        # Normalize if requested
        if settings["normalize"]:
            image_array = image_array.astype(np.float32) / 255.0
            metadata["normalized"] = True
        
        metadata.update({
            "final_size": target_size,
            "final_mode": settings["convert_mode"],
            "array_shape": image_array.shape,
            "array_dtype": str(image_array.dtype)
        })
        
        return ProcessingResult(
            success=True,
            data=image_array,
            processing_time=0.0,
            metadata=metadata,
            errors=errors
        )
    
    async def _preprocess_audio(
        self, 
        audio: Union[str, np.ndarray], 
        settings: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Preprocess audio content."""
        
        settings = {**self.audio_settings, **(settings or {})}
        errors = []
        metadata = {}
        
        # Load audio
        if isinstance(audio, str):
            try:
                audio_data, sample_rate = librosa.load(audio, sr=None)
                metadata["source"] = "file"
                metadata["original_sr"] = sample_rate
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    data=None,
                    processing_time=0.0,
                    metadata={},
                    errors=[f"Cannot load audio: {e}"]
                )
        elif isinstance(audio, np.ndarray):
            audio_data = audio
            sample_rate = settings["target_sr"]  # Assume target sample rate
            metadata["source"] = "array"
        else:
            return ProcessingResult(
                success=False,
                data=None,
                processing_time=0.0,
                metadata={},
                errors=[f"Unsupported audio type: {type(audio)}"]
            )
        
        original_duration = len(audio_data) / sample_rate
        metadata["original_duration"] = original_duration
        
        # Resample if needed
        target_sr = settings["target_sr"]
        if sample_rate != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
            metadata["resampled"] = True
        
        # Trim silence
        if settings["remove_silence"]:
            audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
            metadata["silence_trimmed"] = True
        
        # Normalize volume
        if settings["normalize_volume"]:
            audio_data = librosa.util.normalize(audio_data)
            metadata["volume_normalized"] = True
        
        # Truncate if too long
        max_samples = int(settings["max_duration"] * sample_rate)
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
            metadata["truncated"] = True
        
        final_duration = len(audio_data) / sample_rate
        metadata.update({
            "final_duration": final_duration,
            "final_sr": sample_rate,
            "final_samples": len(audio_data)
        })
        
        return ProcessingResult(
            success=True,
            data=audio_data,
            processing_time=0.0,
            metadata=metadata,
            errors=errors
        )
    
    async def _preprocess_video(
        self, 
        video: Union[str, np.ndarray], 
        settings: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Preprocess video content."""
        
        settings = {**self.video_settings, **(settings or {})}
        errors = []
        metadata = {}
        
        # Currently only support file paths
        if not isinstance(video, str):
            return ProcessingResult(
                success=False,
                data=None,
                processing_time=0.0,
                metadata={},
                errors=["Video preprocessing only supports file paths"]
            )
        
        try:
            video_clip = VideoFileClip(video)
            
            metadata.update({
                "original_duration": video_clip.duration,
                "original_fps": video_clip.fps,
                "original_size": video_clip.size,
                "has_audio": video_clip.audio is not None
            })
            
            # Extract frames at target FPS
            target_fps = settings["target_fps"]
            max_frames = settings["max_frames"]
            
            frame_times = np.arange(0, min(video_clip.duration, max_frames / target_fps), 1.0 / target_fps)
            frames = []
            
            for t in frame_times[:max_frames]:
                try:
                    frame = video_clip.get_frame(t)
                    frames.append(frame)
                except:
                    continue  # Skip problematic frames
            
            # Extract audio if requested
            audio_data = None
            if settings["extract_audio"] and video_clip.audio:
                try:
                    audio_array = video_clip.audio.to_soundarray()
                    if len(audio_array.shape) > 1:
                        audio_data = np.mean(audio_array, axis=1)  # Convert to mono
                    else:
                        audio_data = audio_array
                except:
                    errors.append("Audio extraction failed")
            
            video_clip.close()
            
            processed_data = {
                "frames": frames,
                "frame_times": frame_times[:len(frames)],
                "audio": audio_data
            }
            
            metadata.update({
                "extracted_frames": len(frames),
                "frame_sampling_rate": target_fps,
                "audio_extracted": audio_data is not None
            })
            
            return ProcessingResult(
                success=True,
                data=processed_data,
                processing_time=0.0,
                metadata=metadata,
                errors=errors
            )
        
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=None,
                processing_time=0.0,
                metadata={},
                errors=[f"Video preprocessing failed: {e}"]
            )

class DataProcessor:
    """
    Main data processing coordinator.
    
    Combines validation and preprocessing for complete
    data pipeline management.
    """
    
    def __init__(self, config):
        """Initialize data processor."""
        self.config = config
        self.validator = DataValidator(config)
        self.preprocessor = ContentPreprocessor(config)
        
        # Processing cache
        self.cache = {}
        self.cache_enabled = getattr(config, 'cache_enabled', True)
        self.cache_ttl = getattr(config, 'cache_ttl', 3600)
        
        logger.info("Data Processor initialized")
    
    async def process_content(
        self, 
        content: Any, 
        content_type: str,
        validate: bool = True,
        preprocess: bool = True,
        cache_key: Optional[str] = None
    ) -> Tuple[bool, Any, Dict[str, Any]]:
        """
        Complete content processing pipeline.
        
        Args:
            content: Content to process
            content_type: Type of content
            validate: Whether to validate content
            preprocess: Whether to preprocess content  
            cache_key: Optional cache key
        
        Returns:
            Tuple of (success, processed_data, metadata)
        """
        metadata = {"pipeline_start": datetime.now().isoformat()}
        
        # Check cache first
        if cache_key and self.cache_enabled and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if (datetime.now() - cache_entry["timestamp"]).total_seconds() < self.cache_ttl:
                metadata["cache_hit"] = True
                return True, cache_entry["data"], {**cache_entry["metadata"], **metadata}
        
        try:
            # Validation step
            if validate:
                validation_result = await self.validator.validate_content(content, content_type)
                metadata["validation"] = {
                    "is_valid": validation_result.is_valid,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                    "validation_metadata": validation_result.metadata
                }
                
                if not validation_result.is_valid:
                    return False, None, metadata
            
            # Preprocessing step
            processed_content = content
            if preprocess:
                processing_result = await self.preprocessor.preprocess_content(content, content_type)
                metadata["preprocessing"] = {
                    "success": processing_result.success,
                    "processing_time": processing_result.processing_time,
                    "errors": processing_result.errors or [],
                    "preprocessing_metadata": processing_result.metadata
                }
                
                if not processing_result.success:
                    return False, None, metadata
                
                processed_content = processing_result.data
            
            # Cache result if enabled
            if cache_key and self.cache_enabled:
                self.cache[cache_key] = {
                    "data": processed_content,
                    "metadata": metadata,
                    "timestamp": datetime.now()
                }
            
            metadata["pipeline_end"] = datetime.now().isoformat()
            return True, processed_content, metadata
        
        except Exception as e:
            logger.error(f"Content processing failed: {e}")
            metadata["error"] = str(e)
            return False, None, metadata
    
    def get_content_hash(self, content: Any) -> str:
        """Generate hash for content caching."""
        try:
            if isinstance(content, str):
                content_bytes = content.encode('utf-8')
            elif isinstance(content, bytes):
                content_bytes = content
            elif isinstance(content, np.ndarray):
                content_bytes = content.tobytes()
            elif hasattr(content, '__dict__'):
                content_bytes = json.dumps(content.__dict__, sort_keys=True, default=str).encode('utf-8')
            else:
                content_bytes = str(content).encode('utf-8')
            
            return hashlib.sha256(content_bytes).hexdigest()[:16]
        except:
            return str(hash(str(content)))[:16]
    
    def clear_cache(self) -> None:
        """Clear the processing cache."""
        self.cache.clear()
        logger.info("Processing cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.cache),
            "cache_ttl": self.cache_ttl,
            "memory_usage_estimate": sum(
                len(str(entry)) for entry in self.cache.values()
            )
        }