"""
Configuration management for Project Athena Multimodal Ethics Framework

Handles all configuration aspects including ethics thresholds, model parameters,
Meta integration settings, and multimodal processing configurations.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModalityConfig:
    """Configuration for individual modality processors."""
    enabled: bool = True
    model_name: str = ""
    threshold: float = 0.8
    batch_size: int = 32
    max_input_size: int = 1024
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    postprocessing: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetaIntegrationConfig:
    """Configuration for Meta AI ecosystem integration."""
    llama_models: List[str] = field(default_factory=lambda: ["llama-7b", "llama-13b", "llama-70b"])
    api_endpoint: str = "https://api.meta.ai/v1"
    api_key: Optional[str] = None
    rate_limit: int = 1000
    timeout: int = 30
    retry_attempts: int = 3
    use_pytorch_integration: bool = True
    enable_fair_research_mode: bool = False

@dataclass
class EthicsThresholds:
    """Ethical evaluation thresholds and scoring parameters."""
    harmful_content: float = 0.9
    bias_detection: float = 0.8
    privacy_violation: float = 0.95
    misinformation: float = 0.85
    toxicity: float = 0.8
    hate_speech: float = 0.9
    violence: float = 0.95
    sexual_content: float = 0.9
    child_safety: float = 0.99
    copyright_infringement: float = 0.85

@dataclass
class RLHFConfig:
    """Reinforcement Learning from Human Feedback configuration."""
    enabled: bool = True
    model_path: str = ""
    learning_rate: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 3
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    use_wandb: bool = True
    checkpoint_interval: int = 100

@dataclass
class ConstitutionalAIConfig:
    """Constitutional AI framework configuration."""
    enabled: bool = True
    constitution_path: str = ""
    principles: List[str] = field(default_factory=lambda: [
        "helpfulness", "harmlessness", "honesty", "transparency", "fairness"
    ])
    reasoning_model: str = "meta-llama/Llama-2-70b-chat-hf"
    max_reasoning_steps: int = 5
    confidence_threshold: float = 0.8

class EthicsConfig:
    """
    Main configuration class for Project Athena Multimodal Ethics Framework.
    
    Manages all configuration aspects including modality settings, Meta integration,
    ethics thresholds, and framework parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        
        # Initialize default configurations
        self.text = ModalityConfig(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            threshold=0.8,
            batch_size=32,
            max_input_size=2048
        )
        
        self.image = ModalityConfig(
            model_name="meta-cv/clip-vit-large-patch14",
            threshold=0.85,
            batch_size=16,
            max_input_size=224
        )
        
        self.audio = ModalityConfig(
            model_name="meta-audio/wav2vec2-base",
            threshold=0.8,
            batch_size=8,
            max_input_size=16000
        )
        
        self.video = ModalityConfig(
            model_name="meta-video/videomae-base",
            threshold=0.85,
            batch_size=4,
            max_input_size=16
        )
        
        self.meta_integration = MetaIntegrationConfig()
        self.ethics_thresholds = EthicsThresholds()
        self.rlhf = RLHFConfig()
        self.constitutional_ai = ConstitutionalAIConfig()
        
        # General configuration
        self.logging_level = "INFO"
        self.cache_enabled = True
        self.cache_ttl = 3600
        self.monitoring_enabled = True
        self.metrics_port = 8080
        self.api_port = 8000
        
        # Load configuration from file if provided
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            self._update_from_dict(config_data)
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            config_path: Path to save configuration file
        """
        try:
            config_data = self.to_dict()
            
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if 'text' in config_data:
            self._update_modality_config(self.text, config_data['text'])
        
        if 'image' in config_data:
            self._update_modality_config(self.image, config_data['image'])
        
        if 'audio' in config_data:
            self._update_modality_config(self.audio, config_data['audio'])
        
        if 'video' in config_data:
            self._update_modality_config(self.video, config_data['video'])
        
        if 'meta_integration' in config_data:
            self._update_meta_integration(config_data['meta_integration'])
        
        if 'ethics_thresholds' in config_data:
            self._update_ethics_thresholds(config_data['ethics_thresholds'])
        
        if 'rlhf' in config_data:
            self._update_rlhf_config(config_data['rlhf'])
        
        if 'constitutional_ai' in config_data:
            self._update_constitutional_ai(config_data['constitutional_ai'])
        
        # Update general settings
        for key in ['logging_level', 'cache_enabled', 'cache_ttl', 
                   'monitoring_enabled', 'metrics_port', 'api_port']:
            if key in config_data:
                setattr(self, key, config_data[key])
    
    def _update_modality_config(self, config: ModalityConfig, data: Dict[str, Any]) -> None:
        """Update modality configuration from dictionary."""
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    def _update_meta_integration(self, data: Dict[str, Any]) -> None:
        """Update Meta integration configuration."""
        for key, value in data.items():
            if hasattr(self.meta_integration, key):
                setattr(self.meta_integration, key, value)
    
    def _update_ethics_thresholds(self, data: Dict[str, Any]) -> None:
        """Update ethics thresholds configuration."""
        for key, value in data.items():
            if hasattr(self.ethics_thresholds, key):
                setattr(self.ethics_thresholds, key, value)
    
    def _update_rlhf_config(self, data: Dict[str, Any]) -> None:
        """Update RLHF configuration."""
        for key, value in data.items():
            if hasattr(self.rlhf, key):
                setattr(self.rlhf, key, value)
    
    def _update_constitutional_ai(self, data: Dict[str, Any]) -> None:
        """Update Constitutional AI configuration."""
        for key, value in data.items():
            if hasattr(self.constitutional_ai, key):
                setattr(self.constitutional_ai, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'text': {
                'enabled': self.text.enabled,
                'model_name': self.text.model_name,
                'threshold': self.text.threshold,
                'batch_size': self.text.batch_size,
                'max_input_size': self.text.max_input_size,
                'preprocessing': self.text.preprocessing,
                'postprocessing': self.text.postprocessing,
            },
            'image': {
                'enabled': self.image.enabled,
                'model_name': self.image.model_name,
                'threshold': self.image.threshold,
                'batch_size': self.image.batch_size,
                'max_input_size': self.image.max_input_size,
                'preprocessing': self.image.preprocessing,
                'postprocessing': self.image.postprocessing,
            },
            'audio': {
                'enabled': self.audio.enabled,
                'model_name': self.audio.model_name,
                'threshold': self.audio.threshold,
                'batch_size': self.audio.batch_size,
                'max_input_size': self.audio.max_input_size,
                'preprocessing': self.audio.preprocessing,
                'postprocessing': self.audio.postprocessing,
            },
            'video': {
                'enabled': self.video.enabled,
                'model_name': self.video.model_name,
                'threshold': self.video.threshold,
                'batch_size': self.video.batch_size,
                'max_input_size': self.video.max_input_size,
                'preprocessing': self.video.preprocessing,
                'postprocessing': self.video.postprocessing,
            },
            'meta_integration': {
                'llama_models': self.meta_integration.llama_models,
                'api_endpoint': self.meta_integration.api_endpoint,
                'rate_limit': self.meta_integration.rate_limit,
                'timeout': self.meta_integration.timeout,
                'retry_attempts': self.meta_integration.retry_attempts,
                'use_pytorch_integration': self.meta_integration.use_pytorch_integration,
                'enable_fair_research_mode': self.meta_integration.enable_fair_research_mode,
            },
            'ethics_thresholds': {
                'harmful_content': self.ethics_thresholds.harmful_content,
                'bias_detection': self.ethics_thresholds.bias_detection,
                'privacy_violation': self.ethics_thresholds.privacy_violation,
                'misinformation': self.ethics_thresholds.misinformation,
                'toxicity': self.ethics_thresholds.toxicity,
                'hate_speech': self.ethics_thresholds.hate_speech,
                'violence': self.ethics_thresholds.violence,
                'sexual_content': self.ethics_thresholds.sexual_content,
                'child_safety': self.ethics_thresholds.child_safety,
                'copyright_infringement': self.ethics_thresholds.copyright_infringement,
            },
            'rlhf': {
                'enabled': self.rlhf.enabled,
                'model_path': self.rlhf.model_path,
                'learning_rate': self.rlhf.learning_rate,
                'batch_size': self.rlhf.batch_size,
                'num_epochs': self.rlhf.num_epochs,
                'max_length': self.rlhf.max_length,
                'temperature': self.rlhf.temperature,
                'top_p': self.rlhf.top_p,
                'use_wandb': self.rlhf.use_wandb,
                'checkpoint_interval': self.rlhf.checkpoint_interval,
            },
            'constitutional_ai': {
                'enabled': self.constitutional_ai.enabled,
                'constitution_path': self.constitutional_ai.constitution_path,
                'principles': self.constitutional_ai.principles,
                'reasoning_model': self.constitutional_ai.reasoning_model,
                'max_reasoning_steps': self.constitutional_ai.max_reasoning_steps,
                'confidence_threshold': self.constitutional_ai.confidence_threshold,
            },
            'logging_level': self.logging_level,
            'cache_enabled': self.cache_enabled,
            'cache_ttl': self.cache_ttl,
            'monitoring_enabled': self.monitoring_enabled,
            'metrics_port': self.metrics_port,
            'api_port': self.api_port,
        }
    
    def validate(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate thresholds are between 0 and 1
            threshold_attrs = [
                self.text.threshold, self.image.threshold, 
                self.audio.threshold, self.video.threshold
            ]
            
            ethics_threshold_attrs = [
                self.ethics_thresholds.harmful_content,
                self.ethics_thresholds.bias_detection,
                self.ethics_thresholds.privacy_violation,
                self.ethics_thresholds.misinformation,
                self.ethics_thresholds.toxicity,
                self.ethics_thresholds.hate_speech,
                self.ethics_thresholds.violence,
                self.ethics_thresholds.sexual_content,
                self.ethics_thresholds.child_safety,
                self.ethics_thresholds.copyright_infringement,
            ]
            
            all_thresholds = threshold_attrs + ethics_threshold_attrs
            
            for threshold in all_thresholds:
                if not 0 <= threshold <= 1:
                    logger.error(f"Invalid threshold value: {threshold}")
                    return False
            
            # Validate ports are in valid range
            for port in [self.metrics_port, self.api_port]:
                if not 1024 <= port <= 65535:
                    logger.error(f"Invalid port number: {port}")
                    return False
            
            # Validate required model names are set
            if not all([self.text.model_name, self.image.model_name, 
                       self.audio.model_name, self.video.model_name]):
                logger.error("Missing required model names")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    @classmethod
    def load_default_meta_config(cls) -> 'EthicsConfig':
        """
        Load default configuration optimized for Meta AI ecosystem.
        
        Returns:
            EthicsConfig: Default Meta-optimized configuration
        """
        config = cls()
        
        # Optimize for Meta models
        config.text.model_name = "meta-llama/Llama-2-70b-chat-hf"
        config.text.batch_size = 64
        config.text.threshold = 0.85
        
        config.image.model_name = "meta-cv/clip-vit-large-patch14"
        config.image.batch_size = 32
        config.image.threshold = 0.88
        
        config.audio.model_name = "meta-audio/wav2vec2-large-960h"
        config.audio.batch_size = 16
        config.audio.threshold = 0.82
        
        config.video.model_name = "meta-video/videomae-large"
        config.video.batch_size = 8
        config.video.threshold = 0.87
        
        # Enable all Meta integrations
        config.meta_integration.use_pytorch_integration = True
        config.meta_integration.enable_fair_research_mode = True
        config.meta_integration.rate_limit = 10000
        
        # Higher ethics standards for production
        config.ethics_thresholds.child_safety = 0.995
        config.ethics_thresholds.violence = 0.98
        config.ethics_thresholds.harmful_content = 0.95
        
        return config