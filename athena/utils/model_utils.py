"""
Model Management Utilities for Project Athena

Advanced model loading, caching, optimization, and performance
management for multimodal ethics evaluation models.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
import threading
from pathlib import Path
import time
import gc
import psutil

# ML framework imports
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    import numpy as np
    from torch.utils.data import DataLoader
    import accelerate
    from accelerate import Accelerator
except ImportError as e:
    logging.warning(f"Some ML framework dependencies not available: {e}")

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_id: str
    model_type: str
    framework: str
    size_mb: float
    load_time: float
    last_used: datetime
    usage_count: int = 0
    device: str = "cpu"
    precision: str = "float32"

@dataclass
class PerformanceMetrics:
    """Performance metrics for model operations."""
    inference_time: float
    memory_usage: float
    gpu_utilization: float
    batch_size: int
    throughput: float  # samples per second
    timestamp: datetime = field(default_factory=datetime.now)

class ModelCache:
    """
    Intelligent model caching system.
    
    Manages memory-efficient loading, caching, and eviction
    of ML models with LRU and size-based policies.
    """
    
    def __init__(self, max_memory_mb: int = 8192, max_models: int = 10):
        """
        Initialize model cache.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_models: Maximum number of cached models
        """
        self.max_memory_mb = max_memory_mb
        self.max_models = max_models
        
        self.models = {}  # model_id -> model
        self.model_info = {}  # model_id -> ModelInfo
        self.lock = threading.RLock()
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        
        logger.info(f"Model cache initialized: {max_memory_mb}MB, {max_models} models")
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model from cache."""
        with self.lock:
            if model_id in self.models:
                # Update access time and usage count
                self.model_info[model_id].last_used = datetime.now()
                self.model_info[model_id].usage_count += 1
                self.cache_hits += 1
                return self.models[model_id]
            else:
                self.cache_misses += 1
                return None
    
    def put_model(self, model_id: str, model: Any, model_info: ModelInfo) -> bool:
        """Put model in cache with eviction if necessary."""
        with self.lock:
            # Check if we need to evict models
            while self._should_evict(model_info.size_mb):
                if not self._evict_least_recently_used():
                    logger.warning("Cannot evict models to make space")
                    return False
            
            # Store model and info
            self.models[model_id] = model
            self.model_info[model_id] = model_info
            
            logger.info(f"Cached model {model_id} ({model_info.size_mb:.1f}MB)")
            return True
    
    def _should_evict(self, new_model_size: float) -> bool:
        """Check if eviction is needed."""
        current_memory = sum(info.size_mb for info in self.model_info.values())
        
        return (
            len(self.models) >= self.max_models or
            current_memory + new_model_size > self.max_memory_mb
        )
    
    def _evict_least_recently_used(self) -> bool:
        """Evict the least recently used model."""
        if not self.model_info:
            return False
        
        # Find LRU model
        lru_model_id = min(
            self.model_info.keys(),
            key=lambda k: self.model_info[k].last_used
        )
        
        # Remove model
        del self.models[lru_model_id]
        del self.model_info[lru_model_id]
        self.evictions += 1
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Evicted model {lru_model_id}")
        return True
    
    def remove_model(self, model_id: str) -> bool:
        """Manually remove model from cache."""
        with self.lock:
            if model_id in self.models:
                del self.models[model_id]
                del self.model_info[model_id]
                
                # Force cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return True
            return False
    
    def clear_cache(self) -> None:
        """Clear entire cache."""
        with self.lock:
            self.models.clear()
            self.model_info.clear()
            
            # Force cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("Model cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            current_memory = sum(info.size_mb for info in self.model_info.values())
            
            return {
                "total_models": len(self.models),
                "memory_usage_mb": current_memory,
                "memory_limit_mb": self.max_memory_mb,
                "memory_utilization": current_memory / self.max_memory_mb if self.max_memory_mb > 0 else 0,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                "evictions": self.evictions,
                "models": [
                    {
                        "model_id": model_id,
                        "type": info.model_type,
                        "size_mb": info.size_mb,
                        "usage_count": info.usage_count,
                        "last_used": info.last_used.isoformat()
                    }
                    for model_id, info in self.model_info.items()
                ]
            }

class PerformanceOptimizer:
    """
    Model performance optimization system.
    
    Handles quantization, pruning, distillation, and other
    optimization techniques for efficient inference.
    """
    
    def __init__(self, config):
        """Initialize performance optimizer."""
        self.config = config
        self.optimization_cache = {}
        self.performance_history = []
        
        # Optimization settings
        self.optimization_settings = {
            "enable_quantization": True,
            "quantization_mode": "dynamic",  # dynamic, static, qat
            "enable_pruning": False,
            "pruning_sparsity": 0.1,
            "enable_fusion": True,
            "batch_size_optimization": True,
            "mixed_precision": True
        }
        
        logger.info("Performance Optimizer initialized")
    
    async def optimize_model(
        self, 
        model: Any, 
        model_type: str,
        optimization_level: str = "balanced"  # fast, balanced, quality
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Optimize model for inference performance.
        
        Args:
            model: Model to optimize
            model_type: Type of model (text, image, etc.)
            optimization_level: Level of optimization
        
        Returns:
            Tuple of (optimized_model, optimization_info)
        """
        optimization_info = {
            "original_size": self._get_model_size(model),
            "optimizations_applied": [],
            "performance_gain": 0.0
        }
        
        try:
            optimized_model = model
            
            # Apply optimizations based on level
            if optimization_level in ["fast", "balanced"]:
                # Quantization
                if self.optimization_settings["enable_quantization"] and hasattr(torch, 'quantization'):
                    optimized_model = await self._apply_quantization(optimized_model, model_type)
                    optimization_info["optimizations_applied"].append("quantization")
                
                # Model fusion
                if self.optimization_settings["enable_fusion"]:
                    optimized_model = await self._apply_fusion(optimized_model)
                    optimization_info["optimizations_applied"].append("fusion")
            
            if optimization_level == "fast":
                # Aggressive optimizations for maximum speed
                if self.optimization_settings["enable_pruning"]:
                    optimized_model = await self._apply_pruning(optimized_model)
                    optimization_info["optimizations_applied"].append("pruning")
            
            # Mixed precision
            if self.optimization_settings["mixed_precision"] and torch.cuda.is_available():
                optimized_model = await self._apply_mixed_precision(optimized_model)
                optimization_info["optimizations_applied"].append("mixed_precision")
            
            # Calculate final size and performance gain
            optimization_info["optimized_size"] = self._get_model_size(optimized_model)
            optimization_info["size_reduction"] = (
                optimization_info["original_size"] - optimization_info["optimized_size"]
            ) / optimization_info["original_size"]
            
            logger.info(f"Model optimization completed: {len(optimization_info['optimizations_applied'])} optimizations applied")
            
            return optimized_model, optimization_info
        
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model, optimization_info
    
    async def _apply_quantization(self, model: Any, model_type: str) -> Any:
        """Apply quantization to model."""
        try:
            if hasattr(torch.quantization, 'quantize_dynamic'):
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
                return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
        
        return model
    
    async def _apply_fusion(self, model: Any) -> Any:
        """Apply layer fusion optimizations."""
        try:
            if hasattr(torch.jit, 'script'):
                # Try to script the model for fusion
                scripted_model = torch.jit.script(model)
                return scripted_model
        except Exception as e:
            logger.warning(f"Fusion failed: {e}")
        
        return model
    
    async def _apply_pruning(self, model: Any) -> Any:
        """Apply structured pruning to model."""
        try:
            if hasattr(torch.nn.utils, 'prune'):
                # Apply magnitude-based pruning
                for module in model.modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                        torch.nn.utils.prune.l1_unstructured(
                            module, 
                            name='weight', 
                            amount=self.optimization_settings["pruning_sparsity"]
                        )
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")
        
        return model
    
    async def _apply_mixed_precision(self, model: Any) -> Any:
        """Apply mixed precision optimization."""
        try:
            if hasattr(model, 'half'):
                # Convert to half precision
                model = model.half()
        except Exception as e:
            logger.warning(f"Mixed precision failed: {e}")
        
        return model
    
    def _get_model_size(self, model: Any) -> float:
        """Get model size in MB."""
        try:
            if hasattr(model, 'parameters'):
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB
        except:
            pass
        
        return 0.0
    
    async def benchmark_model(
        self, 
        model: Any, 
        sample_input: Any,
        num_runs: int = 100
    ) -> PerformanceMetrics:
        """Benchmark model performance."""
        
        inference_times = []
        memory_usage = []
        
        # Warm up
        for _ in range(10):
            try:
                with torch.no_grad():
                    _ = model(sample_input)
            except:
                break
        
        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            
            try:
                with torch.no_grad():
                    _ = model(sample_input)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Memory usage
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)  # MB
                else:
                    process = psutil.Process()
                    memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
                
            except Exception as e:
                logger.warning(f"Benchmark iteration failed: {e}")
                break
        
        if not inference_times:
            return PerformanceMetrics(0, 0, 0, 1, 0)
        
        avg_inference_time = np.mean(inference_times)
        avg_memory_usage = np.mean(memory_usage) if memory_usage else 0
        
        # Calculate throughput
        batch_size = getattr(sample_input, 'shape', [1])[0] if hasattr(sample_input, 'shape') else 1
        throughput = batch_size / avg_inference_time if avg_inference_time > 0 else 0
        
        return PerformanceMetrics(
            inference_time=avg_inference_time,
            memory_usage=avg_memory_usage,
            gpu_utilization=0.0,  # Would need nvidia-ml-py for actual GPU util
            batch_size=batch_size,
            throughput=throughput
        )

class ModelManager:
    """
    Comprehensive model management system.
    
    Handles loading, caching, optimization, and lifecycle
    management of ML models for multimodal ethics evaluation.
    """
    
    def __init__(self, config):
        """Initialize model manager."""
        self.config = config
        
        # Initialize components
        self.cache = ModelCache(
            max_memory_mb=getattr(config, 'model_cache_size_mb', 8192),
            max_models=getattr(config, 'max_cached_models', 10)
        )
        self.optimizer = PerformanceOptimizer(config)
        
        # Model registry
        self.model_registry = {}
        self.loading_locks = {}
        
        # Performance tracking
        self.performance_metrics = {}
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Model Manager initialized on device: {self.device}")
    
    async def load_model(
        self, 
        model_id: str, 
        model_path: str,
        model_type: str = "auto",
        optimize: bool = True,
        force_reload: bool = False
    ) -> Optional[Any]:
        """
        Load model with caching and optimization.
        
        Args:
            model_id: Unique identifier for model
            model_path: Path to model or HuggingFace model name
            model_type: Type of model (text, image, audio, video, auto)
            optimize: Whether to apply optimizations
            force_reload: Force reload even if cached
        
        Returns:
            Loaded model or None if failed
        """
        # Check cache first
        if not force_reload:
            cached_model = self.cache.get_model(model_id)
            if cached_model is not None:
                logger.info(f"Using cached model: {model_id}")
                return cached_model
        
        # Prevent concurrent loading of same model
        if model_id in self.loading_locks:
            async with self.loading_locks[model_id]:
                # Check cache again after waiting
                cached_model = self.cache.get_model(model_id)
                if cached_model is not None:
                    return cached_model
        else:
            self.loading_locks[model_id] = asyncio.Lock()
        
        async with self.loading_locks[model_id]:
            try:
                start_time = time.time()
                
                # Load model based on type
                model = await self._load_model_by_type(model_path, model_type)
                
                if model is None:
                    return None
                
                # Move to device
                if hasattr(model, 'to'):
                    model = model.to(self.device)
                
                # Optimize if requested
                optimization_info = {}
                if optimize:
                    model, optimization_info = await self.optimizer.optimize_model(
                        model, model_type
                    )
                
                load_time = time.time() - start_time
                model_size = self._estimate_model_size(model)
                
                # Create model info
                model_info = ModelInfo(
                    model_id=model_id,
                    model_type=model_type,
                    framework="pytorch",
                    size_mb=model_size,
                    load_time=load_time,
                    last_used=datetime.now(),
                    device=str(self.device),
                    precision="float16" if optimization_info.get("mixed_precision") else "float32"
                )
                
                # Cache model
                self.cache.put_model(model_id, model, model_info)
                
                # Register model
                self.model_registry[model_id] = {
                    "path": model_path,
                    "type": model_type,
                    "load_time": load_time,
                    "optimization_info": optimization_info
                }
                
                logger.info(f"Model loaded: {model_id} ({model_size:.1f}MB, {load_time:.2f}s)")
                return model
            
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                return None
            
            finally:
                # Clean up loading lock
                if model_id in self.loading_locks:
                    del self.loading_locks[model_id]
    
    async def _load_model_by_type(self, model_path: str, model_type: str) -> Optional[Any]:
        """Load model based on its type."""
        
        try:
            if model_type in ["text", "auto"]:
                # Try loading as transformers model
                try:
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(model_path)
                    return model
                except:
                    pass
            
            if model_type in ["image", "auto"]:
                # Try loading as vision model
                try:
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(model_path)
                    return model
                except:
                    pass
            
            # Try loading as generic PyTorch model
            if Path(model_path).exists():
                model = torch.load(model_path, map_location=self.device)
                return model
            
            logger.error(f"Could not load model from {model_path}")
            return None
        
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return None
    
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB."""
        return self.optimizer._get_model_size(model)
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload model from cache."""
        success = self.cache.remove_model(model_id)
        
        if success and model_id in self.model_registry:
            del self.model_registry[model_id]
        
        return success
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model IDs."""
        return list(self.cache.model_info.keys())
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.cache.model_info.get(model_id)
    
    async def benchmark_model(
        self, 
        model_id: str, 
        sample_input: Any,
        num_runs: int = 100
    ) -> Optional[PerformanceMetrics]:
        """Benchmark a specific model."""
        
        model = self.cache.get_model(model_id)
        if model is None:
            logger.error(f"Model not found: {model_id}")
            return None
        
        metrics = await self.optimizer.benchmark_model(model, sample_input, num_runs)
        
        # Store metrics
        self.performance_metrics[model_id] = metrics
        
        return metrics
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system and model statistics."""
        
        stats = {
            "cache_stats": self.cache.get_cache_stats(),
            "device": str(self.device),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "total_models_registered": len(self.model_registry),
            "performance_metrics": {
                model_id: {
                    "inference_time": metrics.inference_time,
                    "memory_usage": metrics.memory_usage,
                    "throughput": metrics.throughput
                }
                for model_id, metrics in self.performance_metrics.items()
            }
        }
        
        # Add GPU stats if available
        if torch.cuda.is_available():
            stats["gpu_stats"] = {
                "memory_allocated": torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                "memory_reserved": torch.cuda.memory_reserved() / 1024 / 1024,   # MB
                "device_name": torch.cuda.get_device_name(0)
            }
        
        return stats
    
    async def cleanup(self) -> None:
        """Clean up resources and clear caches."""
        self.cache.clear_cache()
        self.model_registry.clear()
        self.performance_metrics.clear()
        self.loading_locks.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model Manager cleanup completed")