"""
Main Multimodal Ethics Engine for Project Athena

Central coordination system for ethical evaluation across all modalities,
integrating Meta AI ecosystem components and advanced ethical frameworks.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import json

from .config import EthicsConfig
from .evaluator import EthicsEvaluator, EvaluationResult, ComplianceStatus

logger = logging.getLogger(__name__)

@dataclass
class MultimodalContent:
    """Container for multimodal content to be evaluated."""
    text: Optional[str] = None
    image: Optional[Union[str, np.ndarray]] = None
    audio: Optional[Union[str, np.ndarray]] = None
    video: Optional[Union[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_available_modalities(self) -> List[str]:
        """Get list of available modalities in content."""
        modalities = []
        if self.text is not None:
            modalities.append("text")
        if self.image is not None:
            modalities.append("image")
        if self.audio is not None:
            modalities.append("audio")
        if self.video is not None:
            modalities.append("video")
        return modalities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for processing."""
        result = {}
        if self.text is not None:
            result["text"] = self.text
        if self.image is not None:
            result["image"] = self.image
        if self.audio is not None:
            result["audio"] = self.audio
        if self.video is not None:
            result["video"] = self.video
        result["metadata"] = self.metadata
        return result

@dataclass
class BatchEvaluationResult:
    """Result of batch evaluation across multiple content items."""
    results: List[EvaluationResult] = field(default_factory=list)
    batch_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    
    def get_overall_compliance_rate(self) -> float:
        """Calculate overall compliance rate for the batch."""
        if not self.results:
            return 0.0
        
        compliant_count = sum(
            1 for result in self.results 
            if result.compliance_status == ComplianceStatus.COMPLIANT
        )
        return compliant_count / len(self.results)
    
    def get_critical_violations(self) -> List[EvaluationResult]:
        """Get all results with critical violations."""
        return [
            result for result in self.results
            if result.compliance_status == ComplianceStatus.CRITICAL
        ]

class MultimodalEthicsEngine:
    """
    Main Multimodal Ethics Engine for Project Athena.
    
    Coordinates ethical evaluation across all modalities with advanced
    integration for Meta AI ecosystem, RLHF, and Constitutional AI.
    """
    
    def __init__(self, config: Optional[EthicsConfig] = None):
        """
        Initialize the Multimodal Ethics Engine.
        
        Args:
            config: Ethics configuration. If None, uses default Meta config.
        """
        self.config = config or EthicsConfig.load_default_meta_config()
        self.evaluator = EthicsEvaluator(self.config)
        
        # Initialize modality processors
        self._initialize_modality_processors()
        
        # Initialize framework integrations
        self._initialize_frameworks()
        
        # Initialize monitoring
        self._initialize_monitoring()
        
        # Evaluation statistics
        self.evaluation_count = 0
        self.last_evaluation_time = None
        self.performance_metrics = {
            "total_evaluations": 0,
            "avg_processing_time": 0.0,
            "compliance_rate": 0.0,
            "modality_usage": {"text": 0, "image": 0, "audio": 0, "video": 0}
        }
        
        logger.info("Multimodal Ethics Engine initialized successfully")
    
    def _initialize_modality_processors(self) -> None:
        """Initialize modality-specific processors."""
        from ..modalities.text_ethics import TextEthics
        from ..modalities.image_ethics import ImageEthics
        from ..modalities.audio_ethics import AudioEthics
        from ..modalities.video_ethics import VideoEthics
        
        self.modality_processors = {
            "text": TextEthics(self.config),
            "image": ImageEthics(self.config),
            "audio": AudioEthics(self.config),
            "video": VideoEthics(self.config),
        }
        
        logger.info("Modality processors initialized")
    
    def _initialize_frameworks(self) -> None:
        """Initialize ethical framework integrations."""
        self.frameworks = {}
        
        if self.config.rlhf.enabled:
            from ..frameworks.rlhf_integration import RLHFIntegration
            self.frameworks["rlhf"] = RLHFIntegration(self.config)
        
        if self.config.constitutional_ai.enabled:
            from ..frameworks.constitutional_ai import ConstitutionalAI
            self.frameworks["constitutional_ai"] = ConstitutionalAI(self.config)
        
        logger.info(f"Ethical frameworks initialized: {list(self.frameworks.keys())}")
    
    def _initialize_monitoring(self) -> None:
        """Initialize monitoring and dashboard components."""
        if self.config.monitoring_enabled:
            try:
                from ..monitors.content_monitor import ContentMonitor
                from ..monitors.ethics_dashboard import EthicsDashboard
                
                self.content_monitor = ContentMonitor(self.config)
                self.dashboard = EthicsDashboard(self.config)
                
                logger.info("Monitoring systems initialized")
            except ImportError as e:
                logger.warning(f"Monitoring components not available: {e}")
                self.content_monitor = None
                self.dashboard = None
    
    async def evaluate_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate text content for ethical compliance.
        
        Args:
            text: Text content to evaluate
            metadata: Additional metadata for evaluation
        
        Returns:
            EvaluationResult: Comprehensive evaluation result
        """
        content = MultimodalContent(text=text, metadata=metadata or {})
        return await self.evaluate(content)
    
    async def evaluate_image(
        self, 
        image: Union[str, np.ndarray], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate image content for ethical compliance.
        
        Args:
            image: Image content (file path or array)
            metadata: Additional metadata for evaluation
        
        Returns:
            EvaluationResult: Comprehensive evaluation result
        """
        content = MultimodalContent(image=image, metadata=metadata or {})
        return await self.evaluate(content)
    
    async def evaluate_audio(
        self, 
        audio: Union[str, np.ndarray], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate audio content for ethical compliance.
        
        Args:
            audio: Audio content (file path or array)
            metadata: Additional metadata for evaluation
        
        Returns:
            EvaluationResult: Comprehensive evaluation result
        """
        content = MultimodalContent(audio=audio, metadata=metadata or {})
        return await self.evaluate(content)
    
    async def evaluate_video(
        self, 
        video: Union[str, np.ndarray], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate video content for ethical compliance.
        
        Args:
            video: Video content (file path or array)
            metadata: Additional metadata for evaluation
        
        Returns:
            EvaluationResult: Comprehensive evaluation result
        """
        content = MultimodalContent(video=video, metadata=metadata or {})
        return await self.evaluate(content)
    
    async def evaluate_multimodal(
        self, 
        content_dict: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate multimodal content for ethical compliance.
        
        Args:
            content_dict: Dictionary with modality keys and content values
            metadata: Additional metadata for evaluation
        
        Returns:
            EvaluationResult: Comprehensive evaluation result
        """
        content = MultimodalContent(
            text=content_dict.get("text"),
            image=content_dict.get("image"),
            audio=content_dict.get("audio"),
            video=content_dict.get("video"),
            metadata=metadata or {}
        )
        return await self.evaluate(content)
    
    async def evaluate(self, content: MultimodalContent) -> EvaluationResult:
        """
        Main evaluation method for multimodal content.
        
        Args:
            content: MultimodalContent instance to evaluate
        
        Returns:
            EvaluationResult: Comprehensive evaluation result
        """
        start_time = datetime.now()
        
        try:
            # Update usage statistics
            modalities = content.get_available_modalities()
            for modality in modalities:
                self.performance_metrics["modality_usage"][modality] += 1
            
            # Perform core evaluation
            result = await self.evaluator.evaluate(
                content.to_dict(), 
                modalities
            )
            
            # Apply framework enhancements
            if self.frameworks:
                result = await self._apply_framework_enhancements(result, content)
            
            # Update monitoring
            if self.content_monitor:
                await self.content_monitor.log_evaluation(result, content)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(result, processing_time)
            
            logger.info(
                f"Evaluation completed for {modalities} in {processing_time:.3f}s. "
                f"Score: {result.overall_score:.3f}, Status: {result.compliance_status.value}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # Return a safe default result in case of errors
            return EvaluationResult(
                overall_score=0.0,
                compliance_status=ComplianceStatus.CRITICAL,
                metadata={
                    "error": str(e),
                    "evaluation_failed": True,
                    "timestamp": start_time.isoformat()
                }
            )
    
    async def evaluate_batch(
        self, 
        content_list: List[MultimodalContent],
        max_concurrent: int = 10
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple content items in batch.
        
        Args:
            content_list: List of content items to evaluate
            max_concurrent: Maximum concurrent evaluations
        
        Returns:
            BatchEvaluationResult: Batch evaluation results
        """
        start_time = datetime.now()
        
        # Create semaphore to limit concurrent evaluations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(content: MultimodalContent) -> EvaluationResult:
            async with semaphore:
                return await self.evaluate(content)
        
        # Execute evaluations concurrently
        tasks = [evaluate_with_semaphore(content) for content in content_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch evaluation failed for item {i}: {result}")
                # Create error result
                error_result = EvaluationResult(
                    overall_score=0.0,
                    compliance_status=ComplianceStatus.CRITICAL,
                    metadata={"batch_error": str(result), "item_index": i}
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        batch_result = BatchEvaluationResult(
            results=valid_results,
            processing_time=processing_time,
            batch_metadata={
                "total_items": len(content_list),
                "max_concurrent": max_concurrent,
                "avg_item_time": processing_time / len(content_list) if content_list else 0,
                "timestamp": start_time.isoformat()
            }
        )
        
        logger.info(
            f"Batch evaluation completed: {len(content_list)} items in {processing_time:.3f}s. "
            f"Compliance rate: {batch_result.get_overall_compliance_rate():.1%}"
        )
        
        return batch_result
    
    async def _apply_framework_enhancements(
        self, 
        result: EvaluationResult, 
        content: MultimodalContent
    ) -> EvaluationResult:
        """Apply enhancements from ethical frameworks."""
        enhanced_result = result
        
        # Apply RLHF if enabled
        if "rlhf" in self.frameworks:
            enhanced_result = await self.frameworks["rlhf"].enhance_evaluation(
                enhanced_result, content
            )
        
        # Apply Constitutional AI if enabled
        if "constitutional_ai" in self.frameworks:
            enhanced_result = await self.frameworks["constitutional_ai"].enhance_evaluation(
                enhanced_result, content
            )
        
        return enhanced_result
    
    def _update_performance_metrics(
        self, 
        result: EvaluationResult, 
        processing_time: float
    ) -> None:
        """Update internal performance metrics."""
        self.evaluation_count += 1
        self.last_evaluation_time = datetime.now()
        
        # Update averages
        total_evals = self.performance_metrics["total_evaluations"]
        current_avg_time = self.performance_metrics["avg_processing_time"]
        
        self.performance_metrics["total_evaluations"] = total_evals + 1
        self.performance_metrics["avg_processing_time"] = (
            (current_avg_time * total_evals + processing_time) / (total_evals + 1)
        )
        
        # Update compliance rate
        is_compliant = result.compliance_status == ComplianceStatus.COMPLIANT
        current_compliance = self.performance_metrics["compliance_rate"]
        self.performance_metrics["compliance_rate"] = (
            (current_compliance * total_evals + (1.0 if is_compliant else 0.0)) / (total_evals + 1)
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.performance_metrics,
            "engine_uptime": (
                (datetime.now() - self.last_evaluation_time).total_seconds()
                if self.last_evaluation_time else 0
            ),
            "config_version": self.config.to_dict(),
            "framework_status": {
                name: True for name in self.frameworks.keys()
            }
        }
    
    def export_evaluation_history(
        self, 
        filepath: str, 
        format_type: str = "json"
    ) -> None:
        """
        Export evaluation history to file.
        
        Args:
            filepath: Output file path
            format_type: Export format ('json', 'csv')
        """
        history_data = [result.to_dict() for result in self.evaluator.evaluation_history]
        
        if format_type.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
        elif format_type.lower() == "csv":
            import pandas as pd
            
            # Flatten the data for CSV export
            flattened_data = []
            for result_dict in history_data:
                flat_result = {
                    "overall_score": result_dict["overall_score"],
                    "compliance_status": result_dict["compliance_status"],
                    "evaluation_time": result_dict["evaluation_time"],
                    "num_issues": len(result_dict["issues"]),
                    "modalities": ",".join(result_dict["modality_scores"].keys()),
                }
                # Add modality scores
                flat_result.update(result_dict["modality_scores"])
                flattened_data.append(flat_result)
            
            df = pd.DataFrame(flattened_data)
            df.to_csv(filepath, index=False)
        
        logger.info(f"Evaluation history exported to {filepath}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the ethics engine."""
        logger.info("Shutting down Multimodal Ethics Engine...")
        
        # Shutdown monitoring components
        if self.content_monitor:
            await self.content_monitor.shutdown()
        
        if self.dashboard:
            await self.dashboard.shutdown()
        
        # Shutdown framework integrations
        for framework in self.frameworks.values():
            if hasattr(framework, 'shutdown'):
                await framework.shutdown()
        
        logger.info("Multimodal Ethics Engine shutdown complete")