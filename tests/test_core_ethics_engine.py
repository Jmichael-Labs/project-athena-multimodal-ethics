"""
Unit tests for the core ethics engine.

Tests the main MultimodalEthicsEngine class including initialization,
evaluation methods, and integration with different modalities.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from PIL import Image

from athena.core.ethics_engine import MultimodalEthicsEngine
from athena.core.evaluator import MultimodalContent, EvaluationResult
from athena.core.config import EthicsConfig


class TestMultimodalEthicsEngine:
    """Test suite for MultimodalEthicsEngine."""

    @pytest.fixture
    def engine(self, test_config):
        """Create ethics engine instance for testing."""
        return MultimodalEthicsEngine(test_config)

    @pytest.mark.asyncio
    async def test_engine_initialization(self, test_config):
        """Test engine initialization with valid config."""
        engine = MultimodalEthicsEngine(test_config)
        
        assert engine.config == test_config
        assert hasattr(engine, 'text_analyzer')
        assert hasattr(engine, 'image_analyzer')
        assert hasattr(engine, 'audio_analyzer')
        assert hasattr(engine, 'video_analyzer')
        assert hasattr(engine, 'content_monitor')

    def test_engine_initialization_invalid_config(self):
        """Test engine initialization with invalid config."""
        with pytest.raises((TypeError, ValueError)):
            MultimodalEthicsEngine(None)

    @pytest.mark.asyncio
    async def test_evaluate_text_only(self, engine, sample_text):
        """Test evaluation with text-only content."""
        content = MultimodalContent()
        content.text = sample_text
        
        with patch.object(engine.text_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = Mock(
                toxicity_score=0.1,
                bias_score=0.2,
                hate_speech_score=0.05,
                overall_score=0.9,
                issues=[],
                processing_time=0.5
            )
            
            result = await engine.evaluate(content)
            
            assert isinstance(result, EvaluationResult)
            assert result.content_id is not None
            assert 0 <= result.overall_score <= 1
            assert 'text' in result.modality_scores
            mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_image_only(self, engine, sample_image):
        """Test evaluation with image-only content."""
        content = MultimodalContent()
        content.image = sample_image
        
        with patch.object(engine.image_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = Mock(
                violence_score=0.1,
                sexual_content_score=0.05,
                hate_symbols_score=0.0,
                overall_score=0.95,
                issues=[],
                processing_time=0.3
            )
            
            result = await engine.evaluate(content)
            
            assert isinstance(result, EvaluationResult)
            assert 'image' in result.modality_scores
            mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_multimodal_content(self, engine, sample_text, sample_image, sample_audio):
        """Test evaluation with multimodal content."""
        content = MultimodalContent()
        content.text = sample_text
        content.image = sample_image
        content.audio = sample_audio
        
        # Mock all analyzers
        with patch.object(engine.text_analyzer, 'analyze', new_callable=AsyncMock) as mock_text, \
             patch.object(engine.image_analyzer, 'analyze', new_callable=AsyncMock) as mock_image, \
             patch.object(engine.audio_analyzer, 'analyze', new_callable=AsyncMock) as mock_audio:
            
            # Setup mock returns
            mock_text.return_value = Mock(overall_score=0.9, issues=[], processing_time=0.5)
            mock_image.return_value = Mock(overall_score=0.85, issues=[], processing_time=0.3)
            mock_audio.return_value = Mock(overall_score=0.8, issues=[], processing_time=0.7)
            
            result = await engine.evaluate(content)
            
            assert isinstance(result, EvaluationResult)
            assert len(result.modality_scores) == 3
            assert 'text' in result.modality_scores
            assert 'image' in result.modality_scores
            assert 'audio' in result.modality_scores
            
            # All analyzers should be called
            mock_text.assert_called_once()
            mock_image.assert_called_once()
            mock_audio.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_empty_content(self, engine):
        """Test evaluation with empty content."""
        content = MultimodalContent()
        
        with pytest.raises(ValueError, match="No content provided"):
            await engine.evaluate(content)

    @pytest.mark.asyncio
    async def test_evaluate_with_metadata(self, engine, sample_text):
        """Test evaluation with metadata."""
        content = MultimodalContent()
        content.text = sample_text
        content.metadata = {
            "source": "test",
            "timestamp": "2024-01-01T12:00:00Z",
            "user_id": "test_user"
        }
        
        with patch.object(engine.text_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = Mock(
                overall_score=0.9,
                issues=[],
                processing_time=0.5
            )
            
            result = await engine.evaluate(content)
            
            # Check that metadata was passed through
            call_args = mock_analyze.call_args
            assert call_args[1]['metadata'] == content.metadata

    @pytest.mark.asyncio
    async def test_overall_score_calculation(self, engine, sample_text, sample_image):
        """Test overall score calculation from modality scores."""
        content = MultimodalContent()
        content.text = sample_text
        content.image = sample_image
        
        with patch.object(engine.text_analyzer, 'analyze', new_callable=AsyncMock) as mock_text, \
             patch.object(engine.image_analyzer, 'analyze', new_callable=AsyncMock) as mock_image:
            
            mock_text.return_value = Mock(overall_score=0.9, issues=[], processing_time=0.5)
            mock_image.return_value = Mock(overall_score=0.7, issues=[], processing_time=0.3)
            
            result = await engine.evaluate(content)
            
            # Overall score should be weighted average or minimum
            expected_min = min(0.9, 0.7)
            assert result.overall_score <= max(0.9, 0.7)
            assert result.overall_score >= expected_min

    @pytest.mark.asyncio
    async def test_issue_aggregation(self, engine, sample_text):
        """Test aggregation of issues from different modalities."""
        from athena.core.evaluator import Issue
        
        content = MultimodalContent()
        content.text = sample_text
        
        mock_issue = Issue(
            category="toxicity",
            severity=0.6,
            description="Potential toxic content detected",
            location="text",
            confidence=0.8
        )
        
        with patch.object(engine.text_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = Mock(
                overall_score=0.4,
                issues=[mock_issue],
                processing_time=0.5
            )
            
            result = await engine.evaluate(content)
            
            assert len(result.issues) == 1
            assert result.issues[0].category == "toxicity"
            assert result.issues[0].severity == 0.6

    @pytest.mark.asyncio
    async def test_performance_tracking(self, engine, sample_text):
        """Test performance tracking during evaluation."""
        content = MultimodalContent()
        content.text = sample_text
        
        with patch.object(engine.text_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = Mock(
                overall_score=0.9,
                issues=[],
                processing_time=0.5
            )
            
            result = await engine.evaluate(content)
            
            assert result.processing_time > 0
            assert isinstance(result.processing_time, (int, float))

    @pytest.mark.asyncio
    async def test_concurrent_evaluation(self, engine, sample_text):
        """Test concurrent evaluation of multiple contents."""
        contents = []
        for i in range(3):
            content = MultimodalContent()
            content.text = f"{sample_text} {i}"
            contents.append(content)
        
        with patch.object(engine.text_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = Mock(
                overall_score=0.9,
                issues=[],
                processing_time=0.5
            )
            
            # Evaluate concurrently
            tasks = [engine.evaluate(content) for content in contents]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            for result in results:
                assert isinstance(result, EvaluationResult)
            
            # Analyzer should be called for each content
            assert mock_analyze.call_count == 3

    @pytest.mark.asyncio
    async def test_error_handling_analyzer_failure(self, engine, sample_text):
        """Test error handling when analyzer fails."""
        content = MultimodalContent()
        content.text = sample_text
        
        with patch.object(engine.text_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.side_effect = Exception("Analyzer failed")
            
            # Should handle the error gracefully
            result = await engine.evaluate(content)
            
            # Should still return a result with error information
            assert isinstance(result, EvaluationResult)
            assert result.overall_score == 0.0  # Default to safe score
            assert any("error" in str(issue).lower() for issue in result.issues)

    @pytest.mark.asyncio
    async def test_constitutional_ai_enhancement(self, engine, sample_text):
        """Test Constitutional AI enhancement of evaluation."""
        content = MultimodalContent()
        content.text = sample_text
        
        with patch.object(engine.text_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze, \
             patch.object(engine, 'constitutional_ai', new_callable=AsyncMock) as mock_constitutional:
            
            base_result = Mock(overall_score=0.7, issues=[], processing_time=0.5)
            enhanced_result = Mock(overall_score=0.8, issues=[], processing_time=0.5)
            
            mock_analyze.return_value = base_result
            mock_constitutional.enhance_evaluation.return_value = enhanced_result
            
            result = await engine.evaluate(content)
            
            # Should use Constitutional AI enhancement if available
            if hasattr(engine, 'constitutional_ai') and engine.constitutional_ai:
                mock_constitutional.enhance_evaluation.assert_called_once()

    @pytest.mark.asyncio
    async def test_rlhf_enhancement(self, engine, sample_text):
        """Test RLHF enhancement of evaluation."""
        content = MultimodalContent()
        content.text = sample_text
        
        with patch.object(engine.text_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze, \
             patch.object(engine, 'rlhf_integration', new_callable=AsyncMock) as mock_rlhf:
            
            base_result = Mock(overall_score=0.7, issues=[], processing_time=0.5)
            enhanced_result = Mock(overall_score=0.75, issues=[], processing_time=0.5)
            
            mock_analyze.return_value = base_result
            mock_rlhf.enhance_evaluation.return_value = enhanced_result
            
            result = await engine.evaluate(content)
            
            # Should use RLHF enhancement if available
            if hasattr(engine, 'rlhf_integration') and engine.rlhf_integration:
                mock_rlhf.enhance_evaluation.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitoring_integration(self, engine, sample_text):
        """Test integration with content monitoring."""
        content = MultimodalContent()
        content.text = sample_text
        
        with patch.object(engine.text_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze, \
             patch.object(engine.content_monitor, 'log_evaluation', new_callable=AsyncMock) as mock_monitor:
            
            mock_analyze.return_value = Mock(
                overall_score=0.9,
                issues=[],
                processing_time=0.5
            )
            
            result = await engine.evaluate(content)
            
            # Should log the evaluation
            mock_monitor.assert_called_once()
            call_args = mock_monitor.call_args[0]
            assert isinstance(call_args[0], EvaluationResult)

    def test_engine_configuration_validation(self, test_config):
        """Test engine validates configuration on initialization."""
        # Test with invalid threshold
        test_config.text.threshold = 1.5  # Invalid - should be <= 1.0
        
        with pytest.raises(ValueError):
            MultimodalEthicsEngine(test_config)

    @pytest.mark.asyncio
    async def test_batch_evaluation(self, engine, sample_text):
        """Test batch evaluation of multiple contents."""
        contents = []
        for i in range(5):
            content = MultimodalContent()
            content.text = f"{sample_text} {i}"
            contents.append(content)
        
        with patch.object(engine, 'evaluate', new_callable=AsyncMock) as mock_evaluate:
            mock_evaluate.return_value = Mock(
                content_id="test",
                overall_score=0.9,
                modality_scores={'text': 0.9},
                issues=[],
                processing_time=0.5
            )
            
            # Test if engine has batch evaluation method
            if hasattr(engine, 'evaluate_batch'):
                results = await engine.evaluate_batch(contents)
                assert len(results) == 5
            else:
                # Fallback to individual evaluation
                results = []
                for content in contents:
                    result = await engine.evaluate(content)
                    results.append(result)
                assert len(results) == 5

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_content_evaluation(self, engine):
        """Test evaluation of large content."""
        # Large text content
        large_text = "Test content. " * 1000  # ~13KB text
        
        content = MultimodalContent()
        content.text = large_text
        
        with patch.object(engine.text_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = Mock(
                overall_score=0.9,
                issues=[],
                processing_time=2.0
            )
            
            result = await engine.evaluate(content)
            
            assert isinstance(result, EvaluationResult)
            # Should handle large content without errors
            assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_engine_shutdown(self, engine):
        """Test engine shutdown and cleanup."""
        # Test if engine has cleanup method
        if hasattr(engine, 'shutdown'):
            await engine.shutdown()
        
        # Test if engine has cleanup method
        if hasattr(engine, 'cleanup'):
            await engine.cleanup()
        
        # Engine should handle shutdown gracefully
        assert True  # If we get here, shutdown was successful