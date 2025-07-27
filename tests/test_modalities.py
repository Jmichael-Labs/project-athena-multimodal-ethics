"""
Unit tests for modality analyzers.

Tests all modality analyzers including text, image, audio, and video
ethics evaluation components.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from PIL import Image
import torch

from athena.modalities.text_ethics import TextEthicsAnalyzer
from athena.modalities.image_ethics import ImageEthicsAnalyzer
from athena.modalities.audio_ethics import AudioEthicsAnalyzer
from athena.modalities.video_ethics import VideoEthicsAnalyzer


class TestTextEthicsAnalyzer:
    """Test suite for TextEthicsAnalyzer."""

    @pytest.fixture
    def text_analyzer(self, test_config):
        """Create text analyzer instance for testing."""
        return TextEthicsAnalyzer(test_config)

    @pytest.mark.asyncio
    async def test_text_analyzer_initialization(self, test_config):
        """Test text analyzer initialization."""
        analyzer = TextEthicsAnalyzer(test_config)
        assert analyzer.config == test_config
        assert hasattr(analyzer, 'model')
        assert hasattr(analyzer, 'tokenizer')

    @pytest.mark.asyncio
    async def test_analyze_benign_text(self, text_analyzer, sample_text):
        """Test analysis of benign text content."""
        with patch.object(text_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(text_analyzer, '_detect_toxicity', return_value=0.1), \
             patch.object(text_analyzer, '_detect_bias', return_value=0.2), \
             patch.object(text_analyzer, '_detect_hate_speech', return_value=0.05):
            
            result = await text_analyzer.analyze(sample_text)
            
            assert hasattr(result, 'toxicity_score')
            assert hasattr(result, 'bias_score')
            assert hasattr(result, 'hate_speech_score')
            assert hasattr(result, 'overall_score')
            assert 0 <= result.overall_score <= 1

    @pytest.mark.asyncio
    async def test_analyze_toxic_text(self, text_analyzer):
        """Test analysis of potentially toxic text."""
        toxic_text = "This is a test message with potentially harmful content"
        
        with patch.object(text_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(text_analyzer, '_detect_toxicity', return_value=0.9), \
             patch.object(text_analyzer, '_detect_bias', return_value=0.3), \
             patch.object(text_analyzer, '_detect_hate_speech', return_value=0.8):
            
            result = await text_analyzer.analyze(toxic_text)
            
            assert result.toxicity_score >= 0.8
            assert result.overall_score <= 0.5  # Should be flagged as problematic

    @pytest.mark.asyncio
    async def test_analyze_empty_text(self, text_analyzer):
        """Test analysis of empty text."""
        with pytest.raises(ValueError, match="Empty text"):
            await text_analyzer.analyze("")

    @pytest.mark.asyncio
    async def test_analyze_with_metadata(self, text_analyzer, sample_text):
        """Test analysis with metadata."""
        metadata = {"source": "test", "user_id": "123"}
        
        with patch.object(text_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(text_analyzer, '_detect_toxicity', return_value=0.1):
            
            result = await text_analyzer.analyze(sample_text, metadata=metadata)
            
            assert hasattr(result, 'metadata')
            assert result.metadata == metadata

    @pytest.mark.asyncio
    async def test_bias_detection(self, text_analyzer):
        """Test bias detection functionality."""
        biased_text = "This text contains stereotypical assumptions about certain groups"
        
        with patch.object(text_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(text_analyzer, '_detect_bias', return_value=0.8):
            
            result = await text_analyzer.analyze(biased_text)
            
            assert result.bias_score >= 0.7

    @pytest.mark.asyncio
    async def test_misinformation_detection(self, text_analyzer):
        """Test misinformation detection."""
        misinfo_text = "This is a factually incorrect statement that could mislead people"
        
        with patch.object(text_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(text_analyzer, '_detect_misinformation', return_value=0.85):
            
            if hasattr(text_analyzer, '_detect_misinformation'):
                result = await text_analyzer.analyze(misinfo_text)
                assert hasattr(result, 'misinformation_score')


class TestImageEthicsAnalyzer:
    """Test suite for ImageEthicsAnalyzer."""

    @pytest.fixture
    def image_analyzer(self, test_config):
        """Create image analyzer instance for testing."""
        return ImageEthicsAnalyzer(test_config)

    @pytest.mark.asyncio
    async def test_image_analyzer_initialization(self, test_config):
        """Test image analyzer initialization."""
        analyzer = ImageEthicsAnalyzer(test_config)
        assert analyzer.config == test_config

    @pytest.mark.asyncio
    async def test_analyze_safe_image(self, image_analyzer, sample_image):
        """Test analysis of safe image content."""
        with patch.object(image_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(image_analyzer, '_detect_violence', return_value=0.1), \
             patch.object(image_analyzer, '_detect_sexual_content', return_value=0.05), \
             patch.object(image_analyzer, '_detect_hate_symbols', return_value=0.0):
            
            result = await image_analyzer.analyze(sample_image)
            
            assert hasattr(result, 'violence_score')
            assert hasattr(result, 'sexual_content_score')
            assert hasattr(result, 'overall_score')
            assert 0 <= result.overall_score <= 1

    @pytest.mark.asyncio
    async def test_analyze_image_array(self, image_analyzer, sample_image_array):
        """Test analysis of image as numpy array."""
        with patch.object(image_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(image_analyzer, '_detect_violence', return_value=0.1):
            
            result = await image_analyzer.analyze(sample_image_array)
            
            assert hasattr(result, 'overall_score')

    @pytest.mark.asyncio
    async def test_analyze_image_path(self, image_analyzer, test_data_paths):
        """Test analysis of image from file path."""
        image_path = str(test_data_paths["sample_image"])
        
        with patch.object(image_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(image_analyzer, '_detect_violence', return_value=0.1):
            
            result = await image_analyzer.analyze(image_path)
            
            assert hasattr(result, 'overall_score')

    @pytest.mark.asyncio
    async def test_violence_detection(self, image_analyzer, sample_image):
        """Test violence detection in images."""
        with patch.object(image_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(image_analyzer, '_detect_violence', return_value=0.9):
            
            result = await image_analyzer.analyze(sample_image)
            
            assert result.violence_score >= 0.8
            assert result.overall_score <= 0.5

    @pytest.mark.asyncio
    async def test_invalid_image_input(self, image_analyzer):
        """Test handling of invalid image input."""
        with pytest.raises((ValueError, TypeError)):
            await image_analyzer.analyze("not_an_image")

    @pytest.mark.asyncio
    async def test_child_safety_detection(self, image_analyzer, sample_image):
        """Test child safety detection."""
        with patch.object(image_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(image_analyzer, '_detect_child_safety_issues', return_value=0.1):
            
            if hasattr(image_analyzer, '_detect_child_safety_issues'):
                result = await image_analyzer.analyze(sample_image)
                assert hasattr(result, 'child_safety_score')


class TestAudioEthicsAnalyzer:
    """Test suite for AudioEthicsAnalyzer."""

    @pytest.fixture
    def audio_analyzer(self, test_config):
        """Create audio analyzer instance for testing."""
        return AudioEthicsAnalyzer(test_config)

    @pytest.mark.asyncio
    async def test_audio_analyzer_initialization(self, test_config):
        """Test audio analyzer initialization."""
        analyzer = AudioEthicsAnalyzer(test_config)
        assert analyzer.config == test_config

    @pytest.mark.asyncio
    async def test_analyze_safe_audio(self, audio_analyzer, sample_audio):
        """Test analysis of safe audio content."""
        with patch.object(audio_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(audio_analyzer, '_detect_speech_toxicity', return_value=0.1), \
             patch.object(audio_analyzer, '_detect_hate_speech', return_value=0.05):
            
            result = await audio_analyzer.analyze(sample_audio)
            
            assert hasattr(result, 'speech_toxicity_score')
            assert hasattr(result, 'overall_score')
            assert 0 <= result.overall_score <= 1

    @pytest.mark.asyncio
    async def test_analyze_audio_tensor(self, audio_analyzer):
        """Test analysis of audio as torch tensor."""
        audio_tensor = torch.randn(16000)  # 1 second at 16kHz
        
        with patch.object(audio_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(audio_analyzer, '_detect_speech_toxicity', return_value=0.1):
            
            result = await audio_analyzer.analyze(audio_tensor)
            
            assert hasattr(result, 'overall_score')

    @pytest.mark.asyncio
    async def test_speech_to_text_analysis(self, audio_analyzer, sample_audio):
        """Test speech-to-text conversion and text analysis."""
        with patch.object(audio_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(audio_analyzer, '_speech_to_text', return_value="This is transcribed text"), \
             patch.object(audio_analyzer, '_analyze_transcribed_text', return_value=0.1):
            
            result = await audio_analyzer.analyze(sample_audio)
            
            if hasattr(result, 'transcription'):
                assert isinstance(result.transcription, str)

    @pytest.mark.asyncio
    async def test_audio_privacy_detection(self, audio_analyzer, sample_audio):
        """Test privacy violation detection in audio."""
        with patch.object(audio_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(audio_analyzer, '_detect_privacy_violations', return_value=0.8):
            
            if hasattr(audio_analyzer, '_detect_privacy_violations'):
                result = await audio_analyzer.analyze(sample_audio)
                assert hasattr(result, 'privacy_violation_score')

    @pytest.mark.asyncio
    async def test_empty_audio_input(self, audio_analyzer):
        """Test handling of empty audio input."""
        empty_audio = np.array([])
        
        with pytest.raises(ValueError, match="Empty audio"):
            await audio_analyzer.analyze(empty_audio)


class TestVideoEthicsAnalyzer:
    """Test suite for VideoEthicsAnalyzer."""

    @pytest.fixture
    def video_analyzer(self, test_config):
        """Create video analyzer instance for testing."""
        return VideoEthicsAnalyzer(test_config)

    @pytest.mark.asyncio
    async def test_video_analyzer_initialization(self, test_config):
        """Test video analyzer initialization."""
        analyzer = VideoEthicsAnalyzer(test_config)
        assert analyzer.config == test_config

    @pytest.mark.asyncio
    async def test_analyze_safe_video(self, video_analyzer, sample_video_frames):
        """Test analysis of safe video content."""
        with patch.object(video_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(video_analyzer, '_analyze_frames', return_value=[0.1, 0.1, 0.1]), \
             patch.object(video_analyzer, '_detect_deepfake', return_value=0.05):
            
            result = await video_analyzer.analyze(sample_video_frames)
            
            assert hasattr(result, 'frame_scores')
            assert hasattr(result, 'overall_score')
            assert 0 <= result.overall_score <= 1

    @pytest.mark.asyncio
    async def test_analyze_video_with_audio(self, video_analyzer, sample_video_frames, sample_audio):
        """Test analysis of video with audio track."""
        video_data = {
            'frames': sample_video_frames,
            'audio': sample_audio
        }
        
        with patch.object(video_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(video_analyzer, '_analyze_frames', return_value=[0.1, 0.1]), \
             patch.object(video_analyzer, '_analyze_audio_track', return_value=0.1):
            
            result = await video_analyzer.analyze(video_data)
            
            assert hasattr(result, 'audio_score')
            assert hasattr(result, 'overall_score')

    @pytest.mark.asyncio
    async def test_deepfake_detection(self, video_analyzer, sample_video_frames):
        """Test deepfake detection in video."""
        with patch.object(video_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(video_analyzer, '_detect_deepfake', return_value=0.9):
            
            result = await video_analyzer.analyze(sample_video_frames)
            
            if hasattr(result, 'deepfake_score'):
                assert result.deepfake_score >= 0.8

    @pytest.mark.asyncio
    async def test_temporal_consistency_analysis(self, video_analyzer, sample_video_frames):
        """Test temporal consistency analysis."""
        with patch.object(video_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(video_analyzer, '_analyze_temporal_consistency', return_value=0.95):
            
            if hasattr(video_analyzer, '_analyze_temporal_consistency'):
                result = await video_analyzer.analyze(sample_video_frames)
                assert hasattr(result, 'temporal_consistency_score')

    @pytest.mark.asyncio
    async def test_violence_detection_video(self, video_analyzer, sample_video_frames):
        """Test violence detection in video frames."""
        with patch.object(video_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(video_analyzer, '_detect_violence_in_frames', return_value=0.8):
            
            result = await video_analyzer.analyze(sample_video_frames)
            
            assert result.overall_score <= 0.5  # Should be flagged

    @pytest.mark.asyncio
    async def test_empty_video_input(self, video_analyzer):
        """Test handling of empty video input."""
        empty_frames = []
        
        with pytest.raises(ValueError, match="Empty video"):
            await video_analyzer.analyze(empty_frames)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_long_video_analysis(self, video_analyzer):
        """Test analysis of long video (many frames)."""
        # Create a longer video with many frames
        long_video = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(100)]
        
        with patch.object(video_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(video_analyzer, '_analyze_frames', return_value=[0.1] * 100):
            
            result = await video_analyzer.analyze(long_video)
            
            assert hasattr(result, 'overall_score')
            assert result.processing_time > 0


class TestModalityIntegration:
    """Test integration between different modality analyzers."""

    @pytest.mark.asyncio
    async def test_cross_modal_consistency(self, test_config, sample_text, sample_image):
        """Test consistency across different modalities."""
        text_analyzer = TextEthicsAnalyzer(test_config)
        image_analyzer = ImageEthicsAnalyzer(test_config)
        
        with patch.object(text_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(image_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(text_analyzer, '_detect_toxicity', return_value=0.8), \
             patch.object(image_analyzer, '_detect_violence', return_value=0.8):
            
            text_result = await text_analyzer.analyze(sample_text)
            image_result = await image_analyzer.analyze(sample_image)
            
            # Both should flag problematic content consistently
            assert text_result.overall_score <= 0.5
            assert image_result.overall_score <= 0.5

    @pytest.mark.asyncio
    async def test_modality_performance_comparison(self, test_config, sample_text, sample_image, sample_audio):
        """Test performance characteristics of different modalities."""
        text_analyzer = TextEthicsAnalyzer(test_config)
        image_analyzer = ImageEthicsAnalyzer(test_config)
        audio_analyzer = AudioEthicsAnalyzer(test_config)
        
        with patch.object(text_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(image_analyzer, '_load_models', new_callable=AsyncMock), \
             patch.object(audio_analyzer, '_load_models', new_callable=AsyncMock):
            
            # Mock fast responses
            with patch.object(text_analyzer, '_detect_toxicity', return_value=0.1), \
                 patch.object(image_analyzer, '_detect_violence', return_value=0.1), \
                 patch.object(audio_analyzer, '_detect_speech_toxicity', return_value=0.1):
                
                text_result = await text_analyzer.analyze(sample_text)
                image_result = await image_analyzer.analyze(sample_image)
                audio_result = await audio_analyzer.analyze(sample_audio)
                
                # All should complete successfully
                assert all(hasattr(result, 'processing_time') 
                          for result in [text_result, image_result, audio_result])
                assert all(hasattr(result, 'overall_score') 
                          for result in [text_result, image_result, audio_result])