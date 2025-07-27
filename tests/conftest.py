"""
Pytest configuration and shared fixtures for Project Athena tests.

Provides common test fixtures, mock objects, and test utilities
used across the entire test suite.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
import torch

# Import project modules
from athena.core.config import EthicsConfig
from athena.core.ethics_engine import MultimodalEthicsEngine
from athena.utils.data_utils import DataProcessor
from athena.utils.model_utils import ModelManager
from athena.utils.security_utils import SecurityManager
from athena.utils.logging_utils import LogManager

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def test_config():
    """Create a test configuration."""
    config = EthicsConfig()
    
    # Override with test-specific settings
    config.logging_level = "DEBUG"
    config.cache_enabled = False
    config.monitoring_enabled = False
    
    # Use smaller models and batches for testing
    config.text.model_name = "mock-llama-model"
    config.text.batch_size = 2
    config.text.threshold = 0.5
    
    config.image.model_name = "mock-clip-model"
    config.image.batch_size = 1
    config.image.threshold = 0.5
    
    config.audio.model_name = "mock-wav2vec-model"
    config.audio.batch_size = 1
    config.audio.threshold = 0.5
    
    config.video.model_name = "mock-videomae-model"
    config.video.batch_size = 1
    config.video.threshold = 0.5
    
    # Disable external integrations
    config.meta_integration.use_pytorch_integration = False
    config.rlhf.enabled = False
    
    return config

@pytest.fixture
def mock_model():
    """Create a mock ML model for testing."""
    model = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    model.forward = Mock(return_value=torch.tensor([0.7]))
    model.__call__ = Mock(return_value=torch.tensor([0.7]))
    return model

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = Mock(return_value="mock decoded text")
    tokenizer.batch_encode_plus = Mock(return_value={
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    })
    return tokenizer

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "This is a sample text for testing the ethics evaluation system."

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create a simple RGB image
    image = Image.new('RGB', (224, 224), color='red')
    return image

@pytest.fixture
def sample_image_array():
    """Create a sample image as numpy array."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

@pytest.fixture
def sample_audio():
    """Create sample audio data for testing."""
    # Generate simple sine wave
    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio_data

@pytest.fixture
def sample_video_frames():
    """Create sample video frames for testing."""
    # Create 5 simple frames
    frames = []
    for i in range(5):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        frames.append(frame)
    return frames

@pytest.fixture
def mock_ethics_engine(test_config):
    """Create a mock ethics engine for testing."""
    with patch('athena.core.ethics_engine.MultimodalEthicsEngine') as mock_engine:
        engine = Mock(spec=MultimodalEthicsEngine)
        engine.config = test_config
        engine.evaluate = Mock()
        mock_engine.return_value = engine
        yield engine

@pytest.fixture
def mock_data_processor(test_config):
    """Create a mock data processor for testing."""
    with patch('athena.utils.data_utils.DataProcessor') as mock_processor:
        processor = Mock(spec=DataProcessor)
        processor.config = test_config
        processor.validate_content = Mock()
        processor.preprocess_content = Mock()
        mock_processor.return_value = processor
        yield processor

@pytest.fixture
def mock_model_manager(test_config):
    """Create a mock model manager for testing."""
    with patch('athena.utils.model_utils.ModelManager') as mock_manager:
        manager = Mock(spec=ModelManager)
        manager.config = test_config
        manager.load_model = Mock()
        manager.cache = Mock()
        mock_manager.return_value = manager
        yield manager

@pytest.fixture
def mock_security_manager(test_config):
    """Create a mock security manager for testing."""
    with patch('athena.utils.security_utils.SecurityManager') as mock_security:
        security = Mock(spec=SecurityManager)
        security.config = test_config
        security.encrypt_sensitive_data = Mock()
        security.secure_content_evaluation = Mock()
        mock_security.return_value = security
        yield security

@pytest.fixture
def mock_log_manager(test_config):
    """Create a mock log manager for testing."""
    with patch('athena.utils.logging_utils.LogManager') as mock_logger:
        log_manager = Mock(spec=LogManager)
        log_manager.config = test_config
        mock_logger.return_value = log_manager
        yield log_manager

@pytest.fixture
def mock_external_apis():
    """Mock external API calls."""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        
        # Mock successful API responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "data": {}}
        mock_response.text = "success"
        
        mock_get.return_value = mock_response
        mock_post.return_value = mock_response
        
        yield {
            'get': mock_get,
            'post': mock_post,
            'response': mock_response
        }

@pytest.fixture
def mock_meta_api():
    """Mock Meta AI API responses."""
    with patch('athena.frameworks.rlhf_integration.MetaAPI') as mock_api:
        api = Mock()
        api.generate_text = Mock(return_value="Generated text response")
        api.classify_content = Mock(return_value={"score": 0.7, "label": "safe"})
        api.get_embeddings = Mock(return_value=np.random.random((1, 768)))
        mock_api.return_value = api
        yield api

@pytest.fixture
def sample_evaluation_result():
    """Create a sample evaluation result for testing."""
    from athena.core.evaluator import EvaluationResult, Issue
    
    return EvaluationResult(
        content_id="test-content-123",
        overall_score=0.85,
        modality_scores={
            "text": 0.9,
            "image": 0.8,
            "audio": 0.85,
            "video": 0.8
        },
        issues=[
            Issue(
                category="bias",
                severity=0.3,
                description="Minor bias detected in text content",
                location="text",
                confidence=0.75
            )
        ],
        recommendations=[
            "Consider reviewing content for potential bias",
            "Add diverse perspectives to content"
        ],
        processing_time=1.23,
        timestamp="2024-01-01T12:00:00Z"
    )

@pytest.fixture
def performance_benchmarks():
    """Define performance benchmarks for testing."""
    return {
        "text_processing_time": 2.0,  # seconds
        "image_processing_time": 1.5,
        "audio_processing_time": 3.0,
        "video_processing_time": 5.0,
        "memory_usage_mb": 1024,
        "accuracy_threshold": 0.85,
        "throughput_requests_per_second": 10
    }

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables and patches."""
    # Set test environment variables
    monkeypatch.setenv("ATHENA_CONFIG_ENV", "testing")
    monkeypatch.setenv("ATHENA_DEBUG", "true")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.parent))
    
    # Mock torch.cuda functions to avoid GPU dependency
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)

@pytest.fixture
def test_data_paths(temp_dir):
    """Create test data directory structure."""
    data_dir = temp_dir / "test_data"
    data_dir.mkdir()
    
    # Create subdirectories
    (data_dir / "images").mkdir()
    (data_dir / "audio").mkdir()
    (data_dir / "videos").mkdir()
    (data_dir / "text").mkdir()
    
    # Create sample files
    sample_text_file = data_dir / "text" / "sample.txt"
    sample_text_file.write_text("This is a sample text file for testing.")
    
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='blue')
    test_image.save(data_dir / "images" / "test_image.jpg")
    
    return {
        "root": data_dir,
        "images": data_dir / "images",
        "audio": data_dir / "audio", 
        "videos": data_dir / "videos",
        "text": data_dir / "text",
        "sample_text": sample_text_file,
        "sample_image": data_dir / "images" / "test_image.jpg"
    }

@pytest.fixture
def async_test_client():
    """Create async test client for API testing."""
    from httpx import AsyncClient
    
    async def _create_client():
        return AsyncClient(base_url="http://testserver")
    
    return _create_client

# Utility functions for tests
def assert_evaluation_result_valid(result):
    """Assert that an evaluation result has valid structure."""
    assert hasattr(result, 'content_id')
    assert hasattr(result, 'overall_score')
    assert hasattr(result, 'modality_scores')
    assert hasattr(result, 'issues')
    assert hasattr(result, 'recommendations')
    assert hasattr(result, 'processing_time')
    assert hasattr(result, 'timestamp')
    
    assert 0 <= result.overall_score <= 1
    assert result.processing_time >= 0
    assert isinstance(result.modality_scores, dict)
    assert isinstance(result.issues, list)
    assert isinstance(result.recommendations, list)

def create_mock_multimodal_content(include_text=True, include_image=True, 
                                 include_audio=True, include_video=True):
    """Create mock multimodal content for testing."""
    from athena.core.evaluator import MultimodalContent
    
    content = MultimodalContent()
    
    if include_text:
        content.text = "Sample text content for testing"
    
    if include_image:
        content.image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    if include_audio:
        content.audio = np.random.random(16000).astype(np.float32)
    
    if include_video:
        content.video = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(5)
        ]
    
    return content