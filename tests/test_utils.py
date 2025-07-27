"""
Unit tests for utility functions.

Tests all utility modules including data processing, model management,
security, and logging utilities.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from PIL import Image
import torch

from athena.utils.data_utils import DataProcessor, DataValidator, ContentPreprocessor
from athena.utils.model_utils import ModelManager, ModelCache, PerformanceOptimizer
from athena.utils.security_utils import SecurityManager, EncryptionHandler, AccessController
from athena.utils.logging_utils import LogManager, AuditLogger, MetricsCollector


class TestDataUtils:
    """Test suite for data utilities."""

    @pytest.fixture
    def data_processor(self, test_config):
        """Create data processor instance for testing."""
        return DataProcessor(test_config)

    @pytest.fixture
    def data_validator(self, test_config):
        """Create data validator instance for testing."""
        return DataValidator(test_config)

    @pytest.fixture
    def content_preprocessor(self, test_config):
        """Create content preprocessor instance for testing."""
        return ContentPreprocessor(test_config)

    @pytest.mark.asyncio
    async def test_data_validator_text(self, data_validator, sample_text):
        """Test text validation."""
        result = await data_validator.validate_content(sample_text, "text")
        
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'metadata')
        assert result.is_valid is True
        assert isinstance(result.errors, list)

    @pytest.mark.asyncio
    async def test_data_validator_invalid_text(self, data_validator):
        """Test validation of invalid text."""
        invalid_text = ""  # Empty text
        
        result = await data_validator.validate_content(invalid_text, "text")
        
        assert result.is_valid is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_data_validator_image(self, data_validator, sample_image):
        """Test image validation."""
        result = await data_validator.validate_content(sample_image, "image")
        
        assert hasattr(result, 'is_valid')
        assert result.is_valid is True
        assert 'width' in result.metadata
        assert 'height' in result.metadata

    @pytest.mark.asyncio
    async def test_data_validator_audio(self, data_validator, sample_audio):
        """Test audio validation."""
        result = await data_validator.validate_content(sample_audio, "audio")
        
        assert hasattr(result, 'is_valid')
        assert result.is_valid is True
        assert 'duration' in result.metadata

    @pytest.mark.asyncio
    async def test_content_preprocessor_text(self, content_preprocessor, sample_text):
        """Test text preprocessing."""
        result = await content_preprocessor.preprocess_content(sample_text, "text")
        
        assert result.success is True
        assert isinstance(result.data, str)
        assert result.processing_time >= 0
        assert 'final_length' in result.metadata

    @pytest.mark.asyncio
    async def test_content_preprocessor_image(self, content_preprocessor, sample_image):
        """Test image preprocessing."""
        result = await content_preprocessor.preprocess_content(sample_image, "image")
        
        assert result.success is True
        assert isinstance(result.data, np.ndarray)
        assert 'final_size' in result.metadata

    @pytest.mark.asyncio
    async def test_content_preprocessor_audio(self, content_preprocessor, sample_audio):
        """Test audio preprocessing."""
        result = await content_preprocessor.preprocess_content(sample_audio, "audio")
        
        assert result.success is True
        assert isinstance(result.data, np.ndarray)
        assert 'final_duration' in result.metadata

    @pytest.mark.asyncio
    async def test_data_processor_pipeline(self, data_processor, sample_text):
        """Test complete data processing pipeline."""
        success, processed_data, metadata = await data_processor.process_content(
            sample_text, "text", validate=True, preprocess=True
        )
        
        assert success is True
        assert processed_data is not None
        assert 'validation' in metadata
        assert 'preprocessing' in metadata

    def test_data_processor_cache(self, data_processor, sample_text):
        """Test data processor caching functionality."""
        cache_key = data_processor.get_content_hash(sample_text)
        
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

    def test_data_processor_cache_stats(self, data_processor):
        """Test cache statistics."""
        stats = data_processor.get_cache_stats()
        
        assert 'cache_enabled' in stats
        assert 'cache_size' in stats
        assert 'cache_ttl' in stats


class TestModelUtils:
    """Test suite for model utilities."""

    @pytest.fixture
    def model_manager(self, test_config):
        """Create model manager instance for testing."""
        return ModelManager(test_config)

    @pytest.fixture
    def model_cache(self):
        """Create model cache instance for testing."""
        return ModelCache(max_memory_mb=1024, max_models=5)

    @pytest.fixture
    def performance_optimizer(self, test_config):
        """Create performance optimizer instance for testing."""
        return PerformanceOptimizer(test_config)

    def test_model_cache_initialization(self, model_cache):
        """Test model cache initialization."""
        assert model_cache.max_memory_mb == 1024
        assert model_cache.max_models == 5
        assert len(model_cache.models) == 0

    def test_model_cache_put_get(self, model_cache, mock_model):
        """Test putting and getting models from cache."""
        from athena.utils.model_utils import ModelInfo
        from datetime import datetime
        
        model_info = ModelInfo(
            model_id="test-model",
            model_type="text",
            framework="pytorch",
            size_mb=100.0,
            load_time=1.0,
            last_used=datetime.now()
        )
        
        # Put model in cache
        success = model_cache.put_model("test-model", mock_model, model_info)
        assert success is True
        
        # Get model from cache
        retrieved_model = model_cache.get_model("test-model")
        assert retrieved_model == mock_model

    def test_model_cache_eviction(self, model_cache, mock_model):
        """Test model cache eviction."""
        from athena.utils.model_utils import ModelInfo
        from datetime import datetime
        
        # Fill cache beyond capacity
        for i in range(7):  # More than max_models (5)
            model_info = ModelInfo(
                model_id=f"model-{i}",
                model_type="text",
                framework="pytorch",
                size_mb=100.0,
                load_time=1.0,
                last_used=datetime.now()
            )
            model_cache.put_model(f"model-{i}", mock_model, model_info)
        
        # Should not exceed max_models
        assert len(model_cache.models) <= model_cache.max_models

    def test_model_cache_stats(self, model_cache):
        """Test model cache statistics."""
        stats = model_cache.get_cache_stats()
        
        assert 'total_models' in stats
        assert 'memory_usage_mb' in stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'hit_rate' in stats

    @pytest.mark.asyncio
    async def test_performance_optimizer_quantization(self, performance_optimizer, mock_model):
        """Test model quantization optimization."""
        with patch('torch.quantization.quantize_dynamic', return_value=mock_model):
            optimized_model, info = await performance_optimizer.optimize_model(
                mock_model, "text", "fast"
            )
            
            assert optimized_model is not None
            assert 'optimizations_applied' in info
            assert 'original_size' in info

    @pytest.mark.asyncio
    async def test_performance_optimizer_benchmark(self, performance_optimizer, mock_model):
        """Test model benchmarking."""
        sample_input = torch.randn(1, 10)
        
        with patch.object(mock_model, '__call__', return_value=torch.tensor([0.5])):
            metrics = await performance_optimizer.benchmark_model(
                mock_model, sample_input, num_runs=5
            )
            
            assert hasattr(metrics, 'inference_time')
            assert hasattr(metrics, 'memory_usage')
            assert hasattr(metrics, 'throughput')

    @pytest.mark.asyncio
    async def test_model_manager_load_model(self, model_manager, mock_model):
        """Test model loading through model manager."""
        with patch.object(model_manager, '_load_model_by_type', return_value=mock_model):
            loaded_model = await model_manager.load_model(
                "test-model", "mock-model-path", "text"
            )
            
            assert loaded_model is not None

    def test_model_manager_get_loaded_models(self, model_manager):
        """Test getting list of loaded models."""
        models = model_manager.get_loaded_models()
        assert isinstance(models, list)

    def test_model_manager_system_stats(self, model_manager):
        """Test system statistics."""
        stats = model_manager.get_system_stats()
        
        assert 'cache_stats' in stats
        assert 'device' in stats
        assert 'total_models_registered' in stats


class TestSecurityUtils:
    """Test suite for security utilities."""

    @pytest.fixture
    def security_manager(self, test_config):
        """Create security manager instance for testing."""
        return SecurityManager(test_config)

    @pytest.fixture
    def encryption_handler(self):
        """Create encryption handler instance for testing."""
        return EncryptionHandler()

    @pytest.fixture
    def access_controller(self, test_config):
        """Create access controller instance for testing."""
        return AccessController(test_config)

    def test_encryption_handler_initialization(self, encryption_handler):
        """Test encryption handler initialization."""
        assert encryption_handler.master_key is not None
        assert encryption_handler.fernet is not None

    def test_encryption_decryption(self, encryption_handler):
        """Test data encryption and decryption."""
        original_data = b"sensitive information"
        
        # Encrypt data
        encrypted_data = encryption_handler.encrypt_data(original_data)
        assert encrypted_data != original_data
        
        # Decrypt data
        decrypted_data = encryption_handler.decrypt_data(encrypted_data)
        assert decrypted_data == original_data

    def test_string_encryption_decryption(self, encryption_handler):
        """Test string encryption and decryption."""
        original_string = "sensitive text information"
        
        # Encrypt string
        encrypted_string = encryption_handler.encrypt_string(original_string)
        assert encrypted_string != original_string
        
        # Decrypt string
        decrypted_string = encryption_handler.decrypt_string(encrypted_string)
        assert decrypted_string == original_string

    def test_password_hashing(self, encryption_handler):
        """Test password hashing and verification."""
        password = "test_password_123"
        
        # Hash password
        hashed_password = encryption_handler.hash_password(password)
        assert hashed_password != password
        
        # Verify password
        is_valid = encryption_handler.verify_password(password, hashed_password)
        assert is_valid is True
        
        # Verify wrong password
        is_invalid = encryption_handler.verify_password("wrong_password", hashed_password)
        assert is_invalid is False

    def test_hmac_generation_verification(self, encryption_handler):
        """Test HMAC generation and verification."""
        data = b"data to sign"
        
        # Generate HMAC
        signature = encryption_handler.generate_hmac(data)
        assert isinstance(signature, str)
        
        # Verify HMAC
        is_valid = encryption_handler.verify_hmac(data, signature)
        assert is_valid is True
        
        # Verify with wrong data
        is_invalid = encryption_handler.verify_hmac(b"wrong data", signature)
        assert is_invalid is False

    def test_access_controller_user_creation(self, access_controller):
        """Test user creation in access controller."""
        success = access_controller.create_user(
            "test_user", "password123", "evaluator"
        )
        assert success is True
        
        # Test duplicate user creation
        duplicate_success = access_controller.create_user(
            "test_user", "password456", "evaluator"
        )
        assert duplicate_success is False

    def test_access_controller_authentication(self, access_controller):
        """Test user authentication."""
        # Create user first
        access_controller.create_user("auth_test_user", "password123", "evaluator")
        
        # Test valid authentication
        token = access_controller.authenticate_user("auth_test_user", "password123")
        assert token is not None
        
        # Test invalid authentication
        invalid_token = access_controller.authenticate_user("auth_test_user", "wrong_password")
        assert invalid_token is None

    def test_access_controller_token_validation(self, access_controller):
        """Test token validation."""
        # Create user and authenticate
        access_controller.create_user("token_test_user", "password123", "evaluator")
        token = access_controller.authenticate_user("token_test_user", "password123")
        
        # Validate token
        credentials = access_controller.validate_token(token)
        assert credentials is not None
        assert credentials.user_id == "token_test_user"

    def test_access_controller_permissions(self, access_controller):
        """Test permission checking."""
        # Create user and authenticate
        access_controller.create_user("perm_test_user", "password123", "evaluator")
        token = access_controller.authenticate_user("perm_test_user", "password123")
        credentials = access_controller.validate_token(token)
        
        # Test permission checking
        has_permission = access_controller.check_permission(credentials, "content.evaluate")
        assert has_permission is True
        
        has_no_permission = access_controller.check_permission(credentials, "admin.users")
        assert has_no_permission is False

    @pytest.mark.asyncio
    async def test_security_manager_secure_evaluation(self, security_manager):
        """Test secure content evaluation."""
        from athena.utils.security_utils import SecurityCredentials, AccessLevel
        
        credentials = SecurityCredentials(
            user_id="test_user",
            access_level=AccessLevel.RESTRICTED,
            permissions={"content.evaluate"}
        )
        
        content = "test content"
        
        authorized, error = await security_manager.secure_content_evaluation(
            content, credentials, "evaluate"
        )
        
        assert authorized is True
        assert error is None

    def test_security_manager_audit_events(self, security_manager):
        """Test audit event retrieval."""
        events = security_manager.get_audit_events(limit=10)
        assert isinstance(events, list)

    def test_security_manager_security_stats(self, security_manager):
        """Test security statistics."""
        stats = security_manager.get_security_stats()
        
        assert 'total_audit_events' in stats
        assert 'security_alerts' in stats
        assert 'encryption_enabled' in stats


class TestLoggingUtils:
    """Test suite for logging utilities."""

    @pytest.fixture
    def log_manager(self, test_config):
        """Create log manager instance for testing."""
        return LogManager(test_config)

    @pytest.fixture
    def audit_logger(self, test_config):
        """Create audit logger instance for testing."""
        return AuditLogger(test_config)

    @pytest.fixture
    def metrics_collector(self, test_config):
        """Create metrics collector instance for testing."""
        return MetricsCollector(test_config)

    def test_log_manager_initialization(self, log_manager):
        """Test log manager initialization."""
        assert log_manager.config is not None
        assert hasattr(log_manager, 'log_buffer')
        assert hasattr(log_manager, 'log_stats')

    def test_log_manager_get_logs(self, log_manager):
        """Test log retrieval."""
        logs = log_manager.get_logs(level="INFO", limit=10)
        assert isinstance(logs, list)

    def test_log_manager_stats(self, log_manager):
        """Test log statistics."""
        stats = log_manager.get_log_stats()
        
        assert 'total_logs' in stats
        assert 'buffer_size' in stats
        assert 'logs_by_level' in stats

    @pytest.mark.asyncio
    async def test_audit_logger_log_event(self, audit_logger):
        """Test audit event logging."""
        audit_id = await audit_logger.log_audit_event(
            event_type="test_event",
            actor="test_user",
            resource="test_resource",
            action="test_action",
            result="success"
        )
        
        assert isinstance(audit_id, str)
        assert len(audit_id) > 0

    def test_audit_logger_get_audit_trail(self, audit_logger):
        """Test audit trail retrieval."""
        trail = audit_logger.get_audit_trail(limit=10)
        assert isinstance(trail, list)

    def test_audit_logger_compliance_report(self, audit_logger):
        """Test compliance report generation."""
        report = audit_logger.get_compliance_report()
        
        assert 'report_generated' in report
        assert 'compliance_events' in report
        assert 'summary' in report

    @pytest.mark.asyncio
    async def test_metrics_collector_record_metric(self, metrics_collector):
        """Test metric recording."""
        await metrics_collector.record_metric(
            "test.metric",
            42.0,
            labels={"type": "test"},
            metric_type="gauge"
        )
        
        # Should record without error
        assert True

    @pytest.mark.asyncio
    async def test_metrics_collector_evaluation_metrics(self, metrics_collector):
        """Test evaluation metrics recording."""
        await metrics_collector.record_evaluation_metrics(
            modality="text",
            duration=1.5,
            compliance_score=0.85,
            status="success",
            issues=[]
        )
        
        # Should record without error
        assert True

    def test_metrics_collector_get_metrics(self, metrics_collector):
        """Test metrics retrieval."""
        metrics = metrics_collector.get_metrics(limit=10)
        assert isinstance(metrics, list)

    def test_metrics_collector_stats(self, metrics_collector):
        """Test metrics collection statistics."""
        stats = metrics_collector.get_metrics_stats()
        
        assert 'total_metrics' in stats
        assert 'prometheus_enabled' in stats
        assert 'collection_active' in stats

    @pytest.mark.asyncio
    async def test_metrics_collector_start_stop(self, metrics_collector):
        """Test starting and stopping metrics collection."""
        await metrics_collector.start_collection()
        assert metrics_collector.collection_active is True
        
        await metrics_collector.stop_collection()
        assert metrics_collector.collection_active is False


class TestUtilityIntegration:
    """Test integration between different utility modules."""

    @pytest.mark.asyncio
    async def test_data_security_integration(self, test_config, sample_text):
        """Test integration between data processing and security."""
        data_processor = DataProcessor(test_config)
        security_manager = SecurityManager(test_config)
        
        # Test secure data processing
        encrypted_text = security_manager.encrypt_sensitive_data(sample_text)
        assert encrypted_text != sample_text
        
        decrypted_text = security_manager.decrypt_sensitive_data(encrypted_text)
        assert decrypted_text == sample_text

    @pytest.mark.asyncio
    async def test_model_security_integration(self, test_config, mock_model):
        """Test integration between model management and security."""
        model_manager = ModelManager(test_config)
        security_manager = SecurityManager(test_config)
        
        # Test secure model operations
        from athena.utils.security_utils import SecurityCredentials, AccessLevel
        
        credentials = SecurityCredentials(
            user_id="test_user",
            access_level=AccessLevel.RESTRICTED,
            permissions={"content.evaluate"}
        )
        
        authorized, error = await security_manager.secure_content_evaluation(
            "test content", credentials, "evaluate"
        )
        
        assert authorized is True

    def test_logging_monitoring_integration(self, test_config):
        """Test integration between logging and monitoring."""
        log_manager = LogManager(test_config)
        metrics_collector = MetricsCollector(test_config)
        
        # Test that both systems can coexist
        log_stats = log_manager.get_log_stats()
        metrics_stats = metrics_collector.get_metrics_stats()
        
        assert isinstance(log_stats, dict)
        assert isinstance(metrics_stats, dict)