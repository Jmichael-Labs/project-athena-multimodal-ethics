"""
Test Suite for Project Athena Multimodal Ethics Framework

Comprehensive testing infrastructure covering all components including
unit tests, integration tests, performance tests, and end-to-end validation.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATA_DIR = project_root / "tests" / "data"
TEST_FIXTURES_DIR = project_root / "tests" / "fixtures"
TEST_OUTPUT_DIR = project_root / "tests" / "output"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_FIXTURES_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Global test configuration
PYTEST_CONFIG = {
    "test_timeout": 300,  # 5 minutes default timeout
    "slow_test_timeout": 600,  # 10 minutes for slow tests
    "integration_test_timeout": 900,  # 15 minutes for integration tests
    "parallel_workers": 4,
    "coverage_threshold": 85,
}

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )

def pytest_collection_modify_items(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        else:
            item.add_marker(pytest.mark.unit)