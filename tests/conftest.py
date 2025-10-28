"""
pytest Configuration - Shared Fixtures and Setup
================================================

Professional test infrastructure following:
- DRY: Reusable fixtures across all test modules
- Separation of Concerns: Test config separate from tests
- Clean Code: Clear, documented fixtures
- MLOps: Reproducible test environment

Author: MLOps Team 24
Date: October 2025
"""
import pytest
import sys
from pathlib import Path
import logging

# ============================================================================
# PROJECT PATH CONFIGURATION
# ============================================================================

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """
    Configure pytest environment before tests run.
    
    Setup:
        - Logging level
        - Warning filters
        - Test markers
    """
    # Suppress verbose logging during tests
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s'
    )
    
    # Register custom markers
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests"
    )

# ============================================================================
# PYTEST COLLECTION HOOKS
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically.
    
    Auto-marking:
        - Files named test_integration_* → integration marker
        - Files named test_unit_* → unit marker
    """
    for item in items:
        # Auto-mark integration tests
        if "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark unit tests
        if "test_unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

# ============================================================================
# CUSTOM ASSERTIONS
# ============================================================================

def pytest_assertrepr_compare(op, left, right):
    """
    Provide custom assertion failure messages for better debugging.
    """
    if op == "==":
        if isinstance(left, float) and isinstance(right, float):
            return [
                f"Float comparison failed:",
                f"   Left:  {left:.6f}",
                f"   Right: {right:.6f}",
                f"   Diff:  {abs(left - right):.6f}"
            ]

# ============================================================================
# SESSION FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Project root directory.
    
    Scope: session (computed once)
    Returns: Path to project root
    """
    return PROJECT_ROOT

@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """
    Data directory path.
    
    Returns: Path to data directory
    Validates: Directory exists
    """
    data_path = project_root / "data"
    assert data_path.exists(), f"Data directory not found: {data_path}"
    return data_path

@pytest.fixture(scope="session")
def processed_data_dir(data_dir: Path) -> Path:
    """
    Processed data directory.
    
    Returns: Path to processed data
    Validates: Directory exists and contains split files
    """
    processed_path = data_dir / "processed"
    assert processed_path.exists(), f"Processed data directory not found: {processed_path}"
    
    # Validate critical files exist
    required_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
    for filename in required_files:
        filepath = processed_path / filename
        assert filepath.exists(), f"Required file missing: {filepath}"
    
    return processed_path

# ============================================================================
# MODULE FIXTURES  
# ============================================================================

@pytest.fixture(scope="module")
def acoustic_ml_module():
    """
    Import acoustic_ml module.
    
    Scope: module (import once per test module)
    Validates: Module can be imported
    """
    try:
        import acoustic_ml
        return acoustic_ml
    except ImportError as e:
        pytest.fail(f"Failed to import acoustic_ml: {e}")

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

@pytest.fixture(autouse=True)
def test_timer(request):
    """
    Automatically time all tests.
    
    Usage: Runs for every test automatically
    Reports: Slow tests (>2 seconds)
    """
    import time
    
    start_time = time.time()
    yield
    duration = time.time() - start_time
    
    # Warn about slow tests
    if duration > 2.0:
        print(f"\n⚠️  Slow test: {request.node.name} took {duration:.2f}s")

# ============================================================================
# CLEANUP HOOKS
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """
    Cleanup test artifacts after session.
    
    Cleans:
        - Temporary model files
        - Test cache
        - Log files
    """
    yield
    
    # Cleanup after all tests complete
    test_artifacts = [
        PROJECT_ROOT / ".pytest_cache",
        PROJECT_ROOT / "__pycache__",
    ]
    
    # Note: Actual cleanup can be added here if needed
    # For now, we just yield to let tests run

# ============================================================================
# DOCUMENTATION
# ============================================================================

def pytest_report_header(config):
    """
    Add custom header to pytest report.
    
    Displays:
        - Project information
        - Python version
        - Test environment
    """
    return [
        "=" * 70,
        "MLOps Team 24 - Turkish Music Emotion Recognition",
        "Test Suite: Professional MLOps Integration Tests",
        f"Project Root: {PROJECT_ROOT}",
        "=" * 70,
    ]
