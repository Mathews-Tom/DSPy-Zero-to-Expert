"""
Shared utility functions for the DSPy Zero-to-Expert learning repository.

This module provides common utility functions used across all learning modules.
"""

# Standard Library
import importlib
import json
import logging
import sys
import time
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# Local Modules
from .config import Config, get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Environment and Setup Utilities
# =============================================================================


def validate_environment() -> Dict[str, bool]:
    """
    Validate the current environment setup.

    Returns:
        Dictionary with validation results for different components.
    """
    results = {}

    # Check Python version
    results["python_version"] = sys.version_info >= (3, 11)

    # Check required packages
    required_packages = [
        "dspy",
        "marimo",
        "openai",
        "anthropic",
        "numpy",
        "pandas",
        "matplotlib",
        "plotly",
        "sklearn",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)

    results["packages"] = len(missing_packages) == 0
    results["missing_packages"] = missing_packages

    # Check configuration
    config = get_config()
    results["has_llm_config"] = len(config.get_available_llm_providers()) > 0

    return results


def setup_logging(
    level: Optional[str] = None, format_type: Optional[str] = None
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type (simple, detailed, json)
    """
    config = get_config()

    if level:
        config.log_level = level
    if format_type:
        config.log_format = format_type

    from .config import configure_logging

    configure_logging(config)


def check_api_connectivity(provider: str = "openai", timeout: int = 10) -> bool:
    """
    Check if we can connect to the specified LLM provider.

    Args:
        provider: LLM provider name
        timeout: Timeout in seconds

    Returns:
        True if connection is successful, False otherwise
    """
    config = get_config()

    try:
        if provider == "openai" and config.has_openai_config():
            import openai

            client = openai.OpenAI(api_key=config.openai_api_key)
            # Simple API call to check connectivity
            client.models.list()
            return True

        elif provider == "anthropic" and config.has_anthropic_config():
            import anthropic

            client = anthropic.Anthropic(api_key=config.anthropic_api_key)
            # Simple API call to check connectivity
            client.models.list()
            return True

        elif provider == "cohere" and config.has_cohere_config():
            import cohere

            client = cohere.Client(config.cohere_api_key)
            # Simple API call to check connectivity
            client.models.list()
            return True

    except Exception as e:
        logger.warning(f"Failed to connect to {provider}: {e}")
        return False

    return False


# =============================================================================
# File and Path Utilities
# =============================================================================


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_module_path(module_name: str) -> Path:
    """
    Get the path to a specific learning module.

    Args:
        module_name: Module name (e.g., "00-setup", "01-foundations")

    Returns:
        Path to the module directory
    """
    return get_project_root() / module_name


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file safely.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {file_path}: {e}")


def save_json_file(
    data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2
) -> None:
    """
    Save data to a JSON file safely.

    Args:
        data: Data to save
        file_path: Path to save the file
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# =============================================================================
# Performance and Timing Utilities
# =============================================================================


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")

        return result

    return wrapper


def measure_memory_usage(func: Callable) -> Callable:
    """
    Decorator to measure memory usage of a function.

    Args:
        func: Function to measure

    Returns:
        Wrapped function that logs memory usage
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            result = func(*args, **kwargs)

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = memory_after - memory_before

            logger.info(f"{func.__name__} memory usage: {memory_diff:.2f} MB")

            return result

        except ImportError:
            logger.warning("psutil not available, skipping memory measurement")
            return func(*args, **kwargs)

    return wrapper


class ProgressTracker:
    """Simple progress tracker for long-running operations."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, increment: int = 1) -> None:
        """Update progress by the specified increment."""
        self.current += increment
        self._print_progress()

    def _print_progress(self) -> None:
        """Print current progress."""
        if self.total == 0:
            return

        percentage = (self.current / self.total) * 100
        elapsed_time = time.time() - self.start_time

        if self.current > 0:
            eta = (elapsed_time / self.current) * (self.total - self.current)
            eta_str = f"ETA: {timedelta(seconds=int(eta))}"
        else:
            eta_str = "ETA: --:--:--"

        print(
            f"\r{self.description}: {self.current}/{self.total} "
            f"({percentage:.1f}%) {eta_str}",
            end="",
            flush=True,
        )

        if self.current >= self.total:
            print()  # New line when complete


# =============================================================================
# Data Processing Utilities
# =============================================================================


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.

    Args:
        a: Numerator
        b: Denominator
        default: Default value if b is zero

    Returns:
        Result of division or default value
    """
    return a / b if b != 0 else default


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


# =============================================================================
# String and Text Utilities
# =============================================================================


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    import re

    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip("_.")
    return sanitized


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# =============================================================================
# Validation Utilities
# =============================================================================


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    import re

    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def validate_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid, False otherwise
    """
    import re

    pattern = r"^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$"
    return re.match(pattern, url) is not None


# =============================================================================
# System Information Utilities
# =============================================================================


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and diagnostics.

    Returns:
        Dictionary with system information
    """
    import platform

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
    }

    # Add memory information if available
    try:
        import psutil

        memory = psutil.virtual_memory()
        info["memory_total"] = f"{memory.total / (1024**3):.1f} GB"
        info["memory_available"] = f"{memory.available / (1024**3):.1f} GB"
        info["memory_percent"] = f"{memory.percent}%"
    except ImportError:
        info["memory"] = "psutil not available"

    return info


def get_package_versions() -> Dict[str, str]:
    """
    Get versions of key packages.

    Returns:
        Dictionary mapping package names to versions
    """
    packages = [
        "dspy",
        "marimo",
        "openai",
        "anthropic",
        "cohere",
        "numpy",
        "pandas",
        "matplotlib",
        "plotly",
        "sklearn",
        "transformers",
        "datasets",
        "faiss",
        "chromadb",
        "qdrant_client",
    ]

    versions = {}
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            versions[package] = version
        except ImportError:
            versions[package] = "not installed"

    return versions


# =============================================================================
# Error Handling Utilities
# =============================================================================


class DSPyLearningError(Exception):
    """Base exception for DSPy learning repository."""

    pass


class ConfigurationError(DSPyLearningError):
    """Raised when there's a configuration error."""

    pass


class ValidationError(DSPyLearningError):
    """Raised when validation fails."""

    pass


def handle_api_error(func: Callable) -> Callable:
    """
    Decorator to handle common API errors gracefully.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with error handling
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"API error in {func.__name__}: {e}")
            # You might want to implement retry logic here
            raise

    return wrapper
