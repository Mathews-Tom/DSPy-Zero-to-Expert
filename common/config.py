"""
Configuration management for the DSPy Zero-to-Expert learning repository.

This module provides centralized configuration management using Pydantic settings
with support for environment variables and .env files.
"""

# Standard Library
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-Party Library
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class Config(BaseSettings):
    """
    Main configuration class for the DSPy learning repository.

    This class uses Pydantic settings to automatically load configuration
    from environment variables and .env files.
    """

    # =============================================================================
    # Environment Settings
    # =============================================================================

    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Enable debug mode")

    # =============================================================================
    # LLM Provider Configuration
    # =============================================================================

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_org_id: Optional[str] = Field(
        default=None, description="OpenAI organization ID"
    )

    # Anthropic
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )

    # Cohere
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key")

    # Default LLM settings
    default_llm_provider: str = Field(
        default="openai", description="Default LLM provider"
    )
    default_model: str = Field(default="gpt-4o-mini", description="Default model name")

    # =============================================================================
    # Search and Tool APIs
    # =============================================================================

    tavily_api_key: Optional[str] = Field(
        default=None, description="Tavily search API key"
    )

    # =============================================================================
    # Observability and Monitoring
    # =============================================================================

    # Langfuse
    langfuse_public_key: Optional[str] = Field(
        default=None, description="Langfuse public key"
    )
    langfuse_secret_key: Optional[str] = Field(
        default=None, description="Langfuse secret key"
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", description="Langfuse host"
    )

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", description="MLflow tracking URI"
    )
    mlflow_experiment_name: str = Field(
        default="dspy-learning", description="MLflow experiment name"
    )

    # =============================================================================
    # Vector Database Configuration
    # =============================================================================

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")

    # ChromaDB
    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8000, description="ChromaDB port")

    # =============================================================================
    # Application Settings
    # =============================================================================

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="detailed", description="Log format (simple, detailed, json)"
    )

    # Cache
    enable_cache: bool = Field(default=True, description="Enable caching")
    cache_dir: str = Field(default=".cache", description="Cache directory")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # Marimo
    marimo_host: str = Field(default="localhost", description="Marimo host")
    marimo_port: int = Field(default=2718, description="Marimo port")
    marimo_auto_reload: bool = Field(default=True, description="Marimo auto-reload")

    # =============================================================================
    # Development Settings
    # =============================================================================

    # Performance monitoring
    enable_profiling: bool = Field(
        default=False, description="Enable performance profiling"
    )
    profile_output_dir: str = Field(
        default="profiles", description="Profile output directory"
    )

    # Testing
    test_api_calls: bool = Field(
        default=False, description="Make actual API calls during tests"
    )
    test_timeout: int = Field(default=30, description="Test timeout in seconds")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v):
        """Validate log format."""
        valid_formats = ["simple", "detailed", "json"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Log format must be one of: {valid_formats}")
        return v.lower()

    @field_validator("default_llm_provider")
    @classmethod
    def validate_llm_provider(cls, v):
        """Validate LLM provider."""
        valid_providers = ["openai", "anthropic", "cohere"]
        if v.lower() not in valid_providers:
            raise ValueError(f"LLM provider must be one of: {valid_providers}")
        return v.lower()

    def get_cache_dir(self) -> Path:
        """Get the cache directory as a Path object."""
        cache_path = Path(self.cache_dir)
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def get_profile_dir(self) -> Path:
        """Get the profile output directory as a Path object."""
        profile_path = Path(self.profile_output_dir)
        if not profile_path.is_absolute():
            profile_path = PROJECT_ROOT / profile_path
        profile_path.mkdir(parents=True, exist_ok=True)
        return profile_path

    def has_openai_config(self) -> bool:
        """Check if OpenAI is properly configured."""
        return self.openai_api_key is not None

    def has_anthropic_config(self) -> bool:
        """Check if Anthropic is properly configured."""
        return self.anthropic_api_key is not None

    def has_cohere_config(self) -> bool:
        """Check if Cohere is properly configured."""
        return self.cohere_api_key is not None

    def has_tavily_config(self) -> bool:
        """Check if Tavily is properly configured."""
        return self.tavily_api_key is not None

    def has_langfuse_config(self) -> bool:
        """Check if Langfuse is properly configured."""
        return (
            self.langfuse_public_key is not None
            and self.langfuse_secret_key is not None
        )

    def get_available_llm_providers(self) -> List[str]:
        """Get list of available LLM providers based on configuration."""
        providers = []
        if self.has_openai_config():
            providers.append("openai")
        if self.has_anthropic_config():
            providers.append("anthropic")
        if self.has_cohere_config():
            providers.append("cohere")
        return providers

    def get_llm_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM configuration for the specified provider."""
        if provider is None:
            provider = self.default_llm_provider

        provider = provider.lower()

        if provider == "openai":
            if not self.has_openai_config():
                raise ValueError("OpenAI API key not configured")
            config = {"api_key": self.openai_api_key}
            if self.openai_org_id:
                config["organization"] = self.openai_org_id
            return config

        elif provider == "anthropic":
            if not self.has_anthropic_config():
                raise ValueError("Anthropic API key not configured")
            return {"api_key": self.anthropic_api_key}

        elif provider == "cohere":
            if not self.has_cohere_config():
                raise ValueError("Cohere API key not configured")
            return {"api_key": self.cohere_api_key}

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary, excluding sensitive data."""
        data = self.dict()

        # Mask sensitive keys
        sensitive_keys = [
            "openai_api_key",
            "anthropic_api_key",
            "cohere_api_key",
            "tavily_api_key",
            "langfuse_public_key",
            "langfuse_secret_key",
            "qdrant_api_key",
        ]

        for key in sensitive_keys:
            if key in data and data[key]:
                data[key] = "***masked***"

        return data


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    This function implements a singleton pattern to ensure we only
    load the configuration once.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """
    Reload the configuration from environment variables and .env file.

    This is useful during development or testing when you want to
    pick up configuration changes.
    """
    global _config
    _config = Config()
    return _config


def set_config(config: Config) -> None:
    """
    Set the global configuration instance.

    This is primarily used for testing to inject a custom configuration.
    """
    global _config
    _config = config


def configure_logging(config: Optional[Config] = None) -> None:
    """
    Configure logging based on the current configuration.

    Args:
        config: Optional configuration instance. If not provided,
                uses the global configuration.
    """
    if config is None:
        config = get_config()

    # Set up logging format
    if config.log_format == "simple":
        log_format = "%(levelname)s: %(message)s"
    elif config.log_format == "detailed":
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    elif config.log_format == "json":
        # For JSON logging, you might want to use a structured logging library
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set specific logger levels
    if config.debug:
        logging.getLogger("dspy").setLevel(logging.DEBUG)
        logging.getLogger("marimo").setLevel(logging.DEBUG)
    else:
        # Reduce noise from third-party libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)


# Initialize logging when the module is imported
configure_logging()
