"""
Common utilities and shared components for the DSPy Zero-to-Expert learning repository.

This package provides reusable components, configuration management, and utility functions
that are used across all learning modules.
"""

__version__ = "0.1.0"
__author__ = "Tom Mathews"

# Import key components for easy access
from .config import Config, configure_logging, get_config, reload_config

# Import DSPy extensions
from .dspy_extensions import (
    ModuleChain,
    OptimizationTracker,
    ReactiveModule,
    SignatureTester,
    configure_dspy_lm,
    make_reactive,
    setup_dspy_environment,
)

# Import evaluation utilities
from .evaluation_utils import (
    ABTestConfig,
    ABTester,
    ContainsMetric,
    CustomMetric,
    EvaluationSuite,
    ExactMatchMetric,
    NumericMetric,
    PerformanceBenchmark,
    create_evaluation_suite,
    quick_evaluate,
)

# Import Marimo components
from .marimo_components import (
    ComparisonViewer,
    DSPyParameterPanel,
    DSPyResultViewer,
    OptimizationProgressViewer,
    SignatureBuilder,
)
from .marimo_components import SignatureTester as MarimoSignatureTester
from .utils import (
    ConfigurationError,
    DSPyLearningError,
    ProgressTracker,
    ValidationError,
    ensure_directory,
    get_project_root,
    setup_logging,
    timer,
    validate_environment,
)

__all__ = [
    # Configuration
    "Config",
    "get_config",
    "reload_config",
    "configure_logging",
    # Utilities
    "setup_logging",
    "validate_environment",
    "get_project_root",
    "ensure_directory",
    "timer",
    "ProgressTracker",
    # Exceptions
    "DSPyLearningError",
    "ConfigurationError",
    "ValidationError",
    # Marimo Components
    "DSPyParameterPanel",
    "SignatureBuilder",
    "DSPyResultViewer",
    "OptimizationProgressViewer",
    "ComparisonViewer",
    "MarimoSignatureTester",
    # DSPy Extensions
    "ReactiveModule",
    "make_reactive",
    "SignatureTester",
    "OptimizationTracker",
    "configure_dspy_lm",
    "setup_dspy_environment",
    "ModuleChain",
    # Evaluation
    "EvaluationSuite",
    "ExactMatchMetric",
    "ContainsMetric",
    "NumericMetric",
    "CustomMetric",
    "ABTester",
    "ABTestConfig",
    "PerformanceBenchmark",
    "create_evaluation_suite",
    "quick_evaluate",
]
