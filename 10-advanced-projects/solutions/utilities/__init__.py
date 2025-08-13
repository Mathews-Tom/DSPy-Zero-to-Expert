"""
DSPy Learning Framework - Solution Utilities

This package provides shared utilities and helper functions for all solution scripts.
These utilities simplify common tasks and provide consistent functionality across
all exercises and projects.

Author: DSPy Learning Framework
"""

from .dspy_helpers import (
    benchmark_signature_performance,
    create_test_signature,
    run_signature_test,
    setup_dspy_environment,
    validate_signature_output,
)
from .evaluation_helpers import (
    compare_model_performance,
    comprehensive_evaluation,
    create_evaluation_dataset,
    generate_evaluation_report,
    run_evaluation_suite,
)
from .testing_helpers import (
    create_test_suite,
    integration_test,
    performance_test,
    run_unit_tests,
    validate_implementation,
)

__all__ = [
    # DSPy Helpers
    "setup_dspy_environment",
    "create_test_signature",
    "run_signature_test",
    "benchmark_signature_performance",
    "validate_signature_output",
    # Evaluation Helpers
    "comprehensive_evaluation",
    "create_evaluation_dataset",
    "run_evaluation_suite",
    "generate_evaluation_report",
    "compare_model_performance",
    # Testing Helpers
    "create_test_suite",
    "run_unit_tests",
    "validate_implementation",
    "performance_test",
    "integration_test",
]
