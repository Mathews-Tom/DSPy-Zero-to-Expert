#!/usr/bin/env python3
"""
Custom DSPy Module Development Framework

This module provides templates, patterns, and utilities for creating custom DSPy modules.
It includes base classes, validation frameworks, and documentation generation tools
to help developers build robust and reusable DSPy components.

Key Features:
- Base classes for custom DSPy modules
- Module validation and testing framework
- Documentation generation utilities
- Best practices and design patterns
- Performance optimization tools

Author: DSPy Learning Framework
"""

import inspect
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import dspy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModuleMetadata:
    """Metadata for custom DSPy modules"""

    name: str
    version: str
    description: str
    author: str
    tags: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class ModuleValidationResult:
    """Result of module validation"""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)


class CustomModuleBase(ABC):
    """
    Abstract base class for custom DSPy modules

    This class provides a foundation for building custom DSPy modules with
    built-in validation, documentation, and testing capabilities.
    """

    def __init__(self, metadata: ModuleMetadata | None = None):
        """
        Initialize the custom module

        Args:
            metadata: Module metadata information
        """
        self.metadata = metadata or ModuleMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description="Custom DSPy module",
            author="Unknown",
        )
        self.validation_results: list[ModuleValidationResult] = []
        self._performance_history: list[dict[str, float]] = []

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass of the module

        This method must be implemented by subclasses to define
        the core functionality of the custom module.
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """
        Make the module callable

        This method wraps the forward pass with performance tracking
        and validation if enabled.
        """
        start_time = time.time()

        try:
            result = self.forward(*args, **kwargs)
            duration = time.time() - start_time

            # Track performance
            self._performance_history.append(
                {
                    "duration": duration,
                    "timestamp": time.time(),
                    "success": True,
                    "input_size": len(str(args) + str(kwargs)),
                    "output_size": len(str(result)) if result else 0,
                }
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            # Track failed execution
            self._performance_history.append(
                {
                    "duration": duration,
                    "timestamp": time.time(),
                    "success": False,
                    "error": str(e),
                }
            )

            raise

    def validate(self) -> ModuleValidationResult:
        """
        Validate the module implementation

        Returns:
            ModuleValidationResult with validation status and details
        """
        errors = []
        warnings = []

        # Check if forward method is implemented
        if not hasattr(self, "forward") or self.forward == CustomModuleBase.forward:
            errors.append("forward() method must be implemented")

        # Check metadata completeness
        if not self.metadata.name:
            warnings.append("Module name is empty")

        if (
            not self.metadata.description
            or self.metadata.description == "Custom DSPy module"
        ):
            warnings.append("Module description is generic or empty")

        # Check for docstrings
        if not self.__doc__:
            warnings.append("Module class lacks documentation")

        if hasattr(self, "forward") and not self.forward.__doc__:
            warnings.append("forward() method lacks documentation")

        # Performance analysis
        performance_metrics = self._analyze_performance()

        result = ModuleValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            performance_metrics=performance_metrics,
        )

        self.validation_results.append(result)
        return result

    def _analyze_performance(self) -> dict[str, float]:
        """Analyze performance history"""
        if not self._performance_history:
            return {}

        successful_runs = [
            run for run in self._performance_history if run.get("success", False)
        ]

        if not successful_runs:
            return {"success_rate": 0.0}

        durations = [run["duration"] for run in successful_runs]

        return {
            "success_rate": len(successful_runs) / len(self._performance_history),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_runs": len(self._performance_history),
        }

    def get_documentation(self) -> dict[str, Any]:
        """
        Generate comprehensive documentation for the module

        Returns:
            Dictionary containing module documentation
        """
        doc = {
            "metadata": {
                "name": self.metadata.name,
                "version": self.metadata.version,
                "description": self.metadata.description,
                "author": self.metadata.author,
                "tags": self.metadata.tags,
                "requirements": self.metadata.requirements,
            },
            "class_info": {
                "name": self.__class__.__name__,
                "module": self.__class__.__module__,
                "docstring": self.__doc__ or "No documentation available",
                "mro": [cls.__name__ for cls in self.__class__.__mro__],
            },
            "methods": {},
            "performance": self._analyze_performance(),
            "validation": (
                self.validation_results[-1].__dict__
                if self.validation_results
                else None
            ),
        }

        # Document methods
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith("_"):
                doc["methods"][name] = {
                    "docstring": method.__doc__ or "No documentation available",
                    "signature": str(inspect.signature(method)),
                }

        return doc

    def export_config(self) -> dict[str, Any]:
        """
        Export module configuration for serialization

        Returns:
            Dictionary containing module configuration
        """
        return {
            "class_name": self.__class__.__name__,
            "module_name": self.__class__.__module__,
            "metadata": {
                "name": self.metadata.name,
                "version": self.metadata.version,
                "description": self.metadata.description,
                "author": self.metadata.author,
                "tags": self.metadata.tags,
                "requirements": self.metadata.requirements,
            },
            "config_timestamp": time.time(),
        }


class DSPyModuleTemplate(CustomModuleBase):
    """
    Template for creating DSPy-compatible custom modules

    This template provides a structure for creating modules that integrate
    seamlessly with DSPy's signature and prediction system.
    """

    def __init__(
        self, signature: type[dspy.Signature], metadata: ModuleMetadata | None = None
    ):
        """
        Initialize DSPy module template

        Args:
            signature: DSPy signature class defining inputs and outputs
            metadata: Module metadata information
        """
        super().__init__(metadata)
        self.signature = signature
        self.predictor = dspy.Predict(signature)

    def forward(self, **kwargs) -> Any:
        """
        Forward pass using DSPy predictor

        Args:
            **kwargs: Input arguments matching the signature

        Returns:
            Prediction result from DSPy
        """
        return self.predictor(**kwargs)

    def get_signature_info(self) -> dict[str, Any]:
        """
        Get information about the module's signature

        Returns:
            Dictionary containing signature information
        """
        return {
            "signature_name": self.signature.__name__,
            "signature_doc": self.signature.__doc__ or "No documentation available",
            "input_fields": {
                name: {
                    "description": field.desc,
                    "prefix": getattr(field, "prefix", None),
                }
                for name, field in self.signature.input_fields.items()
            },
            "output_fields": {
                name: {
                    "description": field.desc,
                    "prefix": getattr(field, "prefix", None),
                }
                for name, field in self.signature.output_fields.items()
            },
        }


class ModuleValidator:
    """
    Comprehensive validation framework for custom DSPy modules
    """

    def __init__(self):
        self.validation_rules = []
        self.test_cases = []

    def add_validation_rule(self, rule_func: callable, description: str):
        """
        Add a custom validation rule

        Args:
            rule_func: Function that takes a module and returns (bool, str)
            description: Description of the validation rule
        """
        self.validation_rules.append(
            {"function": rule_func, "description": description}
        )

    def add_test_case(
        self,
        inputs: dict[str, Any],
        expected_output: Any = None,
        should_succeed: bool = True,
    ):
        """
        Add a test case for module validation

        Args:
            inputs: Input arguments for the module
            expected_output: Expected output (optional)
            should_succeed: Whether the test should succeed
        """
        self.test_cases.append(
            {
                "inputs": inputs,
                "expected_output": expected_output,
                "should_succeed": should_succeed,
            }
        )

    def validate_module(self, module: CustomModuleBase) -> ModuleValidationResult:
        """
        Perform comprehensive validation of a custom module

        Args:
            module: Module to validate

        Returns:
            ModuleValidationResult with detailed validation information
        """
        errors = []
        warnings = []
        performance_metrics = {}

        # Run basic validation
        basic_result = module.validate()
        errors.extend(basic_result.errors)
        warnings.extend(basic_result.warnings)
        performance_metrics.update(basic_result.performance_metrics)

        # Run custom validation rules
        for rule in self.validation_rules:
            try:
                is_valid, message = rule["function"](module)
                if not is_valid:
                    errors.append(f"{rule['description']}: {message}")
            except Exception as e:
                warnings.append(
                    f"Validation rule '{rule['description']}' failed: {str(e)}"
                )

        # Run test cases
        test_results = self._run_test_cases(module)
        if test_results["failed_tests"]:
            errors.extend(
                [f"Test case failed: {test}" for test in test_results["failed_tests"]]
            )

        performance_metrics.update(test_results["performance"])

        return ModuleValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            performance_metrics=performance_metrics,
        )

    def _run_test_cases(self, module: CustomModuleBase) -> dict[str, Any]:
        """Run test cases against the module"""
        failed_tests = []
        durations = []

        for i, test_case in enumerate(self.test_cases):
            try:
                start_time = time.time()
                result = module(**test_case["inputs"])
                duration = time.time() - start_time
                durations.append(duration)

                # Check if test should succeed
                if not test_case["should_succeed"]:
                    failed_tests.append(f"Test {i+1}: Expected failure but succeeded")

                # Check expected output if provided
                if test_case["expected_output"] is not None:
                    if result != test_case["expected_output"]:
                        failed_tests.append(f"Test {i+1}: Output mismatch")

            except Exception as e:
                duration = time.time() - start_time
                durations.append(duration)

                if test_case["should_succeed"]:
                    failed_tests.append(f"Test {i+1}: Unexpected failure - {str(e)}")

        return {
            "failed_tests": failed_tests,
            "performance": {
                "test_avg_duration": (
                    sum(durations) / len(durations) if durations else 0
                ),
                "test_total_duration": sum(durations),
                "tests_run": len(self.test_cases),
            },
        }


class DocumentationGenerator:
    """
    Automatic documentation generation for custom DSPy modules
    """

    def __init__(self):
        self.templates = {
            "markdown": self._generate_markdown,
            "json": self._generate_json,
            "html": self._generate_html,
        }

    def generate_documentation(
        self, module: CustomModuleBase, format: str = "markdown"
    ) -> str:
        """
        Generate documentation for a custom module

        Args:
            module: Module to document
            format: Output format ('markdown', 'json', 'html')

        Returns:
            Generated documentation as string
        """
        if format not in self.templates:
            raise ValueError(f"Unsupported format: {format}")

        doc_data = module.get_documentation()
        return self.templates[format](doc_data)

    def _generate_markdown(self, doc_data: dict[str, Any]) -> str:
        """Generate Markdown documentation"""
        md = []

        # Header
        md.append(f"# {doc_data['metadata']['name']}")
        md.append(f"**Version:** {doc_data['metadata']['version']}")
        md.append(f"**Author:** {doc_data['metadata']['author']}")
        md.append("")
        md.append(doc_data["metadata"]["description"])
        md.append("")

        # Tags
        if doc_data["metadata"]["tags"]:
            md.append("**Tags:** " + ", ".join(doc_data["metadata"]["tags"]))
            md.append("")

        # Class Information
        md.append("## Class Information")
        md.append(f"**Class Name:** `{doc_data['class_info']['name']}`")
        md.append(f"**Module:** `{doc_data['class_info']['module']}`")
        md.append("")
        md.append("### Description")
        md.append(doc_data["class_info"]["docstring"])
        md.append("")

        # Methods
        if doc_data["methods"]:
            md.append("## Methods")
            for method_name, method_info in doc_data["methods"].items():
                md.append(f"### `{method_name}{method_info['signature']}`")
                md.append(method_info["docstring"])
                md.append("")

        # Performance
        if doc_data["performance"]:
            md.append("## Performance Metrics")
            for metric, value in doc_data["performance"].items():
                md.append(f"- **{metric.replace('_', ' ').title()}:** {value}")
            md.append("")

        # Validation
        if doc_data["validation"]:
            md.append("## Validation Status")
            md.append(
                f"**Valid:** {'✅ Yes' if doc_data['validation']['is_valid'] else '❌ No'}"
            )

            if doc_data["validation"]["errors"]:
                md.append("### Errors")
                for error in doc_data["validation"]["errors"]:
                    md.append(f"- {error}")
                md.append("")

            if doc_data["validation"]["warnings"]:
                md.append("### Warnings")
                for warning in doc_data["validation"]["warnings"]:
                    md.append(f"- {warning}")
                md.append("")

        return "\n".join(md)

    def _generate_json(self, doc_data: dict[str, Any]) -> str:
        """Generate JSON documentation"""
        return json.dumps(doc_data, indent=2, default=str)

    def _generate_html(self, doc_data: dict[str, Any]) -> str:
        """Generate HTML documentation"""
        html = []

        html.append("<!DOCTYPE html>")
        html.append(
            "<html><head><title>{}</title></head><body>".format(
                doc_data["metadata"]["name"]
            )
        )

        # Convert markdown to basic HTML
        markdown_doc = self._generate_markdown(doc_data)

        # Simple markdown to HTML conversion
        html_content = markdown_doc.replace("# ", "<h1>").replace("\n", "</h1>\n", 1)
        html_content = html_content.replace("## ", "<h2>").replace("\n", "</h2>\n")
        html_content = html_content.replace("### ", "<h3>").replace("\n", "</h3>\n")
        html_content = html_content.replace("**", "<strong>", 1).replace(
            "**", "</strong>", 1
        )
        html_content = html_content.replace("- ", "<li>").replace("\n", "</li>\n")

        html.append(html_content)
        html.append("</body></html>")

        return "\n".join(html)


def create_module_template(
    name: str,
    signature_class: type[dspy.Signature] | None = None,
    description: str = "",
    author: str = "Unknown",
) -> str:
    """
    Generate a Python code template for a custom DSPy module

    Args:
        name: Name of the module class
        signature_class: Optional DSPy signature class
        description: Module description
        author: Module author

    Returns:
        Python code template as string
    """
    template = f'''"""
{name} - Custom DSPy Module

{description}

Author: {author}
"""

import dspy
from typing import Any, dict, Optional
from custom_module_template import CustomModuleBase, ModuleMetadata


class {name}(CustomModuleBase):
    """
    {description}

    This is a custom DSPy module that extends CustomModuleBase
    to provide specialized functionality.
    """

    def __init__(self, **config):
        """
        Initialize {name}

        Args:
            **config: Configuration parameters for the module
        """
        metadata = ModuleMetadata(
            name="{name}",
            version="1.0.0",
            description="{description}",
            author="{author}"
        )

        super().__init__(metadata)

        # Initialize your module-specific attributes here
        self.config = config

        # If using a DSPy signature, initialize predictor
        # self.signature = YourSignatureClass
        # self.predictor = dspy.Predict(self.signature)

    def forward(self, **kwargs) -> Any:
        """
        Forward pass of the {name} module

        Args:
            **kwargs: Input arguments

        Returns:
            Module output
        """
        # Implement your module logic here
        # Example:
        # return self.predictor(**kwargs)

        raise NotImplementedError("Implement the forward method")

    def configure(self, **new_config):
        """
        Update module configuration

        Args:
            **new_config: New configuration parameters
        """
        self.config.update(new_config)

    def get_config(self) -> dict[str, Any]:
        """
        Get current module configuration

        Returns:
            Current configuration dictionary
        """
        return self.config.copy()


# Example usage:
if __name__ == "__main__":
    # Create and test your module
    module = {name}()

    # Validate the module
    validation_result = module.validate()
    print(f"Module valid: {{validation_result.is_valid}}")

    if validation_result.errors:
        print("Errors:")
        for error in validation_result.errors:
            print(f"  - {{error}}")

    if validation_result.warnings:
        print("Warnings:")
        for warning in validation_result.warnings:
            print(f"  - {{warning}}")
'''

    return template


def demonstrate_custom_module_framework():
    """
    Demonstrate the custom module development framework
    """
    print("=== Custom DSPy Module Development Framework Demo ===\n")

    # Example 1: Create a simple custom module
    print("1. Creating a simple custom module:")

    class SimpleTextProcessor(CustomModuleBase):
        """A simple text processing module"""

        def __init__(self):
            metadata = ModuleMetadata(
                name="SimpleTextProcessor",
                version="1.0.0",
                description="Processes text by converting to uppercase",
                author="Demo Author",
                tags=["text", "processing", "demo"],
            )
            super().__init__(metadata)

        def forward(self, text: str) -> str:
            """Convert text to uppercase"""
            return text.upper()

    # Test the simple module
    simple_module = SimpleTextProcessor()
    result = simple_module("hello world")
    print(f"Simple module result: {result}")

    # Validate the module
    validation = simple_module.validate()
    print(f"Validation passed: {validation.is_valid}")
    print(f"Performance metrics: {validation.performance_metrics}")
    print()

    # Example 2: Create a DSPy-compatible module
    print("2. Creating a DSPy-compatible module:")

    class SentimentSignature(dspy.Signature):
        """Analyze sentiment of text"""

        text = dspy.InputField(desc="Text to analyze")
        sentiment = dspy.OutputField(desc="Sentiment (positive/negative/neutral)")

    class SentimentAnalyzer(DSPyModuleTemplate):
        """Sentiment analysis module using DSPy"""

        def __init__(self):
            metadata = ModuleMetadata(
                name="SentimentAnalyzer",
                version="1.0.0",
                description="Analyzes sentiment using DSPy",
                author="Demo Author",
                tags=["sentiment", "analysis", "nlp"],
            )
            super().__init__(SentimentSignature, metadata)

    # Note: This would require actual DSPy configuration to work
    # sentiment_module = SentimentAnalyzer()
    print("SentimentAnalyzer created (requires DSPy configuration to run)")
    print()

    # Example 3: Module validation with custom rules
    print("3. Advanced module validation:")

    validator = ModuleValidator()

    # Add custom validation rule
    def check_metadata_tags(module):
        if not module.metadata.tags:
            return False, "Module should have at least one tag"
        return True, "Tags are present"

    validator.add_validation_rule(check_metadata_tags, "Metadata tags check")

    # Add test cases
    validator.add_test_case({"text": "hello"}, "HELLO")
    validator.add_test_case({"text": "world"}, "WORLD")

    # Validate with custom rules
    advanced_validation = validator.validate_module(simple_module)
    print(f"Advanced validation passed: {advanced_validation.is_valid}")
    print(f"Errors: {advanced_validation.errors}")
    print(f"Warnings: {advanced_validation.warnings}")
    print()

    # Example 4: Documentation generation
    print("4. Automatic documentation generation:")

    doc_generator = DocumentationGenerator()

    # Generate markdown documentation
    markdown_doc = doc_generator.generate_documentation(simple_module, "markdown")
    print("Generated Markdown Documentation:")
    print("=" * 50)
    print(markdown_doc[:500] + "..." if len(markdown_doc) > 500 else markdown_doc)
    print("=" * 50)
    print()

    # Example 5: Module template generation
    print("5. Module template generation:")

    template_code = create_module_template(
        name="MyCustomModule",
        description="A custom module for specific functionality",
        author="Your Name",
    )

    print("Generated module template:")
    print("=" * 50)
    print(template_code[:800] + "..." if len(template_code) > 800 else template_code)
    print("=" * 50)

    print("\n✅ Custom module framework demonstration completed!")
    print("\nKey takeaways:")
    print("- CustomModuleBase provides foundation for custom modules")
    print("- DSPyModuleTemplate integrates with DSPy signatures")
    print("- ModuleValidator enables comprehensive testing")
    print("- DocumentationGenerator creates automatic docs")
    print("- Template generation speeds up development")


if __name__ == "__main__":
    demonstrate_custom_module_framework()
