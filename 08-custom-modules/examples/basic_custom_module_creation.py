#!/usr/bin/env python3
"""
Basic Custom Module Creation Example

This example demonstrates how to create a basic custom DSPy module from scratch,
including proper inheritance, initialization, and implementation of core methods.

Learning Objectives:
- Create a custom DSPy module inheriting from CustomModuleBase
- Implement required methods (forward, validation)
- Add metadata and performance tracking
- Test the module with various inputs

Author: DSPy Learning Framework
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import logging
import time

# Configure DSPy with available API keys
from dspy_config import configure_dspy_lm, get_configured_model_info, is_dspy_configured

dspy_configured = configure_dspy_lm()
if dspy_configured:
    model_info = get_configured_model_info()
    print(f"✅ DSPy configured with {model_info['provider']} {model_info['model']}")
else:
    print("ℹ️  Using fallback implementations (no API keys configured)")

import dspy
from custom_module_template import CustomModuleBase, ModuleMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicTextAnalyzer(CustomModuleBase):
    """
    A basic custom DSPy module for text analysis.

    This module demonstrates the fundamental structure of a custom DSPy module,
    including proper initialization, metadata management, and core functionality.
    """

    def __init__(self, analysis_type: str = "basic"):
        """
        Initialize the BasicTextAnalyzer module.

        Args:
            analysis_type: Type of analysis to perform ("basic", "detailed", "summary")
        """
        # Create metadata for the module
        metadata = ModuleMetadata(
            name="Basic Text Analyzer",
            description="A simple text analysis module for learning custom DSPy module creation",
            version="1.0.0",
            author="DSPy Learning Framework",
            tags=["text", "analysis", "basic", "learning"],
            requirements=["dspy>=2.0.0"],
        )

        # Initialize parent class
        super().__init__(metadata)

        # Set module-specific parameters
        self.analysis_type = analysis_type

        # Create DSPy signature for text analysis
        class TextAnalysisSignature(dspy.Signature):
            """Analyze the given text and provide insights"""

            text = dspy.InputField(desc="Text to analyze")
            analysis_type = dspy.InputField(desc="Type of analysis to perform")
            word_count = dspy.OutputField(desc="Number of words in the text")
            character_count = dspy.OutputField(desc="Number of characters in the text")
            analysis_summary = dspy.OutputField(desc="Brief summary of the analysis")

        # Initialize DSPy predictor only if DSPy is properly configured
        try:
            if is_dspy_configured():
                self.analyzer = dspy.ChainOfThought(TextAnalysisSignature)
            else:
                self.analyzer = None
        except Exception as e:
            logger.warning(
                "DSPy predictor initialization failed: %s, using fallback", e
            )
            self.analyzer = None

        # Mark as initialized
        self._initialized = True
        logger.info(
            "BasicTextAnalyzer initialized with analysis_type: %s", analysis_type
        )

    def forward(self, **kwargs) -> dict:
        """
        Forward pass implementation for text analysis.

        Args:
            **kwargs: Input arguments, expecting 'text' key

        Returns:
            dict: Analysis results including word count, character count, and summary
        """
        # Validate inputs
        if "text" not in kwargs:
            raise ValueError("Missing required input: 'text'")

        text = kwargs["text"]

        if not isinstance(text, str):
            raise ValueError("Input 'text' must be a string")

        if not text.strip():
            raise ValueError("Input 'text' cannot be empty or whitespace-only")

        # Perform basic analysis
        word_count = len(text.split())
        character_count = len(text)
        sentence_count = len([s for s in text.split(".") if s.strip()])

        # Try to use DSPy for enhanced analysis
        analysis_summary = ""
        if self.analyzer:
            try:
                result = self.analyzer(text=text, analysis_type=self.analysis_type)
                analysis_summary = result.analysis_summary
            except Exception as e:
                logger.info("DSPy analysis failed: %s, using fallback", e)
                analysis_summary = self._fallback_analysis(text)
        else:
            analysis_summary = self._fallback_analysis(text)

        # Prepare results based on analysis type
        results = {
            "word_count": word_count,
            "character_count": character_count,
            "sentence_count": sentence_count,
            "analysis_type": self.analysis_type,
            "analysis_summary": analysis_summary,
            "text_length_category": self._categorize_text_length(word_count),
            "processing_timestamp": time.time(),
        }

        # Add detailed analysis for specific types
        if self.analysis_type == "detailed":
            results.update(self._detailed_analysis(text))
        elif self.analysis_type == "summary":
            results.update(self._summary_analysis(text))

        return results

    def _fallback_analysis(self, text: str) -> str:
        """
        Fallback analysis when DSPy is not available.

        Args:
            text: Input text to analyze

        Returns:
            str: Basic analysis summary
        """
        word_count = len(text.split())

        if word_count < 10:
            return "Short text with basic content"
        elif word_count < 50:
            return "Medium-length text with moderate content"
        else:
            return "Long text with substantial content"

    def _categorize_text_length(self, word_count: int) -> str:
        """
        Categorize text based on word count.

        Args:
            word_count: Number of words in the text

        Returns:
            str: Length category
        """
        if word_count < 10:
            return "very_short"
        elif word_count < 50:
            return "short"
        elif word_count < 200:
            return "medium"
        elif word_count < 500:
            return "long"
        else:
            return "very_long"

    def _detailed_analysis(self, text: str) -> dict:
        """
        Perform detailed analysis of the text.

        Args:
            text: Input text to analyze

        Returns:
            dict: Detailed analysis results
        """
        import string

        # Character analysis
        punctuation_count = sum(1 for char in text if char in string.punctuation)
        uppercase_count = sum(1 for char in text if char.isupper())
        lowercase_count = sum(1 for char in text if char.islower())
        digit_count = sum(1 for char in text if char.isdigit())

        # Word analysis
        words = text.split()
        unique_words = len({word.lower().strip(string.punctuation) for word in words})
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        return {
            "punctuation_count": punctuation_count,
            "uppercase_count": uppercase_count,
            "lowercase_count": lowercase_count,
            "digit_count": digit_count,
            "unique_words": unique_words,
            "vocabulary_richness": unique_words / len(words) if words else 0,
            "average_word_length": round(avg_word_length, 2),
        }

    def _summary_analysis(self, text: str) -> dict:
        """
        Perform summary analysis of the text.

        Args:
            text: Input text to analyze

        Returns:
            dict: Summary analysis results
        """
        words = text.split()
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        # Calculate readability metrics (simplified)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0

        # Identify key characteristics
        has_questions = "?" in text
        has_exclamations = "!" in text
        has_numbers = any(char.isdigit() for char in text)

        return {
            "average_sentence_length": round(avg_sentence_length, 2),
            "has_questions": has_questions,
            "has_exclamations": has_exclamations,
            "has_numbers": has_numbers,
            "text_complexity": (
                "high"
                if avg_sentence_length > 20
                else "medium" if avg_sentence_length > 10 else "low"
            ),
        }

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate input parameters.

        Args:
            **kwargs: Input arguments to validate

        Returns:
            bool: True if inputs are valid

        Raises:
            ValueError: If inputs are invalid
        """
        if "text" not in kwargs:
            raise ValueError("Missing required input: 'text'")

        text = kwargs["text"]

        if not isinstance(text, str):
            raise ValueError("Input 'text' must be a string")

        if len(text.strip()) == 0:
            raise ValueError("Input 'text' cannot be empty")

        return True

    def get_analysis_capabilities(self) -> dict:
        """
        Get information about the module's analysis capabilities.

        Returns:
            dict: Capabilities information
        """
        return {
            "supported_analysis_types": ["basic", "detailed", "summary"],
            "current_analysis_type": self.analysis_type,
            "features": [
                "word_count",
                "character_count",
                "sentence_count",
                "text_categorization",
                "dspy_integration",
                "fallback_analysis",
            ],
            "detailed_features": (
                [
                    "punctuation_analysis",
                    "character_type_analysis",
                    "vocabulary_analysis",
                    "readability_metrics",
                ]
                if self.analysis_type == "detailed"
                else []
            ),
            "summary_features": (
                [
                    "sentence_length_analysis",
                    "text_characteristics",
                    "complexity_assessment",
                ]
                if self.analysis_type == "summary"
                else []
            ),
        }


def demonstrate_basic_custom_module():
    """Demonstrate the BasicTextAnalyzer custom module"""
    print("=== Basic Custom Module Creation Example ===\n")

    # Test different analysis types
    analysis_types = ["basic", "detailed", "summary"]

    # Sample texts for testing
    test_texts = [
        "Hello, world!",
        "This is a medium-length text that contains multiple sentences. It has various punctuation marks and demonstrates different text characteristics.",
        "What is artificial intelligence? It's a fascinating field! AI systems can process natural language, recognize patterns, and make decisions. The technology has applications in healthcare, finance, education, and many other domains. Researchers continue to push the boundaries of what's possible with machine learning algorithms and neural networks.",
    ]

    for analysis_type in analysis_types:
        print(f"Testing {analysis_type.upper()} Analysis:")
        print("-" * 50)

        # Create analyzer instance
        analyzer = BasicTextAnalyzer(analysis_type=analysis_type)

        # Display module information
        module_info = analyzer.get_documentation()
        print(f"Module: {module_info['metadata']['name']}")
        print(f"Version: {module_info['metadata']['version']}")
        print(f"Description: {module_info['metadata']['description']}")

        # Display capabilities
        capabilities = analyzer.get_analysis_capabilities()
        print(f"Analysis Type: {capabilities['current_analysis_type']}")
        print(f"Features: {', '.join(capabilities['features'])}")

        print("\nTest Results:")

        for i, text in enumerate(test_texts):
            try:
                # Validate inputs
                analyzer.validate_inputs(text=text)

                # Perform analysis
                result = analyzer(text=text)

                print(f"\nTest {i+1}: {text[:50]}{'...' if len(text) > 50 else ''}")
                print(f"  Word Count: {result['word_count']}")
                print(f"  Character Count: {result['character_count']}")
                print(f"  Sentence Count: {result['sentence_count']}")
                print(f"  Length Category: {result['text_length_category']}")
                print(f"  Analysis Summary: {result['analysis_summary']}")

                # Show type-specific results
                if analysis_type == "detailed":
                    print(f"  Unique Words: {result.get('unique_words', 'N/A')}")
                    print(
                        f"  Vocabulary Richness: {result.get('vocabulary_richness', 'N/A'):.2f}"
                    )
                    print(
                        f"  Average Word Length: {result.get('average_word_length', 'N/A')}"
                    )
                elif analysis_type == "summary":
                    print(
                        f"  Average Sentence Length: {result.get('average_sentence_length', 'N/A')}"
                    )
                    print(f"  Text Complexity: {result.get('text_complexity', 'N/A')}")
                    print(f"  Has Questions: {result.get('has_questions', 'N/A')}")

            except Exception as e:
                print(f"\nTest {i+1} failed: {e}")

        # Display performance metrics
        metrics = analyzer._analyze_performance()
        print("\nPerformance Metrics:")
        print(f"  Total Runs: {metrics.get('total_runs', 0)}")
        print(f"  Average Duration: {metrics.get('avg_duration', 0):.4f}s")
        print(f"  Success Rate: {metrics.get('success_rate', 0):.2%}")

        print("\n" + "=" * 60 + "\n")


def test_error_handling():
    """Test error handling capabilities"""
    print("Testing Error Handling:")
    print("-" * 30)

    analyzer = BasicTextAnalyzer()

    # Test cases that should raise errors
    error_test_cases = [
        ({}, "Missing 'text' input"),
        ({"text": None}, "None text input"),
        ({"text": 123}, "Non-string text input"),
        ({"text": ""}, "Empty text input"),
        ({"text": "   "}, "Whitespace-only text input"),
    ]

    for test_input, description in error_test_cases:
        try:
            _ = analyzer(**test_input)
            print(f"❌ {description}: Expected error but got result")
        except Exception as e:
            print(f"✅ {description}: Correctly caught error - {e}")


def test_module_validation():
    """Test module validation using the framework"""
    print("Testing Module Validation:")
    print("-" * 30)

    from custom_module_template import ModuleValidator

    analyzer = BasicTextAnalyzer()

    # Validate the module
    validator = ModuleValidator()
    validation_results = validator.validate_module(analyzer)

    print(f"Module Valid: {validation_results.is_valid}")
    print(f"Performance Metrics: {validation_results.performance_metrics}")

    if validation_results.errors:
        print(f"Errors: {validation_results.errors}")

    if validation_results.warnings:
        print(f"Warnings: {validation_results.warnings}")

    # Test module execution
    test_inputs = [
        {"text": "Hello, world!"},
        {"text": "This is a test sentence."},
        {"text": "Multiple sentences here. With different content!"},
    ]

    print("\nExecution Test Results:")
    passed = 0
    failed = 0

    for i, test_input in enumerate(test_inputs):
        try:
            result = analyzer(**test_input)
            passed += 1
            print(f"  Test {i+1}: ✅ Passed")
        except Exception as e:
            failed += 1
            print(f"  Test {i+1}: ❌ Failed - {e}")

    print(f"  Total Tests: {len(test_inputs)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    """
    Basic Custom Module Creation Example

    This script demonstrates:
    1. Creating a custom DSPy module with proper inheritance
    2. Implementing required methods (forward, validation)
    3. Adding comprehensive metadata and documentation
    4. Integrating DSPy functionality with fallback mechanisms
    5. Testing the module with various inputs and scenarios
    6. Error handling and input validation
    7. Performance tracking and metrics collection
    """

    try:
        demonstrate_basic_custom_module()
        test_error_handling()
        test_module_validation()

        print("✅ Basic Custom Module Creation example completed successfully!")
        print("\nKey Learning Points:")
        print("- Custom modules should inherit from CustomModuleBase")
        print("- Always implement the forward() method for core functionality")
        print("- Include comprehensive metadata for documentation")
        print("- Add input validation to ensure robust operation")
        print("- Integrate DSPy functionality with fallback mechanisms")
        print("- Use performance tracking to monitor module efficiency")
        print("- Test thoroughly with various input scenarios")

    except Exception as e:
        print("\n❌ Basic Custom Module Creation example failed: %s", e)
        logger.exception("Basic Custom Module Creation example execution failed")
