#!/usr/bin/env python3
"""
Module Testing and Validation Strategies Example

This example demonstrates comprehensive testing strategies for custom DSPy modules,
including unit testing, integration testing, performance benchmarking, and quality assessment.

Learning Objectives:
- Create comprehensive test suites for custom modules
- Implement automated testing and validation workflows
- Perform performance benchmarking and load testing
- Assess module quality and reliability
- Generate detailed test reports and metrics

Author: DSPy Learning Framework
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import logging
import statistics
import time
from typing import Any, Dict, List

from component_library import (
    ComponentPipeline,
    NLPComponentSuite,
    SentimentAnalyzerComponent,
    TextCleanerComponent,
    TextSummarizerComponent,
)
from custom_module_template import CustomModuleBase, ModuleMetadata, ModuleValidator
from module_testing_framework import (
    ModuleTestRunner,
    PerformanceBenchmark,
    QualityAssessment,
    TestCase,
    TestCaseGenerator,
    TestResult,
    TestSuite,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestableTextProcessor(CustomModuleBase):
    """A text processor designed specifically for testing demonstrations"""

    def __init__(self, processing_mode: str = "standard", enable_caching: bool = False):
        metadata = ModuleMetadata(
            name="Testable Text Processor",
            description="A text processing module designed for comprehensive testing",
            version="1.0.0",
            author="DSPy Learning Framework",
            tags=["testing", "text", "processor"],
        )
        super().__init__(metadata)

        self.processing_mode = processing_mode
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        self._initialized = True

    def forward(self, **kwargs) -> Dict[str, Any]:
        """Process text with various modes and caching"""
        text = kwargs.get("text", "")

        if not isinstance(text, str):
            raise ValueError("Input 'text' must be a string")

        # Check cache if enabled
        if self.enable_caching and text in self.cache:
            return self.cache[text]

        # Simulate processing time based on mode
        if self.processing_mode == "slow":
            time.sleep(0.1)  # Simulate slow processing
        elif self.processing_mode == "fast":
            time.sleep(0.01)  # Simulate fast processing

        # Process text
        result = self._process_text(text)

        # Cache result if enabled
        if self.enable_caching:
            self.cache[text] = result

        return result

    def _process_text(self, text: str) -> Dict[str, Any]:
        """Internal text processing logic"""
        words = text.split()
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        # Basic statistics
        stats = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "character_count": len(text),
            "average_word_length": (
                sum(len(word) for word in words) / len(words) if words else 0
            ),
        }

        # Mode-specific processing
        if self.processing_mode == "detailed":
            stats.update(self._detailed_processing(text, words))
        elif self.processing_mode == "summary":
            stats.update(self._summary_processing(text, words, sentences))

        # Add processing metadata
        stats.update(
            {
                "processing_mode": self.processing_mode,
                "caching_enabled": self.enable_caching,
                "processed_at": time.time(),
            }
        )

        return stats

    def _detailed_processing(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Detailed processing mode"""
        import string

        # Character analysis
        letters = sum(1 for c in text if c.isalpha())
        digits = sum(1 for c in text if c.isdigit())
        punctuation = sum(1 for c in text if c in string.punctuation)

        # Word analysis
        unique_words = len(
            set(word.lower().strip(string.punctuation) for word in words)
        )

        return {
            "letter_count": letters,
            "digit_count": digits,
            "punctuation_count": punctuation,
            "unique_word_count": unique_words,
            "vocabulary_richness": unique_words / len(words) if words else 0,
        }

    def _summary_processing(
        self, text: str, words: List[str], sentences: List[str]
    ) -> Dict[str, Any]:
        """Summary processing mode"""
        # Text complexity metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0

        # Content characteristics
        has_questions = "?" in text
        has_exclamations = "!" in text

        return {
            "average_sentence_length": avg_sentence_length,
            "has_questions": has_questions,
            "has_exclamations": has_exclamations,
            "complexity_level": (
                "high"
                if avg_sentence_length > 15
                else "medium" if avg_sentence_length > 8 else "low"
            ),
        }

    def clear_cache(self):
        """Clear the processing cache"""
        if self.cache is not None:
            self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache is None:
            return {"caching_enabled": False}

        return {
            "caching_enabled": True,
            "cache_size": len(self.cache),
            "cached_items": list(self.cache.keys())[:5],  # Show first 5 items
        }


def create_comprehensive_test_suite() -> TestSuite:
    """Create a comprehensive test suite for the TestableTextProcessor"""

    # Basic functionality tests
    basic_tests = [
        TestCase(
            name="empty_text_test",
            inputs={"text": ""},
            expected_outputs={
                "word_count": 0,
                "sentence_count": 0,
                "character_count": 0,
            },
            description="Test handling of empty text input",
        ),
        TestCase(
            name="single_word_test",
            inputs={"text": "Hello"},
            expected_outputs={"word_count": 1, "character_count": 5},
            description="Test processing of single word",
        ),
        TestCase(
            name="simple_sentence_test",
            inputs={"text": "Hello, world!"},
            expected_outputs={
                "word_count": 2,
                "sentence_count": 0,  # No period, so no sentences detected
            },
            description="Test processing of simple sentence",
        ),
        TestCase(
            name="multiple_sentences_test",
            inputs={"text": "First sentence. Second sentence. Third sentence."},
            expected_outputs={"sentence_count": 3},
            description="Test processing of multiple sentences",
        ),
    ]

    # Edge case tests
    edge_case_tests = [
        TestCase(
            name="whitespace_only_test",
            inputs={"text": "   \n\t   "},
            expected_outputs={"word_count": 0},
            description="Test handling of whitespace-only input",
        ),
        TestCase(
            name="special_characters_test",
            inputs={"text": "Hello @#$%^&*() World!"},
            validation_func=lambda result, inputs: result["word_count"] == 2,
            description="Test handling of special characters",
        ),
        TestCase(
            name="unicode_text_test",
            inputs={"text": "Hello 世界! 🌍"},
            validation_func=lambda result, inputs: result["character_count"] > 0,
            description="Test handling of Unicode characters",
        ),
        TestCase(
            name="very_long_text_test",
            inputs={"text": "Word " * 1000},
            expected_outputs={"word_count": 1000},
            description="Test handling of very long text",
        ),
    ]

    # Performance tests
    performance_tests = [
        TestCase(
            name="performance_baseline_test",
            inputs={"text": "This is a standard performance test sentence."},
            timeout=1.0,
            tags=["performance"],
            description="Baseline performance test",
        ),
        TestCase(
            name="large_text_performance_test",
            inputs={"text": "Performance test sentence. " * 100},
            timeout=2.0,
            tags=["performance"],
            description="Performance test with large text",
        ),
    ]

    # Combine all tests
    all_tests = basic_tests + edge_case_tests + performance_tests

    return TestSuite(
        name="Comprehensive TestableTextProcessor Test Suite",
        test_cases=all_tests,
        parallel_execution=False,  # Sequential for deterministic results
    )


def demonstrate_unit_testing():
    """Demonstrate unit testing of individual modules"""
    print("=== Module Testing and Validation Strategies Example ===\n")
    print("1. Unit Testing:")
    print("-" * 40)

    # Create test module
    processor = TestableTextProcessor(processing_mode="standard")
    test_runner = ModuleTestRunner(processor)

    # Create and run test suite
    test_suite = create_comprehensive_test_suite()
    results = test_runner.run_test_suite(test_suite)

    print(f"Test Suite: {results['test_suite_name']}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']}")
    print(f"Failed: {results['failed_tests']}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Average Execution Time: {results['execution_times']['average']:.4f}s")

    # Show failed test details
    if results["failed_test_details"]:
        print(f"\nFailed Tests:")
        for failed_test in results["failed_test_details"]:
            print(f"  - {failed_test['name']}: {failed_test['error']}")

    # Show performance summary
    if results["performance_summary"]:
        perf = results["performance_summary"]
        print(f"\nPerformance Summary:")
        if "execution_time_stats" in perf:
            exec_stats = perf["execution_time_stats"]
            print(
                f"  Execution Time - Mean: {exec_stats['mean']:.4f}s, Std: {exec_stats['stdev']:.4f}s"
            )
            print(
                f"  Execution Time - Min: {exec_stats['min']:.4f}s, Max: {exec_stats['max']:.4f}s"
            )


def demonstrate_integration_testing():
    """Demonstrate integration testing of component pipelines"""
    print("\n2. Integration Testing:")
    print("-" * 40)

    # Create a pipeline for integration testing
    pipeline = ComponentPipeline(
        [
            TextCleanerComponent(),
            SentimentAnalyzerComponent(),
            TextSummarizerComponent(),
        ]
    )

    test_runner = ModuleTestRunner(pipeline)

    # Create integration test cases
    integration_tests = [
        TestCase(
            name="pipeline_integration_basic",
            inputs={"text": "This is a great product! I love it."},
            validation_func=lambda result, inputs: (
                result.get("successful_steps", 0) > 0 and "final_result" in result
            ),
            description="Basic pipeline integration test",
        ),
        TestCase(
            name="pipeline_integration_complex",
            inputs={
                "text": "<p>This is an <b>amazing</b> article about AI technology!</p> The content is very informative."
            },
            validation_func=lambda result, inputs: (
                result.get("successful_steps", 0) == result.get("total_steps", 0)
            ),
            description="Complex pipeline integration test with HTML",
        ),
        TestCase(
            name="pipeline_integration_empty",
            inputs={"text": ""},
            validation_func=lambda result, inputs: "pipeline_results" in result,
            description="Pipeline integration test with empty input",
        ),
    ]

    integration_suite = TestSuite(
        name="Pipeline Integration Tests", test_cases=integration_tests
    )

    results = test_runner.run_test_suite(integration_suite)

    print(f"Integration Test Results:")
    print(f"  Total Tests: {results['total_tests']}")
    print(f"  Success Rate: {results['success_rate']:.2%}")
    print(f"  Average Duration: {results['execution_times']['average']:.4f}s")

    # Analyze pipeline-specific results
    for test_result in results["detailed_results"]:
        if test_result.success and test_result.actual_outputs:
            output = test_result.actual_outputs
            if "successful_steps" in output:
                print(
                    f"  {test_result.test_case.name}: {output['successful_steps']}/{output.get('total_steps', 0)} steps"
                )


def demonstrate_performance_benchmarking():
    """Demonstrate performance benchmarking and load testing"""
    print("\n3. Performance Benchmarking:")
    print("-" * 40)

    # Test different processing modes
    processors = {
        "Fast Mode": TestableTextProcessor(processing_mode="fast"),
        "Standard Mode": TestableTextProcessor(processing_mode="standard"),
        "Slow Mode": TestableTextProcessor(processing_mode="slow"),
        "Cached Mode": TestableTextProcessor(
            processing_mode="standard", enable_caching=True
        ),
    }

    # Benchmark each processor
    test_inputs = [
        {"text": "Short test."},
        {"text": "Medium length test with multiple words and sentences."},
        {"text": "Long test text with extensive content and multiple sentences. " * 10},
    ]

    print("Performance Comparison:")

    for name, processor in processors.items():
        benchmark = PerformanceBenchmark(processor)

        # Run benchmark
        start_time = time.time()
        results = []

        for _ in range(10):  # 10 iterations
            for test_input in test_inputs:
                try:
                    exec_start = time.time()
                    result = processor(**test_input)
                    exec_time = time.time() - exec_start
                    results.append(exec_time)
                except Exception as e:
                    logger.warning(f"Benchmark failed for {name}: {e}")

        if results:
            avg_time = statistics.mean(results)
            min_time = min(results)
            max_time = max(results)

            print(f"  {name}:")
            print(f"    Average: {avg_time:.4f}s")
            print(f"    Min: {min_time:.4f}s")
            print(f"    Max: {max_time:.4f}s")
            print(f"    Total Tests: {len(results)}")

    # Load testing
    print(f"\nLoad Testing (Standard Mode):")
    standard_processor = processors["Standard Mode"]
    benchmark = PerformanceBenchmark(standard_processor)

    load_test_results = benchmark.run_load_test(
        test_inputs, concurrent_users=5, duration_seconds=10
    )

    print(f"  Duration: {load_test_results['test_duration']}s")
    print(f"  Concurrent Users: {load_test_results['concurrent_users']}")
    print(f"  Total Requests: {load_test_results['total_requests']}")
    print(f"  Success Rate: {(1 - load_test_results['error_rate']):.2%}")
    print(f"  Throughput: {load_test_results['throughput_rps']:.2f} RPS")

    if "response_time_stats" in load_test_results:
        stats = load_test_results["response_time_stats"]
        print(f"  Response Times:")
        print(f"    Mean: {stats['mean']:.4f}s")
        print(f"    Median: {stats['median']:.4f}s")
        print(f"    95th percentile: {stats['p95']:.4f}s")


def demonstrate_quality_assessment():
    """Demonstrate comprehensive quality assessment"""
    print("\n4. Quality Assessment:")
    print("-" * 40)

    # Test different module configurations
    modules_to_assess = [
        ("Well-configured Module", TestableTextProcessor(processing_mode="standard")),
        ("Slow Module", TestableTextProcessor(processing_mode="slow")),
        (
            "Cached Module",
            TestableTextProcessor(processing_mode="standard", enable_caching=True),
        ),
    ]

    # Generate test cases for quality assessment
    quality_test_cases = TestCaseGenerator.generate_text_processing_tests(10)

    for name, module in modules_to_assess:
        print(f"\nAssessing: {name}")
        print("-" * 30)

        quality_assessor = QualityAssessment(module)
        quality_results = quality_assessor.assess_module_quality(quality_test_cases)

        print(f"Overall Quality Score: {quality_results['overall_quality_score']:.2f}")

        metrics = quality_results["quality_metrics"]
        print(f"Quality Metrics:")
        print(f"  Validation Score: {metrics['validation_score']:.2f}")
        print(f"  Execution Success Rate: {metrics['execution_success_rate']:.2f}")
        print(f"  Consistency Score: {metrics['consistency_score']:.2f}")
        print(f"  Robustness Score: {metrics['robustness_score']:.2f}")

        if quality_results["recommendations"]:
            print(f"Recommendations:")
            for rec in quality_results["recommendations"]:
                print(f"  - {rec}")


def demonstrate_automated_test_generation():
    """Demonstrate automated test case generation"""
    print("\n5. Automated Test Generation:")
    print("-" * 40)

    # Generate different types of test cases
    test_generators = [
        (
            "Text Processing Tests",
            lambda: TestCaseGenerator.generate_text_processing_tests(5),
        ),
        (
            "Classification Tests",
            lambda: TestCaseGenerator.generate_classification_tests(
                ["positive", "negative", "neutral"], 6
            ),
        ),
        (
            "Performance Tests",
            lambda: TestCaseGenerator.generate_performance_tests(
                {"text": "Sample text"}, 5
            ),
        ),
    ]

    for generator_name, generator_func in test_generators:
        print(f"\n{generator_name}:")

        generated_tests = generator_func()

        print(f"  Generated {len(generated_tests)} test cases:")
        for i, test_case in enumerate(generated_tests[:3]):  # Show first 3
            print(f"    {i+1}. {test_case.name}: {test_case.description}")
            if test_case.tags:
                print(f"       Tags: {', '.join(test_case.tags)}")

    # Run generated tests on a module
    print(f"\nRunning Generated Tests:")
    processor = TestableTextProcessor()
    test_runner = ModuleTestRunner(processor)

    # Use text processing tests
    text_tests = TestCaseGenerator.generate_text_processing_tests(8)
    generated_suite = TestSuite(name="Generated Test Suite", test_cases=text_tests)

    results = test_runner.run_test_suite(generated_suite)

    print("  Generated Test Results:")
    print(f"    Total: {results['total_tests']}")
    print(f"    Passed: {results['passed_tests']}")
    print(f"    Success Rate: {results['success_rate']:.2%}")


def demonstrate_test_reporting():
    """Demonstrate comprehensive test reporting"""
    print("\n6. Test Reporting and Analysis:")
    print("-" * 40)

    # Create a comprehensive test scenario
    processor = TestableTextProcessor(processing_mode="detailed")
    test_runner = ModuleTestRunner(processor)

    # Combine different test types
    all_test_cases = []
    all_test_cases.extend(TestCaseGenerator.generate_text_processing_tests(5))
    all_test_cases.extend(
        [
            TestCase(
                name="custom_validation_test",
                inputs={"text": "Custom test with specific validation"},
                validation_func=lambda result, inputs: (
                    result.get("processing_mode") == "detailed"
                    and "vocabulary_richness" in result
                ),
                description="Custom validation test for detailed mode",
            )
        ]
    )

    comprehensive_suite = TestSuite(
        name="Comprehensive Reporting Test Suite", test_cases=all_test_cases
    )

    results = test_runner.run_test_suite(comprehensive_suite)

    # Generate comprehensive report
    print("Comprehensive Test Report:")
    print("=" * 50)
    print(f"Test Suite: {results['test_suite_name']}")
    print(f"Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    print("Summary:")
    print(f"  Total Tests: {results['total_tests']}")
    print(f"  Passed: {results['passed_tests']}")
    print(f"  Failed: {results['failed_tests']}")
    print(f"  Success Rate: {results['success_rate']:.2%}")
    print("")
    print("Execution Times:")
    exec_times = results["execution_times"]
    print(f"  Total: {exec_times['total']:.4f}s")
    print(f"  Average: {exec_times['average']:.4f}s")
    print(f"  Min: {exec_times['min']:.4f}s")
    print(f"  Max: {exec_times['max']:.4f}s")
    print(f"  Median: {exec_times['median']:.4f}s")

    if results["performance_summary"]:
        perf = results["performance_summary"]
        print("")
        print("Performance Analysis:")
        if "execution_time_stats" in perf:
            stats = perf["execution_time_stats"]
            print(f"  Mean Execution Time: {stats['mean']:.4f}s")
            print(f"  Standard Deviation: {stats['stdev']:.4f}s")

        if "memory_usage_stats" in perf:
            mem_stats = perf["memory_usage_stats"]
            print(f"  Average Memory Usage: {mem_stats['mean']} bytes")

    # Module performance metrics
    module_metrics = processor._analyze_performance()
    print("")
    print("Module Performance:")
    print(f"  Total Runs: {module_metrics.get('total_runs', 0)}")
    print(f"  Average Duration: {module_metrics.get('avg_duration', 0):.4f}s")
    print(f"  Success Rate: {module_metrics.get('success_rate', 0):.2%}")


if __name__ == "__main__":
    """
    Module Testing and Validation Strategies Example

    This script demonstrates:
    1. Comprehensive unit testing with custom test cases
    2. Integration testing of component pipelines
    3. Performance benchmarking and load testing
    4. Quality assessment and reliability evaluation
    5. Automated test case generation
    6. Detailed test reporting and analysis
    """

    try:
        demonstrate_unit_testing()
        demonstrate_integration_testing()
        demonstrate_performance_benchmarking()
        demonstrate_quality_assessment()
        demonstrate_automated_test_generation()
        demonstrate_test_reporting()

        print(
            "\n✅ Module Testing and Validation Strategies example completed successfully!"
        )
        print("\nKey Learning Points:")
        print("- Comprehensive testing ensures module reliability and correctness")
        print("- Different test types serve different validation purposes")
        print(
            "- Performance benchmarking identifies bottlenecks and scalability limits"
        )
        print("- Quality assessment provides objective module evaluation")
        print("- Automated test generation reduces manual testing effort")
        print("- Detailed reporting enables data-driven improvement decisions")
        print("- Integration testing validates component interactions")
        print("- Load testing reveals system behavior under stress")

    except Exception as e:
        print(f"\n❌ Module Testing and Validation Strategies example failed: {e}")
        logger.exception(
            "Module Testing and Validation Strategies example execution failed"
        )
