#!/usr/bin/env python3
"""
DSPy Module Testing Framework

This module provides comprehensive testing utilities for custom DSPy modules including
unit tests, integration tests, performance benchmarking, and quality assessment tools.

Learning Objectives:
- Understand testing strategies for DSPy modules
- Learn to create comprehensive test suites for custom modules
- Master performance benchmarking and quality assessment
- Implement automated testing and validation workflows

Author: DSPy Learning Framework
"""

import asyncio
import json
import logging
import statistics
import time
import unittest
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import dspy
from dspy import Example

# Import from our custom module framework
try:
    from .component_library import ComponentPipeline, ComponentRouter, ReusableComponent
    from .custom_module_template import (
        CustomModuleBase,
        ModuleMetadata,
        ModuleValidator,
    )
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))
    from component_library import ComponentPipeline, ComponentRouter, ReusableComponent
    from custom_module_template import CustomModuleBase, ModuleMetadata, ModuleValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a single test case for module testing"""

    name: str
    inputs: dict[str, Any]
    expected_outputs: Optional[dict[str, Any]] = None
    validation_func: Optional[Callable] = None
    timeout: float = 30.0
    tags: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class TestResult:
    """Results from executing a test case"""

    test_case: TestCase
    success: bool
    execution_time: float
    actual_outputs: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    validation_details: Optional[dict[str, Any]] = None
    performance_metrics: Optional[dict[str, Any]] = None


@dataclass
class TestSuite:
    """Collection of test cases for comprehensive module testing"""

    name: str
    test_cases: list[TestCase]
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    parallel_execution: bool = False
    max_workers: int = 4


class ModuleTestRunner:
    """Test runner for executing module test suites"""

    def __init__(self, module: CustomModuleBase):
        self.module = module
        self.test_results = []
        self.performance_data = []

    def run_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single test case"""
        start_time = time.time()

        try:
            # Execute the module with test inputs
            result = self.module(**test_case.inputs)
            execution_time = time.time() - start_time

            # Validate results
            validation_success = True
            validation_details = {}

            if test_case.expected_outputs:
                validation_success, validation_details = self._validate_outputs(
                    result, test_case.expected_outputs
                )

            if test_case.validation_func:
                custom_validation = test_case.validation_func(result, test_case.inputs)
                if isinstance(custom_validation, bool):
                    validation_success = validation_success and custom_validation
                elif isinstance(custom_validation, dict):
                    validation_success = validation_success and custom_validation.get(
                        "success", False
                    )
                    validation_details.update(custom_validation.get("details", {}))

            # Collect performance metrics
            performance_metrics = {
                "execution_time": execution_time,
                "memory_usage": self._estimate_memory_usage(result),
                "output_size": self._calculate_output_size(result),
            }

            return TestResult(
                test_case=test_case,
                success=validation_success,
                execution_time=execution_time,
                actual_outputs=result,
                validation_details=validation_details,
                performance_metrics=performance_metrics,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_case=test_case,
                success=False,
                execution_time=execution_time,
                error=str(e),
            )

    def run_test_suite(self, test_suite: TestSuite) -> dict[str, Any]:
        """Execute a complete test suite"""
        logger.info(f"Running test suite: {test_suite.name}")

        # Setup
        if test_suite.setup_func:
            try:
                test_suite.setup_func()
            except Exception as e:
                logger.error(f"Test suite setup failed: {e}")
                return {"error": f"Setup failed: {e}"}

        # Execute tests
        results = []

        if test_suite.parallel_execution:
            results = self._run_tests_parallel(
                test_suite.test_cases, test_suite.max_workers
            )
        else:
            results = self._run_tests_sequential(test_suite.test_cases)

        # Teardown
        if test_suite.teardown_func:
            try:
                test_suite.teardown_func()
            except Exception as e:
                logger.warning(f"Test suite teardown failed: {e}")

        # Analyze results
        return self._analyze_test_results(test_suite, results)

    def _run_tests_sequential(self, test_cases: list[TestCase]) -> list[TestResult]:
        """Run tests sequentially"""
        results = []
        for i, test_case in enumerate(test_cases):
            logger.info(f"Running test {i+1}/{len(test_cases)}: {test_case.name}")
            result = self.run_test_case(test_case)
            results.append(result)
        return results

    def _run_tests_parallel(
        self, test_cases: list[TestCase], max_workers: int
    ) -> list[TestResult]:
        """Run tests in parallel"""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {
                executor.submit(self.run_test_case, test_case): test_case
                for test_case in test_cases
            }

            for future in as_completed(future_to_test):
                test_case = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed test: {test_case.name}")
                except Exception as e:
                    logger.error(f"Test {test_case.name} failed with exception: {e}")
                    results.append(
                        TestResult(
                            test_case=test_case,
                            success=False,
                            execution_time=0.0,
                            error=str(e),
                        )
                    )

        return results

    def _validate_outputs(
        self, actual: Any, expected: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Validate actual outputs against expected outputs"""
        validation_details = {}

        if not isinstance(actual, dict):
            return False, {"error": "tput is not a dictionary"}

        for key, expected_value in expected.items():
            if key not in actual:
                validation_details[key] = {
                    "status": "missing",
                    "expected": expected_value,
                }
                continue

            actual_value = actual[key]

            if isinstance(expected_value, dict) and "type" in expected_value:
                # Type-based validation
                expected_type = expected_value["type"]
                if not isinstance(actual_value, expected_type):
                    validation_details[key] = {
                        "status": "type_mismatch",
                        "expected_type": expected_type.__name__,
                        "actual_type": type(actual_value).__name__,
                    }
                    continue
            elif actual_value != expected_value:
                validation_details[key] = {
                    "status": "value_mismatch",
                    "expected": expected_value,
                    "actual": actual_value,
                }
                continue

            validation_details[key] = {"status": "match"}

        success = all(
            detail["status"] == "match" for detail in validation_details.values()
        )
        return success, validation_details

    def _estimate_memory_usage(self, result: Any) -> int:
        """Estimate memory usage of result (simplified)"""
        try:
            import sys

            return sys.getsizeof(result)
        except:
            return 0

    def _calculate_output_size(self, result: Any) -> int:
        """Calculate size of output data"""
        try:
            if isinstance(result, str):
                return len(result)
            elif isinstance(result, (list, dict)):
                return len(str(result))
            else:
                return len(str(result))
        except:
            return 0

    def _analyze_test_results(
        self, test_suite: TestSuite, results: list[TestResult]
    ) -> Dict[str, Any]:
        """Analyze and summarize test results"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests

        execution_times = [r.execution_time for r in results]

        analysis = {
            "test_suite_name": test_suite.name,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "execution_times": {
                "total": sum(execution_times),
                "average": statistics.mean(execution_times) if execution_times else 0,
                "min": min(execution_times) if execution_times else 0,
                "max": max(execution_times) if execution_times else 0,
                "median": statistics.median(execution_times) if execution_times else 0,
            },
            "failed_test_details": [
                {
                    "name": r.test_case.name,
                    "error": r.error,
                    "execution_time": r.execution_time,
                }
                for r in results
                if not r.success
            ],
            "performance_summary": self._analyze_performance(results),
            "detailed_results": results,
        }

        return analysis

    def _analyze_performance(self, results: list[TestResult]) -> Dict[str, Any]:
        """Analyze performance metrics from test results"""
        performance_data = []

        for result in results:
            if result.performance_metrics:
                performance_data.append(result.performance_metrics)

        if not performance_data:
            return {}

        # Aggregate performance metrics
        execution_times = [p["execution_time"] for p in performance_data]
        memory_usages = [p["memory_usage"] for p in performance_data]
        output_sizes = [p["output_size"] for p in performance_data]

        return {
            "execution_time_stats": {
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "stdev": (
                    statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                ),
                "min": min(execution_times),
                "max": max(execution_times),
            },
            "memory_usage_stats": {
                "mean": statistics.mean(memory_usages),
                "median": statistics.median(memory_usages),
                "min": min(memory_usages),
                "max": max(memory_usages),
            },
            "output_size_stats": {
                "mean": statistics.mean(output_sizes),
                "median": statistics.median(output_sizes),
                "min": min(output_sizes),
                "max": max(output_sizes),
            },
        }


class PerformanceBenchmark:
    """Performance benchmarking utilities for DSPy modules"""

    def __init__(self, module: CustomModuleBase):
        self.module = module
        self.benchmark_results = []

    def run_load_test(
        self,
        test_inputs: list[Dict[str, Any]],
        concurrent_users: int = 10,
        duration_seconds: int = 60,
    ) -> Dict[str, Any]:
        """Run load testing on the module"""
        logger.info(
            f"Starting load test: {concurrent_users} users for {duration_seconds}s"
        )

        start_time = time.time()
        end_time = start_time + duration_seconds

        results = []
        errors = []

        def worker():
            """Worker function for load testing"""
            worker_results = []
            worker_errors = []

            while time.time() < end_time:
                for test_input in test_inputs:
                    if time.time() >= end_time:
                        break

                    try:
                        exec_start = time.time()
                        result = self.module(**test_input)
                        exec_time = time.time() - exec_start

                        worker_results.append(
                            {
                                "execution_time": exec_time,
                                "timestamp": exec_start,
                                "success": True,
                            }
                        )
                    except Exception as e:
                        worker_errors.append(
                            {
                                "error": str(e),
                                "timestamp": time.time(),
                                "input": test_input,
                            }
                        )

            return worker_results, worker_errors

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(concurrent_users)]

            for future in as_completed(futures):
                worker_results, worker_errors = future.result()
                results.extend(worker_results)
                errors.extend(worker_errors)

        # Analyze results
        total_requests = len(results) + len(errors)
        successful_requests = len(results)
        error_rate = len(errors) / total_requests if total_requests > 0 else 0

        if results:
            execution_times = [r["execution_time"] for r in results]
            throughput = successful_requests / duration_seconds

            return {
                "test_duration": duration_seconds,
                "concurrent_users": concurrent_users,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": len(errors),
                "error_rate": error_rate,
                "throughput_rps": throughput,
                "response_time_stats": {
                    "mean": statistics.mean(execution_times),
                    "median": statistics.median(execution_times),
                    "p95": self._percentile(execution_times, 95),
                    "p99": self._percentile(execution_times, 99),
                    "min": min(execution_times),
                    "max": max(execution_times),
                },
                "errors": errors[:10],  # Sample of errors
            }
        else:
            return {
                "test_duration": duration_seconds,
                "concurrent_users": concurrent_users,
                "total_requests": total_requests,
                "successful_requests": 0,
                "failed_requests": len(errors),
                "error_rate": 1.0,
                "throughput_rps": 0,
                "errors": errors,
            }

    def run_stress_test(
        self, test_input: Dict[str, Any], max_concurrent: int = 100, step_size: int = 10
    ) -> Dict[str, Any]:
        """Run stress testing to find breaking point"""
        logger.info(f"Starting stress test: up to {max_concurrent} concurrent requests")

        stress_results = []

        for concurrent_level in range(step_size, max_concurrent + 1, step_size):
            logger.info(f"Testing with {concurrent_level} concurrent requests")

            start_time = time.time()
            results = []
            errors = []

            def stress_worker():
                try:
                    exec_start = time.time()
                    result = self.module(**test_input)
                    exec_time = time.time() - exec_start
                    return {"success": True, "execution_time": exec_time}
                except Exception as e:
                    return {"success": False, "error": str(e)}

            with ThreadPoolExecutor(max_workers=concurrent_level) as executor:
                futures = [
                    executor.submit(stress_worker) for _ in range(concurrent_level)
                ]

                for future in as_completed(futures):
                    result = future.result()
                    if result["success"]:
                        results.append(result)
                    else:
                        errors.append(result)

            total_time = time.time() - start_time
            success_rate = len(results) / concurrent_level

            stress_results.append(
                {
                    "concurrent_level": concurrent_level,
                    "success_rate": success_rate,
                    "total_time": total_time,
                    "avg_response_time": (
                        statistics.mean([r["execution_time"] for r in results])
                        if results
                        else 0
                    ),
                    "error_count": len(errors),
                }
            )

            # Stop if success rate drops below threshold
            if success_rate < 0.8:
                logger.warning(
                    f"Success rate dropped to {success_rate:.2%} at {concurrent_level} concurrent requests"
                )
                break

        return {
            "max_tested_concurrent": max(r["concurrent_level"] for r in stress_results),
            "breaking_point": next(
                (
                    r["concurrent_level"]
                    for r in stress_results
                    if r["success_rate"] < 0.9
                ),
                None,
            ),
            "stress_test_results": stress_results,
        }

    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0

        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


class QualityAssessment:
    """Quality assessment tools for DSPy modules"""

    def __init__(self, module: CustomModuleBase):
        self.module = module

    def assess_module_quality(self, test_cases: list[TestCase]) -> Dict[str, Any]:
        """Comprehensive quality assessment"""
        logger.info("Starting comprehensive quality assessment")

        # Run basic validation
        validator = ModuleValidator()
        validation_results = validator.validate_module(self.module)

        # Test execution quality
        test_runner = ModuleTestRunner(self.module)
        execution_results = []

        for test_case in test_cases:
            result = test_runner.run_test_case(test_case)
            execution_results.append(result)

        # Analyze consistency
        consistency_score = self._assess_consistency(test_cases)

        # Analyze robustness
        robustness_score = self._assess_robustness(test_cases)

        # Calculate overall quality score
        quality_metrics = {
            "validation_score": 1.0 if validation_results.is_valid else 0.0,
            "execution_success_rate": sum(1 for r in execution_results if r.success)
            / len(execution_results),
            "consistency_score": consistency_score,
            "robustness_score": robustness_score,
        }

        overall_score = sum(quality_metrics.values()) / len(quality_metrics)

        return {
            "overall_quality_score": overall_score,
            "quality_metrics": quality_metrics,
            "validation_results": validation_results,
            "execution_results": execution_results,
            "recommendations": self._generate_quality_recommendations(
                quality_metrics, validation_results
            ),
        }

    def _assess_consistency(self, test_cases: list[TestCase]) -> float:
        """Assess consistency of module outputs"""
        if len(test_cases) < 2:
            return 1.0

        # Run same inputs multiple times and check consistency
        consistency_scores = []

        for test_case in test_cases[:5]:  # Test first 5 cases for consistency
            results = []
            for _ in range(3):  # Run each test 3 times
                try:
                    result = self.module(**test_case.inputs)
                    results.append(result)
                except:
                    results.append(None)

            # Check if results are consistent
            if all(r is not None for r in results):
                if all(str(r) == str(results[0]) for r in results):
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.5)  # Partial consistency
            else:
                consistency_scores.append(0.0)  # Inconsistent (some failed)

        return statistics.mean(consistency_scores) if consistency_scores else 0.0

    def _assess_robustness(self, test_cases: list[TestCase]) -> float:
        """Assess robustness to edge cases and invalid inputs"""
        robustness_tests = [
            # Empty inputs
            {},
            # None values
            {key: None for key in test_cases[0].inputs.keys()} if test_cases else {},
            # Empty strings
            (
                {
                    key: ""
                    for key in test_cases[0].inputs.keys()
                    if isinstance(test_cases[0].inputs[key], str)
                }
                if test_cases
                else {}
            ),
            # Very long strings
            (
                {
                    key: "x" * 10000
                    for key in test_cases[0].inputs.keys()
                    if isinstance(test_cases[0].inputs[key], str)
                }
                if test_cases
                else {}
            ),
        ]

        robustness_scores = []

        for test_input in robustness_tests:
            if not test_input:  # Skip empty test inputs
                continue

            try:
                result = self.module(**test_input)
                # Module handled edge case gracefully
                robustness_scores.append(1.0)
            except Exception as e:
                # Check if it's a reasonable error (validation) vs unexpected crash
                if "required" in str(e).lower() or "invalid" in str(e).lower():
                    robustness_scores.append(0.8)  # Good error handling
                else:
                    robustness_scores.append(0.2)  # Poor error handling

        return statistics.mean(robustness_scores) if robustness_scores else 0.5

    def _generate_quality_recommendations(
        self, quality_metrics: Dict[str, float], validation_results: Dict[str, Any]
    ) -> list[str]:
        """Generate recommendations for improving module quality"""
        recommendations = []

        if quality_metrics["validation_score"] < 1.0:
            if validation_results.get("errors"):
                recommendations.extend(
                    [
                        f"Fix validation error: {error}"
                        for error in validation_results["errors"]
                    ]
                )
            if validation_results.get("warnings"):
                recommendations.extend(
                    [
                        f"Address warning: {warning}"
                        for warning in validation_results["warnings"]
                    ]
                )

        if quality_metrics["execution_success_rate"] < 0.9:
            recommendations.append(
                "Improve error handling to increase execution success rate"
            )

        if quality_metrics["consistency_score"] < 0.8:
            recommendations.append(
                "Improve output consistency - consider deterministic processing"
            )

        if quality_metrics["robustness_score"] < 0.7:
            recommendations.append("Improve input validation and edge case handling")

        if not recommendations:
            recommendations.append(
                "Module quality is good - consider performance optimization"
            )

        return recommendations


# Test Case Generators


class TestCaseGenerator:
    """Utilities for generating test cases automatically"""

    @staticmethod
    def generate_text_processing_tests(num_cases: int = 10) -> list[TestCase]:
        """Generate test cases for text processing modules"""
        test_cases = []

        # Basic text cases
        basic_texts = [
            "Hello, world!",
            "This is a simple test.",
            "Multiple sentences. With punctuation!",
            "Text with numbers 123 and symbols @#$",
            "",  # Empty string
        ]

        for i, text in enumerate(basic_texts):
            test_cases.append(
                TestCase(
                    name=f"basic_text_{i}",
                    inputs={"text": text},
                    tags=["basic", "text_processing"],
                    description=f"Basic text processing test with: {text[:30]}...",
                )
            )

        # Edge cases
        edge_cases = [
            "A" * 10000,  # Very long text
            "   \n\t   ",  # Whitespace only
            "🚀 Unicode text with emojis 🎉",  # Unicode
            "<html><body>HTML content</body></html>",  # HTML
            "Line 1\nLine 2\nLine 3",  # Multi-line
        ]

        for i, text in enumerate(edge_cases):
            test_cases.append(
                TestCase(
                    name=f"edge_case_{i}",
                    inputs={"text": text},
                    tags=["edge_case", "text_processing"],
                    description=f"Edge case test: {type(text).__name__}",
                )
            )

        return test_cases[:num_cases]

    @staticmethod
    def generate_classification_tests(
        categories: list[str], num_cases: int = 10
    ) -> list[TestCase]:
        """Generate test cases for classification modules"""
        test_cases = []

        # Sample texts for different categories
        category_samples = {
            "technology": [
                "The new AI model shows impressive performance",
                "Software development best practices",
                "Machine learning algorithms comparison",
            ],
            "business": [
                "Quarterly revenue increased by 15%",
                "Market analysis and customer insights",
                "Strategic business planning session",
            ],
            "science": [
                "Research findings published in Nature",
                "Experimental methodology and results",
                "Hypothesis testing and data analysis",
            ],
        }

        case_count = 0
        for category in categories:
            if case_count >= num_cases:
                break

            samples = category_samples.get(
                category.lower(), [f"Sample text for {category}"]
            )

            for i, text in enumerate(samples):
                if case_count >= num_cases:
                    break

                test_cases.append(
                    TestCase(
                        name=f"classification_{category}_{i}",
                        inputs={"text": text},
                        expected_outputs={"predicted_category": category},
                        tags=["classification", category],
                        description=f"Classification test for {category}",
                    )
                )
                case_count += 1

        return test_cases

    @staticmethod
    def generate_performance_tests(
        base_input: Dict[str, Any], num_cases: int = 100
    ) -> list[TestCase]:
        """Generate test cases for performance testing"""
        test_cases = []

        for i in range(num_cases):
            # Vary input sizes and complexity
            if "text" in base_input:
                text_length = 100 + (i * 50)  # Increasing text length
                test_input = {
                    **base_input,
                    "text": "Sample text. " * (text_length // 12),
                }
            else:
                test_input = base_input.copy()

            test_cases.append(
                TestCase(
                    name=f"performance_test_{i}",
                    inputs=test_input,
                    tags=["performance"],
                    description=f"Performance test case {i}",
                )
            )

        return test_cases


def demonstrate_testing_framework():
    """Demonstrate the module testing framework"""
    print("=== DSPy Module Testing Framework Demonstration ===\n")

    # Create a sample module for testing
    from component_library import SentimentAnalyzerComponent, TextCleanerComponent

    # Example 1: Basic module testing
    print("1. Basic Module Testing:")
    print("-" * 40)

    cleaner = TextCleanerComponent()
    test_runner = ModuleTestRunner(cleaner)

    # Create test cases
    test_cases = [
        TestCase(
            name="basic_cleaning",
            inputs={"text": "<p>Hello, World!</p>"},
            expected_outputs={"cleaned_text": {"type": str}},
            description="Basic HTML cleaning test",
        ),
        TestCase(
            name="whitespace_normalization",
            inputs={"text": "Multiple   spaces    here"},
            validation_func=lambda result, inputs: len(
                result.get("cleaned_text", "").split()
            )
            == 3,
            description="Whitespace normalization test",
        ),
    ]

    # Create and run test suite
    test_suite = TestSuite(name="Text Cleaner Tests", test_cases=test_cases)

    results = test_runner.run_test_suite(test_suite)

    print(f"Test Suite: {results['test_suite_name']}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']}")
    print(f"Failed: {results['failed_tests']}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Average Execution Time: {results['execution_times']['average']:.3f}s")

    # Example 2: Performance Benchmarking
    print("\n2. Performance Benchmarking:")
    print("-" * 40)

    sentiment_analyzer = SentimentAnalyzerComponent()
    benchmark = PerformanceBenchmark(sentiment_analyzer)

    # Generate performance test inputs
    test_inputs = TestCaseGenerator.generate_performance_tests(
        {"text": "This is a test message"}, num_cases=20
    )

    # Convert TestCase objects to dict inputs for benchmarking
    benchmark_inputs = [tc.inputs for tc in test_inputs]

    # Run load test (shorter duration for demo)
    load_results = benchmark.run_load_test(
        benchmark_inputs[:5],  # Use fewer inputs for demo
        concurrent_users=3,
        duration_seconds=10,
    )

    print("Load Test Results:")
    print(f"  Duration: {load_results['test_duration']}s")
    print(f"  Concurrent Users: {load_results['concurrent_users']}")
    print(f"  Total Requests: {load_results['total_requests']}")
    print(f"  Success Rate: {(1 - load_results['error_rate']):.2%}")
    print(f"  Throughput: {load_results['throughput_rps']:.2f} RPS")
    if "response_time_stats" in load_results:
        print(
            f"  Avg Response Time: {load_results['response_time_stats']['mean']:.3f}s"
        )

    # Example 3: Quality Assessment
    print("\n3. Quality Assessment:")
    print("-" * 40)

    quality_assessor = QualityAssessment(sentiment_analyzer)

    # Generate test cases for quality assessment
    quality_test_cases = TestCaseGenerator.generate_text_processing_tests(5)

    quality_results = quality_assessor.assess_module_quality(quality_test_cases)

    print(f"Overall Quality Score: {quality_results['overall_quality_score']:.2f}")
    print(f"Quality Metrics:")
    for metric, score in quality_results["quality_metrics"].items():
        print(f"  {metric}: {score:.2f}")

    if quality_results["recommendations"]:
        print(f"Recommendations:")
        for rec in quality_results["recommendations"]:
            print(f"  - {rec}")

    # Example 4: Automated Test Generation
    print("\n4. Automated Test Generation:")
    print("-" * 40)

    # Generate classification tests
    classification_tests = TestCaseGenerator.generate_classification_tests(
        ["technology", "business", "science"], num_cases=6
    )

    print(f"Generated {len(classification_tests)} classification test cases:")
    for test_case in classification_tests[:3]:  # Show first 3
        print(f"  - {test_case.name}: {test_case.inputs['text'][:50]}...")

    # Generate text processing tests
    text_tests = TestCaseGenerator.generate_text_processing_tests(5)

    print(f"\nGenerated {len(text_tests)} text processing test cases:")
    for test_case in text_tests[:3]:  # Show first 3
        print(f"  - {test_case.name}: {test_case.description}")


if __name__ == "__main__":
    """
    DSPy Module Testing Framework Demonstration

    This script demonstrates:
    1. Comprehensive module testing with test suites
    2. Performance benchmarking and load testing
    3. Quality assessment and validation
    4. Automated test case generation
    5. Detailed result analysis and reporting
    """

    try:
        demonstrate_testing_framework()

        print("\n✅ Module testing framework demonstration completed successfully!")
        print("\nKey takeaways:")
        print("- Comprehensive testing ensures module reliability and quality")
        print(
            "- Performance benchmarking identifies bottlenecks and scalability limits"
        )
        print("- Quality assessment provides objective module evaluation")
        print("- Automated test generation reduces manual testing effort")
        print("- Detailed reporting enables data-driven optimization decisions")

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logger.exception("Testing framework demonstration failed")
