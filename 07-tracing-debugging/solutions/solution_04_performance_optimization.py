#!/usr/bin/env python3
"""
Solution 04: DSPy Performance Optimization Implementation

This solution demonstrates comprehensive performance optimization techniques
for DSPy systems, including profiling, bottleneck identification, and optimization strategies.

Learning Objectives:
- Implement performance profiling for DSPy modules
- Identify and analyze performance bottlenecks
- Apply optimization strategies and measure improvements
- Develop performance regression detection systems

Author: DSPy Learning Framework
"""

import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import dspy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceProfile:
    """Performance profile for a specific operation"""

    operation_name: str
    total_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    avg_duration: float = 0.0
    p95_duration: float = 0.0
    p99_duration: float = 0.0
    error_count: int = 0
    last_updated: float = field(default_factory=time.time)
    durations: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add_measurement(self, duration: float, success: bool = True):
        """Add a new performance measurement"""
        self.total_calls += 1
        self.durations.append(duration)

        if success:
            self.total_duration += duration
            self.min_duration = min(self.min_duration, duration)
            self.max_duration = max(self.max_duration, duration)
            self.avg_duration = self.total_duration / (
                self.total_calls - self.error_count
            )

            # Calculate percentiles
            if len(self.durations) >= 2:
                sorted_durations = sorted(self.durations)
                self.p95_duration = sorted_durations[int(len(sorted_durations) * 0.95)]
                self.p99_duration = sorted_durations[int(len(sorted_durations) * 0.99)]
        else:
            self.error_count += 1

        self.last_updated = time.time()


@dataclass
class OptimizationResult:
    """Results of a performance optimization"""

    optimization_name: str
    baseline_performance: Dict[str, float]
    optimized_performance: Dict[str, float]
    improvement_percent: float
    optimization_applied: str
    timestamp: float = field(default_factory=time.time)


class PerformanceProfiler:
    """Advanced performance profiler for DSPy modules"""

    def __init__(self):
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.baseline_profiles: Dict[str, PerformanceProfile] = {}
        self.optimization_results: List[OptimizationResult] = []

    def profile_function(self, operation_name: str):
        """Decorator for profiling function execution"""

        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    self.add_measurement(operation_name, duration, success)

            return wrapper

        return decorator

    def add_measurement(
        self, operation_name: str, duration: float, success: bool = True
    ):
        """Add a performance measurement"""
        if operation_name not in self.profiles:
            self.profiles[operation_name] = PerformanceProfile(operation_name)

        self.profiles[operation_name].add_measurement(duration, success)

    def get_profile_summary(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get performance profile summary"""
        if operation_name not in self.profiles:
            return None

        profile = self.profiles[operation_name]
        return {
            "operation_name": profile.operation_name,
            "total_calls": profile.total_calls,
            "avg_duration": profile.avg_duration,
            "min_duration": profile.min_duration,
            "max_duration": profile.max_duration,
            "p95_duration": profile.p95_duration,
            "p99_duration": profile.p99_duration,
            "error_rate": (
                profile.error_count / profile.total_calls
                if profile.total_calls > 0
                else 0
            ),
            "throughput": (
                profile.total_calls / profile.total_duration
                if profile.total_duration > 0
                else 0
            ),
        }

    def identify_bottlenecks(
        self, threshold_seconds: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        for name, profile in self.profiles.items():
            if profile.avg_duration > threshold_seconds:
                severity = (
                    "critical"
                    if profile.avg_duration > 5.0
                    else "high" if profile.avg_duration > 2.0 else "medium"
                )

                bottlenecks.append(
                    {
                        "operation": name,
                        "avg_duration": profile.avg_duration,
                        "p95_duration": profile.p95_duration,
                        "total_calls": profile.total_calls,
                        "severity": severity,
                        "impact_score": profile.avg_duration * profile.total_calls,
                        "suggestions": self._generate_optimization_suggestions(profile),
                    }
                )

        # Sort by impact score (duration * frequency)
        bottlenecks.sort(key=lambda x: x["impact_score"], reverse=True)
        return bottlenecks

    def _generate_optimization_suggestions(
        self, profile: PerformanceProfile
    ) -> List[str]:
        """Generate optimization suggestions based on performance profile"""
        suggestions = []

        if profile.avg_duration > 5.0:
            suggestions.append("Consider using a faster model or reducing max_tokens")
            suggestions.append("Implement caching for repeated requests")

        if profile.avg_duration > 2.0:
            suggestions.append("Optimize prompt length and complexity")
            suggestions.append("Consider batch processing for multiple requests")

        if profile.error_count / profile.total_calls > 0.1:
            suggestions.append("Improve error handling and retry logic")
            suggestions.append("Validate inputs before processing")

        if profile.p99_duration > profile.avg_duration * 3:
            suggestions.append("Investigate outlier requests causing high latency")
            suggestions.append("Implement timeout mechanisms")

        return suggestions

    def set_baseline(self):
        """Set current performance as baseline for comparison"""
        self.baseline_profiles = {
            name: profile for name, profile in self.profiles.items()
        }
        logger.info("Performance baseline set")

    def compare_to_baseline(self) -> Dict[str, Dict[str, float]]:
        """Compare current performance to baseline"""
        comparisons = {}

        for name, current_profile in self.profiles.items():
            if name in self.baseline_profiles:
                baseline_profile = self.baseline_profiles[name]

                avg_change = (
                    (
                        (current_profile.avg_duration - baseline_profile.avg_duration)
                        / baseline_profile.avg_duration
                        * 100
                    )
                    if baseline_profile.avg_duration > 0
                    else 0
                )

                p95_change = (
                    (
                        (current_profile.p95_duration - baseline_profile.p95_duration)
                        / baseline_profile.p95_duration
                        * 100
                    )
                    if baseline_profile.p95_duration > 0
                    else 0
                )

                comparisons[name] = {
                    "avg_duration_change_percent": avg_change,
                    "p95_duration_change_percent": p95_change,
                    "baseline_avg": baseline_profile.avg_duration,
                    "current_avg": current_profile.avg_duration,
                    "baseline_p95": baseline_profile.p95_duration,
                    "current_p95": current_profile.p95_duration,
                }

        return comparisons

    def record_optimization(
        self,
        optimization_name: str,
        baseline_perf: Dict[str, float],
        optimized_perf: Dict[str, float],
        optimization_applied: str,
    ):
        """Record the results of an optimization"""
        improvement = (
            (
                (baseline_perf["avg_duration"] - optimized_perf["avg_duration"])
                / baseline_perf["avg_duration"]
                * 100
            )
            if baseline_perf["avg_duration"] > 0
            else 0
        )

        result = OptimizationResult(
            optimization_name=optimization_name,
            baseline_performance=baseline_perf,
            optimized_performance=optimized_perf,
            improvement_percent=improvement,
            optimization_applied=optimization_applied,
        )

        self.optimization_results.append(result)
        return result


class DSPyOptimizer:
    """DSPy-specific performance optimizer"""

    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.optimization_strategies = {
            "reduce_max_tokens": self._optimize_max_tokens,
            "optimize_prompt": self._optimize_prompt,
            "implement_caching": self._implement_caching,
            "batch_processing": self._implement_batch_processing,
        }

    def _optimize_max_tokens(
        self, module, current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize max_tokens parameter"""
        # Simulate reducing max_tokens by 25%
        optimized_config = current_config.copy()
        current_tokens = optimized_config.get("max_tokens", 500)
        optimized_config["max_tokens"] = int(current_tokens * 0.75)

        return optimized_config

    def _optimize_prompt(
        self, module, current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize prompt structure and length"""
        # Simulate prompt optimization
        optimized_config = current_config.copy()
        optimized_config["prompt_optimized"] = True

        return optimized_config

    def _implement_caching(
        self, module, current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement response caching"""
        optimized_config = current_config.copy()
        optimized_config["caching_enabled"] = True
        optimized_config["cache_ttl"] = 300  # 5 minutes

        return optimized_config

    def _implement_batch_processing(
        self, module, current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement batch processing for multiple requests"""
        optimized_config = current_config.copy()
        optimized_config["batch_size"] = 5
        optimized_config["batch_processing"] = True

        return optimized_config

    def suggest_optimizations(self, operation_name: str) -> List[Dict[str, Any]]:
        """Suggest optimizations based on performance profile"""
        profile_summary = self.profiler.get_profile_summary(operation_name)
        if not profile_summary:
            return []

        suggestions = []

        # High latency optimizations
        if profile_summary["avg_duration"] > 3.0:
            suggestions.append(
                {
                    "strategy": "reduce_max_tokens",
                    "priority": "high",
                    "expected_improvement": "20-30%",
                    "description": "Reduce max_tokens to decrease response time",
                }
            )

            suggestions.append(
                {
                    "strategy": "implement_caching",
                    "priority": "high",
                    "expected_improvement": "50-80%",
                    "description": "Cache responses for repeated requests",
                }
            )

        # Medium latency optimizations
        if profile_summary["avg_duration"] > 1.0:
            suggestions.append(
                {
                    "strategy": "optimize_prompt",
                    "priority": "medium",
                    "expected_improvement": "10-20%",
                    "description": "Optimize prompt structure and length",
                }
            )

        # High throughput optimizations
        if profile_summary["total_calls"] > 100:
            suggestions.append(
                {
                    "strategy": "batch_processing",
                    "priority": "medium",
                    "expected_improvement": "15-25%",
                    "description": "Process multiple requests in batches",
                }
            )

        return suggestions

    def apply_optimization(
        self, operation_name: str, strategy: str, module, current_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """Apply a specific optimization strategy"""
        if strategy not in self.optimization_strategies:
            raise ValueError(f"Unknown optimization strategy: {strategy}")

        optimization_func = self.optimization_strategies[strategy]
        optimized_config = optimization_func(module, current_config)

        description = f"Applied {strategy} optimization to {operation_name}"
        return optimized_config, description


def create_performance_test_modules():
    """Create modules with different performance characteristics for testing"""

    class FastModule:
        """Fast executing module"""

        def __init__(self, config: Optional[Dict] = None):
            self.config = config or {"max_tokens": 100}

        def __call__(self, **kwargs):
            # Simulate fast execution
            delay = 0.1
            if self.config.get("caching_enabled"):
                delay *= 0.2  # Caching makes it much faster
            if self.config.get("max_tokens", 100) < 100:
                delay *= 0.8  # Fewer tokens = faster

            time.sleep(delay)
            return {"result": f"Fast result: {kwargs.get('text', '')[:50]}"}

    class SlowModule:
        """Slow executing module"""

        def __init__(self, config: Optional[Dict] = None):
            self.config = config or {"max_tokens": 500}

        def __call__(self, **kwargs):
            # Simulate slow execution
            delay = 3.0
            if self.config.get("caching_enabled"):
                delay *= 0.3  # Caching helps significantly
            if self.config.get("max_tokens", 500) < 400:
                delay *= 0.7  # Fewer tokens = faster
            if self.config.get("prompt_optimized"):
                delay *= 0.85  # Optimized prompt is faster

            time.sleep(delay)
            return {"result": f"Slow result: {kwargs.get('text', '')[:50]}"}

    class VariableModule:
        """Module with variable performance"""

        def __init__(self, config: Optional[Dict] = None):
            self.config = config or {"max_tokens": 300}
            self.call_count = 0

        def __call__(self, **kwargs):
            self.call_count += 1

            # Variable delay based on call count
            base_delay = 1.5
            if self.call_count % 10 == 0:
                delay = base_delay * 3  # Occasional slow requests
            else:
                delay = base_delay

            # Apply optimizations
            if self.config.get("caching_enabled") and self.call_count % 3 == 0:
                delay *= 0.1  # Cache hit
            if self.config.get("batch_processing"):
                delay *= 0.8  # Batch processing efficiency

            time.sleep(delay)
            return {
                "result": f"Variable result {self.call_count}: {kwargs.get('text', '')[:50]}"
            }

    return {"fast": FastModule(), "slow": SlowModule(), "variable": VariableModule()}


def demonstrate_performance_profiling():
    """Demonstrate comprehensive performance profiling"""
    print("=== Performance Profiling Demonstration ===\n")

    profiler = PerformanceProfiler()
    modules = create_performance_test_modules()

    print("1. Profiling different module types:")
    print("-" * 40)

    # Profile each module type
    test_inputs = [
        "Short test input",
        "Medium length test input for performance analysis",
        "Very long test input that might take more time to process and analyze thoroughly",
    ]

    for module_name, module in modules.items():
        print(f"\nProfiling {module_name} module:")

        # Profile multiple executions
        for i, test_input in enumerate(test_inputs):
            start_time = time.time()
            try:
                result = module(text=test_input)
                duration = time.time() - start_time
                profiler.add_measurement(f"{module_name}_execution", duration, True)
                print(f"  Test {i+1}: {duration:.3f}s - Success")
            except Exception as e:
                duration = time.time() - start_time
                profiler.add_measurement(f"{module_name}_execution", duration, False)
                print(f"  Test {i+1}: {duration:.3f}s - Error: {e}")

    return profiler, modules


def demonstrate_bottleneck_identification(profiler):
    """Demonstrate bottleneck identification and analysis"""
    print("\n2. Bottleneck identification and analysis:")
    print("-" * 40)

    # Get performance summaries
    print("Performance summaries:")
    for operation_name in profiler.profiles.keys():
        summary = profiler.get_profile_summary(operation_name)
        if summary:
            print(f"\n{operation_name}:")
            print(f"  • Average duration: {summary['avg_duration']:.3f}s")
            print(f"  • P95 duration: {summary['p95_duration']:.3f}s")
            print(f"  • Total calls: {summary['total_calls']}")
            print(f"  • Error rate: {summary['error_rate']:.1%}")
            print(f"  • Throughput: {summary['throughput']:.2f} ops/s")

    # Identify bottlenecks
    bottlenecks = profiler.identify_bottlenecks(threshold_seconds=1.0)

    if bottlenecks:
        print(f"\nIdentified {len(bottlenecks)} performance bottlenecks:")
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(
                f"\n{i}. {bottleneck['operation']} ({bottleneck['severity']} severity)"
            )
            print(f"   • Average duration: {bottleneck['avg_duration']:.3f}s")
            print(f"   • Impact score: {bottleneck['impact_score']:.2f}")
            print("   • Optimization suggestions:")
            for suggestion in bottleneck["suggestions"]:
                print(f"     - {suggestion}")
    else:
        print("\n✅ No significant bottlenecks identified")


def demonstrate_optimization_strategies(profiler, modules):
    """Demonstrate optimization strategies and their application"""
    print("\n3. Optimization strategies and application:")
    print("-" * 40)

    optimizer = DSPyOptimizer(profiler)

    # Set baseline performance
    profiler.set_baseline()
    print("Baseline performance established")

    # Get optimization suggestions for slow operations
    for operation_name in profiler.profiles.keys():
        if "slow" in operation_name:
            suggestions = optimizer.suggest_optimizations(operation_name)

            if suggestions:
                print(f"\nOptimization suggestions for {operation_name}:")
                for suggestion in suggestions:
                    print(
                        f"  • {suggestion['strategy']} ({suggestion['priority']} priority)"
                    )
                    print(
                        f"    Expected improvement: {suggestion['expected_improvement']}"
                    )
                    print(f"    Description: {suggestion['description']}")

                # Apply the highest priority optimization
                if suggestions:
                    best_suggestion = suggestions[0]
                    strategy = best_suggestion["strategy"]

                    print(f"\nApplying optimization: {strategy}")

                    # Get baseline performance
                    baseline_summary = profiler.get_profile_summary(operation_name)

                    # Apply optimization to module
                    module_name = operation_name.replace("_execution", "")
                    if module_name in modules:
                        module = modules[module_name]
                        current_config = getattr(module, "config", {})

                        try:
                            optimized_config, description = (
                                optimizer.apply_optimization(
                                    operation_name, strategy, module, current_config
                                )
                            )

                            # Update module configuration
                            module.config = optimized_config
                            print(f"  Applied: {description}")

                            # Test optimized performance
                            print("  Testing optimized performance...")
                            test_runs = 3
                            optimized_durations = []

                            for i in range(test_runs):
                                start_time = time.time()
                                try:
                                    result = module(text="Test input for optimization")
                                    duration = time.time() - start_time
                                    optimized_durations.append(duration)
                                    profiler.add_measurement(
                                        f"{operation_name}_optimized", duration, True
                                    )
                                except Exception as e:
                                    print(f"    Optimization test failed: {e}")

                            if optimized_durations:
                                avg_optimized = statistics.mean(optimized_durations)
                                improvement = (
                                    (baseline_summary["avg_duration"] - avg_optimized)
                                    / baseline_summary["avg_duration"]
                                    * 100
                                )

                                print(f"  Results:")
                                print(
                                    f"    Baseline avg: {baseline_summary['avg_duration']:.3f}s"
                                )
                                print(f"    Optimized avg: {avg_optimized:.3f}s")
                                print(f"    Improvement: {improvement:.1f}%")

                                # Record optimization result
                                optimizer.profiler.record_optimization(
                                    f"{operation_name}_{strategy}",
                                    {"avg_duration": baseline_summary["avg_duration"]},
                                    {"avg_duration": avg_optimized},
                                    description,
                                )

                        except Exception as e:
                            print(f"  Optimization failed: {e}")


def demonstrate_regression_detection(profiler):
    """Demonstrate performance regression detection"""
    print("\n4. Performance regression detection:")
    print("-" * 40)

    # Compare current performance to baseline
    comparisons = profiler.compare_to_baseline()

    if comparisons:
        print("Performance comparison to baseline:")

        regressions_found = False
        improvements_found = False

        for operation_name, comparison in comparisons.items():
            avg_change = comparison["avg_duration_change_percent"]
            p95_change = comparison["p95_duration_change_percent"]

            print(f"\n{operation_name}:")
            print(f"  • Average duration change: {avg_change:+.1f}%")
            print(f"  • P95 duration change: {p95_change:+.1f}%")

            # Detect regressions (>20% slower)
            if avg_change > 20:
                print(
                    f"  🚨 REGRESSION DETECTED: {avg_change:.1f}% slower than baseline"
                )
                regressions_found = True
            elif avg_change < -20:
                print(f"  ✅ IMPROVEMENT: {abs(avg_change):.1f}% faster than baseline")
                improvements_found = True
            else:
                print(f"  ➡️ Performance stable (within 20% of baseline)")

        # Summary
        if regressions_found:
            print("\n🚨 Performance regressions detected - investigation recommended")
        elif improvements_found:
            print("\n✅ Performance improvements detected - optimizations working")
        else:
            print("\n➡️ Performance stable - no significant changes")
    else:
        print("No baseline data available for comparison")


def demonstrate_optimization_results(profiler):
    """Demonstrate optimization results analysis"""
    print("\n5. Optimization results analysis:")
    print("-" * 40)

    if profiler.optimization_results:
        print("Optimization results summary:")

        total_optimizations = len(profiler.optimization_results)
        successful_optimizations = [
            r for r in profiler.optimization_results if r.improvement_percent > 0
        ]

        print(f"  • Total optimizations attempted: {total_optimizations}")
        print(f"  • Successful optimizations: {len(successful_optimizations)}")
        print(
            f"  • Success rate: {len(successful_optimizations) / total_optimizations * 100:.1f}%"
        )

        if successful_optimizations:
            avg_improvement = statistics.mean(
                [r.improvement_percent for r in successful_optimizations]
            )
            best_optimization = max(
                successful_optimizations, key=lambda r: r.improvement_percent
            )

            print(f"  • Average improvement: {avg_improvement:.1f}%")
            print(
                f"  • Best optimization: {best_optimization.optimization_name} ({best_optimization.improvement_percent:.1f}% improvement)"
            )

            print("\nDetailed optimization results:")
            for result in profiler.optimization_results:
                print(f"\n{result.optimization_name}:")
                print(f"  • Applied: {result.optimization_applied}")
                print(f"  • Improvement: {result.improvement_percent:.1f}%")
                print(
                    f"  • Baseline: {result.baseline_performance['avg_duration']:.3f}s"
                )
                print(
                    f"  • Optimized: {result.optimized_performance['avg_duration']:.3f}s"
                )
    else:
        print("No optimization results available")


def demonstrate_performance_best_practices():
    """Demonstrate performance optimization best practices"""
    print("\n6. Performance optimization best practices:")
    print("-" * 40)

    print("Best Practices for DSPy Performance Optimization:")
    print()

    print("1. Measurement and Profiling:")
    print("   • Always measure before optimizing")
    print("   • Profile in production-like conditions")
    print("   • Focus on operations with highest impact")
    print("   • Track multiple metrics (avg, p95, p99)")

    print("\n2. Common Optimization Strategies:")
    print("   • Reduce max_tokens for faster responses")
    print("   • Implement caching for repeated requests")
    print("   • Optimize prompt structure and length")
    print("   • Use batch processing for high throughput")
    print("   • Choose appropriate model for use case")

    print("\n3. Monitoring and Alerting:")
    print("   • Set up performance regression alerts")
    print("   • Monitor key performance indicators")
    print("   • Track optimization effectiveness")
    print("   • Establish performance baselines")

    print("\n4. Optimization Process:")
    print("   • Identify bottlenecks systematically")
    print("   • Apply one optimization at a time")
    print("   • Measure impact of each change")
    print("   • Validate optimizations in production")
    print("   • Document successful optimizations")

    print("\n5. Performance Targets:")
    print("   • Set realistic performance goals")
    print("   • Consider user experience requirements")
    print("   • Balance speed vs. quality")
    print("   • Account for peak load scenarios")


if __name__ == "__main__":
    """
    Exercise Solution: DSPy Performance Optimization

    This script demonstrates:
    1. Comprehensive performance profiling
    2. Bottleneck identification and analysis
    3. Optimization strategies and application
    4. Performance regression detection
    5. Optimization results analysis
    6. Performance optimization best practices
    """

    try:
        # Run the complete performance optimization demonstration
        profiler, modules = demonstrate_performance_profiling()
        demonstrate_bottleneck_identification(profiler)
        demonstrate_optimization_strategies(profiler, modules)
        demonstrate_regression_detection(profiler)
        demonstrate_optimization_results(profiler)
        demonstrate_performance_best_practices()

        print("\n✅ Performance optimization exercise completed successfully!")
        print("\nKey takeaways:")
        print("- Systematic profiling identifies optimization opportunities")
        print("- Bottleneck analysis prioritizes optimization efforts")
        print("- Multiple optimization strategies can be applied")
        print("- Regression detection prevents performance degradation")
        print("- Continuous monitoring ensures sustained performance")
        print("- Best practices guide effective optimization processes")

    except Exception as e:
        print(f"\n❌ Exercise failed: {e}")
        logger.exception("Exercise execution failed")
