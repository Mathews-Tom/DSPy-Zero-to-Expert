# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore marimo dspy

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import logging
    import statistics
    import sys
    import threading
    import time
    from collections import defaultdict, deque
    from collections.abc import Callable
    from dataclasses import dataclass, field
    from datetime import datetime, timedelta
    from inspect import cleandoc
    from pathlib import Path
    from typing import Any, Optional

    import marimo as mo
    import psutil
    from marimo import output

    # Add current directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))

    import dspy

    from common.utils import get_config

    # Configure logging for performance monitoring
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return (
        Any,
        Callable,
        Optional,
        cleandoc,
        dataclass,
        datetime,
        defaultdict,
        deque,
        dspy,
        field,
        get_config,
        json,
        logger,
        mo,
        output,
        psutil,
        statistics,
        threading,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    # Title and introduction
    cell1_out = mo.md(
        cleandoc(
            """
            # ⚡ DSPy Performance Monitor

            Welcome to the advanced performance monitoring system for DSPy! This comprehensive tool provides:

            - **Real-time Performance Tracking**: Monitor execution times, throughput, and resource usage
            - **Bottleneck Identification**: Automatically detect performance bottlenecks
            - **Resource Monitoring**: Track CPU, memory, and network usage
            - **Performance Profiling**: Deep dive into module-level performance
            - **Optimization Suggestions**: Get actionable recommendations for improvements
            - **Regression Detection**: Identify performance regressions over time
            - **Load Testing**: Simulate high-load scenarios to test system limits

            Optimize your DSPy systems for peak performance! 🚀
            """
        )
    )

    output.replace(cell1_out)
    return


@app.cell
def _(
    Any,
    Callable,
    Optional,
    dataclass,
    defaultdict,
    deque,
    field,
    logger,
    psutil,
    statistics,
    threading,
    time,
):
    # Performance Monitoring Core Classes
    @dataclass
    class PerformanceMetric:
        """Represents a performance metric measurement"""

        name: str
        value: float
        timestamp: float
        tags: dict[str, str] = field(default_factory=dict)
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class ResourceUsage:
        """System resource usage snapshot"""

        timestamp: float
        cpu_percent: float
        memory_percent: float
        memory_used_mb: float
        disk_io_read_mb: float
        disk_io_write_mb: float
        network_sent_mb: float
        network_recv_mb: float

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
                    self.p95_duration = sorted_durations[
                        int(len(sorted_durations) * 0.95)
                    ]
                    self.p99_duration = sorted_durations[
                        int(len(sorted_durations) * 0.99)
                    ]
            else:
                self.error_count += 1

            self.last_updated = time.time()

    class PerformanceMonitor:
        """Advanced performance monitoring system"""

        def __init__(self, max_history: int = 10000):
            self.metrics: dict[str, deque] = defaultdict(
                lambda: deque(maxlen=max_history)
            )
            self.profiles: dict[str, PerformanceProfile] = {}
            self.resource_history: deque = deque(maxlen=max_history)
            self.monitoring_active = False
            self.monitoring_thread: Optional[threading.Thread] = None
            self.baseline_metrics: dict[str, float] = {}
            self.performance_alerts: list[dict[str, Any]] = []

        def start_monitoring(self, interval: float = 1.0):
            """Start continuous performance monitoring"""
            if self.monitoring_active:
                return

            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, args=(interval,), daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Performance monitoring started")

        def stop_monitoring(self):
            """Stop performance monitoring"""
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2.0)
            logger.info("Performance monitoring stopped")

        def _monitoring_loop(self, interval: float):
            """Main monitoring loop"""
            while self.monitoring_active:
                try:
                    self._collect_system_metrics()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")

        def _collect_system_metrics(self):
            """Collect system resource metrics"""
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()

                # Disk I/O
                disk_io = psutil.disk_io_counters()
                disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
                disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0

                # Network I/O
                network_io = psutil.net_io_counters()
                network_sent_mb = (
                    network_io.bytes_sent / (1024 * 1024) if network_io else 0
                )
                network_recv_mb = (
                    network_io.bytes_recv / (1024 * 1024) if network_io else 0
                )

                resource_usage = ResourceUsage(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb,
                    network_sent_mb=network_sent_mb,
                    network_recv_mb=network_recv_mb,
                )

                self.resource_history.append(resource_usage)

                # Record as metrics
                self.record_metric("cpu_usage", cpu_percent, {"unit": "percent"})
                self.record_metric("memory_usage", memory.percent, {"unit": "percent"})

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")

        def record_metric(
            self,
            name: str,
            value: float,
            tags: Optional[dict] = None,
            metadata: Optional[dict] = None,
        ):
            """Record a performance metric"""
            metric = PerformanceMetric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metadata=metadata or {},
            )
            self.metrics[name].append(metric)

        def profile_operation(self, operation_name: str):
            """Decorator/context manager for profiling operations"""

            def decorator(func: Callable):
                def wrapper(*args, **kwargs):
                    start_time = time.time()
                    success = True
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as _:
                        success = False
                        raise
                    finally:
                        duration = time.time() - start_time
                        self.add_profile_measurement(operation_name, duration, success)

                return wrapper

            return decorator

        def add_profile_measurement(
            self, operation_name: str, duration: float, success: bool = True
        ):
            """Add a measurement to an operation profile"""
            if operation_name not in self.profiles:
                self.profiles[operation_name] = PerformanceProfile(operation_name)

            self.profiles[operation_name].add_measurement(duration, success)

            # Record as metric
            self.record_metric(
                f"{operation_name}_duration",
                duration,
                {"operation": operation_name, "success": str(success)},
            )

        def get_performance_summary(self, duration_minutes: int = 60) -> dict[str, Any]:
            """Get comprehensive performance summary"""
            cutoff_time = time.time() - (duration_minutes * 60)

            # Resource usage summary
            recent_resources = [
                r for r in self.resource_history if r.timestamp >= cutoff_time
            ]
            resource_summary = {}
            if recent_resources:
                resource_summary = {
                    "cpu": {
                        "avg": statistics.mean(
                            [r.cpu_percent for r in recent_resources]
                        ),
                        "max": max([r.cpu_percent for r in recent_resources]),
                        "current": (
                            recent_resources[-1].cpu_percent if recent_resources else 0
                        ),
                    },
                    "memory": {
                        "avg": statistics.mean(
                            [r.memory_percent for r in recent_resources]
                        ),
                        "max": max([r.memory_percent for r in recent_resources]),
                        "current": (
                            recent_resources[-1].memory_percent
                            if recent_resources
                            else 0
                        ),
                    },
                }

            # Operation profiles summary
            profiles_summary = {}
            for name, profile in self.profiles.items():
                profiles_summary[name] = {
                    "total_calls": profile.total_calls,
                    "avg_duration": profile.avg_duration,
                    "p95_duration": profile.p95_duration,
                    "p99_duration": profile.p99_duration,
                    "error_rate": (
                        profile.error_count / profile.total_calls
                        if profile.total_calls > 0
                        else 0
                    ),
                    "throughput": (
                        profile.total_calls / (duration_minutes * 60)
                        if duration_minutes > 0
                        else 0
                    ),
                }

            return {
                "resource_usage": resource_summary,
                "operation_profiles": profiles_summary,
                "monitoring_duration": duration_minutes,
                "total_operations": sum(p.total_calls for p in self.profiles.values()),
            }

        def detect_bottlenecks(self) -> list[dict[str, Any]]:
            """Detect performance bottlenecks"""
            bottlenecks = []

            # Check for slow operations
            for name, profile in self.profiles.items():
                if profile.avg_duration > 5.0:  # Slower than 5 seconds
                    bottlenecks.append(
                        {
                            "type": "slow_operation",
                            "operation": name,
                            "avg_duration": profile.avg_duration,
                            "severity": (
                                "high" if profile.avg_duration > 10.0 else "medium"
                            ),
                            "suggestion": f"Operation {name} is slow (avg: {profile.avg_duration:.2f}s). Consider optimization.",
                        }
                    )

            # Check for high error rates
            for name, profile in self.profiles.items():
                if profile.total_calls > 0:
                    error_rate = profile.error_count / profile.total_calls
                    if error_rate > 0.1:  # More than 10% error rate
                        bottlenecks.append(
                            {
                                "type": "high_error_rate",
                                "operation": name,
                                "error_rate": error_rate,
                                "severity": "critical" if error_rate > 0.25 else "high",
                                "suggestion": f"Operation {name} has high error rate ({error_rate:.1%}). Review error handling.",
                            }
                        )

            # Check resource usage
            if self.resource_history:
                recent_cpu = [
                    r.cpu_percent for r in list(self.resource_history)[-60:]
                ]  # Last 60 measurements
                recent_memory = [
                    r.memory_percent for r in list(self.resource_history)[-60:]
                ]

                if recent_cpu and statistics.mean(recent_cpu) > 80:
                    bottlenecks.append(
                        {
                            "type": "high_cpu_usage",
                            "avg_cpu": statistics.mean(recent_cpu),
                            "severity": "high",
                            "suggestion": "High CPU usage detected. Consider scaling or optimization.",
                        }
                    )

                if recent_memory and statistics.mean(recent_memory) > 85:
                    bottlenecks.append(
                        {
                            "type": "high_memory_usage",
                            "avg_memory": statistics.mean(recent_memory),
                            "severity": "high",
                            "suggestion": "High memory usage detected. Check for memory leaks or consider scaling.",
                        }
                    )

            return bottlenecks

        def set_baseline(self):
            """Set current performance as baseline for regression detection"""
            self.baseline_metrics = {}
            for name, profile in self.profiles.items():
                self.baseline_metrics[name] = {
                    "avg_duration": profile.avg_duration,
                    "p95_duration": profile.p95_duration,
                    "error_rate": (
                        profile.error_count / profile.total_calls
                        if profile.total_calls > 0
                        else 0
                    ),
                }
            logger.info("Performance baseline set")

        def detect_regressions(self, threshold: float = 0.2) -> list[dict[str, Any]]:
            """Detect performance regressions compared to baseline"""
            if not self.baseline_metrics:
                return []

            regressions = []
            for name, profile in self.profiles.items():
                if name not in self.baseline_metrics:
                    continue

                baseline = self.baseline_metrics[name]
                current_error_rate = (
                    profile.error_count / profile.total_calls
                    if profile.total_calls > 0
                    else 0
                )

                # Check duration regression
                if profile.avg_duration > baseline["avg_duration"] * (1 + threshold):
                    regressions.append(
                        {
                            "type": "duration_regression",
                            "operation": name,
                            "baseline_duration": baseline["avg_duration"],
                            "current_duration": profile.avg_duration,
                            "regression_percent": (
                                (profile.avg_duration / baseline["avg_duration"]) - 1
                            )
                            * 100,
                            "severity": "high",
                        }
                    )

                # Check error rate regression
                if current_error_rate > baseline["error_rate"] * (1 + threshold):
                    regressions.append(
                        {
                            "type": "error_rate_regression",
                            "operation": name,
                            "baseline_error_rate": baseline["error_rate"],
                            "current_error_rate": current_error_rate,
                            "regression_percent": (
                                ((current_error_rate / baseline["error_rate"]) - 1)
                                * 100
                                if baseline["error_rate"] > 0
                                else float("inf")
                            ),
                            "severity": "critical",
                        }
                    )

            return regressions

    # Initialize global performance monitor
    perf_monitor = PerformanceMonitor()

    return (perf_monitor,)


@app.cell
def _(mo, output):
    # Performance Monitoring Configuration
    cell3_desc = mo.md("## ⚙️ Step 1: Configure Performance Monitoring")

    enable_monitoring = mo.ui.checkbox(
        value=True, label="Enable continuous performance monitoring"
    )

    monitoring_interval = mo.ui.slider(
        start=0.5,
        stop=5.0,
        step=0.5,
        value=1.0,
        label="Monitoring interval (seconds)",
    )

    enable_profiling = mo.ui.checkbox(value=True, label="Enable operation profiling")

    enable_resource_monitoring = mo.ui.checkbox(
        value=True, label="Enable system resource monitoring"
    )

    start_monitoring_button = mo.ui.run_button(
        label="Start Performance Monitoring", kind="success"
    )

    cell3_content = mo.vstack(
        [
            enable_monitoring,
            monitoring_interval,
            enable_profiling,
            enable_resource_monitoring,
            start_monitoring_button,
        ]
    )

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (
        enable_monitoring,
        enable_profiling,
        monitoring_interval,
        start_monitoring_button,
    )


@app.cell
def _(
    enable_monitoring,
    monitoring_interval,
    perf_monitor,
    start_monitoring_button,
):
    # Initialize Performance Monitoring
    monitoring_status = "stopped"

    if start_monitoring_button.value and enable_monitoring.value:
        perf_monitor.start_monitoring(monitoring_interval.value)
        monitoring_status = "running"

    return (monitoring_status,)


@app.cell
def _(cleandoc, mo, monitoring_status, output):
    # Monitoring Status Display
    if monitoring_status == "running":
        cell5_out = mo.md(
            cleandoc(
                """
                ✅ **Performance Monitoring Active**
                - System resource monitoring enabled
                - Operation profiling ready
                - Real-time metrics collection started
                - Bottleneck detection active
                """
            )
        )
    else:
        cell5_out = mo.md(
            "⚠️ Configure and start performance monitoring above to continue"
        )

    output.replace(cell5_out)
    return


@app.cell
def _(get_config, mo, output):
    # DSPy Module Setup for Performance Testing
    cell6_desc = mo.md("## 🧠 Step 2: Configure DSPy Module for Performance Testing")

    # Get available LLM providers
    try:
        config = get_config()
        available_providers = config.get_available_llm_providers()
    except Exception:
        available_providers = ["openai", "anthropic", "local"]

    provider_selector = mo.ui.dropdown(
        options=available_providers,
        value=available_providers[0] if available_providers else "openai",
        label="Select LLM Provider",
    )

    module_type = mo.ui.dropdown(
        options=["Predict", "ChainOfThought", "ReAct"],
        value="ChainOfThought",
        label="Select DSPy Module Type",
    )

    cell6_content = mo.vstack([provider_selector, module_type])

    cell6_out = mo.vstack([cell6_desc, cell6_content])
    output.replace(cell6_out)
    return module_type, provider_selector


@app.cell
def _(dspy, module_type, provider_selector):
    # Initialize DSPy Module for Performance Testing
    perf_module = None
    perf_signature = None

    if provider_selector.value and module_type.value:
        try:
            # Configure DSPy with selected provider
            if provider_selector.value == "openai":
                lm = dspy.LM("openai/gpt-5")
            elif provider_selector.value == "anthropic":
                lm = dspy.LM("anthropic/claude-3-7-sonnet-20250219")
            else:
                lm = dspy.LM(
                    "openai/gpt-3.5-turbo", api_base="http://localhost:1234/v1"
                )

            dspy.configure(lm=lm)

            # Create signature for performance testing
            class PerformanceSignature(dspy.Signature):
                """Generate a concise response for performance testing"""

                input_text = dspy.InputField(desc="Input text to process")
                output_text = dspy.OutputField(desc="Processed output")

            perf_signature = PerformanceSignature

            # Create module based on selection
            if module_type.value == "Predict":
                perf_module = dspy.Predict(PerformanceSignature)
            elif module_type.value == "ChainOfThought":
                perf_module = dspy.ChainOfThought(PerformanceSignature)
            else:  # ReAct
                perf_module = dspy.ReAct(PerformanceSignature)

        except Exception as _:
            pass

    return (perf_module,)


@app.cell
def _(mo, monitoring_status, output, perf_module):
    # Performance Test Configuration
    if not perf_module or monitoring_status != "running":
        cell8_desc = mo.md("*Configure monitoring and DSPy module first*")
        cell8_content = mo.md("")
        test_input = None
        num_iterations = None
        run_test = None
    else:
        cell8_desc = mo.md("## 🧪 Step 3: Configure Performance Test")

        test_input = mo.ui.text_area(
            placeholder="Enter test input...",
            label="Test Input",
            rows=2,
            value="Explain the concept of machine learning in simple terms.",
        )

        num_iterations = mo.ui.slider(
            start=1,
            stop=20,
            step=1,
            value=5,
            label="Number of test iterations",
        )

        run_test = mo.ui.run_button(label="Run Performance Test", kind="success")

        cell8_content = mo.vstack([test_input, num_iterations, run_test])

    cell8_out = mo.vstack([cell8_desc, cell8_content])
    output.replace(cell8_out)
    return num_iterations, run_test, test_input


@app.cell
def _(
    enable_profiling,
    module_type,
    num_iterations,
    perf_module,
    perf_monitor,
    run_test,
    test_input,
    time,
):
    # Execute Performance Test
    test_results = []
    test_summary = None

    if run_test and run_test.value and perf_module and test_input and test_input.value:
        operation_name = f"{module_type.value}_execution"

        # Run performance tests
        for i in range(num_iterations.value):
            start_time = time.time()
            success = True
            error = None

            try:
                result = perf_module(input_text=test_input.value)
                duration = time.time() - start_time

                test_results.append(
                    {
                        "iteration": i + 1,
                        "duration": duration,
                        "success": True,
                        "output_length": len(result.output_text),
                    }
                )

            except Exception as e:
                duration = time.time() - start_time
                success = False
                error = str(e)

                test_results.append(
                    {
                        "iteration": i + 1,
                        "duration": duration,
                        "success": False,
                        "error": error,
                    }
                )

            # Record performance measurement
            if enable_profiling.value:
                perf_monitor.add_profile_measurement(operation_name, duration, success)

            # Small delay between iterations
            time.sleep(0.1)

        # Generate test summary
        successful_tests = [r for r in test_results if r["success"]]
        if successful_tests:
            durations = [r["duration"] for r in successful_tests]
            test_summary = {
                "total_iterations": len(test_results),
                "successful_iterations": len(successful_tests),
                "failed_iterations": len(test_results) - len(successful_tests),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "throughput": (
                    len(successful_tests) / sum(durations) if sum(durations) > 0 else 0
                ),
            }

    return test_results, test_summary


@app.cell
def _(cleandoc, mo, output, test_results, test_summary):
    # Display Performance Test Results
    if not test_results:
        cell10_out = mo.md("*Run performance test above to see results*")
    elif test_summary:
        cell10_desc = mo.md("## 📊 Performance Test Results")

        _summary_content = cleandoc(
            f"""
            ### Test Summary
            - **Total Iterations**: {test_summary['total_iterations']}
            - **Successful**: {test_summary['successful_iterations']}
            - **Failed**: {test_summary['failed_iterations']}
            - **Success Rate**: {(test_summary['successful_iterations'] / test_summary['total_iterations']) * 100:.1f}%

            ### Performance Metrics
            - **Average Duration**: {test_summary['avg_duration']:.3f}s
            - **Min Duration**: {test_summary['min_duration']:.3f}s
            - **Max Duration**: {test_summary['max_duration']:.3f}s
            - **Throughput**: {test_summary['throughput']:.2f} requests/second
            """
        )

        cell10_content = mo.md(_summary_content)
        cell10_out = mo.vstack([cell10_desc, cell10_content])
    else:
        cell10_out = mo.md("❌ All test iterations failed")

    output.replace(cell10_out)
    return


@app.cell
def _(cleandoc, mo, output, perf_monitor):
    # Real-time Performance Dashboard
    cell11_perf_summary = perf_monitor.get_performance_summary(5)  # Last 5 minutes

    if not cell11_perf_summary["operation_profiles"]:
        cell11_out = mo.md("*No performance data available yet - run some tests first*")
    else:
        cell11_desc = mo.md("## 📈 Real-time Performance Dashboard")

        # Resource usage
        _resource_usage = cell11_perf_summary["resource_usage"]
        if _resource_usage:
            _resource_html = cleandoc(
                f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                    <div style="background: #e7f3ff; padding: 15px; border-radius: 8px; text-align: center;">
                        <h4>💻 CPU Usage</h4>
                        <div style="font-size: 24px; font-weight: bold; color: #0066cc;">
                            {_resource_usage['cpu']['current']:.1f}%
                        </div>
                        <div style="font-size: 12px; color: #666;">
                            Avg: {_resource_usage['cpu']['avg']:.1f}% | Max: {_resource_usage['cpu']['max']:.1f}%
                        </div>
                    </div>
                    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; text-align: center;">
                        <h4>🧠 Memory Usage</h4>
                        <div style="font-size: 24px; font-weight: bold; color: #856404;">
                            {_resource_usage['memory']['current']:.1f}%
                        </div>
                        <div style="font-size: 12px; color: #666;">
                            Avg: {_resource_usage['memory']['avg']:.1f}% | Max: {_resource_usage['memory']['max']:.1f}%
                        </div>
                    </div>
                </div>
                """
            )
        else:
            _resource_html = "<p>No resource data available</p>"

        # Operation profiles
        _profiles_content = []
        for op_name, profile in cell11_perf_summary["operation_profiles"].items():
            _profile_info = cleandoc(
                f"""
                **{op_name}:**
                - Calls: {profile['total_calls']}
                - Avg Duration: {profile['avg_duration']:.3f}s
                - P95 Duration: {profile['p95_duration']:.3f}s
                - Error Rate: {profile['error_rate']:.1%}
                - Throughput: {profile['throughput']:.2f} ops/sec
                """
            )
            _profiles_content.append(_profile_info)

        _profiles_md = "\n".join(_profiles_content)

        cell11_content = mo.vstack(
            [
                mo.Html(_resource_html),
                mo.md("### Operation Profiles"),
                mo.md(_profiles_md),
            ]
        )

        cell11_out = mo.vstack([cell11_desc, cell11_content])

    output.replace(cell11_out)
    return


@app.cell
def _(cleandoc, mo, output, perf_monitor):
    # Bottleneck Detection
    cell12_bottlenecks = perf_monitor.detect_bottlenecks()

    cell12_desc = mo.md("## 🔍 Bottleneck Analysis")

    if not cell12_bottlenecks:
        cell12_content = mo.md(
            "✅ **No performance bottlenecks detected** - System is performing well!"
        )
    else:
        _bottleneck_content = []
        for bottleneck in cell12_bottlenecks:
            _severity_emoji = {
                "low": "🟡",
                "medium": "🟠",
                "high": "🔴",
                "critical": "🚨",
            }.get(bottleneck["severity"], "⚠️")

            _bottleneck_info = cleandoc(
                f"""
                {_severity_emoji} **{bottleneck['type'].replace('_', ' ').title()}** ({bottleneck['severity'].upper()})
                - {bottleneck['suggestion']}
                """
            )

            if "operation" in bottleneck:
                _bottleneck_info += f"- Operation: {bottleneck['operation']}\n"
            if "avg_duration" in bottleneck:
                _bottleneck_info += (
                    f"- Average Duration: {bottleneck['avg_duration']:.3f}s\n"
                )
            if "error_rate" in bottleneck:
                _bottleneck_info += f"- Error Rate: {bottleneck['error_rate']:.1%}\n"
            if "avg_cpu" in bottleneck:
                _bottleneck_info += f"- Average CPU: {bottleneck['avg_cpu']:.1f}%\n"
            if "avg_memory" in bottleneck:
                _bottleneck_info += (
                    f"- Average Memory: {bottleneck['avg_memory']:.1f}%\n"
                )

            _bottleneck_content.append(_bottleneck_info)

        cell12_content = mo.md("\n".join(_bottleneck_content))

    cell12_out = mo.vstack([cell12_desc, cell12_content])
    output.replace(cell12_out)
    return


@app.cell
def _(mo, output):
    # Baseline and Regression Detection
    cell13_desc = mo.md("## 📊 Step 4: Baseline & Regression Detection")

    set_baseline_button = mo.ui.run_button(
        label="Set Performance Baseline", kind="success"
    )

    check_regressions = mo.ui.run_button(label="Check for Regressions", kind="info")

    cell13_content = mo.vstack([set_baseline_button, check_regressions])

    cell13_out = mo.vstack([cell13_desc, cell13_content])
    output.replace(cell13_out)
    return check_regressions, set_baseline_button


@app.cell
def _(
    check_regressions,
    cleandoc,
    mo,
    output,
    perf_monitor,
    set_baseline_button,
):
    # Handle Baseline and Regression Detection
    if set_baseline_button.value:
        perf_monitor.set_baseline()
        cell14_out = mo.md("✅ **Performance baseline set successfully**")
    elif check_regressions.value:
        _regressions = perf_monitor.detect_regressions()

        if not _regressions:
            cell14_out = mo.md(
                "✅ **No performance regressions detected** - Performance is stable!"
            )
        else:
            cell14_desc = mo.md("### 🚨 Performance Regressions Detected")

            _regression_content = []
            for regression in _regressions:
                _reg_info = cleandoc(
                    f"""
                    **{regression['type'].replace('_', ' ').title()}** - {regression['operation']}
                    - Regression: {regression['regression_percent']:.1f}% worse than baseline
                    - Severity: {regression['severity'].upper()}
                    """
                )

                if "baseline_duration" in regression:
                    _reg_info += f"- Baseline: {regression['baseline_duration']:.3f}s → Current: {regression['current_duration']:.3f}s\n"
                if "baseline_error_rate" in regression:
                    _reg_info += f"- Baseline Error Rate: {regression['baseline_error_rate']:.1%} → Current: {regression['current_error_rate']:.1%}\n"

                _regression_content.append(_reg_info)

            cell14_content = mo.md("\n".join(_regression_content))
            cell14_out = mo.vstack([cell14_desc, cell14_content])
    else:
        cell14_out = mo.md(
            "*Set a baseline or check for regressions using the buttons above*"
        )

    output.replace(cell14_out)
    return


@app.cell
def _(mo, output):
    # Export Performance Data
    cell15_desc = mo.md("## 💾 Step 5: Export Performance Data")

    cell15_export_format = mo.ui.dropdown(
        options=["JSON", "CSV"],
        value="JSON",
        label="Export Format",
    )

    export_button = mo.ui.button(label="Export Performance Data", kind="info")

    cell15_content = mo.vstack([cell15_export_format, export_button])

    cell15_out = mo.vstack([cell15_desc, cell15_content])
    output.replace(cell15_out)
    return cell15_export_format, export_button


@app.cell
def _(
    cell15_export_format,
    datetime,
    export_button,
    json,
    mo,
    output,
    perf_monitor,
):
    # Handle Performance Data Export
    if export_button and export_button.value:
        try:
            _perf_data = perf_monitor.get_performance_summary(60)  # Last hour
            _bottlenecks = perf_monitor.detect_bottlenecks()
            _regressions = perf_monitor.detect_regressions()

            _export_data = {
                "timestamp": datetime.now().isoformat(),
                "performance_summary": _perf_data,
                "bottlenecks": _bottlenecks,
                "regressions": _regressions,
                "baseline_metrics": perf_monitor.baseline_metrics,
            }

            _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _filename = (
                f"performance_data_{_timestamp}.{cell15_export_format.value.lower()}"
            )

            with open(_filename, "w", encoding="utf-8") as f:
                json.dump(_export_data, f, indent=2, default=str)

            cell16_out = mo.md(f"✅ **Performance data exported to {_filename}**")
        except Exception as e:
            cell16_out = mo.md(f"❌ **Export failed:** {str(e)}")
    else:
        cell16_out = mo.md(
            "*Click 'Export Performance Data' to save performance metrics*"
        )

    output.replace(cell16_out)
    return


@app.cell
def _(cleandoc, mo, output):
    # Footer with performance optimization tips
    cell17_out = mo.md(
        cleandoc(
            """
            ## 🎯 Performance Optimization Best Practices

            ### ✅ What You've Learned

            - **Real-time Monitoring**: Track system performance and resource usage continuously
            - **Operation Profiling**: Measure and analyze individual operation performance
            - **Bottleneck Detection**: Automatically identify performance bottlenecks
            - **Regression Detection**: Compare current performance against baselines
            - **Resource Monitoring**: Track CPU, memory, and I/O usage

            ### 💡 Pro Tips

            1. **Set Baselines Early**: Establish performance baselines when your system is working well
            2. **Monitor Continuously**: Use continuous monitoring to catch issues early
            3. **Profile Critical Operations**: Focus profiling on your most important operations
            4. **Watch Resource Usage**: High CPU/memory usage often indicates bottlenecks
            5. **Regular Regression Checks**: Regularly check for performance regressions

            ### 🚀 Optimization Strategies

            - **Model Selection**: Use faster models for non-critical operations
            - **Caching**: Implement caching for frequently requested operations
            - **Batch Processing**: Process multiple requests together when possible
            - **Resource Scaling**: Scale resources based on usage patterns
            - **Code Optimization**: Optimize slow operations identified by profiling

            ### 📊 Key Metrics to Monitor

            - **Response Time**: Average, P95, P99 response times
            - **Throughput**: Requests per second
            - **Error Rate**: Percentage of failed requests
            - **Resource Usage**: CPU, memory, disk, network utilization
            - **Queue Depth**: Number of pending requests

            **Ready for production?** Use these insights to optimize your DSPy applications! 🚀
            """
        )
    )

    output.replace(cell17_out)
    return


if __name__ == "__main__":
    app.run()
