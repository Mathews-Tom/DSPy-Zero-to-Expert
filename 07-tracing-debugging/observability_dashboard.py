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
    import time
    from collections import defaultdict, deque
    from dataclasses import dataclass, field
    from datetime import datetime, timedelta
    from inspect import cleandoc
    from pathlib import Path
    from typing import Any, Optional

    import dspy
    import marimo as mo
    from marimo import output

    from common.utils import get_config

    # Add current directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))

    # Configure logging for observability
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return (
        Any,
        Optional,
        cleandoc,
        dataclass,
        datetime,
        defaultdict,
        deque,
        field,
        get_config,
        json,
        logger,
        mo,
        output,
        statistics,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    # Title and introduction
    cell1_out = mo.md(
        cleandoc(
            """
            # 📊 DSPy Observability Dashboard

            Welcome to the comprehensive observability dashboard for DSPy systems! This real-time monitoring system provides:

            - **System Health Monitoring**: Track overall system performance and health
            - **Real-time Metrics**: Monitor key performance indicators in real-time
            - **Performance Analytics**: Analyze trends and identify patterns
            - **Alert Management**: Set up alerts for critical system events
            - **Resource Utilization**: Monitor CPU, memory, and API usage
            - **Error Tracking**: Track and analyze system errors and failures
            - **Historical Analysis**: View performance trends over time

            Monitor your DSPy systems like a pro! 📈
            """
        )
    )

    output.replace(cell1_out)
    return


@app.cell
def _(
    Any,
    Optional,
    dataclass,
    defaultdict,
    deque,
    field,
    logger,
    statistics,
    time,
):
    # Observability System Core Classes
    @dataclass
    class Metric:
        """Represents a system metric"""

        name: str
        value: float
        timestamp: float
        tags: dict[str, str] = field(default_factory=dict)
        unit: str = ""

    @dataclass
    class Alert:
        """Represents a system alert"""

        alert_id: str
        metric_name: str
        threshold: float
        condition: str  # 'greater_than', 'less_than', 'equals'
        severity: str  # 'low', 'medium', 'high', 'critical'
        message: str
        enabled: bool = True
        triggered: bool = False
        last_triggered: Optional[float] = None

    @dataclass
    class SystemHealth:
        """Overall system health status"""

        status: str  # 'healthy', 'warning', 'critical'
        score: float  # 0-100
        issues: list[str] = field(default_factory=list)
        recommendations: list[str] = field(default_factory=list)
        last_updated: float = field(default_factory=time.time)

    class MetricsCollector:
        """Collects and manages system metrics"""

        def __init__(self, max_history: int = 1000):
            self.metrics: dict[str, deque] = defaultdict(
                lambda: deque(maxlen=max_history)
            )
            self.alerts: dict[str, Alert] = {}
            self.system_health = SystemHealth(status="unknown", score=0)

        def record_metric(
            self, name: str, value: float, tags: Optional[dict] = None, unit: str = ""
        ):
            """Record a new metric value"""
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                unit=unit,
            )
            self.metrics[name].append(metric)
            self._check_alerts(metric)

        def get_metric_history(
            self, name: str, duration_minutes: int = 60
        ) -> list[Metric]:
            """Get metric history for the specified duration"""
            if name not in self.metrics:
                return []

            cutoff_time = time.time() - (duration_minutes * 60)
            return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]

        def get_metric_stats(
            self, name: str, duration_minutes: int = 60
        ) -> dict[str, float]:
            """Get statistical summary of a metric"""
            history = self.get_metric_history(name, duration_minutes)
            if not history:
                return {}

            values = [m.value for m in history]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "latest": values[-1] if values else 0,
            }

        def add_alert(self, alert: Alert):
            """Add a new alert"""
            self.alerts[alert.alert_id] = alert

        def _check_alerts(self, metric: Metric):
            """Check if any alerts should be triggered"""
            for alert in self.alerts.values():
                if not alert.enabled or alert.metric_name != metric.name:
                    continue

                should_trigger = False
                if alert.condition == "greater_than" and metric.value > alert.threshold:
                    should_trigger = True
                elif alert.condition == "less_than" and metric.value < alert.threshold:
                    should_trigger = True
                elif alert.condition == "equals" and metric.value == alert.threshold:
                    should_trigger = True

                if should_trigger and not alert.triggered:
                    alert.triggered = True
                    alert.last_triggered = time.time()
                    logger.warning(f"Alert triggered: {alert.message}")
                elif not should_trigger and alert.triggered:
                    alert.triggered = False

        def update_system_health(self):
            """Update overall system health status"""
            issues = []
            recommendations = []
            score = 100

            # Check for active alerts
            critical_alerts = [
                a
                for a in self.alerts.values()
                if a.triggered and a.severity == "critical"
            ]
            high_alerts = [
                a for a in self.alerts.values() if a.triggered and a.severity == "high"
            ]
            medium_alerts = [
                a
                for a in self.alerts.values()
                if a.triggered and a.severity == "medium"
            ]

            if critical_alerts:
                issues.extend([f"Critical: {a.message}" for a in critical_alerts])
                score -= len(critical_alerts) * 30
            if high_alerts:
                issues.extend([f"High: {a.message}" for a in high_alerts])
                score -= len(high_alerts) * 20
            if medium_alerts:
                issues.extend([f"Medium: {a.message}" for a in medium_alerts])
                score -= len(medium_alerts) * 10

            # Check performance metrics
            response_time_stats = self.get_metric_stats("response_time", 10)
            if response_time_stats and response_time_stats["mean"] > 5.0:
                issues.append("High average response time")
                recommendations.append(
                    "Consider optimizing model parameters or using a faster model"
                )
                score -= 15

            error_rate_stats = self.get_metric_stats("error_rate", 10)
            if error_rate_stats and error_rate_stats["latest"] > 0.1:
                issues.append("High error rate")
                recommendations.append("Review error logs and improve error handling")
                score -= 20

            # Determine status
            if score >= 90:
                status = "healthy"
            elif score >= 70:
                status = "warning"
            else:
                status = "critical"

            self.system_health = SystemHealth(
                status=status,
                score=max(0, score),
                issues=issues,
                recommendations=recommendations,
                last_updated=time.time(),
            )

    class DSPyMonitor:
        """Main monitoring system for DSPy applications"""

        def __init__(self):
            self.collector = MetricsCollector()
            self.monitoring_active = False
            self.start_time = time.time()

        def start_monitoring(self):
            """Start monitoring system"""
            self.monitoring_active = True
            self.start_time = time.time()
            logger.info("DSPy monitoring started")

        def stop_monitoring(self):
            """Stop monitoring system"""
            self.monitoring_active = False
            logger.info("DSPy monitoring stopped")

        def record_execution(
            self,
            module_name: str,
            duration: float,
            success: bool,
            error: Optional[str] = None,
        ):
            """Record a module execution"""
            self.collector.record_metric(
                "response_time", duration, {"module": module_name}, "seconds"
            )
            self.collector.record_metric(
                "request_count", 1, {"module": module_name}, "count"
            )

            if not success:
                self.collector.record_metric(
                    "error_count",
                    1,
                    {"module": module_name, "error": error or "unknown"},
                    "count",
                )

            # Calculate error rate
            total_requests = len(self.collector.metrics["request_count"])
            total_errors = len(self.collector.metrics["error_count"])
            error_rate = total_errors / total_requests if total_requests > 0 else 0
            self.collector.record_metric("error_rate", error_rate, unit="percentage")

        def get_dashboard_data(self) -> dict[str, Any]:
            """Get comprehensive dashboard data"""
            self.collector.update_system_health()

            return {
                "system_health": {
                    "status": self.collector.system_health.status,
                    "score": self.collector.system_health.score,
                    "issues": self.collector.system_health.issues,
                    "recommendations": self.collector.system_health.recommendations,
                },
                "metrics": {
                    "response_time": self.collector.get_metric_stats(
                        "response_time", 60
                    ),
                    "error_rate": self.collector.get_metric_stats("error_rate", 60),
                    "request_count": self.collector.get_metric_stats(
                        "request_count", 60
                    ),
                },
                "alerts": {
                    "active": [
                        a for a in self.collector.alerts.values() if a.triggered
                    ],
                    "total": len(self.collector.alerts),
                },
                "uptime": time.time() - self.start_time,
            }

    # Initialize global monitor
    monitor = DSPyMonitor()

    return Alert, monitor


@app.cell
def _(mo, output):
    # Monitoring Configuration
    cell3_desc = mo.md("## ⚙️ Step 1: Configure Monitoring")

    enable_monitoring = mo.ui.checkbox(value=True, label="Enable real-time monitoring")

    alert_threshold = mo.ui.slider(
        start=1.0,
        stop=10.0,
        step=0.5,
        value=5.0,
        label="Response time alert threshold (seconds)",
    )

    error_threshold = mo.ui.slider(
        start=0.01,
        stop=0.5,
        step=0.01,
        value=0.1,
        label="Error rate alert threshold",
    )

    start_monitoring_button = mo.ui.run_button(label="Start Monitoring", kind="success")

    cell3_content = mo.vstack(
        [
            alert_threshold,
            enable_monitoring,
            error_threshold,
            start_monitoring_button,
        ]
    )

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (
        alert_threshold,
        enable_monitoring,
        error_threshold,
        start_monitoring_button,
    )


@app.cell
def _(
    Alert,
    alert_threshold,
    enable_monitoring,
    error_threshold,
    monitor,
    start_monitoring_button,
):
    # Initialize Monitoring System
    monitoring_status = "stopped"

    if start_monitoring_button.value and enable_monitoring.value:
        # Start monitoring
        monitor.start_monitoring()
        monitoring_status = "running"

        # Add alerts
        response_time_alert = Alert(
            alert_id="high_response_time",
            metric_name="response_time",
            threshold=alert_threshold.value,
            condition="greater_than",
            severity="high",
            message=f"Response time exceeded {alert_threshold.value}s",
        )
        monitor.collector.add_alert(response_time_alert)

        error_rate_alert = Alert(
            alert_id="high_error_rate",
            metric_name="error_rate",
            threshold=error_threshold.value,
            condition="greater_than",
            severity="critical",
            message=f"Error rate exceeded {error_threshold.value * 100}%",
        )
        monitor.collector.add_alert(error_rate_alert)

    return (monitoring_status,)


@app.cell
def _(cleandoc, mo, monitoring_status, output):
    # Monitoring Status Display
    if monitoring_status == "running":
        cell5_out = mo.md(
            cleandoc(
                """
                ✅ **Monitoring System Active**
                - Real-time metrics collection enabled
                - Alert system configured
                - Dashboard ready for data visualization
                """
            )
        )
    else:
        cell5_out = mo.md("⚠️ Configure and start monitoring above to continue")

    output.replace(cell5_out)
    return


@app.cell
def _(get_config, mo, output):
    # DSPy Module Setup for Monitoring
    cell6_desc = mo.md("## 🧠 Step 2: Configure DSPy Module")

    # Get available LLM providers
    try:
        config = get_config()
        cell6_available_providers = config.get_available_llm_providers()
    except Exception as _:
        cell6_available_providers = ["openai", "anthropic", "local"]

    provider_selector = mo.ui.dropdown(
        options=cell6_available_providers,
        value=cell6_available_providers[0] if cell6_available_providers else "openai",
        label="Select LLM Provider",
    )

    module_type = mo.ui.dropdown(
        options=["Predict", "ChainOfThought", "ReAct"],
        value="Predict",
        label="Select DSPy Module Type",
    )

    cell6_content = mo.vstack([provider_selector, module_type])

    cell6_out = mo.vstack([cell6_desc, cell6_content])
    output.replace(cell6_out)
    return module_type, provider_selector


@app.cell
def _(module_type, provider_selector, time):
    # Initialize Mock DSPy Module for Monitoring (to avoid model configuration issues)
    monitor_module = None
    monitor_signature = None

    if provider_selector.value and module_type.value:
        # Create a mock module that simulates DSPy behavior for demonstration
        class MockMonitoringModule:
            def __init__(self, module_type, provider):
                self.module_type = module_type
                self.provider = provider

            def __call__(self, **kwargs):
                # Simulate processing time based on module type and provider
                if self.module_type == "Predict":
                    base_time = 0.3
                elif self.module_type == "ChainOfThought":
                    base_time = 0.8
                else:  # ReAct
                    base_time = 1.2

                # Add provider-based variation
                if self.provider == "anthropic":
                    base_time *= 0.9  # Slightly faster
                elif self.provider == "local":
                    base_time *= 1.5  # Slower for local

                time.sleep(base_time)

                query = kwargs.get("query", "No query provided")

                # Mock response based on module type
                if self.module_type == "Predict":
                    response = f"Direct response to: {query[:50]}..."
                elif self.module_type == "ChainOfThought":
                    response = f"Let me think about '{query[:30]}...' step by step. First, I need to understand the context..."
                else:  # ReAct
                    response = f"I need to reason about '{query[:30]}...' and take appropriate action based on my analysis."

                class MockResult:
                    def __init__(self, response):
                        self.response = response

                return MockResult(response)

        monitor_module = MockMonitoringModule(
            module_type.value, provider_selector.value
        )
        monitor_signature = "MockMonitoringSignature"

    return (monitor_module,)


@app.cell
def _(mo, monitor_module, monitoring_status, output):
    # Test Execution Section
    if not monitor_module or monitoring_status != "running":
        cell8_desc = mo.md("*Configure monitoring and DSPy module first*")
        cell8_content = mo.md("")
        test_query = None
        execute_test = None
    else:
        cell8_desc = mo.md("## 🧪 Step 3: Test Execution with Monitoring")

        test_query = mo.ui.text_area(
            placeholder="Enter test query...",
            label="Test Query",
            rows=2,
            value="What is artificial intelligence?",
        )

        execute_test = mo.ui.run_button(label="Execute Test", kind="success")

        cell8_content = mo.vstack([test_query, execute_test])

    cell8_out = mo.vstack([cell8_desc, cell8_content])
    output.replace(cell8_out)
    return execute_test, test_query


@app.cell
def _(execute_test, module_type, monitor, monitor_module, test_query, time):
    # Execute Test with Monitoring
    test_results = []

    if (
        execute_test is not None
        and execute_test.value
        and monitor_module
        and test_query
        and test_query.value
    ):
        # Execute multiple test runs to generate metrics
        for i in range(3):  # Run 3 tests to generate some data
            start_time = time.time()
            success = True
            error = None

            try:
                _result = monitor_module(query=test_query.value)
                duration = time.time() - start_time
                test_results.append(
                    {
                        "run": i + 1,
                        "duration": duration,
                        "success": True,
                        "response": (
                            _result.response[:100] + "..."
                            if len(_result.response) > 100
                            else _result.response
                        ),
                    }
                )
            except Exception as e:
                duration = time.time() - start_time
                success = False
                error = str(e)
                test_results.append(
                    {
                        "run": i + 1,
                        "duration": duration,
                        "success": False,
                        "error": error,
                    }
                )

            # Record metrics
            monitor.record_execution(
                str(module_type.value),
                duration,
                success,
                error,
            )

    return (test_results,)


@app.cell
def _(cleandoc, mo, output, test_results):
    # Display Test Results
    if not test_results:
        cell10_out = mo.md("*Execute tests above to see results*")
    else:
        cell10_desc = mo.md("## 📋 Test Execution Results")

        _results_content = []
        for result in test_results:
            if result["success"]:
                _result_info = cleandoc(
                    f"""
                    **Run {result['run']}:** ✅ Success
                    - Duration: {result['duration']:.3f}s
                    - Response: {result['response']}
                    """
                )
            else:
                _result_info = cleandoc(
                    f"""
                    **Run {result['run']}:** ❌ Failed
                    - Duration: {result['duration']:.3f}s
                    - Error: {result['error']}
                    """
                )
            _results_content.append(_result_info)

        cell10_content = mo.md("\n\n".join(_results_content))
        cell10_out = mo.vstack([cell10_desc, cell10_content])

    output.replace(cell10_out)
    return


@app.cell
def _(cleandoc, mo, monitor, output):
    # Real-time Dashboard
    cell11_dashboard_data = monitor.get_dashboard_data()

    if not cell11_dashboard_data["metrics"]["response_time"]:
        cell11_out = mo.md("*No metrics data available yet - execute some tests first*")
    else:
        cell11_desc = mo.md("## 📊 Real-time Dashboard")

        # System Health
        _health = cell11_dashboard_data["system_health"]
        _health_color = {
            "healthy": "#28a745",
            "warning": "#ffc107",
            "critical": "#dc3545",
        }.get(_health["status"], "#6c757d")

        _health_html = cleandoc(
            f"""
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 10px 0;">
                <h3>🏥 System Health</h3>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 20px; height: 20px; border-radius: 50%; background: {_health_color};"></div>
                    <span style="font-size: 18px; font-weight: bold;">{_health["status"].title()}</span>
                    <span style="margin-left: 20px;">Score: {_health["score"]:.0f}/100</span>
                </div>
            </div>
            """
        )

        # Metrics Summary
        _metrics = cell11_dashboard_data["metrics"]
        _response_time = _metrics["response_time"]
        _error_rate = _metrics["error_rate"]

        _metrics_html = cleandoc(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 20px 0;">
                <div style="background: #e7f3ff; padding: 15px; border-radius: 8px; text-align: center;">
                    <h4>⚡ Response Time</h4>
                    <div style="font-size: 24px; font-weight: bold; color: #0066cc;">
                        {_response_time.get('latest', 0):.3f}s
                    </div>
                    <div style="font-size: 12px; color: #666;">
                        Avg: {_response_time.get('mean', 0):.3f}s
                    </div>
                </div>
                <div style="background: #fff3cd; padding: 15px; border-radius: 8px; text-align: center;">
                    <h4>📊 Requests</h4>
                    <div style="font-size: 24px; font-weight: bold; color: #856404;">
                        {_metrics.get('request_count', {}).get('count', 0)}
                    </div>
                    <div style="font-size: 12px; color: #666;">
                        Total processed
                    </div>
                </div>
                <div style="background: #f8d7da; padding: 15px; border-radius: 8px; text-align: center;">
                    <h4>🚨 Error Rate</h4>
                    <div style="font-size: 24px; font-weight: bold; color: #721c24;">
                        {_error_rate.get('latest', 0):.1%}
                    </div>
                    <div style="font-size: 12px; color: #666;">
                        Current rate
                    </div>
                </div>
            </div>
            """
        )

        cell11_content = mo.Html(_health_html + _metrics_html)
        cell11_out = mo.vstack([cell11_desc, cell11_content])

    output.replace(cell11_out)
    return


@app.cell
def _(mo, monitor, output):
    # Alerts and Issues
    cell12_dashboard_data = monitor.get_dashboard_data()

    cell12_desc = mo.md("## 🚨 Alerts & Issues")

    _health = cell12_dashboard_data["system_health"]
    _alerts = cell12_dashboard_data["alerts"]

    if _health["issues"] or _alerts["active"]:
        _issues_content = []

        # Active alerts
        if _alerts["active"]:
            _issues_content.append("### 🔔 Active Alerts")
            for alert in _alerts["active"]:
                _issues_content.append(
                    f"- **{alert.severity.upper()}**: {alert.message}"
                )

        # System issues
        if _health["issues"]:
            _issues_content.append("### ⚠️ System Issues")
            for issue in _health["issues"]:
                _issues_content.append(f"- {issue}")

        # Recommendations
        if _health["recommendations"]:
            _issues_content.append("### 💡 Recommendations")
            for rec in _health["recommendations"]:
                _issues_content.append(f"- {rec}")

        cell12_content = mo.md("\n".join(_issues_content))
    else:
        cell12_content = mo.md(
            "✅ **All systems operating normally** - No active alerts or issues"
        )

    cell12_out = mo.vstack([cell12_desc, cell12_content])
    output.replace(cell12_out)
    return


@app.cell
def _(cleandoc, mo, monitor, output):
    # Performance Analytics
    cell13_dashboard_data = monitor.get_dashboard_data()

    if not cell13_dashboard_data["metrics"]["response_time"]:
        cell13_out = mo.md("*No performance data available yet*")
    else:
        cell13_desc = mo.md("## 📈 Performance Analytics")

        _metrics = cell13_dashboard_data["metrics"]
        _response_time = _metrics["response_time"]
        _uptime = cell13_dashboard_data["uptime"]

        _analytics_content = cleandoc(
            f"""
            ### Response Time Analysis
            - **Current**: {_response_time.get('latest', 0):.3f}s
            - **Average**: {_response_time.get('mean', 0):.3f}s
            - **Min/Max**: {_response_time.get('min', 0):.3f}s / {_response_time.get('max', 0):.3f}s
            - **Std Dev**: {_response_time.get('std_dev', 0):.3f}s

            ### System Uptime
            - **Duration**: {_uptime / 60:.1f} minutes
            - **Status**: {'🟢 Stable' if _uptime > 300 else '🟡 Starting up'}

            ### Request Statistics
            - **Total Requests**: {_metrics.get('request_count', {}).get('count', 0)}
            - **Success Rate**: {(1 - _metrics.get('error_rate', {}).get('latest', 0)) * 100:.1f}%
            """
        )

        cell13_content = mo.md(_analytics_content)
        cell13_out = mo.vstack([cell13_desc, cell13_content])

    output.replace(cell13_out)
    return


@app.cell
def _(mo, output):
    # Export Monitoring Data
    cell14_desc = mo.md("## 💾 Step 4: Export Monitoring Data")

    export_format = mo.ui.dropdown(
        options=["JSON", "CSV"],
        value="JSON",
        label="Export Format",
    )

    export_button = mo.ui.run_button(label="Export Data", kind="info")

    cell14_content = mo.vstack([export_format, export_button])

    cell14_out = mo.vstack([cell14_desc, cell14_content])
    output.replace(cell14_out)
    return export_button, export_format


@app.cell
def _(datetime, export_button, export_format, json, mo, monitor, output):
    # Handle Data Export
    if export_button and export_button.value:
        try:
            _dashboard_data = monitor.get_dashboard_data()
            _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _filename = f"observability_data_{_timestamp}.{export_format.value.lower()}"

            if export_format.value == "JSON":
                with open(_filename, "w", encoding="utf-8") as f:
                    json.dump(_dashboard_data, f, indent=2, default=str)
            else:  # CSV format would require more complex conversion
                # For now, just export as JSON with CSV extension
                with open(_filename, "w", encoding="utf-8") as f:
                    json.dump(_dashboard_data, f, indent=2, default=str)

            cell15_out = mo.md(f"✅ **Monitoring data exported to {_filename}**")
        except Exception as e:
            cell15_out = mo.md(f"❌ **Export failed:** {str(e)}")
    else:
        cell15_out = mo.md("*Click 'Export Data' to save monitoring data*")

    output.replace(cell15_out)
    return


@app.cell
def _(cleandoc, mo, output):
    # Footer with observability best practices
    cell16_out = mo.md(
        cleandoc(
            """
            ## 🎯 Observability Best Practices

            ### ✅ What You've Learned

            - **Real-time Monitoring**: Track system performance and health in real-time
            - **Alert Management**: Set up proactive alerts for critical system events
            - **Performance Analytics**: Analyze trends and identify optimization opportunities
            - **System Health Scoring**: Get an overall health score for your DSPy systems
            - **Data Export**: Export monitoring data for further analysis and reporting

            ### 💡 Pro Tips

            1. **Set Appropriate Thresholds**: Configure alert thresholds based on your SLA requirements
            2. **Monitor Key Metrics**: Focus on response time, error rate, and throughput
            3. **Regular Health Checks**: Monitor system health score and address issues promptly
            4. **Historical Analysis**: Export and analyze historical data to identify patterns
            5. **Proactive Monitoring**: Use alerts to catch issues before they impact users

            ### 🚀 Next Steps

            - **Performance Monitor**: Set up continuous performance monitoring
            - **Custom Metrics**: Add domain-specific metrics for your use case
            - **Integration**: Integrate with external monitoring systems (Prometheus, Grafana)
            - **Automated Responses**: Set up automated responses to common issues

            **Ready for production monitoring?** Set up continuous monitoring for your DSPy applications! 🚀
            """
        )
    )

    output.replace(cell16_out)
    return


if __name__ == "__main__":
    app.run()
