#!/usr/bin/env python3
"""
Solution 03: DSPy Observability Metrics Implementation

This solution demonstrates how to implement comprehensive observability metrics
for DSPy systems, including real-time monitoring, alerting, and health assessment.

Learning Objectives:
- Implement comprehensive metrics collection for DSPy systems
- Create real-time monitoring and alerting systems
- Build system health assessment and scoring
- Develop observability dashboards and reporting

Author: DSPy Learning Framework
"""

import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import dspy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Represents a system metric"""

    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
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
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and manages system metrics"""

    def __init__(self, max_history: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.alerts: Dict[str, Alert] = {}
        self.system_health = SystemHealth(status="unknown", score=0)

    def record_metric(
        self, name: str, value: float, tags: Optional[Dict] = None, unit: str = ""
    ):
        """Record a new metric value"""
        metric = Metric(
            name=name, value=value, timestamp=time.time(), tags=tags or {}, unit=unit
        )
        self.metrics[name].append(metric)
        self._check_alerts(metric)

    def get_metric_history(self, name: str, duration_minutes: int = 60) -> List[Metric]:
        """Get metric history for the specified duration"""
        if name not in self.metrics:
            return []

        cutoff_time = time.time() - (duration_minutes * 60)
        return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]

    def get_metric_stats(
        self, name: str, duration_minutes: int = 60
    ) -> Dict[str, float]:
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
            a for a in self.alerts.values() if a.triggered and a.severity == "critical"
        ]
        high_alerts = [
            a for a in self.alerts.values() if a.triggered and a.severity == "high"
        ]
        medium_alerts = [
            a for a in self.alerts.values() if a.triggered and a.severity == "medium"
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


class DSPyObservabilitySystem:
    """Comprehensive observability system for DSPy applications"""

    def __init__(self):
        self.collector = MetricsCollector()
        self.monitoring_active = False
        self.start_time = time.time()
        self.request_counter = 0
        self.error_counter = 0

    def start_monitoring(self):
        """Start monitoring system"""
        self.monitoring_active = True
        self.start_time = time.time()
        logger.info("DSPy observability monitoring started")

    def stop_monitoring(self):
        """Stop monitoring system"""
        self.monitoring_active = False
        logger.info("DSPy observability monitoring stopped")

    def record_request(
        self,
        module_name: str,
        duration: float,
        success: bool,
        input_size: int = 0,
        output_size: int = 0,
        error: Optional[str] = None,
    ):
        """Record a DSPy module request"""
        if not self.monitoring_active:
            return

        self.request_counter += 1

        # Record basic metrics
        self.collector.record_metric(
            "response_time", duration, {"module": module_name}, "seconds"
        )
        self.collector.record_metric(
            "request_count", 1, {"module": module_name}, "count"
        )
        self.collector.record_metric(
            "input_size", input_size, {"module": module_name}, "bytes"
        )
        self.collector.record_metric(
            "output_size", output_size, {"module": module_name}, "bytes"
        )

        if not success:
            self.error_counter += 1
            self.collector.record_metric(
                "error_count",
                1,
                {"module": module_name, "error": error or "unknown"},
                "count",
            )

        # Calculate and record derived metrics
        error_rate = (
            self.error_counter / self.request_counter if self.request_counter > 0 else 0
        )
        self.collector.record_metric("error_rate", error_rate, unit="percentage")

        # Calculate throughput (requests per second over last minute)
        recent_requests = self.collector.get_metric_history("request_count", 1)
        throughput = len(recent_requests) / 60.0  # requests per second
        self.collector.record_metric(
            "throughput", throughput, unit="requests_per_second"
        )

    def record_custom_metric(
        self, name: str, value: float, tags: Optional[Dict] = None, unit: str = ""
    ):
        """Record a custom application metric"""
        if self.monitoring_active:
            self.collector.record_metric(name, value, tags, unit)

    def setup_default_alerts(self):
        """Set up default alerts for common issues"""
        alerts = [
            Alert(
                alert_id="high_response_time",
                metric_name="response_time",
                threshold=5.0,
                condition="greater_than",
                severity="high",
                message="Response time exceeded 5 seconds",
            ),
            Alert(
                alert_id="high_error_rate",
                metric_name="error_rate",
                threshold=0.1,
                condition="greater_than",
                severity="critical",
                message="Error rate exceeded 10%",
            ),
            Alert(
                alert_id="low_throughput",
                metric_name="throughput",
                threshold=0.1,
                condition="less_than",
                severity="medium",
                message="Throughput dropped below 0.1 requests/second",
            ),
        ]

        for alert in alerts:
            self.collector.add_alert(alert)

    def get_dashboard_data(self) -> Dict[str, Any]:
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
                "response_time": self.collector.get_metric_stats("response_time", 60),
                "error_rate": self.collector.get_metric_stats("error_rate", 60),
                "throughput": self.collector.get_metric_stats("throughput", 60),
                "request_count": self.collector.get_metric_stats("request_count", 60),
            },
            "alerts": {
                "active": [a for a in self.collector.alerts.values() if a.triggered],
                "total": len(self.collector.alerts),
            },
            "uptime": time.time() - self.start_time,
            "total_requests": self.request_counter,
            "total_errors": self.error_counter,
        }

    def generate_health_report(self) -> str:
        """Generate a comprehensive health report"""
        dashboard_data = self.get_dashboard_data()

        report = []
        report.append("=== DSPy System Health Report ===")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # System Health
        health = dashboard_data["system_health"]
        report.append(
            f"Overall Health: {health['status'].upper()} (Score: {health['score']:.0f}/100)"
        )

        if health["issues"]:
            report.append("\nIssues:")
            for issue in health["issues"]:
                report.append(f"  • {issue}")

        if health["recommendations"]:
            report.append("\nRecommendations:")
            for rec in health["recommendations"]:
                report.append(f"  • {rec}")

        # Metrics Summary
        report.append("\n=== Metrics Summary ===")
        metrics = dashboard_data["metrics"]

        if metrics["response_time"]:
            rt = metrics["response_time"]
            report.append(
                f"Response Time: {rt['latest']:.3f}s (avg: {rt['mean']:.3f}s, max: {rt['max']:.3f}s)"
            )

        if metrics["error_rate"]:
            er = metrics["error_rate"]
            report.append(f"Error Rate: {er['latest']:.1%}")

        if metrics["throughput"]:
            tp = metrics["throughput"]
            report.append(f"Throughput: {tp['latest']:.2f} req/s")

        # System Stats
        report.append(f"\nUptime: {dashboard_data['uptime'] / 60:.1f} minutes")
        report.append(f"Total Requests: {dashboard_data['total_requests']}")
        report.append(f"Total Errors: {dashboard_data['total_errors']}")

        # Active Alerts
        active_alerts = dashboard_data["alerts"]["active"]
        if active_alerts:
            report.append("\n=== Active Alerts ===")
            for alert in active_alerts:
                report.append(f"  🚨 {alert.severity.upper()}: {alert.message}")
        else:
            report.append("\n✅ No active alerts")

        return "\n".join(report)


def create_sample_dspy_modules():
    """Create sample DSPy modules for observability demonstration"""

    class FastModule:
        """Simulates a fast DSPy module"""

        def __call__(self, **kwargs):
            time.sleep(0.1)  # Fast execution
            text = kwargs.get("text", "")
            return {"result": f"Fast processing: {text[:50]}"}

    class SlowModule:
        """Simulates a slow DSPy module"""

        def __call__(self, **kwargs):
            time.sleep(2.0)  # Slow execution
            text = kwargs.get("text", "")
            return {"result": f"Slow processing: {text[:50]}"}

    class UnreliableModule:
        """Simulates an unreliable DSPy module"""

        def __init__(self):
            self.call_count = 0

        def __call__(self, **kwargs):
            self.call_count += 1
            time.sleep(0.5)

            # Fail every 3rd call
            if self.call_count % 3 == 0:
                raise Exception("Simulated module failure")

            text = kwargs.get("text", "")
            return {"result": f"Unreliable processing: {text[:50]}"}

    return {
        "fast": FastModule(),
        "slow": SlowModule(),
        "unreliable": UnreliableModule(),
    }


def demonstrate_metrics_collection():
    """Demonstrate comprehensive metrics collection"""
    print("=== Metrics Collection Demonstration ===\n")

    obs_system = DSPyObservabilitySystem()
    obs_system.start_monitoring()
    obs_system.setup_default_alerts()

    modules = create_sample_dspy_modules()

    print("1. Recording various module executions:")
    print("-" * 40)

    # Test scenarios
    test_cases = [
        ("fast", "This is a test for the fast module", True),
        ("fast", "Another test for fast processing", True),
        ("slow", "This will take longer to process", True),
        ("unreliable", "Testing unreliable module - attempt 1", True),
        ("unreliable", "Testing unreliable module - attempt 2", True),
        (
            "unreliable",
            "Testing unreliable module - attempt 3",
            False,
        ),  # This will fail
    ]

    for module_name, text, should_succeed in test_cases:
        module = modules[module_name]
        start_time = time.time()

        try:
            result = module(text=text)
            duration = time.time() - start_time
            success = True
            error = None
            print(f"✅ {module_name}: {duration:.3f}s - {result['result'][:30]}...")
        except Exception as e:
            duration = time.time() - start_time
            success = False
            error = str(e)
            print(f"❌ {module_name}: {duration:.3f}s - Error: {error}")

        # Record metrics
        obs_system.record_request(
            module_name=module_name,
            duration=duration,
            success=success,
            input_size=len(text),
            output_size=len(str(result)) if success else 0,
            error=error,
        )

        time.sleep(0.1)  # Small delay between requests

    return obs_system


def demonstrate_real_time_monitoring(obs_system):
    """Demonstrate real-time monitoring and alerting"""
    print("\n2. Real-time monitoring and alerting:")
    print("-" * 40)

    # Get current dashboard data
    dashboard_data = obs_system.get_dashboard_data()

    print("System Health:")
    health = dashboard_data["system_health"]
    print(f"  Status: {health['status'].upper()}")
    print(f"  Score: {health['score']:.0f}/100")

    if health["issues"]:
        print("  Issues:")
        for issue in health["issues"]:
            print(f"    • {issue}")

    if health["recommendations"]:
        print("  Recommendations:")
        for rec in health["recommendations"]:
            print(f"    • {rec}")

    print("\nKey Metrics:")
    metrics = dashboard_data["metrics"]

    if metrics["response_time"]:
        rt = metrics["response_time"]
        print(f"  Response Time: {rt['latest']:.3f}s (avg: {rt['mean']:.3f}s)")

    if metrics["error_rate"]:
        er = metrics["error_rate"]
        print(f"  Error Rate: {er['latest']:.1%}")

    if metrics["throughput"]:
        tp = metrics["throughput"]
        print(f"  Throughput: {tp['latest']:.2f} req/s")

    # Check for active alerts
    active_alerts = dashboard_data["alerts"]["active"]
    if active_alerts:
        print("\nActive Alerts:")
        for alert in active_alerts:
            print(f"  🚨 {alert.severity.upper()}: {alert.message}")
    else:
        print("\n✅ No active alerts")


def demonstrate_custom_metrics(obs_system):
    """Demonstrate custom metrics and advanced monitoring"""
    print("\n3. Custom metrics and advanced monitoring:")
    print("-" * 40)

    # Record custom application metrics
    custom_metrics = [
        ("model_accuracy", 0.95, {"model": "gpt-3.5-turbo"}, "percentage"),
        ("cache_hit_rate", 0.75, {"cache_type": "redis"}, "percentage"),
        ("queue_depth", 5, {"queue": "processing"}, "count"),
        ("memory_usage", 1024, {"component": "model"}, "MB"),
        ("api_quota_remaining", 8500, {"provider": "openai"}, "requests"),
    ]

    print("Recording custom metrics:")
    for name, value, tags, unit in custom_metrics:
        obs_system.record_custom_metric(name, value, tags, unit)
        print(f"  • {name}: {value} {unit}")

    # Demonstrate metric history analysis
    print("\nMetric history analysis:")
    response_time_history = obs_system.collector.get_metric_history("response_time", 5)
    if response_time_history:
        durations = [m.value for m in response_time_history]
        print(f"  Response time trend: {durations}")

        # Identify performance patterns
        if len(durations) >= 3:
            recent_avg = statistics.mean(durations[-3:])
            overall_avg = statistics.mean(durations)

            if recent_avg > overall_avg * 1.2:
                print("  📈 Performance degradation detected")
            elif recent_avg < overall_avg * 0.8:
                print("  📉 Performance improvement detected")
            else:
                print("  ➡️ Performance stable")


def demonstrate_health_assessment(obs_system):
    """Demonstrate comprehensive health assessment"""
    print("\n4. Comprehensive health assessment:")
    print("-" * 40)

    # Generate and display health report
    health_report = obs_system.generate_health_report()
    print(health_report)

    # Demonstrate health scoring logic
    print("\n" + "=" * 50)
    print("Health Scoring Breakdown:")
    print("=" * 50)

    dashboard_data = obs_system.get_dashboard_data()
    health = dashboard_data["system_health"]

    base_score = 100
    print(f"Base score: {base_score}")

    # Show deductions
    active_alerts = dashboard_data["alerts"]["active"]
    for alert in active_alerts:
        deduction = {"critical": 30, "high": 20, "medium": 10, "low": 5}.get(
            alert.severity, 0
        )
        base_score -= deduction
        print(f"  -{deduction} points: {alert.severity} alert - {alert.message}")

    # Performance-based deductions
    metrics = dashboard_data["metrics"]
    if metrics["response_time"] and metrics["response_time"]["mean"] > 5.0:
        print(
            f"  -15 points: High average response time ({metrics['response_time']['mean']:.3f}s)"
        )
        base_score -= 15

    if metrics["error_rate"] and metrics["error_rate"]["latest"] > 0.1:
        print(f"  -20 points: High error rate ({metrics['error_rate']['latest']:.1%})")
        base_score -= 20

    final_score = max(0, base_score)
    print(f"\nFinal health score: {final_score}/100")


def demonstrate_alerting_system(obs_system):
    """Demonstrate advanced alerting system"""
    print("\n5. Advanced alerting system:")
    print("-" * 40)

    # Add custom alerts
    custom_alerts = [
        Alert(
            alert_id="model_accuracy_low",
            metric_name="model_accuracy",
            threshold=0.9,
            condition="less_than",
            severity="high",
            message="Model accuracy dropped below 90%",
        ),
        Alert(
            alert_id="queue_depth_high",
            metric_name="queue_depth",
            threshold=10,
            condition="greater_than",
            severity="medium",
            message="Processing queue depth exceeded 10 items",
        ),
        Alert(
            alert_id="memory_usage_critical",
            metric_name="memory_usage",
            threshold=2048,
            condition="greater_than",
            severity="critical",
            message="Memory usage exceeded 2GB",
        ),
    ]

    for alert in custom_alerts:
        obs_system.collector.add_alert(alert)

    print("Added custom alerts:")
    for alert in custom_alerts:
        print(f"  • {alert.alert_id}: {alert.message}")

    # Simulate conditions that trigger alerts
    print("\nSimulating alert conditions:")

    # Trigger model accuracy alert
    obs_system.record_custom_metric(
        "model_accuracy", 0.85, {"model": "test"}, "percentage"
    )
    print("  • Recorded low model accuracy (0.85)")

    # Trigger queue depth alert
    obs_system.record_custom_metric("queue_depth", 15, {"queue": "processing"}, "count")
    print("  • Recorded high queue depth (15)")

    # Check for triggered alerts
    time.sleep(0.1)  # Allow alerts to process

    dashboard_data = obs_system.get_dashboard_data()
    active_alerts = dashboard_data["alerts"]["active"]

    if active_alerts:
        print("\nTriggered alerts:")
        for alert in active_alerts:
            print(f"  🚨 {alert.severity.upper()}: {alert.message}")
    else:
        print("\nNo alerts triggered")


def demonstrate_data_export(obs_system):
    """Demonstrate metrics data export and reporting"""
    print("\n6. Data export and reporting:")
    print("-" * 40)

    # Export dashboard data
    dashboard_data = obs_system.get_dashboard_data()

    # Save to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"observability_report_{timestamp}.json"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        print(f"✅ Dashboard data exported to {filename}")
    except Exception as e:
        print(f"❌ Export failed: {e}")

    # Generate summary statistics
    print("\nSummary Statistics:")
    print(f"  • Monitoring duration: {dashboard_data['uptime'] / 60:.1f} minutes")
    print(f"  • Total requests processed: {dashboard_data['total_requests']}")
    print(f"  • Total errors encountered: {dashboard_data['total_errors']}")
    print(
        f"  • Overall success rate: {((dashboard_data['total_requests'] - dashboard_data['total_errors']) / dashboard_data['total_requests'] * 100):.1f}%"
    )

    # Metric coverage
    available_metrics = list(obs_system.collector.metrics.keys())
    print(f"  • Metrics collected: {len(available_metrics)}")
    print(f"  • Alert rules configured: {len(obs_system.collector.alerts)}")


if __name__ == "__main__":
    """
    Exercise Solution: DSPy Observability Metrics

    This script demonstrates:
    1. Comprehensive metrics collection for DSPy systems
    2. Real-time monitoring and alerting
    3. Custom metrics and advanced monitoring
    4. Health assessment and scoring
    5. Advanced alerting system
    6. Data export and reporting
    """

    try:
        # Run the complete observability demonstration
        obs_system = demonstrate_metrics_collection()
        demonstrate_real_time_monitoring(obs_system)
        demonstrate_custom_metrics(obs_system)
        demonstrate_health_assessment(obs_system)
        demonstrate_alerting_system(obs_system)
        demonstrate_data_export(obs_system)

        print("\n✅ Observability metrics exercise completed successfully!")
        print("\nKey takeaways:")
        print("- Comprehensive metrics provide deep system insights")
        print("- Real-time monitoring enables proactive issue detection")
        print("- Custom metrics support domain-specific monitoring")
        print("- Health scoring provides overall system assessment")
        print("- Alerting systems enable automated incident response")
        print("- Data export supports analysis and reporting")

    except Exception as e:
        print(f"\n❌ Exercise failed: {e}")
        logger.exception("Exercise execution failed")
