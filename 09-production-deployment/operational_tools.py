#!/usr/bin/env python3
"""
Operational Tools and Utilities for DSPy Applications

This module provides additional operational tools including log management,
performance analysis, security monitoring, and operational dashboards.

Author: DSPy Learning Framework
"""

import asyncio
import gzip
import hashlib
import json
import logging
import re
import statistics
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for analysis"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SecurityEventType(Enum):
    """Types of security events"""

    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_FAILURE = "authz_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MALFORMED_REQUEST = "malformed_request"
    POTENTIAL_ATTACK = "potential_attack"


@dataclass
class LogEntry:
    """Structured log entry"""

    timestamp: datetime
    level: LogLevel
    message: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class SecurityEvent:
    """Security event record"""

    event_id: str
    event_type: SecurityEventType
    timestamp: datetime
    source_ip: str
    user_agent: str
    description: str
    severity: int  # 1-10, 10 being most severe
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class PerformanceMetric:
    """Performance metric data point"""

    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


class LogAnalyzer:
    """Analyze application logs for patterns and issues"""

    def __init__(self, log_directory: str = "logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        self.log_patterns: Dict[str, re.Pattern] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.log_cache: deque = deque(maxlen=10000)
        self._setup_default_patterns()

    def _setup_default_patterns(self):
        """Set up default log patterns"""
        self.log_patterns.update(
            {
                "error": re.compile(
                    r"ERROR|CRITICAL|Exception|Traceback", re.IGNORECASE
                ),
                "warning": re.compile(r"WARNING|WARN", re.IGNORECASE),
                "auth_failure": re.compile(
                    r"authentication.*failed|login.*failed|unauthorized", re.IGNORECASE
                ),
                "rate_limit": re.compile(
                    r"rate.?limit|too.?many.?requests", re.IGNORECASE
                ),
                "timeout": re.compile(r"timeout|timed.?out", re.IGNORECASE),
                "memory_error": re.compile(
                    r"out.?of.?memory|memory.?error|oom", re.IGNORECASE
                ),
                "database_error": re.compile(
                    r"database.*error|connection.*failed|sql.*error", re.IGNORECASE
                ),
            }
        )

    def add_alert_rule(
        self, name: str, pattern: str, threshold: int, time_window_minutes: int
    ):
        """Add a log alert rule"""
        self.alert_rules.append(
            {
                "name": name,
                "pattern": re.compile(pattern, re.IGNORECASE),
                "threshold": threshold,
                "time_window_minutes": time_window_minutes,
                "last_triggered": None,
            }
        )
        logger.info(f"Added log alert rule: {name}")

    async def analyze_log_file(self, log_file: Path) -> Dict[str, Any]:
        """Analyze a single log file"""
        if not log_file.exists():
            return {"error": f"Log file not found: {log_file}"}

        analysis = {
            "file": str(log_file),
            "total_lines": 0,
            "log_levels": defaultdict(int),
            "patterns_found": defaultdict(int),
            "time_range": {"start": None, "end": None},
            "issues": [],
            "recommendations": [],
        }

        try:
            # Handle compressed logs
            if log_file.suffix == ".gz":
                file_opener = gzip.open
            else:
                file_opener = open

            with file_opener(log_file, "rt", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    analysis["total_lines"] += 1

                    # Parse log entry
                    log_entry = self._parse_log_line(line)
                    if log_entry:
                        # Update time range
                        if not analysis["time_range"]["start"]:
                            analysis["time_range"][
                                "start"
                            ] = log_entry.timestamp.isoformat()
                        analysis["time_range"]["end"] = log_entry.timestamp.isoformat()

                        # Count log levels
                        analysis["log_levels"][log_entry.level.value] += 1

                        # Check patterns
                        for pattern_name, pattern in self.log_patterns.items():
                            if pattern.search(line):
                                analysis["patterns_found"][pattern_name] += 1

                        # Cache recent entries
                        self.log_cache.append(log_entry)

            # Generate insights
            analysis["issues"] = self._identify_issues(analysis)
            analysis["recommendations"] = self._generate_recommendations(analysis)

        except Exception as e:
            analysis["error"] = str(e)
            logger.error(f"Failed to analyze log file {log_file}: {e}")

        return analysis

    def _parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a log line into structured format"""
        # Simple log parsing - can be enhanced for specific formats
        line = line.strip()
        if not line:
            return None

        # Try to extract timestamp (ISO format)
        timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})", line)
        if timestamp_match:
            try:
                timestamp_str = timestamp_match.group(1).replace(" ", "T")
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.utcnow()
        else:
            timestamp = datetime.utcnow()

        # Extract log level
        level = LogLevel.INFO  # default
        for log_level in LogLevel:
            if log_level.value in line.upper():
                level = log_level
                break

        # Extract source/logger name
        source_match = re.search(r"\[([^\]]+)\]", line)
        source = source_match.group(1) if source_match else "unknown"

        return LogEntry(timestamp=timestamp, level=level, message=line, source=source)

    def _identify_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify issues from log analysis"""
        issues = []

        # High error rate
        error_count = analysis["log_levels"].get("ERROR", 0) + analysis[
            "log_levels"
        ].get("CRITICAL", 0)
        total_lines = analysis["total_lines"]
        if (
            total_lines > 0 and (error_count / total_lines) > 0.05
        ):  # More than 5% errors
            issues.append(
                f"High error rate: {error_count}/{total_lines} ({error_count/total_lines*100:.1f}%)"
            )

        # Specific pattern issues
        if analysis["patterns_found"]["auth_failure"] > 10:
            issues.append(
                f"Multiple authentication failures: {analysis['patterns_found']['auth_failure']}"
            )

        if analysis["patterns_found"]["timeout"] > 5:
            issues.append(
                f"Multiple timeout errors: {analysis['patterns_found']['timeout']}"
            )

        if analysis["patterns_found"]["memory_error"] > 0:
            issues.append(
                f"Memory errors detected: {analysis['patterns_found']['memory_error']}"
            )

        return issues

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        # Based on issues found
        if analysis["patterns_found"]["auth_failure"] > 10:
            recommendations.append(
                "Review authentication system and implement rate limiting"
            )

        if analysis["patterns_found"]["timeout"] > 5:
            recommendations.append(
                "Investigate timeout issues and optimize response times"
            )

        if analysis["patterns_found"]["memory_error"] > 0:
            recommendations.append(
                "Investigate memory usage and consider scaling up resources"
            )

        if analysis["patterns_found"]["database_error"] > 0:
            recommendations.append("Check database connectivity and performance")

        # General recommendations
        error_rate = (
            analysis["log_levels"].get("ERROR", 0)
            + analysis["log_levels"].get("CRITICAL", 0)
        ) / max(analysis["total_lines"], 1)
        if error_rate > 0.01:
            recommendations.append("Implement better error handling and monitoring")

        return recommendations

    async def analyze_logs_directory(self) -> Dict[str, Any]:
        """Analyze all log files in the directory"""
        log_files = list(self.log_directory.glob("*.log")) + list(
            self.log_directory.glob("*.log.gz")
        )

        if not log_files:
            return {"error": "No log files found"}

        overall_analysis = {
            "files_analyzed": len(log_files),
            "total_lines": 0,
            "overall_log_levels": defaultdict(int),
            "overall_patterns": defaultdict(int),
            "file_analyses": {},
            "summary_issues": [],
            "summary_recommendations": [],
        }

        for log_file in log_files:
            file_analysis = await self.analyze_log_file(log_file)
            overall_analysis["file_analyses"][str(log_file)] = file_analysis

            if "error" not in file_analysis:
                overall_analysis["total_lines"] += file_analysis["total_lines"]

                for level, count in file_analysis["log_levels"].items():
                    overall_analysis["overall_log_levels"][level] += count

                for pattern, count in file_analysis["patterns_found"].items():
                    overall_analysis["overall_patterns"][pattern] += count

                overall_analysis["summary_issues"].extend(file_analysis["issues"])
                overall_analysis["summary_recommendations"].extend(
                    file_analysis["recommendations"]
                )

        # Deduplicate recommendations
        overall_analysis["summary_recommendations"] = list(
            set(overall_analysis["summary_recommendations"])
        )

        return overall_analysis

    async def check_alert_rules(self) -> List[Dict[str, Any]]:
        """Check log alert rules against recent logs"""
        triggered_alerts = []
        current_time = datetime.utcnow()

        for rule in self.alert_rules:
            time_window = timedelta(minutes=rule["time_window_minutes"])
            cutoff_time = current_time - time_window

            # Count pattern matches in time window
            matches = 0
            for log_entry in self.log_cache:
                if log_entry.timestamp >= cutoff_time and rule["pattern"].search(
                    log_entry.message
                ):
                    matches += 1

            # Check if threshold exceeded
            if matches >= rule["threshold"]:
                # Check cooldown period (don't trigger same alert too frequently)
                if not rule["last_triggered"] or current_time - rule[
                    "last_triggered"
                ] > timedelta(minutes=30):

                    triggered_alerts.append(
                        {
                            "rule_name": rule["name"],
                            "matches": matches,
                            "threshold": rule["threshold"],
                            "time_window_minutes": rule["time_window_minutes"],
                            "triggered_at": current_time.isoformat(),
                        }
                    )

                    rule["last_triggered"] = current_time

        return triggered_alerts


class SecurityMonitor:
    """Monitor security events and threats"""

    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.ip_tracking: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "request_count": 0,
                "failed_attempts": 0,
                "last_seen": None,
                "blocked": False,
            }
        )
        self.threat_patterns: List[Dict[str, Any]] = []
        self._setup_threat_patterns()

    def _setup_threat_patterns(self):
        """Set up threat detection patterns"""
        self.threat_patterns = [
            {
                "name": "SQL Injection",
                "pattern": re.compile(
                    r"(union|select|insert|update|delete|drop|exec|script)",
                    re.IGNORECASE,
                ),
                "severity": 8,
            },
            {
                "name": "XSS Attempt",
                "pattern": re.compile(
                    r"(<script|javascript:|onload=|onerror=)", re.IGNORECASE
                ),
                "severity": 7,
            },
            {
                "name": "Path Traversal",
                "pattern": re.compile(r"(\.\./|\.\.\\|%2e%2e)", re.IGNORECASE),
                "severity": 6,
            },
            {
                "name": "Command Injection",
                "pattern": re.compile(r"(;|\||&|`|\$\()", re.IGNORECASE),
                "severity": 8,
            },
        ]

    def analyze_request(self, request_data: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Analyze a request for security threats"""
        source_ip = request_data.get("source_ip", "unknown")
        user_agent = request_data.get("user_agent", "unknown")
        request_path = request_data.get("path", "")
        request_body = request_data.get("body", "")

        # Update IP tracking
        self.ip_tracking[source_ip]["request_count"] += 1
        self.ip_tracking[source_ip]["last_seen"] = datetime.utcnow()

        # Check for threat patterns
        for pattern_info in self.threat_patterns:
            if pattern_info["pattern"].search(request_path) or pattern_info[
                "pattern"
            ].search(request_body):

                event_id = f"sec_{int(time.time())}_{len(self.security_events)}"

                security_event = SecurityEvent(
                    event_id=event_id,
                    event_type=SecurityEventType.POTENTIAL_ATTACK,
                    timestamp=datetime.utcnow(),
                    source_ip=source_ip,
                    user_agent=user_agent,
                    description=f"{pattern_info['name']} detected in request",
                    severity=pattern_info["severity"],
                    metadata={
                        "pattern_name": pattern_info["name"],
                        "request_path": request_path,
                        "request_body": request_body[:500],  # Truncate for storage
                    },
                )

                self.security_events.append(security_event)
                logger.warning(f"Security event detected: {security_event.description}")

                return security_event

        # Check for rate limiting
        if self.ip_tracking[source_ip]["request_count"] > 100:  # More than 100 requests
            time_window = datetime.utcnow() - timedelta(minutes=5)
            if self.ip_tracking[source_ip]["last_seen"] > time_window:
                event_id = f"sec_{int(time.time())}_{len(self.security_events)}"

                security_event = SecurityEvent(
                    event_id=event_id,
                    event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                    timestamp=datetime.utcnow(),
                    source_ip=source_ip,
                    user_agent=user_agent,
                    description=f"Rate limit exceeded: {self.ip_tracking[source_ip]['request_count']} requests",
                    severity=5,
                    metadata={
                        "request_count": self.ip_tracking[source_ip]["request_count"]
                    },
                )

                self.security_events.append(security_event)
                return security_event

        return None

    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [e for e in self.security_events if e.timestamp >= cutoff_time]

        event_types = defaultdict(int)
        severity_distribution = defaultdict(int)
        top_source_ips = defaultdict(int)

        for event in recent_events:
            event_types[event.event_type.value] += 1
            severity_distribution[event.severity] += 1
            top_source_ips[event.source_ip] += 1

        return {
            "total_events": len(recent_events),
            "event_types": dict(event_types),
            "severity_distribution": dict(severity_distribution),
            "top_source_ips": dict(
                sorted(top_source_ips.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "high_severity_events": len([e for e in recent_events if e.severity >= 8]),
            "unresolved_events": len([e for e in recent_events if not e.resolved]),
            "time_period_hours": hours,
        }

    def block_ip(self, ip_address: str, reason: str):
        """Block an IP address"""
        self.ip_tracking[ip_address]["blocked"] = True
        self.ip_tracking[ip_address]["block_reason"] = reason
        logger.info(f"Blocked IP address {ip_address}: {reason}")

    def unblock_ip(self, ip_address: str):
        """Unblock an IP address"""
        if ip_address in self.ip_tracking:
            self.ip_tracking[ip_address]["blocked"] = False
            logger.info(f"Unblocked IP address {ip_address}")


class PerformanceAnalyzer:
    """Analyze system and application performance"""

    def __init__(self):
        self.metrics_history: deque = deque(maxlen=10000)
        self.performance_baselines: Dict[str, float] = {}
        self.anomaly_threshold = 2.0  # Standard deviations

    def record_metric(
        self, metric_name: str, value: float, tags: Dict[str, str] = None
    ):
        """Record a performance metric"""
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
        )
        self.metrics_history.append(metric)

    def calculate_baseline(self, metric_name: str, days: int = 7) -> Dict[str, float]:
        """Calculate performance baseline for a metric"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        values = [
            m.value
            for m in self.metrics_history
            if m.metric_name == metric_name and m.timestamp >= cutoff_time
        ]

        if len(values) < 10:
            return {"error": "Insufficient data for baseline calculation"}

        baseline = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
            "p95": sorted(values)[int(len(values) * 0.95)],
            "p99": sorted(values)[int(len(values) * 0.99)],
            "sample_size": len(values),
        }

        self.performance_baselines[metric_name] = baseline
        return baseline

    def detect_anomalies(
        self, metric_name: str, hours: int = 1
    ) -> List[Dict[str, Any]]:
        """Detect performance anomalies"""
        if metric_name not in self.performance_baselines:
            self.calculate_baseline(metric_name)

        baseline = self.performance_baselines.get(metric_name)
        if not baseline or "error" in baseline:
            return []

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [
            m
            for m in self.metrics_history
            if m.metric_name == metric_name and m.timestamp >= cutoff_time
        ]

        anomalies = []
        for metric in recent_metrics:
            # Check if value is outside normal range
            z_score = abs(metric.value - baseline["mean"]) / baseline["std_dev"]
            if z_score > self.anomaly_threshold:
                anomalies.append(
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "value": metric.value,
                        "baseline_mean": baseline["mean"],
                        "z_score": z_score,
                        "severity": "high" if z_score > 3.0 else "medium",
                    }
                )

        return anomalies

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate performance analysis report"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {"error": "No metrics data available"}

        # Group by metric name
        metrics_by_name = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_name[metric.metric_name].append(metric.value)

        report = {
            "time_period_hours": hours,
            "total_data_points": len(recent_metrics),
            "metrics_analyzed": len(metrics_by_name),
            "metric_summaries": {},
            "anomalies": {},
            "recommendations": [],
        }

        for metric_name, values in metrics_by_name.items():
            if len(values) > 1:
                summary = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                }
                report["metric_summaries"][metric_name] = summary

                # Check for anomalies
                anomalies = self.detect_anomalies(metric_name, hours)
                if anomalies:
                    report["anomalies"][metric_name] = anomalies

        # Generate recommendations
        report["recommendations"] = self._generate_performance_recommendations(report)

        return report

    def _generate_performance_recommendations(
        self, report: Dict[str, Any]
    ) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        # Check for high response times
        if "response_time" in report["metric_summaries"]:
            response_time_summary = report["metric_summaries"]["response_time"]
            if response_time_summary["mean"] > 2.0:  # More than 2 seconds
                recommendations.append(
                    "High average response time detected - consider performance optimization"
                )

        # Check for high error rates
        if "error_rate" in report["metric_summaries"]:
            error_rate_summary = report["metric_summaries"]["error_rate"]
            if error_rate_summary["mean"] > 0.05:  # More than 5% error rate
                recommendations.append(
                    "High error rate detected - investigate error causes"
                )

        # Check for resource utilization
        if "cpu_utilization" in report["metric_summaries"]:
            cpu_summary = report["metric_summaries"]["cpu_utilization"]
            if cpu_summary["mean"] > 80:
                recommendations.append(
                    "High CPU utilization - consider scaling up resources"
                )

        # Check for anomalies
        if report["anomalies"]:
            recommendations.append(
                f"Performance anomalies detected in {len(report['anomalies'])} metrics - investigate unusual patterns"
            )

        return recommendations


class OperationalDashboard:
    """Operational dashboard for system overview"""

    def __init__(
        self,
        log_analyzer: LogAnalyzer,
        security_monitor: SecurityMonitor,
        performance_analyzer: PerformanceAnalyzer,
    ):
        self.log_analyzer = log_analyzer
        self.security_monitor = security_monitor
        self.performance_analyzer = performance_analyzer

    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_overview": await self._get_system_overview(),
            "log_analysis": await self._get_log_summary(),
            "security_status": self._get_security_status(),
            "performance_metrics": self._get_performance_summary(),
            "alerts": await self._get_active_alerts(),
            "recommendations": self._get_operational_recommendations(),
        }

        return dashboard_data

    async def _get_system_overview(self) -> Dict[str, Any]:
        """Get system overview metrics"""
        import psutil

        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": (psutil.disk_usage("/").used / psutil.disk_usage("/").total)
            * 100,
            "uptime_seconds": time.time() - psutil.boot_time(),
            "process_count": len(psutil.pids()),
        }

    async def _get_log_summary(self) -> Dict[str, Any]:
        """Get log analysis summary"""
        log_analysis = await self.log_analyzer.analyze_logs_directory()

        if "error" in log_analysis:
            return {"status": "error", "message": log_analysis["error"]}

        return {
            "total_lines": log_analysis["total_lines"],
            "error_count": log_analysis["overall_log_levels"].get("ERROR", 0),
            "warning_count": log_analysis["overall_log_levels"].get("WARNING", 0),
            "critical_patterns": {
                k: v for k, v in log_analysis["overall_patterns"].items() if v > 0
            },
            "issues_count": len(log_analysis["summary_issues"]),
        }

    def _get_security_status(self) -> Dict[str, Any]:
        """Get security status summary"""
        security_summary = self.security_monitor.get_security_summary(24)

        return {
            "total_events": security_summary["total_events"],
            "high_severity_events": security_summary["high_severity_events"],
            "unresolved_events": security_summary["unresolved_events"],
            "blocked_ips": len(
                [
                    ip
                    for ip, data in self.security_monitor.ip_tracking.items()
                    if data.get("blocked", False)
                ]
            ),
            "threat_level": self._calculate_threat_level(security_summary),
        }

    def _calculate_threat_level(self, security_summary: Dict[str, Any]) -> str:
        """Calculate overall threat level"""
        high_severity = security_summary["high_severity_events"]
        total_events = security_summary["total_events"]

        if high_severity > 10:
            return "critical"
        elif high_severity > 5 or total_events > 50:
            return "high"
        elif total_events > 10:
            return "medium"
        else:
            return "low"

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        performance_report = self.performance_analyzer.get_performance_report(
            1
        )  # Last hour

        if "error" in performance_report:
            return {"status": "error", "message": performance_report["error"]}

        return {
            "data_points": performance_report["total_data_points"],
            "metrics_count": performance_report["metrics_analyzed"],
            "anomalies_count": len(performance_report["anomalies"]),
            "key_metrics": {
                k: v["mean"] for k, v in performance_report["metric_summaries"].items()
            },
        }

    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts from all systems"""
        alerts = []

        # Log alerts
        log_alerts = await self.log_analyzer.check_alert_rules()
        for alert in log_alerts:
            alerts.append(
                {
                    "type": "log",
                    "severity": "warning",
                    "message": f"Log alert: {alert['rule_name']} ({alert['matches']} matches)",
                    "timestamp": alert["triggered_at"],
                }
            )

        # Security alerts
        recent_security_events = [
            e
            for e in self.security_monitor.security_events
            if e.timestamp >= datetime.utcnow() - timedelta(hours=1) and e.severity >= 7
        ]
        for event in recent_security_events:
            alerts.append(
                {
                    "type": "security",
                    "severity": "critical" if event.severity >= 8 else "warning",
                    "message": f"Security event: {event.description}",
                    "timestamp": event.timestamp.isoformat(),
                }
            )

        return alerts

    def _get_operational_recommendations(self) -> List[str]:
        """Get operational recommendations"""
        recommendations = []

        # Performance recommendations
        perf_report = self.performance_analyzer.get_performance_report(24)
        if "recommendations" in perf_report:
            recommendations.extend(perf_report["recommendations"])

        # Security recommendations
        security_summary = self.security_monitor.get_security_summary(24)
        if security_summary["high_severity_events"] > 0:
            recommendations.append(
                "Review and respond to high-severity security events"
            )

        if security_summary["unresolved_events"] > 10:
            recommendations.append("Address unresolved security events")

        return recommendations


async def main():
    """Demonstrate operational tools"""
    print("=== DSPy Operational Tools Demo ===")

    # Create operational tools
    log_analyzer = LogAnalyzer()
    security_monitor = SecurityMonitor()
    performance_analyzer = PerformanceAnalyzer()
    dashboard = OperationalDashboard(
        log_analyzer, security_monitor, performance_analyzer
    )

    print("Setting up operational monitoring...")

    # Add some log alert rules
    log_analyzer.add_alert_rule("High Error Rate", "ERROR|CRITICAL", 10, 5)
    log_analyzer.add_alert_rule(
        "Authentication Failures", "authentication.*failed", 5, 10
    )

    # Simulate some data
    print("\nSimulating operational data...")

    # Simulate performance metrics
    for i in range(50):
        performance_analyzer.record_metric(
            "response_time", 0.5 + (i * 0.01), {"endpoint": "/api/process"}
        )
        performance_analyzer.record_metric(
            "cpu_utilization", 60 + (i % 20), {"node": "worker-1"}
        )
        performance_analyzer.record_metric(
            "error_rate", 0.02 + (i * 0.001), {"service": "dspy-api"}
        )

    # Simulate security events
    for i in range(10):
        request_data = {
            "source_ip": f"192.168.1.{100 + i}",
            "user_agent": "Mozilla/5.0 (compatible; TestBot/1.0)",
            "path": f"/api/test?id={i}",
            "body": "",
        }
        security_monitor.analyze_request(request_data)

    # Simulate a potential attack
    attack_request = {
        "source_ip": "10.0.0.1",
        "user_agent": "AttackBot/1.0",
        "path": "/api/users?id=1' OR '1'='1",
        "body": "",
    }
    security_event = security_monitor.analyze_request(attack_request)
    if security_event:
        print(f"Security event detected: {security_event.description}")

    # Generate dashboard
    print("\nGenerating operational dashboard...")
    dashboard_data = await dashboard.generate_dashboard_data()

    print(f"\n=== Operational Dashboard ===")
    print(f"System Overview:")
    print(f"  CPU: {dashboard_data['system_overview']['cpu_percent']:.1f}%")
    print(f"  Memory: {dashboard_data['system_overview']['memory_percent']:.1f}%")
    print(f"  Disk: {dashboard_data['system_overview']['disk_percent']:.1f}%")

    print(f"\nSecurity Status:")
    print(f"  Total Events: {dashboard_data['security_status']['total_events']}")
    print(
        f"  High Severity: {dashboard_data['security_status']['high_severity_events']}"
    )
    print(f"  Threat Level: {dashboard_data['security_status']['threat_level']}")

    print(f"\nPerformance Metrics:")
    print(f"  Data Points: {dashboard_data['performance_metrics']['data_points']}")
    print(f"  Anomalies: {dashboard_data['performance_metrics']['anomalies_count']}")

    if dashboard_data["alerts"]:
        print(f"\nActive Alerts:")
        for alert in dashboard_data["alerts"]:
            print(f"  [{alert['severity'].upper()}] {alert['message']}")

    if dashboard_data["recommendations"]:
        print(f"\nRecommendations:")
        for rec in dashboard_data["recommendations"]:
            print(f"  • {rec}")

    print(f"\nOperational tools demonstration completed!")


if __name__ == "__main__":
    asyncio.run(main())
