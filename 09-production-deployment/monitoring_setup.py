#!/usr/bin/env python3
"""
Production Monitoring and Alerting System for DSPy Applications

This module provides comprehensive monitoring, alerting, and observability
tools for DSPy applications in production environments.

Learning Objectives:
- Implement comprehensive monitoring for DSPy applications
- Set up alerting systems with multiple notification channels
- Create performance monitoring and metrics collection
- Build observability dashboards and reporting
- Master incident response and automated recovery

Author: DSPy Learning Framework
"""

import asyncio
import json
import logging
import smtplib
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MimeMultipart
from email.mime.text import MimeText
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class AlertChannel(Enum):
    """Alert notification channels"""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"


class MetricType(Enum):
    """Types of metrics to collect"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Alert configuration and data"""

    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    source: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class Metric:
    """Metric data structure"""

    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class MonitoringConfig:
    """Monitoring system configuration"""

    app_name: str
    environment: str
    check_interval: int = 60
    alert_channels: List[AlertChannel] = field(default_factory=list)
    email_config: Dict[str, str] = field(default_factory=dict)
    slack_webhook: Optional[str] = None
    webhook_urls: List[str] = field(default_factory=list)
    metrics_retention_days: int = 30
    enable_auto_recovery: bool = True


class MetricsCollector:
    """Collect and store application metrics"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    labels TEXT,
                    description TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                ON alerts(timestamp)
            """
            )

    def record_metric(self, metric: Metric):
        """Record a metric to the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO metrics (name, metric_type, value, timestamp, labels, description)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    metric.name,
                    metric.metric_type.value,
                    metric.value,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.labels),
                    metric.description,
                ),
            )

    def record_alert(self, alert: Alert):
        """Record an alert to the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO alerts (name, severity, message, timestamp, source, details, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert.name,
                    alert.severity.value,
                    alert.message,
                    alert.timestamp.isoformat(),
                    alert.source,
                    json.dumps(alert.details),
                    alert.resolved,
                ),
            )

    def get_metrics(self, name: str, hours: int = 24) -> List[Metric]:
        """Retrieve metrics from the database"""
        since = datetime.utcnow() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT name, metric_type, value, timestamp, labels, description
                FROM metrics
                WHERE name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """,
                (name, since.isoformat()),
            )

            metrics = []
            for row in cursor.fetchall():
                metrics.append(
                    Metric(
                        name=row[0],
                        metric_type=MetricType(row[1]),
                        value=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        labels=json.loads(row[4]) if row[4] else {},
                        description=row[5] or "",
                    )
                )

            return metrics

    def cleanup_old_metrics(self, retention_days: int = 30):
        """Clean up old metrics beyond retention period"""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM metrics WHERE timestamp < ?
            """,
                (cutoff.isoformat(),),
            )

            deleted_count = cursor.rowcount
            logger.info(f"Cleaned up {deleted_count} old metrics")


class SystemMetricsCollector:
    """Collect system-level metrics"""

    @staticmethod
    def collect_system_metrics() -> List[Metric]:
        """Collect current system metrics"""
        timestamp = datetime.utcnow()
        metrics = []

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(
            Metric(
                name="system_cpu_percent",
                metric_type=MetricType.GAUGE,
                value=cpu_percent,
                timestamp=timestamp,
                description="CPU utilization percentage",
            )
        )

        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(
            Metric(
                name="system_memory_percent",
                metric_type=MetricType.GAUGE,
                value=memory.percent,
                timestamp=timestamp,
                description="Memory utilization percentage",
            )
        )

        metrics.append(
            Metric(
                name="system_memory_available_gb",
                metric_type=MetricType.GAUGE,
                value=memory.available / (1024**3),
                timestamp=timestamp,
                description="Available memory in GB",
            )
        )

        # Disk metrics
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        metrics.append(
            Metric(
                name="system_disk_percent",
                metric_type=MetricType.GAUGE,
                value=disk_percent,
                timestamp=timestamp,
                description="Disk utilization percentage",
            )
        )

        metrics.append(
            Metric(
                name="system_disk_free_gb",
                metric_type=MetricType.GAUGE,
                value=disk.free / (1024**3),
                timestamp=timestamp,
                description="Free disk space in GB",
            )
        )

        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            metrics.append(
                Metric(
                    name="system_network_bytes_sent",
                    metric_type=MetricType.COUNTER,
                    value=network.bytes_sent,
                    timestamp=timestamp,
                    description="Total bytes sent over network",
                )
            )

            metrics.append(
                Metric(
                    name="system_network_bytes_recv",
                    metric_type=MetricType.COUNTER,
                    value=network.bytes_recv,
                    timestamp=timestamp,
                    description="Total bytes received over network",
                )
            )
        except Exception as e:
            logger.warning(f"Failed to collect network metrics: {e}")

        return metrics


class ApplicationMetricsCollector:
    """Collect application-specific metrics"""

    def __init__(self, app_name: str):
        self.app_name = app_name
        self.request_count = 0
        self.error_count = 0
        self.response_times = []

    def record_request(self, response_time: float, status_code: int):
        """Record API request metrics"""
        self.request_count += 1
        self.response_times.append(response_time)

        if status_code >= 400:
            self.error_count += 1

    def collect_app_metrics(self) -> List[Metric]:
        """Collect current application metrics"""
        timestamp = datetime.utcnow()
        metrics = []

        # Request metrics
        metrics.append(
            Metric(
                name="app_requests_total",
                metric_type=MetricType.COUNTER,
                value=self.request_count,
                timestamp=timestamp,
                labels={"app": self.app_name},
                description="Total number of requests",
            )
        )

        metrics.append(
            Metric(
                name="app_errors_total",
                metric_type=MetricType.COUNTER,
                value=self.error_count,
                timestamp=timestamp,
                labels={"app": self.app_name},
                description="Total number of errors",
            )
        )

        # Response time metrics
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            max_response_time = max(self.response_times)

            metrics.append(
                Metric(
                    name="app_response_time_avg",
                    metric_type=MetricType.GAUGE,
                    value=avg_response_time,
                    timestamp=timestamp,
                    labels={"app": self.app_name},
                    description="Average response time in seconds",
                )
            )

            metrics.append(
                Metric(
                    name="app_response_time_max",
                    metric_type=MetricType.GAUGE,
                    value=max_response_time,
                    timestamp=timestamp,
                    labels={"app": self.app_name},
                    description="Maximum response time in seconds",
                )
            )

        # Error rate
        error_rate = (
            (self.error_count / self.request_count * 100)
            if self.request_count > 0
            else 0
        )
        metrics.append(
            Metric(
                name="app_error_rate_percent",
                metric_type=MetricType.GAUGE,
                value=error_rate,
                timestamp=timestamp,
                labels={"app": self.app_name},
                description="Error rate percentage",
            )
        )

        return metrics


class AlertManager:
    """Manage alerts and notifications"""

    def __init__(self, config: MonitoringConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Callable] = []

    def add_alert_rule(self, rule: Callable[[List[Metric]], Optional[Alert]]):
        """Add custom alert rule"""
        self.alert_rules.append(rule)

    async def process_metrics(self, metrics: List[Metric]):
        """Process metrics and generate alerts"""
        # Check built-in alert rules
        alerts = self._check_builtin_rules(metrics)

        # Check custom alert rules
        for rule in self.alert_rules:
            try:
                alert = rule(metrics)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.error(f"Alert rule failed: {e}")

        # Process alerts
        for alert in alerts:
            await self._handle_alert(alert)

    def _check_builtin_rules(self, metrics: List[Metric]) -> List[Alert]:
        """Check built-in alert rules"""
        alerts = []
        timestamp = datetime.utcnow()

        for metric in metrics:
            # High CPU usage
            if metric.name == "system_cpu_percent" and metric.value > 90:
                alerts.append(
                    Alert(
                        name="high_cpu_usage",
                        severity=AlertSeverity.CRITICAL,
                        message=f"High CPU usage: {metric.value:.1f}%",
                        timestamp=timestamp,
                        source="system_monitor",
                        details={"cpu_percent": metric.value},
                    )
                )

            # High memory usage
            elif metric.name == "system_memory_percent" and metric.value > 90:
                alerts.append(
                    Alert(
                        name="high_memory_usage",
                        severity=AlertSeverity.CRITICAL,
                        message=f"High memory usage: {metric.value:.1f}%",
                        timestamp=timestamp,
                        source="system_monitor",
                        details={"memory_percent": metric.value},
                    )
                )

            # High disk usage
            elif metric.name == "system_disk_percent" and metric.value > 85:
                severity = (
                    AlertSeverity.CRITICAL
                    if metric.value > 95
                    else AlertSeverity.WARNING
                )
                alerts.append(
                    Alert(
                        name="high_disk_usage",
                        severity=severity,
                        message=f"High disk usage: {metric.value:.1f}%",
                        timestamp=timestamp,
                        source="system_monitor",
                        details={"disk_percent": metric.value},
                    )
                )

            # High error rate
            elif metric.name == "app_error_rate_percent" and metric.value > 5:
                severity = (
                    AlertSeverity.CRITICAL
                    if metric.value > 10
                    else AlertSeverity.WARNING
                )
                alerts.append(
                    Alert(
                        name="high_error_rate",
                        severity=severity,
                        message=f"High error rate: {metric.value:.1f}%",
                        timestamp=timestamp,
                        source="app_monitor",
                        details={"error_rate": metric.value},
                    )
                )

            # Slow response times
            elif metric.name == "app_response_time_avg" and metric.value > 2.0:
                severity = (
                    AlertSeverity.CRITICAL
                    if metric.value > 5.0
                    else AlertSeverity.WARNING
                )
                alerts.append(
                    Alert(
                        name="slow_response_time",
                        severity=severity,
                        message=f"Slow response time: {metric.value:.2f}s",
                        timestamp=timestamp,
                        source="app_monitor",
                        details={"response_time": metric.value},
                    )
                )

        return alerts
