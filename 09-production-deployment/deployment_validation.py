#!/usr/bin/env python3
"""
Deployment Validation and Health Check System

This module provides comprehensive deployment validation, health checks,
and monitoring utilities for DSPy applications in production.

Author: DSPy Learning Framework
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status types"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Types of health checks"""

    HTTP = "http"
    TCP = "tcp"
    COMMAND = "command"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    RESOURCE = "resource"


@dataclass
class HealthCheck:
    """Individual health check configuration"""

    name: str
    check_type: CheckType
    endpoint: Optional[str] = None
    command: Optional[str] = None
    timeout: int = 30
    interval: int = 60
    retries: int = 3
    expected_status: int = 200
    expected_response: Optional[str] = None
    critical: bool = True


@dataclass
class HealthCheckResult:
    """Result of a health check"""

    check_name: str
    status: HealthStatus
    response_time: float
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class HTTPHealthChecker:
    """HTTP-based health checks"""

    @staticmethod
    async def check_endpoint(check: HealthCheck) -> HealthCheckResult:
        """Perform HTTP health check"""
        start_time = time.time()

        try:
            timeout = aiohttp.ClientTimeout(total=check.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(check.endpoint) as response:
                    response_time = time.time() - start_time
                    response_text = await response.text()

                    # Check status code
                    if response.status != check.expected_status:
                        return HealthCheckResult(
                            check_name=check.name,
                            status=HealthStatus.UNHEALTHY,
                            response_time=response_time,
                            message=f"Expected status {check.expected_status}, got {response.status}",
                            timestamp=datetime.utcnow(),
                            details={
                                "status_code": response.status,
                                "response": response_text[:500],
                            },
                        )

                    # Check response content if specified
                    if (
                        check.expected_response
                        and check.expected_response not in response_text
                    ):
                        return HealthCheckResult(
                            check_name=check.name,
                            status=HealthStatus.DEGRADED,
                            response_time=response_time,
                            message=f"Expected response content not found",
                            timestamp=datetime.utcnow(),
                            details={"response": response_text[:500]},
                        )

                    return HealthCheckResult(
                        check_name=check.name,
                        status=HealthStatus.HEALTHY,
                        response_time=response_time,
                        message="HTTP check passed",
                        timestamp=datetime.utcnow(),
                        details={"status_code": response.status},
                    )

        except asyncio.TimeoutError:
            return HealthCheckResult(
                check_name=check.name,
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                message=f"Request timeout after {check.timeout}s",
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            return HealthCheckResult(
                check_name=check.name,
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                message=f"HTTP check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                details={"error": str(e)},
            )


class CommandHealthChecker:
    """Command-based health checks"""

    @staticmethod
    def check_command(check: HealthCheck) -> HealthCheckResult:
        """Perform command-based health check"""
        start_time = time.time()

        try:
            result = subprocess.run(
                check.command.split(),
                capture_output=True,
                text=True,
                timeout=check.timeout,
            )

            response_time = time.time() - start_time

            if result.returncode == 0:
                return HealthCheckResult(
                    check_name=check.name,
                    status=HealthStatus.HEALTHY,
                    response_time=response_time,
                    message="Command executed successfully",
                    timestamp=datetime.utcnow(),
                    details={
                        "stdout": result.stdout[:500],
                        "stderr": result.stderr[:500],
                    },
                )
            else:
                return HealthCheckResult(
                    check_name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    response_time=response_time,
                    message=f"Command failed with exit code {result.returncode}",
                    timestamp=datetime.utcnow(),
                    details={
                        "stdout": result.stdout[:500],
                        "stderr": result.stderr[:500],
                    },
                )

        except subprocess.TimeoutExpired:
            return HealthCheckResult(
                check_name=check.name,
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                message=f"Command timeout after {check.timeout}s",
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            return HealthCheckResult(
                check_name=check.name,
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                message=f"Command check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                details={"error": str(e)},
            )


class ResourceHealthChecker:
    """System resource health checks"""

    @staticmethod
    def check_system_resources() -> HealthCheckResult:
        """Check system resource utilization"""
        start_time = time.time()

        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Determine health status based on thresholds
            status = HealthStatus.HEALTHY
            messages = []

            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 75:
                status = HealthStatus.DEGRADED
                messages.append(f"Elevated CPU usage: {cpu_percent:.1f}%")

            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                messages.append(f"High memory usage: {memory.percent:.1f}%")
            elif memory.percent > 75:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                messages.append(f"Elevated memory usage: {memory.percent:.1f}%")

            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 90:
                status = HealthStatus.UNHEALTHY
                messages.append(f"High disk usage: {disk_percent:.1f}%")
            elif disk_percent > 80:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                messages.append(f"Elevated disk usage: {disk_percent:.1f}%")

            message = (
                "; ".join(messages)
                if messages
                else "System resources within normal limits"
            )

            return HealthCheckResult(
                check_name="system_resources",
                status=status,
                response_time=time.time() - start_time,
                message=message,
                timestamp=datetime.utcnow(),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk_percent,
                    "available_memory_gb": memory.available / (1024**3),
                    "free_disk_gb": disk.free / (1024**3),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="system_resources",
                status=HealthStatus.UNKNOWN,
                response_time=time.time() - start_time,
                message=f"Resource check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                details={"error": str(e)},
            )


class DeploymentValidator:
    """Comprehensive deployment validation"""

    def __init__(self, health_checks: List[HealthCheck]):
        self.health_checks = health_checks
        self.results_history: List[List[HealthCheckResult]] = []

    async def run_health_checks(self) -> List[HealthCheckResult]:
        """Run all configured health checks"""
        results = []

        for check in self.health_checks:
            for attempt in range(check.retries):
                try:
                    if check.check_type == CheckType.HTTP:
                        result = await HTTPHealthChecker.check_endpoint(check)
                    elif check.check_type == CheckType.COMMAND:
                        result = CommandHealthChecker.check_command(check)
                    elif check.check_type == CheckType.RESOURCE:
                        result = ResourceHealthChecker.check_system_resources()
                    else:
                        result = HealthCheckResult(
                            check_name=check.name,
                            status=HealthStatus.UNKNOWN,
                            response_time=0,
                            message=f"Unsupported check type: {check.check_type}",
                            timestamp=datetime.utcnow(),
                        )

                    # If check passes or it's the last attempt, break
                    if (
                        result.status == HealthStatus.HEALTHY
                        or attempt == check.retries - 1
                    ):
                        results.append(result)
                        break

                    # Wait before retry
                    if attempt < check.retries - 1:
                        await asyncio.sleep(2)

                except Exception as e:
                    if attempt == check.retries - 1:
                        results.append(
                            HealthCheckResult(
                                check_name=check.name,
                                status=HealthStatus.UNHEALTHY,
                                response_time=0,
                                message=f"Health check failed: {str(e)}",
                                timestamp=datetime.utcnow(),
                                details={"error": str(e)},
                            )
                        )

        self.results_history.append(results)
        return results

    def get_overall_health(
        self, results: List[HealthCheckResult]
    ) -> Tuple[HealthStatus, str]:
        """Determine overall health status"""
        if not results:
            return HealthStatus.UNKNOWN, "No health check results"

        critical_checks = [
            r
            for r in results
            if any(c.critical for c in self.health_checks if c.name == r.check_name)
        ]
        non_critical_checks = [
            r
            for r in results
            if not any(c.critical for c in self.health_checks if c.name == r.check_name)
        ]

        # Check critical health checks
        critical_unhealthy = [
            r for r in critical_checks if r.status == HealthStatus.UNHEALTHY
        ]
        critical_degraded = [
            r for r in critical_checks if r.status == HealthStatus.DEGRADED
        ]

        if critical_unhealthy:
            return (
                HealthStatus.UNHEALTHY,
                f"Critical checks failing: {', '.join([r.check_name for r in critical_unhealthy])}",
            )

        # Check all health checks
        all_unhealthy = [r for r in results if r.status == HealthStatus.UNHEALTHY]
        all_degraded = [r for r in results if r.status == HealthStatus.DEGRADED]

        if all_unhealthy:
            return (
                HealthStatus.DEGRADED,
                f"Some checks failing: {', '.join([r.check_name for r in all_unhealthy])}",
            )

        if critical_degraded or all_degraded:
            return (
                HealthStatus.DEGRADED,
                f"Some checks degraded: {', '.join([r.check_name for r in critical_degraded + all_degraded])}",
            )

        return HealthStatus.HEALTHY, "All health checks passing"

    def generate_health_report(
        self, results: List[HealthCheckResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        overall_status, overall_message = self.get_overall_health(results)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status.value,
            "overall_message": overall_message,
            "checks": [
                {
                    "name": result.check_name,
                    "status": result.status.value,
                    "response_time": result.response_time,
                    "message": result.message,
                    "timestamp": result.timestamp.isoformat(),
                    "details": result.details,
                }
                for result in results
            ],
            "summary": {
                "total_checks": len(results),
                "healthy": len(
                    [r for r in results if r.status == HealthStatus.HEALTHY]
                ),
                "degraded": len(
                    [r for r in results if r.status == HealthStatus.DEGRADED]
                ),
                "unhealthy": len(
                    [r for r in results if r.status == HealthStatus.UNHEALTHY]
                ),
                "unknown": len(
                    [r for r in results if r.status == HealthStatus.UNKNOWN]
                ),
                "average_response_time": (
                    sum(r.response_time for r in results) / len(results)
                    if results
                    else 0
                ),
            },
        }


class DeploymentMonitor:
    """Continuous deployment monitoring"""

    def __init__(self, validator: DeploymentValidator, check_interval: int = 60):
        self.validator = validator
        self.check_interval = check_interval
        self.running = False
        self.alert_callbacks: List[callable] = []

    def add_alert_callback(self, callback: callable):
        """Add callback for health alerts"""
        self.alert_callbacks.append(callback)

    async def start_monitoring(self):
        """Start continuous health monitoring"""
        self.running = True
        logger.info("Starting deployment monitoring")

        while self.running:
            try:
                results = await self.validator.run_health_checks()
                overall_status, overall_message = self.validator.get_overall_health(
                    results
                )

                # Log current status
                logger.info(
                    f"Health check completed: {overall_status.value} - {overall_message}"
                )

                # Trigger alerts if needed
                if overall_status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
                    for callback in self.alert_callbacks:
                        try:
                            await callback(overall_status, overall_message, results)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")

                # Wait for next check
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.check_interval)

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.running = False
        logger.info("Stopping deployment monitoring")


class KubernetesValidator:
    """Kubernetes-specific deployment validation"""

    @staticmethod
    def check_pod_status(namespace: str, app_name: str) -> HealthCheckResult:
        """Check Kubernetes pod status"""
        start_time = time.time()

        try:
            # Get pod status
            cmd = [
                "kubectl",
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                f"app={app_name}",
                "-o",
                "json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return HealthCheckResult(
                    check_name="k8s_pod_status",
                    status=HealthStatus.UNHEALTHY,
                    response_time=time.time() - start_time,
                    message=f"kubectl command failed: {result.stderr}",
                    timestamp=datetime.utcnow(),
                )

            pods_data = json.loads(result.stdout)
            pods = pods_data.get("items", [])

            if not pods:
                return HealthCheckResult(
                    check_name="k8s_pod_status",
                    status=HealthStatus.UNHEALTHY,
                    response_time=time.time() - start_time,
                    message="No pods found",
                    timestamp=datetime.utcnow(),
                )

            # Check pod statuses
            running_pods = 0
            total_pods = len(pods)
            pod_details = []

            for pod in pods:
                pod_name = pod["metadata"]["name"]
                pod_status = pod["status"]["phase"]
                pod_details.append({"name": pod_name, "status": pod_status})

                if pod_status == "Running":
                    running_pods += 1

            if running_pods == 0:
                status = HealthStatus.UNHEALTHY
                message = "No pods are running"
            elif running_pods < total_pods:
                status = HealthStatus.DEGRADED
                message = f"{running_pods}/{total_pods} pods running"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {total_pods} pods running"

            return HealthCheckResult(
                check_name="k8s_pod_status",
                status=status,
                response_time=time.time() - start_time,
                message=message,
                timestamp=datetime.utcnow(),
                details={
                    "pods": pod_details,
                    "running": running_pods,
                    "total": total_pods,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="k8s_pod_status",
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                message=f"Kubernetes check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                details={"error": str(e)},
            )

    @staticmethod
    def check_service_endpoints(namespace: str, app_name: str) -> HealthCheckResult:
        """Check Kubernetes service endpoints"""
        start_time = time.time()

        try:
            cmd = [
                "kubectl",
                "get",
                "endpoints",
                "-n",
                namespace,
                f"{app_name}-service",
                "-o",
                "json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return HealthCheckResult(
                    check_name="k8s_service_endpoints",
                    status=HealthStatus.UNHEALTHY,
                    response_time=time.time() - start_time,
                    message=f"kubectl command failed: {result.stderr}",
                    timestamp=datetime.utcnow(),
                )

            endpoints_data = json.loads(result.stdout)
            subsets = endpoints_data.get("subsets", [])

            if not subsets:
                return HealthCheckResult(
                    check_name="k8s_service_endpoints",
                    status=HealthStatus.UNHEALTHY,
                    response_time=time.time() - start_time,
                    message="No service endpoints available",
                    timestamp=datetime.utcnow(),
                )

            total_endpoints = sum(
                len(subset.get("addresses", [])) for subset in subsets
            )

            return HealthCheckResult(
                check_name="k8s_service_endpoints",
                status=HealthStatus.HEALTHY,
                response_time=time.time() - start_time,
                message=f"{total_endpoints} service endpoints available",
                timestamp=datetime.utcnow(),
                details={"endpoints": total_endpoints},
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="k8s_service_endpoints",
                status=HealthStatus.UNHEALTHY,
                response_time=time.time() - start_time,
                message=f"Service endpoints check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                details={"error": str(e)},
            )


async def alert_callback_example(
    status: HealthStatus, message: str, results: List[HealthCheckResult]
):
    """Example alert callback"""
    logger.warning(f"ALERT: {status.value} - {message}")

    # Here you could integrate with:
    # - Slack notifications
    # - Email alerts
    # - PagerDuty
    # - Custom webhook endpoints


async def main():
    """Demonstrate deployment validation"""
    print("=== Deployment Validation System ===")

    # Configure health checks
    health_checks = [
        HealthCheck(
            name="api_health",
            check_type=CheckType.HTTP,
            endpoint="http://localhost:8000/health",
            timeout=10,
            critical=True,
        ),
        HealthCheck(
            name="api_ready",
            check_type=CheckType.HTTP,
            endpoint="http://localhost:8000/ready",
            timeout=10,
            critical=True,
        ),
        HealthCheck(
            name="system_resources", check_type=CheckType.RESOURCE, critical=False
        ),
        HealthCheck(
            name="database_connection",
            check_type=CheckType.COMMAND,
            command="python -c 'import psycopg2; print(\"DB OK\")'",
            timeout=15,
            critical=True,
        ),
    ]

    # Create validator
    validator = DeploymentValidator(health_checks)

    # Run health checks
    print("Running health checks...")
    results = await validator.run_health_checks()

    # Generate report
    report = validator.generate_health_report(results)
    print(f"\nHealth Report:")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Message: {report['overall_message']}")
    print(f"Summary: {report['summary']}")

    # Show individual check results
    print("\nIndividual Check Results:")
    for check_result in report["checks"]:
        print(
            f"- {check_result['name']}: {check_result['status']} ({check_result['response_time']:.3f}s)"
        )
        print(f"  {check_result['message']}")

    print("\n" + "=" * 50)
    print("Deployment validation completed!")


if __name__ == "__main__":
    asyncio.run(main())
