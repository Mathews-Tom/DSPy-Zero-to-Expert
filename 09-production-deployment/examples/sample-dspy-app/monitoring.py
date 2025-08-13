#!/usr/bin/env python3
"""
Production Monitoring Script
"""

import asyncio
import logging
from deployment_validation import (
    HealthCheck, CheckType, DeploymentValidator,
    DeploymentMonitor, HealthStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def alert_callback(status, message, results):
    """Handle health alerts"""
    if status == HealthStatus.UNHEALTHY:
        logger.error(f"CRITICAL ALERT: {message}")
        # Here you would integrate with your alerting system
        # - Send Slack notification
        # - Trigger PagerDuty alert
        # - Send email notification
    elif status == HealthStatus.DEGRADED:
        logger.warning(f"WARNING: {message}")

async def main():
    """Start monitoring"""
    health_checks = [
        HealthCheck(
            name="api_health",
            check_type=CheckType.HTTP,
            endpoint="http://localhost:8000/health",
            timeout=10,
            critical=True
        ),
        HealthCheck(
            name="api_ready",
            check_type=CheckType.HTTP,
            endpoint="http://localhost:8000/ready",
            timeout=10,
            critical=True
        ),
        HealthCheck(
            name="system_resources",
            check_type=CheckType.RESOURCE,
            critical=False
        )
    ]
    
    validator = DeploymentValidator(health_checks)
    monitor = DeploymentMonitor(validator, check_interval=60)
    monitor.add_alert_callback(alert_callback)
    
    logger.info("Starting production monitoring...")
    await monitor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
