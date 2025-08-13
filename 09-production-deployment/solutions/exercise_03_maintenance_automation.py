#!/usr/bin/env python3
"""
Exercise 3: Maintenance Automation and Self-Healing

This exercise demonstrates automated maintenance workflows, self-healing
capabilities, and operational monitoring for DSPy applications.

Learning Objectives:
- Implement automated maintenance workflows
- Configure self-healing systems
- Set up operational monitoring
- Create backup and recovery procedures

Author: DSPy Learning Framework
"""

import asyncio
import json
import random

# Import our production deployment modules
import sys
from datetime import datetime, timedelta

sys.path.append("..")

from maintenance_automation import (
    AutomationRule,
    AutomationTrigger,
    MaintenanceAutomation,
)
from maintenance_operations import (
    MaintenanceOperation,
    MaintenanceOperationsManager,
    MaintenanceStatus,
    MaintenanceTask,
    MaintenanceType,
)
from operational_tools import LogAnalyzer, PerformanceAnalyzer, SecurityMonitor


async def exercise_03_solution():
    """
    Exercise 3 Solution: Maintenance Automation and Self-Healing

    This solution demonstrates:
    1. Setting up automated maintenance workflows
    2. Configuring self-healing systems
    3. Implementing operational monitoring
    4. Creating backup and recovery procedures
    """

    print("=== Exercise 3: Maintenance Automation and Self-Healing ===")
    print("Setting up automated maintenance and self-healing systems...")

    # Step 1: Set up maintenance operations manager
    print("\n1. Setting up maintenance operations...")
    ops_manager = MaintenanceOperationsManager()

    # Start operations in background
    ops_task = asyncio.create_task(ops_manager.start_operations())

    print("✓ Maintenance operations started")
    print("  - Health monitoring active")
    print("  - Backup management ready")
    print("  - Maintenance scheduling enabled")

    # Step 2: Set up maintenance automation
    print("\n2. Configuring maintenance automation...")
    automation = MaintenanceAutomation()

    # Add custom automation rules
    custom_rules = [
        AutomationRule(
            rule_id="high_memory_cleanup",
            name="High Memory Usage Cleanup",
            description="Clean up memory when usage exceeds 85%",
            trigger=AutomationTrigger.THRESHOLD_BASED,
            condition="memory_percent > 85",
            action="cleanup_memory",
            parameters={"cleanup_type": "aggressive"},
        ),
        AutomationRule(
            rule_id="error_rate_response",
            name="High Error Rate Response",
            description="Respond to high error rates",
            trigger=AutomationTrigger.THRESHOLD_BASED,
            condition="error_rate > 0.05",
            action="restart_services",
            parameters={"services": ["api", "worker"]},
        ),
        AutomationRule(
            rule_id="disk_space_management",
            name="Disk Space Management",
            description="Manage disk space when usage is high",
            trigger=AutomationTrigger.THRESHOLD_BASED,
            condition="disk_percent > 80",
            action="cleanup_old_logs",
            parameters={"retention_days": 7},
        ),
    ]

    for rule in custom_rules:
        automation.add_automation_rule(rule)

    # Start automation
    automation.start_automation()

    print("✓ Maintenance automation configured")
    print(f"  - Total rules: {len(automation.automation_rules)}")
    print(f"  - Custom rules: {len(custom_rules)}")
    print(f"  - Self-healing enabled: {automation.self_healing.enabled}")

    # Step 3: Set up operational monitoring
    print("\n3. Setting up operational monitoring...")

    # Log analyzer
    log_analyzer = LogAnalyzer()
    log_analyzer.add_alert_rule("Critical Errors", "CRITICAL|FATAL", 5, 10)
    log_analyzer.add_alert_rule("Memory Issues", "out.?of.?memory|oom", 1, 30)
    log_analyzer.add_alert_rule(
        "Database Errors", "database.*error|connection.*failed", 3, 15
    )

    # Security monitor
    security_monitor = SecurityMonitor()

    # Performance analyzer
    performance_analyzer = PerformanceAnalyzer()

    print("✓ Operational monitoring configured")
    print("  - Log analysis with alert rules")
    print("  - Security monitoring active")
    print("  - Performance analysis enabled")

    # Step 4: Create and execute maintenance tasks
    print("\n4. Creating maintenance tasks...")

    # Create maintenance tasks
    maintenance_tasks = [
        MaintenanceTask(
            task_id="system_health_check",
            name="System Health Check",
            description="Comprehensive system health assessment",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=8,
            estimated_duration_minutes=15,
            validation_checks=["cpu_check", "memory_check", "disk_check"],
        ),
        MaintenanceTask(
            task_id="security_scan",
            name="Security Vulnerability Scan",
            description="Scan for security vulnerabilities",
            maintenance_type=MaintenanceType.PREVENTIVE,
            priority=9,
            estimated_duration_minutes=30,
            validation_checks=["security_baseline", "vulnerability_check"],
        ),
        MaintenanceTask(
            task_id="performance_optimization",
            name="Performance Optimization",
            description="Optimize system performance",
            maintenance_type=MaintenanceType.ADAPTIVE,
            priority=6,
            estimated_duration_minutes=20,
            validation_checks=["performance_baseline", "response_time_check"],
        ),
    ]

    # Create maintenance operation
    maintenance_op = MaintenanceOperation(
        operation_id="scheduled_maintenance_001",
        name="Weekly System Maintenance",
        description="Comprehensive weekly maintenance",
        maintenance_type=MaintenanceType.SCHEDULED,
        status=MaintenanceStatus.PLANNED,
        scheduled_time=datetime.utcnow() + timedelta(seconds=10),
        tasks=maintenance_tasks,
        approval_required=False,  # Auto-approve for demo
    )

    # Schedule maintenance
    ops_manager.maintenance_scheduler.schedule_maintenance(maintenance_op)

    print("✓ Maintenance operation scheduled")
    print(f"  - Operation ID: {maintenance_op.operation_id}")
    print(f"  - Tasks: {len(maintenance_tasks)}")
    print(f"  - Scheduled for: {maintenance_op.scheduled_time.strftime('%H:%M:%S')}")

    # Step 5: Simulate system conditions and trigger automation
    print("\n5. Simulating system conditions...")

    # Simulate various system metrics over time
    simulation_phases = [
        {
            "name": "Normal Operation",
            "duration": 20,
            "conditions": {"memory": 60, "cpu": 50, "disk": 40, "error_rate": 0.01},
        },
        {
            "name": "High Memory Usage",
            "duration": 15,
            "conditions": {"memory": 90, "cpu": 70, "disk": 45, "error_rate": 0.02},
        },
        {
            "name": "High Error Rate",
            "duration": 10,
            "conditions": {"memory": 75, "cpu": 80, "disk": 50, "error_rate": 0.08},
        },
        {
            "name": "Disk Space Issue",
            "duration": 15,
            "conditions": {"memory": 65, "cpu": 60, "disk": 85, "error_rate": 0.03},
        },
        {
            "name": "Recovery",
            "duration": 20,
            "conditions": {"memory": 55, "cpu": 45, "disk": 60, "error_rate": 0.01},
        },
    ]

    automation_events = []
    performance_data = []
    security_events = []

    for phase in simulation_phases:
        print(f"\n  Phase: {phase['name']} ({phase['duration']}s)")
        phase_start = datetime.utcnow()

        while (datetime.utcnow() - phase_start).total_seconds() < phase["duration"]:
            conditions = phase["conditions"]

            # Record performance metrics
            performance_analyzer.record_metric(
                "cpu_utilization", conditions["cpu"] + random.uniform(-5, 5)
            )
            performance_analyzer.record_metric(
                "memory_utilization", conditions["memory"] + random.uniform(-3, 3)
            )
            performance_analyzer.record_metric(
                "disk_utilization", conditions["disk"] + random.uniform(-2, 2)
            )
            performance_analyzer.record_metric(
                "error_rate", conditions["error_rate"] + random.uniform(-0.005, 0.005)
            )
            performance_analyzer.record_metric(
                "response_time",
                0.5 + (conditions["cpu"] / 100) + random.uniform(-0.1, 0.1),
            )

            # Simulate security events occasionally
            if random.random() < 0.1:  # 10% chance
                request_data = {
                    "source_ip": f"192.168.1.{random.randint(100, 200)}",
                    "user_agent": "TestBot/1.0",
                    "path": f"/api/test?id={random.randint(1, 100)}",
                    "body": "",
                }

                # Occasionally simulate attacks
                if random.random() < 0.3:  # 30% of requests are suspicious
                    request_data["path"] = "/api/users?id=1' OR '1'='1"

                security_event = security_monitor.analyze_request(request_data)
                if security_event:
                    security_events.append(security_event)

            # Check automation conditions
            system_metrics = {
                "memory_percent": conditions["memory"],
                "cpu_percent": conditions["cpu"],
                "disk_percent": conditions["disk"],
                "error_rate": conditions["error_rate"],
                "service_health": (
                    "healthy" if conditions["error_rate"] < 0.05 else "warning"
                ),
            }

            # Evaluate self-healing conditions
            triggered_rules = await automation.self_healing.evaluate_healing_conditions(
                system_metrics
            )

            if triggered_rules:
                for rule_name in triggered_rules:
                    healing_result = (
                        await automation.self_healing.execute_healing_actions(rule_name)
                    )
                    automation_events.append(
                        {
                            "phase": phase["name"],
                            "rule": rule_name,
                            "success": healing_result["overall_success"],
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    print(f"    🔧 Self-healing triggered: {rule_name}")

            await asyncio.sleep(1)

    # Step 6: Wait for scheduled maintenance to complete
    print("\n6. Waiting for scheduled maintenance...")
    await asyncio.sleep(15)  # Wait for maintenance to execute

    # Step 7: Generate comprehensive report
    print("\n7. Generating comprehensive report...")

    # Get system status
    ops_status = ops_manager.get_system_status()
    automation_status = automation.get_automation_status()
    security_summary = security_monitor.get_security_summary(24)
    performance_report = performance_analyzer.get_performance_report(1)

    # Calculate baselines
    performance_analyzer.calculate_baseline("cpu_utilization")
    performance_analyzer.calculate_baseline("memory_utilization")
    performance_analyzer.calculate_baseline("response_time")

    # Detect anomalies
    cpu_anomalies = performance_analyzer.detect_anomalies("cpu_utilization")
    memory_anomalies = performance_analyzer.detect_anomalies("memory_utilization")

    report = {
        "maintenance_summary": {
            "operations_completed": ops_status["maintenance_scheduling"][
                "completed_operations"
            ],
            "operations_failed": ops_status["maintenance_scheduling"][
                "failed_operations"
            ],
            "health_checks_registered": ops_status["health_monitoring"][
                "registered_checks"
            ],
            "total_backups": ops_status["backup_management"]["total_backups"],
        },
        "automation_summary": {
            "total_executions": automation_status["total_executions"],
            "successful_executions": automation_status["successful_executions"],
            "self_healing_events": len(automation_events),
            "healing_success_rate": len([e for e in automation_events if e["success"]])
            / max(len(automation_events), 1)
            * 100,
        },
        "security_summary": {
            "total_events": security_summary["total_events"],
            "high_severity_events": security_summary["high_severity_events"],
            "unique_source_ips": len(security_summary["top_source_ips"]),
            "threat_level": security_monitor._calculate_threat_level(security_summary),
        },
        "performance_summary": {
            "metrics_collected": performance_report.get("total_data_points", 0),
            "anomalies_detected": len(cpu_anomalies) + len(memory_anomalies),
            "baselines_calculated": len(performance_analyzer.performance_baselines),
            "recommendations": len(performance_report.get("recommendations", [])),
        },
    }

    print("\n8. Final Report Summary:")
    print(f"  Maintenance Operations:")
    print(
        f"    - Completed operations: {report['maintenance_summary']['operations_completed']}"
    )
    print(
        f"    - Health checks: {report['maintenance_summary']['health_checks_registered']}"
    )
    print(f"    - Backups created: {report['maintenance_summary']['total_backups']}")

    print(f"  Automation Performance:")
    print(f"    - Total executions: {report['automation_summary']['total_executions']}")
    print(
        f"    - Success rate: {report['automation_summary']['successful_executions']}/{report['automation_summary']['total_executions']}"
    )
    print(
        f"    - Self-healing events: {report['automation_summary']['self_healing_events']}"
    )
    print(
        f"    - Healing success rate: {report['automation_summary']['healing_success_rate']:.1f}%"
    )

    print(f"  Security Monitoring:")
    print(f"    - Security events: {report['security_summary']['total_events']}")
    print(f"    - High severity: {report['security_summary']['high_severity_events']}")
    print(f"    - Threat level: {report['security_summary']['threat_level']}")

    print(f"  Performance Analysis:")
    print(
        f"    - Metrics collected: {report['performance_summary']['metrics_collected']}"
    )
    print(
        f"    - Anomalies detected: {report['performance_summary']['anomalies_detected']}"
    )
    print(
        f"    - Baselines calculated: {report['performance_summary']['baselines_calculated']}"
    )

    if automation_events:
        print(f"\n  Self-Healing Events:")
        for event in automation_events[-5:]:  # Show last 5 events
            status = "✓" if event["success"] else "✗"
            print(f"    {status} {event['rule']} ({event['phase']})")

    print("\n✅ Exercise 3 completed successfully!")
    print("\nMaintenance automation and self-healing demonstrated:")
    print("  • Automated maintenance workflows with scheduling")
    print("  • Self-healing system with condition-based triggers")
    print("  • Comprehensive operational monitoring")
    print("  • Security event detection and analysis")
    print("  • Performance baseline calculation and anomaly detection")
    print("  • Backup and recovery management")

    # Cleanup
    automation.stop_automation()
    ops_manager.stop_operations()

    return report


if __name__ == "__main__":
    result = asyncio.run(exercise_03_solution())
    print(f"\nExercise Result Summary:")
    print(
        f"Maintenance Operations: {result['maintenance_summary']['operations_completed']} completed"
    )
    print(
        f"Automation Executions: {result['automation_summary']['total_executions']} total"
    )
    print(
        f"Self-Healing Events: {result['automation_summary']['self_healing_events']} triggered"
    )
    print(f"Security Events: {result['security_summary']['total_events']} detected")
    print(
        f"Performance Anomalies: {result['performance_summary']['anomalies_detected']} found"
    )
