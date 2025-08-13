#!/usr/bin/env python3
"""
Exercise 1: Production Deployment Setup

This exercise demonstrates setting up a complete production deployment
environment for DSPy applications with monitoring, scaling, and maintenance.

Learning Objectives:
- Set up production deployment infrastructure
- Configure monitoring and alerting
- Implement auto-scaling policies
- Set up maintenance workflows

Author: DSPy Learning Framework
"""

import asyncio
import json

# Import our production deployment modules
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append("..")

from maintenance_operations import MaintenanceOperationsManager
from monitoring_system import MonitoringSystem
from scaling_config import ConfigurationManager, DeploymentEnvironment
from scaling_strategies import LoadBalancingStrategy, ScalingOrchestrator, WorkerNode


async def exercise_01_solution():
    """
    Exercise 1 Solution: Complete Production Deployment Setup

    This solution demonstrates:
    1. Setting up a production deployment configuration
    2. Configuring monitoring and alerting
    3. Implementing auto-scaling
    4. Setting up maintenance workflows
    """

    print("=== Exercise 1: Production Deployment Setup ===")
    print("Setting up complete production deployment environment...")

    # Step 1: Create production configuration
    print("\n1. Creating production configuration...")
    config_manager = ConfigurationManager()

    # Create production configuration
    prod_config = config_manager.create_default_config(
        DeploymentEnvironment.PRODUCTION, "dspy-production-service"
    )

    # Customize for production requirements
    prod_config.auto_scaling.min_instances = 3
    prod_config.auto_scaling.max_instances = 20
    prod_config.auto_scaling.target_cpu_utilization = 60.0
    prod_config.auto_scaling.predictive_scaling = True

    # Save configuration
    config_file = config_manager.save_config(prod_config, "production_config.yaml")
    print(f"✓ Production configuration saved: {config_file}")

    # Step 2: Set up monitoring system
    print("\n2. Setting up monitoring system...")
    monitoring = MonitoringSystem()

    # Configure Slack notifications (example)
    # monitoring.setup_slack_notifications("https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK")

    # Start monitoring in background
    monitoring_task = asyncio.create_task(
        monitoring.start_monitoring(
            system_metrics_interval=30,
            alert_evaluation_interval=60,
            enable_prometheus=True,
            enable_dashboard=True,
        )
    )

    print("✓ Monitoring system started")
    print("  - Prometheus metrics: http://localhost:9090/metrics")
    print("  - Dashboard: http://localhost:8080")

    # Step 3: Set up scaling orchestrator
    print("\n3. Setting up auto-scaling...")
    orchestrator = ScalingOrchestrator()

    # Add production worker nodes
    for i in range(prod_config.auto_scaling.min_instances):
        node = WorkerNode(
            node_id=f"prod-worker-{i+1}",
            host="10.0.1.{}".format(10 + i),
            port=8000,
            weight=1.0,
            healthy=True,
        )
        orchestrator.load_balancer.add_node(node)

    # Configure intelligent load balancing
    orchestrator.load_balancer.strategy = LoadBalancingStrategy.INTELLIGENT

    # Start orchestration
    orchestration_task = asyncio.create_task(orchestrator.start_orchestration())
    print("✓ Auto-scaling orchestrator started")

    # Step 4: Set up maintenance and operations
    print("\n4. Setting up maintenance and operations...")
    ops_manager = MaintenanceOperationsManager()

    # Start operations management
    ops_task = asyncio.create_task(ops_manager.start_operations())
    print("✓ Maintenance and operations started")

    # Step 5: Simulate production workload
    print("\n5. Simulating production workload...")

    # Record some production metrics
    for i in range(10):
        # Simulate DSPy requests
        monitoring.record_dspy_request(
            duration=0.3 + (i * 0.05),
            success=True,
            tokens_used=150 + (i * 10),
            model="gpt-4o",
            cached=i % 3 == 0,
        )

        await asyncio.sleep(0.5)

    print("✓ Production workload simulation completed")

    # Step 6: Generate deployment templates
    print("\n6. Generating deployment templates...")
    templates = config_manager.generate_deployment_templates(prod_config)

    # Save templates
    templates_dir = Path("deployment_templates")
    templates_dir.mkdir(exist_ok=True)

    for template_type, template_content in templates.items():
        template_file = templates_dir / f"{template_type}_deployment.yaml"
        with open(template_file, "w") as f:
            f.write(template_content)
        print(f"✓ {template_type.title()} template saved: {template_file}")

    # Step 7: Show system status
    print("\n7. System Status Summary:")

    # Monitoring status
    monitoring_status = monitoring.get_monitoring_status()
    print(f"  Monitoring:")
    print(f"    - Metrics collected: {monitoring_status['metrics_count']}")
    print(f"    - Alerts configured: {monitoring_status['alerts_configured']}")
    print(f"    - System CPU: {monitoring_status['system_status']['cpu_percent']:.1f}%")
    print(
        f"    - System Memory: {monitoring_status['system_status']['memory_percent']:.1f}%"
    )

    # Scaling status
    scaling_status = orchestrator.get_system_status()
    print(f"  Auto-scaling:")
    print(
        f"    - Current capacity: {scaling_status['auto_scaling']['current_capacity']}"
    )
    print(f"    - Healthy nodes: {scaling_status['load_balancing']['healthy_nodes']}")
    print(
        f"    - Load balancing: {scaling_status['load_balancing']['load_balancing_strategy']}"
    )

    # Operations status
    ops_status = ops_manager.get_system_status()
    print(f"  Operations:")
    print(
        f"    - Health monitoring: {'✓' if ops_status['health_monitoring']['running'] else '✗'}"
    )
    print(
        f"    - Backup management: {ops_status['backup_management']['total_backups']} backups"
    )
    print(
        f"    - Maintenance scheduling: {'✓' if ops_status['maintenance_scheduling']['scheduled_operations'] >= 0 else '✗'}"
    )

    print("\n✅ Exercise 1 completed successfully!")
    print("\nProduction deployment environment is now fully configured with:")
    print("  • Auto-scaling with intelligent load balancing")
    print("  • Comprehensive monitoring and alerting")
    print("  • Maintenance and operations management")
    print("  • Deployment templates for multiple platforms")
    print("  • Health monitoring and self-healing capabilities")

    # Cleanup
    await asyncio.sleep(2)
    monitoring.stop_monitoring()
    orchestrator.stop_orchestration()
    ops_manager.stop_operations()

    return {
        "status": "completed",
        "configuration": prod_config.service_name,
        "monitoring_metrics": monitoring_status["metrics_count"],
        "scaling_capacity": scaling_status["auto_scaling"]["current_capacity"],
        "templates_generated": len(templates),
    }


if __name__ == "__main__":
    result = asyncio.run(exercise_01_solution())
    print(f"\nExercise Result: {json.dumps(result, indent=2)}")
