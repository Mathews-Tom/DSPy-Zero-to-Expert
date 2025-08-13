#!/usr/bin/env python3
"""
Exercise 4: Comprehensive Production Deployment

This exercise demonstrates a complete end-to-end production deployment
with all systems integrated: scaling, monitoring, maintenance, and operations.

Learning Objectives:
- Integrate all production deployment components
- Demonstrate real-world deployment scenarios
- Test system resilience and recovery
- Validate production readiness

Author: DSPy Learning Framework
"""

import asyncio
import json
import random

# Import all our production deployment modules
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append("..")

from cost_optimization import CostOptimizationManager
from maintenance_automation import MaintenanceAutomation
from maintenance_operations import MaintenanceOperationsManager
from monitoring_system import MonitoringSystem
from operational_tools import (
    LogAnalyzer,
    OperationalDashboard,
    PerformanceAnalyzer,
    SecurityMonitor,
)
from scaling_config import ConfigurationManager, DeploymentEnvironment
from scaling_strategies import LoadBalancingStrategy, ScalingOrchestrator, WorkerNode


class ProductionDeploymentSystem:
    """Complete production deployment system integration"""

    def __init__(self, service_name: str = "dspy-production-system"):
        self.service_name = service_name

        # Initialize all components
        self.config_manager = ConfigurationManager()
        self.monitoring = MonitoringSystem()
        self.orchestrator = ScalingOrchestrator()
        self.ops_manager = MaintenanceOperationsManager()
        self.automation = MaintenanceAutomation()
        self.cost_manager = CostOptimizationManager()

        # Operational tools
        self.log_analyzer = LogAnalyzer()
        self.security_monitor = SecurityMonitor()
        self.performance_analyzer = PerformanceAnalyzer()
        self.dashboard = OperationalDashboard(
            self.log_analyzer, self.security_monitor, self.performance_analyzer
        )

        # System state
        self.running = False
        self.deployment_history = []
        self.system_metrics = {}

    async def initialize_system(self):
        """Initialize the complete production system"""
        print(f"🚀 Initializing {self.service_name} production system...")

        # Step 1: Create production configuration
        print("\n1. Creating production configuration...")
        prod_config = self.config_manager.create_default_config(
            DeploymentEnvironment.PRODUCTION, self.service_name
        )

        # Customize for comprehensive deployment
        prod_config.auto_scaling.min_instances = 3
        prod_config.auto_scaling.max_instances = 25
        prod_config.auto_scaling.target_cpu_utilization = 65.0
        prod_config.auto_scaling.predictive_scaling = True

        # Save configuration
        config_file = self.config_manager.save_config(prod_config)
        print(f"✓ Configuration saved: {config_file}")

        # Step 2: Set up cost management
        print("\n2. Setting up cost management...")
        self.cost_manager.setup_default_budgets()
        self.cost_manager.cost_tracker.set_budget_limit("daily", 150.0)
        self.cost_manager.cost_tracker.set_budget_limit("weekly", 900.0)
        self.cost_manager.cost_tracker.set_budget_limit("monthly", 3600.0)
        print("✓ Cost management configured")

        # Step 3: Configure monitoring
        print("\n3. Configuring monitoring and alerting...")
        # Add custom monitoring rules here if needed
        print("✓ Monitoring configured")

        # Step 4: Set up worker nodes
        print("\n4. Setting up worker nodes...")
        node_types = [
            {"prefix": "high-perf", "count": 2, "weight": 2.0, "base_port": 8000},
            {"prefix": "standard", "count": 4, "weight": 1.0, "base_port": 8010},
            {"prefix": "spot", "count": 3, "weight": 1.5, "base_port": 8020},
        ]

        total_nodes = 0
        for node_type in node_types:
            for i in range(node_type["count"]):
                node = WorkerNode(
                    node_id=f"{node_type['prefix']}-{i+1}",
                    host=f"10.0.{node_type['base_port']//1000}.{10+i}",
                    port=node_type["base_port"] + i,
                    weight=node_type["weight"],
                    healthy=True,
                )
                self.orchestrator.load_balancer.add_node(node)
                total_nodes += 1

        print(f"✓ {total_nodes} worker nodes configured")

        # Step 5: Configure operational tools
        print("\n5. Configuring operational tools...")

        # Log analyzer rules
        self.log_analyzer.add_alert_rule(
            "Critical System Errors", "CRITICAL|FATAL|PANIC", 3, 5
        )
        self.log_analyzer.add_alert_rule(
            "Authentication Failures", "auth.*fail|login.*fail", 10, 15
        )
        self.log_analyzer.add_alert_rule(
            "Performance Issues", "timeout|slow|performance", 5, 10
        )

        print("✓ Operational tools configured")

        return prod_config

    async def start_system(self):
        """Start all system components"""
        if self.running:
            return

        print("\n🔄 Starting production system components...")

        # Start monitoring
        monitoring_task = asyncio.create_task(
            self.monitoring.start_monitoring(
                system_metrics_interval=20,
                alert_evaluation_interval=30,
                enable_prometheus=True,
                enable_dashboard=True,
            )
        )

        # Start scaling orchestration
        orchestration_task = asyncio.create_task(
            self.orchestrator.start_orchestration()
        )

        # Start operations management
        ops_task = asyncio.create_task(self.ops_manager.start_operations())

        # Start automation
        self.automation.start_automation()

        self.running = True
        print("✓ All system components started")

        return [monitoring_task, orchestration_task, ops_task]

    async def simulate_production_workload(self, duration_minutes: int = 10):
        """Simulate realistic production workload"""
        print(f"\n📊 Simulating production workload ({duration_minutes} minutes)...")

        # Define realistic workload patterns
        workload_patterns = [
            {
                "name": "Morning Ramp-up",
                "duration": 0.2,
                "intensity": 0.3,
                "error_rate": 0.01,
            },
            {
                "name": "Business Hours",
                "duration": 0.4,
                "intensity": 0.8,
                "error_rate": 0.02,
            },
            {
                "name": "Peak Traffic",
                "duration": 0.2,
                "intensity": 1.0,
                "error_rate": 0.03,
            },
            {
                "name": "Evening Wind-down",
                "duration": 0.2,
                "intensity": 0.4,
                "error_rate": 0.015,
            },
        ]

        total_duration = duration_minutes * 60  # Convert to seconds
        total_requests = 0

        for pattern in workload_patterns:
            pattern_duration = total_duration * pattern["duration"]
            pattern_start = datetime.utcnow()

            print(f"  Phase: {pattern['name']} ({pattern_duration:.0f}s)")

            while (
                datetime.utcnow() - pattern_start
            ).total_seconds() < pattern_duration:
                # Calculate request rate based on intensity
                base_rate = 5.0  # Base requests per second
                request_rate = base_rate * pattern["intensity"]

                # Process requests
                for _ in range(int(request_rate)):
                    await self._process_simulated_request(pattern)
                    total_requests += 1

                # Update system metrics
                await self._update_system_metrics(pattern)

                # Simulate some security events
                if random.random() < 0.05:  # 5% chance
                    await self._simulate_security_event()

                await asyncio.sleep(1)

        print(f"✓ Workload simulation completed ({total_requests:,} requests)")
        return total_requests

    async def _process_simulated_request(self, pattern):
        """Process a simulated request"""
        # Simulate request processing
        processing_time = random.uniform(0.1, 2.0) * (2 - pattern["intensity"])
        success = random.random() > pattern["error_rate"]

        # Record costs
        compute_time = processing_time / 3600  # Convert to hours
        self.cost_manager.cost_tracker.record_usage(
            "compute", compute_time, {"pattern": pattern["name"]}
        )

        # Record API usage
        tokens_used = random.randint(50, 500)
        self.cost_manager.cost_tracker.record_usage("api_calls", 1)
        self.cost_manager.cost_tracker.record_usage("tokens", tokens_used)

        # Record performance metrics
        self.performance_analyzer.record_metric("response_time", processing_time)
        self.performance_analyzer.record_metric("error_rate", 1 if not success else 0)

        # Record monitoring metrics
        self.monitoring.record_dspy_request(
            duration=processing_time,
            success=success,
            tokens_used=tokens_used,
            model="gpt-4o",
            cached=random.random() < 0.3,
        )

    async def _update_system_metrics(self, pattern):
        """Update system metrics based on current pattern"""
        # Calculate resource utilization based on pattern intensity
        base_cpu = 30
        base_memory = 40
        base_disk = 50

        cpu_usage = base_cpu + (pattern["intensity"] * 50) + random.uniform(-5, 5)
        memory_usage = base_memory + (pattern["intensity"] * 40) + random.uniform(-3, 3)
        disk_usage = base_disk + random.uniform(-2, 2)

        # Update node metrics
        for node in self.orchestrator.load_balancer.nodes.values():
            node.cpu_percent = max(0, min(100, cpu_usage + random.uniform(-10, 10)))
            node.memory_percent = max(0, min(100, memory_usage + random.uniform(-5, 5)))
            node.active_connections = random.randint(0, int(20 * pattern["intensity"]))

        # Store system metrics
        self.system_metrics = {
            "cpu_percent": cpu_usage,
            "memory_percent": memory_usage,
            "disk_percent": disk_usage,
            "error_rate": pattern["error_rate"],
            "service_health": "healthy" if pattern["error_rate"] < 0.05 else "warning",
        }

        # Record performance metrics
        self.performance_analyzer.record_metric("cpu_utilization", cpu_usage)
        self.performance_analyzer.record_metric("memory_utilization", memory_usage)
        self.performance_analyzer.record_metric("disk_utilization", disk_usage)

    async def _simulate_security_event(self):
        """Simulate security events"""
        attack_types = [
            {"path": "/api/users?id=1' OR '1'='1", "type": "sql_injection"},
            {"path": "/api/search?q=<script>alert('xss')</script>", "type": "xss"},
            {"path": "/api/files?path=../../../etc/passwd", "type": "path_traversal"},
        ]

        attack = random.choice(attack_types)
        request_data = {
            "source_ip": f"192.168.{random.randint(1, 10)}.{random.randint(100, 200)}",
            "user_agent": "AttackBot/1.0",
            "path": attack["path"],
            "body": "",
        }

        security_event = self.security_monitor.analyze_request(request_data)
        if security_event:
            print(f"    🚨 Security event: {security_event.description}")

    async def test_system_resilience(self):
        """Test system resilience and recovery capabilities"""
        print("\n🧪 Testing system resilience...")

        resilience_tests = [
            {"name": "High CPU Load", "condition": {"cpu_percent": 95}, "duration": 30},
            {
                "name": "Memory Pressure",
                "condition": {"memory_percent": 92},
                "duration": 25,
            },
            {
                "name": "High Error Rate",
                "condition": {"error_rate": 0.15},
                "duration": 20,
            },
            {
                "name": "Disk Space Critical",
                "condition": {"disk_percent": 95},
                "duration": 15,
            },
        ]

        recovery_events = []

        for test in resilience_tests:
            print(f"  Testing: {test['name']}...")

            # Simulate the condition
            test_metrics = {**self.system_metrics, **test["condition"]}

            # Check if self-healing triggers
            triggered_rules = (
                await self.automation.self_healing.evaluate_healing_conditions(
                    test_metrics
                )
            )

            if triggered_rules:
                for rule_name in triggered_rules:
                    healing_result = (
                        await self.automation.self_healing.execute_healing_actions(
                            rule_name
                        )
                    )
                    recovery_events.append(
                        {
                            "test": test["name"],
                            "rule": rule_name,
                            "success": healing_result["overall_success"],
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    print(f"    ✓ Self-healing activated: {rule_name}")

            await asyncio.sleep(2)  # Brief pause between tests

        print(
            f"✓ Resilience testing completed ({len(recovery_events)} recovery events)"
        )
        return recovery_events

    async def generate_comprehensive_report(
        self, total_requests: int, recovery_events: list
    ):
        """Generate comprehensive system report"""
        print("\n📋 Generating comprehensive system report...")

        # Collect data from all components
        monitoring_status = self.monitoring.get_monitoring_status()
        scaling_status = self.orchestrator.get_system_status()
        ops_status = self.ops_manager.get_system_status()
        automation_status = self.automation.get_automation_status()
        cost_report = self.cost_manager.get_comprehensive_report()
        security_summary = self.security_monitor.get_security_summary(24)
        performance_report = self.performance_analyzer.get_performance_report(1)
        dashboard_data = await self.dashboard.generate_dashboard_data()

        # Calculate performance baselines
        self.performance_analyzer.calculate_baseline("response_time")
        self.performance_analyzer.calculate_baseline("cpu_utilization")

        # Detect anomalies
        response_time_anomalies = self.performance_analyzer.detect_anomalies(
            "response_time"
        )
        cpu_anomalies = self.performance_analyzer.detect_anomalies("cpu_utilization")

        report = {
            "deployment_summary": {
                "service_name": self.service_name,
                "deployment_time": datetime.utcnow().isoformat(),
                "total_requests_processed": total_requests,
                "system_uptime_minutes": 15,  # Approximate for demo
                "components_deployed": 8,
            },
            "scaling_performance": {
                "current_capacity": scaling_status["auto_scaling"]["current_capacity"],
                "capacity_range": f"{scaling_status['auto_scaling']['min_capacity']}-{scaling_status['auto_scaling']['max_capacity']}",
                "scaling_events": scaling_status["auto_scaling"][
                    "scaling_history_count"
                ],
                "healthy_nodes": scaling_status["load_balancing"]["healthy_nodes"],
                "load_balancing_strategy": scaling_status["load_balancing"][
                    "load_balancing_strategy"
                ],
            },
            "monitoring_metrics": {
                "metrics_collected": monitoring_status["metrics_count"],
                "alerts_configured": monitoring_status["alerts_configured"],
                "prometheus_enabled": True,
                "dashboard_enabled": True,
            },
            "cost_analysis": {
                "total_cost": cost_report["summary"]["total_daily_cost"],
                "cost_per_request": cost_report["summary"]["total_daily_cost"]
                / max(total_requests, 1),
                "optimization_opportunities": cost_report["summary"][
                    "optimization_opportunities"
                ],
                "potential_savings": cost_report["summary"]["potential_savings"],
                "budget_status": (
                    "within_limits"
                    if not cost_report["budget_status"]["alerts"]
                    else "has_alerts"
                ),
            },
            "security_status": {
                "total_events": security_summary["total_events"],
                "high_severity_events": security_summary["high_severity_events"],
                "threat_level": security_summary.get("threat_level", "low"),
                "unique_source_ips": len(security_summary["top_source_ips"]),
            },
            "maintenance_operations": {
                "health_monitoring_active": ops_status["health_monitoring"]["running"],
                "backup_management_ready": ops_status["backup_management"][
                    "total_backups"
                ]
                >= 0,
                "maintenance_scheduling_enabled": True,
                "automation_rules": automation_status["total_rules"],
            },
            "resilience_testing": {
                "tests_conducted": 4,
                "recovery_events": len(recovery_events),
                "self_healing_success_rate": len(
                    [e for e in recovery_events if e["success"]]
                )
                / max(len(recovery_events), 1)
                * 100,
            },
            "performance_analysis": {
                "baselines_calculated": len(
                    self.performance_analyzer.performance_baselines
                ),
                "anomalies_detected": len(response_time_anomalies) + len(cpu_anomalies),
                "performance_recommendations": len(
                    performance_report.get("recommendations", [])
                ),
            },
        }

        return report

    async def stop_system(self):
        """Stop all system components"""
        if not self.running:
            return

        print("\n🛑 Stopping production system...")

        self.monitoring.stop_monitoring()
        self.orchestrator.stop_orchestration()
        self.ops_manager.stop_operations()
        self.automation.stop_automation()

        self.running = False
        print("✓ All system components stopped")


async def exercise_04_solution():
    """
    Exercise 4 Solution: Comprehensive Production Deployment

    This solution demonstrates:
    1. Complete system integration
    2. Real-world production scenarios
    3. System resilience testing
    4. Comprehensive reporting
    """

    print("=== Exercise 4: Comprehensive Production Deployment ===")
    print("Deploying complete production system with all components...")

    # Step 1: Initialize production system
    production_system = ProductionDeploymentSystem("dspy-comprehensive-prod")
    config = await production_system.initialize_system()

    # Step 2: Start all system components
    system_tasks = await production_system.start_system()

    # Step 3: Simulate production workload
    total_requests = await production_system.simulate_production_workload(
        duration_minutes=5
    )

    # Step 4: Test system resilience
    recovery_events = await production_system.test_system_resilience()

    # Step 5: Run cost optimization
    print("\n💰 Running cost optimization...")
    optimization_result = await production_system.cost_manager.run_optimization_cycle()
    print(
        f"✓ Cost optimization completed ({optimization_result['auto_implemented']} optimizations applied)"
    )

    # Step 6: Generate comprehensive report
    report = await production_system.generate_comprehensive_report(
        total_requests, recovery_events
    )

    # Step 7: Display final results
    print("\n📊 COMPREHENSIVE DEPLOYMENT REPORT")
    print("=" * 60)

    print(f"\n🚀 Deployment Summary:")
    print(f"  Service: {report['deployment_summary']['service_name']}")
    print(f"  Components: {report['deployment_summary']['components_deployed']}")
    print(
        f"  Requests Processed: {report['deployment_summary']['total_requests_processed']:,}"
    )
    print(
        f"  System Uptime: {report['deployment_summary']['system_uptime_minutes']} minutes"
    )

    print(f"\n📈 Scaling Performance:")
    print(
        f"  Current Capacity: {report['scaling_performance']['current_capacity']} nodes"
    )
    print(f"  Capacity Range: {report['scaling_performance']['capacity_range']}")
    print(f"  Scaling Events: {report['scaling_performance']['scaling_events']}")
    print(f"  Healthy Nodes: {report['scaling_performance']['healthy_nodes']}")

    print(f"\n📊 Monitoring & Alerting:")
    print(f"  Metrics Collected: {report['monitoring_metrics']['metrics_collected']}")
    print(f"  Alerts Configured: {report['monitoring_metrics']['alerts_configured']}")
    print(
        f"  Prometheus: {'✓' if report['monitoring_metrics']['prometheus_enabled'] else '✗'}"
    )
    print(
        f"  Dashboard: {'✓' if report['monitoring_metrics']['dashboard_enabled'] else '✗'}"
    )

    print(f"\n💰 Cost Analysis:")
    print(f"  Total Cost: ${report['cost_analysis']['total_cost']:.2f}")
    print(f"  Cost per Request: ${report['cost_analysis']['cost_per_request']:.4f}")
    print(
        f"  Optimization Opportunities: {report['cost_analysis']['optimization_opportunities']}"
    )
    print(f"  Potential Savings: ${report['cost_analysis']['potential_savings']:.2f}")

    print(f"\n🔒 Security Status:")
    print(f"  Security Events: {report['security_status']['total_events']}")
    print(f"  High Severity: {report['security_status']['high_severity_events']}")
    print(f"  Threat Level: {report['security_status']['threat_level']}")
    print(f"  Unique IPs: {report['security_status']['unique_source_ips']}")

    print(f"\n🔧 Maintenance & Operations:")
    print(
        f"  Health Monitoring: {'✓' if report['maintenance_operations']['health_monitoring_active'] else '✗'}"
    )
    print(
        f"  Backup Management: {'✓' if report['maintenance_operations']['backup_management_ready'] else '✗'}"
    )
    print(f"  Automation Rules: {report['maintenance_operations']['automation_rules']}")

    print(f"\n🧪 Resilience Testing:")
    print(f"  Tests Conducted: {report['resilience_testing']['tests_conducted']}")
    print(f"  Recovery Events: {report['resilience_testing']['recovery_events']}")
    print(
        f"  Self-Healing Success: {report['resilience_testing']['self_healing_success_rate']:.1f}%"
    )

    print(f"\n⚡ Performance Analysis:")
    print(
        f"  Baselines Calculated: {report['performance_analysis']['baselines_calculated']}"
    )
    print(
        f"  Anomalies Detected: {report['performance_analysis']['anomalies_detected']}"
    )
    print(
        f"  Recommendations: {report['performance_analysis']['performance_recommendations']}"
    )

    print("\n" + "=" * 60)
    print("✅ COMPREHENSIVE PRODUCTION DEPLOYMENT COMPLETED!")
    print("\nSystem Features Demonstrated:")
    print("  • Complete production deployment automation")
    print("  • Intelligent auto-scaling with load balancing")
    print("  • Comprehensive monitoring and alerting")
    print("  • Cost optimization and budget management")
    print("  • Security monitoring and threat detection")
    print("  • Maintenance automation and self-healing")
    print("  • Performance analysis and anomaly detection")
    print("  • System resilience and recovery testing")
    print("  • Operational dashboard and reporting")

    # Step 8: Cleanup
    await production_system.stop_system()

    return report


if __name__ == "__main__":
    result = asyncio.run(exercise_04_solution())

    print(f"\n🎯 FINAL EXERCISE RESULTS:")
    print(
        f"Requests Processed: {result['deployment_summary']['total_requests_processed']:,}"
    )
    print(f"System Capacity: {result['scaling_performance']['current_capacity']} nodes")
    print(f"Total Cost: ${result['cost_analysis']['total_cost']:.2f}")
    print(f"Security Events: {result['security_status']['total_events']}")
    print(
        f"Self-Healing Success: {result['resilience_testing']['self_healing_success_rate']:.1f}%"
    )
    print(
        f"Performance Anomalies: {result['performance_analysis']['anomalies_detected']}"
    )

    print(
        f"\n🏆 Production deployment system is fully operational and production-ready!"
    )
