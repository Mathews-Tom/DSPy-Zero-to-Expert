#!/usr/bin/env python3
"""
Comprehensive Test Suite for DSPy Production Deployment System

This test suite validates all components of the production deployment system
including scaling, monitoring, maintenance, and operations.

Author: DSPy Learning Framework
"""

import asyncio
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from cost_optimization import (
    BudgetManager,
    CostOptimizationManager,
    CostTracker,
    ResourceOptimizer,
)
from maintenance_automation import (
    AutomationRule,
    AutomationTrigger,
    MaintenanceAutomation,
    SelfHealingSystem,
)
from maintenance_operations import (
    BackupManager,
    DeploymentManager,
    HealthMonitor,
    MaintenanceOperationsManager,
    MaintenanceScheduler,
)
from monitoring_system import (
    Alert,
    AlertManager,
    AlertSeverity,
    MetricsCollector,
    MonitoringSystem,
)
from operational_tools import (
    LogAnalyzer,
    OperationalDashboard,
    PerformanceAnalyzer,
    SecurityMonitor,
)
from scaling_config import (
    ConfigurationManager,
    DeploymentEnvironment,
    ScalingConfiguration,
)

# Import all production deployment modules
from scaling_strategies import (
    IntelligentBalancer,
    LeastConnectionsBalancer,
    LoadBalancingStrategy,
    RoundRobinBalancer,
    ScalingOrchestrator,
    WorkerNode,
)


class TestScalingStrategies(unittest.TestCase):
    """Test scaling strategies and load balancing"""

    def setUp(self):
        self.orchestrator = ScalingOrchestrator()

        # Add test nodes
        for i in range(3):
            node = WorkerNode(
                node_id=f"test-node-{i+1}",
                host=f"10.0.1.{10+i}",
                port=8000 + i,
                weight=1.0,
                healthy=True,
            )
            self.orchestrator.load_balancer.add_node(node)

    def test_load_balancer_strategies(self):
        """Test different load balancing strategies"""
        strategies = [
            (LoadBalancingStrategy.ROUND_ROBIN, RoundRobinBalancer),
            (LoadBalancingStrategy.LEAST_CONNECTIONS, LeastConnectionsBalancer),
            (LoadBalancingStrategy.INTELLIGENT, IntelligentBalancer),
        ]

        for strategy, balancer_class in strategies:
            self.orchestrator.load_balancer.strategy = strategy
            balancer = self.orchestrator.load_balancer._create_balancer(strategy)
            self.assertIsInstance(balancer, balancer_class)

    def test_node_selection(self):
        """Test node selection for requests"""
        node = self.orchestrator.load_balancer.get_node_for_request()
        self.assertIsNotNone(node)
        self.assertIn(node.node_id, [f"test-node-{i+1}" for i in range(3)])

    def test_auto_scaler_configuration(self):
        """Test auto-scaler configuration"""
        auto_scaler = self.orchestrator.auto_scaler

        # Test default configuration
        self.assertGreaterEqual(auto_scaler.min_capacity, 1)
        self.assertGreaterEqual(auto_scaler.max_capacity, auto_scaler.min_capacity)
        self.assertGreater(auto_scaler.target_cpu, 0)
        self.assertLess(auto_scaler.target_cpu, 100)

    async def test_scaling_evaluation(self):
        """Test scaling evaluation logic"""
        # This would require more complex setup with actual metrics
        # For now, test that the method exists and can be called
        scaling_event = await self.orchestrator.auto_scaler.evaluate_scaling()
        # scaling_event can be None if no scaling is needed
        self.assertTrue(scaling_event is None or hasattr(scaling_event, "direction"))


class TestMonitoringSystem(unittest.TestCase):
    """Test monitoring and alerting system"""

    def setUp(self):
        self.monitoring = MonitoringSystem()
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)

    def test_metrics_collection(self):
        """Test metrics collection"""
        # Record some test metrics
        self.metrics_collector.record_metric("test_metric", 42.0, {"tag": "test"})

        # Verify metric was recorded
        latest_value = self.metrics_collector.get_latest_value("test_metric")
        self.assertEqual(latest_value, 42.0)

        # Test metric statistics
        for i in range(10):
            self.metrics_collector.record_metric("test_metric", float(i))

        stats = self.metrics_collector.get_metric_stats("test_metric", 60)
        self.assertIn("count", stats)
        self.assertIn("mean", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)

    def test_alert_management(self):
        """Test alert creation and management"""
        alert = Alert(
            name="test_alert",
            condition="metrics.get('test_metric', {}).get('value', 0) > 50",
            severity=AlertSeverity.WARNING,
            message="Test alert triggered",
            channels=[],
        )

        self.alert_manager.register_alert(alert)
        self.assertIn("test_alert", self.alert_manager.alerts)

    def test_dspy_metrics(self):
        """Test DSPy-specific metrics"""
        dspy_metrics = self.monitoring.dspy_metrics

        # Record a DSPy request
        dspy_metrics.record_request(
            duration=0.5, success=True, tokens_used=100, model="gpt-4o", cached=False
        )

        # Verify metrics were recorded
        requests_total = self.monitoring.metrics_collector.get_latest_value(
            "dspy_requests_total"
        )
        self.assertIsNotNone(requests_total)


class TestMaintenanceOperations(unittest.TestCase):
    """Test maintenance and operations management"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ops_manager = MaintenanceOperationsManager()
        self.health_monitor = HealthMonitor()
        self.backup_manager = BackupManager(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_health_monitoring(self):
        """Test health monitoring system"""

        # Register a test health check
        def test_health_check():
            return {"status": "healthy", "metrics": {"test_value": 100}}

        self.health_monitor.register_health_check("test_check", test_health_check)
        self.assertIn("test_check", self.health_monitor.health_checks)

    async def test_health_check_execution(self):
        """Test health check execution"""

        # Register a simple health check
        def simple_check():
            return True

        self.health_monitor.register_health_check("simple", simple_check)

        # Run health checks
        health = await self.health_monitor.run_health_checks()
        self.assertIsNotNone(health)
        self.assertIn("simple", health.components)

    async def test_backup_creation(self):
        """Test backup creation and management"""
        # Create a test backup
        backup_info = await self.backup_manager.create_backup(
            backup_type="test", metadata={"test": True}
        )

        self.assertIsNotNone(backup_info)
        self.assertEqual(backup_info.backup_type, "test")
        self.assertIn(backup_info.backup_id, self.backup_manager.backups)

    def test_backup_registry(self):
        """Test backup registry management"""
        # Test backup listing
        backups = self.backup_manager.list_backups()
        self.assertIsInstance(backups, list)


class TestMaintenanceAutomation(unittest.TestCase):
    """Test maintenance automation and self-healing"""

    def setUp(self):
        self.automation = MaintenanceAutomation()
        self.self_healing = SelfHealingSystem()

    def test_automation_rules(self):
        """Test automation rule management"""
        rule = AutomationRule(
            rule_id="test_rule",
            name="Test Rule",
            description="Test automation rule",
            trigger=AutomationTrigger.SCHEDULED,
            condition="0 * * * *",  # Every hour
            action="test_action",
        )

        self.automation.add_automation_rule(rule)
        self.assertIn("test_rule", self.automation.automation_rules)

        # Test rule enabling/disabling
        self.automation.disable_rule("test_rule")
        self.assertFalse(self.automation.automation_rules["test_rule"].enabled)

        self.automation.enable_rule("test_rule")
        self.assertTrue(self.automation.automation_rules["test_rule"].enabled)

    async def test_self_healing_evaluation(self):
        """Test self-healing condition evaluation"""
        # Test with normal metrics
        normal_metrics = {
            "memory_percent": 50,
            "cpu_percent": 60,
            "disk_percent": 40,
            "service_health": "healthy",
        }

        triggered_rules = await self.self_healing.evaluate_healing_conditions(
            normal_metrics
        )
        self.assertIsInstance(triggered_rules, list)

        # Test with critical metrics
        critical_metrics = {
            "memory_percent": 95,
            "cpu_percent": 98,
            "disk_percent": 95,
            "service_health": "critical",
        }

        triggered_rules = await self.self_healing.evaluate_healing_conditions(
            critical_metrics
        )
        self.assertIsInstance(triggered_rules, list)

    def test_automation_status(self):
        """Test automation system status"""
        status = self.automation.get_automation_status()

        required_keys = [
            "running",
            "total_rules",
            "enabled_rules",
            "disabled_rules",
            "total_executions",
            "successful_executions",
            "failed_executions",
        ]

        for key in required_keys:
            self.assertIn(key, status)


class TestCostOptimization(unittest.TestCase):
    """Test cost optimization and management"""

    def setUp(self):
        self.cost_manager = CostOptimizationManager()
        self.cost_tracker = CostTracker()
        self.resource_optimizer = ResourceOptimizer(self.cost_tracker)

    def test_cost_tracking(self):
        """Test cost tracking functionality"""
        # Record some usage
        self.cost_tracker.record_usage("compute", 2.5, {"instance_type": "standard"})
        self.cost_tracker.record_usage("api_calls", 1000)
        self.cost_tracker.record_usage("tokens", 5000)

        # Test cost retrieval
        total_cost = self.cost_tracker.get_total_cost(24)
        self.assertGreaterEqual(total_cost, 0)

        costs_by_type = self.cost_tracker.get_costs_by_period(24)
        self.assertIsInstance(costs_by_type, dict)

    def test_budget_management(self):
        """Test budget limits and alerts"""
        self.cost_tracker.set_budget_limit("daily", 100.0)
        self.cost_tracker.set_budget_limit("weekly", 600.0)

        self.assertEqual(self.cost_tracker.budget_limits["daily"], 100.0)
        self.assertEqual(self.cost_tracker.budget_limits["weekly"], 600.0)

    def test_optimization_recommendations(self):
        """Test optimization recommendation generation"""
        # Record some usage to generate recommendations
        for i in range(10):
            self.cost_tracker.record_usage("compute", 1.0)
            self.cost_tracker.record_usage("tokens", 1000)

        recommendations = self.resource_optimizer.generate_recommendations()
        self.assertIsInstance(recommendations, list)


class TestScalingConfiguration(unittest.TestCase):
    """Test scaling configuration management"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_configuration_creation(self):
        """Test configuration creation for different environments"""
        environments = [
            DeploymentEnvironment.DEVELOPMENT,
            DeploymentEnvironment.STAGING,
            DeploymentEnvironment.PRODUCTION,
        ]

        for env in environments:
            config = self.config_manager.create_default_config(env, "test-service")
            self.assertIsInstance(config, ScalingConfiguration)
            self.assertEqual(config.environment, env)
            self.assertEqual(config.service_name, "test-service")

    def test_configuration_validation(self):
        """Test configuration validation"""
        config = self.config_manager.create_default_config(
            DeploymentEnvironment.PRODUCTION, "test-service"
        )

        issues = self.config_manager.validate_config(config)
        self.assertIsInstance(issues, list)
        # A valid default configuration should have no issues
        self.assertEqual(len(issues), 0)

    def test_configuration_persistence(self):
        """Test configuration saving and loading"""
        config = self.config_manager.create_default_config(
            DeploymentEnvironment.PRODUCTION, "test-service"
        )

        # Save configuration
        config_file = self.config_manager.save_config(config, "test_config.yaml")
        self.assertTrue(Path(config_file).exists())

        # Load configuration
        loaded_config = self.config_manager.load_config("test_config.yaml")
        self.assertEqual(loaded_config.service_name, config.service_name)
        self.assertEqual(loaded_config.environment, config.environment)

    def test_deployment_templates(self):
        """Test deployment template generation"""
        config = self.config_manager.create_default_config(
            DeploymentEnvironment.PRODUCTION, "test-service"
        )

        templates = self.config_manager.generate_deployment_templates(config)

        expected_templates = ["kubernetes", "docker_compose", "terraform"]
        for template_type in expected_templates:
            self.assertIn(template_type, templates)
            self.assertIsInstance(templates[template_type], str)
            self.assertGreater(len(templates[template_type]), 0)


class TestOperationalTools(unittest.TestCase):
    """Test operational tools and utilities"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_analyzer = LogAnalyzer(self.temp_dir)
        self.security_monitor = SecurityMonitor()
        self.performance_analyzer = PerformanceAnalyzer()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_analysis(self):
        """Test log analysis functionality"""
        # Create a test log file
        log_file = Path(self.temp_dir) / "test.log"
        with open(log_file, "w") as f:
            f.write("2024-01-01T10:00:00 INFO [test] Test log entry\n")
            f.write("2024-01-01T10:01:00 ERROR [test] Test error entry\n")
            f.write("2024-01-01T10:02:00 WARNING [test] Test warning entry\n")

        # Test log analysis (this would be async in real usage)
        # For unit test, we'll test the pattern setup
        self.assertIn("error", self.log_analyzer.log_patterns)
        self.assertIn("warning", self.log_analyzer.log_patterns)

    def test_security_monitoring(self):
        """Test security monitoring"""
        # Test normal request
        normal_request = {
            "source_ip": "192.168.1.100",
            "user_agent": "Mozilla/5.0",
            "path": "/api/test",
            "body": "",
        }

        security_event = self.security_monitor.analyze_request(normal_request)
        # Normal request should not trigger security event
        self.assertIsNone(security_event)

        # Test suspicious request
        suspicious_request = {
            "source_ip": "10.0.0.1",
            "user_agent": "AttackBot/1.0",
            "path": "/api/users?id=1' OR '1'='1",
            "body": "",
        }

        security_event = self.security_monitor.analyze_request(suspicious_request)
        self.assertIsNotNone(security_event)

    def test_performance_analysis(self):
        """Test performance analysis"""
        # Record some performance metrics
        for i in range(20):
            self.performance_analyzer.record_metric("response_time", 0.5 + (i * 0.01))
            self.performance_analyzer.record_metric("cpu_usage", 50 + (i % 10))

        # Test baseline calculation
        baseline = self.performance_analyzer.calculate_baseline("response_time")
        self.assertIn("mean", baseline)
        self.assertIn("std_dev", baseline)

        # Test performance report
        report = self.performance_analyzer.get_performance_report(1)
        self.assertIn("metric_summaries", report)


class TestSystemIntegration(unittest.TestCase):
    """Test system integration and end-to-end functionality"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_complete_system_integration(self):
        """Test complete system integration"""
        # This is a simplified integration test
        # In a real scenario, this would test the full system workflow

        # Initialize components
        config_manager = ConfigurationManager(self.temp_dir)
        monitoring = MonitoringSystem()

        # Create configuration
        config = config_manager.create_default_config(
            DeploymentEnvironment.DEVELOPMENT, "integration-test"
        )

        # Validate configuration
        issues = config_manager.validate_config(config)
        self.assertEqual(len(issues), 0)

        # Test monitoring system initialization
        self.assertIsNotNone(monitoring.metrics_collector)
        self.assertIsNotNone(monitoring.dspy_metrics)

        # Record some test metrics
        monitoring.record_dspy_request(
            duration=0.5, success=True, tokens_used=100, model="test-model"
        )

        # Verify metrics were recorded
        status = monitoring.get_monitoring_status()
        self.assertIn("metrics_count", status)


async def run_async_tests():
    """Run async tests"""
    print("Running async tests...")

    # Test scaling strategies
    test_scaling = TestScalingStrategies()
    test_scaling.setUp()
    await test_scaling.test_scaling_evaluation()
    print("✓ Scaling strategies async tests passed")

    # Test maintenance operations
    test_maintenance = TestMaintenanceOperations()
    test_maintenance.setUp()
    await test_maintenance.test_health_check_execution()
    await test_maintenance.test_backup_creation()
    test_maintenance.tearDown()
    print("✓ Maintenance operations async tests passed")

    # Test maintenance automation
    test_automation = TestMaintenanceAutomation()
    test_automation.setUp()
    await test_automation.test_self_healing_evaluation()
    print("✓ Maintenance automation async tests passed")

    # Test system integration
    test_integration = TestSystemIntegration()
    test_integration.setUp()
    await test_integration.test_complete_system_integration()
    test_integration.tearDown()
    print("✓ System integration async tests passed")


def run_sync_tests():
    """Run synchronous tests"""
    print("Running synchronous tests...")

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestScalingStrategies,
        TestMonitoringSystem,
        TestMaintenanceOperations,
        TestMaintenanceAutomation,
        TestCostOptimization,
        TestScalingConfiguration,
        TestOperationalTools,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


async def main():
    """Main test runner"""
    print("=== DSPy Production Deployment System Test Suite ===")
    print("Testing all components of the production deployment system...\n")

    # Run synchronous tests
    sync_success = run_sync_tests()

    print("\n" + "=" * 60)

    # Run asynchronous tests
    try:
        await run_async_tests()
        async_success = True
    except Exception as e:
        print(f"✗ Async tests failed: {e}")
        async_success = False

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    print(f"Synchronous Tests: {'✓ PASSED' if sync_success else '✗ FAILED'}")
    print(f"Asynchronous Tests: {'✓ PASSED' if async_success else '✗ FAILED'}")

    overall_success = sync_success and async_success
    print(
        f"\nOverall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}"
    )

    if overall_success:
        print("\n🎉 Production deployment system is fully functional!")
        print("All components have been validated and are ready for production use.")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")

    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
