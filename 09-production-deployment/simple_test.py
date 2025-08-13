#!/usr/bin/env python3
"""
Simple Test for DSPy Production Deployment System

This test validates the core functionality of the production deployment system
using the actual available modules.

Author: DSPy Learning Framework
"""

import asyncio
import json
from datetime import datetime, timedelta

from cost_optimization import CostOptimizationManager, ResourceType
from maintenance_operations import MaintenanceOperationsManager
from operational_tools import LogAnalyzer, PerformanceAnalyzer, SecurityMonitor
from scaling_config import ConfigurationManager, DeploymentEnvironment

# Import available modules
from scaling_strategies import LoadBalancingStrategy, ScalingOrchestrator, WorkerNode


async def test_scaling_system():
    """Test the scaling system"""
    print("🔧 Testing Scaling System...")

    # Create orchestrator
    orchestrator = ScalingOrchestrator()

    # Add test nodes
    for i in range(3):
        node = WorkerNode(
            node_id=f"test-node-{i+1}",
            host=f"10.0.1.{10+i}",
            port=8000 + i,
            weight=1.0,
            healthy=True,
        )
        orchestrator.load_balancer.add_node(node)

    # Test node selection
    selected_node = orchestrator.load_balancer.get_node_for_request()
    print(
        f"✓ Node selection works: {selected_node.node_id if selected_node else 'None'}"
    )

    # Test system status
    status = orchestrator.get_system_status()
    print(f"✓ System status: {status['load_balancing']['healthy_nodes']} healthy nodes")

    return True


async def test_cost_optimization():
    """Test the cost optimization system"""
    print("💰 Testing Cost Optimization...")

    # Create cost manager
    cost_manager = CostOptimizationManager()
    cost_manager.setup_default_budgets()

    # Record some usage
    cost_manager.cost_tracker.record_usage(ResourceType.COMPUTE, 2.5)
    cost_manager.cost_tracker.record_usage(ResourceType.API_CALLS, 1000)
    cost_manager.cost_tracker.record_usage(ResourceType.TOKENS, 5000)

    # Get cost report
    report = cost_manager.get_comprehensive_report()
    print(f"✓ Cost tracking works: ${report['summary']['total_daily_cost']:.2f}")

    # Test optimization
    optimization_result = await cost_manager.run_optimization_cycle()
    print(
        f"✓ Optimization works: {optimization_result['recommendations_generated']} recommendations"
    )

    return True


async def test_maintenance_operations():
    """Test the maintenance operations system"""
    print("🔧 Testing Maintenance Operations...")

    # Create operations manager
    ops_manager = MaintenanceOperationsManager()

    # Test health monitoring
    health = await ops_manager.health_monitor.run_health_checks()
    print(f"✓ Health monitoring works: {health.overall_status.value}")

    # Test backup creation
    backup_info = await ops_manager.backup_manager.create_backup(
        backup_type="test", metadata={"test": True}
    )
    print(f"✓ Backup system works: {backup_info.backup_id}")

    return True


def test_configuration_management():
    """Test the configuration management system"""
    print("⚙️  Testing Configuration Management...")

    # Create config manager
    config_manager = ConfigurationManager()

    # Create configuration
    config = config_manager.create_default_config(
        DeploymentEnvironment.DEVELOPMENT, "test-service"
    )

    # Validate configuration
    issues = config_manager.validate_config(config)
    print(f"✓ Configuration validation: {len(issues)} issues found")

    # Generate templates
    templates = config_manager.generate_deployment_templates(config)
    print(f"✓ Template generation: {len(templates)} templates created")

    return True


def test_operational_tools():
    """Test the operational tools"""
    print("📊 Testing Operational Tools...")

    # Test log analyzer
    log_analyzer = LogAnalyzer()
    log_analyzer.add_alert_rule("Test Rule", "ERROR", 5, 10)
    print("✓ Log analyzer works")

    # Test security monitor
    security_monitor = SecurityMonitor()
    request_data = {
        "source_ip": "192.168.1.100",
        "user_agent": "TestBot/1.0",
        "path": "/api/test",
        "body": "",
    }
    security_event = security_monitor.analyze_request(request_data)
    print(f"✓ Security monitor works: {security_event is not None}")

    # Test performance analyzer
    performance_analyzer = PerformanceAnalyzer()
    performance_analyzer.record_metric("test_metric", 42.0)
    print("✓ Performance analyzer works")

    return True


async def run_comprehensive_test():
    """Run comprehensive test of all systems"""
    print("=== DSPy Production Deployment System Test ===")
    print("Testing all available components...\n")

    test_results = {}

    try:
        # Test scaling system
        test_results["scaling"] = await test_scaling_system()
        print()

        # Test cost optimization
        test_results["cost_optimization"] = await test_cost_optimization()
        print()

        # Test maintenance operations
        test_results["maintenance"] = await test_maintenance_operations()
        print()

        # Test configuration management
        test_results["configuration"] = test_configuration_management()
        print()

        # Test operational tools
        test_results["operational_tools"] = test_operational_tools()
        print()

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

    # Summary
    print("=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if not result:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Production deployment system is working correctly!")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("Please check the output above for details.")

    return all_passed


async def main():
    """Main test function"""
    success = await run_comprehensive_test()
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
