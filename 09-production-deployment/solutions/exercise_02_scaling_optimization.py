#!/usr/bin/env python3
"""
Exercise 2: Scaling and Cost Optimization

This exercise demonstrates advanced scaling strategies and cost optimization
techniques for DSPy applications in production.

Learning Objectives:
- Implement intelligent auto-scaling policies
- Optimize costs through resource management
- Configure load balancing strategies
- Monitor and optimize performance

Author: DSPy Learning Framework
"""

import asyncio
import json
import random

# Import our production deployment modules
import sys
from datetime import datetime, timedelta

sys.path.append("..")

from cost_optimization import (
    CostOptimizationManager,
    OptimizationStrategy,
    ResourceType,
)
from scaling_config import ConfigurationManager, DeploymentEnvironment
from scaling_strategies import (
    LoadBalancingStrategy,
    ResourceMetrics,
    ScalingDirection,
    ScalingOrchestrator,
    WorkerNode,
)


async def exercise_02_solution():
    """
    Exercise 2 Solution: Advanced Scaling and Cost Optimization

    This solution demonstrates:
    1. Implementing intelligent auto-scaling policies
    2. Cost optimization strategies
    3. Load balancing optimization
    4. Performance monitoring and optimization
    """

    print("=== Exercise 2: Scaling and Cost Optimization ===")
    print("Implementing advanced scaling and cost optimization...")

    # Step 1: Set up cost optimization manager
    print("\n1. Setting up cost optimization...")
    cost_manager = CostOptimizationManager()
    cost_manager.setup_default_budgets()

    # Set custom budget limits
    cost_manager.cost_tracker.set_budget_limit("daily", 75.0)
    cost_manager.cost_tracker.set_budget_limit("weekly", 450.0)
    cost_manager.cost_tracker.set_budget_limit("monthly", 1800.0)

    print("✓ Cost optimization configured")
    print(f"  - Daily budget: $75.00")
    print(f"  - Weekly budget: $450.00")
    print(f"  - Monthly budget: $1,800.00")

    # Step 2: Configure advanced scaling orchestrator
    print("\n2. Configuring advanced auto-scaling...")
    orchestrator = ScalingOrchestrator()

    # Configure auto-scaler with custom thresholds
    orchestrator.auto_scaler.min_capacity = 2
    orchestrator.auto_scaler.max_capacity = 15
    orchestrator.auto_scaler.target_cpu = 65.0
    orchestrator.auto_scaler.target_memory = 75.0
    orchestrator.auto_scaler.scale_up_threshold = 75.0
    orchestrator.auto_scaler.scale_down_threshold = 25.0
    orchestrator.auto_scaler.cooldown_minutes = 3  # Faster scaling for demo

    # Add worker nodes with different capacities
    node_configs = [
        {"id": "high-perf-1", "host": "10.0.1.10", "weight": 2.0},
        {"id": "high-perf-2", "host": "10.0.1.11", "weight": 2.0},
        {"id": "standard-1", "host": "10.0.1.20", "weight": 1.0},
        {"id": "standard-2", "host": "10.0.1.21", "weight": 1.0},
        {"id": "spot-1", "host": "10.0.1.30", "weight": 1.5},  # Spot instance
    ]

    for config in node_configs:
        node = WorkerNode(
            node_id=config["id"],
            host=config["host"],
            port=8000,
            weight=config["weight"],
            healthy=True,
        )
        orchestrator.load_balancer.add_node(node)

    # Use intelligent load balancing
    orchestrator.load_balancer.strategy = LoadBalancingStrategy.INTELLIGENT

    print("✓ Advanced auto-scaling configured")
    print(
        f"  - Capacity range: {orchestrator.auto_scaler.min_capacity}-{orchestrator.auto_scaler.max_capacity}"
    )
    print(f"  - Target CPU: {orchestrator.auto_scaler.target_cpu}%")
    print(f"  - Load balancing: {orchestrator.load_balancer.strategy.value}")

    # Step 3: Start orchestration
    print("\n3. Starting scaling orchestration...")
    orchestration_task = asyncio.create_task(orchestrator.start_orchestration())

    # Step 4: Simulate varying workload patterns
    print("\n4. Simulating varying workload patterns...")

    workload_phases = [
        {
            "name": "Low Load",
            "duration": 30,
            "request_rate": 1.0,
            "cpu_range": (20, 40),
        },
        {
            "name": "Medium Load",
            "duration": 45,
            "request_rate": 3.0,
            "cpu_range": (50, 70),
        },
        {
            "name": "High Load",
            "duration": 60,
            "request_rate": 8.0,
            "cpu_range": (80, 95),
        },
        {
            "name": "Peak Load",
            "duration": 30,
            "request_rate": 12.0,
            "cpu_range": (90, 98),
        },
        {
            "name": "Cool Down",
            "duration": 45,
            "request_rate": 2.0,
            "cpu_range": (30, 50),
        },
    ]

    total_requests = 0
    total_cost = 0.0

    for phase in workload_phases:
        print(f"\n  Phase: {phase['name']} ({phase['duration']}s)")
        phase_start = datetime.utcnow()

        while (datetime.utcnow() - phase_start).total_seconds() < phase["duration"]:
            # Simulate requests
            for _ in range(int(phase["request_rate"])):
                # Record compute usage
                compute_hours = random.uniform(0.001, 0.005)  # Small increments
                cost_manager.cost_tracker.record_usage(
                    ResourceType.COMPUTE,
                    compute_hours,
                    {"phase": phase["name"], "node_type": "standard"},
                )

                # Record API calls and tokens
                tokens_used = random.randint(100, 1000)
                cost_manager.cost_tracker.record_usage(ResourceType.API_CALLS, 1)
                cost_manager.cost_tracker.record_usage(ResourceType.TOKENS, tokens_used)

                total_requests += 1

            # Update node metrics to simulate load
            for node in orchestrator.load_balancer.nodes.values():
                node.cpu_percent = random.uniform(*phase["cpu_range"])
                node.memory_percent = random.uniform(40, 80)
                node.active_connections = random.randint(0, 20)

            await asyncio.sleep(1)

        # Get current system status
        status = orchestrator.get_system_status()
        current_cost = cost_manager.cost_tracker.get_total_cost(24)

        print(f"    Requests processed: {total_requests}")
        print(f"    Current capacity: {status['auto_scaling']['current_capacity']}")
        print(f"    Current cost: ${current_cost:.2f}")

    # Step 5: Run cost optimization
    print("\n5. Running cost optimization analysis...")
    optimization_result = await cost_manager.run_optimization_cycle()

    print(f"✓ Cost optimization completed")
    print(
        f"  - Recommendations generated: {optimization_result['recommendations_generated']}"
    )
    print(f"  - Auto-implemented: {optimization_result['auto_implemented']}")
    print(f"  - Potential savings: ${optimization_result['potential_savings']:.2f}")

    # Step 6: Generate comprehensive report
    print("\n6. Generating comprehensive report...")

    # Get final system status
    final_status = orchestrator.get_system_status()
    cost_report = cost_manager.get_comprehensive_report()

    # Calculate performance metrics
    final_cost = cost_manager.cost_tracker.get_total_cost(24)
    cost_per_request = final_cost / max(total_requests, 1)

    report = {
        "workload_summary": {
            "total_requests": total_requests,
            "total_phases": len(workload_phases),
            "total_duration_minutes": sum(p["duration"] for p in workload_phases) / 60,
            "average_request_rate": total_requests
            / sum(p["duration"] for p in workload_phases),
        },
        "scaling_performance": {
            "final_capacity": final_status["auto_scaling"]["current_capacity"],
            "capacity_range": f"{orchestrator.auto_scaler.min_capacity}-{orchestrator.auto_scaler.max_capacity}",
            "scaling_events": final_status["auto_scaling"]["scaling_history_count"],
            "healthy_nodes": final_status["load_balancing"]["healthy_nodes"],
            "load_balancing_strategy": final_status["load_balancing"][
                "load_balancing_strategy"
            ],
        },
        "cost_analysis": {
            "total_cost": final_cost,
            "cost_per_request": cost_per_request,
            "cost_trend": cost_report["summary"]["cost_trend"],
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
        "optimization_recommendations": cost_report["recommendations"][:3],  # Top 3
    }

    print("\n7. Final Report Summary:")
    print(f"  Workload Performance:")
    print(f"    - Total requests: {report['workload_summary']['total_requests']:,}")
    print(
        f"    - Average rate: {report['workload_summary']['average_request_rate']:.1f} req/s"
    )
    print(
        f"    - Duration: {report['workload_summary']['total_duration_minutes']:.1f} minutes"
    )

    print(f"  Scaling Performance:")
    print(
        f"    - Final capacity: {report['scaling_performance']['final_capacity']} nodes"
    )
    print(f"    - Scaling events: {report['scaling_performance']['scaling_events']}")
    print(f"    - Healthy nodes: {report['scaling_performance']['healthy_nodes']}")

    print(f"  Cost Analysis:")
    print(f"    - Total cost: ${report['cost_analysis']['total_cost']:.2f}")
    print(f"    - Cost per request: ${report['cost_analysis']['cost_per_request']:.4f}")
    print(
        f"    - Potential savings: ${report['cost_analysis']['potential_savings']:.2f}"
    )
    print(f"    - Budget status: {report['cost_analysis']['budget_status']}")

    if report["optimization_recommendations"]:
        print(f"  Top Recommendations:")
        for i, rec in enumerate(report["optimization_recommendations"], 1):
            print(f"    {i}. {rec['title']} (${rec['potential_savings']:.2f} savings)")

    print("\n✅ Exercise 2 completed successfully!")
    print("\nAdvanced scaling and cost optimization demonstrated:")
    print("  • Intelligent auto-scaling with custom thresholds")
    print("  • Multi-phase workload simulation")
    print("  • Real-time cost tracking and optimization")
    print("  • Load balancing with weighted nodes")
    print("  • Comprehensive performance analysis")

    # Cleanup
    orchestrator.stop_orchestration()

    return report


if __name__ == "__main__":
    result = asyncio.run(exercise_02_solution())
    print(f"\nExercise Result Summary:")
    print(f"Total Requests: {result['workload_summary']['total_requests']:,}")
    print(f"Final Capacity: {result['scaling_performance']['final_capacity']} nodes")
    print(f"Total Cost: ${result['cost_analysis']['total_cost']:.2f}")
    print(f"Potential Savings: ${result['cost_analysis']['potential_savings']:.2f}")
