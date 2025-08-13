#!/usr/bin/env python3
"""
Complete DSPy Scaling System Demonstration

This module demonstrates the complete scaling system including:
- Intelligent auto-scaling
- Load balancing strategies
- Cost optimization
- Monitoring and alerting
- Configuration management

Author: DSPy Learning Framework
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime

from cost_optimization import CostOptimizationManager, ResourceType
from scaling_config import (
    ConfigurationManager,
    DeploymentEnvironment,
    ScalingConfiguration,
)

# Import our scaling modules
from scaling_strategies import (
    LoadBalancingStrategy,
    ResourceMetrics,
    ScalingDirection,
    ScalingOrchestrator,
    WorkerNode,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DSPyScalingDemo:
    """Complete DSPy scaling system demonstration"""

    def __init__(self):
        self.orchestrator = ScalingOrchestrator()
        self.cost_manager = CostOptimizationManager()
        self.config_manager = ConfigurationManager()
        self.demo_running = False
        self.request_counter = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Demo configuration
        self.service_name = "dspy-demo-service"
        self.demo_duration = 300  # 5 minutes
        self.request_interval = 1.0  # seconds between requests

        # Setup cost tracking
        self.cost_manager.setup_default_budgets()

    async def setup_demo_environment(self):
        """Set up the demo environment"""
        logger.info("Setting up DSPy scaling demo environment...")

        # Create configuration for demo
        config = self.config_manager.create_default_config(
            DeploymentEnvironment.DEVELOPMENT, self.service_name
        )

        # Customize for demo
        config.auto_scaling.min_instances = 2
        config.auto_scaling.max_instances = 8
        config.auto_scaling.target_cpu_utilization = 60.0
        config.auto_scaling.scale_up_cooldown = 30  # Faster scaling for demo
        config.auto_scaling.scale_down_cooldown = 60

        # Save configuration
        config_file = self.config_manager.save_config(config, "demo_config.yaml")
        logger.info(f"Demo configuration saved to: {config_file}")

        # Add initial worker nodes
        for i in range(config.auto_scaling.min_instances):
            node = WorkerNode(
                node_id=f"demo-worker-{i+1}",
                host="localhost",
                port=8000 + i + 1,
                weight=1.0,
                healthy=True,
            )
            self.orchestrator.load_balancer.add_node(node)
            logger.info(f"Added demo worker node: {node.node_id}")

        # Configure load balancer strategy
        self.orchestrator.load_balancer.strategy = LoadBalancingStrategy.INTELLIGENT

        logger.info("Demo environment setup completed")

    async def simulate_workload(self):
        """Simulate realistic workload patterns"""
        logger.info("Starting workload simulation...")

        start_time = time.time()

        while self.demo_running and (time.time() - start_time) < self.demo_duration:
            try:
                # Simulate varying load patterns
                current_time = time.time() - start_time

                # Create load spikes at certain intervals
                if (
                    int(current_time) % 60 < 20
                ):  # High load for first 20 seconds of each minute
                    request_rate = 5.0  # 5 requests per second
                    cpu_load = random.uniform(70, 90)
                    memory_load = random.uniform(60, 80)
                elif int(current_time) % 60 < 40:  # Medium load
                    request_rate = 2.0  # 2 requests per second
                    cpu_load = random.uniform(40, 60)
                    memory_load = random.uniform(40, 60)
                else:  # Low load
                    request_rate = 0.5  # 0.5 requests per second
                    cpu_load = random.uniform(20, 40)
                    memory_load = random.uniform(20, 40)

                # Simulate requests
                for _ in range(int(request_rate)):
                    await self.simulate_request(cpu_load, memory_load)

                await asyncio.sleep(1.0 / request_rate if request_rate > 0 else 1.0)

            except Exception as e:
                logger.error(f"Error in workload simulation: {e}")
                await asyncio.sleep(1)

        logger.info("Workload simulation completed")

    async def simulate_request(self, cpu_load: float, memory_load: float):
        """Simulate a single request"""
        self.request_counter += 1
        self.total_requests += 1

        try:
            # Get node for request
            node = self.orchestrator.load_balancer.get_node_for_request()
            if not node:
                self.failed_requests += 1
                logger.warning("No healthy nodes available for request")
                return

            # Simulate request processing
            processing_time = random.uniform(0.1, 2.0)
            success = random.random() > 0.05  # 95% success rate

            # Update node metrics
            node.active_connections += 1
            node.cpu_percent = cpu_load + random.uniform(-10, 10)
            node.memory_percent = memory_load + random.uniform(-10, 10)

            # Simulate processing delay
            await asyncio.sleep(processing_time / 10)  # Speed up for demo

            # Update statistics
            node.active_connections = max(0, node.active_connections - 1)
            self.orchestrator.load_balancer.balancer.update_node_stats(
                node, processing_time, success
            )

            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            # Record costs
            self.cost_manager.cost_tracker.record_usage(
                ResourceType.COMPUTE,
                processing_time / 3600,  # Convert to hours
                {"node_id": node.node_id, "request_id": self.request_counter},
            )

            self.cost_manager.cost_tracker.record_usage(
                ResourceType.API_CALLS, 1, {"endpoint": "process", "success": success}
            )

            if success:
                tokens_used = random.randint(100, 1000)
                self.cost_manager.cost_tracker.record_usage(
                    ResourceType.TOKENS,
                    tokens_used,
                    {
                        "input_tokens": tokens_used * 0.6,
                        "output_tokens": tokens_used * 0.4,
                    },
                )

        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Request simulation failed: {e}")

    async def monitor_system(self):
        """Monitor system metrics and scaling decisions"""
        logger.info("Starting system monitoring...")

        while self.demo_running:
            try:
                # Get system status
                system_status = self.orchestrator.get_system_status()
                cost_report = self.cost_manager.get_comprehensive_report()

                # Log key metrics
                logger.info(f"=== System Status ===")
                logger.info(
                    f"Capacity: {system_status['auto_scaling']['current_capacity']}"
                )
                logger.info(
                    f"Healthy nodes: {system_status['load_balancing']['healthy_nodes']}"
                )
                logger.info(f"Total requests: {self.total_requests}")
                logger.info(
                    f"Success rate: {(self.successful_requests/max(self.total_requests,1))*100:.1f}%"
                )
                logger.info(
                    f"Current cost: ${cost_report['summary']['total_daily_cost']:.2f}"
                )

                # Check for scaling recommendations
                recommendations = system_status.get("recommendations", {})
                if recommendations.get("recommended_action") != "stable":
                    logger.info(
                        f"Scaling recommendation: {recommendations['recommended_action']}"
                    )

                # Log cost optimization opportunities
                if cost_report["summary"]["optimization_opportunities"] > 0:
                    logger.info(
                        f"Cost optimization opportunities: {cost_report['summary']['optimization_opportunities']}"
                    )
                    logger.info(
                        f"Potential savings: ${cost_report['summary']['potential_savings']:.2f}"
                    )

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)

        logger.info("System monitoring stopped")

    async def run_cost_optimization(self):
        """Run periodic cost optimization"""
        logger.info("Starting cost optimization cycle...")

        while self.demo_running:
            try:
                # Run optimization cycle
                result = await self.cost_manager.run_optimization_cycle()

                if result["auto_implemented"] > 0:
                    logger.info(
                        f"Applied {result['auto_implemented']} cost optimizations"
                    )
                    logger.info(
                        f"Estimated savings: ${result['potential_savings']:.2f}"
                    )

                await asyncio.sleep(120)  # Run every 2 minutes

            except Exception as e:
                logger.error(f"Cost optimization error: {e}")
                await asyncio.sleep(60)

        logger.info("Cost optimization stopped")

    async def generate_demo_report(self):
        """Generate comprehensive demo report"""
        logger.info("Generating demo report...")

        # Get final system status
        system_status = self.orchestrator.get_system_status()
        cost_report = self.cost_manager.get_comprehensive_report()

        # Calculate performance metrics
        success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100
        avg_nodes = system_status["load_balancing"]["healthy_nodes"]

        report = {
            "demo_summary": {
                "service_name": self.service_name,
                "duration_minutes": self.demo_duration / 60,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate_percent": success_rate,
                "average_nodes": avg_nodes,
            },
            "scaling_performance": {
                "final_capacity": system_status["auto_scaling"]["current_capacity"],
                "min_capacity": system_status["auto_scaling"]["min_capacity"],
                "max_capacity": system_status["auto_scaling"]["max_capacity"],
                "scaling_events": system_status["auto_scaling"][
                    "scaling_history_count"
                ],
                "load_balancing_strategy": system_status["load_balancing"][
                    "load_balancing_strategy"
                ],
            },
            "cost_analysis": {
                "total_cost": cost_report["summary"]["total_daily_cost"],
                "cost_trend": cost_report["summary"]["cost_trend"],
                "optimization_opportunities": cost_report["summary"][
                    "optimization_opportunities"
                ],
                "potential_savings": cost_report["summary"]["potential_savings"],
                "budget_status": (
                    "within_limits"
                    if not cost_report["budget_status"]["alerts"]
                    else "alerts_present"
                ),
            },
            "recommendations": cost_report["recommendations"][
                :5
            ],  # Top 5 recommendations
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Save report
        report_file = f"demo_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Demo report saved to: {report_file}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"DSPy SCALING SYSTEM DEMO REPORT")
        print(f"{'='*60}")
        print(f"Service: {report['demo_summary']['service_name']}")
        print(f"Duration: {report['demo_summary']['duration_minutes']:.1f} minutes")
        print(f"Total Requests: {report['demo_summary']['total_requests']:,}")
        print(f"Success Rate: {report['demo_summary']['success_rate_percent']:.1f}%")
        print(
            f"Final Capacity: {report['scaling_performance']['final_capacity']} nodes"
        )
        print(f"Scaling Events: {report['scaling_performance']['scaling_events']}")
        print(f"Total Cost: ${report['cost_analysis']['total_cost']:.2f}")
        print(f"Potential Savings: ${report['cost_analysis']['potential_savings']:.2f}")
        print(
            f"Optimization Opportunities: {report['cost_analysis']['optimization_opportunities']}"
        )

        if report["recommendations"]:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"{i}. {rec['title']} (${rec['potential_savings']:.2f} savings)")

        print(f"{'='*60}")

        return report

    async def run_demo(self):
        """Run the complete scaling demo"""
        print(f"{'='*60}")
        print(f"DSPy INTELLIGENT SCALING SYSTEM DEMO")
        print(f"{'='*60}")
        print(f"This demo will showcase:")
        print(f"• Intelligent auto-scaling based on load")
        print(f"• Advanced load balancing strategies")
        print(f"• Real-time cost optimization")
        print(f"• Comprehensive monitoring and alerting")
        print(f"• Configuration management")
        print(f"")
        print(f"Demo duration: {self.demo_duration/60:.1f} minutes")
        print(f"Service: {self.service_name}")
        print(f"{'='*60}")

        try:
            # Setup environment
            await self.setup_demo_environment()

            # Start demo
            self.demo_running = True

            # Start all demo components
            tasks = [
                asyncio.create_task(self.orchestrator.start_orchestration()),
                asyncio.create_task(self.simulate_workload()),
                asyncio.create_task(self.monitor_system()),
                asyncio.create_task(self.run_cost_optimization()),
            ]

            # Wait for demo completion or interruption
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.demo_duration + 60,
                )
            except asyncio.TimeoutError:
                logger.info("Demo completed successfully")

        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.error(f"Demo error: {e}")
        finally:
            # Stop demo
            self.demo_running = False
            self.orchestrator.stop_orchestration()

            # Generate final report
            await self.generate_demo_report()

            print(f"\nDemo completed! Check the generated report for detailed results.")


async def main():
    """Main demo function"""
    demo = DSPyScalingDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
