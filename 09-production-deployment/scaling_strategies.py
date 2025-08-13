#!/usr/bin/env python3
"""
Intelligent Scaling Strategies for DSPy Applications

This module provides comprehensive scaling strategies, load balancing,
resource optimization, and auto-scaling capabilities for DSPy applications
in production environments.

Learning Objectives:
- Implement intelligent auto-scaling algorithms
- Create load balancing strategies for DSPy workloads
- Optimize resource allocation and utilization
- Handle traffic spikes and load distribution
- Implement cost-effective scaling policies

Author: DSPy Learning Framework
"""

import asyncio
import logging
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction"""

    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    INTELLIGENT = "intelligent"


class ScalingPolicy(Enum):
    """Scaling policies"""

    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    HYBRID = "hybrid"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""

    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: float
    active_requests: int
    response_time: float
    error_rate: float
    timestamp: datetime


@dataclass
class ScalingEvent:
    """Scaling event record"""

    timestamp: datetime
    direction: ScalingDirection
    reason: str
    old_capacity: int
    new_capacity: int
    metrics: ResourceMetrics
    success: bool = True
    error_message: str | None = None


@dataclass
class WorkerNode:
    """Worker node representation"""

    node_id: str
    host: str
    port: int
    weight: float = 1.0
    active_connections: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    healthy: bool = True
    last_health_check: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LoadBalancer(ABC):
    """Abstract load balancer"""

    @abstractmethod
    def select_node(
        self, nodes: list[WorkerNode], request_context: dict[str, Any] = None
    ) -> WorkerNode | None:
        """Select a node for the request"""
        pass

    @abstractmethod
    def update_node_stats(self, node: WorkerNode, response_time: float, success: bool):
        """Update node statistics after request completion"""
        pass


class RoundRobinBalancer(LoadBalancer):
    """Round-robin load balancer"""

    def __init__(self):
        self.current_index = 0
        self._lock = threading.Lock()

    def select_node(
        self, nodes: list[WorkerNode], request_context: dict[str, Any] = None
    ) -> WorkerNode | None:
        healthy_nodes = [node for node in nodes if node.healthy]
        if not healthy_nodes:
            return None

        with self._lock:
            node = healthy_nodes[self.current_index % len(healthy_nodes)]
            self.current_index += 1
            return node

    def update_node_stats(self, node: WorkerNode, response_time: float, success: bool):
        node.total_requests += 1
        # Update rolling average response time
        alpha = 0.1  # Smoothing factor
        node.avg_response_time = (
            1 - alpha
        ) * node.avg_response_time + alpha * response_time


class LeastConnectionsBalancer(LoadBalancer):
    """Least connections load balancer"""

    def select_node(
        self, nodes: list[WorkerNode], request_context: dict[str, Any] = None
    ) -> WorkerNode | None:
        healthy_nodes = [node for node in nodes if node.healthy]
        if not healthy_nodes:
            return None

        return min(healthy_nodes, key=lambda n: n.active_connections)

    def update_node_stats(self, node: WorkerNode, response_time: float, success: bool):
        node.total_requests += 1
        alpha = 0.1
        node.avg_response_time = (
            1 - alpha
        ) * node.avg_response_time + alpha * response_time


class WeightedRoundRobinBalancer(LoadBalancer):
    """Weighted round-robin load balancer"""

    def __init__(self):
        self.current_weights: dict[str, float] = {}
        self._lock = threading.Lock()

    def select_node(
        self, nodes: list[WorkerNode], request_context: dict[str, Any] = None
    ) -> WorkerNode | None:
        healthy_nodes = [node for node in nodes if node.healthy]
        if not healthy_nodes:
            return None

        with self._lock:
            # Initialize weights if needed
            for node in healthy_nodes:
                if node.node_id not in self.current_weights:
                    self.current_weights[node.node_id] = 0

            # Find node with highest current weight
            selected_node = max(
                healthy_nodes, key=lambda n: self.current_weights[n.node_id]
            )

            # Update weights
            total_weight = sum(node.weight for node in healthy_nodes)
            for node in healthy_nodes:
                if node.node_id == selected_node.node_id:
                    self.current_weights[node.node_id] -= total_weight
                else:
                    self.current_weights[node.node_id] += node.weight

            return selected_node

    def update_node_stats(self, node: WorkerNode, response_time: float, success: bool):
        node.total_requests += 1
        alpha = 0.1
        node.avg_response_time = (
            1 - alpha
        ) * node.avg_response_time + alpha * response_time


class IntelligentBalancer(LoadBalancer):
    """Intelligent load balancer using multiple factors"""

    def __init__(self):
        self.response_time_weight = 0.4
        self.cpu_weight = 0.3
        self.connection_weight = 0.2
        self.memory_weight = 0.1

    def select_node(
        self, nodes: list[WorkerNode], request_context: dict[str, Any] = None
    ) -> WorkerNode | None:
        healthy_nodes = [node for node in nodes if node.healthy]
        if not healthy_nodes:
            return None

        # Calculate composite score for each node (lower is better)
        def calculate_score(node: WorkerNode) -> float:
            # Normalize metrics (0-1 scale)
            response_time_score = min(
                node.avg_response_time / 2.0, 1.0
            )  # Assume 2s is max
            cpu_score = node.cpu_percent / 100.0
            memory_score = node.memory_percent / 100.0
            connection_score = min(
                node.active_connections / 100.0, 1.0
            )  # Assume 100 is max

            return (
                response_time_score * self.response_time_weight
                + cpu_score * self.cpu_weight
                + memory_score * self.memory_weight
                + connection_score * self.connection_weight
            )

        return min(healthy_nodes, key=calculate_score)

    def update_node_stats(self, node: WorkerNode, response_time: float, success: bool):
        node.total_requests += 1
        alpha = 0.1
        node.avg_response_time = (
            1 - alpha
        ) * node.avg_response_time + alpha * response_time


class ResourceMonitor:
    """Monitor system resources for scaling decisions"""

    def __init__(self, history_size: int = 100):
        self.metrics_history: deque = deque(maxlen=history_size)
        self.current_metrics: ResourceMetrics | None = None
        self._lock = threading.Lock()

    async def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            net_io = psutil.net_io_counters()

            # Application metrics (would be provided by monitoring system)
            active_requests = 0  # Placeholder
            response_time = 0.0  # Placeholder
            error_rate = 0.0  # Placeholder

            metrics = ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                network_io=net_io.bytes_sent + net_io.bytes_recv if net_io else 0,
                active_requests=active_requests,
                response_time=response_time,
                error_rate=error_rate,
                timestamp=datetime.utcnow(),
            )

            with self._lock:
                self.metrics_history.append(metrics)
                self.current_metrics = metrics

            return metrics

        except Exception as e:
            logger.error("Failed to collect metrics: %s", str(e))
            return ResourceMetrics(
                cpu_percent=0,
                memory_percent=0,
                disk_percent=0,
                network_io=0,
                active_requests=0,
                response_time=0,
                error_rate=0,
                timestamp=datetime.utcnow(),
            )

    def get_average_metrics(self, minutes: int = 5) -> ResourceMetrics | None:
        """Get average metrics over specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        with self._lock:
            recent_metrics = [
                m for m in self.metrics_history if m.timestamp >= cutoff_time
            ]

        if not recent_metrics:
            return None

        return ResourceMetrics(
            cpu_percent=statistics.mean(m.cpu_percent for m in recent_metrics),
            memory_percent=statistics.mean(m.memory_percent for m in recent_metrics),
            disk_percent=statistics.mean(m.disk_percent for m in recent_metrics),
            network_io=statistics.mean(m.network_io for m in recent_metrics),
            active_requests=int(
                statistics.mean(m.active_requests for m in recent_metrics)
            ),
            response_time=statistics.mean(m.response_time for m in recent_metrics),
            error_rate=statistics.mean(m.error_rate for m in recent_metrics),
            timestamp=datetime.utcnow(),
        )

    def get_trend(self, metric_name: str, minutes: int = 10) -> str:
        """Get trend for a specific metric"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        with self._lock:
            recent_metrics = [
                m for m in self.metrics_history if m.timestamp >= cutoff_time
            ]

        if len(recent_metrics) < 3:
            return "insufficient_data"

        values = [getattr(m, metric_name) for m in recent_metrics]

        # Simple trend analysis
        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"


class AutoScaler:
    """Intelligent auto-scaling system"""

    def __init__(
        self,
        min_capacity: int = 1,
        max_capacity: int = 10,
        target_cpu: float = 70.0,
        target_memory: float = 80.0,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 30.0,
        cooldown_minutes: int = 5,
    ):

        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.target_cpu = target_cpu
        self.target_memory = target_memory
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_minutes = cooldown_minutes

        self.current_capacity = min_capacity
        self.last_scaling_event: Optional[datetime] = None
        self.scaling_history: list[ScalingEvent] = []
        self.resource_monitor = ResourceMonitor()

        # Predictive scaling
        self.load_patterns: dict[str, list[float]] = defaultdict(list)
        self.enable_predictive = True

    async def evaluate_scaling(self) -> ScalingEvent | None:
        """Evaluate if scaling is needed"""
        current_metrics = await self.resource_monitor.collect_metrics()

        # Check cooldown period
        if (
            self.last_scaling_event
            and datetime.utcnow() - self.last_scaling_event
            < timedelta(minutes=self.cooldown_minutes)
        ):
            return None

        # Get average metrics for decision making
        avg_metrics = self.resource_monitor.get_average_metrics(5)
        if not avg_metrics:
            return None

        # Determine scaling direction
        scaling_direction = self._determine_scaling_direction(avg_metrics)

        if scaling_direction == ScalingDirection.STABLE:
            return None

        # Calculate new capacity
        new_capacity = self._calculate_new_capacity(scaling_direction, avg_metrics)

        if new_capacity == self.current_capacity:
            return None

        # Create scaling event
        scaling_event = ScalingEvent(
            timestamp=datetime.utcnow(),
            direction=scaling_direction,
            reason=self._get_scaling_reason(avg_metrics),
            old_capacity=self.current_capacity,
            new_capacity=new_capacity,
            metrics=current_metrics,
        )

        return scaling_event

    def _determine_scaling_direction(
        self, metrics: ResourceMetrics
    ) -> ScalingDirection:
        """Determine if we should scale up, down, or stay stable"""

        # Scale up conditions
        if (
            metrics.cpu_percent > self.scale_up_threshold
            or metrics.memory_percent > self.scale_up_threshold
            or metrics.response_time > 2.0  # 2 second threshold
            or metrics.error_rate > 0.05
        ):  # 5% error rate threshold
            return ScalingDirection.UP

        # Scale down conditions
        if (
            metrics.cpu_percent < self.scale_down_threshold
            and metrics.memory_percent < self.scale_down_threshold
            and metrics.response_time < 0.5  # Fast response times
            and metrics.error_rate < 0.01
        ):  # Low error rate
            return ScalingDirection.DOWN

        return ScalingDirection.STABLE

    def _calculate_new_capacity(
        self, direction: ScalingDirection, metrics: ResourceMetrics
    ) -> int:
        """Calculate new capacity based on scaling direction and metrics"""

        if direction == ScalingDirection.UP:
            # Calculate scale-up factor based on resource utilization
            cpu_factor = max(1.0, metrics.cpu_percent / self.target_cpu)
            memory_factor = max(1.0, metrics.memory_percent / self.target_memory)

            scale_factor = max(cpu_factor, memory_factor)
            new_capacity = min(
                self.max_capacity,
                max(
                    self.current_capacity + 1, int(self.current_capacity * scale_factor)
                ),
            )

        else:  # Scale down
            # Conservative scale-down
            new_capacity = max(self.min_capacity, self.current_capacity - 1)

        return new_capacity

    def _get_scaling_reason(self, metrics: ResourceMetrics) -> str:
        """Get human-readable reason for scaling"""
        reasons = []

        if metrics.cpu_percent > self.scale_up_threshold:
            reasons.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent < self.scale_down_threshold:
            reasons.append(f"Low CPU usage: {metrics.cpu_percent:.1f}%")

        if metrics.memory_percent > self.scale_up_threshold:
            reasons.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent < self.scale_down_threshold:
            reasons.append(f"Low memory usage: {metrics.memory_percent:.1f}%")

        if metrics.response_time > 2.0:
            reasons.append(f"High response time: {metrics.response_time:.2f}s")

        if metrics.error_rate > 0.05:
            reasons.append(f"High error rate: {metrics.error_rate:.2%}")

        return "; ".join(reasons) if reasons else "Optimization"

    async def execute_scaling(self, scaling_event: ScalingEvent) -> bool:
        """Execute the scaling operation"""
        try:
            logger.info(
                "Executing scaling: %s -> %s",
                scaling_event.old_capacity,
                scaling_event.new_capacity,
            )
            logger.info("Reason: {scaling_event.reason}")

            # Here you would implement actual scaling logic
            # For example: start/stop containers, adjust load balancer, etc.

            # Simulate scaling operation
            await asyncio.sleep(1)

            # Update capacity
            self.current_capacity = scaling_event.new_capacity
            self.last_scaling_event = scaling_event.timestamp
            scaling_event.success = True

            self.scaling_history.append(scaling_event)

            logger.info(
                "Scaling completed successfully. New capacity: %s",
                self.current_capacity,
            )
            return True

        except Exception as e:
            scaling_event.success = False
            scaling_event.error_message = str(e)
            self.scaling_history.append(scaling_event)
            logger.error("Scaling failed: %s", e)
            return False

    def get_scaling_recommendations(self) -> dict[str, Any]:
        """Get scaling recommendations based on current state"""
        current_metrics = self.resource_monitor.current_metrics
        if not current_metrics:
            return {"status": "no_data"}

        avg_metrics = self.resource_monitor.get_average_metrics(10)
        if not avg_metrics:
            return {"status": "insufficient_data"}

        recommendations = {
            "current_capacity": self.current_capacity,
            "recommended_action": self._determine_scaling_direction(avg_metrics).value,
            "current_metrics": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "response_time": current_metrics.response_time,
                "error_rate": current_metrics.error_rate,
            },
            "average_metrics": {
                "cpu_percent": avg_metrics.cpu_percent,
                "memory_percent": avg_metrics.memory_percent,
                "response_time": avg_metrics.response_time,
                "error_rate": avg_metrics.error_rate,
            },
            "trends": {
                "cpu": self.resource_monitor.get_trend("cpu_percent"),
                "memory": self.resource_monitor.get_trend("memory_percent"),
                "response_time": self.resource_monitor.get_trend("response_time"),
            },
            "last_scaling": (
                self.last_scaling_event.isoformat() if self.last_scaling_event else None
            ),
        }

        return recommendations


class LoadBalancingManager:
    """Manage load balancing across worker nodes"""

    def __init__(
        self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.INTELLIGENT
    ):
        self.strategy = strategy
        self.nodes: dict[str, WorkerNode] = {}
        self.balancer = self._create_balancer(strategy)
        self.health_check_interval = 30  # seconds
        self.running = False
        self._lock = threading.Lock()

    def _create_balancer(self, strategy: LoadBalancingStrategy) -> LoadBalancer:
        """Create load balancer based on strategy"""
        balancer_map = {
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinBalancer,
            LoadBalancingStrategy.LEAST_CONNECTIONS: LeastConnectionsBalancer,
            LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinBalancer,
            LoadBalancingStrategy.INTELLIGENT: IntelligentBalancer,
        }

        return balancer_map.get(strategy, IntelligentBalancer)()

    def add_node(self, node: WorkerNode):
        """Add a worker node"""
        with self._lock:
            self.nodes[node.node_id] = node
        logger.info(
            "Added worker node: %s (%s:%s)",
            str(node.node_id),
            str(node.host),
            str(node.port),
        )

    def remove_node(self, node_id: str):
        """Remove a worker node"""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
        logger.info("Removed worker node: %s", str(node_id))

    def get_node_for_request(
        self, request_context: dict[str, Any] = None
    ) -> WorkerNode | None:
        """Get a node to handle the request"""
        with self._lock:
            nodes = list(self.nodes.values())

        return self.balancer.select_node(nodes, request_context)

    async def execute_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Execute a request using load balancing"""
        node = self.get_node_for_request(request_data)
        if not node:
            raise Exception("No healthy nodes available")

        start_time = time.time()
        success = True

        try:
            # Increment active connections
            node.active_connections += 1

            # Simulate request execution
            async with aiohttp.ClientSession() as session:
                url = f"http://{node.host}:{node.port}/process"
                async with session.post(url, json=request_data, timeout=30) as response:
                    result = await response.json()
                    success = response.status == 200

            return result

        except Exception as e:
            success = False
            logger.error("Request failed on node %s: %e", str(node.node_id), str(e))
            raise

        finally:
            # Update node statistics
            response_time = time.time() - start_time
            node.active_connections = max(0, node.active_connections - 1)
            self.balancer.update_node_stats(node, response_time, success)

    async def health_check_node(self, node: WorkerNode) -> bool:
        """Perform health check on a node"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://{node.host}:{node.port}/health"
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        health_data = await response.json()

                        # Update node metrics from health check
                        node.cpu_percent = health_data.get("cpu_percent", 0)
                        node.memory_percent = health_data.get("memory_percent", 0)
                        node.healthy = True
                        node.last_health_check = datetime.utcnow()

                        return True

            return False

        except Exception as e:
            logger.warning(
                "Health check failed for node %s: %s", str(node.node_id), str(e)
            )
            node.healthy = False
            node.last_health_check = datetime.utcnow()
            return False

    async def start_health_checks(self):
        """Start periodic health checks"""
        self.running = True
        logger.info("Starting health checks for worker nodes")

        while self.running:
            with self._lock:
                nodes = list(self.nodes.values())

            # Check all nodes concurrently
            health_tasks = [self.health_check_node(node) for node in nodes]
            if health_tasks:
                await asyncio.gather(*health_tasks, return_exceptions=True)

            await asyncio.sleep(self.health_check_interval)

    def stop_health_checks(self):
        """Stop health checks"""
        self.running = False
        logger.info("Stopped health checks")

    def get_cluster_status(self) -> dict[str, Any]:
        """Get cluster status"""
        with self._lock:
            nodes = list(self.nodes.values())

        healthy_nodes = [n for n in nodes if n.healthy]
        total_requests = sum(n.total_requests for n in nodes)
        avg_response_time = (
            statistics.mean([n.avg_response_time for n in nodes]) if nodes else 0
        )

        return {
            "total_nodes": len(nodes),
            "healthy_nodes": len(healthy_nodes),
            "unhealthy_nodes": len(nodes) - len(healthy_nodes),
            "total_requests": total_requests,
            "average_response_time": avg_response_time,
            "load_balancing_strategy": self.strategy.value,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "host": n.host,
                    "port": n.port,
                    "healthy": n.healthy,
                    "active_connections": n.active_connections,
                    "total_requests": n.total_requests,
                    "avg_response_time": n.avg_response_time,
                    "cpu_percent": n.cpu_percent,
                    "memory_percent": n.memory_percent,
                    "weight": n.weight,
                }
                for n in nodes
            ],
        }


class ScalingOrchestrator:
    """Orchestrate scaling and load balancing"""

    def __init__(self):
        self.auto_scaler = AutoScaler()
        self.load_balancer = LoadBalancingManager()
        self.running = False
        self.scaling_interval = 60  # seconds

    async def start_orchestration(self):
        """Start the scaling orchestration"""
        self.running = True
        logger.info("Starting scaling orchestration")

        # Start health checks
        health_check_task = asyncio.create_task(
            self.load_balancer.start_health_checks()
        )

        # Start scaling loop
        async def scaling_loop():
            while self.running:
                try:
                    scaling_event = await self.auto_scaler.evaluate_scaling()
                    if scaling_event:
                        success = await self.auto_scaler.execute_scaling(scaling_event)
                        if success:
                            await self._adjust_worker_nodes(scaling_event)
                except Exception as e:
                    logger.error("Scaling evaluation failed: %s", str(e))

                await asyncio.sleep(self.scaling_interval)

        scaling_task = asyncio.create_task(scaling_loop())

        try:
            await asyncio.gather(health_check_task, scaling_task)
        except asyncio.CancelledError:
            logger.info("Scaling orchestration stopped")

    def stop_orchestration(self):
        """Stop the scaling orchestration"""
        self.running = False
        self.load_balancer.stop_health_checks()
        logger.info("Scaling orchestration stopped")

    async def _adjust_worker_nodes(self, scaling_event: ScalingEvent):
        """Adjust worker nodes based on scaling event"""
        current_nodes = len(self.load_balancer.nodes)
        target_nodes = scaling_event.new_capacity

        if target_nodes > current_nodes:
            # Scale up: add nodes
            for i in range(target_nodes - current_nodes):
                node_id = f"worker-{current_nodes + i + 1}"
                port = 8000 + current_nodes + i + 1

                node = WorkerNode(
                    node_id=node_id, host="localhost", port=port, weight=1.0
                )

                # In a real implementation, you would start the actual worker process here
                logger.info(
                    "Would start worker node: %s on port %s", str(node_id), str(port)
                )

                self.load_balancer.add_node(node)

        elif target_nodes < current_nodes:
            # Scale down: remove nodes
            nodes_to_remove = current_nodes - target_nodes
            node_ids = list(self.load_balancer.nodes.keys())[-nodes_to_remove:]

            for node_id in node_ids:
                # In a real implementation, you would gracefully stop the worker process here
                logger.info("Would stop worker node: %s", str(node_id))
                self.load_balancer.remove_node(node_id)

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""
        cluster_status = self.load_balancer.get_cluster_status()
        scaling_recommendations = self.auto_scaler.get_scaling_recommendations()

        return {
            "orchestration_running": self.running,
            "auto_scaling": {
                "current_capacity": self.auto_scaler.current_capacity,
                "min_capacity": self.auto_scaler.min_capacity,
                "max_capacity": self.auto_scaler.max_capacity,
                "last_scaling": (
                    self.auto_scaler.last_scaling_event.isoformat()
                    if self.auto_scaler.last_scaling_event
                    else None
                ),
                "scaling_history_count": len(self.auto_scaler.scaling_history),
            },
            "load_balancing": cluster_status,
            "recommendations": scaling_recommendations,
            "timestamp": datetime.utcnow().isoformat(),
        }


async def main():
    """Demonstrate scaling strategies"""
    print("=== DSPy Intelligent Scaling Strategies Demo ===")

    # Create scaling orchestrator
    orchestrator = ScalingOrchestrator()

    # Add some initial worker nodes
    for i in range(2):
        node = WorkerNode(
            node_id=f"worker-{i+1}", host="localhost", port=8000 + i + 1, weight=1.0
        )
        orchestrator.load_balancer.add_node(node)

    print("Starting scaling orchestration...")
    print("- Auto-scaling based on resource utilization")
    print("- Intelligent load balancing")
    print("- Health monitoring")
    print("- Predictive scaling capabilities")

    try:
        # Start orchestration in background
        orchestration_task = asyncio.create_task(orchestrator.start_orchestration())

        # Simulate some load and demonstrate scaling
        print("\nSimulating workload and scaling decisions...")

        for i in range(10):
            # Get current system status
            status = orchestrator.get_system_status()

            print(f"\n--- Status Update {i+1} ---")
            print(f"Current capacity: {status['auto_scaling']['current_capacity']}")
            print(f"Healthy nodes: {status['load_balancing']['healthy_nodes']}")
            print(f"Total requests: {status['load_balancing']['total_requests']}")

            if status["recommendations"].get("recommended_action") != "stable":
                print(
                    f"Scaling recommendation: {status['recommendations']['recommended_action']}"
                )

            # Simulate some requests
            try:
                node = orchestrator.load_balancer.get_node_for_request()
                if node:
                    print(f"Selected node for request: {node.node_id}")
                    # Simulate request completion
                    orchestrator.load_balancer.balancer.update_node_stats(
                        node, 0.5, True
                    )
            except Exception as e:
                print(f"Request simulation failed: {e}")

            await asyncio.sleep(10)  # Wait 10 seconds between updates

        print("\nScaling demonstration completed!")

        # Show final status
        final_status = orchestrator.get_system_status()
        print(f"\nFinal System Status:")
        print(f"- Capacity: {final_status['auto_scaling']['current_capacity']}")
        print(f"- Healthy nodes: {final_status['load_balancing']['healthy_nodes']}")
        print(
            f"- Scaling events: {final_status['auto_scaling']['scaling_history_count']}"
        )

    except KeyboardInterrupt:
        print("\nStopping scaling orchestration...")
    finally:
        orchestrator.stop_orchestration()


if __name__ == "__main__":
    asyncio.run(main())
