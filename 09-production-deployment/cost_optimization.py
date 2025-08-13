#!/usr/bin/env python3
"""
Cost Optimization Tools for DSPy Applications

This module provides cost optimization strategies, resource efficiency analysis,
and budget management tools for DSPy applications in production.

Author: DSPy Learning Framework
"""

import asyncio
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources"""

    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    API_CALLS = "api_calls"
    TOKENS = "tokens"


class OptimizationStrategy(Enum):
    """Cost optimization strategies"""

    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


@dataclass
class CostMetric:
    """Cost tracking metric"""

    resource_type: ResourceType
    usage_amount: float
    unit_cost: float
    total_cost: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""

    title: str
    description: str
    potential_savings: float
    implementation_effort: str  # low, medium, high
    risk_level: str  # low, medium, high
    category: str
    action_items: list[str]
    estimated_impact: dict[str, Any]


class CostTracker:
    """Track costs across different resources"""

    def __init__(self):
        self.cost_history: list[CostMetric] = []
        self.current_rates: dict[ResourceType, float] = {
            ResourceType.COMPUTE: 0.10,  # per hour
            ResourceType.MEMORY: 0.01,  # per GB-hour
            ResourceType.STORAGE: 0.023,  # per GB-month
            ResourceType.NETWORK: 0.09,  # per GB
            ResourceType.API_CALLS: 0.002,  # per 1K calls
            ResourceType.TOKENS: 0.00002,  # per token
        }
        self.budget_limits: dict[str, float] = {}
        self.alerts_enabled = True

    def record_usage(
        self,
        resource_type: ResourceType,
        amount: float,
        metadata: dict[str, Any] = None,
    ):
        """Record resource usage"""
        unit_cost = self.current_rates.get(resource_type, 0.0)
        total_cost = amount * unit_cost

        cost_metric = CostMetric(
            resource_type=resource_type,
            usage_amount=amount,
            unit_cost=unit_cost,
            total_cost=total_cost,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )

        self.cost_history.append(cost_metric)

        # Check budget alerts
        if self.alerts_enabled:
            self._check_budget_alerts()

    def get_costs_by_period(self, hours: int = 24) -> dict[ResourceType, float]:
        """Get costs by resource type for a time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        costs = defaultdict(float)
        for metric in self.cost_history:
            if metric.timestamp >= cutoff_time:
                costs[metric.resource_type] += metric.total_cost

        return dict(costs)

    def get_total_cost(self, hours: int = 24) -> float:
        """Get total cost for a time period"""
        costs = self.get_costs_by_period(hours)
        return sum(costs.values())

    def get_cost_trends(self, days: int = 7) -> dict[str, Any]:
        """Analyze cost trends"""
        daily_costs = defaultdict(float)

        for metric in self.cost_history:
            if metric.timestamp >= datetime.utcnow() - timedelta(days=days):
                day_key = metric.timestamp.strftime("%Y-%m-%d")
                daily_costs[day_key] += metric.total_cost

        if len(daily_costs) < 2:
            return {"trend": "insufficient_data"}

        costs_list = list(daily_costs.values())
        recent_avg = (
            statistics.mean(costs_list[-3:]) if len(costs_list) >= 3 else costs_list[-1]
        )
        older_avg = (
            statistics.mean(costs_list[:-3]) if len(costs_list) >= 6 else costs_list[0]
        )

        if recent_avg > older_avg * 1.1:
            trend = "increasing"
        elif recent_avg < older_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "daily_costs": dict(daily_costs),
            "average_daily_cost": statistics.mean(costs_list),
            "total_period_cost": sum(costs_list),
        }

    def set_budget_limit(self, period: str, limit: float):
        """Set budget limit for a period (daily, weekly, monthly)"""
        self.budget_limits[period] = limit

    def _check_budget_alerts(self):
        """Check if budget limits are exceeded"""
        current_daily = self.get_total_cost(24)
        current_weekly = self.get_total_cost(24 * 7)
        current_monthly = self.get_total_cost(24 * 30)

        alerts = []

        if (
            "daily" in self.budget_limits
            and current_daily > self.budget_limits["daily"]
        ):
            alerts.append(
                f"Daily budget exceeded: ${current_daily:.2f} > ${self.budget_limits['daily']:.2f}"
            )

        if (
            "weekly" in self.budget_limits
            and current_weekly > self.budget_limits["weekly"]
        ):
            alerts.append(
                f"Weekly budget exceeded: ${current_weekly:.2f} > ${self.budget_limits['weekly']:.2f}"
            )

        if (
            "monthly" in self.budget_limits
            and current_monthly > self.budget_limits["monthly"]
        ):
            alerts.append(
                f"Monthly budget exceeded: ${current_monthly:.2f} > ${self.budget_limits['monthly']:.2f}"
            )

        for alert in alerts:
            logger.warning(f"BUDGET ALERT: {alert}")


class ResourceOptimizer:
    """Optimize resource usage for cost efficiency"""

    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker
        self.optimization_history: list[dict[str, Any]] = []

    def analyze_usage_patterns(self, days: int = 7) -> dict[str, Any]:
        """Analyze resource usage patterns"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        # Group usage by hour of day and day of week
        hourly_usage = defaultdict(list)
        daily_usage = defaultdict(list)
        resource_usage = defaultdict(list)

        for metric in self.cost_tracker.cost_history:
            if metric.timestamp >= cutoff_time:
                hour = metric.timestamp.hour
                day = metric.timestamp.strftime("%A")

                hourly_usage[hour].append(metric.total_cost)
                daily_usage[day].append(metric.total_cost)
                resource_usage[metric.resource_type].append(metric.total_cost)

        # Calculate averages
        hourly_avg = {
            hour: statistics.mean(costs) for hour, costs in hourly_usage.items()
        }
        daily_avg = {day: statistics.mean(costs) for day, costs in daily_usage.items()}
        resource_avg = {
            res: statistics.mean(costs) for res, costs in resource_usage.items()
        }

        # Find peak and off-peak hours
        if hourly_avg:
            peak_hour = max(hourly_avg, key=hourly_avg.get)
            off_peak_hour = min(hourly_avg, key=hourly_avg.get)
        else:
            peak_hour = off_peak_hour = None

        return {
            "hourly_patterns": hourly_avg,
            "daily_patterns": daily_avg,
            "resource_breakdown": resource_avg,
            "peak_hour": peak_hour,
            "off_peak_hour": off_peak_hour,
            "analysis_period_days": days,
        }

    def generate_recommendations(
        self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ) -> list[OptimizationRecommendation]:
        """Generate cost optimization recommendations"""
        recommendations = []

        # Analyze current usage
        usage_patterns = self.analyze_usage_patterns()
        cost_trends = self.cost_tracker.get_cost_trends()
        current_costs = self.cost_tracker.get_costs_by_period(24)

        # Compute optimization recommendations
        recommendations.extend(
            self._compute_optimization_recommendations(current_costs, strategy)
        )
        recommendations.extend(
            self._api_optimization_recommendations(current_costs, strategy)
        )
        recommendations.extend(
            self._scheduling_recommendations(usage_patterns, strategy)
        )
        recommendations.extend(
            self._resource_rightsizing_recommendations(usage_patterns, strategy)
        )

        # Sort by potential savings
        recommendations.sort(key=lambda x: x.potential_savings, reverse=True)

        return recommendations

    def _compute_optimization_recommendations(
        self, costs: dict[ResourceType, float], strategy: OptimizationStrategy
    ) -> list[OptimizationRecommendation]:
        """Generate compute optimization recommendations"""
        recommendations = []

        compute_cost = costs.get(ResourceType.COMPUTE, 0)
        memory_cost = costs.get(ResourceType.MEMORY, 0)

        if compute_cost > 10:  # Significant compute costs
            if strategy in [
                OptimizationStrategy.AGGRESSIVE,
                OptimizationStrategy.BALANCED,
            ]:
                recommendations.append(
                    OptimizationRecommendation(
                        title="Implement Auto-Scaling",
                        description="Use auto-scaling to reduce compute costs during low-usage periods",
                        potential_savings=compute_cost * 0.3,
                        implementation_effort="medium",
                        risk_level="low",
                        category="compute",
                        action_items=[
                            "Configure auto-scaling policies",
                            "Set appropriate scaling thresholds",
                            "Monitor scaling behavior",
                        ],
                        estimated_impact={
                            "cost_reduction": "20-40%",
                            "performance_impact": "minimal",
                        },
                    )
                )

            recommendations.append(
                OptimizationRecommendation(
                    title="Use Spot/Preemptible Instances",
                    description="Leverage spot instances for non-critical workloads",
                    potential_savings=compute_cost * 0.6,
                    implementation_effort="high",
                    risk_level="medium",
                    category="compute",
                    action_items=[
                        "Identify fault-tolerant workloads",
                        "Implement spot instance handling",
                        "Add graceful shutdown procedures",
                    ],
                    estimated_impact={
                        "cost_reduction": "50-70%",
                        "availability_impact": "some interruptions",
                    },
                )
            )

        return recommendations

    def _api_optimization_recommendations(
        self, costs: dict[ResourceType, float], strategy: OptimizationStrategy
    ) -> list[OptimizationRecommendation]:
        """Generate API optimization recommendations"""
        recommendations = []

        api_cost = costs.get(ResourceType.API_CALLS, 0)
        token_cost = costs.get(ResourceType.TOKENS, 0)

        if token_cost > 5:  # Significant token costs
            recommendations.append(
                OptimizationRecommendation(
                    title="Implement Response Caching",
                    description="Cache API responses to reduce redundant calls and token usage",
                    potential_savings=token_cost * 0.4,
                    implementation_effort="medium",
                    risk_level="low",
                    category="api",
                    action_items=[
                        "Implement intelligent caching layer",
                        "Set appropriate cache TTL values",
                        "Monitor cache hit rates",
                    ],
                    estimated_impact={
                        "cost_reduction": "30-50%",
                        "response_time": "improved",
                    },
                )
            )

            recommendations.append(
                OptimizationRecommendation(
                    title="Optimize Prompt Engineering",
                    description="Reduce token usage through more efficient prompts",
                    potential_savings=token_cost * 0.25,
                    implementation_effort="low",
                    risk_level="low",
                    category="api",
                    action_items=[
                        "Analyze current prompt efficiency",
                        "Implement prompt compression techniques",
                        "Use few-shot learning more effectively",
                    ],
                    estimated_impact={
                        "cost_reduction": "15-35%",
                        "quality_impact": "maintained or improved",
                    },
                )
            )

        return recommendations

    def _scheduling_recommendations(
        self, patterns: dict[str, Any], strategy: OptimizationStrategy
    ) -> list[OptimizationRecommendation]:
        """Generate scheduling optimization recommendations"""
        recommendations = []

        if patterns.get("peak_hour") and patterns.get("off_peak_hour"):
            peak_cost = patterns["hourly_patterns"].get(patterns["peak_hour"], 0)
            off_peak_cost = patterns["hourly_patterns"].get(
                patterns["off_peak_hour"], 0
            )

            if peak_cost > off_peak_cost * 2:  # Significant difference
                recommendations.append(
                    OptimizationRecommendation(
                        title="Implement Workload Scheduling",
                        description="Schedule non-urgent tasks during off-peak hours",
                        potential_savings=peak_cost * 0.3,
                        implementation_effort="medium",
                        risk_level="low",
                        category="scheduling",
                        action_items=[
                            "Identify deferrable workloads",
                            "Implement job scheduling system",
                            "Set up off-peak processing queues",
                        ],
                        estimated_impact={
                            "cost_reduction": "20-40%",
                            "latency_impact": "some tasks delayed",
                        },
                    )
                )

        return recommendations

    def _resource_rightsizing_recommendations(
        self, patterns: dict[str, Any], strategy: OptimizationStrategy
    ) -> list[OptimizationRecommendation]:
        """Generate resource rightsizing recommendations"""
        recommendations = []

        resource_costs = patterns.get("resource_breakdown", {})

        # Check for over-provisioning
        total_cost = sum(
            cost for cost in resource_costs.values() if isinstance(cost, (int, float))
        )

        if total_cost > 20:  # Significant total costs
            recommendations.append(
                OptimizationRecommendation(
                    title="Right-size Resources",
                    description="Analyze and optimize resource allocation based on actual usage",
                    potential_savings=total_cost * 0.2,
                    implementation_effort="high",
                    risk_level="medium",
                    category="rightsizing",
                    action_items=[
                        "Conduct resource utilization analysis",
                        "Implement resource monitoring",
                        "Gradually adjust resource allocations",
                    ],
                    estimated_impact={
                        "cost_reduction": "15-30%",
                        "performance_impact": "requires monitoring",
                    },
                )
            )

        return recommendations

    def implement_recommendation(
        self, recommendation: OptimizationRecommendation
    ) -> dict[str, Any]:
        """Simulate implementing a recommendation"""
        implementation_result = {
            "recommendation": recommendation.title,
            "status": "implemented",
            "timestamp": datetime.utcnow().isoformat(),
            "estimated_savings": recommendation.potential_savings,
            "implementation_notes": f"Simulated implementation of {recommendation.title}",
        }

        self.optimization_history.append(implementation_result)
        logger.info(f"Implemented optimization: {recommendation.title}")

        return implementation_result


class BudgetManager:
    """Manage budgets and cost controls"""

    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker
        self.budget_policies: dict[str, dict[str, Any]] = {}
        self.spending_forecasts: dict[str, float] = {}

    def create_budget_policy(
        self, name: str, limits: dict[str, float], actions: list[str]
    ):
        """Create a budget policy"""
        self.budget_policies[name] = {
            "limits": limits,
            "actions": actions,
            "created": datetime.utcnow().isoformat(),
            "active": True,
        }

    def forecast_spending(self, days: int = 30) -> dict[str, float]:
        """Forecast spending based on current trends"""
        trends = self.cost_tracker.get_cost_trends(7)

        if trends.get("trend") == "insufficient_data":
            return {"error": "Insufficient data for forecasting"}

        daily_avg = trends.get("average_daily_cost", 0)

        # Simple linear projection
        forecasts = {
            "daily": daily_avg,
            "weekly": daily_avg * 7,
            "monthly": daily_avg * 30,
            f"{days}_day_forecast": daily_avg * days,
        }

        # Adjust based on trend
        trend = trends.get("trend", "stable")
        if trend == "increasing":
            multiplier = 1.2
        elif trend == "decreasing":
            multiplier = 0.8
        else:
            multiplier = 1.0

        for key in forecasts:
            forecasts[key] *= multiplier

        self.spending_forecasts = forecasts
        return forecasts

    def get_budget_status(self) -> dict[str, Any]:
        """Get current budget status"""
        current_costs = {
            "daily": self.cost_tracker.get_total_cost(24),
            "weekly": self.cost_tracker.get_total_cost(24 * 7),
            "monthly": self.cost_tracker.get_total_cost(24 * 30),
        }

        forecasts = self.forecast_spending()

        status = {
            "current_spending": current_costs,
            "forecasted_spending": forecasts,
            "budget_limits": self.cost_tracker.budget_limits,
            "policies": self.budget_policies,
            "alerts": [],
        }

        # Check for budget violations
        for period, limit in self.cost_tracker.budget_limits.items():
            current = current_costs.get(period, 0)
            forecast = forecasts.get(period, 0)

            if current > limit:
                status["alerts"].append(
                    {
                        "type": "budget_exceeded",
                        "period": period,
                        "current": current,
                        "limit": limit,
                        "overage": current - limit,
                    }
                )
            elif forecast > limit:
                status["alerts"].append(
                    {
                        "type": "budget_forecast_exceeded",
                        "period": period,
                        "forecast": forecast,
                        "limit": limit,
                        "projected_overage": forecast - limit,
                    }
                )

        return status


class CostOptimizationManager:
    """Comprehensive cost optimization management"""

    def __init__(self):
        self.cost_tracker = CostTracker()
        self.resource_optimizer = ResourceOptimizer(self.cost_tracker)
        self.budget_manager = BudgetManager(self.cost_tracker)
        self.optimization_schedule: list[dict[str, Any]] = []

    def setup_default_budgets(self):
        """Set up default budget limits"""
        self.cost_tracker.set_budget_limit("daily", 50.0)
        self.cost_tracker.set_budget_limit("weekly", 300.0)
        self.cost_tracker.set_budget_limit("monthly", 1200.0)

        self.budget_manager.create_budget_policy(
            "default_policy",
            {"daily": 50.0, "weekly": 300.0, "monthly": 1200.0},
            ["alert", "scale_down", "pause_non_critical"],
        )

    def get_comprehensive_report(self) -> dict[str, Any]:
        """Get comprehensive cost optimization report"""
        recommendations = self.resource_optimizer.generate_recommendations()
        budget_status = self.budget_manager.get_budget_status()
        usage_patterns = self.resource_optimizer.analyze_usage_patterns()
        cost_trends = self.cost_tracker.get_cost_trends()

        return {
            "summary": {
                "total_daily_cost": self.cost_tracker.get_total_cost(24),
                "total_weekly_cost": self.cost_tracker.get_total_cost(24 * 7),
                "total_monthly_cost": self.cost_tracker.get_total_cost(24 * 30),
                "cost_trend": cost_trends.get("trend", "unknown"),
                "optimization_opportunities": len(recommendations),
                "potential_savings": sum(r.potential_savings for r in recommendations),
            },
            "recommendations": [
                {
                    "title": r.title,
                    "description": r.description,
                    "potential_savings": r.potential_savings,
                    "effort": r.implementation_effort,
                    "risk": r.risk_level,
                    "category": r.category,
                }
                for r in recommendations[:10]  # Top 10 recommendations
            ],
            "budget_status": budget_status,
            "usage_patterns": usage_patterns,
            "cost_trends": cost_trends,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def run_optimization_cycle(self):
        """Run a complete optimization cycle"""
        logger.info("Starting cost optimization cycle")

        # Generate recommendations
        recommendations = self.resource_optimizer.generate_recommendations(
            OptimizationStrategy.BALANCED
        )

        # Implement low-risk, high-impact recommendations automatically
        auto_implement = [
            r
            for r in recommendations
            if r.risk_level == "low" and r.potential_savings > 5.0
        ]

        for rec in auto_implement:
            result = self.resource_optimizer.implement_recommendation(rec)
            logger.info(
                f"Auto-implemented: {rec.title} (savings: ${rec.potential_savings:.2f})"
            )

        # Log summary
        total_savings = sum(r.potential_savings for r in auto_implement)
        logger.info(
            f"Optimization cycle completed. Potential savings: ${total_savings:.2f}"
        )

        return {
            "recommendations_generated": len(recommendations),
            "auto_implemented": len(auto_implement),
            "potential_savings": total_savings,
            "timestamp": datetime.utcnow().isoformat(),
        }


async def main():
    """Demonstrate cost optimization tools"""
    print("=== DSPy Cost Optimization Demo ===")

    # Create cost optimization manager
    cost_manager = CostOptimizationManager()
    cost_manager.setup_default_budgets()

    print("Setting up cost tracking and optimization...")

    # Simulate some usage
    print("\nSimulating resource usage...")
    for i in range(20):
        # Simulate compute usage
        cost_manager.cost_tracker.record_usage(
            ResourceType.COMPUTE,
            2.5 + (i * 0.1),
            {"instance_type": "standard", "region": "us-east-1"},
        )

        # Simulate API calls
        cost_manager.cost_tracker.record_usage(
            ResourceType.API_CALLS,
            1000 + (i * 50),
            {"model": "gpt-4o", "endpoint": "completions"},
        )

        # Simulate token usage
        cost_manager.cost_tracker.record_usage(
            ResourceType.TOKENS,
            5000 + (i * 200),
            {"input_tokens": 3000, "output_tokens": 2000},
        )

        await asyncio.sleep(0.1)  # Small delay for realistic timestamps

    # Run optimization cycle
    print("\nRunning cost optimization cycle...")
    optimization_result = await cost_manager.run_optimization_cycle()

    print(
        f"Generated {optimization_result['recommendations_generated']} recommendations"
    )
    print(f"Auto-implemented {optimization_result['auto_implemented']} optimizations")
    print(f"Potential savings: ${optimization_result['potential_savings']:.2f}")

    # Get comprehensive report
    print("\nGenerating comprehensive cost report...")
    report = cost_manager.get_comprehensive_report()

    print(f"\n=== Cost Summary ===")
    print(f"Daily cost: ${report['summary']['total_daily_cost']:.2f}")
    print(f"Weekly cost: ${report['summary']['total_weekly_cost']:.2f}")
    print(f"Monthly cost: ${report['summary']['total_monthly_cost']:.2f}")
    print(f"Cost trend: {report['summary']['cost_trend']}")
    print(
        f"Optimization opportunities: {report['summary']['optimization_opportunities']}"
    )
    print(f"Total potential savings: ${report['summary']['potential_savings']:.2f}")

    print(f"\n=== Top Recommendations ===")
    for i, rec in enumerate(report["recommendations"][:5], 1):
        print(f"{i}. {rec['title']}")
        print(
            f"   Savings: ${rec['potential_savings']:.2f} | Effort: {rec['effort']} | Risk: {rec['risk']}"
        )
        print(f"   {rec['description']}")

    # Show budget status
    budget_status = report["budget_status"]
    if budget_status["alerts"]:
        print(f"\n=== Budget Alerts ===")
        for alert in budget_status["alerts"]:
            print(f"- {alert['type']}: {alert['period']} period")
    else:
        print(f"\n=== Budget Status ===")
        print("All budgets are within limits")

    print(f"\nCost optimization demonstration completed!")


if __name__ == "__main__":
    asyncio.run(main())
