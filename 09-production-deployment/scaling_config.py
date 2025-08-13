#!/usr/bin/env python3
"""
Scaling Configuration and Management System

This module provides configuration management, policy definitions,
and deployment templates for DSPy scaling strategies.

Author: DSPy Learning Framework
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class DeploymentEnvironment(Enum):
    """Deployment environments"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ScalingTrigger(Enum):
    """Scaling triggers"""

    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingRule:
    """Individual scaling rule configuration"""

    name: str
    trigger: ScalingTrigger
    threshold: float
    comparison: str  # "greater_than", "less_than", "equals"
    action: str  # "scale_up", "scale_down"
    scale_amount: int
    cooldown_seconds: int = 300
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadBalancerConfig:
    """Load balancer configuration"""

    strategy: str = "intelligent"
    health_check_interval: int = 30
    health_check_timeout: int = 5
    health_check_path: str = "/health"
    session_affinity: bool = False
    connection_timeout: int = 30
    max_connections_per_node: int = 1000
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class AutoScalingConfig:
    """Auto-scaling configuration"""

    enabled: bool = True
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_cooldown: int = 300
    scale_down_cooldown: int = 600
    scaling_rules: list[ScalingRule] = field(default_factory=list)
    predictive_scaling: bool = False
    scheduled_scaling: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ResourceLimits:
    """Resource limits configuration"""

    cpu_request: str = "100m"
    cpu_limit: str = "1000m"
    memory_request: str = "128Mi"
    memory_limit: str = "1Gi"
    storage_request: str = "1Gi"
    storage_limit: str = "10Gi"


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""

    enabled: bool = True
    metrics_interval: int = 30
    retention_days: int = 30
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    dashboard_enabled: bool = True
    dashboard_port: int = 8080
    alert_channels: list[str] = field(default_factory=lambda: ["log"])
    custom_metrics: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CostOptimizationConfig:
    """Cost optimization configuration"""

    enabled: bool = True
    budget_limits: dict[str, float] = field(default_factory=dict)
    optimization_strategy: str = "balanced"
    spot_instances_enabled: bool = False
    scheduled_shutdown: dict[str, Any] = field(default_factory=dict)
    resource_rightsizing: bool = True
    cost_alerts_enabled: bool = True


@dataclass
class ScalingConfiguration:
    """Complete scaling configuration"""

    environment: DeploymentEnvironment
    service_name: str
    version: str = "1.0.0"
    auto_scaling: AutoScalingConfig = field(default_factory=AutoScalingConfig)
    load_balancer: LoadBalancerConfig = field(default_factory=LoadBalancerConfig)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cost_optimization: CostOptimizationConfig = field(
        default_factory=CostOptimizationConfig
    )
    custom_settings: dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """Manage scaling configurations"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.configurations: dict[str, ScalingConfiguration] = {}

    def create_default_config(
        self, environment: DeploymentEnvironment, service_name: str
    ) -> ScalingConfiguration:
        """Create default configuration for an environment"""

        # Environment-specific defaults
        if environment == DeploymentEnvironment.DEVELOPMENT:
            auto_scaling = AutoScalingConfig(
                min_instances=1,
                max_instances=3,
                target_cpu_utilization=80.0,
                scale_up_cooldown=180,
                scale_down_cooldown=300,
            )
            resource_limits = ResourceLimits(
                cpu_request="50m",
                cpu_limit="500m",
                memory_request="64Mi",
                memory_limit="512Mi",
            )
            cost_optimization = CostOptimizationConfig(
                budget_limits={"daily": 10.0, "monthly": 200.0},
                spot_instances_enabled=True,
            )

        elif environment == DeploymentEnvironment.STAGING:
            auto_scaling = AutoScalingConfig(
                min_instances=2,
                max_instances=5,
                target_cpu_utilization=70.0,
                predictive_scaling=True,
            )
            resource_limits = ResourceLimits(
                cpu_request="100m",
                cpu_limit="1000m",
                memory_request="128Mi",
                memory_limit="1Gi",
            )
            cost_optimization = CostOptimizationConfig(
                budget_limits={"daily": 25.0, "monthly": 500.0},
                optimization_strategy="balanced",
            )

        else:  # PRODUCTION
            auto_scaling = AutoScalingConfig(
                min_instances=3,
                max_instances=20,
                target_cpu_utilization=60.0,
                predictive_scaling=True,
                scaling_rules=[
                    ScalingRule(
                        name="high_cpu_scale_up",
                        trigger=ScalingTrigger.CPU_UTILIZATION,
                        threshold=80.0,
                        comparison="greater_than",
                        action="scale_up",
                        scale_amount=2,
                        cooldown_seconds=300,
                    ),
                    ScalingRule(
                        name="low_cpu_scale_down",
                        trigger=ScalingTrigger.CPU_UTILIZATION,
                        threshold=30.0,
                        comparison="less_than",
                        action="scale_down",
                        scale_amount=1,
                        cooldown_seconds=600,
                    ),
                    ScalingRule(
                        name="high_response_time",
                        trigger=ScalingTrigger.RESPONSE_TIME,
                        threshold=2.0,
                        comparison="greater_than",
                        action="scale_up",
                        scale_amount=3,
                        cooldown_seconds=180,
                    ),
                ],
            )
            resource_limits = ResourceLimits(
                cpu_request="200m",
                cpu_limit="2000m",
                memory_request="256Mi",
                memory_limit="2Gi",
            )
            cost_optimization = CostOptimizationConfig(
                budget_limits={"daily": 100.0, "monthly": 2000.0},
                optimization_strategy="conservative",
                resource_rightsizing=True,
            )

        config = ScalingConfiguration(
            environment=environment,
            service_name=service_name,
            auto_scaling=auto_scaling,
            resource_limits=resource_limits,
            cost_optimization=cost_optimization,
        )

        return config

    def save_config(self, config: ScalingConfiguration, filename: str | None = None):
        """Save configuration to file"""
        if not filename:
            filename = f"{config.service_name}_{config.environment.value}.yaml"

        filepath = self.config_dir / filename

        # Convert to dict and handle enums
        config_dict = asdict(config)
        config_dict["environment"] = config.environment.value

        # Convert scaling rules
        if config_dict["auto_scaling"]["scaling_rules"]:
            for rule in config_dict["auto_scaling"]["scaling_rules"]:
                rule["trigger"] = (
                    rule["trigger"].value
                    if hasattr(rule["trigger"], "value")
                    else rule["trigger"]
                )

        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        self.configurations[filename] = config
        return filepath

    def load_config(self, filename: str) -> ScalingConfiguration:
        """Load configuration from file"""
        filepath = self.config_dir / filename

        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert environment back to enum
        config_dict["environment"] = DeploymentEnvironment(config_dict["environment"])

        # Convert scaling rules back
        if config_dict["auto_scaling"]["scaling_rules"]:
            scaling_rules = []
            for rule_dict in config_dict["auto_scaling"]["scaling_rules"]:
                rule_dict["trigger"] = ScalingTrigger(rule_dict["trigger"])
                scaling_rules.append(ScalingRule(**rule_dict))
            config_dict["auto_scaling"]["scaling_rules"] = scaling_rules

        # Reconstruct nested objects
        config_dict["auto_scaling"] = AutoScalingConfig(**config_dict["auto_scaling"])
        config_dict["load_balancer"] = LoadBalancerConfig(
            **config_dict["load_balancer"]
        )
        config_dict["resource_limits"] = ResourceLimits(
            **config_dict["resource_limits"]
        )
        config_dict["monitoring"] = MonitoringConfig(**config_dict["monitoring"])
        config_dict["cost_optimization"] = CostOptimizationConfig(
            **config_dict["cost_optimization"]
        )

        config = ScalingConfiguration(**config_dict)
        self.configurations[filename] = config

        return config

    def validate_config(self, config: ScalingConfiguration) -> list[str]:
        """Validate configuration and return any issues"""
        issues = []

        # Validate auto-scaling settings
        if config.auto_scaling.min_instances < 1:
            issues.append("Minimum instances must be at least 1")

        if config.auto_scaling.max_instances < config.auto_scaling.min_instances:
            issues.append("Maximum instances must be greater than minimum instances")

        if (
            config.auto_scaling.target_cpu_utilization <= 0
            or config.auto_scaling.target_cpu_utilization > 100
        ):
            issues.append("Target CPU utilization must be between 0 and 100")

        # Validate resource limits
        try:
            cpu_request = self._parse_resource(config.resource_limits.cpu_request)
            cpu_limit = self._parse_resource(config.resource_limits.cpu_limit)
            if cpu_limit < cpu_request:
                issues.append("CPU limit must be greater than or equal to CPU request")
        except ValueError as e:
            issues.append(f"Invalid CPU resource format: {e}")

        try:
            memory_request = self._parse_memory(config.resource_limits.memory_request)
            memory_limit = self._parse_memory(config.resource_limits.memory_limit)
            if memory_limit < memory_request:
                issues.append(
                    "Memory limit must be greater than or equal to memory request"
                )
        except ValueError as e:
            issues.append(f"Invalid memory resource format: {e}")

        # Validate scaling rules
        for rule in config.auto_scaling.scaling_rules:
            if rule.threshold <= 0:
                issues.append(f"Scaling rule '{rule.name}' threshold must be positive")

            if rule.scale_amount <= 0:
                issues.append(
                    f"Scaling rule '{rule.name}' scale amount must be positive"
                )

            if rule.cooldown_seconds < 0:
                issues.append(
                    f"Scaling rule '{rule.name}' cooldown must be non-negative"
                )

        return issues

    def _parse_resource(self, resource_str: str) -> float:
        """Parse CPU resource string to millicores"""
        if resource_str.endswith("m"):
            return float(resource_str[:-1])
        else:
            return float(resource_str) * 1000

    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string to bytes"""
        units = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4}

        for unit, multiplier in units.items():
            if memory_str.endswith(unit):
                return float(memory_str[: -len(unit)]) * multiplier

        # Assume bytes if no unit
        return float(memory_str)

    def generate_deployment_templates(
        self, config: ScalingConfiguration
    ) -> dict[str, str]:
        """Generate deployment templates for different platforms"""
        templates = {}

        # Kubernetes deployment template
        templates["kubernetes"] = self._generate_k8s_template(config)

        # Docker Compose template
        templates["docker_compose"] = self._generate_docker_compose_template(config)

        # Terraform template
        templates["terraform"] = self._generate_terraform_template(config)

        return templates

    def _generate_k8s_template(self, config: ScalingConfiguration) -> str:
        """Generate Kubernetes deployment template"""
        template = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {config.service_name}
  labels:
    app: {config.service_name}
    environment: {config.environment.value}
spec:
  replicas: {config.auto_scaling.min_instances}
  selector:
    matchLabels:
      app: {config.service_name}
  template:
    metadata:
      labels:
        app: {config.service_name}
    spec:
      containers:
      - name: {config.service_name}
        image: {config.service_name}:{config.version}
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: {config.resource_limits.cpu_request}
            memory: {config.resource_limits.memory_request}
          limits:
            cpu: {config.resource_limits.cpu_limit}
            memory: {config.resource_limits.memory_limit}
        livenessProbe:
          httpGet:
            path: {config.load_balancer.health_check_path}
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: {config.load_balancer.health_check_interval}
        readinessProbe:
          httpGet:
            path: {config.load_balancer.health_check_path}
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: {config.service_name}-service
spec:
  selector:
    app: {config.service_name}
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {config.service_name}-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {config.service_name}
  minReplicas: {config.auto_scaling.min_instances}
  maxReplicas: {config.auto_scaling.max_instances}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {int(config.auto_scaling.target_cpu_utilization)}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {int(config.auto_scaling.target_memory_utilization)}
  behavior:
    scaleUp:
      stabilizationWindowSeconds: {config.auto_scaling.scale_up_cooldown}
    scaleDown:
      stabilizationWindowSeconds: {config.auto_scaling.scale_down_cooldown}
"""
        return template.strip()

    def _generate_docker_compose_template(self, config: ScalingConfiguration) -> str:
        """Generate Docker Compose template"""
        template = f"""
version: '3.8'

services:
  {config.service_name}:
    image: {config.service_name}:{config.version}
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT={config.environment.value}
      - SERVICE_NAME={config.service_name}
    deploy:
      replicas: {config.auto_scaling.min_instances}
      resources:
        limits:
          cpus: '{float(config.resource_limits.cpu_limit.rstrip("m")) / 1000:.2f}'
          memory: {config.resource_limits.memory_limit}
        reservations:
          cpus: '{float(config.resource_limits.cpu_request.rstrip("m")) / 1000:.2f}'
          memory: {config.resource_limits.memory_request}
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000{config.load_balancer.health_check_path}"]
      interval: {config.load_balancer.health_check_interval}s
      timeout: {config.load_balancer.health_check_timeout}s
      retries: 3
      start_period: 30s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - {config.service_name}
    deploy:
      replicas: 1
      restart_policy:
        condition: on-failure

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "{config.monitoring.prometheus_port}:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    deploy:
      replicas: 1
"""
        return template.strip()

    def _generate_terraform_template(self, config: ScalingConfiguration) -> str:
        """Generate Terraform template for AWS"""
        template = f"""
# Auto Scaling Group
resource "aws_autoscaling_group" "{config.service_name}_asg" {{
  name                = "{config.service_name}-asg"
  vpc_zone_identifier = var.subnet_ids
  target_group_arns   = [aws_lb_target_group.{config.service_name}_tg.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300

  min_size         = {config.auto_scaling.min_instances}
  max_size         = {config.auto_scaling.max_instances}
  desired_capacity = {config.auto_scaling.min_instances}

  launch_template {{
    id      = aws_launch_template.{config.service_name}_lt.id
    version = "$Latest"
  }}

  tag {{
    key                 = "Name"
    value               = "{config.service_name}"
    propagate_at_launch = true
  }}

  tag {{
    key                 = "Environment"
    value               = "{config.environment.value}"
    propagate_at_launch = true
  }}
}}

# Launch Template
resource "aws_launch_template" "{config.service_name}_lt" {{
  name_prefix   = "{config.service_name}-"
  image_id      = var.ami_id
  instance_type = var.instance_type

  vpc_security_group_ids = [aws_security_group.{config.service_name}_sg.id]

  user_data = base64encode(templatefile("user_data.sh", {{
    service_name = "{config.service_name}"
    environment  = "{config.environment.value}"
  }}))

  tag_specifications {{
    resource_type = "instance"
    tags = {{
      Name        = "{config.service_name}"
      Environment = "{config.environment.value}"
    }}
  }}
}}

# Auto Scaling Policies
resource "aws_autoscaling_policy" "{config.service_name}_scale_up" {{
  name                   = "{config.service_name}-scale-up"
  scaling_adjustment     = 2
  adjustment_type        = "ChangeInCapacity"
  cooldown              = {config.auto_scaling.scale_up_cooldown}
  autoscaling_group_name = aws_autoscaling_group.{config.service_name}_asg.name
}}

resource "aws_autoscaling_policy" "{config.service_name}_scale_down" {{
  name                   = "{config.service_name}-scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown              = {config.auto_scaling.scale_down_cooldown}
  autoscaling_group_name = aws_autoscaling_group.{config.service_name}_asg.name
}}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "{config.service_name}_cpu_high" {{
  alarm_name          = "{config.service_name}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "{config.auto_scaling.target_cpu_utilization}"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.{config.service_name}_scale_up.arn]

  dimensions = {{
    AutoScalingGroupName = aws_autoscaling_group.{config.service_name}_asg.name
  }}
}}

resource "aws_cloudwatch_metric_alarm" "{config.service_name}_cpu_low" {{
  alarm_name          = "{config.service_name}-cpu-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "30"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.{config.service_name}_scale_down.arn]

  dimensions = {{
    AutoScalingGroupName = aws_autoscaling_group.{config.service_name}_asg.name
  }}
}}

# Load Balancer
resource "aws_lb" "{config.service_name}_lb" {{
  name               = "{config.service_name}-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.{config.service_name}_lb_sg.id]
  subnets            = var.subnet_ids

  enable_deletion_protection = false

  tags = {{
    Environment = "{config.environment.value}"
  }}
}}

resource "aws_lb_target_group" "{config.service_name}_tg" {{
  name     = "{config.service_name}-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = var.vpc_id

  health_check {{
    enabled             = true
    healthy_threshold   = 2
    interval            = {config.load_balancer.health_check_interval}
    matcher             = "200"
    path                = "{config.load_balancer.health_check_path}"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = {config.load_balancer.health_check_timeout}
    unhealthy_threshold = 2
  }}
}}

resource "aws_lb_listener" "{config.service_name}_listener" {{
  load_balancer_arn = aws_lb.{config.service_name}_lb.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {{
    type             = "forward"
    target_group_arn = aws_lb_target_group.{config.service_name}_tg.arn
  }}
}}
"""
        return template.strip()

    def export_config_summary(self, config: ScalingConfiguration) -> dict[str, Any]:
        """Export configuration summary for documentation"""
        return {
            "service_info": {
                "name": config.service_name,
                "version": config.version,
                "environment": config.environment.value,
            },
            "scaling_settings": {
                "auto_scaling_enabled": config.auto_scaling.enabled,
                "min_instances": config.auto_scaling.min_instances,
                "max_instances": config.auto_scaling.max_instances,
                "target_cpu": f"{config.auto_scaling.target_cpu_utilization}%",
                "target_memory": f"{config.auto_scaling.target_memory_utilization}%",
                "scaling_rules_count": len(config.auto_scaling.scaling_rules),
            },
            "resource_allocation": {
                "cpu_request": config.resource_limits.cpu_request,
                "cpu_limit": config.resource_limits.cpu_limit,
                "memory_request": config.resource_limits.memory_request,
                "memory_limit": config.resource_limits.memory_limit,
            },
            "load_balancing": {
                "strategy": config.load_balancer.strategy,
                "health_check_interval": f"{config.load_balancer.health_check_interval}s",
                "session_affinity": config.load_balancer.session_affinity,
            },
            "monitoring": {
                "enabled": config.monitoring.enabled,
                "prometheus_enabled": config.monitoring.prometheus_enabled,
                "dashboard_enabled": config.monitoring.dashboard_enabled,
                "retention_days": config.monitoring.retention_days,
            },
            "cost_optimization": {
                "enabled": config.cost_optimization.enabled,
                "strategy": config.cost_optimization.optimization_strategy,
                "spot_instances": config.cost_optimization.spot_instances_enabled,
                "budget_limits": config.cost_optimization.budget_limits,
            },
        }


def main():
    """Demonstrate configuration management"""
    print("=== DSPy Scaling Configuration Management Demo ===")

    # Create configuration manager
    config_manager = ConfigurationManager()

    # Create configurations for different environments
    environments = [
        DeploymentEnvironment.DEVELOPMENT,
        DeploymentEnvironment.STAGING,
        DeploymentEnvironment.PRODUCTION,
    ]

    service_name = "dspy-text-analyzer"

    for env in environments:
        print(f"\nCreating configuration for {env.value} environment...")

        # Create default config
        config = config_manager.create_default_config(env, service_name)

        # Validate configuration
        issues = config_manager.validate_config(config)
        if issues:
            print(f"Configuration issues found: {issues}")
        else:
            print("Configuration validation passed")

        # Save configuration
        config_file = config_manager.save_config(config)
        print(f"Configuration saved to: {config_file}")

        # Generate deployment templates
        templates = config_manager.generate_deployment_templates(config)
        print(f"Generated templates: {list(templates.keys())}")

        # Export summary
        summary = config_manager.export_config_summary(config)
        print(f"Configuration summary:")
        print(
            f"  - Instances: {summary['scaling_settings']['min_instances']}-{summary['scaling_settings']['max_instances']}"
        )
        print(f"  - CPU Target: {summary['scaling_settings']['target_cpu']}")
        print(
            f"  - Resources: {summary['resource_allocation']['cpu_request']}/{summary['resource_allocation']['memory_request']}"
        )
        print(f"  - Budget: {summary['cost_optimization']['budget_limits']}")

    print(f"\nConfiguration management demonstration completed!")
    print(f"Configuration files saved in: {config_manager.config_dir}")


if __name__ == "__main__":
    main()
