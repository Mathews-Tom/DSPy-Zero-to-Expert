#!/usr/bin/env python3
"""
Production Deployment Guide for DSPy Applications

This module provides comprehensive deployment utilities, containerization support,
and CI/CD integration for DSPy applications in production environments.

Learning Objectives:
- Understand production deployment patterns for DSPy applications
- Learn containerization best practices with Docker
- Implement CI/CD pipelines for automated deployment
- Create deployment validation and health check systems
- Master production configuration management

Author: DSPy Learning Framework
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentStrategy(Enum):
    """Deployment strategy types"""

    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"


@dataclass
class DeploymentConfig:
    """Configuration for deployment settings"""

    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    app_name: str
    version: str
    replicas: int = 3
    resources: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    health_check_path: str = "/health"
    readiness_probe_path: str = "/ready"
    port: int = 8000
    enable_monitoring: bool = True
    enable_logging: bool = True


class DockerManager:
    """Docker containerization management"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dockerfile_path = project_root / "Dockerfile"
        self.dockerignore_path = project_root / ".dockerignore"

    def create_dockerfile(self, config: DeploymentConfig) -> str:
        """Create optimized Dockerfile for DSPy applications"""
        dockerfile_content = f"""# Multi-stage build for DSPy application
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE {config.port}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{config.port}{config.health_check_path} || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{config.port}"]
"""

        with open(self.dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        logger.info(f"Created Dockerfile at {self.dockerfile_path}")
        return dockerfile_content

    def create_dockerignore(self) -> str:
        """Create .dockerignore file"""
        dockerignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Git
.git
.gitignore

# Documentation
docs/
*.md
!README.md

# Tests
tests/
*_test.py
test_*.py

# Development
.pytest_cache/
.coverage
htmlcov/
.tox/
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
"""

        with open(self.dockerignore_path, "w") as f:
            f.write(dockerignore_content)

        logger.info(f"Created .dockerignore at {self.dockerignore_path}")
        return dockerignore_content

    def build_image(self, config: DeploymentConfig, tag: Optional[str] = None) -> bool:
        """Build Docker image"""
        if not tag:
            tag = f"{config.app_name}:{config.version}"

        try:
            cmd = ["docker", "build", "-t", tag, str(self.project_root)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully built Docker image: {tag}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build Docker image: {e.stderr}")
            return False

    def push_image(
        self, config: DeploymentConfig, registry: str, tag: Optional[str] = None
    ) -> bool:
        """Push Docker image to registry"""
        if not tag:
            tag = f"{config.app_name}:{config.version}"

        registry_tag = f"{registry}/{tag}"

        try:
            # Tag for registry
            subprocess.run(["docker", "tag", tag, registry_tag], check=True)

            # Push to registry
            subprocess.run(["docker", "push", registry_tag], check=True)

            logger.info(f"Successfully pushed image to registry: {registry_tag}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push Docker image: {e}")
            return False


class KubernetesManager:
    """Kubernetes deployment management"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.k8s_dir = project_root / "k8s"
        self.k8s_dir.mkdir(exist_ok=True)

    def create_deployment_manifest(
        self, config: DeploymentConfig, image_tag: str
    ) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest"""
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{config.app_name}-deployment",
                "labels": {
                    "app": config.app_name,
                    "version": config.version,
                    "environment": config.environment.value,
                },
            },
            "spec": {
                "replicas": config.replicas,
                "strategy": {
                    "type": (
                        "RollingUpdate"
                        if config.strategy == DeploymentStrategy.ROLLING
                        else "Recreate"
                    ),
                    "rollingUpdate": (
                        {"maxUnavailable": 1, "maxSurge": 1}
                        if config.strategy == DeploymentStrategy.ROLLING
                        else None
                    ),
                },
                "selector": {"matchLabels": {"app": config.app_name}},
                "template": {
                    "metadata": {
                        "labels": {"app": config.app_name, "version": config.version}
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": config.app_name,
                                "image": image_tag,
                                "ports": [
                                    {"containerPort": config.port, "name": "http"}
                                ],
                                "env": [
                                    {"name": k, "value": v}
                                    for k, v in config.environment_variables.items()
                                ],
                                "resources": config.resources,
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": config.health_check_path,
                                        "port": config.port,
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                    "timeoutSeconds": 5,
                                    "failureThreshold": 3,
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": config.readiness_probe_path,
                                        "port": config.port,
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5,
                                    "timeoutSeconds": 3,
                                    "failureThreshold": 3,
                                },
                            }
                        ],
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000,
                        },
                    },
                },
            },
        }

        # Remove None values
        if manifest["spec"]["strategy"]["rollingUpdate"] is None:
            del manifest["spec"]["strategy"]["rollingUpdate"]

        return manifest

    def create_service_manifest(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create Kubernetes service manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{config.app_name}-service",
                "labels": {
                    "app": config.app_name,
                    "environment": config.environment.value,
                },
            },
            "spec": {
                "selector": {"app": config.app_name},
                "ports": [
                    {
                        "port": 80,
                        "targetPort": config.port,
                        "protocol": "TCP",
                        "name": "http",
                    }
                ],
                "type": "ClusterIP",
            },
        }

    def create_ingress_manifest(
        self, config: DeploymentConfig, domain: str
    ) -> Dict[str, Any]:
        """Create Kubernetes ingress manifest"""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{config.app_name}-ingress",
                "labels": {
                    "app": config.app_name,
                    "environment": config.environment.value,
                },
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true",
                },
            },
            "spec": {
                "tls": [{"hosts": [domain], "secretName": f"{config.app_name}-tls"}],
                "rules": [
                    {
                        "host": domain,
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": f"{config.app_name}-service",
                                            "port": {"number": 80},
                                        }
                                    },
                                }
                            ]
                        },
                    }
                ],
            },
        }

    def save_manifests(
        self, config: DeploymentConfig, image_tag: str, domain: Optional[str] = None
    ):
        """Save all Kubernetes manifests to files"""
        # Deployment manifest
        deployment = self.create_deployment_manifest(config, image_tag)
        deployment_path = self.k8s_dir / f"{config.app_name}-deployment.yaml"
        with open(deployment_path, "w") as f:
            yaml.dump(deployment, f, default_flow_style=False)

        # Service manifest
        service = self.create_service_manifest(config)
        service_path = self.k8s_dir / f"{config.app_name}-service.yaml"
        with open(service_path, "w") as f:
            yaml.dump(service, f, default_flow_style=False)

        # Ingress manifest (if domain provided)
        if domain:
            ingress = self.create_ingress_manifest(config, domain)
            ingress_path = self.k8s_dir / f"{config.app_name}-ingress.yaml"
            with open(ingress_path, "w") as f:
                yaml.dump(ingress, f, default_flow_style=False)

        logger.info(f"Saved Kubernetes manifests to {self.k8s_dir}")

    def deploy(self, config: DeploymentConfig) -> bool:
        """Deploy to Kubernetes cluster"""
        try:
            # Apply all manifests in k8s directory
            cmd = ["kubectl", "apply", "-f", str(self.k8s_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            logger.info(f"Successfully deployed {config.app_name} to Kubernetes")
            logger.info(f"Deployment output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to deploy to Kubernetes: {e.stderr}")
            return False


class CICDManager:
    """CI/CD pipeline management"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.github_dir = project_root / ".github" / "workflows"
        self.github_dir.mkdir(parents=True, exist_ok=True)

    def create_github_actions_workflow(
        self, config: DeploymentConfig, registry: str
    ) -> str:
        """Create GitHub Actions workflow for CI/CD"""
        workflow_content = f"""name: Deploy {config.app_name}

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: {registry}
  IMAGE_NAME: {config.app_name}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install uv
      run: pip install uv
    
    - name: Install dependencies
      run: uv sync
    
    - name: Run tests
      run: uv run pytest
    
    - name: Run linting
      run: |
        uv run ruff check .
        uv run black --check .
    
    - name: Run type checking
      run: uv run mypy .

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    outputs:
      image-tag: ${{{{ steps.meta.outputs.tags }}}}
      image-digest: ${{{{ steps.build.outputs.digest }}}}
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ github.actor }}}}
        password: ${{{{ secrets.GITHUB_TOKEN }}}}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{{{branch}}}}-
          type=raw,value=latest,enable={{{{is_default_branch}}}}
    
    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{{{ steps.meta.outputs.tags }}}}
        labels: ${{{{ steps.meta.outputs.labels }}}}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'latest'
    
    - name: Configure kubectl
      run: |
        echo "${{{{ secrets.KUBECONFIG }}}}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to staging
      run: |
        export KUBECONFIG=kubeconfig
        sed -i 's|IMAGE_TAG|${{{{ needs.build.outputs.image-tag }}}}|g' k8s/{config.app_name}-deployment.yaml
        kubectl apply -f k8s/ -n staging
        kubectl rollout status deployment/{config.app_name}-deployment -n staging

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'latest'
    
    - name: Configure kubectl
      run: |
        echo "${{{{ secrets.KUBECONFIG }}}}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to production
      run: |
        export KUBECONFIG=kubeconfig
        sed -i 's|IMAGE_TAG|${{{{ needs.build.outputs.image-tag }}}}|g' k8s/{config.app_name}-deployment.yaml
        kubectl apply -f k8s/ -n production
        kubectl rollout status deployment/{config.app_name}-deployment -n production
    
    - name: Run deployment validation
      run: |
        export KUBECONFIG=kubeconfig
        kubectl wait --for=condition=available --timeout=300s deployment/{config.app_name}-deployment -n production
        kubectl get pods -n production -l app={config.app_name}
"""

        workflow_path = self.github_dir / f"deploy-{config.app_name}.yml"
        with open(workflow_path, "w") as f:
            f.write(workflow_content)

        logger.info(f"Created GitHub Actions workflow at {workflow_path}")
        return workflow_content


class HealthCheckManager:
    """Health check and validation utilities"""

    @staticmethod
    def create_health_check_endpoint() -> str:
        """Create FastAPI health check endpoint"""
        return '''from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import psutil
import time
from datetime import datetime
from typing import Dict, Any

app = FastAPI()

@app.get("/health")
async def health_check() -> JSONResponse:
    """Basic health check endpoint"""
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Basic health indicators
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": time.time() - psutil.boot_time(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": (disk.used / disk.total) * 100,
                "available_memory_gb": memory.available / (1024**3)
            }
        }
        
        # Determine overall health
        if cpu_percent > 90 or memory.percent > 90 or (disk.used / disk.total) * 100 > 90:
            health_status["status"] = "degraded"
        
        return JSONResponse(
            status_code=200 if health_status["status"] == "healthy" else 503,
            content=health_status
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/ready")
async def readiness_check() -> JSONResponse:
    """Readiness check endpoint"""
    try:
        # Check if application is ready to serve requests
        # Add your specific readiness checks here
        
        readiness_status = {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "database": "connected",  # Replace with actual DB check
                "external_apis": "available",  # Replace with actual API checks
                "cache": "connected"  # Replace with actual cache check
            }
        }
        
        return JSONResponse(status_code=200, content=readiness_status)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
'''

    @staticmethod
    def validate_deployment(config: DeploymentConfig, endpoint: str) -> bool:
        """Validate deployment health"""
        import requests

        try:
            # Check health endpoint
            health_response = requests.get(
                f"{endpoint}{config.health_check_path}", timeout=30
            )
            health_response.raise_for_status()

            # Check readiness endpoint
            ready_response = requests.get(
                f"{endpoint}{config.readiness_probe_path}", timeout=30
            )
            ready_response.raise_for_status()

            logger.info("Deployment validation successful")
            return True
        except requests.RequestException as e:
            logger.error(f"Deployment validation failed: {e}")
            return False


class DeploymentAutomation:
    """Main deployment automation orchestrator"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docker_manager = DockerManager(project_root)
        self.k8s_manager = KubernetesManager(project_root)
        self.cicd_manager = CICDManager(project_root)

    def setup_deployment(
        self, config: DeploymentConfig, registry: str, domain: Optional[str] = None
    ) -> bool:
        """Complete deployment setup"""
        try:
            logger.info(f"Setting up deployment for {config.app_name}")

            # Create Docker files
            self.docker_manager.create_dockerfile(config)
            self.docker_manager.create_dockerignore()

            # Create Kubernetes manifests
            image_tag = f"{registry}/{config.app_name}:{config.version}"
            self.k8s_manager.save_manifests(config, image_tag, domain)

            # Create CI/CD pipeline
            self.cicd_manager.create_github_actions_workflow(config, registry)

            # Create health check endpoints
            health_check_code = HealthCheckManager.create_health_check_endpoint()
            health_check_path = self.project_root / "health_endpoints.py"
            with open(health_check_path, "w") as f:
                f.write(health_check_code)

            logger.info("Deployment setup completed successfully")
            return True
        except Exception as e:
            logger.error(f"Deployment setup failed: {e}")
            return False

    def deploy_application(
        self,
        config: DeploymentConfig,
        registry: str,
        build_image: bool = True,
        push_image: bool = True,
        deploy_k8s: bool = True,
    ) -> bool:
        """Deploy application with specified options"""
        try:
            logger.info(f"Deploying {config.app_name} to {config.environment.value}")

            # Build Docker image
            if build_image:
                if not self.docker_manager.build_image(config):
                    return False

            # Push to registry
            if push_image:
                if not self.docker_manager.push_image(config, registry):
                    return False

            # Deploy to Kubernetes
            if deploy_k8s:
                if not self.k8s_manager.deploy(config):
                    return False

            logger.info("Application deployment completed successfully")
            return True
        except Exception as e:
            logger.error(f"Application deployment failed: {e}")
            return False


def main():
    """Demonstration of deployment automation"""
    print("=== DSPy Production Deployment Automation ===")

    # Example deployment configuration
    config = DeploymentConfig(
        environment=DeploymentEnvironment.PRODUCTION,
        strategy=DeploymentStrategy.ROLLING,
        app_name="dspy-app",
        version="1.0.0",
        replicas=3,
        resources={
            "requests": {"cpu": "100m", "memory": "256Mi"},
            "limits": {"cpu": "500m", "memory": "512Mi"},
        },
        environment_variables={
            "ENVIRONMENT": "production",
            "LOG_LEVEL": "INFO",
            "OPENAI_API_KEY": "${OPENAI_API_KEY}",
            "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        },
    )

    # Initialize deployment automation
    project_root = Path.cwd()
    deployment = DeploymentAutomation(project_root)

    # Setup deployment files
    registry = "ghcr.io/your-org"
    domain = "your-app.example.com"

    print(f"Setting up deployment for {config.app_name}...")
    success = deployment.setup_deployment(config, registry, domain)

    if success:
        print("✅ Deployment setup completed successfully!")
        print("\nGenerated files:")
        print("- Dockerfile")
        print("- .dockerignore")
        print("- k8s/ directory with Kubernetes manifests")
        print("- .github/workflows/ directory with CI/CD pipeline")
        print("- health_endpoints.py with health check endpoints")

        print("\nNext steps:")
        print("1. Configure your container registry credentials")
        print("2. Set up Kubernetes cluster access")
        print("3. Configure GitHub Actions secrets")
        print("4. Push code to trigger deployment pipeline")
    else:
        print("❌ Deployment setup failed")


if __name__ == "__main__":
    main()
