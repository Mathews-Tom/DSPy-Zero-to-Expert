#!/usr/bin/env python3
"""
Complete Deployment Example for DSPy Applications

This example demonstrates a full deployment automation workflow including:
- Docker containerization
- Kubernetes deployment
- CI/CD pipeline setup
- Health checks and validation
- Monitoring setup

Author: DSPy Learning Framework
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from cicd_templates import (
    CICDConfig,
    CICDPlatform,
    CICDTemplateManager,
    DeploymentTarget,
)
from deployment_guide import (
    DeploymentAutomation,
    DeploymentConfig,
    DeploymentEnvironment,
    DeploymentStrategy,
)
from deployment_validation import (
    CheckType,
    DeploymentMonitor,
    DeploymentValidator,
    HealthCheck,
    KubernetesValidator,
)
from docker_templates import (
    ApplicationType,
    DockerTemplate,
    DockerTemplateGenerator,
    OptimizationLevel,
)


class CompleteDeploymentDemo:
    """Complete deployment demonstration"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.deployment_automation = DeploymentAutomation(project_root)

    def create_sample_dspy_app(self):
        """Create a sample DSPy application for deployment"""
        # Create main application file
        app_content = '''#!/usr/bin/env python3
"""
Sample DSPy Web API Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import dspy
import logging
from datetime import datetime
import psutil
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DSPy Sample API",
    description="A sample DSPy application for deployment demonstration",
    version="1.0.0"
)

# Configure DSPy (you would normally load this from environment)
try:
    # This would be configured with actual API keys in production
    logger.info("DSPy configuration would be loaded here")
except Exception as e:
    logger.warning(f"DSPy configuration failed: {e}")

# Sample DSPy signature
class TextAnalysisSignature(dspy.Signature):
    """Analyze text and provide insights"""
    text = dspy.InputField(desc="Text to analyze")
    sentiment = dspy.OutputField(desc="Sentiment: positive, negative, or neutral")
    summary = dspy.OutputField(desc="Brief summary of the text")

# Initialize DSPy module (with fallback)
try:
    text_analyzer = dspy.ChainOfThought(TextAnalysisSignature)
except Exception as e:
    logger.warning(f"DSPy module initialization failed: {e}")
    text_analyzer = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "DSPy Sample API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "available_memory_gb": memory.available / (1024**3)
            },
            "dspy_configured": text_analyzer is not None
        }
        
        # Determine overall health
        if cpu_percent > 90 or memory.percent > 90:
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
async def readiness_check():
    """Readiness check endpoint"""
    try:
        readiness_status = {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "dspy_module": "available" if text_analyzer else "unavailable",
                "api": "ready"
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

@app.post("/analyze")
async def analyze_text(request: dict):
    """Analyze text using DSPy"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        if text_analyzer:
            try:
                # Use DSPy for analysis
                result = text_analyzer(text=text)
                return {
                    "text": text,
                    "sentiment": result.sentiment,
                    "summary": result.summary,
                    "method": "dspy",
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.warning(f"DSPy analysis failed: {e}")
        
        # Fallback analysis
        word_count = len(text.split())
        sentiment = "neutral"  # Simple fallback
        summary = text[:100] + "..." if len(text) > 100 else text
        
        return {
            "text": text,
            "sentiment": sentiment,
            "summary": summary,
            "word_count": word_count,
            "method": "fallback",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

        app_path = self.project_root / "main.py"
        with open(app_path, "w") as f:
            f.write(app_content)

        # Create requirements file
        requirements_content = """fastapi==0.104.1
uvicorn[standard]==0.24.0
dspy-ai==2.4.9
psutil==5.9.6
aiohttp==3.9.1
"""

        requirements_path = self.project_root / "requirements.txt"
        with open(requirements_path, "w") as f:
            f.write(requirements_content)

        # Create pyproject.toml
        pyproject_content = """[project]
name = "dspy-sample-api"
version = "1.0.0"
description = "Sample DSPy API for deployment demonstration"
dependencies = [
    "fastapi>=0.104.1",
    "uvicorn[standard]>=0.24.0",
    "dspy-ai>=2.4.9",
    "psutil>=5.9.6",
    "aiohttp>=3.9.1",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.3",
    "pytest-asyncio>=0.21.1",
    "httpx>=0.25.2",
    "ruff>=0.1.6",
    "black>=23.11.0",
    "mypy>=1.7.1",
]
"""

        pyproject_path = self.project_root / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            f.write(pyproject_content)

        print("✅ Created sample DSPy application files")

    def setup_docker_configuration(self):
        """Set up Docker configuration"""
        print("\n=== Setting up Docker Configuration ===")

        # Create Docker template
        template = DockerTemplate(
            app_type=ApplicationType.WEB_API,
            optimization=OptimizationLevel.PRODUCTION,
            port=8000,
            environment_vars={"ENVIRONMENT": "production", "LOG_LEVEL": "INFO"},
        )

        # Generate Dockerfile
        dockerfile_content = DockerTemplateGenerator.generate_dockerfile(template)
        dockerfile_path = self.project_root / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        # Generate .dockerignore
        dockerignore_content = DockerTemplateGenerator.get_dockerignore_template(
            template
        )
        dockerignore_path = self.project_root / ".dockerignore"
        with open(dockerignore_path, "w") as f:
            f.write(dockerignore_content)

        # Generate docker-compose.yml for development
        compose_content = DockerTemplateGenerator.get_docker_compose_template(
            template, ["redis", "postgres"]
        )
        compose_path = self.project_root / "docker-compose.yml"
        with open(compose_path, "w") as f:
            f.write(compose_content)

        print("✅ Docker configuration created")
        print("  - Dockerfile")
        print("  - .dockerignore")
        print("  - docker-compose.yml")

    def setup_kubernetes_manifests(self):
        """Set up Kubernetes deployment manifests"""
        print("\n=== Setting up Kubernetes Manifests ===")

        # Create deployment configuration
        config = DeploymentConfig(
            environment=DeploymentEnvironment.PRODUCTION,
            strategy=DeploymentStrategy.ROLLING,
            app_name="dspy-sample-api",
            version="1.0.0",
            replicas=3,
            resources={
                "requests": {"cpu": "100m", "memory": "256Mi"},
                "limits": {"cpu": "500m", "memory": "512Mi"},
            },
            environment_variables={"ENVIRONMENT": "production", "LOG_LEVEL": "INFO"},
        )

        # Generate Kubernetes manifests
        registry = "ghcr.io/your-org"
        domain = "dspy-api.example.com"

        self.deployment_automation.k8s_manager.save_manifests(
            config, f"{registry}/{config.app_name}:IMAGE_TAG", domain
        )

        print("✅ Kubernetes manifests created in k8s/ directory")

    def setup_cicd_pipeline(self):
        """Set up CI/CD pipeline"""
        print("\n=== Setting up CI/CD Pipeline ===")

        # Create CI/CD configuration
        cicd_config = CICDConfig(
            platform=CICDPlatform.GITHUB_ACTIONS,
            target=DeploymentTarget.KUBERNETES,
            app_name="dspy-sample-api",
            registry="ghcr.io/your-org",
            environments=["staging", "production"],
            test_commands=[
                "pytest --cov=. --cov-report=xml",
                "ruff check .",
                "black --check .",
                "mypy .",
            ],
            secrets=["KUBECONFIG", "SLACK_WEBHOOK_URL"],
        )

        # Generate GitHub Actions workflow
        workflow_content = CICDTemplateManager.get_deployment_specific_pipeline(
            cicd_config
        )

        # Create .github/workflows directory
        workflows_dir = self.project_root / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        workflow_path = workflows_dir / "deploy.yml"
        with open(workflow_path, "w") as f:
            f.write(workflow_content)

        print("✅ CI/CD pipeline created")
        print("  - .github/workflows/deploy.yml")

    async def setup_health_checks(self):
        """Set up health check and monitoring system"""
        print("\n=== Setting up Health Checks ===")

        # Configure health checks
        health_checks = [
            HealthCheck(
                name="api_health",
                check_type=CheckType.HTTP,
                endpoint="http://localhost:8000/health",
                timeout=10,
                critical=True,
            ),
            HealthCheck(
                name="api_ready",
                check_type=CheckType.HTTP,
                endpoint="http://localhost:8000/ready",
                timeout=10,
                critical=True,
            ),
            HealthCheck(
                name="system_resources", check_type=CheckType.RESOURCE, critical=False
            ),
        ]

        # Create validator
        validator = DeploymentValidator(health_checks)

        # Create monitoring configuration script
        monitoring_script = '''#!/usr/bin/env python3
"""
Production Monitoring Script
"""

import asyncio
import logging
from deployment_validation import (
    HealthCheck, CheckType, DeploymentValidator,
    DeploymentMonitor, HealthStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def alert_callback(status, message, results):
    """Handle health alerts"""
    if status == HealthStatus.UNHEALTHY:
        logger.error(f"CRITICAL ALERT: {message}")
        # Here you would integrate with your alerting system
        # - Send Slack notification
        # - Trigger PagerDuty alert
        # - Send email notification
    elif status == HealthStatus.DEGRADED:
        logger.warning(f"WARNING: {message}")

async def main():
    """Start monitoring"""
    health_checks = [
        HealthCheck(
            name="api_health",
            check_type=CheckType.HTTP,
            endpoint="http://localhost:8000/health",
            timeout=10,
            critical=True
        ),
        HealthCheck(
            name="api_ready",
            check_type=CheckType.HTTP,
            endpoint="http://localhost:8000/ready",
            timeout=10,
            critical=True
        ),
        HealthCheck(
            name="system_resources",
            check_type=CheckType.RESOURCE,
            critical=False
        )
    ]
    
    validator = DeploymentValidator(health_checks)
    monitor = DeploymentMonitor(validator, check_interval=60)
    monitor.add_alert_callback(alert_callback)
    
    logger.info("Starting production monitoring...")
    await monitor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
'''

        monitoring_path = self.project_root / "monitoring.py"
        with open(monitoring_path, "w") as f:
            f.write(monitoring_script)

        print("✅ Health check system configured")
        print("  - monitoring.py script created")

        # Run a sample health check
        print("\nRunning sample health checks...")
        try:
            results = await validator.run_health_checks()
            report = validator.generate_health_report(results)

            print(f"Health Status: {report['overall_status']}")
            print(f"Message: {report['overall_message']}")
            print(f"Checks completed: {report['summary']['total_checks']}")
        except Exception as e:
            print(f"Health check failed (expected if app not running): {e}")

    def create_deployment_documentation(self):
        """Create deployment documentation"""
        print("\n=== Creating Deployment Documentation ===")

        readme_content = """# DSPy Sample API - Deployment Guide

This is a sample DSPy application demonstrating production deployment best practices.

## Quick Start

### Local Development

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the application:
   ```bash
   uv run python main.py
   ```

3. Test the API:
   ```bash
   curl http://localhost:8000/health
   ```

### Docker Development

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. Test the containerized API:
   ```bash
   curl http://localhost:8000/health
   ```

## Production Deployment

### Prerequisites

- Docker and Docker registry access
- Kubernetes cluster access
- kubectl configured
- GitHub Actions secrets configured

### Deployment Process

1. **Push to GitHub**: The CI/CD pipeline will automatically:
   - Run tests and security scans
   - Build and push Docker image
   - Deploy to staging (develop branch)
   - Deploy to production (main branch, manual approval)

2. **Manual Deployment**:
   ```bash
   # Build Docker image
   docker build -t dspy-sample-api:latest .
   
   # Deploy to Kubernetes
   kubectl apply -f k8s/
   
   # Check deployment status
   kubectl rollout status deployment/dspy-sample-api-deployment
   ```

### Health Checks

The application provides health check endpoints:

- `/health` - Overall application health
- `/ready` - Readiness for traffic
- `/` - Basic connectivity test

### Monitoring

Start the monitoring system:
```bash
python monitoring.py
```

This will continuously monitor the application and alert on issues.

## Configuration

### Environment Variables

- `ENVIRONMENT`: Deployment environment (development/staging/production)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `OPENAI_API_KEY`: OpenAI API key for DSPy (optional)
- `ANTHROPIC_API_KEY`: Anthropic API key for DSPy (optional)

### Kubernetes Resources

- **CPU**: 100m request, 500m limit
- **Memory**: 256Mi request, 512Mi limit
- **Replicas**: 3 (production)

## Troubleshooting

### Common Issues

1. **Pod not starting**: Check resource limits and image availability
2. **Health checks failing**: Verify application configuration
3. **High resource usage**: Check for memory leaks or CPU-intensive operations

### Debugging Commands

```bash
# Check pod status
kubectl get pods -l app=dspy-sample-api

# View pod logs
kubectl logs -l app=dspy-sample-api

# Check service endpoints
kubectl get endpoints dspy-sample-api-service

# Port forward for local testing
kubectl port-forward service/dspy-sample-api-service 8000:80
```

## Security

- Application runs as non-root user
- Resource limits enforced
- Security scanning in CI/CD pipeline
- TLS termination at ingress

## Scaling

The application is configured for horizontal scaling:

```bash
# Scale replicas
kubectl scale deployment dspy-sample-api-deployment --replicas=5

# Auto-scaling (optional)
kubectl autoscale deployment dspy-sample-api-deployment --min=3 --max=10 --cpu-percent=70
```
"""

        readme_path = self.project_root / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        print("✅ Deployment documentation created")
        print("  - README.md with comprehensive deployment guide")

    async def run_complete_demo(self):
        """Run the complete deployment demonstration"""
        print("🚀 DSPy Production Deployment - Complete Example")
        print("=" * 60)

        try:
            # Step 1: Create sample application
            self.create_sample_dspy_app()

            # Step 2: Set up Docker configuration
            self.setup_docker_configuration()

            # Step 3: Set up Kubernetes manifests
            self.setup_kubernetes_manifests()

            # Step 4: Set up CI/CD pipeline
            self.setup_cicd_pipeline()

            # Step 5: Set up health checks and monitoring
            await self.setup_health_checks()

            # Step 6: Create documentation
            self.create_deployment_documentation()

            print("\n" + "=" * 60)
            print("🎉 Complete Deployment Setup Finished!")
            print("\nGenerated Files:")
            print("📁 Application:")
            print("  - main.py (FastAPI application)")
            print("  - pyproject.toml (dependencies)")
            print("  - requirements.txt (pip dependencies)")

            print("\n📁 Docker:")
            print("  - Dockerfile (production-optimized)")
            print("  - .dockerignore (build optimization)")
            print("  - docker-compose.yml (development)")

            print("\n📁 Kubernetes:")
            print("  - k8s/dspy-sample-api-deployment.yaml")
            print("  - k8s/dspy-sample-api-service.yaml")
            print("  - k8s/dspy-sample-api-ingress.yaml")

            print("\n📁 CI/CD:")
            print("  - .github/workflows/deploy.yml")

            print("\n📁 Monitoring:")
            print("  - monitoring.py (health check system)")

            print("\n📁 Documentation:")
            print("  - README.md (deployment guide)")

            print("\n🔧 Next Steps:")
            print("1. Configure your container registry credentials")
            print("2. Set up Kubernetes cluster access")
            print("3. Configure GitHub Actions secrets:")
            print("   - KUBECONFIG (base64 encoded)")
            print("   - SLACK_WEBHOOK_URL (optional)")
            print("4. Push code to GitHub to trigger deployment")
            print("5. Monitor deployment with: python monitoring.py")

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise


async def main():
    """Run the complete deployment example"""
    project_root = Path.cwd() / "sample-dspy-app"
    project_root.mkdir(exist_ok=True)

    demo = CompleteDeploymentDemo(project_root)
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
