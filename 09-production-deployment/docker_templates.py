#!/usr/bin/env python3
"""
Docker Templates and Configuration for DSPy Applications

This module provides Docker configuration templates optimized for different
deployment scenarios and DSPy application types.

Author: DSPy Learning Framework
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ApplicationType(Enum):
    """Types of DSPy applications"""

    WEB_API = "web_api"
    BATCH_PROCESSOR = "batch_processor"
    STREAMING = "streaming"
    NOTEBOOK_SERVER = "notebook_server"


class OptimizationLevel(Enum):
    """Docker optimization levels"""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    MINIMAL = "minimal"


@dataclass
class DockerTemplate:
    """Docker template configuration"""

    app_type: ApplicationType
    optimization: OptimizationLevel
    python_version: str = "3.11"
    base_image: str = "python:3.11-slim"
    port: int = 8000
    additional_packages: List[str] = None
    environment_vars: Dict[str, str] = None


class DockerTemplateGenerator:
    """Generate Docker templates for different application types"""

    @staticmethod
    def get_base_dockerfile(template: DockerTemplate) -> str:
        """Generate base Dockerfile content"""
        additional_packages = template.additional_packages or []
        env_vars = template.environment_vars or {}

        # System packages based on application type
        system_packages = {
            ApplicationType.WEB_API: ["curl", "nginx"],
            ApplicationType.BATCH_PROCESSOR: ["cron"],
            ApplicationType.STREAMING: ["curl"],
            ApplicationType.NOTEBOOK_SERVER: ["curl", "nodejs", "npm"],
        }

        packages = system_packages.get(template.app_type, ["curl"])
        packages.extend(additional_packages)
        packages_str = " \\\n    ".join(packages)

        # Environment variables
        env_section = ""
        if env_vars:
            env_section = "\n".join([f"ENV {k}={v}" for k, v in env_vars.items()])
            env_section = f"\n# Application environment variables\n{env_section}\n"

        return f"""# {template.app_type.value.title()} DSPy Application
# Optimization level: {template.optimization.value}

FROM {template.base_image} as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
{env_section}
# Install system dependencies
RUN apt-get update && apt-get install -y \\
    {packages_str} \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Install uv for fast Python package management
RUN pip install uv

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --frozen {"--no-dev" if template.optimization == OptimizationLevel.PRODUCTION else ""}

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE {template.port}
"""

    @staticmethod
    def get_web_api_dockerfile(template: DockerTemplate) -> str:
        """Generate Dockerfile for web API applications"""
        base = DockerTemplateGenerator.get_base_dockerfile(template)

        web_specific = f"""
# Health check for web API
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{template.port}/health || exit 1

# Run web application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{template.port}"]
"""
        return base + web_specific

    @staticmethod
    def get_batch_processor_dockerfile(template: DockerTemplate) -> str:
        """Generate Dockerfile for batch processing applications"""
        base = DockerTemplateGenerator.get_base_dockerfile(template)

        batch_specific = """
# Create directories for batch processing
RUN mkdir -p /app/data /app/logs /app/output

# Health check for batch processor
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \\
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run batch processor
CMD ["python", "batch_processor.py"]
"""
        return base + batch_specific

    @staticmethod
    def get_streaming_dockerfile(template: DockerTemplate) -> str:
        """Generate Dockerfile for streaming applications"""
        base = DockerTemplateGenerator.get_base_dockerfile(template)

        streaming_specific = f"""
# Health check for streaming application
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \\
    CMD curl -f http://localhost:{template.port}/health || exit 1

# Run streaming application
CMD ["python", "streaming_processor.py"]
"""
        return base + streaming_specific

    @staticmethod
    def get_notebook_server_dockerfile(template: DockerTemplate) -> str:
        """Generate Dockerfile for notebook server applications"""
        base = DockerTemplateGenerator.get_base_dockerfile(template)

        notebook_specific = f"""
# Install Jupyter and Marimo
RUN uv add jupyter marimo

# Create notebook directory
RUN mkdir -p /app/notebooks

# Health check for notebook server
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \\
    CMD curl -f http://localhost:{template.port}/ || exit 1

# Run notebook server
CMD ["marimo", "edit", "--host", "0.0.0.0", "--port", "{template.port}", "--no-token"]
"""
        return base + notebook_specific

    @staticmethod
    def generate_dockerfile(template: DockerTemplate) -> str:
        """Generate complete Dockerfile based on template"""
        generators = {
            ApplicationType.WEB_API: DockerTemplateGenerator.get_web_api_dockerfile,
            ApplicationType.BATCH_PROCESSOR: DockerTemplateGenerator.get_batch_processor_dockerfile,
            ApplicationType.STREAMING: DockerTemplateGenerator.get_streaming_dockerfile,
            ApplicationType.NOTEBOOK_SERVER: DockerTemplateGenerator.get_notebook_server_dockerfile,
        }

        generator = generators.get(
            template.app_type, DockerTemplateGenerator.get_web_api_dockerfile
        )
        return generator(template)

    @staticmethod
    def get_docker_compose_template(
        template: DockerTemplate, services: List[str] = None
    ) -> str:
        """Generate docker-compose.yml template"""
        services = services or []

        # Base service configuration
        app_service = f"""  app:
    build: .
    ports:
      - "{template.port}:{template.port}"
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
    volumes:
      - .:/app
      - /app/.venv
    depends_on:"""

        # Add dependencies
        dependencies = []
        if "redis" in services:
            dependencies.append("      - redis")
        if "postgres" in services:
            dependencies.append("      - postgres")
        if "mongodb" in services:
            dependencies.append("      - mongodb")

        if dependencies:
            app_service += "\n" + "\n".join(dependencies)
        else:
            app_service += " []"

        # Additional services
        additional_services = ""

        if "redis" in services:
            additional_services += """
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
"""

        if "postgres" in services:
            additional_services += """
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: dspy_app
      POSTGRES_USER: dspy_user
      POSTGRES_PASSWORD: dspy_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
"""

        if "mongodb" in services:
            additional_services += """
  mongodb:
    image: mongo:7
    environment:
      MONGO_INITDB_ROOT_USERNAME: dspy_user
      MONGO_INITDB_ROOT_PASSWORD: dspy_password
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
"""

        # Volumes section
        volumes_section = ""
        volume_names = []
        if "redis" in services:
            volume_names.append("redis_data")
        if "postgres" in services:
            volume_names.append("postgres_data")
        if "mongodb" in services:
            volume_names.append("mongodb_data")

        if volume_names:
            volumes_section = f"""
volumes:
  {chr(10).join([f"  {name}:" for name in volume_names])}
"""

        return f"""version: '3.8'

services:
{app_service}
{additional_services}{volumes_section}"""

    @staticmethod
    def get_dockerignore_template(template: DockerTemplate) -> str:
        """Generate .dockerignore template"""
        base_ignore = """# Python
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

        # Application-specific ignores
        app_specific = {
            ApplicationType.WEB_API: """
# Web API specific
static/
media/
uploads/
""",
            ApplicationType.BATCH_PROCESSOR: """
# Batch processing specific
data/
output/
processed/
""",
            ApplicationType.STREAMING: """
# Streaming specific
streams/
checkpoints/
""",
            ApplicationType.NOTEBOOK_SERVER: """
# Notebook specific
.ipynb_checkpoints/
*.ipynb
notebooks/output/
""",
        }

        return base_ignore + app_specific.get(template.app_type, "")


def main():
    """Demonstrate Docker template generation"""
    print("=== Docker Template Generator ===")

    # Example templates for different application types
    templates = [
        DockerTemplate(
            app_type=ApplicationType.WEB_API,
            optimization=OptimizationLevel.PRODUCTION,
            port=8000,
            environment_vars={"API_VERSION": "v1", "MAX_WORKERS": "4"},
        ),
        DockerTemplate(
            app_type=ApplicationType.BATCH_PROCESSOR,
            optimization=OptimizationLevel.PRODUCTION,
            additional_packages=["cron", "logrotate"],
        ),
        DockerTemplate(
            app_type=ApplicationType.NOTEBOOK_SERVER,
            optimization=OptimizationLevel.DEVELOPMENT,
            port=8888,
        ),
    ]

    for template in templates:
        print(f"\n=== {template.app_type.value.title()} Template ===")

        # Generate Dockerfile
        dockerfile = DockerTemplateGenerator.generate_dockerfile(template)
        print("Dockerfile:")
        print(dockerfile[:500] + "..." if len(dockerfile) > 500 else dockerfile)

        # Generate docker-compose.yml
        if template.app_type == ApplicationType.WEB_API:
            compose = DockerTemplateGenerator.get_docker_compose_template(
                template, ["redis", "postgres"]
            )
            print("\ndocker-compose.yml:")
            print(compose[:300] + "..." if len(compose) > 300 else compose)

        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
