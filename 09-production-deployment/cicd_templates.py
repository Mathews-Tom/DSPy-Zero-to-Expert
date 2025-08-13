#!/usr/bin/env python3
"""
CI/CD Pipeline Templates for DSPy Applications

This module provides comprehensive CI/CD pipeline templates for different
platforms and deployment strategies.

Author: DSPy Learning Framework
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class CICDPlatform(Enum):
    """CI/CD platform types"""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    AZURE_DEVOPS = "azure_devops"
    JENKINS = "jenkins"


class DeploymentTarget(Enum):
    """Deployment target types"""

    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"
    AWS_ECS = "aws_ecs"
    AZURE_CONTAINER_INSTANCES = "azure_container_instances"
    GOOGLE_CLOUD_RUN = "google_cloud_run"


@dataclass
class CICDConfig:
    """CI/CD configuration"""

    platform: CICDPlatform
    target: DeploymentTarget
    app_name: str
    registry: str
    environments: List[str] = None
    test_commands: List[str] = None
    build_args: Dict[str, str] = None
    secrets: List[str] = None


class GitHubActionsGenerator:
    """Generate GitHub Actions workflows"""

    @staticmethod
    def get_basic_workflow(config: CICDConfig) -> str:
        """Generate basic GitHub Actions workflow"""
        environments = config.environments or ["staging", "production"]
        test_commands = config.test_commands or [
            "pytest",
            "ruff check .",
            "black --check .",
        ]
        secrets = config.secrets or ["KUBECONFIG", "REGISTRY_TOKEN"]

        test_steps = "\n    ".join(
            [
                f"- name: {cmd.split()[0].title()}\n      run: uv run {cmd}"
                for cmd in test_commands
            ]
        )

        return f"""name: CI/CD Pipeline for {config.app_name}

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: {config.registry}
  IMAGE_NAME: {config.app_name}

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python ${{{{ matrix.python-version }}}}
      uses: actions/setup-python@v4
      with:
        python-version: ${{{{ matrix.python-version }}}}
    
    - name: Install uv
      run: pip install uv
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/uv
        key: ${{{{ runner.os }}}}-uv-${{{{ hashFiles('**/uv.lock') }}}}
        restore-keys: |
          ${{{{ runner.os }}}}-uv-
    
    - name: Install dependencies
      run: uv sync --all-extras
    
    {test_steps}
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      if: matrix.python-version == '3.11'

  security:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: security-scan-results.sarif
    
    - name: Dependency vulnerability scan
      run: |
        pip install safety
        safety check --json --output safety-report.json || true

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    outputs:
      image-tag: ${{{{ steps.meta.outputs.tags }}}}
      image-digest: ${{{{ steps.build.outputs.digest }}}}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
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
          type=semver,pattern={{{{version}}}}
          type=semver,pattern={{{{major}}}}.{{{{minor}}}}
    
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
        platforms: linux/amd64,linux/arm64

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add staging deployment commands here
    
    - name: Run smoke tests
      run: |
        echo "Running smoke tests"
        # Add smoke test commands here

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Deploy to production
      run: |
        echo "Deploying to production environment"
        # Add production deployment commands here
    
    - name: Run health checks
      run: |
        echo "Running health checks"
        # Add health check commands here
    
    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{{{ job.status }}}}
        channel: '#deployments'
      env:
        SLACK_WEBHOOK_URL: ${{{{ secrets.SLACK_WEBHOOK_URL }}}}
"""

    @staticmethod
    def get_kubernetes_workflow(config: CICDConfig) -> str:
        """Generate Kubernetes-specific GitHub Actions workflow"""
        return f"""name: Deploy to Kubernetes

on:
  push:
    branches: [ main, develop ]

env:
  REGISTRY: {config.registry}
  IMAGE_NAME: {config.app_name}

jobs:
  deploy:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        environment: [staging, production]
        include:
          - environment: staging
            branch: develop
            namespace: staging
          - environment: production
            branch: main
            namespace: production
    
    if: github.ref == format('refs/heads/{{0}}', matrix.branch)
    environment: ${{{{ matrix.environment }}}}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'latest'
    
    - name: Configure kubectl
      run: |
        echo "${{{{ secrets.KUBECONFIG }}}}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        kubectl config current-context
    
    - name: Deploy to Kubernetes
      run: |
        export KUBECONFIG=kubeconfig
        
        # Update image tag in deployment
        sed -i 's|IMAGE_TAG|${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}:${{{{ github.sha }}}}|g' k8s/deployment.yaml
        
        # Apply manifests
        kubectl apply -f k8s/ -n ${{{{ matrix.namespace }}}}
        
        # Wait for rollout
        kubectl rollout status deployment/{config.app_name}-deployment -n ${{{{ matrix.namespace }}}} --timeout=300s
    
    - name: Verify deployment
      run: |
        export KUBECONFIG=kubeconfig
        
        # Check pod status
        kubectl get pods -n ${{{{ matrix.namespace }}}} -l app={config.app_name}
        
        # Check service endpoints
        kubectl get endpoints -n ${{{{ matrix.namespace }}}} -l app={config.app_name}
        
        # Run health check
        kubectl exec -n ${{{{ matrix.namespace }}}} deployment/{config.app_name}-deployment -- curl -f http://localhost:8000/health
    
    - name: Rollback on failure
      if: failure()
      run: |
        export KUBECONFIG=kubeconfig
        kubectl rollout undo deployment/{config.app_name}-deployment -n ${{{{ matrix.namespace }}}}
        kubectl rollout status deployment/{config.app_name}-deployment -n ${{{{ matrix.namespace }}}}
"""


class GitLabCIGenerator:
    """Generate GitLab CI/CD pipelines"""

    @staticmethod
    def get_basic_pipeline(config: CICDConfig) -> str:
        """Generate basic GitLab CI pipeline"""
        test_commands = config.test_commands or [
            "pytest",
            "ruff check .",
            "black --check .",
        ]
        test_script = "\n    ".join([f"- uv run {cmd}" for cmd in test_commands])

        return f"""# GitLab CI/CD Pipeline for {config.app_name}

stages:
  - test
  - security
  - build
  - deploy-staging
  - deploy-production

variables:
  REGISTRY: {config.registry}
  IMAGE_NAME: {config.app_name}
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

# Test stage
test:
  stage: test
  image: python:3.11-slim
  services:
    - docker:dind
  before_script:
    - pip install uv
    - uv sync --all-extras
  script:
    {test_script}
  coverage: '/TOTAL.*\\s+(\\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - htmlcov/
    expire_in: 1 week
  only:
    - branches
    - merge_requests

# Security scanning
security:
  stage: security
  image: python:3.11-slim
  before_script:
    - pip install safety bandit
  script:
    - safety check --json --output safety-report.json || true
    - bandit -r . -f json -o bandit-report.json || true
  artifacts:
    reports:
      security:
        - safety-report.json
        - bandit-report.json
    expire_in: 1 week
  only:
    - branches
    - merge_requests

# Build Docker image
build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - echo $CI_REGISTRY_PASSWORD | docker login -u $CI_REGISTRY_USER --password-stdin $REGISTRY
  script:
    - docker build -t $REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA .
    - docker build -t $REGISTRY/$IMAGE_NAME:latest .
    - docker push $REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA
    - docker push $REGISTRY/$IMAGE_NAME:latest
  only:
    - main
    - develop

# Deploy to staging
deploy-staging:
  stage: deploy-staging
  image: bitnami/kubectl:latest
  before_script:
    - echo "$KUBECONFIG_STAGING" | base64 -d > kubeconfig
    - export KUBECONFIG=kubeconfig
  script:
    - sed -i "s|IMAGE_TAG|$REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA|g" k8s/deployment.yaml
    - kubectl apply -f k8s/ -n staging
    - kubectl rollout status deployment/{config.app_name}-deployment -n staging --timeout=300s
  environment:
    name: staging
    url: https://staging.{config.app_name}.example.com
  only:
    - develop

# Deploy to production
deploy-production:
  stage: deploy-production
  image: bitnami/kubectl:latest
  before_script:
    - echo "$KUBECONFIG_PRODUCTION" | base64 -d > kubeconfig
    - export KUBECONFIG=kubeconfig
  script:
    - sed -i "s|IMAGE_TAG|$REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA|g" k8s/deployment.yaml
    - kubectl apply -f k8s/ -n production
    - kubectl rollout status deployment/{config.app_name}-deployment -n production --timeout=300s
  environment:
    name: production
    url: https://{config.app_name}.example.com
  when: manual
  only:
    - main
"""


class AzureDevOpsGenerator:
    """Generate Azure DevOps pipelines"""

    @staticmethod
    def get_basic_pipeline(config: CICDConfig) -> str:
        """Generate basic Azure DevOps pipeline"""
        return f"""# Azure DevOps Pipeline for {config.app_name}

trigger:
  branches:
    include:
      - main
      - develop

pr:
  branches:
    include:
      - main

variables:
  registry: '{config.registry}'
  imageName: '{config.app_name}'
  dockerfilePath: '$(Build.SourcesDirectory)/Dockerfile'
  tag: '$(Build.BuildId)'

stages:
- stage: Test
  displayName: 'Test Stage'
  jobs:
  - job: Test
    displayName: 'Run Tests'
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.11'
        displayName: 'Use Python 3.11'
    
    - script: |
        pip install uv
        uv sync --all-extras
      displayName: 'Install dependencies'
    
    - script: |
        uv run pytest --junitxml=junit/test-results.xml --cov=. --cov-report=xml
      displayName: 'Run tests'
    
    - task: PublishTestResults@2
      condition: succeededOrFailed()
      inputs:
        testResultsFiles: '**/test-*.xml'
        testRunTitle: 'Publish test results for Python $(python.version)'
    
    - task: PublishCodeCoverageResults@1
      inputs:
        codeCoverageTool: Cobertura
        summaryFileLocation: '$(System.DefaultWorkingDirectory)/**/coverage.xml'

- stage: Build
  displayName: 'Build Stage'
  dependsOn: Test
  condition: succeeded()
  jobs:
  - job: Build
    displayName: 'Build and Push Docker Image'
    pool:
      vmImage: 'ubuntu-latest'
    steps:
    - task: Docker@2
      displayName: 'Build and push Docker image'
      inputs:
        command: buildAndPush
        repository: $(imageName)
        dockerfile: $(dockerfilePath)
        containerRegistry: 'dockerRegistryServiceConnection'
        tags: |
          $(tag)
          latest

- stage: DeployStaging
  displayName: 'Deploy to Staging'
  dependsOn: Build
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/develop'))
  jobs:
  - deployment: DeployStaging
    displayName: 'Deploy to Staging Environment'
    pool:
      vmImage: 'ubuntu-latest'
    environment: 'staging'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: KubernetesManifest@0
            displayName: 'Deploy to Kubernetes'
            inputs:
              action: deploy
              kubernetesServiceConnection: 'k8s-staging'
              namespace: staging
              manifests: |
                $(Pipeline.Workspace)/k8s/deployment.yaml
                $(Pipeline.Workspace)/k8s/service.yaml

- stage: DeployProduction
  displayName: 'Deploy to Production'
  dependsOn: Build
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  jobs:
  - deployment: DeployProduction
    displayName: 'Deploy to Production Environment'
    pool:
      vmImage: 'ubuntu-latest'
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: KubernetesManifest@0
            displayName: 'Deploy to Kubernetes'
            inputs:
              action: deploy
              kubernetesServiceConnection: 'k8s-production'
              namespace: production
              manifests: |
                $(Pipeline.Workspace)/k8s/deployment.yaml
                $(Pipeline.Workspace)/k8s/service.yaml
"""


class CICDTemplateManager:
    """Manage CI/CD template generation"""

    @staticmethod
    def generate_pipeline(config: CICDConfig) -> str:
        """Generate CI/CD pipeline based on platform"""
        generators = {
            CICDPlatform.GITHUB_ACTIONS: GitHubActionsGenerator.get_basic_workflow,
            CICDPlatform.GITLAB_CI: GitLabCIGenerator.get_basic_pipeline,
            CICDPlatform.AZURE_DEVOPS: AzureDevOpsGenerator.get_basic_pipeline,
        }

        generator = generators.get(config.platform)
        if not generator:
            raise ValueError(f"Unsupported CI/CD platform: {config.platform}")

        return generator(config)

    @staticmethod
    def get_deployment_specific_pipeline(config: CICDConfig) -> str:
        """Generate deployment-specific pipeline"""
        if (
            config.platform == CICDPlatform.GITHUB_ACTIONS
            and config.target == DeploymentTarget.KUBERNETES
        ):
            return GitHubActionsGenerator.get_kubernetes_workflow(config)
        else:
            return CICDTemplateManager.generate_pipeline(config)

    @staticmethod
    def get_pipeline_filename(platform: CICDPlatform) -> str:
        """Get appropriate filename for pipeline"""
        filenames = {
            CICDPlatform.GITHUB_ACTIONS: ".github/workflows/ci-cd.yml",
            CICDPlatform.GITLAB_CI: ".gitlab-ci.yml",
            CICDPlatform.AZURE_DEVOPS: "azure-pipelines.yml",
        }
        return filenames.get(platform, "pipeline.yml")


def main():
    """Demonstrate CI/CD template generation"""
    print("=== CI/CD Template Generator ===")

    # Example configurations
    configs = [
        CICDConfig(
            platform=CICDPlatform.GITHUB_ACTIONS,
            target=DeploymentTarget.KUBERNETES,
            app_name="dspy-web-api",
            registry="ghcr.io/your-org",
            environments=["staging", "production"],
            test_commands=["pytest --cov=.", "ruff check .", "mypy ."],
            secrets=["KUBECONFIG", "SLACK_WEBHOOK_URL"],
        ),
        CICDConfig(
            platform=CICDPlatform.GITLAB_CI,
            target=DeploymentTarget.KUBERNETES,
            app_name="dspy-batch-processor",
            registry="registry.gitlab.com/your-org",
            test_commands=["pytest", "bandit -r ."],
        ),
    ]

    for config in configs:
        print(f"\n=== {config.platform.value} Pipeline ===")

        # Generate pipeline
        pipeline = CICDTemplateManager.generate_pipeline(config)
        filename = CICDTemplateManager.get_pipeline_filename(config.platform)

        print(f"Filename: {filename}")
        print("Pipeline content (first 500 chars):")
        print(pipeline[:500] + "..." if len(pipeline) > 500 else pipeline)

        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
