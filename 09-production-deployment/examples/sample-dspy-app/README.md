# DSPy Sample API - Deployment Guide

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
