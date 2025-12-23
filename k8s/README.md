# Kubernetes Deployment Manifests

This directory contains Kubernetes manifests for deploying ConformAI to Azure AKS.

## Directory Structure

```
k8s/
├── staging/                    # Staging environment manifests
│   ├── 01-api-gateway.yaml    # API Gateway deployment + service
│   ├── 02-rag-service.yaml    # RAG Service deployment + service + HPA
│   ├── 03-retrieval-service.yaml  # Retrieval Service + HPA
│   └── 04-infrastructure.yaml # Qdrant, PostgreSQL, Redis
└── production/                 # Production environment manifests
    ├── 01-api-gateway.yaml
    ├── 02-rag-service.yaml
    ├── 03-retrieval-service.yaml
    └── 04-infrastructure.yaml
```

## Differences: Staging vs Production

| Component | Staging | Production |
|-----------|---------|------------|
| API Gateway replicas | 2 | 3 |
| RAG Service replicas | 3 | 5 |
| Retrieval Service replicas | 2 | 4 |
| HPA max replicas | 10 | 20 |
| Resource limits | Lower | Higher |
| LLM Model | gpt-4o-mini | gpt-4o |
| Log level | INFO | WARNING |

## Prerequisites

1. **AKS Cluster**: Running Azure Kubernetes cluster
2. **kubectl**: Configured to access your cluster
3. **Secrets**: Created in Kubernetes (done by CI/CD)
4. **Storage Class**: `managed-premium` available

## Manual Deployment

### Deploy to Staging

```bash
# Create namespace
kubectl create namespace conformai-staging

# Create secrets (replace with actual values)
kubectl create secret generic conformai-secrets \
  --from-literal=openai-api-key="sk-proj-..." \
  --from-literal=anthropic-api-key="sk-ant-..." \
  --from-literal=opik-api-key="..." \
  --from-literal=postgres-password="..." \
  --namespace conformai-staging

# Create ConfigMap
kubectl create configmap conformai-config \
  --from-literal=environment="staging" \
  --from-literal=log-level="INFO" \
  --from-literal=llm-provider="openai" \
  --from-literal=llm-model="gpt-4o-mini" \
  --namespace conformai-staging

# Apply manifests
kubectl apply -f k8s/staging/ -n conformai-staging

# Watch deployment
kubectl get pods -n conformai-staging -w
```

### Deploy to Production

```bash
# Create namespace
kubectl create namespace conformai-prod

# Create secrets (use production values)
kubectl create secret generic conformai-secrets \
  --from-literal=openai-api-key="sk-proj-..." \
  --from-literal=postgres-password="..." \
  --namespace conformai-prod

# Create ConfigMap
kubectl create configmap conformai-config \
  --from-literal=environment="production" \
  --from-literal=log-level="WARNING" \
  --from-literal=llm-provider="openai" \
  --from-literal=llm-model="gpt-4o" \
  --namespace conformai-prod

# Apply manifests
kubectl apply -f k8s/production/ -n conformai-prod
```

## Service Architecture

### Networking

```
Internet → LoadBalancer (API Gateway) → ClusterIP Services
                                            ├── RAG Service
                                            ├── Retrieval Service
                                            ├── Qdrant
                                            ├── PostgreSQL
                                            └── Redis
```

### Ports

| Service | Internal Port | External Port | Type |
|---------|---------------|---------------|------|
| API Gateway | 8000 | 80 | LoadBalancer |
| RAG Service | 8000 | 8001 | ClusterIP |
| Retrieval Service | 8000 | 8002 | ClusterIP |
| Qdrant | 6333, 6334 | - | ClusterIP |
| PostgreSQL | 5432 | - | ClusterIP |
| Redis | 6379 | - | ClusterIP |

## Scaling

### Horizontal Pod Autoscaler (HPA)

Automatically scales based on:
- CPU utilization (70% target)
- Memory utilization (80% target)

```bash
# Check HPA status
kubectl get hpa -n conformai-staging

# Manually scale
kubectl scale deployment rag-service --replicas=5 -n conformai-staging
```

### Vertical Scaling

Update resource requests/limits in YAML files:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## Storage

### Persistent Volumes

- **Qdrant**: 50Gi (managed-premium)
- **PostgreSQL**: 20Gi (managed-premium)

```bash
# List PVCs
kubectl get pvc -n conformai-staging

# Check PV usage
kubectl exec -it qdrant-0 -n conformai-staging -- df -h /qdrant/storage
```

## Monitoring

### Health Checks

```bash
# Get LoadBalancer IP
export API_IP=$(kubectl get svc api-gateway -n conformai-staging -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Health check
curl http://$API_IP/health

# Readiness check
curl http://$API_IP/ready
```

### Pod Logs

```bash
# View logs
kubectl logs -f deployment/rag-service -n conformai-staging

# View logs from all pods
kubectl logs -f deployment/rag-service --all-containers -n conformai-staging

# Previous logs (after crash)
kubectl logs deployment/rag-service --previous -n conformai-staging
```

### Pod Status

```bash
# Get all pods
kubectl get pods -n conformai-staging

# Describe pod
kubectl describe pod <pod-name> -n conformai-staging

# Get events
kubectl get events -n conformai-staging --sort-by='.lastTimestamp'
```

## Troubleshooting

### Pods not starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n conformai-staging

# Common issues:
# 1. Image pull errors → Check ACR credentials
# 2. Secret not found → Verify secret exists
# 3. Resource limits → Check node capacity
```

### Service not accessible

```bash
# Check service
kubectl get svc -n conformai-staging

# Check endpoints
kubectl get endpoints -n conformai-staging

# Port forward for testing
kubectl port-forward svc/api-gateway 8000:80 -n conformai-staging
```

### Database connection errors

```bash
# Check PostgreSQL pod
kubectl logs statefulset/postgres -n conformai-staging

# Connect to PostgreSQL
kubectl exec -it postgres-0 -n conformai-staging -- psql -U conformai

# Check Qdrant
kubectl exec -it qdrant-0 -n conformai-staging -- curl localhost:6333/collections
```

## Updating Deployments

### Rolling Update

```bash
# Update image
kubectl set image deployment/rag-service \
  rag-service=conformaiacr.azurecr.io/conformai-rag-service:v2.0.0 \
  -n conformai-staging

# Check rollout status
kubectl rollout status deployment/rag-service -n conformai-staging

# Pause rollout
kubectl rollout pause deployment/rag-service -n conformai-staging

# Resume rollout
kubectl rollout resume deployment/rag-service -n conformai-staging
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/rag-service -n conformai-staging

# Rollback to specific revision
kubectl rollout undo deployment/rag-service --to-revision=2 -n conformai-staging

# Check rollout history
kubectl rollout history deployment/rag-service -n conformai-staging
```

## Cleanup

### Delete Staging

```bash
# Delete all resources
kubectl delete namespace conformai-staging
```

### Delete Production

```bash
# ⚠️ CAUTION: This deletes production data!
kubectl delete namespace conformai-prod
```

## Cost Optimization

### Azure Reserved Instances
- Use for production nodes (up to 72% savings)
- Commit to 1-year or 3-year terms

### Spot Instances
- Use for non-critical workloads
- 90% cost reduction

```bash
# Add spot node pool
az aks nodepool add \
  --resource-group conformai-rg \
  --cluster-name conformai-aks \
  --name spotnodepool \
  --priority Spot \
  --eviction-policy Delete \
  --spot-max-price -1 \
  --node-count 2
```

### Right-sizing

Monitor and adjust:
```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n conformai-staging

# Adjust based on usage
# - Reduce replicas during low traffic
# - Lower CPU/memory requests if underutilized
```

## Security

### Network Policies

```bash
# Apply network policies (create separate file)
kubectl apply -f k8s/network-policies.yaml -n conformai-staging
```

### Pod Security

- Use non-root users in containers
- Read-only root filesystem
- Drop capabilities

### Secrets Management

Consider Azure Key Vault:
```bash
# Install CSI driver
helm repo add csi-secrets-store-provider-azure https://azure.github.io/secrets-store-csi-driver-provider-azure/charts
helm install csi csi-secrets-store-provider-azure/csi-secrets-store-provider-azure
```

## CI/CD Integration

Deployments are automated via GitHub Actions:
- `.github/workflows/deploy.yml` handles deployments
- Secrets stored in GitHub Secrets
- Manual approval required for production

See: [GITHUB_SECRETS_SETUP.md](../GITHUB_SECRETS_SETUP.md)

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Azure AKS Best Practices](https://learn.microsoft.com/en-us/azure/aks/best-practices)
- [Kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
