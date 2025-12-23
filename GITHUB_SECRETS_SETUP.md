# GitHub Secrets Setup Guide

This document provides comprehensive instructions for setting up all required GitHub Secrets for deploying ConformAI to Azure AKS.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Required Secrets](#required-secrets)
3. [Azure Setup](#azure-setup)
4. [Adding Secrets to GitHub](#adding-secrets-to-github)
5. [Environment-Specific Secrets](#environment-specific-secrets)
6. [Verification](#verification)
7. [Security Best Practices](#security-best-practices)

---

## Prerequisites

Before setting up secrets, ensure you have:

- GitHub repository with admin access
- Azure subscription with appropriate permissions
- Azure CLI installed locally
- OpenAI API key
- Opik (Comet ML) account and API key

---

## Required Secrets

### 1. Azure Container Registry (ACR) Secrets

| Secret Name | Description | How to Obtain |
|------------|-------------|---------------|
| `ACR_LOGIN_SERVER` | ACR login server URL | `az acr show --name <acr-name> --query loginServer -o tsv` |
| `ACR_USERNAME` | ACR username | `az acr credential show --name <acr-name> --query username -o tsv` |
| `ACR_PASSWORD` | ACR password | `az acr credential show --name <acr-name> --query passwords[0].value -o tsv` |

**Setup Commands:**
```bash
# Create ACR
az acr create --resource-group conformai-rg --name conformaiacr --sku Standard

# Enable admin user
az acr update --name conformaiacr --admin-enabled true

# Get credentials
az acr credential show --name conformaiacr
```

---

### 2. Azure Kubernetes Service (AKS) Secrets

| Secret Name | Description | How to Obtain |
|------------|-------------|---------------|
| `AZURE_CREDENTIALS` | Service principal credentials JSON | See below |

**Setup Commands:**
```bash
# Create service principal for GitHub Actions
az ad sp create-for-rbac --name "github-actions-conformai" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/conformai-rg \
  --sdk-auth

# Output will be JSON - use this as AZURE_CREDENTIALS value
```

**Expected JSON format:**
```json
{
  "clientId": "<client-id>",
  "clientSecret": "<client-secret>",
  "subscriptionId": "<subscription-id>",
  "tenantId": "<tenant-id>",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

---

### 3. LLM API Keys

| Secret Name | Description | How to Obtain |
|------------|-------------|---------------|
| `OPENAI_API_KEY` | OpenAI API key | [OpenAI Platform](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) API key | [Anthropic Console](https://console.anthropic.com/) |

**Format:**
- OpenAI: `sk-proj-...` (starts with `sk-proj-` or `sk-`)
- Anthropic: `sk-ant-api03-...` (starts with `sk-ant-`)

---

### 4. Observability Secrets (Opik/Comet ML)

| Secret Name | Description | How to Obtain |
|------------|-------------|---------------|
| `OPIK_API_KEY` | Opik API key | [Comet ML Settings](https://www.comet.com/api/my/settings/) |
| `COMET_WORKSPACE` | Comet ML workspace name | Your workspace name from Comet ML dashboard |

**Setup:**
1. Go to https://www.comet.com/
2. Sign up / Log in
3. Navigate to Account Settings → API Keys
4. Generate new API key
5. Note your workspace name (usually your username)

---

### 5. Database Secrets

| Secret Name | Description | How to Generate |
|------------|-------------|-----------------|
| `POSTGRES_PASSWORD` | PostgreSQL password (staging) | `openssl rand -base64 32` |
| `POSTGRES_PASSWORD_PROD` | PostgreSQL password (production) | `openssl rand -base64 32` |

**Important:** Use different passwords for staging and production.

---

### 6. Vector Database Secrets

| Secret Name | Description | How to Obtain |
|------------|-------------|---------------|
| `QDRANT_API_KEY` | Qdrant Cloud API key (optional) | [Qdrant Cloud](https://cloud.qdrant.io/) |
| `QDRANT_API_KEY_PROD` | Qdrant Cloud API key for production | [Qdrant Cloud](https://cloud.qdrant.io/) |

**Note:** If using self-hosted Qdrant (from K8s manifests), these are optional.

---

### 7. Storage Secrets (MinIO / Azure Storage)

| Secret Name | Description | How to Generate |
|------------|-------------|-----------------|
| `S3_ACCESS_KEY` | S3-compatible access key | `openssl rand -hex 20` |
| `S3_SECRET_KEY` | S3-compatible secret key | `openssl rand -base64 40` |

**For Azure Blob Storage:**
```bash
# Get storage account key
az storage account keys list \
  --resource-group conformai-rg \
  --account-name conformaistorage \
  --query '[0].value' -o tsv
```

---

### 8. Airflow Secrets

| Secret Name | Description | How to Generate |
|------------|-------------|-----------------|
| `AIRFLOW_FERNET_KEY` | Airflow encryption key | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `AIRFLOW_WEBSERVER_SECRET_KEY` | Airflow webserver secret | `openssl rand -hex 32` |

**Generate Fernet Key:**
```bash
# Install cryptography if needed
pip install cryptography

# Generate key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

---

### 9. JWT Secrets

| Secret Name | Description | How to Generate |
|------------|-------------|-----------------|
| `JWT_SECRET_KEY` | JWT signing key (staging) | `openssl rand -hex 64` |
| `JWT_SECRET_KEY_PROD` | JWT signing key (production) | `openssl rand -hex 64` |

**Important:** Use different keys for staging and production.

---

## Azure Setup

### Step 1: Create Resource Group
```bash
az group create --name conformai-rg --location eastus
```

### Step 2: Create Azure Container Registry
```bash
# Create ACR
az acr create \
  --resource-group conformai-rg \
  --name conformaiacr \
  --sku Standard \
  --admin-enabled true

# Login to ACR
az acr login --name conformaiacr
```

### Step 3: Create AKS Cluster
```bash
# Create AKS cluster
az aks create \
  --resource-group conformai-rg \
  --name conformai-aks \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --attach-acr conformaiacr \
  --generate-ssh-keys

# Get credentials
az aks get-credentials \
  --resource-group conformai-rg \
  --name conformai-aks
```

### Step 4: Create Service Principal
```bash
# Get subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

# Create service principal with contributor role
az ad sp create-for-rbac \
  --name "github-actions-conformai" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/conformai-rg \
  --sdk-auth > azure-credentials.json

# View credentials
cat azure-credentials.json
```

---

## Adding Secrets to GitHub

### Method 1: GitHub Web UI

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Enter secret name and value
5. Click **Add secret**

### Method 2: GitHub CLI
```bash
# Install GitHub CLI
brew install gh  # macOS
# or
sudo apt install gh  # Ubuntu

# Authenticate
gh auth login

# Add secrets
gh secret set OPENAI_API_KEY --body "sk-proj-..."
gh secret set ANTHROPIC_API_KEY --body "sk-ant-..."
gh secret set OPIK_API_KEY --body "XJUo5793..."

# Add from file
gh secret set AZURE_CREDENTIALS < azure-credentials.json

# Add ACR credentials
gh secret set ACR_LOGIN_SERVER --body "conformaiacr.azurecr.io"
gh secret set ACR_USERNAME --body "conformaiacr"
gh secret set ACR_PASSWORD --body "$(az acr credential show --name conformaiacr --query 'passwords[0].value' -o tsv)"

# Add database passwords
gh secret set POSTGRES_PASSWORD --body "$(openssl rand -base64 32)"
gh secret set POSTGRES_PASSWORD_PROD --body "$(openssl rand -base64 32)"

# Add JWT secrets
gh secret set JWT_SECRET_KEY --body "$(openssl rand -hex 64)"
gh secret set JWT_SECRET_KEY_PROD --body "$(openssl rand -hex 64)"

# Add Airflow secrets
gh secret set AIRFLOW_FERNET_KEY --body "$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
gh secret set AIRFLOW_WEBSERVER_SECRET_KEY --body "$(openssl rand -hex 32)"

# Add storage secrets
gh secret set S3_ACCESS_KEY --body "$(openssl rand -hex 20)"
gh secret set S3_SECRET_KEY --body "$(openssl rand -base64 40)"
```

---

## Environment-Specific Secrets

### Staging Environment
Create environment in GitHub:
1. Go to **Settings** → **Environments**
2. Click **New environment**
3. Name it `staging`
4. Add environment-specific secrets if needed

### Production Environment
1. Create `production` environment
2. Add **Required reviewers** (recommended)
3. Add environment-specific secrets:
   - `POSTGRES_PASSWORD_PROD`
   - `QDRANT_API_KEY_PROD`
   - `JWT_SECRET_KEY_PROD`

---

## Complete Secret Checklist

### Repository Secrets (Required for All Environments)
- [ ] `ACR_LOGIN_SERVER`
- [ ] `ACR_USERNAME`
- [ ] `ACR_PASSWORD`
- [ ] `AZURE_CREDENTIALS`
- [ ] `OPENAI_API_KEY`
- [ ] `ANTHROPIC_API_KEY` (optional)
- [ ] `OPIK_API_KEY`
- [ ] `COMET_WORKSPACE`
- [ ] `POSTGRES_PASSWORD`
- [ ] `POSTGRES_PASSWORD_PROD`
- [ ] `JWT_SECRET_KEY`
- [ ] `JWT_SECRET_KEY_PROD`
- [ ] `AIRFLOW_FERNET_KEY`
- [ ] `AIRFLOW_WEBSERVER_SECRET_KEY`
- [ ] `S3_ACCESS_KEY`
- [ ] `S3_SECRET_KEY`

### Optional Secrets
- [ ] `QDRANT_API_KEY` (if using Qdrant Cloud)
- [ ] `QDRANT_API_KEY_PROD`

---

## Verification

### Test Secrets Setup
```bash
# Trigger workflow manually
gh workflow run deploy.yml -f environment=staging

# Watch workflow run
gh run watch

# Check deployment
kubectl get pods -n conformai-staging
kubectl get svc -n conformai-staging
```

### Verify Deployments
```bash
# Get API Gateway URL
kubectl get svc api-gateway -n conformai-staging -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Test health endpoint
curl http://<API_GATEWAY_IP>/health

# Check logs
kubectl logs -n conformai-staging deployment/api-gateway
kubectl logs -n conformai-staging deployment/rag-service
```

---

## Security Best Practices

### 1. **Rotate Secrets Regularly**
- Rotate API keys every 90 days
- Rotate database passwords every 180 days
- Rotate JWT secrets after security incidents

### 2. **Use Different Secrets for Each Environment**
- Never reuse production secrets in staging
- Use weak/test values only in development
- Keep production secrets offline in password manager

### 3. **Limit Secret Access**
- Only add secrets needed for deployment
- Use environment-specific secrets when possible
- Enable branch protection rules

### 4. **Audit Secret Usage**
- Review GitHub Actions logs regularly
- Monitor failed authentication attempts
- Set up alerts for secret access

### 5. **Backup Secrets Securely**
- Store secrets in password manager (1Password, LastPass)
- Encrypt secret backups
- Use Azure Key Vault for production secrets

### 6. **Emergency Procedures**
```bash
# If secrets compromised, rotate immediately:

# 1. Regenerate API keys
# 2. Update GitHub secrets
gh secret set OPENAI_API_KEY --body "new-key"

# 3. Restart deployments
kubectl rollout restart deployment/api-gateway -n conformai-prod
kubectl rollout restart deployment/rag-service -n conformai-prod

# 4. Audit access logs
kubectl logs -n conformai-prod deployment/api-gateway | grep "401\|403"
```

---

## Quick Setup Script

Save this as `setup-github-secrets.sh`:

```bash
#!/bin/bash

set -e

echo "🔐 ConformAI GitHub Secrets Setup"
echo "=================================="

# Check prerequisites
command -v gh >/dev/null 2>&1 || { echo "❌ GitHub CLI not found. Install: brew install gh"; exit 1; }
command -v az >/dev/null 2>&1 || { echo "❌ Azure CLI not found. Install: brew install azure-cli"; exit 1; }

# Authenticate
gh auth status || gh auth login
az account show || az login

# Get Azure resources
ACR_NAME="conformaiacr"
RESOURCE_GROUP="conformai-rg"

echo "📦 Fetching Azure credentials..."
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)

echo "🔑 Setting GitHub secrets..."
gh secret set ACR_LOGIN_SERVER --body "$ACR_LOGIN_SERVER"
gh secret set ACR_USERNAME --body "$ACR_USERNAME"
gh secret set ACR_PASSWORD --body "$ACR_PASSWORD"

# Generate database passwords
gh secret set POSTGRES_PASSWORD --body "$(openssl rand -base64 32)"
gh secret set POSTGRES_PASSWORD_PROD --body "$(openssl rand -base64 32)"

# Generate JWT secrets
gh secret set JWT_SECRET_KEY --body "$(openssl rand -hex 64)"
gh secret set JWT_SECRET_KEY_PROD --body "$(openssl rand -hex 64)"

# Generate Airflow secrets
gh secret set AIRFLOW_FERNET_KEY --body "$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
gh secret set AIRFLOW_WEBSERVER_SECRET_KEY --body "$(openssl rand -hex 32)"

# Generate storage secrets
gh secret set S3_ACCESS_KEY --body "$(openssl rand -hex 20)"
gh secret set S3_SECRET_KEY --body "$(openssl rand -base64 40)"

echo ""
echo "⚠️  Manual steps required:"
echo "1. Add AZURE_CREDENTIALS: Run 'az ad sp create-for-rbac --sdk-auth'"
echo "2. Add OPENAI_API_KEY from https://platform.openai.com/api-keys"
echo "3. Add OPIK_API_KEY from https://www.comet.com/api/my/settings/"
echo "4. Add COMET_WORKSPACE (your Comet ML workspace name)"
echo ""
echo "✅ Automated secrets configured successfully!"
```

**Run the script:**
```bash
chmod +x setup-github-secrets.sh
./setup-github-secrets.sh
```

---

## Troubleshooting

### Common Issues

**1. ACR Authentication Failed**
```bash
# Verify ACR credentials
az acr login --name conformaiacr

# Check if admin user is enabled
az acr update --name conformaiacr --admin-enabled true
```

**2. AKS Access Denied**
```bash
# Verify service principal has correct permissions
az role assignment list --assignee <client-id> --scope /subscriptions/<subscription-id>/resourceGroups/conformai-rg
```

**3. Secret Not Found**
```bash
# List all secrets
gh secret list

# Check secret value (won't show actual value)
gh secret set OPENAI_API_KEY --body "new-value"
```

**4. Deployment Failing**
```bash
# Check pod status
kubectl describe pod <pod-name> -n conformai-staging

# Check secret mounting
kubectl get secret conformai-secrets -n conformai-staging -o yaml

# Verify ConfigMap
kubectl get configmap conformai-config -n conformai-staging -o yaml
```

---

## Next Steps

After setting up all secrets:

1. **Test the pipeline**: Trigger a manual workflow run
2. **Monitor deployment**: Watch logs and pod status
3. **Verify health**: Test API endpoints
4. **Set up monitoring**: Configure Azure Monitor alerts
5. **Document custom changes**: Update this file with any environment-specific modifications

---

## Support

For issues:
- GitHub Actions: Check workflow logs in **Actions** tab
- Azure issues: Use `az` CLI or Azure Portal
- Application logs: `kubectl logs` in AKS

**Remember**: Never commit secrets to Git. Always use GitHub Secrets or Azure Key Vault.
