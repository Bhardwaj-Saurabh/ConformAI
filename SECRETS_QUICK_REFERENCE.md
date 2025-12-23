# GitHub Secrets Quick Reference

## Essential Secrets (Required)

Copy and paste these commands to set up all secrets:

```bash
# 1. Azure ACR Credentials
gh secret set ACR_LOGIN_SERVER --body "conformaiacr.azurecr.io"
gh secret set ACR_USERNAME --body "conformaiacr"
gh secret set ACR_PASSWORD --body "$(az acr credential show --name conformaiacr --query 'passwords[0].value' -o tsv)"

# 2. Azure Service Principal
az ad sp create-for-rbac --name "github-actions-conformai" \
  --role contributor \
  --scopes /subscriptions/$(az account show --query id -o tsv)/resourceGroups/conformai-rg \
  --sdk-auth > azure-creds.json
gh secret set AZURE_CREDENTIALS < azure-creds.json
rm azure-creds.json

# 3. LLM API Keys (manually add from dashboards)
gh secret set OPENAI_API_KEY --body "sk-proj-YOUR_KEY_HERE"
gh secret set ANTHROPIC_API_KEY --body "sk-ant-YOUR_KEY_HERE"

# 4. Opik Observability
gh secret set OPIK_API_KEY --body "YOUR_OPIK_KEY_HERE"
gh secret set COMET_WORKSPACE --body "your-workspace-name"

# 5. Database Passwords
gh secret set POSTGRES_PASSWORD --body "$(openssl rand -base64 32)"
gh secret set POSTGRES_PASSWORD_PROD --body "$(openssl rand -base64 32)"

# 6. JWT Secrets
gh secret set JWT_SECRET_KEY --body "$(openssl rand -hex 64)"
gh secret set JWT_SECRET_KEY_PROD --body "$(openssl rand -hex 64)"

# 7. Airflow Secrets
pip install cryptography
gh secret set AIRFLOW_FERNET_KEY --body "$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
gh secret set AIRFLOW_WEBSERVER_SECRET_KEY --body "$(openssl rand -hex 32)"

# 8. Storage Secrets
gh secret set S3_ACCESS_KEY --body "$(openssl rand -hex 20)"
gh secret set S3_SECRET_KEY --body "$(openssl rand -base64 40)"

# 9. Optional: Qdrant Cloud (if using cloud version)
gh secret set QDRANT_API_KEY --body "YOUR_QDRANT_KEY"
gh secret set QDRANT_API_KEY_PROD --body "YOUR_QDRANT_PROD_KEY"
```

## Verification

```bash
# List all secrets
gh secret list

# Test deployment
gh workflow run deploy.yml -f environment=staging

# Watch progress
gh run watch
```

## Required Manual Steps

1. **Get OpenAI API Key**: https://platform.openai.com/api-keys
2. **Get Opik API Key**: https://www.comet.com/api/my/settings/
3. **Get Comet Workspace**: Your username on Comet ML
4. **Create Azure Resources**: Run commands from GITHUB_SECRETS_SETUP.md

## All Secrets Checklist

- [ ] ACR_LOGIN_SERVER
- [ ] ACR_USERNAME
- [ ] ACR_PASSWORD
- [ ] AZURE_CREDENTIALS
- [ ] OPENAI_API_KEY
- [ ] ANTHROPIC_API_KEY
- [ ] OPIK_API_KEY
- [ ] COMET_WORKSPACE
- [ ] POSTGRES_PASSWORD
- [ ] POSTGRES_PASSWORD_PROD
- [ ] JWT_SECRET_KEY
- [ ] JWT_SECRET_KEY_PROD
- [ ] AIRFLOW_FERNET_KEY
- [ ] AIRFLOW_WEBSERVER_SECRET_KEY
- [ ] S3_ACCESS_KEY
- [ ] S3_SECRET_KEY

## Test After Setup

```bash
# Check AKS cluster
az aks get-credentials --resource-group conformai-rg --name conformai-aks
kubectl get nodes

# Trigger deployment
gh workflow run deploy.yml -f environment=staging

# Monitor deployment
kubectl get pods -n conformai-staging -w

# Test API
export API_IP=$(kubectl get svc api-gateway -n conformai-staging -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$API_IP/health
```

## Emergency Rotation

If secrets are compromised:

```bash
# Rotate immediately
gh secret set OPENAI_API_KEY --body "new-key"
gh secret set JWT_SECRET_KEY_PROD --body "$(openssl rand -hex 64)"

# Restart pods
kubectl rollout restart deployment -n conformai-prod
```

Full documentation: [GITHUB_SECRETS_SETUP.md](./GITHUB_SECRETS_SETUP.md)
