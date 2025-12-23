# Security Best Practices

This document outlines security best practices for the ConformAI project.

---

## 🔒 Environment Variables & Secrets Management

### Important Security Rules

1. **NEVER commit `.env` files to version control**
   - `.env` is in `.gitignore` to prevent accidental commits
   - Only `.env.example` should be in git (with placeholder values)

2. **Use strong, unique credentials for each environment**
   - Development
   - Staging
   - Production

3. **Rotate credentials regularly**
   - API keys (every 90 days)
   - Database passwords (every 90 days)
   - JWT secrets (every 90 days)
   - Fernet keys (every 90 days)

---

## 📝 Setting Up Environment Variables

### Step 1: Copy Example File

```bash
cp .env.example .env
```

### Step 2: Replace Placeholder Values

Look for these placeholder patterns and replace them:

| Placeholder | What to Replace With |
|------------|---------------------|
| `YOUR_KEY_HERE` | Your actual API key |
| `CHANGE_THIS_PASSWORD` | A strong password |
| `GENERATE_FERNET_KEY_HERE` | Generated Fernet key |
| `GENERATE_SECRET_KEY_HERE` | Generated secret key |
| `example.com` | Your actual domain |

### Step 3: Generate Secure Keys

**Fernet Key** (for Airflow):
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Secret Key** (for Airflow webserver):
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**JWT Secret Key**:
```bash
openssl rand -hex 32
```

**Database Password**:
```bash
openssl rand -base64 32
```

---

## 🔑 API Keys

### Anthropic API Key

1. Sign up at https://console.anthropic.com/
2. Navigate to "API Keys"
3. Create a new key
4. Copy to `.env` as `ANTHROPIC_API_KEY`

**Key format**: `sk-ant-api03-...`

### OpenAI API Key

1. Sign up at https://platform.openai.com/
2. Navigate to "API Keys"
3. Create a new key
4. Copy to `.env` as `OPENAI_API_KEY`

**Key format**: `sk-...`

### Opik API Key (Optional)

1. Sign up at https://www.comet.com/
2. Create a project for Opik
3. Get your API key
4. Copy to `.env` as `OPIK_API_KEY`

---

## 📧 SMTP Configuration

### Using Gmail

1. Enable 2-Factor Authentication on your Google account
2. Generate an App Password:
   - Go to https://myaccount.google.com/security
   - Navigate to "2-Step Verification"
   - Scroll down to "App passwords"
   - Generate a password for "Mail"
3. Use these settings in `.env`:

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-16-char-app-password
SMTP_FROM=your-email@gmail.com
```

### Using Other SMTP Providers

| Provider | SMTP Host | Port |
|----------|-----------|------|
| Gmail | smtp.gmail.com | 587 |
| Outlook | smtp-mail.outlook.com | 587 |
| SendGrid | smtp.sendgrid.net | 587 |
| Mailgun | smtp.mailgun.org | 587 |
| AWS SES | email-smtp.us-east-1.amazonaws.com | 587 |

---

## 🛡️ Production Security Checklist

### Before Deployment

- [ ] All placeholder values in `.env` replaced
- [ ] Strong, unique passwords generated
- [ ] API keys created and added
- [ ] Fernet and secret keys generated
- [ ] SMTP credentials configured
- [ ] `.env` file permissions set to 600 (`chmod 600 .env`)
- [ ] `.env` confirmed in `.gitignore`
- [ ] No secrets committed to git history

### Docker Compose Security

- [ ] Change default PostgreSQL password
- [ ] Change default MinIO credentials
- [ ] Use Docker secrets for sensitive data
- [ ] Restrict container network access
- [ ] Enable TLS for external services

### Kubernetes Security

- [ ] Use Kubernetes Secrets for all credentials
- [ ] Enable RBAC (Role-Based Access Control)
- [ ] Use Network Policies to restrict traffic
- [ ] Enable Pod Security Policies
- [ ] Use separate namespaces for different environments
- [ ] Encrypt secrets at rest

---

## 🔍 GitGuardian & Secret Scanning

### What is GitGuardian?

GitGuardian scans your GitHub repositories for accidentally committed secrets (API keys, passwords, etc.).

### If You Get an Alert

1. **Immediately rotate the exposed credential**
   - Create a new API key
   - Generate a new password
   - Update your `.env` file

2. **Remove the secret from git history**
   ```bash
   # For recent commits
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch path/to/file" \
     --prune-empty --tag-name-filter cat -- --all

   # Force push (WARNING: This rewrites history)
   git push origin --force --all
   ```

3. **Prevent future exposure**
   - Double-check `.gitignore` includes `.env`
   - Use pre-commit hooks to scan for secrets
   - Enable GitHub secret scanning

### Recommended: Pre-commit Hook

Install `gitleaks` to scan for secrets before committing:

```bash
# Install gitleaks
brew install gitleaks  # macOS
# or
# Download from: https://github.com/gitleaks/gitleaks/releases

# Add pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
gitleaks protect --staged --verbose
EOF

chmod +x .git/hooks/pre-commit
```

---

## 📋 Environment Variables Reference

### Critical Secrets (MUST be changed)

| Variable | Purpose | How to Generate |
|----------|---------|----------------|
| `ANTHROPIC_API_KEY` | Claude API access | console.anthropic.com |
| `OPENAI_API_KEY` | OpenAI API access | platform.openai.com |
| `POSTGRES_PASSWORD` | Database password | `openssl rand -base64 32` |
| `AIRFLOW__CORE__FERNET_KEY` | Airflow encryption | See "Generate Secure Keys" |
| `AIRFLOW__WEBSERVER__SECRET_KEY` | Airflow webserver | See "Generate Secure Keys" |
| `JWT_SECRET_KEY` | JWT token signing | `openssl rand -hex 32` |
| `SMTP_PASSWORD` | Email password | Gmail App Password |

### Moderate Sensitivity

| Variable | Purpose | Notes |
|----------|---------|-------|
| `QDRANT_API_KEY` | Qdrant access (if using cloud) | Optional for local |
| `S3_ACCESS_KEY` | MinIO/S3 access | Change from default |
| `S3_SECRET_KEY` | MinIO/S3 secret | Change from default |
| `OPIK_API_KEY` | Observability platform | Optional |

### Low Sensitivity (Can use defaults)

| Variable | Purpose | Notes |
|----------|---------|-------|
| `LOG_LEVEL` | Logging verbosity | INFO, DEBUG, ERROR |
| `POSTGRES_USER` | Database username | conformai |
| `POSTGRES_DB` | Database name | conformai |
| `LLM_MODEL` | Model selection | claude-3-5-sonnet-20241022 |

---

## 🔐 Additional Security Measures

### 1. Rate Limiting

Configure rate limits in `.env`:

```bash
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

### 2. API Authentication

Enable API key authentication:

```bash
API_KEYS_ENABLED=true
API_KEYS=key1,key2,key3  # Comma-separated list
```

### 3. Enable HTTPS

For production, use HTTPS:

```bash
# Use a reverse proxy (nginx, traefik) with TLS certificates
# Example with Let's Encrypt:
# - Certbot for automated certificate management
# - Configure nginx/traefik to terminate TLS
```

### 4. Database Encryption

Enable PostgreSQL SSL:

```bash
# In PostgreSQL config
ssl = on
ssl_cert_file = '/path/to/server.crt'
ssl_key_file = '/path/to/server.key'
```

### 5. Secrets Management (Production)

Use a dedicated secrets manager:

- **HashiCorp Vault**
- **AWS Secrets Manager**
- **Azure Key Vault**
- **Google Secret Manager**
- **Kubernetes Secrets** (with encryption at rest)

---

## 📞 Security Incident Response

If you suspect a security breach:

1. **Immediately rotate all credentials**
2. **Review access logs** for unauthorized access
3. **Check git history** for exposed secrets
4. **Notify affected users** if data was compromised
5. **Update security measures** to prevent recurrence

---

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [12-Factor App Security](https://12factor.net/config)
- [GitGuardian Documentation](https://docs.gitguardian.com/)
- [Secrets Management Best Practices](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)

---

## ✅ Quick Security Audit Checklist

Run this checklist before every deployment:

```bash
# Check for exposed secrets
git log --all --full-history --source -- .env

# Verify .env is not tracked
git ls-files | grep "^\.env$" || echo "✓ .env not tracked"

# Check file permissions
ls -la .env | grep "^-rw-------" && echo "✓ .env permissions correct" || echo "⚠ Fix: chmod 600 .env"

# Scan for secrets with gitleaks (if installed)
gitleaks detect --verbose || echo "⚠ Install gitleaks for secret scanning"

# Verify strong passwords
grep "CHANGE_THIS" .env && echo "⚠ Replace placeholder values!" || echo "✓ No placeholders found"

# Check SSL/TLS configuration
curl -I https://your-domain.com | grep "Strict-Transport-Security" || echo "⚠ Enable HSTS"
```

---

**Remember**: Security is an ongoing process, not a one-time setup. Regularly review and update your security practices.
