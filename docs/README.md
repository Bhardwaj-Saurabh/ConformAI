# ConformAI Documentation

Welcome to the ConformAI documentation hub. This directory contains all comprehensive guides and documentation for the system.

## 📚 Documentation Index

### Getting Started

| Document | Description |
|----------|-------------|
| [Getting Started Guide](GETTING_STARTED.md) | Complete guide to get ConformAI up and running |
| [Architecture Overview](ARCHITECTURE.md) | System architecture with Mermaid diagrams |
| [Quick Start (Production)](../QUICK_START_PRODUCTION.md) | Fast production deployment guide |
| [Production Readiness](../PRODUCTION_READINESS.md) | Production checklist and guidelines |

### Infrastructure & Deployment

| Document | Description |
|----------|-------------|
| [GitHub Secrets Setup](GITHUB_SECRETS_SETUP.md) | Comprehensive guide for setting up GitHub Secrets for CI/CD |
| [Secrets Quick Reference](SECRETS_QUICK_REFERENCE.md) | One-page quick reference for secrets setup |
| [Docker Compose](../infrastructure/docker-compose.yml) | Local development environment |
| [Kubernetes Manifests](../infrastructure/k8s/) | Production deployment on Azure AKS |

### Data Pipeline & Airflow

| Document | Description |
|----------|-------------|
| [Airflow Quick Start](AIRFLOW_QUICKSTART.md) | Get Airflow running in 5 minutes |
| [Airflow Implementation](AIRFLOW_IMPLEMENTATION_SUMMARY.md) | Complete Airflow implementation details |
| [Data Pipeline Logging](DATA_PIPELINE_LOGGING_SUMMARY.md) | Logging strategy for data pipeline |

### Testing & Quality

| Document | Description |
|----------|-------------|
| [Testing Guide](TESTING.md) | Comprehensive testing guide (unit, integration, E2E) |
| [Security Documentation](SECURITY.md) | Security best practices and guidelines |

### Observability & Monitoring

| Document | Description |
|----------|-------------|
| [Complete Logging Guide](LOGGING_GUIDE.md) | RAG and data pipeline logging |
| [Logging Implementation](LOGGING_IMPLEMENTATION_SUMMARY.md) | RAG logging implementation details |
| [Production Improvements](PRODUCTION_IMPROVEMENTS_SUMMARY.md) | Production-ready improvements |
| [Opik Evaluation System](../shared/evaluation/README.md) | Evaluation datasets and metrics with Opik |

## 📖 Quick Navigation

### For New Users
1. Start with [Getting Started Guide](GETTING_STARTED.md)
2. Review [Architecture Overview](ARCHITECTURE.md)
3. Follow [Airflow Quick Start](AIRFLOW_QUICKSTART.md)
4. Read [Testing Guide](TESTING.md)

### For Deployment
1. Review [Production Readiness](../PRODUCTION_READINESS.md)
2. Follow [GitHub Secrets Setup](GITHUB_SECRETS_SETUP.md)
3. Use [Secrets Quick Reference](SECRETS_QUICK_REFERENCE.md) for quick setup
4. Deploy using [Kubernetes Manifests](../infrastructure/k8s/)

### For Development
1. Check [Architecture Overview](ARCHITECTURE.md)
2. Review [Logging Guide](LOGGING_GUIDE.md)
3. Follow [Testing Guide](TESTING.md)
4. Use [Evaluation System](../shared/evaluation/README.md) for testing RAG performance

### For Operations
1. Read [Production Improvements](PRODUCTION_IMPROVEMENTS_SUMMARY.md)
2. Review [Security Documentation](SECURITY.md)
3. Set up [Opik Evaluation](../shared/evaluation/README.md)
4. Monitor using [Logging Guide](LOGGING_GUIDE.md)

## 🔗 External Resources

### APIs & Endpoints
- **API Gateway**: http://localhost:8000/docs (when running)
- **ReDoc**: http://localhost:8000/redoc (when running)
- **Airflow**: http://localhost:8080 (admin/admin)
- **Qdrant**: http://localhost:6333/dashboard

### Tools & Platforms
- [Opik Workspace](https://www.comet.com/) - Evaluation and observability
- [Azure Portal](https://portal.azure.com/) - Cloud resources
- [GitHub Actions](https://github.com/Bhardwaj-Saurabh/ConformAI/actions) - CI/CD pipelines

## 📝 Document Conventions

### File Naming
- `*_GUIDE.md` - Step-by-step guides
- `*_SUMMARY.md` - Implementation summaries
- `README.md` - Directory overviews
- `QUICK_*.md` - Quick reference guides

### Structure
Each guide includes:
- **Overview** - What the document covers
- **Prerequisites** - What you need before starting
- **Step-by-step instructions** - Numbered steps
- **Examples** - Code examples and commands
- **Troubleshooting** - Common issues and solutions
- **Next steps** - What to do after completing the guide

## 🆘 Getting Help

### Common Issues
1. **Can't start services** → Check [Getting Started Guide](GETTING_STARTED.md)
2. **Docker errors** → Review [Docker Compose](../infrastructure/docker-compose.yml)
3. **Airflow not working** → Follow [Airflow Quick Start](AIRFLOW_QUICKSTART.md)
4. **Tests failing** → Read [Testing Guide](TESTING.md)
5. **Deployment issues** → Check [GitHub Secrets Setup](GITHUB_SECRETS_SETUP.md)

### Support Channels
- **Issues**: [GitHub Issues](https://github.com/Bhardwaj-Saurabh/ConformAI/issues)
- **Documentation**: This directory
- **API Docs**: http://localhost:8000/docs

## 🤝 Contributing to Documentation

When adding new documentation:
1. Follow the naming conventions above
2. Include all standard sections (Overview, Prerequisites, etc.)
3. Add examples and code snippets
4. Update this README.md with a link
5. Test all commands and examples

## 📋 Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| Getting Started | ✅ Complete | 2025-12-23 |
| Architecture | ✅ Complete | 2025-12-23 |
| GitHub Secrets Setup | ✅ Complete | 2025-12-23 |
| Secrets Quick Reference | ✅ Complete | 2025-12-23 |
| Airflow Quick Start | ✅ Complete | 2025-12-20 |
| Testing Guide | ✅ Complete | 2025-12-20 |
| Security | ✅ Complete | 2025-12-20 |
| Logging Guide | ✅ Complete | 2025-12-20 |
| Evaluation System | ✅ Complete | 2025-12-23 |

---

**Note**: All documentation is maintained in this `docs/` directory for easy access and organization. Infrastructure files are in `infrastructure/`, and code documentation is co-located with the code.
