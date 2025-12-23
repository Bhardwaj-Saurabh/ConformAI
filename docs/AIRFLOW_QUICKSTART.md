# Airflow Quick Start Guide

Get the automated EU Legal Documents data pipeline running in 5 minutes.

---

## Prerequisites

✅ **Docker and Docker Compose installed**
✅ **API Keys ready**:
   - OpenAI API key (for embeddings)
   - Anthropic API key (for LLM)
✅ **At least 4GB RAM available**

---

## 1. Quick Setup (Automated)

### Step 1: Clone and Configure

```bash
# Navigate to project directory
cd /path/to/ConformAI

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your favorite editor
```

**Required variables in `.env`**:
```bash
# API Keys (REQUIRED)
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here

# Database (use defaults or customize)
POSTGRES_USER=conformai
POSTGRES_PASSWORD=conformai_password
POSTGRES_DB=conformai

# Email notifications (OPTIONAL but recommended)
ALERT_EMAIL=your-email@example.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=noreply@conformai.com
```

### Step 2: Run Initialization Script

```bash
# Make script executable (if not already)
chmod +x airflow/scripts/init-airflow.sh

# Run initialization
./airflow/scripts/init-airflow.sh
```

The script will:
- ✅ Check dependencies
- ✅ Generate security keys
- ✅ Initialize Airflow database
- ✅ Create admin user
- ✅ Start all services

**That's it!** 🎉

---

## 2. Access Airflow UI

Open your browser: **http://localhost:8080**

**Login credentials**:
- Username: `admin`
- Password: (what you set during initialization, default: `admin`)

---

## 3. Run the Pipeline

### Option A: Via UI (Recommended for first time)

1. Find `eu_legal_documents_pipeline` in the DAG list
2. Click the **toggle switch** to enable it (turns blue)
3. Click the **"Trigger DAG"** button (▶️ play icon)
4. Watch the task execution in real-time!

### Option B: Via Command Line

```bash
# Trigger the pipeline
docker-compose exec airflow-scheduler airflow dags trigger eu_legal_documents_pipeline

# Monitor execution
docker-compose logs -f airflow-worker
```

---

## 4. Monitor Pipeline Execution

### Real-time Task View

In the Airflow UI:
1. Click on the DAG name
2. Select **Graph View** or **Grid View**
3. Click on individual tasks to see logs

**Task stages**:
```
check_prerequisites
    ↓
discover_documents
    ↓
download_documents
    ↓
parse_documents
    ↓
chunk_documents
    ↓
generate_embeddings
    ↓
index_to_qdrant
    ↓
send_completion_notification
```

### View Logs

**Via UI**: Click task → View Log

**Via CLI**:
```bash
# Scheduler logs
docker-compose logs -f airflow-scheduler

# Worker logs (actual task execution)
docker-compose logs -f airflow-worker

# Specific task log
docker-compose exec airflow-scheduler airflow tasks logs \
  eu_legal_documents_pipeline discover_documents 2025-12-20
```

---

## 5. Understanding the Pipeline

### What It Does

1. **Checks prerequisites** - Validates API keys, database connections, disk space
2. **Discovers documents** - Queries EUR-Lex API for new/updated regulations
3. **Downloads documents** - Fetches full XML documents (default: 5 documents)
4. **Parses documents** - Extracts legal structure (chapters, articles, paragraphs)
5. **Chunks documents** - Creates legal-aware text chunks
6. **Generates embeddings** - Creates vector embeddings using OpenAI
7. **Indexes to Qdrant** - Stores vectors in Qdrant database
8. **Sends notification** - Emails summary with success/failure metrics

### Default Schedule

- **When**: Daily at 2:00 AM UTC
- **Automatic retries**: 2 attempts with exponential backoff
- **Email on failure**: Yes (if SMTP configured)

### Customize Pipeline Parameters

```bash
# Run with custom parameters
docker-compose exec airflow-scheduler airflow dags trigger \
  eu_legal_documents_pipeline \
  --conf '{
    "limit": 10,
    "recreate_collection": false,
    "start_date": "2024-01-01"
  }'
```

---

## 6. Verify Pipeline Results

### Check Qdrant Database

```bash
# Check number of indexed documents
docker-compose exec qdrant wget -qO- http://localhost:6333/collections/eu_legal_documents

# Or use Python
docker-compose exec airflow-worker python -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='http://qdrant:6333')
info = client.get_collection('eu_legal_documents')
print(f'Total points indexed: {info.points_count}')
"
```

### Check Pipeline Metrics

In Airflow UI:
1. Click on the completed DAG run
2. Click on `send_completion_notification` task
3. View the log to see full summary

**Example summary**:
```
Pipeline Execution Summary
==========================
✓ Documents processed: 5
✓ Total chunks: 1,234
✓ Embeddings generated: 1,234
✓ Points indexed: 1,234
✓ Success rate: 100%
✓ Total duration: 3m 45s
```

---

## 7. Troubleshooting

### Pipeline not visible

```bash
# Check for DAG parsing errors
docker-compose exec airflow-scheduler airflow dags list

# Check scheduler logs
docker-compose logs airflow-scheduler | tail -50

# Restart scheduler
docker-compose restart airflow-scheduler
```

### Tasks failing

```bash
# View detailed error logs
docker-compose logs airflow-worker | grep ERROR

# Check specific task
docker-compose exec airflow-scheduler airflow tasks test \
  eu_legal_documents_pipeline check_prerequisites 2025-12-20

# Verify environment variables
docker-compose exec airflow-worker env | grep -E "API_KEY|QDRANT"
```

### Database connection issues

```bash
# Check if services are running
docker-compose ps

# Restart all services
docker-compose down
docker-compose up -d

# Check Qdrant is accessible
curl http://localhost:6333/collections
```

### Email notifications not working

```bash
# Test SMTP configuration
docker-compose exec airflow-scheduler python -c "
import smtplib
from email.mime.text import MIMEText
msg = MIMEText('Test from Airflow')
msg['Subject'] = 'Test'
msg['From'] = 'noreply@conformai.com'
msg['To'] = 'your-email@example.com'
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('your-email@gmail.com', 'your-app-password')
server.send_message(msg)
server.quit()
print('Email sent!')
"
```

**For Gmail**: Use an [App Password](https://support.google.com/accounts/answer/185833), not your regular password.

---

## 8. Stop/Restart Services

### Stop all services

```bash
docker-compose down
```

### Stop only Airflow (keep database)

```bash
docker-compose stop airflow-webserver airflow-scheduler airflow-worker
```

### Restart Airflow

```bash
docker-compose restart airflow-webserver airflow-scheduler airflow-worker
```

### Complete reset (⚠️ deletes all data)

```bash
docker-compose down -v  # Deletes volumes
./airflow/scripts/init-airflow.sh  # Re-initialize
```

---

## 9. Production Checklist

Before deploying to production:

- [ ] Change default admin password
- [ ] Generate new Fernet key and webserver secret
- [ ] Configure proper SMTP for email alerts
- [ ] Set up log aggregation (ELK, Loki)
- [ ] Enable HTTPS for Airflow UI
- [ ] Configure proper backup strategy for PostgreSQL
- [ ] Set resource limits in docker-compose
- [ ] Enable authentication/authorization
- [ ] Monitor disk space and logs rotation
- [ ] Set up external secrets management

See [airflow/README.md](./airflow/README.md) for detailed production setup.

---

## 10. Next Steps

### Customize the Pipeline

Edit [`airflow/dags/eu_legal_documents_pipeline.py`](./airflow/dags/eu_legal_documents_pipeline.py):

- Adjust schedule: Change `schedule_interval`
- Modify retry logic: Change `retries` and `retry_delay`
- Customize email notifications: Update email templates
- Add new tasks: Extend the DAG with additional processing

### Integrate with RAG Service

The indexed documents are now available for querying:

```bash
# Start RAG service
docker-compose up -d rag-service

# Query the indexed documents
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the obligations for high-risk AI systems?",
    "conversation_id": null
  }'
```

### Monitor and Maintain

- **Daily**: Check DAG runs in UI
- **Weekly**: Review email notifications
- **Monthly**: Clean up old logs and DAG runs
- **Quarterly**: Update dependencies and security patches

---

## Quick Reference

### Useful Commands

```bash
# View all DAGs
docker-compose exec airflow-scheduler airflow dags list

# Trigger pipeline
docker-compose exec airflow-scheduler airflow dags trigger eu_legal_documents_pipeline

# View DAG runs
docker-compose exec airflow-scheduler airflow dags list-runs -d eu_legal_documents_pipeline

# Pause/Unpause DAG
docker-compose exec airflow-scheduler airflow dags pause eu_legal_documents_pipeline
docker-compose exec airflow-scheduler airflow dags unpause eu_legal_documents_pipeline

# View logs
docker-compose logs -f airflow-scheduler
docker-compose logs -f airflow-worker

# Check service health
docker-compose ps
```

### Important URLs

- **Airflow UI**: http://localhost:8080
- **Qdrant UI**: http://localhost:6333/dashboard
- **RAG Service**: http://localhost:8001
- **API Gateway**: http://localhost:8000

### Documentation

- [Airflow Setup README](./airflow/README.md) - Detailed setup guide
- [Data Pipeline Logging Guide](./LOGGING_GUIDE.md) - Logging documentation
- [Production Readiness](./PRODUCTION_READINESS.md) - Production checklist
- [Apache Airflow Docs](https://airflow.apache.org/docs/) - Official documentation

---

## Support

**Issues with setup?**
1. Check the [Troubleshooting](#7-troubleshooting) section
2. Review logs: `docker-compose logs -f`
3. Consult [airflow/README.md](./airflow/README.md)

**Need help?**
- Check Airflow UI task logs
- Review Docker Compose service status
- Verify environment variables are set correctly

---

**🎉 Happy pipelining!**
