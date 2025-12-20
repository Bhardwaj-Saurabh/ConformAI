# Airflow Setup for ConformAI Data Pipeline

This directory contains Apache Airflow configuration for automated orchestration of the EU Legal Documents data pipeline.

---

## Overview

The Airflow setup provides:
- **Automated daily data pipeline execution** (2 AM UTC)
- **Retry logic with exponential backoff**
- **Email notifications on failure**
- **Comprehensive monitoring and logging**
- **Task dependency management**
- **Production-ready error handling**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Airflow Infrastructure                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Webserver  │  │  Scheduler   │  │    Worker    │       │
│  │  (UI/API)   │  │ (Orchestrate)│  │  (Execute)   │       │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                │                  │                │
│         └────────────────┼──────────────────┘                │
│                          │                                   │
│                    ┌─────▼─────┐                            │
│                    │ PostgreSQL │  (Metadata DB)             │
│                    └───────────┘                             │
│                          │                                   │
│                    ┌─────▼─────┐                            │
│                    │   Redis    │  (Message Broker)          │
│                    └───────────┘                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## DAGs

### 1. `eu_legal_documents_pipeline.py`

**Purpose**: Complete automated data pipeline for EU legal documents

**Schedule**: Daily at 2 AM UTC

**Tasks**:
1. **check_prerequisites** - Validate API keys, database connections, disk space
2. **discover_documents** - Find new documents from EUR-Lex API
3. **download_documents** - Download documents in XML format
4. **parse_documents** - Extract legal structure (chapters, articles)
5. **chunk_documents** - Create legal-aware text chunks
6. **generate_embeddings** - Generate embeddings using OpenAI
7. **index_to_qdrant** - Index vectors into Qdrant database
8. **send_completion_notification** - Send email summary with metrics

**Features**:
- 2 automatic retries with exponential backoff
- Email notifications on failure
- XCom for inter-task data transfer
- Comprehensive error handling
- Progress tracking and metrics

---

## Quick Start

### 1. Prerequisites

**Required Environment Variables**:
```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Database Configuration
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=...  # Optional
QDRANT_COLLECTION_NAME=eu_legal_documents

# PostgreSQL (for Airflow metadata)
POSTGRES_USER=conformai
POSTGRES_PASSWORD=conformai_password
POSTGRES_DB=conformai

# Email Notifications (Optional but recommended)
ALERT_EMAIL=admin@conformai.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=noreply@conformai.com

# Airflow Security Keys (Generate new ones for production!)
AIRFLOW__CORE__FERNET_KEY=...  # Generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
AIRFLOW__WEBSERVER__SECRET_KEY=...  # Any random string
```

### 2. Initialize Airflow Database

**First time setup**:
```bash
# Initialize Airflow database
docker-compose run --rm airflow-webserver airflow db init

# Create admin user
docker-compose run --rm airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@conformai.com \
    --password admin
```

Or use the initialization script:
```bash
bash airflow/scripts/init-airflow.sh
```

### 3. Start Airflow Services

```bash
# Start all Airflow components
docker-compose up -d airflow-webserver airflow-scheduler airflow-worker

# Check logs
docker-compose logs -f airflow-webserver
docker-compose logs -f airflow-scheduler
docker-compose logs -f airflow-worker
```

### 4. Access Airflow UI

Open http://localhost:8080

**Default credentials**:
- Username: `admin`
- Password: `admin` (or what you set during initialization)

---

## Using the DAG

### Enable the DAG

1. Navigate to http://localhost:8080
2. Find `eu_legal_documents_pipeline` in the DAG list
3. Toggle the switch to **ON** (unpause)

### Trigger Manual Execution

**Via UI**:
1. Click on the DAG name
2. Click "Trigger DAG" button (play icon)
3. Optionally configure parameters
4. Click "Trigger"

**Via CLI**:
```bash
# Trigger DAG
docker-compose exec airflow-scheduler airflow dags trigger eu_legal_documents_pipeline

# Trigger with custom config
docker-compose exec airflow-scheduler airflow dags trigger eu_legal_documents_pipeline \
  --conf '{"limit": 10, "recreate_collection": false}'
```

### Monitor Execution

**Real-time monitoring**:
```bash
# View DAG run status
docker-compose exec airflow-scheduler airflow dags list-runs -d eu_legal_documents_pipeline

# View task instance status
docker-compose exec airflow-scheduler airflow tasks list eu_legal_documents_pipeline

# View logs for specific task
docker-compose exec airflow-scheduler airflow tasks logs eu_legal_documents_pipeline discover_documents 2025-12-20
```

**Via UI**:
1. Click on DAG name
2. Click on specific run (Graph/Tree/Gantt view)
3. Click on task to see logs and details

---

## Configuration

### DAG Parameters

Edit [`dags/eu_legal_documents_pipeline.py`](./dags/eu_legal_documents_pipeline.py) to customize:

```python
# Schedule (cron format)
schedule_interval="0 2 * * *"  # Daily at 2 AM UTC

# Retry configuration
"retries": 2,
"retry_delay": timedelta(minutes=5),
"retry_exponential_backoff": True,
"max_retry_delay": timedelta(minutes=30),

# Email configuration
"email": os.getenv("ALERT_EMAIL", "admin@conformai.com"),
"email_on_failure": True,
"email_on_retry": False,
```

### Pipeline Parameters

Configure via environment variables or DAG config:

```python
# In task functions, access via:
limit = context.get("params", {}).get("limit", 5)
recreate_collection = context.get("params", {}).get("recreate_collection", False)
```

**Trigger with parameters**:
```bash
docker-compose exec airflow-scheduler airflow dags trigger eu_legal_documents_pipeline \
  --conf '{"limit": 20, "recreate_collection": true, "start_date": "2024-01-01"}'
```

---

## Troubleshooting

### Common Issues

#### 1. DAG not appearing in UI

**Symptoms**: DAG doesn't show up in Airflow UI

**Solutions**:
```bash
# Check for Python errors in DAG file
docker-compose exec airflow-scheduler airflow dags list

# Check scheduler logs
docker-compose logs airflow-scheduler | grep eu_legal_documents_pipeline

# Force DAG refresh
docker-compose restart airflow-scheduler
```

#### 2. Import errors

**Symptoms**: Tasks fail with `ModuleNotFoundError`

**Solutions**:
```bash
# Verify shared modules are mounted
docker-compose exec airflow-worker ls -la /opt/airflow/shared

# Check Python path
docker-compose exec airflow-worker python -c "import sys; print('\n'.join(sys.path))"

# Rebuild Airflow containers
docker-compose build airflow-webserver airflow-scheduler airflow-worker
docker-compose up -d airflow-webserver airflow-scheduler airflow-worker
```

#### 3. Database connection errors

**Symptoms**: Tasks fail with Qdrant/PostgreSQL connection errors

**Solutions**:
```bash
# Check if services are running
docker-compose ps qdrant postgres

# Test Qdrant connection from worker
docker-compose exec airflow-worker python -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='http://qdrant:6333')
print(client.get_collections())
"

# Verify environment variables
docker-compose exec airflow-worker env | grep -E "QDRANT|OPENAI|ANTHROPIC"
```

#### 4. Email notifications not working

**Symptoms**: No emails received on task failure

**Solutions**:
```bash
# Test SMTP configuration
docker-compose exec airflow-scheduler python -c "
import smtplib
from email.mime.text import MIMEText
import os

msg = MIMEText('Test email from Airflow')
msg['Subject'] = 'Airflow Email Test'
msg['From'] = os.getenv('SMTP_FROM', 'noreply@conformai.com')
msg['To'] = os.getenv('ALERT_EMAIL', 'admin@conformai.com')

with smtplib.SMTP(os.getenv('SMTP_HOST', 'smtp.gmail.com'), int(os.getenv('SMTP_PORT', 587))) as server:
    server.starttls()
    server.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASSWORD'))
    server.send_message(msg)
print('Email sent successfully!')
"

# Configure Airflow email settings (add to docker-compose.yml)
# environment:
#   - AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
#   - AIRFLOW__SMTP__SMTP_PORT=587
#   - AIRFLOW__SMTP__SMTP_USER=your-email@gmail.com
#   - AIRFLOW__SMTP__SMTP_PASSWORD=your-app-password
#   - AIRFLOW__SMTP__SMTP_MAIL_FROM=noreply@conformai.com
```

#### 5. Tasks stuck in "running" state

**Symptoms**: Tasks show as running but no progress

**Solutions**:
```bash
# Check worker health
docker-compose exec airflow-worker airflow celery worker --help

# Check Redis connection
docker-compose exec airflow-worker redis-cli -h redis ping

# Clear task state (DANGEROUS - use with caution)
docker-compose exec airflow-scheduler airflow tasks clear eu_legal_documents_pipeline

# Restart worker
docker-compose restart airflow-worker
```

---

## Monitoring and Maintenance

### View Logs

**Scheduler logs**:
```bash
docker-compose logs -f airflow-scheduler
```

**Worker logs**:
```bash
docker-compose logs -f airflow-worker
```

**Task logs** (stored in volume):
```bash
docker-compose exec airflow-scheduler ls -la /opt/airflow/logs/dag_id=eu_legal_documents_pipeline/
```

**View specific task log**:
```bash
docker-compose exec airflow-scheduler cat /opt/airflow/logs/dag_id=eu_legal_documents_pipeline/run_id=manual__2025-12-20T00:00:00+00:00/task_id=discover_documents/attempt=1.log
```

### Database Maintenance

**Check DAG runs**:
```bash
docker-compose exec airflow-scheduler airflow dags list-runs -d eu_legal_documents_pipeline --state success
docker-compose exec airflow-scheduler airflow dags list-runs -d eu_legal_documents_pipeline --state failed
```

**Clean up old DAG runs** (keeps last 30 days):
```bash
docker-compose exec airflow-scheduler airflow db clean --clean-before-timestamp "$(date -d '30 days ago' '+%Y-%m-%d')" --yes
```

### Performance Monitoring

**Check task duration**:
```sql
-- Via PostgreSQL
docker-compose exec postgres psql -U conformai -d conformai -c "
SELECT task_id, AVG(duration) as avg_duration_sec
FROM task_instance
WHERE dag_id = 'eu_legal_documents_pipeline'
  AND state = 'success'
GROUP BY task_id
ORDER BY avg_duration_sec DESC;
"
```

**Check success rate**:
```bash
docker-compose exec airflow-scheduler airflow dags show eu_legal_documents_pipeline
```

---

## Production Deployment

### Security Hardening

1. **Generate secure Fernet key**:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

2. **Use secrets management** (Kubernetes, AWS Secrets Manager, etc.)

3. **Enable RBAC** (Role-Based Access Control):
```python
# In airflow.cfg or environment
AIRFLOW__WEBSERVER__RBAC = True
```

4. **Configure HTTPS for webserver**:
```python
AIRFLOW__WEBSERVER__WEB_SERVER_SSL_CERT = /path/to/cert.pem
AIRFLOW__WEBSERVER__WEB_SERVER_SSL_KEY = /path/to/key.pem
```

### Scaling

**Increase workers**:
```bash
docker-compose up -d --scale airflow-worker=3
```

**Configure worker concurrency**:
```python
# In docker-compose.yml
environment:
  - AIRFLOW__CELERY__WORKER_CONCURRENCY=8
```

### Backup and Recovery

**Backup Airflow metadata database**:
```bash
docker-compose exec postgres pg_dump -U conformai conformai > airflow_backup_$(date +%Y%m%d).sql
```

**Restore database**:
```bash
docker-compose exec -T postgres psql -U conformai conformai < airflow_backup_20251220.sql
```

---

## Development

### Adding New DAGs

1. Create new DAG file in `airflow/dags/`
2. Follow naming convention: `<name>_pipeline.py`
3. Use shared utilities from `shared/` directory
4. Test locally before deploying

**Template**:
```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "conformai",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="my_new_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

def my_task(**context):
    print("Executing task...")

task1 = PythonOperator(
    task_id="my_task",
    python_callable=my_task,
    dag=dag,
)
```

### Testing DAGs Locally

```bash
# Test DAG parsing
docker-compose exec airflow-scheduler python /opt/airflow/dags/eu_legal_documents_pipeline.py

# Test specific task
docker-compose exec airflow-scheduler airflow tasks test eu_legal_documents_pipeline check_prerequisites 2025-12-20

# Run full DAG in backfill mode
docker-compose exec airflow-scheduler airflow dags backfill eu_legal_documents_pipeline \
  --start-date 2025-12-20 \
  --end-date 2025-12-20
```

---

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Celery Executor](https://airflow.apache.org/docs/apache-airflow/stable/executor/celery.html)
- [ConformAI Data Pipeline Logging Guide](../LOGGING_GUIDE.md)
- [Data Pipeline Logging Summary](../DATA_PIPELINE_LOGGING_SUMMARY.md)

---

## Support

For issues related to:
- **Airflow setup**: Check logs and troubleshooting section above
- **Pipeline failures**: Review task logs and error messages
- **Configuration**: Refer to environment variables section
- **Performance**: Monitor task durations and adjust worker concurrency

**Logs location**: `/opt/airflow/logs/` (inside Airflow containers)
