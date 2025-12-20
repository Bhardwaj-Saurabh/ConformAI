# ConformAI Logging & Debugging Guide

**Comprehensive guide to viewing, understanding, and debugging ConformAI system logs**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Log Locations](#log-locations)
- [Log Levels](#log-levels)
- [Viewing Logs](#viewing-logs)
- [Log Format](#log-format)
- [RAG Pipeline Logs](#rag-pipeline-logs)
- [Debugging Common Issues](#debugging-common-issues)
- [Production Log Management](#production-log-management)
- [Log Analysis Tools](#log-analysis-tools)

---

## Overview

ConformAI uses **production-grade structured logging** with:
- ✅ JSON format (machine-readable)
- ✅ Request ID tracking
- ✅ Performance metrics
- ✅ Audit trails
- ✅ Error context
- ✅ Conversation tracking

All logs include rich context for easy debugging and monitoring.

---

## Log Locations

### Development (Docker Compose)

**Console Logs** (default):
```bash
# View all services
docker-compose logs -f

# View specific service
docker-compose logs -f rag-service
docker-compose logs -f retrieval-service
docker-compose logs -f api-gateway

# View with timestamps
docker-compose logs -f --timestamps rag-service
```

**Container Logs**:
```bash
# Access container
docker exec -it conformai-rag-service-1 /bin/bash

# View logs inside container
cat /var/log/conformai/rag-service.log
```

### Production

**Centralized Logging** (recommended):
- **ELK Stack**: Elasticsearch + Logstash + Kibana
- **Loki**: Grafana Loki + Promtail
- **CloudWatch**: AWS CloudWatch Logs
- **Datadog**: Datadog Logs

**File Logs**:
```bash
# Default location
/var/log/conformai/{service-name}.log

# Rotate with logrotate
/etc/logrotate.d/conformai
```

---

## Log Levels

ConformAI uses standard Python logging levels:

| Level | Use Case | Example |
|-------|----------|---------|
| `DEBUG` | Detailed debugging, state transitions | Graph node entry/exit, LLM calls |
| `INFO` | Normal operations, key events | Pipeline start/complete, retrievals |
| `WARNING` | Potential issues, fallbacks | Max iterations reached, validation failed |
| `ERROR` | Error conditions, failures | LLM API errors, retrieval failures |
| `CRITICAL` | System failures | Service crash, database down |

**Configure Level** via `.env`:
```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

---

## Viewing Logs

### 1. Real-Time Monitoring

**Follow logs as they happen:**
```bash
# All services
docker-compose logs -f

# With grep for filtering
docker-compose logs -f | grep ERROR
docker-compose logs -f | grep "request_id=abc123"

# Pretty-print JSON logs (requires jq)
docker-compose logs -f rag-service | jq '.'
```

### 2. Search Historical Logs

**Search within timeframe:**
```bash
# Since specific time
docker-compose logs --since="2025-12-20T14:00:00" rag-service

# Last N minutes
docker-compose logs --since=30m rag-service

# Last N lines
docker-compose logs --tail=100 rag-service
```

### 3. Filter by Context

**Using jq for JSON logs:**
```bash
# Find all errors
docker-compose logs rag-service | jq 'select(.level=="ERROR")'

# Find by request ID
docker-compose logs rag-service | jq 'select(.request_id=="abc123")'

# Find slow queries (> 5 seconds)
docker-compose logs rag-service | jq 'select(.duration_ms > 5000)'

# Find by conversation
docker-compose logs rag-service | jq 'select(.conversation_id=="conv_456")'
```

---

## Log Format

### JSON Structure (Production)

```json
{
  "timestamp": "2025-12-20T14:30:22.123Z",
  "level": "INFO",
  "logger": "graph.graph",
  "message": "RAG pipeline completed successfully",
  "module": "graph",
  "function": "run_rag_pipeline",
  "line": 342,
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "conversation_id": "conv_user123_20251220",
  "processing_time_ms": 2341.5,
  "confidence_score": 0.87,
  "iterations_used": 3,
  "chunks_retrieved": 8,
  "citations_count": 5
}
```

### Human-Readable Format (Development)

```
2025-12-20 14:30:22.123 | INFO | graph.graph:run_rag_pipeline:342 - RAG pipeline completed successfully
  request_id=a1b2c3d4 processing_time_ms=2341.5 confidence=0.87 iterations=3 chunks=8 citations=5
```

---

## RAG Pipeline Logs

### Pipeline Execution Flow

The RAG pipeline logs follow this structure:

```
╔═══════════════════════════════════════════════════════════════════╗
║                 STARTING RAG PIPELINE EXECUTION                   ║
╚═══════════════════════════════════════════════════════════════════╝

┌─ [ANALYSIS] analyze_query - STARTED
  [analyze_query] Analyzing query intent and complexity
  [analyze_query] LLM call completed (234ms)
  [analyze_query] Analysis complete: intent=compliance_question, complexity=medium
└─ [ANALYSIS] analyze_query - COMPLETED (245ms)

┌─ [ANALYSIS] decompose_query - STARTED
  [decompose_query] Decomposing complex query
  [decompose_query] Generated 3 sub-queries
    1. 🔴 [P1] obligations: What are the obligations for high-risk AI?
    2. 🟡 [P2] documentation: What documentation is required?
    3. 🟢 [P3] compliance: What are the compliance procedures?
└─ [ANALYSIS] decompose_query - COMPLETED (456ms)

┌─ [ANALYSIS] safety_check - STARTED
  [safety_check] Checking query safety
  [safety_check] Safety check passed
└─ [ANALYSIS] safety_check - COMPLETED (12ms)

┌─ [AGENT] react_plan - STARTED
  [react_plan] Planning next action (iteration 1/5)
  [react_plan] Decided to retrieve information
└─ [AGENT] react_plan - COMPLETED (189ms)

┌─ [AGENT] react_act - STARTED
  [react_act] Step 1: retrieve_legal_chunks
  [react_act] Retrieved 8 chunks (top_score: 0.92)
└─ [AGENT] react_act - COMPLETED (567ms)

┌─ [AGENT] react_observe - STARTED
  [react_observe] Observing retrieval results
  [react_observe] Agent state: continue
└─ [AGENT] react_observe - COMPLETED (123ms)

... (ReAct loop continues) ...

┌─ [SYNTHESIS] synthesize_answer - STARTED
  [synthesize_answer] Synthesizing final answer
  [synthesize_answer] Answer synthesized: 1234 chars, 5 citations
└─ [SYNTHESIS] synthesize_answer - COMPLETED (1234ms)

┌─ [VALIDATION] validate_grounding - STARTED
  [validate_grounding] Validating answer grounding
  [validate_grounding] Validation PASSED ✓ (score: 0.89)
└─ [VALIDATION] validate_grounding - COMPLETED (345ms)

┌─ [SYNTHESIS] format_response - STARTED
  [format_response] Formatting final response
└─ [SYNTHESIS] format_response - COMPLETED (23ms)

╔═══════════════════════════════════════════════════════════════════╗
║              RAG PIPELINE COMPLETED SUCCESSFULLY                  ║
╚═══════════════════════════════════════════════════════════════════╝

Pipeline results summary:
  - Processing time: 3,456ms
  - Confidence: 0.87
  - Iterations: 3/5
  - Chunks retrieved: 8
  - Citations: 5
  - Answer length: 1234 chars
  - Grounding validated: ✓
```

### Performance Logs

```json
{
  "timestamp": "2025-12-20T14:30:22.456Z",
  "level": "INFO",
  "logger": "shared.utils.logger",
  "message": "Performance metric",
  "operation": "rag_pipeline_execution",
  "duration_ms": 3456.7,
  "confidence_score": 0.87,
  "iterations_used": 3,
  "chunks_retrieved": 8,
  "citations_count": 5,
  "answer_length": 1234,
  "grounding_validated": true
}
```

### Audit Logs

```json
{
  "timestamp": "2025-12-20T14:30:22.456Z",
  "level": "INFO",
  "logger": "shared.utils.logger",
  "message": "Audit event",
  "action": "rag_pipeline_completed",
  "resource": "compliance_query",
  "result": "success",
  "request_id": "a1b2c3d4",
  "processing_time_ms": 3456.7,
  "confidence_score": 0.87
}
```

### Error Logs

```json
{
  "timestamp": "2025-12-20T14:30:25.789Z",
  "level": "ERROR",
  "logger": "graph.nodes.react_agent",
  "message": "Error in node 'react_act'",
  "node_name": "react_act",
  "node_type": "agent",
  "error_type": "ConnectionError",
  "error_message": "Failed to connect to retrieval service",
  "iteration": 2,
  "stack_trace": "...",
  "request_id": "a1b2c3d4"
}
```

---

## Debugging Common Issues

### 1. Slow Query Performance

**Find slow queries:**
```bash
# Queries over 5 seconds
docker-compose logs rag-service | jq 'select(.processing_time_ms > 5000)'
```

**What to check:**
- Number of iterations used (high = inefficient)
- Chunks retrieved (too many = slow)
- LLM call durations
- Retrieval scores (low = irrelevant results)

**Log indicators:**
```json
{
  "processing_time_ms": 12345.6,  // >5s is slow
  "iterations_used": 5,  // Max iterations hit
  "chunks_retrieved": 50,  // Too many chunks
  "confidence_score": 0.45  // Low confidence
}
```

### 2. Low Confidence Answers

**Find low-confidence results:**
```bash
# Confidence below 0.5
docker-compose logs rag-service | jq 'select(.confidence_score < 0.5)'
```

**What to check:**
- Retrieval scores (check if relevant chunks found)
- Sub-queries (check decomposition quality)
- Grounding validation (check if failed)

### 3. Retrieval Failures

**Find retrieval errors:**
```bash
# Errors in retrieval
docker-compose logs rag-service | grep -i "retrieval.*error"
```

**Common causes:**
- Qdrant connection issues
- Empty collection (no documents indexed)
- Embedding service down
- Query too vague

### 4. LLM API Errors

**Find LLM errors:**
```bash
# LLM API failures
docker-compose logs rag-service | jq 'select(.logger | contains("llm"))'
```

**Common causes:**
- API key invalid
- Rate limit exceeded
- Model not available
- Timeout

### 5. Validation Failures

**Find grounding validation failures:**
```bash
# Failed validations
docker-compose logs rag-service | jq 'select(.grounding_validated == false)'
```

**What to check:**
- Are chunks actually relevant?
- Is answer too speculative?
- Are citations missing?

---

## Production Log Management

### 1. Log Aggregation

**ELK Stack Setup:**
```yaml
# docker-compose.logging.yml
version: '3.8'
services:
  elasticsearch:
    image: elasticsearch:8.11.0
    ports:
      - "9200:9200"

  logstash:
    image: logstash:8.11.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline

  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
```

**Logstash Pipeline:**
```
input {
  tcp {
    port => 5000
    codec => json
  }
}

filter {
  # Parse ConformAI logs
  if [service] == "rag-service" {
    mutate {
      add_tag => ["rag"]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "conformai-%{+YYYY.MM.dd}"
  }
}
```

### 2. Log Rotation

**logrotate config** (`/etc/logrotate.d/conformai`):
```
/var/log/conformai/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 conformai conformai
    sharedscripts
    postrotate
        docker-compose restart rag-service
    endscript
}
```

### 3. Log Retention

**Recommended retention periods:**
- **INFO logs**: 30 days
- **WARNING logs**: 90 days
- **ERROR logs**: 1 year
- **Audit logs**: 7 years (compliance)

---

## Log Analysis Tools

### 1. CLI Tools

**jq** (JSON processor):
```bash
# Pretty print
docker-compose logs rag-service | jq '.'

# Filter and format
docker-compose logs rag-service | jq '.message, .duration_ms'

# Complex queries
docker-compose logs rag-service | jq 'select(.level=="ERROR" and .duration_ms > 1000)'
```

**grep** (pattern matching):
```bash
# Case-insensitive search
docker-compose logs -f | grep -i error

# Context (show 5 lines before/after)
docker-compose logs -f | grep -C 5 "request_id=abc123"

# Multiple patterns
docker-compose logs -f | grep -E "(ERROR|WARNING)"
```

**awk** (text processing):
```bash
# Extract specific fields
docker-compose logs rag-service | awk '/duration_ms/{print $NF}'

# Calculate averages
docker-compose logs rag-service | awk '/duration_ms/{sum+=$NF; count++} END{print sum/count}'
```

### 2. Web UI

**Kibana** (ELK Stack):
- Navigate to http://localhost:5601
- Create index pattern: `conformai-*`
- Use Discover for log exploration
- Build dashboards for monitoring

**Grafana Loki**:
- Navigate to http://localhost:3000
- Add Loki data source
- Use LogQL for queries:
  ```
  {service="rag-service"} |= "ERROR"
  ```

### 3. Custom Scripts

**Performance analyzer:**
```bash
#!/bin/bash
# analyze_performance.sh

# Extract performance metrics
docker-compose logs rag-service | \
  jq -r 'select(.processing_time_ms) | "\(.processing_time_ms),\(.confidence_score),\(.iterations_used)"' | \
  awk -F, '{
    sum_time+=$1; sum_conf+=$2; sum_iter+=$3; count++
  } END {
    printf "Average Time: %.2fms\n", sum_time/count
    printf "Average Confidence: %.2f\n", sum_conf/count
    printf "Average Iterations: %.2f\n", sum_iter/count
  }'
```

**Error rate calculator:**
```bash
#!/bin/bash
# error_rate.sh

total=$(docker-compose logs rag-service | jq -r '.level' | wc -l)
errors=$(docker-compose logs rag-service | jq -r 'select(.level=="ERROR")' | wc -l)

echo "Total logs: $total"
echo "Errors: $errors"
echo "Error rate: $(awk "BEGIN {printf \"%.2f%%\", ($errors/$total)*100}")"
```

---

## Quick Reference

### Common Log Queries

```bash
# Find all errors for a request
docker-compose logs rag-service | jq 'select(.request_id=="abc123" and .level=="ERROR")'

# Track full conversation
docker-compose logs rag-service | jq 'select(.conversation_id=="conv_456")'

# Find queries that hit max iterations
docker-compose logs rag-service | jq 'select(.iterations_used >= .max_iterations)'

# Monitor real-time error rate
docker-compose logs -f rag-service | grep ERROR | wc -l

# Get performance stats
docker-compose logs rag-service | jq 'select(.processing_time_ms) | {time: .processing_time_ms, conf: .confidence_score}'
```

### Log Monitoring Alerts

**Set up alerts for:**
- Error rate > 5%
- Average processing time > 5s
- Confidence score < 0.6 consistently
- Max iterations hit frequently (> 30%)
- API rate limiting triggered
- Health check failures

---

## Support

**For logging issues:**
1. Check log level in `.env` (set to DEBUG for detailed logs)
2. Verify logging configuration in `shared/utils/logger.py`
3. Check disk space for log files
4. Review log rotation configuration
5. Open issue with log excerpts (sanitize sensitive data first)

---

**Happy Debugging! 🐛🔍**
