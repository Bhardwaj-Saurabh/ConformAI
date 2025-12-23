# ConformAI Memory System

Complete guide to conversation persistence and user-specific long-term memory in the ConformAI RAG system.

## Overview

The ConformAI memory system provides:

1. **Conversation Memory** - Multi-turn conversations with persistent message history
2. **User Memory** - Long-term storage of user facts, preferences, and context
3. **LangGraph State Persistence** - Checkpointing of RAG pipeline state
4. **Memory-Aware RAG** - Context-enriched query processing using conversation and user history

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Pipeline                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Memory Retrieval → Query Analysis → ReAct Agent      │  │
│  │      ↓                                     ↓            │  │
│  │  User Context → Synthesis → Memory Storage            │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
           ↓                                    ↓
┌──────────────────────┐          ┌───────────────────────────┐
│ Conversation Memory  │          │     User Memory           │
├──────────────────────┤          ├───────────────────────────┤
│ - Conversation       │          │ - Facts                   │
│ - Messages           │          │ - Preferences             │
│ - Summaries          │          │ - Interactions            │
└──────────────────────┘          │ - Context                 │
           ↓                       └───────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│              PostgreSQL Database                             │
│  - users                                                      │
│  - conversations                                              │
│  - messages                                                   │
│  - user_memories                                              │
│  - conversation_summaries                                     │
│  - langgraph_checkpoints (LangGraph state persistence)       │
└──────────────────────────────────────────────────────────────┘
```

### Database Schema

#### users
- `id` (PK): Internal user ID
- `user_id`: External user identifier (unique)
- `email`: User email (optional)
- `full_name`: User's full name
- `created_at`: Account creation timestamp
- `last_active`: Last activity timestamp

#### conversations
- `id` (PK): Internal conversation ID
- `conversation_id`: External conversation identifier (unique)
- `user_id` (FK): References users.id
- `title`: Conversation title
- `message_count`: Total messages in conversation
- `created_at`: Conversation creation timestamp
- `updated_at`: Last update timestamp
- `last_message_at`: Last message timestamp
- `is_active`: Active status (0 = archived)

#### messages
- `id` (PK): Internal message ID
- `conversation_id` (FK): References conversations.id
- `role`: Message role (user/assistant/system)
- `content`: Message content
- `metadata`: JSON metadata (citations, confidence, etc.)
- `sequence_number`: Order within conversation
- `created_at`: Message timestamp

#### user_memories
- `id` (PK): Internal memory ID
- `user_id` (FK): References users.id
- `memory_type`: Type (fact/preference/interaction/context)
- `key`: Memory key
- `value`: Memory value
- `source_conversation_id`: Originating conversation
- `confidence`: Confidence score (1-10)
- `importance`: Importance score (1-10)
- `access_count`: Number of times accessed
- `last_accessed`: Last access timestamp
- `created_at`: Memory creation timestamp
- `updated_at`: Last update timestamp

#### conversation_summaries
- `id` (PK): Internal summary ID
- `conversation_id` (FK): References conversations.id
- `summary_text`: Conversation summary
- `summary_type`: Summary type (periodic/final)
- `message_range_start`: Start sequence number
- `message_range_end`: End sequence number
- `created_at`: Summary creation timestamp

## Setup

### 1. Initialize Database Tables

```bash
# Using Makefile
make db-init-memory

# Or directly
python scripts/init_memory_database.py
```

This creates:
- users
- conversations
- messages
- user_memories
- conversation_summaries
- langgraph_checkpoints (created by LangGraph)

### 2. Configure Settings

Ensure your `.env` file has the correct PostgreSQL URL:

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/conformai
```

### 3. Start Services

```bash
# Start PostgreSQL (if using Docker Compose)
make docker-up

# Start RAG service
cd services/rag-service
python -m src.api.main
```

## Usage

### API Endpoints

#### 1. Create Conversation

```bash
POST /api/v1/conversations/create
{
  "user_id": "user-123",
  "title": "EU AI Act Compliance Questions"
}
```

**Response:**
```json
{
  "conversation_id": "conv-456",
  "user_id": "user-123",
  "title": "EU AI Act Compliance Questions",
  "created_at": "2025-01-15T10:30:00Z"
}
```

#### 2. Query with Memory

```bash
POST /api/v1/query
{
  "query": "What are high-risk AI systems?",
  "user_id": "user-123",
  "conversation_id": "conv-456",
  "max_iterations": 5
}
```

The RAG pipeline will:
1. Retrieve conversation history (last 10 messages)
2. Load user's long-term memories
3. Process query with enriched context
4. Store query and response in conversation
5. Extract and save new user memories

**Response:**
```json
{
  "success": true,
  "query": "What are high-risk AI systems?",
  "answer": "High-risk AI systems are...",
  "citations": [...],
  "metadata": {
    "confidence_score": 0.92,
    "processing_time_ms": 2500,
    "iterations": 3
  }
}
```

#### 3. List User Conversations

```bash
POST /api/v1/conversations/list
{
  "user_id": "user-123",
  "limit": 20,
  "include_archived": false
}
```

**Response:**
```json
{
  "user_id": "user-123",
  "total": 5,
  "conversations": [
    {
      "conversation_id": "conv-456",
      "title": "EU AI Act Compliance Questions",
      "message_count": 12,
      "created_at": "2025-01-15T10:30:00Z",
      "updated_at": "2025-01-15T12:45:00Z",
      "is_active": true
    }
  ]
}
```

#### 4. Get Conversation History

```bash
POST /api/v1/conversations/history
{
  "conversation_id": "conv-456",
  "limit": 10
}
```

**Response:**
```json
{
  "conversation_id": "conv-456",
  "total_messages": 10,
  "messages": [
    {
      "role": "user",
      "content": "What are high-risk AI systems?",
      "created_at": "2025-01-15T10:35:00Z",
      "sequence_number": 1,
      "metadata": null
    },
    {
      "role": "assistant",
      "content": "High-risk AI systems are...",
      "created_at": "2025-01-15T10:35:03Z",
      "sequence_number": 2,
      "metadata": {
        "confidence_score": 0.92,
        "citations_count": 5
      }
    }
  ]
}
```

#### 5. Get User Memories

```bash
POST /api/v1/conversations/memories
{
  "user_id": "user-123",
  "memory_type": "preference",
  "limit": 50
}
```

**Response:**
```json
{
  "user_id": "user-123",
  "total": 8,
  "memories": [
    {
      "type": "preference",
      "key": "preferred_regulation",
      "value": "EU AI Act",
      "confidence": 9,
      "importance": 8,
      "created_at": "2025-01-15T10:35:05Z",
      "updated_at": "2025-01-15T10:35:05Z"
    },
    {
      "type": "context",
      "key": "primary_ai_domain",
      "value": "biometrics",
      "confidence": 8,
      "importance": 7,
      "created_at": "2025-01-15T10:40:12Z",
      "updated_at": "2025-01-15T10:40:12Z"
    }
  ]
}
```

#### 6. Archive Conversation

```bash
POST /api/v1/conversations/conv-456/archive
```

**Response:**
```json
{
  "success": true,
  "conversation_id": "conv-456",
  "status": "archived"
}
```

## How Memory Works

### Conversation Flow

```
┌─────────────────────────────────────────────────────────────┐
│ User Query                                                   │
│ - query: "What are biometric AI regulations?"               │
│ - user_id: "user-123"                                        │
│ - conversation_id: "conv-456"                                │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Memory Retrieval Phase                                       │
├─────────────────────────────────────────────────────────────┤
│ 1. Retrieve Conversation Context                            │
│    - Load last 10 messages from conversation                │
│    - Generate conversation summary                          │
│                                                              │
│ 2. Retrieve User Memory                                     │
│    - Load user facts (e.g., "works in recruitment")        │
│    - Load preferences (e.g., "prefers EU AI Act focus")     │
│    - Load interaction history                               │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Query Analysis (Memory-Aware)                               │
├─────────────────────────────────────────────────────────────┤
│ - Analyze intent with conversation context                  │
│ - Consider user's domain preferences                        │
│ - Identify follow-up questions                              │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ ReAct Agent Loop                                             │
│ - Plan with user context                                    │
│ - Retrieve relevant legal sources                           │
│ - Synthesize answer                                          │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Memory Storage Phase                                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Store Conversation Message                               │
│    - Save user query as message                             │
│    - Save assistant response with metadata                  │
│    - Update conversation timestamp                          │
│                                                              │
│ 2. Extract User Memories                                    │
│    - Extract facts from conversation                        │
│      (e.g., "user is deploying biometric systems")          │
│    - Extract preferences                                    │
│      (e.g., "user interested in GDPR compliance")           │
│    - Store with confidence and importance scores            │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Response to User                                             │
│ - Final answer with citations                               │
│ - Metadata (confidence, processing time, etc.)              │
└─────────────────────────────────────────────────────────────┘
```

### Memory Types

#### 1. Conversation Memory (Short-Term)
- **Scope**: Within a single conversation thread
- **Duration**: Until conversation is archived/deleted
- **Purpose**: Maintain context across multiple queries in a conversation
- **Example**:
  - User: "What are high-risk AI systems?"
  - Assistant: "High-risk AI systems include..."
  - User: "What are the documentation requirements?" ← Uses context from previous question

#### 2. User Memory (Long-Term)
- **Scope**: Across all user's conversations
- **Duration**: Persistent until explicitly deleted
- **Purpose**: Remember user facts, preferences, and recurring context
- **Types**:
  - **fact**: Objective information about the user
    - Example: "User works in healthcare AI"
  - **preference**: User's explicit preferences
    - Example: "Prefers detailed citations"
  - **interaction**: Interaction patterns
    - Example: "Frequently asks about GDPR"
  - **context**: Ongoing context
    - Example: "Building biometric AI system"

### Memory Extraction

The `extract_user_memories` node uses an LLM to analyze conversations and extract:

1. **User Facts**
   - Industry/domain
   - Role/position
   - Current projects

2. **User Preferences**
   - Preferred regulations
   - Desired level of detail
   - Citation preferences

3. **Recurring Context**
   - AI domain focus (biometrics, recruitment, etc.)
   - Risk categories of interest
   - Specific use cases

### Memory Usage in Query Processing

When a query is processed with memory context:

1. **Conversation Context Summary**:
   ```
   "User has been asking about high-risk AI systems in recruitment.
   Previous questions covered: obligations, documentation, risk assessment."
   ```

2. **User Profile**:
   ```json
   {
     "domain": "recruitment",
     "regulation_focus": "EU AI Act",
     "use_case": "CV screening system"
   }
   ```

3. **Enhanced Query Analysis**:
   - Detects follow-up questions
   - Maintains consistent domain focus
   - Personalizes answer depth based on user expertise

## LangGraph State Persistence

The system uses [LangGraph PostgreSQL Checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#postgresaver) for state persistence.

### What is Checkpointed

- Full RAGState at each node transition
- Agent actions and observations
- Retrieved chunks and citations
- Intermediate reasoning steps

### Benefits

1. **Resume Interrupted Queries**: Resume processing if service crashes
2. **Replay State**: Debug issues by replaying state transitions
3. **Audit Trail**: Complete history of query processing
4. **Thread-Based Conversations**: Each conversation_id gets its own thread

### Thread Configuration

```python
# Each conversation_id maps to a LangGraph thread
config = {
    "configurable": {
        "thread_id": conversation_id,  # "conv-456"
        "checkpoint_ns": "rag_pipeline",
    }
}

# Invoke graph with checkpointing
result = await graph.ainvoke(initial_state, config=config)
```

## Programmatic Usage

### Python SDK

```python
from shared.memory.manager import get_conversation_manager, get_user_memory_manager
from shared.models.conversation import MessageRole

# Initialize managers
conv_manager = get_conversation_manager()
memory_manager = get_user_memory_manager()

# Create conversation
conversation_id = conv_manager.create_conversation(
    user_id="user-123",
    title="AI Compliance Questions"
)

# Add messages
conv_manager.add_message(
    conversation_id=conversation_id,
    role=MessageRole.USER,
    content="What are high-risk AI systems?"
)

conv_manager.add_message(
    conversation_id=conversation_id,
    role=MessageRole.ASSISTANT,
    content="High-risk AI systems include...",
    metadata={
        "confidence_score": 0.92,
        "citations_count": 5
    }
)

# Retrieve conversation history
messages = conv_manager.get_conversation_history(
    conversation_id=conversation_id,
    limit=10
)

# Store user memory
memory_manager.store_memory(
    user_id="user-123",
    memory_type="context",
    key="primary_ai_domain",
    value="biometrics",
    confidence=8,
    importance=7
)

# Retrieve user memories
memories = memory_manager.get_user_memories(
    user_id="user-123",
    memory_type="preference",
    limit=50
)

# Get specific memory
memory = memory_manager.get_memory(
    user_id="user-123",
    key="primary_ai_domain"
)

# Delete memory
success = memory_manager.delete_memory(
    user_id="user-123",
    key="outdated_preference"
)
```

## Best Practices

### 1. User Identification

- Use consistent `user_id` across sessions
- For anonymous users, generate a UUID and store in client session
- Consider user authentication for production

### 2. Conversation Management

- Create new conversation for each distinct topic
- Archive old conversations to keep active list manageable
- Use descriptive conversation titles

### 3. Memory Hygiene

- Periodically review and clean up outdated memories
- Update confidence/importance scores as needed
- Delete sensitive information when no longer needed

### 4. Privacy Considerations

- Only store essential user information
- Implement data retention policies
- Provide user access to view/delete their data
- Consider GDPR compliance for EU users

### 5. Performance Optimization

- Limit conversation history retrieval (default: 10 messages)
- Cache frequently accessed user memories
- Archive inactive conversations
- Use database indexes on user_id, conversation_id

## Monitoring

### Key Metrics

```python
# Log memory-related metrics
logger.info("Memory retrieval stats", extra={
    "conversation_messages": len(messages),
    "user_memories": len(memories),
    "memory_types": list(set(m["type"] for m in memories))
})
```

### Database Performance

```sql
-- Check conversation counts per user
SELECT user_id, COUNT(*) as conversation_count
FROM conversations
WHERE is_active = 1
GROUP BY user_id
ORDER BY conversation_count DESC;

-- Check message volume
SELECT
    c.conversation_id,
    c.title,
    COUNT(m.id) as message_count
FROM conversations c
LEFT JOIN messages m ON c.id = m.conversation_id
GROUP BY c.id
ORDER BY message_count DESC
LIMIT 10;

-- Check user memory distribution
SELECT
    memory_type,
    COUNT(*) as count,
    AVG(importance) as avg_importance
FROM user_memories
GROUP BY memory_type;
```

## Troubleshooting

### Issue: Memory not being retrieved

**Solution**:
1. Verify user_id and conversation_id are correct
2. Check database tables exist: `make db-init-memory`
3. Verify DATABASE_URL is correct in `.env`
4. Check logs for memory retrieval errors

### Issue: LangGraph checkpointer errors

**Solution**:
1. Ensure PostgreSQL checkpoints table exists
2. Check database connection permissions
3. Verify LangGraph version compatibility

### Issue: Conversation history incomplete

**Solution**:
1. Check `limit` parameter (default: 10)
2. Verify messages were stored successfully
3. Check conversation_id matches

### Issue: User memories not updating

**Solution**:
1. Verify `extract_user_memories` node is executing
2. Check LLM is successfully analyzing conversations
3. Review confidence/importance thresholds

## Future Enhancements

- [ ] Conversation summarization for long threads
- [ ] Semantic search over conversation history
- [ ] Memory importance decay over time
- [ ] Multi-modal memory (images, documents)
- [ ] Cross-user shared knowledge base
- [ ] Memory-based user segmentation
- [ ] Automated memory cleanup policies
- [ ] Real-time memory updates via WebSockets

## References

- [LangGraph Checkpointers](https://langchain-ai.github.io/langgraph/reference/checkpoints/)
- [PostgreSQL Best Practices](https://wiki.postgresql.org/wiki/Don%27t_Do_This)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/en/20/orm/)
- [Pydantic Models](https://docs.pydantic.dev/latest/)

## Support

For issues or questions:
- Check [GitHub Issues](https://github.com/yourusername/ConformAI/issues)
- Review logs in `/logs/` directory
- Enable DEBUG logging: `LOG_LEVEL=DEBUG` in `.env`
