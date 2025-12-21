## Testing Guide for ConformAI

Comprehensive guide to running and writing tests for the ConformAI project.

---

## 📋 Table of Contents

- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Writing Tests](#writing-tests)
- [Test Fixtures](#test-fixtures)
- [Mocking](#mocking)
- [Coverage Reports](#coverage-reports)
- [CI/CD Integration](#cicd-integration)
- [Best Practices](#best-practices)

---

## 🏗️ Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_models.py
│   ├── test_eurlex_client.py
│   ├── test_embeddings.py
│   └── ...
├── integration/             # Integration tests (require services)
│   ├── test_data_pipeline.py
│   └── ...
├── e2e/                     # End-to-end tests (full system)
│   ├── test_rag_pipeline.py
│   └── test_api_endpoints.py
└── fixtures/                # Test data and fixtures
```

---

## 🚀 Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=services --cov=shared --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Using the Test Runner Script

```bash
# Run unit tests only (fast)
./scripts/run_tests.sh --unit

# Run integration tests
./scripts/run_tests.sh --integration

# Run E2E tests
./scripts/run_tests.sh --e2e

# Run with coverage
./scripts/run_tests.sh --unit --coverage

# Verbose output
./scripts/run_tests.sh --unit --verbose
```

### Running Specific Test Files

```bash
# Run specific test file
pytest tests/unit/test_models.py

# Run specific test class
pytest tests/unit/test_models.py::TestRegulation

# Run specific test function
pytest tests/unit/test_models.py::TestRegulation::test_regulation_creation
```

### Running Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only RAG tests
pytest -m rag

# Run only API tests
pytest -m api

# Run tests NOT requiring API keys
pytest -m "not requires_api_keys"

# Combine markers
pytest -m "unit and rag"
```

---

## 📚 Test Categories

### Unit Tests (`-m unit`)

**Purpose**: Test individual components in isolation

**Characteristics**:
- Fast execution (< 1 second each)
- No external dependencies
- Use mocks for external services
- High test coverage

**Examples**:
```bash
pytest -m unit
pytest tests/unit/test_models.py
pytest tests/unit/test_eurlex_client.py
```

**What they test**:
- Data models and validation
- Business logic
- Utility functions
- Individual class methods

### Integration Tests (`-m integration`)

**Purpose**: Test component interactions

**Characteristics**:
- Moderate speed (1-5 seconds each)
- Require some services (Qdrant, PostgreSQL)
- Test real database interactions
- Mock external APIs

**Examples**:
```bash
pytest -m integration
pytest tests/integration/test_data_pipeline.py
```

**Prerequisites**:
```bash
# Start required services
docker-compose up -d qdrant postgres redis
```

**What they test**:
- Data pipeline workflows
- Database operations
- Service integrations
- Error handling across components

### End-to-End Tests (`-m e2e`)

**Purpose**: Test complete workflows

**Characteristics**:
- Slower execution (5-30 seconds each)
- Require all services
- Test user-facing scenarios
- May use real APIs (when keys provided)

**Examples**:
```bash
pytest -m e2e
pytest tests/e2e/test_rag_pipeline.py
pytest tests/e2e/test_api_endpoints.py
```

**What they test**:
- Complete RAG pipeline
- API endpoints
- Full data pipeline flow
- User scenarios

---

## ✍️ Writing Tests

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestMyComponent:
    """Tests for MyComponent."""

    @pytest.fixture
    def component(self):
        """Create component instance."""
        return MyComponent()

    def test_basic_functionality(self, component):
        """Test basic component functionality."""
        result = component.do_something()
        assert result is not None

    def test_error_handling(self, component):
        """Test component handles errors."""
        with pytest.raises(ValueError):
            component.do_something_invalid()
```

### Using Fixtures

```python
def test_with_sample_data(sample_chunk, sample_regulation):
    """Test using fixtures from conftest.py."""
    assert sample_chunk.text is not None
    assert sample_regulation.celex_id == "32016R0679"
```

### Async Tests

```python
import pytest

class TestAsyncFunction:
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async function."""
        result = await async_function()
        assert result is not None
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("GDPR", RegulationType.REGULATION),
    ("AI Act", RegulationType.REGULATION),
    ("Directive", RegulationType.DIRECTIVE),
])
def test_regulation_types(input, expected):
    """Test multiple inputs with parametrize."""
    result = parse_regulation_type(input)
    assert result == expected
```

---

## 🎭 Test Fixtures

### Available Fixtures (from `conftest.py`)

**Configuration**:
- `test_settings` - Test settings instance
- `mock_settings` - Mocked settings

**Sample Data**:
- `sample_regulation` - Sample EU regulation
- `sample_chunk` - Sample chunk without embedding
- `sample_chunk_with_embedding` - Sample chunk with embedding
- `sample_chunks` - List of sample chunks
- `sample_document` - Sample document

**Mock APIs**:
- `mock_anthropic_client` - Mocked Anthropic API
- `mock_openai_client` - Mocked OpenAI API
- `mock_qdrant_client` - Mocked Qdrant client
- `mock_eurlex_response` - Mock EUR-Lex response
- `mock_eurlex_document` - Mock EUR-Lex XML

**Databases** (integration tests):
- `qdrant_test_client` - Real Qdrant client
- `qdrant_test_collection` - Test collection in Qdrant

**File System**:
- `temp_data_dir` - Temporary data directory
- `sample_xml_file` - Sample XML file

**Generators**:
- `generate_chunks(count, with_embeddings)` - Generate test chunks

### Creating Custom Fixtures

```python
# In conftest.py or test file
@pytest.fixture
def my_custom_fixture():
    """Create custom test data."""
    # Setup
    data = create_test_data()

    yield data

    # Cleanup (optional)
    cleanup_test_data()
```

---

## 🎭 Mocking

### Mocking External APIs

```python
from unittest.mock import Mock, patch

def test_with_mocked_api(mock_openai_client):
    """Test with mocked OpenAI API."""
    # Mock is already set up in fixture
    result = call_function_using_openai()
    assert result is not None
```

### Patching Methods

```python
def test_with_patch():
    """Test with method patch."""
    with patch('services.data-pipeline.src.clients.eurlex_client.EURLexClient.download_document') as mock_download:
        mock_download.return_value = b"<root>Test</root>"

        result = download_and_process()
        assert result is not None
        mock_download.assert_called_once()
```

### Mock Return Values

```python
def test_mock_return_values():
    """Test with specific return values."""
    mock_client = Mock()
    mock_client.get_data.return_value = {"key": "value"}
    mock_client.count_items.return_value = 42

    assert mock_client.get_data() == {"key": "value"}
    assert mock_client.count_items() == 42
```

---

## 📊 Coverage Reports

### Generating Coverage

```bash
# Generate HTML coverage report
pytest --cov=services --cov=shared --cov-report=html

# Generate terminal coverage report
pytest --cov=services --cov=shared --cov-report=term-missing

# Generate XML coverage (for CI/CD)
pytest --cov=services --cov=shared --cov-report=xml
```

### Viewing Coverage

```bash
# Open HTML report
open htmlcov/index.html

# View in terminal
pytest --cov=services --cov=shared --cov-report=term
```

### Coverage Targets

- **Minimum**: 70% overall coverage
- **Target**: 85% overall coverage
- **Critical paths**: 95%+ coverage
  - Data models
  - API endpoints
  - Core RAG logic

### Excluding from Coverage

```python
def debug_function():  # pragma: no cover
    """This function is excluded from coverage."""
    print("Debug information")
```

---

## 🔄 CI/CD Integration

### GitHub Actions Workflow

Tests run automatically on:
- Push to `master`, `main`, or `develop`
- Pull requests
- Manual trigger

### Test Jobs

1. **Unit Tests** - Fast, isolated tests
2. **Integration Tests** - Tests with services
3. **E2E Tests** - Full system tests
4. **Lint** - Code quality checks
5. **Security** - Security scans

### Status Badges

Add to README:
```markdown
![Tests](https://github.com/yourusername/ConformAI/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/yourusername/ConformAI/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/ConformAI)
```

---

## ✅ Best Practices

### General Guidelines

1. **Test One Thing** - Each test should test one specific behavior
2. **Clear Names** - Test names should describe what they test
3. **AAA Pattern** - Arrange, Act, Assert
4. **Independent** - Tests should not depend on each other
5. **Fast** - Unit tests should run in < 1 second

### Naming Conventions

```python
# Good names
def test_regulation_creation_with_valid_data()
def test_chunk_validation_fails_with_empty_text()
def test_embedding_generation_handles_api_error()

# Bad names
def test_1()
def test_regulation()
def test_stuff()
```

### Test Organization

```python
class TestRegulation:
    """Tests for Regulation model."""

    def test_creation(self):
        """Test creating a regulation."""
        pass

    def test_validation(self):
        """Test validation rules."""
        pass

    def test_serialization(self):
        """Test serialization."""
        pass
```

### Arrange-Act-Assert Pattern

```python
def test_chunk_creation():
    """Test chunk creation."""
    # Arrange
    metadata = ChunkMetadata(...)
    text = "Test chunk text"

    # Act
    chunk = Chunk(text=text, metadata=metadata)

    # Assert
    assert chunk.text == text
    assert chunk.metadata == metadata
```

### Error Testing

```python
def test_handles_invalid_input():
    """Test handling of invalid input."""
    with pytest.raises(ValidationError):
        create_object_with_invalid_data()

def test_logs_error_on_failure():
    """Test error logging."""
    with pytest.raises(Exception) as exc_info:
        dangerous_operation()

    assert "expected error message" in str(exc_info.value)
```

### Mocking Best Practices

```python
# Good - Mock at the boundary
@patch('services.my_service.external_api.call')
def test_service_logic(mock_api):
    """Test service logic with mocked API."""
    pass

# Bad - Mock internal methods
@patch('services.my_service.MyClass._internal_method')
def test_something(mock_method):
    """This makes tests brittle."""
    pass
```

---

## 🐛 Debugging Tests

### Running Tests in Debug Mode

```bash
# Run single test with print statements
pytest tests/unit/test_models.py::test_regulation_creation -s

# Run with debugging
pytest --pdb tests/unit/test_models.py

# Increase verbosity
pytest -vvv tests/unit/test_models.py
```

### Common Issues

**Issue**: Tests pass locally but fail in CI
- **Solution**: Check environment variables, ensure services are running

**Issue**: Flaky tests (pass/fail randomly)
- **Solution**: Remove dependencies between tests, fix race conditions

**Issue**: Slow test suite
- **Solution**: Mock external services, use smaller test data

**Issue**: Import errors
- **Solution**: Check PYTHONPATH, ensure `__init__.py` files exist

---

## 📝 Examples

### Complete Test Example

```python
"""
Example test module demonstrating best practices.
"""

import pytest
from unittest.mock import Mock, patch

from shared.models import Chunk, ChunkMetadata, RegulationType


class TestChunkCreation:
    """Tests for Chunk model creation."""

    @pytest.fixture
    def valid_metadata(self):
        """Create valid chunk metadata."""
        return ChunkMetadata(
            regulation_name="GDPR",
            celex_id="32016R0679",
            regulation_type=RegulationType.REGULATION,
            article_number="22",
            chunk_index=0,
            total_chunks=1,
        )

    def test_create_chunk_with_valid_data(self, valid_metadata):
        """Test creating chunk with valid data."""
        # Arrange
        text = "The data subject shall have the right..."

        # Act
        chunk = Chunk(text=text, metadata=valid_metadata)

        # Assert
        assert chunk.text == text
        assert chunk.metadata == valid_metadata
        assert chunk.embedding is None

    def test_create_chunk_with_embedding(self, valid_metadata):
        """Test creating chunk with embedding."""
        # Arrange
        text = "Test text"
        embedding = [0.1] * 1024

        # Act
        chunk = Chunk(
            text=text,
            metadata=valid_metadata,
            embedding=embedding,
        )

        # Assert
        assert chunk.embedding == embedding
        assert len(chunk.embedding) == 1024

    def test_chunk_validation_empty_text(self, valid_metadata):
        """Test validation fails with empty text."""
        # Arrange
        text = ""

        # Act & Assert
        with pytest.raises(ValidationError):
            Chunk(text=text, metadata=valid_metadata)


class TestEmbeddingGeneration:
    """Tests for embedding generation."""

    @pytest.fixture
    def generator(self, mock_openai_client):
        """Create embedding generator."""
        from services.data-pipeline.src.embeddings.embedding_generator import EmbeddingGenerator

        return EmbeddingGenerator(show_progress=False)

    def test_generate_embeddings(self, generator, sample_chunk, mock_openai_client):
        """Test generating embeddings for chunks."""
        # Arrange
        chunks = [sample_chunk]
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1024)]
        mock_openai_client.embeddings.create.return_value = mock_response

        # Act
        result = generator.generate_embeddings(chunks)

        # Assert
        assert len(result) == 1
        assert result[0].embedding is not None
        assert len(result[0].embedding) == 1024
```

---

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Python unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

**Happy Testing!** 🎉

For questions or issues, please create an issue on GitHub.
