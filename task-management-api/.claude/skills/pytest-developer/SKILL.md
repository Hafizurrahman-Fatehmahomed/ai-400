---
name: pytest-developer
description: Write professional pytest test suites from unit tests to production-ready systems. Use when creating tests, writing fixtures, parametrizing test cases, mocking dependencies, testing FastAPI/async code, database testing, or setting up CI/CD test pipelines. Covers fixtures, markers, parametrization, monkeypatching, coverage, and test architecture.
---

# Pytest Developer

Write production-ready test suites following pytest best practices for clean architecture, readability, scalability, and maintainability.

## Quick Start

### Basic Test Structure

```python
# tests/test_example.py
import pytest

def test_simple_assertion():
    """Test names should describe the expected behavior."""
    result = 1 + 1
    assert result == 2

class TestUserService:
    """Group related tests in classes."""

    def test_create_user_returns_user_object(self):
        user = create_user("alice")
        assert user.name == "alice"

    def test_create_user_with_empty_name_raises_error(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            create_user("")
```

### Run Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific file/directory
pytest tests/test_api.py
pytest tests/unit/

# Run tests matching pattern
pytest -k "test_user"

# Run marked tests
pytest -m "not slow"

# Run with coverage
pytest --cov=app --cov-report=html
```

## Instructions

### 1. Project Structure

Follow this standard test directory layout:

```
project/
├── app/                    # Application code
│   ├── __init__.py
│   ├── models.py
│   ├── services.py
│   └── api.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Shared fixtures (root level)
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── conftest.py     # Unit test fixtures
│   │   ├── test_models.py
│   │   └── test_services.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── conftest.py     # Integration fixtures
│   │   └── test_api.py
│   └── e2e/
│       ├── __init__.py
│       └── test_workflows.py
├── pyproject.toml
└── pytest.ini (optional)
```

### 2. Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "-ra",                    # Show summary of all except passed
    "-q",                     # Quiet mode
    "--strict-markers",       # Error on unknown markers
    "--strict-config",        # Error on config issues
]
testpaths = ["tests"]
pythonpath = ["."]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: integration tests requiring external services",
    "e2e: end-to-end tests",
]
filterwarnings = [
    "error",                  # Treat warnings as errors
    "ignore::DeprecationWarning",
]
```

### 3. Writing Fixtures

Place fixtures in `conftest.py` at the appropriate scope level:

```python
# tests/conftest.py - Available to ALL tests
import pytest

@pytest.fixture
def sample_user_data():
    """Simple data fixture."""
    return {"name": "Alice", "email": "alice@example.com"}

@pytest.fixture
def db_session(tmp_path):
    """Fixture with setup and teardown using yield."""
    db_path = tmp_path / "test.db"
    session = create_session(db_path)
    yield session
    session.close()

@pytest.fixture(scope="module")
def expensive_resource():
    """Module-scoped fixture - created once per module."""
    resource = create_expensive_resource()
    yield resource
    resource.cleanup()

@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Autouse fixture - runs automatically for all tests."""
    monkeypatch.setenv("TESTING", "true")
```

### 4. Test Naming Conventions

Follow this naming pattern: `test_<unit>_<scenario>_<expected_result>`

```python
# Good: Descriptive test names
def test_calculate_total_with_discount_returns_reduced_price(): ...
def test_user_login_with_invalid_password_raises_auth_error(): ...
def test_api_endpoint_returns_404_when_resource_not_found(): ...

# Bad: Vague test names
def test_calculate(): ...
def test_user(): ...
def test_api(): ...
```

### 5. Assertion Best Practices

```python
# Use specific assertions
assert user.name == "Alice"                    # Good
assert user.name                               # Bad - not specific

# Use pytest.raises for exceptions
with pytest.raises(ValueError, match=r"invalid.*format"):
    parse_date("not-a-date")

# Use pytest.approx for floats
assert 0.1 + 0.2 == pytest.approx(0.3)

# Assert on collections
assert "admin" in user.roles
assert len(results) == 3
assert results == [1, 2, 3]                    # Order matters
assert set(results) == {1, 2, 3}               # Order doesn't matter
```

### 6. When to Use Each Fixture Scope

| Scope | Use Case | Example |
|-------|----------|---------|
| `function` (default) | Test isolation needed | User objects, test data |
| `class` | Shared across class methods | Class-level setup |
| `module` | Expensive, read-only resources | Database schema |
| `package` | Package-level resources | Shared connections |
| `session` | Global resources | Docker containers, server |

### 7. Mocking Strategy

Use `monkeypatch` for simple mocking, `unittest.mock` for complex scenarios:

```python
# Environment variables
def test_config_uses_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    config = load_config()
    assert config.api_key == "test-key"

# Function replacement
def test_api_call(monkeypatch):
    monkeypatch.setattr("app.client.fetch", lambda url: {"data": "mocked"})
    result = fetch_user_data()
    assert result["data"] == "mocked"

# Complex mocking with unittest.mock
from unittest.mock import Mock, patch, AsyncMock

def test_service_calls_repository():
    mock_repo = Mock()
    mock_repo.get_user.return_value = User(id=1, name="Alice")

    service = UserService(repository=mock_repo)
    user = service.get_user(1)

    mock_repo.get_user.assert_called_once_with(1)
    assert user.name == "Alice"
```

## Quick Reference

### Essential Markers

```python
@pytest.mark.skip(reason="Not implemented")
@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires 3.11+")
@pytest.mark.xfail(reason="Known bug #123")
@pytest.mark.parametrize("input,expected", [(1, 2), (2, 4)])
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("db_session")
```

### Command Line Options

```bash
pytest -v                    # Verbose
pytest -x                    # Stop on first failure
pytest --lf                  # Run last failed
pytest --ff                  # Run failed first
pytest -k "pattern"          # Filter by name
pytest -m "marker"           # Filter by marker
pytest --pdb                 # Debug on failure
pytest -n auto               # Parallel (pytest-xdist)
pytest --cov=app             # Coverage (pytest-cov)
```

### Required Dependencies

```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.21",
    "pytest-xdist>=3.0",      # Parallel execution
    "httpx>=0.24",            # Async HTTP client for FastAPI
    "factory-boy>=3.0",       # Test factories
]
```

## See Also

- [REFERENCE.md](REFERENCE.md) - Complete API reference for fixtures, markers, hooks
- [EXAMPLES.md](EXAMPLES.md) - Comprehensive examples for all testing scenarios
- [templates/](templates/) - Starter templates for common test patterns
