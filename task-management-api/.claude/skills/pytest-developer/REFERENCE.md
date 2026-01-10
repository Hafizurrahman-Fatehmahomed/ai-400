# Pytest Reference

Complete API reference for fixtures, markers, hooks, assertions, and configuration.

## Table of Contents

1. [Fixtures](#fixtures)
2. [Markers](#markers)
3. [Assertions](#assertions)
4. [Parametrization](#parametrization)
5. [Monkeypatching](#monkeypatching)
6. [Hooks](#hooks)
7. [Configuration](#configuration)
8. [Built-in Fixtures](#built-in-fixtures)
9. [Plugins](#plugins)

---

## Fixtures

### @pytest.fixture Decorator

```python
@pytest.fixture(
    scope="function",      # "function", "class", "module", "package", "session"
    params=None,           # List of parameter values for parametrized fixtures
    autouse=False,         # If True, activate for all tests in scope
    ids=None,              # List of string IDs for params
    name=None,             # Override fixture name
)
def fixture_function():
    pass
```

### Fixture Scopes

| Scope | Instantiation | Teardown |
|-------|---------------|----------|
| `function` | Once per test function | After each test |
| `class` | Once per test class | After all class methods |
| `module` | Once per .py file | After all tests in file |
| `package` | Once per directory | After all tests in dir |
| `session` | Once per test session | After all tests complete |

### Fixture Execution Order

Fixtures execute in scope order: session → package → module → class → function

```python
@pytest.fixture(scope="session")
def session_resource():
    print("1. Session setup")
    yield
    print("5. Session teardown")

@pytest.fixture(scope="module")
def module_resource():
    print("2. Module setup")
    yield
    print("4. Module teardown")

@pytest.fixture
def function_resource():
    print("3. Function setup")
    yield
    print("3. Function teardown")
```

### Fixture Finalization (Teardown)

**Using yield (recommended):**
```python
@pytest.fixture
def database():
    db = create_database()
    yield db              # Test runs here
    db.close()            # Cleanup after test
```

**Using request.addfinalizer:**
```python
@pytest.fixture
def database(request):
    db = create_database()
    request.addfinalizer(lambda: db.close())
    return db
```

### Fixture Factories

Return a factory function when tests need multiple instances:

```python
@pytest.fixture
def make_user():
    created_users = []

    def _make_user(name, email=None):
        user = User(name=name, email=email or f"{name}@test.com")
        created_users.append(user)
        return user

    yield _make_user

    # Cleanup all created users
    for user in created_users:
        user.delete()
```

### Parametrized Fixtures

```python
@pytest.fixture(params=["sqlite", "postgres", "mysql"])
def database(request):
    """Test runs 3 times, once for each database type."""
    db_type = request.param
    db = create_database(db_type)
    yield db
    db.close()

# Custom IDs for better test output
@pytest.fixture(
    params=[
        pytest.param("admin", id="admin-user"),
        pytest.param("guest", id="guest-user"),
    ]
)
def user_role(request):
    return request.param
```

### Accessing Fixture Information

```python
@pytest.fixture
def my_fixture(request):
    # Access test information
    test_name = request.node.name
    test_module = request.module
    test_class = request.cls  # None if not in a class

    # Access fixture information
    fixture_name = request.fixturename
    scope = request.scope

    # Access markers
    markers = list(request.node.iter_markers())

    yield
```

---

## Markers

### Built-in Markers

```python
# Skip unconditionally
@pytest.mark.skip(reason="Not implemented yet")
def test_future(): ...

# Skip conditionally
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix only"
)
def test_unix_feature(): ...

# Expected failure
@pytest.mark.xfail(
    reason="Known bug",
    strict=True,           # Fail if test passes
    raises=ValueError,     # Expected exception type
    run=True,              # Whether to run the test
)
def test_known_bug(): ...

# Parametrize
@pytest.mark.parametrize("x,y,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_add(x, y, expected):
    assert x + y == expected

# Use fixtures
@pytest.mark.usefixtures("setup_db", "mock_api")
def test_integration(): ...

# Filter warnings
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_legacy_code(): ...
```

### Custom Markers

Register in configuration:

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: tests requiring external services",
    "e2e: end-to-end tests",
    "smoke: critical path tests",
    "regression: regression tests",
]
```

Usage:
```python
@pytest.mark.slow
@pytest.mark.integration
def test_full_workflow(): ...
```

### Marker Expressions

```bash
pytest -m "slow"                    # Run slow tests
pytest -m "not slow"                # Skip slow tests
pytest -m "integration and not e2e" # Integration but not e2e
pytest -m "smoke or regression"     # Either smoke or regression
```

### Accessing Markers in Fixtures

```python
@pytest.fixture
def setup_based_on_marker(request):
    if request.node.get_closest_marker("integration"):
        # Setup for integration tests
        return create_real_database()
    else:
        # Setup for unit tests
        return create_mock_database()
```

---

## Assertions

### pytest.raises

```python
# Basic exception check
with pytest.raises(ValueError):
    int("not a number")

# Check exception message with regex
with pytest.raises(ValueError, match=r"invalid literal.*base 10"):
    int("abc")

# Access exception info
with pytest.raises(ValueError) as exc_info:
    raise ValueError("custom error")

assert exc_info.type is ValueError
assert "custom" in str(exc_info.value)
assert exc_info.value.args[0] == "custom error"

# Check callable with arguments
pytest.raises(ZeroDivisionError, lambda: 1/0)
```

### pytest.warns

```python
# Check warning is raised
with pytest.warns(DeprecationWarning):
    deprecated_function()

# Check warning message
with pytest.warns(UserWarning, match="will be removed"):
    warn_user()
```

### pytest.approx

```python
# Float comparison with tolerance
assert 0.1 + 0.2 == pytest.approx(0.3)
assert 0.1 + 0.2 == pytest.approx(0.3, rel=1e-9)  # Relative tolerance
assert 0.1 + 0.2 == pytest.approx(0.3, abs=1e-12) # Absolute tolerance

# Works with sequences
assert [0.1 + 0.2, 0.2 + 0.4] == pytest.approx([0.3, 0.6])

# Works with dicts
assert {"a": 0.1 + 0.2} == pytest.approx({"a": 0.3})
```

### Assertion Introspection

Pytest provides detailed assertion messages automatically:

```python
def test_detailed_failure():
    data = {"name": "Alice", "age": 30}
    assert data == {"name": "Bob", "age": 30}
    # Output shows exactly which key differs
```

### Custom Assertion Helpers

```python
# In conftest.py
def assert_valid_user(user):
    """Custom assertion helper."""
    assert user is not None, "User should not be None"
    assert user.id > 0, f"User ID should be positive, got {user.id}"
    assert "@" in user.email, f"Invalid email: {user.email}"

# Register for better tracebacks
pytest.register_assert_rewrite("tests.helpers")
```

---

## Parametrization

### Basic Parametrization

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("World", "WORLD"),
    ("", ""),
])
def test_upper(input, expected):
    assert input.upper() == expected
```

### Multiple Parameter Sets

```python
# Cartesian product: runs 2 x 3 = 6 tests
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [10, 20, 30])
def test_multiply(x, y):
    assert x * y == x * y
```

### pytest.param for IDs and Marks

```python
@pytest.mark.parametrize("input,expected", [
    pytest.param("hello", "HELLO", id="lowercase"),
    pytest.param("WORLD", "WORLD", id="already-upper"),
    pytest.param("MiXeD", "MIXED", id="mixed-case"),
    pytest.param(
        "special!@#",
        "SPECIAL!@#",
        id="special-chars",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        None,
        None,
        id="none-input",
        marks=pytest.mark.xfail(raises=AttributeError),
    ),
])
def test_upper_detailed(input, expected):
    assert input.upper() == expected
```

### Indirect Parametrization (via Fixtures)

```python
@pytest.fixture
def user(request):
    """Create user based on parametrized role."""
    role = request.param
    return User(name="Test", role=role)

@pytest.mark.parametrize("user", ["admin", "editor", "viewer"], indirect=True)
def test_user_permissions(user):
    assert user.role in ["admin", "editor", "viewer"]
```

### Parametrizing Fixtures

```python
@pytest.fixture(params=[
    pytest.param("sqlite:///test.db", id="sqlite"),
    pytest.param("postgresql://localhost/test", id="postgres"),
])
def database_url(request):
    return request.param

def test_database_connection(database_url):
    # Runs twice: once with sqlite, once with postgres
    conn = connect(database_url)
    assert conn.is_connected()
```

---

## Monkeypatching

### monkeypatch Fixture Methods

```python
def test_monkeypatch_examples(monkeypatch):
    # Set attribute
    monkeypatch.setattr(obj, "attribute", value)
    monkeypatch.setattr("module.Class.method", mock_method)

    # Delete attribute
    monkeypatch.delattr(obj, "attribute")
    monkeypatch.delattr("module.function")

    # Set dictionary item
    monkeypatch.setitem(dict_obj, "key", value)
    monkeypatch.delitem(dict_obj, "key")

    # Set environment variable
    monkeypatch.setenv("VAR_NAME", "value")
    monkeypatch.delenv("VAR_NAME", raising=False)

    # Change working directory
    monkeypatch.chdir(path)

    # Modify sys.path
    monkeypatch.syspath_prepend(path)

    # Undo all patches (rarely needed)
    monkeypatch.undo()
```

### Common Patterns

```python
# Mock environment variables
def test_config(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
    monkeypatch.setenv("DEBUG", "true")
    config = load_config()
    assert config.debug is True

# Mock function return value
def test_api_call(monkeypatch):
    def mock_fetch(*args, **kwargs):
        return {"status": "ok", "data": [1, 2, 3]}

    monkeypatch.setattr("app.client.fetch_data", mock_fetch)
    result = get_data()
    assert result["status"] == "ok"

# Mock class method
def test_service(monkeypatch):
    monkeypatch.setattr(
        UserRepository,
        "find_by_id",
        lambda self, id: User(id=id, name="Mocked")
    )
    service = UserService()
    user = service.get_user(42)
    assert user.name == "Mocked"

# Mock datetime
def test_time_sensitive(monkeypatch):
    import datetime

    class MockDatetime:
        @classmethod
        def now(cls):
            return datetime.datetime(2024, 1, 1, 12, 0, 0)

    monkeypatch.setattr("datetime.datetime", MockDatetime)
```

### Context-Specific Monkeypatching

```python
@pytest.fixture
def mock_external_api(monkeypatch):
    """Reusable fixture for mocking external API."""
    responses = {}

    def mock_request(url, **kwargs):
        if url in responses:
            return responses[url]
        raise ValueError(f"Unmocked URL: {url}")

    monkeypatch.setattr("requests.get", mock_request)
    return responses  # Allow tests to configure responses

def test_user_fetch(mock_external_api):
    mock_external_api["https://api.example.com/users/1"] = {
        "id": 1,
        "name": "Alice"
    }
    user = fetch_user(1)
    assert user.name == "Alice"
```

---

## Hooks

### Configuration Hooks

```python
# conftest.py

def pytest_configure(config):
    """Called after command line options parsed."""
    config.addinivalue_line("markers", "e2e: end-to-end tests")

def pytest_unconfigure(config):
    """Called before process exit."""
    pass

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--env",
        action="store",
        default="test",
        help="Environment to run tests against"
    )
```

### Collection Hooks

```python
def pytest_collection_modifyitems(config, items):
    """Modify collected tests."""
    # Add marker to slow tests automatically
    for item in items:
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)

    # Reorder tests
    items.sort(key=lambda x: x.name)

def pytest_ignore_collect(collection_path, config):
    """Return True to prevent collection."""
    if "legacy" in str(collection_path):
        return True
```

### Test Execution Hooks

```python
def pytest_runtest_setup(item):
    """Called before each test."""
    markers = [m.name for m in item.iter_markers()]
    if "integration" in markers:
        # Check external service availability
        if not is_service_available():
            pytest.skip("External service unavailable")

def pytest_runtest_teardown(item, nextitem):
    """Called after each test."""
    pass

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Access test results."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        # Do something on failure
        pass
```

### Reporting Hooks

```python
def pytest_report_header(config):
    """Add info to test report header."""
    return f"Testing environment: {config.getoption('--env')}"

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary at end of test run."""
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    terminalreporter.write_line(f"Custom summary: {passed} passed, {failed} failed")
```

---

## Configuration

### pyproject.toml (Recommended)

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "-ra",
    "-q",
    "--strict-markers",
    "--strict-config",
    "-p", "no:cacheprovider",  # Disable plugin
]
testpaths = ["tests"]
pythonpath = [".", "src"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow",
    "integration: integration tests",
]
filterwarnings = [
    "error",
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
]
xfail_strict = true
log_cli = true
log_cli_level = "INFO"
log_cli_format = "%(asctime)s [%(levelname)8s] %(message)s"
```

### pytest.ini

```ini
[pytest]
minversion = 7.0
addopts = -ra -q --strict-markers
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: slow tests
    integration: integration tests
filterwarnings =
    error
    ignore::DeprecationWarning
```

### conftest.py Configuration

```python
# tests/conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests",
    )
    parser.addoption(
        "--env",
        action="store",
        default="test",
        choices=["test", "staging", "prod"],
        help="environment to test against",
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

@pytest.fixture
def env(request):
    return request.config.getoption("--env")
```

---

## Built-in Fixtures

### tmp_path / tmp_path_factory

```python
def test_create_file(tmp_path):
    """tmp_path is a pathlib.Path to a temp directory."""
    file = tmp_path / "test.txt"
    file.write_text("content")
    assert file.read_text() == "content"

@pytest.fixture(scope="session")
def session_temp_dir(tmp_path_factory):
    """Create temp dir that persists for entire session."""
    return tmp_path_factory.mktemp("session_data")
```

### capsys / capfd / caplog

```python
def test_capture_stdout(capsys):
    print("hello")
    captured = capsys.readouterr()
    assert captured.out == "hello\n"
    assert captured.err == ""

def test_capture_logs(caplog):
    import logging
    logger = logging.getLogger(__name__)

    with caplog.at_level(logging.INFO):
        logger.info("test message")

    assert "test message" in caplog.text
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "INFO"
```

### monkeypatch

See [Monkeypatching](#monkeypatching) section above.

### request

```python
@pytest.fixture
def resource(request):
    # Access test node
    test_name = request.node.name
    test_class = request.node.cls

    # Access fixture scope
    scope = request.scope

    # Access configuration
    option = request.config.getoption("--env")

    # Access markers
    marker = request.node.get_closest_marker("slow")

    # Add finalizer
    request.addfinalizer(lambda: cleanup())

    return Resource()
```

### cache

```python
def test_with_cache(cache):
    # Store value between test runs
    cache.set("key", "value")
    assert cache.get("key", None) == "value"

@pytest.fixture(scope="session")
def expensive_data(cache):
    data = cache.get("expensive/data", None)
    if data is None:
        data = compute_expensive_data()
        cache.set("expensive/data", data)
    return data
```

---

## Plugins

### Essential Plugins

| Plugin | Purpose | Install |
|--------|---------|---------|
| `pytest-cov` | Code coverage | `pip install pytest-cov` |
| `pytest-xdist` | Parallel execution | `pip install pytest-xdist` |
| `pytest-asyncio` | Async test support | `pip install pytest-asyncio` |
| `pytest-mock` | Enhanced mocking | `pip install pytest-mock` |
| `pytest-timeout` | Test timeouts | `pip install pytest-timeout` |
| `pytest-randomly` | Random test order | `pip install pytest-randomly` |
| `pytest-html` | HTML reports | `pip install pytest-html` |
| `pytest-env` | Environment management | `pip install pytest-env` |

### pytest-cov Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = ["app"]
branch = true
omit = ["*/tests/*", "*/__init__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
fail_under = 80
show_missing = true
```

```bash
# Run with coverage
pytest --cov=app --cov-report=html --cov-report=term-missing
```

### pytest-xdist Configuration

```bash
# Run tests in parallel
pytest -n auto                    # Auto-detect CPU count
pytest -n 4                       # Use 4 workers
pytest -n auto --dist loadscope  # Group by module
pytest -n auto --dist loadfile   # Group by file
```

### pytest-asyncio Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # or "strict"
```

```python
import pytest

# With asyncio_mode = "auto", no decorator needed
async def test_async_function():
    result = await async_operation()
    assert result == expected

# With asyncio_mode = "strict", decorator required
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result == expected

# Async fixtures
@pytest.fixture
async def async_client():
    async with AsyncClient() as client:
        yield client
```
