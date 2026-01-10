# Pytest Examples

Comprehensive examples for all testing scenarios from unit tests to production-ready test suites.

## Table of Contents

1. [Unit Testing](#unit-testing)
2. [Fixtures](#fixtures)
3. [Mocking](#mocking)
4. [FastAPI Testing](#fastapi-testing)
5. [Async Testing](#async-testing)
6. [Database Testing](#database-testing)
7. [Parametrization](#parametrization)
8. [Test Organization](#test-organization)
9. [CI/CD Integration](#cicd-integration)

---

## Unit Testing

### Basic Unit Tests

```python
# tests/unit/test_calculator.py
import pytest
from app.calculator import Calculator


class TestCalculator:
    """Test suite for Calculator class."""

    def test_add_positive_numbers(self):
        calc = Calculator()
        assert calc.add(2, 3) == 5

    def test_add_negative_numbers(self):
        calc = Calculator()
        assert calc.add(-2, -3) == -5

    def test_add_mixed_numbers(self):
        calc = Calculator()
        assert calc.add(-2, 5) == 3

    def test_divide_returns_float(self):
        calc = Calculator()
        assert calc.divide(10, 4) == 2.5

    def test_divide_by_zero_raises_error(self):
        calc = Calculator()
        with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
            calc.divide(10, 0)


# Function-based tests for simpler cases
def test_calculator_multiply():
    calc = Calculator()
    assert calc.multiply(3, 4) == 12


def test_calculator_subtract():
    calc = Calculator()
    assert calc.subtract(10, 4) == 6
```

### Testing Pure Functions

```python
# tests/unit/test_utils.py
import pytest
from app.utils import slugify, parse_email, validate_password


class TestSlugify:
    def test_converts_spaces_to_hyphens(self):
        assert slugify("hello world") == "hello-world"

    def test_converts_to_lowercase(self):
        assert slugify("Hello World") == "hello-world"

    def test_removes_special_characters(self):
        assert slugify("hello@world!") == "helloworld"

    def test_handles_empty_string(self):
        assert slugify("") == ""

    def test_handles_unicode(self):
        assert slugify("café") == "cafe"


class TestParseEmail:
    def test_extracts_username_and_domain(self):
        result = parse_email("user@example.com")
        assert result == {"username": "user", "domain": "example.com"}

    def test_handles_subdomain(self):
        result = parse_email("user@mail.example.com")
        assert result["domain"] == "mail.example.com"

    def test_invalid_email_raises_error(self):
        with pytest.raises(ValueError, match="Invalid email format"):
            parse_email("not-an-email")


class TestValidatePassword:
    @pytest.mark.parametrize("password,expected", [
        ("short", False),              # Too short
        ("alllowercase123", False),    # No uppercase
        ("ALLUPPERCASE123", False),    # No lowercase
        ("NoNumbers", False),          # No digits
        ("ValidPass123", True),        # Valid
        ("Another$Valid1", True),      # Valid with special char
    ])
    def test_password_validation(self, password, expected):
        assert validate_password(password) == expected
```

### Testing Classes with State

```python
# tests/unit/test_shopping_cart.py
import pytest
from app.cart import ShoppingCart, Product


class TestShoppingCart:
    @pytest.fixture
    def cart(self):
        """Fresh cart for each test."""
        return ShoppingCart()

    @pytest.fixture
    def sample_product(self):
        return Product(id=1, name="Widget", price=9.99)

    def test_new_cart_is_empty(self, cart):
        assert cart.is_empty()
        assert len(cart) == 0

    def test_add_item_increases_count(self, cart, sample_product):
        cart.add(sample_product)
        assert len(cart) == 1
        assert not cart.is_empty()

    def test_add_same_item_increases_quantity(self, cart, sample_product):
        cart.add(sample_product)
        cart.add(sample_product)
        assert len(cart) == 1
        assert cart.get_quantity(sample_product) == 2

    def test_remove_item_decreases_count(self, cart, sample_product):
        cart.add(sample_product)
        cart.remove(sample_product)
        assert cart.is_empty()

    def test_total_calculates_correctly(self, cart, sample_product):
        cart.add(sample_product, quantity=3)
        assert cart.total == pytest.approx(29.97)

    def test_clear_empties_cart(self, cart, sample_product):
        cart.add(sample_product, quantity=5)
        cart.clear()
        assert cart.is_empty()
```

---

## Fixtures

### Basic Fixtures

```python
# tests/conftest.py
import pytest
from app.models import User, Organization
from app.database import Session


@pytest.fixture
def user_data():
    """Simple data fixture."""
    return {
        "name": "Alice",
        "email": "alice@example.com",
        "role": "admin",
    }


@pytest.fixture
def user(user_data):
    """Fixture depending on another fixture."""
    return User(**user_data)


@pytest.fixture
def organization():
    return Organization(name="Acme Corp", plan="enterprise")
```

### Fixtures with Setup and Teardown

```python
# tests/conftest.py
import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_directory():
    """Create temp directory, clean up after test."""
    dir_path = Path(tempfile.mkdtemp())
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def config_file(temp_directory):
    """Create a temporary config file."""
    config_path = temp_directory / "config.yaml"
    config_path.write_text("""
database:
  host: localhost
  port: 5432
    """)
    yield config_path
    # Cleanup handled by temp_directory fixture
```

### Scoped Fixtures

```python
# tests/conftest.py
import pytest


@pytest.fixture(scope="session")
def database_engine():
    """Create database engine once per test session."""
    from sqlalchemy import create_engine
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def database_tables(database_engine):
    """Create tables once per test module."""
    from app.models import Base
    Base.metadata.create_all(database_engine)
    yield
    Base.metadata.drop_all(database_engine)


@pytest.fixture
def db_session(database_engine, database_tables):
    """Create fresh session for each test."""
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=database_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()
```

### Fixture Factories

```python
# tests/conftest.py
import pytest
from app.models import User, Post


@pytest.fixture
def make_user(db_session):
    """Factory fixture for creating users."""
    created_users = []

    def _make_user(
        name="Test User",
        email=None,
        role="user",
        **kwargs
    ):
        if email is None:
            email = f"{name.lower().replace(' ', '.')}@test.com"

        user = User(name=name, email=email, role=role, **kwargs)
        db_session.add(user)
        db_session.commit()
        created_users.append(user)
        return user

    yield _make_user

    # Cleanup
    for user in created_users:
        db_session.delete(user)
    db_session.commit()


@pytest.fixture
def make_post(db_session, make_user):
    """Factory that depends on another factory."""
    created_posts = []

    def _make_post(title="Test Post", content="Content", author=None):
        if author is None:
            author = make_user()

        post = Post(title=title, content=content, author_id=author.id)
        db_session.add(post)
        db_session.commit()
        created_posts.append(post)
        return post

    yield _make_post

    for post in created_posts:
        db_session.delete(post)
    db_session.commit()


# Usage in tests
def test_user_can_create_multiple_posts(make_user, make_post):
    author = make_user(name="Alice")
    post1 = make_post(title="First Post", author=author)
    post2 = make_post(title="Second Post", author=author)

    assert len(author.posts) == 2
```

### Autouse Fixtures

```python
# tests/conftest.py
import pytest
import os


@pytest.fixture(autouse=True)
def set_test_environment(monkeypatch):
    """Automatically set test environment for all tests."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DEBUG", "false")


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    from app.config import Config
    Config._instance = None
    yield
    Config._instance = None


# Scoped autouse fixture
@pytest.fixture(scope="module", autouse=True)
def module_setup():
    """Runs once at the start of each test module."""
    print("\nSetting up module")
    yield
    print("\nTearing down module")
```

---

## Mocking

### Monkeypatch Examples

```python
# tests/unit/test_service.py
import pytest
from app.services import UserService, EmailService


class TestUserService:
    def test_get_user_from_api(self, monkeypatch):
        """Mock external API call."""
        def mock_fetch(url):
            return {"id": 1, "name": "Alice", "email": "alice@test.com"}

        monkeypatch.setattr("app.services.fetch_json", mock_fetch)

        service = UserService()
        user = service.get_user(1)

        assert user.name == "Alice"

    def test_config_from_environment(self, monkeypatch):
        """Mock environment variables."""
        monkeypatch.setenv("API_URL", "https://test.api.com")
        monkeypatch.setenv("API_KEY", "test-key-123")

        service = UserService()

        assert service.api_url == "https://test.api.com"
        assert service.api_key == "test-key-123"

    def test_missing_env_raises_error(self, monkeypatch):
        """Test behavior when env var is missing."""
        monkeypatch.delenv("API_KEY", raising=False)

        with pytest.raises(ValueError, match="API_KEY not configured"):
            UserService()
```

### unittest.mock Integration

```python
# tests/unit/test_email_service.py
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from app.services import EmailService, NotificationService


class TestEmailService:
    def test_send_email_calls_smtp(self):
        """Mock SMTP client."""
        mock_smtp = Mock()
        service = EmailService(smtp_client=mock_smtp)

        service.send("user@test.com", "Subject", "Body")

        mock_smtp.send_message.assert_called_once()
        call_args = mock_smtp.send_message.call_args
        assert call_args[0][0]["To"] == "user@test.com"

    def test_send_email_retries_on_failure(self):
        """Test retry logic."""
        mock_smtp = Mock()
        mock_smtp.send_message.side_effect = [
            ConnectionError("Failed"),
            ConnectionError("Failed"),
            None,  # Success on third try
        ]

        service = EmailService(smtp_client=mock_smtp, max_retries=3)
        service.send("user@test.com", "Subject", "Body")

        assert mock_smtp.send_message.call_count == 3

    @patch("app.services.smtp_connect")
    def test_with_patch_decorator(self, mock_connect):
        """Use patch decorator."""
        mock_client = Mock()
        mock_connect.return_value = mock_client

        service = EmailService()
        service.send("user@test.com", "Subject", "Body")

        mock_connect.assert_called_once()
        mock_client.send_message.assert_called_once()


class TestNotificationService:
    def test_sends_to_all_channels(self):
        """Mock multiple dependencies."""
        mock_email = Mock()
        mock_sms = Mock()
        mock_push = Mock()

        service = NotificationService(
            email_client=mock_email,
            sms_client=mock_sms,
            push_client=mock_push,
        )

        service.notify_user(user_id=1, message="Hello")

        mock_email.send.assert_called_once()
        mock_sms.send.assert_called_once()
        mock_push.send.assert_called_once()

    def test_continues_on_channel_failure(self):
        """Test graceful degradation."""
        mock_email = Mock()
        mock_email.send.side_effect = Exception("Email failed")
        mock_sms = Mock()

        service = NotificationService(
            email_client=mock_email,
            sms_client=mock_sms,
        )

        # Should not raise, should continue to SMS
        service.notify_user(user_id=1, message="Hello")

        mock_sms.send.assert_called_once()
```

### pytest-mock Plugin

```python
# tests/unit/test_with_mocker.py
import pytest


def test_with_mocker_fixture(mocker):
    """pytest-mock provides mocker fixture."""
    # Create mock
    mock_func = mocker.patch("app.services.external_api_call")
    mock_func.return_value = {"status": "ok"}

    # Spy on existing function
    spy = mocker.spy(some_module, "some_function")

    # Call code under test
    result = do_something()

    # Assertions
    mock_func.assert_called_once_with(expected_args)
    spy.assert_called()


def test_mock_context_manager(mocker):
    """Mock context manager."""
    mock_file = mocker.mock_open(read_data="file content")
    mocker.patch("builtins.open", mock_file)

    result = read_config_file("config.txt")

    mock_file.assert_called_once_with("config.txt", "r")
    assert result == "file content"
```

---

## FastAPI Testing

### Basic FastAPI Tests

```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestUsersAPI:
    def test_create_user(self, client):
        response = client.post(
            "/users/",
            json={"name": "Alice", "email": "alice@test.com"}
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Alice"
        assert "id" in data

    def test_create_user_invalid_email(self, client):
        response = client.post(
            "/users/",
            json={"name": "Alice", "email": "not-an-email"}
        )
        assert response.status_code == 422

    def test_get_user(self, client):
        # Create user first
        create_response = client.post(
            "/users/",
            json={"name": "Bob", "email": "bob@test.com"}
        )
        user_id = create_response.json()["id"]

        # Get user
        response = client.get(f"/users/{user_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "Bob"

    def test_get_nonexistent_user(self, client):
        response = client.get("/users/99999")
        assert response.status_code == 404

    def test_list_users(self, client):
        response = client.get("/users/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
```

### FastAPI with Database Override

```python
# tests/integration/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import get_db, Base


# Create in-memory SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Create tables and provide session."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create test client with database override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


# tests/integration/test_tasks_api.py
class TestTasksAPI:
    def test_create_task(self, client):
        response = client.post(
            "/tasks/",
            json={"title": "Test Task", "description": "A test task"}
        )
        assert response.status_code == 201
        assert response.json()["title"] == "Test Task"

    def test_list_tasks_empty(self, client):
        response = client.get("/tasks/")
        assert response.status_code == 200
        assert response.json() == []

    def test_complete_task_workflow(self, client):
        # Create
        create_resp = client.post(
            "/tasks/",
            json={"title": "Workflow Task"}
        )
        task_id = create_resp.json()["id"]

        # Update
        update_resp = client.put(
            f"/tasks/{task_id}",
            json={"title": "Updated Task", "completed": True}
        )
        assert update_resp.json()["completed"] is True

        # Delete
        delete_resp = client.delete(f"/tasks/{task_id}")
        assert delete_resp.status_code == 204

        # Verify deleted
        get_resp = client.get(f"/tasks/{task_id}")
        assert get_resp.status_code == 404
```

### Testing Authentication

```python
# tests/integration/test_auth.py
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def auth_headers(client):
    """Get authentication headers."""
    response = client.post(
        "/auth/login",
        data={"username": "testuser", "password": "testpass"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def authenticated_client(client, auth_headers):
    """Client with auth headers set."""
    client.headers.update(auth_headers)
    return client


class TestProtectedEndpoints:
    def test_protected_route_without_auth(self, client):
        response = client.get("/users/me")
        assert response.status_code == 401

    def test_protected_route_with_auth(self, authenticated_client):
        response = authenticated_client.get("/users/me")
        assert response.status_code == 200
        assert "email" in response.json()

    def test_admin_route_requires_admin_role(self, client, auth_headers):
        response = client.get("/admin/users", headers=auth_headers)
        assert response.status_code == 403  # Regular user, not admin
```

---

## Async Testing

### Basic Async Tests

```python
# tests/unit/test_async.py
import pytest
import asyncio
from app.async_services import fetch_user, fetch_all_users


@pytest.mark.asyncio
async def test_async_function():
    """Basic async test."""
    result = await fetch_user(1)
    assert result["id"] == 1


@pytest.mark.asyncio
async def test_async_with_timeout():
    """Test with timeout."""
    async with asyncio.timeout(5.0):
        result = await slow_operation()
        assert result is not None


@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test multiple concurrent async operations."""
    results = await asyncio.gather(
        fetch_user(1),
        fetch_user(2),
        fetch_user(3),
    )
    assert len(results) == 3
    assert all(r is not None for r in results)
```

### Async Fixtures

```python
# tests/conftest.py
import pytest
import asyncio
from app.database import AsyncSession, async_engine


@pytest.fixture
async def async_session():
    """Async database session fixture."""
    async with AsyncSession(async_engine) as session:
        yield session
        await session.rollback()


@pytest.fixture
async def async_client():
    """Async HTTP client fixture."""
    import httpx
    async with httpx.AsyncClient() as client:
        yield client


# Usage
@pytest.mark.asyncio
async def test_with_async_fixtures(async_session, async_client):
    # Both fixtures are async
    user = await async_session.get(User, 1)
    response = await async_client.get("https://api.example.com/users/1")
```

### Async FastAPI Testing with httpx

```python
# tests/integration/test_async_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
async def async_client():
    """Async test client for FastAPI."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_async_endpoint(async_client):
    response = await async_client.get("/async-endpoint")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_websocket():
    """Test WebSocket endpoint."""
    from httpx_ws import aconnect_ws

    async with aconnect_ws("ws://test/ws", app) as ws:
        await ws.send_text("hello")
        message = await ws.receive_text()
        assert message == "echo: hello"
```

### Testing Async Context Managers

```python
# tests/unit/test_async_context.py
import pytest
from app.resources import AsyncDatabasePool


@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async context manager."""
    async with AsyncDatabasePool() as pool:
        conn = await pool.acquire()
        assert conn is not None
        await pool.release(conn)


@pytest.mark.asyncio
async def test_async_generator():
    """Test async generator."""
    from app.streams import async_data_stream

    results = []
    async for item in async_data_stream():
        results.append(item)
        if len(results) >= 5:
            break

    assert len(results) == 5
```

---

## Database Testing

### SQLAlchemy/SQLModel Testing

```python
# tests/integration/conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from app.models import User, Post, Comment


@pytest.fixture(scope="session")
def engine():
    """Create test database engine."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    """Create new session for each test with rollback."""
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


# tests/integration/test_models.py
class TestUserModel:
    def test_create_user(self, session):
        user = User(name="Alice", email="alice@test.com")
        session.add(user)
        session.commit()

        assert user.id is not None
        assert user.created_at is not None

    def test_user_posts_relationship(self, session):
        user = User(name="Alice", email="alice@test.com")
        post = Post(title="Test Post", content="Content", author=user)

        session.add(user)
        session.add(post)
        session.commit()

        assert len(user.posts) == 1
        assert user.posts[0].title == "Test Post"

    def test_cascade_delete(self, session):
        user = User(name="Alice", email="alice@test.com")
        post = Post(title="Test Post", content="Content", author=user)

        session.add_all([user, post])
        session.commit()

        post_id = post.id
        session.delete(user)
        session.commit()

        # Post should be deleted with user
        assert session.get(Post, post_id) is None
```

### Repository Pattern Testing

```python
# tests/unit/test_repository.py
import pytest
from app.repositories import UserRepository
from app.models import User


class TestUserRepository:
    @pytest.fixture
    def repo(self, session):
        return UserRepository(session)

    def test_create(self, repo):
        user = repo.create(name="Alice", email="alice@test.com")
        assert user.id is not None

    def test_get_by_id(self, repo):
        created = repo.create(name="Bob", email="bob@test.com")
        found = repo.get_by_id(created.id)
        assert found.name == "Bob"

    def test_get_by_id_not_found(self, repo):
        result = repo.get_by_id(99999)
        assert result is None

    def test_get_by_email(self, repo):
        repo.create(name="Charlie", email="charlie@test.com")
        found = repo.get_by_email("charlie@test.com")
        assert found.name == "Charlie"

    def test_list_all(self, repo):
        repo.create(name="User1", email="user1@test.com")
        repo.create(name="User2", email="user2@test.com")

        users = repo.list_all()
        assert len(users) == 2

    def test_update(self, repo):
        user = repo.create(name="Old Name", email="test@test.com")
        updated = repo.update(user.id, name="New Name")
        assert updated.name == "New Name"

    def test_delete(self, repo):
        user = repo.create(name="ToDelete", email="delete@test.com")
        user_id = user.id

        repo.delete(user_id)

        assert repo.get_by_id(user_id) is None
```

### Testing with Factory Boy

```python
# tests/factories.py
import factory
from factory.alchemy import SQLAlchemyModelFactory
from app.models import User, Post, Comment


class BaseFactory(SQLAlchemyModelFactory):
    class Meta:
        abstract = True
        sqlalchemy_session = None  # Set in conftest
        sqlalchemy_session_persistence = "commit"


class UserFactory(BaseFactory):
    class Meta:
        model = User

    name = factory.Faker("name")
    email = factory.LazyAttribute(lambda o: f"{o.name.lower().replace(' ', '.')}@test.com")
    role = "user"

    class Params:
        admin = factory.Trait(role="admin")


class PostFactory(BaseFactory):
    class Meta:
        model = Post

    title = factory.Faker("sentence", nb_words=5)
    content = factory.Faker("paragraph")
    author = factory.SubFactory(UserFactory)


# tests/conftest.py
@pytest.fixture(autouse=True)
def set_factory_session(session):
    """Set session for all factories."""
    from tests.factories import BaseFactory
    BaseFactory._meta.sqlalchemy_session = session


# tests/integration/test_with_factories.py
from tests.factories import UserFactory, PostFactory


class TestWithFactories:
    def test_create_user(self, session):
        user = UserFactory()
        assert user.id is not None

    def test_create_admin(self, session):
        admin = UserFactory(admin=True)
        assert admin.role == "admin"

    def test_user_with_posts(self, session):
        user = UserFactory()
        posts = PostFactory.create_batch(5, author=user)

        assert len(user.posts) == 5

    def test_bulk_create(self, session):
        users = UserFactory.create_batch(10)
        assert len(users) == 10
```

---

## Parametrization

### Basic Parametrization

```python
# tests/unit/test_parametrized.py
import pytest


@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("Python", "PYTHON"),
    ("", ""),
])
def test_upper(input, expected):
    assert input.upper() == expected


@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
])
def test_addition(a, b, expected):
    assert a + b == expected
```

### Parametrization with IDs and Marks

```python
import pytest
import sys


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
])
def test_upper_with_ids(input, expected):
    assert input.upper() == expected


@pytest.mark.parametrize("value", [
    pytest.param(1, id="positive"),
    pytest.param(-1, id="negative"),
    pytest.param(0, id="zero"),
    pytest.param(
        None,
        id="none",
        marks=pytest.mark.xfail(raises=TypeError),
    ),
])
def test_absolute_value(value):
    assert abs(value) >= 0
```

### Combining Multiple Parametrize

```python
import pytest


# Cartesian product: 3 x 3 = 9 test cases
@pytest.mark.parametrize("x", [1, 2, 3])
@pytest.mark.parametrize("y", [10, 20, 30])
def test_multiply(x, y):
    result = x * y
    assert result == x * y


# Parametrize class
@pytest.mark.parametrize("browser", ["chrome", "firefox", "safari"])
class TestBrowserCompatibility:
    def test_page_load(self, browser):
        # browser is available in all methods
        pass

    def test_form_submit(self, browser):
        pass
```

### Indirect Parametrization

```python
import pytest


@pytest.fixture
def user(request):
    """Create user with specified role."""
    role = request.param
    return {"name": "Test User", "role": role}


@pytest.mark.parametrize("user", ["admin", "editor", "viewer"], indirect=True)
def test_user_permissions(user):
    assert user["role"] in ["admin", "editor", "viewer"]


# Multiple indirect fixtures
@pytest.fixture
def database(request):
    db_type = request.param
    return create_database(db_type)


@pytest.fixture
def cache(request):
    cache_type = request.param
    return create_cache(cache_type)


@pytest.mark.parametrize(
    "database,cache",
    [
        ("postgres", "redis"),
        ("mysql", "memcached"),
        ("sqlite", "memory"),
    ],
    indirect=True,
)
def test_with_different_backends(database, cache):
    pass
```

---

## Test Organization

### conftest.py Hierarchy

```
tests/
├── conftest.py              # Root: shared fixtures for ALL tests
│   └── Contains: db_session, client, make_user, etc.
│
├── unit/
│   ├── conftest.py          # Unit-specific: mocks, simple fixtures
│   │   └── Contains: mock_api, mock_email_service
│   ├── test_models.py
│   └── test_services.py
│
├── integration/
│   ├── conftest.py          # Integration: real connections
│   │   └── Contains: real_db_session, api_client
│   ├── test_api.py
│   └── test_database.py
│
└── e2e/
    ├── conftest.py          # E2E: full stack setup
    │   └── Contains: browser, selenium_driver
    └── test_workflows.py
```

### Root conftest.py Example

```python
# tests/conftest.py
"""
Shared fixtures available to all tests.
"""
import pytest
import os


# ============ Configuration ============
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "e2e: end-to-end tests")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --runslow is passed."""
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="need --runslow option")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests",
    )


# ============ Environment ============
@pytest.fixture(autouse=True)
def test_environment(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


# ============ Database ============
@pytest.fixture(scope="session")
def database_url():
    """Test database URL."""
    return os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")


# ============ Factories ============
@pytest.fixture
def make_user():
    """User factory fixture."""
    def _make_user(name="Test", email=None, **kwargs):
        from app.models import User
        email = email or f"{name.lower()}@test.com"
        return User(name=name, email=email, **kwargs)
    return _make_user
```

### Marker-Based Test Selection

```python
# tests/unit/test_services.py
import pytest


class TestUserService:
    def test_create_user(self):
        """Fast unit test - runs by default."""
        pass

    @pytest.mark.slow
    def test_bulk_import(self):
        """Slow test - skipped unless --runslow."""
        pass


# tests/integration/test_api.py
@pytest.mark.integration
class TestAPIEndpoints:
    def test_health_check(self):
        pass

    @pytest.mark.slow
    def test_large_payload(self):
        """Both integration AND slow."""
        pass


# Run commands:
# pytest                           # All fast tests
# pytest --runslow                 # Include slow tests
# pytest -m integration            # Only integration
# pytest -m "not integration"      # Skip integration
# pytest -m "slow and integration" # Slow integration tests
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run linting
        run: uv run ruff check .

      - name: Run type checking
        run: uv run mypy app

      - name: Run tests with coverage
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/testdb
        run: |
          uv run pytest \
            --cov=app \
            --cov-report=xml \
            --cov-report=html \
            --junitxml=junit.xml \
            -v

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
          fail_ci_if_error: true

      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results-${{ matrix.python-version }}
          path: |
            junit.xml
            htmlcov/
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - coverage

variables:
  POSTGRES_DB: testdb
  POSTGRES_USER: test
  POSTGRES_PASSWORD: test
  DATABASE_URL: postgresql://test:test@postgres:5432/testdb

test:
  stage: test
  image: python:3.11
  services:
    - postgres:15
  before_script:
    - pip install uv
    - uv sync --all-extras
  script:
    - uv run pytest --cov=app --cov-report=xml --junitxml=report.xml
  artifacts:
    reports:
      junit: report.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
  coverage: '/TOTAL.*\s+(\d+%)/'

coverage:
  stage: coverage
  image: python:3.11
  needs: [test]
  script:
    - pip install coverage
    - coverage html
  artifacts:
    paths:
      - htmlcov/
```

### pyproject.toml for CI

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "integration: integration tests",
    "e2e: end-to-end tests",
]
filterwarnings = [
    "error",
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["app"]
branch = true
omit = [
    "*/tests/*",
    "*/__init__.py",
    "*/migrations/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
]
fail_under = 80
show_missing = true

[tool.coverage.html]
directory = "htmlcov"
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: uv run pytest --tb=short -q
        language: system
        pass_filenames: false
        always_run: true
        stages: [push]  # Only on push, not commit

      - id: pytest-fast
        name: pytest-fast
        entry: uv run pytest -x -q --tb=line -m "not slow"
        language: system
        pass_filenames: false
        stages: [commit]  # Fast tests on commit
```
