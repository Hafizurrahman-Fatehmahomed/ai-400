import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine
from sqlmodel.pool import StaticPool

from app.main import app
from app.database import get_session
from app.models import Task


@pytest.fixture(name="session")
def session_fixture():
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(session: Session):
    """Create a test client with the session dependency overridden."""

    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_task(session: Session) -> Task:
    """Create a sample task in the database."""
    task = Task(title="Test Task", description="Test Description")
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


@pytest.fixture
def multiple_tasks(session: Session) -> list[Task]:
    """Create multiple tasks in the database."""
    tasks = [
        Task(title="Task 1", description="Description 1"),
        Task(title="Task 2", description="Description 2"),
        Task(title="Task 3", description=None),
    ]
    for task in tasks:
        session.add(task)
    session.commit()
    for task in tasks:
        session.refresh(task)
    return tasks
