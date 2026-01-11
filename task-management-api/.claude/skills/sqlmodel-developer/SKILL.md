---
name: sqlmodel-developer
description: Build production-grade database layers with SQLModel, SQLAlchemy, and Pydantic. Use when designing models, creating CRUD operations, defining relationships, setting up migrations, implementing async patterns, connection pooling, security hardening, or testing database code. Covers single-table to enterprise multi-database systems.
---

# SQLModel Developer

Build production-ready database layers using SQLModel—the library that unifies SQLAlchemy's power with Pydantic's validation.

## Quick Reference

- [REFERENCE.md](REFERENCE.md) - API patterns, connection pooling, async configuration
- [EXAMPLES.md](EXAMPLES.md) - Production scenarios and enterprise patterns
- [templates/](templates/) - Boilerplate code for common patterns

## Core Principles

1. **Single Source of Truth**: One model class serves as database table, Pydantic schema, and Python dataclass
2. **Type Safety**: Use Python type hints throughout; leverage editor autocompletion
3. **Security First**: Always use parameterized queries; validate all inputs; separate API models
4. **Production Ready**: Configure connection pooling, migrations, and proper error handling

## Instructions

### 1. Model Design

When creating SQLModel models, follow these patterns:

**Basic Table Model:**
```python
from sqlmodel import Field, SQLModel

class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True, max_length=255)
    name: str = Field(max_length=100)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

**Key Design Rules:**
- Primary keys: Use `int | None` with `Field(default=None, primary_key=True)` for auto-increment
- Required fields: Type without `| None` and no default
- Optional fields: Type with `| None` and `= None` default
- Always add `index=True` to frequently queried fields
- Use `max_length` constraints for string fields
- Add `unique=True` for business keys (email, username, etc.)

### 2. Separate API Models (Security Critical)

**Never expose database models directly to APIs.** Create separate models:

```python
# Database model (internal)
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str
    hashed_password: str  # NEVER expose
    role: str = "user"

# API input model (no id, no sensitive fields)
class UserCreate(SQLModel):
    email: str
    password: str  # Plaintext for input only

# API response model (excludes secrets)
class UserPublic(SQLModel):
    id: int
    email: str

# API update model (all fields optional)
class UserUpdate(SQLModel):
    email: str | None = None
    password: str | None = None
```

### 3. Relationships

**One-to-Many:**
```python
from sqlmodel import Relationship

class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    members: list["User"] = Relationship(back_populates="team")

class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    team_id: int | None = Field(default=None, foreign_key="team.id")
    team: Team | None = Relationship(back_populates="members")
```

**Many-to-Many:**
```python
class UserProjectLink(SQLModel, table=True):
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    project_id: int = Field(foreign_key="project.id", primary_key=True)

class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    projects: list["Project"] = Relationship(
        back_populates="users",
        link_model=UserProjectLink
    )

class Project(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    users: list["User"] = Relationship(
        back_populates="projects",
        link_model=UserProjectLink
    )
```

### 4. Engine and Session Configuration

**Development:**
```python
from sqlmodel import create_engine, Session

engine = create_engine("sqlite:///dev.db", echo=True)
```

**Production (PostgreSQL with pooling):**
```python
from sqlmodel import create_engine

engine = create_engine(
    "postgresql+psycopg2://user:pass@host:5432/db",
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    echo=False,
)
```

**With PGBouncer (disable app pooling):**
```python
from sqlalchemy.pool import NullPool

engine = create_engine(
    "postgresql+psycopg2://user:pass@pgbouncer:6432/db",
    poolclass=NullPool,
)
```

### 5. FastAPI Integration

**Session Dependency:**
```python
from fastapi import Depends
from sqlmodel import Session

def get_session():
    with Session(engine) as session:
        yield session

@app.post("/users/", response_model=UserPublic)
def create_user(
    user: UserCreate,
    session: Session = Depends(get_session)
):
    hashed = hash_password(user.password)
    db_user = User(email=user.email, hashed_password=hashed)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user
```

### 6. CRUD Operations

**Create:**
```python
def create_item(session: Session, item: ItemCreate) -> Item:
    db_item = Item.model_validate(item)
    session.add(db_item)
    session.commit()
    session.refresh(db_item)
    return db_item
```

**Read (with filtering):**
```python
from sqlmodel import select

def get_items(
    session: Session,
    skip: int = 0,
    limit: int = 100,
    is_active: bool | None = None
) -> list[Item]:
    statement = select(Item)
    if is_active is not None:
        statement = statement.where(Item.is_active == is_active)
    statement = statement.offset(skip).limit(limit)
    return session.exec(statement).all()
```

**Update:**
```python
def update_item(session: Session, item_id: int, data: ItemUpdate) -> Item | None:
    db_item = session.get(Item, item_id)
    if not db_item:
        return None
    update_data = data.model_dump(exclude_unset=True)
    db_item.sqlmodel_update(update_data)
    session.add(db_item)
    session.commit()
    session.refresh(db_item)
    return db_item
```

**Delete:**
```python
def delete_item(session: Session, item_id: int) -> bool:
    db_item = session.get(Item, item_id)
    if not db_item:
        return False
    session.delete(db_item)
    session.commit()
    return True
```

### 7. Input Validation (Security)

**Field-Level Validation:**
```python
from pydantic import field_validator, EmailStr
from sqlmodel import SQLModel, Field

class UserCreate(SQLModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8)

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v.lower()

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v
```

**Model-Level Validation:**
```python
from pydantic import model_validator
from typing_extensions import Self

class DateRange(SQLModel):
    start_date: date
    end_date: date

    @model_validator(mode='after')
    def validate_dates(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError('end_date must be after start_date')
        return self
```

### 8. Async Support

**Async Engine and Session:**
```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlmodel import SQLModel

async_engine = create_async_engine(
    "postgresql+asyncpg://user:pass@host/db",
    pool_size=20,
    max_overflow=10,
)

async_session_factory = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_async_session():
    async with async_session_factory() as session:
        yield session
```

**Async CRUD:**
```python
from sqlmodel import select

async def get_user(session: AsyncSession, user_id: int) -> User | None:
    result = await session.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()

async def create_user(session: AsyncSession, user: UserCreate) -> User:
    db_user = User.model_validate(user)
    session.add(db_user)
    await session.commit()
    await session.refresh(db_user)
    return db_user
```

### 9. Migrations with Alembic

**Setup:**
```bash
alembic init migrations
```

**Configure env.py:**
```python
# migrations/env.py
from sqlmodel import SQLModel
from app.models import *  # Import all models

target_metadata = SQLModel.metadata

def run_migrations_online():
    # Add these to context.configure():
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # Required for SQLite
        user_module_prefix="sqlmodel.sql.sqltypes.",
    )
```

**Workflow:**
```bash
# Generate migration
alembic revision --autogenerate -m "Add users table"

# Review the generated migration file!

# Apply migration
alembic upgrade head

# Rollback (if needed)
alembic downgrade -1
```

### 10. Testing

**Test Fixtures:**
```python
import pytest
from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

@pytest.fixture(name="session")
def session_fixture():
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
    def override_get_session():
        return session

    app.dependency_overrides[get_session] = override_get_session
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
```

**Test Examples:**
```python
def test_create_user(client: TestClient):
    response = client.post(
        "/users/",
        json={"email": "test@example.com", "password": "SecurePass123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "hashed_password" not in data  # Security check

def test_create_user_invalid_email(client: TestClient):
    response = client.post(
        "/users/",
        json={"email": "invalid", "password": "SecurePass123"}
    )
    assert response.status_code == 422
```

## Security Checklist

- [ ] Use parameterized queries (never string interpolation)
- [ ] Separate input/output models from database models
- [ ] Never expose sensitive fields (passwords, tokens, internal IDs)
- [ ] Validate all inputs with Pydantic validators
- [ ] Use least-privilege database users
- [ ] Enable `pool_pre_ping` for connection health checks
- [ ] Set appropriate `pool_recycle` for connection lifetime
- [ ] Review Alembic migrations before applying
- [ ] Test for SQL injection in custom queries
- [ ] Audit all raw SQL usage with `text()`

## Common Patterns

| Pattern | Use Case |
|---------|----------|
| `session.get(Model, id)` | Fetch by primary key |
| `session.exec(select(...)).all()` | Fetch multiple records |
| `session.exec(select(...)).first()` | Fetch first or None |
| `session.exec(select(...)).one()` | Fetch exactly one (raises if not found) |
| `Model.model_validate(data)` | Create model from dict/Pydantic |
| `model.sqlmodel_update(data)` | Update model from dict |
| `data.model_dump(exclude_unset=True)` | Get only provided fields |

## When to Use This Skill

- Designing new database models
- Implementing CRUD operations
- Setting up FastAPI + SQLModel integration
- Configuring connection pooling for production
- Creating database migrations
- Writing async database code
- Implementing input validation
- Security hardening database layers
- Testing database operations
- Scaling to enterprise multi-database systems

## Next Steps

- See [REFERENCE.md](REFERENCE.md) for detailed API patterns
- See [EXAMPLES.md](EXAMPLES.md) for production scenarios
- Use [templates/](templates/) for quick starts
