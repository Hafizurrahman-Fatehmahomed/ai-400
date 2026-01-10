---
name: fastapi-developer
description: Build FastAPI applications from Hello World to production-ready systems. Use when creating REST APIs, microservices, CRUD endpoints, authentication, database integration, or any Python web API. Covers project structure, SQLModel/SQLAlchemy, OAuth2/JWT security, testing, and deployment.
---

# FastAPI Developer

Build professional FastAPI applications following best practices for clean architecture, security, scalability, and maintainability.

## Quick Reference

- [REFERENCE.md](REFERENCE.md) - Security patterns, middleware, production configs
- [EXAMPLES.md](EXAMPLES.md) - Code examples from basic to advanced
- [templates/](templates/) - Project structure templates

## Instructions

### 1. Project Initialization

When starting a new FastAPI project:

1. **Determine project complexity**:
   - **Minimal**: Single `main.py` for prototypes/learning
   - **Standard**: Modular structure with routers, models, config
   - **Enterprise**: Layered architecture with services, repositories, background tasks

2. **Set up project structure** based on complexity (see templates/)

3. **Initialize with uv** (preferred) or pip:
   ```bash
   uv init project-name
   cd project-name
   uv add fastapi sqlmodel pydantic-settings
   uv add --dev pytest httpx pytest-asyncio
   ```

4. **Create environment configuration**:
   ```bash
   # .env.local (gitignored)
   DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
   SECRET_KEY=<generate with: openssl rand -hex 32>
   ```

### 2. Application Structure

**Always follow this layered pattern for non-trivial projects**:

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, lifespan, middleware
│   ├── config.py            # Settings with pydantic-settings
│   ├── database.py          # Engine, session, dependencies
│   ├── models/              # SQLModel/Pydantic models
│   │   ├── __init__.py
│   │   └── user.py
│   ├── schemas/             # Request/Response schemas (if separate from models)
│   ├── routers/             # API route handlers
│   │   ├── __init__.py
│   │   └── users.py
│   ├── services/            # Business logic layer
│   ├── repositories/        # Data access layer (optional)
│   └── core/                # Security, utils, exceptions
│       ├── security.py
│       └── exceptions.py
├── tests/
│   ├── conftest.py          # Fixtures
│   └── test_users.py
├── .env.example
├── .gitignore
└── pyproject.toml
```

### 3. Core Components

#### Configuration (config.py)
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    database_url: str
    secret_key: str
    debug: bool = False

settings = Settings()
```

#### Database Setup (database.py)
```python
from typing import Annotated
from fastapi import Depends
from sqlmodel import Session, SQLModel, create_engine

engine = create_engine(settings.database_url)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]
```

#### Lifespan Events (main.py)
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    SQLModel.metadata.create_all(engine)
    yield
    # Shutdown
    engine.dispose()

app = FastAPI(lifespan=lifespan)
```

### 4. Models and Schemas

**Use SQLModel for unified ORM + Pydantic models**:

```python
from sqlmodel import SQLModel, Field
from typing import Optional

# Base model (shared fields)
class UserBase(SQLModel):
    email: str = Field(index=True, unique=True)
    full_name: Optional[str] = None

# Database model (table=True)
class User(UserBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str

# Request schemas
class UserCreate(UserBase):
    password: str

# Response schemas
class UserPublic(UserBase):
    id: int
```

### 5. Router Organization

```python
from fastapi import APIRouter, HTTPException, status

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}}
)

@router.get("/", response_model=list[UserPublic])
def list_users(session: SessionDep, skip: int = 0, limit: int = 100):
    return session.exec(select(User).offset(skip).limit(limit)).all()

@router.get("/{user_id}", response_model=UserPublic)
def get_user(user_id: int, session: SessionDep):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

**Include routers in main.py**:
```python
from app.routers import users, items

app.include_router(users.router, prefix="/api/v1")
app.include_router(items.router, prefix="/api/v1")
```

### 6. Security Implementation

**ALWAYS implement security for production**. See [REFERENCE.md](REFERENCE.md) for complete patterns.

**Minimum security checklist**:
- [ ] OAuth2 + JWT authentication
- [ ] Password hashing with Argon2id
- [ ] CORS configuration (explicit origins, never "*")
- [ ] HTTPS enforcement
- [ ] Rate limiting
- [ ] Input validation (Pydantic handles this)
- [ ] Security headers (HSTS, CSP, X-Frame-Options)

**Quick JWT setup**:
```python
from datetime import datetime, timedelta, timezone
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({**data, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    # Fetch and return user from database
```

### 7. Exception Handling

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class AppException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
```

### 8. Testing

**Always write tests. Use pytest + httpx**:

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session
from sqlmodel.pool import StaticPool

from app.main import app
from app.database import get_session

@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

@pytest.fixture
def client(session):
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
```

```python
# tests/test_users.py
def test_create_user(client):
    response = client.post("/api/v1/users/", json={
        "email": "test@example.com",
        "password": "secret123"
    })
    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"
```

**Async testing**:
```python
import pytest
from httpx import ASGITransport, AsyncClient

@pytest.mark.anyio
async def test_async_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.get("/")
        assert response.status_code == 200
```

### 9. Production Deployment

**Running with workers**:
```bash
# Development
uv run fastapi dev app/main.py

# Production with multiple workers
uv run fastapi run app/main.py --workers 4

# Or with uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Docker deployment** (see templates/Dockerfile):
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app ./app
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--host", "0.0.0.0"]
```

### 10. Background Tasks

**Simple background tasks**:
```python
from fastapi import BackgroundTasks

def send_email(email: str, message: str):
    # Email sending logic
    pass

@router.post("/notify/")
async def notify(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(send_email, email, "Welcome!")
    return {"status": "queued"}
```

**For complex jobs**: Use Celery, ARQ, or similar task queues.

## Best Practices Summary

| Area | Practice |
|------|----------|
| **Structure** | Layered architecture: routers → services → repositories |
| **Models** | SQLModel for unified ORM + validation |
| **Config** | pydantic-settings with .env files |
| **Security** | OAuth2/JWT, Argon2id passwords, CORS, rate limiting |
| **Validation** | Pydantic models for all inputs |
| **Errors** | Custom exception handlers, proper HTTP codes |
| **Testing** | pytest + TestClient, dependency overrides |
| **Async** | Use async for I/O-bound operations |
| **Docs** | Let FastAPI generate OpenAPI, add descriptions |

## Common Patterns

### Pagination
```python
@router.get("/items/")
def list_items(skip: int = 0, limit: int = Query(default=100, le=100)):
    return session.exec(select(Item).offset(skip).limit(limit)).all()
```

### Filtering
```python
@router.get("/items/")
def list_items(
    status: Optional[str] = None,
    category: Optional[str] = None
):
    query = select(Item)
    if status:
        query = query.where(Item.status == status)
    if category:
        query = query.where(Item.category == category)
    return session.exec(query).all()
```

### Dependency Injection
```python
def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

CurrentUser = Annotated[User, Depends(get_current_active_user)]

@router.get("/me")
def read_current_user(user: CurrentUser):
    return user
```

## When to Use This Skill

Invoke this skill when:
- Creating new FastAPI projects or APIs
- Adding endpoints, routers, or models
- Implementing authentication/authorization
- Setting up database integration
- Writing API tests
- Preparing for production deployment
- Troubleshooting FastAPI issues
