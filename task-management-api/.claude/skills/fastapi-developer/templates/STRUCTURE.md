# FastAPI Project Structure Templates

## Minimal (Single File)

For prototypes, learning, or very simple APIs.

```
project/
├── main.py          # Everything in one file
├── .env.local       # Environment variables (gitignored)
├── .gitignore
└── pyproject.toml
```

**Use when:** Quick prototypes, learning FastAPI, APIs with < 5 endpoints.

---

## Standard (Modular)

For most production applications.

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app instance, lifespan, middleware
│   ├── config.py            # Settings with pydantic-settings
│   ├── database.py          # Engine, session factory, dependencies
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py          # User model
│   │   └── item.py          # Item model
│   ├── schemas/             # Optional: separate request/response schemas
│   │   └── __init__.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py          # Authentication endpoints
│   │   ├── users.py         # User CRUD
│   │   └── items.py         # Item CRUD
│   └── core/
│       ├── __init__.py
│       ├── security.py      # JWT, password hashing
│       └── exceptions.py    # Custom exceptions
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures
│   ├── test_auth.py
│   ├── test_users.py
│   └── test_items.py
├── .env.example             # Example environment variables
├── .env.local               # Actual environment variables (gitignored)
├── .gitignore
├── pyproject.toml
└── README.md
```

**Use when:** Production APIs, team projects, applications with authentication.

---

## Enterprise (Layered Architecture)

For large-scale applications with complex business logic.

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py              # App factory, middleware stack
│   ├── config.py            # Settings with environment profiles
│   ├── database.py          # Database configuration
│   │
│   ├── api/                 # API Layer (presentation)
│   │   ├── __init__.py
│   │   ├── deps.py          # Shared dependencies
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py    # API v1 router aggregation
│   │       ├── auth.py
│   │       ├── users.py
│   │       └── items.py
│   │
│   ├── models/              # Database models (ORM)
│   │   ├── __init__.py
│   │   ├── base.py          # Base model class
│   │   ├── user.py
│   │   └── item.py
│   │
│   ├── schemas/             # Pydantic schemas (DTOs)
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   │
│   ├── services/            # Business logic layer
│   │   ├── __init__.py
│   │   ├── user_service.py
│   │   └── item_service.py
│   │
│   ├── repositories/        # Data access layer
│   │   ├── __init__.py
│   │   ├── base.py          # Generic repository
│   │   ├── user_repository.py
│   │   └── item_repository.py
│   │
│   ├── core/                # Core utilities
│   │   ├── __init__.py
│   │   ├── security.py      # Authentication, authorization
│   │   ├── exceptions.py    # Custom exceptions
│   │   └── events.py        # Lifespan events
│   │
│   └── tasks/               # Background tasks
│       ├── __init__.py
│       └── email.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/                # Unit tests
│   │   ├── test_services/
│   │   └── test_repositories/
│   ├── integration/         # Integration tests
│   │   └── test_api/
│   └── e2e/                 # End-to-end tests
│
├── alembic/                 # Database migrations
│   ├── versions/
│   ├── env.py
│   └── alembic.ini
│
├── scripts/                 # Utility scripts
│   ├── seed_db.py
│   └── create_admin.py
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml
│
├── .env.example
├── .env.local
├── .gitignore
├── pyproject.toml
└── README.md
```

**Use when:** Large teams, complex domains, microservices, applications requiring strict separation of concerns.

---

## File Templates

### pyproject.toml

```toml
[project]
name = "my-api"
version = "1.0.0"
description = "FastAPI application"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "sqlmodel>=0.0.22",
    "pydantic-settings>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "pyjwt>=2.8.0",
    "passlib[argon2]>=1.7.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
    "ruff>=0.5.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/
ENV/

# Environment
.env
.env.local
.env.*.local

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.coverage
htmlcov/
.pytest_cache/

# Build
dist/
build/
*.egg-info/

# Logs
logs/
*.log

# Database
*.db
*.sqlite

# Docker
docker-compose.override.yml
```

### .env.example

```bash
# Application
APP_NAME=MyAPI
DEBUG=false
ENVIRONMENT=development

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mydb

# Security
SECRET_KEY=your-secret-key-here-generate-with-openssl-rand-hex-32
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# CORS
CORS_ORIGINS=["http://localhost:3000"]
```

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application
COPY app ./app

# Expose port
EXPOSE 8000

# Run application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env.local
    depends_on:
      - db
      - redis
    volumes:
      - ./app:/app/app  # Hot reload in development

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${DB_USER:-postgres}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-postgres}
      POSTGRES_DB: ${DB_NAME:-mydb}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

volumes:
  postgres_data:
  redis_data:
```
