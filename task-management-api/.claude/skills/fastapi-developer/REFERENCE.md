# FastAPI Reference Guide

Detailed patterns, configurations, and best practices for production FastAPI applications.

## Table of Contents

1. [Authentication & Authorization](#authentication--authorization)
2. [Security Middleware](#security-middleware)
3. [CORS Configuration](#cors-configuration)
4. [Database Patterns](#database-patterns)
5. [Error Handling](#error-handling)
6. [Middleware Stack](#middleware-stack)
7. [Production Configuration](#production-configuration)
8. [Performance Optimization](#performance-optimization)

---

## Authentication & Authorization

### Complete OAuth2 + JWT Implementation

```python
# app/core/security.py
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel, ValidationError

from app.config import settings

# Password hashing with Argon2id (recommended)
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# OAuth2 scheme with scopes
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="api/v1/auth/token",
    scopes={
        "read": "Read access",
        "write": "Write access",
        "admin": "Admin access"
    }
)

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list[str] = []

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme)
) -> User:
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )

    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(username=username, scopes=token_scopes)
    except (InvalidTokenError, ValidationError):
        raise credentials_exception

    user = get_user_by_username(token_data.username)
    if user is None:
        raise credentials_exception

    # Check scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )

    return user

async def get_current_active_user(
    current_user: User = Security(get_current_user, scopes=["read"])
) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Type aliases for dependency injection
CurrentUser = Annotated[User, Depends(get_current_active_user)]
AdminUser = Annotated[User, Security(get_current_user, scopes=["admin"])]
```

### Login Endpoint

```python
# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: SessionDep
):
    user = authenticate_user(session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user.username, "scopes": form_data.scopes}
    )
    return Token(access_token=access_token)

def authenticate_user(session: Session, username: str, password: str) -> Optional[User]:
    user = session.exec(select(User).where(User.username == username)).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user
```

### Protected Endpoints with Scopes

```python
@router.get("/users/me", response_model=UserPublic)
async def read_users_me(current_user: CurrentUser):
    return current_user

@router.get("/admin/users", response_model=list[UserPublic])
async def list_all_users(
    admin_user: AdminUser,  # Requires admin scope
    session: SessionDep
):
    return session.exec(select(User)).all()

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Security(get_current_user, scopes=["admin", "write"]),
    session: SessionDep
):
    # Requires both admin and write scopes
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    session.delete(user)
    session.commit()
    return {"deleted": True}
```

---

## Security Middleware

### FastAPI-Guard Integration

For comprehensive security, use fastapi-guard:

```bash
pip install fastapi-guard
```

```python
# app/core/security_middleware.py
from fastapi import FastAPI
from guard import SecurityConfig, SecurityMiddleware

def configure_security(app: FastAPI) -> None:
    config = SecurityConfig(
        # === IP Access Control ===
        whitelist=["127.0.0.1", "10.0.0.0/8"],  # Always allowed
        blacklist=[],  # Always blocked

        # === Rate Limiting ===
        enable_rate_limiting=True,
        rate_limit=100,              # Requests per window
        rate_limit_window=60,        # Window in seconds

        # === Auto-Banning ===
        enable_ip_banning=True,
        auto_ban_threshold=5,        # Suspicious requests before ban
        auto_ban_duration=3600,      # Ban duration in seconds

        # === Penetration Detection ===
        enable_penetration_detection=True,
        blocked_user_agents=["sqlmap", "nikto", "nmap", "masscan"],

        # === Security Headers (OWASP) ===
        security_headers={
            "enabled": True,
            "csp": {
                "default-src": ["'self'"],
                "script-src": ["'self'"],
                "style-src": ["'self'", "'unsafe-inline'"],
                "img-src": ["'self'", "data:", "https:"],
                "font-src": ["'self'"],
                "connect-src": ["'self'"],
                "frame-ancestors": ["'none'"],
            },
            "hsts": {
                "max_age": 31536000,
                "include_subdomains": True,
                "preload": True
            },
            "frame_options": "DENY",
            "content_type_options": "nosniff",
            "referrer_policy": "strict-origin-when-cross-origin",
        },

        # === HTTPS Enforcement ===
        enforce_https=True,

        # === Redis for Distributed State ===
        enable_redis=True,
        redis_url="redis://localhost:6379/0",
        redis_prefix="myapp:guard:",

        # === Logging ===
        custom_log_file="logs/security.log",
        log_request_level="INFO",
        log_suspicious_level="WARNING",

        # === Excluded Paths ===
        exclude_paths=["/health", "/metrics", "/docs", "/redoc", "/openapi.json"],
    )

    app.add_middleware(SecurityMiddleware, config=config)
```

### Custom Rate Limiting (Without fastapi-guard)

```python
# app/core/rate_limit.py
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import HTTPException, Request

class RateLimiter:
    def __init__(self, requests: int, window: int):
        self.requests = requests
        self.window = timedelta(seconds=window)
        self.clients: dict[str, list[datetime]] = defaultdict(list)

    async def __call__(self, request: Request):
        client_ip = request.client.host
        now = datetime.now()

        # Clean old requests
        self.clients[client_ip] = [
            req_time for req_time in self.clients[client_ip]
            if now - req_time < self.window
        ]

        if len(self.clients[client_ip]) >= self.requests:
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )

        self.clients[client_ip].append(now)

# Usage
rate_limiter = RateLimiter(requests=100, window=60)

@router.get("/endpoint", dependencies=[Depends(rate_limiter)])
async def limited_endpoint():
    return {"status": "ok"}
```

---

## CORS Configuration

### Production CORS Setup

```python
# app/main.py
from fastapi.middleware.cors import CORSMiddleware

# NEVER use ["*"] in production!
ALLOWED_ORIGINS = [
    "https://myapp.com",
    "https://www.myapp.com",
    "https://admin.myapp.com",
]

# Development origins (conditionally added)
if settings.debug:
    ALLOWED_ORIGINS.extend([
        "http://localhost:3000",
        "http://localhost:5173",
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining"],
    max_age=600,  # Preflight cache duration
)
```

---

## Database Patterns

### Repository Pattern

```python
# app/repositories/base.py
from typing import Generic, TypeVar, Optional, Type
from sqlmodel import Session, select, SQLModel

ModelType = TypeVar("ModelType", bound=SQLModel)

class BaseRepository(Generic[ModelType]):
    def __init__(self, model: Type[ModelType], session: Session):
        self.model = model
        self.session = session

    def get(self, id: int) -> Optional[ModelType]:
        return self.session.get(self.model, id)

    def get_all(self, skip: int = 0, limit: int = 100) -> list[ModelType]:
        return self.session.exec(
            select(self.model).offset(skip).limit(limit)
        ).all()

    def create(self, obj: ModelType) -> ModelType:
        self.session.add(obj)
        self.session.commit()
        self.session.refresh(obj)
        return obj

    def update(self, db_obj: ModelType, update_data: dict) -> ModelType:
        for key, value in update_data.items():
            setattr(db_obj, key, value)
        self.session.add(db_obj)
        self.session.commit()
        self.session.refresh(db_obj)
        return db_obj

    def delete(self, id: int) -> bool:
        obj = self.get(id)
        if obj:
            self.session.delete(obj)
            self.session.commit()
            return True
        return False
```

```python
# app/repositories/user.py
from app.models.user import User

class UserRepository(BaseRepository[User]):
    def __init__(self, session: Session):
        super().__init__(User, session)

    def get_by_email(self, email: str) -> Optional[User]:
        return self.session.exec(
            select(User).where(User.email == email)
        ).first()

    def get_by_username(self, username: str) -> Optional[User]:
        return self.session.exec(
            select(User).where(User.username == username)
        ).first()
```

### Service Layer Pattern

```python
# app/services/user.py
from fastapi import HTTPException

class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def create_user(self, user_create: UserCreate) -> User:
        # Check if email exists
        existing = self.repository.get_by_email(user_create.email)
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Hash password
        hashed_password = get_password_hash(user_create.password)

        # Create user
        user = User(
            email=user_create.email,
            username=user_create.username,
            hashed_password=hashed_password
        )
        return self.repository.create(user)

    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self.repository.get_by_username(username)
        if not user or not verify_password(password, user.hashed_password):
            return None
        return user
```

### Dependency Injection with Services

```python
# app/dependencies.py
from typing import Annotated
from fastapi import Depends

def get_user_repository(session: SessionDep) -> UserRepository:
    return UserRepository(session)

def get_user_service(
    repository: UserRepository = Depends(get_user_repository)
) -> UserService:
    return UserService(repository)

UserServiceDep = Annotated[UserService, Depends(get_user_service)]

# Usage in router
@router.post("/users/", response_model=UserPublic, status_code=201)
def create_user(user_create: UserCreate, service: UserServiceDep):
    return service.create_user(user_create)
```

---

## Error Handling

### Structured Exception Handling

```python
# app/core/exceptions.py
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None

class AppException(Exception):
    def __init__(
        self,
        status_code: int,
        error: str,
        detail: Optional[str] = None,
        code: Optional[str] = None
    ):
        self.status_code = status_code
        self.error = error
        self.detail = detail
        self.code = code

class NotFoundError(AppException):
    def __init__(self, resource: str, id: Any):
        super().__init__(
            status_code=404,
            error=f"{resource} not found",
            detail=f"{resource} with id {id} does not exist",
            code="NOT_FOUND"
        )

class ValidationError(AppException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=422,
            error="Validation error",
            detail=detail,
            code="VALIDATION_ERROR"
        )

class AuthenticationError(AppException):
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=401,
            error="Authentication error",
            detail=detail,
            code="AUTH_ERROR"
        )

class AuthorizationError(AppException):
    def __init__(self, detail: str = "Permission denied"):
        super().__init__(
            status_code=403,
            error="Authorization error",
            detail=detail,
            code="FORBIDDEN"
        )
```

### Exception Handlers

```python
# app/main.py
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.error,
            detail=exc.detail,
            code=exc.code
        ).model_dump()
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=str(exc.detail)).model_dump()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append(f"{field}: {error['msg']}")

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation error",
            detail="; ".join(errors),
            code="VALIDATION_ERROR"
        ).model_dump()
    )
```

---

## Middleware Stack

### Recommended Middleware Order

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI()

# Order matters! First added = last executed (outermost)

# 1. Trusted Host (security - outermost)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["myapp.com", "*.myapp.com"]
)

# 2. Security Middleware (rate limiting, penetration detection)
# app.add_middleware(SecurityMiddleware, config=security_config)

# 3. CORS
app.add_middleware(CORSMiddleware, ...)

# 4. GZip Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 5. Request ID (innermost - closest to route handlers)
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

---

## Production Configuration

### Settings for Production

```python
# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application
    app_name: str = "MyAPI"
    debug: bool = False
    environment: str = "production"

    # Database
    database_url: str
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # CORS
    cors_origins: list[str] = []

    # Redis
    redis_url: Optional[str] = None

    # Logging
    log_level: str = "INFO"

settings = Settings()
```

### Production Database Engine

```python
# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections after 1 hour
)
```

### Logging Configuration

```python
# app/core/logging.py
import logging
import sys
from typing import Any

def setup_logging() -> None:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/app.log"),
        ]
    )

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
```

---

## Performance Optimization

### Async Database Operations

```python
# For async operations, use asyncpg with SQLAlchemy async
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

async_engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
    pool_size=settings.db_pool_size,
)

AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_async_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

### Response Caching

```python
from fastapi import Response
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_data(key: str) -> dict:
    # Expensive computation
    return {"data": "cached"}

@router.get("/cached")
async def cached_endpoint(response: Response):
    response.headers["Cache-Control"] = "public, max-age=300"
    return get_cached_data("key")
```

### Pagination with Cursor

```python
from typing import Optional
from pydantic import BaseModel

class CursorPage(BaseModel):
    items: list
    next_cursor: Optional[str]
    has_more: bool

@router.get("/items", response_model=CursorPage)
async def list_items(
    cursor: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    session: SessionDep
):
    query = select(Item).order_by(Item.id)

    if cursor:
        cursor_id = int(cursor)
        query = query.where(Item.id > cursor_id)

    items = session.exec(query.limit(limit + 1)).all()

    has_more = len(items) > limit
    items = items[:limit]

    return CursorPage(
        items=items,
        next_cursor=str(items[-1].id) if items and has_more else None,
        has_more=has_more
    )
```
