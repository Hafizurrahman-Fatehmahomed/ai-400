# FastAPI Examples

Concrete code examples ranging from minimal Hello World to production-ready patterns.

## Table of Contents

1. [Minimal Examples](#minimal-examples)
2. [CRUD Operations](#crud-operations)
3. [Authentication](#authentication)
4. [Database Integration](#database-integration)
5. [Testing](#testing)
6. [Production Patterns](#production-patterns)
7. [Advanced Patterns](#advanced-patterns)

---

## Minimal Examples

### Hello World (Single File)

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Run: uvicorn main:app --reload
```

### With Path Parameters

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

### With Request Body (Pydantic)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

@app.post("/items/")
def create_item(item: Item):
    return {"item_name": item.name, "item_price": item.price}
```

### With Query Parameters and Validation

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
def read_items(
    skip: int = 0,
    limit: int = Query(default=10, le=100, description="Max items to return"),
    search: str = Query(default=None, min_length=3, max_length=50)
):
    return {"skip": skip, "limit": limit, "search": search}
```

---

## CRUD Operations

### Complete CRUD with SQLModel

```python
# app/models/item.py
from typing import Optional
from sqlmodel import SQLModel, Field

class ItemBase(SQLModel):
    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    price: float = Field(gt=0)
    is_available: bool = True

class Item(ItemBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

class ItemCreate(ItemBase):
    pass

class ItemUpdate(SQLModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    price: Optional[float] = Field(default=None, gt=0)
    is_available: Optional[bool] = None

class ItemPublic(ItemBase):
    id: int
```

```python
# app/routers/items.py
from fastapi import APIRouter, HTTPException, status, Query
from sqlmodel import select
from typing import Optional

from app.database import SessionDep
from app.models.item import Item, ItemCreate, ItemUpdate, ItemPublic

router = APIRouter(prefix="/items", tags=["items"])

@router.get("/", response_model=list[ItemPublic])
def list_items(
    session: SessionDep,
    skip: int = 0,
    limit: int = Query(default=20, le=100),
    is_available: Optional[bool] = None
):
    query = select(Item)
    if is_available is not None:
        query = query.where(Item.is_available == is_available)
    return session.exec(query.offset(skip).limit(limit)).all()

@router.get("/{item_id}", response_model=ItemPublic)
def get_item(item_id: int, session: SessionDep):
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@router.post("/", response_model=ItemPublic, status_code=status.HTTP_201_CREATED)
def create_item(item_create: ItemCreate, session: SessionDep):
    item = Item.model_validate(item_create)
    session.add(item)
    session.commit()
    session.refresh(item)
    return item

@router.put("/{item_id}", response_model=ItemPublic)
def update_item(item_id: int, item_update: ItemUpdate, session: SessionDep):
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    update_data = item_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(item, key, value)

    session.add(item)
    session.commit()
    session.refresh(item)
    return item

@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int, session: SessionDep):
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    session.delete(item)
    session.commit()
```

---

## Authentication

### Basic OAuth2 Password Flow

```python
# app/routers/auth.py
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlmodel import select

from app.config import settings
from app.database import SessionDep
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["authentication"])

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm="HS256")

def authenticate_user(session, username: str, password: str):
    user = session.exec(select(User).where(User.username == username)).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: SessionDep = None
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception

    user = session.exec(select(User).where(User.username == username)).first()
    if user is None:
        raise credentials_exception
    return user

CurrentUser = Annotated[User, Depends(get_current_user)]

@router.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), session: SessionDep = None):
    user = authenticate_user(session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return Token(access_token=access_token)

@router.get("/me")
def read_users_me(current_user: CurrentUser):
    return {"username": current_user.username, "email": current_user.email}
```

### API Key Authentication

```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key required"
        )
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return api_key

@router.get("/protected", dependencies=[Depends(get_api_key)])
def protected_endpoint():
    return {"message": "Access granted"}
```

---

## Database Integration

### Complete Database Setup

```python
# app/database.py
from typing import Annotated
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

engine = create_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables
    SQLModel.metadata.create_all(engine)
    yield
    # Shutdown: Dispose engine
    engine.dispose()
```

### Relationships

```python
# app/models/user.py
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)
    hashed_password: str

    # Relationship to posts
    posts: List["Post"] = Relationship(back_populates="author")

# app/models/post.py
class Post(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    content: str
    author_id: Optional[int] = Field(default=None, foreign_key="user.id")

    # Relationship to user
    author: Optional[User] = Relationship(back_populates="posts")
```

### Eager Loading

```python
from sqlmodel import select
from sqlalchemy.orm import selectinload

@router.get("/users/{user_id}/with-posts")
def get_user_with_posts(user_id: int, session: SessionDep):
    statement = (
        select(User)
        .where(User.id == user_id)
        .options(selectinload(User.posts))
    )
    user = session.exec(statement).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

---

## Testing

### Test Configuration

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine
from sqlmodel.pool import StaticPool

from app.main import app
from app.database import get_session
from app.core.security import get_password_hash
from app.models.user import User

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
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

@pytest.fixture
def test_user(session: Session) -> User:
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("testpassword")
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user

@pytest.fixture
def auth_headers(client: TestClient, test_user: User) -> dict:
    response = client.post(
        "/auth/token",
        data={"username": "testuser", "password": "testpassword"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
```

### Test Examples

```python
# tests/test_items.py
from fastapi.testclient import TestClient

def test_create_item(client: TestClient, auth_headers: dict):
    response = client.post(
        "/items/",
        json={"name": "Test Item", "price": 9.99},
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Item"
    assert data["price"] == 9.99
    assert "id" in data

def test_read_item(client: TestClient, auth_headers: dict):
    # Create item first
    create_response = client.post(
        "/items/",
        json={"name": "Test Item", "price": 9.99},
        headers=auth_headers
    )
    item_id = create_response.json()["id"]

    # Read item
    response = client.get(f"/items/{item_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Test Item"

def test_read_item_not_found(client: TestClient):
    response = client.get("/items/99999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Item not found"

def test_update_item(client: TestClient, auth_headers: dict):
    # Create item
    create_response = client.post(
        "/items/",
        json={"name": "Original", "price": 10.00},
        headers=auth_headers
    )
    item_id = create_response.json()["id"]

    # Update item
    response = client.put(
        f"/items/{item_id}",
        json={"name": "Updated"},
        headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Updated"
    assert response.json()["price"] == 10.00  # Unchanged

def test_delete_item(client: TestClient, auth_headers: dict):
    # Create item
    create_response = client.post(
        "/items/",
        json={"name": "To Delete", "price": 5.00},
        headers=auth_headers
    )
    item_id = create_response.json()["id"]

    # Delete item
    response = client.delete(f"/items/{item_id}", headers=auth_headers)
    assert response.status_code == 204

    # Verify deleted
    get_response = client.get(f"/items/{item_id}")
    assert get_response.status_code == 404

def test_list_items_with_filter(client: TestClient):
    response = client.get("/items/?is_available=true&limit=10")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
```

### Async Tests

```python
# tests/test_async.py
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app

@pytest.mark.anyio
async def test_async_root():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.get("/")
        assert response.status_code == 200
```

---

## Production Patterns

### Health Check Endpoint

```python
# app/routers/health.py
from fastapi import APIRouter, Depends
from sqlmodel import text
from pydantic import BaseModel
from datetime import datetime

from app.database import SessionDep

router = APIRouter(tags=["health"])

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    database: str
    version: str

@router.get("/health", response_model=HealthCheck)
def health_check(session: SessionDep):
    # Check database connectivity
    try:
        session.exec(text("SELECT 1"))
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"

    return HealthCheck(
        status="healthy" if db_status == "healthy" else "degraded",
        timestamp=datetime.utcnow(),
        database=db_status,
        version="1.0.0"
    )

@router.get("/ready")
def readiness_check(session: SessionDep):
    """Kubernetes readiness probe"""
    try:
        session.exec(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/live")
def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive"}
```

### Request Logging Middleware

```python
# app/middleware/logging.py
import time
import logging
import uuid
from fastapi import Request

logger = logging.getLogger(__name__)

async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Add request ID to state for access in handlers
    request.state.request_id = request_id

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status={response.status_code} "
        f"duration={process_time:.3f}s"
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Usage in main.py
app.middleware("http")(log_requests)
```

### Background Tasks

```python
# app/tasks/email.py
import logging
from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)

def send_email(to: str, subject: str, body: str):
    """Simulated email sending"""
    logger.info(f"Sending email to {to}: {subject}")
    # Actual email logic here
    pass

def send_welcome_email(email: str, username: str):
    send_email(
        to=email,
        subject="Welcome!",
        body=f"Hello {username}, welcome to our platform!"
    )

# Usage in router
@router.post("/users/", response_model=UserPublic, status_code=201)
def create_user(
    user_create: UserCreate,
    background_tasks: BackgroundTasks,
    session: SessionDep
):
    user = create_user_in_db(session, user_create)
    background_tasks.add_task(send_welcome_email, user.email, user.username)
    return user
```

---

## Advanced Patterns

### WebSocket Example

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Client {client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client {client_id} left")
```

### File Upload

```python
from fastapi import File, UploadFile
import shutil
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="File type not allowed")

    # Validate file size (10MB max)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    # Save file
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "size": file_size}
```

### Streaming Response

```python
from fastapi.responses import StreamingResponse
import asyncio

async def generate_data():
    for i in range(10):
        yield f"data: {i}\n\n"
        await asyncio.sleep(1)

@router.get("/stream")
async def stream():
    return StreamingResponse(
        generate_data(),
        media_type="text/event-stream"
    )
```

### Caching with Redis

```python
import redis
import json
from functools import wraps

redis_client = redis.from_url(settings.redis_url)

def cache(expire: int = 300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Try to get from cache
            cached = redis_client.get(key)
            if cached:
                return json.loads(cached)

            # Execute function and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(key, expire, json.dumps(result))
            return result
        return wrapper
    return decorator

@router.get("/expensive-data")
@cache(expire=300)
async def get_expensive_data():
    # Expensive computation here
    return {"data": "expensive result"}
```

### Dependency Injection with Yield (Cleanup)

```python
from typing import Generator

def get_db_transaction(session: SessionDep) -> Generator:
    """Dependency that provides a transaction with automatic rollback on error"""
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise

@router.post("/transfer/")
def transfer_money(
    from_account: int,
    to_account: int,
    amount: float,
    session: Session = Depends(get_db_transaction)
):
    # Both operations in same transaction
    debit_account(session, from_account, amount)
    credit_account(session, to_account, amount)
    return {"status": "success"}
```
