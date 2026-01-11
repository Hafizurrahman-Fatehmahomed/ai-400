# SQLModel Production Examples

Real-world production scenarios from simple CRUD to enterprise multi-database systems.

## Table of Contents

1. [Simple CRUD API](#simple-crud-api)
2. [User Authentication System](#user-authentication-system)
3. [E-Commerce Order System](#e-commerce-order-system)
4. [Multi-Tenant SaaS Application](#multi-tenant-saas-application)
5. [Event Sourcing Pattern](#event-sourcing-pattern)
6. [Audit Trail System](#audit-trail-system)
7. [Async High-Performance API](#async-high-performance-api)
8. [Read Replica Pattern](#read-replica-pattern)
9. [Soft Delete Pattern](#soft-delete-pattern)
10. [Full-Text Search](#full-text-search)

---

## Simple CRUD API

A complete task management API with FastAPI.

### Models

```python
# app/models/task.py
from datetime import datetime
from enum import Enum
from sqlmodel import Field, SQLModel

class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"

class TaskBase(SQLModel):
    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    status: TaskStatus = Field(default=TaskStatus.TODO)
    due_date: datetime | None = None

class Task(TaskBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class TaskCreate(TaskBase):
    pass

class TaskUpdate(SQLModel):
    title: str | None = None
    description: str | None = None
    status: TaskStatus | None = None
    due_date: datetime | None = None

class TaskPublic(TaskBase):
    id: int
    created_at: datetime
    updated_at: datetime
```

### Database Setup

```python
# app/database.py
from collections.abc import Generator
from sqlmodel import Session, SQLModel, create_engine
from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
```

### CRUD Operations

```python
# app/crud/task.py
from sqlmodel import Session, select
from app.models.task import Task, TaskCreate, TaskUpdate

def create_task(session: Session, task: TaskCreate) -> Task:
    db_task = Task.model_validate(task)
    session.add(db_task)
    session.commit()
    session.refresh(db_task)
    return db_task

def get_task(session: Session, task_id: int) -> Task | None:
    return session.get(Task, task_id)

def get_tasks(
    session: Session,
    skip: int = 0,
    limit: int = 100,
    status: str | None = None
) -> list[Task]:
    statement = select(Task)
    if status:
        statement = statement.where(Task.status == status)
    statement = statement.offset(skip).limit(limit)
    return session.exec(statement).all()

def update_task(
    session: Session,
    task_id: int,
    task_update: TaskUpdate
) -> Task | None:
    db_task = session.get(Task, task_id)
    if not db_task:
        return None
    update_data = task_update.model_dump(exclude_unset=True)
    db_task.sqlmodel_update(update_data)
    db_task.updated_at = datetime.utcnow()
    session.add(db_task)
    session.commit()
    session.refresh(db_task)
    return db_task

def delete_task(session: Session, task_id: int) -> bool:
    db_task = session.get(Task, task_id)
    if not db_task:
        return False
    session.delete(db_task)
    session.commit()
    return True
```

### API Endpoints

```python
# app/routers/tasks.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session
from app.database import get_session
from app.models.task import Task, TaskCreate, TaskUpdate, TaskPublic
from app.crud import task as crud

router = APIRouter(prefix="/tasks", tags=["tasks"])

@router.post("/", response_model=TaskPublic, status_code=201)
def create_task(
    task: TaskCreate,
    session: Session = Depends(get_session)
):
    return crud.create_task(session, task)

@router.get("/", response_model=list[TaskPublic])
def list_tasks(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    status: str | None = None,
    session: Session = Depends(get_session)
):
    return crud.get_tasks(session, skip, limit, status)

@router.get("/{task_id}", response_model=TaskPublic)
def get_task(
    task_id: int,
    session: Session = Depends(get_session)
):
    task = crud.get_task(session, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task

@router.patch("/{task_id}", response_model=TaskPublic)
def update_task(
    task_id: int,
    task_update: TaskUpdate,
    session: Session = Depends(get_session)
):
    task = crud.update_task(session, task_id, task_update)
    if not task:
        raise HTTPException(404, "Task not found")
    return task

@router.delete("/{task_id}", status_code=204)
def delete_task(
    task_id: int,
    session: Session = Depends(get_session)
):
    if not crud.delete_task(session, task_id):
        raise HTTPException(404, "Task not found")
```

---

## User Authentication System

Complete JWT-based authentication with password hashing.

### Models

```python
# app/models/user.py
from datetime import datetime
from sqlmodel import Field, SQLModel
from pydantic import EmailStr, field_validator

class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    full_name: str = Field(max_length=100)
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)

class User(UserBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: datetime | None = None

class UserCreate(SQLModel):
    email: EmailStr
    full_name: str = Field(max_length=100)
    password: str = Field(min_length=8)

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        if not any(c in '!@#$%^&*()' for c in v):
            raise ValueError('Password must contain special character')
        return v

class UserUpdate(SQLModel):
    email: EmailStr | None = None
    full_name: str | None = None
    password: str | None = None

class UserPublic(UserBase):
    id: int
    created_at: datetime

class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"

class TokenPayload(SQLModel):
    sub: int
    exp: datetime
```

### Security Utilities

```python
# app/core/security.py
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import jwt, JWTError
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(user_id: int, expires_delta: timedelta | None = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))
    to_encode = {"sub": str(user_id), "exp": expire}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")

def decode_token(token: str) -> int | None:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return int(payload["sub"])
    except JWTError:
        return None
```

### Authentication Dependencies

```python
# app/deps.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session
from app.database import get_session
from app.models.user import User
from app.core.security import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session)
) -> User:
    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = session.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user

def get_current_superuser(
    current_user: User = Depends(get_current_user)
) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user
```

### Auth Endpoints

```python
# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select
from app.database import get_session
from app.models.user import User, UserCreate, UserPublic, Token
from app.core.security import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserPublic, status_code=201)
def register(user: UserCreate, session: Session = Depends(get_session)):
    # Check if user exists
    existing = session.exec(
        select(User).where(User.email == user.email)
    ).first()
    if existing:
        raise HTTPException(400, "Email already registered")

    db_user = User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hash_password(user.password),
    )
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user

@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session)
):
    user = session.exec(
        select(User).where(User.email == form_data.username)
    ).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(401, "Incorrect email or password")

    if not user.is_active:
        raise HTTPException(400, "User is inactive")

    # Update last login
    user.last_login = datetime.utcnow()
    session.add(user)
    session.commit()

    return Token(access_token=create_access_token(user.id))

@router.get("/me", response_model=UserPublic)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
```

---

## E-Commerce Order System

Complex relationships with order, items, and products.

### Models

```python
# app/models/ecommerce.py
from datetime import datetime
from decimal import Decimal
from enum import Enum
from sqlmodel import Field, Relationship, SQLModel

class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class Product(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(max_length=200, index=True)
    description: str | None = Field(default=None, max_length=2000)
    price: Decimal = Field(max_digits=10, decimal_places=2)
    stock_quantity: int = Field(default=0, ge=0)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    order_items: list["OrderItem"] = Relationship(back_populates="product")

class Order(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    total_amount: Decimal = Field(max_digits=12, decimal_places=2, default=0)
    shipping_address: str = Field(max_length=500)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    user: "User" = Relationship(back_populates="orders")
    items: list["OrderItem"] = Relationship(
        back_populates="order",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )

class OrderItem(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    order_id: int = Field(foreign_key="order.id", index=True)
    product_id: int = Field(foreign_key="product.id", index=True)
    quantity: int = Field(ge=1)
    unit_price: Decimal = Field(max_digits=10, decimal_places=2)
    subtotal: Decimal = Field(max_digits=12, decimal_places=2)

    order: Order = Relationship(back_populates="items")
    product: Product = Relationship(back_populates="order_items")
```

### Order Service with Transaction

```python
# app/services/order.py
from decimal import Decimal
from sqlmodel import Session, select
from sqlalchemy.exc import IntegrityError
from app.models.ecommerce import Order, OrderItem, Product, OrderStatus

class InsufficientStockError(Exception):
    def __init__(self, product_id: int, requested: int, available: int):
        self.product_id = product_id
        self.requested = requested
        self.available = available
        super().__init__(
            f"Insufficient stock for product {product_id}: "
            f"requested {requested}, available {available}"
        )

class OrderService:
    def __init__(self, session: Session):
        self.session = session

    def create_order(
        self,
        user_id: int,
        items: list[dict],  # [{"product_id": 1, "quantity": 2}, ...]
        shipping_address: str
    ) -> Order:
        """Create order with stock validation and reservation."""

        # Start transaction
        order = Order(user_id=user_id, shipping_address=shipping_address)
        self.session.add(order)
        self.session.flush()  # Get order.id without committing

        total = Decimal("0")

        for item_data in items:
            product = self.session.get(Product, item_data["product_id"])
            if not product or not product.is_active:
                raise ValueError(f"Product {item_data['product_id']} not found")

            quantity = item_data["quantity"]

            # Check stock (with row lock for concurrent safety)
            if product.stock_quantity < quantity:
                raise InsufficientStockError(
                    product.id, quantity, product.stock_quantity
                )

            # Reserve stock
            product.stock_quantity -= quantity

            # Create order item
            subtotal = product.price * quantity
            order_item = OrderItem(
                order_id=order.id,
                product_id=product.id,
                quantity=quantity,
                unit_price=product.price,
                subtotal=subtotal,
            )
            self.session.add(order_item)
            total += subtotal

        order.total_amount = total
        self.session.commit()
        self.session.refresh(order)
        return order

    def cancel_order(self, order_id: int) -> Order:
        """Cancel order and restore stock."""
        order = self.session.get(Order, order_id)
        if not order:
            raise ValueError("Order not found")

        if order.status in [OrderStatus.SHIPPED, OrderStatus.DELIVERED]:
            raise ValueError("Cannot cancel shipped/delivered order")

        # Restore stock
        for item in order.items:
            product = self.session.get(Product, item.product_id)
            product.stock_quantity += item.quantity

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        self.session.commit()
        self.session.refresh(order)
        return order
```

---

## Multi-Tenant SaaS Application

Row-level security with tenant isolation.

### Models

```python
# app/models/tenant.py
from datetime import datetime
from sqlmodel import Field, Relationship, SQLModel

class Tenant(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(max_length=100, unique=True)
    slug: str = Field(max_length=50, unique=True, index=True)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    users: list["TenantUser"] = Relationship(back_populates="tenant")
    projects: list["Project"] = Relationship(back_populates="tenant")

class TenantUser(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenant.id", index=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    role: str = Field(default="member")  # owner, admin, member

    tenant: Tenant = Relationship(back_populates="users")
    user: "User" = Relationship(back_populates="tenant_memberships")

class Project(SQLModel, table=True):
    """Tenant-scoped resource."""
    id: int | None = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenant.id", index=True)
    name: str = Field(max_length=200)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    tenant: Tenant = Relationship(back_populates="projects")
```

### Tenant Context

```python
# app/core/tenant.py
from contextvars import ContextVar
from fastapi import Depends, HTTPException, Header
from sqlmodel import Session, select
from app.database import get_session
from app.models.tenant import Tenant, TenantUser
from app.deps import get_current_user

current_tenant: ContextVar[Tenant | None] = ContextVar("current_tenant", default=None)

def get_tenant_from_header(
    x_tenant_id: int = Header(...),
    session: Session = Depends(get_session),
    current_user = Depends(get_current_user)
) -> Tenant:
    """Extract and validate tenant from header."""
    tenant = session.get(Tenant, x_tenant_id)
    if not tenant or not tenant.is_active:
        raise HTTPException(404, "Tenant not found")

    # Verify user belongs to tenant
    membership = session.exec(
        select(TenantUser).where(
            TenantUser.tenant_id == tenant.id,
            TenantUser.user_id == current_user.id
        )
    ).first()

    if not membership:
        raise HTTPException(403, "Not a member of this tenant")

    current_tenant.set(tenant)
    return tenant

def require_tenant_role(required_roles: list[str]):
    """Decorator/dependency for role-based access."""
    def dependency(
        tenant: Tenant = Depends(get_tenant_from_header),
        current_user = Depends(get_current_user),
        session: Session = Depends(get_session)
    ):
        membership = session.exec(
            select(TenantUser).where(
                TenantUser.tenant_id == tenant.id,
                TenantUser.user_id == current_user.id
            )
        ).first()

        if membership.role not in required_roles:
            raise HTTPException(403, f"Requires role: {required_roles}")

        return membership

    return Depends(dependency)
```

### Tenant-Scoped Queries

```python
# app/crud/project.py
from sqlmodel import Session, select
from app.models.tenant import Project, Tenant

def get_projects(session: Session, tenant: Tenant) -> list[Project]:
    """Get all projects for a tenant."""
    return session.exec(
        select(Project).where(Project.tenant_id == tenant.id)
    ).all()

def create_project(
    session: Session,
    tenant: Tenant,
    name: str
) -> Project:
    """Create project scoped to tenant."""
    project = Project(tenant_id=tenant.id, name=name)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project

def get_project(
    session: Session,
    tenant: Tenant,
    project_id: int
) -> Project | None:
    """Get project with tenant validation."""
    project = session.get(Project, project_id)
    if project and project.tenant_id != tenant.id:
        return None  # Don't leak existence
    return project
```

---

## Audit Trail System

Track all changes to models with before/after values.

### Models

```python
# app/models/audit.py
from datetime import datetime
from enum import Enum
from sqlmodel import Field, SQLModel, Column
from sqlalchemy import JSON

class AuditAction(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"

class AuditLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    table_name: str = Field(max_length=100, index=True)
    record_id: int = Field(index=True)
    action: AuditAction
    user_id: int | None = Field(default=None, foreign_key="user.id", index=True)
    old_values: dict | None = Field(default=None, sa_column=Column(JSON))
    new_values: dict | None = Field(default=None, sa_column=Column(JSON))
    ip_address: str | None = Field(default=None, max_length=45)
    user_agent: str | None = Field(default=None, max_length=500)
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### Audit Mixin

```python
# app/models/mixins.py
from sqlalchemy import event
from sqlmodel import Session
from app.models.audit import AuditLog, AuditAction
from app.core.context import get_current_user_id, get_request_info

class AuditMixin:
    """Mixin to add audit logging to models."""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if hasattr(cls, '__tablename__'):
            event.listen(cls, 'after_insert', cls._audit_insert)
            event.listen(cls, 'after_update', cls._audit_update)
            event.listen(cls, 'after_delete', cls._audit_delete)

    @classmethod
    def _audit_insert(cls, mapper, connection, target):
        cls._create_audit_log(
            connection,
            target,
            AuditAction.CREATE,
            old_values=None,
            new_values=target.model_dump()
        )

    @classmethod
    def _audit_update(cls, mapper, connection, target):
        # Get old values from history
        state = target.__dict__.get('_sa_instance_state')
        old_values = {}
        new_values = {}

        for attr in state.mapper.columns:
            hist = state.get_history(attr.key, True)
            if hist.has_changes():
                old_values[attr.key] = hist.deleted[0] if hist.deleted else None
                new_values[attr.key] = hist.added[0] if hist.added else None

        if new_values:
            cls._create_audit_log(
                connection,
                target,
                AuditAction.UPDATE,
                old_values=old_values,
                new_values=new_values
            )

    @classmethod
    def _audit_delete(cls, mapper, connection, target):
        cls._create_audit_log(
            connection,
            target,
            AuditAction.DELETE,
            old_values=target.model_dump(),
            new_values=None
        )

    @classmethod
    def _create_audit_log(cls, connection, target, action, old_values, new_values):
        user_id, ip_address, user_agent = get_request_info()

        connection.execute(
            AuditLog.__table__.insert().values(
                table_name=cls.__tablename__,
                record_id=target.id,
                action=action,
                user_id=user_id,
                old_values=old_values,
                new_values=new_values,
                ip_address=ip_address,
                user_agent=user_agent,
            )
        )
```

### Usage

```python
# app/models/document.py
from app.models.mixins import AuditMixin

class Document(SQLModel, AuditMixin, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str
    content: str
    # All changes will be automatically logged
```

---

## Async High-Performance API

Optimized async patterns for high throughput.

### Setup

```python
# app/database.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlmodel import SQLModel

async_engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=50,
    max_overflow=100,
    pool_pre_ping=True,
    pool_recycle=3600,
)

async_session_factory = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

async def get_async_session():
    async with async_session_factory() as session:
        yield session
```

### Optimized CRUD

```python
# app/crud/async_crud.py
from sqlmodel import select
from sqlalchemy import func
from sqlalchemy.orm import selectinload

async def get_users_with_orders(
    session: AsyncSession,
    skip: int = 0,
    limit: int = 100
) -> tuple[list[User], int]:
    """Fetch users with orders in single query + count."""

    # Count query
    count_result = await session.execute(select(func.count(User.id)))
    total = count_result.scalar_one()

    # Data query with eager loading
    result = await session.execute(
        select(User)
        .options(selectinload(User.orders))
        .offset(skip)
        .limit(limit)
    )
    users = result.scalars().all()

    return users, total

async def bulk_create_users(
    session: AsyncSession,
    users_data: list[UserCreate]
) -> list[User]:
    """Bulk insert with single commit."""
    users = [User.model_validate(data) for data in users_data]
    session.add_all(users)
    await session.commit()

    # Refresh all
    for user in users:
        await session.refresh(user)

    return users

async def update_user_status(
    session: AsyncSession,
    user_ids: list[int],
    is_active: bool
) -> int:
    """Bulk update without loading objects."""
    from sqlalchemy import update

    result = await session.execute(
        update(User)
        .where(User.id.in_(user_ids))
        .values(is_active=is_active, updated_at=datetime.utcnow())
    )
    await session.commit()
    return result.rowcount
```

### Async Endpoints

```python
# app/routers/async_users.py
from fastapi import APIRouter, Depends, Query
from app.database import get_async_session
from app.crud.async_crud import get_users_with_orders, bulk_create_users

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/")
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    session: AsyncSession = Depends(get_async_session)
):
    users, total = await get_users_with_orders(session, skip, limit)
    return {
        "data": users,
        "total": total,
        "skip": skip,
        "limit": limit,
    }

@router.post("/bulk", status_code=201)
async def create_users_bulk(
    users: list[UserCreate],
    session: AsyncSession = Depends(get_async_session)
):
    if len(users) > 1000:
        raise HTTPException(400, "Maximum 1000 users per request")
    created = await bulk_create_users(session, users)
    return {"created": len(created)}
```

---

## Read Replica Pattern

Route reads to replica, writes to primary.

```python
# app/database.py
from enum import Enum
from contextvars import ContextVar
from sqlmodel import Session, create_engine

class DatabaseRole(str, Enum):
    PRIMARY = "primary"
    REPLICA = "replica"

current_db_role: ContextVar[DatabaseRole] = ContextVar(
    "current_db_role", default=DatabaseRole.REPLICA
)

primary_engine = create_engine(
    "postgresql://user:pass@primary-host/db",
    pool_size=20,
    max_overflow=10,
)

replica_engine = create_engine(
    "postgresql://user:pass@replica-host/db",
    pool_size=50,
    max_overflow=20,
)

def get_session() -> Session:
    """Get session based on current context."""
    role = current_db_role.get()
    engine = primary_engine if role == DatabaseRole.PRIMARY else replica_engine
    return Session(engine)

def use_primary():
    """Context manager for write operations."""
    token = current_db_role.set(DatabaseRole.PRIMARY)
    try:
        yield
    finally:
        current_db_role.reset(token)
```

### Usage in Endpoints

```python
from app.database import get_session, use_primary, DatabaseRole, current_db_role

@router.get("/users/")
def list_users(session: Session = Depends(get_session)):
    # Automatically uses replica
    return session.exec(select(User)).all()

@router.post("/users/")
def create_user(user: UserCreate):
    # Force primary for writes
    current_db_role.set(DatabaseRole.PRIMARY)
    with Session(primary_engine) as session:
        db_user = User.model_validate(user)
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
        return db_user
```

---

## Soft Delete Pattern

Mark records as deleted instead of removing.

### Mixin

```python
# app/models/mixins.py
from datetime import datetime
from sqlmodel import Field, SQLModel

class SoftDeleteMixin(SQLModel):
    deleted_at: datetime | None = Field(default=None, index=True)
    deleted_by: int | None = Field(default=None, foreign_key="user.id")

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def soft_delete(self, user_id: int | None = None):
        self.deleted_at = datetime.utcnow()
        self.deleted_by = user_id

    def restore(self):
        self.deleted_at = None
        self.deleted_by = None
```

### Model Usage

```python
class Document(SoftDeleteMixin, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str
    content: str
```

### Filtered Queries

```python
def get_active_documents(session: Session) -> list[Document]:
    """Get only non-deleted documents."""
    return session.exec(
        select(Document).where(Document.deleted_at == None)
    ).all()

def get_all_documents(session: Session, include_deleted: bool = False) -> list[Document]:
    """Get documents with optional deleted."""
    statement = select(Document)
    if not include_deleted:
        statement = statement.where(Document.deleted_at == None)
    return session.exec(statement).all()

def delete_document(session: Session, doc_id: int, user_id: int) -> Document | None:
    """Soft delete a document."""
    doc = session.get(Document, doc_id)
    if doc and not doc.is_deleted:
        doc.soft_delete(user_id)
        session.add(doc)
        session.commit()
        session.refresh(doc)
    return doc

def restore_document(session: Session, doc_id: int) -> Document | None:
    """Restore a soft-deleted document."""
    doc = session.get(Document, doc_id)
    if doc and doc.is_deleted:
        doc.restore()
        session.add(doc)
        session.commit()
        session.refresh(doc)
    return doc
```

---

## Full-Text Search

PostgreSQL full-text search integration.

### Model with Search Vector

```python
# app/models/article.py
from sqlalchemy import Column, Index
from sqlalchemy.dialects.postgresql import TSVECTOR

class Article(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str = Field(max_length=200)
    content: str
    search_vector: str | None = Field(
        default=None,
        sa_column=Column(TSVECTOR)
    )

    __table_args__ = (
        Index(
            'ix_article_search_vector',
            'search_vector',
            postgresql_using='gin'
        ),
    )
```

### Search Trigger (Alembic Migration)

```python
# migrations/versions/xxxx_add_search_trigger.py
from alembic import op

def upgrade():
    # Create function to update search vector
    op.execute("""
        CREATE OR REPLACE FUNCTION article_search_vector_update() RETURNS trigger AS $$
        BEGIN
            NEW.search_vector := to_tsvector('english', COALESCE(NEW.title, '') || ' ' || COALESCE(NEW.content, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Create trigger
    op.execute("""
        CREATE TRIGGER article_search_vector_trigger
        BEFORE INSERT OR UPDATE ON article
        FOR EACH ROW EXECUTE FUNCTION article_search_vector_update();
    """)

    # Update existing rows
    op.execute("""
        UPDATE article SET search_vector = to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content, ''));
    """)

def downgrade():
    op.execute("DROP TRIGGER IF EXISTS article_search_vector_trigger ON article;")
    op.execute("DROP FUNCTION IF EXISTS article_search_vector_update;")
```

### Search Queries

```python
# app/crud/search.py
from sqlalchemy import func, text

def search_articles(
    session: Session,
    query: str,
    limit: int = 20
) -> list[Article]:
    """Full-text search with ranking."""
    search_query = func.plainto_tsquery('english', query)

    statement = (
        select(Article)
        .where(Article.search_vector.op('@@')(search_query))
        .order_by(
            func.ts_rank(Article.search_vector, search_query).desc()
        )
        .limit(limit)
    )

    return session.exec(statement).all()

def search_articles_with_highlight(
    session: Session,
    query: str,
    limit: int = 20
) -> list[dict]:
    """Search with highlighted snippets."""
    result = session.execute(text("""
        SELECT
            id,
            title,
            ts_headline('english', content, plainto_tsquery('english', :query),
                'StartSel=<mark>, StopSel=</mark>, MaxWords=50') as snippet,
            ts_rank(search_vector, plainto_tsquery('english', :query)) as rank
        FROM article
        WHERE search_vector @@ plainto_tsquery('english', :query)
        ORDER BY rank DESC
        LIMIT :limit
    """), {"query": query, "limit": limit})

    return [
        {"id": row.id, "title": row.title, "snippet": row.snippet, "rank": row.rank}
        for row in result
    ]
```

---

## Testing Patterns

### Comprehensive Test Setup

```python
# tests/conftest.py
import pytest
from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_session

@pytest.fixture(name="engine", scope="session")
def engine_fixture():
    """Create test engine once per session."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine

@pytest.fixture(name="session")
def session_fixture(engine):
    """Create fresh session per test with rollback."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(name="client")
def client_fixture(session):
    """Create test client with session override."""
    def override_get_session():
        return session

    app.dependency_overrides[get_session] = override_get_session

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()

@pytest.fixture
def sample_user(session):
    """Create sample user for tests."""
    user = User(
        email="test@example.com",
        full_name="Test User",
        hashed_password="hashed_password"
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user

@pytest.fixture
def auth_headers(sample_user):
    """Generate auth headers for authenticated requests."""
    from app.core.security import create_access_token
    token = create_access_token(sample_user.id)
    return {"Authorization": f"Bearer {token}"}
```

### Test Examples

```python
# tests/test_users.py
import pytest

def test_create_user(client):
    response = client.post(
        "/auth/register",
        json={
            "email": "new@example.com",
            "full_name": "New User",
            "password": "SecurePass123!"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "new@example.com"
    assert "hashed_password" not in data

def test_create_user_duplicate_email(client, sample_user):
    response = client.post(
        "/auth/register",
        json={
            "email": sample_user.email,
            "full_name": "Another User",
            "password": "SecurePass123!"
        }
    )
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"]

def test_login_success(client, sample_user, session):
    # Update with real hashed password
    from app.core.security import hash_password
    sample_user.hashed_password = hash_password("TestPass123!")
    session.add(sample_user)
    session.commit()

    response = client.post(
        "/auth/login",
        data={"username": sample_user.email, "password": "TestPass123!"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_get_me_authenticated(client, auth_headers):
    response = client.get("/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"

def test_get_me_unauthenticated(client):
    response = client.get("/auth/me")
    assert response.status_code == 401
```
