# SQLModel Reference

Comprehensive API reference for SQLModel, SQLAlchemy, and Pydantic integration patterns.

## Table of Contents

1. [Field Configuration](#field-configuration)
2. [Connection Pooling](#connection-pooling)
3. [Async Configuration](#async-configuration)
4. [Query Patterns](#query-patterns)
5. [Relationship Configuration](#relationship-configuration)
6. [Validation Patterns](#validation-patterns)
7. [Transaction Management](#transaction-management)
8. [Error Handling](#error-handling)
9. [Performance Optimization](#performance-optimization)
10. [Multi-Database Patterns](#multi-database-patterns)

---

## Field Configuration

### Field Parameters

```python
from sqlmodel import Field

Field(
    default=None,              # Default value
    default_factory=list,      # Factory for mutable defaults
    primary_key=True,          # Mark as primary key
    foreign_key="table.column",# Foreign key reference
    unique=True,               # Unique constraint
    index=True,                # Create index
    nullable=True,             # Allow NULL (inferred from type)
    sa_column=Column(...),     # Raw SQLAlchemy column
    max_length=255,            # String max length
    min_length=1,              # String min length
    ge=0,                      # Greater than or equal (numbers)
    le=100,                    # Less than or equal (numbers)
    gt=0,                      # Greater than (numbers)
    lt=100,                    # Less than (numbers)
    regex=r"^[a-z]+$",         # Regex pattern (strings)
    description="Field desc",  # OpenAPI description
    title="Field Title",       # OpenAPI title
    examples=["value1"],       # OpenAPI examples
)
```

### Common Field Types

```python
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from uuid import UUID
from typing import Optional

class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

class Example(SQLModel, table=True):
    # Primary Keys
    id: int | None = Field(default=None, primary_key=True)
    uuid: UUID = Field(default_factory=uuid4, primary_key=True)

    # Strings
    name: str = Field(max_length=100, index=True)
    description: str | None = Field(default=None, max_length=1000)
    email: str = Field(unique=True, max_length=255)

    # Numbers
    count: int = Field(default=0, ge=0)
    price: Decimal = Field(max_digits=10, decimal_places=2)
    rating: float = Field(ge=0.0, le=5.0)

    # Dates/Times
    created_at: datetime = Field(default_factory=datetime.utcnow)
    birth_date: date | None = None
    start_time: time | None = None

    # Enums
    status: Status = Field(default=Status.ACTIVE)

    # Boolean
    is_active: bool = Field(default=True)

    # JSON (requires sa_column)
    metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
```

### Custom Column Types

```python
from sqlalchemy import Column, Text, LargeBinary, ARRAY, JSON
from sqlalchemy.dialects.postgresql import JSONB, INET, CIDR, UUID as PG_UUID

class Advanced(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    # Large text
    content: str = Field(sa_column=Column(Text))

    # Binary data
    file_data: bytes = Field(sa_column=Column(LargeBinary))

    # PostgreSQL-specific
    tags: list[str] = Field(sa_column=Column(ARRAY(String)))
    data: dict = Field(sa_column=Column(JSONB))
    ip_address: str = Field(sa_column=Column(INET))
```

---

## Connection Pooling

### QueuePool (Default)

```python
from sqlmodel import create_engine

engine = create_engine(
    "postgresql+psycopg2://user:pass@localhost/db",

    # Pool Size
    pool_size=5,           # Persistent connections (default: 5)
    max_overflow=10,       # Temporary connections above pool_size (default: 10)

    # Timeouts
    pool_timeout=30,       # Wait time for connection (default: 30s)
    pool_recycle=1800,     # Recycle connections after N seconds (default: -1/disabled)

    # Health Checks
    pool_pre_ping=True,    # Test connection before use (default: False)

    # Connection Order
    pool_use_lifo=True,    # Use LIFO for better idle timeout handling

    # Debugging
    echo=False,            # Log SQL statements (default: False)
    echo_pool="debug",     # Log pool events: False, True, "debug"
)
```

### Pool Size Guidelines

| Application Type | pool_size | max_overflow | Notes |
|------------------|-----------|--------------|-------|
| Small app (< 10 req/s) | 5 | 5 | Default works |
| Medium app (10-100 req/s) | 10-20 | 10 | Monitor connections |
| Large app (100+ req/s) | 20-50 | 20 | Consider PGBouncer |
| Worker/Background | 2-5 | 2 | Limited concurrency |

### NullPool (External Pooler)

```python
from sqlalchemy.pool import NullPool

# Use with PGBouncer, pgpool, or other external poolers
engine = create_engine(
    "postgresql+psycopg2://user:pass@pgbouncer:6432/db",
    poolclass=NullPool,  # Disable internal pooling
)
```

### StaticPool (Testing)

```python
from sqlalchemy.pool import StaticPool

# Single connection for in-memory databases
engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
```

### Pool Events

```python
from sqlalchemy import event

@event.listens_for(engine, "connect")
def on_connect(dbapi_connection, connection_record):
    """Called when a new connection is created."""
    cursor = dbapi_connection.cursor()
    cursor.execute("SET timezone='UTC'")
    cursor.close()

@event.listens_for(engine, "checkout")
def on_checkout(dbapi_connection, connection_record, connection_proxy):
    """Called when a connection is retrieved from pool."""
    pass

@event.listens_for(engine, "checkin")
def on_checkin(dbapi_connection, connection_record):
    """Called when a connection is returned to pool."""
    pass
```

---

## Async Configuration

### Async Engine Setup

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlmodel import SQLModel

# PostgreSQL with asyncpg
async_engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

# MySQL with aiomysql
async_engine = create_async_engine(
    "mysql+aiomysql://user:pass@localhost/db"
)

# SQLite with aiosqlite
async_engine = create_async_engine(
    "sqlite+aiosqlite:///./database.db"
)
```

### Async Session Factory

```python
async_session_factory = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Important: prevents lazy-load issues
    autocommit=False,
    autoflush=False,
)
```

### FastAPI Async Dependency

```python
from fastapi import Depends

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()

@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    session: AsyncSession = Depends(get_async_session)
) -> User:
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    return user
```

### Async CRUD Operations

```python
from sqlmodel import select
from sqlalchemy import func

async def create(session: AsyncSession, data: CreateSchema) -> Model:
    obj = Model.model_validate(data)
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return obj

async def get_by_id(session: AsyncSession, id: int) -> Model | None:
    result = await session.execute(select(Model).where(Model.id == id))
    return result.scalar_one_or_none()

async def get_all(
    session: AsyncSession,
    skip: int = 0,
    limit: int = 100
) -> list[Model]:
    result = await session.execute(
        select(Model).offset(skip).limit(limit)
    )
    return result.scalars().all()

async def update(
    session: AsyncSession,
    id: int,
    data: UpdateSchema
) -> Model | None:
    obj = await get_by_id(session, id)
    if not obj:
        return None
    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(obj, key, value)
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return obj

async def delete(session: AsyncSession, id: int) -> bool:
    obj = await get_by_id(session, id)
    if not obj:
        return False
    await session.delete(obj)
    await session.commit()
    return True

async def count(session: AsyncSession) -> int:
    result = await session.execute(select(func.count(Model.id)))
    return result.scalar_one()
```

### Async Relationship Loading

```python
from sqlalchemy.orm import selectinload, joinedload

# Eager load relationships in async
async def get_user_with_orders(session: AsyncSession, user_id: int) -> User:
    result = await session.execute(
        select(User)
        .options(selectinload(User.orders))  # For collections
        .where(User.id == user_id)
    )
    return result.scalar_one_or_none()

# For nested relationships
result = await session.execute(
    select(User)
    .options(
        selectinload(User.orders).selectinload(Order.items)
    )
    .where(User.id == user_id)
)
```

---

## Query Patterns

### Basic Queries

```python
from sqlmodel import select, col, or_, and_, not_

# Select all
statement = select(User)
users = session.exec(statement).all()

# Select with conditions
statement = select(User).where(User.is_active == True)
statement = select(User).where(User.age >= 18)
statement = select(User).where(col(User.name).contains("john"))

# Multiple conditions (AND)
statement = select(User).where(
    User.is_active == True,
    User.age >= 18
)

# OR conditions
statement = select(User).where(
    or_(User.role == "admin", User.role == "moderator")
)

# NOT
statement = select(User).where(not_(User.is_banned))

# NULL checks
statement = select(User).where(User.deleted_at == None)
statement = select(User).where(User.deleted_at != None)

# IN clause
statement = select(User).where(User.id.in_([1, 2, 3]))

# BETWEEN
statement = select(User).where(User.age.between(18, 65))

# LIKE / ILIKE
statement = select(User).where(User.name.like("%john%"))
statement = select(User).where(User.name.ilike("%john%"))  # Case-insensitive
```

### Ordering and Pagination

```python
from sqlmodel import select, desc, asc

# Order by single column
statement = select(User).order_by(User.created_at)
statement = select(User).order_by(desc(User.created_at))

# Order by multiple columns
statement = select(User).order_by(User.last_name, User.first_name)

# Pagination
statement = select(User).offset(20).limit(10)

# Combined
statement = (
    select(User)
    .where(User.is_active == True)
    .order_by(desc(User.created_at))
    .offset(0)
    .limit(20)
)
```

### Aggregations

```python
from sqlalchemy import func

# Count
statement = select(func.count(User.id))
count = session.exec(statement).one()

# Sum, Avg, Min, Max
statement = select(func.sum(Order.total))
statement = select(func.avg(Product.price))
statement = select(func.min(User.age), func.max(User.age))

# Group By
statement = (
    select(User.role, func.count(User.id))
    .group_by(User.role)
)

# Group By with Having
statement = (
    select(User.role, func.count(User.id).label("count"))
    .group_by(User.role)
    .having(func.count(User.id) > 5)
)
```

### Joins

```python
from sqlmodel import select

# Implicit join (with relationship)
statement = (
    select(User, Team)
    .where(User.team_id == Team.id)
)

# Explicit join
statement = (
    select(User)
    .join(Team)
    .where(Team.name == "Engineering")
)

# Left outer join
statement = (
    select(User, Team)
    .join(Team, isouter=True)
)

# Join with specific condition
statement = (
    select(User, Team)
    .join(Team, User.team_id == Team.id)
)

# Multiple joins
statement = (
    select(User)
    .join(Team)
    .join(Department, Team.department_id == Department.id)
)
```

### Subqueries

```python
from sqlmodel import select

# Subquery for filtering
subquery = (
    select(Order.user_id)
    .where(Order.total > 1000)
    .distinct()
    .subquery()
)
statement = select(User).where(User.id.in_(select(subquery)))

# Scalar subquery
order_count = (
    select(func.count(Order.id))
    .where(Order.user_id == User.id)
    .correlate(User)
    .scalar_subquery()
)
statement = select(User, order_count.label("order_count"))
```

### Raw SQL (Use Carefully)

```python
from sqlalchemy import text

# Parameterized raw SQL (SAFE)
statement = text("SELECT * FROM users WHERE email = :email")
result = session.exec(statement.params(email=user_email))

# With model mapping
statement = text("SELECT * FROM user WHERE role = :role")
result = session.exec(statement.params(role="admin"))
users = [User.model_validate(dict(row._mapping)) for row in result]

# Execute raw SQL
session.exec(text("UPDATE users SET is_active = false WHERE last_login < :date").params(
    date=datetime.utcnow() - timedelta(days=365)
))
session.commit()
```

---

## Relationship Configuration

### One-to-One

```python
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    profile: "Profile" = Relationship(
        back_populates="user",
        sa_relationship_kwargs={"uselist": False}
    )

class Profile(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", unique=True)
    user: User = Relationship(back_populates="profile")
```

### One-to-Many

```python
class Parent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    children: list["Child"] = Relationship(back_populates="parent")

class Child(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    parent_id: int = Field(foreign_key="parent.id")
    parent: Parent = Relationship(back_populates="children")
```

### Many-to-Many

```python
class UserRoleLink(SQLModel, table=True):
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    role_id: int = Field(foreign_key="role.id", primary_key=True)
    assigned_at: datetime = Field(default_factory=datetime.utcnow)

class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    roles: list["Role"] = Relationship(
        back_populates="users",
        link_model=UserRoleLink
    )

class Role(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    users: list["User"] = Relationship(
        back_populates="roles",
        link_model=UserRoleLink
    )
```

### Self-Referential

```python
class Category(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    parent_id: int | None = Field(default=None, foreign_key="category.id")

    parent: "Category | None" = Relationship(
        back_populates="children",
        sa_relationship_kwargs={"remote_side": "Category.id"}
    )
    children: list["Category"] = Relationship(back_populates="parent")
```

### Cascade Delete

```python
class Parent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    children: list["Child"] = Relationship(
        back_populates="parent",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "passive_deletes": True,
        }
    )

class Child(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    parent_id: int = Field(
        foreign_key="parent.id",
        ondelete="CASCADE"  # Database-level cascade
    )
    parent: Parent = Relationship(back_populates="children")
```

### Lazy Loading Strategies

```python
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    # Lazy load (default) - loads on access
    orders: list["Order"] = Relationship(back_populates="user")

    # Eager load - always loads with parent
    profile: "Profile" = Relationship(
        back_populates="user",
        sa_relationship_kwargs={"lazy": "joined"}
    )

    # Subquery load - separate query, good for collections
    posts: list["Post"] = Relationship(
        back_populates="author",
        sa_relationship_kwargs={"lazy": "selectin"}
    )
```

---

## Validation Patterns

### Field Validators

```python
from pydantic import field_validator, ValidationInfo
from typing import Any

class UserCreate(SQLModel):
    username: str
    email: str
    password: str
    password_confirm: str

    @field_validator('username')
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError('must be alphanumeric')
        return v.lower()

    @field_validator('email')
    @classmethod
    def email_valid(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('invalid email format')
        return v.lower()

    @field_validator('password_confirm')
    @classmethod
    def passwords_match(cls, v: str, info: ValidationInfo) -> str:
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('passwords do not match')
        return v
```

### Before/After Validators

```python
from pydantic import field_validator
from typing import Any

class Product(SQLModel):
    name: str
    price: float

    @field_validator('name', mode='before')
    @classmethod
    def strip_name(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator('price', mode='after')
    @classmethod
    def round_price(cls, v: float) -> float:
        return round(v, 2)
```

### Model Validators

```python
from pydantic import model_validator
from typing_extensions import Self

class DateRange(SQLModel):
    start_date: date
    end_date: date

    @model_validator(mode='after')
    def check_dates(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError('end_date must be >= start_date')
        return self

class User(SQLModel):
    first_name: str | None = None
    last_name: str | None = None
    full_name: str | None = None

    @model_validator(mode='before')
    @classmethod
    def set_full_name(cls, data: dict) -> dict:
        if isinstance(data, dict):
            if not data.get('full_name'):
                first = data.get('first_name', '')
                last = data.get('last_name', '')
                data['full_name'] = f"{first} {last}".strip()
        return data
```

### Custom Types

```python
from typing import Annotated
from pydantic import AfterValidator, BeforeValidator

def validate_phone(v: str) -> str:
    cleaned = ''.join(c for c in v if c.isdigit())
    if len(cleaned) != 10:
        raise ValueError('Phone must be 10 digits')
    return cleaned

PhoneNumber = Annotated[str, AfterValidator(validate_phone)]

class Contact(SQLModel):
    phone: PhoneNumber
```

---

## Transaction Management

### Basic Transactions

```python
from sqlmodel import Session

# Automatic commit with context manager
with Session(engine) as session:
    user = User(name="John")
    session.add(user)
    session.commit()
    # Auto-close on exit

# Explicit transaction control
with Session(engine) as session:
    try:
        session.add(user1)
        session.add(user2)
        session.commit()
    except Exception:
        session.rollback()
        raise
```

### Nested Transactions (Savepoints)

```python
with Session(engine) as session:
    session.add(user)

    # Create savepoint
    with session.begin_nested():
        try:
            session.add(risky_operation)
            # If this fails, only this block rolls back
        except Exception:
            pass  # Savepoint rolled back, outer transaction intact

    session.commit()  # Commits user, not risky_operation if it failed
```

### Async Transactions

```python
async with async_session_factory() as session:
    async with session.begin():
        session.add(user)
        session.add(profile)
        # Auto-commit at end of block
        # Auto-rollback on exception
```

---

## Error Handling

### Common Exceptions

```python
from sqlalchemy.exc import (
    IntegrityError,
    OperationalError,
    NoResultFound,
    MultipleResultsFound,
)
from fastapi import HTTPException

def create_user(session: Session, user: UserCreate) -> User:
    try:
        db_user = User.model_validate(user)
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
        return db_user
    except IntegrityError as e:
        session.rollback()
        if "unique constraint" in str(e).lower():
            raise HTTPException(400, "Email already exists")
        raise HTTPException(400, "Database constraint violation")
    except OperationalError:
        session.rollback()
        raise HTTPException(503, "Database unavailable")

def get_user(session: Session, user_id: int) -> User:
    try:
        statement = select(User).where(User.id == user_id)
        return session.exec(statement).one()
    except NoResultFound:
        raise HTTPException(404, "User not found")
    except MultipleResultsFound:
        raise HTTPException(500, "Data integrity error")
```

### Custom Exception Classes

```python
class DatabaseError(Exception):
    """Base database exception."""
    pass

class RecordNotFoundError(DatabaseError):
    """Record not found in database."""
    def __init__(self, model: str, id: int):
        self.model = model
        self.id = id
        super().__init__(f"{model} with id {id} not found")

class DuplicateRecordError(DatabaseError):
    """Duplicate record constraint violation."""
    def __init__(self, field: str, value: str):
        self.field = field
        self.value = value
        super().__init__(f"Duplicate value '{value}' for field '{field}'")
```

---

## Performance Optimization

### Eager Loading

```python
from sqlalchemy.orm import selectinload, joinedload, subqueryload

# For single relationships (one-to-one, many-to-one)
statement = select(User).options(joinedload(User.profile))

# For collections (one-to-many, many-to-many)
statement = select(User).options(selectinload(User.orders))

# Nested eager loading
statement = (
    select(User)
    .options(
        selectinload(User.orders).selectinload(Order.items)
    )
)

# Multiple relationships
statement = (
    select(User)
    .options(
        joinedload(User.profile),
        selectinload(User.orders),
    )
)
```

### Bulk Operations

```python
# Bulk insert
users = [User(name=f"User {i}") for i in range(1000)]
session.add_all(users)
session.commit()

# Bulk update (more efficient)
session.execute(
    update(User)
    .where(User.is_active == False)
    .values(deleted_at=datetime.utcnow())
)
session.commit()

# Bulk delete
session.execute(
    delete(User)
    .where(User.last_login < datetime.utcnow() - timedelta(days=365))
)
session.commit()
```

### Query Optimization

```python
# Select only needed columns
statement = select(User.id, User.name, User.email)

# Use exists() for existence checks
from sqlalchemy import exists
statement = select(exists().where(User.email == email))
result = session.exec(statement).one()

# Use distinct
statement = select(User.role).distinct()

# Pagination with total count
from sqlalchemy import func

total = session.exec(select(func.count(User.id))).one()
users = session.exec(
    select(User).offset(skip).limit(limit)
).all()
```

---

## Multi-Database Patterns

### Multiple Engines

```python
# Primary database (read-write)
primary_engine = create_engine("postgresql://user:pass@primary/db")

# Read replica
replica_engine = create_engine("postgresql://user:pass@replica/db")

def get_session(readonly: bool = False):
    engine = replica_engine if readonly else primary_engine
    with Session(engine) as session:
        yield session

@app.get("/users/")
def list_users(session: Session = Depends(lambda: get_session(readonly=True))):
    return session.exec(select(User)).all()

@app.post("/users/")
def create_user(user: UserCreate, session: Session = Depends(get_session)):
    # Uses primary
    pass
```

### Database Routing

```python
from contextvars import ContextVar

current_db: ContextVar[str] = ContextVar("current_db", default="default")

engines = {
    "default": create_engine("postgresql://localhost/default"),
    "analytics": create_engine("postgresql://localhost/analytics"),
    "archive": create_engine("postgresql://localhost/archive"),
}

def get_session(db_name: str = "default"):
    engine = engines[db_name]
    with Session(engine) as session:
        yield session

@app.get("/analytics/")
def get_analytics(session: Session = Depends(lambda: get_session("analytics"))):
    pass
```

### Horizontal Sharding (Basic)

```python
def get_shard_engine(user_id: int) -> Engine:
    shard_id = user_id % 4  # 4 shards
    return engines[f"shard_{shard_id}"]

def get_user_session(user_id: int):
    engine = get_shard_engine(user_id)
    with Session(engine) as session:
        yield session
```

---

## Database URL Formats

```python
# PostgreSQL
"postgresql://user:password@localhost:5432/dbname"
"postgresql+psycopg2://user:password@localhost/dbname"
"postgresql+asyncpg://user:password@localhost/dbname"  # async

# MySQL
"mysql://user:password@localhost:3306/dbname"
"mysql+pymysql://user:password@localhost/dbname"
"mysql+aiomysql://user:password@localhost/dbname"  # async

# SQLite
"sqlite:///./database.db"  # File
"sqlite://"  # In-memory
"sqlite+aiosqlite:///./database.db"  # async

# SQL Server
"mssql+pyodbc://user:password@server/dbname?driver=ODBC+Driver+17+for+SQL+Server"

# With SSL
"postgresql://user:password@localhost/dbname?sslmode=require"
```
