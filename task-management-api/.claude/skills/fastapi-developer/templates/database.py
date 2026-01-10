# FastAPI Database Configuration Template
# Copy to: app/database.py

from typing import Annotated, Generator

from fastapi import Depends
from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy.pool import QueuePool

from app.config import settings


# Create database engine
engine = create_engine(
    settings.database_url,
    echo=settings.db_echo,  # Log SQL queries (disable in production)
    poolclass=QueuePool,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections after 1 hour
)


def get_session() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session.

    Usage:
        @router.get("/items")
        def get_items(session: SessionDep):
            return session.exec(select(Item)).all()
    """
    with Session(engine) as session:
        yield session


# Type alias for dependency injection
SessionDep = Annotated[Session, Depends(get_session)]


def get_transactional_session() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session with automatic
    commit on success and rollback on error.

    Usage:
        @router.post("/items")
        def create_item(item: Item, session: Session = Depends(get_transactional_session)):
            session.add(item)
            # Commits automatically if no exception
    """
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def init_db() -> None:
    """
    Initialize database tables.

    Call this during application startup.
    """
    SQLModel.metadata.create_all(engine)


def close_db() -> None:
    """
    Close database connections.

    Call this during application shutdown.
    """
    engine.dispose()


# ============================================================================
# Async Database Configuration (Optional)
# Uncomment if you need async database operations
# ============================================================================

# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
#
# # Convert sync URL to async URL
# async_database_url = settings.database_url.replace(
#     "postgresql://", "postgresql+asyncpg://"
# )
#
# async_engine = create_async_engine(
#     async_database_url,
#     echo=settings.db_echo,
#     pool_size=settings.db_pool_size,
#     max_overflow=settings.db_max_overflow,
# )
#
# AsyncSessionLocal = async_sessionmaker(
#     async_engine,
#     class_=AsyncSession,
#     expire_on_commit=False,
# )
#
# async def get_async_session() -> AsyncSession:
#     async with AsyncSessionLocal() as session:
#         yield session
#
# AsyncSessionDep = Annotated[AsyncSession, Depends(get_async_session)]
