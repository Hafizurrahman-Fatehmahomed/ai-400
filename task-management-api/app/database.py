from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from sqlmodel import SQLModel, Session, create_engine

from app.config import get_settings

settings = get_settings()
engine = create_engine(settings.database_url, echo=True)


def create_tables():
    SQLModel.metadata.create_all(engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield
    engine.dispose()


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
