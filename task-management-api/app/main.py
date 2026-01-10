from fastapi import FastAPI

from app.config import get_settings
from app.database import lifespan
from app.routers import tasks

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

app.include_router(tasks.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
