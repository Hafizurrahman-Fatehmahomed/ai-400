# FastAPI Main Application Template
# Copy to: app/main.py

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import SQLModel

from app.config import settings
from app.database import engine

# Import routers
# from app.routers import auth, users, items

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(f"Starting {settings.app_name}...")
    SQLModel.metadata.create_all(engine)
    logger.info("Database tables created")
    yield
    # Shutdown
    logger.info(f"Shutting down {settings.app_name}...")
    engine.dispose()
    logger.info("Database connections closed")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.app_name,
        description="A production-ready FastAPI application",
        version="1.0.0",
        lifespan=lifespan,
        # Disable docs in production if needed
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    # Add middleware (order matters: first added = outermost)
    configure_middleware(app)

    # Add exception handlers
    configure_exception_handlers(app)

    # Include routers
    configure_routes(app)

    return app


def configure_middleware(app: FastAPI) -> None:
    """Configure application middleware."""

    # CORS - must be added before other middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        import time
        import uuid

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.time()

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
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        return response


def configure_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers."""

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.debug else None
            }
        )


def configure_routes(app: FastAPI) -> None:
    """Configure application routes."""

    # Health check endpoints
    @app.get("/health", tags=["health"])
    def health_check():
        return {"status": "healthy", "environment": settings.environment}

    @app.get("/ready", tags=["health"])
    def readiness_check():
        # Add database connectivity check here
        return {"status": "ready"}

    @app.get("/live", tags=["health"])
    def liveness_check():
        return {"status": "alive"}

    # Include API routers
    # app.include_router(auth.router, prefix=settings.api_prefix, tags=["auth"])
    # app.include_router(users.router, prefix=settings.api_prefix, tags=["users"])
    # app.include_router(items.router, prefix=settings.api_prefix, tags=["items"])


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
    )
