# Task Management API

FastAPI-based REST API for task management with PostgreSQL backend.

## Tech Stack

- **Python 3.13+** with uv package manager
- **FastAPI** - Web framework
- **SQLModel** - ORM (SQLAlchemy + Pydantic)
- **PostgreSQL** - Database (psycopg2-binary driver)

## Project Structure

```
task-management-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app entry point
│   ├── config.py        # Settings and environment loading
│   ├── database.py      # Engine, session, lifespan setup
│   ├── models/
│   │   ├── __init__.py
│   │   └── task.py      # Task model
│   └── routers/
│       ├── __init__.py
│       └── tasks.py     # Task CRUD endpoints
├── tests/
│   └── __init__.py
├── .env.example
├── .gitignore
├── CLAUDE.md
├── pyproject.toml
└── README.md
```

## Commands

```bash
# Install dependencies
uv sync

# Run development server
uv run fastapi dev app/main.py

# Run production server
uv run fastapi run app/main.py
```

## Environment Variables

Create `.env.local` with:
```
DATABASE_URL=postgresql://user:password@localhost:5432/taskdb
GEMINI_API_KEY=your_key_here  # optional
```

## Database Model

**Task**: `id` (int, PK), `title` (str), `description` (str, optional)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /tasks/ | List all tasks |
| GET | /tasks/{id} | Get single task |
| POST | /tasks/ | Create a task |
| PUT | /tasks/{id} | Update a task |
| DELETE | /tasks/{id} | Delete a task |
| GET | /health | Health check |
