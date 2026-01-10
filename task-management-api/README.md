# Task Management API

A FastAPI-based REST API for task management with PostgreSQL backend.

## Demo

Watch the project in action: [YouTube Demo](https://www.youtube.com/watch?v=kdlMooUwnfY)

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
│   │   └── task.py      # Task model
│   └── routers/
│       └── tasks.py     # Task CRUD endpoints
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- PostgreSQL database

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd task-management-api
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env.local
   ```

4. Edit `.env.local` with your database credentials:
   ```
   DATABASE_URL=postgresql://user:password@localhost:5432/taskdb
   GEMINI_API_KEY=your_key_here  # optional
   ```

### Running the Server

**Development mode** (with auto-reload):
```bash
uv run python -m uvicorn app.main:app --reload
```

The server will start at `http://localhost:8000`

**Production mode** (without auto-reload):
```bash
uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Stopping the server**:

Press `Ctrl + C` in the terminal where the server is running.

## Testing the API

### Interactive API Documentation

FastAPI provides built-in interactive documentation. Once the server is running, open your browser and navigate to:

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | **Swagger UI** - Interactive API documentation where you can test all endpoints directly |
| http://localhost:8000/redoc | **ReDoc** - Alternative API documentation with a clean interface |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tasks/` | List all tasks |
| GET | `/tasks/{id}` | Get a single task |
| POST | `/tasks/` | Create a new task |
| PUT | `/tasks/{id}` | Update a task |
| DELETE | `/tasks/{id}` | Delete a task |
| GET | `/health` | Health check |

### Quick Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Create a task
curl -X POST http://localhost:8000/tasks/ \
  -H "Content-Type: application/json" \
  -d '{"title": "My First Task", "description": "This is a test task"}'

# List all tasks
curl http://localhost:8000/tasks/

# Get a specific task
curl http://localhost:8000/tasks/1

# Update a task
curl -X PUT http://localhost:8000/tasks/1 \
  -H "Content-Type: application/json" \
  -d '{"title": "Updated Task", "description": "Updated description"}'

# Delete a task
curl -X DELETE http://localhost:8000/tasks/1
```

## Database Model

**Task**
| Field | Type | Description |
|-------|------|-------------|
| id | Integer | Primary key, auto-generated |
| title | String | Task title (required) |
| description | String | Task description (optional) |

## License

MIT
