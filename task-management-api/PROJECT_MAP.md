# Project File Map

## Directory Structure

```
app/
├── main.py           # App entry point, creates FastAPI instance
├── config.py         # Environment variables (DATABASE_URL, etc.)
├── database.py       # DB connection, session management, startup/shutdown
│
├── models/
│   ├── __init__.py   # Exports all models for easy imports
│   ├── task.py       # Task table + validation schemas
│   └── user.py       # User table + validation schemas
│
└── routers/
    ├── __init__.py   # Exports all routers
    └── tasks.py      # /tasks/ endpoints (CRUD operations)

tests/                # Test files

Root files:
├── .env.local        # Secrets (DB password, API keys) - NOT committed
├── .env.example      # Template for .env.local
├── pyproject.toml    # Dependencies + project config
├── CLAUDE.md         # Project docs for AI assistants
└── README.md         # Project docs for humans
```

## Quick Mental Model

| Folder/File | Purpose |
|-------------|---------|
| `models/` | **What** - Data structures and validation |
| `routers/` | **How** - API endpoints and logic |
| `config.py` | **Where** - Connections and settings |
| `main.py` | **Glue** - Ties everything together |
