from sqlmodel import SQLModel, Field


class TaskBase(SQLModel):
    """Base schema with shared fields."""

    title: str
    description: str | None = None


class Task(TaskBase, table=True):
    """Database model for tasks."""

    id: int | None = Field(default=None, primary_key=True)


class TaskCreate(TaskBase):
    """Schema for creating a task."""

    pass


class TaskUpdate(SQLModel):
    """Schema for updating a task (all fields optional)."""

    title: str | None = None
    description: str | None = None


class TaskPublic(TaskBase):
    """Schema for task responses."""

    id: int
