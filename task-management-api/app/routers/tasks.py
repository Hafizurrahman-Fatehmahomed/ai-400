from fastapi import APIRouter, HTTPException, status
from sqlmodel import select

from app.database import SessionDep
from app.models import Task, TaskCreate, TaskUpdate, TaskPublic

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    responses={404: {"description": "Task not found"}},
)


@router.get("/", response_model=list[TaskPublic])
def get_tasks(session: SessionDep) -> list[Task]:
    tasks = session.exec(select(Task)).all()
    return tasks


@router.get("/{task_id}", response_model=TaskPublic)
def get_task(task_id: int, session: SessionDep) -> Task:
    task = session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/", response_model=TaskPublic, status_code=status.HTTP_201_CREATED)
def create_task(task: TaskCreate, session: SessionDep) -> Task:
    db_task = Task.model_validate(task)
    session.add(db_task)
    session.commit()
    session.refresh(db_task)
    return db_task


@router.put("/{task_id}", response_model=TaskPublic)
def update_task(task_id: int, task_data: TaskUpdate, session: SessionDep) -> Task:
    task = session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    update_data = task_data.model_dump(exclude_unset=True)
    task.sqlmodel_update(update_data)
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: int, session: SessionDep) -> None:
    task = session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    session.delete(task)
    session.commit()
