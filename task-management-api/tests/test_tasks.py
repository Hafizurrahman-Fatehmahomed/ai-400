from fastapi.testclient import TestClient

from app.models import Task


class TestGetTasks:
    """Tests for GET /tasks/ endpoint."""

    def test_get_tasks_empty_returns_empty_list(self, client: TestClient):
        response = client.get("/tasks/")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_tasks_returns_all_tasks(self, client: TestClient, multiple_tasks: list[Task]):
        response = client.get("/tasks/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["title"] == "Task 1"
        assert data[1]["title"] == "Task 2"
        assert data[2]["title"] == "Task 3"


class TestGetTask:
    """Tests for GET /tasks/{task_id} endpoint."""

    def test_get_task_returns_task(self, client: TestClient, sample_task: Task):
        response = client.get(f"/tasks/{sample_task.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_task.id
        assert data["title"] == "Test Task"
        assert data["description"] == "Test Description"

    def test_get_task_not_found_returns_404(self, client: TestClient):
        response = client.get("/tasks/999")
        assert response.status_code == 404
        assert response.json()["detail"] == "Task not found"


class TestCreateTask:
    """Tests for POST /tasks/ endpoint."""

    def test_create_task_returns_created_task(self, client: TestClient):
        response = client.post(
            "/tasks/",
            json={"title": "New Task", "description": "New Description"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "New Task"
        assert data["description"] == "New Description"
        assert "id" in data

    def test_create_task_without_description(self, client: TestClient):
        response = client.post("/tasks/", json={"title": "Task Without Description"})
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Task Without Description"
        assert data["description"] is None

    def test_create_task_without_title_returns_422(self, client: TestClient):
        response = client.post("/tasks/", json={"description": "No title"})
        assert response.status_code == 422


class TestUpdateTask:
    """Tests for PUT /tasks/{task_id} endpoint."""

    def test_update_task_returns_updated_task(self, client: TestClient, sample_task: Task):
        response = client.put(
            f"/tasks/{sample_task.id}",
            json={"title": "Updated Title", "description": "Updated Description"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
        assert data["description"] == "Updated Description"

    def test_update_task_partial_update(self, client: TestClient, sample_task: Task):
        response = client.put(
            f"/tasks/{sample_task.id}",
            json={"title": "Only Title Updated"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Only Title Updated"
        assert data["description"] == "Test Description"

    def test_update_task_not_found_returns_404(self, client: TestClient):
        response = client.put(
            "/tasks/999",
            json={"title": "Updated Title"},
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Task not found"


class TestDeleteTask:
    """Tests for DELETE /tasks/{task_id} endpoint."""

    def test_delete_task_returns_204(self, client: TestClient, sample_task: Task):
        response = client.delete(f"/tasks/{sample_task.id}")
        assert response.status_code == 204

    def test_delete_task_removes_from_database(self, client: TestClient, sample_task: Task):
        client.delete(f"/tasks/{sample_task.id}")
        response = client.get(f"/tasks/{sample_task.id}")
        assert response.status_code == 404

    def test_delete_task_not_found_returns_404(self, client: TestClient):
        response = client.delete("/tasks/999")
        assert response.status_code == 404
        assert response.json()["detail"] == "Task not found"


class TestHealthCheck:
    """Tests for GET /health endpoint."""

    def test_health_check_returns_ok(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
