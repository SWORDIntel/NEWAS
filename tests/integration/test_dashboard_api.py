import pytest
from fastapi.testclient import TestClient
from main import NEMWASCore
from src.api.server import create_app

@pytest.fixture(scope="module")
def test_app():
    core = NEMWASCore(config_path='config/default.yaml')
    app = create_app(core)
    with TestClient(app) as client:
        yield client

def test_get_overview_metrics(test_app):
    response = test_app.get("/dashboard/overview")
    assert response.status_code == 200
    data = response.json()
    assert "agents" in data
    assert "tasks" in data
    assert "performance" in data
    assert "resources" in data
    assert "total" in data["agents"]
    assert "active" in data["agents"]
    assert "idle" in data["agents"]
    assert "failed" in data["agents"]
    assert "queued" in data["tasks"]
    assert "processing" in data["tasks"]
    assert "completed24h" in data["tasks"]
    assert "failed24h" in data["tasks"]
    assert "npuUtilization" in data["resources"]
    assert "gpuUtilization" in data["resources"]
    assert "cpuUtilization" in data["resources"]
    assert "memoryUsage" in data["resources"]
