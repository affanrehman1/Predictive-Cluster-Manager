from fastapi.testclient import TestClient

import api
from conftest import FakeDockerManager


def test_health_endpoint_reports_component_state():
    api.state.model = None
    api.state.docker_manager = FakeDockerManager()

    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data == {
        "status": "healthy",
        "model_loaded": False,
        "docker_available": False,
    }


def test_status_endpoint_returns_cluster_state():
    api.state.docker_manager = FakeDockerManager(["cluster-node-1"])
    api.state.current_cpu = 0.25
    api.state.current_memory = 0.15

    client = TestClient(api.app)
    response = client.get("/status")

    assert response.status_code == 200
    data = response.json()
    assert data["running_nodes"] == 1
    assert data["current_cpu"] == 0.25
    assert data["current_memory"] == 0.15
    assert data["node_details"][0]["name"] == "cluster-node-1"
    assert data["node_details"][0]["status"] == "running (test)"


def test_predict_and_scale_uses_fallback_prediction_and_scales_up():
    api.state.docker_manager = FakeDockerManager([])

    client = TestClient(api.app)
    response = client.post(
        "/predict-and-scale",
        json={
            "avg_cpu": 0.45,
            "avg_memory": 0.30,
            "headroom": 1.2,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["scaling_direction"] == "UP"
    assert data["required_nodes"] == 5
    assert data["final_node_count"] == 5
    assert data["scaling_plan"][0] == "Boot cluster-node-1"
    assert len(data["execution_results"]) == 5


def test_simulate_stress_scales_up_from_direct_workload():
    api.state.docker_manager = FakeDockerManager([])

    client = TestClient(api.app)
    response = client.post(
        "/simulate-stress",
        json={
            "cpu_spike": 0.45,
            "memory_spike": 0.30,
            "headroom": 1.2,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_cpu"] == 0.45
    assert data["predicted_memory"] == 0.30
    assert data["scaling_direction"] == "UP"
    assert data["required_nodes"] == 4
    assert data["final_node_count"] == 4


def test_simulate_stress_scales_down_when_cluster_is_too_large():
    api.state.docker_manager = FakeDockerManager(
        [
            "cluster-node-1",
            "cluster-node-2",
            "cluster-node-3",
            "cluster-node-4",
        ]
    )

    client = TestClient(api.app)
    response = client.post(
        "/simulate-stress",
        json={
            "cpu_spike": 0.10,
            "memory_spike": 0.08,
            "headroom": 1.2,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["scaling_direction"] == "DOWN"
    assert data["required_nodes"] == 1
    assert data["final_node_count"] == 1
    assert data["scaling_plan"] == [
        "Shutdown cluster-node-4",
        "Shutdown cluster-node-3",
        "Shutdown cluster-node-2",
    ]


def test_cleanup_endpoint_removes_all_fake_nodes():
    api.state.docker_manager = FakeDockerManager(
        ["cluster-node-1", "cluster-node-2"]
    )

    client = TestClient(api.app)
    response = client.post("/cleanup")

    assert response.status_code == 200
    data = response.json()
    assert data["removed"] == 2
    assert len(data["details"]) == 2
