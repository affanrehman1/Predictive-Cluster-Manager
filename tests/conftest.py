import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeDockerManager:
    """In-memory Docker replacement for fast, safe API tests."""

    def __init__(self, names=None):
        self.is_available = False
        self.nodes = list(names or [])

    def get_running_nodes(self):
        return [
            SimpleNamespace(
                name=name,
                container_id=f"fake-{index:04d}",
                status="running (test)",
            )
            for index, name in enumerate(self.nodes, start=1)
        ]

    def get_running_count(self):
        return len(self.nodes)

    def get_running_names(self):
        return list(self.nodes)

    def execute_scaling_plan(self, plan):
        results = []

        for name in plan.nodes_to_boot:
            if name not in self.nodes:
                self.nodes.append(name)
            results.append(
                SimpleNamespace(
                    action="boot",
                    node_name=name,
                    success=True,
                    message=f"Booted {name}",
                    duration_ms=1.0,
                )
            )

        for name in plan.nodes_to_shutdown:
            if name in self.nodes:
                self.nodes.remove(name)
            results.append(
                SimpleNamespace(
                    action="shutdown",
                    node_name=name,
                    success=True,
                    message=f"Shutdown {name}",
                    duration_ms=1.0,
                )
            )

        return results

    def cleanup_all(self):
        results = [
            SimpleNamespace(
                action="shutdown",
                node_name=name,
                success=True,
                message=f"Removed {name}",
                duration_ms=1.0,
            )
            for name in self.nodes
        ]
        self.nodes.clear()
        return results


@pytest.fixture(autouse=True)
def reset_api_state():
    """Keep API tests isolated from one another."""
    import api

    api.state.model = None
    api.state.input_scaler = None
    api.state.target_scaler = None
    api.state.postprocess_config = None
    api.state.docker_manager = FakeDockerManager()
    api.state.synthetic_workload = None
    api.state.current_cpu = 0.10
    api.state.current_memory = 0.08
    yield
