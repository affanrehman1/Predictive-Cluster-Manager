from docker_manager import DockerClusterManager


def test_docker_manager_simulated_boot_and_shutdown(monkeypatch):
    def fake_init(self):
        self._available = False
        self._simulated_nodes = {}

    monkeypatch.setattr(DockerClusterManager, "__init__", fake_init)

    manager = DockerClusterManager()

    boot = manager.boot_node("cluster-node-1")
    assert boot.success is True
    assert boot.action == "boot"
    assert manager.get_running_count() == 1
    assert manager.get_running_names() == ["cluster-node-1"]

    shutdown = manager.shutdown_node("cluster-node-1")
    assert shutdown.success is True
    assert shutdown.action == "shutdown"
    assert manager.get_running_count() == 0


def test_docker_manager_simulated_shutdown_missing_node(monkeypatch):
    def fake_init(self):
        self._available = False
        self._simulated_nodes = {}

    monkeypatch.setattr(DockerClusterManager, "__init__", fake_init)

    manager = DockerClusterManager()
    result = manager.shutdown_node("cluster-node-404")

    assert result.success is False
    assert "not found" in result.message


def test_cleanup_all_removes_simulated_nodes(monkeypatch):
    def fake_init(self):
        self._available = False
        self._simulated_nodes = {}

    monkeypatch.setattr(DockerClusterManager, "__init__", fake_init)

    manager = DockerClusterManager()
    manager.boot_node("cluster-node-1")
    manager.boot_node("cluster-node-2")

    results = manager.cleanup_all()

    assert len(results) == 2
    assert all(result.success for result in results)
    assert manager.get_running_count() == 0
