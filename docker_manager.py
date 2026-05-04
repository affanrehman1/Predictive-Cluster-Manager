"""
Phase 6 — Docker Simulation Layer

Uses the Docker Python SDK to manage lightweight Alpine containers that
act as simulated Google Cloud cluster nodes.

Each container is named ``cluster-node-<N>`` and runs a simple ``sleep``
command — the point is to demonstrate real OS-level container lifecycle
management driven by AI predictions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import docker
from docker.errors import APIError, NotFound

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONTAINER_IMAGE = "alpine:latest"
CONTAINER_PREFIX = "cluster-node-"
CONTAINER_COMMAND = "sleep infinity"  # keeps the container alive
CONTAINER_LABELS = {"managed-by": "predictive-cluster-manager"}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NodeInfo:
    """Metadata for a running simulated node."""
    name: str
    container_id: str
    status: str
    created: str


@dataclass
class OperationResult:
    """Outcome of a single boot / shutdown operation."""
    action: str  # "boot" | "shutdown"
    node_name: str
    success: bool
    message: str
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Docker Manager
# ---------------------------------------------------------------------------
class DockerClusterManager:
    """Manages simulated cluster nodes as Docker containers."""

    def __init__(self) -> None:
        try:
            self.client = docker.from_env()
            self.client.ping()
            self._available = True
        except Exception as exc:
            print(f"[WARN] Docker daemon not available: {exc}")
            print("  Falling back to simulated mode (no real containers).")
            self._available = False
            self._simulated_nodes: Dict[str, str] = {}

    @property
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_running_nodes(self) -> List[NodeInfo]:
        """List all managed cluster-node containers."""
        if not self._available:
            return [
                NodeInfo(name=name, container_id=cid[:12], status="running (simulated)", created="N/A")
                for name, cid in self._simulated_nodes.items()
            ]

        containers = self.client.containers.list(
            all=True,
            filters={"label": "managed-by=predictive-cluster-manager"},
        )
        nodes = []
        for c in containers:
            nodes.append(
                NodeInfo(
                    name=c.name,
                    container_id=c.short_id,
                    status=c.status,
                    created=str(c.attrs.get("Created", "unknown")),
                )
            )
        # Sort by name so cluster-node-1 appears before cluster-node-2.
        nodes.sort(key=lambda n: n.name)
        return nodes

    def get_running_count(self) -> int:
        """Return the number of running (not stopped/exited) nodes."""
        if not self._available:
            return len(self._simulated_nodes)
        containers = self.client.containers.list(
            filters={
                "label": "managed-by=predictive-cluster-manager",
                "status": "running",
            },
        )
        return len(containers)

    def get_running_names(self) -> List[str]:
        """Return sorted list of running node names."""
        if not self._available:
            return sorted(self._simulated_nodes.keys())
        containers = self.client.containers.list(
            filters={
                "label": "managed-by=predictive-cluster-manager",
                "status": "running",
            },
        )
        return sorted(c.name for c in containers)

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------
    def boot_node(self, name: str) -> OperationResult:
        """Start a new container acting as a cluster node."""
        t0 = time.perf_counter()

        if not self._available:
            self._simulated_nodes[name] = f"sim-{len(self._simulated_nodes):04d}"
            return OperationResult(
                action="boot",
                node_name=name,
                success=True,
                message=f"Simulated node {name} started.",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            # Check if a container with this name already exists.
            try:
                existing = self.client.containers.get(name)
                if existing.status == "running":
                    return OperationResult(
                        action="boot",
                        node_name=name,
                        success=True,
                        message=f"Node {name} is already running.",
                        duration_ms=(time.perf_counter() - t0) * 1000,
                    )
                # Restart a stopped container.
                existing.start()
                return OperationResult(
                    action="boot",
                    node_name=name,
                    success=True,
                    message=f"Restarted existing stopped container {name}.",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
            except NotFound:
                pass

            # Pull image if missing.
            try:
                self.client.images.get(CONTAINER_IMAGE)
            except Exception:
                print(f"  Pulling image {CONTAINER_IMAGE}...")
                self.client.images.pull(CONTAINER_IMAGE)

            container = self.client.containers.run(
                CONTAINER_IMAGE,
                CONTAINER_COMMAND,
                name=name,
                labels=CONTAINER_LABELS,
                detach=True,
                mem_limit="32m",
                cpu_quota=10000,  # ~10% of one CPU
            )
            elapsed = (time.perf_counter() - t0) * 1000
            return OperationResult(
                action="boot",
                node_name=name,
                success=True,
                message=f"Node {name} started (container {container.short_id}).",
                duration_ms=elapsed,
            )
        except APIError as exc:
            return OperationResult(
                action="boot",
                node_name=name,
                success=False,
                message=f"Docker API error: {exc.explanation}",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            return OperationResult(
                action="boot",
                node_name=name,
                success=False,
                message=str(exc),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

    def shutdown_node(self, name: str) -> OperationResult:
        """Stop and remove a cluster-node container."""
        t0 = time.perf_counter()

        if not self._available:
            if name in self._simulated_nodes:
                del self._simulated_nodes[name]
                return OperationResult(
                    action="shutdown",
                    node_name=name,
                    success=True,
                    message=f"Simulated node {name} shut down.",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
            return OperationResult(
                action="shutdown",
                node_name=name,
                success=False,
                message=f"Simulated node {name} not found.",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            container = self.client.containers.get(name)
            container.stop(timeout=5)
            container.remove(force=True)
            elapsed = (time.perf_counter() - t0) * 1000
            return OperationResult(
                action="shutdown",
                node_name=name,
                success=True,
                message=f"Node {name} stopped and removed.",
                duration_ms=elapsed,
            )
        except NotFound:
            return OperationResult(
                action="shutdown",
                node_name=name,
                success=False,
                message=f"Node {name} not found.",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            return OperationResult(
                action="shutdown",
                node_name=name,
                success=False,
                message=str(exc),
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

    def execute_scaling_plan(self, plan) -> List[OperationResult]:
        """Execute a ScalingPlan from the A* algorithm.

        Parameters
        ----------
        plan : ScalingPlan
            Output from ``astar_scaler.compute_scaling_plan``.

        Returns
        -------
        list[OperationResult]
        """
        results: List[OperationResult] = []
        for node_name in plan.nodes_to_boot:
            results.append(self.boot_node(node_name))
        for node_name in plan.nodes_to_shutdown:
            results.append(self.shutdown_node(node_name))
        return results

    def cleanup_all(self) -> List[OperationResult]:
        """Shut down and remove ALL managed cluster-node containers."""
        results: List[OperationResult] = []
        if not self._available:
            names = list(self._simulated_nodes.keys())
            for name in names:
                results.append(self.shutdown_node(name))
            return results

        containers = self.client.containers.list(
            all=True,
            filters={"label": "managed-by=predictive-cluster-manager"},
        )
        for c in containers:
            results.append(self.shutdown_node(c.name))
        return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Docker Cluster Manager — Self-Test")
    print("=" * 60)

    manager = DockerClusterManager()
    print(f"\nDocker available: {manager.is_available}")
    print(f"Running nodes: {manager.get_running_count()}")

    # Boot 3 nodes.
    for i in range(1, 4):
        result = manager.boot_node(f"cluster-node-{i}")
        print(f"  {result.action}: {result.node_name} — {result.message} ({result.duration_ms:.0f}ms)")

    print(f"\nRunning nodes after boot: {manager.get_running_count()}")
    for node in manager.get_running_nodes():
        print(f"  {node.name} [{node.status}]")

    # Shutdown node 3.
    result = manager.shutdown_node("cluster-node-3")
    print(f"\n  {result.action}: {result.node_name} — {result.message}")
    print(f"Running nodes after shutdown: {manager.get_running_count()}")

    # Cleanup.
    print("\nCleaning up all nodes...")
    for r in manager.cleanup_all():
        print(f"  {r.action}: {r.node_name} — {r.message}")
    print(f"Running nodes after cleanup: {manager.get_running_count()}")
