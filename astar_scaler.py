"""
Phase 4 — A* Search Algorithm for Cluster Scaling

Computes the mathematically cheapest sequence of boot/shutdown actions
to transition from the current cluster size (Initial State) to the
AI-predicted required cluster size (Goal State).

Costs:
    Boot   = 1.0  (provisioning a new node takes more resources)
    Shutdown = 0.5  (graceful teardown is cheaper)

Heuristic:
    h(n) = abs(current_nodes - goal_nodes)
    Admissible because every step changes the node count by exactly 1,
    so the true remaining cost is always >= h(n).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Cost constants
# ---------------------------------------------------------------------------
BOOT_COST: float = 1.0
SHUTDOWN_COST: float = 0.5

# Capacity of a single simulated node.
NODE_CPU_CAPACITY: float = 0.15
NODE_MEMORY_CAPACITY: float = 0.10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(order=True)
class _PrioritisedState:
    """Min-heap element wrapping an A* search state."""
    f_score: float
    state: int = field(compare=False)
    g_score: float = field(compare=False)
    actions: list = field(compare=False, default_factory=list)


@dataclass
class ScalingPlan:
    """Result produced by the A* planner."""
    initial_nodes: int
    goal_nodes: int
    actions: List[str]
    total_cost: float
    nodes_to_boot: List[str]
    nodes_to_shutdown: List[str]

    def summary(self) -> str:
        if not self.actions:
            return (
                f"Cluster is already at optimal size ({self.initial_nodes} nodes). "
                "No scaling actions required."
            )
        direction = "UP" if self.goal_nodes > self.initial_nodes else "DOWN"
        return (
            f"Scale {direction}: {self.initial_nodes} -> {self.goal_nodes} nodes | "
            f"{len(self.actions)} actions | total cost: {self.total_cost:.1f}\n"
            + "\n".join(f"  {i+1}. {a}" for i, a in enumerate(self.actions))
        )


# ---------------------------------------------------------------------------
# Heuristic & neighbours
# ---------------------------------------------------------------------------
def _heuristic(current: int, goal: int) -> float:
    """Admissible heuristic: minimum number of steps × cheapest action cost."""
    diff = abs(current - goal)
    # Lower-bound using the cheapest possible per-step cost.
    return diff * min(BOOT_COST, SHUTDOWN_COST)


def _neighbours(
    state: int,
    goal: int,
    existing_node_names: List[str],
    max_nodes: int = 50,
) -> List[Tuple[int, float, str]]:
    """Return (next_state, step_cost, action_description) for each valid move."""
    result: List[Tuple[int, float, str]] = []

    if state < goal and state < max_nodes:
        next_id = state + 1
        name = f"cluster-node-{next_id}"
        result.append((next_id, BOOT_COST, f"Boot {name}"))

    if state > goal and state > 0:
        # Shut down the highest-numbered node first.
        name = (
            existing_node_names[state - 1]
            if state - 1 < len(existing_node_names)
            else f"cluster-node-{state}"
        )
        result.append((state - 1, SHUTDOWN_COST, f"Shutdown {name}"))

    return result


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------
def compute_scaling_plan(
    current_nodes: int,
    goal_nodes: int,
    existing_node_names: Optional[List[str]] = None,
    max_nodes: int = 50,
) -> ScalingPlan:
    """Run A* to find the cheapest path from *current_nodes* to *goal_nodes*.

    Parameters
    ----------
    current_nodes : int
        Number of nodes currently running in the cluster.
    goal_nodes : int
        Target number of nodes (derived from AI prediction).
    existing_node_names : list[str] | None
        Names of the currently running containers (used for shutdown labelling).
    max_nodes : int
        Upper safety cap on cluster size.

    Returns
    -------
    ScalingPlan
        Ordered sequence of boot/shutdown actions with cost metadata.
    """
    if existing_node_names is None:
        existing_node_names = [f"cluster-node-{i}" for i in range(1, current_nodes + 1)]

    goal_nodes = max(0, min(goal_nodes, max_nodes))

    # Trivial case — already at goal.
    if current_nodes == goal_nodes:
        return ScalingPlan(
            initial_nodes=current_nodes,
            goal_nodes=goal_nodes,
            actions=[],
            total_cost=0.0,
            nodes_to_boot=[],
            nodes_to_shutdown=[],
        )

    # Priority queue: (f_score, state, g_score, action_history)
    open_set: list[_PrioritisedState] = []
    heapq.heappush(
        open_set,
        _PrioritisedState(
            f_score=_heuristic(current_nodes, goal_nodes),
            state=current_nodes,
            g_score=0.0,
            actions=[],
        ),
    )
    visited: set[int] = set()

    while open_set:
        current = heapq.heappop(open_set)

        if current.state == goal_nodes:
            # Reconstruct boot / shutdown lists.
            boots = [a.split(" ", 1)[1] for a in current.actions if a.startswith("Boot")]
            shutdowns = [a.split(" ", 1)[1] for a in current.actions if a.startswith("Shutdown")]
            return ScalingPlan(
                initial_nodes=current_nodes,
                goal_nodes=goal_nodes,
                actions=current.actions,
                total_cost=current.g_score,
                nodes_to_boot=boots,
                nodes_to_shutdown=shutdowns,
            )

        if current.state in visited:
            continue
        visited.add(current.state)

        for next_state, step_cost, action in _neighbours(
            current.state, goal_nodes, existing_node_names, max_nodes
        ):
            if next_state in visited:
                continue
            new_g = current.g_score + step_cost
            new_f = new_g + _heuristic(next_state, goal_nodes)
            heapq.heappush(
                open_set,
                _PrioritisedState(
                    f_score=new_f,
                    state=next_state,
                    g_score=new_g,
                    actions=current.actions + [action],
                ),
            )

    # Fallback (should never be reached for valid inputs).
    return ScalingPlan(
        initial_nodes=current_nodes,
        goal_nodes=goal_nodes,
        actions=[],
        total_cost=0.0,
        nodes_to_boot=[],
        nodes_to_shutdown=[],
    )


# ---------------------------------------------------------------------------
# Workload → node-count mapping
# ---------------------------------------------------------------------------
def predicted_workload_to_nodes(
    predicted_cpu: float,
    predicted_memory: float,
    cpu_capacity: float = NODE_CPU_CAPACITY,
    mem_capacity: float = NODE_MEMORY_CAPACITY,
    min_nodes: int = 1,
    max_nodes: int = 50,
    headroom: float = 1.2,
) -> int:
    """Convert a CPU/memory prediction into a required node count.

    Applies a 20% headroom buffer by default so the cluster is never at 100%
    capacity at the predicted load.
    """
    cpu_nodes = math.ceil((predicted_cpu * headroom) / cpu_capacity) if cpu_capacity > 0 else 1
    mem_nodes = math.ceil((predicted_memory * headroom) / mem_capacity) if mem_capacity > 0 else 1
    return max(min_nodes, min(max(cpu_nodes, mem_nodes), max_nodes))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("A* Cluster Scaling — Self-Test")
    print("=" * 60)

    # Scenario 1: Scale UP from 2 → 5
    plan = compute_scaling_plan(current_nodes=2, goal_nodes=5)
    print(f"\n{plan.summary()}")

    # Scenario 2: Scale DOWN from 5 → 2
    plan = compute_scaling_plan(
        current_nodes=5,
        goal_nodes=2,
        existing_node_names=[f"cluster-node-{i}" for i in range(1, 6)],
    )
    print(f"\n{plan.summary()}")

    # Scenario 3: Already at target
    plan = compute_scaling_plan(current_nodes=3, goal_nodes=3)
    print(f"\n{plan.summary()}")

    # Scenario 4: From workload prediction
    pred_cpu, pred_mem = 0.45, 0.30
    needed = predicted_workload_to_nodes(pred_cpu, pred_mem)
    plan = compute_scaling_plan(current_nodes=2, goal_nodes=needed)
    print(f"\nPredicted CPU={pred_cpu}, Memory={pred_mem} -> need {needed} nodes")
    print(plan.summary())
