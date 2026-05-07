import pytest

from astar_scaler import compute_scaling_plan, predicted_workload_to_nodes


def test_compute_scaling_plan_scales_up():
    plan = compute_scaling_plan(current_nodes=2, goal_nodes=5)

    assert plan.initial_nodes == 2
    assert plan.goal_nodes == 5
    assert plan.actions == [
        "Boot cluster-node-3",
        "Boot cluster-node-4",
        "Boot cluster-node-5",
    ]
    assert plan.total_cost == pytest.approx(3.0)
    assert plan.nodes_to_boot == [
        "cluster-node-3",
        "cluster-node-4",
        "cluster-node-5",
    ]
    assert plan.nodes_to_shutdown == []


def test_compute_scaling_plan_scales_down_using_existing_names():
    existing_names = [f"cluster-node-{i}" for i in range(1, 6)]

    plan = compute_scaling_plan(
        current_nodes=5,
        goal_nodes=2,
        existing_node_names=existing_names,
    )

    assert plan.actions == [
        "Shutdown cluster-node-5",
        "Shutdown cluster-node-4",
        "Shutdown cluster-node-3",
    ]
    assert plan.total_cost == pytest.approx(1.5)
    assert plan.nodes_to_boot == []
    assert plan.nodes_to_shutdown == [
        "cluster-node-5",
        "cluster-node-4",
        "cluster-node-3",
    ]


def test_compute_scaling_plan_when_already_at_goal():
    plan = compute_scaling_plan(current_nodes=3, goal_nodes=3)

    assert plan.actions == []
    assert plan.total_cost == pytest.approx(0.0)
    assert plan.nodes_to_boot == []
    assert plan.nodes_to_shutdown == []


def test_compute_scaling_plan_clamps_goal_to_max_nodes():
    plan = compute_scaling_plan(current_nodes=48, goal_nodes=99, max_nodes=50)

    assert plan.goal_nodes == 50
    assert plan.actions == [
        "Boot cluster-node-49",
        "Boot cluster-node-50",
    ]


def test_predicted_workload_to_nodes_uses_largest_capacity_need():
    nodes = predicted_workload_to_nodes(
        predicted_cpu=0.45,
        predicted_memory=0.30,
        headroom=1.2,
    )

    assert nodes == 4


def test_predicted_workload_to_nodes_respects_min_and_max():
    assert predicted_workload_to_nodes(0.0, 0.0, min_nodes=1) == 1
    assert predicted_workload_to_nodes(99.0, 99.0, max_nodes=10) == 10
