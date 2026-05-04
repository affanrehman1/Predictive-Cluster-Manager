"""
Phase 7 -- Automated Stress Test

Simulates a realistic workload cycle against the running FastAPI server:
    1. Starts with a calm baseline (low CPU/memory)
    2. Ramps up to a spike (high CPU/memory) -> watches the autoscaler boot nodes
    3. Returns to calm -> watches the autoscaler shut nodes down
    4. Cleans up at the end

Run this after starting the API:
    uvicorn api:app --host 0.0.0.0 --port 8000
    python stress_test.py
"""

from __future__ import annotations

import sys
import time

import requests

API_BASE = "http://localhost:8000"

SCENARIOS = [
    {
        "label": "[1] CALM -- Normal baseline load",
        "endpoint": "/simulate-stress",
        "body": {"cpu_spike": 0.10, "memory_spike": 0.08, "headroom": 1.2},
    },
    {
        "label": "[2] RISING -- Moderate load increase",
        "endpoint": "/simulate-stress",
        "body": {"cpu_spike": 0.35, "memory_spike": 0.25, "headroom": 1.2},
    },
    {
        "label": "[3] SPIKE -- Heavy traffic surge!",
        "endpoint": "/simulate-stress",
        "body": {"cpu_spike": 0.85, "memory_spike": 0.65, "headroom": 1.2},
    },
    {
        "label": "[4] PEAK -- Maximum load!",
        "endpoint": "/simulate-stress",
        "body": {"cpu_spike": 1.20, "memory_spike": 0.90, "headroom": 1.3},
    },
    {
        "label": "[5] COOLING -- Load dropping",
        "endpoint": "/simulate-stress",
        "body": {"cpu_spike": 0.40, "memory_spike": 0.30, "headroom": 1.2},
    },
    {
        "label": "[6] CALM -- Back to normal",
        "endpoint": "/simulate-stress",
        "body": {"cpu_spike": 0.10, "memory_spike": 0.08, "headroom": 1.2},
    },
]

SEPARATOR = "-" * 70


def check_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        data = r.json()
        print(f"   Status:           {data['status']}")
        print(f"   Model loaded:     {data['model_loaded']}")
        print(f"   Docker available: {data['docker_available']}")
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def get_status():
    r = requests.get(f"{API_BASE}/status", timeout=10)
    return r.json()


def run_scenario(scenario: dict):
    r = requests.post(
        f"{API_BASE}{scenario['endpoint']}",
        json=scenario["body"],
        timeout=60,
    )
    if r.status_code != 200:
        print(f"   [FAIL] Error: {r.status_code} -- {r.text}")
        return None
    return r.json()


def cleanup():
    r = requests.post(f"{API_BASE}/cleanup", timeout=60)
    return r.json()


def main():
    print()
    print("=" * 70)
    print("  PREDICTIVE CLUSTER MANAGER -- AUTOMATED STRESS TEST")
    print("=" * 70)

    # Health check.
    print(f"\n{SEPARATOR}")
    print("[HEALTH] Health Check")
    print(SEPARATOR)
    if not check_health():
        print("   [FAIL] API is not running. Start it with:")
        print("      uvicorn api:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # Initial status.
    print(f"\n{SEPARATOR}")
    print("[STATUS] Initial Cluster Status")
    print(SEPARATOR)
    status = get_status()
    print(f"   Running nodes: {status['running_nodes']}")
    for node in status.get("node_details", []):
        print(f"     * {node['name']} [{node['status']}]")

    # Run through scenarios.
    for i, scenario in enumerate(SCENARIOS):
        print(f"\n{SEPARATOR}")
        print(f"[SCENARIO] {scenario['label']}")
        print(f"   CPU: {scenario['body']['cpu_spike']}  |  Memory: {scenario['body']['memory_spike']}")
        print(SEPARATOR)

        result = run_scenario(scenario)
        if result is None:
            continue

        print(f"   Predicted CPU:      {result['predicted_cpu']:.4f}")
        print(f"   Predicted Memory:   {result['predicted_memory']:.4f}")
        print(f"   Current nodes:      {result['current_nodes']}")
        print(f"   Required nodes:     {result['required_nodes']}")
        print(f"   Scaling direction:  {result['scaling_direction']}")
        print(f"   A* plan cost:       {result['total_cost']}")

        if result["scaling_plan"]:
            print("   Actions:")
            for action in result["scaling_plan"]:
                print(f"     > {action}")

        if result["execution_results"]:
            print("   Execution:")
            for er in result["execution_results"]:
                icon = "[OK]" if er["success"] else "[FAIL]"
                print(f"     {icon} {er['action']} {er['node_name']} -- {er['message']} ({er['duration_ms']:.0f}ms)")

        print(f"   Final node count:   {result['final_node_count']}")

        # Pause between scenarios to simulate real time passing.
        if i < len(SCENARIOS) - 1:
            print("   Waiting 2s before next scenario...")
            time.sleep(2)

    # Final status.
    print(f"\n{SEPARATOR}")
    print("[STATUS] Final Cluster Status")
    print(SEPARATOR)
    status = get_status()
    print(f"   Running nodes: {status['running_nodes']}")
    for node in status.get("node_details", []):
        print(f"     * {node['name']} [{node['status']}]")

    # Cleanup.
    print(f"\n{SEPARATOR}")
    print("[CLEANUP] Removing all cluster nodes")
    print(SEPARATOR)
    result = cleanup()
    print(f"   Removed: {result['removed']} containers")
    for d in result.get("details", []):
        print(f"     * {d['node_name']} -- {d['message']}")

    print(f"\n{'=' * 70}")
    print("  [DONE] STRESS TEST COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
