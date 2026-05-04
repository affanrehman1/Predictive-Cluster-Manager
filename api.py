"""
Phase 5 — FastAPI Backend Integration

Wires together the LSTM prediction model, A* scaling planner, and Docker
cluster manager into a modern async REST API.

Endpoints:
    GET  /health             — Health check
    GET  /status             — Current cluster state
    POST /predict-and-scale  — End-to-end: predict → plan → execute
    POST /simulate-stress    — Inject a synthetic workload spike
    POST /cleanup            — Remove all managed containers
"""

from __future__ import annotations

import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False
    print("INFO: TensorFlow not installed. Running in simulation-only mode.")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from astar_scaler import (
    ScalingPlan,
    compute_scaling_plan,
    predicted_workload_to_nodes,
    NODE_CPU_CAPACITY,
    NODE_MEMORY_CAPACITY,
)
from docker_manager import DockerClusterManager, NodeInfo, OperationResult

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "workload_lstm_model.keras")
INPUT_SCALER_PATH = os.path.join(BASE_DIR, "input_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(BASE_DIR, "target_scaler.pkl")
CALIBRATION_PATH = os.path.join(BASE_DIR, "prediction_calibration.pkl")
POSTPROCESS_CONFIG_PATH = os.path.join(BASE_DIR, "prediction_postprocess_config.pkl")
SPIKE_GUARD_PATH = os.path.join(BASE_DIR, "spike_guard_config.pkl")

# ---------------------------------------------------------------------------
# Feature / model constants (must match train_model.py)
# ---------------------------------------------------------------------------
LOOK_BACK = 60

BASE_FEATURES = ['avg_cpu', 'avg_memory', 'max_cpu', 'assigned_memory', 'cpi']
ENGINEERED_FEATURES = [
    'cpu_delta', 'mem_delta', 'cpu_abs_delta', 'mem_abs_delta',
    'cpu_spike_pressure', 'time_delta',
    'cpu_roll_mean_3', 'cpu_roll_std_3', 'cpu_roll_max_3',
    'mem_roll_mean_3', 'mem_roll_std_3', 'mem_roll_max_3',
    'cpu_roll_mean_5', 'cpu_roll_std_5', 'cpu_roll_max_5',
    'mem_roll_mean_5', 'mem_roll_std_5', 'mem_roll_max_5',
    'cpu_roll_mean_10', 'cpu_roll_std_10', 'cpu_roll_max_10',
    'mem_roll_mean_10', 'mem_roll_std_10', 'mem_roll_max_10',
    'cpu_roll_mean_30', 'cpu_roll_std_30', 'cpu_roll_max_30',
    'mem_roll_mean_30', 'mem_roll_std_30', 'mem_roll_max_30',
    'cpu_ewm_gap', 'mem_ewm_gap',
]
INPUT_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
VALUE_TARGETS = ['avg_cpu', 'avg_memory']
DELTA_TARGETS = ['cpu_delta', 'mem_delta']
TARGET_FEATURES = VALUE_TARGETS + DELTA_TARGETS
NUM_INPUTS = len(INPUT_FEATURES)


# ---------------------------------------------------------------------------
# Global state (loaded at startup)
# ---------------------------------------------------------------------------
class AppState:
    model: Optional[Any] = None
    input_scaler: Optional[Any] = None
    target_scaler: Optional[Any] = None
    postprocess_config: Optional[Dict] = None
    docker_manager: Optional[DockerClusterManager] = None
    # In-memory synthetic workload buffer for simulation mode.
    synthetic_workload: Optional[np.ndarray] = None
    # Track latest metrics for the visual dashboard
    current_cpu: float = 0.10
    current_memory: float = 0.08


state = AppState()


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artefacts and initialise the Docker manager at startup."""
    print("[STARTUP] Loading ML model and scalers...")
    t0 = time.perf_counter()

    # Load model.
    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        state.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"   Model loaded from {MODEL_PATH}")
    elif not TF_AVAILABLE:
        print(f"   TensorFlow not available. Model will not be loaded (simulation mode).")
    else:
        print(f"   Model not found at {MODEL_PATH}. /predict-and-scale will use synthetic predictions.")

    # Load scalers.
    for attr, path, label in [
        ("input_scaler", INPUT_SCALER_PATH, "Input scaler"),
        ("target_scaler", TARGET_SCALER_PATH, "Target scaler"),
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                setattr(state, attr, pickle.load(f))
            print(f"   [OK] {label} loaded")
        else:
            print(f"   [WARN] {label} not found at {path}")

    # Load postprocess config.
    if os.path.exists(POSTPROCESS_CONFIG_PATH):
        with open(POSTPROCESS_CONFIG_PATH, "rb") as f:
            state.postprocess_config = pickle.load(f)
        print("   [OK] Postprocess config loaded")
    else:
        print("   [WARN] Postprocess config not found -- using defaults")
        state.postprocess_config = {
            "blend": np.array([0.6, 0.2], dtype=np.float32),
            "ensemble_alpha": np.ones(2, dtype=np.float32),
            "baseline_kind": "last",
            "use_calibration": False,
            "calibration": {"scale": np.ones(2, dtype=np.float32), "offset": np.zeros(2, dtype=np.float32)},
            "use_spike_guard": False,
            "spike_guard_config": None,
        }

    # Initialise Docker manager.
    state.docker_manager = DockerClusterManager()
    elapsed = time.perf_counter() - t0
    print(f"[READY] Startup complete in {elapsed:.1f}s  |  Docker available: {state.docker_manager.is_available}")

    yield  # Application runs here.

    # Shutdown: clean up containers.
    if state.docker_manager:
        print("[SHUTDOWN] Cleaning up cluster nodes...")
        state.docker_manager.cleanup_all()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Predictive Cluster Manager",
    description=(
        "AI-powered cloud infrastructure autoscaler.  "
        "Uses an LSTM model to predict future workload, an A* algorithm to "
        "compute the optimal scaling plan, and the Docker SDK to execute it."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    docker_available: bool

class NodeInfoResponse(BaseModel):
    name: str
    container_id: str
    status: str

class ClusterStatus(BaseModel):
    running_nodes: int
    current_cpu: float
    current_memory: float
    node_details: List[NodeInfoResponse]

class PredictionInput(BaseModel):
    """Optional override for workload values.  If omitted, a synthetic
    window is generated for demonstration."""
    avg_cpu: Optional[float] = Field(None, description="Current average CPU usage (0-1 scale)")
    avg_memory: Optional[float] = Field(None, description="Current average memory usage (0-1 scale)")
    headroom: float = Field(1.2, description="Safety buffer multiplier for capacity planning")

class ScalingActionResponse(BaseModel):
    action: str
    node_name: str
    success: bool
    message: str
    duration_ms: float

class PredictAndScaleResponse(BaseModel):
    predicted_cpu: float
    predicted_memory: float
    current_nodes: int
    required_nodes: int
    scaling_direction: str
    scaling_plan: List[str]
    total_cost: float
    execution_results: List[ScalingActionResponse]
    final_node_count: int

class StressRequest(BaseModel):
    cpu_spike: float = Field(0.85, ge=0.0, le=2.0, description="Simulated CPU usage spike")
    memory_spike: float = Field(0.60, ge=0.0, le=2.0, description="Simulated memory usage spike")
    headroom: float = Field(1.2, description="Safety buffer multiplier")

class CleanupResponse(BaseModel):
    removed: int
    details: List[ScalingActionResponse]


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def _generate_synthetic_window(
    avg_cpu: float = 0.25,
    avg_memory: float = 0.15,
) -> np.ndarray:
    """Create a synthetic (LOOK_BACK, NUM_INPUTS) window for demo purposes.

    This fills in plausible feature values centred on the given CPU/memory
    levels so the model produces a meaningful (if approximate) prediction.
    """
    rng = np.random.RandomState(42)
    window = np.zeros((LOOK_BACK, NUM_INPUTS), dtype=np.float32)

    for t in range(LOOK_BACK):
        noise_cpu = rng.normal(0, 0.01)
        noise_mem = rng.normal(0, 0.005)
        cpu = max(0.0, avg_cpu + noise_cpu)
        mem = max(0.0, avg_memory + noise_mem)
        max_cpu = cpu * 1.1
        assigned_mem = mem * 1.2
        cpi = 0.8 + rng.normal(0, 0.05)

        base = [cpu, mem, max_cpu, assigned_mem, cpi]
        # Fill engineered features with reasonable proxies.
        eng = [0.0] * len(ENGINEERED_FEATURES)
        window[t] = base + eng

    return window


def _predict_single(window: np.ndarray) -> tuple[float, float]:
    """Run a single prediction through the loaded model.

    Returns (predicted_cpu, predicted_memory) in original scale.
    """
    if state.model is None or state.input_scaler is None or state.target_scaler is None:
        # Fallback: return the last row's CPU/memory with slight increase.
        return float(window[-1, 0] * 1.15), float(window[-1, 1] * 1.10)

    # Scale the input window.
    scaled = state.input_scaler.transform(window)  # (LOOK_BACK, NUM_INPUTS)
    X = scaled[np.newaxis, ...]  # (1, LOOK_BACK, NUM_INPUTS)

    pred_scaled = state.model.predict(X, verbose=0)  # (1, 4)

    # Inverse-transform to get [avg_cpu, avg_memory, cpu_delta, mem_delta].
    pred_all = state.target_scaler.inverse_transform(pred_scaled)
    direct_cpu = float(pred_all[0, 0])
    direct_mem = float(pred_all[0, 1])

    # Apply blend from postprocess config.
    blend = state.postprocess_config.get("blend", np.array([0.6, 0.2]))
    delta_cpu = float(pred_all[0, 2])
    delta_mem = float(pred_all[0, 3])
    last_cpu = float(window[-1, 0])
    last_mem = float(window[-1, 1])

    blended_cpu = max(0.0, (1.0 - blend[0]) * direct_cpu + blend[0] * (last_cpu + delta_cpu))
    blended_mem = max(0.0, (1.0 - blend[1]) * direct_mem + blend[1] * (last_mem + delta_mem))

    return blended_cpu, blended_mem


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check service health and component availability."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.model is not None,
        docker_available=state.docker_manager.is_available if state.docker_manager else False,
    )


@app.get("/status", response_model=ClusterStatus, tags=["Cluster"])
async def cluster_status():
    """Return current cluster state — running containers and their metadata."""
    if not state.docker_manager:
        raise HTTPException(status_code=503, detail="Docker manager not initialised.")
    nodes = state.docker_manager.get_running_nodes()
    return ClusterStatus(
        running_nodes=len(nodes),
        current_cpu=state.current_cpu,
        current_memory=state.current_memory,
        node_details=[
            NodeInfoResponse(name=n.name, container_id=n.container_id, status=n.status)
            for n in nodes
        ],
    )


@app.post("/predict-and-scale", response_model=PredictAndScaleResponse, tags=["Core"])
async def predict_and_scale(body: PredictionInput = PredictionInput()):
    """End-to-end pipeline: predict workload → A* plan → Docker execution."""
    if not state.docker_manager:
        raise HTTPException(status_code=503, detail="Docker manager not initialised.")

    # 1. Build input window.
    avg_cpu = body.avg_cpu if body.avg_cpu is not None else 0.25
    avg_memory = body.avg_memory if body.avg_memory is not None else 0.15
    state.current_cpu = avg_cpu
    state.current_memory = avg_memory
    window = _generate_synthetic_window(avg_cpu, avg_memory)

    # 2. Predict next time-step.
    pred_cpu, pred_mem = _predict_single(window)

    # 3. Convert prediction to required nodes.
    current_nodes = state.docker_manager.get_running_count()
    required_nodes = predicted_workload_to_nodes(
        pred_cpu,
        pred_mem,
        headroom=body.headroom,
    )

    # 4. A* scaling plan.
    existing_names = state.docker_manager.get_running_names()
    plan = compute_scaling_plan(
        current_nodes=current_nodes,
        goal_nodes=required_nodes,
        existing_node_names=existing_names,
    )

    # 5. Execute the plan.
    results = state.docker_manager.execute_scaling_plan(plan)
    final_count = state.docker_manager.get_running_count()

    direction = "UP" if required_nodes > current_nodes else ("DOWN" if required_nodes < current_nodes else "STABLE")

    return PredictAndScaleResponse(
        predicted_cpu=round(pred_cpu, 6),
        predicted_memory=round(pred_mem, 6),
        current_nodes=current_nodes,
        required_nodes=required_nodes,
        scaling_direction=direction,
        scaling_plan=plan.actions,
        total_cost=plan.total_cost,
        execution_results=[
            ScalingActionResponse(
                action=r.action,
                node_name=r.node_name,
                success=r.success,
                message=r.message,
                duration_ms=round(r.duration_ms, 1),
            )
            for r in results
        ],
        final_node_count=final_count,
    )


@app.post("/simulate-stress", response_model=PredictAndScaleResponse, tags=["Simulation"])
async def simulate_stress(body: StressRequest = StressRequest()):
    """Inject a synthetic workload spike and watch the autoscaler react.

    This endpoint is designed for demonstrations and stress testing — it
    bypasses the model and directly uses the provided CPU/memory values as
    the "predicted" workload.
    """
    if not state.docker_manager:
        raise HTTPException(status_code=503, detail="Docker manager not initialised.")

    pred_cpu = body.cpu_spike
    pred_mem = body.memory_spike
    state.current_cpu = pred_cpu
    state.current_memory = pred_mem
    current_nodes = state.docker_manager.get_running_count()
    required_nodes = predicted_workload_to_nodes(
        pred_cpu,
        pred_mem,
        headroom=body.headroom,
    )

    existing_names = state.docker_manager.get_running_names()
    plan = compute_scaling_plan(
        current_nodes=current_nodes,
        goal_nodes=required_nodes,
        existing_node_names=existing_names,
    )
    results = state.docker_manager.execute_scaling_plan(plan)
    final_count = state.docker_manager.get_running_count()

    direction = "UP" if required_nodes > current_nodes else ("DOWN" if required_nodes < current_nodes else "STABLE")

    return PredictAndScaleResponse(
        predicted_cpu=round(pred_cpu, 6),
        predicted_memory=round(pred_mem, 6),
        current_nodes=current_nodes,
        required_nodes=required_nodes,
        scaling_direction=direction,
        scaling_plan=plan.actions,
        total_cost=plan.total_cost,
        execution_results=[
            ScalingActionResponse(
                action=r.action,
                node_name=r.node_name,
                success=r.success,
                message=r.message,
                duration_ms=round(r.duration_ms, 1),
            )
            for r in results
        ],
        final_node_count=final_count,
    )


@app.post("/cleanup", response_model=CleanupResponse, tags=["Cluster"])
async def cleanup_cluster():
    """Remove all managed cluster-node containers."""
    if not state.docker_manager:
        raise HTTPException(status_code=503, detail="Docker manager not initialised.")

    results = state.docker_manager.cleanup_all()
    return CleanupResponse(
        removed=sum(1 for r in results if r.success),
        details=[
            ScalingActionResponse(
                action=r.action,
                node_name=r.node_name,
                success=r.success,
                message=r.message,
                duration_ms=round(r.duration_ms, 1),
            )
            for r in results
        ],
    )
