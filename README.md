# 🚀 Predictive Cluster Manager

> **AI-powered cloud infrastructure autoscaler** — uses deep learning to predict future workload, A\* search to compute the optimal scaling plan, and Docker to physically execute it.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Pipeline Phases](#pipeline-phases)
- [Model Architecture](#model-architecture)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Stress Test Demo](#stress-test-demo)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)

---

## Project Overview

This project implements a **complete end-to-end predictive autoscaling system** that:

1. **Predicts** future CPU and memory usage using an LSTM neural network trained on Google Cluster Trace data
2. **Plans** the optimal scaling action sequence using the A\* search algorithm
3. **Executes** the plan by physically spinning up or tearing down Docker containers that simulate cloud servers

It bridges **predictive AI** with **deterministic algorithms** and **real infrastructure automation**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (api.py)                       │
│                                                                  │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │  LSTM Model   │───▶│  A* Planner   │───▶│  Docker Manager  │  │
│  │  (Predict)    │    │  (Optimize)   │    │  (Execute)       │  │
│  └──────────────┘    └───────────────┘    └──────────────────┘  │
│        │                     │                      │            │
│   .keras model         Cost-optimal          Alpine containers   │
│   .pkl scalers         action path           on host daemon      │
└──────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Docker Daemon     │
                    │  ┌───┐ ┌───┐ ┌───┐ │
                    │  │ 1 │ │ 2 │ │ 3 │ │  ← Simulated Servers
                    │  └───┘ └───┘ └───┘ │
                    └────────────────────┘
```

---

## Dataset

This project uses the **Google Cluster-Usage Traces v3 (2019)**, tracking resource utilisation across ~12,000 machines.

- **Kaggle Source:** [Google Cluster Data](https://www.kaggle.com/datasets/google/cluster-usage)
- **Official Docs:** [Google Cluster Data GitHub](https://github.com/google/cluster-data/blob/master/ClusterData2019.md)

---

## Pipeline Phases

### Phase 1–3: Data & Model Training *(Completed)*

| Phase | File | Description |
|-------|------|-------------|
| **Data Ingestion** | `data_preprocessing.py` | Parses compressed JSON traces, aggregates by timestamp, computes spike-aware features |
| **Model Training** | `train_model.py` | Trains Conv1D + LSTM architecture with multi-GPU support, custom loss, and validation-selected postprocessing |
| **Visualisation** | `visualize_results.py` | Generates prediction accuracy plots and spike detection analysis |

### Phase 4: A\* Search Algorithm

| File | `astar_scaler.py` |
|------|-------------------|
| **Purpose** | Computes the mathematically cheapest path from current cluster size to AI-predicted target |
| **Costs** | Boot = 1.0 (provisioning), Shutdown = 0.5 (teardown) |
| **Heuristic** | `h(n) = abs(current - goal) × min_cost` — admissible & consistent |
| **Output** | Ordered list of scaling actions with total cost |

### Phase 5: FastAPI Backend

| File | `api.py` |
|------|----------|
| **Purpose** | Wires prediction model + A\* planner + Docker manager into REST endpoints |
| **Endpoints** | `/health`, `/status`, `/predict-and-scale`, `/simulate-stress`, `/cleanup` |
| **Features** | Loads `.keras` model and `.pkl` scalers at startup; graceful fallback if model unavailable |

### Phase 6: Docker Simulation

| File | `docker_manager.py` |
|------|----------------------|
| **Purpose** | Uses Docker Python SDK to manage lightweight Alpine containers as simulated servers |
| **Containers** | Named `cluster-node-1`, `cluster-node-2`, etc., with 32MB memory limit |
| **Fallback** | Simulated mode when Docker daemon is unavailable |

### Phase 7: Testing & Documentation

| File | Description |
|------|-------------|
| `stress_test.py` | Automated 6-scenario traffic cycle: calm → rising → spike → peak → cooling → calm |
| `Dockerfile` | Containerised FastAPI application |
| `docker-compose.yml` | One-command deployment with Docker socket mount |
| `README.md` | This file |

---

## Model Architecture

```
Input(60, 36) → LayerNorm → Conv1D(64) → Conv1D(64) → SpatialDropout
    → LSTM(192) → Dropout → LSTM(96) → Dense(128) → Dense(64) → Dense(4)
```

- **Optimizer:** Adam with gradient clipping
- **Loss:** Output-specific peak-weighted Huber loss
- **Outputs:** `[avg_cpu, avg_memory, cpu_delta, mem_delta]`
- **Regularisation:** L2 + Dropout + EarlyStopping + ReduceLROnPlateau

---

## Quick Start

### Option A: Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000

# 3. Open interactive docs
#    http://localhost:8000/docs

# 4. Run the automated stress test (in another terminal)
python stress_test.py
```

### Option B: Run with Docker Compose

```bash
# Build and start
docker-compose up --build

# Run stress test against the container
python stress_test.py

# Shut down
docker-compose down
```

### Test Individual Components

```bash
# Test A* algorithm
python astar_scaler.py

# Test Docker manager
python docker_manager.py
```

---

## API Documentation

Once the server is running, visit **http://localhost:8000/docs** for the interactive Swagger UI.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — model loaded? Docker available? |
| `GET` | `/status` | Current cluster state (running containers) |
| `POST` | `/predict-and-scale` | **Full pipeline**: predict → A\* plan → Docker execute |
| `POST` | `/simulate-stress` | Inject synthetic spike → watch autoscaler react |
| `POST` | `/cleanup` | Remove all managed containers |

### Example: Predict & Scale

```bash
curl -X POST http://localhost:8000/predict-and-scale \
  -H "Content-Type: application/json" \
  -d '{"avg_cpu": 0.45, "avg_memory": 0.30}'
```

**Response:**
```json
{
  "predicted_cpu": 0.4823,
  "predicted_memory": 0.3145,
  "current_nodes": 0,
  "required_nodes": 4,
  "scaling_direction": "UP",
  "scaling_plan": [
    "Boot cluster-node-1",
    "Boot cluster-node-2",
    "Boot cluster-node-3",
    "Boot cluster-node-4"
  ],
  "total_cost": 4.0,
  "execution_results": [...],
  "final_node_count": 4
}
```

---

## Stress Test Demo

The stress test simulates a complete traffic cycle:

```
Calm (0.10) → Rising (0.35) → Spike (0.85) → Peak (1.20) → Cooling (0.40) → Calm (0.10)
```

At each stage, the system:
1. Receives the workload signal
2. Computes the required node count
3. Runs A\* to find the cheapest scaling path
4. Executes boot/shutdown commands via Docker

---

## Project Structure

```
Predictive-Cluster-Manager/
├── data_preprocessing.py       # Phase 1: Data ingestion & feature engineering
├── train_model.py              # Phase 2-3: LSTM model training pipeline
├── visualize_results.py        # Phase 3: Prediction visualisation
├── astar_scaler.py             # Phase 4: A* search algorithm
├── api.py                      # Phase 5: FastAPI backend
├── docker_manager.py           # Phase 6: Docker simulation layer
├── stress_test.py              # Phase 7: Automated stress test
├── Dockerfile                  # Phase 7: Container for the API
├── docker-compose.yml          # Phase 7: One-command deployment
├── requirements.txt            # Python dependencies
├── workload_lstm_model.keras   # Trained LSTM model
├── input_scaler.pkl            # Fitted input StandardScaler
├── target_scaler.pkl           # Fitted target StandardScaler
├── prediction_calibration.pkl  # Prediction bias calibration
├── prediction_postprocess_config.pkl  # Selected postprocessing config
├── spike_guard_config.pkl      # Spike guard thresholds
├── README.md                   # This file
└── .gitignore                  # Repository hygiene
```

---

## Prerequisites

- **Python 3.11+**
- **Docker Desktop** (for container simulation)
- **pip** packages: see `requirements.txt`

### Training (optional — model is pre-trained)

- **Kaggle Notebook** with GPU T4 x2 accelerator
- Upload `clean_workload_data.csv` as a dataset
- Run `train_model.py` contents in the notebook

---

## License

This project is for educational and portfolio purposes.
