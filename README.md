# Workload Forecasting with LSTM

## Project Overview
This repository implements a high-performance preprocessing pipeline for workload forecasting using the Google Cluster Trace dataset (v3). The system is designed to transform raw cluster telemetry into 3D temporal tensors optimized for Long Short-Term Memory (LSTM) network training.

## Dataset
This project utilizes the Google Cluster-Usage Traces v3 (2019), which tracks resource utilization across approximately 12,000 machines.

*   **Kaggle Source:** [Google Cluster Data - Kaggle](https://www.kaggle.com/datasets/google/cluster-usage)
*   **Official Documentation:** [Google Cluster Data GitHub](https://github.com/google/cluster-data/blob/master/ClusterData2019.md)

## Prerequisites
*   Python 3.8+
*   NumPy
*   Pandas
*   Scikit-learn
*   TensorFlow 2.x (Planned for model implementation)

## Pipeline Architecture

### 1. Data Ingestion (`data_preprocessing.py`)
Parses compressed GZIP JSON traces. It performs initial cleaning, extracts core telemetry (CPU/Memory), and exports a chronologically sorted CSV.

### 2. Feature Engineering (`lstm_preprocessing.py`)
Processes the cleaned telemetry into a format suitable for recurrent architectures:
*   **Min-Max Normalization**: Scales CPU usage to [0, 1] to ensure numerical stability and prevent activation saturation.
*   **Vectorization**: Implements a sliding-window algorithm to generate temporal sequences of shape `(N, T, F)`.
*   **Chronological Partitioning**: Splits data into 80% training and 20% testing sets while preserving the temporal sequence to prevent data leakage.

## Repository Hygiene
A comprehensive `.gitignore` is implemented to ensure repository integrity. The following artifacts are excluded from version control:
*   **Large Data**: All `.csv`, `.gz`, and raw data directories.
*   **Model Weights**: Saved model files (`.h5`, `.keras`, `.pth`).
*   **Virtual Environments**: Local environment directories (`.venv`, `env`).
*   **Secrets**: Environment variable files (`.env`).

## Execution Flow
1.  Place raw `instance_usage-*.json.gz` files in the project root.
2.  Run `python data_preprocessing.py` to generate the normalized CSV.
3.  Run `python lstm_preprocessing.py` to prepare tensors for model training.
