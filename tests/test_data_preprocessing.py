import pandas as pd
import pytest

from data_preprocessing import add_spike_features, aggregate_workload_by_time


def test_aggregate_workload_by_time_groups_rows_by_timestamp():
    df = pd.DataFrame(
        [
            {
                "time": 1,
                "avg_cpu": 0.10,
                "avg_memory": 0.20,
                "max_cpu": 0.30,
                "assigned_memory": 0.40,
                "cpi": 1.0,
            },
            {
                "time": 1,
                "avg_cpu": 0.20,
                "avg_memory": 0.30,
                "max_cpu": 0.40,
                "assigned_memory": 0.50,
                "cpi": 3.0,
            },
            {
                "time": 2,
                "avg_cpu": 0.40,
                "avg_memory": 0.50,
                "max_cpu": 0.60,
                "assigned_memory": 0.70,
                "cpi": 5.0,
            },
        ]
    )

    result = aggregate_workload_by_time(df)

    assert list(result["time"]) == [1, 2]
    assert result.loc[0, "avg_cpu"] == pytest.approx(0.30)
    assert result.loc[0, "avg_memory"] == pytest.approx(0.50)
    assert result.loc[0, "max_cpu"] == pytest.approx(0.70)
    assert result.loc[0, "assigned_memory"] == pytest.approx(0.90)
    assert result.loc[0, "cpi"] == pytest.approx(2.0)


def test_aggregate_workload_by_time_raises_for_missing_columns():
    df = pd.DataFrame([{"time": 1, "avg_cpu": 0.10}])

    with pytest.raises(ValueError, match="Missing columns"):
        aggregate_workload_by_time(df)


def test_add_spike_features_creates_temporal_feature_columns():
    df = pd.DataFrame(
        [
            {
                "time": 1,
                "avg_cpu": 0.10,
                "avg_memory": 0.20,
                "max_cpu": 0.15,
                "assigned_memory": 0.30,
                "cpi": 1.0,
            },
            {
                "time": 2,
                "avg_cpu": 0.50,
                "avg_memory": 0.40,
                "max_cpu": 0.70,
                "assigned_memory": 0.60,
                "cpi": 1.2,
            },
        ]
    )

    result = add_spike_features(df)

    expected_columns = {
        "cpu_delta",
        "mem_delta",
        "cpu_abs_delta",
        "mem_abs_delta",
        "cpu_spike_pressure",
        "time_delta",
        "cpu_roll_mean_3",
        "mem_roll_std_30",
        "cpu_ewm_gap",
        "mem_ewm_gap",
    }

    assert expected_columns.issubset(result.columns)
    assert result.loc[1, "cpu_delta"] == pytest.approx(0.40)
    assert result.loc[1, "mem_delta"] == pytest.approx(0.20)
    assert result.loc[1, "cpu_spike_pressure"] == pytest.approx(0.20)
    assert result.isna().sum().sum() == 0
