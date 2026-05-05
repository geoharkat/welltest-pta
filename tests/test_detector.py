"""Detector tests — V8.1 pipeline correctness."""

import numpy as np
import pandas as pd
import pytest

from welltest_pta import EventDetector, EventDetectorConfig, detect_events


def test_detector_runs_on_synthetic(synth_df):
    """Detector returns expected columns and labels."""
    annotated, det = detect_events(synth_df)
    assert "p_smooth" in annotated.columns
    assert "elapsed_hr" in annotated.columns
    assert "event" in annotated.columns
    assert annotated["event"].isin(["drawdown", "buildup", "non_pta"]).all()


def test_detector_finds_at_least_one_pta(synth_df):
    """Synthetic DD-BU sequence must yield at least 1 DD and 1 BU."""
    annotated, _ = detect_events(synth_df)
    n_dd = (annotated["event"] == "drawdown").sum()
    n_bu = (annotated["event"] == "buildup").sum()
    assert n_dd > 0, f"No drawdowns detected (got {n_dd})"
    assert n_bu > 0, f"No buildups detected (got {n_bu})"


def test_detector_p_res_close_to_truth(synth_df):
    """Detected P_res should be within 5 % of the synthetic value."""
    _, det = detect_events(synth_df)
    p_res = det._p_res
    truth = 4500.0
    assert 0.95 * truth < p_res < 1.05 * truth


def test_detector_config_overrides():
    """Custom config must override defaults."""
    cfg = EventDetectorConfig(
        hampel_sigma=4.5,
        spike_percentile=98.0,
        min_pta_dp_psi=25.0,
    )
    assert cfg.hampel_sigma == 4.5
    assert cfg.spike_percentile == 98.0
    det = EventDetector(cfg=cfg)
    assert det.cfg.hampel_sigma == 4.5


def test_detector_rejects_too_few_samples():
    """Detector must error out with < 20 samples."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-15 10:00", periods=10, freq="5s"),
        "pressure": np.linspace(4500, 4505, 10),
    })
    det = EventDetector()
    with pytest.raises(ValueError):
        det.detect(df)
