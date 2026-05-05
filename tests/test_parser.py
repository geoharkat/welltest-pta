"""Smoke test for the WellTestParser on a synthetic ASCII file."""

import io
import pandas as pd
import pytest

from welltest_pta import parse, WellTestParser
from welltest_pta.utils.synthetic import generate_synthetic_dst


def test_parser_handles_csv(tmp_path):
    """Parser should handle a clean CSV with timestamp + pressure."""
    df = generate_synthetic_dst(n_samples=500, sample_period_s=10.0, seed=7)
    df_to_save = df.drop(columns=["true_event"], errors="ignore")
    p = tmp_path / "fake.csv"
    df_to_save.to_csv(p, index=False)

    out = parse(p)
    assert "timestamp" in out.columns
    assert "pressure" in out.columns
    assert len(out) == len(df_to_save)


def test_parser_handles_semicolon_decimal_comma(tmp_path):
    """Parser should handle European-style ';' delimiter and ',' decimal."""
    p = tmp_path / "euro.txt"
    p.write_text(
        "timestamp;pressure;temperature\n"
        "2025-01-15 10:00:00;4500,12;180,5\n"
        "2025-01-15 10:00:05;4500,15;180,6\n"
        "2025-01-15 10:00:10;4500,18;180,5\n"
        "2025-01-15 10:00:15;4500,11;180,7\n"
        "2025-01-15 10:00:20;4500,09;180,8\n",
        encoding="utf-8",
    )
    out = parse(p)
    assert len(out) == 5
    assert out["pressure"].dtype.kind == "f"
    assert abs(out["pressure"].iloc[0] - 4500.12) < 1e-6


def test_parser_class_metadata(tmp_path):
    """Parser exposes metadata + column mapping."""
    df = generate_synthetic_dst(n_samples=300, sample_period_s=10.0, seed=3)
    p = tmp_path / "x.csv"
    df.drop(columns=["true_event"], errors="ignore").to_csv(p, index=False)
    parser = WellTestParser()
    out = parser.parse(p)
    assert isinstance(parser.metadata, dict)
    assert isinstance(parser.mapping, dict)
    assert len(out) == len(df)
