"""Tests for the Event and EventCollection classes."""

import pandas as pd
import pytest

from welltest_pta import Event, EventCollection


def test_eventcollection_indexable_by_int(fitted_wt):
    """EventCollection should be indexable like a list."""
    if len(fitted_wt.events) == 0:
        pytest.skip("no events in fixture")
    e0 = fitted_wt.events[0]
    assert isinstance(e0, Event)


def test_eventcollection_indexable_by_id(fitted_wt):
    """EventCollection should support lookup by event_id."""
    if len(fitted_wt.events) == 0:
        pytest.skip("no events in fixture")
    eid = fitted_wt.events[0].event_id
    e_by_id = fitted_wt.events[eid]
    assert e_by_id.event_id == eid


def test_eventcollection_iter_and_len(fitted_wt):
    n = sum(1 for _ in fitted_wt.events)
    assert n == len(fitted_wt.events)


def test_eventcollection_filters(fitted_wt):
    """drawdowns / buildups properties should partition the collection."""
    n_dd = len(fitted_wt.events.drawdowns)
    n_bu = len(fitted_wt.events.buildups)
    assert n_dd + n_bu == len(fitted_wt.events)


def test_event_summary_keys(fitted_wt):
    """summary() must contain the documented keys."""
    if len(fitted_wt.events) == 0:
        pytest.skip("no events")
    s = fitted_wt.events[0].summary()
    expected_keys = {
        "event_id", "type", "duration_hr", "p_initial",
        "p_final", "delta_p", "n_points",
    }
    assert expected_keys.issubset(s.keys())


def test_event_export_csv(fitted_wt, tmp_path):
    """Event.export() should write a CSV that round-trips."""
    if len(fitted_wt.events) == 0:
        pytest.skip("no events")
    e = fitted_wt.events[0]
    p = tmp_path / "ev.csv"
    e.export(p)
    out = pd.read_csv(p)
    assert "p_smooth" in out.columns
    assert len(out) == len(e.data)


def test_eventcollection_to_dataframe(fitted_wt):
    df = fitted_wt.events.to_dataframe()
    assert "event_id" in df.columns
    assert len(df) == len(fitted_wt.events)


def test_buildup_horner_runs(fitted_wt):
    """The longest buildup must yield a finite Horner P*."""
    bu = fitted_wt.events.longest_buildup
    if bu is None or bu.preceding_dd_dur_hr is None:
        pytest.skip("no buildup or no preceding DD")
    h = bu.horner()
    assert pd.notna(h["p_star"])
    assert pd.notna(h["slope_m"])
