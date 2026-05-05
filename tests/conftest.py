"""Shared pytest fixtures."""

import pytest

from welltest_pta.utils.synthetic import generate_synthetic_dst


@pytest.fixture(scope="module")
def synth_df():
    """
    A small synthetic DST DataFrame used across tests.

    Uses the default 4-event sequence (DD-BU-DD-BU with the long final BU
    of 6 hr) which the V8.1 detector reliably classifies on this density.
    """
    return generate_synthetic_dst(
        n_samples=8_000,
        sample_period_s=4.0,
        seed=42,
    )


@pytest.fixture(scope="module")
def fitted_wt(synth_df):
    """A WellTest with detection already run."""
    from welltest_pta import WellTest
    return WellTest.from_dataframe(synth_df, auto_detect=True)
