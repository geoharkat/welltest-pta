"""
manual_split.py — manual override of the auto-detector.

Demonstrates the workflow when the cross-validation score is marginal:
1. Run auto-detect first (required — sets up p_smooth / elapsed_hr columns).
2. Inspect the catalogue and decide which events you want to override.
3. Pass a list of (type, t_start, t_end) tuples to wt.split_manual().
"""

from welltest_pta import WellTest, EventDetectorConfig
from welltest_pta.utils.synthetic import generate_synthetic_dst


# ── 1. Synthetic test with 4 events (DD-BU-DD-BU) ────────────────────────
df = generate_synthetic_dst(
    n_samples=20_000,
    sample_period_s=3.0,
    sequence=[
        ("DD", 0.5, 3300.0),   # short flow #1
        ("BU", 1.0, 4490.0),   # short BU
        ("DD", 1.0, 3000.0),   # main flow
        ("BU", 8.0, 4495.0),   # extended BU
    ],
)

# ── 2. Auto-detection first ──────────────────────────────────────────────
wt = WellTest.from_dataframe(df, auto_detect=True)
print("[AUTO]")
wt.events.print()

# ── 3. Suppose the auto-detector misclassified BU-1 as part of BU-2.
#     We override with explicit timestamps.
ts = df["timestamp"]

# Find indices roughly bracketing the four events (the synthetic generator
# places them in order with small inter-segment gaps).
idx_dd1_a, idx_dd1_b = 4_500, 5_500
idx_bu1_a, idx_bu1_b = 5_700, 7_500
idx_dd2_a, idx_dd2_b = 7_700, 9_000
idx_bu2_a, idx_bu2_b = 9_100, 18_500

wt.split_manual([
    ("DD", ts.iloc[idx_dd1_a], ts.iloc[idx_dd1_b]),
    ("BU", ts.iloc[idx_bu1_a], ts.iloc[idx_bu1_b]),
    ("DD", ts.iloc[idx_dd2_a], ts.iloc[idx_dd2_b]),
    ("BU", ts.iloc[idx_bu2_a], ts.iloc[idx_bu2_b]),
])

print("\n[MANUAL OVERRIDE]")
wt.events.print()

# ── 4. Per-event analysis is now identical to the auto-detect case ─────
bu_long = wt.events.longest_buildup
print(f"\nLongest BU = {bu_long.event_id}, duration = {bu_long.duration_hr:.2f} hr")
print(f"Preceding tp = {bu_long.preceding_dd_dur_hr:.3f} hr")

h = bu_long.horner()
print(f"Horner: P* = {h['p_star']:.1f} psi, m = {h['slope_m']:.2f}, R² = {h['r2']:.4f}")
