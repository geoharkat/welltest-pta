"""
quick_start.py — three-line tour of welltest-pta.

This example uses synthetic data so it can run without any external file.
Replace the synthetic load with WellTest.from_file("yourfile.txt") for real
gauge data.
"""

from welltest_pta import WellTest
from welltest_pta.utils.synthetic import generate_synthetic_dst


# ── 1. Load (here: synthetic; in practice: from a gauge file) ────────────
df = generate_synthetic_dst(n_samples=10_000, sample_period_s=4.0)
wt = WellTest.from_dataframe(df)

# ── 2. Inspect the event catalogue ───────────────────────────────────────
wt.print_summary()

# ── 3. Per-event analysis on the longest buildup ─────────────────────────
bu = wt.events.longest_buildup
bu.print()

print("\n>>> Bourdet log-derivative")
dt, deriv = bu.bourdet(L=0.2)
print(f"    {len(dt)} derivative points covering Δt = "
      f"[{dt[0]:.4f}, {dt[-1]:.2f}] hr")

print("\n>>> Horner extrapolation")
h = bu.horner()
print(f"    P*    = {h['p_star']:.1f} psi")
print(f"    slope = {h['slope_m']:.2f} psi/cycle")
print(f"    R²    = {h['r2']:.4f}")

print("\n>>> Reservoir parameters (oilfield units)")
params = bu.reservoir_params(
    q=850, mu=0.45, B=1.18,
    h=18, phi=0.12, ct=1.2e-5, rw=0.108,
    method="horner",
)
print(f"    k    = {params['k']:.3f} mD")
print(f"    kh   = {params['kh']:.1f} mD·ft")
print(f"    skin = {params['skin']:+.3f}")

print("\n>>> Flow-regime identification")
for reg in bu.flow_regimes():
    print(f"    {reg['regime']:<22s} slope={reg['slope_mean']:+.2f}  "
          f"Δt = [{reg['dt_start']:.3f}, {reg['dt_end']:.3f}] hr")
