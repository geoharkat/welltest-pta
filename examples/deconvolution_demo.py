"""
deconvolution_demo.py — multi-event deconvolution merging multiple buildups.

This example shows how deconvolve() takes all DD/BU events from a single
test and merges them into one equivalent unit-rate response — the
"diagnostic master plot" of the entire test.
"""

import matplotlib.pyplot as plt

from welltest_pta import WellTest, deconvolve
from welltest_pta.utils.synthetic import generate_synthetic_dst


# ── 1. Multi-rate synthetic test with three buildups ─────────────────────
df = generate_synthetic_dst(
    n_samples=24_000,
    sample_period_s=4.0,
    sequence=[
        ("DD", 0.4, 3500.0),
        ("BU", 0.8, 4480.0),
        ("DD", 0.8, 3100.0),
        ("BU", 1.5, 4490.0),
        ("DD", 1.0, 2900.0),
        ("BU", 8.0, 4495.0),
    ],
)
wt = WellTest.from_dataframe(df)
wt.print_summary()

# ── 2. Deconvolve ────────────────────────────────────────────────────────
print("\n→ Running vSH04 deconvolution …")
res = deconvolve(
    wt.events,
    default_q=850,                # STB/D for drawdowns
    nu=1e-2,                      # regularisation
    n_response_nodes=60,
    fit_p_initial=True,
)
print(f"   converged = {res.converged}, iters = {res.iterations}")
print(f"   p_initial = {res.p_initial:.1f} psi   ||r|| = {res.residual_norm:.2f} psi")

# ── 3. Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: observed pressure + reconstruction
axes[0].plot(res.obs_time, res.obs_pressure, "o", ms=1.5,
             color="#888888", label="Observed")
axes[0].plot(res.obs_time, res.fit_pressure, "-", lw=1.0,
             color="#1f77b4", label="Reconstructed")
axes[0].set_xlabel("Elapsed time (hr)")
axes[0].set_ylabel("Pressure (psi)")
axes[0].set_title("Pressure: observed vs deconvolution reconstruction")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: log-log diagnostic of the recovered unit-rate response
res.plot(ax=axes[1])

plt.tight_layout()
plt.savefig("deconvolution_demo.png", dpi=200, bbox_inches="tight")
print("\n→ Saved deconvolution_demo.png")
