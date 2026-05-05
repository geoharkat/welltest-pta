"""
full_workflow.py — end-to-end pipeline demo.

Shows everything in sequence:
  load → cross-validate → manual override → per-event analytics →
  reservoir parameters → deconvolution → composite report → bulk export.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from welltest_pta import WellTest, deconvolve, EventDetectorConfig
from welltest_pta.utils.synthetic import generate_synthetic_dst


OUTPUT_DIR = Path("welltest_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── 1. Load gauge data ──────────────────────────────────────────────────
print("\n[1/7]  Loading gauge data…")
df = generate_synthetic_dst(
    n_samples=20_000,
    sample_period_s=4.0,
    sequence=[
        ("DD", 0.5, 3300.0),
        ("BU", 1.2, 4485.0),
        ("DD", 1.0, 3000.0),
        ("BU", 8.0, 4495.0),
    ],
)
df.drop(columns="true_event").to_csv(OUTPUT_DIR / "synthetic_input.csv", index=False)


# ─── 2. WellTest construction with cross-validation ─────────────────────
print("\n[2/7]  Auto-detect + cross-validate…")
cfg = EventDetectorConfig(
    hampel_sigma=3.0,
    spike_percentile=95.0,
    min_pta_dp_psi=15.0,
    tail_trim_enabled=True,
)
wt = WellTest.from_dataframe(df, cfg=cfg)
wt.cross_validate(n_bootstrap=4)


# ─── 3. Print catalogue ──────────────────────────────────────────────────
print("\n[3/7]  Event catalogue")
wt.print_summary()


# ─── 4. Per-event analysis ───────────────────────────────────────────────
print("\n[4/7]  Per-event analytics")
for ev in wt.events:
    print(f"\n   {ev}")
    if ev.event_type != "buildup":
        continue
    h = ev.horner()
    print(f"     Horner P*  = {h['p_star']:.1f} psi   m = {h['slope_m']:.2f}  R² = {h['r2']:.4f}")
    m = ev.mdh()
    print(f"     MDH p_1hr  = {m['intercept_p1hr']:.1f} psi   m = {m['slope_m']:.2f}  R² = {m['r2']:.4f}")
    fr = ev.flow_regimes()
    if fr:
        names = ", ".join(r["regime"] for r in fr)
        print(f"     Regimes    : {names}")


# ─── 5. Reservoir parameters on the longest BU ──────────────────────────
print("\n[5/7]  Reservoir parameters (longest BU)")
bu = wt.events.longest_buildup
params = bu.reservoir_params(
    q=850, mu=0.45, B=1.18,
    h=18, phi=0.12, ct=1.2e-5, rw=0.108,
    method="horner",
)
for k, v in params.items():
    if v is None or (isinstance(v, float) and v != v):  # NaN check
        print(f"     {k:<8s} = (n/a)")
    elif isinstance(v, float):
        print(f"     {k:<8s} = {v:+.4f}")
    else:
        print(f"     {k:<8s} = {v}")


# ─── 6. Deconvolution ────────────────────────────────────────────────────
print("\n[6/7]  Multi-event deconvolution")
res = deconvolve(wt.events, default_q=850, nu=1e-2, n_response_nodes=50)
print(f"     converged = {res.converged}, ||r|| = {res.residual_norm:.2f} psi")
res.export(OUTPUT_DIR / "deconvolution.csv")


# ─── 7. Plots + bulk export ──────────────────────────────────────────────
print("\n[7/7]  Plots and bulk export")
fig = wt.plot_composite(out_path=OUTPUT_DIR / "composite_report.pdf")
plt.close(fig)
fig = bu.plot_loglog(); plt.savefig(OUTPUT_DIR / "longest_BU_loglog.png", dpi=180); plt.close(fig)
fig = bu.plot_horner(); plt.savefig(OUTPUT_DIR / "longest_BU_horner.png", dpi=180); plt.close(fig)
fig = res.plot(); plt.savefig(OUTPUT_DIR / "deconvolution.png", dpi=180); plt.close(fig)

paths = wt.export_all(OUTPUT_DIR, prefix="full_workflow", per_event=True)
print(f"\n   Wrote {len(paths)} files to {OUTPUT_DIR}/")
print("\n✓ Pipeline complete.")
