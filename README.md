# welltest-pta

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![pip install welltest-pta](https://img.shields.io/badge/pip-install%20welltest--pta-orange.svg)](#installation)

**Pressure Transient Analysis (PTA) and Drill-Stem Test (DST) toolkit for Python.**

A complete, batteries-included pipeline for petroleum well-test interpretation:
robust ASCII parsing → automatic event detection → per-event analytics
(Bourdet derivative, Horner, MDH) → flow-regime identification → reservoir
parameters (`k`, `kh`, skin, wellbore storage) → multi-event deconvolution →
publication-quality plots.

---

## Highlights

- **Robust parser** — auto-detects delimiter / encoding / decimal style;
  handles raw electronic-gauge ASCII files, Sonatrach exports, KAPPA outputs,
  and most generic CSVs.
- **V8.1 event detector** — Hampel-filter despike + Savitzky–Golay smoothing,
  spike-boundary detection, net-ΔP signed classification, post-plateau tail
  trimming. Optional cross-validation reports a **0–100 confidence score**.
- **Manual override** — drop in your own list of `(type, t_start, t_end)`
  tuples whenever the auto-detector misclassifies.
- **Per-event API** — every drawdown / buildup is a first-class `Event`
  object with `.bourdet()`, `.horner()`, `.mdh()`, `.flow_regimes()`,
  `.reservoir_params()`, `.plot_loglog()`, `.plot_horner()`, `.export()`.
- **Multi-event deconvolution** — von Schroeter–Hollaender–Gringarten (2004)
  encoded `z`-formulation with second-difference regularisation. Merges
  any number of buildups + drawdowns into a single equivalent unit-rate
  response, dramatically extending the radius of investigation.
- **Publication-quality plots** — single-panel overview + 4-panel composite
  report (pressure, temperature, log–log diagnostic, BU pressure histogram).
- **Command-line interface** — `welltest-pta analyze DST.txt --cv --plot`.

---

## Installation

```bash
pip install welltest-pta
```

Or for development:

```bash
git clone https://github.com/geoharkat/welltest-pta.git
cd welltest-pta
pip install -e ".[dev]"
```

Optional Excel export support:

```bash
pip install "welltest-pta[excel]"
```

---

## Quick start

### 1. From an ASCII gauge file

```python
from welltest_pta import WellTest

wt = WellTest.from_file("DST_WELL-6.txt", cross_validate=True)
wt.print_summary()
```

```
════════════════════════════════════════════════════════════════════════
  WELL TEST SUMMARY
════════════════════════════════════════════════════════════════════════
  File:           DST_WELL-6.txt
  Samples:        58 432
  P_reservoir:    4 752.18 psi
  Noise floor:    0.94 psi
  Events:         8  (4 DD, 4 BU)
  CV score:       82.4 / 100  (HIGHLY ROBUST)
════════════════════════════════════════════════════════════════════════
event_id      type  duration_hr  p_initial  p_final  delta_p  rate_psi_hr   n_points
   DD-1  drawdown         0.50    4750.12  3300.40 -1449.72     -2899.45        408
   BU-1   buildup         1.00    3301.22  4488.10  1186.88     +1186.88        720
   DD-2  drawdown         1.00    4486.60  3010.55 -1476.05     -1476.05        720
   BU-2   buildup         8.42    3010.55  4495.30  1484.75      +176.41       6048
   ...
```

### 2. Per-event analysis

```python
bu = wt.events["BU-2"]                     # access by id
bu.print()
bu.plot_loglog()                           # log-log diagnostic

# Horner extrapolation
h = bu.horner()
print(f"P* = {h['p_star']:.1f} psi, m = {h['slope_m']:.2f} psi/cycle, R² = {h['r2']:.4f}")

# Reservoir parameters (oilfield units)
params = bu.reservoir_params(
    q=850, mu=0.45, B=1.18,                 # rate, viscosity, FVF
    h=18, phi=0.12, ct=1.2e-5, rw=0.108,    # net pay, porosity, ct, rw
    method="horner",
)
print(f"k = {params['k']:.2f} mD,  kh = {params['kh']:.1f} mD·ft,  skin = {params['skin']:+.2f}")
```

### 3. Manual splitting

When the auto-detector's CV score is marginal:

```python
wt.split_manual([
    ("DD",  "2025-01-15 10:00", "2025-01-15 12:30"),
    ("BU",  "2025-01-15 12:30", "2025-01-15 18:00"),
    ("DD",  "2025-01-15 18:00", "2025-01-15 20:00"),
    ("BU",  "2025-01-15 20:00", "2025-01-16 04:00"),
])
```

### 4. Multi-event deconvolution

```python
from welltest_pta import deconvolve

result = deconvolve(
    wt.events,                 # all DDs + BUs
    default_q=850,             # STB/D for any drawdown without ev.rate
    nu=1e-2,                   # regularisation weight
    n_response_nodes=60,
)
result.plot()                  # log-log of merged response
result.export("decon.csv")
```

### 5. Composite report + bulk export

```python
wt.plot_composite(out_path="DST_WELL-6_report.pdf")

wt.export_all(
    out_dir="./output",
    prefix="DST_WELL-6",
    per_event=True,            # also write CSV per event
)
```

### 6. Command-line

```bash
# Full analysis
welltest-pta analyze DST.txt -o ./results --cv --plot --per-event

# Just print catalogue
welltest-pta detect DST.txt

# Multi-event deconvolution
welltest-pta deconvolve DST.txt --q 850 --nu 1e-2 --plot decon.png

# Generate synthetic data for testing
welltest-pta synthetic -o synth.csv --n 10000
```

---

## Pipeline architecture

```
ASCII file
   │
   ▼  parse()
DataFrame  ┐
           │
           ▼  detect()      ──▶  cross_validate_detector()
                                 [CV score 0–100]
   annotated_df
           │
           ▼  EventCollection.from_annotated_dataframe()
   wt.events
   ├── DD-1 ──┐
   ├── BU-1   │  per event:
   ├── DD-2   ├──▶ .bourdet()   .horner()   .mdh()
   ├── BU-2   │   .flow_regimes()  .reservoir_params()
   └── ...   ─┘   .plot_loglog()   .export()
           │
           ▼  deconvolve(events)
   DeconvolutionResult
   (unit-rate response merging all events)
```

---

## Detector — V8.1 algorithm

The `WellTestEventDetector V8.1` has been validated on Rhourde Nouss
(Algeria) DSTs and offshore Qatar North Field tests:

| Phase | Step                                                              |
|-------|-------------------------------------------------------------------|
| 0     | Hampel-filter despike → Savitzky–Golay smoothing → noise σ̂       |
| 1     | Reservoir-pressure plateau detection                              |
| 2     | RIH / POOH edge masking                                           |
| 3     | Spike-boundary + turning-point detection (validated on ±5 σ̂)     |
| 4     | Zone classification using net-ΔP signed logic                     |
| 5     | Pause absorption → same-type merge → edge trimming                |
| 5b    | **V8.1**: Post-plateau tail trim (H→I→J spike before POOH)        |

Configuration via the `EventDetectorConfig` dataclass (defaults shown):

```python
from welltest_pta import EventDetectorConfig, WellTest

cfg = EventDetectorConfig(
    hampel_sigma=3.0,
    spike_percentile=95.0,
    min_pta_dp_psi=15.0,
    min_pta_duration_hr=0.10,
    tail_trim_enabled=True,
    tail_trim_min_dur_hr=4.0,
)

wt = WellTest.from_file("DST.txt", cfg=cfg)
```

---

## Cross-validation

Three independent stability checks merge into a 0–100 confidence index:

| Check                  | Method                                              | Weight |
|------------------------|-----------------------------------------------------|--------|
| Bootstrap event count  | K-fold downsample replicas → σ of n_DD, n_BU        | 0.40   |
| Jaccard edge overlap   | Overlap of "is-PTA" mask between bootstrap and ref  | 0.40   |
| Parameter sensitivity  | ±20 % sweep on key detector parameters              | 0.20   |

| Score    | Grade                                       |
|----------|---------------------------------------------|
| 80–100   | **HIGHLY ROBUST** — manual review optional  |
| 60–80    | **REASONABLE** — spot-check critical events |
| 40–60    | **MARGINAL** — recommend manual splitting   |
| 0–40     | **UNSTABLE** — manual splitting strongly advised |

---

## Deconvolution — vSH04

The implementation follows von Schroeter, Hollaender & Gringarten (2004),
solving the encoded non-linear least-squares problem

$$
\min_{\mathbf{z}, p_i}\;\| \mathbf{y} - C(q,\mathbf{z}) - p_i \|^2 + \nu\,\| D\,\mathbf{z}\|^2
$$

with $z(\sigma) = \ln\!\left[dp_u/d\ln t\right]$ on a log-spaced response
grid $\sigma = \ln t$. Positivity of the derivative is enforced by
construction; second-difference regularisation $D$ controls smoothness.

```python
from welltest_pta import deconvolve

res = deconvolve(
    wt.events,
    default_q=850,            # STB/D
    nu=1e-2,                  # regularisation
    n_response_nodes=60,      # log-spaced grid
    fit_p_initial=True,       # solve p_i jointly with z
)

print(f"converged = {res.converged}, iters = {res.iterations}")
print(f"||r|| = {res.residual_norm:.2f} psi,  p_i = {res.p_initial:.1f} psi")
```

---

## Citation

If you use `welltest-pta` in research, please cite:

```bibtex
@software{harkat2026welltestpta,
  author = {Harkat, Ismail},
  title  = {welltest-pta: Pressure Transient Analysis Toolkit for Python},
  year   = {2026},
  url    = {https://github.com/geoharkat/welltest-pta},
  version = {0.1.0},
}
```

Algorithmic references:

- Bourdet, D., Ayoub, J. A., & Pirard, Y. M. (1989).
  *Use of pressure derivative in well-test interpretation.* SPE Formation
  Evaluation **4** (2), 293–302.
- Horner, D. R. (1951). *Pressure build-up in wells.* Proc. 3rd World
  Petroleum Congress, The Hague.
- von Schroeter, T., Hollaender, F., & Gringarten, A. C. (2004).
  *Deconvolution of well-test data as a nonlinear total least-squares
  problem.* SPE Journal **9** (4), 375–390.

---

## Roadmap

- [ ] Type-curve matching (Gringarten / Bourdet)
- [ ] Boundary-model detection (linear, parallel, intersecting faults)
- [ ] Gas / two-phase pseudopressure
- [ ] KAPPA-Saphir LAS round-trip export
- [ ] Web UI (FastAPI + Plotly Dash)

Pull requests welcome.

---

## License

MIT — see [LICENSE](LICENSE).

## Author

**Ismail Harkat** — Senior Wellsite/Operations Geologist, Sonatrach
(Rhourde Nouss field, Algeria). Adapted from the production V8.1 detector
+ analytics pipeline used on WELL-6 wells.
