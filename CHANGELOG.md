# Changelog

All notable changes to **welltest-pta** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-05-04

### Added

- Initial public release.
- `parse()` / `WellTestParser` — robust ASCII gauge file parser.
- `EventDetector` (V8.1) with `EventDetectorConfig` — Hampel + Savitzky–Golay,
  spike-boundary detection, net-ΔP signed classification, post-plateau tail
  trimming.
- `Event` and `EventCollection` — first-class objects with per-event
  `.bourdet()`, `.horner()`, `.mdh()`, `.flow_regimes()`,
  `.reservoir_params()`, `.export()`, and full plotting suite.
- `WellTest` orchestrator with `.from_file()`, `.from_dataframe()`,
  `.detect()`, `.split_manual()`, `.cross_validate()`, `.export_all()`,
  `.plot_composite()`.
- Analysis utilities:
  - `bourdet_derivative()` — log-derivative with smoothing parameter L
  - `horner_extrapolation()` / `horner_diagnostic_line()`
  - `mdh_extrapolation()`
  - `identify_flow_regimes()` — auto-classifies WBS / bilinear / linear /
    IARF / spherical / boundary
  - `reservoir_parameters()` — k, kh, skin, wellbore storage; plus
    `dimensionless_storage()` and `radius_of_investigation()`
  - `deconvolve()` / `DeconvolutionResult` — von Schroeter–Hollaender–
    Gringarten (2004) encoded `z`-formulation
- `cross_validate_detector()` — 3-component CV (bootstrap, Jaccard,
  parameter sensitivity) producing a 0–100 confidence score with grade.
- `welltest_pta.utils.synthetic.generate_synthetic_dst()` — multi-rate
  synthetic gauge data for testing and tutorials.
- Command-line tool `welltest-pta` with sub-commands `analyze`, `detect`,
  `deconvolve`, `synthetic`.
- Publication-quality 4-panel composite report
  (`welltest_pta.visualization.composite`).
