r"""
welltest_pta.cli
================
Command-line interface for the welltest-pta package.

After installation, a ``welltest-pta`` command becomes available::

    $ welltest-pta DST_file.txt --output ./results --plot --cv

Sub-commands
------------
analyze      Full pipeline: parse → detect → CV → plot → export
detect       Detection only — print event catalogue
deconvolve   Run multi-event deconvolution on a previously analysed test
synthetic    Generate a synthetic DST file (for testing / demos)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("welltest_pta")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────────────
# analyze
# ─────────────────────────────────────────────────────────────────────────────

def cmd_analyze(args: argparse.Namespace) -> int:
    from welltest_pta import WellTest

    out_dir = Path(args.output) if args.output else Path("welltest_pta_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n→ Loading {args.file}")
    wt = WellTest.from_file(
        args.file,
        cross_validate=args.cv,
        cv_n_bootstrap=args.cv_n,
        cv_print=True,
    )
    wt.print_summary()

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            print("\n→ Composite figure")
            fig = wt.plot_composite(
                out_path=out_dir / f"{Path(args.file).stem}_composite.pdf"
            )
            plt.close(fig)
        except Exception as e:
            print(f"  ⚠️  Plotting failed: {e}", file=sys.stderr)

    print(f"\n→ Exporting to {out_dir}")
    paths = wt.export_all(out_dir, prefix=Path(args.file).stem,
                          per_event=args.per_event)
    for label, p in paths.items():
        print(f"   • {label:<22s}{p}")

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# detect
# ─────────────────────────────────────────────────────────────────────────────

def cmd_detect(args: argparse.Namespace) -> int:
    from welltest_pta import WellTest

    wt = WellTest.from_file(args.file)
    wt.events.print()
    print()
    if args.export:
        wt.events.export(args.export)
        print(f"   → catalogue exported to {args.export}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# deconvolve
# ─────────────────────────────────────────────────────────────────────────────

def cmd_deconvolve(args: argparse.Namespace) -> int:
    from welltest_pta import WellTest, deconvolve

    wt = WellTest.from_file(args.file)
    print(f"\n→ Deconvolving {len(wt.events)} events (q = {args.q} STB/D, ν = {args.nu})")
    res = deconvolve(
        wt.events,
        default_q=args.q,
        nu=args.nu,
        n_response_nodes=args.n_nodes,
        verbose=args.verbose,
    )
    print(f"   converged = {res.converged}, iters = {res.iterations}, "
          f"||r|| = {res.residual_norm:.2f} psi")
    if args.export:
        res.export(args.export)
        print(f"   → response saved to {args.export}")
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig = res.plot()
            outfile = args.plot if isinstance(args.plot, str) else "deconvolution.png"
            fig.savefig(outfile, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"   → plot saved to {outfile}")
        except Exception as e:
            print(f"  ⚠️  Plot failed: {e}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# synthetic
# ─────────────────────────────────────────────────────────────────────────────

def cmd_synthetic(args: argparse.Namespace) -> int:
    from welltest_pta.utils.synthetic import generate_synthetic_dst

    df = generate_synthetic_dst(
        n_samples=args.n,
        sample_period_s=args.dt,
        seed=args.seed,
    )
    out_path = Path(args.output)
    df_to_save = df.drop(columns=["true_event"], errors="ignore")
    df_to_save.to_csv(out_path, index=False)
    print(f"   → synthetic DST written to {out_path}  ({len(df)} rows)")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="welltest-pta",
        description="Pressure Transient Analysis & DST toolkit (V8.1 detector + vSH04 deconvolution).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")

    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p1 = sub.add_parser("analyze", help="Full pipeline (parse → detect → CV → plot → export).")
    p1.add_argument("file", help="ASCII gauge file to analyse.")
    p1.add_argument("--output", "-o", default=None, help="Output directory.")
    p1.add_argument("--cv", action="store_true", help="Run cross-validation.")
    p1.add_argument("--cv-n", type=int, default=8, help="Bootstrap replicas (default 8).")
    p1.add_argument("--plot", action="store_true", help="Save composite PDF figure.")
    p1.add_argument("--per-event", action="store_true",
                    help="Also export per-event CSVs.")
    p1.set_defaults(func=cmd_analyze)

    # detect
    p2 = sub.add_parser("detect", help="Print the event catalogue and (optionally) export.")
    p2.add_argument("file", help="ASCII gauge file.")
    p2.add_argument("--export", default=None,
                    help="Path for the catalogue CSV/Excel/JSON.")
    p2.set_defaults(func=cmd_detect)

    # deconvolve
    p3 = sub.add_parser("deconvolve", help="Multi-event deconvolution.")
    p3.add_argument("file", help="ASCII gauge file.")
    p3.add_argument("--q", type=float, required=True, help="Flow rate (STB/D) for drawdowns.")
    p3.add_argument("--nu", type=float, default=1e-2, help="Regularisation weight ν.")
    p3.add_argument("--n-nodes", type=int, default=60, help="Number of log-spaced response nodes.")
    p3.add_argument("--export", default=None, help="Save response CSV/Excel/JSON.")
    p3.add_argument("--plot", default=None, nargs="?", const="deconvolution.png",
                    help="Save log-log diagnostic of the recovered response.")
    p3.set_defaults(func=cmd_deconvolve)

    # synthetic
    p4 = sub.add_parser("synthetic", help="Generate a synthetic DST CSV file.")
    p4.add_argument("--output", "-o", default="synthetic_dst.csv", help="Output file path.")
    p4.add_argument("--n", type=int, default=18000, help="Number of samples.")
    p4.add_argument("--dt", type=float, default=5.0, help="Sampling period (s).")
    p4.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p4.set_defaults(func=cmd_synthetic)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
