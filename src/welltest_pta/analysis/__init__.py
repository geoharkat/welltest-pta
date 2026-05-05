"""Analysis submodule — Bourdet, Horner, MDH, flow regimes, reservoir, deconvolution."""

from welltest_pta.analysis.bourdet import bourdet_derivative
from welltest_pta.analysis.horner import horner_extrapolation, horner_diagnostic_line
from welltest_pta.analysis.mdh import mdh_extrapolation
from welltest_pta.analysis.flow_regimes import identify_flow_regimes
from welltest_pta.analysis.reservoir import reservoir_parameters
from welltest_pta.analysis.deconvolution import deconvolve, DeconvolutionResult

__all__ = [
    "bourdet_derivative",
    "horner_extrapolation",
    "horner_diagnostic_line",
    "mdh_extrapolation",
    "identify_flow_regimes",
    "reservoir_parameters",
    "deconvolve",
    "DeconvolutionResult",
]
