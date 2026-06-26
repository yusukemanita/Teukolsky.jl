"""
    BHPtoolkit — Mano-Suzuki-Takasugi formalism for the Teukolsky equation

Computes:
- Renormalized angular momentum ν
- Asymptotic amplitudes B^inc, B^ref, B^trans, C^trans
- Homogeneous radial solutions R_in (ingoing) and R_up (upgoing)

Following Sasaki & Tagoshi, Living Rev. Rel. 6 (2003) 6.
Convention: M = 1 (black hole mass), G = c = 1.
"""
module BHPtoolkit

using LinearAlgebra, SpecialFunctions, Printf

export MSTParams, compute_nu, compute_fn, compute_fn_truncated
export compute_amplitudes, compute_amplitudes_nufixed
export compute_amplitudes_mero, compute_amplitudes_nufixed_mero
export compute_q, compute_qtilde
export Rin, dRin, Rup, dRup, Rdown
export scan_Binc, spectral_Binc_inv
export pochhammer
export compute_lambda, sYlm, swsh_coefficients
export SpinWeightedSpheroidalEigenvalue, SpinWeightedSpheroidalHarmonicS
export TeukolskyRadial, TeukolskyRadialFunction
export NumericalIntegrationRadial

include("params.jl")
include("utils.jl")
include("recurrence.jl")
include("nu_solver.jl")
include("amplitudes.jl")
include("branch_cut.jl")
include("hypergeometric.jl")
include("radial_in.jl")
include("radial_up.jl")
include("radial_down.jl")
include("spheroidal.jl")
include("teukolsky_radial.jl")
include("numint_radial.jl")
include("waveform.jl")

using .Waveform
export WaveformParams, compute_waveform, green_function

end  # module BHPtoolkit
