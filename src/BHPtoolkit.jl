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
export Rin, dRin, Rup, dRup
export scan_Binc, spectral_Binc_inv
export pochhammer

include("params.jl")
include("utils.jl")
include("recurrence.jl")
include("nu_solver.jl")
include("amplitudes.jl")
include("hypergeometric.jl")
include("radial_in.jl")
include("radial_up.jl")

end  # module BHPtoolkit
