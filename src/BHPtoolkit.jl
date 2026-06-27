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
export rf, rd, rj, rc
export ellK, ellE, ellF, ellEinc, ellPi
export jacobi_sn, jacobi_am
export kerr_geo_energy, kerr_geo_angular_momentum
export kerr_geo_carter_constant, kerr_geo_constants_of_motion
export kerr_geo_radial_roots, kerr_geo_polar_roots
export kerr_geo_mino_frequencies, kerr_geo_boyer_lindquist_frequencies
export kerr_geo_frequencies
export KerrGeoOrbit, KerrGeoOrbitFunction
export kerr_geo_isco, kerr_geo_photon_sphere_radius
export kerr_geo_ibso, kerr_geo_isso, kerr_geo_separatrix
export PNSeries, pneps, pnconst, getcoeff, evalseries, gamma_ratio
export nu_pn, an_pn, lambda_pn
export KerrCircularOrbit, convolve_source_circular, TeukolskyPointParticleMode

include("params.jl")
include("utils.jl")
include("elliptic_integrals.jl")  # shared elliptic/Jacobi (before geodesic files)
include("recurrence.jl")
include("pn_series.jl")   # PNSeries ring (B6)
include("pn.jl")          # post-Newtonian low-frequency series (B6)
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
include("kerr_geo_constants.jl")  # geodesic constants of motion (after numint_radial)
include("kerr_geo_frequencies.jl")  # radial/polar roots + Mino & BL frequencies
include("special_orbits.jl")        # ISCO, photon sphere, IBSO, ISSO, separatrix
include("kerr_geo_orbit.jl")        # callable KerrGeoOrbit trajectory
include("teukolsky_mode.jl")        # B5: source convolution + fluxes
include("waveform.jl")

using .Waveform
export WaveformParams, compute_waveform, green_function

end  # module BHPtoolkit
