# ============================================================
#  Point-particle source convolution + fluxes  (Track B5)
#
#  s = -2 circular equatorial orbits: convolve the Teukolsky point-particle
#  source with the homogeneous radial solutions (B1/B3) and the spin-weighted
#  spheroidal harmonic (B2), giving the asymptotic amplitudes Z^∞ (𝓘) and Z^H
#  (𝓗), then the energy and angular-momentum fluxes at infinity and the horizon.
#
#  Transcribed from Teukolsky-1.1.1 ConvolveSourcePointParticleCircular and
#  EnergyFlux/AngularMomentumFlux (M = 1). The orbit integral collapses to a
#  single evaluation at r0 = p, θ0 = π/2.
# ============================================================

"""
    KerrCircularOrbit(a, p; prograde=true)

Circular equatorial Kerr geodesic at radius p (reusing the B4 geodesics), with
energy E, axial angular momentum L, axial frequency Ωφ and Mino Υ_t.
"""
struct KerrCircularOrbit{R<:AbstractFloat}
    a::R; p::R; prograde::Bool
    E::R; L::R; Ωφ::R; Υt::R
end

function KerrCircularOrbit(a, p; prograde::Bool=true)
    R = float(promote_type(typeof(a), typeof(p)))
    x = prograde ? one(R) : -one(R)
    E, L, _ = kerr_geo_constants_of_motion(R(a), R(p), zero(R), x)
    Ωφ = kerr_geo_frequencies(R(a), R(p), zero(R), x).Omega_phi
    Υt = kerr_geo_mino_frequencies(R(a), R(p), zero(R), x).Upsilon_t
    KerrCircularOrbit{R}(R(a), R(p), prograde, E, L, Ωφ, Υt)
end

"""
    convolve_source_circular(tr, orbit) -> (ZInf, ZHor)

Z^∞ (→ infinity, 𝓘) and Z^H (→ horizon, 𝓗) for the s=-2 point particle on the
circular equatorial `orbit`, given the radial solution object `tr =
TeukolskyRadial(-2,l,m,a,ω)` (ω = m Ωφ).
"""
function convolve_source_circular(tr, orbit::KerrCircularOrbit)
    s, l, m = tr.s, tr.l, tr.m
    s == -2 || throw(ArgumentError("convolve_source_circular: only s=-2 implemented"))
    a, ω, λ = tr.a, tr.ω, tr.λ
    R  = real(typeof(ω))
    sqrt2 = sqrt(R(2))
    E, L, Υt = orbit.E, orbit.L, orbit.Υt
    r0 = orbit.p
    θ0 = R(π) / 2
    Δ  = r0^2 - 2r0 + a^2
    Kt = (r0^2 + a^2) * ω - m * a
    W  = 2im * ω * tr.amplitudes.Binc                     # invariant Wronskian

    # radial solutions at r0 (d²R from the radial ODE — reuse B3)
    RIn, dRIn = tr.In(r0), tr.In(r0; deriv=1)
    RUp, dRUp = tr.Up(r0), tr.Up(r0; deriv=1)
    d2RIn = _teuk_radial_rhs(r0, RIn, dRIn, s, m, a, ω, λ)[2]
    d2RUp = _teuk_radial_rhs(r0, RUp, dRUp, s, m, a, ω, λ)[2]

    # spheroidal harmonic and its θ-derivatives at θ0 (φ=0)
    S0   = SpinWeightedSpheroidalHarmonicS(s, l, m, a, ω, θ0; deriv=0)
    dS0  = SpinWeightedSpheroidalHarmonicS(s, l, m, a, ω, θ0; deriv=1)
    d2S0 = SpinWeightedSpheroidalHarmonicS(s, l, m, a, ω, θ0; deriv=2)

    sθ, cθ = sin(θ0), cos(θ0)
    L1  = -m/sθ + a*ω*sθ + cθ/sθ
    L2  = -m/sθ + a*ω*sθ + 2cθ/sθ
    L2S = dS0 + L2*S0
    L2p = m*cθ/sθ^2 + a*ω*cθ - 2/sθ^2
    L1Sp = d2S0 + L1*dS0
    L1L2S = L1Sp + L2p*S0 + L2*dS0 + L1*L2*S0

    ρ    = -1 / (r0 - im*a*cθ)
    ρbar = -1 / (r0 + im*a*cθ)
    Σ    = 1 / (ρ * ρbar)

    Ann0 = -ρ^(-2) * ρbar^(-1) * (sqrt2*Δ)^(-2) *
           (ρ^(-1)*L1L2S + 3im*a*sθ*L1*S0 + 3im*a*cθ*S0 + 2im*a*sθ*dS0 - im*a*sθ*L2*S0)
    Anmbar0 = ρ^(-3) * (sqrt2*Δ)^(-1) *
              ((ρ + ρbar - im*Kt/Δ)*L2S + (ρ - ρbar)*a*sθ*Kt/Δ*S0)
    Anmbar1 = -ρ^(-3) * (sqrt2*Δ)^(-1) * (L2S + im*(ρ - ρbar)*a*sθ*S0)
    Ambarmbar0 = (Kt^2*S0*ρbar)/(4Δ^2*ρ^3) +
                 (im*Kt*S0*(1 - r0 + Δ*ρ)*ρbar)/(2Δ^2*ρ^3) +
                 (im*r0*S0*ρbar*ω)/(2Δ*ρ^3)
    Ambarmbar1 = -ρ^(-3)*ρbar*S0/2 * (im*Kt/Δ - ρ)
    Ambarmbar2 = -ρ^(-3)*ρbar*S0/4

    rcomp = (E*(r0^2 + a^2) - a*L) / (2Σ)
    θcomp = ρ * (im*sθ*(a*E - L/sθ^2)) / sqrt2
    Cnn, Cnmbar, Cmbarmbar = rcomp^2, rcomp*θcomp, θcomp^2

    base(Rr, dRr, d2Rr) =
        (Ann0*Cnn + Anmbar0*Cnmbar + Ambarmbar0*Cmbarmbar)*Rr -
        (Anmbar1*Cnmbar + Ambarmbar1*Cmbarmbar)*dRr +
        Ambarmbar2*Cmbarmbar*d2Rr
    αIn = base(RIn, dRIn, d2RIn)
    αUp = base(RUp, dRUp, d2RUp)

    ZInf = -8*R(π)*αIn / W / Υt        # 𝓘 (→ ∞)  uses αIn
    ZHor = -8*R(π)*αUp / W / Υt        # 𝓗 (→ horizon) uses αUp
    return (ZInf=ZInf, ZHor=ZHor)
end

# Teukolsky-Starobinsky |C|² and the energy/ang-mom fluxes (s=-2), real ω.
function _fluxes_s2(l, m, a, ω, λ, ZInf, ZHor)
    rh = 1 + sqrt(1 - a^2)
    Ωh = a / (2rh)
    κ  = ω - m*Ωh
    ϵ  = sqrt(1 - a^2) / (4rh)
    # π at FULL working precision: `4π` is a Float64 literal (typeof(4π) ==
    # Float64), which floored BigFloat fluxes at ~3.9e-17 relative error.
    fourπ = 4 * promote_type(typeof(float(abs2(ZInf))), typeof(float(real(ω))))(π)
    FInf = abs2(ZInf) / (fourπ * ω^2)                      # ω^{2(1-|s|)} = ω^{-2}
    AbsCSq = ((λ+2)^2 + 4a*m*ω - 4a^2*ω^2) * (λ^2 + 36m*a*ω - 36a^2*ω^2) +
             (2λ+3) * (96a^2*ω^2 - 48m*a*ω) + 144*ω^2*(1 - a^2)
    α    = (256*(2rh)^5 * κ*(κ^2+4ϵ^2)*(κ^2+16ϵ^2)*ω^3) / AbsCSq
    FHor = α * abs2(ZHor) / (fourπ * ω^2)
    return (Inf=FInf, Hor=FHor)
end

"""
    TeukolskyPointParticleMode(s, l, m, a, p; prograde=true, nmax=80)

Sourced (s=-2) Teukolsky mode for a point particle on a circular equatorial
orbit at radius `p`. Returns a NamedTuple with the asymptotic amplitudes `Z`
(`ZInf`/`ZHor`), the `EnergyFlux` and `AngularMomentumFlux` (each `(Inf,Hor)`),
and the mode data (ω, λ, ...).
"""
function TeukolskyPointParticleMode(s::Int, l::Int, m::Int, a, p;
                                    prograde::Bool=true, nmax::Int=80)
    s == -2 || throw(ArgumentError("TeukolskyPointParticleMode: only s=-2 implemented"))
    orbit = KerrCircularOrbit(a, p; prograde=prograde)
    ω = m * orbit.Ωφ
    tr = TeukolskyRadial(s, l, m, a, ω; nmax=nmax)
    Z  = convolve_source_circular(tr, orbit)
    ωr, λr = real(tr.ω), real(tr.λ)
    EF = _fluxes_s2(l, m, a, ωr, λr, Z.ZInf, Z.ZHor)
    LF = ωr == 0 ? (Inf=zero(EF.Inf), Hor=zero(EF.Hor)) :
                   (Inf=EF.Inf*m/ωr, Hor=EF.Hor*m/ωr)
    return (s=s, l=l, m=m, a=tr.a, p=orbit.p, ω=ω, λ=tr.λ, orbit=orbit,
            Z=Z, EnergyFlux=EF, AngularMomentumFlux=LF)
end
