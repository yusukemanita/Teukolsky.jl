# ============================================================
#  Spin-weighted spheroidal harmonics  (Track B2)
#
#  S_{slm}(θ, φ; aω) = Σ_{ℓ′} C_{ℓ′} · ₛY_{ℓ′m}(θ, φ)
#
#  - eigenvalue λ and coefficients C from the BCS spectral matrix (_swsh_eigen)
#  - ₛY_{ℓm} spin-weighted spherical harmonics via the Goldberg et al. (1967)
#    closed form, evaluated to the working float precision
#
#  Normalization: ∫_{S²} |S|² dΩ = 1  (the C are unit 2-norm and the ₛY are
#  orthonormal). Phase: S_lm → ₛY_lm as aω → 0 (ℓ′=ℓ coefficient real positive).
# ============================================================

"""
    sYlm(s, l, m, θ; φ=0)

Spin-weighted spherical harmonic ₛY_{lm}(θ, φ), evaluated to the precision of
`θ` (Float64, BigFloat, …). Uses the Goldberg et al. (1967) finite sum

    ₛY_{lm} = (-1)^m √[(l+m)!(l-m)!(2l+1) / (4π (l+s)!(l-s)!)] e^{imφ}
              Σ_r C(l-s,r) C(l+s, r+s-m) (-1)^{l-r-s}
                  cos(θ/2)^{2r+s-m} sin(θ/2)^{2l-2r-s+m}

(orthonormal over the sphere; reduces to the usual Yₗₘ for s=0).
"""
# `deriv`-th θ-derivative of cos(θ/2)^p sin(θ/2)^q (h=θ/2 ⇒ each d/dθ carries ½).
# Zero coefficients (p,q small) kill the would-be negative exponents.
function _cshalf_deriv(cz::R, sz::R, p::Int, q::Int, deriv::Int) where {R}
    pw(b, e) = e < 0 ? zero(R) : b^e
    if deriv == 0
        return cz^p * sz^q
    elseif deriv == 1
        return (q * pw(cz, p+1) * pw(sz, q-1) - p * pw(cz, p-1) * pw(sz, q+1)) / 2
    else
        c0 = q*(q-1); c1 = q*(p+1) + p*(q+1); c2 = p*(p-1)
        return (c0 * pw(cz, p+2) * pw(sz, q-2) -
                c1 * pw(cz, p)   * pw(sz, q)   +
                c2 * pw(cz, p-2) * pw(sz, q+2)) / 4
    end
end

# θ-only (real) part of the Goldberg sum: pref · Σ_r …, in the precision of θ.
function _sYlm_theta(s::Int, l::Int, m::Int, θ, deriv::Int)
    R = typeof(float(real(θ)))
    fac(n) = R(factorial(big(n)))
    pref = R((-1)^m) * sqrt(fac(l+m) * fac(l-m) * R(2l+1) /
                            (4 * R(π) * fac(l+s) * fac(l-s)))

    half = R(θ) / 2
    cz, sz = cos(half), sin(half)
    total = zero(R)
    for r in max(0, m - s):min(l - s, l + m)
        ce = 2r + s - m          # cos(θ/2) exponent
        se = 2l - 2r - s + m     # sin(θ/2) exponent
        (ce < 0 || se < 0) && continue
        total += R(binomial(big(l - s), big(r))) *
                 R(binomial(big(l + s), big(r + s - m))) *
                 R((-1)^(l - r - s)) * _cshalf_deriv(cz, sz, ce, se, deriv)
    end
    return pref * total
end

# The alternating Goldberg sum loses ≈ 1.05·l bits to cancellation (measured:
# 22 bits at l=20, 104 at l=100, 154 at l=150 — worst over θ, s, m grids), so a
# direct Float64 evaluation degrades from ~5e-13 rel. error at l=12 to O(1)
# garbage at l=60 and factorial-overflow NaN beyond l≈120.  Above this threshold
# Float64/Float32/Float16 inputs are evaluated internally in BigFloat with
# l + 64 guard bits over the target precision (covers the cancellation up to
# l ≈ 1200) and rounded back.  Measured direct-path worst generic-point rel.
# error at l = 8 is 4.7e-13; the promoted path is ≤ ~1e-15 for all l tested
# (≤ 150).  BigFloat/Arb inputs keep the exact type-generic path unchanged.
const _SYLM_DIRECT_LMAX = 8

function sYlm(s::Int, l::Int, m::Int, θ; φ=0, deriv::Int=0)
    R = typeof(float(real(θ)))
    (l < abs(s) || abs(m) > l) && return zero(Complex{R})
    deriv in (0, 1, 2) || throw(ArgumentError("deriv must be 0, 1, or 2"))

    y = if R <: Union{Float16, Float32, Float64} && l > _SYLM_DIRECT_LMAX
        setprecision(BigFloat, l + 64 + precision(R)) do
            R(_sYlm_theta(s, l, m, BigFloat(R(θ)), deriv))
        end
    else
        _sYlm_theta(s, l, m, θ, deriv)
    end
    return Complex{R}(y) * cis(R(m) * R(φ))
end

"""
    SpinWeightedSpheroidalEigenvalue(s, l, m, γ)

Angular separation constant λ = A_lm of the spin-weighted spheroidal equation
with oblateness `γ = aω` (a thin alias for [`compute_lambda`](@ref) taking γ
directly, matching the Mathematica `SpinWeightedSpheroidalEigenvalue` signature).
"""
SpinWeightedSpheroidalEigenvalue(s::Int, l::Int, m::Int, γ; l_max::Int=0) =
    compute_lambda(s, l, m, one(real(typeof(float(real(complex(γ)))))), γ; l_max=l_max)

"""
    swsh_coefficients(s, l, m, a, ω; l_max=0) -> (ells, C)

Spherical-harmonic expansion coefficients of S_{slm}: `S = Σ C[i]·ₛY_{ells[i],m}`.
Unit 2-norm, phase-fixed so the ℓ′=ℓ term is real positive.
`l_max ≤ 0` (default) sizes the ℓ′ basis adaptively (see [`compute_lambda`](@ref)).
"""
swsh_coefficients(s::Int, l::Int, m::Int, a, ω; l_max::Int=0) =
    (e = _swsh_eigen(s, l, m, a, ω; l_max=l_max); (e[2], e[3]))

"""
    SpinWeightedSpheroidalHarmonicS(s, l, m, a, ω, θ; φ=0, deriv=0, l_max=0)

Spin-weighted spheroidal harmonic S_{slm}(θ, φ) for oblateness aω, to working
precision. `deriv` ∈ {0,1,2} returns the θ-derivative ∂^deriv S / ∂θ^deriv.
(Pass BigFloat a/ω/θ for a BigFloat harmonic.)
`l_max ≤ 0` (default) sizes the ℓ′ basis adaptively (see [`compute_lambda`](@ref)).
"""
function SpinWeightedSpheroidalHarmonicS(s::Int, l::Int, m::Int, a, ω, θ;
                                         φ=0, deriv::Int=0, l_max::Int=0)
    ells, C = swsh_coefficients(s, l, m, a, ω; l_max=l_max)
    R = promote_type(typeof(float(real(a))), typeof(float(real(complex(ω)))),
                     typeof(float(real(θ))))
    S = zero(Complex{R})
    for (i, lp) in enumerate(ells)
        ci = C[i]
        iszero(ci) && continue
        S += Complex{R}(ci) * sYlm(s, lp, m, R(θ); φ=φ, deriv=deriv)
    end
    return S
end
