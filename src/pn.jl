# ============================================================
#  Post-Newtonian (low-frequency) series layer  (Track B6)
#
#  Strategy: run the EXISTING, Wolfram-validated MST 3-term recurrence
#  (αn/βn/γn) over the truncated power-series ring PNSeries{T} in ε = 2ω.
#  Because the minimal solution has fₙ = O(ε^{|n|}), the continued fractions
#  for Rₙ=fₙ/f_{n-1} and Lₙ=fₙ/f_{n+1} TERMINATE at |n| > order — so they are
#  finite recurrences over the ring (no convergence loop needed), and the
#  renormalized ν falls out by a modified Newton iteration whose leading
#  derivative is ∂_ν[ν(ν+1)] = 2l+1.
#
#  SCOPE (this version): Schwarzschild (a=0), l≥1. There the angular eigenvalue
#  is exactly λ = l(l+1)-s(s+1) (constant), and the PN ν/aₙ series match the
#  Wolfram reference to full precision. Kerr (a≠0) requires the spin-weighted
#  spheroidal eigenvalue's c=aω expansion in the exact SWSH convention (the
#  O(c²) seed used elsewhere does NOT reproduce it) and is not yet supported;
#  l=0 (scalar monopole) hits a small-index recurrence-pole straddle. Both are
#  guarded with a clear error rather than returning wrong values.
# ============================================================

function _pn_guard(s::Int, l::Int, a)
    iszero(a) || throw(ArgumentError(
        "PN series: Kerr (a≠0) not yet supported — needs the spin-weighted " *
        "spheroidal eigenvalue c=aω expansion; use a=0 (Schwarzschild)."))
    l ≥ 1 || throw(ArgumentError(
        "PN series: l=0 monopole not supported (small-index recurrence pole); use l≥1."))
end

# Angular eigenvalue λ = A_lm as a (constant) PNSeries — Schwarzschild, exact.
_pn_lambda(s::Int, l::Int, m::Int, a, order::Int, ::Type{T}) where {T} =
    pnconst(T(l*(l+1) - s*(s+1)), order, T)

# PNSeries-valued MST context usable by the generic αn/βn/γn.
function _pn_context(s::Int, l::Int, m::Int, a, order::Int, ::Type{T}) where {T}
    ε = pneps(order, T)
    q = T(a)
    κ = sqrt(pnconst(one(T) - q^2, order, T))
    τ = (ε - m*q) / κ
    λ = _pn_lambda(s, l, m, a, order, T)
    return (s = s, m = m, q = q, ϵ = ε, κ = κ, τ = τ, λ = λ)
end

# Finite continued fractions over the series ring (terminate beyond ±nmax).
function _pn_R(ctx, ν, n::Int, nmax::Int)
    R = zero(ν)
    for k in nmax:-1:n
        R = -γn(ctx, ν, k) / (βn(ctx, ν, k) + αn(ctx, ν, k) * R)
    end
    return R
end
function _pn_L(ctx, ν, n::Int, nmax::Int)
    L = zero(ν)
    for k in -nmax:n
        L = -αn(ctx, ν, k) / (βn(ctx, ν, k) + γn(ctx, ν, k) * L)
    end
    return L
end

# Characteristic function g(ν) = β₀ + α₀ R₁ + γ₀ L₋₁ (its series must vanish).
_pn_g(ctx, ν, nmax::Int) =
    βn(ctx, ν, 0) + αn(ctx, ν, 0) * _pn_R(ctx, ν, 1, nmax) +
                    γn(ctx, ν, 0) * _pn_L(ctx, ν, -1, nmax)

"""
    nu_pn(s, l, m, a; order=4, T=Complex{BigFloat}) -> PNSeries

Renormalized angular momentum ν as a low-frequency series, ν = l + Σ_{i≥2} dᵢ εⁱ
(ε = 2ω). Exact for Schwarzschild; reliable through ε² for Kerr (see module note).
"""
function nu_pn(s::Int, l::Int, m::Int, a; order::Int=4, T::Type=Complex{BigFloat})
    _pn_guard(s, l, a)
    ctx  = _pn_context(s, l, m, a, order, T)
    nmax = order + 2
    # Seed with a nonzero ε² term: at ν=l the index n=-l gives n+ν=0 exactly and
    # the βₙ coupling term 1/((n+ν)(n+ν+1)) is 0/0. A nonzero ε² makes n+ν have
    # valuation 2 (Laurent-safe); Newton then corrects the coefficient.
    ν    = pnconst(T(l), order, T) + pneps(order, T)^2
    invd = one(T) / T(2l + 1)                  # 1/g'(ν)|_{leading}
    for _ in 1:(order + 6)                      # modified Newton: +1 order/step
        ν = ν - _pn_g(ctx, ν, nmax) * invd
    end
    return ν
end

"""
    an_pn(s, l, m, a; order=4, T=Complex{BigFloat}) -> Dict{Int,PNSeries}

MST series coefficients a_n^ν (== f_n^ν, with a₀=1) as low-frequency series.
"""
function an_pn(s::Int, l::Int, m::Int, a; order::Int=4, T::Type=Complex{BigFloat})
    _pn_guard(s, l, a)
    ctx  = _pn_context(s, l, m, a, order, T)
    nmax = order + 2
    ν    = nu_pn(s, l, m, a; order=order, T=T)
    f    = Dict{Int,PNSeries{T}}()
    f[0] = one(ν)
    for n in 1:nmax
        f[n] = _pn_R(ctx, ν, n, nmax) * f[n-1]
    end
    for n in -1:-1:-nmax
        f[n] = _pn_L(ctx, ν, n, nmax) * f[n+1]
    end
    return f
end

"""
    lambda_pn(s, l, m, a; order=4, T=Complex{BigFloat}) -> PNSeries

Angular eigenvalue λ = A_lm as a low-frequency series (see spin note).
"""
lambda_pn(s::Int, l::Int, m::Int, a; order::Int=4, T::Type=Complex{BigFloat}) =
    _pn_lambda(s, l, m, a, order, T)
