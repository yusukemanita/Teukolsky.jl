# ============================================================
#  Hypergeometric functions for MST radial solutions
#
#  - Gauss 2F1 for Rin (via HypergeometricFunctions.jl + recurrence)
#  - Tricomi U for Rup (via Kummer relation + recurrence)
#
#  Teukolsky parameters:
#    2F1: aF = ν+1-iτ, bF = -ν-iτ, cF = 1-s-i(ε+τ)
#    U:   aU = ν+s+1-iε, bU = 2ν+2, argument c = -2i·ẑ
# ============================================================

using HypergeometricFunctions: _₂F₁, _₁F₁

# ============================================================
#  HypergeometricU via Kummer relation
#  U(a,b,z) = Γ(1-b)/Γ(a+1-b) M(a,b,z)
#           + Γ(b-1)/Γ(a) z^(1-b) M(a+1-b, 2-b, z)
# ============================================================

"""
    hypergeometric_U_asymptotic(a, b, z; nterms=80)

Asymptotic expansion of U(a, b, z) for large |z| with optimal truncation:

    U(a, b, z) ~ z^{-a} Σ_{k=0}^{N} (a)_k (a-b+1)_k / (k! (-z)^k)

The series is truncated at the term of smallest magnitude (optimal truncation
for asymptotic series).  Returns (value, min_term_ratio) where min_term_ratio
is |smallest term| / |partial sum| — a rough accuracy estimate.
"""
function hypergeometric_U_asymptotic(a, b, z; nterms::Int=80)
    epsR = eps(real(promote_type(typeof(complex(a)), typeof(complex(b)), typeof(complex(z)))))
    inv_mz = -1 / z  # 1/(-z)
    term = one(complex(a))
    s = term
    min_abs_term = abs(term)
    s_at_min = s
    prev_abs = abs(term)
    for k in 1:nterms
        term *= (a + k - 1) * (a - b + k) * inv_mz / k
        at = abs(term)
        if at > prev_abs && k > 1
            # Terms are growing — stop before adding this term (optimal truncation)
            break
        end
        s += term
        if at < min_abs_term
            min_abs_term = at
            s_at_min = s
        end
        prev_abs = at
        at < epsR * abs(s) && break
    end
    accuracy = abs(s) > 0 ? min_abs_term / abs(s) : Inf
    return z^(-a) * s_at_min, accuracy
end

"""
    hypergeometric_U_asymptotic_accuracy(a, b, z)

Estimate the accuracy of the asymptotic expansion for U(a, b, z)
without computing the full expansion. Returns the estimated relative
accuracy (smaller is better).
"""
function hypergeometric_U_asymptotic_accuracy(a, b, z)
    abs(z) < 5.0 && return Inf
    _, accuracy = hypergeometric_U_asymptotic(a, b, z)
    return accuracy
end

# Complex{Arb}: use Arb's OWN rigorous acb_hypgeom_u (Arblib.hypgeom_u!) instead
# of the Kummer/asymptotic construction below.  At large |z| the Kummer relation
# catastrophically cancels (off by ~1e42 at |z|≈77) and its near-integer-b Γ-pole
# guard `round(Int, ·)` NaNs; acb_hypgeom_u is accurate to full precision and
# handles every b.  This makes the generic Rup recurrence (whose seeds route
# through here) both correct AND fast at large |ω| for the Arb backend.
function hypergeometric_U(a::Complex{Arb}, b::Complex{Arb}, z::Complex{Arb})
    prec = precision(Arb)
    Uv = Acb(0)
    Arblib.hypgeom_u!(Uv, Acb(real(a), imag(a); prec=prec),
                          Acb(real(b), imag(b); prec=prec),
                          Acb(real(z), imag(z); prec=prec); prec=prec)
    return Complex{Arb}(Uv)
end

# Complex{BigFloat}: same routing through Arb's rigorous acb_hypgeom_u, at the
# working BigFloat precision (+53 guard bits absorbed by Arb's internal ball
# growth).  This matters because the BigFloat backend is what the precision
# predictor selects for |ω| ≳ 3.5, exactly where the generic path below is
# worst: the Kummer relation loses ~|z|·log₂e bits to cancellation and the
# asymptotic series is capped at ~e^(−|z|) accuracy, so at branch-cut
# frequencies the per-term U evaluations dominate Rup/dRup wall-clock AND
# silently force precision inflation.  acb_hypgeom_u selects its small /
# intermediate / asymptotic zones internally with rigorous error bounds.
function hypergeometric_U(a::Complex{BigFloat}, b::Complex{BigFloat},
                          z::Complex{BigFloat})
    prec = precision(BigFloat) + 53
    Uv = Acb(0; prec=prec)
    Arblib.hypgeom_u!(Uv, Acb(Arb(real(a); prec=prec), Arb(imag(a); prec=prec)),
                          Acb(Arb(real(b); prec=prec), Arb(imag(b); prec=prec)),
                          Acb(Arb(real(z); prec=prec), Arb(imag(z); prec=prec));
                      prec=prec)
    return Complex{BigFloat}(BigFloat(Arb(real(Uv))), BigFloat(Arb(imag(Uv))))
end

function hypergeometric_U(a, b, z)
    R = real(promote_type(typeof(complex(a)), typeof(complex(z))))
    # Asymptotic-vs-Kummer gate. The asymptotic series is fundamentally limited
    # to ~exp(-|z|) (optimal truncation); accept it only if it meets the working
    # precision. For Float64 keep the loose 1e-6 gate (Kummer catastrophically
    # cancels beyond |z|≈15 with only 53 bits); at higher precision the extra
    # digits absorb the Kummer cancellation, so demand near-eps from asymptotic.
    acc_tol = (R === Float64 || R === Float32) ? 1e-6 : eps(R)^(3//4)
    if abs(z) > 10
        val, accuracy = hypergeometric_U_asymptotic(a, b, z)
        accuracy < acc_tol && return val
        # Asymptotic not accurate enough — fall through to the convergent Kummer form.
    end

    # Non-finite parameters: propagate NaN instead of crashing (issue R14a —
    # round(Int, real(b)) throws InexactError for NaN/Inf b, and huge finite
    # real(b) would overflow Int; U with a non-finite parameter is undefined).
    if !(isfinite(complex(a)) && isfinite(complex(b)) && isfinite(complex(z)))
        return complex(R(NaN), R(NaN))
    end

    # Kummer relation for moderate/small |z|
    # For near-integer b, add a small (precision-scaled) perturbation off the Γ pole.
    # `round(real(b))` (float round, not Int) stays exact for arbitrarily large b.
    b_pert = b
    b_int = round(real(b))
    if abs(b - b_int) < sqrt(eps(R))
        b_pert = b + sqrt(eps(R)) * im
    end

    term1 = _cgamma(complex(1 - b_pert)) / _cgamma(complex(a + 1 - b_pert)) * _₁F₁(a, b_pert, z)
    term2 = _cgamma(complex(b_pert - 1)) / _cgamma(complex(a)) * z^(1 - b_pert) * _₁F₁(a + 1 - b_pert, 2 - b_pert, z)
    return term1 + term2
end

# ============================================================
#  2F1 evaluation with DLMF recurrence
#  Follows MST.m lines 139-167
# ============================================================

# ── Robust ₂F₁ for the MST radial sum ───────────────────────────────────────
# HypergeometricFunctions._₂F₁ routes |x|>1 through a connection formula that
# evaluates `loggamma` on the parameter combinations. For near-real parameters
# (real ν, i.e. the low-frequency regime ω = mΩφ ≪ 1) those combinations land on
# the negative real axis and loggamma throws a DomainError. We catch that and
# evaluate instead via a gamma-free Pfaff transformation,
#     ₂F₁(a,b;c;x) = (1-x)^(-a) ₂F₁(a, c-b; c; x/(x-1)),
# whose argument z' = x/(x-1) ∈ (0,1) for x<0, summed as a direct Maclaurin series
# (no Γ at all). This only engages on the rare instability-fallback / seed terms
# where HGF fails, so the well-conditioned-series regime (moderate |n|) is exactly
# where it is used.
function _h2f1_pfaff(a, b, c, x)
    z  = x / (x - 1)
    bb = c - b
    T  = promote_type(typeof(complex(a)), typeof(complex(b)),
                      typeof(complex(c)), typeof(complex(z)))
    term = one(T)
    s    = term
    tol  = eps(real(T))
    # The Maclaurin series converges geometrically at rate |z|, so reaching `tol`
    # needs ≈ log(tol)/log|z| terms. Size the cap to the working precision and z
    # (instead of a fixed window) so deep-precision / large-|z| (large orbit radius)
    # cases still converge, and warn rather than silently truncate if the cap is hit.
    az   = abs(z)
    kmax = 200_000
    if az < 1
        est  = log(tol) / log(az) + 64
        kmax = est < kmax ? ceil(Int, est) : kmax
    end
    converged = false
    for k in 0:kmax-1
        term *= (a + k) * (bb + k) / ((c + k) * (k + 1)) * z
        s += term
        if abs(term) ≤ tol * abs(s)
            converged = true
            break
        end
    end
    converged || @warn "_h2f1_pfaff: Maclaurin series hit the $kmax-term cap without \
                        reaching tol (|z|=$(Float64(az))); result may be sub-precision." maxlog=3
    return (1 - x)^(-a) * s
end

@inline function _h2f1_robust(a, b, c, x)
    try
        return _₂F₁(a, b, c, x)
    catch e
        e isa DomainError || rethrow(e)
        return _h2f1_pfaff(a, b, c, x)
    end
end

# Complex{Arb}: use Arb's OWN rigorous acb_hypgeom_2f1 (mirrors the
# hypergeometric_U routing above).  HypergeometricFunctions' |x|>1 connection
# formulas run Γ-heavy generic series on Arb — slow at high precision and prone
# to the loggamma DomainError → Pfaff detour; acb_hypgeom_2f1 selects the
# small/intermediate/large-|x| zones internally with rigorous error bounds and
# handles the near-integer parameter combinations without perturbation guards.
# Strictly more specific than the generic method, so Float64/BigFloat/MultiFloat
# dispatch is untouched.
function _h2f1_robust(a::Complex{Arb}, b::Complex{Arb}, c::Complex{Arb},
                      x::Complex{Arb})
    prec = precision(Arb)
    F = Acb(0; prec=prec)
    Arblib.hypgeom_2f1!(F, Acb(real(a), imag(a); prec=prec),
                           Acb(real(b), imag(b); prec=prec),
                           Acb(real(c), imag(c); prec=prec),
                           Acb(real(x), imag(x); prec=prec);
                        flags=0, prec=prec)
    return Complex{Arb}(F)
end

struct H2F1Params{T<:Complex}
    aF::T  # ν + 1 - iτ
    bF::T  # -ν - iτ
    cF::T  # 1 - s - i(ε+τ)
    x::T   # (r+ - r) / (2κ)
end

function H2F1Params(p::MSTParams, ν, x)
    aF = ν + 1 - im * p.τ
    bF = -ν - im * p.τ
    cF = 1 - p.s - im * (p.ϵ + p.τ)
    xc = complex(x)
    T  = promote_type(typeof(aF), typeof(bF), typeof(cF), typeof(xc))
    H2F1Params{T}(convert(T, aF), convert(T, bF), convert(T, cF), convert(T, xc))
end

function h2f1_exact(hp::H2F1Params, n::Int)
    _h2f1_robust(n + hp.aF, hp.bF - n, hp.cF, hp.x)
end

"""
3-term recurrence for 2F1 going upward in n (n >= 2).
Returns (term1, term2) such that H2F1[n] = term1 + term2.
Requires H2F1[n-2] and H2F1[n-1] already computed.
"""
function h2f1_up(hp::H2F1Params, n::Int, h2f1_nm2, h2f1_nm1)
    a, b, c, x = hp.aF, hp.bF, hp.cF, hp.x
    denom = (3 - a + b - 2n) * (1 + b - c - n) * (-1 + a + n)
    t1 = -(1 - a + b - 2n) * (1 + b - n) * (-1 + a - c + n) * h2f1_nm2
    t2 = -((-2 + a - b + 2n) * (-2 + 2a - 2b + 2a*b + c - a*c - b*c +
          4n - 2a*n + 2b*n - 2n^2 + 3x - 4a*x + a^2*x + 4b*x -
          2a*b*x + b^2*x - 8n*x + 4a*n*x - 4b*n*x + 4n^2*x)) * h2f1_nm1
    return t1 / denom, t2 / denom
end

"""
3-term recurrence for 2F1 going downward in n (n <= -1).
Returns (term1, term2) such that H2F1[n] = term1 + term2.
Requires H2F1[n+2] and H2F1[n+1] already computed.
"""
function h2f1_down(hp::H2F1Params, n::Int, h2f1_np2, h2f1_np1)
    a, b, c, x = hp.aF, hp.bF, hp.cF, hp.x
    denom = (-3 - a + b - 2n) * (-1 + b - n) * (1 + a - c + n)
    t1 = (-2 - a + b - 2n) * (-2 - 2a + 2b + 2a*b + c - a*c - b*c -
          4n - 2a*n + 2b*n - 2n^2 + 3x + 4a*x + a^2*x - 4b*x -
          2a*b*x + b^2*x + 8n*x + 4a*n*x - 4b*n*x + 4n^2*x) * h2f1_np1
    t2 = -(-1 - a + b - 2n) * (-1 + b - c - n) * (1 + a + n) * h2f1_np2
    return t1 / denom, t2 / denom
end

# ============================================================
#  2F1 derivative: d/dx [₂F₁(n+a, b-n, c, x)] = (n+a)(b-n)/c · ₂F₁(n+a+1, b-n+1, c+1, x)
# ============================================================

function dh2f1_exact(hp::H2F1Params, n::Int)
    (n + hp.aF) * (hp.bF - n) / hp.cF *
        _h2f1_robust(n + hp.aF + 1, hp.bF - n + 1, hp.cF + 1, hp.x)
end

"""
3-term recurrence for d(2F1)/dx going upward in n.
Returns (term1, term2, term3) where term3 is the H2F1 mixing term.
"""
function dh2f1_up(hp::H2F1Params, n::Int, dh2f1_nm2, dh2f1_nm1, h2f1_nm1)
    a, b, c, x = hp.aF, hp.bF, hp.cF, hp.x
    denom_outer = (1 + b - c - n) * (-1 + a + n)
    coeff_nm2 = -((1 - a + b - 2n) * (1 + b - n) * (-1 + a - c + n)) / (3 - a + b - 2n)
    poly = (-2 + 2a - 2b + 2a*b + c - a*c - b*c + 4n - 2a*n + 2b*n - 2n^2 +
            3x - 4a*x + a^2*x + 4b*x - 2a*b*x + b^2*x - 8n*x + 4a*n*x - 4b*n*x + 4n^2*x)
    coeff_nm1 = (2 - a + b - 2n) * poly / (3 - a + b - 2n)
    coeff_h = (1 - a + b - 2n) * (2 - a + b - 2n)
    return coeff_nm2 * dh2f1_nm2 / denom_outer,
           coeff_nm1 * dh2f1_nm1 / denom_outer,
           coeff_h * h2f1_nm1 / denom_outer
end

"""
3-term recurrence for d(2F1)/dx going downward in n.
Returns (term1, term2, term3) where term3 is the H2F1 mixing term.
"""
function dh2f1_down(hp::H2F1Params, n::Int, dh2f1_np2, dh2f1_np1, h2f1_np1)
    a, b, c, x = hp.aF, hp.bF, hp.cF, hp.x
    denom_outer = (-1 + b - n) * (1 + a - c + n)
    poly = (-2 - 2a + 2b + 2a*b + c - a*c - b*c - 4n - 2a*n + 2b*n - 2n^2 +
            3x + 4a*x + a^2*x - 4b*x - 2a*b*x + b^2*x + 8n*x + 4a*n*x - 4b*n*x + 4n^2*x)
    coeff_np1 = (-2 - a + b - 2n) * poly / (-3 - a + b - 2n)
    coeff_np2 = -((-1 - a + b - 2n) * (-1 + b - c - n) * (1 + a + n)) / (-3 - a + b - 2n)
    coeff_h = (-2 - a + b - 2n) * (-1 - a + b - 2n)
    return coeff_np1 * dh2f1_np1 / denom_outer,
           coeff_np2 * dh2f1_np2 / denom_outer,
           coeff_h * h2f1_np1 / denom_outer
end

# ============================================================
#  HypergeometricU evaluation with recurrence
#  HU[n] = c^n * U(n+aU, 2n+bU, c)
#  where aU = ν+s+1-iε, bU = 2ν+2, c = -2i·ẑ
# ============================================================

struct HUParams{T<:Complex}
    aU::T  # ν + s + 1 - iε
    bU::T  # 2ν + 2
    c::T   # -2i * ẑ
end

# Promoting constructor (also used by radial_down's direct call).
function HUParams(aU, bU, c)
    aUc, bUc, cc = complex(aU), complex(bU), complex(c)
    T = promote_type(typeof(aUc), typeof(bUc), typeof(cc))
    HUParams{T}(convert(T, aUc), convert(T, bUc), convert(T, cc))
end

function HUParams(p::MSTParams, ν, zhat)
    aU = ν + p.s + 1 - im * p.ϵ
    bU = 2ν + 2
    c = -2im * zhat
    HUParams(aU, bU, c)
end

function hu_exact(hp::HUParams, n::Int)
    hp.c^n * hypergeometric_U(n + hp.aU, 2n + hp.bU, hp.c)
end

"""
3-term recurrence for HU going upward in n (n >= 2).
Returns (term1, term2) such that HU[n] = term1 + term2.
"""
function hu_up(hp::HUParams, n::Int, hu_nm2, hu_nm1)
    a, b, c = hp.aU, hp.bU, hp.c
    denom = (-1 + a + n) * (-4 + b + 2n)
    t1 = (-2 - a + b + n) * (-2 + b + 2n) * hu_nm2
    t2 = (-3 + b + 2n) * (8 + (b + 2n)^2 + 2(a + n)*c - (b + 2n)*(6 + c)) * hu_nm1 / c
    return t1 / denom, t2 / denom
end

"""
3-term recurrence for HU going downward in n (n <= -1).
Returns (term1, term2) such that HU[n] = term1 + term2.
"""
function hu_down(hp::HUParams, n::Int, hu_np2, hu_np1)
    a, b, c = hp.aU, hp.bU, hp.c
    denom = (-a + b + n) * (2 + b + 2n)
    t1 = -((1 + b + 2n) * (b^2 + 4n*(1 + n) + b*(2 + 4n - c) + 2a*c)) * hu_np1 / c
    t2 = (1 + a + n) * (b + 2n) * hu_np2
    return t1 / denom, t2 / denom
end

# ============================================================
#  HypergeometricU derivative with respect to ẑ
#  dHU[n] = d/dẑ [c^n U(n+a, 2n+b, c)]  with dc/dẑ = -2i
# ============================================================

# dHU[n] = -2i ( c^{n-1} n U(a+n,b+2n,c) - c^n (a+n) U(1+a+n,1+b+2n,c) ).
# The first confluent-U is exactly hu_exact(hp,n)/c^n, i.e. c^{n-1} n U = (n/c)·HU[n]
# (optimization B): pass the already-computed base HU[n] to skip re-evaluating it.
function dhu_exact(hp::HUParams, n::Int, hu_n)
    a, b, c = hp.aU, hp.bU, hp.c
    -2im * ((n / c) * hu_n -
            c^n * (a + n) * hypergeometric_U(1 + a + n, 1 + b + 2n, c))
end
dhu_exact(hp::HUParams, n::Int) = dhu_exact(hp, n, hu_exact(hp, n))

"""
3-term recurrence for dHU going upward in n (n >= 2).
Returns (term1, term2, term3) where term3 is the HU mixing term.
"""
function dhu_up(hp::HUParams, n::Int, dhu_nm2, dhu_nm1, hu_nm1)
    a, b, c = hp.aU, hp.bU, hp.c
    denom = -1 + a + n
    t1 = (-2 - a + b + n) * (-2 + b + 2n) * dhu_nm2 / (-4 + b + 2n)
    poly = 8 + b^2 + 4(-3 + n)*n + b*(-6 + 4n - c) + 2a*c
    t2 = (-3 + b + 2n) * poly * dhu_nm1 / ((-4 + b + 2n) * c)
    t3 = 2im * (-3 + b + 2n) * (-2 + b + 2n) * hu_nm1 / c^2
    return t1 / denom, t2 / denom, t3 / denom
end

"""
3-term recurrence for dHU going downward in n (n <= -1).
Returns (term1, term2, term3) where term3 is the HU mixing term.
"""
function dhu_down(hp::HUParams, n::Int, dhu_np2, dhu_np1, hu_np1)
    a, b, c = hp.aU, hp.bU, hp.c
    denom = (a - b - n) * (2 + b + 2n) * c^2
    poly = b^2 + 4n*(1 + n) + b*(2 + 4n - c) + 2a*c
    t1 = (1 + b + 2n) * c * poly * dhu_np1
    t2 = (b + 2n) * (-(1 + a + n) * c^2) * dhu_np2
    t3 = (b + 2n) * 2im * (1 + b + 2n) * (2 + b + 2n) * hu_np1
    return t1 / denom, t2 / denom, t3 / denom
end

# ============================================================
#  Certified HU / dHU evaluation with stable outward marching
#  (Complex{Arb} and Complex{BigFloat} backends)
#
#  MEASURED FACTS (arbiter: per-n Arblib.hypgeom_u! with rigorous,
#  rel_accuracy_bits-verified balls; see test/test_hu_evaluation.jl):
#
#  * The 3-term n-recurrence marched OUTWARD from seeds at n ∈ {0,1} is
#    uniformly stable for HU[n]: across σ ∈ [0.5, 16] (ω = iσ PIA and the
#    θ ∈ {0°,30°,60°} complex angles at |ω| ∈ {5,10}), r ∈ {4, 10},
#    physical and arbitrary ν, the max relative drift over |n| ≤ 60 from
#    exact seeds is ≤ 23 bits (worst at real ω; ≤ 3 bits in the PIA
#    regimes) at every working precision tested (256/512/1024 bits).
#    U is the dominant-or-neutral solution of the recurrence in BOTH
#    outward directions (up for n>0, down for n<0), so outward marching
#    cannot lose it; INWARD marching is exponentially unstable (errors
#    grow ~1e70 over 60 steps at small σ) and is never used.
#
#  * The old per-step guard max(|t1/val|,|t2/val|) > 2.0 sits exactly on
#    the natural mid-band plateau of the term ratios (K ≈ 0.9–5.7,
#    hovering at 2.0 for |ω| ≳ 5): in BigFloat it fires almost every
#    step — each dhu fallback costing TWO rigorous U calls — while for
#    Complex{Arb} the BALL comparison becomes undecidable once radii
#    grow and silently never fires.  The ratio plateau is NOT an
#    instability (see above); the guard was miscalibrated.
#
#  * acb_hypgeom_u at the working precision loses ~0.6·(|c|+Re₊c) up to
#    ~2.2·|c| bits INTERNALLY in its convergent-Kummer zone (the rigorous
#    ball reports it: e.g. 25 good bits out of 256 at σ=16, r=4, PIA), so
#    `hu_exact` at working precision — the old guard's fallback AND its
#    seeds — was the real accuracy bottleneck, not the recurrence.
#
#  SCHEME
#    Seeds HU[0], HU[1] (and the dHU companions) are evaluated by
#    precision ESCALATION on Arblib.hypgeom_u!: call at
#    need + 64 + 0.7(|c|+Re₊c) bits, verify the rigorous ball via
#    rel_accuracy_bits ≥ need ≈ 0.94·prec, escalate by the measured
#    deficit until certified.  Every other n comes from the outward
#    march at working precision.  A cheap ComplexF64 shadow of the
#    contaminant solution tracks the true dominant/wanted growth S(n),
#    and a certified refresh replaces a marched value if either the
#    running error estimate crosses the trip threshold or a single step
#    cancels catastrophically (near-vanishing recurrence denominators at
#    2ν ∈ ℤ / a+n−1 ≈ 0).  This keeps the exact-fallback safety net of
#    the old scheme, now with a decidable (Float64) trigger and a
#    certified (escalated) fallback value.
#
#  Float64 / MultiFloat paths keep the legacy guard scheme verbatim
#  (_hu_dhu_evaluators_legacy below) — at 53–212 bits the mid-band drift
#  budget is real and per-step exact fallback remains the right call.
# ============================================================

_hu_certified_backend(::Type) = false
_hu_certified_backend(::Type{Complex{Arb}}) = true
_hu_certified_backend(::Type{Complex{BigFloat}}) = true

_hu_bits(hp::HUParams) = precision(real(hp.aU))

_mid_f64(x::Arb) = Float64(Arblib.midref(x))
_mid_f64(x) = Float64(x)
_mid_c64(z::Complex) = complex(_mid_f64(real(z)), _mid_f64(imag(z)))
_mag_f64(z::Complex) = abs(_mid_c64(z))

# ── Decidable recurrence-instability guard (issue R6) ────────────────────────
# The old per-step guard `iszero(val) || max(abs(t1/val),abs(t2/val)) > 2.0`
# is UNDECIDABLE on Complex{Arb} balls: Arb's `>` is certainly-greater, so an
# uncertain comparison returns `false` and the exact-recompute fallback can
# NEVER fire once ball radii grow (verified: Arb(10±100) > 2.0 === false).
# Same bug class as the pre-certified HU guard.  Decide on the Float64
# MIDPOINT of the working-precision ratios t/val instead — the ratios are
# O(1) near the trip point, so the Float64 cast cannot over/underflow there —
# and be conservative on degenerate values: a non-finite or exactly-zero
# `val`, or a non-finite ratio midpoint (Arb division by a ball containing
# zero yields NaN midpoints), always trips the guard.
#
# `_RECUR_GUARD_TRIP_COUNT` is a test hook (not thread-safe) so regression
# tests can assert the fallback actually fires on the Arb backend.
const _RECUR_GUARD_TRIP_COUNT = Ref{Int}(0)

@inline function _recur_guard_trips(val, terms...)
    ok = isfinite(real(val)) && isfinite(imag(val)) && !iszero(val)
    if ok
        for t in terms
            r = _mag_f64(t / val)
            if !(isfinite(r) && r <= 2.0)
                ok = false
                break
            end
        end
    end
    ok && return false
    _RECUR_GUARD_TRIP_COUNT[] += 1
    return true
end

# Exact midpoint lift to an Acb at precision p (the working-precision midpoints
# DEFINE the evaluation problem; p ≥ source bits ⇒ conversion is exact).
function _acb_at(z::Complex{Arb}, p::Int)
    re = Arb(0; prec=p); im_ = Arb(0; prec=p)
    Arblib.set!(re, Arblib.midref(real(z)))
    Arblib.set!(im_, Arblib.midref(imag(z)))
    return Acb(re, im_)
end
_acb_at(z::Complex{BigFloat}, p::Int) = Acb(Arb(real(z); prec=p), Arb(imag(z); prec=p))

# Round an escalated-precision Acb back to the working scalar type.
function _from_acb(::Type{Complex{Arb}}, H::Acb, pb::Int)
    W = Acb(0; prec=pb)
    Arblib.set_round!(W, H, pb)
    return Complex{Arb}(W)
end
function _from_acb(::Type{Complex{BigFloat}}, H::Acb, pb::Int)
    return Complex{BigFloat}(BigFloat(Arb(real(H))), BigFloat(Arb(imag(H))))
end

# Pre-estimate (bits) of acb_hypgeom_u's internal Kummer-zone cancellation.
# Measured: ~1.0|c| for PIA (c real > 0, moderate a), up to ~2.2|c| when
# Re(a) ≫ |c|/4; ~0.2–0.8|c| for imaginary c (real ω).  0.7(|c|+Re₊c)+64
# lands the first call at-or-above the certified target in the common cases;
# the escalation loop absorbs the rest by the measured deficit.
# Non-finite midpoints (degenerate inputs) and absurdly large |c| are clamped
# so the estimate can never throw (ceil(Int, NaN) → InexactError) or drive the
# seed precision past any sane bound (issue R14a).
function _u_loss_estimate(c64::ComplexF64)
    y = 0.7 * (abs(c64) + max(real(c64), 0.0))
    return 64 + (isfinite(y) ? ceil(Int, min(y, 2.0^20)) : 0)
end

# Certified accuracy (bits) of an escalation-seed ball, made SAFE for the
# escalation arithmetic (issue R14a): Arblib.rel_accuracy_bits returns
# ±(2^63−1) for degenerate balls — +typemax for exact zero/NaN/Inf midpoints
# (radius 0), −typemax for zero/non-finite midpoints with a finite radius —
# and the raw value would either falsely certify a non-finite ball or
# overflow `p + (need − acc)` into a negative precision (→ crash).  Clamp to
# [−2^20, p−4] and treat any non-finite ball as having no usable accuracy.
function _seed_acc_bits(H::Acb, p::Int)
    isfinite(H) || return -(1 << 20)
    return Int(clamp(Arblib.rel_accuracy_bits(H), -(1 << 20), p - 4))
end

"""
    _hu_seed_acb(hp, n, need) -> (H::Acb, acc::Int)

Certified HU[n] = c^n U(n+aU, 2n+bU, c): escalate Arblib.hypgeom_u! precision
until the rigorous result ball has `rel_accuracy_bits ≥ need`.  Returns the
raw escalated-precision ball (caller rounds to working precision) and the
certified accuracy in bits.
"""
function _hu_seed_acb(hp::HUParams, n::Int, need::Int)
    p = need + _u_loss_estimate(_mid_c64(hp.c))
    best = nothing
    bacc = typemin(Int)
    for _ in 1:5
        aA = _acb_at(hp.aU, p); bA = _acb_at(hp.bU, p); cA = _acb_at(hp.c, p)
        U = Acb(0; prec=p)
        Arblib.hypgeom_u!(U, aA + n, bA + 2n, cA; prec=p)
        H = Acb(0; prec=p)
        Arblib.pow!(H, cA, n; prec=p)
        Arblib.mul!(H, H, U; prec=p)
        acc = _seed_acc_bits(H, p)
        acc >= need && return H, acc
        if acc > bacc
            best = H; bacc = acc
        end
        p = min(p + (need - acc) + 128, 4p + 1024)
    end
    @warn "_hu_seed_acb: escalation exhausted at n=$n (achieved $bacc of $need bits)" maxlog=2
    return best, max(bacc, 1)
end

"""
    _dhu_seed_acb(hp, n, need, Hraw) -> (D::Acb, acc::Int)

Certified dHU[n] = -2i[(n/c)·HU[n] − c^n (aU+n) U(1+aU+n, 1+bU+2n, c)].
`Hraw` (the raw certified ball from `_hu_seed_acb`, or `nothing`) is reused on
the first attempt so the common path costs ONE extra hypgeom_u call; the ball
arithmetic propagates its rigorous radius, so any cancellation between the two
terms is captured honestly by `rel_accuracy_bits` and triggers escalation
(which then recomputes both U's).
"""
function _dhu_seed_acb(hp::HUParams, n::Int, need::Int, Hraw::Union{Nothing,Acb})
    p = need + _u_loss_estimate(_mid_c64(hp.c))
    Hraw !== nothing && (p = max(p, precision(Hraw)))
    best = nothing
    bacc = typemin(Int)
    for it in 1:5
        aA = _acb_at(hp.aU, p); bA = _acb_at(hp.bU, p); cA = _acb_at(hp.c, p)
        U2 = Acb(0; prec=p)
        Arblib.hypgeom_u!(U2, aA + (n + 1), bA + (2n + 1), cA; prec=p)
        cn = Acb(0; prec=p)
        Arblib.pow!(cn, cA, n; prec=p)
        term2 = cn * (aA + n) * U2
        Hn = if it == 1 && Hraw !== nothing
            Hraw
        else
            UH = Acb(0; prec=p)
            Arblib.hypgeom_u!(UH, aA + n, bA + 2n, cA; prec=p)
            H = Acb(0; prec=p)
            Arblib.pow!(H, cA, n; prec=p)
            Arblib.mul!(H, H, UH; prec=p)
            H
        end
        D = Acb(0, -2; prec=p) * ((n / cA) * Hn - term2)
        acc = _seed_acc_bits(D, p)
        acc >= need && return D, acc
        if acc > bacc
            best = D; bacc = acc
        end
        p = min(p + (need - acc) + 128, 4p + 1024)
    end
    @warn "_dhu_seed_acb: escalation exhausted at n=$n (achieved $bacc of $need bits)" maxlog=2
    return best, max(bacc, 1)
end

# Per-direction march state: PURE-ComplexF64 twin marches of (i) the
# contaminant solution g (seeded (0,1) — the second solution of the
# recurrence) and (ii) the wanted solution h itself (seeded from the
# certified-seed midpoints — valid as a MAGNITUDE tracker precisely because
# the outward march is stable).  All per-step safety bookkeeping is Float64;
# no Arb→Float64 conversions in the hot loop.
#   (ga, gb)·2^logg : contaminant pair at positions (n∓2, n∓1)
#   (ha, hb)·2^logh : wanted-solution twin pair
#   mind            : min over past positions of (log2|g| − log2|h|)
#   errb            : certified error bits (log2 rel err) of the current pair
#   logg / logh accumulate only the RESCALE corrections, so both twins live in
#   units normalized at the seed (d(seed) ≈ 0 and S measures pure relative
#   growth); logh0 records the absolute scale |h(seed)| needed to convert true
#   HU values into twin units for the dhu mixing term.
mutable struct _HUMarchState
    ga::ComplexF64
    gb::ComplexF64
    logg::Float64
    ha::ComplexF64
    hb::ComplexF64
    logh::Float64
    logh0::Float64
    mind::Float64
    errb::Float64
    steps::Int
end
function _HUMarchState(errb::Float64, va::Complex, vb::Complex)
    ha = _mid_c64(va); hb = _mid_c64(vb)
    s = abs(hb)
    if isfinite(s) && s > 0
        return _HUMarchState(complex(0.0), complex(1.0), 0.0, ha/s, hb/s, 0.0, log2(s),
                             0.0, errb, 0)
    end
    return _HUMarchState(complex(0.0), complex(1.0), 0.0, complex(1.0), complex(1.0), 0.0, 0.0,
                         0.0, errb, 0)
end

@inline function _rescale2(a::ComplexF64, b::ComplexF64, m::Float64)
    if isfinite(m) && (m > 1e100 || (m > 0 && m < 1e-100))
        return a / m, b / m, log2(m)
    end
    return a, b, 0.0
end

# advance both twins one step; returns (est_bits, q3_est) — the running
# relative-error estimate in bits (Inf when the twin broke / the step
# cancelled catastrophically) and |t3|/|val| for the dhu mixing term
# (0.0 for the plain hu march).
function _shadow_advance!(st::_HUMarchState, hp64::HUParams{ComplexF64},
                          n::Int, upward::Bool, pb::Int; dhu::Bool=false,
                          hu64::ComplexF64=complex(0.0))
    st.steps += 1
    local gv, hv, qnum, q3::Float64
    if dhu
        # contaminant g obeys the HOMOGENEOUS part (zero HU mixing); the
        # wanted-solution twin h gets the true HU value, rescaled into the
        # twin's 2^-(logh0+logh) working scale.
        hmix = hu64 * exp2(-(st.logh0 + st.logh))
        g1, g2, _  = upward ? dhu_up(hp64, n, st.ga, st.gb, complex(0.0)) :
                              dhu_down(hp64, n, st.ga, st.gb, complex(0.0))
        h1, h2, h3 = upward ? dhu_up(hp64, n, st.ha, st.hb, hmix) :
                              dhu_down(hp64, n, st.ha, st.hb, hmix)
        gv = g1 + g2; hv = h1 + h2 + h3
        qnum = abs(h1) + abs(h2) + abs(h3)
        q3 = abs(h3)
    else
        g1, g2 = upward ? hu_up(hp64, n, st.ga, st.gb) : hu_down(hp64, n, st.ga, st.gb)
        h1, h2 = upward ? hu_up(hp64, n, st.ha, st.hb) : hu_down(hp64, n, st.ha, st.hb)
        gv = g1 + g2; hv = h1 + h2
        qnum = abs(h1) + abs(h2)
        q3 = 0.0
    end
    ah = abs(hv); ag = abs(gv)
    (isfinite(ah) && ah > 0 && isfinite(ag)) || return Inf, q3
    q = qnum / ah
    (isfinite(q) && q < 2.0^24) || return Inf, q3
    d = (st.logg + (ag > 0 ? log2(ag) : -Inf)) - (st.logh + log2(ah))
    st.mind = min(st.mind, d)
    S = d - st.mind
    st.ga, gv, dg = _rescale2(st.gb, gv, ag)
    st.gb = gv; st.logg += dg
    st.ha, hv2, dh = _rescale2(st.hb, hv, ah)
    st.hb = hv2; st.logh += dh
    q3 = ah > 0 ? q3 / ah : 0.0
    return max(st.errb, -pb + log2(1.0 + st.steps) + 2) + S + 1, q3
end

# in-place radius strip (no allocation; the input is a freshly built value)
@inline function _strip_radius!(z::Complex{Arb})
    Arblib.zero!(Arblib.radref(real(z)))
    Arblib.zero!(Arblib.radref(imag(z)))
    return z
end
@inline _strip_radius!(z::Complex) = z

"""
    _hu_dhu_evaluators(hp::HUParams{T}) -> (get_hu, get_dhu)

Memoized evaluators for HU[n] and dHU[n] used by `Rup`/`dRup`.  For the
Complex{Arb} and Complex{BigFloat} backends this is the certified-seed +
outward-march scheme documented above; for all other scalar types it is the
legacy exact-seeded recurrence with the per-step ratio guard (verbatim the
pre-rewrite behavior).
"""
function _hu_dhu_evaluators(hp::HUParams{T}) where {T<:Complex}
    _hu_certified_backend(T) || return _hu_dhu_evaluators_legacy(hp)

    pb   = _hu_bits(hp)
    need = min(pb, ceil(Int, 0.94 * pb) + 40)
    trip = -0.86 * pb                       # refresh when est. rel-err bits exceed this
    hp64 = HUParams(_mid_c64(hp.aU), _mid_c64(hp.bU), _mid_c64(hp.c))

    hu_cache  = Dict{Int,T}()
    dhu_cache = Dict{Int,T}()
    raw_seeds = Dict{Int,Acb}()

    up  = Ref{Union{Nothing,_HUMarchState}}(nothing)
    dn  = Ref{Union{Nothing,_HUMarchState}}(nothing)
    dup = Ref{Union{Nothing,_HUMarchState}}(nothing)
    ddn = Ref{Union{Nothing,_HUMarchState}}(nothing)

    function seed_hu!(n::Int)
        H, acc = _hu_seed_acb(hp, n, need)
        raw_seeds[n] = H
        v = _from_acb(T, H, pb)
        hu_cache[n] = v
        return v, Float64(-(min(acc, pb) - 1))
    end
    function seed_dhu!(n::Int)
        haskey(hu_cache, n) || get_hu(n)     # populate raw_seeds[n] for n ∈ {0,1}
        D, acc = _dhu_seed_acb(hp, n, need, get(raw_seeds, n, nothing))
        v = _from_acb(T, D, pb)
        dhu_cache[n] = v
        return v, Float64(-(min(acc, pb) - 1))
    end
    seederr() = Float64(-(need > pb ? pb : need) + 1)

    function get_hu(n::Int)
        v = get(hu_cache, n, nothing)
        v === nothing || return v
        if n == 0 || n == 1
            v, _ = seed_hu!(n)
            return v
        end
        upward = n >= 2
        va = upward ? get_hu(n - 2) : get_hu(n + 2)
        vb = upward ? get_hu(n - 1) : get_hu(n + 1)
        stref = upward ? up : dn
        if stref[] === nothing
            stref[] = _HUMarchState(seederr(), va, vb)
        end
        st = stref[]
        t1, t2 = upward ? hu_up(hp, n, va, vb) : hu_down(hp, n, va, vb)
        # point arithmetic: marched Complex{Arb} balls accumulate triangle-
        # inequality radii ~K^n that make Arb TRUNCATE the midpoints (~1 bit per
        # step at complex ω); the midpoint march itself is stable (measured
        # ≤ 23-bit drift), so strip radii like the Arb Lentz CF does and let the
        # shadow tracker + certified refresh carry the error accounting.
        val = _strip_radius!(t1 + t2)
        est, _ = _shadow_advance!(st, hp64, n, upward, pb)
        (isfinite(real(val)) && isfinite(imag(val))) || (est = Inf)
        if est > trip
            # safety net: certify BOTH pair values and restart the shadow
            m = upward ? n - 1 : n + 1
            _, em = seed_hu!(m)
            v2, e2 = seed_hu!(n)
            stref[] = _HUMarchState(max(em, e2), hu_cache[m], v2)
            return v2
        end
        hu_cache[n] = val
        return val
    end

    function get_dhu(n::Int)
        v = get(dhu_cache, n, nothing)
        v === nothing || return v
        if n == 0 || n == 1
            v, _ = seed_dhu!(n)
            return v
        end
        get_hu(n)                            # keep the hu march (and shadow) ahead
        upward = n >= 2
        dva = upward ? get_dhu(n - 2) : get_dhu(n + 2)
        dvb = upward ? get_dhu(n - 1) : get_dhu(n + 1)
        hub = upward ? get_hu(n - 1) : get_hu(n + 1)
        stref = upward ? dup : ddn
        if stref[] === nothing
            stref[] = _HUMarchState(seederr(), dva, dvb)
        end
        st = stref[]
        t1, t2, t3 = upward ? dhu_up(hp, n, dva, dvb, hub) :
                              dhu_down(hp, n, dva, dvb, hub)
        val = _strip_radius!(t1 + t2 + t3)   # point arithmetic (see get_hu)
        est, q3 = _shadow_advance!(st, hp64, n, upward, pb;
                                   dhu=true, hu64=_mid_c64(hub))
        (isfinite(real(val)) && isfinite(imag(val))) || (est = Inf)
        # fold the HU-mixing error path: hu values are certified-march accurate
        # (≥ 0.9·pb bits), entering through t3 with weight |t3|/|val|
        if isfinite(est) && q3 > 0
            est = max(est, -0.9 * pb + log2(q3) + 1)
        end
        if est > trip
            m = upward ? n - 1 : n + 1
            _, em = seed_dhu!(m)
            v2, e2 = seed_dhu!(n)
            stref[] = _HUMarchState(max(em, e2), dhu_cache[m], v2)
            return v2
        end
        dhu_cache[n] = val
        return val
    end

    return get_hu, get_dhu
end

# Legacy evaluators (verbatim pre-rewrite behavior): exact seeds at working
# precision, per-step ratio guard > 2.0 with hu_exact/dhu_exact fallback, and
# the all-exact shortcut when the asymptotic series already meets the working
# tolerance.  Kept for Float64/Float32/MultiFloat backends.
function _hu_dhu_evaluators_legacy(hp::HUParams{T}) where {T<:Complex}
    R = real(T)
    hu_cache  = Dict{Int,T}()
    dhu_cache = Dict{Int,T}()
    _asymp_acc = hypergeometric_U_asymptotic_accuracy(hp.aU, hp.bU, hp.c)
    _acc_tol = (R === Float64 || R === Float32) ? 1e-6 : eps(R)^(3//4)
    use_exact_all = _asymp_acc < _acc_tol

    function get_hu(n::Int)
        haskey(hu_cache, n) && return hu_cache[n]
        if use_exact_all || n == 0 || n == 1
            val = hu_exact(hp, n)
        elseif n >= 2
            t1, t2 = hu_up(hp, n, get_hu(n-2), get_hu(n-1))
            val = t1 + t2
            if iszero(val) || max(abs(t1/val), abs(t2/val)) > 2.0
                val = hu_exact(hp, n)
            end
        else
            t1, t2 = hu_down(hp, n, get_hu(n+2), get_hu(n+1))
            val = t1 + t2
            if iszero(val) || max(abs(t1/val), abs(t2/val)) > 2.0
                val = hu_exact(hp, n)
            end
        end
        hu_cache[n] = val
        return val
    end

    function get_dhu(n::Int)
        haskey(dhu_cache, n) && return dhu_cache[n]
        if use_exact_all || n == 0 || n == 1
            val = dhu_exact(hp, n, get_hu(n))   # reuse base HU[n] (optimization B)
        elseif n >= 2
            t1, t2, t3 = dhu_up(hp, n, get_dhu(n-2), get_dhu(n-1), get_hu(n-1))
            val = t1 + t2 + t3
            if iszero(val) || max(abs(t1/val), abs(t2/val), abs(t3/val)) > 2.0
                val = dhu_exact(hp, n)
            end
        else
            t1, t2, t3 = dhu_down(hp, n, get_dhu(n+2), get_dhu(n+1), get_hu(n+1))
            val = t1 + t2 + t3
            if iszero(val) || max(abs(t1/val), abs(t2/val), abs(t3/val)) > 2.0
                val = dhu_exact(hp, n)
            end
        end
        dhu_cache[n] = val
        return val
    end

    return get_hu, get_dhu
end
