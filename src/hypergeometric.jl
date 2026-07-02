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

    # Kummer relation for moderate/small |z|
    # For near-integer b, add a small (precision-scaled) perturbation off the Γ pole.
    b_pert = b
    b_int = round(Int, real(b))
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
