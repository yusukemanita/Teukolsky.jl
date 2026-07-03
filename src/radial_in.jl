# ============================================================
#  Rin: Ingoing radial solution (near horizon, 2F1-based)
#
#  Sasaki-Tagoshi Eqs. (116), (120); MST.m lines 506-540
#  Teukolsky case only.
#
#  R_in(r) = prefac(x) * Σ_n fIn_n * ₂F₁(n+aF, bF-n, cF, x)
#
#  where x = (r+ - r) / (2κ)
#  prefac = (-x)^{-s - i(ε+τ)/2} (1-x)^{i(ε-τ)/2} exp(iεκx)
#  fIn_n = fn[n]  (for Teukolsky)
# ============================================================

# ── InTrans / Btrans factor (shared by Rin and dRin) ─────────────────────────
# Σfn runs over EVERY entry of the (possibly adaptively extended) fn dict, in
# deterministic index order; f_n is the minimal solution and decays
# super-exponentially, so the sum is converged long before the dict boundary.
function _intrans(p::MSTParams, fn)
    s, ε, τ, κ = p.s, p.ϵ, p.τ, p.κ
    Σfn = sum(fn[n] for n in sort!(collect(keys(fn))))
    prefac = (4 * one(ε))^s * κ^(2s) * exp(im * (ε + τ) * κ * (0.5 + log(κ) / (1 + κ)))
    return prefac * Σfn
end

# ============================================================
#  Converge-or-error MST series summation (issue R1)
#
#  The horizon-side 2F1 series terms GROW with n (peak n ≈ 2.3·|x|,
#  x = (r₊−r)/(2κ)) before the super-exponential decay of f_n takes over, so
#  any FIXED nmax silently returns garbage once r is large enough that the
#  peak sits at/beyond the truncation (verified: rel-err 2.3e15 at r=30 for
#  the old fixed nmax=80 default, s=-2 l=m=2 a=0.7 ω=0.8, 256-bit — and the
#  Float64 path agreed with the wrong value to 2.6e-11, i.e. fully silent).
#
#  `_sum_mst_series!` therefore sums until the last-term criterion
#      |term| < tol·|result| + tol_abs
#  holds on two consecutive nonzero terms, extending `fn` outward on demand
#  (continued-fraction ratio chaining — exactly how `compute_fn` builds it),
#  and ERRORS instead of ever returning a silently unconverged sum: when the
#  coefficients terminate early (deliberately truncated or underflowed f_n
#  tail) with the last term still above tolerance, or when `nmax_hard` is
#  reached.  Convergence decisions use Float64 MIDPOINTS of the O(1)
#  criterion ratio, so they stay decidable on Complex{Arb} balls (an Arb `<`
#  is certainly-less: an uncertain comparison is `false`, which would
#  otherwise disable the stopping test exactly like the dead guards of
#  issue R6).
# ============================================================

"""
    _extend_fn_dir!(fn, p, ν, dir, N) -> Bool

Extend the series-coefficient dict `fn` outward to |n| = `N` on the side
`dir` (+1: positive n, −1: negative n) by continued-fraction ratio chaining
(f_n = R_n·f_{n−1} resp. f_{−n} = L_{−n}·f_{−n+1}), exactly how `compute_fn`
builds it.  Existing entries are never modified, so caller-supplied
coefficients stay authoritative.  Returns `false` when the boundary entry is
zero (deliberately truncated or underflowed tail — chaining would be
identically zero), `true` otherwise.
"""
function _extend_fn_dir!(fn::Dict{Int,T}, p::MSTParams, ν, dir::Int, N::Int) where {T}
    b = dir > 0 ? maximum(keys(fn)) : minimum(keys(fn))
    (dir > 0 ? N <= b : -N >= b) && return true
    iszero(fn[b]) && return false
    ratios = _cf_ratios(p, ν, dir, N)
    if dir > 0
        for n in b+1:N
            fn[n] = ratios[n] * fn[n-1]
        end
    else
        for n in b-1:-1:-N
            fn[n] = ratios[-n] * fn[n+1]
        end
    end
    return true
end

"""
    _sum_mst_series!(termfn, fn, p, ν, dir, tol, tol_abs, n_first_ext,
                     nmax_hard, what)

Sum `Σ_n fn[n]·termfn(n)` over one direction (`dir = +1`: n = 0,1,2,…;
`dir = −1`: n = −1,−2,…) until two consecutive nonzero terms satisfy
`|term| < tol·|result| + tol_abs`, adaptively extending `fn` (mutates the
dict) via [`_extend_fn_dir!`](@ref).

Never returns unconverged: raises an error naming `what` if the
coefficients terminate early or |n| exceeds `nmax_hard`.

Returns `(result, smax)` where `smax` is the running maximum of |term| and
|partial sum| — the caller MUST pass it (combined over both directions) to
[`_certify_mst_sum`](@ref): the 2F1 term peak can exceed the final sum by
many decades (measured 1.4e14× at s=-2 l=m=2 a=0.7 ω=0.8, r=30: the tail
cancels the peak), in which case the term criterion alone converges happily
onto pure rounding noise (Float64 there is wrong by 2e4× RELATIVE despite a
perfectly converged-looking tail, and 256-bit BigFloat is floor-limited to
~5e-62 ≈ eps·peak/|result| — measured, not modeled).
"""
function _sum_mst_series!(termfn::F, fn::Dict{Int,T}, p::MSTParams, ν, dir::Int,
                          tol, tol_abs, n_first_ext::Int, nmax_hard::Int,
                          what::AbstractString) where {F,T}
    RT = real(T)
    result = zero(T)
    smax = zero(RT)   # running max of |term|, |partial sum| — cancellation-floor scale
    consec = 0        # consecutive nonzero terms below the tol criterion
    lastq  = Inf      # criterion ratio |term|/(tol·|result|+tol_abs) of the last nonzero term

    n = dir > 0 ? 0 : -1
    while true
        if abs(n) > nmax_hard
            error("$what: series not converged by |n| = $nmax_hard " *
                  "(last |term|/(tol·|sum|) ≈ $(lastq)); increase nmax_hard, " *
                  "loosen tol, or use a higher-precision backend.")
        end
        if !haskey(fn, n)
            target = min(max(2 * abs(n), n_first_ext), nmax_hard)
            if !_extend_fn_dir!(fn, p, ν, dir, target)
                # f_n terminated (explicit zero tail: deliberate truncation or
                # working-precision underflow).  Accept ONLY if the trailing
                # nonzero term already met the tolerance — otherwise the
                # partial sum is silent garbage and we must error.
                lastq < 1.0 && return result, smax
                error("$what: f_n coefficients ended in a zero tail at n = $n " *
                      "while the last term was still $(lastq)× the stopping " *
                      "tolerance — f_n truncated too aggressively or |f_n| " *
                      "underflowed at working precision before the 2F1/U term " *
                      "peak (large r needs a higher-precision backend or an " *
                      "untruncated fn with larger nmax).")
            end
            continue
        end
        fn_n = fn[n]
        if !iszero(fn_n)
            term = termfn(n) * fn_n
            old = result
            result += term
            abs_term = abs(term)
            abs_res  = abs(result)
            smax = max(max(smax, abs_term), abs_res)
            if result == old
                # exact stagnation at working precision — as converged as this
                # arithmetic can express
                consec += 1
                lastq = 0.0
            else
                q = _mid_f64(abs_term / (tol * abs_res + tol_abs))
                lastq = q
                consec = (isfinite(q) && q < 1.0) ? consec + 1 : 0
            end
            consec >= 2 && return result, smax
        end
        n += dir
    end
end

"""
    _certify_mst_sum(total, smax, ctol, ctol_abs, what) -> total

Cancellation-floor certification of a term-converged bidirectional MST sum:
require

    eps(R)·smax ≤ ctol·|total| + ctol_abs

(decidable Float64-midpoint comparison, conservative on non-finite values),
where `smax` is the max over both directions of |term| and |partial sum|.
The left side estimates the accumulated rounding error of the summation —
VALIDATED against converged BigFloat references: across s=-2 l=m=2
(a,ω) ∈ {(0.7,0.8),(0,0.5)}, r = 5…30, the Float64 error tracks
eps·smax/|total| within a factor ≈ 4 over ten decades (e.g. r=25: estimate
2.5, measured 5.6).  Failing the test means `ctol` is unattainable at this
working precision, so the honest outcome is an error — never a silently
floor-limited value — reporting the achievable relative accuracy.
"""
function _certify_mst_sum(total, smax, ctol, ctol_abs, what::AbstractString)
    epsR = eps(typeof(smax))
    qf = _mid_f64(epsR * smax / (ctol * abs(total) + ctol_abs))
    if !(isfinite(qf) && qf <= 1.0)
        achievable = _mid_f64(epsR * smax / abs(total))
        error("$what: series converged termwise, but the cancellation floor " *
              "eps·max|partial| exceeds the accepted tolerance by ≈ $(qf)× " *
              "(term peak $(_mid_f64(smax / abs(total)))× the final sum; " *
              "achievable relative accuracy ≈ $(achievable)).  Raise the " *
              "working precision or pass tol ≥ the achievable accuracy.")
    end
    return total
end

# Default certification (floor-acceptance) tolerance: at least HALF the
# working digits must survive the series cancellation, or we error.  The
# actual certified tolerance is max(tol, √eps) so an explicitly loosened
# `tol` is always honored (see Rin docstring).
_default_floor_tol(RT::Type) = sqrt(eps(RT))

# First-extension target for the horizon-side 2F1 series: cover the measured
# term peak n ≈ 2.3·|x| with margin, so the common case needs a single fn
# extension.  (The tol tail beyond the peak is handled by the 2·|n| doubling;
# non-finite/huge |x| midpoints are clamped — the call site clamps to
# nmax_hard anyway.)
function _rin_next_ext(nmax::Int, x)
    m = _mag_f64(x)
    m = isfinite(m) ? min(m, 1.0e6) : 1.0e6
    return max(2 * nmax, 64 + ceil(Int, 2.5 * m))
end

# ── Raw (un-normalized) MST series — internal use only ───────────────────────
function _Rin_raw(p::MSTParams, ν, fn, r; nmax::Int=80, tol::Real=100*eps(real(typeof(p.ϵ))),
                  nmax_hard::Int=50_000,
                  floor_tol::Real=_default_floor_tol(real(typeof(p.ϵ))))
    κ = p.κ
    rp = p.rp
    x = complex((rp - r) / (2κ))

    hp = H2F1Params(p, ν, x)

    ϵ, τ, s = p.ϵ, p.τ, p.s
    prefac = (-x)^(-s - im*(ϵ + τ)/2) * (1 - x)^(im*(ϵ - τ)/2) * exp(im*ϵ*κ*x)

    h2f1_cache = Dict{Int, typeof(p.ϵ)}()

    function get_h2f1(n::Int)
        haskey(h2f1_cache, n) && return h2f1_cache[n]

        if n == 0 || n == 1
            val = h2f1_exact(hp, n)
        elseif n >= 2
            v_nm2 = get_h2f1(n - 2)
            v_nm1 = get_h2f1(n - 1)
            t1, t2 = h2f1_up(hp, n, v_nm2, v_nm1)
            val = t1 + t2
            if _recur_guard_trips(val, t1, t2)
                val = h2f1_exact(hp, n)
            end
        else
            v_np2 = get_h2f1(n + 2)
            v_np1 = get_h2f1(n + 1)
            t1, t2 = h2f1_down(hp, n, v_np2, v_np1)
            val = t1 + t2
            if _recur_guard_trips(val, t1, t2)
                val = h2f1_exact(hp, n)
            end
        end

        h2f1_cache[n] = val
        return val
    end

    # prefac is hoisted out of the term loops (one complex multiply per term
    # saved); the absolute part of the stopping test is rescaled by 1/|prefac|
    # so the truncation decisions are identical to the un-hoisted form.
    tol_abs = tol / abs(prefac)

    n_ext = _rin_next_ext(nmax, x)
    res_up, smax_up = _sum_mst_series!(get_h2f1, fn, p, ν, +1, tol, tol_abs,
                                       n_ext, nmax_hard, "Rin")
    res_down, smax_dn = _sum_mst_series!(get_h2f1, fn, p, ν, -1, tol, tol_abs,
                                         n_ext, nmax_hard, "Rin")

    ctol = max(tol, floor_tol)
    total = _certify_mst_sum(res_up + res_down, max(smax_up, smax_dn),
                             ctol, ctol / abs(prefac), "Rin")
    return prefac * total
end

"""
    Rin(p::MSTParams, ν, fn, r; nmax=80, tol=100·eps, nmax_hard=50_000,
        floor_tol=√eps)

Transmission-normalized ingoing Teukolsky solution at r, matching
Mathematica's `TeukolskyRadial["In",...]` convention:

    Rin = Rin_raw / Btrans

where `Btrans = 4^s κ^{2s} exp(i(ε+τ)κ(½ + logκ/(1+κ))) Σfn`.

This is smooth across ν branch transitions (real / half-integer / integer).

Convergence policy (converge-or-error, issue R1):

* The horizon-side 2F1 series is summed until the last-term criterion
  `|term| < tol·|sum|` holds — the terms PEAK near n ≈ 2.3·(r−r₊)/(2|κ|)
  before decaying, so far more than `nmax` terms can be required at large
  r.  `fn` is adaptively extended IN PLACE (continued-fraction ratio
  chaining) when more coefficients are needed; an error is raised if the
  series cannot converge within `nmax_hard` terms or the supplied `fn`
  terminates too early.
* The tail of the series cancels the peak (by 1.4e14× already at r=30 for
  s=-2 l=m=2 a=0.7 ω=0.8), so the result is additionally CERTIFIED against
  the cancellation floor: an error is raised unless
  `eps·max|partial sum| ≤ max(tol, floor_tol)·|sum|`.  The default
  `floor_tol = √eps` demands that at least half the working digits
  survive; pass a looser `tol` (it is always honored, `max(tol,
  floor_tol)`) to accept a lower certified accuracy, or raise the working
  precision.  The floor estimate `eps·max|partial|/|sum|` tracks the true
  error within ≈ 4× (validated against converged BigFloat references).
"""
function Rin(p::MSTParams, ν, fn, r; nmax::Int=80, tol::Real=100*eps(real(typeof(p.ϵ))),
             nmax_hard::Int=50_000,
             floor_tol::Real=_default_floor_tol(real(typeof(p.ϵ))))
    raw = _Rin_raw(p, ν, fn, r; nmax=nmax, tol=tol, nmax_hard=nmax_hard,
                   floor_tol=floor_tol)
    return raw / _intrans(p, fn)
end

"""
    dRin(p::MSTParams, ν, fn, r; nmax=80, tol=100·eps, nmax_hard=50_000,
         floor_tol=√eps)

Compute dR_in/dr at Boyer-Lindquist radius r, transmission-normalized
(i.e. d/dr[Rin_raw / Btrans]).  Same converge-or-error summation and
cancellation-floor certification as [`Rin`](@ref) (adaptive fn extension,
hard error past `nmax_hard` or below the certified floor).
"""
function dRin(p::MSTParams, ν, fn, r; nmax::Int=80, tol::Real=100*eps(real(typeof(p.ϵ))),
              nmax_hard::Int=50_000,
              floor_tol::Real=_default_floor_tol(real(typeof(p.ϵ))))
    κ = p.κ
    rp = p.rp
    x = complex((rp - r) / (2κ))
    dxdr = complex(-1 / (2κ))

    hp = H2F1Params(p, ν, x)

    ϵ, τ, s = p.ϵ, p.τ, p.s

    pow_neg_x = (-x)^(-s - im*(ϵ + τ)/2)
    pow_1_x = (1 - x)^(im*(ϵ - τ)/2)
    exp_part = exp(im*ϵ*κ*x)
    prefac = pow_neg_x * pow_1_x * exp_part

    α_exp = -s - im*(ϵ + τ)/2
    dpow_neg_x = α_exp / x * pow_neg_x

    β_exp = im*(ϵ - τ)/2
    dpow_1_x = -β_exp / (1 - x) * pow_1_x

    dexp_part = im*ϵ*κ * exp_part

    dprefac_dx = dpow_neg_x * pow_1_x * exp_part +
                 pow_neg_x * dpow_1_x * exp_part +
                 pow_neg_x * pow_1_x * dexp_part

    dprefac = dprefac_dx * dxdr
    prefac_dxdr = prefac * dxdr

    h2f1_cache = Dict{Int, typeof(p.ϵ)}()
    dh2f1_cache = Dict{Int, typeof(p.ϵ)}()

    function get_h2f1(n::Int)
        haskey(h2f1_cache, n) && return h2f1_cache[n]
        if n == 0 || n == 1
            val = h2f1_exact(hp, n)
        elseif n >= 2
            t1, t2 = h2f1_up(hp, n, get_h2f1(n-2), get_h2f1(n-1))
            val = t1 + t2
            if _recur_guard_trips(val, t1, t2)
                val = h2f1_exact(hp, n)
            end
        else
            t1, t2 = h2f1_down(hp, n, get_h2f1(n+2), get_h2f1(n+1))
            val = t1 + t2
            if _recur_guard_trips(val, t1, t2)
                val = h2f1_exact(hp, n)
            end
        end
        h2f1_cache[n] = val
        return val
    end

    function get_dh2f1(n::Int)
        haskey(dh2f1_cache, n) && return dh2f1_cache[n]
        if n == 0 || n == 1
            val = dh2f1_exact(hp, n)
        elseif n >= 2
            t1, t2, t3 = dh2f1_up(hp, n, get_dh2f1(n-2), get_dh2f1(n-1), get_h2f1(n-1))
            val = t1 + t2 + t3
            if _recur_guard_trips(val, t1, t2, t3)
                val = dh2f1_exact(hp, n)
            end
        else
            t1, t2, t3 = dh2f1_down(hp, n, get_dh2f1(n+2), get_dh2f1(n+1), get_h2f1(n+1))
            val = t1 + t2 + t3
            if _recur_guard_trips(val, t1, t2, t3)
                val = dh2f1_exact(hp, n)
            end
        end
        dh2f1_cache[n] = val
        return val
    end

    dterm(n::Int) = dprefac * get_h2f1(n) + prefac_dxdr * get_dh2f1(n)

    n_ext = _rin_next_ext(nmax, x)
    res_up, smax_up = _sum_mst_series!(dterm, fn, p, ν, +1, tol, tol,
                                       n_ext, nmax_hard, "dRin")
    res_down, smax_dn = _sum_mst_series!(dterm, fn, p, ν, -1, tol, tol,
                                         n_ext, nmax_hard, "dRin")

    ctol = max(tol, floor_tol)
    total = _certify_mst_sum(res_up + res_down, max(smax_up, smax_dn),
                             ctol, ctol, "dRin")
    return total / _intrans(p, fn)
end
