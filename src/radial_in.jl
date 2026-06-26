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
function _intrans(p::MSTParams, fn, nmax::Int)
    s, ε, τ, κ = p.s, p.ϵ, p.τ, p.κ
    Σfn = sum(get(fn, n, zero(typeof(p.ϵ))) for n in -nmax:nmax)
    prefac = (4 * one(ε))^s * κ^(2s) * exp(im * (ε + τ) * κ * (0.5 + log(κ) / (1 + κ)))
    return prefac * Σfn
end

# ── Raw (un-normalized) MST series — internal use only ───────────────────────
function _Rin_raw(p::MSTParams, ν, fn, r; nmax::Int=80, tol::Real=100*eps(real(typeof(p.ϵ))))
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
            if iszero(val) || max(abs(t1/val), abs(t2/val)) > 2.0
                val = h2f1_exact(hp, n)
            end
        else
            v_np2 = get_h2f1(n + 2)
            v_np1 = get_h2f1(n + 1)
            t1, t2 = h2f1_down(hp, n, v_np2, v_np1)
            val = t1 + t2
            if iszero(val) || max(abs(t1/val), abs(t2/val)) > 2.0
                val = h2f1_exact(hp, n)
            end
        end

        h2f1_cache[n] = val
        return val
    end

    result = zero(typeof(p.ϵ))
    for n in 0:nmax
        fn_n = get(fn, n, zero(typeof(p.ϵ)))
        iszero(fn_n) && break
        term = prefac * fn_n * get_h2f1(n)
        old = result
        result += term
        result == old && break
        n > 0 && abs(term) < tol * abs(result) + tol && break
    end

    res_down = zero(typeof(p.ϵ))
    for n in -1:-1:-nmax
        fn_n = get(fn, n, zero(typeof(p.ϵ)))
        iszero(fn_n) && break
        term = prefac * fn_n * get_h2f1(n)
        old = res_down
        res_down += term
        res_down == old && break
        abs(term) < tol * abs(res_down) + tol && break
    end

    return result + res_down
end

"""
    Rin(p::MSTParams, ν, fn, r; nmax=80, tol=1e-14)

Transmission-normalized ingoing Teukolsky solution at r, matching
Mathematica's `TeukolskyRadial["In",...]` convention:

    Rin = Rin_raw / Btrans

where `Btrans = 4^s κ^{2s} exp(i(ε+τ)κ(½ + logκ/(1+κ))) Σfn`.

This is smooth across ν branch transitions (real / half-integer / integer).
"""
function Rin(p::MSTParams, ν, fn, r; nmax::Int=80, tol::Real=100*eps(real(typeof(p.ϵ))))
    raw = _Rin_raw(p, ν, fn, r; nmax=nmax, tol=tol)
    return raw / _intrans(p, fn, nmax)
end

"""
    dRin(p::MSTParams, ν, fn, r; nmax=80, tol=1e-14)

Compute dR_in/dr at Boyer-Lindquist radius r, transmission-normalized
(i.e. d/dr[Rin_raw / Btrans]).
"""
function dRin(p::MSTParams, ν, fn, r; nmax::Int=80, tol::Real=100*eps(real(typeof(p.ϵ))))
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
            if iszero(val) || max(abs(t1/val), abs(t2/val)) > 2.0
                val = h2f1_exact(hp, n)
            end
        else
            t1, t2 = h2f1_down(hp, n, get_h2f1(n+2), get_h2f1(n+1))
            val = t1 + t2
            if iszero(val) || max(abs(t1/val), abs(t2/val)) > 2.0
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
            if iszero(val) || max(abs(t1/val), abs(t2/val), abs(t3/val)) > 2.0
                val = dh2f1_exact(hp, n)
            end
        else
            t1, t2, t3 = dh2f1_down(hp, n, get_dh2f1(n+2), get_dh2f1(n+1), get_h2f1(n+1))
            val = t1 + t2 + t3
            if iszero(val) || max(abs(t1/val), abs(t2/val), abs(t3/val)) > 2.0
                val = dh2f1_exact(hp, n)
            end
        end
        dh2f1_cache[n] = val
        return val
    end

    result = zero(typeof(p.ϵ))
    for n in 0:nmax
        fn_n = get(fn, n, zero(typeof(p.ϵ)))
        iszero(fn_n) && break
        term = fn_n * (dprefac * get_h2f1(n) + prefac_dxdr * get_dh2f1(n))
        old = result
        result += term
        result == old && break
        n > 0 && abs(term) < tol * abs(result) + tol && break
    end

    res_down = zero(typeof(p.ϵ))
    for n in -1:-1:-nmax
        fn_n = get(fn, n, zero(typeof(p.ϵ)))
        iszero(fn_n) && break
        term = fn_n * (dprefac * get_h2f1(n) + prefac_dxdr * get_dh2f1(n))
        old = res_down
        res_down += term
        res_down == old && break
        abs(term) < tol * abs(res_down) + tol && break
    end

    return (result + res_down) / _intrans(p, fn, nmax)
end
