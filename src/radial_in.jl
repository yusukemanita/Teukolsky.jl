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

"""
    Rin(p::MSTParams, ν, fn, r; nmax=40, tol=1e-14)

Compute the ingoing radial Teukolsky solution at Boyer-Lindquist radius r.
Uses the MST series expansion in Hypergeometric 2F1 functions.
"""
function Rin(p::MSTParams, ν, fn, r; nmax::Int=40, tol::Float64=1e-14)
    κ = p.κ
    rp = p.rp
    x = complex((rp - r) / (2κ))

    hp = H2F1Params(p, ν, x)

    # Prefactor: (-x)^{-s - i(ε+τ)/2} (1-x)^{i(ε-τ)/2} exp(iεκx)
    ϵ, τ, s = p.ϵ, p.τ, p.s
    prefac = (-x)^(-s - im*(ϵ + τ)/2) * (1 - x)^(im*(ϵ - τ)/2) * exp(im*ϵ*κ*x)

    # Evaluate H2F1 with recurrence + cancellation fallback
    h2f1_cache = Dict{Int, ComplexF64}()

    function get_h2f1(n::Int)
        haskey(h2f1_cache, n) && return h2f1_cache[n]

        if n == 0 || n == 1
            val = h2f1_exact(hp, n)
        elseif n >= 2
            v_nm2 = get_h2f1(n - 2)
            v_nm1 = get_h2f1(n - 1)
            t1, t2 = h2f1_up(hp, n, v_nm2, v_nm1)
            val = t1 + t2
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val)) > 2.0
                val = h2f1_exact(hp, n)
            end
        else  # n <= -1
            v_np2 = get_h2f1(n + 2)
            v_np1 = get_h2f1(n + 1)
            t1, t2 = h2f1_down(hp, n, v_np2, v_np1)
            val = t1 + t2
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val)) > 2.0
                val = h2f1_exact(hp, n)
            end
        end

        h2f1_cache[n] = val
        return val
    end

    # Sum bidirectionally
    result = complex(0.0)

    # Upward: n = 0, 1, 2, ...
    for n in 0:nmax
        fn_n = get(fn, n, complex(0.0))
        iszero(fn_n) && continue
        term = prefac * fn_n * get_h2f1(n)
        result += term
        n > 0 && abs(term) < tol * abs(result) + tol && break
    end

    # Downward: n = -1, -2, ...
    res_down = complex(0.0)
    for n in -1:-1:-nmax
        fn_n = get(fn, n, complex(0.0))
        iszero(fn_n) && continue
        term = prefac * fn_n * get_h2f1(n)
        res_down += term
        abs(term) < tol * abs(res_down) + tol && break
    end

    return result + res_down
end

"""
    dRin(p::MSTParams, ν, fn, r; nmax=40, tol=1e-14)

Compute dR_in/dr at Boyer-Lindquist radius r.
"""
function dRin(p::MSTParams, ν, fn, r; nmax::Int=40, tol::Float64=1e-14)
    κ = p.κ
    rp = p.rp
    x = complex((rp - r) / (2κ))
    dxdr = complex(-1 / (2κ))

    hp = H2F1Params(p, ν, x)

    ϵ, τ, s = p.ϵ, p.τ, p.s

    # Prefactor and its x-derivative
    pow_neg_x = (-x)^(-s - im*(ϵ + τ)/2)
    pow_1_x = (1 - x)^(im*(ϵ - τ)/2)
    exp_part = exp(im*ϵ*κ*x)
    prefac = pow_neg_x * pow_1_x * exp_part

    # dprefac/dx via product rule on three factors
    # d/dx (-x)^α = (-x)^α * α * d/dx[log(-x)] = (-x)^α * α / x
    α_exp = -s - im*(ϵ + τ)/2
    dpow_neg_x = α_exp / x * pow_neg_x

    # d/dx (1-x)^β = -β (1-x)^{β-1}
    β_exp = im*(ϵ - τ)/2
    dpow_1_x = -β_exp / (1 - x) * pow_1_x

    # d/dx exp(iεκx) = iεκ exp(iεκx)
    dexp_part = im*ϵ*κ * exp_part

    dprefac_dx = dpow_neg_x * pow_1_x * exp_part +
                 pow_neg_x * dpow_1_x * exp_part +
                 pow_neg_x * pow_1_x * dexp_part

    dprefac = dprefac_dx * dxdr
    prefac_dxdr = prefac * dxdr

    # H2F1 and dH2F1 caches
    h2f1_cache = Dict{Int, ComplexF64}()
    dh2f1_cache = Dict{Int, ComplexF64}()

    function get_h2f1(n::Int)
        haskey(h2f1_cache, n) && return h2f1_cache[n]
        if n == 0 || n == 1
            val = h2f1_exact(hp, n)
        elseif n >= 2
            t1, t2 = h2f1_up(hp, n, get_h2f1(n-2), get_h2f1(n-1))
            val = t1 + t2
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val)) > 2.0
                val = h2f1_exact(hp, n)
            end
        else
            t1, t2 = h2f1_down(hp, n, get_h2f1(n+2), get_h2f1(n+1))
            val = t1 + t2
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val)) > 2.0
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
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val), abs(t3/val)) > 2.0
                val = dh2f1_exact(hp, n)
            end
        else
            t1, t2, t3 = dh2f1_down(hp, n, get_dh2f1(n+2), get_dh2f1(n+1), get_h2f1(n+1))
            val = t1 + t2 + t3
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val), abs(t3/val)) > 2.0
                val = dh2f1_exact(hp, n)
            end
        end
        dh2f1_cache[n] = val
        return val
    end

    # Sum: dRin/dr = Σ fIn_n * (dprefac * H2F1[n] + prefac * dH2F1[n] * dxdr)
    result = complex(0.0)
    for n in 0:nmax
        fn_n = get(fn, n, complex(0.0))
        iszero(fn_n) && continue
        term = fn_n * (dprefac * get_h2f1(n) + prefac_dxdr * get_dh2f1(n))
        result += term
        n > 0 && abs(term) < tol * abs(result) + tol && break
    end

    res_down = complex(0.0)
    for n in -1:-1:-nmax
        fn_n = get(fn, n, complex(0.0))
        iszero(fn_n) && continue
        term = fn_n * (dprefac * get_h2f1(n) + prefac_dxdr * get_dh2f1(n))
        res_down += term
        abs(term) < tol * abs(res_down) + tol && break
    end

    return result + res_down
end
