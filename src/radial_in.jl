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

Compute the (un-normalized) ingoing radial Teukolsky solution at r.

Summation matches Teukolsky package (MST.m lines 531-535):
- stop when fn[n] = 0 (beyond the computed dict range)
- stop when the sum no longer changes in Float64 (machine-precision convergence)
- stop when |term| < tol * |sum| + tol (relative+absolute tolerance)

For the transmission-normalized version (matching Mathematica's
`TeukolskyRadial["In",...]`), call `Rin_phys` instead.
"""
function Rin(p::MSTParams, ν, fn, r; nmax::Int=40, tol::Float64=1e-14)
    κ = p.κ
    rp = p.rp
    x = complex((rp - r) / (2κ))

    hp = H2F1Params(p, ν, x)

    # Prefactor: (-x)^{-s - i(ε+τ)/2} (1-x)^{i(ε-τ)/2} exp(iεκx)
    ϵ, τ, s = p.ϵ, p.τ, p.s
    prefac = (-x)^(-s - im*(ϵ + τ)/2) * (1 - x)^(im*(ϵ - τ)/2) * exp(im*ϵ*κ*x)

    # H2F1 cache with recurrence + Teukolsky-compatible cancellation fallback.
    # Fix #3: when recurrence gives val=0, Max[{t1,t2}/0]=∞>2 in Mathematica
    # → always fall back to H2F1Exact. Replicate by checking iszero(val).
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
            # Fix #3: fall back when val=0 OR significant cancellation
            if iszero(val) || max(abs(t1/val), abs(t2/val)) > 2.0
                val = h2f1_exact(hp, n)
            end
        else  # n <= -1
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

    # ── Upward sum: n = 0, 1, 2, … ─────────────────────────────────────────
    # Fix #2: match Teukolsky's adaptive termination (MST.m line 532):
    #   While[resUp != (resUp += term[nUp]) && |term| > tol, nUp++]
    # i.e. stop when sum stops changing in Float64 OR term is within tolerance.
    # Also: fn[n]=0 (beyond dict) → term=0 → sum unchanged → stop (Fix #4).
    result = complex(0.0)
    for n in 0:nmax
        fn_n = get(fn, n, complex(0.0))
        iszero(fn_n) && break  # Fix #4: break (not continue) to match Teukolsky
        term = prefac * fn_n * get_h2f1(n)
        old = result
        result += term
        result == old && break  # Fix #2: float convergence
        n > 0 && abs(term) < tol * abs(result) + tol && break
    end

    # ── Downward sum: n = -1, -2, … ─────────────────────────────────────────
    res_down = complex(0.0)
    for n in -1:-1:-nmax
        fn_n = get(fn, n, complex(0.0))
        iszero(fn_n) && break  # Fix #4
        term = prefac * fn_n * get_h2f1(n)
        old = res_down
        res_down += term
        res_down == old && break  # Fix #2
        abs(term) < tol * abs(res_down) + tol && break
    end

    return result + res_down
end

"""
    Rin_phys(p::MSTParams, ν, fn, r; nmax=40, tol=1e-14)

Transmission-normalized ingoing Teukolsky solution, matching Mathematica's
`TeukolskyRadial["In",...]` convention:

    Rin_phys = Rin_raw / InTrans

where `InTrans = Btrans = 4^s κ^{2s} exp(i(ε+τ)κ(½ + logκ/(1+κ))) Σfn`.

This quantity is smooth across ν branch transitions (real / half-integer /
integer), whereas the raw `Rin` can jump by a large factor at each branch
boundary because it carries the ν-dependent normalization.

Physical note: the waveform formula G = Rin_raw × Bref / (2iω Binc)
is already correct as written (Btrans cancels), but when comparing directly
with Mathematica's MSTRadialIn output, use Rin_phys.
"""
function Rin_phys(p::MSTParams, ν, fn, r; nmax::Int=40, tol::Float64=1e-14)
    raw = Rin(p, ν, fn, r; nmax=nmax, tol=tol)
    # InTrans = Btrans (MST.m Teukolsky case, line 106)
    s, ε, τ, κ = p.s, p.ϵ, p.τ, p.κ
    Σfn = sum(get(fn, n, complex(0.0)) for n in -nmax:nmax)
    prefac_it = ComplexF64(4)^s * κ^(2s) * exp(im * (ε + τ) * κ * (0.5 + log(κ) / (1 + κ)))
    InTrans = prefac_it * Σfn
    return raw / InTrans
end

"""
    dRin(p::MSTParams, ν, fn, r; nmax=40, tol=1e-14)

Compute dR_in/dr at Boyer-Lindquist radius r.
Applies the same Teukolsky-compatible summation and cancellation fixes as Rin.
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

    # H2F1 and dH2F1 caches — same cancellation fix as Rin
    h2f1_cache = Dict{Int, ComplexF64}()
    dh2f1_cache = Dict{Int, ComplexF64}()

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

    # Upward sum with same termination rules as Rin
    result = complex(0.0)
    for n in 0:nmax
        fn_n = get(fn, n, complex(0.0))
        iszero(fn_n) && break
        term = fn_n * (dprefac * get_h2f1(n) + prefac_dxdr * get_dh2f1(n))
        old = result
        result += term
        result == old && break
        n > 0 && abs(term) < tol * abs(result) + tol && break
    end

    res_down = complex(0.0)
    for n in -1:-1:-nmax
        fn_n = get(fn, n, complex(0.0))
        iszero(fn_n) && break
        term = fn_n * (dprefac * get_h2f1(n) + prefac_dxdr * get_dh2f1(n))
        old = res_down
        res_down += term
        res_down == old && break
        abs(term) < tol * abs(res_down) + tol && break
    end

    return result + res_down
end
