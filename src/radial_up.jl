# ============================================================
#  Rup: Upgoing radial solution (at infinity, HypergeometricU-based)
#
#  Sasaki-Tagoshi Eqs. (153), (159); MST.m lines 616-652
#  Teukolsky case only.
#
#  R_up(r) = prefac(ẑ) * Σ_n fUp_n * HU[n]
#
#  where ẑ = ε(r - r-)/2,  c = -2iẑ
#  HU[n] = c^n U(n+aU, 2n+bU, c)
#  aU = ν+s+1-iε, bU = 2ν+2
#
#  fUp_n = (-1)^n Pochhammer(ν+1+s-iε, n) / Pochhammer(ν+1-s+iε, n) * fn[n]
#
#  prefac = 2^ν e^{-πε} e^{-iπ(ν+1)} e^{iẑ} ẑ^{ν+i(ε+τ)/2}
#           (ẑ-εκ)^{-i(ε+τ)/2-s} e^{-iπs}
# ============================================================

"""
    Rup(p::MSTParams, ν, fn, r; nmax=40, tol=1e-14)

Compute the upgoing radial Teukolsky solution at Boyer-Lindquist radius r.
Uses the MST series expansion in HypergeometricU functions.
"""
function Rup(p::MSTParams, ν, fn, r; nmax::Int=40, tol::Float64=1e-14)
    ϵ, κ, τ, s = p.ϵ, p.κ, p.τ, p.s
    rm = p.rm
    zhat = complex(ϵ * (r - rm) / 2)

    hp = HUParams(p, ν, zhat)

    # Prefactor (Teukolsky, MST.m line 90)
    prefac = 2^ν * exp(-π*ϵ) * exp(-im*π*(ν + 1)) *
             exp(im*zhat) * zhat^(ν + im*(ϵ + τ)/2) *
             (zhat - ϵ*κ)^(-im*(ϵ + τ)/2) *
             exp(-im*π*s) * (zhat - ϵ*κ)^(-s)

    # HU cache with recurrence + fallback
    hu_cache = Dict{Int, ComplexF64}()

    function get_hu(n::Int)
        haskey(hu_cache, n) && return hu_cache[n]
        if n == 0 || n == 1
            val = hu_exact(hp, n)
        elseif n >= 2
            t1, t2 = hu_up(hp, n, get_hu(n-2), get_hu(n-1))
            val = t1 + t2
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val)) > 2.0
                val = hu_exact(hp, n)
            end
        else
            t1, t2 = hu_down(hp, n, get_hu(n+2), get_hu(n+1))
            val = t1 + t2
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val)) > 2.0
                val = hu_exact(hp, n)
            end
        end
        hu_cache[n] = val
        return val
    end

    # fUp_n coefficient
    function fup(n::Int)
        fn_n = get(fn, n, complex(0.0))
        iszero(fn_n) && return complex(0.0)
        (-1)^n * pochhammer(ν + 1 + s - im*ϵ, n) /
                 pochhammer(ν + 1 - s + im*ϵ, n) * fn_n
    end

    # Sum bidirectionally
    result = complex(0.0)
    for n in 0:nmax
        fu = fup(n)
        iszero(fu) && continue
        term = prefac * fu * get_hu(n)
        result += term
        n > 0 && abs(term) < tol * abs(result) + tol && break
    end

    res_down = complex(0.0)
    for n in -1:-1:-nmax
        fu = fup(n)
        iszero(fu) && continue
        term = prefac * fu * get_hu(n)
        res_down += term
        abs(term) < tol * abs(res_down) + tol && break
    end

    return result + res_down
end

"""
    dRup(p::MSTParams, ν, fn, r; nmax=40, tol=1e-14)

Compute dR_up/dr at Boyer-Lindquist radius r.
"""
function dRup(p::MSTParams, ν, fn, r; nmax::Int=40, tol::Float64=1e-14)
    ϵ, κ, τ, s = p.ϵ, p.κ, p.τ, p.s
    rm = p.rm
    zhat = complex(ϵ * (r - rm) / 2)
    dzhatdr = ϵ / 2

    hp = HUParams(p, ν, zhat)

    # Prefactor components
    A = 2^ν * exp(-π*ϵ) * exp(-im*π*(ν + 1)) * exp(-im*π*s)
    exp_z = exp(im*zhat)
    pow_z = zhat^(ν + im*(ϵ + τ)/2)
    zmek = zhat - ϵ*κ
    pow_zmek = zmek^(-im*(ϵ + τ)/2 - s)

    prefac = A * exp_z * pow_z * pow_zmek

    # dprefac/dẑ via product rule on three ẑ-dependent factors
    α_z = ν + im*(ϵ + τ)/2
    β_zmek = -im*(ϵ + τ)/2 - s

    # d/dẑ exp(iẑ) = i exp(iẑ)
    dexp_z = im * exp_z
    # d/dẑ ẑ^α = α/ẑ * ẑ^α
    dpow_z = α_z / zhat * pow_z
    # d/dẑ (ẑ-εκ)^β = β/(ẑ-εκ) * (ẑ-εκ)^β
    dpow_zmek = β_zmek / zmek * pow_zmek

    dprefac_dzhat = A * (dexp_z * pow_z * pow_zmek +
                         exp_z * dpow_z * pow_zmek +
                         exp_z * pow_z * dpow_zmek)

    dprefac = dprefac_dzhat * dzhatdr
    prefac_dzdr = prefac * dzhatdr

    # HU and dHU caches
    hu_cache = Dict{Int, ComplexF64}()
    dhu_cache = Dict{Int, ComplexF64}()

    function get_hu(n::Int)
        haskey(hu_cache, n) && return hu_cache[n]
        if n == 0 || n == 1
            val = hu_exact(hp, n)
        elseif n >= 2
            t1, t2 = hu_up(hp, n, get_hu(n-2), get_hu(n-1))
            val = t1 + t2
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val)) > 2.0
                val = hu_exact(hp, n)
            end
        else
            t1, t2 = hu_down(hp, n, get_hu(n+2), get_hu(n+1))
            val = t1 + t2
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val)) > 2.0
                val = hu_exact(hp, n)
            end
        end
        hu_cache[n] = val
        return val
    end

    function get_dhu(n::Int)
        haskey(dhu_cache, n) && return dhu_cache[n]
        if n == 0 || n == 1
            val = dhu_exact(hp, n)
        elseif n >= 2
            t1, t2, t3 = dhu_up(hp, n, get_dhu(n-2), get_dhu(n-1), get_hu(n-1))
            val = t1 + t2 + t3
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val), abs(t3/val)) > 2.0
                val = dhu_exact(hp, n)
            end
        else
            t1, t2, t3 = dhu_down(hp, n, get_dhu(n+2), get_dhu(n+1), get_hu(n+1))
            val = t1 + t2 + t3
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val), abs(t3/val)) > 2.0
                val = dhu_exact(hp, n)
            end
        end
        dhu_cache[n] = val
        return val
    end

    function fup(n::Int)
        fn_n = get(fn, n, complex(0.0))
        iszero(fn_n) && return complex(0.0)
        (-1)^n * pochhammer(ν + 1 + s - im*ϵ, n) /
                 pochhammer(ν + 1 - s + im*ϵ, n) * fn_n
    end

    # Sum: dRup/dr = Σ fUp_n * (dprefac * HU[n] + prefac * dHU[n] * dzhatdr)
    result = complex(0.0)
    for n in 0:nmax
        fu = fup(n)
        iszero(fu) && continue
        term = fu * (dprefac * get_hu(n) + prefac_dzdr * get_dhu(n))
        result += term
        n > 0 && abs(term) < tol * abs(result) + tol && break
    end

    res_down = complex(0.0)
    for n in -1:-1:-nmax
        fu = fup(n)
        iszero(fu) && continue
        term = fu * (dprefac * get_hu(n) + prefac_dzdr * get_dhu(n))
        res_down += term
        abs(term) < tol * abs(res_down) + tol && break
    end

    return result + res_down
end
