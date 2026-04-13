# ============================================================
#  Three-term recurrence coefficients, Eq. (124)
# ============================================================

function αn(p::MSTParams, ν, n)
    nν = n + ν
    s, ϵ, κ, τ = p.s, p.ϵ, p.κ, p.τ
    im * ϵ * κ * (nν + 1 + s + im*ϵ) * (nν + 1 + s - im*ϵ) * (nν + 1 + im*τ) /
        ((nν + 1) * (2nν + 3))
end

function βn(p::MSTParams, ν, n)
    nν = n + ν
    s, ϵ, τ, λ, m, q = p.s, p.ϵ, p.τ, p.λ, p.m, p.q
    -λ - s*(s+1) + nν*(nν+1) + ϵ^2 + ϵ*(ϵ - m*q) +
        ϵ*(ϵ - m*q) * (s^2 + ϵ^2) / (nν * (nν + 1))
end

function γn(p::MSTParams, ν, n)
    nν = n + ν
    s, ϵ, κ, τ = p.s, p.ϵ, p.κ, p.τ
    -im * ϵ * κ * (nν - s + im*ϵ) * (nν - s - im*ϵ) * (nν - im*τ) /
        (nν * (2nν - 1))
end

# ============================================================
#  Continued fractions for R_n and L_n, Eqs. (127)-(128)
# ============================================================

"""
    Rn_cf(p, ν, n; nmax=150)

Compute R_n = f_n/f_{n-1} via continued fraction going to +∞.
"""
function Rn_cf(p::MSTParams, ν, n; nmax=150)
    R = zero(p.ϵ)
    for k in nmax:-1:n
        α_k = αn(p, ν, k)
        β_k = βn(p, ν, k)
        γ_k = γn(p, ν, k)
        R = -γ_k / (β_k + α_k * R)
    end
    return R
end

"""
    Ln_cf(p, ν, n; nmax=150)

Compute L_n = f_n/f_{n+1} via continued fraction going to -∞.
"""
function Ln_cf(p::MSTParams, ν, n; nmax=150)
    L = zero(p.ϵ)
    for k in -nmax:n
        α_k = αn(p, ν, k)
        β_k = βn(p, ν, k)
        γ_k = γn(p, ν, k)
        L = -α_k / (β_k + γ_k * L)
    end
    return L
end

# ============================================================
#  Compute minimal solution {f^ν_n}, Eqs. (123), (134)
# ============================================================

"""
    compute_fn(p, ν; nmax=60)

Compute the minimal solution f^ν_n for -nmax ≤ n ≤ nmax.
Normalized so that f_0 = 1.
"""
function compute_fn(p::MSTParams, ν; nmax::Int=60)
    T = typeof(p.ϵ)
    f = Dict{Int, T}()
    f[0] = one(T)

    for n in 1:nmax
        R = Rn_cf(p, ν, n)
        f[n] = R * f[n-1]
    end

    for n in -1:-1:-nmax
        L = Ln_cf(p, ν, n)
        f[n] = L * f[n+1]
    end

    return f
end

"""
    compute_fn_truncated(p, ν, nmin; nmax=60)

Like `compute_fn` but with f_n = 0 for n < nmin.
"""
function compute_fn_truncated(p::MSTParams, ν, nmin::Int; nmax::Int=60)
    T = typeof(p.ϵ)
    f = Dict{Int, T}()

    for n in -nmax:nmin-1
        f[n] = zero(T)
    end

    n0 = max(0, nmin)
    f[n0] = one(T)

    for n in n0+1:nmax
        R = Rn_cf(p, ν, n)
        f[n] = R * f[n-1]
    end

    for n in n0-1:-1:nmin
        L = Ln_cf(p, ν, n)
        f[n] = L * f[n+1]
    end

    return f
end
