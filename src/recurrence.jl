# ============================================================
#  Three-term recurrence coefficients, Eq. (124)
# ============================================================

function αn(p, ν, n)
    nν = n + ν
    s, ϵ, κ, τ = p.s, p.ϵ, p.κ, p.τ
    im * ϵ * κ * (nν + 1 + s + im*ϵ) * (nν + 1 + s - im*ϵ) * (nν + 1 + im*τ) /
        ((nν + 1) * (2nν + 3))
end

function βn(p, ν, n)
    nν = n + ν
    s, ϵ, τ, λ, m, q = p.s, p.ϵ, p.τ, p.λ, p.m, p.q
    -λ - s*(s+1) + nν*(nν+1) + ϵ^2 + ϵ*(ϵ - m*q) +
        ϵ*(ϵ - m*q) * (s^2 + ϵ^2) / (nν * (nν + 1))
end

function γn(p, ν, n)
    nν = n + ν
    s, ϵ, κ, τ = p.s, p.ϵ, p.κ, p.τ
    -im * ϵ * κ * (nν - s + im*ϵ) * (nν - s - im*ϵ) * (nν - im*τ) /
        (nν * (2nν - 1))
end

# ============================================================
#  Continued fractions for R_n and L_n, Eqs. (127)-(128)
#
#  Evaluated by the modified Lentz algorithm with a convergence check (instead
#  of a fixed backward-recurrence window): self-sizing and accurate to the
#  working precision, with a warning if the iteration cap is hit.
# ============================================================

"""
    _lentz_cf(a, b, T; tol, maxiter)

Modified Lentz evaluation (Numerical Recipes §5.2) of the continued fraction

    a(1) / (b(1) + a(2) / (b(2) + a(3) / (b(3) + ⋯)))

iterating until the update factor is within `tol` of 1. Returns `(value, converged)`.
"""
function _lentz_cf(a, b, ::Type{T}; tol, maxiter::Int) where {T<:Complex}
    tiny = T(eps(real(T))^2)
    f = tiny
    C = f
    D = zero(T)
    for j in 1:maxiter
        aj = a(j); bj = b(j)
        D = bj + aj * D; iszero(D) && (D = tiny); D = inv(D)
        C = bj + aj / C; iszero(C) && (C = tiny)
        Δ = C * D
        f *= Δ
        abs(Δ - one(real(T))) < tol && return f, true
    end
    return f, false
end

"""
    Rn_cf(p, ν, n; nmax=2000, tol=-1)

Compute R_n = f_n/f_{n-1} via the continued fraction going to +∞, evaluated by
convergence-checked Lentz iteration. `nmax` is the iteration cap; `tol` defaults
to ~16·eps of the working precision.
"""
function Rn_cf(p::MSTParams, ν, n; nmax::Int=2000, tol::Real=-1)
    T = typeof(p.ϵ)
    tol_use = tol < 0 ? 16 * eps(real(T)) : real(T)(tol)
    a(j) = j == 1 ? -γn(p, ν, n) : -αn(p, ν, n + j - 2) * γn(p, ν, n + j - 1)
    b(j) = βn(p, ν, n + j - 1)
    R, conv = _lentz_cf(a, b, T; tol=tol_use, maxiter=nmax)
    conv || @warn "Rn_cf: continued fraction did not converge in $nmax terms (n=$n)"
    return R
end

"""
    Ln_cf(p, ν, n; nmax=2000, tol=-1)

Compute L_n = f_n/f_{n+1} via the continued fraction going to −∞, evaluated by
convergence-checked Lentz iteration.
"""
function Ln_cf(p::MSTParams, ν, n; nmax::Int=2000, tol::Real=-1)
    T = typeof(p.ϵ)
    tol_use = tol < 0 ? 16 * eps(real(T)) : real(T)(tol)
    a(j) = j == 1 ? -αn(p, ν, n) : -γn(p, ν, n - j + 2) * αn(p, ν, n - j + 1)
    b(j) = βn(p, ν, n - j + 1)
    L, conv = _lentz_cf(a, b, T; tol=tol_use, maxiter=nmax)
    conv || @warn "Ln_cf: continued fraction did not converge in $nmax terms (n=$n)"
    return L
end

# ============================================================
#  Compute minimal solution {f^ν_n}, Eqs. (123), (134)
# ============================================================

"""
    compute_fn(p, ν; nmax=80)

Compute the minimal solution f^ν_n for -nmax ≤ n ≤ nmax.
Normalized so that f_0 = 1.
"""
function compute_fn(p::MSTParams, ν; nmax::Int=80, nmax_cf::Int=2000)
    T = typeof(p.ϵ)
    f = Dict{Int, T}()
    f[0] = one(T)

    for n in 1:nmax
        R = Rn_cf(p, ν, n; nmax=nmax_cf)
        f[n] = R * f[n-1]
    end

    for n in -1:-1:-nmax
        L = Ln_cf(p, ν, n; nmax=nmax_cf)
        f[n] = L * f[n+1]
    end

    return f
end

"""
    compute_fn_truncated(p, ν, nmin; nmax=80)

Like `compute_fn` but with f_n = 0 for n < nmin.
"""
function compute_fn_truncated(p::MSTParams, ν, nmin::Int; nmax::Int=80,
                              nmax_cf::Int=2000)
    T = typeof(p.ϵ)
    f = Dict{Int, T}()

    for n in -nmax:nmin-1
        f[n] = zero(T)
    end

    n0 = max(0, nmin)
    f[n0] = one(T)

    for n in n0+1:nmax
        R = Rn_cf(p, ν, n; nmax=nmax_cf)
        f[n] = R * f[n-1]
    end

    for n in n0-1:-1:nmin
        L = Ln_cf(p, ν, n; nmax=nmax_cf)
        f[n] = L * f[n+1]
    end

    return f
end
