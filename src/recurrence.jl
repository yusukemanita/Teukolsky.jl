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
function compute_fn(p::MSTParams, ν; nmax::Int=80, nmax_cf::Int=2000,
                   tol::Real=-1, min_terms::Int=8, n_consec::Int=3)
    T  = typeof(p.ϵ)
    RT = real(T)
    # Adaptive early termination (optimization C): the minimal solution f^ν_n
    # decays away from n=0, so once the tail is negligible we stop instead of
    # always evaluating a fixed 2·nmax continued fractions.  nmax stays a HARD
    # upper cap (can only shorten, never extend).  tol defaults to eps(real T) —
    # the machine-precision floor — NOT eps^{3/4}: downstream amplitude sums
    # (compute_Aminus, compute_Knu) weight f_n by Pochhammer/Γ factors that GROW
    # like n^{2ν+2s}, so a term negligible in |f_n| is not automatically negligible
    # in the weighted summand.  Truncating only at the precision floor keeps the
    # dropped weighted tail below round-off for all l/s while still cutting ~1.6×
    # of the continued-fraction work.  f_n is not perfectly monotone, so we break
    # only after n_consec CONSECUTIVE terms fall below tol·(running max |f|) and at
    # least min_terms terms have been taken.
    τol = tol < 0 ? eps(RT) : RT(tol)
    f = Dict{Int, T}()
    f[0] = one(T)

    # Forward: stop the (expensive) continued-fraction evals once the tail is
    # negligible, then fill the rest of the range with exact zeros so downstream
    # consumers that index fn[n] directly (e.g. Σfn in compute_amplitudes) still
    # see the full -nmax:nmax range.  The zeroed terms are < τol·|f|max, so this
    # is below the precision floor.
    nstop = nmax; fmax = one(RT); nsmall = 0
    for n in 1:nmax
        R = Rn_cf(p, ν, n; nmax=nmax_cf)
        f[n] = R * f[n-1]
        af = abs(f[n]); fmax = max(fmax, af)
        nsmall = af < τol * fmax ? nsmall + 1 : 0
        if n >= min_terms && nsmall >= n_consec
            nstop = n; break
        end
    end
    for n in nstop+1:nmax
        f[n] = zero(T)
    end

    nstop = -nmax; fmax = one(RT); nsmall = 0   # independent decay on the n<0 side
    for n in -1:-1:-nmax
        L = Ln_cf(p, ν, n; nmax=nmax_cf)
        f[n] = L * f[n+1]
        af = abs(f[n]); fmax = max(fmax, af)
        nsmall = af < τol * fmax ? nsmall + 1 : 0
        if -n >= min_terms && nsmall >= n_consec
            nstop = n; break
        end
    end
    for n in nstop-1:-1:-nmax
        f[n] = zero(T)
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
