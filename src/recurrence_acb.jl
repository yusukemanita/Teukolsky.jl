# ============================================================
#  Native-Acb minimal-solution kernel  f^ν_n  (M3)
#
#  In-place native-Acb clone of compute_fn (recurrence.jl): the three-term
#  coefficients αn/βn/γn (Eq. 124), the modified-Lentz continued fractions
#  Rn/Ln (Eqs. 127-128), and the f_n = ratio·f_{n∓1} recurrence — with the
#  per-op-allocating Complex{Arb} arithmetic replaced by preallocated Acb
#  registers + Arblib in-place `!` ops (every op carries prec=P).  Follows the
#  M2 monodromy kernel (nu_solver.jl).  Reached via compute_fn(...; native=true)
#  when the working type is Arb; the Float64/BigFloat/generic-Arb paths are
#  untouched.
#
#  Coefficient algebra (folded so conjugate-pair products become real ε²):
#    x  = n + ν,   g = x + 1
#    αn = (iεκ)·((g+s)² + ε²)·(g + iτ) / ( g·(2g+1) )
#    βn = Cβ + x·g + D/(x·g),   Cβ = -λ - s(s+1) + ε² + ε(ε-mq),
#                               D  = ε(ε-mq)(s² + ε²)
#    γn = -(iεκ)·((x-s)² + ε²)·(x - iτ) / ( x·(2x-1) )
# ============================================================

"""
    _FnCtxAcb

Preallocated native-Acb context for the f^ν_n recurrence.  Holds the
n-independent Acb constants (built once from `p`, `ν`) plus scratch registers
reused across every αn/βn/γn evaluation and Lentz step.  All Acb fields are
MUTATED in place; never reassigned.
"""
mutable struct _FnCtxAcb
    ν::Acb                       # renormalized angular momentum
    iεκ::Acb; ε2::Acb; iτ::Acb   # recurrence constants
    Cβ::Acb; D::Acb
    s::Int
    # α/β/γ outputs (must not alias the shared scratch)
    αo::Acb; βo::Acb; γo::Acb
    # shared scratch for the coefficient evaluators
    x::Acb; g::Acb; t1::Acb; t2::Acb; t3::Acb
    # Lentz registers
    aj::Acb; bj::Acb; Cr::Acb; Dr::Acb; Δ::Acb; fr::Acb; lt::Acb
    prec::Int
end

# Build the constants in Complex{BigFloat} (validated arithmetic, no hot loop),
# convert to Acb once.  `p`/`ν` may carry Arb or BigFloat components.
function _build_fn_ctx_acb(p, ν; prec::Int=precision(Arb))
    s = p.s
    νb = Complex{BigFloat}(0); iεκ = ε2 = iτ = Cβ = D = Complex{BigFloat}(0)
    setprecision(BigFloat, prec) do
        C  = Complex{BigFloat}
        ε  = C(BigFloat(real(p.ϵ)), BigFloat(imag(p.ϵ)))
        κ  = C(BigFloat(real(p.κ)), BigFloat(imag(p.κ)))
        τ  = C(BigFloat(real(p.τ)), BigFloat(imag(p.τ)))
        λ  = C(BigFloat(real(p.λ)), BigFloat(imag(p.λ)))
        q  = BigFloat(real(p.q)); m = p.m
        νb  = C(BigFloat(real(ν)), BigFloat(imag(ν)))
        iεκ = im*ε*κ
        ε2  = ε^2
        iτ  = im*τ
        εεmq = ε*(ε - m*q)
        Cβ  = -λ - s*(s+1) + ε2 + εεmq
        D   = εεmq * (s^2 + ε2)
    end
    toacb(z) = Acb(z; prec=prec)
    _FnCtxAcb(toacb(νb), toacb(iεκ), toacb(ε2), toacb(iτ), toacb(Cβ), toacb(D), s,
              Acb(0), Acb(0), Acb(0),
              Acb(0), Acb(0), Acb(0), Acb(0), Acb(0),
              Acb(0), Acb(0), Acb(0), Acb(0), Acb(0), Acb(0), Acb(0),
              prec)
end

# x = ν + k  (into ctx.x);  g = x + 1  (into ctx.g)
@inline function _set_x!(ctx::_FnCtxAcb, k::Int)
    P = ctx.prec
    Arblib.add!(ctx.x, ctx.ν, k; prec=P)
    Arblib.add!(ctx.g, ctx.x, 1; prec=P)
    return nothing
end

# αn(k) → ctx.αo.  Uses g,x already? No — computes x,g from k.  Scratch t1,t2,t3.
function _alpha_acb!(ctx::_FnCtxAcb, k::Int)
    P = ctx.prec; _set_x!(ctx, k)
    g = ctx.g
    # t1 = (g+s)^2 + ε2
    Arblib.add!(ctx.t1, g, ctx.s; prec=P)
    Arblib.mul!(ctx.t1, ctx.t1, ctx.t1; prec=P)     # square (aliasing FLINT-safe)
    Arblib.add!(ctx.t1, ctx.t1, ctx.ε2; prec=P)
    # t2 = g + iτ
    Arblib.add!(ctx.t2, g, ctx.iτ; prec=P)
    # t1 = iεκ * t1 * t2
    Arblib.mul!(ctx.t1, ctx.t1, ctx.t2; prec=P)
    Arblib.mul!(ctx.t1, ctx.t1, ctx.iεκ; prec=P)
    # t3 = g*(2g+1)
    Arblib.mul!(ctx.t3, g, 2; prec=P)
    Arblib.add!(ctx.t3, ctx.t3, 1; prec=P)
    Arblib.mul!(ctx.t3, ctx.t3, g; prec=P)
    Arblib.div!(ctx.αo, ctx.t1, ctx.t3; prec=P)
    return nothing
end

# γn(k) → ctx.γo.  Scratch t1,t2,t3.
function _gamma_acb!(ctx::_FnCtxAcb, k::Int)
    P = ctx.prec; _set_x!(ctx, k)
    x = ctx.x
    # t1 = (x-s)^2 + ε2
    Arblib.sub!(ctx.t1, x, ctx.s; prec=P)
    Arblib.mul!(ctx.t1, ctx.t1, ctx.t1; prec=P)     # square (aliasing FLINT-safe)
    Arblib.add!(ctx.t1, ctx.t1, ctx.ε2; prec=P)
    # t2 = x - iτ
    Arblib.sub!(ctx.t2, x, ctx.iτ; prec=P)
    # t1 = -iεκ * t1 * t2
    Arblib.mul!(ctx.t1, ctx.t1, ctx.t2; prec=P)
    Arblib.mul!(ctx.t1, ctx.t1, ctx.iεκ; prec=P)
    Arblib.neg!(ctx.t1, ctx.t1)
    # t3 = x*(2x-1)
    Arblib.mul!(ctx.t3, x, 2; prec=P)
    Arblib.sub!(ctx.t3, ctx.t3, 1; prec=P)
    Arblib.mul!(ctx.t3, ctx.t3, x; prec=P)
    Arblib.div!(ctx.γo, ctx.t1, ctx.t3; prec=P)
    return nothing
end

# βn(k) → ctx.βo.  Scratch t1,t2.
function _beta_acb!(ctx::_FnCtxAcb, k::Int)
    P = ctx.prec; _set_x!(ctx, k)
    # t1 = x*g  (= x*(x+1))
    Arblib.mul!(ctx.t1, ctx.x, ctx.g; prec=P)
    # βo = Cβ + t1 + D/t1
    Arblib.div!(ctx.t2, ctx.D, ctx.t1; prec=P)
    Arblib.add!(ctx.βo, ctx.Cβ, ctx.t1; prec=P)
    Arblib.add!(ctx.βo, ctx.βo, ctx.t2; prec=P)
    return nothing
end

# Modified-Lentz value of a continued fraction whose (aj,bj) are supplied by
# `coef!(ctx, j)` writing into (ctx.aj, ctx.bj).  Returns (value::Acb-in-ctx.fr,
# converged::Bool); midpoint convergence like the M2 kernel.
function _lentz_acb!(ctx::_FnCtxAcb, coef!, maxiter::Int, tol::BigFloat)
    P = ctx.prec
    tiny = ldexp(BigFloat(1), -2P)     # ~ eps^2
    Arblib.set!(ctx.fr, Acb(tiny; prec=P))
    Arblib.set!(ctx.Cr, ctx.fr)
    Arblib.zero!(ctx.Dr)
    tinyacb = Acb(tiny; prec=P)
    conv = false
    for j in 1:maxiter
        coef!(ctx, j)                                   # → ctx.aj, ctx.bj
        # Dr = bj + aj*Dr ; if 0 tiny ; Dr = 1/Dr
        Arblib.mul!(ctx.lt, ctx.aj, ctx.Dr; prec=P)
        Arblib.add!(ctx.Dr, ctx.bj, ctx.lt; prec=P)
        iszero(ctx.Dr) && Arblib.set!(ctx.Dr, tinyacb)
        Arblib.inv!(ctx.Dr, ctx.Dr; prec=P)
        # Cr = bj + aj/Cr ; if 0 tiny
        Arblib.div!(ctx.lt, ctx.aj, ctx.Cr; prec=P)
        Arblib.add!(ctx.Cr, ctx.bj, ctx.lt; prec=P)
        iszero(ctx.Cr) && Arblib.set!(ctx.Cr, tinyacb)
        # Δ = Cr*Dr ; fr *= Δ
        Arblib.mul!(ctx.Δ, ctx.Cr, ctx.Dr; prec=P)
        Arblib.mul!(ctx.fr, ctx.fr, ctx.Δ; prec=P)
        # midpoint convergence: |mid(Δ) - 1| < tol  (midpoints via realref/imagref)
        d = setprecision(BigFloat, P) do
            dr = BigFloat(Arblib.realref(ctx.Δ)) - 1
            di = BigFloat(Arblib.imagref(ctx.Δ))
            hypot(dr, di)
        end
        if d < tol
            conv = true; break
        end
    end
    return conv
end

# a(j),b(j) for Rn (forward, +∞): a(1)=-γ(n); a(j≥2)=-α(n+j-2)γ(n+j-1); b(j)=β(n+j-1)
function _make_R_coef(n::Int)
    return function (ctx::_FnCtxAcb, j::Int)
        P = ctx.prec
        if j == 1
            _gamma_acb!(ctx, n)
            Arblib.neg!(ctx.aj, ctx.γo)
        else
            _alpha_acb!(ctx, n + j - 2)                 # → αo (preserved by _gamma!)
            _gamma_acb!(ctx, n + j - 1)                 # → γo
            Arblib.mul!(ctx.aj, ctx.αo, ctx.γo; prec=P)
            Arblib.neg!(ctx.aj, ctx.aj)
        end
        _beta_acb!(ctx, n + j - 1)
        Arblib.set!(ctx.bj, ctx.βo)
        return nothing
    end
end

# a(j),b(j) for Ln (backward, -∞): a(1)=-α(n); a(j≥2)=-γ(n-j+2)α(n-j+1); b(j)=β(n-j+1)
function _make_L_coef(n::Int)
    return function (ctx::_FnCtxAcb, j::Int)
        P = ctx.prec
        if j == 1
            _alpha_acb!(ctx, n)
            Arblib.neg!(ctx.aj, ctx.αo)
        else
            _gamma_acb!(ctx, n - j + 2)                 # → γo (preserved by _alpha!)
            _alpha_acb!(ctx, n - j + 1)                 # → αo
            Arblib.mul!(ctx.aj, ctx.γo, ctx.αo; prec=P)
            Arblib.neg!(ctx.aj, ctx.aj)
        end
        _beta_acb!(ctx, n - j + 1)
        Arblib.set!(ctx.bj, ctx.βo)
        return nothing
    end
end

"""
    compute_fn_acb(p, ν; nmax=80, nmax_cf=2000, tol=-1) -> Dict{Int,Complex{Arb}}

Native-Acb evaluation of the minimal solution f^ν_n for -nmax ≤ n ≤ nmax
(f_0 = 1).  Full 2·nmax sum (no truncation).  Drop-in for `compute_fn` on an
`MSTParams{Arb}`; values returned as `Complex{Arb}` to match the generic path.
"""
function compute_fn_acb(p, ν; nmax::Int=80, nmax_cf::Int=2000, tol::Real=-1)
    prec = precision(Arb)
    ctx  = _build_fn_ctx_acb(p, ν; prec=prec)
    tolbf = tol > 0 ? BigFloat(tol) : ldexp(BigFloat(16), -prec)   # 16·eps
    f = Dict{Int, Complex{Arb}}()
    f[0] = Complex{Arb}(1)
    fprev = Acb(1; prec=prec)     # running f_{n-1} (forward) / f_{n+1} (backward)
    # forward n = 1..nmax:  f_n = R_n · f_{n-1}
    Arblib.set!(fprev, Acb(1; prec=prec))
    for n in 1:nmax
        conv = _lentz_acb!(ctx, _make_R_coef(n), nmax_cf, tolbf)
        conv || @warn "compute_fn_acb: Rn CF not converged (n=$n)"
        Arblib.mul!(ctx.fr, ctx.fr, fprev; prec=prec)   # f_n = R_n * f_{n-1}
        f[n] = Complex{Arb}(ctx.fr)
        Arblib.set!(fprev, ctx.fr)
    end
    # backward n = -1..-nmax:  f_n = L_n · f_{n+1}
    Arblib.set!(fprev, Acb(1; prec=prec))
    for n in -1:-1:-nmax
        conv = _lentz_acb!(ctx, _make_L_coef(n), nmax_cf, tolbf)
        conv || @warn "compute_fn_acb: Ln CF not converged (n=$n)"
        Arblib.mul!(ctx.fr, ctx.fr, fprev; prec=prec)   # f_n = L_n * f_{n+1}
        f[n] = Complex{Arb}(ctx.fr)
        Arblib.set!(fprev, ctx.fr)
    end
    return f
end
