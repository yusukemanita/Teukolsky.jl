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

# Midpoint magnitude of an Acb as a Float64 (guard arithmetic only — the guard
# compares magnitude RATIOS, so Float64 range/precision is ample; overflow to Inf
# makes the guard trip, which falls back conservatively to a fresh Lentz call).
@inline _absmid64(x::Acb) = hypot(Float64(Arblib.realref(x)), Float64(Arblib.imagref(x)))

"""
    _cf_ratios_acb!(ratios, ctx, dir, nmax, nmax_cf, tolbf) -> ratios

Native-Acb CF-ratio peeling — the in-place mirror of the generic `_cf_ratios`
(recurrence.jl): ONE anchor Lentz evaluation at n = ±nmax, then interior ratios
peeled by the CF's own backward recursion

    R_n = -γ_n / (β_n + α_n R_{n+1}),     L_n = -α_n / (β_n + γ_n L_{n-1}),

one in-place division each.  `ratios[k]` receives R_k (dir=+1) resp. L_{-k}
(dir=-1).  The same fresh-start guard as the generic version: on a non-finite
or heavily-cancelling peel denominator (near-integer-ν CF poles) that single n
falls back to a direct in-place Lentz evaluation, degrading exactly to the old
per-n behavior where peeling would lose bits.  Guard magnitudes are Float64
midpoints (ample for a ratio test; underflows only beyond ~4000 bits, where the
guard then trips on non-finiteness alone).
"""
function _cf_ratios_acb!(ratios::Vector{Acb}, ctx::_FnCtxAcb, dir::Int,
                         nmax::Int, nmax_cf::Int, tolbf::BigFloat)
    P = ctx.prec
    nmax == 0 && return ratios
    cancel_floor = exp2((1.0 - P) / 4)        # ≈ (eps at prec)^(1/4), cf. generic
    # anchor: one full Lentz at the far end
    conv = _lentz_acb!(ctx, dir == +1 ? _make_R_coef(nmax) : _make_L_coef(-nmax),
                       nmax_cf, tolbf)
    conv || @warn "compute_fn_acb: anchor CF not converged (dir=$dir, |n|=$nmax)"
    Arblib.set!(ratios[nmax], ctx.fr)
    for k in nmax-1:-1:1
        n = dir == +1 ? k : -k
        if dir == +1
            _alpha_acb!(ctx, n)                                # → αo
            Arblib.mul!(ctx.Dr, ctx.αo, ratios[k+1]; prec=P)   # α_n·R_{n+1}
        else
            _gamma_acb!(ctx, n)                                # → γo
            Arblib.mul!(ctx.Dr, ctx.γo, ratios[k+1]; prec=P)   # γ_n·L_{n-1}
        end
        wing = _absmid64(ctx.Dr)
        _beta_acb!(ctx, n)                                     # → βo (Dr preserved)
        Arblib.add!(ctx.Dr, ctx.βo, ctx.Dr; prec=P)            # den = β_n + wing
        den = _absmid64(ctx.Dr)
        if !isfinite(den) || den < cancel_floor * (_absmid64(ctx.βo) + wing)
            # fresh start: direct Lentz for this single n (clobbers ctx registers;
            # everything needed next iteration is recomputed from ratios[k])
            conv = _lentz_acb!(ctx, dir == +1 ? _make_R_coef(n) : _make_L_coef(n),
                               nmax_cf, tolbf)
            conv || @warn "compute_fn_acb: CF not converged (n=$n)"
            Arblib.set!(ratios[k], ctx.fr)
            continue
        end
        if dir == +1
            _gamma_acb!(ctx, n)                                # → γo (Dr preserved)
            Arblib.div!(ratios[k], ctx.γo, ctx.Dr; prec=P)
        else
            _alpha_acb!(ctx, n)                                # → αo (Dr preserved)
            Arblib.div!(ratios[k], ctx.αo, ctx.Dr; prec=P)
        end
        Arblib.neg!(ratios[k], ratios[k])
    end
    return ratios
end

"""
    _compute_fn_acb_vec(p, ν; nmax=80, nmax_cf=2000, tol=-1) -> Vector{Acb}

Core-private variant of [`compute_fn_acb`](@ref): identical values, but the
minimal solution is returned as a DENSE `Vector{Acb}` `fv` with
`fv[n + nmax + 1] = f^ν_n` (n = -nmax…nmax), skipping the per-value
`Complex{Arb}` boxing of the public Dict.  The CF ratios come from ONE anchor
Lentz call per direction plus O(1) in-place peeling per n
([`_cf_ratios_acb!`](@ref)).  Consumed directly by the internal A^ν_± vector
kernels (`_Aplus_acb`/`_Aminus_acb`) and by `compute_mst_core_acb`, which
builds the public Dict once via [`_fn_dict_from_vec`](@ref).
"""
function _compute_fn_acb_vec(p, ν; nmax::Int=80, nmax_cf::Int=2000, tol::Real=-1)
    prec = precision(Arb)
    # NEAR-INTEGER-ν GATE: with ν within ~1e-3 of a real integer AND small |ε|
    # (deep-IR branch cut, where ν(l) → l), the backward peel crosses CF poles
    # (|L_n| swings ~8 orders within a few n) and the kernel's FOLDED coefficient
    # algebra ((x−s)²+ε² …) loses ~7 digits crossing them — the products then
    # amplify to O(1) in f_{-n} (caught by the Gpia T0b Wolfram cross-check at
    # σ = 5e-4, l′ = 5).  The generic Complex{Arb} ratio peel evaluates the
    # unfolded coefficients and stays exact there, so route this rare corner
    # through it (same gating pattern as the monodromy resonance gate).
    νm = Complex{BigFloat}(ν)
    if abs(imag(νm)) < 1e-3 && abs(real(νm) - round(real(νm))) < 1e-3
        f = compute_fn(p, ν; nmax=nmax, nmax_cf=nmax_cf, tol=tol)
        fv = Vector{Acb}(undef, 2*nmax + 1)
        for n in -nmax:nmax
            fv[n + nmax + 1] = Acb(f[n]; prec=prec)
        end
        return fv
    end
    ctx  = _build_fn_ctx_acb(p, ν; prec=prec)
    tolbf = tol > 0 ? BigFloat(tol) : ldexp(BigFloat(16), -prec)   # 16·eps
    off = nmax + 1                       # fv[n + off] = f_n
    fv = Vector{Acb}(undef, 2*nmax + 1)
    fv[off] = Acb(1; prec=prec)          # f_0 = 1
    nmax == 0 && return fv
    ratios = [Acb(0; prec=prec) for _ in 1:nmax]
    # forward n = 1..nmax:  f_n = R_n · f_{n-1}
    _cf_ratios_acb!(ratios, ctx, +1, nmax, nmax_cf, tolbf)
    for n in 1:nmax
        fv[n + off] = Acb(0; prec=prec)
        Arblib.mul!(fv[n + off], fv[n - 1 + off], ratios[n]; prec=prec)
    end
    # backward n = -1..-nmax:  f_{-k} = L_{-k} · f_{-k+1}
    _cf_ratios_acb!(ratios, ctx, -1, nmax, nmax_cf, tolbf)
    for k in 1:nmax
        fv[-k + off] = Acb(0; prec=prec)
        Arblib.mul!(fv[-k + off], fv[-k + 1 + off], ratios[k]; prec=prec)
    end
    return fv
end

"""
    _fn_dict_from_vec(fv, nmax) -> Dict{Int,Complex{Arb}}

Public-contract Dict (`fn[n] = f^ν_n`) built once from the dense Acb vector of
[`_compute_fn_acb_vec`](@ref).
"""
_fn_dict_from_vec(fv::Vector{Acb}, nmax::Int) = Dict{Int, Complex{Arb}}(
    n => Complex{Arb}(fv[n + nmax + 1]) for n in -nmax:nmax)

"""
    compute_fn_acb(p, ν; nmax=80, nmax_cf=2000, tol=-1) -> Dict{Int,Complex{Arb}}

Native-Acb evaluation of the minimal solution f^ν_n for -nmax ≤ n ≤ nmax
(f_0 = 1).  Full 2·nmax sum (no truncation).  Drop-in for `compute_fn` on an
`MSTParams{Arb}`; values returned as `Complex{Arb}` to match the generic path.

The CF ratios come from ONE anchor Lentz call per direction plus O(1) in-place
peeling per n ([`_cf_ratios_acb!`](@ref)) — the same O(nmax·depth) → O(nmax+depth)
strategy as the generic `_cf_ratios`, on top of the kernel's zero-allocation
arithmetic.
"""
compute_fn_acb(p, ν; nmax::Int=80, nmax_cf::Int=2000, tol::Real=-1) =
    _fn_dict_from_vec(
        _compute_fn_acb_vec(p, ν; nmax=nmax, nmax_cf=nmax_cf, tol=tol), nmax)
