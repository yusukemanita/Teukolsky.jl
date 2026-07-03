# ============================================================
#  Monodromy method for cos(2πν)
# ============================================================

# ------------------------------------------------------------
#  Reusable monodromy context
#
#  The recurrence arrays a1[k], a2[k], Poch_p[k], Poch_m[k] are
#  TRUNCATION-INDEPENDENT: each entry depends only on lower-index entries and
#  on k, never on the chosen truncation n.  Only the closing sums depend on n.
#  So the arrays are built ONCE up to a target depth and the closed-form value
#  of cos(2πν) is evaluated at any n ≤ depth — and the depth can be EXTENDED by
#  continuing the recurrence (no rebuild from scratch).  See `_monodromy_value`.
# ------------------------------------------------------------
mutable struct _MonodromyCtx{R<:AbstractFloat, C<:Complex{R}}
    αε::C; γCH::C; δCH::C; εCH::C; qCH::C
    μ1C::C; μ2C::C
    a1::Vector{C}; a2::Vector{C}
    Poch_p::Vector{C}; Poch_m::Vector{C}
    nfilled::Int          # recurrence computed through truncation `nfilled`
                          # (valid indices 1 … nfilled+1 in every array)
end

"""
    _build_monodromy_ctx(s, l, m, a, ω, λ, nmax)

Build the (truncation-independent) monodromy recurrence arrays up to truncation
`nmax`.  Returns a `_MonodromyCtx` from which `_monodromy_value(ctx, n)` gives
cos(2πν) for any `n ≤ nmax`, and `_extend_monodromy_ctx!(ctx, n2)` deepens it.
"""
function _build_monodromy_ctx(s, _l, m, a, ω, λ, nmax::Int)
    # Infer complex type from inputs
    R = promote_type(typeof(float(real(a))), typeof(float(real(complex(ω)))))
    C = Complex{R}

    q  = R(a)
    ε  = 2 * C(ω)
    κ  = sqrt(C(1 - q^2))
    τ  = (ε - m*q) / κ

    γCH = 1 - s - im*ε - im*τ
    δCH = 1 + s + im*ε - im*τ
    εCH = 2im*ε*κ
    αε  = C(1 - s) + im*(ε - τ)
    qCH = s*(s+1) - ε^2 + im*(1-2s)*ε*κ + λ + im*τ + τ^2

    μ1C = αε - (γCH + δCH)
    μ2C = -αε

    cap = max(nmax, 1) + 2
    a1 = zeros(C, cap)
    a2 = zeros(C, cap)
    a1[1] = one(C); a2[1] = one(C)
    Poch_p = ones(C, cap)
    Poch_m = ones(C, cap)

    ctx = _MonodromyCtx{R,C}(αε, γCH, δCH, εCH, qCH, μ1C, μ2C,
                             a1, a2, Poch_p, Poch_m, 0)
    return _extend_monodromy_ctx!(ctx, nmax)
end

"""
    _extend_monodromy_ctx!(ctx, target)

Continue the monodromy recurrence in-place from `ctx.nfilled` up to truncation
`target` (no-op if already deep enough).  No values are recomputed — the
recurrence simply marches forward, so this is cheap relative to a rebuild.
"""
function _extend_monodromy_ctx!(ctx::_MonodromyCtx{R,C}, target::Int) where {R,C}
    target ≤ ctx.nfilled && return ctx

    a1, a2 = ctx.a1, ctx.a2
    Poch_p, Poch_m = ctx.Poch_p, ctx.Poch_m
    need = target + 2
    if length(a1) < need
        for v in (a1, a2)
            old = length(v); resize!(v, need); @views v[old+1:end] .= zero(C)
        end
        for v in (Poch_p, Poch_m)
            old = length(v); resize!(v, need); @views v[old+1:end] .= one(C)
        end
    end

    αε, γCH, δCH = ctx.αε, ctx.γCH, ctx.δCH
    εCH, qCH     = ctx.εCH, ctx.qCH
    μ1C, μ2C     = ctx.μ1C, ctx.μ2C

    # `_strip_radius` keeps the Complex{Arb} recurrence in POINT arithmetic
    # (midpoints computed exactly to working precision, exactly as MPFR/BigFloat):
    # carrying ball radii through the factorially-growing a1/a2 terms corrupts the
    # MIDPOINT at large |ω| (the cancellation in the cos(2πν) closed form pushes
    # the certified radius past the midpoint, and at large Im(ν) the midpoint
    # itself goes NaN), so the ν extraction breaks at some high-|ω| angles.  It is
    # a strict no-op for Float64/BigFloat (generic `_strip_radius(::Complex)=z`),
    # so those paths stay byte-for-byte identical.
    for n in (ctx.nfilled + 1):target
        a1p = a1[n];  a1pp = n >= 2 ? a1[n-1] : zero(C)
        c2 = (αε - (n-1+δCH)) * (αε - (n-2+γCH+δCH)) * εCH / n
        c1 = (αε^2 + αε*(1-2n-γCH-δCH+εCH) +
              (n^2 - qCH + n*(-1+γCH+δCH-εCH) + εCH - δCH*εCH)) / n
        a1[n+1] = _strip_radius(c2*a1pp - c1*a1p)

        a2p = a2[n];  a2pp = n >= 2 ? a2[n-1] : zero(C)
        d2 = (αε + (n-2)) * (αε + (n-1-γCH)) * εCH / n
        d1 = (αε^2 + (n^2 - qCH + γCH + δCH - n*(1+γCH+δCH-εCH) - εCH) +
              αε*(-1+2n-γCH-δCH+εCH)) / n
        a2[n+1] = _strip_radius(-d2*a2pp + d1*a2p)

        # Pochhammer factors (original loop index i ≡ n): Poch_x[i+1] = (…)·Poch_x[i]
        Poch_p[n+1] = _strip_radius((-μ2C + μ1C + n - 1) * Poch_p[n])
        Poch_m[n+1] = _strip_radius(( μ2C - μ1C + n - 1) * Poch_m[n])
    end
    ctx.nfilled = target
    return ctx
end

"""
    _monodromy_value(ctx, n)

Evaluate cos(2πν) from a prebuilt `ctx` at truncation `n` (requires
`n ≤ ctx.nfilled`).  Identical closed form to `monodromy_cos2pi_nu`, with a
resonance-safe fallback: if the standard factored form comes back non-finite
(the PIA resonance — see `_monodromy_value_safe`), the pole-free marching-Γ
evaluation of the SAME expression is used instead.
"""
function _monodromy_value(ctx::_MonodromyCtx{R,C}, n::Int) where {R,C}
    a1, a2 = ctx.a1, ctx.a2
    Poch_p, Poch_m = ctx.Poch_p, ctx.Poch_m
    μ1C, μ2C = ctx.μ1C, ctx.μ2C

    # PIA-resonance gate: when μd = μ1C − μ2C sits (numerically) ON an integer —
    # exactly what happens for ω = iσ with 4σ ∈ ℤ, where μd = −2s − 4σ cancels
    # EXACTLY — the factored Γ(±μd)·Poch form below is an exact 0·∞.  Depending
    # on precision that yields NaN or, worse, FINITE GARBAGE (verified: at
    # σ = 2, 300 bits it returned a finite ν whose CF residual is O(10)), so
    # detect the resonance up front rather than testing the result.
    if _monodromy_resonant(μ1C - μ2C)
        return _monodromy_value_safe(ctx, n)
    end

    jmax = cld(n, 2)
    a1sum = _cgamma(-μ2C + μ1C) * sum(a1[j+1] * Poch_p[n-j+1] for j in 0:jmax)
    a2sum = _cgamma( μ2C - μ1C) * sum((-1)^j * a2[j+1] * Poch_m[n-j+1] for j in 0:jmax)

    # NOTE: `2π^2` is a Float64 literal (≈19.7392 to 16 digits) and would cap
    # the BigFloat path at ~1e-16; use the full-precision `2*R(π)^2`.
    # `complex(...)` keeps the numerator's float type: Base promotes
    # `BigFloat/Complex{BigFloat}` to Complex{BigFloat} on its own, but Arblib
    # promotes `Arb/Complex{Arb}` to the foreign `Acb` — which then breaks the
    # complex-ω branch (`complex(::Acb)`) downstream.  Making it complex first
    # keeps the result Complex{R} for every R; byte-identical for Float64/BigFloat.
    v = cos(π*(μ1C - μ2C)) +
        (complex(2*R(π)^2) / (a1sum * a2sum)) * (-1)^(n-1) * a1[n+1] * a2[n+1]
    (isfinite(real(v)) && isfinite(imag(v))) && return v
    return _monodromy_value_safe(ctx, n)
end

# Is μd = μ1C − μ2C numerically ON an integer (⇒ Γ(±μd) pole / exact-zero
# Pochhammer factor in the factored monodromy form)?  A generous window is safe:
# the marching-Γ form matches the factored form to working precision everywhere,
# so a false positive only changes rounding at the ~eps level.
function _monodromy_resonant(μd)
    return abs(imag(μd)) < 1e-3 && abs(real(μd) - round(real(μd))) < 1e-3
end

"""
    _monodromy_value_safe(ctx, n)

Pole-free evaluation of the SAME cos(2πν) closed form, for the PIA resonance.

For purely-imaginary ω = iσ the parameter μd = μ1C − μ2C = −2s + 2iε = −2s − 4σ
is REAL and hits an exact integer whenever 4σ ∈ ℤ (independent of s, l, m, a —
2s is an even integer).  There the standard factored form degenerates:
Γ(±μd) sits exactly on a pole while the Pochhammer products (±μd)_k contain an
exact zero factor, so `Γ · Σ(Poch·a)` is an exact 0·∞ → NaN.  The pole is an
artifact of the FACTORING, not of the expression: Γ(z)·(z)_k = Γ(z+k) exactly,
and every Γ argument appearing in the sums is μd + (n−j) with n−j ≥ n/2 ≫ |μd|
— far from any pole.  So evaluate the products directly: seed g = Γ(±μd + n)
once and march it down with one division per term, Γ(z−1) = Γ(z)/(z−1).

Off resonance this matches the standard form to working precision (verified to
~1e-76 at 256 bits); it is only invoked when the standard form is non-finite.
"""
function _monodromy_value_safe(ctx::_MonodromyCtx{R,C}, n::Int) where {R,C}
    a1, a2 = ctx.a1, ctx.a2
    μd = ctx.μ1C - ctx.μ2C

    jmax = cld(n, 2)
    g1 = _cgamma(μd + n)         # Γ(μd)·(μd)_{n−j}  ≡ Γ(μd + n − j); j = 0 term
    g2 = _cgamma(-μd + n)        # Γ(−μd)·(−μd)_{n−j} ≡ Γ(−μd + n − j)
    t1 = zero(C); t2 = zero(C)
    for j in 0:jmax
        t1 += a1[j+1] * g1
        t2 += (iseven(j) ? a2[j+1] : -a2[j+1]) * g2
        g1 /= (μd + n - j - 1)   # Γ(z−1) = Γ(z)/(z−1); argument stays ≥ n/2 − |μd|
        g2 /= (-μd + n - j - 1)
    end

    return cos(π*μd) +
           (complex(2*R(π)^2) / (t1 * t2)) * (-1)^(n-1) * a1[n+1] * a2[n+1]
end

"""
    monodromy_cos2pi_nu(s, l, m, a, ω, λ; nmax=60)

Compute cos(2πν) from the monodromy of the confluent Heun equation.
Works for any numeric precision (Float64, BigFloat, etc.).
"""
function monodromy_cos2pi_nu(s, _l, m, a, ω, λ; nmax::Int=60)
    ctx = _build_monodromy_ctx(s, _l, m, a, ω, λ, nmax)
    return _monodromy_value(ctx, nmax)
end

"""
    _monodromy_adaptive(s, l, m, a, ω, λ; R, nmax0=60)

Compute cos(2πν) with a precision-aware series length. For Float64 the series
length is fixed at `nmax0` (raising it overflows the factorial/Pochhammer
products to NaN). For higher precision the length is chosen from the working
precision and verified by reusing a SINGLE set of recurrence arrays (no
rebuild-from-scratch doubling): cos(2πν) is evaluated at the target `nmax` and
at `nmax−Δ` from the same arrays, and if they have not yet agreed to ~eps(R)
the arrays are EXTENDED (the recurrence marches forward) and re-checked.
"""
function _monodromy_adaptive(s, l, m, a, ω, λ; R, nmax0::Int=60)
    if R === Float64 || R === Float32
        return monodromy_cos2pi_nu(s, l, m, a, ω, λ; nmax=nmax0)
    end
    if R <: MultiFloats.MultiFloat
        # MultiFloat extends the MANTISSA but keeps Float64's EXPONENT range, so
        # the deep monodromy series (factorial/Pochhammer products at
        # nmax ≈ 4.71·prec) overflows to Inf/NaN exactly as Float64 does.  Run the
        # single scalar cos(2πν) series in BigFloat at the MultiFloat working
        # precision and convert back — the rest of the MST pipeline (f_n
        # recurrence, amplitude sums) stays native MultiFloat.
        prec = precision(R)
        ωc   = complex(ω)
        cbig = setprecision(BigFloat, prec) do
            _monodromy_adaptive(s, l, m,
                BigFloat(real(a)),
                Complex{BigFloat}(BigFloat(real(ωc)), BigFloat(imag(ωc))),
                Complex{BigFloat}(BigFloat(real(λ)), BigFloat(imag(λ)));
                R=BigFloat, nmax0=nmax0)
        end
        return Complex{R}(R(real(cbig)), R(imag(cbig)))
    end
    prec = precision(R)
    Δ    = 128
    # Truncation depth: MEASURED envelope, not the old 4.71·prec floor.
    # Mapping |value(n) − value(n_deep)| against n (n_deep = 4.71·prec + 512,
    # recurrence at prec+64 guard bits) over σ ∈ {0.25 … 20} (ω = iσ, PIA),
    # real/complex ω, prec ∈ {256 … 1536}, s ∈ {−2,2}, l ∈ {2,4,10,20},
    # a ∈ {0.7,0.9} shows a clean geometric decay of ~0.9 bits/step down to the
    # rounding floor — no plateaus or humps — with the 2^(−prec) crossing at
    #     n*(prec, |ε|) ≤ 1.04·prec + 13·|ε| + 40      (|ε| = 2|ω|)
    # across the whole grid (worst slack ≈ 39; l and a are irrelevant; s = +2
    # costs ≤ 60 over s = −2, absorbed in the constants).  The old 4.71·prec
    # start was a ~4× overshoot (its "0.064 digits/step" fit does not match the
    # actual series).  Start at the envelope + 2Δ so the THREE-point acceptance
    # below passes on the first try; the verify-and-extend loop remains the
    # backstop for anything outside the mapped envelope.
    # See test/test_monodromy_depth.jl for the arbiter-backed validation.
    ωc   = complex(ω)
    εf   = 2 * hypot(Float64(real(ωc)), Float64(imag(ωc)))   # |ε| = 2|ω|
    nmax = clamp(ceil(Int, 1.04 * prec + 13 * εf + 40) + 2Δ,
                 max(nmax0, 120) + 2Δ, 4000)
    # Acceptance tolerance: RELATIVE 2^(−0.85·prec).  The old 16·eps(R)
    # (= 2^(−prec+4)) sits BELOW the point-arithmetic noise floor of the closed
    # form (measured 2^(−prec+6…9) for |c(n)−c(n−Δ)|/|c| at ANY depth), so it
    # could never be satisfied and the driver always churned to the 4000 cap.
    # 0.85·prec is comfortably above the noise floor and far below the ν gate
    # (values still agree with a deep reference to ≲2^(−prec+10)).
    tol  = ldexp(one(R), -fld(17 * prec, 20))

    ctx = _build_monodromy_ctx(s, l, m, a, ω, λ, nmax)
    c   = _monodromy_value(ctx, nmax)
    for _ in 1:32
        # THREE-depth acceptance (n, n−Δ, n−2Δ): two evaluations 128 apart could
        # in principle agree spuriously on a slow stretch; a third point 256
        # deep (≈230 bits of decay at the measured rate) rules that out.
        nlo  = max(nmax0, nmax - Δ)
        nlo2 = max(nmax0, nmax - 2Δ)
        clo  = _monodromy_value(ctx, nlo)
        clo2 = _monodromy_value(ctx, nlo2)
        # Compare MIDPOINTS (`_strip_radius` is a no-op for BigFloat): for
        # R = Arb the closed-form value is a ball whose radius can exceed the
        # tolerance at large |ω|, making the rigorous `≤` unsatisfiable even
        # though the midpoint — the value every consumer actually uses (the
        # caller strips the radius) — has long converged.
        cm   = _strip_radius(c)
        cl1  = _strip_radius(clo)
        cl2  = _strip_radius(clo2)
        (abs(cm - cl1) ≤ tol * abs(cm) && abs(cm - cl2) ≤ tol * abs(cm)) && return c
        nmax ≥ 4000 && return c                  # safety cap: best effort
        nmax = min(nmax + 2Δ, 4000)
        _extend_monodromy_ctx!(ctx, nmax)
        c = _monodromy_value(ctx, nmax)
    end
    return c
end

"""
    nu_initial_guess(c2pn, l)

From cos(2πν), compute the initial guess for ν and the search type.
Uses full complex arccos to handle Im(c2pn) ≠ 0 (complex ω).
"""
function nu_initial_guess(c2pn, l)
    rc = real(c2pn)
    # Full complex arccos: cos(2πν₀) = c2pn exactly on the principal branch.
    ν0 = ComplexF64(l) - acos(complex(c2pn)) / (2π)
    if -1 ≤ rc ≤ 1
        return ν0, :real
    elseif rc < -1
        return ν0, :half
    else
        return ν0, :integer
    end
end

# ============================================================
#  Solve for ν, Eq. (136): β₀ + α₀R₁ + γ₀L₋₁ = 0
# ============================================================

"""
    compute_nu(s, l, m, a, ω; nmax_cf=150, tol=-1, maxiter=200, precision=64,
               ν_init=nothing, method="Monodromy")

Solve for ν (renormalized angular momentum).

# Methods

- `"Monodromy"` (default): Compute cos(2πν) from the monodromy of the confluent
  Heun equation, then extract ν directly via the branch formula. No Newton
  refinement. Matches Wolfram Teukolsky package default behavior.
  - Real branch   (|rc| ≤ 1):   ν = l − arccos(rc)/(2π)
  - Half-integer  (rc < −1):    ν = 1/2 + i·acosh(−rc)/(2π)
  - Integer       (rc > 1):     ν = −i·acosh(rc)/(2π)

- `"Newton"`: Monodromy initial guess followed by Newton refinement of the
  3-term continued-fraction equation g(ν) = 0. More accurate but slower.

# Other options

- `tol`: convergence tolerance (Newton only). Default auto-scales: ~100·eps(R).
- `precision`: bits of floating-point (64 = Float64, ≥128 = BigFloat).
- `ν_init`: optional initial guess (Newton only, useful for branch tracking).
"""
function compute_nu(s::Int, l::Int, m::Int, a, ω;
                    nmax_cf::Int=150, tol::Real=-1, maxiter::Int=200,
                    precision::Int=64, ν_init=nothing, method::String="Monodromy",
                    backend::Symbol=:bigfloat, l_max::Int=0)
    # ADDITIVE precision-type backends (opt-in): :float64 and :multifloat convert
    # the inputs to the chosen working float type and recurse through the generic
    # path (backend=:bigfloat + precision=64, so neither the :arb/:acb blocks nor
    # the BigFloat-hijack below re-fire).  The code is generic over
    # R<:AbstractFloat, so the MultiFloat type then flows everywhere, with the
    # multifloat_compat.jl shims supplying Γ and the 2-arg atan.  The default
    # :bigfloat path is byte-for-byte unchanged.
    if backend === :float64 || backend === :multifloat
        return _with_backend(backend, precision, a, ω) do a_w, ω_w
            compute_nu(s, l, m, a_w, ω_w;
                       nmax_cf=nmax_cf, tol=tol, maxiter=maxiter, precision=64,
                       ν_init = ν_init === nothing ? nothing : complex(ν_init),
                       method=method, backend=:bigfloat, l_max=l_max)
        end
    end
    # ADDITIVE Arb backend (M1): opt-in via backend=:arb.  Must precede the
    # precision>64 BigFloat-hijack block below.  Default :bigfloat never enters
    # here, so the Float64/BigFloat control flow is byte-for-byte unchanged.
    if backend === :arb
        method == "Monodromy" || error("backend=:arb supports only method=\"Monodromy\" (M1 scope).")
        return setprecision(Arb, precision) do
            ωc = complex(ω)
            _compute_nu_monodromy(s, l, m, Arb(a),
                Complex{Arb}(Arb(real(ωc)), Arb(imag(ωc))); l_max=l_max)
        end
    end
    # ADDITIVE native-Acb in-place backend (M2): opt-in via backend=:acb.  Same
    # plumbing as :arb (returns Complex{Arb} + MSTParams{Arb}) but the monodromy
    # recurrence/value runs through a preallocated Vector{Acb} + Arblib in-place
    # kernel (zero heap boxes/step).  Default :bigfloat never enters; the
    # Float64/BigFloat/:arb control flow is byte-for-byte unchanged.
    if backend === :acb
        method == "Monodromy" || error("backend=:acb supports only method=\"Monodromy\" (M2 scope).")
        return setprecision(Arb, precision) do
            ωc = complex(ω)
            _compute_nu_monodromy_acb(s, l, m, Arb(a),
                Complex{Arb}(Arb(real(ωc)), Arb(imag(ωc))); l_max=l_max)
        end
    end
    if precision > 64
        return setprecision(BigFloat, precision) do
            compute_nu(s, l, m, BigFloat(a), Complex{BigFloat}(complex(ω));
                       nmax_cf=nmax_cf, tol=tol, maxiter=maxiter, precision=64,
                       ν_init=ν_init === nothing ? nothing : Complex{BigFloat}(ν_init),
                       method=method, l_max=l_max)
        end
    end
    if method == "Monodromy"
        return _compute_nu_monodromy(s, l, m, a, ω; l_max=l_max)
    else
        return _compute_nu_impl(s, l, m, a, ω; nmax_cf=nmax_cf, tol=tol,
                                maxiter=maxiter, ν_init=ν_init, l_max=l_max)
    end
end

"""
    _compute_nu_monodromy(s, l, m, a, ω; nmax_mono=60)

Compute ν directly from the monodromy formula (no Newton refinement).
Matches Wolfram Teukolsky package "Monodromy" method.

`nmax_mono` controls the truncation of the monodromy series (default 60,
same as `monodromy_cos2pi_nu` default). This is independent of `nmax_cf`
used in the continued-fraction Newton solver.
"""
# Generic no-op: Float64/BigFloat carry no ball radius.  Complex{Arb}/Acb get the
# midpoint-extracting overrides in arb_compat.jl.
_strip_radius(z::Complex) = z

function _compute_nu_monodromy(s::Int, l::Int, m::Int, a, ω; nmax_mono::Int=60,
                               l_max::Int=0)
    p    = MSTParams(s, l, m, a, ω; l_max=l_max)
    R    = typeof(real(p.ϵ))
    # Point-estimate the converged cos(2πν): at large |ω| its rigorous ball is
    # blown by cancellation, but its midpoint matches BigFloat — see _strip_radius.
    c2pn = _strip_radius(_monodromy_adaptive(s, l, m, a, ω, p.λ; R=R, nmax0=nmax_mono))
    rc   = real(c2pn)
    twoπ = 2 * R(π)   # full-precision 2π (the Float64 literal caps ν at ~1e-16)

    # Branch selection based on rc = Re(cos(2πν)).
    #
    # For complex ω (Im(ω) ≠ 0): use "l − acos(c2pn)/(2π)" — the l-offset form.
    # This gives Im(ν) > 0 for integer/half-integer branches, which ensures the
    # fn 3-term recurrence converges properly.  Wolfram uses "acos(c2pn)/(2π)"
    # (no l-offset), which gives Im(ν) < 0 for integer branch and causes the
    # upward fn recurrence to diverge numerically.
    #
    # For real ω: use Wolfram's real-axis conventions:
    #   Real branch: ν = l − arccos(rc)/(2π)
    #   Half-integer (rc < −1): ν = 1/2 + i·acosh(−rc)/(2π)
    #   Integer (rc > 1):       ν = −i·acosh(rc)/(2π)
    ν = if imag(complex(ω)) != 0
        # Complex ω: l-offset form gives Im(ν) > 0 → stable fn recurrence
        R(l) - acos(complex(c2pn)) / twoπ
    elseif -1 ≤ rc ≤ 1
        # Real branch
        R(l) - acos(complex(rc)) / twoπ
    elseif rc < -1
        # Half-integer branch: Im(ν) > 0 (Wolfram real-ω convention)
        Complex(R(1) / 2, +acosh(-rc) / twoπ)
    else
        # Integer branch: Im(ν) < 0 (Wolfram real-ω convention:
        # ν = -I Im[ArcCos[rc]/(2π)], and ArcCos[rc]=i·acosh(rc) for rc>1)
        Complex(R(0), -acosh(rc) / twoπ)
    end

    return ν, p
end

function _compute_nu_impl(s::Int, l::Int, m::Int, a, ω;
                           nmax_cf::Int=150, tol::Real=-1, maxiter::Int=200,
                           ν_init=nothing, l_max::Int=0)
    p = MSTParams(s, l, m, a, ω; l_max=l_max)
    R = typeof(p.a)

    # Auto-scale tolerance and finite-difference step with precision
    tol_use  = tol < 0 ? R(100) * eps(R) : R(tol)
    δ        = cbrt(eps(R))   # finite-difference step for Newton

    function g0(ν)
        R1  = Rn_cf(p, ν, 1;  nmax=nmax_cf)
        Lm1 = Ln_cf(p, ν, -1; nmax=nmax_cf)
        βn(p, ν, 0) + αn(p, ν, 0) * R1 + γn(p, ν, 0) * Lm1
    end

    # Unconstrained 2D complex Newton
    function newton_from(ν0; max_step=R(2))
        ν = Complex{R}(ν0)
        for _ in 1:maxiter
            g = g0(ν)
            !isfinite(g) && return ν, false
            abs(g) < tol_use && return ν, true
            gp = (g0(ν + δ) - g0(ν - δ)) / (2δ)
            abs(gp) < R(1e-30) && return ν, false
            Δν = -g / gp
            abs(Δν) > max_step && (Δν *= max_step / abs(Δν))
            ν += Δν
            abs(Δν) < tol_use && return ν, true
        end
        g_final = g0(ν)
        return ν, isfinite(g_final) && abs(g_final) < sqrt(tol_use)
    end

    # Constrained 1D real Newton: Re(ν) fixed, search Im(ν) only.
    # Solves Re(g0(ν_re + i·η)) = 0 for real η.
    # Matches Mathematica's FindRoot[Re[g[1/2 + I νi]] == 0, {νi, ...}].
    function newton_1d(ν_re, η0; max_step=1.0)
        η = R(η0)
        ν_re_R = R(ν_re)
        for _ in 1:maxiter
            ν  = Complex{R}(ν_re_R, η)
            g  = real(g0(ν))
            abs(g) < tol_use && return Complex{R}(ν_re_R, η), true
            gp = real(g0(Complex{R}(ν_re_R, η + δ)) - g0(Complex{R}(ν_re_R, η - δ))) / (2δ)
            abs(gp) < 1e-30 && break
            Δη = -g / gp
            abs(Δη) > max_step && (Δη *= max_step / abs(Δη))
            η += Δη
            abs(Δη) < tol_use && return Complex{R}(ν_re_R, η), true
        end
        ν_fin = Complex{R}(ν_re_R, η)
        return ν_fin, abs(real(g0(ν_fin))) < sqrt(tol_use)
    end

    # Compute monodromy first to classify the branch
    c2pn = monodromy_cos2pi_nu(s, l, m, a, ω, p.λ)
    rc   = real(c2pn)

    if imag(complex(ω)) != 0
        # Complex ω: ν_init tracking is safe, use unconstrained 2D Newton
        if ν_init !== nothing
            ν2, c2 = newton_from(ν_init)
            c2 && return ν2, p
        end
        # Use l-offset form (same as real branch) so that ν₀ ≈ l for small Im(ω).
        # Without the offset, acos(c2pn)/(2π) ≈ 0 and Newton may converge to the
        # spurious root ν=0 instead of the physical ν≈l root.
        ν0 = ComplexF64(l) - acos(complex(c2pn)) / (2π)
        ν, converged = newton_from(ν0)
        converged && return ν, p
        ν2, c2 = newton_from(conj(ν0));  c2 && return ν2, p
        ν2, c2 = newton_from(ComplexF64(l) - ν0);  c2 && return ν2, p

    elseif -1 ≤ rc ≤ 1
        # Real ν branch.
        # ν_init tracking is only safe when ν_init itself was on the real branch
        # (Im(ν_init) ≈ 0). If ν_init came from a half/integer branch (Im large),
        # Newton can stray to degenerate integer roots (ν=1,2,…) where the fn
        # recurrence is ill-conditioned and Btrans → 0 spuriously.
        # Use monodromy formula first; fall back to ν_init as secondary seed.
        ν0 = ComplexF64(l) - acos(complex(rc)) / (2π)
        ν, converged = newton_from(ν0)
        converged && return ν, p
        if ν_init !== nothing
            ν2, c2 = newton_from(ν_init)
            c2 && return ν2, p
        end

    elseif rc < -1
        # Half-integer case: cos(2πν) < -1  →  Re(ν) = n + 1/2
        # Mathematica always uses Re(ν) = 1/2 (not l-1/2).
        # Convention: ν = 1/2 - Im(arccos(rc)/(2π))*i
        # arccos(rc) for rc < -1 (principal branch) = π - i*acosh(-rc),
        # so Im(arccos(rc)/(2π)) = -acosh(-rc)/(2π) < 0  →  η0 > 0.
        η0 = R(acosh(max(-rc, one(R))) / (2π))   # > 0
        # Try Re(ν) = 1/2 first (Mathematica convention), then l±1/2 as fallbacks
        for ν_re_try in R[R(0.5), l - R(0.5), l + R(0.5), l - R(1.5), R(-0.5)]
            ν2, c2 = newton_1d(ν_re_try,  η0)
            c2 && return ν2, p
            ν2, c2 = newton_1d(ν_re_try, -η0)
            c2 && return ν2, p
        end
        # Fallback: unconstrained 2D Newton from monodromy guess
        ν0 = ComplexF64(l) - acos(complex(c2pn)) / (2π)
        ν, converged = newton_from(ν0);  converged && return ν, p
        ν2, c2 = newton_from(conj(ν0));  c2 && return ν2, p

    else  # rc > 1
        # Integer (pure-imaginary) case: cos(2πν) > 1  →  Re(ν) = 0
        # Mathematica uses Re(ν) = 0.
        # arccos(rc) for rc > 1 (principal branch) = -i*acosh(rc),
        # so Im(arccos(rc)/(2π)) = -acosh(rc)/(2π) < 0  →  η0 > 0.
        η0 = R(acosh(max(rc, one(R))) / (2π))   # > 0
        for ν_re_try in R[R(0), R(1), R(-1), l, l - R(1)]
            ν2, c2 = newton_1d(ν_re_try,  η0)
            c2 && return ν2, p
            ν2, c2 = newton_1d(ν_re_try, -η0)
            c2 && return ν2, p
        end
        ν0 = ComplexF64(l) - acos(complex(c2pn)) / (2π)
        ν, converged = newton_from(ν0);  converged && return ν, p
    end

    # Final fallback: try a grid of seeds
    ν0_fb = ComplexF64(l) - acos(complex(c2pn)) / (2π)
    im_ω  = imag(complex(ω))
    for ν_try in [ν0_fb, conj(ν0_fb), real(ν0_fb) + 0im,
                   Complex{R}(l, im_ω), Complex{R}(l, -im_ω),
                   Complex{R}(l - R(0.5), 0), Complex{R}(l + R(0.5), 0)]
        ν2, c2 = newton_from(ν_try)
        c2 && return ν2, p
    end

    @warn "compute_nu: Newton did not converge, |g| = $(abs(g0(ν0_fb)))"
    ν_best, _ = newton_from(ν0_fb)
    return ν_best, p
end

# ============================================================
#  Native-Acb in-place monodromy kernel (M2)
#
#  A bit-faithful clone of the BigFloat monodromy path
#  (_MonodromyCtx / _extend_monodromy_ctx! / _monodromy_value /
#  _monodromy_adaptive), with the per-op-allocating Complex{Arb}/BigFloat
#  arithmetic replaced by preallocated Vector{Acb} + a fixed register set +
#  Arblib in-place `!` ops (every op carries prec=P).  Hot loop = 0 heap
#  boxes/step.  Reached ONLY via compute_nu(...; backend=:acb); the
#  Float64/BigFloat/:arb paths are untouched.
#
#  SCOPE: monodromy recurrence + value ONLY.  λ comes from the existing
#  compute_lambda (via MSTParams) and is converted to Acb once at the boundary;
#  params.jl is NOT modified and no warm-start is added.
# ============================================================

"""
    _MonodromyCtxAcb

In-place native-Acb monodromy context (M2 analogue of `_MonodromyCtx`).  All Acb
fields are MUTATED in place (`Arblib.<op>!(field, …)`), never reassigned, so the
underlying FLINT object identity stays stable across extend/value calls.
"""
mutable struct _MonodromyCtxAcb
    a1::Vector{Acb}; a2::Vector{Acb}
    Poch_p::Vector{Acb}; Poch_m::Vector{Acb}
    # recurrence-invariant Acb constants (built once at construction)
    ACαε::Acb; ACεCH::Acb
    AC_aδ::Acb; AC_aγδ::Acb; AC_aγ::Acb
    ACK1::Acb; ACL1::Acb; ACKd::Acb; ACLd::Acb
    ACcMp::Acb; ACcMm::Acb
    # value-invariant Acb (built once)
    gMp::Acb; gMm::Acb; πAcb::Acb; twoπ2::Acb
    # per-step scratch (reused across every n and every extend call)
    t1::Acb; t2::Acb; acc::Acb
    c1r::Acb; c2r::Acb; d1r::Acb; d2r::Acb
    nfilled::Int
    prec::Int
end

"""
    _build_monodromy_ctx_acb(s, l, m, a, ω, λ, nmax; prec=precision(Arb))

Build the native-Acb monodromy context up to truncation `nmax`.  The O(1) scalar
setup is done in `Complex{BigFloat}` (the de-risked, validated arithmetic — zero
perf benefit from going native off the hot loop), then converted to Acb once at
the boundary.  `a`, `ω`, `λ` may be Arb/Complex{Arb} (production) or
BigFloat/Complex{BigFloat}; only their numeric value is used.
"""
function _build_monodromy_ctx_acb(s::Int, _l::Int, m::Int, a, ω, λ, nmax::Int;
                                  prec::Int=precision(Arb))
    # --- O(1) scalar setup in Complex{BigFloat} (exact mirror of
    #     _build_monodromy_ctx), regrouped so the n-dependence is integer ops. ---
    ACαε = ACεCH = AC_aδ = AC_aγδ = AC_aγ = Acb(0)
    ACK1 = ACL1 = ACKd = ACLd = ACcMp = ACcMm = Acb(0)
    gMp = gMm = πAcb = twoπ2 = Acb(0)
    setprecision(BigFloat, prec) do
        C   = Complex{BigFloat}
        ωc  = complex(ω)
        q   = BigFloat(real(a))
        ε   = 2 * C(BigFloat(real(ωc)), BigFloat(imag(ωc)))
        κ   = sqrt(C(1 - q^2))
        τ   = (ε - m*q) / κ
        γCH = 1 - s - im*ε - im*τ
        δCH = 1 + s + im*ε - im*τ
        εCH = 2im*ε*κ
        αε  = C(1 - s) + im*(ε - τ)
        λbf = C(BigFloat(real(λ)), BigFloat(imag(λ)))
        qCH = s*(s+1) - ε^2 + im*(1-2s)*ε*κ + λbf + im*τ + τ^2
        μ1C = αε - (γCH + δCH)
        μ2C = -αε

        αε2   = αε^2
        gd    = γCH + δCH
        c_aδ  = αε - δCH                  # c2 = (c_aδ-(n-1))*(c_aγδ-(n-2))*εCH/n
        c_aγδ = αε - γCH - δCH
        cA    = 1 - γCH - δCH + εCH       # (1-2n-γCH-δCH+εCH) = cA - 2n
        cC    = εCH - δCH*εCH
        K1    = αε2 + αε*cA - qCH + cC    # c1 = (K1 + n*L1 + n^2)/n
        L1    = -cA - 2*αε
        c_aγ  = αε - γCH                  # d2 = (αε+(n-2))*(c_aγ+(n-1))*εCH/n
        cS    = 1 + γCH + δCH - εCH
        Kd    = αε2 - qCH + gd - εCH - αε*cS   # d1 = (Kd + n*Ld + n^2)/n
        Ld    = -cS + 2*αε
        cMp   = μ1C - μ2C                 # = -μ2C + μ1C
        cMm   = μ2C - μ1C

        toacb(z) = Acb(z; prec=prec)
        ACαε   = toacb(αε);   ACεCH  = toacb(εCH)
        AC_aδ  = toacb(c_aδ); AC_aγδ = toacb(c_aγδ); AC_aγ = toacb(c_aγ)
        ACK1   = toacb(K1);   ACL1   = toacb(L1)
        ACKd   = toacb(Kd);   ACLd   = toacb(Ld)
        ACcMp  = toacb(cMp);  ACcMm  = toacb(cMm)
        # value-invariant constants
        gMp = Acb(0); gMm = Acb(0)
        Arblib.gamma!(gMp, ACcMp; prec=prec)
        Arblib.gamma!(gMm, ACcMm; prec=prec)
        πAcb  = Acb(Arb(π); prec=prec)
        twoπ2 = Acb(2 * Arb(π)^2; prec=prec)
        return nothing
    end

    cap = max(nmax, 1) + 2
    a1 = [Acb(0) for _ in 1:cap]
    a2 = [Acb(0) for _ in 1:cap]
    Poch_p = [Acb(1) for _ in 1:cap]
    Poch_m = [Acb(1) for _ in 1:cap]
    Arblib.set!(a1[1], 1); Arblib.set!(a2[1], 1)
    # Poch_p[1] = Poch_m[1] = 1 already (Acb(1)).

    ctx = _MonodromyCtxAcb(a1, a2, Poch_p, Poch_m,
                           ACαε, ACεCH, AC_aδ, AC_aγδ, AC_aγ,
                           ACK1, ACL1, ACKd, ACLd, ACcMp, ACcMm,
                           gMp, gMm, πAcb, twoπ2,
                           Acb(0), Acb(0), Acb(0),
                           Acb(0), Acb(0), Acb(0), Acb(0),
                           0, prec)
    return _extend_monodromy_ctx_acb!(ctx, nmax; prec=prec)
end

"""
    _extend_monodromy_ctx_acb!(ctx, target; prec=ctx.prec)

March the native-Acb recurrence in-place from `ctx.nfilled` up to truncation
`target` (no-op if already deep enough).  Appended Vector{Acb} slots are FILLED
with fresh Acb objects (never left undef).  0 heap boxes per step (all `!` ops
write into preallocated registers / array slots; output aliasing is FLINT-safe).
"""
function _extend_monodromy_ctx_acb!(ctx::_MonodromyCtxAcb, target::Int;
                                    prec::Int=ctx.prec)
    target ≤ ctx.nfilled && return ctx

    a1, a2 = ctx.a1, ctx.a2
    Poch_p, Poch_m = ctx.Poch_p, ctx.Poch_m
    need = target + 2
    if length(a1) < need
        for v in (a1, a2)
            old = length(v); resize!(v, need)
            for k in (old+1):need; v[k] = Acb(0); end
        end
        for v in (Poch_p, Poch_m)
            old = length(v); resize!(v, need)
            for k in (old+1):need; v[k] = Acb(1); end
        end
    end

    ACαε, ACεCH = ctx.ACαε, ctx.ACεCH
    AC_aδ, AC_aγδ, AC_aγ = ctx.AC_aδ, ctx.AC_aγδ, ctx.AC_aγ
    ACK1, ACL1, ACKd, ACLd = ctx.ACK1, ctx.ACL1, ctx.ACKd, ctx.ACLd
    ACcMp, ACcMm = ctx.ACcMp, ctx.ACcMm
    t1, t2, acc = ctx.t1, ctx.t2, ctx.acc
    c1r, c2r, d1r, d2r = ctx.c1r, ctx.c2r, ctx.d1r, ctx.d2r

    for n in (ctx.nfilled + 1):target
        # c2 = (c_aδ-(n-1))*(c_aγδ-(n-2))*εCH/n
        Arblib.sub!(t1, AC_aδ,  n-1; prec=prec)
        Arblib.sub!(t2, AC_aγδ, n-2; prec=prec)
        Arblib.mul!(t1, t1, t2;     prec=prec)
        Arblib.mul!(t1, t1, ACεCH;  prec=prec)
        Arblib.div!(c2r, t1, n;     prec=prec)
        # c1 = (K1 + n*L1 + n^2)/n
        Arblib.mul!(acc, ACL1, n;   prec=prec)
        Arblib.add!(acc, acc, ACK1; prec=prec)
        Arblib.add!(acc, acc, n*n;  prec=prec)
        Arblib.div!(c1r, acc, n;    prec=prec)
        # a1[n+1] = c2*a1[n-1] - c1*a1[n]
        if n >= 2
            Arblib.mul!(a1[n+1], c2r, a1[n-1]; prec=prec)
        else
            Arblib.zero!(a1[n+1])
        end
        Arblib.submul!(a1[n+1], c1r, a1[n]; prec=prec)

        # d2 = (αε+(n-2))*(c_aγ+(n-1))*εCH/n
        Arblib.add!(t1, ACαε,  n-2; prec=prec)
        Arblib.add!(t2, AC_aγ, n-1; prec=prec)
        Arblib.mul!(t1, t1, t2;     prec=prec)
        Arblib.mul!(t1, t1, ACεCH;  prec=prec)
        Arblib.div!(d2r, t1, n;     prec=prec)
        # d1 = (Kd + n*Ld + n^2)/n
        Arblib.mul!(acc, ACLd, n;   prec=prec)
        Arblib.add!(acc, acc, ACKd; prec=prec)
        Arblib.add!(acc, acc, n*n;  prec=prec)
        Arblib.div!(d1r, acc, n;    prec=prec)
        # a2[n+1] = d1*a2[n] - d2*a2[n-1]
        Arblib.mul!(a2[n+1], d1r, a2[n]; prec=prec)
        if n >= 2
            Arblib.submul!(a2[n+1], d2r, a2[n-1]; prec=prec)
        end

        # Poch_p[n+1] = (cMp + n-1)*Poch_p[n]
        Arblib.add!(t1, ACcMp, n-1; prec=prec)
        Arblib.mul!(Poch_p[n+1], t1, Poch_p[n]; prec=prec)
        # Poch_m[n+1] = (cMm + n-1)*Poch_m[n]
        Arblib.add!(t2, ACcMm, n-1; prec=prec)
        Arblib.mul!(Poch_m[n+1], t2, Poch_m[n]; prec=prec)
    end
    ctx.nfilled = target
    return ctx
end

"""
    _monodromy_value_acb(ctx, n; prec=ctx.prec) -> Complex{Arb}

Evaluate cos(2πν) from a prebuilt native-Acb `ctx` at truncation `n` (requires
`n ≤ ctx.nfilled`).  Same closed form as `_monodromy_value`; returns
`Complex{Arb}` so the adaptive driver and branch extraction match the M1 path.
"""
function _monodromy_value_acb(ctx::_MonodromyCtxAcb, n::Int; prec::Int=ctx.prec)
    # PIA-resonance gate (mirrors _monodromy_value): μd = cMp on an integer ⇒
    # the factored gMp/gMm·Poch form is 0·∞ (NaN or finite garbage) — use the
    # pole-free marching-Γ form instead.
    if _monodromy_resonant(Complex{Arb}(ctx.ACcMp))
        return _monodromy_value_acb_safe(ctx, n; prec=prec)
    end
    a1, a2 = ctx.a1, ctx.a2
    Poch_p, Poch_m = ctx.Poch_p, ctx.Poch_m

    jmax = cld(n, 2)
    s1 = Acb(0); s2 = Acb(0)
    # a1sum = Σ_{j=0..jmax} a1[j+1]*Poch_p[n-j+1]
    for j in 0:jmax
        Arblib.addmul!(s1, a1[j+1], Poch_p[n-j+1]; prec=prec)
    end
    # a2sum = Σ_{j=0..jmax} (-1)^j a2[j+1]*Poch_m[n-j+1]
    for j in 0:jmax
        if iseven(j)
            Arblib.addmul!(s2, a2[j+1], Poch_m[n-j+1]; prec=prec)
        else
            Arblib.submul!(s2, a2[j+1], Poch_m[n-j+1]; prec=prec)
        end
    end
    Arblib.mul!(s1, ctx.gMp, s1; prec=prec)
    Arblib.mul!(s2, ctx.gMm, s2; prec=prec)

    # cos(π*(μ1C-μ2C)) = cos(π*cMp)
    argreg = Acb(0); cosreg = Acb(0)
    Arblib.mul!(argreg, ctx.ACcMp, ctx.πAcb; prec=prec)
    Arblib.cos!(cosreg, argreg; prec=prec)

    # term = (2π²/(s1*s2))*(-1)^(n-1)*a1[n+1]*a2[n+1]
    term = Acb(0); den = Acb(0)
    Arblib.mul!(den, s1, s2; prec=prec)
    Arblib.div!(term, ctx.twoπ2, den; prec=prec)
    Arblib.mul!(term, term, a1[n+1]; prec=prec)
    Arblib.mul!(term, term, a2[n+1]; prec=prec)
    iseven(n) && Arblib.neg!(term, term)   # (-1)^(n-1): n even → -1

    res = Acb(0)
    Arblib.add!(res, cosreg, term; prec=prec)
    out = Complex{Arb}(res)
    # PIA-resonance gate (mirrors the generic _monodromy_value): the factored
    # Γ(±cMp)·Poch form is an exact 0·∞ when μd = μ1C−μ2C = −2s−4σ ∈ ℤ (ω = iσ,
    # 4σ ∈ ℤ) — fall back to the pole-free marching-Γ evaluation.
    (isfinite(real(out)) && isfinite(imag(out))) && return out
    return _monodromy_value_acb_safe(ctx, n; prec=prec)
end

"""
    _monodromy_value_acb_safe(ctx, n; prec=ctx.prec) -> Complex{Arb}

Rare-path (PIA resonance) analogue of [`_monodromy_value_safe`](@ref) for the
native-Acb context: the same pole-free marching-Γ form, run in Complex{Arb}
arithmetic on the Acb context arrays.  Only reached when the in-place factored
form returns a non-finite value, so clarity is preferred over in-place speed.
"""
function _monodromy_value_acb_safe(ctx::_MonodromyCtxAcb, n::Int; prec::Int=ctx.prec)
    C  = Complex{Arb}
    μd = C(ctx.ACcMp)                 # ACcMp = cMp = μ1C − μ2C
    jmax = cld(n, 2)
    g1 = _cgamma(μd + n)              # Γ(μd)·(μd)_{n−j} ≡ Γ(μd + n − j)
    g2 = _cgamma(-μd + n)
    t1 = zero(C); t2 = zero(C)
    for j in 0:jmax
        a1j = C(ctx.a1[j+1]); a2j = C(ctx.a2[j+1])
        t1 += a1j * g1
        t2 += (iseven(j) ? a2j : -a2j) * g2
        g1 /= (μd + n - j - 1)        # Γ(z−1) = Γ(z)/(z−1)
        g2 /= (-μd + n - j - 1)
    end
    return cos(π*μd) +
           (complex(2*Arb(π)^2) / (t1 * t2)) * (-1)^(n-1) *
           C(ctx.a1[n+1]) * C(ctx.a2[n+1])
end

"""
    _monodromy_adaptive_acb(s, l, m, a, ω, λ; prec=precision(Arb), nmax0=60)

Native-Acb analogue of `_monodromy_adaptive`, with one Acb-specific fix that
avoids a ~7× wasted-work trap:

  MIDPOINT convergence.  The BigFloat driver stops when `|c−clo| ≤ 16·eps·|c|`.
  On Arb that LHS is a BALL whose radius (~2^(−0.28·prec), from the accumulating
  products) is orders of magnitude larger than 16·eps·|c|, so the rigorous `≤`
  is NEVER satisfiable and the driver would extend to the 4000 safety cap on
  EVERY call.  The series MIDPOINT — the value every downstream consumer and the
  BigFloat cross-check actually use — converges long before the radius matters,
  so we compare midpoints (in BigFloat at the working precision) to
  ~2^(−0.85·prec).  This makes the driver terminate at the true convergence
  point (≈ the 4.71·prec start for normal modes) instead of churning to 4000.

The starting `nmax` is the MEASURED mode-dependent envelope (see the twin
comment in `_monodromy_adaptive` and test/test_monodromy_depth.jl):
n*(prec, |ε|) ≤ 1.04·prec + 13·|ε| + 40 across the mapped grid
(σ ∈ {0.25…20} PIA + real/complex ω, prec ∈ {256…1536}, s ∈ {−2,2},
l ∈ {2,4,10,20}, a ∈ {0.7,0.9}), started 2Δ deeper so the three-point
acceptance passes first try.  The old 4.71·prec start was a ~4× overshoot.
The verify-and-extend loop is the backstop for any mode outside the envelope,
with a THREE-depth acceptance (n, n−Δ, n−2Δ) so two evaluations cannot agree
spuriously on a slow stretch.

Returns `Complex{Arb}`.
"""
function _monodromy_adaptive_acb(s::Int, l::Int, m::Int, a, ω, λ;
                                 prec::Int=precision(Arb), nmax0::Int=60)
    Δ    = 128
    ωc   = complex(ω)
    εf   = 2 * hypot(Float64(real(ωc)), Float64(imag(ωc)))   # |ε| = 2|ω|
    nmax = clamp(ceil(Int, 1.04 * prec + 13 * εf + 40) + 2Δ,
                 max(nmax0, 120) + 2Δ, 4000)

    # Midpoint-relative convergence: |mid(c)−mid(clo)| ≤ 2^(−0.85·prec)·|mid(c)|.
    # 0.85·prec leaves margin above the achievable midpoint-agreement floor (so
    # the test triggers) and far below the ν gate (so accuracy stays ≳0.8·prec
    # digits — validated against the BigFloat path across the mode grid).
    converged(c1, c2) = setprecision(BigFloat, prec) do
        m1 = Complex{BigFloat}(c1); m2 = Complex{BigFloat}(c2)
        abs(m1 - m2) ≤ ldexp(one(BigFloat), -fld(17 * prec, 20)) * abs(m1)
    end

    ctx = _build_monodromy_ctx_acb(s, l, m, a, ω, λ, nmax; prec=prec)
    c   = _monodromy_value_acb(ctx, nmax; prec=prec)
    for _ in 1:24
        # Three-depth acceptance (n, n−Δ, n−2Δ); see _monodromy_adaptive.
        nlo  = max(nmax0, nmax - Δ)
        nlo2 = max(nmax0, nmax - 2Δ)
        clo  = _monodromy_value_acb(ctx, nlo;  prec=prec)
        clo2 = _monodromy_value_acb(ctx, nlo2; prec=prec)
        (converged(c, clo) && converged(c, clo2)) && return c
        nmax ≥ 4000 && return c                  # safety cap: best effort
        nmax = min(nmax + 2Δ, 4000)
        _extend_monodromy_ctx_acb!(ctx, nmax; prec=prec)
        c = _monodromy_value_acb(ctx, nmax; prec=prec)
    end
    return c
end

"""
    _compute_nu_monodromy_acb(s, l, m, a::Arb, ω::Complex{Arb}; nmax_mono=60)

ν via the native-Acb monodromy kernel.  Structurally identical to
`_compute_nu_monodromy` (R = Arb): λ comes from the existing `compute_lambda`
(via `MSTParams`; params.jl untouched), cos(2πν) from `_monodromy_adaptive_acb`,
then the SAME branch-selection block (real / half-integer / integer / complex-ω).
Returns `(ν::Complex{Arb}, p::MSTParams{Arb})`, consistent with the M1 backend.
"""
function _compute_nu_monodromy_acb(s::Int, l::Int, m::Int, a::Arb, ω::Complex{Arb};
                                   nmax_mono::Int=60, l_max::Int=0)
    p    = MSTParams(s, l, m, a, ω; l_max=l_max)  # R = Arb; p.λ::Complex{Arb}
    R    = typeof(real(p.ϵ))                # === Arb
    c2pn = _strip_radius(_monodromy_adaptive_acb(s, l, m, a, ω, p.λ;
                                   prec=precision(Arb), nmax0=nmax_mono))
    rc   = real(c2pn)
    twoπ = 2 * R(π)

    # Branch selection — verbatim from _compute_nu_monodromy (R = Arb).
    ν = if imag(complex(ω)) != 0
        R(l) - acos(complex(c2pn)) / twoπ
    elseif -1 ≤ rc ≤ 1
        R(l) - acos(complex(rc)) / twoπ
    elseif rc < -1
        Complex(R(1) / 2, +acosh(-rc) / twoπ)
    else
        Complex(R(0), -acosh(rc) / twoπ)
    end

    return ν, p
end
