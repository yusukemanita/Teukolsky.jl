# ============================================================
#  Asymptotic amplitudes A^ν_±, Eqs. (157)-(158)
# ============================================================

# A^ν_+ prefactor, shared by the exact-window and adaptive paths.
function _Aplus_prefactor(p::MSTParams, ν)
    s, ϵ = p.s, p.ϵ
    T = typeof(ϵ)
    # π at FULL working precision.  The old `exp(π*im*(ν+1-s)/2)` evaluated
    # `π*im` FIRST, promoting π through Complex{Bool} to Float64 — a silent
    # 1.2e-16 relative error in the prefactor phase at EVERY precision
    # (BigFloat/Arb included).  Found by the native-Acb prefactor arbiter.
    πT = real(T)(π)
    return exp(-πT*ϵ/2) * exp(im*πT*(ν+1-s)/2) *
           T(2)^(-1+s-im*ϵ) *
           _cgamma(ν + 1 - s + im*ϵ) / _cgamma(ν + 1 + s - im*ϵ)
end

# Exact-window A^ν_+ over n = nmin:nmax — the arbiter primitive (generic vs
# native kernels must agree on IDENTICAL windows; see test_native_acb.jl).
# Production callers use the adaptive converge-or-error `compute_Aplus`.
function _Aplus_window(p::MSTParams, ν, fn, nmin::Int, nmax::Int)
    Σ = sum(fn[n] for n in nmin:nmax)
    return _Aplus_prefactor(p, ν) * Σ
end

"""
    compute_Aplus(p, ν, fn; nmax=80, tol=100·eps, nmax_hard=50_000,
                  floor_tol=√eps, nmin=nothing)

Asymptotic amplitude A^ν_+ = prefactor · Σ_n f_n (Sasaki-Tagoshi Eq. 157).

Converge-or-error (same contract as [`Rin`](@ref)/[`Rup`](@ref)): `nmax` is
only the initial window hint — the sum runs until two consecutive terms pass
`tol`, adaptively extending `fn` (the dict is mutated) up to `nmax_hard`, and
is certified against the cancellation floor.  The old fixed ±nmax window
silently truncated on the positive imaginary axis (measured at s=-2 l=m=2
a=0.7 with the `suggest_mst_precision` settings: 1.3e-38 at ω=4.3i up to
5.0e-07 at ω=16i — far above the working-precision floor).

Passing `nmin` explicitly selects the legacy EXACT window `nmin:nmax`
(no extension, no certification) — the generic-vs-native arbiter path.
"""
function compute_Aplus(p::MSTParams, ν, fn; nmax::Int=80,
                       tol::Real=100*eps(real(typeof(p.ϵ))),
                       nmax_hard::Int=50_000,
                       floor_tol::Real=_default_floor_tol(real(typeof(p.ϵ))),
                       nmin::Union{Nothing,Int}=nothing)
    nmin === nothing || return _Aplus_window(p, ν, fn, nmin, nmax)
    T = typeof(p.ϵ)
    one_T = one(T)
    n_ext = max(2 * nmax, 64)
    res_up, smax_up = _sum_mst_series!(n -> one_T, fn, p, ν, +1,
                                       tol, zero(tol), n_ext, nmax_hard, "A+")
    res_dn, smax_dn = _sum_mst_series!(n -> one_T, fn, p, ν, -1,
                                       tol, zero(tol), n_ext, nmax_hard, "A+")
    ctol = max(tol, floor_tol)
    Σ = _certify_mst_sum(res_up + res_dn, max(smax_up, smax_dn),
                         ctol, zero(ctol), "A+")
    return _Aplus_prefactor(p, ν) * Σ
end

# A^ν_- prefactor, shared by the exact-window and adaptive paths.
function _Aminus_prefactor(p::MSTParams, ν)
    s, ϵ = p.s, p.ϵ
    T = typeof(ϵ)
    # Full-precision π — see _Aplus_prefactor.
    πT = real(T)(π)
    return T(2)^(-1-s+im*ϵ) *
           exp(-im*πT*(ν+1+s)/2) *
           exp(-πT*ϵ/2)
end

# Exact-window A^ν_- over n = nmin:nmax — arbiter primitive (see _Aplus_window).
function _Aminus_window(p::MSTParams, ν, fn, nmin::Int, nmax::Int)
    s, ϵ = p.s, p.ϵ
    T = typeof(ϵ)
    prefactor = _Aminus_prefactor(p, ν)

    # Weights w(n) = (-1)^n (aw)_n/(bw)_n built by the incremental ratios
    #   w(n) = -w(n-1)·(aw+n-1)/(bw+n-1)   (ascending),
    #   w(n) = -w(n+1)·(bw+n)/(aw+n)       (descending)
    # — O(nmax) multiplies total instead of the O(nmax²) of fresh pochhammer(·,n)
    # calls, with the same near-pole conditioning (the descending ratio divides by
    # the same (aw+n) factors pochhammer would).  Summation order (nmin → nmax,
    # small far-tail terms first) is preserved via the weight vector.
    # `_strip_radius` keeps the chained Complex{Arb} products in point arithmetic
    # (the nu_solver marching-Pochhammer convention): at large |ω| the ν ball has
    # only ~56 accurate bits and compounding weight radii through the chain turns
    # the cancellation-heavy sums into zero-containing balls (NaN on division)
    # where the fresh-pochhammer form scraped by.  No-op for Float64/BigFloat.
    aw = ν + 1 + s - im*ϵ
    bw = ν + 1 - s + im*ϵ
    off = 1 - nmin
    w = Vector{T}(undef, nmax - nmin + 1)
    n0 = clamp(0, nmin, nmax)   # anchor (n0 = 0 for every in-tree call site)
    w[n0 + off] = T((-1)^n0 * pochhammer(aw, n0) / pochhammer(bw, n0))
    for n in n0+1:nmax
        w[n + off] = _strip_radius(-w[n - 1 + off] * (aw + (n - 1)) / (bw + (n - 1)))
    end
    for n in n0-1:-1:nmin
        w[n + off] = _strip_radius(-w[n + 1 + off] * (bw + n) / (aw + n))
    end

    Σ = sum(
        begin
            fn_n = fn[n]
            iszero(fn_n) ? zero(T) : w[n + off] * fn_n
        end
        for n in nmin:nmax
    )

    return prefactor * Σ
end

"""
    compute_Aminus(p, ν, fn; nmax=80, tol=100·eps, nmax_hard=50_000,
                   floor_tol=√eps, nmin=nothing)

Asymptotic amplitude A^ν_- = prefactor · Σ_n (-1)^n (aw)_n/(bw)_n · f_n
(Sasaki-Tagoshi Eq. 158), with the Pochhammer-ratio weight marched
incrementally (see `_Aminus_window` for the ratio derivation).

Converge-or-error, same contract as [`compute_Aplus`](@ref).  The fixed
±nmax window was the WORST silent truncation on the positive imaginary
axis: the weight grows polynomially like n^(2|ϵ|+2s) there (−2iϵ is real
positive), pushing the weighted peak far beyond fn's own decay — measured
1.1e-33 (ω=4.3i) up to 2.6e-20 (ω=16i) at the `suggest_mst_precision`
settings, and this error enters EVERY `Rup`/`q̃` value through Ctrans.

Passing `nmin` explicitly selects the legacy EXACT window `nmin:nmax`
(no extension, no certification) — the generic-vs-native arbiter path.
"""
function compute_Aminus(p::MSTParams, ν, fn; nmax::Int=80,
                        tol::Real=100*eps(real(typeof(p.ϵ))),
                        nmax_hard::Int=50_000,
                        floor_tol::Real=_default_floor_tol(real(typeof(p.ϵ))),
                        nmin::Union{Nothing,Int}=nothing)
    nmin === nothing || return _Aminus_window(p, ν, fn, nmin, nmax)
    s, ϵ = p.s, p.ϵ
    aw = ν + 1 + s - im*ϵ
    bw = ν + 1 - s + im*ϵ
    n_ext = max(2 * nmax, 64)
    w_up = _fup_weight_stepper(aw, bw, +1)
    res_up, smax_up = _sum_mst_series!(w_up, fn, p, ν, +1,
                                       tol, zero(tol), n_ext, nmax_hard, "A-")
    w_dn = _fup_weight_stepper(aw, bw, -1)
    res_dn, smax_dn = _sum_mst_series!(w_dn, fn, p, ν, -1,
                                       tol, zero(tol), n_ext, nmax_hard, "A-")
    ctol = max(tol, floor_tol)
    Σ = _certify_mst_sum(res_up + res_dn, max(smax_up, smax_dn),
                         ctol, zero(ctol), "A-")
    return _Aminus_prefactor(p, ν) * Σ
end

# Rup / amplitude normalization constant (Sasaki-Tagoshi; MST.m UpTrans)
#   Ctrans = ω^{-1-2s} A^ν_- exp(i(ε logε − (1−κ)/2 ε))
# Single source of truth: Rup/dRup fallback, compute_amplitudes(_nufixed), and
# mst_ctrans all call this so the convention can never drift between call sites.
_ctrans(p::MSTParams, Am) =
    p.ω^(-1 - 2*p.s) * Am * exp(im * (p.ϵ * log(p.ϵ) - (1 - p.κ) / 2 * p.ϵ))

# ============================================================
#  Matching coefficient K_ν, Eq. (165)
# ============================================================

function compute_Knu(p::MSTParams, ν, fn; nmax::Int=80, r::Int=0)
    s, ϵ, κ, τ = p.s, p.ϵ, p.κ, p.τ
    ϵp = p.ϵp
    T = typeof(ϵ)

    # Numerator  Σ_{n≥r} (-1)^n · G(n) · f_n  with the gamma product
    #   G(n) = Γ(n+r+2ν+1)/Γ(n-r+1) · Γ(n+ν+1+s+iϵ)/Γ(n+ν+1-s-iϵ)
    #                                · Γ(n+ν+1+iτ)/Γ(n+ν+1-iτ).
    # Building the consecutive-n arguments by an incremental Pochhammer product —
    #   G(n)/G(n-1) = (n+r+2ν)(n+ν+s+iϵ)(n+ν+iτ) / [(n-r)(n+ν-s-iϵ)(n+ν-iτ)]
    # — replaces six full-precision Γ evaluations per term with three mults / three
    # divides, leaving only the FIVE Γ in the n=r base (Γ(n-r+1)=Γ(1)=1 there). The
    # ratio stays polynomially bounded (Γ(n+2ν+1)/Γ(n+1) ~ n^{2ν}), so no overflow.
    sν, dν = ν + 1 + s + im*ϵ, ν + 1 - s - im*ϵ
    tν, uν = ν + 1 + im*τ,     ν + 1 - im*τ
    G = _cgamma(T(2r) + 2ν + 1) *
        _cgamma(T(r) + sν) / _cgamma(T(r) + dν) *
        _cgamma(T(r) + tν) / _cgamma(T(r) + uν)        # G(r)
    sgn = isodd(r) ? -one(T) : one(T)                  # (-1)^r
    num_sum = zero(T)
    fr = get(fn, r, zero(T))
    iszero(fr) || (num_sum += sgn * G * fr)
    for n in (r+1):nmax
        G *= (T(n + r) + 2ν) * (T(n - 1) + sν) * (T(n - 1) + tν) /
             ((T(n - r)) * (T(n - 1) + dν) * (T(n - 1) + uν))
        sgn = -sgn
        fn_n = get(fn, n, zero(T))
        iszero(fn_n) && continue
        num_sum += sgn * G * fn_n
    end

    # Denominator weights v(n) = (-1)^n / Γ(r-n+1) / (e)_n · (c)_n/(d)_n built
    # incrementally from the n = r anchor by the descending ratio
    #   v(n) = -v(n+1)·(e+n)(d+n) / ((r-n)(c+n)),
    # replacing three O(|n|) pochhammer(·,n) calls per term (O(nmax²) total) with
    # O(1) work per n; the near-pole conditioning is unchanged (the ratio divides
    # by the same (c+n) factors pochhammer would).  Summation order (-nmax → r,
    # small far-tail terms first) is preserved via the weight vector.
    cd = ν + 1 + s - im*ϵ
    dd = ν + 1 - s + im*ϵ
    ew = T(r) + 2ν + 2
    voff = 1 + nmax
    v = Vector{T}(undef, nmax + r + 1)
    v[r + voff] = T((-1)^r / pochhammer(ew, r) *
                    pochhammer(cd, r) / pochhammer(dd, r))
    for n in (r-1):-1:-nmax
        v[n + voff] = _strip_radius(-v[n + 1 + voff] * (ew + n) * (dd + n) /
                                    (T(r - n) * (cd + n)))
    end
    den_sum = zero(T)
    for n in -nmax:r
        fn_n = fn[n]
        iszero(fn_n) && continue
        den_sum += v[n + voff] * fn_n
    end

    prefactor = exp(im*ϵ*κ) * (2*ϵ*κ)^(s - ν - r) * T(2)^(-s) /
                im^r *
                _cgamma(T(1) - s - 2im*ϵp) *
                _cgamma(T(r) + 2ν + 2) /
                (_cgamma(T(r) + ν + 1 - s + im*ϵ) *
                 _cgamma(T(r) + ν + 1 + im*τ) *
                 _cgamma(T(r) + ν + 1 + s + im*ϵ))

    return prefactor * num_sum / den_sum
end

# ============================================================
#  Full asymptotic amplitudes, Eqs. (167)-(170)
# ============================================================

"""
    compute_amplitudes(s, l, m, a, ω; nmax=80, nmax_cf=150, ν_init=nothing, method="Monodromy")

Compute (B^inc, B^ref, B^trans, C^trans) using the MST formalism.
Returns a NamedTuple with fields: Binc, Bref, Btrans, Ctrans, ν, fn, Ap, Am, Kν, Kνn

`Binc` and `Bref` are normalized by `Btrans` (i.e., Binc_raw/Btrans), matching the
Wolfram Teukolsky package convention. The physical Green's function is then simply
`G = Rin(r) / (2iω × Binc)`.

`method` is passed to `compute_nu`: `"Monodromy"` (default) or `"Newton"`.

`backend` selects the working precision: `:float64`, `:bigfloat`, `:multifloat`,
`:arb`, or `:acb` (Arb ball arithmetic at `precision` bits; for amplitudes `:arb`
and `:acb` are equivalent). Default `:auto` leaves the inputs untouched.
"""
function compute_amplitudes(s::Int, l::Int, m::Int, a, ω;
                            nmax::Int=80, nmax_cf::Int=150, ν_init=nothing,
                            method::String="Monodromy",
                            backend::Symbol=:auto, precision::Int=256,
                            l_max::Int=0)
    # ADDITIVE precision-backend dispatch: backend ∈ {:float64,:bigfloat,:multifloat}
    # converts the inputs to the chosen working float type and recurses through the
    # generic (type-driven) path.  Default :auto leaves the inputs untouched, so
    # existing callers are byte-for-byte unchanged.
    if backend !== :auto
        return _with_backend(backend, precision, a, ω) do a_w, ω_w
            compute_amplitudes(s, l, m, a_w, ω_w; nmax=nmax, nmax_cf=nmax_cf,
                ν_init = ν_init === nothing ? nothing : complex(ν_init),
                method=method, backend=:auto, l_max=l_max)
        end
    end
    ν, p = compute_nu(s, l, m, a, ω; nmax_cf=nmax_cf, ν_init=ν_init, method=method,
                      l_max=l_max)

    # εp = (ε+τ)/2 = 0 at the superradiance boundary ω = mΩH.
    # There the transmission amplitude Btrans → 0 and Binc, Bref are not well-defined.
    # Matches Wolfram Teukolsky package: returns Indeterminate for Inc/Ref at εp=0.
    if abs(p.ϵp) ≤ 100 * eps(typeof(real(p.ϵp)))
        @warn "compute_amplitudes: εp = (ε+τ)/2 = 0 (ω = mΩH, superradiance boundary). " *
              "Binc, Bref, Kν are not well-defined. Returning NaN for amplitude fields."
        nan = complex(NaN, NaN)
        fn_d = compute_fn(p, ν; nmax=nmax)
        return (Binc=nan, Bref=nan, Btrans=nan, Ctrans=nan,
                ν=ν, fn=fn_d, Ap=nan, Am=nan, Kν=nan, Kνn=nan)
    end

    fn = compute_fn(p, ν; nmax=nmax)

    Ap = compute_Aplus(p, ν, fn; nmax=nmax)
    Am = compute_Aminus(p, ν, fn; nmax=nmax)

    Kν = compute_Knu(p, ν, fn; nmax=nmax)

    fn_negν = compute_fn(p, -ν - 1; nmax=nmax)
    Kνn = compute_Knu(p, -ν - 1, fn_negν; nmax=nmax)

    ω_c = p.ω
    ϵ = p.ϵ
    κ = p.κ
    τ = p.τ
    phase = exp(-im * (ϵ * log(ϵ) - (1 - κ) / 2 * ϵ))
    phase_conj = exp(im * (ϵ * log(ϵ) - (1 - κ) / 2 * ϵ))

    πT = real(typeof(ϵ))(π)   # full-precision π (π*im would round through ComplexF64)
    sinν_factor = sin(πT * (ν - s + im*ϵ)) / sin(πT * (ν + s - im*ϵ))
    Binc = ω_c^(-1) * (Kν - im * exp(-im*πT*ν) * sinν_factor * Kνn) * Ap * phase

    Bref = ω_c^(-1 - 2s) * (Kν + im * exp(im*πT*ν) * Kνn) * Am * phase_conj

    Σfn = sum(fn[n] for n in -nmax:nmax)
    Btrans = (ϵ * κ / ω_c)^(2s) *
             exp(im * (ϵ + τ) * κ * (0.5 + log(κ) / (1 + κ))) * Σfn

    Ctrans = _ctrans(p, Am)

    return (Binc=Binc/Btrans, Bref=Bref/Btrans, Btrans=Btrans, Ctrans=Ctrans,
            ν=ν, fn=fn, Ap=Ap, Am=Am, Kν=Kν, Kνn=Kνn)
end

# ============================================================
#  ν-fixed mode
# ============================================================

"""
    compute_amplitudes_nufixed(s, l, m, a, ω, ν_fixed; nmax=80)

Same as `compute_amplitudes` but with ν fixed (no ν solver).
"""
function compute_amplitudes_nufixed(s::Int, l::Int, m::Int, a, ω,
                                     ν_fixed; nmax::Int=80,
                                     backend::Symbol=:auto, precision::Int=256,
                                     l_max::Int=0)
    if backend !== :auto
        return _with_backend(backend, precision, a, ω) do a_w, ω_w
            compute_amplitudes_nufixed(s, l, m, a_w, ω_w, ν_fixed;
                                       nmax=nmax, backend=:auto, l_max=l_max)
        end
    end
    p = MSTParams(s, l, m, a, ω; l_max=l_max)
    R = typeof(p.a)
    # Step off exact integer/half-integer ν (removable Γ-pole). δ=√eps balances
    # the O(δ) bias against the O(eps/δ) near-pole cancellation and, crucially,
    # scales with precision (the old fixed 1e-10 capped BigFloat results at 1e-10).
    δ = sqrt(eps(R))
    ν = Complex{R}(ν_fixed + δ)

    if abs(p.ϵp) ≤ 100 * eps(typeof(real(p.ϵp)))
        @warn "compute_amplitudes_nufixed: εp = 0 (ω = mΩH). Returning NaN for amplitude fields."
        nan = complex(NaN, NaN)
        fn_d = compute_fn(p, ν; nmax=nmax)
        return (Binc=nan, Bref=nan, Btrans=nan, Ctrans=nan,
                ν=ν, fn=fn_d, Ap=nan, Am=nan, Kν=nan, Kνn=nan)
    end

    fn = compute_fn(p, ν; nmax=nmax)

    Ap = compute_Aplus(p, ν, fn; nmax=nmax)
    Am = compute_Aminus(p, ν, fn; nmax=nmax)

    Kν = compute_Knu(p, ν, fn; nmax=nmax)

    ν_neg = -ν - 1
    fn_negν = compute_fn(p, ν_neg; nmax=nmax)
    Kνn = compute_Knu(p, ν_neg, fn_negν; nmax=nmax)

    ω_c = p.ω
    ϵ = p.ϵ
    κ = p.κ
    τ = p.τ
    phase = exp(-im * (ϵ * log(ϵ) - (1 - κ) / 2 * ϵ))
    phase_conj = exp(im * (ϵ * log(ϵ) - (1 - κ) / 2 * ϵ))

    πT = real(typeof(ϵ))(π)   # full-precision π (π*im would round through ComplexF64)
    sinν_factor = sin(πT * (ν - s + im*ϵ)) / sin(πT * (ν + s - im*ϵ))
    Binc = ω_c^(-1) * (Kν - im * exp(-im*πT*ν) * sinν_factor * Kνn) * Ap * phase
    Bref = ω_c^(-1 - 2s) * (Kν + im * exp(im*πT*ν) * Kνn) * Am * phase_conj

    Σfn = sum(fn[n] for n in -nmax:nmax)
    Btrans = (ϵ * κ / ω_c)^(2s) *
             exp(im * (ϵ + τ) * κ * (0.5 + log(κ) / (1 + κ))) * Σfn
    Ctrans = _ctrans(p, Am)

    return (Binc=Binc/Btrans, Bref=Bref/Btrans, Btrans=Btrans, Ctrans=Ctrans,
            ν=ν, fn=fn, Ap=Ap, Am=Am, Kν=Kν, Kνn=Kνn)
end

# ============================================================
#  Meromorphic K_ν
# ============================================================

function compute_Knu_mero(p::MSTParams, ν, fn; nmax::Int=80, r::Int=0)
    s, ϵ, κ, τ = p.s, p.ϵ, p.κ, p.τ
    ϵp = p.ϵp
    T = typeof(ϵ)

    # Numerator  Σ_{n≥r} (-1)^n · G(n) · f_n  with the gamma product
    #   G(n) = Γ(n+r+2ν+1)/Γ(n-r+1) · Γ(n+ν+1+s+iϵ)/Γ(n+ν+1-s-iϵ)
    #                                · Γ(n+ν+1+iτ)/Γ(n+ν+1-iτ).
    # Building the consecutive-n arguments by an incremental Pochhammer product —
    #   G(n)/G(n-1) = (n+r+2ν)(n+ν+s+iϵ)(n+ν+iτ) / [(n-r)(n+ν-s-iϵ)(n+ν-iτ)]
    # — replaces six full-precision Γ evaluations per term with three mults / three
    # divides, leaving only the FIVE Γ in the n=r base (Γ(n-r+1)=Γ(1)=1 there). The
    # ratio stays polynomially bounded (Γ(n+2ν+1)/Γ(n+1) ~ n^{2ν}), so no overflow.
    sν, dν = ν + 1 + s + im*ϵ, ν + 1 - s - im*ϵ
    tν, uν = ν + 1 + im*τ,     ν + 1 - im*τ
    G = _cgamma(T(2r) + 2ν + 1) *
        _cgamma(T(r) + sν) / _cgamma(T(r) + dν) *
        _cgamma(T(r) + tν) / _cgamma(T(r) + uν)        # G(r)
    sgn = isodd(r) ? -one(T) : one(T)                  # (-1)^r
    num_sum = zero(T)
    fr = get(fn, r, zero(T))
    iszero(fr) || (num_sum += sgn * G * fr)
    for n in (r+1):nmax
        G *= (T(n + r) + 2ν) * (T(n - 1) + sν) * (T(n - 1) + tν) /
             ((T(n - r)) * (T(n - 1) + dν) * (T(n - 1) + uν))
        sgn = -sgn
        fn_n = get(fn, n, zero(T))
        iszero(fn_n) && continue
        num_sum += sgn * G * fn_n
    end

    # Incremental denominator weights — see compute_Knu for the derivation.
    cd = ν + 1 + s - im*ϵ
    dd = ν + 1 - s + im*ϵ
    ew = T(r) + 2ν + 2
    voff = 1 + nmax
    v = Vector{T}(undef, nmax + r + 1)
    v[r + voff] = T((-1)^r / pochhammer(ew, r) *
                    pochhammer(cd, r) / pochhammer(dd, r))
    for n in (r-1):-1:-nmax
        v[n + voff] = _strip_radius(-v[n + 1 + voff] * (ew + n) * (dd + n) /
                                    (T(r - n) * (cd + n)))
    end
    den_sum = zero(T)
    for n in -nmax:r
        fn_n = fn[n]
        iszero(fn_n) && continue
        den_sum += v[n + voff] * fn_n
    end

    prefactor = exp(im*ϵ*κ) * (2*ϵ*κ)^(s - r) * T(2)^(-s) /
                im^r *
                _cgamma(T(1) - s - 2im*ϵp) *
                _cgamma(T(r) + 2ν + 2) /
                (_cgamma(T(r) + ν + 1 - s + im*ϵ) *
                 _cgamma(T(r) + ν + 1 + im*τ) *
                 _cgamma(T(r) + ν + 1 + s + im*ϵ))

    return prefactor * num_sum / den_sum
end

# ============================================================
#  Meromorphic amplitudes
# ============================================================

"""
    compute_amplitudes_mero(s, l, m, a, ω; nmax=80, nmax_cf=150)

Meromorphic mode: amplitudes with branch-cut factors removed.
"""
function compute_amplitudes_mero(s::Int, l::Int, m::Int, a, ω;
                                  nmax::Int=80, nmax_cf::Int=150, method::String="Monodromy",
                                  backend::Symbol=:auto, precision::Int=256,
                                  l_max::Int=0)
    if backend !== :auto
        return _with_backend(backend, precision, a, ω) do a_w, ω_w
            compute_amplitudes_mero(s, l, m, a_w, ω_w;
                nmax=nmax, nmax_cf=nmax_cf, method=method, backend=:auto,
                l_max=l_max)
        end
    end
    ν, p = compute_nu(s, l, m, a, ω; nmax_cf=nmax_cf, method=method, l_max=l_max)

    if abs(p.ϵp) ≤ 100 * eps(typeof(real(p.ϵp)))
        @warn "compute_amplitudes_mero: εp = 0 (ω = mΩH). Returning NaN for amplitude fields."
        nan = complex(NaN, NaN)
        fn_d = compute_fn(p, ν; nmax=nmax)
        return (Binc=nan, Bref=nan, Btrans=nan, Ctrans=nan,
                ν=ν, fn=fn_d, Ap=nan, Am=nan, Kν=nan, Kνn=nan)
    end

    fn = compute_fn(p, ν; nmax=nmax)

    Ap = compute_Aplus(p, ν, fn; nmax=nmax)
    Am = compute_Aminus(p, ν, fn; nmax=nmax)

    Kν = compute_Knu_mero(p, ν, fn; nmax=nmax)
    fn_negν = compute_fn(p, -ν - 1; nmax=nmax)
    Kνn = compute_Knu_mero(p, -ν - 1, fn_negν; nmax=nmax)

    ω_c = p.ω
    ϵ = p.ϵ
    κ = p.κ
    τ = p.τ
    phase_mero = exp(-im * (-(1 - κ) / 2 * ϵ))
    phase_conj_mero = exp(im * (-(1 - κ) / 2 * ϵ))

    πT = real(typeof(ϵ))(π)   # full-precision π (π*im would round through ComplexF64)
    sinν_factor = sin(πT * (ν - s + im*ϵ)) / sin(πT * (ν + s - im*ϵ))
    Binc = ω_c^(-1) * (Kν - im * exp(-im*πT*ν) * sinν_factor * Kνn) * Ap * phase_mero
    Bref = ω_c^(-1 - 2s) * (Kν + im * exp(im*πT*ν) * Kνn) * Am * phase_conj_mero

    Σfn = sum(fn[n] for n in -nmax:nmax)
    Btrans = (ϵ * κ / ω_c)^(2s) *
             exp(im * (ϵ + τ) * κ * (0.5 + log(κ) / (1 + κ))) * Σfn
    Ctrans = ω_c^(-1 - 2s) * Am * phase_conj_mero

    return (Binc=Binc/Btrans, Bref=Bref/Btrans, Btrans=Btrans, Ctrans=Ctrans,
            ν=ν, fn=fn, Ap=Ap, Am=Am, Kν=Kν, Kνn=Kνn)
end

"""
    compute_amplitudes_nufixed_mero(s, l, m, a, ω, ν_fixed; nmax=80)

Meromorphic mode with ν fixed.
"""
function compute_amplitudes_nufixed_mero(s::Int, l::Int, m::Int, a, ω,
                                          ν_fixed; nmax::Int=80,
                                          backend::Symbol=:auto, precision::Int=256,
                                          l_max::Int=0)
    if backend !== :auto
        return _with_backend(backend, precision, a, ω) do a_w, ω_w
            compute_amplitudes_nufixed_mero(s, l, m, a_w, ω_w, ν_fixed;
                                            nmax=nmax, backend=:auto, l_max=l_max)
        end
    end
    p = MSTParams(s, l, m, a, ω; l_max=l_max)
    R = typeof(p.a)
    # Step off exact integer/half-integer ν (removable Γ-pole). δ=√eps balances
    # the O(δ) bias against the O(eps/δ) near-pole cancellation and, crucially,
    # scales with precision (the old fixed 1e-10 capped BigFloat results at 1e-10).
    δ = sqrt(eps(R))
    ν = Complex{R}(ν_fixed + δ)

    if abs(p.ϵp) ≤ 100 * eps(typeof(real(p.ϵp)))
        @warn "compute_amplitudes_nufixed_mero: εp = 0 (ω = mΩH). Returning NaN for amplitude fields."
        nan = complex(NaN, NaN)
        fn_d = compute_fn(p, ν; nmax=nmax)
        return (Binc=nan, Bref=nan, Btrans=nan, Ctrans=nan,
                ν=ν, fn=fn_d, Ap=nan, Am=nan, Kν=nan, Kνn=nan)
    end

    fn = compute_fn(p, ν; nmax=nmax)

    Ap = compute_Aplus(p, ν, fn; nmax=nmax)
    Am = compute_Aminus(p, ν, fn; nmax=nmax)

    Kν = compute_Knu_mero(p, ν, fn; nmax=nmax)

    ν_neg = -ν - 1
    fn_negν = compute_fn(p, ν_neg; nmax=nmax)
    Kνn = compute_Knu_mero(p, ν_neg, fn_negν; nmax=nmax)

    ω_c = p.ω
    ϵ = p.ϵ
    κ = p.κ
    τ = p.τ
    phase_mero = exp(-im * (-(1 - κ) / 2 * ϵ))
    phase_conj_mero = exp(im * (-(1 - κ) / 2 * ϵ))

    πT = real(typeof(ϵ))(π)   # full-precision π (π*im would round through ComplexF64)
    sinν_factor = sin(πT * (ν - s + im*ϵ)) / sin(πT * (ν + s - im*ϵ))
    Binc = ω_c^(-1) * (Kν - im * exp(-im*πT*ν) * sinν_factor * Kνn) * Ap * phase_mero
    Bref = ω_c^(-1 - 2s) * (Kν + im * exp(im*πT*ν) * Kνn) * Am * phase_conj_mero

    Σfn = sum(fn[n] for n in -nmax:nmax)
    Btrans = (ϵ * κ / ω_c)^(2s) *
             exp(im * (ϵ + τ) * κ * (0.5 + log(κ) / (1 + κ))) * Σfn
    Ctrans = ω_c^(-1 - 2s) * Am * phase_conj_mero

    return (Binc=Binc/Btrans, Bref=Bref/Btrans, Btrans=Btrans, Ctrans=Ctrans,
            ν=ν, fn=fn, Ap=Ap, Am=Am, Kν=Kν, Kνn=Kνn)
end
