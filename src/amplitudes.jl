# ============================================================
#  Asymptotic amplitudes A^ν_±, Eqs. (157)-(158)
# ============================================================

function compute_Aplus(p::MSTParams, ν, fn; nmax::Int=80, nmin::Int=-nmax)
    s, ϵ = p.s, p.ϵ
    T = typeof(ϵ)

    prefactor = exp(-π*ϵ/2) * exp(π*im*(ν+1-s)/2) *
                T(2)^(-1+s-im*ϵ) *
                _cgamma(ν + 1 - s + im*ϵ) / _cgamma(ν + 1 + s - im*ϵ)

    Σ = sum(fn[n] for n in nmin:nmax)

    return prefactor * Σ
end

function compute_Aminus(p::MSTParams, ν, fn; nmax::Int=80, nmin::Int=-nmax)
    s, ϵ = p.s, p.ϵ
    T = typeof(ϵ)

    prefactor = T(2)^(-1-s+im*ϵ) *
                exp(-π*im*(ν+1+s)/2) *
                exp(-π*ϵ/2)

    Σ = sum(
        begin
            fn_n = fn[n]
            iszero(fn_n) ? zero(T) :
            (-1)^n * pochhammer(ν + 1 + s - im*ϵ, n) /
                     pochhammer(ν + 1 - s + im*ϵ, n) * fn_n
        end
        for n in nmin:nmax
    )

    return prefactor * Σ
end

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

    den_sum = zero(T)
    for n in -nmax:r
        fn_n = fn[n]
        iszero(fn_n) && continue
        term = (-1)^n / _cgamma(T(r - n + 1)) /
               pochhammer(r + 2ν + 2, n) *
               pochhammer(ν + 1 + s - im*ϵ, n) /
               pochhammer(ν + 1 - s + im*ϵ, n) *
               fn_n
        den_sum += term
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
"""
function compute_amplitudes(s::Int, l::Int, m::Int, a, ω;
                            nmax::Int=80, nmax_cf::Int=150, ν_init=nothing,
                            method::String="Monodromy",
                            backend::Symbol=:auto, precision::Int=256)
    # ADDITIVE precision-backend dispatch: backend ∈ {:float64,:bigfloat,:multifloat}
    # converts the inputs to the chosen working float type and recurses through the
    # generic (type-driven) path.  Default :auto leaves the inputs untouched, so
    # existing callers are byte-for-byte unchanged.
    if backend !== :auto
        return _with_backend(backend, precision, a, ω) do a_w, ω_w
            compute_amplitudes(s, l, m, a_w, ω_w; nmax=nmax, nmax_cf=nmax_cf,
                ν_init = ν_init === nothing ? nothing : complex(ν_init),
                method=method, backend=:auto)
        end
    end
    ν, p = compute_nu(s, l, m, a, ω; nmax_cf=nmax_cf, ν_init=ν_init, method=method)

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

    sinν_factor = sin(π * (ν - s + im*ϵ)) / sin(π * (ν + s - im*ϵ))
    Binc = ω_c^(-1) * (Kν - im * exp(-im*π*ν) * sinν_factor * Kνn) * Ap * phase

    Bref = ω_c^(-1 - 2s) * (Kν + im * exp(im*π*ν) * Kνn) * Am * phase_conj

    Σfn = sum(fn[n] for n in -nmax:nmax)
    Btrans = (ϵ * κ / ω_c)^(2s) *
             exp(im * (ϵ + τ) * κ * (0.5 + log(κ) / (1 + κ))) * Σfn

    Ctrans = ω_c^(-1 - 2s) * Am * phase_conj

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
                                     backend::Symbol=:auto, precision::Int=256)
    if backend !== :auto
        return _with_backend(backend, precision, a, ω) do a_w, ω_w
            compute_amplitudes_nufixed(s, l, m, a_w, ω_w, ν_fixed;
                                       nmax=nmax, backend=:auto)
        end
    end
    p = MSTParams(s, l, m, a, ω)
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

    sinν_factor = sin(π * (ν - s + im*ϵ)) / sin(π * (ν + s - im*ϵ))
    Binc = ω_c^(-1) * (Kν - im * exp(-im*π*ν) * sinν_factor * Kνn) * Ap * phase
    Bref = ω_c^(-1 - 2s) * (Kν + im * exp(im*π*ν) * Kνn) * Am * phase_conj

    Σfn = sum(fn[n] for n in -nmax:nmax)
    Btrans = (ϵ * κ / ω_c)^(2s) *
             exp(im * (ϵ + τ) * κ * (0.5 + log(κ) / (1 + κ))) * Σfn
    Ctrans = ω_c^(-1 - 2s) * Am * phase_conj

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

    den_sum = zero(T)
    for n in -nmax:r
        fn_n = fn[n]
        iszero(fn_n) && continue
        term = (-1)^n / _cgamma(T(r - n + 1)) /
               pochhammer(r + 2ν + 2, n) *
               pochhammer(ν + 1 + s - im*ϵ, n) /
               pochhammer(ν + 1 - s + im*ϵ, n) *
               fn_n
        den_sum += term
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
                                  backend::Symbol=:auto, precision::Int=256)
    if backend !== :auto
        return _with_backend(backend, precision, a, ω) do a_w, ω_w
            compute_amplitudes_mero(s, l, m, a_w, ω_w;
                nmax=nmax, nmax_cf=nmax_cf, method=method, backend=:auto)
        end
    end
    ν, p = compute_nu(s, l, m, a, ω; nmax_cf=nmax_cf, method=method)

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

    sinν_factor = sin(π * (ν - s + im*ϵ)) / sin(π * (ν + s - im*ϵ))
    Binc = ω_c^(-1) * (Kν - im * exp(-im*π*ν) * sinν_factor * Kνn) * Ap * phase_mero
    Bref = ω_c^(-1 - 2s) * (Kν + im * exp(im*π*ν) * Kνn) * Am * phase_conj_mero

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
                                          backend::Symbol=:auto, precision::Int=256)
    if backend !== :auto
        return _with_backend(backend, precision, a, ω) do a_w, ω_w
            compute_amplitudes_nufixed_mero(s, l, m, a_w, ω_w, ν_fixed;
                                            nmax=nmax, backend=:auto)
        end
    end
    p = MSTParams(s, l, m, a, ω)
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

    sinν_factor = sin(π * (ν - s + im*ϵ)) / sin(π * (ν + s - im*ϵ))
    Binc = ω_c^(-1) * (Kν - im * exp(-im*π*ν) * sinν_factor * Kνn) * Ap * phase_mero
    Bref = ω_c^(-1 - 2s) * (Kν + im * exp(im*π*ν) * Kνn) * Am * phase_conj_mero

    Σfn = sum(fn[n] for n in -nmax:nmax)
    Btrans = (ϵ * κ / ω_c)^(2s) *
             exp(im * (ϵ + τ) * κ * (0.5 + log(κ) / (1 + κ))) * Σfn
    Ctrans = ω_c^(-1 - 2s) * Am * phase_conj_mero

    return (Binc=Binc/Btrans, Bref=Bref/Btrans, Btrans=Btrans, Ctrans=Ctrans,
            ν=ν, fn=fn, Ap=Ap, Am=Am, Kν=Kν, Kνn=Kνn)
end
