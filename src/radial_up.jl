# ============================================================
#  Rup: Upgoing radial solution (at infinity, HypergeometricU-based)
#
#  Sasaki-Tagoshi Eqs. (153), (159); MST.m lines 616-652
#  Teukolsky case only.
#
#  R_up(r) = prefac(ẑ) / Ctrans * Σ_n fUp_n * HU[n]
#
#  where ẑ = ε(r - r-)/2,  c = -2iẑ
#  HU[n] = c^n U(n+aU, 2n+bU, c)
#  aU = ν+s+1-iε, bU = 2ν+2
#
#  fUp_n = (-1)^n Pochhammer(ν+1+s-iε, n) / Pochhammer(ν+1-s+iε, n) * fn[n]
#
#  prefac = 2^ν e^{-πε} e^{-iπ(ν+1)} e^{iẑ} ẑ^{ν+i(ε+τ)/2}
#           (ẑ-εκ)^{-i(ε+τ)/2-s} e^{-iπs}
#
#  Ctrans = ω^{-1-2s} A^ν_- exp(i(ε log ε - (1-κ)/2 ε))
#
#  Normalized so that at infinity:
#    Rup ~ r^{-1-2s} e^{+iωr*}
#  matching Mathematica's MSTRadialUp (MST.m: prefac/UpTrans).
# ============================================================

"""
    Rup(p::MSTParams, ν, fn, r; nmax=80, tol=1e-14, ctrans=nothing)

Compute the upgoing radial Teukolsky solution at Boyer-Lindquist radius r.
Normalized by Ctrans = ω^{-1-2s} A^ν_- exp(i(ε log ε - (1-κ)/2 ε)) so that:

    Rup ~ r^{-1-2s} e^{+iωr*}  at r → ∞

Matches Mathematica's MSTRadialUp convention (norm = UpTrans).

If `ctrans` is provided (e.g. `amp.Ctrans` from `compute_amplitudes`), it is
used directly and `A^ν_-` is not recomputed.
"""
function Rup(p::MSTParams, ν, fn, r; nmax::Int=80, tol::Real=100*eps(real(typeof(p.ϵ))),
             ctrans=nothing)
    ϵ, κ, τ, s = p.ϵ, p.κ, p.τ, p.s
    rm = p.rm
    zhat = complex(ϵ * (r - rm) / 2)

    hp = HUParams(p, ν, zhat)

    # Prefactor (Teukolsky, MST.m line 90)
    # πT: π at the working precision.  `-π*ϵ` / `-im*π*(ν+1)` materialize
    # Float64(π) (unary minus / im product on the Irrational), silently
    # polluting every Arb/BigFloat Rup value at ~ϵ·1.2e-16 relative — caught
    # by the independent-π direct-sum arbiter in test_hu_evaluation.jl.
    πT = π * one(real(ν))
    prefac = 2^ν * exp(-πT*ϵ) * exp(-im*πT*(ν + 1)) *
             exp(im*zhat) * zhat^(ν + im*(ϵ + τ)/2) *
             (zhat - ϵ*κ)^(-im*(ϵ + τ)/2) *
             exp(-im*πT*s) * (zhat - ϵ*κ)^(-s)

    # HU evaluator: certified seeds + stable outward march for the Arb/BigFloat
    # backends, legacy exact-seeded recurrence + ratio guard otherwise.
    # See the "Certified HU / dHU evaluation" section in hypergeometric.jl.
    get_hu, _ = _hu_dhu_evaluators(hp)

    # fUp_n weight (-1)^n (aw)_n/(bw)_n carried incrementally per direction —
    # O(1) work per term instead of two O(|n|) pochhammer(·,n) calls (O(nmax²)
    # total at full precision); see compute_Aminus for the ratio derivation.
    T = typeof(p.ϵ)
    aw = ν + 1 + s - im*ϵ
    bw = ν + 1 - s + im*ϵ

    # Sum bidirectionally
    result = zero(T)
    w = one(T)                       # (-1)^n (aw)_n/(bw)_n at the current n
    for n in 0:nmax
        n > 0 && (w = _strip_radius(-w * (aw + (n - 1)) / (bw + (n - 1))))
        fn_n = get(fn, n, zero(T))
        iszero(fn_n) && continue
        term = prefac * (w * fn_n) * get_hu(n)
        result += term
        n > 0 && abs(term) < tol * abs(result) + tol && break
    end

    res_down = zero(T)
    w = one(T)
    for n in -1:-1:-nmax
        w = _strip_radius(-w * (bw + n) / (aw + n))
        fn_n = get(fn, n, zero(T))
        iszero(fn_n) && continue
        term = prefac * (w * fn_n) * get_hu(n)
        res_down += term
        abs(term) < tol * abs(res_down) + tol && break
    end

    raw = result + res_down

    # Normalize by Ctrans = ω^{-1-2s} A^ν_- exp(i(ε log ε - (1-κ)/2 ε))
    # Matches Mathematica: prefac/UpTrans (MST.m line 640)
    # Use caller-supplied ctrans when available to avoid recomputing A^ν_-.
    ct = if ctrans !== nothing
        ctrans
    else
        _ctrans(p, compute_Aminus(p, ν, fn; nmax=nmax))
    end

    return raw / ct
end

"""
    dRup(p::MSTParams, ν, fn, r; nmax=80, tol=1e-14, ctrans=nothing)

Compute dR_up/dr at Boyer-Lindquist radius r.
Normalized by the same Ctrans as `Rup`.

If `ctrans` is provided (e.g. `amp.Ctrans` from `compute_amplitudes`), it is
used directly and `A^ν_-` is not recomputed.
"""
function dRup(p::MSTParams, ν, fn, r; nmax::Int=80, tol::Real=100*eps(real(typeof(p.ϵ))),
              ctrans=nothing)
    ϵ, κ, τ, s = p.ϵ, p.κ, p.τ, p.s
    rm = p.rm
    zhat = complex(ϵ * (r - rm) / 2)
    dzhatdr = ϵ / 2

    hp = HUParams(p, ν, zhat)

    # Prefactor components (πT: working-precision π — see Rup above)
    πT = π * one(real(ν))
    A = 2^ν * exp(-πT*ϵ) * exp(-im*πT*(ν + 1)) * exp(-im*πT*s)
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

    # HU/dHU evaluators: certified seeds + stable outward march for the
    # Arb/BigFloat backends, legacy exact-seeded recurrence + ratio guard
    # otherwise (see hypergeometric.jl, "Certified HU / dHU evaluation").
    get_hu, get_dhu = _hu_dhu_evaluators(hp)

    # fUp_n weight carried incrementally per direction (see Rup above).
    T = typeof(p.ϵ)
    aw = ν + 1 + s - im*ϵ
    bw = ν + 1 - s + im*ϵ

    # Sum: dRup/dr = Σ fUp_n * (dprefac * HU[n] + prefac * dHU[n] * dzhatdr)
    result = zero(T)
    w = one(T)                       # (-1)^n (aw)_n/(bw)_n at the current n
    for n in 0:nmax
        n > 0 && (w = _strip_radius(-w * (aw + (n - 1)) / (bw + (n - 1))))
        fn_n = get(fn, n, zero(T))
        iszero(fn_n) && continue
        term = (w * fn_n) * (dprefac * get_hu(n) + prefac_dzdr * get_dhu(n))
        result += term
        n > 0 && abs(term) < tol * abs(result) + tol && break
    end

    res_down = zero(T)
    w = one(T)
    for n in -1:-1:-nmax
        w = _strip_radius(-w * (bw + n) / (aw + n))
        fn_n = get(fn, n, zero(T))
        iszero(fn_n) && continue
        term = (w * fn_n) * (dprefac * get_hu(n) + prefac_dzdr * get_dhu(n))
        res_down += term
        abs(term) < tol * abs(res_down) + tol && break
    end

    raw = result + res_down

    # Normalize by Ctrans (same as Rup)
    ct = if ctrans !== nothing
        ctrans
    else
        _ctrans(p, compute_Aminus(p, ν, fn; nmax=nmax))
    end

    return raw / ct
end
