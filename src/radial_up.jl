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

# ── Converge-or-error summation for the U-side series (issue R1, Rup leg) ────
#
# Same policy as _Rin_raw (radial_in.jl): the old fixed-nmax loops carried a
# last-term break that was never verified to have fired, and `get(fn,n,zero)`
# silently treated fn-window exhaustion as a zero tail.  Measured at
# s=-2 l=m=2 a=0.7 on the positive imaginary axis: ω=10i, r=4 → 13% relative
# error at the default nmax=80; ω=16i, r=4 → 100% wrong at nmax=80 and still
# 11% at nmax=120 — all returned as plausible-looking numbers.  The U-side
# terms |c^n U(n+aU,2n+bU,c)|·|fUp_n·fn| start decaying only past
# n ≈ (3.5–4.3)·|ẑ| (c = −2iẑ), so small r + large |ϵ| needs far more than
# 80 terms.  Both directional sums now run through _sum_mst_series!
# (adaptive fn extension via _extend_fn_dir!, errors instead of returning
# unconverged) and the total is certified against the cancellation floor by
# _certify_mst_sum.

# First-extension target: cover the measured onset of U-term decay with
# margin so the common case needs at most one fn extension.  Non-finite/huge
# |ẑ| midpoints are clamped (the series driver clamps to nmax_hard anyway).
function _rup_next_ext(nmax::Int, zhat)
    m = _mag_f64(zhat)
    m = isfinite(m) ? min(m, 1.0e6) : 1.0e6
    return max(2 * nmax, 64 + ceil(Int, 4.5 * m))
end

# fUp weight w(n) = (-1)^n (aw)_n/(bw)_n marched incrementally in direction
# `dir` — O(1) per step instead of two O(|n|) pochhammer calls (see
# compute_Aminus for the ratio derivation and the _strip_radius rationale).
# Returned closure gives w at any n reached monotonically in `dir`; it also
# advances across n's the series driver skips (zero fn entries never reach
# the termfn, but the weight must still march past them).
function _fup_weight_stepper(aw::T, bw::T, dir::Int) where {T}
    w = Ref(one(T))
    at = Ref(0)
    return function (n::Int)
        if dir > 0
            while at[] < n
                w[] = _strip_radius(-w[] * (aw + at[]) / (bw + at[]))
                at[] += 1
            end
        else
            while at[] > n
                at[] -= 1
                w[] = _strip_radius(-w[] * (bw + at[]) / (aw + at[]))
            end
        end
        return w[]
    end
end

"""
    Rup(p::MSTParams, ν, fn, r; nmax=80, tol=100·eps, ctrans=nothing,
        nmax_hard=50_000, floor_tol=√eps)

Compute the upgoing radial Teukolsky solution at Boyer-Lindquist radius r.
Normalized by Ctrans = ω^{-1-2s} A^ν_- exp(i(ε log ε - (1-κ)/2 ε)) so that:

    Rup ~ r^{-1-2s} e^{+iωr*}  at r → ∞

Matches Mathematica's MSTRadialUp convention (norm = UpTrans).

If `ctrans` is provided (e.g. `amp.Ctrans` from `compute_amplitudes`), it is
used directly and `A^ν_-` is not recomputed.

Convergence policy (converge-or-error, same contract as [`Rin`](@ref)):
`nmax` is only the initial window hint — the U-side series is summed until
two consecutive terms pass the `tol` criterion, adaptively extending `fn`
(the dict is mutated) up to `nmax_hard`, and the result is certified against
the cancellation floor `eps·max|partial| ≤ max(tol, floor_tol)·|sum|`.
Errors — never returns silent garbage — when the series cannot converge or
certify at the working precision.  `tol ≤ 0` is the arbiter escape hatch:
exact fixed ±nmax window, no stopping test, no extension, no certification.
"""
function Rup(p::MSTParams, ν, fn, r; nmax::Int=80, tol::Real=100*eps(real(typeof(p.ϵ))),
             ctrans=nothing, nmax_hard::Int=50_000,
             floor_tol::Real=_default_floor_tol(real(typeof(p.ϵ))))
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

    # tol ≤ 0: arbiter escape hatch — exact fixed-window sum, no stopping
    # test, no fn extension, no certification.  Used by the direct-sum
    # arbiter tests (test_hu_evaluation.jl) to eliminate ALL truncation
    # decisions when measuring per-n HU errors.
    if tol <= 0
        result = zero(T)
        w = one(T)
        for n in 0:nmax
            n > 0 && (w = _strip_radius(-w * (aw + (n - 1)) / (bw + (n - 1))))
            fn_n = get(fn, n, zero(T))
            iszero(fn_n) && continue
            result += (w * fn_n) * get_hu(n)
        end
        w = one(T)
        for n in -1:-1:-nmax
            w = _strip_radius(-w * (bw + n) / (aw + n))
            fn_n = get(fn, n, zero(T))
            iszero(fn_n) && continue
            result += (w * fn_n) * get_hu(n)
        end
        raw = prefac * result
    else
        # Bidirectional converge-or-error sums.  prefac is hoisted out of the
        # term loops (one complex multiply per term saved); the absolute part
        # of the stopping/certification tests is rescaled by 1/|prefac| so the
        # truncation decisions match the un-hoisted form (same as _Rin_raw).
        tol_abs = tol / abs(prefac)
        n_ext = _rup_next_ext(nmax, zhat)

        w_up = _fup_weight_stepper(aw, bw, +1)
        res_up, smax_up = _sum_mst_series!(n -> w_up(n) * get_hu(n), fn, p, ν,
                                           +1, tol, tol_abs, n_ext, nmax_hard,
                                           "Rup")
        w_dn = _fup_weight_stepper(aw, bw, -1)
        res_down, smax_dn = _sum_mst_series!(n -> w_dn(n) * get_hu(n), fn, p, ν,
                                             -1, tol, tol_abs, n_ext, nmax_hard,
                                             "Rup")

        ctol = max(tol, floor_tol)
        raw = prefac * _certify_mst_sum(res_up + res_down,
                                        max(smax_up, smax_dn),
                                        ctol, ctol / abs(prefac), "Rup")
    end

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
    dRup(p::MSTParams, ν, fn, r; nmax=80, tol=100·eps, ctrans=nothing,
         nmax_hard=50_000, floor_tol=√eps)

Compute dR_up/dr at Boyer-Lindquist radius r.
Normalized by the same Ctrans as `Rup`.

If `ctrans` is provided (e.g. `amp.Ctrans` from `compute_amplitudes`), it is
used directly and `A^ν_-` is not recomputed.

Same converge-or-error contract as [`Rup`](@ref): `nmax` is only the initial
window hint; the series extends `fn` adaptively up to `nmax_hard` and errors
instead of returning an unconverged or floor-limited value.  `tol ≤ 0` is the
arbiter escape hatch (exact fixed window, no checks).
"""
function dRup(p::MSTParams, ν, fn, r; nmax::Int=80, tol::Real=100*eps(real(typeof(p.ϵ))),
              ctrans=nothing, nmax_hard::Int=50_000,
              floor_tol::Real=_default_floor_tol(real(typeof(p.ϵ))))
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

    # tol ≤ 0: arbiter escape hatch — exact fixed-window sum (see Rup).
    if tol <= 0
        raw = zero(T)
        w = one(T)
        for n in 0:nmax
            n > 0 && (w = _strip_radius(-w * (aw + (n - 1)) / (bw + (n - 1))))
            fn_n = get(fn, n, zero(T))
            iszero(fn_n) && continue
            raw += (w * fn_n) * (dprefac * get_hu(n) + prefac_dzdr * get_dhu(n))
        end
        w = one(T)
        for n in -1:-1:-nmax
            w = _strip_radius(-w * (bw + n) / (aw + n))
            fn_n = get(fn, n, zero(T))
            iszero(fn_n) && continue
            raw += (w * fn_n) * (dprefac * get_hu(n) + prefac_dzdr * get_dhu(n))
        end
    else
        # Sum: dRup/dr = Σ fUp_n * (dprefac * HU[n] + prefac * dHU[n] * dzhatdr)
        # Converge-or-error, same as Rup.  The two prefactors stay inside the
        # term (no single hoistable scale), so the absolute tolerance parts
        # are unscaled — matching the old loop's `+ tol` semantics.
        n_ext = _rup_next_ext(nmax, zhat)

        w_up = _fup_weight_stepper(aw, bw, +1)
        res_up, smax_up = _sum_mst_series!(
            n -> w_up(n) * (dprefac * get_hu(n) + prefac_dzdr * get_dhu(n)),
            fn, p, ν, +1, tol, tol, n_ext, nmax_hard, "dRup")
        w_dn = _fup_weight_stepper(aw, bw, -1)
        res_down, smax_dn = _sum_mst_series!(
            n -> w_dn(n) * (dprefac * get_hu(n) + prefac_dzdr * get_dhu(n)),
            fn, p, ν, -1, tol, tol, n_ext, nmax_hard, "dRup")

        ctol = max(tol, floor_tol)
        raw = _certify_mst_sum(res_up + res_down, max(smax_up, smax_dn),
                               ctol, ctol, "dRup")
    end

    # Normalize by Ctrans (same as Rup)
    ct = if ctrans !== nothing
        ctrans
    else
        _ctrans(p, compute_Aminus(p, ν, fn; nmax=nmax))
    end

    return raw / ct
end
