# ============================================================
#  Fully-native-Acb MST core solve  (M3)
#
#  Composes the native pieces into one Arb-native core:
#    ν  via the native-Acb monodromy kernel  (compute_nu backend=:acb, M2)
#    fₙ via compute_fn_acb                    (M3)
#    A± via compute_Aplus_acb/compute_Aminus_acb (M3)
#  The R^up seeds (acb_hypgeom_u) and the recurrence stay in the generic radial
#  code, which is already Arb-correct+fast once fed this core.  Drop-in shape for
#  `compute_mst_core`: returns `(p, ν, fn, Ap, Am)` with Complex{Arb} fields.
#
#  ν on the PIA: the historical pure-imaginary-ω fragility was the 4σ∈ℤ
#  monodromy resonance, SOLVED by the _monodromy_resonant gate (nu_solver.jl);
#  the :acb ν kernel is safe on the whole axis.  `ν` can still be supplied to
#  skip the solve (e.g. warm-started sweeps).
# ============================================================

"""
    compute_mst_core_acb(s, l, m, a, ω; ν=nothing, nmax=80, nmax_cf=2000,
                         precision=256, ν_init=nothing)

Native-Acb analogue of [`compute_mst_core`](@ref): ν via the native-Acb monodromy
kernel (or the supplied `ν`), then fₙ and A^ν_± via the in-place Acb kernels
(anchor Lentz + in-place CF peeling).  Runs inside `setprecision(Arb, precision)`.
Returns `(p, ν, fn, Ap, Am)` with Complex{Arb} fields.

`nmax_cf` is the Lentz iteration cap for the CF anchors (matching
`compute_fn`'s own default of 2000 — NOT the generic core's `nmax_cf=150`,
which only feeds the ν solver there).
"""
function compute_mst_core_acb(s::Int, l::Int, m::Int, a, ω;
                              ν=nothing, nmax::Int=80, nmax_cf::Int=2000,
                              precision::Int=256, ν_init=nothing)
    setprecision(Arb, precision) do
        ωc = complex(ω)
        νv, p = if ν === nothing
            compute_nu(s, l, m, a, ωc; backend=:acb, precision=precision,
                       ν_init = ν_init === nothing ? nothing : complex(ν_init))
        else
            pp = MSTParams(s, l, m, Arb(real(a)),
                           Complex{Arb}(Arb(real(ωc)), Arb(imag(ωc))))
            (Complex{Arb}(Arb(real(ν)), Arb(imag(ν))), pp)
        end
        # Internal dense-Acb path: fₙ as a Vector{Acb} feeds the A± kernels
        # directly (no Dict → Acb round-trip); the public Dict is built once
        # for the returned NamedTuple (fn::Dict is the Rup/qtilde contract).
        prec = precision   # the working precision (kwarg) = precision(Arb) here

        # Converge-or-error A± window (same doctrine as the generic
        # compute_Aplus/compute_Aminus): the fixed ±nmax window silently
        # truncated the weighted sums on the positive imaginary axis
        # (measured 1.1e-33 at ω=4.3i up to 2.6e-20 at ω=16i at the
        # `suggest_mst_precision` settings — this error enters every Rup/q̃
        # through A±/Ctrans).  Grow the dense fₙ vector until the last two
        # weighted terms of BOTH legs of BOTH sums sit 2^-tailbits below the
        # sums (tailbits ≈ the tol=100·eps criterion), then certify the
        # cancellation floor at √eps like _certify_mst_sum.
        tailbits = prec - 7
        nmax_hard = 50_000
        cur = nmax
        local fv, ApA, AmA
        while true
            fv = _compute_fn_acb_vec(p, νv; nmax=cur, nmax_cf=nmax_cf)
            off = cur + 1
            ApA, okp, cp = _Aplus_acb(p, νv, fv, off, -cur, cur, prec;
                                      tailbits=tailbits)
            AmA, okm, cm = _Aminus_acb(p, νv, fv, off, -cur, cur, prec;
                                       tailbits=tailbits)
            if okp && okm
                cmax = max(cp, cm)
                if !(isfinite(cmax) && log2(cmax) <= prec / 2)
                    error("compute_mst_core_acb: A± converged termwise but " *
                          "the cancellation floor eps·max|partial| exceeds " *
                          "√eps·|A±| (max|partial|/|Σ| ≈ $(cmax)); raise the " *
                          "working precision.")
                end
                break
            end
            cur >= nmax_hard && error("compute_mst_core_acb: A± window not " *
                                      "converged by nmax = $nmax_hard; " *
                                      "increase precision or nmax_hard.")
            cur = min(2 * cur, nmax_hard)
        end
        fn = _fn_dict_from_vec(fv, cur)
        Ap = Complex{Arb}(ApA)
        Am = Complex{Arb}(AmA)
        return (p=p, ν=νv, fn=fn, Ap=Ap, Am=Am)
    end
end

# ── Type-driven fast path ────────────────────────────────────────────────────
# `compute_mst_core` on Arb inputs routes to the native in-place chain: values
# are equivalent to the generic Complex{Arb} path at working precision (fn/A±
# to ~1e-88; end-to-end q̃·R^up bit-identical in ComplexF64), measured 2.8–7.6×
# faster end-to-end at σ = 4.3–10.  This is what makes the precision predictor's
# `backend = :acb` recommendation directly consumable by type-generic drivers:
# `setprecision(Arb, bits) do ... compute_mst_core(s,l,m,Arb(a),Complex{Arb}(ω))`.
# Options the native core does not support fall back to the generic method.
function compute_mst_core(s::Int, l::Int, m::Int, a::Arb, ω::Complex{Arb};
                          nmax::Int=80, nmax_cf::Int=150, ν_init=nothing,
                          method::String="Monodromy", fn_tol::Real=-1)
    if method != "Monodromy" || fn_tol > 0
        return invoke(compute_mst_core, Tuple{Int,Int,Int,Any,Any}, s, l, m, a, ω;
                      nmax=nmax, nmax_cf=nmax_cf, ν_init=ν_init,
                      method=method, fn_tol=fn_tol)
    end
    return compute_mst_core_acb(s, l, m, a, ω; nmax=nmax,
                                precision=precision(Arb), ν_init=ν_init)
end
