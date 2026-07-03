# ============================================================
#  Rdown: Downgoing radial solution (at infinity, HypergeometricU-based)
#
#  Rdown = R^ν_+ / norm,  where
#    norm = A^ν_+ · ω^{-1} · exp(-i(ε ln ε - (1-κ)/2 · ε))
#
#  R^ν_+ is the MST Coulomb-wave expansion (Sasaki-Tagoshi eq. 159):
#
#  R^ν_+(r) = prefac(ẑ) × Σ_n i^n f^ν_n (2ẑ)^n Ψ(n+ν+1-s+iε, 2n+2ν+2; 2iẑ)
#
#  where ẑ = ε(r - r_-)/2
#
#  prefac = 2^ν e^{-πε} e^{iπ(ν+1-s)} Γ(ν+1-s+iε)/Γ(ν+1+s-iε)
#           × e^{-iẑ} ẑ^{ν+iε_+} (ẑ-εκ)^{-s-iε_+}
#
#  Compare with R^ν_- (= Rup, radial_up.jl):
#    - U argument: +2iẑ  (vs -2iẑ)
#    - U first param: ν+1-s+iε  (vs ν+1+s-iε)
#    - series coeff: i^n fn  (vs (-1)^n Poch/Poch fn)
#    - prefac sign of ẑ exponent: e^{-iẑ}  (vs e^{+iẑ})
# ============================================================

# Rdown normalization constant (single source of truth, mirroring `_ctrans`
# for Rup so the convention cannot drift between call sites):
#   Dtrans = ω^{-1} A^ν_+ exp(-i(ε logε − (1−κ)/2 ε))
_dtrans(p::MSTParams, Ap) =
    Ap * p.ω^(-1) * exp(-im * (p.ϵ * log(p.ϵ) - (1 - p.κ) / 2 * p.ϵ))

"""
    Rdown(p::MSTParams, ν, fn, r; nmax=80, tol=100·eps, nmax_hard=50_000,
          floor_tol=√eps)

Compute the downgoing radial Teukolsky solution at Boyer-Lindquist radius r.
Rdown = R^ν_+ / norm, normalized so that at infinity:

    Rdown ~ r^{-1} e^{-iωr*}

(pure ingoing wave at infinity with unit amplitude; the ingoing solution of
the spin-s Teukolsky equation falls off as r^{-1}, while it is the OUTgoing
solution that falls off as r^{-1-2s} — cf. Sasaki-Tagoshi Eq. (21)).

norm = A^ν_+ · ω^{-1} · exp(-i(ε ln ε - (1-κ)/2 · ε))

The HU[n] = (2iẑ)^n U(n+ν+1-s+iε, 2n+2ν+2, 2iẑ) values are produced by the
same certified evaluator machinery as `Rup` (`_hu_dhu_evaluators`: certified
escalated seeds + stable outward march for the Arb/BigFloat backends, legacy
exact-seeded recurrence with a decidable guard otherwise), and the series is
summed converge-or-error (adaptive fn extension, hard error past
`nmax_hard`) exactly like `Rin`.
"""
function Rdown(p::MSTParams, ν, fn, r; nmax::Int=80, tol::Real=100*eps(real(typeof(p.ϵ))),
               nmax_hard::Int=50_000,
               floor_tol::Real=_default_floor_tol(real(typeof(p.ϵ))))
    ϵ, κ, τ, s = p.ϵ, p.κ, p.τ, p.s
    rm = p.rm
    zhat = complex(ϵ * (r - rm) / 2)

    # HUParams for R^ν_+: aU = ν+1-s+iε, bU = 2ν+2, c = +2iẑ
    hp = HUParams(ν + 1 - s + im*ϵ, 2ν + 2, 2im * zhat)

    ϵp = p.ϵp  # = (ε+τ)/2

    # Prefactor for R^ν_+  (πT: full-precision π — -π*ϵ / im*π*… would round
    # π through Float64/ComplexF64 and cap every high-precision Rdown at ~1e-16)
    πT = real(typeof(p.ϵ))(π)
    prefac = 2^ν * exp(-πT*ϵ) * exp(im*πT*(ν + 1 - s)) *
             _cgamma(complex(ν + 1 - s + im*ϵ)) / _cgamma(complex(ν + 1 + s - im*ϵ)) *
             exp(-im*zhat) * zhat^(ν + im*ϵp) *
             (zhat - ϵ*κ)^(-s - im*ϵp)

    # HU evaluator: certified seeds + stable outward march for the Arb/BigFloat
    # backends, legacy exact-seeded recurrence + decidable ratio guard
    # otherwise — the same wiring as Rup (see hypergeometric.jl, "Certified
    # HU / dHU evaluation").
    get_hu, _ = _hu_dhu_evaluators(hp)

    # Series coefficient: just fn
    # (the i^n from the image formula is already absorbed into
    #  hu_exact via c^n = (2iẑ)^n = i^n (2ẑ)^n)
    term(n::Int) = prefac * get_hu(n)

    # Sum bidirectionally, converge-or-error (see _sum_mst_series!).
    n_ext = max(2 * nmax, 64)
    result, smax_up = _sum_mst_series!(term, fn, p, ν, +1, tol, tol,
                                       n_ext, nmax_hard, "Rdown")
    res_down, smax_dn = _sum_mst_series!(term, fn, p, ν, -1, tol, tol,
                                         n_ext, nmax_hard, "Rdown")

    ctol = max(tol, floor_tol)
    Rnu_plus = _certify_mst_sum(result + res_down, max(smax_up, smax_dn),
                                ctol, ctol, "Rdown")

    # Normalization: Rdown = R^ν_+ / Dtrans (shared helper _dtrans above)
    Ap = compute_Aplus(p, ν, fn; nmax=nmax)
    return Rnu_plus / _dtrans(p, Ap)
end
