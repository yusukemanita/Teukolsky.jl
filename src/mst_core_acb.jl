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
#  NOTE ON ν: the monodromy solver (all backends) has a known convergence
#  fragility for PURELY IMAGINARY ω = iσ — the branch-cut / PIA regime — where it
#  NaNs for many (σ, precision) combinations (real and generic-complex ω are
#  fine).  This is pre-existing and orthogonal to the native fₙ/A±/R^up kernels.
#  Supply `ν` directly (e.g. from a precision that converges, or a warm-started
#  solve) to bypass the solver and exercise the fully-native downstream core.
# ============================================================

"""
    compute_mst_core_acb(s, l, m, a, ω; ν=nothing, nmax=80, nmax_cf=150,
                         precision=256, ν_init=nothing)

Native-Acb analogue of [`compute_mst_core`](@ref): ν via the native-Acb monodromy
kernel (or the supplied `ν`), then fₙ and A^ν_± via the in-place Acb kernels.
Runs inside `setprecision(Arb, precision)`.  Returns `(p, ν, fn, Ap, Am)` with
Complex{Arb} fields.

Pass `ν` to skip the (PIA-fragile) monodromy solve; `p` is then built directly
from `(s,l,m,a,ω)`.
"""
function compute_mst_core_acb(s::Int, l::Int, m::Int, a, ω;
                              ν=nothing, nmax::Int=80, nmax_cf::Int=150,
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
        fn = compute_fn_acb(p, νv; nmax=nmax, nmax_cf=nmax_cf)
        Ap = compute_Aplus_acb(p, νv, fn; nmax=nmax)
        Am = compute_Aminus_acb(p, νv, fn; nmax=nmax)
        return (p=p, ν=νv, fn=fn, Ap=Ap, Am=Am)
    end
end
