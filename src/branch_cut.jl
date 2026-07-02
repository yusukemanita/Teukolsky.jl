# ============================================================
#  Branch-cut coefficient q(ω)
#
#  Defined by (MST notation):
#
#    q(ω) = i (A_+^ν / A_-^ν) ω^{2s} ε^{-2iε} e^{iε(1−κ)} (1 − e^{2π(ε−iν)})
#
#  where
#    ε = p.ϵ = 2Mω  (M = 1)
#    κ = p.κ = √(1 − a²)
#    A_+^ν, A_-^ν = MST normalization sums (Sasaki-Tagoshi eqs. 157-158)
#    ν = renormalized angular momentum
#
#  q(ω) appears in the spectral decomposition of the retarded Green's function:
#  the branch-cut contribution along the negative imaginary ω-axis is
#  proportional to q(ω).
# ============================================================

"""
    compute_q(s, l, m, a, ω; nmax=80, nmax_cf=150)

Compute the branch-cut coefficient q(ω) for the MST Teukolsky solution.

# Formula

    q(ω) = i (A_+^ν / A_-^ν) ω^{2s} ε^{-2iε} e^{iε(1−κ)} (1 − e^{2π(ε−iν)})

where ε = 2Mω (M=1), κ = √(1−a²), and A_±^ν are the MST normalization
sums (Sasaki-Tagoshi 2003, eqs. 157-158).

# Returns

Named tuple `(q, ν, p, Ap, Am)`:
- `q`: complex branch-cut coefficient
- `ν`: renormalized angular momentum
- `p`: MSTParams struct
- `Ap`: A_+^ν normalization sum
- `Am`: A_-^ν normalization sum
"""
function compute_q(s::Int, l::Int, m::Int, a, ω;
                   nmax::Int=80, nmax_cf::Int=150, ν_init=nothing, method::String="Monodromy",
                   fn_tol::Real=-1)
    core = compute_mst_core(s, l, m, a, ω; nmax=nmax, nmax_cf=nmax_cf,
                            ν_init=ν_init, method=method, fn_tol=fn_tol)
    return (q=q_from_core(core), ν=core.ν, p=core.p, Ap=core.Ap, Am=core.Am)
end

"""
    q_from_core(core) -> q

The branch-cut coefficient q(ω) evaluated from a [`compute_mst_core`](@ref) result
(no re-solve).  Identical value to `compute_q(...).q`.
"""
function q_from_core(core)
    p = core.p
    s = p.s
    ε, κ, ω_c = p.ϵ, p.κ, p.ω    # ε = 2ω (M=1)
    branch    = exp(-2im * ε * log(ε))     # ε^{-2iε}, principal branch of log
    phase     = exp(im * ε * (1 - κ))      # e^{iε(1-κ)}
    twoπ      = 2 * real(typeof(ε))(π)    # 2π at working precision (2π is Float64)
    monodromy = 1 - exp(twoπ * (ε - im*core.ν))  # 1 - e^{2π(ε-iν)}
    return im * (core.Ap / core.Am) * ω_c^(2s) * branch * phase * monodromy
end

# ============================================================
#  Branch-cut coefficient q̃(ω)  (for R^down / Rdown)
#
#  Defined by (branchcut_note.tex, eq. qtilde):
#
#    q̃(ω) = +i (A_-^ν / A_+^ν) ω^{-2s} ε^{+2iε} e^{-iε(1−κ)} (1 − e^{2π(ε+iν)})
#
#  q̃ is the branch-cut strength for R^down, the analogue of q for R^up.
#  The sign differences vs. q: Am/Ap, ω^{-2s}, ε^{+2iε}, e^{-iε(1-κ)},
#  and ν→-ν in the monodromy factor.
# ============================================================

"""
    compute_mst_core(s, l, m, a, ω; nmax=80, nmax_cf=150, ν_init=nothing, method="Monodromy")

Shared MST core solve.  Returns the named tuple `(p, ν, fn, Ap, Am)` — the pieces
common to `compute_qtilde` and the radial solutions `Rup`/`Rin`.  Compute this ONCE
per `(s,l,m,a,ω)` and feed it to both, instead of each independently re-solving
`ν` + `fn` + `A±` (as `compute_qtilde` and `TeukolskyRadial` do separately).

Use with [`qtilde_from_core`](@ref), [`mst_ctrans`](@ref), and `Rup(core.p, core.ν, core.fn, r; ctrans=...)`.
"""
function compute_mst_core(s::Int, l::Int, m::Int, a, ω;
                          nmax::Int=80, nmax_cf::Int=150, ν_init=nothing, method::String="Monodromy",
                          fn_tol::Real=-1)
    ν, p = compute_nu(s, l, m, a, ω; nmax_cf=nmax_cf, ν_init=ν_init, method=method)
    fn   = compute_fn(p, ν; nmax=nmax, tol=fn_tol)
    Ap   = compute_Aplus(p, ν, fn; nmax=nmax)
    Am   = compute_Aminus(p, ν, fn; nmax=nmax)
    return (p=p, ν=ν, fn=fn, Ap=Ap, Am=Am)
end

"""
    qtilde_from_core(core) -> q̃

The branch-cut coefficient q̃(ω) evaluated from a [`compute_mst_core`](@ref) result
(no re-solve).  Identical value to `compute_qtilde(...).qtilde`.  The spin `s` is
taken from `core.p.s`, so it cannot disagree with the solve that produced `core`.
"""
function qtilde_from_core(core)
    p = core.p
    s = p.s
    ε, κ, ω_c = p.ϵ, p.κ, p.ω
    branch    = exp(2im * ε * log(ε))      # ε^{+2iε}  (sign flipped vs q)
    phase     = exp(-im * ε * (1 - κ))    # e^{-iε(1-κ)} (sign flipped)
    twoπ      = 2 * real(typeof(ε))(π)    # 2π at working precision (2π is Float64)
    monodromy = 1 - exp(twoπ * (ε + im*core.ν))  # 1 - e^{2π(ε+iν)} (ν sign flipped)
    return im * (core.Am / core.Ap) * ω_c^(-2s) * branch * phase * monodromy
end

"""
    mst_ctrans(core) -> Ctrans

The `Rup` normalization constant `Ctrans = ω^{-1-2s} A^ν_- e^{i(ε logε − (1−κ)/2 ε)}`
built from a shared [`compute_mst_core`](@ref) (reuses `core.Am`).  Pass as
`Rup(core.p, core.ν, core.fn, r; ctrans=mst_ctrans(core))` so `Rup` does not
recompute `A^ν_-`.  The spin `s` is taken from `core.p.s`.
"""
mst_ctrans(core) = _ctrans(core.p, core.Am)

"""
    compute_qtilde(s, l, m, a, ω; nmax=80, nmax_cf=150)

Compute the branch-cut coefficient q̃(ω) for the R^down MST solution.

# Formula

    q̃(ω) = +i (A_-^ν / A_+^ν) ω^{-2s} ε^{+2iε} e^{-iε(1−κ)} (1 − e^{2π(ε+iν)})

where ε = 2Mω (M=1), κ = √(1−a²).  Compare with q (for R^up):

    q(ω)  = +i (A_+^ν / A_-^ν) ω^{+2s} ε^{-2iε} e^{+iε(1−κ)} (1 − e^{2π(ε−iν)})

# Returns

Named tuple `(qtilde, ν, p, Ap, Am)`.
"""
function compute_qtilde(s::Int, l::Int, m::Int, a, ω;
                        nmax::Int=80, nmax_cf::Int=150, ν_init=nothing, method::String="Monodromy")
    core = compute_mst_core(s, l, m, a, ω; nmax=nmax, nmax_cf=nmax_cf, ν_init=ν_init, method=method)
    return (qtilde=qtilde_from_core(core), ν=core.ν, p=core.p, Ap=core.Ap, Am=core.Am)
end
