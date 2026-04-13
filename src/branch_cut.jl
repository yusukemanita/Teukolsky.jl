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
                   nmax::Int=80, nmax_cf::Int=150, ν_init=nothing)
    ν, p = compute_nu(s, l, m, a, ω; nmax_cf=nmax_cf, ν_init=ν_init)
    fn   = compute_fn(p, ν; nmax=nmax)

    Ap = compute_Aplus(p, ν, fn; nmax=nmax)
    Am = compute_Aminus(p, ν, fn; nmax=nmax)

    ε   = p.ϵ    # = 2ω (M=1)
    κ   = p.κ
    ω_c = p.ω

    # ε^{-2iε} = exp(-2iε log ε),  principal branch of log
    branch    = exp(-2im * ε * log(ε))

    # e^{iε(1-κ)}
    phase     = exp(im * ε * (1 - κ))

    # 1 - e^{2π(ε - iν)}
    monodromy = 1 - exp(2π * (ε - im*ν))

    q_val = im * (Ap / Am) * ω_c^(2s) * branch * phase * monodromy

    return (q=q_val, ν=ν, p=p, Ap=Ap, Am=Am)
end
