# ============================================================
#  Derived quantities
# ============================================================

struct MSTParams{R<:AbstractFloat}
    s::Int
    l::Int
    m::Int
    a::R
    ω::Complex{R}
    # derived
    ϵ::Complex{R}
    κ::Complex{R}
    q::R
    τ::Complex{R}
    ϵp::Complex{R}
    ϵm::Complex{R}
    λ::Complex{R}
    rp::R
    rm::R
end

# ============================================================
#  Angular eigenvalue via spherical-spheroidal decomposition matrix
#
#  Following Appendix of Berti, Cardoso, Starinets (2009) arXiv:0905.2975
#  and the qnm Python package (Stein 2019).
#
#  The spin-weighted spheroidal harmonic equation has the eigenvalue A_lm
#  (angular separation constant = λ + s(s+1) in our convention).
#  It is solved by diagonalizing the pentadiagonal matrix M whose
#  elements couple modes l, l±1, l±2 via angular momentum algebra.
# ============================================================

# Coupling coefficients, evaluated in the working real type R (algebraic
# √(rational) numbers — kept generic so the high-precision matrix is exact to eps(R)).
function _calF(R::Type, s::Int, l::Int, m::Int)
    (s == 0 && l + 1 == 0) && return zero(R)
    return sqrt(R((l+1)^2 - m^2) / (R(2l+3) * R(2l+1))) *
           sqrt(R((l+1)^2 - s^2) / R((l+1)^2))
end

function _calG(R::Type, s::Int, l::Int, m::Int)
    l == 0 && return zero(R)
    return sqrt(R(l^2 - m^2) / R(4l^2 - 1)) * sqrt(one(R) - R(s^2)/R(l^2))
end

function _calH(R::Type, s::Int, l::Int, m::Int)
    (l == 0 || s == 0) && return zero(R)
    return R(-m*s) / R(l*(l+1))
end

_calA(R, s, l, m) = _calF(R, s, l, m) * _calF(R, s, l+1, m)
_calB(R, s, l, m) = _calF(R, s, l, m)*_calG(R, s, l+1, m) + _calG(R, s, l, m)*_calF(R, s, l-1, m) + _calH(R, s, l, m)^2
_calC(R, s, l, m) = _calG(R, s, l, m) * _calG(R, s, l-1, m)
_calD(R, s, l, m) = _calF(R, s, l, m) * (_calH(R, s, l+1, m) + _calH(R, s, l, m))
_calE(R, s, l, m) = _calG(R, s, l, m) * (_calH(R, s, l-1, m) + _calH(R, s, l, m))

function M_matrix_elem(s::Int, c, m::Int, l::Int, lprime::Int)
    R  = real(typeof(c))
    A0 = lprime*(lprime+1) - s*(s+1)   # A_lm at c=0 (diagonal shift)
    if     lprime == l - 2; return -c^2 * _calA(R, s, lprime, m)
    elseif lprime == l - 1; return -c^2 * _calD(R, s, lprime, m) + 2c*s*_calF(R, s, lprime, m)
    elseif lprime == l    ; return A0   - c^2 * _calB(R, s, lprime, m) + 2c*s*_calH(R, s, lprime, m)
    elseif lprime == l + 1; return -c^2 * _calE(R, s, lprime, m) + 2c*s*_calG(R, s, lprime, m)
    elseif lprime == l + 2; return -c^2 * _calC(R, s, lprime, m)
    else;                    return zero(c)
    end
end

"""
    _rayleigh_refine(M, μ, v; maxiter=8)

Rayleigh-quotient iteration: refine an eigenpair (μ, v) of `M` from a Float64
seed to the working precision of `M` (`Complex{R}`). Converges cubically, so a
few steps suffice; the inverse solve becomes intentionally ill-conditioned as μ
nears the eigenvalue (the error lies along v and is removed by normalization).
"""
function _rayleigh_refine(M::AbstractMatrix{Complex{R}}, μ::Complex{R},
                          v::AbstractVector{Complex{R}}; maxiter::Int=8) where R
    Id = Matrix{Complex{R}}(I, size(M))
    v  = v / norm(v)
    for _ in 1:maxiter
        w = (M - μ*Id) \ v          # generic LU in Complex{R}
        (any(!isfinite, w)) && break  # μ hit the eigenvalue exactly — already converged
        v = w / norm(w)
        μ_new = (v' * (M * v))       # Rayleigh quotient (v is unit-norm)
        Δ = abs(μ_new - μ); μ = μ_new
        Δ ≤ 4*eps(R)*abs(μ) && break
    end
    return μ, v
end

# ── Acb-native Rayleigh refinement (Arb working precision) ───────────────────
# Strictly MORE SPECIFIC than the generic _rayleigh_refine above (Complex{Arb} is
# a concrete instantiation of Complex{R}), so it is auto-selected whenever the
# spectral matrix is Complex{Arb} — i.e. for BOTH the :arb (M1) and :acb (M2)
# backends.  The Complex{BigFloat}/Float64 matrices keep hitting the generic
# method, so those paths stay byte-for-byte unchanged.  Two reasons it exists:
#   (1) ROBUSTNESS — generic partial-pivot LU on Complex{Arb} BALLS makes
#       undecidable midpoint/ball pivot comparisons and returns NaN at high ω
#       (e.g. ω≥1.5: `matrix contains Infs or NaNs` / 0+NaN·im).  Arblib's
#       approx_solve! pivots on ball MIDPOINTS (a definite Float compare), so it
#       never NaNs.  Inverse iteration needs only the solution DIRECTION (the
#       error along v is removed by the next normalization), so discarding ball
#       radii is exactly right here.
#   (2) SPEED — the O(N³) inverse solve runs through Arblib's AcbMatrix LU
#       instead of generic allocating Complex{Arb} `\`.
function _rayleigh_refine(M::AbstractMatrix{Complex{Arb}}, μ::Complex{Arb},
                          v::AbstractVector{Complex{Arb}}; maxiter::Int=8)
    prec = precision(Arb)
    N    = size(M, 1)
    # M as an AcbMatrix, built once (M itself is iteration-invariant).
    Amat = Arblib.AcbMatrix(N, N; prec=prec)
    for i in 1:N, j in 1:N
        Amat[i, j] = Acb(M[i, j]; prec=prec)
    end
    Ashift = Arblib.AcbMatrix(N, N; prec=prec)
    bvec   = Arblib.AcbMatrix(N, 1; prec=prec)
    xvec   = Arblib.AcbMatrix(N, 1; prec=prec)
    Mv     = Arblib.AcbMatrix(N, 1; prec=prec)

    v = v / norm(v)
    for _ in 1:maxiter
        # Ashift = M - μ I
        for i in 1:N, j in 1:N
            Ashift[i, j] = Amat[i, j]
        end
        μacb = Acb(μ; prec=prec)
        for i in 1:N
            Ashift[i, i] = Ashift[i, i] - μacb
        end
        for i in 1:N
            bvec[i, 1] = Acb(v[i]; prec=prec)
        end
        # approx_solve! returns nonzero on success, 0 when (M-μI) is singular
        # (μ hit the eigenvalue exactly → already converged).
        flag = Arblib.approx_solve!(xvec, Ashift, bvec; prec=prec)
        w = Complex{Arb}[Complex{Arb}(xvec[i, 1]) for i in 1:N]
        (flag == 0 || any(!isfinite, w)) && break
        v = w / norm(w)
        # Rayleigh quotient μ_new = v'(Mv); Mv via the Acb matrix (no generic
        # Complex{Arb} matmul allocation).
        for i in 1:N
            bvec[i, 1] = Acb(v[i]; prec=prec)
        end
        Arblib.mul!(Mv, Amat, bvec; prec=prec)
        μ_new = zero(Complex{Arb})
        for i in 1:N
            μ_new += conj(v[i]) * Complex{Arb}(Mv[i, 1])
        end
        Δ = abs(μ_new - μ); μ = μ_new
        Δ ≤ 4*eps(Arb)*abs(μ) && break
    end
    return μ, v
end

# ── Adaptive angular basis size (Issues A4/A5) ───────────────────────────────
# Truncating the spherical-spheroidal ℓ′ basis at l_max leaves a super-
# exponentially small error in λ.  Measured decay (calibrated at prec ∈
# {53, 256, 512} bits and |c| ∈ {1, 3.5, 7, 10, 14}, worst case l = 2; see
# test/test_lmax_adequacy.jl) follows the perturbative estimate
#     err(k) ≈ (e·|c| / 4k)^{2k}      for k = l_max − l − ⌈|c|⌉ buffer levels.
# `_swsh_lmax_margin` picks the smallest k with
#     2k · ln(4k / (e·max(|c|, 0.5))) ≥ 1.2·prec·ln2 + 16,
# which exceeds the measured need at every calibration point (by 4–14 levels,
# i.e. many orders of magnitude of the super-exponential decay).
function _swsh_lmax_margin(prec::Int, cabs::Real)
    c      = max(Float64(cabs), 0.5)
    target = 1.2 * prec * log(2.0) + 16.0
    k      = 8
    while 2k * log(max(4k / (ℯ * c), 1.25)) < target
        k += 2
        k > 100_000 && error("_swsh_lmax_margin: no adequate angular basis " *
                             "margin found (|c| = $cabs, prec = $prec bits)")
    end
    return k
end

# Effective l_max: at least the user request, at least the legacy default (20),
# and always large enough that the λ truncation floor sits below 2^-prec of the
# working type.  `l_max ≤ 0` (the default) means "automatic only".
function _swsh_lmax_auto(R::Type, l::Int, cabs::Real, l_max::Int)
    # NB: not plain `round(Int, -log2(Float64(eps(R))))` — Float64(eps(R))
    # underflows to 0.0 above 1074 bits (Arb/BigFloat), turning prec into
    # round(Int, Inf) and crashing every MSTParams construction at the
    # 1280/1536-bit ladder rungs `suggest_mst_precision` picks for |ω| ≳ 15.
    e = Float64(eps(R))
    prec = e > 0 ? round(Int, -log2(e)) : precision(R)
    auto = l + ceil(Int, Float64(cabs)) + _swsh_lmax_margin(prec, cabs)
    return max(l_max, auto, 20)
end

"""
    _swsh_eigen(s, l, m, a, ω; l_max=0) -> (λ, ells, C)

Solve the spherical-spheroidal coupling eigenproblem (BCS 2009, arXiv:0905.2975)
and return the spin-weighted spheroidal eigenvalue `λ` (= A_lm, the angular
separation constant), the ℓ′ range `ells`, and the spherical-harmonic expansion
coefficients `C` (unit 2-norm, phase-fixed so the ℓ′=ℓ component is real positive,
giving S_lm → ₛY_lm as aω→0). The spheroidal harmonic is then

    S_lm(θ,φ) = Σ_{ℓ′} C[ℓ′] · ₛY_{ℓ′m}(θ,φ).

`l_max ≤ 0` (default) sizes the ℓ′ basis automatically so the truncation error
in λ sits below the working precision (calibrated; see `_swsh_lmax_margin`).
An explicit `l_max > 0` acts as a lower bound on the basis and is widened when
inadequate, so `l` can never fall on or beyond the basis edge.

At higher precision the Float64 LAPACK eigenpair is refined to working precision
by Rayleigh-quotient iteration.
"""
function _swsh_eigen(s::Int, l::Int, m::Int, a, ω; l_max::Int=0)
    R = promote_type(typeof(float(real(a))), typeof(float(real(complex(ω)))))
    c = R(a) * Complex{R}(complex(ω))

    l_min = max(abs(m), abs(s))
    l ≥ l_min || throw(ArgumentError("_swsh_eigen: need l ≥ max(|m|, |s|) = " *
                                     "$l_min (got s=$s, l=$l, m=$m)"))
    l_max_eff = _swsh_lmax_auto(R, l, abs(c), l_max)
    # Unreachable by construction (auto ≥ l + margin ≥ l + 8); kept as a hard
    # guard so a future regression fails loudly instead of returning a silently
    # truncated λ (the old l == l_max zero-buffer bug) or a raw BoundsError.
    l_max_eff ≥ l + 4 && l_max_eff ≥ l_min + 1 ||
        error("_swsh_eigen: angular basis l_max=$l_max_eff cannot resolve " *
              "l=$l (s=$s, m=$m, |aω|=$(Float64(abs(c)))) — auto-widening failed")
    ells  = l_min:l_max_eff
    il    = l - l_min + 1          # index of the ℓ′=ℓ (dominant) component
    N     = length(ells)

    # Float64 LAPACK eigendecomposition → seed for branch selection / refinement.
    c64 = ComplexF64(c)

    # Eigenvalues of M are SWSHEigenvalueSpectral.
    # SpinWeightedSpheroidalEigenvalue = SWSHEigenvalueSpectral - 2m*c + c²
    #
    # BRANCH SELECTION: analytic continuation from c = 0 along the ray t·c,
    # matching eigenVECTORS by overlap at each step.  At c = 0 the matrix is
    # diagonal, so the l-branch is exactly the unit vector e_il; each step picks
    # the eigencolumn maximizing |⟨v_prev, v⟩| and carries it forward.  This is
    # the defining label of the spheroidal harmonic (the branch continuously
    # connected to ₛY_lm) and is exact along PIA/real-ω sweeps, whose paths are
    # precisely this ray.
    #
    # The previous scheme — argmin distance to the O(c²) perturbative λ — is
    # DEGENERATE at branch crossings far from c = 0: at a=0.7, l=m=2, ω=iσ near
    # σ ≈ 4.1885 two well-separated eigenvalues (Δλ ≈ 1.16) sit at |λ−λ_pert| =
    # 2.802 vs 2.810 and the argmin flips, jumping λ (and hence ν by ≈0.06)
    # discontinuously across the sweep.
    F, idx = let
        v_prev = zeros(ComplexF64, N); v_prev[il] = 1     # e_il = exact c→0 branch
        Floc = nothing; k = 0
        h    = 0.25                                        # max |Δc| per step
        hmin = h / 64
        t    = 0.0
        warned = false
        while t < 1.0
            t2 = min(t + h/max(abs(c64), eps()), 1.0)
            Mt = [M_matrix_elem(s, t2*c64, m, li, lj) for li in ells, lj in ells]
            Ft = eigen(Mt)
            ovl = [abs(dot(v_prev, view(Ft.vectors, :, j))) for j in 1:N]
            k2  = argmax(ovl)
            if ovl[k2] < 0.75
                if (t2 - t)*abs(c64) > hmin
                    h /= 2                                 # ambiguous match → refine step
                    continue
                elseif !warned
                    # Step is at the hmin floor (or the final clamped step): the
                    # ambiguous match is accepted, as before — but no longer
                    # silently, so near-degenerate λ crossings are diagnosable.
                    @warn "_swsh_eigen: ambiguous eigenvector match accepted at " *
                          "the minimum continuation step (near-degenerate λ " *
                          "crossing?)" overlap=ovl[k2] s=s l=l m=m aω=c64 t=t2
                    warned = true
                end
            end
            v_prev = Ft.vectors[:, k2]
            Floc, k = Ft, k2
            t = t2
            ovl[k2] > 0.95 && (h = min(2h, 0.25))          # relax step when clean
        end
        Floc === nothing ? (eigen([M_matrix_elem(s, c64, m, li, lj)
                                   for li in ells, lj in ells]), il) : (Floc, k)
    end
    λ_vals = F.values .- 2*m*c64 .+ c64^2

    # phase-fix to (ℓ′=ℓ component real positive) and unit norm
    fixphase(v) = (w = v / norm(v); phref = w[il]; iszero(phref) ? w : w * (conj(phref)/abs(phref)))

    if R === Float64 || iszero(c)
        # Float64 path / a=0 (diagonal matrix): no refinement needed.
        return Complex{R}(λ_vals[idx]), ells, fixphase(Complex{R}.(F.vectors[:, idx]))
    end

    # Higher precision: refine the selected spectral eigenpair on the Complex{R}
    # matrix via Rayleigh-quotient iteration (Float64 LAPACK gives only ~1e-15).
    MR = Matrix{Complex{R}}(undef, N, N)
    for i in 1:N, j in 1:N
        MR[i, j] = M_matrix_elem(s, c, m, ells[i], ells[j])
    end
    μ, v = _rayleigh_refine(MR, Complex{R}(F.values[idx]), Complex{R}.(F.vectors[:, idx]))
    return μ - 2*m*c + c^2, ells, fixphase(v)
end

"""
    compute_lambda(s, l, m, a, ω; l_max=0)

Spin-weighted spheroidal eigenvalue λ = A_lm (angular separation constant), to
working precision (BCS 2009 spectral matrix + Rayleigh-quotient refinement).
`l_max ≤ 0` (default) sizes the ℓ′ basis adaptively so the truncation error
sits below the working precision; an explicit `l_max > 0` is a lower bound on
the basis (widened when inadequate).
"""
compute_lambda(s::Int, l::Int, m::Int, a, ω; l_max::Int=0) =
    _swsh_eigen(s, l, m, a, ω; l_max=l_max)[1]

# ============================================================
#  MSTParams constructor
# ============================================================

"""
    MSTParams(s, l, m, a, ω; solve_lambda=true, l_max=0)

Construct MST parameters. By default, solves the angular eigenvalue λ
self-consistently via matrix diagonalization (recommended for |aω| > 0.1).
Set `solve_lambda=false` to use the O((aω)²) perturbative approximation.
`l_max ≤ 0` (default) sizes the angular basis adaptively (see
[`compute_lambda`](@ref)); an explicit `l_max > 0` is a lower bound.
"""
function MSTParams(s::Int, l::Int, m::Int, a, ω;
                   solve_lambda::Bool=true, l_max::Int=0)
    R   = promote_type(typeof(float(real(a))), typeof(float(real(complex(ω)))))
    a_r = R(a)
    ω_c = Complex{R}(complex(ω))
    q   = a_r
    κ   = sqrt(Complex{R}(1 - q^2))
    ϵ   = 2 * ω_c
    τ   = (ϵ - m*q) / κ
    ϵp  = (ϵ + τ) / 2
    ϵm  = (ϵ - τ) / 2
    rp  = R(1) + sqrt(R(1) - a_r^2)
    rm  = R(1) - sqrt(R(1) - a_r^2)

    if solve_lambda
        λ_val = compute_lambda(s, l, m, a_r, ω_c; l_max=l_max)
    else
        # O((aω)²) perturbative approximation (only valid for small |aω|)
        aω   = a_r * ω_c
        λ0   = R(l*(l+1) - s*(s+1))
        λ1   = R(-2m * (1 + s^2 / (l*(l+1))))
        H(ℓ) = R(2*(ℓ^2 - m^2) * (ℓ^2 - s^2)) / R((2ℓ-1) * ℓ^3 * (2ℓ+1))
        λ2   = l > 0 ? H(l+1) - H(l) : zero(R)
        λ_val = λ0 + aω * λ1 + aω^2 * λ2
    end

    MSTParams{R}(s, l, m, a_r, ω_c, ϵ, κ, q, τ, ϵp, ϵm, λ_val, rp, rm)
end
