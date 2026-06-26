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

"""
    compute_lambda(s, l, m, a, ω; l_max=20)

Compute the angular eigenvalue λ = A_lm - s(s+1) by diagonalizing the
spherical-spheroidal decomposition matrix (BCS 2009, arXiv:0905.2975).
Selects the eigenvalue closest to the l(l+1)-s(s+1) guess.
Works for any precision (Float64, BigFloat).
"""
function compute_lambda(s::Int, l::Int, m::Int, a, ω; l_max::Int=20)
    R = promote_type(typeof(float(real(a))), typeof(float(real(complex(ω)))))
    c = R(a) * Complex{R}(complex(ω))

    # Build ℓ range: all ℓ ≥ max(|m|,|s|) up to l_max
    l_min = max(abs(m), abs(s))
    ells  = l_min:l_max

    N = length(ells)
    # Float64 LAPACK eigendecomposition → seed for branch selection (and, at
    # higher precision, for Rayleigh-quotient refinement).
    c64 = ComplexF64(c)
    M64 = zeros(ComplexF64, N, N)
    for (i, li) in enumerate(ells), (j, lj) in enumerate(ells)
        M64[i, j] = M_matrix_elem(s, c64, m, li, lj)
    end

    # Eigenvalues of M are SWSHEigenvalueSpectral.
    # SpinWeightedSpheroidalEigenvalue = SWSHEigenvalueSpectral - 2m*c + c²
    # (same correction as in the Mathematica SpinWeightedSpheroidalHarmonics package)
    #
    # Selection strategy: compare corrected eigenvalues (= λ) to a perturbative
    # estimate λ_pert(c).  Using a fixed reference λ₀ = l(l+1)-s(s+1) fails at
    # large |c| because the true λ can be far from λ₀ and a different branch
    # gets selected.  The perturbative estimate tracks the correct branch much
    # better for moderate |c|.
    #
    # Perturbative λ to O(c²):
    #   λ ≈ λ₀ + c·λ₁ + c²·λ₂
    #   λ₁ = -2m(1 + s²/(l(l+1)))
    #   λ₂ = H(l+1) - H(l),   H(ℓ) = 2(ℓ²-m²)(ℓ²-s²)/((2ℓ-1)ℓ³(2ℓ+1))
    λ₀ = l*(l+1) - s*(s+1)
    λ₁ = l > 0 ? -2*m*(1 + s^2 / (l*(l+1))) : zero(Float64)
    H(ℓ) = ℓ == 0 ? 0.0 : 2*(ℓ^2 - m^2)*(ℓ^2 - s^2) / ((2ℓ-1) * ℓ^3 * (2ℓ+1))
    λ₂ = l > 0 ? H(l+1) - H(l) : zero(Float64)
    λ_pert = ComplexF64(λ₀ + c64*λ₁ + c64^2*λ₂)

    F      = eigen(M64)
    λ_vals = F.values .- 2*m*c64 .+ c64^2   # SpinWeightedSpheroidalEigenvalue
    idx    = argmin(abs.(λ_vals .- λ_pert))

    # Float64 path (or a=0, where c=0 makes M diagonal and λ exact): no refinement.
    if R === Float64 || iszero(c)
        return Complex{R}(λ_vals[idx])
    end

    # Higher precision: refine the selected spectral eigenpair on the Complex{R}
    # matrix via Rayleigh-quotient iteration (Float64 LAPACK gives only ~1e-15).
    MR = Matrix{Complex{R}}(undef, N, N)
    for i in 1:N, j in 1:N
        MR[i, j] = M_matrix_elem(s, c, m, ells[i], ells[j])
    end
    μ0 = Complex{R}(F.values[idx])
    v0 = Complex{R}.(F.vectors[:, idx])
    μ, _ = _rayleigh_refine(MR, μ0, v0)
    return μ - 2*m*c + c^2
end

# ============================================================
#  MSTParams constructor
# ============================================================

"""
    MSTParams(s, l, m, a, ω; solve_lambda=true, l_max=20)

Construct MST parameters. By default, solves the angular eigenvalue λ
self-consistently via matrix diagonalization (recommended for |aω| > 0.1).
Set `solve_lambda=false` to use the O((aω)²) perturbative approximation.
"""
function MSTParams(s::Int, l::Int, m::Int, a, ω;
                   solve_lambda::Bool=true, l_max::Int=20)
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
