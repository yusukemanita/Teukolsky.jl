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

function _calF(s::Int, l::Int, m::Int)
    (s == 0 && l + 1 == 0) && return 0.0
    return sqrt(((l+1)^2 - m^2) / (2l+3) / (2l+1)) *
           sqrt(((l+1)^2 - s^2) / (l+1)^2)
end

function _calG(s::Int, l::Int, m::Int)
    l == 0 && return 0.0
    return sqrt((l^2 - m^2) / (4l^2 - 1)) * sqrt(1 - s^2/l^2)
end

function _calH(s::Int, l::Int, m::Int)
    (l == 0 || s == 0) && return 0.0
    return -m*s / (l*(l+1))
end

_calA(s, l, m) = _calF(s, l, m) * _calF(s, l+1, m)
_calB(s, l, m) = _calF(s, l, m)*_calG(s, l+1, m) + _calG(s, l, m)*_calF(s, l-1, m) + _calH(s, l, m)^2
_calC(s, l, m) = _calG(s, l, m) * _calG(s, l-1, m)
_calD(s, l, m) = _calF(s, l, m) * (_calH(s, l+1, m) + _calH(s, l, m))
_calE(s, l, m) = _calG(s, l, m) * (_calH(s, l-1, m) + _calH(s, l, m))

function M_matrix_elem(s::Int, c, m::Int, l::Int, lprime::Int)
    # A_lm at c=0 (diagonal shift)
    A0 = lprime*(lprime+1) - s*(s+1)
    if     lprime == l - 2; return -c^2 * _calA(s, lprime, m)
    elseif lprime == l - 1; return -c^2 * _calD(s, lprime, m) + 2c*s*_calF(s, lprime, m)
    elseif lprime == l    ; return A0   - c^2 * _calB(s, lprime, m) + 2c*s*_calH(s, lprime, m)
    elseif lprime == l + 1; return -c^2 * _calE(s, lprime, m) + 2c*s*_calG(s, lprime, m)
    elseif lprime == l + 2; return -c^2 * _calC(s, lprime, m)
    else;                    return zero(c)
    end
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
    # Matrix diagonalization uses Float64 (LAPACK); sufficient for initial guess
    c64 = ComplexF64(c)
    M   = zeros(ComplexF64, N, N)
    for (i, li) in enumerate(ells)
        for (j, lj) in enumerate(ells)
            M[i, j] = M_matrix_elem(s, c64, m, li, lj)
        end
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

    evals  = eigvals(M)
    λ_vals = evals .- 2*m*c64 .+ c64^2   # corrected eigenvalues = SpinWeightedSpheroidalEigenvalue
    idx    = argmin(abs.(λ_vals .- λ_pert))

    return Complex{R}(λ_vals[idx])
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
