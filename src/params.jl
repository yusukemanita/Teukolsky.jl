# ============================================================
#  Derived quantities
# ============================================================

struct MSTParams
    s::Int          # spin weight
    l::Int          # angular mode number
    m::Int          # azimuthal mode number
    a::Float64      # Kerr spin parameter (0 ≤ a < 1)
    ω::ComplexF64   # frequency
    # derived
    ϵ::ComplexF64   # 2Mω
    κ::ComplexF64   # √(1 - q²)
    q::Float64      # a/M
    τ::ComplexF64   # (ϵ - mq)/κ
    ϵp::ComplexF64  # (ϵ + τ)/2
    ϵm::ComplexF64  # (ϵ - τ)/2
    λ::ComplexF64   # angular eigenvalue
    rp::Float64     # r+
    rm::Float64     # r-
end

function MSTParams(s::Int, l::Int, m::Int, a::Float64, ω)
    ω_c = complex(ω)
    q = a  # a/M with M=1
    κ = sqrt(1 - q^2)
    ϵ = 2ω_c  # 2Mω with M=1
    τ = (ϵ - m*q) / κ
    ϵp = (ϵ + τ) / 2
    ϵm = (ϵ - τ) / 2
    rp = 1 + sqrt(1 - a^2)
    rm = 1 - sqrt(1 - a^2)

    # Angular eigenvalue λ to O((aω)²), Eq. (110)-(112)
    aω = a * ω_c
    λ0 = l*(l+1) - s*(s+1)
    λ1 = -2m * (1 + s^2 / (l*(l+1)))
    H(ℓ) = 2(ℓ^2 - m^2) * (ℓ^2 - s^2) / (2ℓ - 1) / ℓ^3 / (2ℓ + 1)
    λ2 = l > 0 ? H(l+1) - H(l) : 0.0
    λ_val = λ0 + aω * λ1 + aω^2 * λ2

    MSTParams(s, l, m, a, ω_c, ϵ, κ, q, τ, ϵp, ϵm, λ_val, rp, rm)
end
