# ============================================================
#  Complex Gamma to full working precision
#
#  SpecialFunctions.gamma(::Complex{T}) is accurate to full precision for any
#  T<:AbstractFloat (Float64, BigFloat, …) — EXCEPT when imag(z)==0 exactly,
#  where it routes to the real-argument path that throws a DomainError for
#  x < 0.  _cgamma handles the real axis explicitly (reflection for negative
#  reals) and otherwise defers to gamma.
#
#  (The previous hand-rolled Stirling series floored the BigFloat result at
#  ~1e-20; SpecialFunctions.gamma reaches ~eps — verified to ~1e-78 at 256 bits.)
# ============================================================

"""
    _cgamma(z)

Complex Gamma function, accurate to full working precision for any float type
(Float64, BigFloat, …), including the negative real axis where the generic
`gamma` would throw a DomainError.
"""
function _cgamma(z::Complex{T}) where T<:AbstractFloat
    if iszero(imag(z))
        x = real(z)
        # gamma(::BigFloat) throws for x < 0; use Γ(x) = π / (sin(πx) Γ(1-x))
        # (1-x > 1 is handled by the real path).
        x < 0 && return Complex{T}(T(π) / (sin(T(π) * x) * gamma(T(1) - x)))
        return Complex{T}(gamma(x))
    end
    return gamma(z)
end

# Real arguments route through the safe complex path (avoids gamma's neg-real throw).
_cgamma(z::Real) = real(_cgamma(complex(float(z))))
_cgamma(z) = gamma(z)

# ============================================================
#  Pochhammer symbol (a)_n = Γ(a+n)/Γ(a)
# ============================================================

function pochhammer(a, n::Int)
    if n == 0
        return complex(1.0)
    elseif n > 0
        result = complex(1.0)
        for k in 0:n-1
            result *= (a + k)
        end
        return result
    else  # n < 0
        result = complex(1.0)
        for k in 1:(-n)
            result /= (a - k)
        end
        return result
    end
end

# ============================================================
#  Convenience: compute B^inc on real axis
# ============================================================

"""
    scan_Binc(s, l, m, a, ωrange)

Compute B^inc for a range of real frequencies.
Returns vectors (ω_vals, Binc_vals).
"""
function scan_Binc(s::Int, l::Int, m::Int, a::Float64, ωrange)
    ω_vals = collect(ωrange)
    Binc_vals = ComplexF64[]

    for ω in ω_vals
        try
            result = compute_amplitudes(s, l, m, a, ω)
            push!(Binc_vals, result.Binc)
        catch e
            @warn "Failed at ω = $ω: $e"
            push!(Binc_vals, NaN + NaN*im)
        end
    end

    return ω_vals, Binc_vals
end

"""
    spectral_Binc_inv(s, l, m, a, ω)

Compute 1/(2iω B^inc) — the key spectral quantity for the Green function.
"""
function spectral_Binc_inv(s::Int, l::Int, m::Int, a::Float64, ω)
    result = compute_amplitudes(s, l, m, a, ω)
    return 1.0 / (2im * ω * result.Binc)
end
