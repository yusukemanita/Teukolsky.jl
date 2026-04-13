# ============================================================
#  BigFloat-safe gamma
#
#  SpecialFunctions.gamma(x::BigFloat) throws DomainError for x < 0.
#  When ω is on the negative imaginary axis, many gamma arguments become
#  negative reals (zero imaginary part), triggering this bug.
#  Fix: reflection formula Γ(x) = π / (sin(πx) Γ(1-x)) for x < 0.
# ============================================================

function _cgamma(z::Complex{T}) where T<:AbstractFloat
    if iszero(imag(z)) && real(z) < 0
        x = real(z)
        return Complex{T}(T(π) / (sin(T(π) * x) * gamma(T(1) - x)))
    end
    return gamma(z)
end
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
