# ============================================================
#  BigFloat-safe gamma
#
#  SpecialFunctions.gamma(x::BigFloat) throws DomainError for x < 0.
#  When ω is on the negative imaginary axis, many gamma arguments become
#  negative reals (zero imaginary part), triggering this bug.
#  Fix: reflection formula Γ(x) = π / (sin(πx) Γ(1-x)) for x < 0.
# ============================================================

"""
    _loggamma_stirling(z)

Stirling series for log Γ(z), valid for |z| >> 1:
    log Γ(z) ≈ (z-1/2)log(z) - z + log(2π)/2 + Σ B_{2k}/[2k(2k-1) z^{2k-1}]

Uses the first 10 Bernoulli terms for high precision.
"""
function _loggamma_stirling(z::Complex{T}) where T
    # Bernoulli numbers B_{2k} / (2k*(2k-1))
    coeffs = T.([
        1//12, -1//360, 1//1260, -1//1680, 1//1188,
        -691//360360, 1//156, -3617//122400, 43867//244188, -174611//125400
    ])
    s = zero(Complex{T})
    zn = z
    for c in coeffs
        s += c / zn
        zn *= z * z
    end
    return (z - T(1)/2) * log(z) - z + log(T(2) * T(π)) / 2 + s
end

"""
    _cgamma(z::Complex{T})

BigFloat-safe complex Gamma function.
Uses recurrence to shift Re(z) ≥ 10, then Stirling series.
Falls back to SpecialFunctions.gamma for Float64/Float32.
"""
function _cgamma(z::Complex{T}) where T<:AbstractFloat
    # For Float64, try SpecialFunctions first
    if T === Float64 || T === Float32
        if iszero(imag(z)) && real(z) < 0
            x = real(z)
            return Complex{T}(T(π) / (sin(T(π) * x) * gamma(T(1) - x)))
        end
        return gamma(z)
    end

    # BigFloat path: recurrence + Stirling
    # Reflection formula for Re(z) < 0
    if real(z) < 0
        # Γ(z) = π / (sin(πz) Γ(1-z))
        return T(π) / (sin(T(π) * z) * _cgamma(one(Complex{T}) - z))
    end

    # Recurrence Γ(z) = Γ(z+n) / (z(z+1)...(z+n-1)) to shift Re(z) ≥ 10
    shift = max(0, ceil(Int, 10 - real(z)))
    z_shifted = z + shift
    log_gamma = _loggamma_stirling(z_shifted)
    result = exp(log_gamma)

    # Undo the shift: Γ(z) = Γ(z+n) / Π_{k=0}^{n-1} (z+k)
    for k in (shift-1):-1:0
        result /= (z + k)
    end
    return result
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
