# ============================================================
#  Monodromy method for cos(2πν)
# ============================================================

"""
    monodromy_cos2pi_nu(s, l, m, a, ω, λ; nmax=60)

Compute cos(2πν) from the monodromy of the confluent Heun equation.
"""
function monodromy_cos2pi_nu(s, l, m, a, ω, λ; nmax::Int=60)
    q  = Float64(a)
    ε  = 2*complex(ω)
    κ  = sqrt(complex(1 - q^2))
    τ  = (ε - m*q) / κ

    γCH = 1 - s - im*ε - im*τ
    δCH = 1 + s + im*ε - im*τ
    εCH = 2im*ε*κ
    αε  = complex(1 - s) + im*(ε - τ)
    qCH = s*(s+1) - ε^2 + im*(1-2s)*ε*κ + λ + im*τ + τ^2

    μ1C = αε - (γCH + δCH)
    μ2C = -αε

    a1 = zeros(ComplexF64, nmax + 2)
    a2 = zeros(ComplexF64, nmax + 2)
    a1[1] = 1.0; a2[1] = 1.0

    for n in 1:nmax
        a1p = a1[n];  a1pp = n >= 2 ? a1[n-1] : zero(ComplexF64)
        c2 = (αε - (n-1+δCH)) * (αε - (n-2+γCH+δCH)) * εCH / n
        c1 = (αε^2 + αε*(1-2n-γCH-δCH+εCH) +
              (n^2 - qCH + n*(-1+γCH+δCH-εCH) + εCH - δCH*εCH)) / n
        a1[n+1] = c2*a1pp - c1*a1p

        a2p = a2[n];  a2pp = n >= 2 ? a2[n-1] : zero(ComplexF64)
        d2 = (αε + (n-2)) * (αε + (n-1-γCH)) * εCH / n
        d1 = (αε^2 + (n^2 - qCH + γCH + δCH - n*(1+γCH+δCH-εCH) - εCH) +
              αε*(-1+2n-γCH-δCH+εCH)) / n
        a2[n+1] = -d2*a2pp + d1*a2p
    end

    Poch_p = ones(ComplexF64, nmax + 2)
    Poch_m = ones(ComplexF64, nmax + 2)
    for i in 1:nmax
        Poch_p[i+1] = (-μ2C + μ1C + i - 1) * Poch_p[i]
        Poch_m[i+1] = ( μ2C - μ1C + i - 1) * Poch_m[i]
    end

    n    = nmax
    jmax = cld(n, 2)
    a1sum = gamma(-μ2C + μ1C) * sum(a1[j+1] * Poch_p[n-j+1] for j in 0:jmax)
    a2sum = gamma( μ2C - μ1C) * sum((-1)^j * a2[j+1] * Poch_m[n-j+1] for j in 0:jmax)

    return cos(π*(μ1C - μ2C)) + (2π^2 / (a1sum * a2sum)) * (-1)^(n-1) * a1[n+1] * a2[n+1]
end

"""
    nu_initial_guess(c2pn, l)

From cos(2πν), compute the initial guess for ν and the search type.
"""
function nu_initial_guess(c2pn, l)
    rc = real(c2pn)
    # Use full complex arccos: cos(2πν₀) = c2pn exactly on the principal branch.
    # Using only real(c2pn) loses Im(c2pn), which is large for complex ω and causes
    # Newton to start far from the correct root.
    ν0 = ComplexF64(l) - acos(complex(c2pn)) / (2π)
    if -1 ≤ rc ≤ 1
        return ν0, :real
    elseif rc < -1
        return ν0, :half
    else
        return ν0, :integer
    end
end

# ============================================================
#  Solve for ν, Eq. (136): β₀ + α₀R₁ + γ₀L₋₁ = 0
# ============================================================

"""
    compute_nu(s, l, m, a, ω; nmax_cf=150, tol=1e-12, maxiter=200)

Solve for ν using monodromy method + Newton refinement.
"""
function compute_nu(s::Int, l::Int, m::Int, a::Float64, ω;
                    nmax_cf::Int=150, tol::Float64=1e-12, maxiter::Int=200)
    p = MSTParams(s, l, m, a, ω)

    function g0(ν)
        R1  = Rn_cf(p, ν, 1;  nmax=nmax_cf)
        Lm1 = Ln_cf(p, ν, -1; nmax=nmax_cf)
        βn(p, ν, 0) + αn(p, ν, 0) * R1 + γn(p, ν, 0) * Lm1
    end

    function newton_from(ν0; max_step=2.0)
        ν = complex(ν0); δ = 1e-7
        for _ in 1:maxiter
            g = g0(ν)
            !isfinite(g) && return ν, false
            abs(g) < tol && return ν, true
            gp = (g0(ν + δ) - g0(ν - δ)) / (2δ)
            abs(gp) < 1e-30 && return ν, false
            Δν = -g / gp
            abs(Δν) > max_step && (Δν *= max_step / abs(Δν))
            ν += Δν
            abs(Δν) < tol && return ν, true
        end
        # Relaxed acceptance: |g| < √tol ≈ 1e-6 for default tol
        g_final = g0(ν)
        return ν, isfinite(g_final) && abs(g_final) < sqrt(tol)
    end

    # Fix: pass full complex p.λ — real(p.λ) loses imaginary part for complex ω
    c2pn = monodromy_cos2pi_nu(s, l, m, a, ω, p.λ)
    ν0, case = nu_initial_guess(c2pn, l)

    ν, converged = newton_from(ν0)
    converged && return ν, p

    # Fallback 1: conjugate of monodromy guess
    if case != :real
        ν2, c2 = newton_from(conj(ν0))
        c2 && return ν2, p
    end

    # Fallback 2: purely real part of monodromy guess
    ν2, c2 = newton_from(real(ν0))
    c2 && return ν2, p

    # Fallback 3: canonical guesses seeded from imaginary part of ω
    im_ω = imag(complex(ω))
    for ν_try in [ComplexF64(l, im_ω), ComplexF64(l, -im_ω),
                   ComplexF64(l + 0.5, im_ω), ComplexF64(l - 0.5, im_ω)]
        ν2, c2 = newton_from(ν_try)
        c2 && return ν2, p
    end

    @warn "compute_nu: Newton did not converge, |g| = $(abs(g0(ν)))"
    return ν, p
end
