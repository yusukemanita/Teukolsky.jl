# ============================================================
#  Monodromy method for cos(2πν)
# ============================================================


"""
    monodromy_cos2pi_nu(s, l, m, a, ω, λ; nmax=60)

Compute cos(2πν) from the monodromy of the confluent Heun equation.
Works for any numeric precision (Float64, BigFloat, etc.).
"""
function monodromy_cos2pi_nu(s, _l, m, a, ω, λ; nmax::Int=60)
    # Infer complex type from inputs
    R = promote_type(typeof(float(real(a))), typeof(float(real(complex(ω)))))
    C = Complex{R}

    q  = R(a)
    ε  = 2 * C(ω)
    κ  = sqrt(C(1 - q^2))
    τ  = (ε - m*q) / κ

    γCH = 1 - s - im*ε - im*τ
    δCH = 1 + s + im*ε - im*τ
    εCH = 2im*ε*κ
    αε  = C(1 - s) + im*(ε - τ)
    qCH = s*(s+1) - ε^2 + im*(1-2s)*ε*κ + λ + im*τ + τ^2

    μ1C = αε - (γCH + δCH)
    μ2C = -αε

    a1 = zeros(C, nmax + 2)
    a2 = zeros(C, nmax + 2)
    a1[1] = one(C); a2[1] = one(C)

    for n in 1:nmax
        a1p = a1[n];  a1pp = n >= 2 ? a1[n-1] : zero(C)
        c2 = (αε - (n-1+δCH)) * (αε - (n-2+γCH+δCH)) * εCH / n
        c1 = (αε^2 + αε*(1-2n-γCH-δCH+εCH) +
              (n^2 - qCH + n*(-1+γCH+δCH-εCH) + εCH - δCH*εCH)) / n
        a1[n+1] = c2*a1pp - c1*a1p

        a2p = a2[n];  a2pp = n >= 2 ? a2[n-1] : zero(C)
        d2 = (αε + (n-2)) * (αε + (n-1-γCH)) * εCH / n
        d1 = (αε^2 + (n^2 - qCH + γCH + δCH - n*(1+γCH+δCH-εCH) - εCH) +
              αε*(-1+2n-γCH-δCH+εCH)) / n
        a2[n+1] = -d2*a2pp + d1*a2p
    end

    Poch_p = ones(C, nmax + 2)
    Poch_m = ones(C, nmax + 2)
    for i in 1:nmax
        Poch_p[i+1] = (-μ2C + μ1C + i - 1) * Poch_p[i]
        Poch_m[i+1] = ( μ2C - μ1C + i - 1) * Poch_m[i]
    end

    n    = nmax
    jmax = cld(n, 2)
    a1sum = _cgamma(-μ2C + μ1C) * sum(a1[j+1] * Poch_p[n-j+1] for j in 0:jmax)
    a2sum = _cgamma( μ2C - μ1C) * sum((-1)^j * a2[j+1] * Poch_m[n-j+1] for j in 0:jmax)

    return cos(π*(μ1C - μ2C)) + (2π^2 / (a1sum * a2sum)) * (-1)^(n-1) * a1[n+1] * a2[n+1]
end

"""
    nu_initial_guess(c2pn, l)

From cos(2πν), compute the initial guess for ν and the search type.
Uses full complex arccos to handle Im(c2pn) ≠ 0 (complex ω).
"""
function nu_initial_guess(c2pn, l)
    rc = real(c2pn)
    # Full complex arccos: cos(2πν₀) = c2pn exactly on the principal branch.
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
    compute_nu(s, l, m, a, ω; nmax_cf=150, tol=-1, maxiter=200, precision=64, ν_init=nothing)

Solve for ν using monodromy method + Newton refinement.

- `tol`: convergence tolerance. Default (`tol=-1`) auto-scales: ~100·eps(R).
- `precision`: bits of floating-point (64 = Float64, ≥128 = BigFloat).
- `ν_init`: optional initial guess for ν (tried first, before monodromy guess).
  Useful for branch tracking along a parameter path.
"""
function compute_nu(s::Int, l::Int, m::Int, a, ω;
                    nmax_cf::Int=150, tol::Real=-1, maxiter::Int=200,
                    precision::Int=64, ν_init=nothing)
    if precision > 64
        return setprecision(BigFloat, precision) do
            compute_nu(s, l, m, BigFloat(a), Complex{BigFloat}(complex(ω));
                       nmax_cf=nmax_cf, tol=tol, maxiter=maxiter, precision=64,
                       ν_init=ν_init === nothing ? nothing : Complex{BigFloat}(ν_init))
        end
    end
    _compute_nu_impl(s, l, m, a, ω; nmax_cf, tol, maxiter, ν_init)
end

function _compute_nu_impl(s::Int, l::Int, m::Int, a, ω;
                           nmax_cf::Int=150, tol::Real=-1, maxiter::Int=200,
                           ν_init=nothing)
    p = MSTParams(s, l, m, a, ω)
    R = typeof(p.a)

    # Auto-scale tolerance and finite-difference step with precision
    tol_use  = tol < 0 ? R(100) * eps(R) : R(tol)
    δ        = cbrt(eps(R))   # finite-difference step for Newton

    function g0(ν)
        R1  = Rn_cf(p, ν, 1;  nmax=nmax_cf)
        Lm1 = Ln_cf(p, ν, -1; nmax=nmax_cf)
        βn(p, ν, 0) + αn(p, ν, 0) * R1 + γn(p, ν, 0) * Lm1
    end

    # Unconstrained 2D complex Newton
    function newton_from(ν0; max_step=R(2))
        ν = Complex{R}(ν0)
        for _ in 1:maxiter
            g = g0(ν)
            !isfinite(g) && return ν, false
            abs(g) < tol_use && return ν, true
            gp = (g0(ν + δ) - g0(ν - δ)) / (2δ)
            abs(gp) < R(1e-30) && return ν, false
            Δν = -g / gp
            abs(Δν) > max_step && (Δν *= max_step / abs(Δν))
            ν += Δν
            abs(Δν) < tol_use && return ν, true
        end
        g_final = g0(ν)
        return ν, isfinite(g_final) && abs(g_final) < sqrt(tol_use)
    end

    # Constrained 1D real Newton: Re(ν) fixed, search Im(ν) only.
    # Solves Re(g0(ν_re + i·η)) = 0 for real η.
    # Matches Mathematica's FindRoot[Re[g[1/2 + I νi]] == 0, {νi, ...}].
    function newton_1d(ν_re, η0; max_step=1.0)
        η = R(η0)
        ν_re_R = R(ν_re)
        for _ in 1:maxiter
            ν  = Complex{R}(ν_re_R, η)
            g  = real(g0(ν))
            abs(g) < tol_use && return Complex{R}(ν_re_R, η), true
            gp = real(g0(Complex{R}(ν_re_R, η + δ)) - g0(Complex{R}(ν_re_R, η - δ))) / (2δ)
            abs(gp) < 1e-30 && break
            Δη = -g / gp
            abs(Δη) > max_step && (Δη *= max_step / abs(Δη))
            η += Δη
            abs(Δη) < tol_use && return Complex{R}(ν_re_R, η), true
        end
        ν_fin = Complex{R}(ν_re_R, η)
        return ν_fin, abs(real(g0(ν_fin))) < sqrt(tol_use)
    end

    # Compute monodromy first to classify the branch
    c2pn = monodromy_cos2pi_nu(s, l, m, a, ω, p.λ)
    rc   = real(c2pn)

    if imag(complex(ω)) != 0
        # Complex ω: ν_init tracking is safe, use unconstrained 2D Newton
        if ν_init !== nothing
            ν2, c2 = newton_from(ν_init)
            c2 && return ν2, p
        end
        # Use l-offset form (same as real branch) so that ν₀ ≈ l for small Im(ω).
        # Without the offset, acos(c2pn)/(2π) ≈ 0 and Newton may converge to the
        # spurious root ν=0 instead of the physical ν≈l root.
        ν0 = ComplexF64(l) - acos(complex(c2pn)) / (2π)
        ν, converged = newton_from(ν0)
        converged && return ν, p
        ν2, c2 = newton_from(conj(ν0));  c2 && return ν2, p
        ν2, c2 = newton_from(ComplexF64(l) - ν0);  c2 && return ν2, p

    elseif -1 ≤ rc ≤ 1
        # Real ν branch.
        # ν_init tracking is only safe when ν_init itself was on the real branch
        # (Im(ν_init) ≈ 0). If ν_init came from a half/integer branch (Im large),
        # Newton can stray to degenerate integer roots (ν=1,2,…) where the fn
        # recurrence is ill-conditioned and Btrans → 0 spuriously.
        # Use monodromy formula first; fall back to ν_init as secondary seed.
        ν0 = ComplexF64(l) - acos(complex(rc)) / (2π)
        ν, converged = newton_from(ν0)
        converged && return ν, p
        if ν_init !== nothing
            ν2, c2 = newton_from(ν_init)
            c2 && return ν2, p
        end

    elseif rc < -1
        # Half-integer case: cos(2πν) < -1  →  Re(ν) = n + 1/2
        # Mathematica always uses Re(ν) = 1/2 (not l-1/2).
        # Convention: ν = 1/2 - Im(arccos(rc)/(2π))*i
        # arccos(rc) for rc < -1 (principal branch) = π - i*acosh(-rc),
        # so Im(arccos(rc)/(2π)) = -acosh(-rc)/(2π) < 0  →  η0 > 0.
        η0 = R(acosh(max(-rc, one(R))) / (2π))   # > 0
        # Try Re(ν) = 1/2 first (Mathematica convention), then l±1/2 as fallbacks
        for ν_re_try in R[R(0.5), l - R(0.5), l + R(0.5), l - R(1.5), R(-0.5)]
            ν2, c2 = newton_1d(ν_re_try,  η0)
            c2 && return ν2, p
            ν2, c2 = newton_1d(ν_re_try, -η0)
            c2 && return ν2, p
        end
        # Fallback: unconstrained 2D Newton from monodromy guess
        ν0 = ComplexF64(l) - acos(complex(c2pn)) / (2π)
        ν, converged = newton_from(ν0);  converged && return ν, p
        ν2, c2 = newton_from(conj(ν0));  c2 && return ν2, p

    else  # rc > 1
        # Integer (pure-imaginary) case: cos(2πν) > 1  →  Re(ν) = 0
        # Mathematica uses Re(ν) = 0.
        # arccos(rc) for rc > 1 (principal branch) = -i*acosh(rc),
        # so Im(arccos(rc)/(2π)) = -acosh(rc)/(2π) < 0  →  η0 > 0.
        η0 = R(acosh(max(rc, one(R))) / (2π))   # > 0
        for ν_re_try in R[R(0), R(1), R(-1), l, l - R(1)]
            ν2, c2 = newton_1d(ν_re_try,  η0)
            c2 && return ν2, p
            ν2, c2 = newton_1d(ν_re_try, -η0)
            c2 && return ν2, p
        end
        ν0 = ComplexF64(l) - acos(complex(c2pn)) / (2π)
        ν, converged = newton_from(ν0);  converged && return ν, p
    end

    # Final fallback: try a grid of seeds
    ν0_fb = ComplexF64(l) - acos(complex(c2pn)) / (2π)
    im_ω  = imag(complex(ω))
    for ν_try in [ν0_fb, conj(ν0_fb), real(ν0_fb) + 0im,
                   Complex{R}(l, im_ω), Complex{R}(l, -im_ω),
                   Complex{R}(l - R(0.5), 0), Complex{R}(l + R(0.5), 0)]
        ν2, c2 = newton_from(ν_try)
        c2 && return ν2, p
    end

    @warn "compute_nu: Newton did not converge, |g| = $(abs(g0(ν0_fb)))"
    ν_best, _ = newton_from(ν0_fb)
    return ν_best, p
end
