# Monodromy truncation-depth envelope (Mandate 3).
#
# The adaptive monodromy drivers (_monodromy_adaptive / _monodromy_adaptive_acb)
# now start at the MEASURED convergence envelope
#     n̂(prec, |ε|) = 1.04·prec + 13·|ε| + 40 + 2Δ        (|ε| = 2|ω|, Δ = 128)
# instead of the old 4.71·prec floor (a ~4× overshoot: the actual series error
# decays at ~0.9 bits/step, monotonically, straight down to the rounding floor
# — mapped over σ ∈ {0.25…20} (ω = iσ), real/complex ω, prec ∈ {256…1536},
# s ∈ {−2,2}, l ∈ {2,4,10,20}, a ∈ {0.7,0.9}).  Acceptance is a THREE-depth
# agreement (n, n−Δ, n−2Δ ≈ 230 bits of decay span) so two evaluations cannot
# agree spuriously on a slow stretch, with the verify-and-extend loop as the
# backstop.
#
# This test guards the envelope against under-truncation:
#   (a) ν from the new start must match a DEEP reference (old 4.71·prec floor
#       + 256, evaluated at prec+64 guard bits) to 2^(−0.8·prec), across the
#       hostile PIA grid (incl. exact resonances 4σ ∈ ℤ), real ω, complex ω;
#   (b) ν must satisfy the CF equation g(ν) = β₀ + α₀R₁ + γ₀L₋₁ = 0 — an
#       arbiter fully independent of the monodromy series.  NOTE the residual
#       SCALE grows steeply with σ (|dg/dν| amplification ~10¹⁵ at σ = 10–12);
#       thresholds below were calibrated on the pre-change code, which produces
#       byte-identical residuals.
using Test
using Teukolsky
using Arblib
using Arblib: Arb, Acb

const _TD = Teukolsky

# ν from cos(2πν) — mirrors the branch block in _compute_nu_monodromy*.
function _td_nu_from_c(c2pn, l, ω, R)
    twoπ = 2 * R(π); rc = real(c2pn)
    imag(complex(ω)) != 0 && return R(l) - acos(complex(c2pn)) / twoπ
    (-1 ≤ rc ≤ 1) && return R(l) - acos(complex(rc)) / twoπ
    rc < -1 && return Complex(R(1) / 2, +acosh(-rc) / twoπ)
    return Complex(R(0), -acosh(rc) / twoπ)
end

# Deep-truncation reference: value at the OLD floor + 256, at the SAME working
# precision and the SAME λ as the production solve.  This isolates the quantity
# under test — the series TRUNCATION depth — from the (σ-dependent) precision
# of λ and of the scalar setup, which are common to both and cancel.
function _td_deep_nu(s, l, m, a, ω, prec)
    setprecision(Arb, prec) do
        ωc = Complex{Arb}(Arb(real(ω)), Arb(imag(ω)))
        aA = Arb(a)
        p = _TD.MSTParams(s, l, m, aA, ωc)
        ndeep = ceil(Int, 4.71 * prec) + 256
        ctx = _TD._build_monodromy_ctx_acb(s, l, m, aA, ωc, p.λ, ndeep; prec=prec)
        c = _TD._strip_radius(_TD._monodromy_value_acb(ctx, ndeep; prec=prec))
        _td_nu_from_c(c, l, ωc, Arb)
    end
end

function _td_dnu_log2(νn, νd, prec)
    setprecision(BigFloat, prec + 64) do
        d = abs(Complex{BigFloat}(νn) - Complex{BigFloat}(νd))
        Float64(log2(max(d, BigFloat(2)^(-2 * prec))))
    end
end

function _td_cf_residual(ν, s, l, m, a, ω; bits=300, nmax=800)
    setprecision(BigFloat, bits) do
        νb = Complex{BigFloat}(ν)
        pb = _TD.MSTParams(s, l, m, BigFloat(a),
                           Complex{BigFloat}(BigFloat(real(ω)), BigFloat(imag(ω))))
        R1  = _TD.Rn_cf(pb, νb, 1;  nmax=nmax)
        Lm1 = _TD.Ln_cf(pb, νb, -1; nmax=nmax)
        Float64(abs(_TD.βn(pb, νb, 0) + _TD.αn(pb, νb, 0) * R1 +
                    _TD.γn(pb, νb, 0) * Lm1))
    end
end

@testset "monodromy depth envelope" begin
    s_, l_, m_ = -2, 2, 2
    a_ = 0.7

    @testset ":acb vs deep reference (PIA grid incl. resonances)" begin
        # σ grid mixes exact resonances (4σ ∈ ℤ: 0.25, 2, 12 → safe marching-Γ
        # path) and off-resonance (4.3), at both s signs.
        for prec in (256, 640), σ in (0.25, 2.0, 4.3, 12.0), s in (-2, 2)
            ω = complex(0.0, σ)
            νn, _ = compute_nu(s, l_, m_, a_, ω; precision=prec, backend=:acb)
            νd = _td_deep_nu(s, l_, m_, a_, ω, prec)
            @test _td_dnu_log2(νn, νd, prec) <= -0.8 * prec
        end
    end

    @testset ":acb vs deep reference (real / complex ω)" begin
        for prec in (256, 640), ω in (complex(0.3, 0.0), complex(1.1, 0.0),
                                      complex(0.5, 0.3))
            νn, _ = compute_nu(s_, l_, m_, a_, ω; precision=prec, backend=:acb)
            νd = _td_deep_nu(s_, l_, m_, a_, ω, prec)
            @test _td_dnu_log2(νn, νd, prec) <= -0.8 * prec
        end
    end

    @testset "BigFloat backend vs :acb (new generic driver)" begin
        # NOTE: ω is given as Float64 and promoted, so both backends see the
        # IDENTICAL binary value (big"1.1" ≠ Float64(1.1) would dominate Δν).
        for prec in (256, 640), ω0 in (complex(0.0, 2.0), complex(1.1, 0.0),
                                       complex(0.5, 0.3))
            νb, _ = compute_nu(s_, l_, m_, a_, Complex{BigFloat}(ω0); precision=prec)
            νa, _ = compute_nu(s_, l_, m_, a_, ω0; precision=prec,
                               backend=:acb)
            @test _td_dnu_log2(νb, νa, prec) <= -0.8 * prec
        end
    end

    @testset "CF-residual arbiter (independent of the monodromy series)" begin
        # Moderate σ: residual scale ~2^(-prec)·amplification; 1e-55 is ~15
        # digits of slack at prec=256 (measured ~1e-66…1e-73).
        for σ in (0.5, 2.0, 4.3), s in (-2, 2)
            ω = complex(0.0, σ)
            νn, _ = compute_nu(s, l_, m_, a_, ω; precision=256, backend=:acb)
            @test _td_cf_residual(νn, s, l_, m_, a_, ω) < 1e-55
        end
        for ω in (complex(0.3, 0.0), complex(1.1, 0.0), complex(0.5, 0.3))
            νn, _ = compute_nu(s_, l_, m_, a_, ω; precision=256, backend=:acb)
            @test _td_cf_residual(νn, s_, l_, m_, a_, ω) < 1e-55
        end
        # Large σ: |dg/dν| amplification eats ~50 digits at σ=10–12, so solve
        # deeper (768 bits) and arbitrate at 900 bits.  Measured residuals
        # ~1e-190…1e-205 (identical to the pre-change code); 1e-160 is slack.
        for (σ, s) in ((10.0, -2), (12.0, 2))
            ω = complex(0.0, σ)
            νn, _ = compute_nu(s, l_, m_, a_, ω; precision=768, backend=:acb)
            @test _td_cf_residual(νn, s, l_, m_, a_, ω; bits=900, nmax=1500) < 1e-160
        end
    end

    @testset "high-l / high-a mode inside envelope" begin
        # l-dependence of the convergence depth measured ≈ 0 (l up to 20);
        # guard one high-l, near-extremal point anyway.
        for (l, m) in ((10, 10), (10, -10))
            ω = complex(0.0, 8.0)
            νn, _ = compute_nu(-2, l, m, 0.9, ω; precision=320, backend=:acb)
            νd = _td_deep_nu(-2, l, m, 0.9, ω, 320)
            @test _td_dnu_log2(νn, νd, 320) <= -0.8 * 320
        end
    end
end
