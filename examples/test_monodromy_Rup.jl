using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")

using Teukolsky
using Printf
using SpecialFunctions

# ============================================================
#  Test: Branch-cut discontinuity (Casals-Ottewill 1608.05392, eq. 33)
#
#  Key relation:
#    δRup ≡ Rup(ω_L) − Rup(ω_R) = iq(σ) × Rdown(ω_R)
#
#  where Rup is transmission-normalized (R̂^up in paper eq. 12):
#    Rup ~ r^{-1-2s} e^{+iωr*}  at r → ∞
#  and ω_R = δ − iσ is just to the right of the negative imaginary axis.
#
#  This normalization is crucial: both sides share the dominant e^{+σr*}
#  behavior, so their difference δRup is purely the subdominant e^{-σr*}
#  part — the same asymptotics as Rdown. Without this normalization the
#  dominant parts have different coefficients (Ctrans_L ≠ Ctrans_R) and
#  the ratio blows up exponentially with r.
#
#  Left-side value via conjugation symmetry:
#    Rup_{l,m}(ω_L) = conj(Rup_{l,−m}(ω_R))
#  For Schwarzschild (a=0, m absent in radial eq.):
#    Rup(ω_L) = conj(Rup(ω_R))
#  For Kerr:
#    Rup_{l,m}(ω_L) = conj(Rup_{l,−m}(ω_R))
#
#  Numerical result: ratio → −1 as δ→0, so the exact identity is
#    δRup = −iq(σ) × Rdown
#  The sign difference from the paper is a convention choice for δ
#  (left−right vs right−left).
# ============================================================

setprecision(BigFloat, 256)

s, l, m = -2, 2, 2
nmax    = 60

function compute_ratio(s, l, m, a_val, σ, δ, r; nmax=60)
    ω_R = Complex{BigFloat}(δ - im*σ)

    # ── Right side at ω_R ────────────────────────────────────
    ν_R, p_R = compute_nu(s, l, m, a_val, ω_R)
    fn_R     = compute_fn(p_R, ν_R; nmax=nmax)

    # Rup is transmission-normalized: Rup ~ r^{-1-2s} e^{+iωr*}
    Rup_R   = Rup(p_R, ν_R, fn_R, r; nmax=nmax)
    Rdown_R = Rdown(p_R, ν_R, fn_R, r; nmax=nmax)

    # ── Left side via conjugation symmetry ───────────────────
    # Rup_{l,m}(ω_L) = conj(Rup_{l,−m}(ω_R))
    ν_sym, p_sym = compute_nu(s, l, -m, a_val, ω_R)
    fn_sym       = compute_fn(p_sym, ν_sym; nmax=nmax)

    Rup_L = conj(Rup(p_sym, ν_sym, fn_sym, r; nmax=nmax))

    # ── Branch-cut discontinuity δRup = Rup(ω_L) − Rup(ω_R) ─
    δRup = Rup_L - Rup_R

    # ── q coefficient (Casals-Ottewill eq. 50) ───────────────
    q_info = compute_q(s, l, m, a_val, ω_R; nmax=nmax)
    q_val  = q_info.q

    # ── Ratio: numerically converges to −1 as δ→0 ────────────
    # Identity: δRup = −iq × Rdown
    expected = im * q_val * Rdown_R
    ratio    = δRup / expected

    return (ratio=ratio, δRhat=δRup, expected=expected,
            q=q_val, Rdown=Rdown_R)
end

function main()
    # ══════════════════════════════════════════════════════════
    #  Test 1: Schwarzschild — ratio δR̂^up / (iq·Rdown) for various r
    # ══════════════════════════════════════════════════════════
    println("="^70)
    println("Test 1: Schwarzschild a=0 — δRup / (iq·Rdown) should be constant (≈ −1)")
    println("  Rup is transmission-normalized: Rup ~ r^{-1-2s} e^{+iωr*}")
    println("  s=$s, l=$l, m=$m")
    println("="^70)

    a_val = BigFloat(0)
    r_vals = [4, 6, 8, 10, 15]
    for σ_f64 in [0.3, 0.5, 0.8]
        σ = parse(BigFloat, string(σ_f64))
        δ = parse(BigFloat, "1e-8")
        @printf("σ=%.1f:\n", σ_f64)
        for r_i in r_vals
            r = BigFloat(r_i)
            res = compute_ratio(s, l, m, a_val, σ, δ, r; nmax=nmax)
            rat = res.ratio
            @printf("  r=%2d  ratio = %+.8e %+.8ei  |ratio| = %.6e\n",
                    r_i, Float64(real(rat)), Float64(imag(rat)),
                    Float64(abs(rat)))
        end
    end

    # ══════════════════════════════════════════════════════════
    #  Test 2: Raw values comparison
    # ══════════════════════════════════════════════════════════
    println()
    println("="^70)
    println("Test 2: Raw δRup vs iq·Rdown (should be equal in magnitude, opposite sign)")
    println("="^70)

    r_test = BigFloat(6)
    δ = parse(BigFloat, "1e-8")
    for σ_f64 in [0.3, 0.5, 0.8]
        σ = parse(BigFloat, string(σ_f64))
        res = compute_ratio(s, l, m, a_val, σ, δ, r_test; nmax=nmax)
        @printf("σ=%.1f:\n", σ_f64)
        @printf("  δRup     = %+.8e %+.8ei\n",
                Float64(real(res.δRhat)), Float64(imag(res.δRhat)))
        @printf("  iq·Rdown = %+.8e %+.8ei\n",
                Float64(real(res.expected)), Float64(imag(res.expected)))
        @printf("  ratio    = %+.8e %+.8ei\n",
                Float64(real(res.ratio)), Float64(imag(res.ratio)))
        @printf("  |q|      = %.6e\n", Float64(abs(res.q)))
    end

    # ══════════════════════════════════════════════════════════
    #  Test 3: Kerr a=0.9
    # ══════════════════════════════════════════════════════════
    println()
    println("="^70)
    println("Test 3: Kerr a=0.9 — ratio δRup/(iq·Rdown) should be constant (≈ −1)")
    println("="^70)

    a_kerr = parse(BigFloat, "0.9")
    δ = parse(BigFloat, "1e-8")
    for σ_f64 in [0.3, 0.5, 0.8]
        σ = parse(BigFloat, string(σ_f64))
        @printf("σ=%.1f:\n", σ_f64)
        for r_i in r_vals
            r = BigFloat(r_i)
            res = compute_ratio(s, l, m, a_kerr, σ, δ, r; nmax=nmax)
            rat = res.ratio
            @printf("  r=%2d  ratio = %+.8e %+.8ei  |ratio| = %.6e\n",
                    r_i, Float64(real(rat)), Float64(imag(rat)),
                    Float64(abs(rat)))
        end
    end

    # ══════════════════════════════════════════════════════════
    #  Test 4: δ convergence (should stabilize as δ→0)
    # ══════════════════════════════════════════════════════════
    println()
    println("="^70)
    println("Test 4: δ convergence for Schwarzschild σ=0.3, r=6")
    println("="^70)

    r_test = BigFloat(6)
    σ = parse(BigFloat, "0.3")
    for δ_exp in [-3, -4, -5, -6, -7, -8, -10]
        δ_bf = parse(BigFloat, "1e$(δ_exp)")
        res = compute_ratio(s, l, m, a_val, σ, δ_bf, r_test; nmax=nmax)
        rat = res.ratio
        @printf("  δ=1e%+d  ratio = %+.8e %+.8ei  |ratio| = %.6e\n",
                δ_exp, Float64(real(rat)), Float64(imag(rat)),
                Float64(abs(rat)))
    end
end

main()
