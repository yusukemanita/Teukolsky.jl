using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Printf
using SpecialFunctions

# ============================================================
#  Test: Branch-cut discontinuity via Teukolsky conjugation symmetry
#
#  Claim (Leaver 1986 eq. 31 + conjugation symmetry):
#
#    ΔR^up ≡ conj(R^up_{l,−m}(ω_R)) − R^up_{l,m}(ω_R)  =  −K(ω_R) · R^down(ω_R)
#
#  where ω_R = δ − iσ is just to the right of the negative imaginary axis.
#
#  The symmetry  R^up_{l,−m}(ω)* = R^up_{l,m}(−ω*)  means that
#  conj(R^up_{l,−m}(ω_R)) = R^up_{l,m}(−ω_R*) = R^up_{l,m}(−δ − iσ) = R^up_{l,m}(ω_L)
#  i.e. it gives the value on the LEFT side of the negative imaginary axis.
#
#  For Schwarzschild (a=0), m does not enter the radial equation so
#  R^up_{l,−m} = R^up_{l,m} and the identity reduces to checking
#  conj(R^up(ω_R)) − R^up(ω_R)  =  −K(ω_R) · R^down(ω_R).
#
#  Precision: BigFloat with 256 bits (≈ 77 decimal digits) throughout.
# ============================================================

setprecision(BigFloat, 256)

s, l, m = -2, 2, 2
a_val   = BigFloat(0)
nmax    = 60

# ------------------------------------------------------------------
#  Core function: ratio ΔR^up / (−K·R^down)
#  Should equal 1 + 0i if the identity holds.
# ------------------------------------------------------------------
function compute_ratio(s, l, m, a_val, σ, δ, r; nmax=60)
    ω_R = Complex{BigFloat}(δ - im*σ)

    # ── Right-side quantities at ω_R ─────────────────────────────
    ν_R, p_R = compute_nu(s, l, m, a_val, ω_R)
    fn_R     = compute_fn(p_R, ν_R; nmax=nmax)

    Rup_R  = Rup(p_R, ν_R, fn_R, r; nmax=nmax)
    Rdown_R = Rdown(p_R, ν_R, fn_R, r; nmax=nmax)
        q_info = compute_q(s, l, m, a_val, ω_R; nmax=nmax)
        q_val  = q_info.q
        K_val  = -im * q_val

    # ── Left-side value via conjugation symmetry ──────────────────
    # R^up_{l,−m}(ω_R)* = R^up_{l,m}(ω_L)  (branch cut "left side")
    # For a=0: R^up_{l,−m} = R^up_{l,m}, so left side = conj(R^up_{l,m}(ω_R))
    ν_sym, p_sym = compute_nu(s, l, -m, a_val, ω_R)
    fn_sym       = compute_fn(p_sym, ν_sym; nmax=nmax)
    Rup_sym      = Rup(p_sym, ν_sym, fn_sym, r; nmax=nmax)
    Rup_L        = conj(Rup_sym)          # = R^up_{l,m}(ω_L) by symmetry

    ΔRup  = Rup_L - Rup_R                 # discontinuity
        denom = -K_val * Rdown_R              # should equal ΔRup

    ratio    = ΔRup / denom
    residual = ΔRup - denom               # should be ≈ 0

    return (ratio=ratio, residual=residual, ΔRup=ΔRup, KRdown=denom,
            K=K_val, Rdown=Rdown_R)
end

function main()
        # ══════════════════════════════════════════════════════════════════
        #  Test 1: ratio ΔR^up / (−K·R^down)  should be 1 for various r
        # ══════════════════════════════════════════════════════════════════
        println("="^70)
        println("Test 1: ratio ΔR^up / (−K·R^down) for various r  (should be 1+0i)")
        println("        BigFloat 256-bit, a=0, s=−2, l=2, m=2")
        println("="^70)

        r_vals = [4, 6, 8, 10, 15]
        for σ_f64 in [0.3, 0.5, 0.8]
                σ = parse(BigFloat, string(σ_f64))
                δ = parse(BigFloat, "1e-5")
                @printf("σ=%.1f:\n", σ_f64)
                for r_i in r_vals
                        r = BigFloat(r_i)
                        res = compute_ratio(s, l, m, a_val, σ, δ, r; nmax=nmax)
                        rat = res.ratio
                        @printf("  r=%2d  ratio = %+.6e %+.6ei  |ratio−1| = %.2e\n",
                                        r_i, Float64(real(rat)), Float64(imag(rat)),
                                        Float64(abs(rat - 1)))
                end
        end

# ══════════════════════════════════════════════════════════════════
#  Test 2: δ → 0 convergence of ratio  (should stay ≈ 1, not → 0)
# ══════════════════════════════════════════════════════════════════
        println()
        println("="^70)
        println("Test 2: ratio vs δ  (should converge to 1 as δ→0)")
        println("="^70)

        r_test = BigFloat(6)
        for σ_f64 in [0.3, 0.5, 0.8]
                σ = parse(BigFloat, string(σ_f64))
                @printf("σ=%.1f:\n", σ_f64)
                for δ_exp in [-2, -3, -4, -5, -6, -7, -8]
                        δ = parse(BigFloat, "1e$(δ_exp)")
                        res = compute_ratio(s, l, m, a_val, σ, δ, r_test; nmax=nmax)
                        rat = res.ratio
                        @printf("  δ=1e%+d  ratio = %+.6e %+.6ei  |ratio−1| = %.2e\n",
                                        δ_exp, Float64(real(rat)), Float64(imag(rat)),
                                        Float64(abs(rat - 1)))
                end
        end

# ══════════════════════════════════════════════════════════════════
#  Test 3: Print raw ΔR^up and −K·R^down to compare directly
# ══════════════════════════════════════════════════════════════════
    println()
    println("="^70)
    println("Test 3: raw values of ΔR^up and −K·R^down  (should match)")
    println("  NOTE: For Schwarzschild (a=0), Rup_{l,-m}=Rup_{l,m} (m absent in radial eq.)")
    println("        so ΔRup = conj(Rup)-Rup = -2i Im(Rup) is purely imaginary,")
    println("        while -K·Rdown is complex → identity cannot hold for a=0.")
    println("="^70)

    r_test = BigFloat(6)
    δ = parse(BigFloat, "1e-6")
    for σ_f64 in [0.3, 0.5, 0.8]
        σ = parse(BigFloat, string(σ_f64))
        res = compute_ratio(s, l, m, a_val, σ, δ, r_test; nmax=nmax)
        @printf("σ=%.1f:\n", σ_f64)
        @printf("  ΔR^up       = %+.8e %+.8ei\n",
                Float64(real(res.ΔRup)), Float64(imag(res.ΔRup)))
        @printf("  −K·R^down   = %+.8e %+.8ei\n",
                Float64(real(res.KRdown)), Float64(imag(res.KRdown)))
        @printf("  residual    = %+.4e %+.4ei\n",
                Float64(real(res.residual)), Float64(imag(res.residual)))
        @printf("  |K|         = %.6e\n", Float64(abs(res.K)))
        @printf("  |R^down|    = %.6e\n", Float64(abs(res.Rdown)))
    end

# ══════════════════════════════════════════════════════════════════
#  Test 4: Kerr (a=0.9) — m enters the radial equation
#
#  For Kerr: Rup_{l,-m}(ω) ≠ Rup_{l,m}(ω).
#  The symmetry gives: conj(Rup_{l,-m}(ω_R)) = Rup_{l,m}(ω_L)
#  i.e. the left-side value.  The identity should then read:
#    Rup_{l,m}(ω_L) − Rup_{l,m}(ω_R) = −K_{l,m}(ω_R) · R^down_{l,m}(ω_R)
# ══════════════════════════════════════════════════════════════════
    println()
    println("="^70)
    println("Test 4: Kerr a=0.9 — ratio should approach 1")
    println("        For Kerr, m enters the radial eq. via angular eigenvalue,")
    println("        so Rup_{l,-m}(ω_R)* = Rup_{l,m}(ω_L) ← true L/R discontinuity")
    println("="^70)

    a_kerr = parse(BigFloat, "0.9")

    r_test = BigFloat(6)
    δ = parse(BigFloat, "1e-6")
    for σ_f64 in [0.3, 0.5, 0.8]
        σ = parse(BigFloat, string(σ_f64))
        res = compute_ratio(s, l, m, a_kerr, σ, δ, r_test; nmax=nmax)
        rat = res.ratio
        @printf("σ=%.1f  ratio = %+.6e %+.6ei  |ratio−1| = %.3e\n",
                σ_f64, Float64(real(rat)), Float64(imag(rat)),
                Float64(abs(rat - 1)))
        @printf("        ΔR^up=%+.3e%+.3ei   -K·Rdown=%+.3e%+.3ei\n",
                Float64(real(res.ΔRup)), Float64(imag(res.ΔRup)),
                Float64(real(res.KRdown)), Float64(imag(res.KRdown)))
    end

    println()
    println("δ convergence for Kerr a=0.9, σ=0.5:")
    σ = parse(BigFloat, "0.5")
    for δ_exp in [-3, -4, -5, -6, -7, -8]
        δ_bf = parse(BigFloat, "1e$(δ_exp)")
        res = compute_ratio(s, l, m, a_kerr, σ, δ_bf, r_test; nmax=nmax)
        rat = res.ratio
        @printf("  δ=1e%+d  ratio = %+.6e %+.6ei  |ratio−1| = %.3e\n",
                δ_exp, Float64(real(rat)), Float64(imag(rat)),
                Float64(abs(rat - 1)))
    end
end

main()
