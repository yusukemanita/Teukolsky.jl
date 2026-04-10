using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Printf

# ============================================================
#  Tests for K(ω) — three independent checks
#
#  Background: In Leaver's notation (ω-frequency, Schwarzschild, s=-2):
#    R^up   → purely outgoing at ∞  (computed by Rup in the code)
#    R^down → purely ingoing  at ∞  = (Rin - Bref × Rup) / Binc
#             Note: Rup(-ν-1) is also outgoing at ∞, NOT R^down!
#    K defined by: W[R^up, R^in](ω e^{2πi}) = W[R^up, R^in](ω) × (1 + K·Bref/Binc)
#
#  Test A: W[Rup, Rin] is r-independent
#    Confirms Rup and Rin solve the same ODE.
#    Passes if δ < 1e-8 for r ∈ [4, 20].
#
#  Test B: K = 0 when ν − ε̃ ∈ ℤ  (monodromy factor vanishes)
#    Search numerically for ω where Re(ν - ε) ≈ 1 (the first zero above ω=0).
#    Pass: |K| / max(|mono_factor|, 1e-10) << 1  near that ω.
#
#  Test C: Low-frequency limit K(ω → 0) ≈ −2πiε e^{iℓπ}
#    Already passed in test_monodromy_K.jl, included here for completeness.
# ============================================================

s, l, m = -2, 2, 2
nmax    = 40

Δ_BL(r, a) = r^2 - 2r + a^2

function wronskian_up_in(p, ν, fn, r; nmax=40)
    f1  = Rup(p, ν, fn, r; nmax=nmax)
    f1p = dRup(p, ν, fn, r; nmax=nmax)
    f2  = Rin(p, ν, fn, r; nmax=nmax)
    f2p = dRin(p, ν, fn, r; nmax=nmax)
    Δ_r = Δ_BL(r, p.a)^(p.s + 1)
    return Δ_r * (f1*f2p - f1p*f2)
end

# ── Test A: W[Rup, Rin] is r-independent ─────────────────────────────────────
println("="^70)
println("Test A: W[Rup, Rin] is constant in r")
println("  W = Δ^{s+1}(Rup·Rin' - Rup'·Rin)")
println("="^70)

# Note: Rin uses 2F1 which converges well for small |x| = |(r+-r)/(2κ)|.
# For Schwarzschild a=0: r+ = 2, x = (2-r)/2 → convergence good for r ≲ 10.
# Test at r ∈ [4, 8] where BOTH Rup (HU) and Rin (2F1) converge well.
a_test = 0.0
for ω_val in [0.1, 0.2, 0.3]
    ω    = complex(ω_val)
    ν, p = compute_nu(s, l, m, a_test, ω)
    fn   = compute_fn(p, ν; nmax=nmax)

    r_ref = 4.0
    W_ref = wronskian_up_in(p, ν, fn, r_ref; nmax=nmax)
    max_δ = 0.0
    for r in [5.0, 6.0, 7.0, 8.0]
        W = wronskian_up_in(p, ν, fn, r; nmax=nmax)
        δ = abs(W - W_ref) / abs(W_ref)
        max_δ = max(max_δ, δ)
    end
    status = max_δ < 1e-5 ? "PASS" : "FAIL"
    @printf("  a=0.0 ω=%.2f  ν=%.5f  max_δ=%.2e  [%s]\n", ω_val, real(ν), max_δ, status)
end

# Kerr: r+ = 1+√(1-a²) ≈ 1.436 for a=0.9, κ≈0.436
for (a_val, ω_val) in [(0.9, 0.2), (0.9, 0.25)]
    ω    = complex(ω_val)
    ν, p = compute_nu(s, l, m, a_val, ω)
    fn   = compute_fn(p, ν; nmax=nmax)

    r_ref = 3.0
    W_ref = wronskian_up_in(p, ν, fn, r_ref; nmax=nmax)
    max_δ = 0.0
    for r in [4.0, 5.0, 6.0, 7.0]
        W = wronskian_up_in(p, ν, fn, r; nmax=nmax)
        δ = abs(W - W_ref) / abs(W_ref)
        max_δ = max(max_δ, δ)
    end
    status = max_δ < 1e-5 ? "PASS" : "FAIL"
    @printf("  a=%.1f ω=%.2f  ν=%.5f%+.3fi  max_δ=%.2e  [%s]\n",
            a_val, ω_val, real(ν), imag(ν), max_δ, status)
end

# ── Test B: K = 0 when ν − ε ∈ ℤ ────────────────────────────────────────────
println()
println("="^70)
println("Test B: K = 0 when ν − ε ∈ ℤ  (monodromy factor vanishes)")
println("  For Schwarzschild, ν decreases from l=2 as ω increases.")
println("  Search for ω* where Re(ν − ε) ≈ 1  (first zero of mono_factor)")
println("="^70)

# Scan ω and look at mono_factor = 1 - exp(2πi(ε-ν))
println("ω scan:")
@printf("  %-6s  %-10s  %-12s  %-10s  %-10s\n",
        "ω", "ν", "Re(ν-ε)", "Im(ν-ε)", "|mono|")
ω_scan = range(0.05, 0.5; step=0.05)
local prev_re_val = nothing
for ω_val in ω_scan
    ω    = complex(ω_val)
    ν, p = compute_nu(s, l, m, 0.0, ω)
    ε    = p.ϵ
    nu_minus_eps = ν - ε
    mono = 1 - exp(2π * im * (ε - ν))
    @printf("  %-6.3f  %-10.5f  %-12.5f  %-10.5f  %-10.4e\n",
            ω_val, real(ν), real(nu_minus_eps), imag(nu_minus_eps), abs(mono))
    prev_re_val = real(nu_minus_eps)
end

# Bisection to find ω* where Re(ν-ε) = 1  (between 0.30 and 0.35)
println()
println("Bisection for ω* where Re(ν-ε) = 1:")
function bisect_nu_eps(s, l, m, ω_lo0, ω_hi0; niter=60)
    lo, hi = ω_lo0, ω_hi0
    for _ in 1:niter
        mid = (lo + hi) / 2
        ν_m, p_m = compute_nu(s, l, m, 0.0, complex(mid))
        if real(ν_m - p_m.ϵ) - 1 > 0
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end
ω_star = bisect_nu_eps(s, l, m, 0.30, 0.35)
ν_star, p_star = compute_nu(s, l, m, 0.0, complex(ω_star))
res_K_star = compute_monodromy_K(s, l, m, 0.0, complex(ω_star))
mono_star  = 1 - exp(2π * im * (p_star.ϵ - ν_star))
@printf("  ω* = %.8f\n", ω_star)
@printf("  ν* = %.8f + %.2ei\n", real(ν_star), imag(ν_star))
@printf("  Re(ν*-ε*) = %.2e  (should ≈ 1)\n", real(ν_star - p_star.ϵ) - 1)
@printf("  |mono_factor| = %.4e  (should ≈ 0)\n", abs(mono_star))
@printf("  |K|           = %.4e  (should ≈ 0)\n", abs(res_K_star.K))
status_B = abs(res_K_star.K) < 1e-3 ? "PASS" : "FAIL"
@printf("  [%s]\n", status_B)

# Show K as a function of ω near ω*
println()
println("  K near the zero (ω* ± 0.02):")
for ω_val in [ω_star - 0.04, ω_star - 0.02, ω_star, ω_star + 0.02, ω_star + 0.04]
    res = compute_monodromy_K(s, l, m, 0.0, complex(ω_val))
    @printf("    ω=%.5f  |K|=%.4e  Re(ν-ε)-1=%+.4e\n",
            ω_val, abs(res.K), real(res.ν - res.p.ϵ) - 1)
end

# ── Test C: Low-frequency limit ───────────────────────────────────────────────
println()
println("="^70)
println("Test C: K(ω→0) / (−2πiε e^{iℓπ}) → 1  (ε = 2ω, M=1)")
println("="^70)

for ω_val in [0.01, 0.001, 0.0001]
    res = compute_monodromy_K(s, l, m, 0.0, complex(ω_val))
    K   = res.K
    ε   = 2ω_val
    K_expected = -2π * im * ε * exp(im * π * l)
    ratio = K / K_expected
    status_C = abs(ratio - 1) < 20 * sqrt(ω_val) ? "PASS" : "FAIL"
    @printf("  ω=%.4f  ratio=%.6f%+.6fi  [%s]\n",
            ω_val, real(ratio), imag(ratio), status_C)
end
