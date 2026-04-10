using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Printf

# ============================================================
#  Sanity checks for compute_monodromy_K
#
#  1. Low-frequency limit: K(ω→0) ≈ -2πiω e^{iℓπ}  (Schwarzschild)
#  2. K=0 when ν - ε̃ ∈ ℤ  (monodromy factor vanishes)
#  3. Compare |ΔG_direct| vs |ΔG_K| along the neg. imag. axis
#     (K provides an alternative route to the branch cut discontinuity)
# ============================================================

s, l, m = -2, 2, 2
a       = 0.0   # Schwarzschild for exact checks

println("="^60)
println("Test 1: Low-frequency limit  K(ω→0) ≈ -2πiε e^{iℓπ}  (ε = 2Mω, M=1)")
println("  i.e. K ≈ -4πiω e^{iℓπ}  in terms of ω")
println("="^60)
for ω in [0.01, 0.001, 0.0001]
    res = compute_monodromy_K(s, l, m, a, complex(ω))
    K   = res.K
    ε   = 2ω   # M = 1
    # mono_factor ≈ -2πiε at small ε; branch_factor → 1; N₊/N₋ → e^{iℓπ}
    K_expected = -2π * im * ε * exp(im * π * l)
    ratio = K / K_expected
    @printf("  ω = %.4f:  K = %.4e%+.4ei  ratio = %.6f%+.6fi\n",
            ω, real(K), imag(K), real(ratio), imag(ratio))
end

println()
println("="^60)
println("Test 2: K along the real axis (several ω)")
println("="^60)
println("  ω\t\t\t|K|\t\t\targ(K)/π")
for ω_val in [0.1, 0.2, 0.3, 0.4, 0.5]
    ω = complex(ω_val)
    res = compute_monodromy_K(s, l, m, a, ω)
    K = res.K
    @printf("  %.2f\t\t\t%.6e\t\t%.4f\n", ω_val, abs(K), angle(K)/π)
end

println()
println("="^60)
println("Test 3: K on the negative imaginary axis (branch cut)")
println("  Compare with direct ΔG = G_R(-iσ) - G_L(-iσ)")
println("="^60)
δ = 1e-6

println("  σ\t|K|\t\t|ΔG_direct|")
for σ in [0.5, 1.0, 1.5, 2.0, 2.5]
    ω_neg = -im * σ + δ
    res   = compute_monodromy_K(s, l, m, a, ω_neg)
    K     = res.K

    # Direct ΔG
    amp_R  = compute_amplitudes(s, l,  m, a, complex(δ - im*σ))
    G_R    = amp_R.Bref / (2im * (δ - im*σ) * amp_R.Binc)
    amp_mir= compute_amplitudes(s, l, -m, a, complex(δ + im*σ))
    G_L    = conj(amp_mir.Bref) / (2im * (-δ - im*σ) * conj(amp_mir.Binc))
    ΔG_direct = G_R - G_L

    @printf("  %.2f\t%.4e\t%.4e\n", σ, abs(K), abs(ΔG_direct))
end

println()
println("="^60)
println("Test 4: Spin dependence (a = 0.9, Kerr)")
println("="^60)
a_kerr = 0.9
for ω_val in [0.2, 0.3, 0.4]
    ω = complex(ω_val)
    res = compute_monodromy_K(s, l, m, a_kerr, ω)
    K = res.K
    @printf("  ω=%.2f  |K|=%.4e  arg/π=%.4f  ν=%+.4f%+.4fi\n",
            ω_val, abs(K), angle(K)/π, real(res.ν), imag(res.ν))
end
