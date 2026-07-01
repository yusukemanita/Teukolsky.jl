using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")
using Teukolsky
using Printf

# Test parameters: Schwarzschild, s=-2, l=2, m=2, ω=0.3
s, l, m, a = -2, 2, 2, 0.0
ω = 0.3

ν, p = compute_nu(s, l, m, a, ω)
fn = compute_fn(p, ν; nmax=40)
amp = compute_amplitudes(s, l, m, a, ω; nmax=40)

println("ν = $ν")
println("Binc = $(amp.Binc)")
println("Bref = $(amp.Bref)")
println("Ctrans = $(amp.Ctrans)")
println("Ap = $(amp.Ap)")
println()

# Test at multiple r values
r_vals = [4.0, 6.0, 8.0, 10.0, 15.0, 20.0]

println("="^90)
println("Test 1: Rin/Rup Wronskian (should be ∝ Δ^{-s} = Δ^2)")
println("="^90)
δr = 1e-6
for r in r_vals
    Rin_val = Rin(p, ν, fn, r)
    Rup_val = Rup(p, ν, fn, r)
    dRin = (Rin(p, ν, fn, r+δr) - Rin(p, ν, fn, r-δr)) / (2δr)
    dRup = (Rup(p, ν, fn, r+δr) - Rup(p, ν, fn, r-δr)) / (2δr)
    W = Rin_val * dRup - Rup_val * dRin
    Δ = r^2 - 2r + a^2
    @printf("r=%5.1f  W = %+.8e %+.8ei  W/Δ^2 = %+.8e %+.8ei\n",
            r, real(W), imag(W), real(W/Δ^2), imag(W/Δ^2))
end

println()
println("="^90)
println("Test 2: R^ν_+ raw (radial_down.jl) at multiple r")
println("="^90)
for r in r_vals
    Rd = Rdown(p, ν, fn, r)
    @printf("r=%5.1f  Rdown = %+.8e %+.8ei  |Rdown| = %.8e\n",
            r, real(Rd), imag(Rd), abs(Rd))
end

println()
println("="^90)
println("Test 3: 'Expected' Rdown from Rin and Rup")
println("  Rdown_expected = (Rin - Binc*Rup) / Bref")
println("  Rup is transmission-normalized: Rup ~ r^{-1-2s} e^{+iωr*}")
println("="^90)
# At infinity (s=-2):
#   Rup  ~ r^{-1-2s} e^{+iωr*} = r^3 e^{+iωr*}   (outgoing, transmission-normalized)
#   Rdown ~ r^{-1}   e^{-iωr*}                     (ingoing)
#   Rin  ~ Binc × r^{-1} e^{-iωr*} + Bref × r^3 e^{+iωr*}
# Rin = some_linear_combination of R^ν_+ and R^ν_-
# From S&T: Rin = K^ν R^ν_C + K^{-ν-1} R^{-ν-1}_C
# where R^ν_C is the Coulomb wave function series (NOT R^ν_+ or R^ν_-)
# R^ν_C = Σ g_n F_{n+ν}(...) (regular Coulomb wave function)
# Using Φ = coeff1 × Ψ(+) + coeff2 × e^x Ψ(-), we get:
# R^ν_C = α_ν R^ν_+ + β_ν R^ν_-

# Let me take a different approach: compute the Wronskian W[Rin, R^ν_+_raw]
# If R^ν_+ is correct, W should be ∝ Δ^{-s}
println()
println("="^90)
println("Test 4: Wronskian W[Rin, R^ν_+_raw]  (should be ∝ Δ^{-s} = Δ^2)")
println("="^90)

# Need R^ν_+ raw (before dividing by norm)
# Modify: compute the raw R^ν_+ by multiplying Rdown by norm
Ap = amp.Ap
ω_c = p.ω
ϵ = p.ϵ
κ = p.κ
phase = exp(-im * (ϵ * log(ϵ) - (1 - κ) / 2 * ϵ))
norm_val = Ap * ω_c^(-1) * phase

for r in r_vals
    Rin_val = Rin(p, ν, fn, r)
    Rd = Rdown(p, ν, fn, r)
    Rp_raw = Rd * norm_val  # R^ν_+ raw

    dRin = (Rin(p, ν, fn, r+δr) - Rin(p, ν, fn, r-δr)) / (2δr)
    dRd = (Rdown(p, ν, fn, r+δr) - Rdown(p, ν, fn, r-δr)) / (2δr)
    dRp_raw = dRd * norm_val

    W = Rin_val * dRp_raw - Rp_raw * dRin
    Δ = r^2 - 2r + a^2
    @printf("r=%5.1f  W = %+.8e %+.8ei  W/Δ^2 = %+.8e %+.8ei\n",
            r, real(W), imag(W), real(W/Δ^2), imag(W/Δ^2))
end

# Test 5: Also compute W[R^ν_+, R^ν_-] (Rup raw)
println()
println("="^90)
println("Test 5: Wronskian W[R^ν_+_raw, R^ν_-_raw] (should be ∝ Δ^{-s})")
println("="^90)
for r in r_vals
    Rp_raw = Rdown(p, ν, fn, r) * norm_val
    Rm_raw = Rup(p, ν, fn, r)  # raw R^ν_-

    dRp = (Rdown(p, ν, fn, r+δr) - Rdown(p, ν, fn, r-δr)) / (2δr) * norm_val
    dRm = (Rup(p, ν, fn, r+δr) - Rup(p, ν, fn, r-δr)) / (2δr)

    W = Rp_raw * dRm - Rm_raw * dRp
    Δ = r^2 - 2r + a^2
    @printf("r=%5.1f  W = %+.8e %+.8ei  W/Δ^2 = %+.8e %+.8ei\n",
            r, real(W), imag(W), real(W/Δ^2), imag(W/Δ^2))
end

# Test 6: Compare HU values using recurrence vs hu_exact for c=+2iẑ
println()
println("="^90)
println("Test 6: HU recurrence vs exact for R^ν_+ (c = +2iẑ)")
println("="^90)
r = 6.0
rm = p.rm
zhat = complex(ϵ * (r - rm) / 2)
hp_plus = Teukolsky.HUParams(ν + 1 - s + im*ϵ, 2ν + 2, 2im * zhat)
hp_minus = Teukolsky.HUParams(p, ν, zhat)  # for Rup: c = -2iẑ

println("c_plus  = $(hp_plus.c)")
println("c_minus = $(hp_minus.c)")
println()

for n in -5:10
    hu_exact_plus = Teukolsky.hu_exact(hp_plus, n)
    hu_exact_minus = Teukolsky.hu_exact(hp_minus, n)
    @printf("n=%3d  |HU_+(exact)| = %.6e  |HU_-(exact)| = %.6e\n",
            n, abs(hu_exact_plus), abs(hu_exact_minus))
end
