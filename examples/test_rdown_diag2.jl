using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")
using Teukolsky
using Printf

s, l, m, a = -2, 2, 2, 0.0
ω = 0.3

ν, p = compute_nu(s, l, m, a, ω)
fn = compute_fn(p, ν; nmax=40)
amp = compute_amplitudes(s, l, m, a, ω; nmax=40)

r_vals = [4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 50.0]
δr = 1e-6

# For Teukolsky equation: Δ^{s+1} W = const, so W ∝ 1/Δ^{s+1}
# s=-2 → W ∝ Δ^{-(-1)} = Δ
# Check: W/Δ should be constant

println("="^90)
println("Wronskian W / Δ  (should be r-independent for s=-2)")
println("="^90)
@printf("%-6s | %-30s | %-30s | %-30s\n", "r", "W[Rin,Rup]/Δ", "W[Rin,R+]/Δ", "W[R+,R-]/Δ")
println("-"^90)

# Get norm for R^ν_+
Ap = amp.Ap
ω_c = p.ω
ϵ = p.ϵ
κ = p.κ
phase_norm = exp(-im * (ϵ * log(ϵ) - (1 - κ) / 2 * ϵ))
norm_val = Ap * ω_c^(-1) * phase_norm

for r in r_vals
    Δ = r^2 - 2r + a^2

    Rin_val = Rin(p, ν, fn, r)
    Rup_val = Rup(p, ν, fn, r)
    Rd = Rdown(p, ν, fn, r)
    Rp_raw = Rd * norm_val  # R^ν_+ raw

    dRin = (Rin(p, ν, fn, r+δr) - Rin(p, ν, fn, r-δr)) / (2δr)
    dRup = (Rup(p, ν, fn, r+δr) - Rup(p, ν, fn, r-δr)) / (2δr)
    dRp_raw = (Rdown(p, ν, fn, r+δr) - Rdown(p, ν, fn, r-δr)) / (2δr) * norm_val

    W_in_up = (Rin_val * dRup - Rup_val * dRin) / Δ
    W_in_rp = (Rin_val * dRp_raw - Rp_raw * dRin) / Δ
    W_rp_rm = (Rp_raw * dRup - Rup_val * dRp_raw) / Δ

    @printf("r=%4.0f | %+.6e%+.6ei | %+.6e%+.6ei | %+.6e%+.6ei\n",
            r, real(W_in_up), imag(W_in_up), real(W_in_rp), imag(W_in_rp),
            real(W_rp_rm), imag(W_rp_rm))
end

# Decompose Rin = c1 * R^ν_+ + c2 * R^ν_-
println()
println("="^90)
println("Decomposition: Rin = c1 * R^ν_+ + c2 * Rup")
println("c1 = W[Rin, Rup] / W[R+, Rup],  c2 = -W[Rin, R+] / W[R+, Rup]")
println("="^90)
for r in r_vals
    Δ = r^2 - 2r + a^2

    Rin_val = Rin(p, ν, fn, r)
    Rup_val = Rup(p, ν, fn, r)
    Rp_raw = Rdown(p, ν, fn, r) * norm_val

    dRin = (Rin(p, ν, fn, r+δr) - Rin(p, ν, fn, r-δr)) / (2δr)
    dRup = (Rup(p, ν, fn, r+δr) - Rup(p, ν, fn, r-δr)) / (2δr)
    dRp_raw = (Rdown(p, ν, fn, r+δr) - Rdown(p, ν, fn, r-δr)) / (2δr) * norm_val

    W_in_up = Rin_val * dRup - Rup_val * dRin
    W_in_rp = Rin_val * dRp_raw - Rp_raw * dRin
    W_rp_up = Rp_raw * dRup - Rup_val * dRp_raw

    c1 = W_in_up / W_rp_up
    c2 = -W_in_rp / W_rp_up

    # Verify: Rin - c1*R+ - c2*R- should be ~0
    residual = Rin_val - c1 * Rp_raw - c2 * Rup_val

    @printf("r=%4.0f  c1=%+.8e%+.8ei  c2=%+.8e%+.8ei  |res|/|Rin|=%.2e\n",
            r, real(c1), imag(c1), real(c2), imag(c2), abs(residual)/abs(Rin_val))
end

# Compare c1 to Binc/norm and c2 to Bref/Ctrans
println()
println("Expected: c1 ≈ Binc/norm  (since Rdown = R+/norm, R+ = norm*Rdown)")
c1_expected = amp.Binc / norm_val
println("Binc / norm = $c1_expected")
c2_expected = amp.Bref / amp.Ctrans
println("Bref / Ctrans = $c2_expected")

# Check with actual from Rin decomposition at r=10
r = 10.0
Δ = r^2 - 2r + a^2
Rin_val = Rin(p, ν, fn, r)
Rup_val = Rup(p, ν, fn, r)
Rp_raw = Rdown(p, ν, fn, r) * norm_val
dRin = (Rin(p, ν, fn, r+δr) - Rin(p, ν, fn, r-δr)) / (2δr)
dRup = (Rup(p, ν, fn, r+δr) - Rup(p, ν, fn, r-δr)) / (2δr)
dRp_raw = (Rdown(p, ν, fn, r+δr) - Rdown(p, ν, fn, r-δr)) / (2δr) * norm_val
W_in_up = Rin_val * dRup - Rup_val * dRin
W_in_rp = Rin_val * dRp_raw - Rp_raw * dRin
W_rp_up = Rp_raw * dRup - Rup_val * dRp_raw
c1_actual = W_in_up / W_rp_up
c2_actual = -W_in_rp / W_rp_up

println()
println("From Wronskian at r=10:")
println("  c1 = $c1_actual")
println("  c2 = $c2_actual")
println("  c1/c1_expected = $(c1_actual / c1_expected)")
println("  c2/c2_expected = $(c2_actual / c2_expected)")
