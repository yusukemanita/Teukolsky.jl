using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Printf
using Plots
using LaTeXStrings

# ============================================================
#  Test: Branch-cut structure of R^up in the MST implementation
#
#  The monodromy relation (Leaver 1986 eq. 31) states:
#    R^up(ωe^{2πi}) = R^up(ω) − K(ω) R^down(ω)
#
#  Numerically, R^up(ω) computed via MST with Julia's principal-branch
#  log distributes the branch cut across TWO rays from ω = 0:
#    (i)  Negative imaginary ω-axis: from U(a,b,c) with c = −2iẑ
#    (ii) Negative real ω-axis: from prefactor ẑ^α
#
#  The FULL monodromy K·R^down requires going around ω = 0 (crossing
#  both cuts). Crossing only one cut gives a partial discontinuity.
#
#  Tests below verify:
#    1. ΔR^up / (K·R^down) is r-independent (proportionality to R^down)
#    2. ΔR^up / (K·R^down) is δ-independent (finite discontinuity)
#    3. Negative-real-axis cut gives a DIFFERENT (larger) contribution
# ============================================================

s, l, m = -2, 2, 2
a_val   = 0.0
nmax    = 40

function compute_ratio_imag_axis(s, l, m, a_val, σ, δ, r; nmax=40)
    # Cross the negative imaginary ω-axis: ω_R = δ−iσ vs ω_L = −δ−iσ
    ω_R = complex(δ - im*σ)
    ω_L = complex(-δ - im*σ)

    ν_R, p_R = compute_nu(s, l, m, a_val, ω_R)
    fn_R = compute_fn(p_R, ν_R; nmax=nmax)
    amp_R = compute_amplitudes(s, l, m, a_val, ω_R; nmax=nmax)

    Rup_R = Rup(p_R, ν_R, fn_R, r; nmax=nmax)
    Rin_R = Rin(p_R, ν_R, fn_R, r; nmax=nmax)
    Rdown = (Rin_R - amp_R.Binc * Rup_R) / amp_R.Bref
    K_val = compute_monodromy_K(s, l, m, a_val, ω_R).K

    ν_L, p_L = compute_nu(s, l, m, a_val, ω_L)
    fn_L = compute_fn(p_L, ν_L; nmax=nmax)
    Rup_L = Rup(p_L, ν_L, fn_L, r; nmax=nmax)

    ΔRup = Rup_L - Rup_R
    return ΔRup / (K_val * Rdown)
end

function compute_ratio_real_axis(s, l, m, a_val, σ_real, δ, r; nmax=40)
    # Cross the negative real ω-axis: ω_above = −σ+iδ vs ω_below = −σ−iδ
    ω_above = complex(-σ_real + im*δ)
    ω_below = complex(-σ_real - im*δ)

    ν_a, p_a = compute_nu(s, l, m, a_val, ω_above)
    fn_a = compute_fn(p_a, ν_a; nmax=nmax)
    amp_a = compute_amplitudes(s, l, m, a_val, ω_above; nmax=nmax)

    Rup_a = Rup(p_a, ν_a, fn_a, r; nmax=nmax)
    Rin_a = Rin(p_a, ν_a, fn_a, r; nmax=nmax)
    Rdown = (Rin_a - amp_a.Binc * Rup_a) / amp_a.Bref
    K_val = compute_monodromy_K(s, l, m, a_val, ω_above).K

    ν_b, p_b = compute_nu(s, l, m, a_val, ω_below)
    fn_b = compute_fn(p_b, ν_b; nmax=nmax)
    Rup_b = Rup(p_b, ν_b, fn_b, r; nmax=nmax)

    ΔRup = Rup_a - Rup_b
    return ΔRup / (K_val * Rdown)
end

# ── Test 1: r-independence of ratio (negative imaginary axis) ────────
println("="^70)
println("Test 1: ΔR^up(r) / (K·R^down(r)) across neg. imag. axis")
println("        Ratio should be constant in r (proportional to R^down)")
println("="^70)

r_vals = [4.0, 6.0, 8.0, 10.0, 15.0]
for σ in [0.3, 0.5, 0.8]
    δ = 1e-5
    ratios = [compute_ratio_imag_axis(s, l, m, a_val, σ, δ, r) for r in r_vals]
    @printf("σ=%.1f δ=%.0e: ", σ, δ)
    for (i, r) in enumerate(r_vals)
        @printf("r=%g→%.4e%+.4ei  ", r, real(ratios[i]), imag(ratios[i]))
    end
    spread = maximum(abs.(ratios .- ratios[1])) / abs(ratios[1])
    @printf("\n  max spread = %.2e  [%s]\n", spread, spread < 1e-2 ? "PASS" : "FAIL")
end

# ── Test 2: δ-independence (negative imaginary axis) ──────────────────
println()
println("="^70)
println("Test 2: |ratio| converges as δ→0 (finite discontinuity from U branch cut)")
println("="^70)
r = 6.0
for σ in [0.3, 0.5, 0.8, 1.0]
    @printf("σ=%.1f:\n", σ)
    for δ in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        ratio = compute_ratio_imag_axis(s, l, m, a_val, σ, δ, r)
        @printf("  δ=%.0e  |ratio|=%.4e  ratio=%+.4e%+.4ei\n",
                δ, abs(ratio), real(ratio), imag(ratio))
    end
end

# ── Test 3: Negative real axis cut (prefactor contribution) ──────────
println()
println("="^70)
println("Test 3: ΔR^up across neg. REAL axis (prefactor branch cut)")
println("        Shows that the prefactor cut gives a DIFFERENT, larger contribution")
println("="^70)
for σ_real in [0.3, 0.5, 0.8]
    @printf("ω_real=%.1f:\n", σ_real)
    for δ in [1e-3, 1e-4, 1e-5]
        ratio_r = compute_ratio_real_axis(s, l, m, a_val, σ_real, δ, r)
        ratio_i = compute_ratio_imag_axis(s, l, m, a_val, σ_real, δ, r)
        @printf("  δ=%.0e  |ratio_real|=%.4e  |ratio_imag|=%.4e  (ratio: %.1f×)\n",
                δ, abs(ratio_r), abs(ratio_i), abs(ratio_r)/abs(ratio_i))
    end
end

# ── Plots ────────────────────────────────────────────────────────────
println("\nGenerating plots...")
gr()

# --- Plot 1: |ratio| vs σ for the two cuts ---
σ_scan = collect(range(0.1, 1.2; length=30))
δ_plot = 1e-5

ratio_imag = [abs(compute_ratio_imag_axis(s, l, m, a_val, σ, δ_plot, 6.0)) for σ in σ_scan]
ratio_real = [abs(compute_ratio_real_axis(s, l, m, a_val, σ, δ_plot, 6.0)) for σ in σ_scan]

p1 = plot(σ_scan, ratio_imag, label="Neg. imag. axis (U cut)",
          lw=2, color=:blue, yscale=:log10,
          xlabel=L"\sigma\ (=|\mathrm{Im}\,\omega|)", ylabel=L"|\Delta R^{\mathrm{up}}| / |K R^{\mathrm{down}}|",
          title="Partial discontinuities: two branch cuts of MST R^up",
          framestyle=:box, dpi=150, legend=:topleft)
plot!(p1, σ_scan, ratio_real, label="Neg. real axis (prefactor cut)",
      lw=2, color=:red, ls=:dash)

# --- Plot 2: δ-convergence for several σ ---
δ_vals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
p2 = plot(xlabel=L"\delta", ylabel=L"|\Delta R^{\mathrm{up}}| / |K R^{\mathrm{down}}|",
          title="Convergence of partial discontinuity (neg. imag. axis)",
          xscale=:log10, framestyle=:box, dpi=150, legend=:topright)
colors = [:blue, :red, :green]
for (i, σ) in enumerate([0.3, 0.5, 0.8])
    ratios_δ = [abs(compute_ratio_imag_axis(s, l, m, a_val, σ, δ, 6.0)) for δ in δ_vals]
    plot!(p2, δ_vals, ratios_δ, label=L"\sigma=%$σ", lw=2, color=colors[i], marker=:circle)
end

# --- Plot 3: r-independence ---
r_plot = collect(range(3.5, 15.0; length=40))
p3 = plot(xlabel=L"r\ [M]",
          ylabel=L"\Delta R^{\mathrm{up}} / (K R^{\mathrm{down}})",
          title="r-independence of ratio (neg. imag. axis, δ=10⁻⁵)",
          framestyle=:box, dpi=150, legend=:topright)
for (i, σ) in enumerate([0.3, 0.5, 0.8])
    ratios_r = [compute_ratio_imag_axis(s, l, m, a_val, σ, 1e-5, r) for r in r_plot]
    plot!(p3, r_plot, real.(ratios_r), label=L"\mathrm{Re},\ \sigma=%$σ", lw=2, color=colors[i])
    plot!(p3, r_plot, imag.(ratios_r), label=L"\mathrm{Im},\ \sigma=%$σ", lw=2, ls=:dash, color=colors[i])
end

fig = plot(p1, p2, p3, layout=(3, 1), size=(900, 1300))
savefig(fig, "monodromy_Rup_test.png")
println("Saved: monodromy_Rup_test.png")
fig
