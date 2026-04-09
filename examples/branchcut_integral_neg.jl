using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Plots
using LaTeXStrings
using Printf

# ============================================================
#  Branch cut integral along the positive imaginary ω-axis (t < 0)
#
#  For t < 0, the Fourier contour closes in the upper half ω-plane.
#  G(ω) has a branch cut at ω = +iσ (σ > 0), giving:
#
#    ψ_BC(t) = (1/2π) ∫₀^∞ ΔG_+(σ) e^{+σt} dσ     (t < 0)
#
#  where ΔG_+(σ) = G_R(+iσ) - G_L(+iσ) is the discontinuity
#  across the cut (right minus left side).
#
#  Symmetry: B_{lm}(-δ+iσ) = conj(B_{l,-m}(+δ+iσ))
#    because -ω_L* = -(−δ+iσ)* = δ+iσ = ω_R.
#  So we evaluate (s,l,m) and (s,l,-m) at the SAME point ω_R = δ+iσ.
# ============================================================

# ── パラメータ ────────────────────────────────────────────────
s, l, m = -2, 2, 2
# a_bf  = BigFloat("0.0")
# δ     = BigFloat("1e-6")   # ブランチカットからのオフセット
# σ_max = BigFloat("5")      # σ の積分上限
a_bf = 0.0
δ    = 1e-6   # ブランチカットからのオフセット
σ_max = 5.0   # σ の積分上限
Nσ    = 1000                # σ のグリッド点数
Nt    = 500                # 時刻グリッド点数
t_ini = -100.0             # 時刻の始点 (t < 0)
t_max = -1.0               # 時刻の終点 (t < 0)

# ── σ グリッド ────────────────────────────────────────────────
Δσ     = σ_max / Nσ
σ_grid = range(Δσ, σ_max; length=Nσ)   # σ=0 を避ける

# ── ΔG_+(σ) = G_R(+iσ) - G_L(+iσ) を計算 ───────────────────
println("ΔG_+(σ) を計算中 ($Nσ 点, BigFloat) ...")
ΔG = Vector{Complex{BigFloat}}(undef, Nσ)

for (i, σ) in enumerate(σ_grid)
    # 右側: ω_R = +δ + iσ  (Re(ω) > 0 側)
    ω_R   = δ + im * σ
    amp_R = compute_amplitudes(s, l, m, a_bf, ω_R)
    G_R   = amp_R.Bref / (2im * ω_R * amp_R.Binc)

    # 左側: ω_L = -δ + iσ  (Re(ω) < 0 側)
    # -ω_L* = δ + iσ = ω_R なので B_{lm}(ω_L) = conj(B_{l,-m}(ω_R))
    amp_mirror = compute_amplitudes(s, l, -m, a_bf, ω_R)
    Binc_L = conj(amp_mirror.Binc)
    Bref_L = conj(amp_mirror.Bref)
    ω_L    = -δ + im * σ
    G_L    = Bref_L / (2im * ω_L * Binc_L)

    ΔG[i] = G_R - G_L
    i % 10 == 0 && @printf("  σ=%.2f  |ΔG_+|=%.3e\n", Float64(σ), Float64(abs(ΔG[i])))
end
println("完了")

# ── 時刻積分: ψ_BC(t) = (Δσ/2π) Σ_i ΔG_+(σ_i) e^{+σ_i t}  (t < 0) ──
t_grid = range(t_ini, t_max; length=Nt)
ψ_BC   = Vector{ComplexF64}(undef, Nt)

println("時刻積分中 ($Nt 点) ...")
for (k, t) in enumerate(t_grid)
    s_val = zero(Complex{BigFloat})
    for i in 1:Nσ
        s_val += ΔG[i] * exp(σ_grid[i] * BigFloat(t))   # +σt, t<0 → 収束
    end
    ψ_BC[k] = ComplexF64(Δσ / (2π) * s_val)
    k % 50 == 0 && (print("."); flush(stdout))
end
println("\n完了")

# ── プロット ──────────────────────────────────────────────────
σ_f64  = Float64.(collect(σ_grid))
ΔG_abs = Float64.(abs.(ΔG))
t_f64  = collect(t_grid)

# 1. 被積分関数 |ΔG_+(σ)| * exp(σt) for いくつかの t < 0
t_plot_vals = [-1.0, -3.0, -5.0, -10.0, -20.0, -50.0]
p1 = plot(
    xlabel     = L"\sigma",
    ylabel     = L"|\Delta G_+(\sigma)|\, e^{\sigma t}",
    yscale     = :log10,
    ylim       = (1e-15, 1e3),
    title      = "Branch cut integrand (upper, t<0)  (s=$s, l=$l, m=$m, a=$(Float64(a_bf)))",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 120,
    legend     = :topright)

for t in t_plot_vals
    integrand = ΔG_abs .* exp.(σ_f64 .* t)   # exp(σt) with t<0
    plot!(p1, σ_f64, integrand, label = L"t = %$t", lw = 1.5)
end

# 2. ψ_BC(t) — |t| でlog-log プロット
abs_t = abs.(t_f64)
p2 = plot(t_f64, abs.(real.(ψ_BC)),
    xlabel     = L"|t|\ [M]",
    ylabel     = L"|\Re[\psi_{\mathrm{BC}}(t)]|",
    label      = L"|\Re[\psi_{\mathrm{BC}}]|",
    lw         = 1.5,
    # xscale     = :log10,
    yscale     = :log10,
    title      = "Branch cut contribution, t < 0",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 120)

fig = plot(p1, p2, layout = (2, 1), size = (800, 900))
savefig(fig, "branchcut_integral_neg.png")
println("branchcut_integral_neg.png を保存しました")
fig
