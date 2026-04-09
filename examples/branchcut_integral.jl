using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Plots
using LaTeXStrings
using Printf

# ============================================================
#  Branch cut integral along the negative imaginary ω-axis
#
#  G(ω) = Bref / (2iω Binc) has a branch cut at ω = -iσ (σ>0).
#  The branch cut contribution to the time-domain waveform is:
#
#    ψ_BC(t) = (1/2π) ∫₀^∞ ΔG(σ) e^{-σt} dσ
#
#  where ΔG(σ) = G_R(-iσ) - G_L(-iσ) is the discontinuity
#  across the cut (right minus left side).
#
#  Left side via symmetry: B_{lm}(-ω) = B_{l,-m}(-ω*)^*
#  so G_L is obtained from compute_amplitudes(s, l, -m, a, δ+iσ).
# ============================================================

# 256-bit BigFloat: 128bitでは σ≈3 の消去誤差を回避できないため256bit必要
setprecision(BigFloat, 256)

# ── パラメータ ────────────────────────────────────────────────
s, l, m = -2, 2, 2
a_bf  = BigFloat("0.9")
δ     = BigFloat("1e-6")   # ブランチカットからのオフセット
σ_max = BigFloat("5")      # σ の積分上限 (BigFloat で σ>3 まで安定)
Nσ    = 100                # σ のグリッド点数
Nt    = 500                # 時刻グリッド点数
t_ini = 1.0                # 時刻の始点
t_max = 100.0              # 時刻の終点

# ── σ グリッド ────────────────────────────────────────────────
Δσ     = σ_max / Nσ
σ_grid = collect(range(Δσ, σ_max; length=Nσ))   # σ=0 を避ける

# ── ΔG(σ) = G_R(-iσ) - G_L(-iσ) を計算 (並列) ────────────────
println("ΔG(σ) を計算中 ($Nσ 点, $(precision(BigFloat))-bit BigFloat) ...")
ΔG = Vector{Complex{BigFloat}}(undef, Nσ)

for i in 1:Nσ
    σ = σ_grid[i]
    ω_R      = δ - im * σ
    amp_R    = compute_amplitudes(s, l, m, a_bf, ω_R)
    G_R      = amp_R.Bref / (2im * ω_R * amp_R.Binc)

    ω_mirror  = δ + im * σ
    amp_mirror = compute_amplitudes(s, l, -m, a_bf, ω_mirror)
    Binc_L   = conj(amp_mirror.Binc)
    Bref_L   = conj(amp_mirror.Bref)
    ω_L      = -δ - im * σ
    G_L      = Bref_L / (2im * ω_L * Binc_L)

    ΔG[i] = G_R - G_L
end
println("完了")
for i in 1:10:Nσ
    @printf("  σ=%.2f  |ΔG|=%.3e\n", Float64(σ_grid[i]), Float64(abs(ΔG[i])))
end

# ── 時刻積分: ψ_BC(t) = (Δσ/2π) Σ_i ΔG(σ_i) e^{-σ_i t} ────
t_grid = range(t_ini, t_max; length=Nt)
ψ_BC   = Vector{ComplexF64}(undef, Nt)

println("時刻積分中 ($Nt 点) ...")
for k in 1:Nt
    t = t_grid[k]
    s_val = zero(Complex{BigFloat})
    for i in 1:Nσ
        s_val += ΔG[i] * exp(-σ_grid[i] * BigFloat(t))
    end
    ψ_BC[k] = ComplexF64(im * Δσ / (2π) * s_val)   # 因子 i: Hankel contour の向き
end
println("完了")

# ── プロット ──────────────────────────────────────────────────
σ_f64  = Float64.(σ_grid)
ΔG_abs = Float64.(abs.(ΔG))
t_f64  = collect(t_grid)

# 1. 被積分関数 |ΔG(σ)| * exp(-σt) for いくつかの t
t_plot_vals = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0]
p1 = plot(
    xlabel     = L"\sigma",
    ylabel     = L"|\Delta G(\sigma)|\, e^{-\sigma t}",
    yscale     = :log10,
    ylim       = (1e-15, 1e3),
    title      = "Branch cut integrand  (s=$s, l=$l, m=$m, a=$(Float64(a_bf)))",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 120,
    legend     = :topright)

for t in t_plot_vals
    integrand = ΔG_abs .* exp.(-σ_f64 .* t)
    plot!(p1, σ_f64, integrand, label = L"t = %$t", lw = 1.5)
end

# 2. ψ_BC(t) — log-log で power-law tail を確認
p2 = plot(t_f64, abs.(real.(ψ_BC)),
    xlabel     = L"t\ [M]",
    ylabel     = L"|\Re[\psi_{\mathrm{BC}}(t)]|",
    label      = L"|\Re[\psi_{\mathrm{BC}}]|",
    lw         = 1.5,
    yscale     = :log10,
    title      = "Branch cut contribution (power-law tail)",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 120)

fig = plot(p1, p2, layout = (2, 1), size = (800, 900))
savefig(fig, "branchcut_integral.png")
println("branchcut_integral.png を保存しました")
fig
