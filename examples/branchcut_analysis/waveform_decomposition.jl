using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Plots
using LaTeXStrings
using Printf
using CSV
using DataFrames

# ============================================================
#  Branch cut decomposition of the Kerr waveform
#
#  ψ(t) = ψ_QNM(t) + ψ_BC⁻(t>0) + ψ_BC⁺(t<0) + (transient)
#
#  Numerical reference: high-precision real-axis integral (CSV)
#
#  Green's function:  G(ω) = B^ref / (2iω B^inc)
#
#  Branch cut (neg. imag. axis, t>0):
#    ψ_BC⁻(t) = (i/2π) ∫₀^σ_max ΔG⁻(σ) e^{-σt} dσ
#    ΔG⁻(σ) = G(+δ-iσ) - G(-δ-iσ)
#    [requires 256-bit BigFloat for σ ≳ 3 due to cancellation]
#
#  Branch cut (pos. imag. axis, t<0):
#    ψ_BC⁺(t) = (-i/2π) ∫₀^σ_max ΔG⁺(σ) e^{+σt} dσ
#    ΔG⁺(σ) = G(+δ+iσ) - G(-δ+iσ)
#    [Float64 stable via symmetry B_{lm}(ω_L) = conj(B_{l,-m}(ω_R))]
#
#  QNM (t>0):
#    ψ_QNM(t) = -Σ_{n,±m} B_n e^{-iω_n t}
# ============================================================

# ── helpers ──────────────────────────────────────────────────

function load_qnm_data(l::Int, m::Int, n::Int, a_target::Float64)
    filepath = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$n.dat"
    df  = CSV.read(filepath, DataFrame, header=false)
    idx = argmin(abs.(df[!, 1] .- a_target))
    ω = (df[idx, 2] + im * df[idx, 3]) / 2
    B = (df[idx, 8] + im * df[idx, 9]) / 16
    return ω, B
end

# ── parameters ───────────────────────────────────────────────

s, l, m = -2, 2, 2
a        = 0.9

# ── Part 1: 数値波形 (CSV) ────────────────────────────────────

println("=== Part 1: 数値波形 ===")
wf_df = CSV.read(
    "/Users/yusuke/work/matrix_pencile/data/psi4_waveform_l2m2a0.900.csv",
    DataFrame)
t_all = wf_df[!, 1]
ψ_num = wf_df[!, 2] .+ im .* wf_df[!, 3]
println("  $(length(t_all)) 点, t ∈ [$(t_all[1]), $(t_all[end])]")

# ── Part 2: ψ_BC⁺  正の虚軸 (t < 0, Float64) ────────────────

println("\n=== Part 2: ψ_BC⁺  正の虚軸ブランチカット (t<0) ===")
Nσ_p    = 300
σ_min_p = 1e-3
σ_max_p = 10.0
δ_f64   = 1e-6

σ_p   = exp.(range(log(σ_min_p), log(σ_max_p); length=Nσ_p))
Δσ_p  = diff([0.0; (σ_p[1:end-1] .+ σ_p[2:end]) ./ 2; σ_p[end]])
ΔG_p  = Vector{ComplexF64}(undef, Nσ_p)

for i in 1:Nσ_p
    σ      = σ_p[i]
    ω_R    = δ_f64 + im*σ
    amp_R  = compute_amplitudes(s, l,  m, a, ω_R)
    G_R    = amp_R.Bref / (2im * ω_R * amp_R.Binc)
    amp_m  = compute_amplitudes(s, l, -m, a, ω_R)
    ω_L    = -δ_f64 + im*σ
    G_L    = conj(amp_m.Bref) / (2im * ω_L * conj(amp_m.Binc))
    ΔG_p[i] = G_R - G_L
    i % 60 == 0 && (print("."); flush(stdout))
end
println(" 完了")

t_neg    = range(-100.0, -0.5; length=800)
ψ_BC_pos = [-im/(2π) * sum(ΔG_p[i]*Δσ_p[i]*exp(σ_p[i]*t) for i in 1:Nσ_p)
            for t in t_neg]

# ── Part 3: ψ_BC⁻  負の虚軸 (t > 0, 256-bit BigFloat) ───────

println("\n=== Part 3: ψ_BC⁻  負の虚軸ブランチカット (t>0, BigFloat) ===")
setprecision(BigFloat, 256)
a_bf    = BigFloat(string(a))
δ_bf    = BigFloat("1e-6")
Nσ_n    = 150
σ_min_n = BigFloat("1e-3")
σ_max_n = BigFloat("5")

σ_n_f64  = exp.(range(log(1e-3), log(5.0); length=Nσ_n))
σ_n      = BigFloat.(σ_n_f64)
Δσ_n     = BigFloat.(diff([0.0;
               (σ_n_f64[1:end-1] .+ σ_n_f64[2:end]) ./ 2;
               5.0]))
ΔG_n     = Vector{Complex{BigFloat}}(undef, Nσ_n)

for i in 1:Nσ_n
    σ      = σ_n[i]
    ω_R    = δ_bf - im*σ
    amp_R  = compute_amplitudes(s, l,  m, a_bf, ω_R)
    G_R    = amp_R.Bref / (2im * ω_R * amp_R.Binc)
    ω_mir  = δ_bf + im*σ
    amp_m  = compute_amplitudes(s, l, -m, a_bf, ω_mir)
    G_L    = conj(amp_m.Bref) / (2im*(-δ_bf - im*σ)*conj(amp_m.Binc))
    ΔG_n[i] = G_R - G_L
    i % 15 == 0 && (print("."); flush(stdout))
end
println(" 完了")

t_pos    = range(1.0, 600.0; length=1000)
ψ_BC_neg = Vector{ComplexF64}(undef, length(t_pos))
for (k, t) in enumerate(t_pos)
    s_val = zero(Complex{BigFloat})
    for i in 1:Nσ_n
        s_val += ΔG_n[i] * Δσ_n[i] * exp(-σ_n[i]*BigFloat(t))
    end
    ψ_BC_neg[k] = ComplexF64(im/(2π) * s_val)
    k % 100 == 0 && (print("."); flush(stdout))
end
println(" 完了")

# ── Part 4: QNM (t > 0) ──────────────────────────────────────

println("\n=== Part 4: QNM ===")
qnm_pro   = [load_qnm_data(l,  m, n, Float64(a)) for n in 0:7]
qnm_retro = [load_qnm_data(l, -m, n, Float64(a)) for n in 0:7]
all_modes = vcat(qnm_pro, qnm_retro)

for (n, (ω_n, B_n)) in enumerate(qnm_pro)
    @printf("  pro  n=%d: ω=%+.5f%+.5fi  |B|=%.3e\n",
            n-1, real(ω_n), imag(ω_n), abs(B_n))
end

ψ_QNM = [-sum(B_n*exp(-im*ω_n*t) for (ω_n, B_n) in all_modes)
         for t in t_pos]

# ── QNM + BC⁻ の合計 ─────────────────────────────────────────

ψ_sum = ψ_QNM .+ ψ_BC_neg

# ── プロット ──────────────────────────────────────────────────

println("\nプロット中 ...")

# --- 上段: 全成分の重ね合わせ ---
p1 = plot(
    xlabel     = L"t/M",
    ylabel     = L"|\Re[\psi_4]|",
    yscale     = :log10,
    title      = "Waveform decomposition  (s=$s, l=$l, m=$m, a=$a)",
    framestyle = :box, grid = true,
    legend     = :topright,
    size       = (900, 400))

plot!(p1, collect(t_all), abs.(real.(ψ_num)),
    label = "Numerical (high-prec.)", lw = 2, color = :steelblue)
plot!(p1, collect(t_neg), abs.(real.(ψ_BC_pos)),
    label = L"\psi_{BC}^{+}\ (t<0)", lw = 1.5, color = :crimson, ls = :dash)
plot!(p1, collect(t_pos), abs.(real.(ψ_BC_neg)),
    label = L"\psi_{BC}^{-}\ (t>0)", lw = 1.5, color = :darkorange, ls = :dash)
plot!(p1, collect(t_pos), abs.(real.(ψ_QNM)),
    label = L"\psi_{QNM}\ (n=0\text{--}7,\ \pm m)",
    lw = 1.5, color = :darkgreen, ls = :dash)
plot!(p1, collect(t_pos), abs.(real.(ψ_sum)),
    label = L"\psi_{QNM}+\psi_{BC}^{-}", lw = 1.5, color = :purple, ls = :dot)
vline!(p1, [0.0], label = "", color = :black, lw = 0.8, ls = :dash)

# --- 下段: 残差 |ψ_num - (QNM + BC⁻)|  (t>0) ---
# t_posとt_allのインデックスを対応付け
t_pos_f64 = collect(t_pos)
t_all_f64 = collect(t_all)

# t_all のうち t > 0 の部分を抽出しt_posと共通範囲で補間
idx_pos = findall(t_all_f64 .> 0.0)
t_ref    = t_all_f64[idx_pos]
ψ_ref    = ψ_num[idx_pos]

# t_pos に最近傍でスナップして残差計算
residual = Vector{Float64}(undef, length(t_pos_f64))
for (k, t) in enumerate(t_pos_f64)
    j = argmin(abs.(t_ref .- t))
    residual[k] = abs(real(ψ_ref[j]) - real(ψ_sum[k]))
end

p2 = plot(
    xlabel     = L"t/M",
    ylabel     = L"|\Re[\psi_{num}] - \Re[\psi_{QNM}+\psi_{BC}^-]|",
    yscale     = :log10,
    title      = "Residual (t>0)",
    framestyle = :box, grid = true,
    legend     = :topright,
    size       = (900, 300))
plot!(p2, t_pos_f64, residual,
    label = "residual", lw = 1.5, color = :gray)
plot!(p2, collect(t_all), abs.(real.(ψ_num)),
    label = "numerical (ref.)", lw = 1, color = :steelblue, alpha = 0.5)

fig = plot(p1, p2, layout = (2,1), size = (900, 750), dpi = 120,
           fontfamily = "Computer Modern")
savefig(fig, joinpath(@__DIR__, "waveform_decomposition.png"))
println("waveform_decomposition.png を保存しました")
fig
