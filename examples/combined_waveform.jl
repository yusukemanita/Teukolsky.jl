using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Plots
using LaTeXStrings
using Printf
using CSV
using DataFrames

# ============================================================
#  3成分の重ね合わせプロット
#
#  1. 実軸積分 (全時刻):
#       ψ_direct(t) = (Δω/2π) Σ G(ω) e^{-iωt}
#
#  2. 正の虚軸ブランチカット (-100 < t < 0, Float64):
#       ψ_BC_pos(t) = (Δσ/2π) Σ ΔG_+(σ) e^{+σt}
#       ΔG_+(σ) = G_R(+iσ) - G_L(+iσ),  t < 0 → 収束
#
#  3. 負の虚軸ブランチカット (0 < t < 600, BigFloat):
#       ψ_BC_neg(t) = (Δσ/2π) Σ ΔG_-(σ) e^{-σt}
#       ΔG_-(σ) = G_R(-iσ) - G_L(-iσ),  t > 0 → 収束
# ============================================================

function load_qnm_data(l::Int, m::Int, n::Int, a_target::Float64)
    filepath = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$n.dat"
    df = CSV.read(filepath, DataFrame, header=false)
    a = df[!, 1]
    idx = argmin(abs.(a .- a_target))
    ω = (df[idx, 2] + im * df[idx, 3]) / 2
    B = (df[idx, 8] + im * df[idx, 9]) / 16
    return ω, B
end

s, l, m = -2, 2, 2
a       = 0.9

# ────────────────────────────────────────────────────────────
# Part 1: 数値波形をCSVから読み込み
# ────────────────────────────────────────────────────────────
println("=== Part 1: 波形データ読み込み ===")
wf_df  = CSV.read("/Users/yusuke/work/matrix_pencile/data/psi4_waveform_l2m2a0.900.csv", DataFrame)
t_all  = wf_df[!, 1]
ψ_dir  = wf_df[!, 2] .+ im .* wf_df[!, 3]
println("  $(length(t_all)) 点, t ∈ [$(t_all[1]), $(t_all[end])]  完了")

# ── MST実軸積分 ──────────────────────────────────────────────
# println("=== Part 1 (MST): 実軸積分 ===")
# N      = 3000
# ω_max  = 3.0
# Δω     = 2ω_max / N
# ω_grid = [(n - N÷2 + 0.5) * Δω for n in 0:N-1]
# GF = Vector{ComplexF64}(undef, N)
# println("G(ω) を計算中 ($N 点) ...")
# for i in (N÷2 + 1):N
#     amp    = compute_amplitudes(s, l, m, a, ω_grid[i])
#     GF[i]  = amp.Bref / (2im * ω_grid[i] * amp.Binc) / (2π)
#     i % 400 == 0 && (print("."); flush(stdout))
# end
# for i in 1:(N÷2)
#     GF[i] = conj(GF[N + 1 - i])
# end
# println(" 完了")
# t_all  = range(-100.0, 600.0; length=3500)
# ψ_dir  = Vector{ComplexF64}(undef, length(t_all))
# println("実軸フーリエ積分中 ...")
# for (k, t) in enumerate(t_all)
#     s_val = zero(ComplexF64)
#     @inbounds for n in 1:N
#         s_val += GF[n] * exp(-im * ω_grid[n] * t)
#     end
#     ψ_dir[k] = Δω / (2π) * s_val
#     k % 500 == 0 && (print("."); flush(stdout))
# end
# println(" 完了")

# ────────────────────────────────────────────────────────────
# Part 2: 正の虚軸ブランチカット (t < 0, Float64)  [一時コメントアウト]
# ────────────────────────────────────────────────────────────

println("\n=== Part 2: 正の虚軸ブランチカット (t < 0) ===")
Nσ_pos   = 200
σ_min_p  = 1e-3
σ_max_p  = 5.0
δ_f64    = 1e-6
σ_grid_p = exp.(range(log(σ_min_p), log(σ_max_p); length=Nσ_pos))
Δσ_p_arr = diff([0.0; (σ_grid_p[1:end-1] .+ σ_grid_p[2:end]) ./ 2; σ_grid_p[end]])
ΔG_pos = Vector{ComplexF64}(undef, Nσ_pos)
println("ΔG_+(σ) を計算中 ($Nσ_pos 点, Float64, log-spaced) ...")
for i in 1:Nσ_pos
    σ = σ_grid_p[i]
    ω_R      = δ_f64 + im * σ
    amp_R    = compute_amplitudes(s, l,  m, a, ω_R)
    G_R      = amp_R.Bref / (2im * ω_R * amp_R.Binc)
    amp_mir  = compute_amplitudes(s, l, -m, a, ω_R)
    Binc_L   = conj(amp_mir.Binc)
    Bref_L   = conj(amp_mir.Bref)
    ω_L      = -δ_f64 + im * σ
    G_L      = Bref_L / (2im * ω_L * Binc_L)
    ΔG_pos[i] = G_R - G_L
    i % 40 == 0 && (print("."); flush(stdout))
end
println(" 完了")
t_neg  = range(-100.0, -0.5; length=500)
ψ_BC_pos = Vector{ComplexF64}(undef, length(t_neg))
println("t<0 時刻積分中 ...")
for (k, t) in enumerate(t_neg)
    ψ_BC_pos[k] = -im / (2π) * sum(ΔG_pos[i] * Δσ_p_arr[i] * exp(σ_grid_p[i] * t) for i in 1:Nσ_pos)
end
println("完了")


# ────────────────────────────────────────────────────────────
# Part 3: 負の虚軸ブランチカット (t > 0, BigFloat)
# ────────────────────────────────────────────────────────────
t_pos  = range(1.0, 600.0; length=3000)

println("\n=== Part 3: 負の虚軸ブランチカット (t > 0, BigFloat) ===")
setprecision(BigFloat, 256)
a_bf    = BigFloat(string(a))
δ_bf    = BigFloat("1e-6")
Nσ_neg   = 100
σ_min_n  = BigFloat("1e-6")
σ_max_n  = BigFloat("5")
σ_grid_n = exp.(range(log(Float64(σ_min_n)), log(Float64(σ_max_n)); length=Nσ_neg))
σ_grid_n = BigFloat.(σ_grid_n)
Δσ_n_arr = BigFloat.(diff([0.0; (Float64.(σ_grid_n[1:end-1]) .+ Float64.(σ_grid_n[2:end])) ./ 2; Float64(σ_max_n)]))
ΔG_neg = Vector{Complex{BigFloat}}(undef, Nσ_neg)
println("ΔG_-(σ) を計算中 ($Nσ_neg 点, 256-bit BigFloat, log-spaced) ...")
for i in 1:Nσ_neg
    σ = σ_grid_n[i]
    ω_R      = δ_bf - im * σ
    amp_R    = compute_amplitudes(s, l,  m, a_bf, ω_R)
    G_R      = amp_R.Bref / (2im * ω_R * amp_R.Binc)
    ω_mir    = δ_bf + im * σ
    amp_mir  = compute_amplitudes(s, l, -m, a_bf, ω_mir)
    G_L      = conj(amp_mir.Bref) / (2im * (-δ_bf - im*σ) * conj(amp_mir.Binc))
    ΔG_neg[i] = G_R - G_L
    i % 10 == 0 && (print("."); flush(stdout))
end
println(" 完了")
ψ_BC_neg = Vector{ComplexF64}(undef, length(t_pos))
println("t>0 時刻積分中 ...")
for (k, t) in enumerate(t_pos)
    s_val = zero(Complex{BigFloat})
    for i in 1:Nσ_neg
        s_val += ΔG_neg[i] * Δσ_n_arr[i] * exp(-σ_grid_n[i] * BigFloat(t))
    end
    ψ_BC_neg[k] = ComplexF64(im / (2π) * s_val)
    k % 50 == 0 && (print("."); flush(stdout))
end
println(" 完了")

# ────────────────────────────────────────────────────────────
# Part 4: QNM寄与 (t > 0)
# ────────────────────────────────────────────────────────────
println("\n=== Part 4: QNM寄与 ===")
# prograde (m>0) と retrograde (m<0, ω_{l,-m,n}=-ω_{lmn}*) を両方ロード
qnm_pro   = [load_qnm_data(l,  m, n, Float64(a)) for n in 0:7]
qnm_retro = [load_qnm_data(l, -m, n, Float64(a)) for n in 0:7]

println("  --- prograde (m=$m) ---")
for (n, (ω_n, B_n)) in enumerate(qnm_pro)
    @printf("  n=%d: ω = %+.6f %+.6fi,  B = %.4e %+.4ei\n",
            n-1, real(ω_n), imag(ω_n), real(B_n), imag(B_n))
end
println("  --- retrograde (m=$(-m)) ---")
for (n, (ω_n, B_n)) in enumerate(qnm_retro)
    @printf("  n=%d: ω = %+.6f %+.6fi,  B = %.4e %+.4ei\n",
            n-1, real(ω_n), imag(ω_n), real(B_n), imag(B_n))
end

ψ_QNM = Vector{ComplexF64}(undef, length(t_pos))
println("QNM波形を計算中 ...")
for (k, t) in enumerate(t_pos)
    ψ_QNM[k] = -(sum(B_n * exp(-im * ω_n * t) for (ω_n, B_n) in qnm_pro) +
                     sum(B_n * exp(-im * ω_n * t) for (ω_n, B_n) in qnm_retro))
end
println("完了")

# ────────────────────────────────────────────────────────────
# プロット
# ────────────────────────────────────────────────────────────
println("\nプロット中 ...")

fig = plot(
    xlabel     = L"t\ [M]",
    ylabel     = L"|\Re[\psi(t)]|",
    yscale     = :log10,
    title      = "Waveform components  (s=$s, l=$l, m=$m, a=$a)",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 120,
    legend     = :topright,
    size       = (900, 500))

# 1. 数値波形
plot!(fig, collect(t_all), abs.(real.(ψ_dir)),
    label = L"\psi_4\ \mathrm{(numerical)}",
    lw = 2, color = :steelblue, alpha = 0.7)

# 2. 正の虚軸ブランチカット (t < 0)
plot!(fig, collect(t_neg), abs.(real.(ψ_BC_pos)),
    label = L"\psi_{BC}^{+}\ (t<0)",
    lw = 1, color = :crimson, ls = :dash)

# 3. 負の虚軸ブランチカット (t > 0)
plot!(fig, collect(t_pos), abs.(real.(ψ_BC_neg)),
    label = L"\psi_{BC}^{-}\ (t>0)",
    lw = 1, color = :darkorange, ls = :dash)

# 4. QNM (t > 0)
plot!(fig, collect(t_pos), abs.(real.(ψ_QNM)),
    label = L"\psi_{\rm QNM}\ (n=\pm 0,\cdots,7)",
    lw = 1, color = :darkgreen, ls = :dash, alpha = 0.9)

# 縦線: t=0
vline!(fig, [0.0], label = "", color = :black, lw = 0.5, ls = :dash)

savefig(fig, "combined_waveform.png")
println("combined_waveform.png を保存しました")
fig
