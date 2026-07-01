using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Plots
using LaTeXStrings
using Printf
using CSV
using DataFrames
using Measures

# ============================================================
#  3成分の重ね合わせプロット
#
#  1. 数値波形 (CSV から読み込み)
#
#  2. 正の虚軸ブランチカット (t < 0, Float64):
#       ψ_BC_pos(t) = (-i/2π) Σ ΔG_+(σ) Δσ e^{+σt}
#       ΔG_+(σ) = G(ω_R) - G(ω_L),  ω_R = δ+iσ, ω_L = -δ+iσ
#
#  3. 負の虚軸ブランチカット (t > 0, Float64):
#       ψ_BC_neg(t) = (i/2π) Σ ΔG_-(σ) Δσ e^{-σt}
#       ΔG_-(σ) = iq Bref² / (2iω Binc (Binc + iq Bref))
#
#  4. QNM (t > 0): prograde + retrograde, n=0..7
# ============================================================

function load_qnm_data(l::Int, m::Int, n::Int, a_target::Float64)
    filepath = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$n.dat"
    df  = CSV.read(filepath, DataFrame, header=false)
    idx = argmin(abs.(df[!, 1] .- a_target))
    ω = (df[idx, 2] + im * df[idx, 3]) / 2
    B = (df[idx, 8] + im * df[idx, 9]) / 16
    return ω, B
end

s, l, m = -2, 2, 2
a       = 0.9

# ────────────────────────────────────────────────────────────
# Part 1: 数値波形
# ────────────────────────────────────────────────────────────
println("=== Part 1: 波形データ読み込み ===")
wf_df = CSV.read("/Users/yusuke/work/matrix_pencile/data/psi4_waveform_l2m2a0.900.csv", DataFrame)
t_all = wf_df[!, 1]
ψ_dir = wf_df[!, 2] .+ im .* wf_df[!, 3]
println("  $(length(t_all)) 点, t ∈ [$(t_all[1]), $(t_all[end])]  完了")

# ────────────────────────────────────────────────────────────
# Part 2: 正の虚軸ブランチカット (t < 0, Float64)
# ────────────────────────────────────────────────────────────
println("\n=== Part 2: 正の虚軸ブランチカット (t < 0) ===")
Nσ_pos   = 200
σ_min_p  = 1e-3
σ_max_p  = 5.0
δ_pos    = 1e-6
σ_grid_p = exp.(range(log(σ_min_p), log(σ_max_p); length=Nσ_pos))
Δσ_p_arr = diff([0.0; (σ_grid_p[1:end-1] .+ σ_grid_p[2:end]) ./ 2; σ_max_p])

ΔG_pos = Vector{ComplexF64}(undef, Nσ_pos)
println("ΔG_+(σ) を計算中 ($Nσ_pos 点, Float64, log-spaced) ...")
for i in 1:Nσ_pos
    σ     = σ_grid_p[i]
    ω_R   = δ_pos + im*σ
    amp_R = compute_amplitudes(s, l,  m, a, ω_R)
    G_R   = amp_R.Bref / (2im * ω_R * amp_R.Binc)

    ω_L   = -δ_pos + im*σ
    amp_m = compute_amplitudes(s, l, -m, a, δ_pos + im*σ)  # symmetry: G_L = conj(G(l,-m,ω_R))
    G_L   = conj(amp_m.Bref) / (2im * ω_L * conj(amp_m.Binc))

    ΔG_pos[i] = G_R - G_L
    i % 40 == 0 && (print("."); flush(stdout))
end
println(" 完了")

t_neg    = range(-100.0, -0.5; length=500)
ψ_BC_pos = Vector{ComplexF64}(undef, length(t_neg))
println("t<0 時刻積分中 ...")
for (k, t) in enumerate(t_neg)
    ψ_BC_pos[k] = -im / (2π) * sum(ΔG_pos[i] * Δσ_p_arr[i] * exp(σ_grid_p[i] * t) for i in 1:Nσ_pos)
end
println(" 完了")

# ────────────────────────────────────────────────────────────
# Part 3: 負の虚軸ブランチカット (t > 0, Float64)
# ────────────────────────────────────────────────────────────
println("\n=== Part 3: 負の虚軸ブランチカット (t > 0) ===")
Nσ_neg   = 200
σ_min_n  = 1e-3
σ_max_n  = 5.0
δ_neg    = 1e-8
σ_grid_n = exp.(range(log(σ_min_n), log(σ_max_n); length=Nσ_neg))
Δσ_n_arr = diff([0.0; (σ_grid_n[1:end-1] .+ σ_grid_n[2:end]) ./ 2; σ_max_n])

ΔG_neg = Vector{ComplexF64}(undef, Nσ_neg)
println("ΔG_-(σ) を計算中 ($Nσ_neg 点, Float64, log-spaced) ...")
for i in 1:Nσ_neg
    σ      = σ_grid_n[i]
    ω_R    = - im*σ
    amp_R  = compute_amplitudes(s, l, m, a, ω_R)
    q_info = compute_q(s, l, m, a, ω_R; nmax=60)
    q_val  = q_info.q
    # ΔG_-(σ) = iq Bref² / (2iω Binc (Binc + iq Bref))
    ΔG_neg[i] = im * q_val * amp_R.Bref^2 /
                (2im * ω_R * amp_R.Binc * (amp_R.Binc + im * q_val * amp_R.Bref))
    i % 40 == 0 && (print("."); flush(stdout))
end
println(" 完了")

t_pos    = range(1.0, 600.0; length=3000)
ψ_BC_neg = Vector{ComplexF64}(undef, length(t_pos))
println("t>0 時刻積分中 ...")
for (k, t) in enumerate(t_pos)
    ψ_BC_neg[k] = im / (2π) * sum(ΔG_neg[i] * Δσ_n_arr[i] * exp(-σ_grid_n[i] * t) for i in 1:Nσ_neg)
    k % 500 == 0 && (print("."); flush(stdout))
end
println(" 完了")

# ────────────────────────────────────────────────────────────
# Part 4: QNM寄与 (t > 0)
# ────────────────────────────────────────────────────────────
println("\n=== Part 4: QNM寄与 ===")
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

ψ_QNM = [-(sum(B_n * exp(-im * ω_n * t) for (ω_n, B_n) in qnm_pro) +
            sum(B_n * exp(-im * ω_n * t) for (ω_n, B_n) in qnm_retro))
          for t in t_pos]
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
    size       = (600, 500),
    ylims      = (1e-15, 1e0),
    margin     = 5mm)

plot!(fig, collect(t_all), abs.(real.(ψ_dir)),
    label = L"\psi_4\ \mathrm{(numerical)}",
    lw = 2, color = :steelblue, alpha = 0.7)

plot!(fig, collect(t_neg), abs.(real.(ψ_BC_pos)),
    label = L"\psi_{BC}^{+}\ (t<0)",
    lw = 1, color = :crimson, ls = :dash)

plot!(fig, collect(t_pos), abs.(real.(ψ_BC_neg)),
    label = L"\psi_{BC}^{-}\ (t>0)",
    lw = 1, color = :darkorange, ls = :dash)

plot!(fig, collect(t_pos), abs.(real.(ψ_QNM)),
    label = L"\psi_{\rm QNM}\ (n=\pm 0,\cdots,7)",
    lw = 1, color = :darkgreen, ls = :dash, alpha = 0.9)

vline!(fig, [0.0], label = "", color = :black, lw = 0.5, ls = :dash)

savefig(fig, "combined_waveform.png")
println("combined_waveform.png を保存しました")
fig
