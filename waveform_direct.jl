using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Plots
using LaTeXStrings
using CSV, DataFrames

# ── パラメータ ────────────────────────────────────────────────
s, l, m, a = -2, 2, 2, 0.9
N     = 4000          # 周波数グリッド点数
ω_max = 6.0           # 周波数の最大値 [M⁻¹]
t_ini = -100.0        # 時刻の始点 [M]
t_max =  600.0        # 時刻の終点 [M]
Nt    = 4000          # 時刻グリッド点数

# ── 周波数グリッド（半整数シフト: ω=0 を避ける）─────────────
Δω     = 2ω_max / N
ω_grid = [(n - N÷2 + 0.5) * Δω for n in 0:N-1]
# ω_grid[i] = -ω_grid[N+1-i]  が成立する（後述のミラーに利用）

# ── G(ω) = Bref / (2iω Binc) を計算 ─────────────────────────
# 正の周波数のみ計算し、G(-ω) = conj(G(ω)) でミラー
println("G(ω) を計算中 ($N 点) ...")
GF = Vector{ComplexF64}(undef, N)

for i in (N÷2 + 1):N
    ω  = ω_grid[i]
    amp = compute_amplitudes(s, l, m, a, ω)
    GF[i] = amp.Bref / (2im * ω * amp.Binc)
    i % 200 == 0 && (print("."); flush(stdout))
end
# 負の周波数: G(-ω) = conj(G(ω))
for i in 1:(N÷2)
    GF[i] = conj(GF[N + 1 - i])
end
println("\n完了")

# ── 時刻積分: ψ(t) = (Δω/2π) Σ_n G(ω_n) e^{-iω_n t} ────────
t_grid = range(t_ini, t_max; length=Nt)
ψ      = Vector{ComplexF64}(undef, Nt)
prefac = Δω / (2π)

println("時刻積分中 ($Nt 点) ...")
for (k, t) in enumerate(t_grid)
    s_val = zero(ComplexF64)
    @inbounds for n in 1:N
        s_val += GF[n] * exp(-im * ω_grid[n] * t)
    end
    ψ[k] = prefac * s_val
    k % 100 == 0 && (print("."); flush(stdout))
end
println("\n完了")

# ── プロット ──────────────────────────────────────────────────
ψ_real = real.(ψ)

fig = plot(collect(t_grid), abs.(ψ_real),
    xlabel = L"t\ [M]",
    ylabel = L"|\Re[\psi_4(t)]|",
    label  = L"|\Re[\psi_4]|",
    lw     = 1.0,
    yscale = :log10,
    title  = "Ringdown waveform  (s=$s, l=$l, m=$m, a=$a)",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 120)
fig2 = plot(ω_grid, abs.(GF), yscale=:log10, 
    label="", 
    xlabel=L"\omega", 
    ylabel=L"|G(\omega)|", 
    title="Frequency-domain Green's function", 
    framestyle=:box, grid=true, 
    fontfamily="Computer Modern", 
    dpi=120,
    ylim=(1e-10, 1e0),
    xlim=(-2.5,2.5))

savefig(fig, "waveform_direct.png")
println("waveform_direct.png を保存しました")
fig2
