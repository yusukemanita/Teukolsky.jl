using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Plots, LaTeXStrings, Printf

# ============================================================
#  実軸積分による波形計算
#
#    ψ(u) = ∫ dω/2π  [Rin(r'=r_src) Bref / (2iω Binc)] e^{-iωu}
#
#  u = retarded time  (= t at null infinity)
#  r_src = 10M  (source location)
# ============================================================

s, l, m   = -2, 2, 2
a         = 0.9
r_src     = 10.0        # source position r' [M]

N         = 4000        # 周波数点数（正負合わせて）
ω_max     = 6.0         # 周波数打ち切り [M⁻¹]
taper_frac = 0.1        # Planck taperの割合

t_ini     = -100.0      # 時刻始点 [M]
t_max     =  600.0      # 時刻終点 [M]
Nt        =  7000       # 時刻点数

# ── 周波数グリッド（半整数シフト: ω=0 を回避）───────────────
Δω     = 2ω_max / N
ω_grid = [(n - N÷2 + 0.5) * Δω for n in 0:N-1]

# ── Planck taper ─────────────────────────────────────────────
@inline function planck_taper(x::Real)
    x ≤ 0.0 && return 1.0
    x ≥ 1.0 && return 0.0
    return 1.0 / (exp(1/x - 1/(1-x)) + 1)
end
@inline function taper_weight(ω::Real, ω_max::Real, frac::Real)
    frac ≤ 0.0 && return 1.0
    x = (abs(ω) / ω_max - (1 - frac)) / frac
    return planck_taper(x)
end

# ── G(ω) = Rin(r_src; ω) * Bref / (2iω Binc) を計算 ─────────
# ブランチトラッキング: 正の周波数を小→大の順に計算し
#   1. ν_init=ν_prev で前のブランチを引き継ぐ
#   2. Im(ν)の符号反転を検出したら共役を使って符号を維持
# 負の周波数は G(-ω) = conj(G(ω)) でミラー

function compute_GF(s, l, m, a, ω_grid, N, r_src, taper_frac, ω_max)
    GF = Vector{ComplexF64}(undef, N)
    local ν_prev = nothing
    local n_jumps = 0
    for i in (N÷2 + 1):N
        ω = ω_grid[i]
        w = taper_weight(ω, ω_max, taper_frac)
        if iszero(w)
            GF[i] = zero(ComplexF64)
            continue
        end

        # Step 1: ν_init = ν_prev でブランチを引き継ぐ
        amp = compute_amplitudes(s, l, m, a, ω; ν_init=ν_prev)
        ν   = amp.ν

        # Step 2: Im(ν)の符号反転を検出したら共役ブランチを試みる
        if ν_prev !== nothing && imag(ν_prev) * imag(ν) < -1e-10
            ν_try   = conj(ν)
            amp_try = compute_amplitudes(s, l, m, a, ω; ν_init=ν_try)
            if imag(amp_try.ν) * imag(ν_prev) ≥ 0
                amp = amp_try
                ν   = amp.ν
                n_jumps += 1
            end
        end

        # Rin(r_src; ω) を評価
        p       = MSTParams(s, l, m, a, ω)
        Rin_val = Rin(p, ν, amp.fn, r_src)

        GF[i]  = Rin_val * amp.Bref / (2im * ω * amp.Binc) * w
        ν_prev = ν

        i % 200 == 0 && (print("."); flush(stdout))
    end
    println("\n完了 (符号反転修正: $n_jumps 回)")

    # 負周波数: G(-ω) = conj(G(ω)) （ψが実数値になる条件）
    for i in 1:(N÷2)
        GF[i] = conj(GF[N + 1 - i])
    end
    return GF
end

println("G(ω) を計算中 ($N 点, r_src = $r_src M, branch-tracked) ...")
GF = compute_GF(s, l, m, a, ω_grid, N, r_src, taper_frac, ω_max)

# ── 時刻積分: ψ(u) = Δω/2π * Σ G(ω_n) e^{-iω_n u} ──────────
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

# ── tortoise座標 r*(r_src) を計算（ピーク時刻の推定）──────────
rp_bh = 1 + sqrt(1 - a^2)
rm_bh = 1 - sqrt(1 - a^2)
rstar_src = r_src + (rp_bh/(rp_bh - rm_bh))*log(abs(r_src - rp_bh)) -
                    (rm_bh/(rp_bh - rm_bh))*log(abs(r_src - rm_bh))
@printf "r*(r_src=%.1f) = %.3f M\n" r_src rstar_src

# ── プロット ──────────────────────────────────────────────────
gr()
t_arr = collect(t_grid)

# Fig 1a: 全時刻域（log scale）
fig1a = plot(t_arr, abs.(real.(ψ)),
    xlabel     = L"u\ [M]",
    ylabel     = L"|\mathrm{Re}[\psi(u)]|",
    label      = latexstring("r'=$(r_src)M"),
    yscale     = :log10,
    title      = "Waveform  (s=$s, l=$l, m=$m, a=$a)",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150,
    size = (750, 400))
vline!(fig1a, [rstar_src], label = latexstring("u = r_*({r_src}M)"),
    color = :red, lw = 1, ls = :dash)

# Fig 1b: u ≈ r*(r_src) 付近のズーム（QNM立ち上がり確認）
u_zoom_lo = rstar_src - 20.0
u_zoom_hi = rstar_src + 200.0
mask_zoom  = u_zoom_lo .≤ t_arr .≤ u_zoom_hi
fig1b = plot(t_arr[mask_zoom], abs.(real.(ψ[mask_zoom])),
    xlabel     = L"u\ [M]",
    ylabel     = L"|\mathrm{Re}[\psi(u)]|",
    label      = latexstring("r'=$(r_src)M"),
    yscale     = :log10,
    title      = "Zoom: u near r_*(r_src)",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150,
    size = (750, 400))
vline!(fig1b, [rstar_src], label = latexstring("u = r_*({r_src}M)"),
    color = :red, lw = 1, ls = :dash)

# Fig 2: 周波数域 Green関数
fig2 = plot(ω_grid, abs.(GF),
    xlabel     = L"\omega\ [M^{-1}]",
    ylabel     = L"|G(\omega)|",
    label      = latexstring("R_{\\rm in}(r'=$(r_src)) B^{\\rm ref} / (2i\\omega B^{\\rm inc})"),
    yscale     = :log10,
    xlim       = (-ω_max, ω_max),
    title      = "Frequency-domain integrand",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150,
    size = (750, 400))

outdir = @__DIR__
savefig(fig1a, joinpath(outdir, "waveform_real_time.png"))
savefig(fig1b, joinpath(outdir, "waveform_real_zoom.png"))
savefig(fig2,  joinpath(outdir, "waveform_real_freq.png"))
println("waveform_real_time.png, waveform_real_zoom.png, waveform_real_freq.png を保存しました")

# 複合プロット
fig_all = plot(fig1a, fig1b, fig2, layout=(3,1), size=(800, 1100))
savefig(fig_all, joinpath(outdir, "waveform_real.png"))
println("waveform_real.png を保存しました")
fig_all