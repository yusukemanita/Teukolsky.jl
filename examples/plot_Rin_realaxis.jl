using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")

using Teukolsky
using Plots, LaTeXStrings, Printf

# ============================================================
#  Rin(r=10M) をω実軸の少し上 ω = ω_real + iδ で掃引
#
#  プロット内容:
#    (1) |Rin(r=10, ω+iδ)|  vs  Re(ω)   — 絶対値
#    (2) arg(Rin) / π          vs  Re(ω)   — 位相
#    (3) Re(ν), Im(ν)          vs  Re(ω)   — 枝の追跡確認
#    (4) |G(ω)| = |Rin Bref / (2iω Binc)| vs Re(ω)  (参考)
# ============================================================

s, l, m = -2, 2, 2
a       = 0.0
r_src   = 10.0
δ       = 1e-3        # 実軸からの浮かし量 Im(ω) = δ

N       = 1200         # ω点数
ω_min   = -1.0
ω_max   = 1.0

ω_real  = range(ω_min, ω_max; length=N)
ω_grid  = ω_real .+ im*δ   # ω = ω_real + iδ のグリッド

# ── 計算 ──────────────────────────────────────────────────────
function compute_Rin_sweep(s, l, m, a, ω_grid, r_src; nmax=60)
    N = length(ω_grid)
    Rin_vals = Vector{ComplexF64}(undef, N)
    ν_vals   = Vector{ComplexF64}(undef, N)
    GF_vals  = Vector{ComplexF64}(undef, N)
    local ν_prev = nothing
    for i in 1:N
        ω = ω_grid[i]

        amp = compute_amplitudes(s, l, m, a, ω; nmax=nmax, ν_init=ν_prev)
        ν   = amp.ν

        p_mst = MSTParams(s, l, m, a, ω)
        Rin_vals[i] = Rin(p_mst, ν, amp.fn, r_src; nmax=nmax)
        GF_vals[i]  = Rin_vals[i] * amp.Bref / (2im * ω * amp.Binc)
        ν_vals[i]   = ν
        ν_prev      = ν

        i % 60 == 0 && (print("."); flush(stdout))
    end
    return Rin_vals, ν_vals, GF_vals
end

# ν ジャンプ検出：隣接点間の |Δν| を表示
function report_nu_jumps(ωr, ν_vals; threshold=0.05)
    println("\n── ν ジャンプ報告（|Δν| > $threshold）──")
    any_jump = false
    for i in 2:length(ν_vals)
        dν = abs(ν_vals[i] - ν_vals[i-1])
        if dν > threshold
            @printf "  ω = %.4f → %.4f : |Δν| = %.4f  (ν_prev=%.4f+%.4fi → ν=%.4f+%.4fi)\n" ωr[i-1] ωr[i] dν real(ν_vals[i-1]) imag(ν_vals[i-1]) real(ν_vals[i]) imag(ν_vals[i])
            any_jump = true
        end
    end
    any_jump || println("  ジャンプなし")
end

nmax    = 60           # MST 級数の打ち切り項数（ω>0.7 では40では不十分）
println("計算中 ($N 点, δ = $δ, r = $r_src M, nmax = $nmax) ...")
Rin_vals, ν_vals, GF_vals = compute_Rin_sweep(s, l, m, a, ω_grid, r_src; nmax=nmax)
println("\n完了")
report_nu_jumps(collect(real.(ω_grid)), ν_vals)

ωr = collect(real.(ω_grid))   # = ω_real

# 位相アンラップ: Julia標準のcumsum差分法
function unwrap_phase(φ::Vector{Float64})
    out = copy(φ)
    for i in 2:length(out)
        Δ = out[i] - out[i-1]
        out[i] -= 2π * round(Δ / (2π))
    end
    return out
end

# ── プロット ─────────────────────────────────────────────────
gr()

# --- (1) |Rin| ---
fig1 = plot(ωr, abs.(Rin_vals),
    xlabel     = L"\mathrm{Re}(\omega)\ [M^{-1}]",
    ylabel     = L"|R_{\rm in}(r=10M,\,\omega+i\delta)|",
    yscale     = :log10,
    label      = latexstring("\\delta=$(δ)"),
    title      = latexstring("|R_{\\rm in}|\\; (r=10M,\\; s=$(s),\\; l=$(l),\\; m=$(m),\\; a=$(a))"),
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150, size = (750, 350))

# --- (2) 位相: アンラップ + r*(r_src)·ω の線形トレンド除去 ---
rp_bh    = 1 + sqrt(1 - a^2)
rm_bh    = 1 - sqrt(1 - a^2)
rstar_src = r_src + (rp_bh/(rp_bh - rm_bh))*log(abs(r_src - rp_bh)) -
                    (rm_bh/(rp_bh - rm_bh))*log(abs(r_src - rm_bh))
@printf "r*(r_src=%.1f) = %.4f M\n" r_src rstar_src

phase_raw     = angle.(Rin_vals)               # wrapped ∈ (-π, π]
phase_unwrap  = unwrap_phase(phase_raw)         # unwrapped
# Rin の漸近形 ~ e^{i·ω·r*}  (符号は正の周波数側)
phase_trend   = ωr .* rstar_src                # 線形トレンド（rad）
phase_residual = phase_unwrap .- phase_trend    # 残差

fig2 = plot(ωr, phase_unwrap ./ π,
    xlabel     = L"\mathrm{Re}(\omega)\ [M^{-1}]",
    ylabel     = L"\phi(\omega)/\pi",
    label      = "unwrapped",
    title      = latexstring("\\mathrm{Unwrapped\\ phase\\ of\\ } R_{\\rm in}\\; (r=10M)"),
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150, size = (750, 320))
plot!(fig2, ωr, (ωr .* rstar_src) ./ π,
    label = latexstring("\\omega r_*(10M)/\\pi\\; (r_*=$(round(rstar_src,digits=2)))"),
    ls = :dash, color = :red, lw = 1)

fig2b = plot(ωr, phase_residual ./ π,
    xlabel     = L"\mathrm{Re}(\omega)\ [M^{-1}]",
    ylabel     = L"(\phi - \omega r_*)/\pi",
    label      = "residual",
    title      = "Phase residual (after removing linear trend)",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150, size = (750, 300))

# --- (3) ν の実部・虚部 ---
fig3 = plot(ωr, real.(ν_vals),
    xlabel     = L"\mathrm{Re}(\omega)\ [M^{-1}]",
    ylabel     = L"\nu",
    label      = L"\mathrm{Re}(\nu)",
    title      = latexstring("\\nu(\\omega+i\\delta)\\; (l=$(l))"),
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150, size = (750, 300))
plot!(fig3, ωr, imag.(ν_vals), label = L"\mathrm{Im}(\nu)", ls = :dash)

# --- (4) |G(ω)| = |Rin Bref/(2iω Binc)| ---
fig4 = plot(ωr, abs.(GF_vals),
    xlabel     = L"\mathrm{Re}(\omega)\ [M^{-1}]",
    ylabel     = L"|G(\omega)|",
    yscale     = :log10,
    label      = latexstring("\\delta=$(δ)"),
    title      = latexstring("|G(\\omega)|\\; (r'=10M)"),
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150, size = (750, 350))

# 複合プロット
fig_all = plot(fig1, fig2, fig2b, fig3, fig4,
    layout = (5, 1), size = (800, 1600))

outdir = @__DIR__
savefig(fig1,    joinpath(outdir, "Rin_abs.png"))
savefig(fig2,    joinpath(outdir, "Rin_phase.png"))
savefig(fig2b,   joinpath(outdir, "Rin_phase_residual.png"))
savefig(fig3,    joinpath(outdir, "nu_track.png"))
savefig(fig4,    joinpath(outdir, "GF_abs.png"))
savefig(fig_all, joinpath(outdir, "Rin_realaxis.png"))
println("保存完了: Rin_abs.png, Rin_phase.png, Rin_phase_residual.png, nu_track.png, GF_abs.png, Rin_realaxis.png")

fig_all
