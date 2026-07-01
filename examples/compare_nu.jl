using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")

using Teukolsky
using Plots, LaTeXStrings, CSV, DataFrames, Printf

# ============================================================
#  ν(ω) の比較: Teukolsky.jl  vs  Teukolsky package (Mathematica)
# ============================================================

s, l, m = -2, 2, 2
a       = 0.9
N_pts   = 300
ω_min   = 0.05
ω_max   = 6.0
ω_grid  = range(ω_min, ω_max; length=N_pts)

# ── Teukolsky でν を計算 ──────────────────────────────────
function compute_nu_sweep(s, l, m, a, ω_grid)
    N = length(ω_grid)
    ν_vals = Vector{ComplexF64}(undef, N)
    local ν_prev = nothing
    for i in 1:N
        ω = ω_grid[i]
        ν, _ = compute_nu(s, l, m, a, ω; ν_init=ν_prev)
        ν_vals[i] = ν
        ν_prev    = ν
        i % 30 == 0 && (print("."); flush(stdout))
    end
    println()
    return ν_vals
end

println("Teukolsky: computing ν ($N_pts 点) ...")
ν_bhp = compute_nu_sweep(s, l, m, a, ω_grid)
println("完了")

# ── ジャンプ報告 ───────────────────────────────────────────
function report_jumps(label, ωr, ν_vals; thr=0.05)
    println("\n── $label ジャンプ（|Δν| > $thr）──")
    any = false
    for i in 2:length(ν_vals)
        d = abs(ν_vals[i] - ν_vals[i-1])
        if d > thr
            @printf "  ω=%.4f→%.4f: |Δν|=%.4f  (%.4f%+.4fi → %.4f%+.4fi)\n" ωr[i-1] ωr[i] d real(ν_vals[i-1]) imag(ν_vals[i-1]) real(ν_vals[i]) imag(ν_vals[i])
            any = true
        end
    end
    any || println("  ジャンプなし")
end
ωr = collect(ω_grid)
report_jumps("Teukolsky", ωr, ν_bhp)

# ── Mathematica データ読み込み ──────────────────────────────
mma_file = joinpath(@__DIR__, "nu_mathematica.csv")
has_mma  = isfile(mma_file)
if has_mma
    df     = CSV.read(mma_file, DataFrame)
    ω_mma  = df[!, :omega]
    ν_mma  = df[!, :Re_nu] .+ im .* df[!, :Im_nu]
    println("\nMathematica データ読み込み完了: $(length(ω_mma)) 点")
    report_jumps("Mathematica", ω_mma, ν_mma)
else
    println("\nMathematica CSV が見つかりません。先に export_nu_mathematica.wls を実行してください。")
end

# ── プロット ───────────────────────────────────────────────
gr()

# Re(ν)
fig_re = plot(ωr, real.(ν_bhp),
    label      = "Teukolsky.jl",
    xlabel     = L"\omega\ [M^{-1}]",
    ylabel     = L"\mathrm{Re}(\nu)",
    title      = latexstring("\\mathrm{Re}(\\nu)\\; (s=$(s),\\,l=$(l),\\,m=$(m),\\,a=$(a))"),
    lw = 2, color = :steelblue,
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150, size = (750, 320))
if has_mma
    plot!(fig_re, ω_mma, real.(ν_mma),
        label = "Teukolsky (Mathematica)",
        lw = 1.5, color = :crimson, ls = :dash)
end

# Im(ν)
fig_im = plot(ωr, imag.(ν_bhp),
    label      = "Teukolsky.jl",
    xlabel     = L"\omega\ [M^{-1}]",
    ylabel     = L"\mathrm{Im}(\nu)",
    title      = latexstring("\\mathrm{Im}(\\nu)\\; (s=$(s),\\,l=$(l),\\,m=$(m),\\,a=$(a))"),
    lw = 2, color = :steelblue,
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150, size = (750, 320))
if has_mma
    plot!(fig_im, ω_mma, imag.(ν_mma),
        label = "Teukolsky (Mathematica)",
        lw = 1.5, color = :crimson, ls = :dash)
end

# 差分（Mathematicaがある場合）
if has_mma
    # BHPとMMAを共通ωグリッドで比較（最近傍補間）
    ν_bhp_interp = [ν_bhp[argmin(abs.(ωr .- ω))] for ω in ω_mma]
    Δν = ν_bhp_interp .- ν_mma
    fig_diff = plot(ω_mma, abs.(real.(Δν)),
        label      = L"|\Delta\,\mathrm{Re}(\nu)|",
        xlabel     = L"\omega\ [M^{-1}]",
        ylabel     = L"|\Delta\nu|",
        title      = "Difference: Teukolsky − Mathematica",
        yscale     = :log10,
        lw = 1.5, color = :steelblue,
        framestyle = :box, grid = true,
        fontfamily = "Computer Modern", dpi = 150, size = (750, 300))
    plot!(fig_diff, ω_mma, abs.(imag.(Δν)),
        label = L"|\Delta\,\mathrm{Im}(\nu)|",
        lw = 1.5, color = :darkorange, ls = :dash)

    fig_all = plot(fig_re, fig_im, fig_diff, layout=(3,1), size=(800, 950))
else
    fig_all = plot(fig_re, fig_im, layout=(2,1), size=(800, 650))
end

outdir = @__DIR__
savefig(fig_re,  joinpath(outdir, "nu_compare_re.png"))
savefig(fig_im,  joinpath(outdir, "nu_compare_im.png"))
savefig(fig_all, joinpath(outdir, "nu_compare.png"))
has_mma && savefig(fig_diff, joinpath(outdir, "nu_compare_diff.png"))
println("\n保存完了: nu_compare.png")
fig_all
