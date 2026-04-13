using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")
using BHPtoolkit
using Plots, LaTeXStrings, Printf

# ν = 0.5 + i×η のとき fn[n] が何項必要か調べる
s, l, m = -2, 2, 2
a = 0.0
r_src = 10.0

# テスト周波数を複数選ぶ
ω_tests = [0.1, 0.3, 0.5, 0.8, 1.0]

gr()
fig = plot(xlabel = "n", ylabel = L"|f_n|", yscale = :log10,
    title = "fn[n] magnitude  (s=$s, l=$l, m=$m, a=$a, r=$r_src)",
    framestyle = :box, grid = true, legend = :topright,
    fontfamily = "Computer Modern", dpi = 150, size = (800, 450))

println("ω        Re(ν)    Im(ν)    |fn[40]|/|fn[0]|  still_large?")
println("─"^70)

for ω in ω_tests
    p   = MSTParams(s, l, m, a, ω)
    ν, _ = compute_nu(s, l, m, a, ω)

    # 大きめの nmax でfnを計算
    fn_big = compute_fn(p, ν; nmax=120)

    n_range = -120:120
    fn_abs  = [abs(get(fn_big, n, 0.0+0.0im)) for n in n_range]

    fn0 = abs(get(fn_big, 0, 1.0+0.0im))
    fn40 = abs(get(fn_big, 40, 0.0+0.0im))
    ratio = fn40 / (fn0 + 1e-300)
    still_large = ratio > 1e-6

    @printf "%.3f    %.4f   %.4f    %.2e          %s\n" ω real(ν) imag(ν) ratio (still_large ? "YES ← problem" : "no")

    lbl = @sprintf("ω=%.1f, ν=%.2f%+.2fi", ω, real(ν), imag(ν))
    plot!(fig, collect(n_range), fn_abs .+ 1e-300, label = lbl, lw = 1.5)
end

# nmax=40 の打ち切り位置をマーク
vline!(fig, [40, -40], label = "nmax=40 cutoff", color=:black, ls=:dash, lw=1)

outdir = @__DIR__
savefig(fig, joinpath(outdir, "fn_convergence.png"))
println("\n保存: fn_convergence.png")

# ── 追加: x=-4 での 2F1 の精度を BigFloat で検証 ──────────────────
println("\n── 2F1 精度チェック (x = -4, ν = 0.5+1i) ──")
using HypergeometricFunctions: _₂F₁

ω0 = 0.5
p0 = MSTParams(s, l, m, a, ω0)
ν0 = ComplexF64(0.5 + 1.0im)
τ0 = p0.τ; ε0 = p0.ϵ
aF = ν0 + 1 - im*τ0
bF = -ν0 - im*τ0
cF = ComplexF64(1 - s) - im*(ε0 + τ0)
x  = ComplexF64(-4.0)

val_f64 = _₂F₁(aF, bF, cF, x)
println("  2F1(Float64): ", val_f64)

# BigFloat で高精度計算
setprecision(BigFloat, 256) do
    aF_b = Complex{BigFloat}(aF)
    bF_b = Complex{BigFloat}(bF)
    cF_b = Complex{BigFloat}(cF)
    x_b  = Complex{BigFloat}(x)
    val_bf = _₂F₁(aF_b, bF_b, cF_b, x_b)
    println("  2F1(BigFloat): ", ComplexF64(val_bf))
    println("  相対誤差: ", abs(val_f64 - ComplexF64(val_bf)) / abs(ComplexF64(val_bf)))
end
