using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Plots
using LaTeXStrings
using Printf
using CSV
using DataFrames

# ============================================================
#  Figure generator for branchcut_note.tex
#
#  Produces:
#    waveform_decomposition.pdf  (Fig. 1)
#    convergence_integrand.pdf   (Fig. 2)
#    convergence_sigma_max.pdf   (Fig. 3)
# ============================================================

const OUTDIR = @__DIR__
const s, l, m = -2, 2, 2
const a = 0.9

gr()   # GR backend — saves PDF/PNG without extra dependencies

# ── helper: load QNM excitation factor ───────────────────────

function load_qnm_data(l::Int, m::Int, n::Int, a_target::Float64)
    filepath = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$n.dat"
    df  = CSV.read(filepath, DataFrame, header=false)
    idx = argmin(abs.(df[!, 1] .- a_target))
    ω   = (df[idx, 2] + im * df[idx, 3]) / 2
    B   = (df[idx, 8] + im * df[idx, 9]) / 16
    return ω, B
end

# ── helper: compute ΔG⁻(σ) on neg. imag. axis (BigFloat) ────

function compute_DG_neg(σ_grid::Vector{BigFloat}, a_bf, δ_bf)
    Nσ = length(σ_grid)
    ΔG = Vector{Complex{BigFloat}}(undef, Nσ)
    for i in 1:Nσ
        σ      = σ_grid[i]
        ω_R    = δ_bf - im*σ
        amp_R  = compute_amplitudes(s, l,  m, a_bf, ω_R)
        G_R    = amp_R.Bref / (2im * ω_R * amp_R.Binc)
        ω_mir  = δ_bf + im*σ
        amp_m  = compute_amplitudes(s, l, -m, a_bf, ω_mir)
        G_L    = conj(amp_m.Bref) / (2im*(-δ_bf - im*σ)*conj(amp_m.Binc))
        ΔG[i]  = G_R - G_L
        i % 20 == 0 && (print("."); flush(stdout))
    end
    return ΔG
end

# ── helper: compute ΔG⁺(σ) on pos. imag. axis (Float64) ─────

function compute_DG_pos(σ_grid::AbstractVector{Float64}, a_f64, δ_f64)
    Nσ = length(σ_grid)
    ΔG = Vector{ComplexF64}(undef, Nσ)
    for i in 1:Nσ
        σ      = σ_grid[i]
        ω_R    = δ_f64 + im*σ
        amp_R  = compute_amplitudes(s, l,  m, a_f64, ω_R)
        G_R    = amp_R.Bref / (2im * ω_R * amp_R.Binc)
        amp_m  = compute_amplitudes(s, l, -m, a_f64, ω_R)
        ω_L    = -δ_f64 + im*σ
        G_L    = conj(amp_m.Bref) / (2im * ω_L * conj(amp_m.Binc))
        ΔG[i]  = G_R - G_L
    end
    return ΔG
end

# ── helper: trapezoidal weights for log-spaced grid ──────────

function logspaced_weights(σ_min, σ_max, Nσ)
    σ = exp.(range(log(σ_min), log(σ_max); length=Nσ))
    Δσ = diff([0.0; (σ[1:end-1] .+ σ[2:end]) ./ 2; σ[end]])
    return σ, Δσ
end

# ============================================================
# Shared computation: ΔG on both axes
# ============================================================

println("====== 共有計算: ΔG を計算 ======")

# ── Neg. imag. axis (BigFloat 256-bit) ───────────────────────
println("\n--- ΔG⁻(σ)  [neg. imag. axis, BigFloat] ---")
setprecision(BigFloat, 256)
a_bf  = BigFloat(string(a))
δ_bf  = BigFloat("1e-6")
Nσ_n  = 150
σ_n_f64, Δσ_n_f64 = logspaced_weights(1e-3, 5.0, Nσ_n)
σ_n   = BigFloat.(σ_n_f64)
Δσ_n  = BigFloat.(Δσ_n_f64)
ΔG_n  = compute_DG_neg(σ_n, a_bf, δ_bf)
println(" 完了")

# ── Pos. imag. axis (Float64) ─────────────────────────────────
println("\n--- ΔG⁺(σ)  [pos. imag. axis, Float64] ---")
Nσ_p  = 300
σ_p, Δσ_p = logspaced_weights(1e-3, 10.0, Nσ_p)
ΔG_p  = compute_DG_pos(σ_p, Float64(a), 1e-6)
println(" 完了")

# ── Numerical waveform ────────────────────────────────────────
wf_df = CSV.read(
    "/Users/yusuke/work/matrix_pencile/data/psi4_waveform_l2m2a0.900.csv",
    DataFrame)
t_all = collect(wf_df[!, 1])
ψ_num = wf_df[!, 2] .+ im .* wf_df[!, 3]

# ── QNM ──────────────────────────────────────────────────────
qnm_pro   = [load_qnm_data(l,  m, n, Float64(a)) for n in 0:7]
qnm_retro = [load_qnm_data(l, -m, n, Float64(a)) for n in 0:7]
all_modes  = vcat(qnm_pro, qnm_retro)

# ── Branch cut integrals ──────────────────────────────────────
t_neg = collect(range(-100.0, -0.5; length=800))
ψ_BC_pos = [-im/(2π) * sum(ΔG_p[i]*Δσ_p[i]*exp(σ_p[i]*t) for i in 1:Nσ_p)
            for t in t_neg]

t_pos = collect(range(1.0, 600.0; length=1000))
ψ_BC_neg = Vector{ComplexF64}(undef, length(t_pos))
println("\n--- 時刻積分 ψ_BC⁻ ---")
for (k, t) in enumerate(t_pos)
    sv = zero(Complex{BigFloat})
    for i in 1:Nσ_n
        sv += ΔG_n[i] * Δσ_n[i] * exp(-σ_n[i]*BigFloat(t))
    end
    ψ_BC_neg[k] = ComplexF64(im/(2π) * sv)
    k % 100 == 0 && (print("."); flush(stdout))
end
println(" 完了")

ψ_QNM = [-sum(B_n*exp(-im*ω_n*t) for (ω_n, B_n) in all_modes)
         for t in t_pos]
ψ_sum = ψ_QNM .+ ψ_BC_neg

# ============================================================
# Figure 1: waveform_decomposition.pdf
# ============================================================

println("\n====== Fig.1: waveform_decomposition ======")

# residual on t_pos grid (nearest-neighbour to CSV)
idx_pos  = findall(t_all .> 0.0)
t_ref    = t_all[idx_pos]
ψ_ref    = ψ_num[idx_pos]
residual = [abs(real(ψ_ref[argmin(abs.(t_ref .- t))]) - real(ψ_sum[k]))
            for (k, t) in enumerate(t_pos)]

p1 = plot(
    xlabel = L"t/M",
    ylabel = L"|\mathrm{Re}[\psi_4]|",
    yscale = :log10, ylim = (1e-14, 1e-2),
    framestyle = :box, grid = true, legend = :topright)
plot!(p1, t_all, abs.(real.(ψ_num)),
    label = "Numerical (high-prec.)", lw = 2, color = :steelblue)
plot!(p1, t_neg, abs.(real.(ψ_BC_pos)),
    label = L"\psi_{BC}^{+}\ (t<0)", lw = 1.5, color = :crimson, ls = :dash)
plot!(p1, t_pos, abs.(real.(ψ_BC_neg)),
    label = L"\psi_{BC}^{-}\ (t>0)", lw = 1.5, color = :darkorange, ls = :dash)
plot!(p1, t_pos, abs.(real.(ψ_QNM)),
    label = L"\psi_{QNM}\ (n\leq 7,\ \pm m)",
    lw = 1.5, color = :darkgreen, ls = :dash)
plot!(p1, t_pos, abs.(real.(ψ_sum)),
    label = L"\psi_{QNM}+\psi_{BC}^{-}",
    lw = 1.5, color = :purple, ls = :dot)
vline!(p1, [0.0], label = "", color = :black, lw = 0.8, ls = :dash)

p2 = plot(
    xlabel = L"t/M",
    ylabel = L"|\mathrm{residual}|",
    yscale = :log10,
    framestyle = :box, grid = true, legend = :topright)
plot!(p2, t_pos, residual,
    label = L"|\psi_{num} - \psi_{QNM} - \psi_{BC}^-|",
    lw = 1.5, color = :gray)
plot!(p2, t_all, abs.(real.(ψ_num)),
    label = "Numerical (ref.)", lw = 1, color = :steelblue, alpha = 0.4)

fig1 = plot(p1, p2, layout = (2,1), size = (800, 700), dpi = 150,
            fontfamily = "Computer Modern",
            title = ["Waveform decomposition  (s=$s, l=$l, m=$m, a=$a)" ""])
savefig(fig1, joinpath(OUTDIR, "waveform_decomposition.pdf"))
println("  → waveform_decomposition.pdf")

# ============================================================
# Figure 2: convergence_integrand.pdf
# ============================================================

println("====== Fig.2: convergence_integrand ======")

ΔG_n_abs = Float64.(abs.(ΔG_n))
t_show   = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
colors2  = [:navy, :royalblue, :steelblue, :seagreen, :darkorange, :crimson]

fig2 = plot(
    xlabel = L"\sigma",
    ylabel = L"|\Delta G^-(\sigma)|\,e^{-\sigma t}",
    yscale = :log10, ylim = (1e-20, 1e2),
    title  = "Integrand decay vs. \$\\sigma\$ for several \$t\$  (neg. axis)",
    framestyle = :box, grid = true, legend = :topright,
    size = (750, 450), dpi = 150, fontfamily = "Computer Modern")

for (t, c) in zip(t_show, colors2)
    integrand = ΔG_n_abs .* exp.(-σ_n_f64 .* t)
    plot!(fig2, σ_n_f64, integrand,
          label = latexstring("t = $t\\,M"), lw = 1.5, color = c)
end
vline!(fig2, [5.0], label = L"\sigma_{\max}=5",
       color = :black, lw = 1, ls = :dash)

savefig(fig2, joinpath(OUTDIR, "convergence_integrand.pdf"))
println("  → convergence_integrand.pdf")

# ============================================================
# Figure 3: convergence_sigma_max.pdf
# ============================================================
# Uses ψ_BC⁺ (pos. imag. axis, Float64-stable) to demonstrate
# σ_max dependence without BigFloat overhead.

println("====== Fig.3: convergence_sigma_max ======")

σ_max_vals = [1.0, 2.0, 5.0, 10.0, 20.0]
colors3    = [:navy, :royalblue, :seagreen, :darkorange, :crimson]
t_conv     = collect(range(-30.0, -0.5; length=600))

fig3 = plot(
    xlabel = L"|t|/M",
    ylabel = L"|\mathrm{Re}[\psi_{BC}^+(t)]|",
    xscale = :log10, yscale = :log10,
    title  = "Convergence of \$\\psi_{BC}^+\$ with \$\\sigma_{\\max}\$  (pos. axis)",
    framestyle = :box, grid = true, legend = :topright,
    size = (750, 450), dpi = 150, fontfamily = "Computer Modern")

# reference: numerical waveform for t < 0
idx_neg_ref = findall(t_all .< -0.5)
plot!(fig3, abs.(t_all[idx_neg_ref]), abs.(real.(ψ_num[idx_neg_ref])),
    label = "Numerical (ref.)", lw = 2, color = :steelblue, alpha = 0.6)

for (σ_max, c) in zip(σ_max_vals, colors3)
    # recompute ΔG_p on grid up to σ_max
    Nσ_c  = max(50, round(Int, 300 * σ_max / 20.0))
    σ_c, Δσ_c = logspaced_weights(1e-3, σ_max, Nσ_c)
    ΔG_c  = compute_DG_pos(σ_c, Float64(a), 1e-6)
    ψ_c   = [-im/(2π) * sum(ΔG_c[i]*Δσ_c[i]*exp(σ_c[i]*t) for i in 1:Nσ_c)
             for t in t_conv]
    plot!(fig3, abs.(t_conv), abs.(real.(ψ_c)),
          label = latexstring("\\sigma_{\\max}=$(σ_max)"),
          lw = 1.5, color = c, ls = :dash)
end

savefig(fig3, joinpath(OUTDIR, "convergence_sigma_max.pdf"))
println("  → convergence_sigma_max.pdf")

println("\n全図を $(OUTDIR) に保存しました。")
