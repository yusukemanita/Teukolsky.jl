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

OUTDIR = @__DIR__
s, l, m = -2, 2, 2
a = 0.9

gr()   # GR backend — saves PDF/PNG without extra dependencies

# ── helper: load QNM excitation factor ───────────────────────

function load_qnm_data(l::Int, m::Int, n::Int, a_target::Float64)
    filepath = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$(n).dat"
    println("Loading QNM data from: $filepath")
    df  = CSV.read(filepath, DataFrame, header=false)
    idx = argmin(abs.(df[!, 1] .- a_target))
    ω   = (df[idx, 2] + im * df[idx, 3]) / 2
    B   = (df[idx, 8] + im * df[idx, 9]) / 16
    return ω, B
end

# ── helper: compute ΔG⁻(σ) on neg. imag. axis ────────────────
# use_bigfloat=true  → BigFloat arithmetic (slow, high precision)
# use_bigfloat=false → Float64 arithmetic (fast)

function compute_DG_neg(σ_grid::Vector{Float64}, a_val; use_bigfloat::Bool=true)
    Nσ = length(σ_grid)
    if use_bigfloat
        ΔG  = Vector{Complex{BigFloat}}(undef, Nσ)
        a_c = BigFloat(a_val)
    else
        ΔG  = Vector{ComplexF64}(undef, Nσ)
        a_c = Float64(a_val)
    end
    ν_prev = nothing   # branch tracking: carry ν from previous σ
    for i in 1:Nσ
        σ      = use_bigfloat ? BigFloat(σ_grid[i]) : σ_grid[i]
        ω_R    = -im*σ
        # Branch tracking: first point uses Monodromy; subsequent points use
        # Newton continuation from ν_prev to avoid acos branch-cut jumps.
        if ν_prev === nothing
            q_info = compute_q(s, l, m, a_c, ω_R; nmax=100)
        else
            q_info = compute_q(s, l, m, a_c, ω_R; nmax=100,
                               ν_init=ν_prev, method="Newton")
        end
        ν_cur  = q_info.ν

        # Use the same ν as q_info to ensure Bref/Binc are consistent with q.
        # compute_amplitudes with free ν would use a different (Monodromy) ν.
        amp_R  = compute_amplitudes_nufixed(s, l, m, a_c, ω_R, ν_cur; nmax=100)
        q_val  = q_info.q
        ν_prev = ν_cur   # update branch for next step
        ΔG[i]  =  im * q_val * amp_R.Bref^2 /
                (2im * ω_R * amp_R.Binc * (amp_R.Binc + im * q_val * amp_R.Bref))
        i % 20 == 0 && (print("."); flush(stdout))
    end
    return ΔG
end

# ── helper: compute ΔG⁺(σ) on pos. imag. axis (Float64) ─────

# ΔG on the positive imaginary axis via G_R - G_L (Float64, δ-offset)
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

# ── Neg. imag. axis ───────────────────────────────────────────
USE_BIGFLOAT_NEG = false   # ← set false for Float64, true for BigFloat
println("\n--- ΔG⁻(σ)  [neg. imag. axis, $(USE_BIGFLOAT_NEG ? "BigFloat" : "Float64")] ---")
Nσ_n  = 500
σ_n_f64, Δσ_n_f64 = logspaced_weights(1e-3, 5.0, Nσ_n)
ΔG_n  = compute_DG_neg(σ_n_f64, a; use_bigfloat=USE_BIGFLOAT_NEG)
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
        sv += ΔG_n[i] * Δσ_n_f64[i] * exp(-σ_n_f64[i]*BigFloat(t))
    end
    ψ_BC_neg[k] = ComplexF64(im/(2π) * sv)
    k % 100 == 0 && (print("."); flush(stdout))
end
println(" 完了")

ψ_QNM = [-sum(B_n*exp(-im*ω_n*t) for (ω_n, B_n) in qnm_pro) for t in t_all] .+ 
    [-sum(B_n*exp(-im*ω_n*t) for (ω_n, B_n) in qnm_retro) for t in t_all]
# ψ_sum = ψ_QNM .+ ψ_BC_neg

# ============================================================
# Figure 1: waveform_decomposition.pdf
# ============================================================

println("\n====== Fig.1: waveform_decomposition ======")

# residual on t_pos grid (nearest-neighbour to CSV)
idx_pos  = findall(t_all .> 0.0)
t_ref    = t_all[idx_pos]
ψ_ref    = ψ_num[idx_pos]
ψ_num_minus_QNM = ψ_ref .- ψ_QNM[idx_pos]

p1 = plot(
    xlabel = L"t/M",
    ylabel = L"|\mathrm{Re}[\psi_4]|",
    yscale = :log10, ylim = (1e-14, 1e-2),
    framestyle = :box, grid = true, legend = :topright)
plot!(p1, t_all, abs.(real.(ψ_num)),
    label = "Numerical (high-prec.)", lw = 2, color = :steelblue)
plot!(p1, t_neg, abs.(real.(ψ_BC_pos)),
    label = L"\psi_{PIA}\ (t<0)", lw = 1.5, color = :crimson, ls = :dash)
plot!(p1, t_pos, abs.(real.(ψ_BC_neg)),
    label = L"\psi_{NIA}\ (t>0)", lw = 1.5, color = :darkorange, ls = :solid)
plot!(p1, t_ref, abs.(real.(ψ_QNM[idx_pos])),
    label = L"\psi_{QNM}\ (n\leq 7,\ \pm m)",
    lw = 1, color = :darkgreen, ls = :dash)
plot!(p1, t_ref, abs.(real.(ψ_num_minus_QNM)),
    label = L"\psi_{num}-\psi_{QNM}",
    lw = 1, color = :magenta, ls = :dash)
vline!(p1, [0.0], label = "", color = :black, lw = 0.8, ls = :dash)

savefig(p1, joinpath(OUTDIR, "waveform_decomposition.pdf"))
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
    ylabel = L"|\Delta G^+_{NIA}(\sigma)|\,e^{-\sigma t}",
    yscale = :log10, ylim = (1e-20, 1e2),
    title  = "Integrand decay vs. \$\\sigma\$ for several \$t\$  (neg. axis)",
    framestyle = :box, grid = true, legend = :bottomright,
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
# Uses ψ_PIA (pos. imag. axis, Float64-stable) to demonstrate
# σ_max dependence without BigFloat overhead.

println("====== Fig.3: convergence_sigma_max ======")

σ_max_vals = [1.0, 2.0, 5.0, 10.0, 20.0]
colors3    = [:navy, :royalblue, :seagreen, :darkorange, :crimson]
t_conv     = collect(range(-30.0, -0.5; length=600))

fig3 = plot(
    xlabel = L"|t|/M",
    ylabel = L"|\mathrm{Re}[\psi_{PIA}(t)]|",
    xscale = :log10, yscale = :log10,
    title  = "Convergence of \$\\psi_{PIA}\$ with \$\\sigma_{\\max}\$  (pos. axis)",
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
