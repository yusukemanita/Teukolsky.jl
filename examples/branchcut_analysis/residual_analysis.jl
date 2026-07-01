using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using CSV, DataFrames, Plots, LaTeXStrings, Printf

# ============================================================
#  Residual analysis of the waveform decomposition
#
#  t > 0:  residual = ψ_num − ψ_QNM − ψ_NIA
#  t < 0:  residual = ψ_num − ψ_PIA
#
#  NIA (neg. imag. axis, t>0):
#    ψ_NIA(t) = (i/2π) ∫ ΔG⁻(σ) e^{−σt} dσ
#  PIA (pos. imag. axis, t<0):
#    ψ_PIA(t) = (−i/2π) ∫ ΔG⁺(σ) e^{+σt} dσ
# ============================================================

OUTDIR = @__DIR__
s, l, m, a = -2, 2, 2, 0.9
δ = 1e-6

# ── helpers ───────────────────────────────────────────────────
function logspaced_weights(σ_min, σ_max, Nσ)
    σ = exp.(range(log(σ_min), log(σ_max); length=Nσ))
    Δσ = diff([0.0; (σ[1:end-1] .+ σ[2:end]) ./ 2; σ[end]])
    return σ, Δσ
end

function load_qnm_data(l::Int, m::Int, n::Int, a_target::Float64)
    fp = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$(n).dat"
    df = CSV.read(fp, DataFrame, header=false)
    idx = argmin(abs.(df[!, 1] .- a_target))
    ω = (df[idx, 2] + im * df[idx, 3]) / 2
    B = (df[idx, 8] + im * df[idx, 9]) / 16
    return ω, B
end

# ── Numerical waveform ────────────────────────────────────────
println("Loading numerical waveform ...")
wf_df = CSV.read(
    "/Users/yusuke/work/matrix_pencile/data/psi4_waveform_l2m2a0.900.csv",
    DataFrame)
t_csv = wf_df[!, 1]
ψ_csv = wf_df[!, 2] .+ im .* wf_df[!, 3]
println("  $(length(t_csv)) points, t ∈ [$(t_csv[1]), $(t_csv[end])]")

# ── ΔG⁻  (neg. imag. axis, used for ψ_NIA, t>0) ─────────────
println("\nComputing ΔG⁻(σ) ...")
Nσ_n = 500
σ_n, Δσ_n = logspaced_weights(1e-3, 5.0, Nσ_n)
ΔG_n = Vector{ComplexF64}(undef, Nσ_n)
function _fill_DG_neg!(ΔG, σ_grid, s, l, m, a)
    ν_prev = nothing
    for i in eachindex(σ_grid)
        σ   = σ_grid[i]
        ω_R = -im * σ
        if ν_prev === nothing
            qi = compute_q(s, l, m, a, ω_R; nmax=100)
        else
            qi = compute_q(s, l, m, a, ω_R; nmax=100, ν_init=ν_prev, method="Newton")
        end
        ν_prev = qi.ν
        amp_R  = compute_amplitudes_nufixed(s, l, m, a, ω_R, ν_prev; nmax=100)
        q_val  = qi.q
        ΔG[i]  = im * q_val * amp_R.Bref^2 /
                 (2im * ω_R * amp_R.Binc * (amp_R.Binc + im * q_val * amp_R.Bref))
        i % 100 == 0 && (print("."); flush(stdout))
    end
end
_fill_DG_neg!(ΔG_n, σ_n, s, l, m, a)
println(" done")

# ── ΔG⁺  (pos. imag. axis, used for ψ_PIA, t<0) ─────────────
println("\nComputing ΔG⁺(σ) ...")
Nσ_p = 300
σ_p, Δσ_p = logspaced_weights(1e-3, 10.0, Nσ_p)
ΔG_p = Vector{ComplexF64}(undef, Nσ_p)
function _fill_DG_pos!(ΔG, σ_grid, s, l, m, a, δ)
    for i in eachindex(σ_grid)
        σ   = σ_grid[i]
        ω_R = δ + im * σ;  ω_L = -δ + im * σ
        amp_R = compute_amplitudes(s, l,  m, a, ω_R)
        G_R   = amp_R.Bref / (2im * ω_R * amp_R.Binc)
        amp_m = compute_amplitudes(s, l, -m, a, ω_R)
        G_L   = conj(amp_m.Bref) / (2im * ω_L * conj(amp_m.Binc))
        ΔG[i] = G_R - G_L
        i % 60 == 0 && (print("."); flush(stdout))
    end
end
_fill_DG_pos!(ΔG_p, σ_p, s, l, m, a, δ)
println(" done")

# ── QNM ──────────────────────────────────────────────────────
println("\nLoading QNM data ...")
qnm_pro   = [load_qnm_data(l,  m, n, Float64(a)) for n in 0:7]
qnm_retro = [load_qnm_data(l, -m, n, Float64(a)) for n in 0:7]
all_qnm   = vcat(qnm_pro, qnm_retro)
for (n, (ω_n, B_n)) in enumerate(qnm_pro)
    @printf("  pro  n=%d: ω=%+.5f%+.5fi  |B|=%.3e\n",
            n-1, real(ω_n), imag(ω_n), abs(B_n))
end

# ── Subsampled t grids (no need for all 70k CSV points in plot) ──
# Use every 7th point from CSV for t > 0, every 7th for t < 0
step = 7
idx_pos_raw = findall(t_csv .>  0.5)
idx_neg_raw = findall(t_csv .< -0.5)
idx_pos = idx_pos_raw[1:step:end]
idx_neg = idx_neg_raw[1:step:end]
t_pos = t_csv[idx_pos]
t_neg = t_csv[idx_neg]
ψ_pos = ψ_csv[idx_pos]
ψ_neg = ψ_csv[idx_neg]
println("\n  t>0 plot points: $(length(t_pos))")
println("  t<0 plot points: $(length(t_neg))")

# ── Branch cut integrals via matrix multiply ──────────────────
# ψ_NIA(t) = (i/2π) Σ_j ΔG⁻_j Δσ_j e^{-σ_j t}
# ψ_PIA(t) = (-i/2π) Σ_j ΔG⁺_j Δσ_j e^{+σ_j t}
#
# Efficient: weight vector w = ΔG .* Δσ,  kernel K_{jk} = exp(±σ_j t_k)
# ψ = (±i/2π) * w' * K

println("\nComputing ψ_NIA  ($(length(t_pos)) × $(Nσ_n) matrix) ...")
w_n = ΔG_n .* Δσ_n
E_n = exp.(-σ_n .* t_pos')   # Nσ_n × N_t
ψ_NIA = (im / (2π)) .* vec(w_n' * E_n)
println("  done")

println("Computing ψ_PIA  ($(length(t_neg)) × $(Nσ_p) matrix) ...")
w_p = ΔG_p .* Δσ_p
E_p = exp.(σ_p .* t_neg')    # Nσ_p × N_t
ψ_PIA = (-im / (2π)) .* vec(w_p' * E_p)
println("  done")

# ── QNM on plot grids ─────────────────────────────────────────
println("Computing ψ_QNM on t>0 grid ...")
ψ_QNM_pos = [-sum(B * exp(-im * ω * t) for (ω, B) in all_qnm) for t in t_pos]
println("  done")

# ── Residuals ─────────────────────────────────────────────────
resid_pos = ψ_pos .- ψ_QNM_pos .- ψ_NIA
resid_neg = ψ_neg .- ψ_PIA

# ── Figures ──────────────────────────────────────────────────
println("\nPlotting ...")

# Panel A: t > 0
pA = plot(
    xlabel = L"t / M",
    ylabel = L"|\mathrm{Re}[\cdot]|",
    yscale = :log10, xlim = (0, 600), ylim = (:auto, 1e-2),
    title  = L"t > 0",
    framestyle = :box, grid = true, legend = :topright,
    fontfamily = "Computer Modern")

plot!(pA, t_pos, abs.(real.(ψ_pos)),
      label = L"\psi_\mathrm{num}", lw = 2, color = :steelblue)
plot!(pA, t_pos, abs.(real.(ψ_QNM_pos .+ ψ_NIA)),
      label = L"\psi_\mathrm{QNM}+\psi_\mathrm{NIA}", lw = 1.5, color = :seagreen, ls = :dash)
plot!(pA, t_pos, abs.(real.(resid_pos)),
      label = L"\psi_\mathrm{num}-\psi_\mathrm{QNM}-\psi_\mathrm{NIA}", lw = 1.5, color = :crimson)

# Panel B: t < 0
pB = plot(
    xlabel = L"t / M",
    ylabel = L"|\mathrm{Re}[\cdot]|",
    yscale = :log10, xlim = (-100, 0), ylim = (1e-14, 1e-2),
    title  = L"t < 0",
    framestyle = :box, grid = true, legend = :topright,
    fontfamily = "Computer Modern")

plot!(pB, t_neg, abs.(real.(ψ_neg)),
      label = L"\psi_\mathrm{num}", lw = 2, color = :steelblue)
plot!(pB, t_neg, abs.(real.(ψ_PIA)),
      label = L"\psi_\mathrm{PIA}", lw = 1.5, color = :crimson, ls = :dash)
plot!(pB, t_neg, abs.(real.(resid_neg)),
      label = L"\psi_\mathrm{num}-\psi_\mathrm{PIA}", lw = 1.5, color = :darkorange)

fig = plot(pA, pB, layout = (2, 1), size = (900, 800), dpi = 150)
savefig(fig, joinpath(OUTDIR, "residual_analysis.pdf"))
savefig(fig, joinpath(OUTDIR, "residual_analysis.png"))
println("Saved: residual_analysis.pdf / .png")
display(fig)
