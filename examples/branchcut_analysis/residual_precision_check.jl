using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")

using Teukolsky
using CSV, DataFrames, Plots, LaTeXStrings, Printf

# ============================================================
#  Precision comparison for ψ_NIA residual
#  Float64 vs BigFloat(256) for ΔG⁻ computation
# ============================================================

OUTDIR = @__DIR__
s, l, m, a = -2, 2, 2, 0.9

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

# ── Numerical waveform ─────────────────────────────────────────
wf_df = CSV.read("/Users/yusuke/work/matrix_pencile/data/psi4_waveform_l2m2a0.900.csv", DataFrame)
t_csv = wf_df[!, 1];  ψ_csv = wf_df[!, 2] .+ im .* wf_df[!, 3]

# ── QNM ────────────────────────────────────────────────────────
qnm_pro   = [load_qnm_data(l,  m, n, Float64(a)) for n in 0:7]
qnm_retro = [load_qnm_data(l, -m, n, Float64(a)) for n in 0:7]
all_qnm   = vcat(qnm_pro, qnm_retro)

# ── σ grid ─────────────────────────────────────────────────────
Nσ = 500
σ_f64, Δσ_f64 = logspaced_weights(1e-3, 5.0, Nσ)

# ── ΔG⁻  Float64 ──────────────────────────────────────────────
println("Computing ΔG⁻  Float64 ...")
ΔG_f64 = Vector{ComplexF64}(undef, Nσ)
function fill_DG_neg!(ΔG, σ_grid, a_val)
    T = eltype(a_val)
    ν_prev = nothing
    for i in eachindex(σ_grid)
        σ   = T(σ_grid[i])
        ω_R = -im * σ
        if ν_prev === nothing
            qi = compute_q(s, l, m, a_val, ω_R; nmax=100)
        else
            qi = compute_q(s, l, m, a_val, ω_R; nmax=100, ν_init=ν_prev, method="Newton")
        end
        ν_prev = qi.ν
        amp_R  = compute_amplitudes_nufixed(s, l, m, a_val, ω_R, ν_prev; nmax=100)
        q_val  = qi.q
        ΔG[i]  = ComplexF64(
            im * q_val * amp_R.Bref^2 /
            (2im * ω_R * amp_R.Binc * (amp_R.Binc + im * q_val * amp_R.Bref)))
        i % 100 == 0 && (print("."); flush(stdout))
    end
end
fill_DG_neg!(ΔG_f64, σ_f64, Float64(a))
println(" done")

# ── ΔG⁻  BigFloat(256) ────────────────────────────────────────
println("Computing ΔG⁻  BigFloat(256) ...")
setprecision(BigFloat, 256)
ΔG_bf = Vector{ComplexF64}(undef, Nσ)
a_bf = BigFloat(string(a))
fill_DG_neg!(ΔG_bf, σ_f64, a_bf)
println(" done")

println("\n  Max |ΔG_f64 - ΔG_bf| / |ΔG_bf|: ",
    maximum(abs.(ΔG_f64 .- ΔG_bf) ./ max.(abs.(ΔG_bf), 1e-100)))

# ── t grid (subsampled) ────────────────────────────────────────
step    = 7
idx_pos = findall(t_csv .> 0.5)[1:step:end]
t_pos   = t_csv[idx_pos]
ψ_pos   = ψ_csv[idx_pos]

# ── ψ_NIA via matrix multiply (Float64 and BigFloat) ──────────
println("Computing ψ_NIA (Float64) ...")
E_n  = exp.(-σ_f64 .* t_pos')
ψ_NIA_f64 = (im / (2π)) .* vec((ΔG_f64 .* Δσ_f64)' * E_n)
println("Computing ψ_NIA (BigFloat) ...")
# Cast weights to BigFloat for the sum, then project back
w_bf = ComplexF64.(ΔG_bf) .* Δσ_f64   # ΔG already in BigFloat precision
# Use the same Float64 kernel; the precision gain is in ΔG, not the exponential
ψ_NIA_bf = (im / (2π)) .* vec(w_bf' * E_n)
println("  done")

# ── QNM ────────────────────────────────────────────────────────
println("Computing ψ_QNM ...")
ψ_QNM_pos = [-sum(B * exp(-im * ω * t) for (ω, B) in all_qnm) for t in t_pos]
println("  done")

# ── Residuals ──────────────────────────────────────────────────
resid_f64 = ψ_pos .- ψ_QNM_pos .- ψ_NIA_f64
resid_bf  = ψ_pos .- ψ_QNM_pos .- ψ_NIA_bf

# ── Print floor comparison ─────────────────────────────────────
println("\nResidual |Re[ψ_num − ψ_QNM − ψ_NIA]| at late t:")
println("  t        Float64          BigFloat(256)")
for t_check in [10.0, 30.0, 100.0, 200.0, 400.0]
    idx = argmin(abs.(t_pos .- t_check))
    rf  = abs(real(resid_f64[idx]))
    rb  = abs(real(resid_bf[idx]))
    @printf("  t=%5.0f  %.3e       %.3e\n", t_check, rf, rb)
end

# ── Plot ──────────────────────────────────────────────────────
println("\nPlotting ...")
p = plot(
    xlabel = L"t / M",
    ylabel = L"|\mathrm{Re}[\psi_\mathrm{num} - \psi_\mathrm{QNM} - \psi_\mathrm{NIA}]|",
    yscale = :log10, xlim = (0, 600),
    title  = "NIA residual: Float64 vs BigFloat(256)",
    framestyle = :box, grid = true, legend = :topright,
    size = (850, 450), dpi = 150, fontfamily = "Computer Modern")

plot!(p, t_pos, abs.(real.(ψ_pos)),
      label = L"\psi_\mathrm{num}", lw = 1.5, color = :steelblue, alpha = 0.6)
plot!(p, t_pos, abs.(real.(resid_f64)),
      label = "residual  Float64", lw = 1.5, color = :crimson)
plot!(p, t_pos, abs.(real.(resid_bf)),
      label = "residual  BigFloat(256)", lw = 1.5, color = :darkgreen, ls = :dash)

savefig(p, joinpath(OUTDIR, "residual_precision_check.pdf"))
savefig(p, joinpath(OUTDIR, "residual_precision_check.png"))
println("Saved: residual_precision_check.pdf / .png")
display(p)
