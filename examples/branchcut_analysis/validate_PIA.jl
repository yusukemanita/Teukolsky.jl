using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")

using Teukolsky
using CSV, DataFrames, Plots, LaTeXStrings, Printf

# ============================================================
#  Validate ΔG⁺ computation by comparing ψ_PIA from
#  (a) Julia MST direct computation
#  (b) Wolfram TeukolskyRadial CSV
#  against the numerical waveform (reference)
# ============================================================

s, l, m, a = -2, 2, 2, 0.9
δ = 1e-6

# ── Load Wolfram CSV ──────────────────────────────────────────
wf_csv = joinpath(@__DIR__, "DG_pos_wolfram.csv")
df_wf  = CSV.read(wf_csv, DataFrame, types=Dict(:sigma=>Float64))

function safe_f64(s)
    try; return parse(Float64, replace(string(s), r"(?<=[0-9])\.(?=[^0-9eE]|$)" => ".0")); catch; return NaN; end
end
σ_wf   = df_wf.sigma
ReDG   = safe_f64.(string.(df_wf.ReDG))
ImDG   = safe_f64.(string.(df_wf.ImDG))
valid  = .!isnan.(ReDG) .& .!isnan.(ImDG)
σ_w    = σ_wf[valid]
ΔG_w   = ReDG[valid] .+ im .* ImDG[valid]
println("Wolfram: $(length(σ_w)) valid σ, max σ = $(maximum(σ_w))")

# Wolfram trapezoidal weights
Δσ_w = diff([0.0; (σ_w[1:end-1] .+ σ_w[2:end]) ./ 2; σ_w[end]])

# ── Compute Julia ΔG⁺ on the Wolfram σ grid ──────────────────
println("Computing Julia ΔG⁺ (direct, ν≈2 branch) ...")
function compute_DG_direct(σ_grid, s, l, m, a, δ)
    N = length(σ_grid)
    ΔG = Vector{ComplexF64}(undef, N)
    for i in 1:N
        σ    = σ_grid[i]
        ω_R  = δ + im*σ;  ω_L = -δ + im*σ
        amp_R = compute_amplitudes(s, l,  m, a, ω_R)
        G_R   = amp_R.Bref / (2im * ω_R * amp_R.Binc)
        amp_m = compute_amplitudes(s, l, -m, a, ω_R)
        G_L   = conj(amp_m.Bref) / (2im * ω_L * conj(amp_m.Binc))
        ΔG[i] = G_R - G_L
        i % 40 == 0 && (print("."); flush(stdout))
    end
    return ΔG
end
ΔG_j   = compute_DG_direct(σ_w, s, l, m, a, δ)
println(" done")

# ── Numerical waveform ────────────────────────────────────────
wf_df = CSV.read(
    "/Users/yusuke/work/matrix_pencile/data/psi4_waveform_l2m2a0.900.csv",
    DataFrame)
t_csv = wf_df[!, 1]
ψ_num = wf_df[!, 2] .+ im .* wf_df[!, 3]

# ── Compute ψ_PIA from both ──────────────────────────────────
t_neg = range(-50.0, -0.5; length=300)
function psia_integral(σ_grid, ΔG, Δσ, t)
    -im/(2π) * sum(ΔG[i]*Δσ[i]*exp(σ_grid[i]*t) for i in eachindex(σ_grid))
end

println("Computing ψ_PIA (Wolfram & Julia) ...")
ψ_W = [psia_integral(σ_w, ΔG_w, Δσ_w, t) for t in t_neg]
ψ_J = [psia_integral(σ_w, ΔG_j, Δσ_w, t) for t in t_neg]
println(" done")

# ── Plot ──────────────────────────────────────────────────────
idx_neg = findall(t_csv .< -0.5)
p = plot(
    xlabel = L"|t|/M",
    ylabel = L"|\mathrm{Re}[\psi]|",
    xscale = :log10, yscale = :log10,
    ylim   = (1e-12, 1e-2),
    title  = "ψ_PIA comparison: Wolfram vs Julia  (σ_max ≈ 4.9)",
    framestyle = :box, grid = true, legend = :topright,
    size = (850, 500), dpi = 150, fontfamily = "Computer Modern")

plot!(p, abs.(t_csv[idx_neg]), abs.(real.(ψ_num[idx_neg])),
      label = "Numerical ref.", lw = 2, color = :steelblue, alpha = 0.8)
plot!(p, abs.(collect(t_neg)), abs.(real.(ψ_W)),
      label = "Wolfram ΔG⁺", lw = 2, color = :crimson)
plot!(p, abs.(collect(t_neg)), abs.(real.(ψ_J)),
      label = "Julia ΔG⁺ (ν≈2)", lw = 2, color = :darkorange, ls = :dash)

savefig(p, joinpath(@__DIR__, "validate_PIA.pdf"))
println("Saved: validate_PIA.pdf")
display(p)

# ── Summary statistics ────────────────────────────────────────
println("\n|ΔG| ratio (Julia/Wolfram) at selected σ:")
for i in [10, 30, 60, 100, 130, 155]
    @printf("  σ=%.3f  |Julia|=%.3e  |Wolfram|=%.3e  ratio=%.2f\n",
            σ_w[i], abs(ΔG_j[i]), abs(ΔG_w[i]), abs(ΔG_j[i])/max(abs(ΔG_w[i]),1e-30))
end
