using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using CSV, DataFrames
using Plots
using LaTeXStrings
using Printf

# ============================================================
#  Compare ΔG⁺(σ) from Wolfram TeukolskyRadial vs. Julia MST
#  Valid range: σ ≤ 4.9  (both methods fail beyond ~5)
# ============================================================

s, l, m, a = -2, 2, 2, 0.9

# ── Load Wolfram CSV ──────────────────────────────────────────
wf_csv = joinpath(@__DIR__, "DG_pos_wolfram.csv")
df_wf  = CSV.read(wf_csv, DataFrame, types=Dict(:sigma=>Float64))

# Keep only rows with numeric (non-Indeterminate) ΔG values
# Columns ReDG, ImDG are String31 due to Indeterminate entries
function try_parse_f64(s::AbstractString)
    # Handle Wolfram notations: "1.e1" → 10.0, "-0." → 0.0
    s2 = replace(s, r"(?<=[0-9])\.(?=[^0-9]|$)" => ".0")  # trailing dot
    try; return parse(Float64, s2); catch; return NaN; end
end
try_parse_f64(x::Float64) = x

σ_wf_all  = df_wf.sigma
ReDG_all  = try_parse_f64.(string.(df_wf.ReDG))
ImDG_all  = try_parse_f64.(string.(df_wf.ImDG))

valid = .!isnan.(ReDG_all) .& .!isnan.(ImDG_all)
σ_wf  = σ_wf_all[valid]
ΔG_wf = ReDG_all[valid] .+ im .* ImDG_all[valid]
println("Wolfram valid rows: $(length(σ_wf)),  σ_max = $(maximum(σ_wf))")

# ── Compute Julia ΔG⁺ at the same σ grid ─────────────────────
println("\nComputing Julia ΔG⁺ at Wolfram σ grid ...")
function compute_DG_wf_grid(σ_grid, s, l, m, a)
    ΔG = Vector{ComplexF64}(undef, length(σ_grid))
    ν_prev = nothing
    for i in eachindex(σ_grid)
        σ = σ_grid[i]
        ω = im * σ
        if ν_prev === nothing
            qt = compute_qtilde(s, l, m, a, ω; nmax=100)
        else
            qt = compute_qtilde(s, l, m, a, ω; nmax=100,
                                ν_init=ν_prev, method="Newton")
        end
        ν_prev = qt.ν
        ΔG[i]  = -qt.qtilde / (2 * ω)
        i % 30 == 0 && (print("."); flush(stdout))
    end
    return ΔG
end
ΔG_julia = compute_DG_wf_grid(σ_wf, s, l, m, a)
println(" done")

# ── Comparison table (selected σ) ────────────────────────────
println("\n σ          Wolfram ΔG                 Julia ΔG               |diff|")
for i in 10:20:length(σ_wf)
    dg_wf  = ΔG_wf[i]
    dg_jl  = ΔG_julia[i]
    d      = abs(dg_wf - dg_jl)
    @printf("  %6.3f   %+8.4f %+8.4fi   %+8.4f %+8.4fi   %.2e\n",
            σ_wf[i], real(dg_wf), imag(dg_wf),
            real(dg_jl), imag(dg_jl), d)
end

# ── Plot: Re[ΔG] and |ΔG| comparison ─────────────────────────
p1 = plot(xlabel=L"\sigma", ylabel=L"\mathrm{Re}[\Delta G^+]",
          title="Re[ΔG⁺]: Wolfram vs Julia  (pos. imag. axis)",
          framestyle=:box, grid=true, legend=:topright,
          size=(850,400), dpi=150, fontfamily="Computer Modern")
plot!(p1, σ_wf, real.(ΔG_wf),
      label="Wolfram", lw=2, color=:steelblue)
scatter!(p1, σ_wf[1:5:end], real.(ΔG_julia[1:5:end]),
         label="Julia MST", marker=:circle, ms=3, color=:crimson, alpha=0.8)

p2 = plot(xlabel=L"\sigma", ylabel=L"|\Delta G^+|",
          title="|ΔG⁺|: Wolfram vs Julia",
          yscale=:log10,
          framestyle=:box, grid=true, legend=:topleft,
          size=(850,400), dpi=150, fontfamily="Computer Modern")
plot!(p2, σ_wf, abs.(ΔG_wf),
      label="Wolfram", lw=2, color=:steelblue)
scatter!(p2, σ_wf[1:5:end], abs.(ΔG_julia[1:5:end]),
         label="Julia MST", marker=:circle, ms=3, color=:crimson, alpha=0.8)

fig = plot(p1, p2, layout=(2,1), size=(850,800))
savefig(fig, joinpath(@__DIR__, "compare_wolfram_julia.pdf"))
println("Saved: compare_wolfram_julia.pdf")
display(fig)
