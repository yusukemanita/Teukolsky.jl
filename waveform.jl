using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Plots
using LaTeXStrings
using CSV, DataFrames

# ── Parameters ──────────────────────────────────────────────
p = WaveformParams(
    s=-2, l=2, m=2, a=0.7,
    N=3000, ω_max=6.0,
    t_ini=-100.0, t_max=600.0, Nt=3000,taper_frac=0.1
)

# ── Compute ─────────────────────────────────────────────────
t_grid, ψ, GF, ω_grid = compute_waveform(p)

ψ_real = real.(ψ)
ψ_imag = imag.(ψ)
println("max|Im/Re| = $(round(maximum(abs.(ψ_imag)) / (maximum(abs.(ψ_real)) + 1e-30), sigdigits=3))")

# ── QNM reference ────────────────────────────────────────────
function load_qnm(l, m, n, a_target)
    fp = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$n.dat"
    df = CSV.read(fp, DataFrame, header=false)
    idx = argmin(abs.(df[!, 1] .- a_target))
    return (df[idx, 2] + im * df[idx, 3]) / 2
end
ω_qnm = [load_qnm(p.l, p.m, n, p.a) for n in 0:3]

# ── Plot ─────────────────────────────────────────────────────
fig = plot(t_grid, abs.(ψ_real),
    xlabel=L"t\ [M]", ylabel=L"|\psi_4(t)|",
    label=L"|\Re[\psi_4]|", lw=1.0,
    title="Ringdown waveform  (s=$(p.s), l=$(p.l), m=$(p.m), a=$(p.a))",
    framestyle=:box, grid=true, fontfamily="Computer Modern",
    dpi=120, yscale=:log10)
savefig(fig, "waveform.png")
println("Saved waveform.png")
fig
