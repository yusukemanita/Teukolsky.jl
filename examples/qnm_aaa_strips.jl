using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")

using Teukolsky
using BaryRational
using Plots
using LinearAlgebra
using Printf
using LaTeXStrings
using CSV, DataFrames

# ── Physical parameters ──
s, l, m, a = -2, 2, 2, 0.9

omega = filter(!iszero, -3.0:0.01:3.0)

function Gapp(ω)
    if real(ω) > 0
        amp = compute_amplitudes(s, l, m, a, ω)
        return amp.Bref / (2im * ω * amp.Binc)
    else
        amp = compute_amplitudes(s, l, -m, a, -conj(ω))
        return conj(amp.Bref) / (2im * ω * conj(amp.Binc))
    end
end

# ── QNM reference data ──────────────────────────────────────────────
function load_qnm_data(l::Int, m::Int, n::Int, a_target::Float64)
    filepath = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$n.dat"
    df = CSV.read(filepath, DataFrame, header=false)
    idx = argmin(abs.(df[!, 1] .- a_target))
    ω = (df[idx, 2] + im * df[idx, 3]) / 2
    B = (df[idx, 8] + im * df[idx, 9]) / 16
    return ω, B
end

ω22  = [load_qnm_data(l,  m, n, a)[1] for n in 0:7]
ω22R = [-conj(load_qnm_data(l, -m, n, a)[1]) for n in 0:7]
ω_known = [ω22; ω22R]

# ── 4 strips ─────────────────────────────────────────────────────────
strip_imags = [0.0, -0.3, -0.6, -0.9]

panels = map(strip_imags) do σ
    omega_strip = ComplexF64.(omega) .+ σ*im
    GF = [Gapp(ω) for ω in omega_strip]

    approx = aaa(omega_strip, GF; tol=1e-13, mmax=150)
    poles, res, zeros = prz(approx)
    println("Im=$σ  support=$(length(approx.x))  poles=$(length(poles))  zeros=$(length(zeros))")

    res_log = log10.(clamp.(abs.(res), 1e-30, Inf))

    # strip line for visual reference
    p = scatter(real.(ω_known), imag.(ω_known),
        label="QNM ref", marker=:star5, ms=7, color=:blue, alpha=0.8,
        xlabel=L"\Re(\omega)", ylabel=L"\Im(\omega)",
        title=latexstring("\\mathrm{Im}(\\omega_0)=$(σ)"),
        xlim=(-2.0, 2.0), ylim=(-2.0, 0.3),
        framestyle=:box, grid=true, legend=:topright,
        fontfamily="Computer Modern", dpi=120)

    # strip line
    hline!(p, [σ], lw=1, ls=:dash, color=:gray, label="strip")

    scatter!(p, real.(poles), imag.(poles),
        label="poles", marker=:circle, ms=5,
        marker_z=res_log, color=:plasma,
        colorbar=false)

    scatter!(p, real.(zeros), imag.(zeros),
        label="zeros", marker=:xcross, ms=4, color=:green, alpha=0.7)

    p
end

fig = plot(panels..., layout=(2, 2), size=(1100, 800))
savefig(fig, joinpath(@__DIR__, "..", "figures", "qnm_strips.png"))
println("Saved: figures/qnm_strips.png")
display(fig)
