using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")

using Teukolsky
using Plots
using LaTeXStrings
using Printf
using CSV
using DataFrames

# ============================================================
#  Slide figure: numerical waveform + tail (ψ_NIA)
#
#  ψ_NIA uses the exact analytical MST monodromy formula
#  (matches combined_waveform.jl) to avoid the symmetry-based
#  BigFloat approximation of ΔG_- used in waveform_decomposition.jl
#  which overshoots the late-time tail by a factor of ~2-3.
#
#    G(ω)      = Bref / (2iω Binc)
#    ΔG_-(σ)   = iq · Bref² / (2iω · Binc · (Binc + iq Bref)),   ω = -iσ
#    ψ_NIA(t)  = (i/2π) ∫₀^σmax ΔG_-(σ) e^{-σt} dσ
# ============================================================

s, l, m = -2, 2, 2
a       = 0.9

# ── Part 1: numerical waveform (CSV) ────────────────────────
println("=== Part 1: numerical waveform ===")
wf_df = CSV.read(
    "/Users/yusuke/work/matrix_pencile/data/psi4_waveform_l2m2a0.900.csv",
    DataFrame)
t_all = wf_df[!, 1]
ψ_num = wf_df[!, 2] .+ im .* wf_df[!, 3]
println("  $(length(t_all)) points, t ∈ [$(t_all[1]), $(t_all[end])]")

# ── Part 2: ψ_NIA via analytical q formula (Float64) ────────
println("\n=== Part 2: ψ_NIA (analytical MST q formula, Float64) ===")
Nσ       = 300
σ_min    = 1e-3
σ_max    = 5.0
σ_grid   = exp.(range(log(σ_min), log(σ_max); length=Nσ))
Δσ_arr   = diff([0.0; (σ_grid[1:end-1] .+ σ_grid[2:end]) ./ 2; σ_max])

ΔG = Vector{ComplexF64}(undef, Nσ)
print("ΔG_-(σ) ($Nσ points) ")
for i in 1:Nσ
    σ      = σ_grid[i]
    ω      = -im * σ
    amp    = compute_amplitudes(s, l, m, a, ω; nmax=100)
    q_val  = compute_q(s, l, m, a, ω; nmax=100).q
    ΔG[i]  = im * q_val * amp.Bref^2 /
             (2im * ω * amp.Binc * (amp.Binc + im * q_val * amp.Bref))
    i % 40 == 0 && (print("."); flush(stdout))
end
println(" done")

t_pos  = range(5.0, 600.0; length=1000)
ψ_tail = [im / (2π) * sum(ΔG[i] * Δσ_arr[i] * exp(-σ_grid[i] * t) for i in 1:Nσ)
          for t in t_pos]

# ── Plot ─────────────────────────────────────────────────────
println("\nplotting ...")

p1 = plot(
    xlabel = L"t/M",
    ylabel = L"|\mathrm{Re}[\psi_4]|",
    yscale = :log10, ylim = (1e-14, 1e-2),
    framestyle = :box, grid = true, legend = :topright,
    legendfontsize = 12, tickfontsize = 11, guidefontsize = 13)
plot!(p1, t_all, abs.(real.(ψ_num)),
    label = "Numerical", lw = 2.5, color = :steelblue)
plot!(p1, collect(t_pos), abs.(real.(ψ_tail)),
    label = "tail", lw = 2.5, color = :darkorange, ls = :dash)

fig = plot(p1, size = (900, 500), dpi = 150, fontfamily = "Computer Modern")
savefig(fig, joinpath(@__DIR__, "waveform_slide.png"))
savefig(fig, joinpath(@__DIR__, "waveform_slide.pdf"))
println("saved waveform_slide.png / .pdf")

# ── quick numerical check vs CSV tail ───────────────────────
println("\n-- late-time tail comparison --")
for t_check in (400.0, 500.0, 600.0)
    idx  = findall(x -> abs(x - t_check) < 2.0, t_all)
    env  = isempty(idx) ? NaN : maximum(abs.(real.(ψ_num[idx])))
    j    = argmin(abs.(collect(t_pos) .- t_check))
    @printf("t=%.0f : numerical max|Re|=%.3e   tail=%.3e   ratio=%.2f\n",
            t_check, env, abs(real(ψ_tail[j])),
            abs(real(ψ_tail[j])) / env)
end

display(fig)
