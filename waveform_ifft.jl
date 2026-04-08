using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using FFTW
using Plots
using LaTeXStrings

# ============================================================
#  Parameters
# ============================================================
s, l, m, a = -2, 2, 2, 0.9

# Frequency grid: N points, resolution Δω = ω_max / (N/2)
# Time resolution: Δt = 2π/ω_max
# Total time:      T   = 2π/Δω = N * Δt
N      = 4096          # number of frequency points (power of 2 for FFT)
ω_max  = 6.0           # maximum frequency (captures high-frequency content)
σ      = 0.1           # strip offset: evaluate on Im(ω) = -σ for stability
                        # compensate by multiplying waveform by exp(σ t)

Δω = 2ω_max / N
ω_grid = range(-ω_max + Δω, ω_max; length=N)   # symmetric, avoids ω=0

# ============================================================
#  Green's function G(ω) = Bref / (2iω Binc)
#  Symmetry for real waveform: G(-ω-iσ) = conj(G(ω-iσ))  [time-reversal]
#  We evaluate on the strip Im(ω) = -σ and later multiply by exp(σt).
# ============================================================
function Gapp(ω)
    if real(ω) > 0
        amp = compute_amplitudes(s, l, m, a, ω)
        return amp.Bref / (2im * ω * amp.Binc)
    else
        # Symmetry: G(ω) = conj(G(-conj(ω)))  for the retarded GF
        amp = compute_amplitudes(s, l, -m, a, -conj(ω))
        return conj(amp.Bref) / (2im * ω * conj(amp.Binc))
    end
end

println("Evaluating G(ω) on $N points, ω ∈ [-$ω_max, $ω_max], strip σ=$σ ...")
flush(stdout)

omega_strip = ComplexF64.(ω_grid) .- σ*im
GF = Vector{ComplexF64}(undef, N)
for (i, ω) in enumerate(omega_strip)
    GF[i] = Gapp(ω)
    i % 200 == 0 && (print("."); flush(stdout))
end
println("\nDone.")

# ============================================================
#  Inverse FFT → time domain
#
#  Convention:  G(t) = (Δω/2π) Σ_n G(ω_n) exp(-i ω_n t)
#               ↔ ifft with ω_n arranged in FFTW order and
#                 an overall phase factor from the grid offset.
#
#  Steps:
#   1. IFFT of G(ω) on FFTW-ordered grid gives ψ̃(t_k)
#   2. ψ(t_k) = ψ̃(t_k) * exp(σ t_k)  ← strip decontamination
#   3. Time axis: t_k = 2π k / ω_max / 2,  k = 0,…,N-1
# ============================================================

# Rearrange to FFTW order: [0, Δω, 2Δω, ..., ω_max-Δω, -ω_max, ..., -Δω]
# Our ω_grid is already centered; shift with fftshift
GF_fftorder = ifftshift(GF)

# IFFT: ψ̃(t) = N * Δω/(2π) * ifft(G)[k]  (unnormalized IFFT sums over n)
psi_raw = ifft(GF_fftorder)
norm_factor = N * Δω / (2π)
psi_raw .*= norm_factor

# Time grid corresponding to FFTW output
Δt = 2π / (2ω_max)        # = 2π / (N * Δω) * ... wait, standard: Δt = 1/(N*Δω/(2π))
# More carefully: sample spacing in t is Δt = 2π/(N*Δω)
Δt_exact = 2π / (N * Δω)  # = 2π / (2 ω_max) ← same
T_total   = N * Δt_exact
t_grid    = range(0.0, T_total - Δt_exact; length=N)

# Strip decontamination: multiply by exp(+σ t)
psi_t = psi_raw .* exp.(σ .* t_grid)

# The physical waveform is real; imaginary part should be small
# (it's nonzero due to our finite-grid / one-sided-source approximation)
ψ_real = real.(psi_t)
ψ_imag = imag.(psi_t)

println("T_total = $(round(T_total, digits=1)) M")
println("Δt      = $(round(Δt_exact, sigdigits=4)) M")
println("max|Im/Re| = $(round(maximum(abs.(ψ_imag)) / maximum(abs.(ψ_real)), sigdigits=3))")

# ============================================================
#  Plot
# ============================================================

# Focus on the ringdown window: t ∈ [0, 200 M]
t_plot_max = 200.0
idx_plot = t_grid .< t_plot_max

p1 = plot(t_grid[idx_plot], ψ_real[idx_plot],
    xlabel=L"t \ [M]", ylabel=L"\psi_4(t)",
    label=L"\Re[\psi_4]", lw=1.2,
    title="Time-domain waveform  (s=$s, l=$l, m=$m, a=$a)",
    framestyle=:box, grid=true, fontfamily="Computer Modern", dpi=120)

p2 = plot(t_grid[idx_plot], log10.(clamp.(abs.(ψ_real[idx_plot]), 1e-20, Inf)),
    xlabel=L"t \ [M]", ylabel=L"\log_{10}|\psi_4|",
    label=L"|\Re[\psi_4]|", lw=1.2, color=:red,
    framestyle=:box, grid=true, fontfamily="Computer Modern", dpi=120)

# Overlay QNM ringdown frequencies as vertical decay rate lines
try
    using CSV, DataFrames
    function load_qnm(l, m, n, a_target)
        fp = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$n.dat"
        df = CSV.read(fp, DataFrame, header=false)
        idx = argmin(abs.(df[!, 1] .- a_target))
        return (df[idx, 2] + im * df[idx, 3]) / 2
    end
    ω_qnm = [load_qnm(l, m, n, a) for n in 0:2]
    for (n, ωq) in enumerate(ω_qnm)
        γ = -imag(ωq)          # decay rate
        ω_r = real(ωq)
        t_ref = 20.0           # reference time for envelope
        t_range = t_grid[idx_plot]
        envelope = log10.(exp.(-γ .* (t_range .- t_ref))) .+ log10(abs(ψ_real[findfirst(t_grid .> t_ref)]) + 1e-20)
        plot!(p2, t_range, envelope, ls=:dash, lw=1, alpha=0.6,
              label="QNM n=$(n-1): γ=$(round(γ, sigdigits=3))")
    end
catch e
    @warn "Could not overlay QNM lines: $e"
end

plot(p1, p2, layout=(2,1), size=(900, 650))
savefig("waveform_a$(a).pdf")
println("Saved waveform_a$(a).pdf")
