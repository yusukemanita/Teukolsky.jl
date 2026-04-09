using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using BaryRational
using FFTW
using Plots
using LaTeXStrings
using CSV, DataFrames

# ============================================================
#  Step 1: Compute time-domain waveform ψ(t)
# ============================================================
params = WaveformParams(
    s=-2, l=2, m=2, a=0.9,
    N=4096, ω_max=6.0,
    t_ini=0.0, t_max=600.0, Nt=4096,
    verbose=true,
)

t_grid, ψ, _, _ = compute_waveform(params)
t_arr = collect(t_grid)
Δt    = t_arr[2] - t_arr[1]
Nt    = length(t_arr)

# ============================================================
#  Step 2: FFT on strip Im(ω) = +σ  (σ > 0, above real axis)
#
#  G(ω + iσ) = ∫₀^∞ ψ(t) e^{+iωt} e^{-σt} dt
#            ≈ Δt · Σ_k ψ(t_k) e^{-σ t_k} e^{+iω_k t_k}
#
#  σ > 0: e^{-σt} is a DECAYING envelope → FFT converges cleanly.
#  σ < 0 would make it e^{|σ|t} → integral diverges (ψ decays
#  as e^{-γt} with γ ≈ 0.07, which is slower than any e^{|σ|t}).
#
#  So we shift the contour UPWARD (σ > 0), not downward.
#  AAA then finds the QNM poles in the lower half plane by
#  rational analytic continuation from the upper strip data.
#
#  Sign convention:
#    G(ω) = ∫ ψ(t) e^{+iωt} dt
#    FFTW computes Σ x[k] e^{-2πi·nk/N}  (negative exponent)
#    → use conj(fft(conj(x))) to get positive exponent.
# ============================================================

# QNM reference
function load_qnm_data(l, m, n, a_target)
    fp = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$n.dat"
    df = CSV.read(fp, DataFrame, header=false)
    idx = argmin(abs.(df[!, 1] .- a_target))
    return (df[idx, 2] + im * df[idx, 3]) / 2
end
ω22   = [load_qnm_data(params.l,  params.m, n, params.a) for n in 0:7]
ω22R  = [-conj(load_qnm_data(params.l, -params.m, n, params.a)) for n in 0:7]
ω_known = [ω22; ω22R]

# Frequency axis in centered order
ω_raw = fftshift(fftfreq(Nt, 1/Δt) .* 2π)   # rad/M

# Keep physical window (avoid aliasing edges)
ω_cut = params.ω_max * 0.85
mask  = abs.(ω_raw) .< ω_cut
ω_phys = ω_raw[mask]

# Strips: σ > 0 shifts contour ABOVE real axis
# σ = 0: raw real-axis FFT (most noise, poles hardest to find)
# σ → γ₀ ≈ 0.07: signal decays faster, cleaner
# σ > γ₀: higher overtones start to be suppressed
strip_sigmas = [0.0, 0.05, 0.10, 0.20]

panels = map(strip_sigmas) do σ
    # Multiply by decaying envelope e^{-σt}; σ > 0 ensures convergence
    ψ_env = ψ .* exp.(-σ .* t_arr)

    # FFT with +iωt sign: conj(fft(conj(x))) · Δt
    G_centered = conj.(fftshift(fft(conj.(ψ_env)))) .* Δt

    G_aaa = G_centered[mask]

    # AAA on the strip Im(ω) = +σ
    ω_strip = ComplexF64.(ω_phys) .+ σ*im
    approx  = aaa(ω_strip, G_aaa; tol=1e-10, mmax=150)
    poles, res, zeros_r = prz(approx)

    println("σ=+$σ  support=$(length(approx.x))  poles=$(length(poles))  zeros=$(length(zeros_r))")

    res_log = log10.(clamp.(abs.(res), 1e-30, Inf))

    pan = scatter(real.(ω_known), imag.(ω_known),
        label="QNM ref", marker=:star5, ms=7, color=:blue, alpha=0.8,
        xlabel=L"\Re(\omega)", ylabel=L"\Im(\omega)",
        title=latexstring("\\sigma = +$(σ)\\ (\\mathrm{strip\\ above\\ real\\ axis})"),
        xlim=(-2.0, 2.0), ylim=(-1.5, 0.5),
        framestyle=:box, grid=true, legend=:topright,
        fontfamily="Computer Modern", dpi=120)

    # Mark the strip Im(ω) = +σ
    hline!(pan, [σ], lw=1, ls=:dash, color=:gray, label="strip Im(ω)=+σ")
    # Mark the real axis for reference
    σ > 0 && hline!(pan, [0.0], lw=0.5, ls=:dot, color=:black, label="real axis")

    scatter!(pan, real.(poles), imag.(poles),
        label="AAA poles", marker=:circle, ms=5,
        marker_z=res_log, color=:plasma, colorbar=false)

    scatter!(pan, real.(zeros_r), imag.(zeros_r),
        label="AAA zeros", marker=:xcross, ms=4, color=:green, alpha=0.7)

    pan
end

fig = plot(panels..., layout=(2, 2), size=(1200, 900))
savefig(fig, "waveform_aaa_strips.png")
println("Saved waveform_aaa_strips.png")
fig
