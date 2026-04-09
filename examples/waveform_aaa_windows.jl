using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using BaryRational
using FFTW
using Plots
using LaTeXStrings
using CSV, DataFrames

# ============================================================
#  Step 1: Compute ψ(t)
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
σ     = 0.05   # fixed strip height (above real axis)

# ============================================================
#  QNM reference  (a=0.9, l=m=2, s=-2)
# ============================================================
function load_qnm_data(l, m, n, a_target)
    fp = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$n.dat"
    df = CSV.read(fp, DataFrame, header=false)
    idx = argmin(abs.(df[!, 1] .- a_target))
    return (df[idx, 2] + im * df[idx, 3]) / 2
end
ω_known_p = [load_qnm_data(params.l,  params.m, n, params.a) for n in 0:7]
ω_known_m = [-conj(load_qnm_data(params.l, -params.m, n, params.a)) for n in 0:7]
ω_known   = [ω_known_p; ω_known_m]

# QNM decay rates for reference lines
γ_qnm = -imag.(ω_known_p[1:4])
println("γ_n = $(round.(γ_qnm, sigdigits=3))")

# ============================================================
#  Step 2: FFT helper on windowed, strip-shifted signal
#
#  G(ω + iσ) ≈ Δt Σ_{t<T_w} ψ(t) e^{-σt} e^{+iωt}
#
#  Time window [0, T_w]:
#    - Long  T_w → fundamental dominates (overtones decayed)
#    - Short T_w → overtones are relatively stronger
#    - But too short → poor frequency resolution Δω ~ 2π/T_w
#
#  Window function: Tukey (flat-top + cosine taper) to reduce
#  spectral leakage from the hard truncation at T_w.
# ============================================================
function tukey_window(N; α=0.2)
    w = ones(N)
    n_taper = round(Int, α * N / 2)
    for i in 1:n_taper
        w[i]         = 0.5 * (1 - cos(π * (i-1) / n_taper))
        w[N - i + 1] = w[i]
    end
    return w
end

function windowed_fft_aaa(ψ, t_arr, Δt, σ, T_w; tol=1e-10, mmax=150, ω_cut_frac=0.85)
    mask_t   = t_arr .<= T_w
    ψ_win    = ψ[mask_t]
    t_win    = t_arr[mask_t]
    N_win    = length(ψ_win)

    win      = tukey_window(N_win; α=0.15)
    ψ_tapered = ψ_win .* win .* exp.(-σ .* t_win)

    # FFT with +iωt sign
    G_fft    = conj.(fftshift(fft(conj.(ψ_tapered)))) .* Δt

    # Frequency axis
    ω_fft    = fftshift(fftfreq(N_win, 1/Δt) .* 2π)
    ω_cut    = (2π / (2Δt)) * ω_cut_frac
    mask_ω   = abs.(ω_fft) .< ω_cut

    ω_strip  = ComplexF64.(ω_fft[mask_ω]) .+ σ*im
    G_aaa    = G_fft[mask_ω]

    approx   = aaa(ω_strip, G_aaa; tol=tol, mmax=mmax)
    poles, res, zeros_r = prz(approx)
    return ω_strip, G_aaa, approx, poles, res, zeros_r, N_win
end

# ============================================================
#  Step 3: Four panels with different time windows
#
#  Overtone-to-fundamental amplitude ratio at T_w:
#    n=1: exp(-(0.28-0.09)*T_w) → need T_w < 1/(0.19) ≈ 5 M for 1/e
#    n=2: exp(-(0.48-0.09)*T_w) → need T_w < 1/(0.39) ≈ 3 M for 1/e
#
#  So windows of 30, 60, 150, 600 M progressively reveal fewer overtones.
# ============================================================
T_windows = [30.0, 60.0, 150.0, 600.0]

panels = map(T_windows) do T_w
    ω_strip, G_aaa, approx, poles, res, zeros_r, N_win =
        windowed_fft_aaa(ψ, t_arr, Δt, σ, T_w)

    # Frequency resolution and overtone ratios
    Δω_res = 2π / (N_win * Δt)

    println("T_w=$T_w M  N_win=$N_win  Δω=$(round(Δω_res,sigdigits=3))  " *
            "support=$(length(approx.x))  poles=$(length(poles))")

    res_log = log10.(clamp.(abs.(res), 1e-30, Inf))

    pan = scatter(real.(ω_known), imag.(ω_known),
        label="QNM ref", marker=:star5, ms=7, color=:blue, alpha=0.9,
        xlabel=L"\Re(\omega)", ylabel=L"\Im(\omega)",
        title=latexstring("T_w = $(T_w)\\ M,\\ \\Delta\\omega = $(round(Δω_res,sigdigits=2))"),
        xlim=(-2.0, 2.0), ylim=(-1.5, 0.5),
        framestyle=:box, grid=true, legend=:topright,
        fontfamily="Computer Modern", dpi=120)

    # Strip line
    hline!(pan, [σ],   lw=1, ls=:dash,  color=:gray,  label="strip Im=+σ")
    hline!(pan, [0.0], lw=0.5, ls=:dot, color=:black, label="real axis")

    # QNM decay rates as horizontal reference lines
    for (n, γ) in enumerate(γ_qnm[2:end])
        hline!(pan, [-γ], lw=0.8, ls=:dashdot, color=:orange, alpha=0.5, label="")
    end

    scatter!(pan, real.(poles), imag.(poles),
        label="AAA poles", marker=:circle, ms=5,
        marker_z=res_log, color=:plasma, colorbar=false)

    scatter!(pan, real.(zeros_r), imag.(zeros_r),
        label="AAA zeros", marker=:xcross, ms=4, color=:green, alpha=0.7)

    pan
end

fig = plot(panels..., layout=(2, 2), size=(1200, 900))
savefig(fig, "waveform_aaa_windows.png")
println("Saved waveform_aaa_windows.png")
fig
