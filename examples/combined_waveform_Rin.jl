using Pkg; Pkg.activate("/Users/yusuke/work/Teukolsky.jl")
using Teukolsky, Plots, LaTeXStrings, Printf

# ============================================================
#  Combined: real-axis waveform + branch cut integrals
#
#  Green function with source:
#    G(ω) = Rin(r_src; ω) × Bref / (2iω Binc)
#
#  Real-axis waveform:
#    ψ_real(u) = ∫ dω/2π  G(ω) e^{-iωu}
#
#  Branch cut (positive imaginary axis, t < 0):
#    ψ_BC+(t) = (-i/2π) ∫ dσ  ΔG_+(σ) e^{+σt}
#    ΔG_+(σ) = G(+δ+iσ) - G(-δ+iσ)
#
#  Branch cut (negative imaginary axis, t > 0):
#    ψ_BC-(t) = (+i/2π) ∫ dσ  ΔG_-(σ) e^{-σt}
#    ΔG_-(σ) = G(+δ-iσ) - G(-δ-iσ)
# ============================================================

# ── Precision for real-axis waveform (Part 1) ────────────────
# Set T = Float64 for speed, or T = BigFloat for extended precision.
# Parts 2 & 3 (branch cuts) always use Float64.
T    = Float64
PREC = 256       # bits; only used when T = BigFloat
T == BigFloat && setprecision(BigFloat, PREC)

s, l, m = -2, 2, 2
a       = 0.9
r_src   = 10.0

# ── Shared helper: G(ω) = Rin(r_src;ω) × Bref / (2iω Binc) ──
function compute_G(s, l, m, a, ω, r_src; nmax=60, ν_init=nothing)
    amp = compute_amplitudes(s, l, m, a, ω; nmax=nmax, ν_init=ν_init)
    p   = MSTParams(s, l, m, a, ω)
    return Rin(p, amp.ν, amp.fn, r_src; nmax=nmax) * amp.Bref / (2im * ω * amp.Binc), amp.ν
end

# ════════════════════════════════════════════════════════════
# Part 1: Real-axis waveform
# ════════════════════════════════════════════════════════════
println("=== Part 1: Real-axis waveform ===")

N      = 4000
ω_max  = T(2.0)
Δω     = 2ω_max / N
ω_grid = T[(n - N÷2 + 0.5) * Δω for n in 0:N-1]

function compute_GF_real_axis(s, l, m, a, ω_grid::Vector{T}, r_src; nmax=100) where T
    N  = length(ω_grid)
    GF = Vector{Complex{T}}(undef, N)
    ν_prev = nothing
    for i in (N÷2 + 1):N
        ω = ω_grid[i]
        GF[i], ν_new = compute_G(s, l, m, a, ω, r_src; nmax=nmax, ν_init=ν_prev)
        ν_prev = ν_new
        i % 400 == 0 && (print("."); flush(stdout))
    end
    for i in 1:(N÷2)
        GF[i] = conj(GF[N + 1 - i])
    end
    return GF
end

print("G(ω) ($N points, T=$T) ")
GF = compute_GF_real_axis(s, l, m, a, ω_grid, r_src)
println(" done")

t_ini  = T(-100.0)
t_max  = T(600.0)
Nt     = 7000
t_grid = range(t_ini, t_max; length=Nt)
ψ_real = Vector{Complex{T}}(undef, Nt)
prefac = Δω / T(2π)

print("ψ_real ($Nt points) ")
for (k, t) in enumerate(t_grid)
    s_val = zero(Complex{T})
    @inbounds for n in 1:N
        s_val += GF[n] * exp(-im * ω_grid[n] * t)
    end
    ψ_real[k] = prefac * s_val
    k % 2000 == 0 && (print("."); flush(stdout))
end
println(" done")

# ════════════════════════════════════════════════════════════
# Part 2: Branch cut — positive imaginary axis (t < 0)
#   ΔG_+(σ) = G(+δ+iσ) - G(-δ+iσ)
# ════════════════════════════════════════════════════════════
println("\n=== Part 2: Branch cut, positive imaginary axis (t < 0) ===")

Nσ_pos   = 300
σ_min_p  = 1e-3
σ_max_p  = 5.0
δ_pos    = 1e-6
σ_grid_p = exp.(range(log(σ_min_p), log(σ_max_p); length=Nσ_pos))
Δσ_p     = diff([0.0; (σ_grid_p[1:end-1] .+ σ_grid_p[2:end]) ./ 2; σ_max_p])

ΔG_pos = Vector{ComplexF64}(undef, Nσ_pos)
print("ΔG_+(σ) ($Nσ_pos points) ")
for i in 1:Nσ_pos
    σ     = σ_grid_p[i]
    ω_R   = δ_pos + im*σ
    amp_R = compute_amplitudes(s, l,  m, a, ω_R)
    pR = MSTParams(s, l, m, a, ω_R)
    G_R   = Rin(pR, amp_R.ν, amp_R.fn, r_src) * amp_R.Bref / (2im * ω_R * amp_R.Binc)

    ω_L   = -δ_pos + im*σ
    amp_m = compute_amplitudes(s, l, -m, a, δ_pos + im*σ)  # symmetry: G_L = conj(G(l,-m,ω_R))
    pL = MSTParams(s, l, -m, a, ω_L)
    G_L   = conj(Rin(pL, amp_m.ν, amp_m.fn, r_src) * amp_m.Bref / (2im * ω_L * amp_m.Binc))

    ΔG_pos[i] = G_R - G_L
    i % 40 == 0 && (print("."); flush(stdout))
end
println(" done")

t_neg    = range(-100.0, -0.5; length=500)
ψ_BC_pos = [-im/(2π) * sum(ΔG_pos[i] * Δσ_p[i] * exp(σ_grid_p[i] * t) for i in 1:Nσ_pos)
            for t in t_neg]

# ════════════════════════════════════════════════════════════
# Part 3: Branch cut — negative imaginary axis (t > 0)
# ════════════════════════════════════════════════════════════
println("\n=== Part 3: Branch cut, negative imaginary axis (t > 0) ===")

Nσ_neg   = 300
σ_min_n  = 1e-3
σ_max_n  = 2.0
δ_neg    = 1e-6
σ_grid_n = exp.(range(log(σ_min_n), log(σ_max_n); length=Nσ_neg))
Δσ_n     = diff([0.0; (σ_grid_n[1:end-1] .+ σ_grid_n[2:end]) ./ 2; σ_max_n])

ΔG_neg = Vector{ComplexF64}(undef, Nσ_neg)
print("ΔG_-(σ) ($Nσ_neg points) ")
for i in 1:Nσ_neg
    σ      = σ_grid_n[i]
    ω_R    = - im*σ
    amp_R  = compute_amplitudes(s, l, m, a, ω_R; nmax=100)
    q_info = compute_q(s, l, m, a, ω_R; nmax=100)
    q_val  = q_info.q
    # ΔG_-(σ) = iq Bref² / (2iω Binc (Binc + iq Bref))
    pR = MSTParams(s, l, m, a, ω_R)
    ΔG_neg[i] = Rin(pR, amp_R.ν, amp_R.fn, r_src) * im * q_val * amp_R.Bref^2 /
                (2im * ω_R * amp_R.Binc * (amp_R.Binc + im * q_val * amp_R.Bref))
    i % 40 == 0 && (print("."); flush(stdout))
end
println(" done")

t_pos    = range(1.0, 600.0; length=3000)
ψ_BC_neg = [im/(2π) * sum(ΔG_neg[i] * Δσ_n[i] * exp(-σ_grid_n[i] * t) for i in 1:Nσ_neg)
            for t in t_pos]

# ════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════
println("\nPlotting ...")

t_arr  = Float64.(collect(t_grid))
GF_f64 = ComplexF64.(GF)

# ── Plot 1: time-domain waveform + branch cuts ───────────────
fig = plot(
    xlabel     = L"u\ [M]",
    ylabel     = L"|\mathrm{Re}[\psi(u)]|",
    yscale     = :log10,
    title      = "Waveform + branch cut  (s=$s, l=$l, m=$m, a=$a, r'=$(r_src)M)",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150,
    legend     = :topright,
    size       = (900, 500),
    ylims      = (:auto, 2e0))

plot!(fig, t_arr, abs.(real.(ComplexF64.(ψ_real)));
    label  = L"\psi_{\rm real}\ (\omega\ \mathrm{integral})",
    lw = 2, color = :steelblue)

plot!(fig, collect(t_neg), abs.(real.(ψ_BC_pos));
    label  = L"\psi_{BC}^{+}\ (t<0,\ +\mathrm{Im}\,\omega\ \mathrm{axis})",
    lw = 1.5, color = :crimson, ls = :dash)

plot!(fig, collect(t_pos), abs.(real.(ψ_BC_neg));
    label  = L"\psi_{BC}^{-}\ (t>0,\ -\mathrm{Im}\,\omega\ \mathrm{axis})",
    lw = 1.5, color = :darkorange, ls = :dash)

vline!(fig, [0.0]; label = "", color = :black, lw = 0.8, ls = :dot)

# ── Plot 2: frequency-domain spectrum ────────────────────────
fig_freq = plot(
    Float64.(ω_grid), abs.(GF_f64);
    xlabel     = L"\omega\ [M^{-1}]",
    ylabel     = L"|G(\omega)|",
    label      = L"G(\omega) = R_{\rm in} B^{\rm ref}/(2i\omega B^{\rm inc})",
    yscale     = :log10,
    xlim       = (-ω_max, ω_max),
    ylim       = (1e-14, Inf),
    title      = "Frequency-domain spectrum",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150,
    size       = (800, 400))

outdir = @__DIR__
savefig(fig,      joinpath(outdir, "combined_waveform_Rin.png"))
savefig(fig_freq, joinpath(outdir, "combined_waveform_Rin_freq.png"))
println("Saved: combined_waveform_Rin.png, combined_waveform_Rin_freq.png")

display(fig)
# display(fig_freq)
