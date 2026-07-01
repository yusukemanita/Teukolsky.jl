using Pkg; Pkg.activate("/Users/yusuke/work/Teukolsky.jl")
using Teukolsky, Plots, LaTeXStrings, Printf

# ============================================================
#  Waveform via real-axis frequency integral
#
#    ψ(u) = ∫ dω/2π  G(ω) e^{-iωu}
#    G(ω) = Rin(r_src; ω) / (2iω Binc_norm)  where Binc_norm = Binc_raw / Btrans
#
#  u = retarded time, r_src = source location [M]
# ============================================================

# ── Parameters ───────────────────────────────────────────────
T = Float64                # precision: Float64 or BigFloat

s, l, m   = -2, 2, 2
a         = T(0.0)
r_src     = T(10.0)

N         = 1000           # number of frequency points
ω_max     = T(3.0)         # frequency cutoff [M⁻¹]

t_ini     = T(-100.0)      # start time [M]
t_max     = T(600.0)       # end time [M]
Nt        = 7000           # number of time points

# ── Frequency grid (half-integer shift to avoid ω=0) ─────────
Δω     = 2ω_max / N
ω_grid = T[(n - N÷2 + 0.5) * Δω for n in 0:N-1]

# ── Compute G(ω) for positive ω, mirror to negative ─────────
function compute_GF(s, l, m, a::T, ω_grid, r_src::T; nmax=200) where T
    N  = length(ω_grid)
    GF = Vector{Complex{T}}(undef, N)

    ν_prev = nothing
    for i in (N÷2 + 1):N
        ω   = ω_grid[i]
        amp = compute_amplitudes(s, l, m, a, ω; nmax=nmax)
        ν   = amp.ν
        if !isfinite(real(ν)) || !isfinite(imag(ν))
            GF[i] = i > N÷2 + 1 ? GF[i-1] : zero(Complex{T})
            continue
        end
        p   = MSTParams(s, l, m, a, ω)
        val = try
            # G = Rin / (2iω Binc) — Binc is normalized by Btrans (Wolfram convention)
            amp.Bref * Rup(p, ν, amp.fn, r_src; nmax=nmax) / (2im * ω * amp.Binc) +
            Rdown(p, ν, amp.fn, r_src; nmax=nmax) / (2im * ω)
        catch
            i > N÷2 + 1 ? GF[i-1] : zero(Complex{T})
        end
        GF[i]  = isfinite(real(val)) ? val : (i > N÷2 + 1 ? GF[i-1] : zero(Complex{T}))
        ν_prev = ν
        i % 200 == 0 && (print("."); flush(stdout))
    end
    println(" done")

    # G(-ω) = conj(G(ω))  (ψ is real-valued)
    for i in 1:(N÷2)
        GF[i] = conj(GF[N + 1 - i])
    end
    return GF
end

# function compute_Bref(s, l, m, a::T, ω_grid, r_src::T; nmax=100) where T
#     N  = length(ω_grid)
#     Brefs = Vector{Complex{T}}(undef, N)

#     ν_prev = nothing
#     for i in (N÷2 + 1):N
#         ω   = ω_grid[i]
#         amp = compute_amplitudes(s, l, m, a, ω; nmax=nmax)
#         ν   = amp.ν
#         if !isfinite(real(ν)) || !isfinite(imag(ν))
#             Brefs[i] = i > N÷2 + 1 ? Brefs[i-1] : zero(Complex{T})
#             continue
#         end
#         p   = MSTParams(s, l, m, a, ω)
#         val = try
#             amp.Bref
#         catch
#             i > N÷2 + 1 ? Brefs[i-1] : zero(Complex{T})
#         end
#         Brefs[i]  = isfinite(real(val)) ? val : (i > N÷2 + 1 ? Brefs[i-1] : zero(Complex{T}))
#         ν_prev = ν
#         i % 200 == 0 && (print("."); flush(stdout))
#     end
#     println(" done")

#     # G(-ω) = conj(G(ω))  (ψ is real-valued)
#     for i in 1:(N÷2)
#         Brefs[i] = conj(Brefs[N + 1 - i])
#     end
#     return Brefs
# end

# function compute_Rdown(s, l, m, a::T, ω::T, r_src::T; nmax=200) where T
#     amp = compute_amplitudes(s, l, m, a, ω; nmax=nmax)
#     p   = MSTParams(s, l, m, a, ω)
#     return Rdown(p, amp.ν, amp.fn, r_src; nmax=nmax)
# end

# function compute_Rup(s, l, m, a::T, ω::T, r_src::T; nmax=200) where T
#     amp = compute_amplitudes(s, l, m, a, ω; nmax=nmax)
#     p   = MSTParams(s, l, m, a, ω)
#     return Rup(p, amp.ν, amp.fn, r_src; nmax=nmax)
# end

println("Computing G(ω)  ($N points, r_src = $r_src M) ...")
GF = compute_GF(s, l, m, a, ω_grid, r_src)

# ── Time-domain waveform: ψ(u) = Δω/2π Σ G(ω_n) e^{-iω_n u} ─
t_grid = range(t_ini, t_max; length=Nt)
ψ      = Vector{Complex{T}}(undef, Nt)
prefac = Δω / (2π)

println("Computing ψ(u)  ($Nt points) ...")
for (k, t) in enumerate(t_grid)
    s_val = zero(Complex{T})
    @inbounds for n in 1:N
        s_val += GF[n] * exp(-im * ω_grid[n] * t)
    end
    ψ[k] = prefac * s_val
    k % 1000 == 0 && (print("."); flush(stdout))
end
println(" done")

# ── Tortoise coordinate r*(r_src) ────────────────────────────
rp_bh    = 1 + sqrt(1 - a^2)
rm_bh    = 1 - sqrt(1 - a^2)
rstar_src = r_src + (rp_bh/(rp_bh - rm_bh)) * log(abs(r_src - rp_bh)) -
                    (rm_bh/(rp_bh - rm_bh)) * log(abs(r_src - rm_bh))
@printf "r*(%.1f M) = %.3f M\n" r_src rstar_src

# ── Plot 1: time-domain waveform ─────────────────────────────
t_arr = collect(Float64.(t_grid))

fig_time = plot(
    t_arr, abs.(real.(ψ));
    xlabel     = L"u\ [M]",
    ylabel     = L"|\mathrm{Re}[\psi(u)]|",
    label      = latexstring("s=$(s),\\ l=$(l),\\ m=$(m),\\ a=$(a)"),
    yscale     = :log10,
    title      = "Time-domain waveform",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150,
    size       = (800, 400))
vline!(fig_time, [Float64(rstar_src)];
    label  = latexstring("u = r_*($(r_src)M) = $(round(rstar_src; digits=1))"),
    color  = :red, lw = 1.2, ls = :dash)

# ── Plot 2: frequency-domain spectrum ────────────────────────
ω_arr  = Float64.(ω_grid)
GF_abs = abs.(GF)

fig_freq = plot(
    ω_arr, GF_abs;
    xlabel     = L"\omega\ [M^{-1}]",
    ylabel     = L"|G(\omega)|",
    label      = latexstring("G(\\omega) = R_{\\rm in}(r'=$(r_src)) / (2i\\omega B^{\\rm inc})"),
    yscale     = :log10,
    xlim       = (-Float64(ω_max), Float64(ω_max)),
    ylim       = (1e-14, Inf),
    title      = "Frequency-domain spectrum",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150,
    size       = (800, 400))

# Brefs = compute_Bref(s, l, m, a, ω_grid, r_src)
# fig_Bref = plot(
#     ω_arr, abs.(Brefs);
#     xlabel     = L"\omega\ [M^{-1}]",
#     ylabel     = L"|B^{\rm ref}(\omega)|",
#     label      = latexstring("B^{\\rm ref}(r'=$(r_src))"),
#     yscale     = :log10,
#     xlim       = (-Float64(ω_max), Float64(ω_max)),
#     ylim       = (1e-14, Inf),
#     title      = "Reference amplitude B^{ref}",
#     framestyle = :box, grid = true,
#     fontfamily = "Computer Modern", dpi = 150,
#     size       = (800, 400))

# Rdown_arr = [compute_Rdown(s, l, m, a, ω, r_src) for ω in ω_grid]
# fig_Rdown = plot(
#     ω_arr, abs.(Rdown_arr);
#     xlabel     = L"\omega\ [M^{-1}]",
#     ylabel     = L"|R_{\rm down}(\omega)|",
#     label      = latexstring("R_{\\rm down}(r'=$(r_src))"),
#     yscale     = :log10,
#     xlim       = (-Float64(ω_max), Float64(ω_max)),
#     ylim       = (1e-14, Inf),
#     title      = "Outgoing amplitude R_down",
#     framestyle = :box, grid = true,
#     fontfamily = "Computer Modern", dpi = 150,
#     size       = (800, 400))

# nu_arr = [compute_nu(s, l, m, a, ω)[1] for ω in ω_grid]
# fig_nu = plot(
#     ω_arr, real.(nu_arr);
#     xlabel     = L"\omega\ [M^{-1}]",
#     ylabel     = L"\mathrm{Re}(\nu)",
#     label      = latexstring("\\nu(\\omega)"),
#     xlim       = (-Float64(ω_max), Float64(ω_max)),
#     title      = "Renormalized angular momentum ν",
#     framestyle = :box, grid = true,
#     fontfamily = "Computer Modern", dpi = 150,     
#     size       = (800, 400))
# plot!(fig_nu, ω_arr, imag.(nu_arr);
#     label      = latexstring("\\mathrm{Im}(\\nu)"),
#     xlim       = (-Float64(ω_max), Float64(ω_max)),
#     title      = "Renormalized angular momentum ν",
#     framestyle = :box, grid = true,
#     fontfamily = "Computer Modern", dpi = 150,     
#     size       = (800, 400))

# ── Save ─────────────────────────────────────────────────────
outdir = @__DIR__
savefig(fig_time, joinpath(outdir, "waveform_time.png"))
savefig(fig_freq, joinpath(outdir, "waveform_freq.png"))
println("Saved: waveform_time.png, waveform_freq.png")

p = plot(fig_time, fig_freq; layout=(2,1), size=(800, 700))
display(p)
