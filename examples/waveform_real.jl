using Pkg; Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")
using BHPtoolkit, Plots, LaTeXStrings, Printf

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
ω_max     = T(2.0)         # frequency cutoff [M⁻¹]

t_ini     = T(-100.0)      # start time [M]
t_max     = T(600.0)       # end time [M]
Nt        = 7000           # number of time points

# ── Frequency grid (half-integer shift to avoid ω=0) ─────────
Δω     = 2ω_max / N
ω_grid = T[(n - N÷2 + 0.5) * Δω for n in 0:N-1]

# ── Compute G(ω) for positive ω, mirror to negative ─────────
function compute_GF(s, l, m, a::T, ω_grid, r_src::T; nmax=100) where T
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
            # Physical G = Rin_raw / (2iω Binc_raw) = Rin_norm × Btrans / (2iω Binc_raw)
            # Julia's Binc is the raw amplitude; Rin() returns the transmission-normalized form.
            Rin(p, ν, amp.fn, r_src; nmax=nmax) * amp.Btrans / (2im * ω * amp.Binc)
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

println("Computing G(ω)  ($N points, r_src = $r_src M) ...")
GF = compute_GF(s, l, m, a, ω_grid, r_src)

# ── Time-domain waveform: ψ(u) = Δω/2π Σ G(ω_n) e^{-iω_n u} ─
t_grid = range(t_ini, t_max; length=Nt)
ψ      = Vector{ComplexF64}(undef, Nt)
prefac = Δω / (2π)

println("Computing ψ(u)  ($Nt points) ...")
for (k, t) in enumerate(t_grid)
    s_val = zero(ComplexF64)
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

# ── Save ─────────────────────────────────────────────────────
outdir = @__DIR__
savefig(fig_time, joinpath(outdir, "waveform_time.png"))
savefig(fig_freq, joinpath(outdir, "waveform_freq.png"))
println("Saved: waveform_time.png, waveform_freq.png")

p = plot(fig_time, fig_freq; layout=(2,1), size=(800, 700))
display(p)
