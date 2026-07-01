#!/usr/bin/env julia
# ============================================================
#  Compare G(ω) = Rin(r=10) / (2iω Binc)
#  Julia (Teukolsky) vs Wolfram Teukolsky package
#
#  Parameters: s=-2, l=2, m=2, a=0
#  ω grid: 0.01k + 0.001i,  k=1..200
#  Wolfram CSV: scripts/GF_wolfram.csv  (run scripts/generate_GF_comparison.wls first)
# ============================================================
using Pkg; Pkg.activate("/Users/yusuke/work/Teukolsky.jl")
using Teukolsky, Plots, LaTeXStrings, Printf, DelimitedFiles, Statistics

# ── Parameters ───────────────────────────────────────────────
s, l, m = -2, 2, 2
a       = 0.0
r_src   = 10.0
nmax    = 80
nmax_cf = 150
Im_ω    = 1e-3   # imaginary part shift

# ── ω grid (same as Wolfram script) ─────────────────────────
ω_re = [0.01k for k in 1:200]
ωs   = complex.(ω_re, Im_ω)

# ── Compute G(ω) via Julia ───────────────────────────────────
println("Computing G(ω) via Julia  (n=$(length(ωs))) ...")
G_julia = Vector{ComplexF64}(undef, length(ωs))

for (i, ω) in enumerate(ωs)
    amp = compute_amplitudes(s, l, m, a, ω; nmax=nmax, nmax_cf=nmax_cf, method="Monodromy")
    ν   = amp.ν
    p   = MSTParams(s, l, m, a, ω)
    fn  = compute_fn(p, ν; nmax=nmax)
    # Binc is normalized by Btrans (matches Wolfram convention)
    # G = Rin / (2iω Binc) directly
    G_julia[i] = Rin(p, ν, fn, r_src; nmax=nmax) / (2im * ω * amp.Binc)
    i % 40 == 0 && (print("."); flush(stdout))
end
println(" done")

# ── Load Wolfram reference CSV ───────────────────────────────
wolfram_csv = joinpath(@__DIR__, "..", "scripts", "GF_wolfram.csv")
G_wolfram   = nothing
ω_wolfram   = nothing

if isfile(wolfram_csv)
    data      = readdlm(wolfram_csv, ','; skipstart=1)
    ω_wolfram = complex.(Float64.(data[:, 1]), Float64.(data[:, 2]))
    G_wolfram = complex.(Float64.(data[:, 3]), Float64.(data[:, 4]))
    println("Loaded Wolfram data: $(size(data,1)) points from $wolfram_csv")
else
    println("WARNING: Wolfram CSV not found at $wolfram_csv")
    println("  Run: wolframscript scripts/generate_GF_comparison.wls")
end

# ── Plot ─────────────────────────────────────────────────────
fig = plot(; xlabel=L"\mathrm{Re}(\omega)\ [M^{-1}]",
             ylabel=L"|G(\omega)|",
             yscale=:log10,
             title=L"G(\omega) = R_{\rm in}(r'=10)/(2i\omega B^{\rm inc}),\ s{=}{-2},\ l{=}m{=}2,\ a{=}0",
             framestyle=:box, grid=true,
             fontfamily="Computer Modern", dpi=150, size=(900, 450))

plot!(fig, ω_re, abs.(G_julia);
      label="Julia (Monodromy)", lw=2, color=:blue)

if G_wolfram !== nothing
    plot!(fig, real.(ω_wolfram), abs.(G_wolfram);
          label="Wolfram", lw=2, ls=:dash, color=:red)
end

savefig(fig, joinpath(@__DIR__, "compare_GF_abs.png"))
println("Saved: compare_GF_abs.png")
display(fig)

# ── Relative error (if Wolfram data available) ───────────────
if G_wolfram !== nothing
    # interpolate to same grid (they should be the same)
    fig2 = plot(; xlabel=L"\mathrm{Re}(\omega)\ [M^{-1}]",
                  ylabel=L"|G_{\rm Julia}/G_{\rm Wolfram} - 1|",
                  yscale=:log10,
                  title="Relative error",
                  framestyle=:box, grid=true,
                  fontfamily="Computer Modern", dpi=150, size=(900, 400))

    # assume same ω grid
    rel_err = abs.(G_julia ./ G_wolfram .- 1)
    plot!(fig2, ω_re, rel_err; label="", lw=1.5, color=:black)

    savefig(fig2, joinpath(@__DIR__, "compare_GF_error.png"))
    println("Saved: compare_GF_error.png")
    display(fig2)

    # Print summary statistics
    println("\nRelative error summary:")
    @printf "  median  = %.2e\n"  median(rel_err)
    @printf "  max     = %.2e  at ω = %.3f\n"  maximum(rel_err)  ω_re[argmax(rel_err)]
end

# ── Also plot ν and |Binc| to diagnose branch transitions ────
println("\nComputing ν and |Binc| along path ...")
ν_arr    = Vector{ComplexF64}(undef, length(ωs))
Binc_arr = Vector{ComplexF64}(undef, length(ωs))

for (i, ω) in enumerate(ωs)
    amp          = compute_amplitudes(s, l, m, a, ω; nmax=nmax, nmax_cf=nmax_cf, method="Monodromy")
    ν_arr[i]    = amp.ν
    Binc_arr[i] = amp.Binc   # already normalized by Btrans (Wolfram convention)
end

fig3 = plot(
    plot(ω_re, real.(ν_arr); xlabel=L"\mathrm{Re}(\omega)", ylabel=L"\mathrm{Re}(\nu)",
         label="", title="Re(ν) along path", lw=1.5, framestyle=:box),
    plot(ω_re, abs.(Binc_arr); xlabel=L"\mathrm{Re}(\omega)", ylabel=L"|B^{\rm inc}|",
         label="", title="|Binc| along path", yscale=:log10, lw=1.5, framestyle=:box);
    layout=(1,2), size=(1000, 400), dpi=150)

savefig(fig3, joinpath(@__DIR__, "compare_GF_nu_Binc.png"))
println("Saved: compare_GF_nu_Binc.png")
display(fig3)
