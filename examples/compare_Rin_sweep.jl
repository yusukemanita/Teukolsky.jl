using Pkg; Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")
using BHPtoolkit, Plots, LaTeXStrings, DelimitedFiles, Printf, Statistics

# ============================================================
#  Compare Rin(r=10M) vs omega: BHPtoolkit vs Mathematica Teukolsky package
#  s=-2, l=2, m=2, a=0.0
# ============================================================

s, l, m, a, r = -2, 2, 2, 0.0, 10.0

# ── Load Mathematica reference ────────────────────────────────────────────────
ref = readdlm(joinpath(@__DIR__, "Rin_sweep_reference.csv"), ','; skipstart=1)
ω_ref   = Float64.(ref[:, 1])
Rin_ref = complex.(Float64.(ref[:, 2]), Float64.(ref[:, 3]))

# ── Julia computation ────────────────────────────────────────────────────────
nω = length(ω_ref)

function compute_julia_Rin(s, l, m, a, r, ω_grid)
    n = length(ω_grid)
    Rin_out = Vector{ComplexF64}(undef, n)
    ν_out   = Vector{ComplexF64}(undef, n)
    ν_prev  = nothing
    for i in 1:n
        ω = ω_grid[i]
        ν, p = compute_nu(s, l, m, a, ω; ν_init=ν_prev)
        fn = compute_fn(p, ν)
        Rin_out[i] = Rin(p, ν, fn, r)
        ν_out[i]   = ν
        ν_prev = ν
        i % 20 == 0 && (print("."); flush(stdout))
    end
    println(" done")
    return Rin_out, ν_out
end

println("Computing Julia Rin sweep (n=$nω points)...")
Rin_julia, ν_vals = compute_julia_Rin(s, l, m, a, r, ω_ref)

# ── Relative error ────────────────────────────────────────────────────────────
rel_err = abs.(Rin_julia .- Rin_ref) ./ abs.(Rin_ref)
println("Max relative error: $(maximum(rel_err))")
println("Mean relative error: $(mean(rel_err))")

# ── Detect ν branch transitions ───────────────────────────────────────────────
branch = Vector{Symbol}(undef, nω)
for i in 1:nω
    ω = ω_ref[i]
    p = MSTParams(s, l, m, a, ω)
    rc = real(BHPtoolkit.monodromy_cos2pi_nu(s, l, m, a, ω, p.λ))
    branch[i] = rc < -1 ? :half : (rc > 1 ? :int : :real)
end

trans_real2half = Float64[]
trans_half2int  = Float64[]
for i in 2:nω
    if branch[i-1] == :real && branch[i] == :half
        push!(trans_real2half, (ω_ref[i-1] + ω_ref[i]) / 2)
    end
    if branch[i-1] == :half && branch[i] == :int
        push!(trans_half2int, (ω_ref[i-1] + ω_ref[i]) / 2)
    end
end

# ── Figure 1: |Rin| comparison ───────────────────────────────────────────────
fig1 = plot(layout=(3,1), size=(850, 900),
            left_margin=10Plots.mm, bottom_margin=5Plots.mm,
            fontfamily="Computer Modern", dpi=150)

# Upper: |Rin| both
plot!(fig1[1],
    ω_ref, abs.(Rin_ref),
    label=L"|R_\mathrm{in}|\ \mathrm{(Mathematica)}",
    color=:steelblue, lw=2.5,
    yscale=:log10,
    ylabel=L"|R_\mathrm{in}|",
    title=latexstring("R_\\mathrm{in}(r=10M),\\ s=$(s),\\,l=$(l),\\,m=$(m),\\,a=$(a)"),
    framestyle=:box, grid=true)
plot!(fig1[1],
    ω_ref, abs.(Rin_julia),
    label=L"|R_\mathrm{in}|\ \mathrm{(BHPtoolkit)}",
    color=:crimson, lw=1.5, ls=:dash)
for ω0 in trans_real2half
    vline!(fig1[1], [ω0], color=:gray, ls=:dot, lw=1, label="")
end
for ω0 in trans_half2int
    vline!(fig1[1], [ω0], color=:black, ls=:dot, lw=1, label="")
end

# Middle: Re(Rin) both
plot!(fig1[2],
    ω_ref, real.(Rin_ref),
    label=L"\mathrm{Re}(R_\mathrm{in})\ \mathrm{(Mathematica)}",
    color=:steelblue, lw=2.5,
    ylabel=L"\mathrm{Re}(R_\mathrm{in})",
    framestyle=:box, grid=true)
plot!(fig1[2],
    ω_ref, real.(Rin_julia),
    label=L"\mathrm{Re}(R_\mathrm{in})\ \mathrm{(BHPtoolkit)}",
    color=:crimson, lw=1.5, ls=:dash)
for ω0 in trans_real2half
    vline!(fig1[2], [ω0], color=:gray, ls=:dot, lw=1, label="")
end
for ω0 in trans_half2int
    vline!(fig1[2], [ω0], color=:black, ls=:dot, lw=1, label="")
end

# Lower: Im(Rin) both
plot!(fig1[3],
    ω_ref, imag.(Rin_ref),
    label=L"\mathrm{Im}(R_\mathrm{in})\ \mathrm{(Mathematica)}",
    color=:steelblue, lw=2.5,
    xlabel=L"\omega\ [M^{-1}]",
    ylabel=L"\mathrm{Im}(R_\mathrm{in})",
    framestyle=:box, grid=true)
plot!(fig1[3],
    ω_ref, imag.(Rin_julia),
    label=L"\mathrm{Im}(R_\mathrm{in})\ \mathrm{(BHPtoolkit)}",
    color=:crimson, lw=1.5, ls=:dash)
for ω0 in trans_real2half
    vline!(fig1[3], [ω0], color=:gray, ls=:dot, lw=1, label="real→½")
end
for ω0 in trans_half2int
    vline!(fig1[3], [ω0], color=:black, ls=:dot, lw=1, label="½→0")
end

outdir = @__DIR__
savefig(fig1, joinpath(outdir, "compare_Rin_sweep.png"))
println("Saved: compare_Rin_sweep.png")

# ── Figure 2: Relative error ─────────────────────────────────────────────────
fig2 = plot(size=(800, 350),
            left_margin=10Plots.mm, bottom_margin=6Plots.mm,
            fontfamily="Computer Modern", dpi=150)

plot!(fig2,
    ω_ref, rel_err,
    label=L"|R_\mathrm{in}^\mathrm{Julia} - R_\mathrm{in}^\mathrm{Math}| / |R_\mathrm{in}^\mathrm{Math}|",
    color=:darkorange, lw=1.5, yscale=:log10,
    xlabel=L"\omega\ [M^{-1}]",
    ylabel="Relative error",
    title=L"R_\mathrm{in}\ \mathrm{relative\ error}\ (r=10M,\ s=-2,\ l=2,\ m=2,\ a=0)",
    framestyle=:box, grid=true)
hline!(fig2, [1e-6], color=:red, ls=:dash, lw=1, label="tol = 1e-6")
for ω0 in trans_real2half
    vline!(fig2, [ω0], color=:gray, ls=:dot, lw=1, label="")
end
for ω0 in trans_half2int
    vline!(fig2, [ω0], color=:black, ls=:dot, lw=1, label="")
end

savefig(fig2, joinpath(outdir, "compare_Rin_error.png"))
println("Saved: compare_Rin_error.png")

display(fig1)
display(fig2)
