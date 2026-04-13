using Pkg; Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")
using BHPtoolkit, Plots, LaTeXStrings, Printf

# ============================================================
#  Plot Rin_raw vs Rin_phys for a=0.0, r=10M
#  s=-2, l=2, m=2
# ============================================================

s, l, m = -2, 2, 2
a    = 0.0
r    = 10.0
nω   = 200
ω_grid = range(0.05, 1.0; length=nω)

Rin_raw_vals  = Vector{ComplexF64}(undef, nω)
Rin_phys_vals = Vector{ComplexF64}(undef, nω)
branch_vals   = Vector{Symbol}(undef, nω)
ν_vals        = Vector{ComplexF64}(undef, nω)

function compute_Rin_sweep(s, l, m, a, r, ω_grid, nω)
    Rin_raw_vals  = Vector{ComplexF64}(undef, nω)
    Rin_phys_vals = Vector{ComplexF64}(undef, nω)
    branch_vals   = Vector{Symbol}(undef, nω)
    ν_vals        = Vector{ComplexF64}(undef, nω)
    ν_prev = nothing
    for i in 1:nω
    ω = ω_grid[i]
    p = MSTParams(s, l, m, a, ω)
    ν, _ = compute_nu(s, l, m, a, ω; ν_init=ν_prev)
    fn = compute_fn(p, ν; nmax=40)

    rc = real(BHPtoolkit.monodromy_cos2pi_nu(s, l, m, a, ω, p.λ))
    branch_vals[i] = rc < -1 ? :half : (rc > 1 ? :int : :real)

    Rin_raw_vals[i]  = Rin(p, ν, fn, r)
    Rin_phys_vals[i] = Rin_phys(p, ν, fn, r)
    ν_vals[i]        = ν
        ν_prev = ν
        i % 40 == 0 && print("."); flush(stdout)
    end
    println(" done")
    return Rin_raw_vals, Rin_phys_vals, branch_vals, ν_vals
end

Rin_raw_vals, Rin_phys_vals, branch_vals, ν_vals =
    compute_Rin_sweep(s, l, m, a, r, ω_grid, nω)

ωr = collect(ω_grid)

# branch transition ωの特定
real2half_ω = nothing; half2int_ω = nothing
for i in 2:nω
    if branch_vals[i-1] == :real && branch_vals[i] == :half
        real2half_ω = (ωr[i-1] + ωr[i]) / 2
    end
    if branch_vals[i-1] == :half && branch_vals[i] == :int
        half2int_ω = (ωr[i-1] + ωr[i]) / 2
    end
end

# ── Figure 1: |Rin_raw| vs |Rin_phys| 比較 ──────────────────────────────
fig = plot(layout=(2,1), size=(800, 700),
           left_margin=8Plots.mm, bottom_margin=6Plots.mm,
           fontfamily="Computer Modern", dpi=150)

# 上段: 両方
plot!(fig[1],
    ωr, abs.(Rin_raw_vals),
    label     = L"|R_\mathrm{in}|\ \mathrm{(raw)}",
    color     = :steelblue, lw = 1.5, yscale = :log10,
    xlabel    = "", ylabel = L"|R_\mathrm{in}|",
    title     = latexstring("\\mathrm{Ingoing\\ Rin}\\ (s=$(s),\\,l=$(l),\\,m=$(m),\\,a=$(a),\\,r=$(r)M)"),
    framestyle = :box, grid = true)

plot!(fig[1],
    ωr, abs.(Rin_phys_vals),
    label  = L"|R_\mathrm{in}^\mathrm{phys}| = |R_\mathrm{in}|/B_\mathrm{trans}",
    color  = :crimson, lw = 1.5, ls = :dash)

# branch transition vlines
if !isnothing(real2half_ω)
    vline!(fig[1], [real2half_ω], color=:gray, ls=:dot, lw=1.2,
           label=L"\mathrm{real}\to\tfrac{1}{2}")
end
if !isnothing(half2int_ω)
    vline!(fig[1], [half2int_ω], color=:black, ls=:dot, lw=1.2,
           label=L"\tfrac{1}{2}\to 0")
end

# 下段: Rin_phys のみ (Teukolsky と同じ量)
plot!(fig[2],
    ωr, abs.(Rin_phys_vals),
    label     = L"|R_\mathrm{in}^\mathrm{phys}|\ \mathrm{(Teukolsky\ convention)}",
    color     = :crimson, lw = 1.8, yscale = :log10,
    xlabel    = L"\omega\ [M^{-1}]",
    ylabel    = L"|R_\mathrm{in}^\mathrm{phys}|",
    title     = L"R_\mathrm{in}^\mathrm{phys} = R_\mathrm{in}/B_\mathrm{trans}\ \ (\mathrm{smooth\ across\ branches})",
    framestyle = :box, grid = true)

if !isnothing(real2half_ω)
    vline!(fig[2], [real2half_ω], color=:gray, ls=:dot, lw=1.2, label="")
end
if !isnothing(half2int_ω)
    vline!(fig[2], [half2int_ω], color=:black, ls=:dot, lw=1.2, label="")
end

outdir = @__DIR__
savefig(fig, joinpath(outdir, "Rin_phys_comparison.png"))
println("Saved: $(joinpath(outdir, "Rin_phys_comparison.png"))")

# ── Figure 2: Re(ν) と Im(ν) ──────────────────────────────────────────────
fig_nu = plot(layout=(2,1), size=(800, 500),
              left_margin=8Plots.mm, bottom_margin=6Plots.mm,
              fontfamily="Computer Modern", dpi=150)

plot!(fig_nu[1], ωr, real.(ν_vals),
    label=L"\mathrm{Re}(\nu)", color=:steelblue, lw=1.5,
    xlabel="", ylabel=L"\mathrm{Re}(\nu)",
    title=latexstring("\\nu(\\omega)\\ (s=$(s),\\,l=$(l),\\,m=$(m),\\,a=$(a))"),
    framestyle=:box, grid=true)
if !isnothing(real2half_ω)
    vline!(fig_nu[1], [real2half_ω], color=:gray, ls=:dot, lw=1.2, label="")
end
if !isnothing(half2int_ω)
    vline!(fig_nu[1], [half2int_ω], color=:black, ls=:dot, lw=1.2, label="")
end

plot!(fig_nu[2], ωr, imag.(ν_vals),
    label=L"\mathrm{Im}(\nu)", color=:darkorange, lw=1.5,
    xlabel=L"\omega\ [M^{-1}]", ylabel=L"\mathrm{Im}(\nu)",
    framestyle=:box, grid=true)
if !isnothing(real2half_ω)
    vline!(fig_nu[2], [real2half_ω], color=:gray, ls=:dot, lw=1.2, label="real→½")
end
if !isnothing(half2int_ω)
    vline!(fig_nu[2], [half2int_ω], color=:black, ls=:dot, lw=1.2, label="½→0")
end

savefig(fig_nu, joinpath(outdir, "Rin_phys_nu.png"))
println("Saved: $(joinpath(outdir, "Rin_phys_nu.png"))")

display(fig)
display(fig_nu)
