using HDF5, Plots, LaTeXStrings

# ── Load data ────────────────────────────────────────────────
h5file = joinpath(@__DIR__, "..", "scri_l2_m2.h5")
t, re_ψ, im_ψ = h5open(h5file, "r") do f
    read(f["times"]), read(f["mode_re"]), read(f["mode_im"])
end

abs_re = abs.(re_ψ)

# ── Plot 1: linear scale ────────────────────────────────────
fig1 = plot(
    t, abs_re;
    xlabel     = L"t\ [M]",
    ylabel     = L"|\mathrm{Re}[\psi]|",
    label      = L"l=2,\ m=2",
    title      = L"|\mathrm{Re}[\psi_{22}]|\ \mathrm{at}\ \mathcal{I}^+",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150,
    size       = (800, 400),
)

# ── Plot 2: log scale (skip zeros) ──────────────────────────
mask = abs_re .> 0
fig2 = plot(
    t[mask], abs_re[mask];
    xlabel     = L"t\ [M]",
    ylabel     = L"|\mathrm{Re}[\psi]|",
    label      = L"l=2,\ m=2",
    yscale     = :log10,
    title      = L"|\mathrm{Re}[\psi_{22}]|\ \mathrm{at}\ \mathcal{I}^+\ (\log)",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150,
    size       = (800, 400),
    ylim       = (1e-14, Inf),
)

# ── Combined ─────────────────────────────────────────────────
p = plot(fig1, fig2; layout=(2,1), size=(800, 700))

outdir = @__DIR__
savefig(p, joinpath(outdir, "scri_l2_m2_re.png"))
println("Saved: scri_l2_m2_re.png")
display(p)
