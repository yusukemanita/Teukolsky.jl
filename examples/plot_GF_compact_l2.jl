using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using HDF5
using Plots
using Printf

# ============================================================
#  Plot GF_compact_modes.h5 — only ℓ=2
#    x-axis:  (t - 20) / M         (shifted)
#    y-axis:  |Re ψ|  (log scale)
#    Overlays all resolution groups (GF_256 / 512 / 1024)
#    for both scri⁺ (lin_f_*) and horizon (lin_p_*).
# ============================================================

const DATAFILE = joinpath(@__DIR__, "..", "data", "GF_compact_modes.h5")
const OUTPNG   = joinpath(@__DIR__, "GF_compact_l2_reabs_log.png")
const T_SHIFT  = 20.0   # x-axis shift:  t → t - T_SHIFT

gr()

h5open(DATAFILE, "r") do f
    groups = sort(collect(keys(f)))
    println("groups in $(basename(DATAFILE)): ", groups)

    p = plot(xlabel = "(t - $T_SHIFT) / M",
             ylabel = "|Re ψ|   (ℓ=2, m=2, s=-2, a=0)",
             yscale = :log10, ylim = (1e-10, 1e5),
             title  = "GF_compact_modes — ℓ=2",
             size   = (900, 520),
             framestyle = :box, grid = true, legend = :topright)

    for k in groups
        g = f[k]
        t = read(g["times"])
        l = read(g["l_vals"])
        j = findfirst(==(2), l)
        j === nothing && continue

        fre = read(g["lin_f_re"])[:, j]
        pre = read(g["lin_p_re"])[:, j]

        plot!(p, t .- T_SHIFT, abs.(fre) .+ 1e-300;
              label = "$k  scri⁺", lw = 1.6)
    end

    savefig(p, OUTPNG)
    println("saved ", OUTPNG)
end
