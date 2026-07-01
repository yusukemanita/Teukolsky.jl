using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")

using HDF5
using Plots
using Plots.PlotMeasures

const DATAFILE = joinpath(@__DIR__, "..", "data", "GFdelta_256_compact.h5")
const OUTPNG   = joinpath(@__DIR__, "GFdelta_256_l2_reabs_log.png")
const T_SHIFT  = 20.0

gr()

h5open(DATAFILE, "r") do f
    groups = sort(collect(keys(f)))
    println("groups: ", groups)

    p = plot(xlabel = "(t - $T_SHIFT) / M",
             ylabel = "|Re ψ|   (ℓ=2, m=2, s=-2, a=0)",
             yscale = :log10, xlim=(-20, 20), ylim = (1e-5, :auto),
             title  = "GFdelta_256_compact — ℓ=2",
             size   = (900, 520), margin = 5mm,
             framestyle = :box, grid = true, legend = :topleft)

    for k in groups
        g   = f[k]
        t   = read(g["times"])
        lv  = read(g["l_vals"])
        j   = findfirst(==(2), lv)
        j === nothing && continue
        fre = read(g["lin_f_re"])[:, j]
        pre = read(g["lin_p_re"])[:, j]
        plot!(p, t .- T_SHIFT, abs.(fre) .+ 1e-300;
              label = "$k  scri⁺", lw = 1.6, color = :steelblue)
    end

    savefig(p, OUTPNG)
    println("saved ", OUTPNG)
end
