using HDF5
using Plots
using Printf
using LinearAlgebra
using Statistics

const DATAFILE = joinpath(@__DIR__, "..", "data", "delta_compact_modes.h5")

"""
Load one (a, source) group from delta_compact_modes.h5.

Returns a NamedTuple with:
  a             : Kerr spin parameter
  m             : azimuthal number
  s             : spin weight
  t             : (Nt,) time array
  l             : (Nℓ,) array of ℓ values
  psi_scri      : (Nt, Nℓ) complex waveform at future null infinity (ψ₄ × r)
  psi_hor       : (Nt, Nℓ) complex waveform at the horizon
"""
function load_group(h5file::AbstractString, group::AbstractString)
    h5open(h5file, "r") do f
        g = f[group]
        m = read(HDF5.attributes(g)["mv"])
        s = read(HDF5.attributes(g)["spin_weight"])
        a = parse(Float64, match(r"a([0-9.]+)_", group).captures[1])
        t = read(g["times"])
        l = read(g["l_vals"])
        fre, fim = read(g["lin_f_re"]), read(g["lin_f_im"])
        pre, pim = read(g["lin_p_re"]), read(g["lin_p_im"])
        return (; a, m, s, t, l,
                  psi_scri = complex.(fre, fim),
                  psi_hor  = complex.(pre, pim))
    end
end

"List top-level groups in the file."
list_groups(h5file::AbstractString) = h5open(h5file, "r") do f; collect(keys(f)); end

"""
Fit a single-mode damped sinusoid ψ(t) ≈ A exp(-iω t) on the log-amplitude slope
over the window [t1, t2]. Returns (ω_R, ω_I) from log|ψ| and arg ψ.
"""
function fit_qnm(t, psi; t1=60.0, t2=120.0)
    mask = (t .>= t1) .& (t .<= t2)
    tt = t[mask]
    ab = log.(abs.(psi[mask]))
    ph = unwrap(angle.(psi[mask]))
    X  = hcat(ones(length(tt)), tt)
    c_ab = X \ ab    # ab ≈ c[1] + c[2]*t   → ω_I = -c[2]
    c_ph = X \ ph    # ph ≈ c[1] + c[2]*t   → ω_R =  c[2]
    return (ωR = c_ph[2], ωI = -c_ab[2])
end

function unwrap(ϕ::AbstractVector)
    out = similar(ϕ)
    idx = eachindex(ϕ)
    out[first(idx)] = ϕ[first(idx)]
    prev = first(idx)
    @inbounds for i in Iterators.drop(idx, 1)
        d = ϕ[i] - ϕ[prev]
        d -= 2π * round(d / (2π))
        out[i] = out[prev] + d
        prev = i
    end
    return out
end

function summarize(data)
    (; a, m, s, t, l, psi_scri, psi_hor) = data
    println("a = $a   m = $m   s = $s")
    @printf("times: [%.2f, %.2f]  dt = %.4f  Nt = %d\n",
            t[1], t[end], t[2]-t[1], length(t))
    println("ℓ values : ", l)
    println("\nℓ    |ψ_scri|_max   t*@max      |ψ_hor|_max     fit_ω (ℓ=2..): (ωR, -ωI)")
    for (j, ℓ) in enumerate(l)
        a_s = abs.(psi_scri[:, j]);   i_s = argmax(a_s)
        a_h = abs.(psi_hor[:,  j]);   i_h = argmax(a_h)
        fit = fit_qnm(t, psi_scri[:, j]; t1=80.0, t2=150.0)
        @printf("%2d    %.3e   %7.2f      %.3e      (%.4f, %.4f)\n",
                ℓ, a_s[i_s], t[i_s], a_h[i_h], fit.ωR, fit.ωI)
    end
end

function plot_overview(data; outdir=@__DIR__)
    (; a, m, s, t, l, psi_scri, psi_hor) = data
    # ---- scri+ log|ψ| vs t for each ℓ ---------------------------------
    p1 = plot(xlabel="t / M", ylabel="log₁₀|ψ_scri|",
              title="delta-source response — scri+, a=$a, m=$m, s=$s",
              legend=:outerright, size=(900, 520))
    for (j, ℓ) in enumerate(l)
        plot!(p1, t, log10.(abs.(psi_scri[:, j]) .+ 1e-300);
              label="ℓ=$ℓ")
    end
    # ---- horizon log|ψ| ----------------------------------------------
    p2 = plot(xlabel="t / M", ylabel="log₁₀|ψ_hor|",
              title="delta-source response — horizon, a=$a, m=$m, s=$s",
              legend=:outerright, size=(900, 520))
    for (j, ℓ) in enumerate(l)
        plot!(p2, t, log10.(abs.(psi_hor[:, j]) .+ 1e-300);
              label="ℓ=$ℓ")
    end
    # ---- Re ψ for ℓ=2 (typical QNM ringdown view) --------------------
    p3 = plot(xlabel="t / M", ylabel="Re ψ_scri",
              title="ℓ=2 scri+ waveform (re/im)", size=(900, 360))
    plot!(p3, t, real.(psi_scri[:, 1]); label="Re")
    plot!(p3, t, imag.(psi_scri[:, 1]); label="Im", ls=:dash)
    savefig(p1, joinpath(outdir, "delta_compact_scri_logabs.png"))
    savefig(p2, joinpath(outdir, "delta_compact_hor_logabs.png"))
    savefig(p3, joinpath(outdir, "delta_compact_l2_reim.png"))
    println("saved figures to $outdir")
    return p1, p2, p3
end

if abspath(PROGRAM_FILE) == @__FILE__
    groups = list_groups(DATAFILE)
    println("Groups in $(basename(DATAFILE)): ", groups)
    data = load_group(DATAFILE, first(groups))
    summarize(data)
    plot_overview(data)
end
