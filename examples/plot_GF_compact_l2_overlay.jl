using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using HDF5
using Plots
using LaTeXStrings
using Printf

# ============================================================
#  Overlay ψ_tot = ψ_PIA + ψ_C  (from psi_combined_rprime_C.jl)
#  on top of GF_compact_l2_reabs_log.png
# ============================================================

const DATAFILE = joinpath(@__DIR__, "..", "data", "GF_compact_modes.h5")
const OUTPNG   = joinpath(@__DIR__, "GF_compact_l2_reabs_log_overlay.png")
const T_SHIFT  = 20.0

# ──── branch-cut integrand parameters (same as psi_combined_rprime_C.jl) ────
const s, l, m = -2, 2, 2
const a       = 0.0
const M       = 1.0
const rplus   = M + sqrt(M^2 - a^2)
const rminus  = M - sqrt(M^2 - a^2)
const rprime  = 10.0
const q_spin  = a / M
const κ_spin  = sqrt(1 - q_spin^2)
const τ0      = -m * q_spin / κ_spin
# overall angular factor 2·{}_{-2}S_{22}(π/2) in a=0 limit (SWSH = SWSH_spherical)
const S_FACTOR = 2 * sqrt(5) / (8 * sqrt(π))

kerr_delta(r, a; M=1.0) = r^2 - 2M*r + a^2
Δfactor(r) = (r - rplus) * (r - rminus)

# Kerr tortoise coordinate; a=0 limit:  r* = r + 2M ln((r-2M)/(2M))
function rstar(r; M_=M, rp=rplus, rm=rminus)
    denom = rp - rm
    t1 = 2M_*rp/denom * log((r - rp)/(2M_))
    t2 = rm == 0 ? 0.0 : 2M_*rm/denom * log((r - rm)/(2M_))
    return r + t1 - t2
end

function Phi_rprime(rprime, rp, rm, s::Integer, l::Integer, τ0)
    base = (rprime - rm) / (rprime - rp)
    return (rprime - rp)^(-s) * (rprime - rm)^(-(l + 1)) *
           base^(im * τ0 / 2)
end
function I_leading(u, rprime, rp, rm, s::Integer, l::Integer, τ0)
    pref = (-1)^(l + s) * factorial(2l) /
           (2.0^(l + s + 1) * factorial(l - s) * factorial(l + s))
    return pref * Phi_rprime(rprime, rp, rm, s, l, τ0) *
           (rprime - rm + u)^(l + s)
end
psi_C(u, rprime, rp, rm, s::Integer, l::Integer, τ0) =
    -I_leading(u, rprime, rp, rm, s, l, τ0) * Δfactor(rprime)^s

function logspaced_weights(σ_min, σ_max, Nσ)
    σ  = exp.(range(log(σ_min), log(σ_max); length=Nσ))
    Δσ = diff([0.0; (σ[1:end-1] .+ σ[2:end]) ./ 2; σ[end]])
    return σ, Δσ
end

function compute_integrand(σ_grid, a_val, rp; nmax::Int = 100)
    Nσ  = length(σ_grid)
    F   = Vector{ComplexF64}(undef, Nσ)
    Δinv2 = 1.0 / kerr_delta(rp, a_val)^2
    for i in 1:Nσ
        σ = σ_grid[i]; ω = im * σ
        qt = compute_qtilde(s, l, m, Float64(a_val), ω; nmax=nmax)
        val = zero(ComplexF64)
        if isfinite(qt.ν) && isfinite(qt.qtilde)
            try
                fn = compute_fn(qt.p, qt.ν; nmax=nmax)
                Rv = Rup(qt.p, qt.ν, fn, rp; nmax=nmax)
                cand = Δinv2 * qt.qtilde * Rv / (2im * ω)
                isfinite(cand) && (val = cand)
            catch
            end
        end
        F[i] = val
        i % 25 == 0 && (print("."); flush(stdout))
    end
    println()
    return F
end

# ──── compute ψ_tot(u) on the overlay x-range ────────────────
println("=== computing ψ_tot(u) for overlay ===")
Nσ = 300
σ_grid, Δσ = logspaced_weights(1e-3, 2.0, Nσ)
F = compute_integrand(σ_grid, a, rprime)

# x-axis of GF plot is (t - 20)/M ≡ u here; use u ∈ [-r*', +r*']
rstar_p = rstar(rprime)
println("r*(r'=$rprime) = ", round(rstar_p; digits=6))
u_grid = collect(range(-rstar_p, rstar_p; length=401))
ψ_PIA = Vector{ComplexF64}(undef, length(u_grid))
ψ_Cv  = Vector{ComplexF64}(undef, length(u_grid))
for (k, u) in enumerate(u_grid)
    acc = zero(ComplexF64)
    @inbounds for i in 1:Nσ
        acc += F[i] * Δσ[i] * exp(σ_grid[i] * u)
    end
    ψ_PIA[k] = acc / (2π)
    ψ_Cv[k]  = psi_C(u, rprime, rplus, rminus, s, l, τ0)
end
ψ_tot = S_FACTOR .* (ψ_PIA .+ ψ_Cv)

# keep only where PIA truncation has not blown up (σ_max·u ≲ ~20)
const U_MAX_RELIABLE = 5.0
mask = u_grid .<= U_MAX_RELIABLE

# user-chosen display rescaling (matches psi_combined_rprime_C.jl line 168)
scale = 1e5 * (rprime^2 + 2 * rprime)
ψ_plot_y = abs.(real.(scale .* ψ_tot))

# ──── build overlay plot ─────────────────────────────────────
gr()
h5open(DATAFILE, "r") do f
    groups = sort(collect(keys(f)))
    println("groups in $(basename(DATAFILE)): ", groups)

    p = plot(xlabel = "(t - $T_SHIFT) / M",
             ylabel = "|Re ψ|   (ℓ=2, m=2, s=-2, a=0)",
             yscale = :log10, ylim = (1e-10, 1e5),
             title  = "GF_compact_modes + ψ_PIA+ψ_C overlay",
             size   = (900, 520),
             framestyle = :box, grid = true, legend = :topright)

    for k in groups
        g = f[k]
        t = read(g["times"])
        lv = read(g["l_vals"])
        j = findfirst(==(2), lv)
        j === nothing && continue
        fre = read(g["lin_f_re"])[:, j]
        plot!(p, t .- T_SHIFT, abs.(fre) .+ 1e-300;
              label = "$k  scri⁺", lw = 1.4, alpha = 0.7)
    end

    plot!(p, u_grid[mask], ψ_plot_y[mask];
          label = L"|\mathrm{Re}[2\,_{-2}S_{22}(\pi/2)\,(\psi_{\rm PIA}+\psi_C)]|\times 10^5(r'^2{+}2r')",
          lw = 2.2, color = :black, ls = :solid)

    savefig(p, OUTPNG)
    println("saved ", OUTPNG)
end
