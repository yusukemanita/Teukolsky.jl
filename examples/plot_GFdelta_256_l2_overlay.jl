using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")

using Teukolsky
using HDF5
using Plots
using Plots.PlotMeasures
using LaTeXStrings
using Printf
using SpecialFunctions

# ============================================================
#  Overlay  ψ_tot = 2·{}_{-2}S_{22}(π/2) (ψ_PIA + ψ_C)
#  (branch-cut + leading-Kerr, a=0) on top of GFdelta_256_compact.h5
#  |Re ψ| at ℓ=2 (scri⁺).   x-axis: (t - 20)/M  ≡  u.
# ============================================================

const DATAFILE = joinpath(@__DIR__, "..", "data", "GFdelta_256_compact.h5")
const OUTPNG   = joinpath(@__DIR__, "GFdelta_256_l2_reabs_log_overlay.png")
const T_SHIFT  = 20.0

# ── branch-cut integrand parameters (same as psi_combined_rprime_C.jl) ──
const s, l, m = -2, 2, 2
const a       = 0.0
const M       = 1.0
const rplus   = M + sqrt(M^2 - a^2)
const rminus  = M - sqrt(M^2 - a^2)
const rprime  = 10.0
const q_spin  = a / M
const κ_spin  = sqrt(1 - q_spin^2)
const τ0      = -m * q_spin / κ_spin
const ε_plus  = -m * q_spin / (2 * κ_spin)        # ε_+ at ω = 0:  (τ + ε)/2
const S_FACTOR = 2 * sqrt(5) / (8 * sqrt(π))      # 2·{}_{-2}Y_{22}(π/2, 0)

kerr_delta(r, a_; M_=1.0) = r^2 - 2M_*r + a_^2
Δfactor(r) = (r - rplus) * (r - rminus)

function rstar(r; M_=M, rp=rplus, rm=rminus)
    denom = rp - rm
    t1 = 2M_*rp/denom * log((r - rp)/(2M_))
    t2 = rm == 0 ? 0.0 : 2M_*rm/denom * log((r - rm)/(2M_))
    return r + t1 - t2
end

# I_{ℓm}(r) = 2^{-ℓ-s-1} (-i)^{ℓ+s} (r-r_+)^{-s}/(r-r_-)^{ℓ+1}
#             · ((r-r_-)/(r-r_+))^{iε_+}
#             · Γ(1+ℓ-s)/Γ(1+ℓ+s)
#             · Σ_n α̃_n Γ(1+2ℓ+2n)/Γ(1+ℓ+n-s) (-iM/(r-r_-))^n
#
# α̃_0 = 1,
# α̃_n = i(n+ℓ-s)² [(n+ℓ)κ + i m q] / [n(n+ℓ)(n+2ℓ+1)(2n+2ℓ-1)] α̃_{n-1}.
function I_lm(r, rp, rm, s::Integer, l::Integer, ε_p, M_::Real,
              q::Real, κ::Real, m::Integer; nmax::Int = 200,
              tol::Float64 = 1e-15)
    pref = 2.0^(-l - s - 1) * (-im)^(l + s) *
           (r - rp)^(-s) / (r - rm)^(l + 1) *
           ((r - rm) / (r - rp))^(im * ε_p) *
           gamma(1 + l - s) / gamma(1 + l + s)
    z   = -im * M_ / (r - rm)
    T   = complex(gamma(1 + 2l) / gamma(1 + l - s))   # n=0 term:  α̃_0=1
    acc = T
    for n in 1:nmax
        # ratio T_n / T_{n-1} = α̃-recurrence × Γ-ratio × z, simplified:
        #   = 2 i (n+ℓ-s) [(n+ℓ)κ + i m q] / [n (n+2ℓ+1)] · z
        ratio = 2im * (n + l - s) * ((n + l) * κ + im * m * q) /
                (n * (n + 2l + 1)) * z
        T   *= ratio
        acc += T
        abs(T) < tol * abs(acc) && break
    end
    return pref * acc
end

# ψ_C(r') = I_{ℓm}(r') · Δ(r')^s  (ω=0 contour result; u-independent)
psi_C(rprime, rp, rm, s::Integer, l::Integer, ε_p, M_::Real,
      q::Real, κ::Real, m::Integer) =
    I_lm(rprime, rp, rm, s, l, ε_p, M_, q, κ, m) * Δfactor(rprime)^s

function logspaced_weights(σ_min, σ_max, Nσ)
    σ  = exp.(range(log(σ_min), log(σ_max); length=Nσ))
    Δσ = diff([0.0; (σ[1:end-1] .+ σ[2:end]) ./ 2; σ[end]])
    return σ, Δσ
end

function compute_integrand(σ_grid, a_val, rp; nmax::Int = 100)
    Nσ    = length(σ_grid)
    F     = Vector{ComplexF64}(undef, Nσ)
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

# ── compute ψ_tot(u) ────────────────────────────────────────
println("=== computing ψ_tot(u) for overlay ===")
Nσ = 300
σ_grid, Δσ = logspaced_weights(1e-3, 2.0, Nσ)
F = compute_integrand(σ_grid, a, rprime)

rstar_p = rstar(rprime)
println("r*(r'=$rprime) = ", round(rstar_p; digits=6))
u_grid = collect(range(-rstar_p, rstar_p; length=401))

ψ_C_const = psi_C(rprime, rplus, rminus, s, l, ε_plus, M, q_spin, κ_spin, m)
@printf("ψ_C(r'=%.3f) = %+.6e %+.6ei  (u-independent)\n",
        rprime, real(ψ_C_const), imag(ψ_C_const))

ψ_PIA = Vector{ComplexF64}(undef, length(u_grid))
ψ_Cv  = fill(ψ_C_const, length(u_grid))
for (k, u) in enumerate(u_grid)
    acc = zero(ComplexF64)
    @inbounds for i in 1:Nσ
        acc += F[i] * Δσ[i] * exp(σ_grid[i] * u)
    end
    ψ_PIA[k] = acc / (2π)
end
ψ_tot = S_FACTOR .* (ψ_PIA .+ ψ_Cv)

# ── build overlay plot ─────────────────────────────────────
gr()
h5open(DATAFILE, "r") do f
    groups = sort(collect(keys(f)))
    println("groups: ", groups)

    p = plot(xlabel = "(t - $T_SHIFT) / M   =   u / M",
             ylabel = "|Re ψ|   (ℓ=2, m=2, s=-2, a=0)",
             yscale = :log10, xlim = (-20, 20), ylim = (1e-5, :auto),
             title  = "GFdelta_256 scri⁺ + ψ_PIA+ψ_C overlay",
             size   = (900, 520), margin = 5mm,
             framestyle = :box, grid = true, legend = :topleft)

    for k in groups
        g   = f[k]
        t   = read(g["times"])
        lv  = read(g["l_vals"])
        j   = findfirst(==(2), lv)
        j === nothing && continue
        fre = read(g["lin_f_re"])[:, j]
        plot!(p, t .- T_SHIFT, abs.(fre) .+ 1e-300;
              label = "$k  scri⁺", lw = 1.6, color = :steelblue)
    end

    plot!(p, u_grid, abs.(real.(ψ_tot));
          label = L"|\mathrm{Re}[2\,_{-2}S_{22}(\pi/2)\,(\psi_{\rm PIA}+\psi_C)]|",
          lw = 2.0, color = :black)

    savefig(p, OUTPNG)
    println("saved ", OUTPNG)
end
