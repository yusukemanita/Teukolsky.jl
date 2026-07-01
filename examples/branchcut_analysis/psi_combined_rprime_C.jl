using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")

using Teukolsky
using Plots
using LaTeXStrings
using Printf

# ============================================================
#  ψ_total(u; r') = ψ_PIA(u; r') + ψ_C(u; r')
#
#   • ψ_PIA(u; r') : branch-cut integral on the positive
#                    imaginary ω-axis (same as psi_rprime_integral.jl)
#
#            ψ_PIA(u; r') = (1/2π) ∫dσ  Δ(r')^{-2} q̃(iσ) R^up(r', iσ)
#                                       ─────────────────────────── e^{σu}
#                                                   2 i ω
#
#   • G_C (leading-Kerr) — ω = 0 analytic piece:
#
#          G_C = Δ(r')^s · {}_sY_{lm}(θ) · {}_sY_{lm}(θ')
#                × (-1)^{l+s}(2l)! / [2^{l+s+1}(l-s)!(l+s)!]
#                × (r'-r_+)^{-s} (r'-r_- + u - t')^{l+s} / (r'-r_-)^{l+1}
#                × ((r'-r_-)/(r'-r_+))^{iτ₀/2}
#
#   Both evaluated on u ∈ [-10, 10] and summed.
# ============================================================

const OUTDIR = @__DIR__
const s, l, m = -2, 2, 2
const a       = 0.0
const M       = 1.0
const rplus   = M + sqrt(M^2 - a^2)
const rminus  = M - sqrt(M^2 - a^2)
const rprime  = 10.0

# zero-frequency horizon parameter:  τ₀ = -m q / κ,   q = a/M,  κ = √(1-q²)
const q_spin = a / M
const κ_spin = sqrt(1 - q_spin^2)
const τ0     = -m * q_spin / κ_spin

# Angular factor:  {}_sY_{lm}(θ, 0) · {}_sY_{lm}(θ', 0)  with θ = θ' = π/2.
# At a=0 the spin-weighted spheroidal reduces to the spin-weighted spherical
# harmonic:
#   {}_{-2}Y_{22}(π/2, 0) = √(5/(4π)) · cos⁴(π/4) = √5 / (8√π).
# Two factors (observer and source angles) → [√5/(8√π)]² = 5/(64π).
# (For a ≠ 0 this would need the SWSH eigenvector at c = aω.)
const θ_obs = π/2
const θ_src = π/2
sYlm_pi2(θ) = sqrt(5 / (4π)) * cos(θ/2)^4          # {}_{-2}Y_{22}(θ, 0)
const Y_FACTOR = sYlm_pi2(θ_obs) * sYlm_pi2(θ_src)  # ≈ 5/(64π) ≈ 0.02487
const tprime   = 0.0                                # source time t'

gr()

# ─── Δ(r) in two conventions (equivalent) ────────────────────
kerr_delta(r, a; M=1.0) = r^2 - 2M*r + a^2
Δfactor(r) = (r - rplus) * (r - rminus)

# Kerr tortoise coordinate:
#   r* = r + [2Mr_+/(r_+-r_-)] ln((r-r_+)/(2M))
#          - [2Mr_-/(r_+-r_-)] ln((r-r_-)/(2M))
# (a=0 limit:  r* = r + 2M ln((r-2M)/(2M)) )
function rstar(r; M_=M, rp=rplus, rm=rminus)
    denom = rp - rm
    t1 = 2M_*rp/denom * log((r - rp)/(2M_))
    t2 = rm == 0 ? 0.0 : 2M_*rm/denom * log((r - rm)/(2M_))
    return r + t1 - t2
end

# ─── ψ_C leading-Kerr piece ──────────────────────────────────
#   G_C = Δ(r')^s · {}_sY_{lm}(θ) · {}_sY_{lm}(θ')
#         × (-1)^{l+s}(2l)! / [2^{l+s+1}(l-s)!(l+s)!]
#         × (r'-r_+)^{-s} (r'-r_- + u - t')^{l+s} / (r'-r_-)^{l+1}
#         × ((r'-r_-)/(r'-r_+))^{iτ₀/2}
# Here psi_C returns the radial part (everything except the two Y_{lm}),
# applied after ψ_PIA/ψ_C evaluation as a common Y_FACTOR.
function psi_C_radial(u, rprime, rp, rm, s::Integer, l::Integer, τ0;
                      tprime = 0.0)
    pref = (-1)^(l + s) * factorial(2l) /
           (2.0^(l + s + 1) * factorial(l - s) * factorial(l + s))
    base = (rprime - rm) / (rprime - rp)
    Φ    = (rprime - rp)^(-s) * (rprime - rm + u - tprime)^(l + s) /
           (rprime - rm)^(l + 1)
    return Δfactor(rprime)^s * pref * Φ * base^(im * τ0 / 2)
end

# ─── ψ_PIA σ-integrand builder  (from psi_rprime_integral.jl) ─
function logspaced_weights(σ_min, σ_max, Nσ)
    σ  = exp.(range(log(σ_min), log(σ_max); length=Nσ))
    Δσ = diff([0.0; (σ[1:end-1] .+ σ[2:end]) ./ 2; σ[end]])
    return σ, Δσ
end

function compute_integrand(σ_grid::AbstractVector{Float64}, a_val, rp;
                           nmax::Int = 100)
    Nσ        = length(σ_grid)
    integrand = Vector{ComplexF64}(undef, Nσ)
    Δinv2     = 1.0 / kerr_delta(rp, a_val)^2

    for i in 1:Nσ
        σ = σ_grid[i]
        ω = im * σ
        # Fresh solve every σ (no Newton continuation): more robust
        # for small a where branch-selection can drift.
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
        val == zero(ComplexF64) && (print("x"); flush(stdout))
        integrand[i] = val
        i % 25 == 0 && (print("."); flush(stdout))
    end
    println()
    return integrand
end

# ============================================================
# Main
# ============================================================
println("===== ψ_total(u;r'=$rprime) = ψ_PIA + ψ_C =====")
println("s=$s, l=$l, m=$m, a=$a, r'=$rprime,  τ₀(ω=0)=$(round(τ0;digits=6))")
println("r_+=$(round(rplus;digits=6))  r_-=$(round(rminus;digits=6))")
println("Δ(r') = ", kerr_delta(rprime, a))

# σ grid (same as psi_rprime_integral.jl).
Nσ = 300
σ_grid, Δσ = logspaced_weights(1e-3, 2.0, Nσ)

println("\n--- building σ-integrand F(σ) ---")
F = compute_integrand(σ_grid, a, rprime)
println("done.  |F| range: ", extrema(abs.(F)))

# ── u grid on [-r*', +r*'] ───────────────────────────────────
rstar_p = rstar(rprime)
println("r*(r'=$rprime) = ", round(rstar_p; digits=6))
u_grid = collect(range(-rstar_p, rstar_p; length=401))

ψ_PIA = Vector{ComplexF64}(undef, length(u_grid))
ψ_Cv  = Vector{ComplexF64}(undef, length(u_grid))
for (k, u) in enumerate(u_grid)
    # ψ_PIA: truncated σ-integral.  For u > 0 the integrand grows
    # with σ; values there are σ_max-dependent by construction.
    acc = zero(ComplexF64)
    @inbounds for i in 1:Nσ
        acc += F[i] * Δσ[i] * exp(σ_grid[i] * u)
    end
    ψ_PIA[k] = acc / (2π)
    ψ_Cv[k]  = psi_C_radial(u, rprime, rplus, rminus, s, l, τ0;
                            tprime = tprime)
end
ψ_tot = ψ_PIA .+ ψ_Cv

# Apply angular factor  {}_sY_{lm}(θ) · {}_sY_{lm}(θ')
ψ_PIA .*= Y_FACTOR
ψ_Cv  .*= Y_FACTOR
ψ_tot .*= Y_FACTOR
println("\napplied Y(θ)·Y(θ') = ", round(Y_FACTOR; sigdigits=6),
        "   (θ=θ'=π/2)")

# ── quick report at a few u values ─────────────────────────
println("\n--- sample values ---")
for u_tgt in (-rstar_p, -rstar_p/2, 0.0, rstar_p/2, rstar_p)
    k = argmin(abs.(u_grid .- u_tgt))
    @printf("u=%+6.2f  ψ_PIA=%+.3e%+.3ei   ψ_C=%+.3e%+.3ei   ψ_tot=%+.3e%+.3ei\n",
            u_grid[k],
            real(ψ_PIA[k]), imag(ψ_PIA[k]),
            real(ψ_Cv[k]),  imag(ψ_Cv[k]),
            real(ψ_tot[k]), imag(ψ_tot[k]))
end

# ── plots: real and imaginary parts, linear axes ──────────────
p = plot(xlabel = L"u/M", ylabel = L"\mathrm{Re}\,\psi",
            title  = latexstring("\\mathrm{Re}\\,\\psi\\ (r'=$(rprime),\\ a=$(a),\\ (s,l,m)=($s,$l,$m))"),
            framestyle = :box, grid = true, legend = :topright, 
            yscale=:log10, ylim=(1e-10, :auto),
            size = (900, 420), dpi = 150)
plot!(p, u_grid, abs.(real.(ψ_tot)); lw=1.8, label=L"\psi_{\rm PIA}+\psi_C", color=:black)

OUTPNG = joinpath(OUTDIR, "psi_combined_rprime_C.png")
savefig(p, OUTPNG)
println("\nsaved ", OUTPNG)
