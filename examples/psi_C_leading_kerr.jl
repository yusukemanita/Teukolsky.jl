using Plots
using Printf

# ============================================================
#   ψ_C (leading-Kerr I(u) with Φ(r')) — small-frequency form
#
#     Φ(r')  = (r'-r_+)^{-s} (r'-r_-)^{-(l+1)}
#              × ((r'-r_-)/(r'-r_+))^{iτ₀/2}
#
#     I(u)|_{l,s}^{leading Kerr}
#            = (-1)^{l+s} (2l)! / [2^{l+s+1} (l-s)! (l+s)!]
#              × Φ(r') × (r' - r_- + u)^{l+s}
#
#     ψ_C(u) = -I(u) Δ(r')^s,   Δ(r') = (r'-r_+)(r'-r_-)
# ============================================================

Δ(r, rp, rm) = (r - rp) * (r - rm)

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

function psi_C(u, rprime, rp, rm, s::Integer, l::Integer, τ0)
    return -I_leading(u, rprime, rp, rm, s, l, τ0) *
           Δ(rprime, rp, rm)^s
end

# ------------------------------------------------------------
#  Test plot:  r' = 10,  u ∈ [-10, 10],  Kerr (a = 0.7, M = 1)
# ------------------------------------------------------------
const M   = 1.0
const a   = 0.7
const rp  = M + sqrt(M^2 - a^2)
const rm  = M - sqrt(M^2 - a^2)
const τ0  = 1.0
const l   = 2
const rprime = 10.0

us = range(-10.0, 10.0; length = 801)

gr()
p = plot(xlabel = "u",
         ylabel = "ψ_C(u)",
         title  = @sprintf("ψ_C leading-Kerr  (r' = %.1f, a = %.2f, τ₀ = %.2f)",
                           rprime, a, τ0),
         size   = (900, 520),
         framestyle = :box, grid = true, legend = :topright)

for s in (-2, 0, 2)
    vals = [psi_C(u, rprime, rp, rm, s, l, τ0) for u in us]
    plot!(p, us, real.(vals); lw = 1.6, ls = :solid,
          label = "Re ψ_C  (s = $s)")
    plot!(p, us, imag.(vals); lw = 1.6, ls = :dash,
          label = "Im ψ_C  (s = $s)")
end

const OUTPNG = joinpath(@__DIR__, "psi_C_leading_kerr.png")
savefig(p, OUTPNG)
println("saved ", OUTPNG)

# ψ_C printout at a few u for the s = -2 gravitational case
println("\nψ_C(u)  at r' = $rprime,  l = $l,  s = -2,  a = $a,  τ₀ = $τ0")
for u in (-5.0, 0.0, 5.0)
    v = psi_C(u, rprime, rp, rm, -2, l, τ0)
    @printf("  u = %+.1f :  ψ_C = %+.6e  %+.6ei\n", u, real(v), imag(v))
end
