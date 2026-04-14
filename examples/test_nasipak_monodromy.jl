using Pkg; Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")
using BHPtoolkit, Plots, LaTeXStrings, Printf, Statistics, SpecialFunctions

# ============================================================
#  Test: Daalhuis-Olver monodromy method for cos(2πν)
#        (ported from Mathematica Teukolsky paclet,
#         which implements the algorithm from reference [26]:
#         Daalhuis & Olver 1995 / Nasipak arXiv:2412.06503)
#
#  CHE parameters (Mathematica convention):
#    ε = 2ω,  κ = √(1-a²),  τ = (ε-m·a)/κ
#    γCH = 1 - s - iε - iτ
#    δCH = 1 + s + iε - iτ
#    εCH = 2iεκ
#    αεCH = αCH·εCH = 1 - s + i(ε-τ)   (product, not separate)
#    qCH  = -(-s(1+s) + ε² + i(2s-1)εκ - Λ - τ(i+τ))
#           where Λ = full separation constant A_{slm}
#
#  μ₁ = αεCH - (γCH+δCH),  μ₂ = -αεCH
#
#  Two series (a1, a2) built by two-step recurrences.
#  Sums with Pochhammer symbols, then:
#
#  cos(2πν) = cos(π(μ₁-μ₂)) + 2π²/(a1sum·a2sum) · (-1)^{n-1} a1[n] a2[n]
#
#  NOTE: requires arbitrary precision (BigFloat) for reliable results.
# ============================================================

"""
    monodromy_cos2pi_nu_DH(spin, l, m, a, ω, λ_ang; nmax=60, prec=256, tol=1e-10)

Compute cos(2πν) using the Daalhuis-Olver method (ported from Mathematica
Teukolsky paclet νRCHMonodromy).

`λ_ang` should be `p.λ` (= full separation constant A minus s*(s+1)).

Returns `(cos2πν, converged, nmax_used)`.
"""
function monodromy_cos2pi_nu_DH(spin::Int, l::Int, m::Int, a, ω, λ_ang;
                                  nmax::Int=80, prec::Int=256, tol::Real=1e-10)
    result = setprecision(BigFloat, prec) do
        _dh_core(spin, l, m, a, ω, λ_ang; nmax=nmax, tol=tol)
    end
    return result
end

function _dh_core(spin::Int, l::Int, m::Int, a, ω, λ_ang;
                   nmax::Int=80, tol::Real=1e-10)
    BF = BigFloat
    C  = Complex{BF}

    ε  = BF(2) * BF(real(ω)) + BF(2) * im * BF(imag(ω))
    q  = BF(real(a))
    κ  = sqrt(C(1 - q^2))
    τ  = (ε - m*q) / κ

    # λ_ang = p.λ is already the full Mathematica Λ (= A_{lm}, eigenvalue of angular eq.)
    # BHPtoolkit convention: p.λ ≡ Λ (not Λ-s(s+1)); matches existing monodromy_cos2pi_nu
    Λ  = BF(λ_ang)

    # CHE parameters (Mathematica convention)
    γCH  = C(1 - spin) - im*ε - im*τ
    δCH  = C(1 + spin) + im*ε - im*τ
    εCH  = C(2) * im * ε * κ
    αεCH = C(1 - spin) + im*(ε - τ)   # = αCH * εCH
    qCH  = -(- C(spin*(1+spin)) + ε^2 + im*C(2*spin - 1)*ε*κ - Λ - τ*(im + τ))

    μ1 = αεCH - (γCH + δCH)
    μ2 = -αεCH
    Δμ = μ1 - μ2   # = 2αεCH - γCH - δCH

    π_BF = BF(π)

    # ── Build a1 and a2 arrays ──────────────────────────────────
    a1 = zeros(C, nmax + 2)   # a1[n+1] = a1 coefficient for index n
    a2 = zeros(C, nmax + 2)
    a1[1] = one(C)   # n=0
    a2[1] = one(C)   # n=0
    # n=-1 boundary: a1[-1]=0, a2[-1]=0 (already zero)

    for n in 1:nmax+1
        # a1 recurrence
        f2_a1 = (αεCH - (n - 1 + δCH)) * (αεCH - (n - 2 + γCH + δCH)) * εCH
        f1_a1 = αεCH^2 + αεCH*(1 - 2n - γCH - δCH + εCH) +
                n^2 - qCH + n*(-1 + γCH + δCH - εCH) + εCH - δCH*εCH
        a1prev = n >= 2 ? a1[n-1] : zero(C)
        a1[n+1] = (f2_a1 * a1prev - f1_a1 * a1[n]) / n

        # a2 recurrence
        g2_a2 = (αεCH + (n - 2)) * (αεCH + (n - 1 - γCH)) * εCH
        g1_a2 = αεCH^2 + n^2 - qCH + γCH + δCH - n*(1 + γCH + δCH - εCH) - εCH +
                αεCH*(-1 + 2n - γCH - δCH + εCH)
        a2prev = n >= 2 ? a2[n-1] : zero(C)
        a2[n+1] = (-g2_a2 * a2prev + g1_a2 * a2[n]) / n
    end

    # ── Pochhammer vectors: poch_p1m2[k+1] = (Δμ)_k, poch_m1p2[k+1] = (-Δμ)_k ─
    # poch_p1m2[k+1] = Δμ*(Δμ+1)*...*(Δμ+k-1)  [k-th Pochhammer of Δμ]
    poch_p1m2 = zeros(C, nmax + 2)
    poch_m1p2 = zeros(C, nmax + 2)
    poch_p1m2[1] = one(C)   # k=0
    poch_m1p2[1] = one(C)
    for k in 1:nmax+1
        poch_p1m2[k+1] = (Δμ + k - 1) * poch_p1m2[k]
        poch_m1p2[k+1] = (-Δμ + k - 1) * poch_m1p2[k]
    end

    # ── Sums and cos(2πν) via reflection formula ─────────────────
    # a1sum = Γ(Δμ)  * Σ_{j=0..ceil(n/2)} a1[j] * (Δμ)_{n-j}
    # a2sum = Γ(-Δμ) * Σ_{j=0..ceil(n/2)} (-1)^j a2[j] * (-Δμ)_{n-j}
    #
    # Use Γ(Δμ)·Γ(-Δμ) = -π/(Δμ·sin(π·Δμ)) to avoid gamma(Complex{BigFloat}):
    #
    # (2π²)/(a1sum·a2sum) = (2π²)·(-Δμ·sin(πΔμ)/π) / (sum1·sum2)
    #                     = -2π·Δμ·sin(πΔμ) / (sum1·sum2)
    #
    # so: cos(2πν) = cos(πΔμ) - 2π·Δμ·sin(πΔμ)·(-1)^{n-1}·a1[n]·a2[n] / (sum1·sum2)

    cos2pinu_prev = C(NaN)
    cos2pinu      = C(NaN)
    converged     = false
    nmax_used     = nmax

    for n in 2:nmax+1
        jmax = cld(n, 2)

        sum1 = zero(C)
        sum2 = zero(C)
        for j in 0:jmax
            j > n && break
            k = n - j
            sum1 +=        a1[j+1] * poch_p1m2[k+1]
            sum2 += (-1)^j * a2[j+1] * poch_m1p2[k+1]
        end

        (!isfinite(real(sum1)) || !isfinite(real(sum2)) ||
         abs(sum1) < 1e-300 || abs(sum2) < 1e-300) && continue

        # Reflection formula: Γ(Δμ)·Γ(-Δμ) = -π/(Δμ·sin(πΔμ))
        # (2π²)/(Γ(Δμ)·Γ(-Δμ)·sum1·sum2) = -2π·Δμ·sin(πΔμ)/(sum1·sum2)
        factor = -2 * π_BF * Δμ * sin(π_BF * Δμ) / (sum1 * sum2)
        corr = factor * (-1)^(n-1) * a1[n+1] * a2[n+1]
        cos2pinu_new = cos(π_BF * Δμ) + corr

        !isfinite(real(cos2pinu_new)) && continue

        if n > 2 && abs(cos2pinu_new - cos2pinu_prev) < tol * max(one(BF), abs(real(cos2pinu_new)))
            converged = true
            nmax_used = n
            cos2pinu  = cos2pinu_new
            break
        end
        cos2pinu_prev = cos2pinu_new
        cos2pinu      = cos2pinu_new
    end

    return ComplexF64(cos2pinu), converged, nmax_used
end

# ============================================================
#  Comparison table
# ============================================================
println("="^72)
println("monodromy_cos2pi_nu_DH  vs  monodromy_cos2pi_nu  (s=-2, l=2, m=2, a=0)")
println("="^72)
@printf("%-8s | %-22s | %-22s | %s\n",
        "ω", "Re(cos2πν) existing", "Re(cos2πν) DH", "diff")
println("-"^72)

test_omegas = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.3, 2.5, 3.0]
spin_test, l_test, m_test, a_test = -2, 2, 2, 0.0

for ω in test_omegas
    p   = MSTParams(spin_test, l_test, m_test, a_test, ω)
    λ   = p.λ

    c2pn_exist = BHPtoolkit.monodromy_cos2pi_nu(spin_test, l_test, m_test, a_test, ω, λ)
    c2pn_dh, conv, n_use = monodromy_cos2pi_nu_DH(spin_test, l_test, m_test, a_test, ω, λ)

    diff = abs(c2pn_exist - c2pn_dh)
    conv_str = conv ? "(n=$n_use)" : "(NC)"
    @printf("ω=%5.2f | %+22.10e | %+22.10e | %.2e  %s\n",
            ω, real(c2pn_exist), real(c2pn_dh), diff, conv_str)
end
println("-"^72)

# ============================================================
#  Also test for Kerr (a=0.9)
# ============================================================
println()
println("="^72)
println("monodromy_cos2pi_nu_DH  vs  monodromy_cos2pi_nu  (s=-2, l=2, m=2, a=0.9)")
println("="^72)
@printf("%-8s | %-22s | %-22s | %s\n",
        "ω", "Re(cos2πν) existing", "Re(cos2πν) DH", "diff")
println("-"^72)

for ω in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5]
    p   = MSTParams(spin_test, l_test, m_test, 0.9, ω)
    λ   = p.λ

    c2pn_exist = BHPtoolkit.monodromy_cos2pi_nu(spin_test, l_test, m_test, 0.9, ω, λ)
    c2pn_dh, conv, n_use = monodromy_cos2pi_nu_DH(spin_test, l_test, m_test, 0.9, ω, λ)

    diff = abs(c2pn_exist - c2pn_dh)
    conv_str = conv ? "(n=$n_use)" : "(NC)"
    @printf("ω=%5.2f | %+22.10e | %+22.10e | %.2e  %s\n",
            ω, real(c2pn_exist), real(c2pn_dh), diff, conv_str)
end
println("-"^72)

# ============================================================
#  Sweep plot: ω ∈ [0.05, 3.0]
# ============================================================
println("\nSweep plot (200 points) ...")
ω_sweep    = range(0.05, 3.0; length=200)
rc_exist   = Float64[]
rc_dh      = Float64[]

for ω in ω_sweep
    p  = MSTParams(spin_test, l_test, m_test, a_test, ω)
    λ  = p.λ
    c_ex = BHPtoolkit.monodromy_cos2pi_nu(spin_test, l_test, m_test, a_test, ω, λ)
    c_dh, _, _ = monodromy_cos2pi_nu_DH(spin_test, l_test, m_test, a_test, ω, λ)
    push!(rc_exist, real(c_ex))
    push!(rc_dh,    real(c_dh))
end

ω_arr = collect(Float64.(ω_sweep))

clamp_val = 5.0
rc_ex_c = clamp.(rc_exist, -clamp_val, clamp_val)
rc_dh_c = clamp.(rc_dh,    -clamp_val, clamp_val)

fig = plot(
    ω_arr, rc_dh_c;
    label      = "Daalhuis-Olver (BigFloat)",
    color      = :steelblue, lw = 2,
    xlabel     = L"\omega\ [M^{-1}]",
    ylabel     = L"\mathrm{Re}[\cos(2\pi\nu)]\ (\mathrm{clamped\ to} \pm 5)",
    title      = L"s=-2,\ l=2,\ m=2,\ a=0",
    framestyle = :box, grid = true,
    fontfamily = "Computer Modern", dpi = 150,
    size       = (900, 450),
    ylims      = (-clamp_val - 0.3, clamp_val + 0.3))

plot!(fig, ω_arr, rc_ex_c;
    label  = "Sasaki-Tagoshi (nmax=60)",
    color  = :crimson, lw = 1.5, ls = :dash, alpha = 0.8)

hline!(fig, [ 1.0]; label = "rc = +1", color = :black, lw = 0.8, ls = :dot)
hline!(fig, [-1.0]; label = "rc = -1", color = :gray,  lw = 0.8, ls = :dot)
vline!(fig, [2.0];  label = "ω ≈ 2",  color = :orange, lw = 1.0, ls = :dash)

outdir = @__DIR__
savefig(fig, joinpath(outdir, "nasipak_comparison.png"))
println("Saved: nasipak_comparison.png")
display(fig)

# ============================================================
#  Stability check near ω≈2
# ============================================================
println("\nStability check near ω≈2 (21 points in [1.9, 2.1]):")

rc_exist_vals = Float64[]
rc_dh_vals    = Float64[]
for ω in range(1.9, 2.1; length=21)
    p = MSTParams(spin_test, l_test, m_test, a_test, ω)
    push!(rc_exist_vals, real(BHPtoolkit.monodromy_cos2pi_nu(spin_test, l_test, m_test, a_test, ω, p.λ)))
    c, _, _ = monodromy_cos2pi_nu_DH(spin_test, l_test, m_test, a_test, ω, p.λ)
    push!(rc_dh_vals, real(c))
end

@printf("  std(rc) existing  : %.4e\n", std(rc_exist_vals))
@printf("  std(rc) DH method : %.4e\n", std(rc_dh_vals))
ratio = std(rc_exist_vals) / max(std(rc_dh_vals), 1e-30)
@printf("  std ratio (exist/DH): %.1f\n", ratio)
if ratio > 5
    println("  DH method is >5x smoother near ω≈2 ✓")
else
    println("  ~ Methods have similar smoothness near ω≈2")
end

# ── Agreement table near the real branch (ω small, |rc|<1) ──────────────────
println("\nPrecision comparison on real branch (|rc|≤1), a=0:")
@printf("%-8s | %-22s | %-22s | %-12s | %s\n",
        "ω", "existing (Float64)", "DH (BigFloat)", "rel_diff", "conv?")
println("-"^82)
for ω in range(0.05, 0.35; length=11)
    p = MSTParams(spin_test, l_test, m_test, a_test, ω)
    c_ex = BHPtoolkit.monodromy_cos2pi_nu(spin_test, l_test, m_test, a_test, ω, p.λ)
    c_dh, cv, nu = monodromy_cos2pi_nu_DH(spin_test, l_test, m_test, a_test, ω, p.λ)
    rdiff = abs(c_ex - c_dh) / max(abs(c_dh), 1e-30)
    @printf("ω=%5.3f | %+22.14e | %+22.14e | %.3e  | %s\n",
            ω, real(c_ex), real(c_dh), rdiff, cv ? "conv(n=$nu)" : "NC")
end
