using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")

using Teukolsky
using Plots
using LaTeXStrings
using Printf

# ============================================================
#  Branch-cut integral evaluated at finite radius r'
#
#    ψ(u; r') = (1/2π) ∫ dσ  Δ(r')^{-2} q̃(iσ) R^up(r', iσ)
#                             ────────────────────────────── e^{σu}
#                                        2 i ω
#
#  with  ω = iσ  (positive imaginary axis, same convention as
#  ΔG⁺ in generate_figures.jl).
# ============================================================

const OUTDIR = @__DIR__
const s, l, m = -2, 2, 2
const a       = 0.9
const rp      = 10.0   # r'
const M       = 1.0

gr()

# Δ(r) = r^2 - 2Mr + a^2
function kerr_delta(r, a; M=1.0)
    return r^2 - 2M*r + a^2
end

# trapezoidal weights on log-spaced grid (same helper as generate_figures.jl)
function logspaced_weights(σ_min, σ_max, Nσ)
    σ  = exp.(range(log(σ_min), log(σ_max); length=Nσ))
    Δσ = diff([0.0; (σ[1:end-1] .+ σ[2:end]) ./ 2; σ[end]])
    return σ, Δσ
end

# ── Build the σ-integrand at r'  ─────────────────────────────
# integrand(σ)  :=  Δ(r')^{-2} q̃(iσ) R^up(r', iσ) / (2 i ω)
#
# Called once per σ; Newton-continues ν from the previous σ.
function compute_integrand(σ_grid::AbstractVector{Float64}, a_val, rp;
                           nmax::Int = 100)
    Nσ        = length(σ_grid)
    integrand = Vector{ComplexF64}(undef, Nσ)
    qt_arr    = Vector{ComplexF64}(undef, Nσ)
    Rup_arr   = Vector{ComplexF64}(undef, Nσ)
    ν_prev    = nothing

    Δinv2 = 1.0 / kerr_delta(rp, a_val)^2

    for i in 1:Nσ
        σ = σ_grid[i]
        ω = im * σ

        qt = ν_prev === nothing ?
             compute_qtilde(s, l, m, Float64(a_val), ω; nmax=nmax) :
             compute_qtilde(s, l, m, Float64(a_val), ω; nmax=nmax,
                            ν_init=ν_prev, method="Newton")
        ν_prev = qt.ν
        fn = compute_fn(qt.p, qt.ν; nmax=nmax)
        Rv = Rup(qt.p, qt.ν, fn, rp; nmax=nmax)

        qt_arr[i]    = qt.qtilde
        Rup_arr[i]   = Rv
        integrand[i] = Δinv2 * qt.qtilde * Rv / (2im * ω)

        i % 25 == 0 && (print("."); flush(stdout))
    end
    println()
    return (; integrand, qtilde = qt_arr, Rup = Rup_arr, Δinv2)
end

# ============================================================
# Main
# ============================================================

println("===== ψ(u;r'=$rp)  via branch-cut PIA integral =====")
println("s=$s, l=$l, m=$m, a=$a, r'=$rp")
println("Δ(r') = ", kerr_delta(rp, a), "   Δ(r')^{-2} = ",
        1/kerr_delta(rp, a)^2)

# σ grid.  σ_max = 2 is enough here: the Float64-clean integrand |F(σ)| has
# already fallen below ~1e-15 near σ ≈ 1, and the σ > 2 tail contributes
# < 1e-4 relative to the total even at u = -0.5.  Going further into σ is
# where the MST sums (R^up, q̃) start losing digits to cancellation and the
# "bumps" appear — that's the Float64 precision wall for this integrand.
# For full-precision large-σ work we'd need to make Rup BigFloat-capable
# (src/radial_up.jl:53 hardcodes Dict{Int, ComplexF64}).
Nσ = 300
σ_grid, Δσ = logspaced_weights(1e-3, 2.0, Nσ)

println("\n--- building integrand F(σ) at r'=$rp ---")
res = compute_integrand(σ_grid, a, rp)
F   = res.integrand
println("done.  |F| range: ", extrema(abs.(F)))

# ── ψ(u) for a grid of u ──────────────────────────────────────
u_grid = collect(range(-80.0, -0.5; length=400))
ψ_u = Vector{ComplexF64}(undef, length(u_grid))
for (k, u) in enumerate(u_grid)
    s_sum = zero(ComplexF64)
    @inbounds for i in 1:Nσ
        s_sum += F[i] * Δσ[i] * exp(σ_grid[i] * u)
    end
    ψ_u[k] = s_sum / (2π)
end

# ── report --------------------------------------------------
println("\n--- ψ(u;r'=$rp) summary ---")
@printf("u=%-7.2f   ψ = %.6e %+.6e i   |ψ|=%.3e\n",
        u_grid[end], real(ψ_u[end]), imag(ψ_u[end]), abs(ψ_u[end]))
@printf("u=%-7.2f   ψ = %.6e %+.6e i   |ψ|=%.3e\n",
        -10.0, real(ψ_u[argmin(abs.(u_grid .+ 10.0))]),
        imag(ψ_u[argmin(abs.(u_grid .+ 10.0))]),
        abs(ψ_u[argmin(abs.(u_grid .+ 10.0))]))
@printf("u=%-7.2f   ψ = %.6e %+.6e i   |ψ|=%.3e\n",
        u_grid[1], real(ψ_u[1]), imag(ψ_u[1]), abs(ψ_u[1]))

# ── plots ---------------------------------------------------
# (1) integrand |F(σ)|
p1 = plot(σ_grid, abs.(F),
    xscale=:log10, yscale=:log10,
    xlabel=L"\sigma", ylabel=L"|F(\sigma)|",
    title = latexstring("\\mathrm{Integrand\\ at\\ } r'=$(rp): ",
                        "\\ |\\Delta^{-2}\\,\\tilde q\\,R^{up}/(2i\\omega)|"),
    lw=1.8, label="", framestyle=:box, grid=true,
    size=(780, 440), dpi=150)
savefig(p1, joinpath(OUTDIR, "psi_rprime_integrand.png"))

# (2) ψ(u) at r'
p2 = plot(abs.(u_grid), abs.(real.(ψ_u)),
    xscale=:log10, yscale=:log10,
    xlabel=L"|u|/M", ylabel=L"|\mathrm{Re}\,\psi(u;r')|",
    title = latexstring("\\psi(u;r'=$(rp)) — branch cut (PIA)"),
    lw=1.8, label="Re ψ", color=:crimson,
    framestyle=:box, grid=true, legend=:topright,
    size=(780, 440), dpi=150)
plot!(p2, abs.(u_grid), abs.(imag.(ψ_u));
    lw=1.5, label="Im ψ", color=:steelblue, ls=:dash)
plot!(p2, abs.(u_grid), abs.(ψ_u);
    lw=1.2, label="|ψ|", color=:black, ls=:dot)
savefig(p2, joinpath(OUTDIR, "psi_rprime_vs_u.png"))

println("\nSaved:")
println("  ", joinpath(OUTDIR, "psi_rprime_integrand.png"))
println("  ", joinpath(OUTDIR, "psi_rprime_vs_u.png"))
