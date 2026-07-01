# ============================================================================
#  Arb-backend validation harness for the MST radial quantities
#  (B^inc, B^ref, R^in, R^up) at large complex frequency.
#
#  Self-consistency methodology
#  ----------------------------
#  The literal cross-check against Mathematica's Teukolsky package at 100 digits
#  is INFEASIBLE for complex ω: that package returns `Indeterminate` for B^inc,
#  B^ref, R^in, R^up at any Im(ω) ≠ 0 (it only resolves ν there, to ~15 digits).
#  So we validate the Arb backend against the independent BigFloat (MPFR) path of
#  THIS library at the SAME working precision — a genuine self-consistency test:
#  the two backends share no arithmetic kernels (native Arb ball ops vs MPFR), so
#  agreement to working precision certifies the Arb code path (type dispatch,
#  Acb/loggamma bridges, point-arithmetic continued fraction).  The radial
#  functions get an additional, reference-free check: the Wronskian
#      W = Δ^{s+1}(R^in (R^up)' − (R^in)' R^up)
#  is r-independent for any pair of homogeneous solutions, so |W(r₁)−W(r₂)|/|W|
#  measures how well the truncated MST series actually solves the equation.
#
#  Run:  julia --project=. examples/arb_validation/arb_amplitude_validation.jl
#  Produces the figures in ../../figures and a results table next to this file.
# ============================================================================

using Teukolsky
using Arblib: Arb
using Printf
using Plots

const B = Teukolsky
const S, L, M = -2, 2, 2
const OUTDIR  = joinpath(@__DIR__, "..", "..", "figures")
const RESULTS = joinpath(@__DIR__, "results.txt")

gr()
default(linewidth=2, markersize=5, legend=:best, framestyle=:box,
        guidefontsize=11, tickfontsize=9, legendfontsize=8)

# Number of agreeing digits between an Arb result and the BigFloat reference,
# both compared as midpoints at `prec` bits.  Capped at the working precision.
digits_agree(x_arb, x_bf, prec) = setprecision(BigFloat, prec) do
    d = abs(Complex{BigFloat}(x_arb) - x_bf) / abs(x_bf)
    d == 0 ? Float64(prec) * log10(2) : clamp(-log10(Float64(d)), 0.0, prec * log10(2))
end

# Arb and BigFloat amplitudes at one (a, ω) point, returning agreement digits.
function amp_point(a, re, im, prec; nmax=80)
    ra = setprecision(Arb, prec) do
        compute_amplitudes(S, L, M, Arb(a), Complex{Arb}(Arb(re), Arb(im)); nmax=nmax)
    end
    rb = setprecision(BigFloat, prec) do
        compute_amplitudes(S, L, M, BigFloat(a), Complex{BigFloat}(re, im); nmax=nmax)
    end
    (binc = digits_agree(ra.Binc, rb.Binc, prec),
     bref = digits_agree(ra.Bref, rb.Bref, prec))
end

# Radial Wronskian relative deviation |W(r₁)−W(r₂)|/|W(r₁)|, Arb backend.
function wronskian_reldev(a, re, im, prec; nmax=80, rs=(8.0, 20.0))
    setprecision(Arb, prec) do
        aA = Arb(a); ωA = Complex{Arb}(Arb(re), Arb(im))
        ν, p = B._compute_nu_monodromy(S, L, M, aA, ωA)
        fn = B.compute_fn(p, ν; nmax=nmax)
        W(r) = begin
            rA = Arb(r); Δ = rA^2 - 2rA + aA^2
            Δ^(S + 1) * (B.Rin(p, ν, fn, rA; nmax=nmax) * B.dRup(p, ν, fn, rA; nmax=nmax) -
                         B.dRin(p, ν, fn, rA; nmax=nmax) * B.Rup(p, ν, fn, rA; nmax=nmax))
        end
        w1 = W(rs[1]); w2 = W(rs[2])
        Float64(setprecision(BigFloat, prec) do
            abs(Complex{BigFloat}(w1) - Complex{BigFloat}(w2)) / abs(Complex{BigFloat}(w1))
        end)
    end
end

# Same Wronskian self-consistency in the BigFloat backend (reference: the MST
# algorithm itself converges with nmax — isolating that the Arb shortfall at
# |ω|≥2 is the hypergeometric ball-arithmetic path, not the radial formalism).
function wronskian_reldev_bf(a, re, im, prec; nmax=80, rs=(8.0, 20.0))
    setprecision(BigFloat, prec) do
        p = B.MSTParams(S, L, M, BigFloat(a), Complex{BigFloat}(re, im))
        ν, _ = compute_nu(S, L, M, a, complex(re, im); precision=prec)
        fn = B.compute_fn(p, ν; nmax=nmax)
        W(r) = begin
            rB = BigFloat(r); Δ = rB^2 - 2rB + BigFloat(a)^2
            Δ^(S + 1) * (B.Rin(p, ν, fn, rB; nmax=nmax) * B.dRup(p, ν, fn, rB; nmax=nmax) -
                         B.dRin(p, ν, fn, rB; nmax=nmax) * B.Rup(p, ν, fn, rB; nmax=nmax))
        end
        w1 = W(rs[1]); w2 = W(rs[2])
        Float64(abs(w1 - w2) / abs(w1))
    end
end

results = IOBuffer()
logln(args...) = (s = string(args...); println(s); println(results, s))

# ----------------------------------------------------------------------------
# (1) θ sweep of the amplitude agreement at |ω| = 2 and |ω| = 10, 256-bit.
# ----------------------------------------------------------------------------
const THETAS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 89.99]
const SPINS  = [(0.0, "Schwarzschild a=0"), (0.9, "Kerr a=0.9")]
const PREC0  = 256

function theta_sweep(absω, prec)
    out = Dict{Float64, Vector{NamedTuple}}()
    for (a, _) in SPINS
        row = NamedTuple[]
        for θ in THETAS
            re = absω * cosd(θ); im = absω * sind(θ)
            push!(row, amp_point(a, re, im, prec))
        end
        out[a] = row
    end
    return out
end

logln("="^72)
logln("Arb vs BigFloat MST amplitudes — agreement digits (s=$S, l=m=$M, $(PREC0)-bit)")
logln("θ sweep: ω = |ω|·e^{iθ}, |ω| ∈ {2, 10}")
logln("="^72)

sweep_data = Dict{Int, Any}()
for absω in (2, 10)
    global sweep_data
    logln("\n|ω| = $absω")
    sw = theta_sweep(absω, PREC0)
    sweep_data[absω] = sw
    for (a, lbl) in SPINS
        logln("  $lbl:")
        for (k, θ) in enumerate(THETAS)
            r = sw[a][k]
            logln(@sprintf("    θ=%6.2f°   Binc: %5.1f digits   Bref: %5.1f digits",
                           θ, r.binc, r.bref))
        end
    end
end

# Figure 1 & 2: agreement digits vs θ for each |ω|.
for absω in (2, 10)
    plt = plot(title="Arb vs BigFloat amplitude agreement, |ω|=$absω  (s=-2, l=m=2, 256-bit)",
               xlabel="θ  (deg),   ω = |ω| e^{iθ}", ylabel="agreeing digits",
               ylim=(0, 80))
    for (a, lbl) in SPINS
        bincs = [sweep_data[absω][a][k].binc for k in eachindex(THETAS)]
        brefs = [sweep_data[absω][a][k].bref for k in eachindex(THETAS)]
        plot!(plt, THETAS, bincs, marker=:circle, label="$lbl — Binc")
        plot!(plt, THETAS, brefs, marker=:square, linestyle=:dash, label="$lbl — Bref")
    end
    savefig(plt, joinpath(OUTDIR, "arb_amp_agreement_w$(absω).pdf"))
    savefig(plt, joinpath(OUTDIR, "arb_amp_agreement_w$(absω).png"))
end

# ----------------------------------------------------------------------------
# (2) Precision recovery at |ω| = 10 (the conditioning loss is constant in bits,
#     so the agreement grows ~linearly with the working precision).
# ----------------------------------------------------------------------------
const PRECS = [256, 512, 1024]
logln("\n" * "="^72)
logln("Precision recovery at |ω| = 10, θ = 45° (Schw) / 60° (Kerr)")
logln("="^72)
recov = Dict{Float64, Vector{Float64}}()
recov_modes = [(0.0, 45.0, "Schwarzschild a=0, θ=45°"),
               (0.9, 60.0, "Kerr a=0.9, θ=60°")]
for (a, θ, lbl) in recov_modes
    re = 10 * cosd(θ); im = 10 * sind(θ)
    ds = [amp_point(a, re, im, p).binc for p in PRECS]
    recov[a] = ds
    logln("  $lbl:")
    for (p, d) in zip(PRECS, ds)
        # conditioning loss in BITS = (full digits − achieved digits) / log10(2)
        loss_bits = (p * log10(2) - d) / log10(2)
        logln(@sprintf("    prec=%5d bits   Binc: %6.1f digits   (conditioning loss ≈ %4.0f bits)",
                       p, d, loss_bits))
    end
end
let plt = plot(title="Arb amplitude precision recovery at |ω|=10",
               xlabel="working precision (bits)", ylabel="Binc agreement digits",
               legend=:topleft)
    for (a, _, lbl) in recov_modes
        plot!(plt, PRECS, recov[a], marker=:circle, label=lbl)
    end
    plot!(plt, PRECS, [p * log10(2) for p in PRECS], linestyle=:dot, color=:black,
          label="full precision (prec·log₁₀2)")
    savefig(plt, joinpath(OUTDIR, "arb_amp_precision_recovery.pdf"))
    savefig(plt, joinpath(OUTDIR, "arb_amp_precision_recovery.png"))
end

# ----------------------------------------------------------------------------
# (3) Radial Wronskian self-consistency vs nmax (radial series truncation).
# ----------------------------------------------------------------------------
const NMAXES = [80, 160, 240, 320, 400]
logln("\n" * "="^72)
logln("Radial Wronskian self-consistency |W(8)-W(20)|/|W|  vs nmax (256-bit)")
logln("Arb (solid) vs BigFloat (dashed) — the formalism converges (BigFloat);")
logln("at |ω|≥2 the Arb hypergeometric (2F1) ball path is conditioning-limited.")
logln("="^72)
wron_modes = [(0.0, 0.5, 0.0, "Schw |ω|=0.5"),
              (0.0, 2*cosd(45), 2*sind(45), "Schw |ω|=2, 45°"),
              (0.9, 2*cosd(45), 2*sind(45), "Kerr |ω|=2, 45°")]
let plt = plot(title="Radial Wronskian self-consistency vs nmax (256-bit)",
               xlabel="nmax (radial MST truncation)", ylabel="|W(8)-W(20)| / |W|",
               yscale=:log10, legend=:outertopright, size=(760, 460))
    for (i, (a, re, im, lbl)) in enumerate(wron_modes)
        devs_a = [max(wronskian_reldev(a, re, im, PREC0; nmax=n), 1e-99) for n in NMAXES]
        devs_b = [max(wronskian_reldev_bf(a, re, im, PREC0; nmax=n), 1e-99) for n in NMAXES]
        logln("  $lbl:")
        for (n, da, db) in zip(NMAXES, devs_a, devs_b)
            logln(@sprintf("    nmax=%4d   Arb rel-dev = %.3e   BigFloat rel-dev = %.3e", n, da, db))
        end
        plot!(plt, NMAXES, devs_a, marker=:circle, color=i, label="$lbl (Arb)")
        plot!(plt, NMAXES, devs_b, marker=:square, linestyle=:dash, color=i, label="$lbl (BigFloat)")
    end
    savefig(plt, joinpath(OUTDIR, "arb_radial_wronskian_nmax.pdf"))
    savefig(plt, joinpath(OUTDIR, "arb_radial_wronskian_nmax.png"))
end

open(RESULTS, "w") do io
    write(io, String(take!(results)))
end
println("\nWrote results to $RESULTS and figures to $OUTDIR")
