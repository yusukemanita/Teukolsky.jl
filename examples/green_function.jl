"""
Compute the frequency-domain Green's function G_lmω(r, r')
for the radial Teukolsky equation.

Parameters: s=-2, l=m=2, a=0 (Schwarzschild), r'=10M, r=30M
Scan ω ∈ (-3, 3), skipping ω ≈ 0.

The Green's function is:
    G(ω, r, r') = R_in(r_<) R_up(r_>) / W(r_ref)

where W(r) = R_in(r) R'_up(r) - R_up(r) R'_in(r)
and r_< = min(r, r'), r_> = max(r, r').
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BHPtoolkit
using Printf

# ── Parameters ──
s, l, m, a = -2, 2, 2, 0.0
r_prime = 10.0   # source point (r')
r_field = 30.0   # field point (r)
r_less = min(r_prime, r_field)   # r_< = 10
r_greater = max(r_prime, r_field)  # r_> = 30

# ── Frequency grid: avoid ω = 0 ──
ω_neg = range(-3.0, -0.02, length=150)
ω_pos = range(0.02, 3.0, length=150)
ω_all = vcat(collect(ω_neg), collect(ω_pos))

println("=" ^ 70)
println("  Green's function G_{lmω}(r, r')")
println("  s=$s, l=$l, m=$m, a=$a")
@printf("  r' = %.1fM (source),  r = %.1fM (field)\n", r_prime, r_field)
@printf("  r_< = %.1fM,  r_> = %.1fM\n", r_less, r_greater)
println("  ω ∈ [-3, -0.02] ∪ [0.02, 3]  (300 points)")
println("=" ^ 70)

# ── Compute ──
results = []
n_fail = 0

@printf("\n%10s  %24s  %24s  %12s\n",
        "ω", "Re[G]", "Im[G]", "|G|")
println("-" ^ 76)

for ω in ω_all
    try
        ν, p = compute_nu(s, l, m, a, ω)
        fn = compute_fn(p, ν)

        # Evaluate radial solutions
        rin_less = Rin(p, ν, fn, r_less)
        rup_greater = Rup(p, ν, fn, r_greater)

        # Wronskian at r_< (could use any r, but r_< is well-converged)
        rin_w = rin_less
        drin_w = dRin(p, ν, fn, r_less)
        rup_w = Rup(p, ν, fn, r_less)
        drup_w = dRup(p, ν, fn, r_less)
        W = rin_w * drup_w - rup_w * drin_w

        G = rin_less * rup_greater / W

        push!(results, (ω=ω, G=G, ν=ν))

        if abs(ω - round(ω; digits=1)) < 0.011 || abs(ω) < 0.03
            @printf("%+10.4f  %+24.12e  %+24.12e  %12.6e\n",
                    ω, real(G), imag(G), abs(G))
        end
    catch e
        global n_fail += 1
        push!(results, (ω=ω, G=NaN+NaN*im, ν=NaN+NaN*im))
        if abs(ω - round(ω; digits=1)) < 0.011
            @printf("%+10.4f  FAILED: %s\n", ω, sprint(showerror, e))
        end
    end
end

println("-" ^ 76)
@printf("Completed: %d/%d successful", length(ω_all) - n_fail, length(ω_all))
n_fail > 0 && @printf(", %d failed", n_fail)
println()

# ── Summary statistics ──
valid = filter(r -> isfinite(r.G), results)
if !isempty(valid)
    abs_G = [abs(r.G) for r in valid]
    idx_max = argmax(abs_G)
    println("\nPeak |G|:")
    r_peak = valid[idx_max]
    @printf("  ω = %+.6f,  |G| = %.6e\n", r_peak.ω, abs(r_peak.G))
    @printf("  G  = %+.8e %+.8ei\n", real(r_peak.G), imag(r_peak.G))
    @printf("  ν  = %+.8f %+.8fi\n", real(r_peak.ν), imag(r_peak.ν))
end

# ── Save data for plotting ──
output_file = joinpath(@__DIR__, "green_function_data.dat")
open(output_file, "w") do io
    @printf(io, "# Green's function G_{lmω}(r,r') for Teukolsky equation\n")
    @printf(io, "# s=%d, l=%d, m=%d, a=%.1f, r'=%.1fM, r=%.1fM\n",
            s, l, m, a, r_prime, r_field)
    @printf(io, "# Columns: ω  Re[G]  Im[G]  |G|  Re[ν]  Im[ν]\n")
    for r in valid
        @printf(io, "%+.8e  %+.16e  %+.16e  %.16e  %+.16e  %+.16e\n",
                r.ω, real(r.G), imag(r.G), abs(r.G), real(r.ν), imag(r.ν))
    end
end
println("\nData saved to: $output_file")
