using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")
using BHPtoolkit
using Printf

# ============================================================
#  Test: Rdown vs linear combination of Rin and Rup
#
#  From Sasaki-Tagoshi:
#    Rin = Binc × Rdown + Bref × Rup_normed
#
#  where Rup_normed = Rup / Ctrans, Rdown = R^ν_+ / norm.
#  Therefore:
#    Rdown_expected = (Rin - Bref × Rup_normed) / Binc
#
#  This tests that Rdown is a valid Teukolsky solution with
#  the correct normalization.
# ============================================================

test_cases = [
    # (s, l, m, a, ω, label)
    # Real ω — Schwarzschild
    (-2, 2, 2, 0.0, 0.3,           "Schw  ω=0.3"),
    (-2, 2, 2, 0.0, 0.5,           "Schw  ω=0.5"),
    (-2, 2, 1, 0.0, 0.3,           "Schw  m=1 ω=0.3"),
    (-2, 3, 3, 0.0, 0.3,           "Schw  l=3 ω=0.3"),
    # Real ω — Kerr
    (-2, 2, 2, 0.5, 0.3,           "Kerr  a=0.5 ω=0.3"),
    (-2, 2, 2, 0.9, 0.3,           "Kerr  a=0.9 ω=0.3"),
    (-2, 2, 2, 0.9, 0.5,           "Kerr  a=0.9 ω=0.5"),
    (-2, 2, 1, 0.9, 0.3,           "Kerr  a=0.9 m=1 ω=0.3"),
    # Complex ω — Schwarzschild
    (-2, 2, 2, 0.0, 0.3 - 0.1im,   "Schw  ω=0.3-0.1i"),
    (-2, 2, 2, 0.0, 0.5 + 0.2im,   "Schw  ω=0.5+0.2i"),
    (-2, 2, 2, 0.0, 0.1 - 0.3im,   "Schw  ω=0.1-0.3i"),
    # Complex ω — Kerr
    (-2, 2, 2, 0.5, 0.3 - 0.1im,   "Kerr  a=0.5 ω=0.3-0.1i"),
    (-2, 2, 2, 0.9, 0.3 - 0.1im,   "Kerr  a=0.9 ω=0.3-0.1i"),
    (-2, 2, 2, 0.9, 0.5 + 0.2im,   "Kerr  a=0.9 ω=0.5+0.2i"),
    (-2, 2, 1, 0.9, 0.3 - 0.2im,   "Kerr  a=0.9 m=1 ω=0.3-0.2i"),
]

r_test = [4.0, 6.0, 8.0, 10.0]
tol = 1e-6

println("="^90)
println("Rdown vs (Rin - Bref*Rup/Ctrans) / Binc")
println("="^90)
@printf("%-30s | %-6s | %-14s %-14s | %s\n",
        "Case", "r", "|Rdown|", "|Rdown_exp|", "rel_err")
println("-"^90)

n_pass = 0
n_fail = 0

for (s, l, m, a, ω, label) in test_cases
    ν, p = compute_nu(s, l, m, a, ω)
    fn = compute_fn(p, ν; nmax=40)
    amp = compute_amplitudes(s, l, m, a, ω; nmax=40)

    for r in r_test
        r <= p.rp + 0.1 && continue

        Rdown_val = Rdown(p, ν, fn, r; nmax=40)
        Rin_val   = Rin(p, ν, fn, r; nmax=40)
        Rup_val   = Rup(p, ν, fn, r; nmax=40)

        # Rin_norm = Binc_norm × Rdown + Bref_norm × Rup  (all normalized by Btrans)
        Rdown_exp = (Rin_val - amp.Bref * Rup_val) / amp.Binc

        rel_err = abs(Rdown_val - Rdown_exp) / abs(Rdown_exp)
        status = rel_err < tol ? "PASS" : "FAIL"

        if rel_err < tol
            global n_pass += 1
        else
            global n_fail += 1
        end

        @printf("%-30s | r=%4.1f | %14.6e %14.6e | %.2e  [%s]\n",
                label, r, abs(Rdown_val), abs(Rdown_exp), rel_err, status)

        if rel_err >= tol
            @printf("  Rdown:     %+.10e %+.10ei\n", real(Rdown_val), imag(Rdown_val))
            @printf("  Expected:  %+.10e %+.10ei\n", real(Rdown_exp), imag(Rdown_exp))
        end
    end
end

println("-"^90)
@printf("Results: %d PASS, %d FAIL out of %d tests (tol = %.0e)\n",
        n_pass, n_fail, n_pass + n_fail, tol)
