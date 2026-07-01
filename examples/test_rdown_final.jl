using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")
using Teukolsky
using Printf

println("="^80)
println("Rdown verification: Wronskian constancy + normalization check")
println("="^80)

test_cases = [
    # (s, l, m, a, ω, label)
    (-2, 2, 2, 0.0, 0.3, "Schwarzschild ω=0.3"),
    (-2, 2, 2, 0.0, 0.5, "Schwarzschild ω=0.5"),
    (-2, 2, 2, 0.5, 0.3, "Kerr a=0.5 ω=0.3"),
    (-2, 2, 2, 0.9, 0.3, "Kerr a=0.9 ω=0.3"),
    (-2, 2, 2, 0.0, 0.3-0.1im, "Schwarzschild ω=0.3-0.1i"),
    (-2, 2, 2, 0.5, 0.3-0.1im, "Kerr a=0.5 ω=0.3-0.1i"),
]

δr = 1e-6
r_test = [4.0, 6.0, 10.0, 15.0]

for (s, l, m, a, ω, label) in test_cases
    println("\n--- $label ---")

    ν, p = compute_nu(s, l, m, a, ω)
    fn = compute_fn(p, ν; nmax=40)
    amp = compute_amplitudes(s, l, m, a, ω; nmax=40)

    Ap = amp.Ap
    ω_c = p.ω
    ϵ = p.ϵ
    κ = p.κ
    phase_norm = exp(-im * (ϵ * log(ϵ) - (1 - κ) / 2 * ϵ))
    norm_val = Ap * ω_c^(-1) * phase_norm

    rp = p.rp
    r_min = rp + 0.5

    # Compute W[R+, R-]/Δ at several r and check constancy
    w_vals = ComplexF64[]
    for r in r_test
        r < r_min && continue
        Δ = r^2 - 2r + a^2
        Rp = Rdown(p, ν, fn, r) * norm_val
        Rm = Rup(p, ν, fn, r)
        dRp = (Rdown(p, ν, fn, r+δr) - Rdown(p, ν, fn, r-δr)) / (2δr) * norm_val
        dRm = (Rup(p, ν, fn, r+δr) - Rup(p, ν, fn, r-δr)) / (2δr)
        W = (Rp * dRm - Rm * dRp) / Δ
        push!(w_vals, W)
        @printf("  r=%5.1f  W[R+,R-]/Δ = %+.8e %+.8ei\n", r, real(W), imag(W))
    end

    # Check constancy
    if length(w_vals) >= 2
        max_var = maximum(abs(w - w_vals[1]) / abs(w_vals[1]) for w in w_vals[2:end])
        @printf("  Max variation: %.2e  %s\n", max_var, max_var < 1e-4 ? "PASS" : "FAIL")
    end

    # Check c1 = Binc/norm
    r = 6.0
    Δ = r^2 - 2r + a^2
    Rin_val = Rin(p, ν, fn, r)
    Rup_val = Rup(p, ν, fn, r)
    Rp_raw = Rdown(p, ν, fn, r) * norm_val
    dRin = (Rin(p, ν, fn, r+δr) - Rin(p, ν, fn, r-δr)) / (2δr)
    dRup = (Rup(p, ν, fn, r+δr) - Rup(p, ν, fn, r-δr)) / (2δr)
    dRp_raw = (Rdown(p, ν, fn, r+δr) - Rdown(p, ν, fn, r-δr)) / (2δr) * norm_val
    W_in_up = Rin_val * dRup - Rup_val * dRin
    W_rp_up = Rp_raw * dRup - Rup_val * dRp_raw
    c1 = W_in_up / W_rp_up
    c1_expected = amp.Binc / norm_val
    rel_err_c1 = abs(c1 - c1_expected) / abs(c1_expected)
    @printf("  Binc/norm check: rel_err = %.2e  %s\n", rel_err_c1, rel_err_c1 < 1e-4 ? "PASS" : "FAIL")
end

println("\n" * "="^80)
println("All tests complete.")
