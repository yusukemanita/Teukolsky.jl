using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")
using BHPtoolkit
using Printf

# Reference from qnm Python package (M=1 units, same convention)
const QNM_REF = [
    0.67161427 - 0.06486924im,
    0.66765755 - 0.19525207im,
    0.65982668 - 0.32751838im,
    0.64786903 - 0.46286425im,
    0.62983606 - 0.60329505im,
    0.53691672 - 0.74872346im,
    0.60222748 - 0.77033698im,
    0.61879884 - 0.93317832im,
]

s, l, m, a = -2, 2, 2, 0.9

println("="^90)
println("Comparing MST Binc zeros with qnm package reference (s=$s, l=$l, m=$m, a=$a)")
println("="^90)

# ── Test 1: |Binc| at reference frequencies ──────────────────────────────────
println("\n── Test 1: |Binc(ω_ref)| at qnm reference frequencies ──")
println(@sprintf("%-4s  %-26s  %-12s  %-12s  %-12s",
        "n", "ω_ref", "|Binc| F64", "|Binc| 128", "pass?"))
println("-"^80)

for (n, ωref) in enumerate(QNM_REF)
    n -= 1
    b64  = abs(compute_amplitudes(s, l, m, a, ωref).Binc)
    b128 = Float64(abs(compute_amplitudes(s, l, m, a, ωref; precision=128).Binc))
    pass = b128 < 1e-4 ? "✓" : "✗ NOT A ZERO"
    @printf("n=%-2d  (%+.6f%+.6fim)  %10.3e  %10.3e  %s\n",
            n, real(ωref), imag(ωref), b64, b128, pass)
end

# ── Test 2: compute_nu at reference frequencies ───────────────────────────────
println("\n── Test 2: compute_nu ν at reference frequencies ──")
println(@sprintf("%-4s  %-26s  %-30s  %-12s",
        "n", "ω_ref", "ν (F64)", "|g0(ν)|"))
println("-"^90)

for (n, ωref) in enumerate(QNM_REF)
    n -= 1
    try
        ν, p = compute_nu(s, l, m, a, ωref)
        # Evaluate residual of the CF equation
        R1  = BHPtoolkit.Rn_cf(p, ν, 1)
        Lm1 = BHPtoolkit.Ln_cf(p, ν, -1)
        g0  = BHPtoolkit.βn(p, ν, 0) + BHPtoolkit.αn(p, ν, 0)*R1 + BHPtoolkit.γn(p, ν, 0)*Lm1
        @printf("n=%-2d  (%+.6f%+.6fim)  ν=(%+.6f%+.6fim)  |g0|=%.2e\n",
                n, real(ωref), imag(ωref), real(ν), imag(ν), abs(g0))
    catch e
        @printf("n=%-2d  ERROR: %s\n", n, e)
    end
end

# ── Test 3: Newton refinement of Binc=0 ──────────────────────────────────────
println("\n── Test 3: Newton refinement of Binc(ω)=0 starting from qnm ref ──")
println(@sprintf("%-4s  %-26s  %-30s  %-12s  %-10s",
        "n", "ω_ref", "ω_refined (128bit)", "|Binc|", "Δω"))
println("-"^100)

function refine_binc_zero(ω0; prec=128, maxiter=50, tol=1e-25)
    setprecision(BigFloat, prec) do
        ω = Complex{BigFloat}(ω0)
        δ = BigFloat("1e-7")
        for _ in 1:maxiter
            f  = compute_amplitudes(s, l, m, a, ω; precision=prec).Binc
            fp = (compute_amplitudes(s, l, m, a, ω+δ; precision=prec).Binc -
                  compute_amplitudes(s, l, m, a, ω-δ; precision=prec).Binc) / (2δ)
            abs(fp) < 1e-30 && break
            Δ  = -f / fp
            ω += Δ
            Float64(abs(Δ)) < Float64(tol) && break
        end
        return ω
    end
end

for (n, ωref) in enumerate(QNM_REF)
    n -= 1
    try
        ω_refined = refine_binc_zero(ωref)
        b_check   = Float64(abs(compute_amplitudes(s, l, m, a, ω_refined; precision=128).Binc))
        Δω        = abs(Float64(ω_refined) - ωref)
        pass      = b_check < 1e-10 ? "✓" : "✗"
        @printf("n=%-2d  (%+.6f%+.6fim)  →(%+.6f%+.6fim)  %.2e  %.2e  %s\n",
                n, real(ωref), imag(ωref),
                Float64(real(ω_refined)), Float64(imag(ω_refined)),
                b_check, Δω, pass)
    catch e
        @printf("n=%-2d  ERROR: %s\n", n, e)
    end
end
