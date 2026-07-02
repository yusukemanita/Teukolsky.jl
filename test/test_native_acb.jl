# Rigorous confluent-U for the Arb radial solution.
#
# The generic Rup recurrence seeds its HypergeometricU via `hypergeometric_U`,
# which for Complex{Arb} now routes through Arb's rigorous acb_hypgeom_u
# (src/hypergeometric.jl).  The old Kummer/asymptotic seed catastrophically
# cancels at large |z| — off by ~1e14 at σ=4 — and its near-integer-b Γ-pole
# guard NaNs.  Here we verify the Arb Rup (300 bit) matches a 700-bit reference
# across σ where the old seed failed.  Arbitrary complex ν (both paths share it),
# decoupled from compute_nu's PIA fragility.
using Test
using Teukolsky
using Arblib: Arb

const _relC = (A, g) -> Float64(abs(Complex{BigFloat}(A) - Complex{BigFloat}(g)) /
                                max(abs(Complex{BigFloat}(g)), BigFloat("1e-300")))

@testset "Arb Rup via rigorous acb_hypgeom_u" begin
    nmax = 60
    for (s,l,m,σ,ν0) in ((-2,2,2,0.5, Complex(0.5, 0.30)),
                         (-2,3,2,2.0, Complex(1.7,-0.40)),
                         (-2,2,2,4.0, Complex(1.5,-0.47)),   # old Kummer seed: ~1e14 off
                         ( 2,2,2,1.0, Complex(0.8, 0.20)),
                         (-1,3,1,1.5, Complex(2.1, 0.90)))
        ref_rup = setprecision(Arb, 700) do
            pr = MSTParams(s, l, m, Arb(7)/10, Complex{Arb}(Arb(0), Arb(σ)))
            νr = Complex{Arb}(Arb(real(ν0)), Arb(imag(ν0)))
            fr = compute_fn(pr, νr; nmax=nmax)
            ctr = Teukolsky._ctrans(pr, Teukolsky.compute_Aminus(pr, νr, fr; nmax=nmax))
            Complex{BigFloat}(Rup(pr, νr, fr, Arb(10); nmax=nmax, ctrans=ctr))
        end
        setprecision(Arb, 300) do
            p = MSTParams(s, l, m, Arb(7)/10, Complex{Arb}(Arb(0), Arb(σ)))
            ν = Complex{Arb}(Arb(real(ν0)), Arb(imag(ν0)))
            fg = compute_fn(p, ν; nmax=nmax)
            ct = Teukolsky._ctrans(p, Teukolsky.compute_Aminus(p, ν, fg; nmax=nmax))
            ra = Rup(p, ν, fg, Arb(10); nmax=nmax, ctrans=ct)
            @testset "Rup s=$s l=$l σ=$σ" begin
                @test isfinite(_relC(ra, ra))
                @test _relC(ra, ref_rup) < 1e-30     # ≥40 digits where old seed gave garbage
            end
        end
    end
end
