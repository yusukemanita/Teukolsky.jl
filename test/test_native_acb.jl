# Native-Acb kernels (M3):
#   * f^ν_n and A^ν_± : native in-place Acb vs the generic Complex{Arb} path
#     (element-wise / value equivalence).
#   * R^up : the generic recurrence, whose confluent-U seeds now route through
#     Arb's rigorous acb_hypgeom_u for Complex{Arb} — verified against a 700-bit
#     reference (the OLD Kummer seed was off by ~1e14 at σ=4).
# Decoupled from compute_nu's precision fragility by using an ARBITRARY complex ν
# with a valid p — the recurrences/sums are well-defined for any ν and every path
# uses the SAME ν.
using Test
using Teukolsky
using Arblib: Arb, Acb

const _relC = (A, g) -> Float64(abs(Complex{BigFloat}(A) - Complex{BigFloat}(g)) /
                                max(abs(Complex{BigFloat}(g)), BigFloat("1e-300")))

@testset "native-Acb pipeline (M3)" begin
    cases = ((-2,2,2,0.5, Complex(0.5, 0.30)),
             (-2,3,2,2.0, Complex(1.7,-0.40)),
             (-2,2,2,4.0, Complex(1.5,-0.47)),
             ( 2,2,2,1.0, Complex(0.8, 0.20)),
             (-1,3,1,1.5, Complex(2.1, 0.90)))
    nmax = 60
    for (s,l,m,σ,ν0) in cases
        # High-precision (700-bit) R^up reference at the SAME nmax (isolating
        # precision from truncation), computed via the generic path — accurate at
        # 700 bit even where 300-bit generic fails.
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
            fa = Teukolsky.compute_fn_acb(p, ν; nmax=nmax)
            @testset "fn s=$s l=$l σ=$σ" begin
                @test Set(keys(fa)) == Set(keys(fg))
                @test maximum(_relC(fa[n], fg[n]) for n in -nmax:nmax) < 1e-70
            end

            Apg = Teukolsky.compute_Aplus(p, ν, fg; nmax=nmax)
            Amg = Teukolsky.compute_Aminus(p, ν, fg; nmax=nmax)
            Apa = Teukolsky.compute_Aplus_acb(p, ν, fa; nmax=nmax)
            Ama = Teukolsky.compute_Aminus_acb(p, ν, fa; nmax=nmax)
            @testset "A± s=$s l=$l σ=$σ" begin
                @test _relC(Apa, Apg) < 1e-70
                @test _relC(Ama, Amg) < 1e-70
            end

            # R^up (Arb) at a representative radius, shared Ctrans from A^ν_-.
            # With the acb_hypgeom_u seed dispatch this matches the 700-bit
            # reference to ≳40 digits (the Complex{Arb} recurrence loses ~half the
            # 300-bit mantissa at σ=4); the OLD Kummer seed gave ~1e14 error.
            ct = Teukolsky._ctrans(p, Amg)
            ra = Rup(p, ν, fa, Arb(10); nmax=nmax, ctrans=ct)
            @testset "Rup s=$s l=$l σ=$σ" begin
                @test isfinite(_relC(ra, ra))
                @test _relC(ra, ref_rup) < 1e-30
            end
        end
    end
end
