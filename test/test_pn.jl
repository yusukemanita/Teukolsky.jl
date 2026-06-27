using Test
using BHPtoolkit

# ============================================================
#  B6 — post-Newtonian (low-frequency) series
#  Scope: Schwarzschild (a=0), l≥1. ν / aₙ as ε=2ω series via the MST
#  recurrence run over the PNSeries ring; validated vs Wolfram pn_ref.txt
#  (Teukolsky`PN`) — at q=0 the q-dependent terms vanish, leaving exact
#  rational coefficients.
# ============================================================

@testset "B6 post-Newtonian series (Schwarzschild, l≥1)" begin
    # ν coefficients vs the Wolfram reference (q=0).
    refs = Dict(
        (-2, 2, 2) => (0 => 2.0, 2 => -107/210, 4 => -1695233/9261000),
        (-2, 2, 0) => (0 => 2.0, 2 => -107/210, 4 => -1695233/9261000),
        (-1, 1, 1) => (0 => 1.0, 2 => -47/60,   4 => -43908007/71064000),
    )
    @testset "ν coefficients s=$s l=$l m=$m" for ((s, l, m), cs) in refs
        ν = nu_pn(s, l, m, 0.0; order=4)
        for (k, c) in cs
            @test isapprox(real(getcoeff(ν, k)), c; atol=1e-13, rtol=1e-12)
            @test abs(imag(getcoeff(ν, k))) < 1e-13
        end
        # odd ε powers vanish at q=0
        @test abs(getcoeff(ν, 1)) < 1e-13
        @test abs(getcoeff(ν, 3)) < 1e-13
    end

    @testset "aₙ sanity + small-ω regression vs full solver" begin
        f = an_pn(-2, 2, 2, 0.0; order=4)
        @test real(getcoeff(f[0], 0)) ≈ 1.0            # a₀ = 1
        @test all(isfinite, [ComplexF64(getcoeff(f[1], k)) for k in 0:4])
        # PN series at small ω matches the validated numeric compute_nu to truncation.
        for (s, l, m) in [(-2, 2, 2), (-2, 3, 2), (-1, 1, 1)]
            ν  = nu_pn(s, l, m, 0.0; order=4); ω = 0.02
            vp = evalseries(ν, 2ω); vex, _ = compute_nu(s, l, m, 0.0, ω)
            @test min(abs(vp - vex), abs(vp - (-vex - 1))) < 1e-7   # O(ε⁵)
        end
    end

    @testset "unsupported cases error clearly" begin
        @test_throws ArgumentError nu_pn(-2, 2, 2, 0.5)   # Kerr a≠0
        @test_throws ArgumentError nu_pn(0, 0, 0, 0.0)    # l=0 monopole
        @test_throws ArgumentError an_pn(-2, 2, 2, 0.9)
    end
end
