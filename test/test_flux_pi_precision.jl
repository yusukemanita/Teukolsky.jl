# ============================================================
#  Regression: full-precision π in the fluxes (Issue A7b).
#
#  _fluxes_s2 used `abs2(Z)/(4π ω²)` with typeof(4π) == Float64, which
#  floored BigFloat fluxes at ~3.9e-17 relative error.  Arbiter: an
#  independent rebuild of the same formulas with explicit BigFloat(π)
#  at a HIGHER precision (so a Float64-π regression on either side
#  cannot cancel), compared at 256 bits.
# ============================================================

using Test
using Teukolsky

@testset "flux π precision (A7b)" begin
    l, m = 2, 2

    # Independent exact-π rebuild of the s=-2 flux formulas (works at the
    # ambient BigFloat precision of its arguments).
    function fluxes_ref(l, m, a, ω, λ, ZI, ZH)
        πb = oftype(a, π)                       # exact π at working precision
        rh = 1 + sqrt(1 - a^2)
        Ωh = a / (2rh)
        κ = ω - m * Ωh
        ϵ = sqrt(1 - a^2) / (4rh)
        FInf = abs2(ZI) / (4πb * ω^2)
        AbsCSq = ((λ + 2)^2 + 4a * m * ω - 4a^2 * ω^2) *
                 (λ^2 + 36m * a * ω - 36a^2 * ω^2) +
                 (2λ + 3) * (96a^2 * ω^2 - 48m * a * ω) + 144 * ω^2 * (1 - a^2)
        α = (256 * (2rh)^5 * κ * (κ^2 + 4ϵ^2) * (κ^2 + 16ϵ^2) * ω^3) / AbsCSq
        FHor = α * abs2(ZH) / (4πb * ω^2)
        return (Inf=FInf, Hor=FHor)
    end

    # Rational inputs representable exactly at every precision.
    mk(prec) = setprecision(BigFloat, prec) do
        a = BigFloat(7) / 10
        ω = BigFloat(3) / 10
        λ = BigFloat(2)
        ZI = complex(BigFloat(11) / 10, BigFloat(3) / 10)
        ZH = complex(BigFloat(2) / 10, BigFloat(5) / 100)
        (a, ω, λ, ZI, ZH)
    end

    @testset "BigFloat-256 vs exact-π rebuild at 320 bits" begin
        F256 = setprecision(BigFloat, 256) do
            Teukolsky._fluxes_s2(l, m, mk(256)...)
        end
        Fref = setprecision(BigFloat, 320) do
            fluxes_ref(l, m, mk(320)...)
        end
        # Old Float64-4π floor: 3.9e-17.  New: rounding-level at 256 bits.
        @test abs(F256.Inf - Fref.Inf) / abs(Fref.Inf) < big"1e-70"
        @test abs(F256.Hor - Fref.Hor) / abs(Fref.Hor) < big"1e-70"
    end

    @testset "Float64 path agrees with BigFloat to rounding" begin
        a, ω, λv = 0.7, 0.3, 2.0
        ZI, ZH = 1.1 + 0.3im, 0.2 + 0.05im
        F64 = Teukolsky._fluxes_s2(l, m, a, ω, λv, ZI, ZH)
        Fbig = setprecision(BigFloat, 256) do
            fluxes_ref(l, m, mk(256)...)
        end
        @test abs(F64.Inf - Float64(Fbig.Inf)) / Float64(Fbig.Inf) < 1e-14
        @test abs(F64.Hor - Float64(Fbig.Hor)) / abs(Float64(Fbig.Hor)) < 1e-14
    end

    @testset "no Float64-π literals survive in teukolsky_mode.jl" begin
        src = read(joinpath(@__DIR__, "..", "src", "teukolsky_mode.jl"), String)
        code = join([split(line, '#')[1] for line in split(src, '\n')], "\n")
        # any bare `4π`-style Irrational-times-Int literal is a regression
        @test !occursin(r"[0-9]π", code)
        @test !occursin(r"π\s*\*\s*im", code)
    end
end
