using Test
using BHPtoolkit
using LinearAlgebra: norm

# ============================================================
#  B2 — spin-weighted spheroidal harmonics
# ============================================================
@testset "B2 spin-weighted spheroidal harmonics" begin
    @testset "ₛYlm known values" begin
        for θ in (0.3, 1.0, 2.0)
            @test sYlm(0, 0, 0, θ) ≈ 1 / sqrt(4π)                  rtol=1e-13
            @test sYlm(0, 1, 0, θ) ≈ sqrt(3/(4π)) * cos(θ)         rtol=1e-13
            # ₋₂Y₂₂ = ½√(5/π) cos⁴(θ/2)
            @test sYlm(-2, 2, 2, θ) ≈ 0.5*sqrt(5/π)*cos(θ/2)^4     rtol=1e-13
        end
    end

    @testset "vs Wolfram SpinWeightedSpheroidalHarmonicS" begin
        ref = joinpath(@__DIR__, "swsh_ref.txt")
        for ln in readlines(ref)
            f = split(strip(ln), ";"); (isempty(strip(ln)) || f[6] == "ERR") && continue
            s = parse(Int, f[1]); l = parse(Int, f[2]); m = parse(Int, f[3])
            g = parse(Float64, f[4]); θ = parse(Float64, f[5])
            r = complex(parse(Float64, f[6]), parse(Float64, f[7]))
            jl = SpinWeightedSpheroidalHarmonicS(s, l, m, 1.0, g, θ)   # a=1 ⇒ aω=g
            @test isapprox(jl, r; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "normalization, c→0 limit, BigFloat" begin
        ells, C = swsh_coefficients(-2, 2, 2, 0.9, 0.5)
        @test norm(C) ≈ 1.0 rtol=1e-13                 # ∫|S|²dΩ = 1
        # a=0 reduces to the spin-weighted spherical harmonic
        @test SpinWeightedSpheroidalHarmonicS(-2, 3, 2, 0.0, 0.7, 1.1) ≈
              sYlm(-2, 3, 2, 1.1) rtol=1e-12
        # BigFloat harmonic flows through
        v = setprecision(BigFloat, 128) do
            SpinWeightedSpheroidalHarmonicS(-2, 2, 2, big"0.9", Complex{BigFloat}(big"0.5"),
                                            BigFloat(π) / 3)
        end
        @test v isa Complex{BigFloat}
    end
end

# ============================================================
#  B1 — callable TeukolskyRadial object
# ============================================================
@testset "B1 TeukolskyRadial object" begin
    s, l, m, a, ω = -2, 2, 2, 0.0, 0.5
    tr = TeukolskyRadial(s, l, m, a, ω)
    ν, p = compute_nu(s, l, m, a, ω); fn = compute_fn(p, ν)

    @testset "callable matches bare radial solutions" begin
        for r in (3.0, 5.0, 10.0)
            @test tr.In(r)            ≈ Rin(p, ν, fn, r)   rtol=1e-12
            @test tr.Up(r)            ≈ Rup(p, ν, fn, r)   rtol=1e-12
            @test tr.In(r; deriv=1)   ≈ dRin(p, ν, fn, r)  rtol=1e-12
            @test tr.Up(r; deriv=1)   ≈ dRup(p, ν, fn, r)  rtol=1e-12
        end
    end

    @testset "metadata and key access" begin
        @test tr.ν == tr.In["nu"] == tr.In["ν"]
        @test tr.λ == tr.In["lambda"]
        @test tr.In["BoundaryCondition"] == "In"
        @test tr.Up["BoundaryCondition"] == "Up"
        @test tr.In["s"] == s && tr.In["l"] == l && tr.In["m"] == m
        @test tr.amplitudes.Binc == tr.In["Amplitudes"].Binc
        @test_throws KeyError tr.In["nonsense"]
        @test_throws ArgumentError tr.In(10.0; deriv=2)
        @test tr.S(1.0) ≈ SpinWeightedSpheroidalHarmonicS(s, l, m, a, ω, 1.0) rtol=1e-12
    end

    @testset "input validation" begin
        @test_throws ArgumentError TeukolskyRadial(-2, 1, 0, 0.0, 0.5)   # l < |s|
        @test_throws ArgumentError TeukolskyRadial(-2, 2, 3, 0.0, 0.5)   # |m| > l
        @test_throws ArgumentError TeukolskyRadial(-2, 2, 2, 1.0, 0.5)   # |a| ≥ 1
    end

    @testset "BigFloat solution" begin
        trb = setprecision(BigFloat, 128) do
            TeukolskyRadial(-2, 2, 2, big"0.9", Complex{BigFloat}(big"0.5"))
        end
        val = trb.In(BigFloat(10))
        @test val isa Complex{BigFloat}
        @test isfinite(val)
    end
end
