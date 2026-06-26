using Test
using BHPtoolkit

@testset "BHPtoolkit" begin

    @testset "ν computation — Schwarzschild s=-2, l=m=2" begin
        s, l, m, a = -2, 2, 2, 0.0
        ω = 0.3
        ν, p = compute_nu(s, l, m, a, ω)
        @test isfinite(ν)
        @test abs(imag(ν)) < 0.1  # should be nearly real for small ω
    end

    @testset "Amplitudes — Schwarzschild" begin
        s, l, m, a = -2, 2, 2, 0.0
        ω = 0.3
        result = compute_amplitudes(s, l, m, a, ω)
        @test isfinite(result.Binc)
        @test isfinite(result.Bref)
        @test isfinite(result.Btrans)
        @test isfinite(result.Ctrans)

        # Binc is normalized by Btrans; 2iω Binc should be O(1)
        W = 2im * ω * result.Binc
        @test isfinite(W)
        @test abs(W) > 1e-10
    end

    @testset "ν scan — Sasaki-Tagoshi Table 1" begin
        s, l, m, a = -2, 2, 2, 0.0
        for Mω in [1.0, 2.0, 3.0]
            ν, _ = compute_nu(s, l, m, a, Mω)
            @test isfinite(ν)
        end
    end

    @testset "Amplitudes — Kerr a=0.9" begin
        s, l, m = -2, 2, 2
        a = 0.9
        for Mω in [0.1, 0.2, 0.3]
            result = compute_amplitudes(s, l, m, a, Mω)
            @test isfinite(result.Binc)
            @test isfinite(result.Btrans)
        end
    end

    @testset "Meromorphic amplitudes" begin
        s, l, m, a = -2, 2, 2, 0.0
        ω = 0.3
        result = compute_amplitudes_mero(s, l, m, a, ω)
        @test isfinite(result.Binc)
        @test isfinite(result.Btrans)
    end

    @testset "Rin — basic evaluation" begin
        s, l, m, a = -2, 2, 2, 0.0
        ω = 0.3
        ν, p = compute_nu(s, l, m, a, ω)
        fn = compute_fn(p, ν)

        for r in [3.0, 5.0, 10.0, 50.0]
            val = Rin(p, ν, fn, r)
            @test isfinite(val)
        end
    end

    @testset "dRin — basic evaluation" begin
        s, l, m, a = -2, 2, 2, 0.0
        ω = 0.3
        ν, p = compute_nu(s, l, m, a, ω)
        fn = compute_fn(p, ν)

        for r in [3.0, 5.0, 10.0]
            val = dRin(p, ν, fn, r)
            @test isfinite(val)
        end
    end

    @testset "Rup — basic evaluation" begin
        s, l, m, a = -2, 2, 2, 0.0
        ω = 0.3
        ν, p = compute_nu(s, l, m, a, ω)
        fn = compute_fn(p, ν)

        for r in [3.0, 5.0, 10.0, 50.0]
            val = Rup(p, ν, fn, r)
            @test isfinite(val)
        end
    end

    @testset "dRup — basic evaluation" begin
        s, l, m, a = -2, 2, 2, 0.0
        ω = 0.3
        ν, p = compute_nu(s, l, m, a, ω)
        fn = compute_fn(p, ν)

        for r in [3.0, 5.0, 10.0]
            val = dRup(p, ν, fn, r)
            @test isfinite(val)
        end
    end

    @testset "Wronskian constancy — Schwarzschild" begin
        s, l, m, a = -2, 2, 2, 0.0
        ω = 0.3
        ν, p = compute_nu(s, l, m, a, ω)
        fn = compute_fn(p, ν)

        # W(r) = Δ^{s+1} (Rin dRup - Rup dRin) should be r-independent
        # (Abel identity for Teukolsky equation)
        # Use moderate r values where both series converge well
        rs = [4.0, 5.0, 8.0]
        Ws = ComplexF64[]
        for r in rs
            Δ = r^2 - 2r + a^2
            rin = Rin(p, ν, fn, r)
            drin = dRin(p, ν, fn, r)
            rup = Rup(p, ν, fn, r)
            drup = dRup(p, ν, fn, r)
            W = Δ^(s+1) * (rin * drup - rup * drin)
            push!(Ws, W)
        end

        for i in 2:length(Ws)
            rel_diff = abs(Ws[i] - Ws[1]) / abs(Ws[1])
            @test rel_diff < 1e-6
        end
    end

    @testset "Wronskian constancy — Kerr a=0.9" begin
        s, l, m = -2, 2, 2
        a = 0.9
        ω = 0.3
        ν, p = compute_nu(s, l, m, a, ω)
        fn = compute_fn(p, ν)

        rs = [3.0, 5.0, 10.0]
        Ws = ComplexF64[]
        for r in rs
            Δ = r^2 - 2r + a^2
            rin = Rin(p, ν, fn, r)
            drin = dRin(p, ν, fn, r)
            rup = Rup(p, ν, fn, r)
            drup = dRup(p, ν, fn, r)
            W = Δ^(s+1) * (rin * drup - rup * drin)
            push!(Ws, W)
        end

        for i in 2:length(Ws)
            rel_diff = abs(Ws[i] - Ws[1]) / abs(Ws[1])
            @test rel_diff < 1e-6
        end
    end

    @testset "Numerical derivative consistency" begin
        s, l, m, a = -2, 2, 2, 0.0
        ω = 0.3
        ν, p = compute_nu(s, l, m, a, ω)
        fn = compute_fn(p, ν)
        r = 3.0; h = 1e-6

        drin_num = (Rin(p, ν, fn, r+h) - Rin(p, ν, fn, r-h)) / (2h)
        drin_ana = dRin(p, ν, fn, r)
        @test abs(drin_num - drin_ana) / abs(drin_ana) < 1e-6

        drup_num = (Rup(p, ν, fn, r+h) - Rup(p, ν, fn, r-h)) / (2h)
        drup_ana = dRup(p, ν, fn, r)
        @test abs(drup_num - drup_ana) / abs(drup_ana) < 1e-6
    end

    @testset "A0 bug-fix regressions" begin
        # H7: even N enforced for the G(-ω)=conj G(ω) waveform grid mirror
        wp = WaveformParams(s=-2, l=2, m=2, a=0.0, N=99, Nt=4, verbose=false)
        @test iseven(wp.N)                       # constructor rounds odd N up (99→100)
        wp_odd = BHPtoolkit.Waveform.WaveformParams(
            -2, 2, 2, 0.0, 99, 6.0, -10.0, 10.0, 4, 0.1, false)
        @test_throws ArgumentError compute_waveform(wp_odd)

        # M3: f_n continued-fraction window guard — out-of-window index errors
        # instead of silently returning 0.
        ν, p = compute_nu(-2, 2, 2, 0.0, 0.3)
        @test_throws ArgumentError BHPtoolkit.Rn_cf(p, ν, 200; nmax=150)
        @test_throws ArgumentError BHPtoolkit.Ln_cf(p, ν, -200; nmax=150)
        # compute_fn auto-sizes the window so a large nmax never hits the guard.
        @test compute_fn(p, ν; nmax=160) isa Dict
    end

end

# Quantitative cross-check against the Wolfram Teukolsky package.
include("test_wolfram_grid.jl")

# Forward-looking BigFloat precision gate for the Track-A refactor.
include("test_precision_bigfloat.jl")
