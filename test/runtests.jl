using Test
using Teukolsky

@testset "Teukolsky" begin

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
        wp_odd = Teukolsky.Waveform.WaveformParams(
            -2, 2, 2, 0.0, 99, 6.0, -10.0, 10.0, 4, 0.1, false)
        @test_throws ArgumentError compute_waveform(wp_odd)

        # M3 / A3: f_n continued fractions now use convergence-checked Lentz
        # iteration (no fixed window / silent-zero). Any index converges to a
        # finite, nonzero ratio.
        ν, p = compute_nu(-2, 2, 2, 0.0, 0.3)
        @test isfinite(Teukolsky.Rn_cf(p, ν, 200)) && !iszero(Teukolsky.Rn_cf(p, ν, 200))
        @test isfinite(Teukolsky.Ln_cf(p, ν, -200))
        @test compute_fn(p, ν; nmax=160) isa Dict
    end

    @testset "A6 waveform precision" begin
        # Float64 waveform runs and ψ is real (G(-ω)=conj G(ω)) to rounding.
        wp = WaveformParams(s=-2, l=2, m=2, a=0.0, N=20, ω_max=2.0, Nt=4, verbose=false)
        _, ψ, _, _ = compute_waveform(wp)
        @test eltype(ψ) == ComplexF64
        @test all(isfinite, ψ)
        @test maximum(abs, imag.(ψ)) ≤ 1e-9 * maximum(abs, real.(ψ)) + 1e-30

        # BigFloat: the parametric type flows through to a Complex{BigFloat} waveform.
        wpb = WaveformParams(s=-2, l=2, m=2, a=big"0.0", N=8, ω_max=big"2.0", Nt=2, verbose=false)
        @test wpb isa WaveformParams{BigFloat}
        _, ψb, _, _ = setprecision(() -> compute_waveform(wpb), BigFloat, 128)
        @test eltype(ψb) == Complex{BigFloat}
        @test all(isfinite, ψb)
    end

end

# Quantitative cross-check against the Wolfram Teukolsky package.
include("test_wolfram_grid.jl")

# Forward-looking BigFloat precision gate for the Track-A refactor.
include("test_precision_bigfloat.jl")

# Track B: spheroidal harmonics (B2) and the callable radial object (B1).
include("test_spheroidal_radial.jl")

# Track B3: NumericalIntegration radial backend.
include("test_numint_radial.jl")

# Shared elliptic-integral / Jacobi module (geodesic prerequisite).
include("test_elliptic.jl")

# Kerr geodesic constants of motion (E, L, Q).
include("test_geodesics.jl")

# Track B6: post-Newtonian (low-frequency) series.
include("test_pn.jl")

# Track B5: point-particle source convolution + fluxes.
include("test_fluxes.jl")

# Arb backend validation for Binc/Bref/Rin/Rup at large complex frequency
# (self-consistency vs the BigFloat path + radial Wronskian).
include("test_arb_amplitudes.jl")

# |ω|-driven precision predictor: structure + calibration guard (predicted
# precision reproduces the branch-cut MST core).
include("test_precision_hint.jl")

# Native in-place Acb kernels (M3): f^ν_n, A^ν_± equivalence vs the generic
# path, the Arb-Lentz stall-exit regression (σ≳13.3 backward CF), and R^up vs
# a 700-bit reference.
include("test_native_acb.jl")

# PIA monodromy resonance (4σ ∈ ℤ ⇒ exact Γ·Poch 0·∞ in the factored form):
# compute_nu must return CF-residual-validated ν on the resonant grid.
include("test_pia_resonance.jl")

# Large-ω MST performance work: CF-ratio peeling ≡ per-n Lentz, the
# hypergeometric_U BigFloat→Arb bridge, and incremental Pochhammer weights.
include("test_mst_perf_opt.jl")
