# ============================================================
#  MultiFloat precision-backend tests
#
#  Verifies the three-way precision option (:float64 / :bigfloat / :multifloat)
#  on compute_nu and compute_amplitudes:
#    - all backends run and return the expected float types;
#    - the MultiFloat result matches a high-precision BigFloat REFERENCE to a
#      relative accuracy consistent with the MultiFloat working precision
#      (≈ N·53 bits), i.e. MultiFloat genuinely beats Float64;
#    - the default (:auto / :bigfloat) paths are unchanged (backward compat).
# ============================================================

using Test
using Teukolsky
using MultiFloats: MultiFloat, Float64x2, Float64x4

# relative error of a complex value vs a BigFloat reference
relerr(x, ref) = abs(ComplexF64((Complex{BigFloat}(x) - ref) / ref))

@testset "MultiFloat precision backend" begin
    s, l, m = -2, 2, 2
    # (a, ω) grid: Schwarzschild + Kerr, real ω.  Branch coverage:
    #   (0.0,0.5) half-integer ν,  (0.7,0.4) integer ν,  (0.5,0.3) real ν.
    modes = [(0.0, 0.5), (0.7, 0.4), (0.5, 0.3)]

    @testset "type plumbing" begin
        νf, _ = compute_nu(s, l, m, 0.5, 0.3; backend=:float64)
        νb, _ = compute_nu(s, l, m, 0.5, 0.3; backend=:bigfloat,   precision=128)
        ν2, _ = compute_nu(s, l, m, 0.5, 0.3; backend=:multifloat, precision=80)
        ν4, _ = compute_nu(s, l, m, 0.5, 0.3; backend=:multifloat, precision=200)
        @test νf isa Complex{Float64}
        @test νb isa Complex{BigFloat}
        @test ν2 isa Complex{Float64x2}    # ⌈80/53⌉  = 2 limbs
        @test ν4 isa Complex{Float64x4}    # ⌈200/53⌉ = 4 limbs
    end

    @testset "ν matches BigFloat reference" begin
        for (a, ω) in modes
            begin
                refhi, _ = compute_nu(s, l, m, a, ω; backend=:bigfloat, precision=512)
                ν2, _ = compute_nu(s, l, m, a, ω; backend=:multifloat, precision=106)
                ν4, _ = compute_nu(s, l, m, a, ω; backend=:multifloat, precision=212)
                νf, _ = compute_nu(s, l, m, a, ω; backend=:float64)
                # Float64x2 ≈ 106 bits, Float64x4 ≈ 212 bits.  Thresholds allow a
                # margin for acos/acosh branch-point conditioning in the ν formula
                # (mode-dependent, inherent — same loss BigFloat would see).
                @test relerr(ν2, Complex{BigFloat}(refhi)) < 1e-24
                @test relerr(ν4, Complex{BigFloat}(refhi)) < 1e-52
                # MultiFloat must beat plain Float64
                @test relerr(ν4, Complex{BigFloat}(refhi)) < relerr(νf, Complex{BigFloat}(refhi))
            end
        end
    end

    @testset "amplitudes match BigFloat reference" begin
        for (a, ω) in modes
            refhi = compute_amplitudes(s, l, m, a, ω; backend=:bigfloat, precision=512)
            A2 = compute_amplitudes(s, l, m, a, ω; backend=:multifloat, precision=106)
            A4 = compute_amplitudes(s, l, m, a, ω; backend=:multifloat, precision=212)
            for f in (:Binc, :Bref, :Btrans, :Ctrans)
                r = Complex{BigFloat}(getfield(refhi, f))
                @test relerr(getfield(A2, f), r) < 1e-24
                @test relerr(getfield(A4, f), r) < 1e-52
            end
        end
    end

    @testset "backward compatibility (default paths unchanged)" begin
        # :auto default must equal an explicit Float64 input call
        A_default = compute_amplitudes(s, l, m, 0.5, 0.3)
        A_f64     = compute_amplitudes(s, l, m, 0.5, 0.3; backend=:float64)
        @test A_default.Binc ≈ A_f64.Binc
        # nufixed + mero variants accept the backend kw and run in MultiFloat
        νfix = 2.0
        Anf = compute_amplitudes_nufixed(s, l, m, 0.5, 0.3, νfix; backend=:multifloat, precision=106)
        @test Anf.Binc isa Complex{Float64x2}
        Am = compute_amplitudes_mero(s, l, m, 0.5, 0.3; backend=:multifloat, precision=106)
        @test Am.Binc isa Complex{Float64x2}
    end

    @testset "compat shims" begin
        # 2-argument atan (the gap use_bigfloat_transcendentals misses)
        y, x = Float64x4(1.0), Float64x4(-1.0)
        @test Float64(atan(y, x)) ≈ atan(1.0, -1.0)
        # complex log / acos / acosh on Complex{MultiFloat} (depend on 2-arg atan)
        z = Complex{Float64x4}(Float64x4(1.3), Float64x4(0.7))
        @test Float64(real(log(z)))   ≈ real(log(ComplexF64(z)))
        @test Float64(real(acos(z)))  ≈ real(acos(ComplexF64(z)))
        @test Float64(real(acosh(z))) ≈ real(acosh(ComplexF64(z)))
    end
end
