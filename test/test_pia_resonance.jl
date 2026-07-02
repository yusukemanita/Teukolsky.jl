# PIA monodromy resonance regression (compute_nu on the positive imaginary axis).
#
# For purely-imaginary ω = iσ the monodromy parameter μd = μ1C − μ2C = −2s − 4σ
# is REAL and cancels to an EXACT integer whenever 4σ ∈ ℤ (independent of
# s, l, m, a).  There the factored Γ(±μd)·Pochhammer closed form of cos(2πν) is
# an exact 0·∞: depending on precision it returned NaN or — worse — FINITE
# GARBAGE (e.g. σ=2, 300 bits gave a ν whose continued-fraction residual was
# O(10)).  The resonance gate now routes these through the pole-free marching-Γ
# form (_monodromy_value_safe), which is the SAME expression with Γ(z)(z)_k
# recombined into Γ(z+k).
#
# Validation: ν must satisfy the MST three-term continued-fraction equation
# g(ν) = β₀ + α₀·R₁ + γ₀·L₋₁ = 0 to ~working precision — an arbiter fully
# independent of the monodromy series.
using Test
using Teukolsky
using Arblib: Arb

@testset "PIA resonance (4σ ∈ ℤ) in compute_nu" begin
    s, l, m = -2, 2, 2
    a = 7//10

    # residual of the CF equation at ν (computed at 300 bits regardless of the
    # precision ν was solved at — the CF ratios are trustworthy at these σ)
    function cf_residual(ν, σq)
        setprecision(BigFloat, 300) do
            νb = Complex{BigFloat}(ν)
            pb = MSTParams(s, l, m, BigFloat(7)/10,
                           Complex{BigFloat}(0, BigFloat(numerator(σq))/denominator(σq)))
            R1  = Teukolsky.Rn_cf(pb, νb, 1;  nmax=600)
            Lm1 = Teukolsky.Ln_cf(pb, νb, -1; nmax=600)
            Float64(abs(Teukolsky.βn(pb, νb, 0) + Teukolsky.αn(pb, νb, 0)*R1 +
                        Teukolsky.γn(pb, νb, 0)*Lm1))
        end
    end

    @testset "resonant grid, BigFloat backend" begin
        for σq in (1//4, 1//2, 3//4, 1//1, 3//2, 2//1, 3//1)
            for prec in (128, 256)
                ν, p = compute_nu(s, l, m, a, im*big(numerator(σq))/denominator(σq);
                                  precision=prec)
                @test isfinite(Float64(real(ν))) && isfinite(Float64(imag(ν)))
                # residual scales ~2^-prec; generous ceilings per precision
                @test cf_residual(ν, σq) < (prec == 128 ? 1e-25 : 1e-60)
            end
        end
    end

    @testset "resonant grid, :acb backend" begin
        for σq in (1//2, 1//1, 2//1)
            ν, _ = compute_nu(s, l, m, a, im*big(numerator(σq))/denominator(σq);
                              backend=:acb, precision=256)
            @test isfinite(Float64(real(ν))) && isfinite(Float64(imag(ν)))
            @test cf_residual(Complex{BigFloat}(ν), σq) < 1e-60
        end
    end

    @testset "near/off-resonance unchanged" begin
        for σ in (0.5001, 0.6, 1.37)
            ν, _ = compute_nu(s, l, m, a, im*big(σ); precision=256)
            @test isfinite(Float64(real(ν)))
            r = setprecision(BigFloat, 300) do
                νb = Complex{BigFloat}(ν)
                pb = MSTParams(s, l, m, BigFloat(7)/10, Complex{BigFloat}(0, BigFloat(σ)))
                R1  = Teukolsky.Rn_cf(pb, νb, 1;  nmax=600)
                Lm1 = Teukolsky.Ln_cf(pb, νb, -1; nmax=600)
                Float64(abs(Teukolsky.βn(pb, νb, 0) + Teukolsky.αn(pb, νb, 0)*R1 +
                            Teukolsky.γn(pb, νb, 0)*Lm1))
            end
            @test r < 1e-60
        end
    end

    @testset "real/complex ω regression" begin
        # validated regime — values must stay on the established branches
        ν1, _ = compute_nu(s, l, m, a, big"0.3"; precision=256)
        @test abs(Float64(real(ν1)) - 1.816016744299) < 1e-9   # real branch
        @test abs(Float64(imag(ν1))) < 1e-20
        ν2, _ = compute_nu(s, l, m, a, big"1.1"; precision=256)
        @test abs(Float64(real(ν2)) - 0.5) < 1e-20             # half-integer branch
    end
end
