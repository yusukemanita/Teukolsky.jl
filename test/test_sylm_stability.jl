# ============================================================
#  Regression: sYlm high-l stability (Issue A8).
#
#  The alternating Goldberg sum loses ≈ 1.05·l bits to cancellation, so
#  the old direct Float64 evaluation was wrong by 4e-6 (l=40), O(1)
#  (l=60), unbounded (|Y| ~ 4e3 at l=80 vs the exact bound ~3.6) and
#  factorial-overflow NaN beyond l ≈ 120.  Float64-facing calls above
#  l = 8 now promote internally to BigFloat with l + 64 guard bits.
#
#  Arbiters (all computed in-test):
#   - BigFloat-512 evaluation of the same harmonic (the BigFloat path is
#     the unchanged type-generic code; at 512 bits its own cancellation
#     loss ≤ ~160 bits for l ≤ 150, leaving ≥ 350 accurate bits),
#   - the uniform bound |sYlm| ≤ sqrt((2l+1)/4π),
#   - Gauss–Legendre orthonormality of the l-basis.
# ============================================================

using Test, LinearAlgebra
using Teukolsky

@testset "sYlm high-l stability (A8)" begin
    θs = [0.05, 0.4, 0.9, 1.3, π / 2, 1.8, 2.3, 2.8, 3.09]
    cases = [(-2, 2), (-2, 0), (0, 0), (2, -3), (-2, -2), (1, 1), (0, 5), (-2, 17)]

    ybig(s, l, m, θ, d) = setprecision(BigFloat, 512) do
        sYlm(s, l, m, BigFloat(θ); deriv=d)
    end

    @testset "rel-err ≤ 1e-12 vs BigFloat-512 (l ≤ 120, deriv 0–2)" begin
        for l in (10, 25, 40, 60, 80, 100, 120)
            worst = 0.0
            for (s, m) in cases, θ in θs, d in (0, 1, 2)
                (l ≥ abs(s) && abs(m) ≤ l) || continue
                yf = sYlm(s, l, m, θ; deriv=d)
                yb = ComplexF64(ybig(s, l, m, θ, d))
                # skip near-zeros of the harmonic (rel-err is undefined
                # there in ANY fixed-precision arithmetic); the scale of a
                # deriv-d value is ~ l^d × the value bound
                abs(yb) > 1e-3 * sqrt((2l + 1) / (4π)) * max(1, l)^d || continue
                worst = max(worst, abs(ComplexF64(yf) - yb) / abs(yb))
            end
            @test worst ≤ 1e-12
        end
    end

    @testset "uniform bound and finiteness (old NaN region l ≥ 120)" begin
        for l in (60, 100, 150, 200)
            bound = sqrt((2l + 1) / (4π))
            for (s, m) in cases, θ in θs
                (l ≥ abs(s) && abs(m) ≤ l) || continue
                y = sYlm(s, l, m, θ)
                @test isfinite(y)
                @test abs(y) ≤ bound * (1 + 1e-10)
            end
        end
    end

    @testset "orthonormality (Gauss–Legendre, in-test nodes)" begin
        n = 400
        β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
        E = eigen(SymTridiagonal(zeros(n), β))
        xs = E.values
        ws = 2 .* (E.vectors[1, :] .^ 2)
        θn = acos.(clamp.(xs, -1, 1))
        for (s, m, l1, l2) in ((-2, 2, 60, 60), (-2, 2, 60, 62),
                               (0, 0, 100, 100), (0, 0, 100, 98),
                               (-2, -2, 80, 80))
            I = 0.0
            for (θ, w) in zip(θn, ws)
                I += w * real(conj(sYlm(s, l1, m, θ)) * sYlm(s, l2, m, θ))
            end
            I *= 2π
            @test abs(I - (l1 == l2 ? 1.0 : 0.0)) < 1e-10
        end
    end

    @testset "type-generic paths unchanged" begin
        # BigFloat input stays on the direct generic path and returns
        # Complex{BigFloat} at the ambient precision.
        y = setprecision(BigFloat, 256) do
            sYlm(-2, 30, 2, BigFloat(11) / 10)
        end
        @test y isa Complex{BigFloat}
        @test precision(real(y)) == 256
        # Small-l Float64 stays on the fast direct path and matches the
        # closed form ₂Y₂₂ = (1/2)√(5/π) cos⁴(θ/2) e^{2iφ} for s=-2.
        θ = 0.83
        @test sYlm(-2, 2, 2, θ) ≈ 0.5 * sqrt(5 / π) * cos(θ / 2)^4 rtol = 1e-13
    end
end
