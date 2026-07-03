# ============================================================
#  Regression: adaptive angular basis size (Issues A4 + A5) and the
#  branch-walk ambiguity diagnostics (Issue A6-edge).
#
#  A4(a): l beyond the old hard-wired l_max=20 used to throw a raw
#         BoundsError from _swsh_eigen (v_prev[il]).
#  A4(b): l == l_max used to run with a ZERO basis buffer and return λ
#         silently wrong by ~1.1e-2 — an explicit l_max is now a lower
#         bound and is auto-widened to adequacy.
#  A5   : l_max=20 hard-wired through the pipeline floored the λ
#         truncation error at ~1e-28 for aω=3.5 at 256 bits.  The
#         adaptive default (calibrated margin, see _swsh_lmax_margin)
#         keeps the truncation floor below the working precision, and
#         the SAME l_max is threaded through ν/amplitudes/λ/S in
#         TeukolskyRadial.
#
#  All references are computed in-test (self-arbitrating): higher
#  working precision (whose own truncation and rounding are both far
#  below the target's precision) — never pinned constants.
# ============================================================

using Test, Logging
using Teukolsky

@testset "adaptive l_max (A4+A5)" begin

    @testset "A4a: l beyond old default no longer errors, value arbitrated" begin
        # Old behavior: BoundsError.  New: valid λ, checked against a
        # 256-bit computation (independent of the Float64 eigen path:
        # BigFloat goes through Rayleigh-quotient refinement).
        λ64 = compute_lambda(-2, 25, 2, 0.7, 0.5)
        λbig = setprecision(BigFloat, 256) do
            compute_lambda(-2, 25, 2, BigFloat(7) / 10, BigFloat(1) / 2)
        end
        @test isfinite(λ64)
        @test abs(λ64 - ComplexF64(λbig)) / abs(λbig) < 1e-13
    end

    @testset "A4b: explicit l_max is a lower bound (auto-widened)" begin
        # l == l_max used to return λ off by ~1.1e-2.  Now any explicit
        # l_max must agree with a much larger basis to rounding level.
        λ20 = compute_lambda(-2, 20, 2, 0.7, 0.5; l_max=20)
        λ60 = compute_lambda(-2, 20, 2, 0.7, 0.5; l_max=60)
        @test abs(λ20 - λ60) / abs(λ60) < 1e-13
    end

    @testset "A5: truncation floor below working precision" begin
        # λ at prec bits (adaptive basis) vs λ at prec+128 bits: the
        # reference's truncation AND rounding are ≲ 2^-(prec+127), so the
        # difference measures the full error of the prec-bit run.  The old
        # hard-wired l_max=20 floor was ~1e-28 (aω = 3.5, 256 bits) —
        # ~3e49 times the working eps; the bound below is 64·eps.
        for (prec, cval) in ((256, 3.5), (256, 10.0), (512, 3.5))
            for c in (complex(cval), im * cval)
                λd = setprecision(BigFloat, prec) do
                    compute_lambda(-2, 2, 2, BigFloat(1), Complex{BigFloat}(c))
                end
                λr = setprecision(BigFloat, prec + 128) do
                    compute_lambda(-2, 2, 2, BigFloat(1), Complex{BigFloat}(c))
                end
                @test abs(λd - λr) / abs(λr) < 64 * big(2.0)^(-prec)
            end
        end
    end

    @testset "A5: one λ everywhere (l_max threading)" begin
        # TeukolskyRadial must use a single λ for ν, fn, amplitudes and p.
        tr = TeukolskyRadial(-2, 2, 2, 0.7, 0.5; l_max=40)
        ν40, p40 = compute_nu(-2, 2, 2, 0.7, 0.5; l_max=40)
        @test tr.λ == p40.λ
        @test tr.λ == compute_lambda(-2, 2, 2, 0.7, 0.5; l_max=40)
        @test tr.ν == ν40
        # Default (auto) path is self-consistent too.
        tra = TeukolskyRadial(-2, 2, 2, 0.7, 0.5)
        νa, pa = compute_nu(-2, 2, 2, 0.7, 0.5)
        @test tra.λ == pa.λ && tra.ν == νa
    end

    @testset "A4: clear error for invalid l (never a BoundsError)" begin
        @test_throws ArgumentError compute_lambda(-2, 1, 0, 0.7, 0.5)  # l < |s|
        @test_throws ArgumentError compute_lambda(0, 1, 2, 0.7, 0.5)   # l < |m|
    end

    @testset "A6-edge: branch walk is quiet on nominal parameters" begin
        # The ambiguous-accept path now warns; nominal calls (including the
        # formerly problematic σ≈4.19 crossing region and an exactly
        # degenerate large-c pair) must stay quiet AND finite.
        @test_logs min_level = Logging.Warn begin
            compute_lambda(-2, 2, 2, 0.7, 0.5)
            compute_lambda(-2, 2, 2, 0.7, 4.1885im)   # near λ-crossing (tracked)
            compute_lambda(-2, 4, 0, 1.0, 28.0)       # exactly degenerate pair
        end
        @test isfinite(compute_lambda(-2, 4, 0, 1.0, 28.0))
    end
end
