# ============================================================
#  Issue R14a — degenerate-input robustness in hypergeometric.jl
#
#  (a) escalation seeds: Arblib.rel_accuracy_bits returns ±(2^63−1) for
#      degenerate balls (+typemax for exact zero/NaN/Inf midpoints,
#      −typemax for zero/non-finite midpoints with finite radius).  The raw
#      value either falsely certified a non-finite ball or overflowed
#      `p + (need − acc)` into a NEGATIVE precision → crash on the next
#      Acb(0; prec=p).  `_seed_acc_bits` clamps to [−2^20, p−4] and treats
#      non-finite balls as no-accuracy.
#  (b) generic hypergeometric_U: round(Int, real(b)) threw InexactError for
#      NaN (and Inf / huge) b — now propagates NaN.
# ============================================================

using Test
using Teukolsky
using Arblib: Arb, Acb
import Arblib

@testset "R14a: degenerate-input robustness" begin
    @testset "(a) _seed_acc_bits clamps degenerate rel_accuracy_bits" begin
        setprecision(Arb, 256) do
            p = 512
            # exact NaN ball: rel_accuracy_bits = +typemax → must NOT certify
            @test Teukolsky._seed_acc_bits(Acb(NaN), p) == -(1 << 20)
            # non-finite midpoint with radius: −typemax → clamped, no overflow
            z = Arb(NaN); Arblib.add_error!(z, Arb(1))
            @test Teukolsky._seed_acc_bits(Acb(z, Arb(0)), p) == -(1 << 20)
            # zero midpoint with radius: −typemax → clamped
            z0 = Arb(0); Arblib.add_error!(z0, Arb(1))
            @test Teukolsky._seed_acc_bits(Acb(z0, Arb(0)), p) == -(1 << 20)
            # exact zero: +typemax → certified at the p−4 cap (an exact zero
            # IS exact)
            @test Teukolsky._seed_acc_bits(Acb(0), p) == p - 4
            # sane ball: passes through (clamped only at p−4)
            @test Teukolsky._seed_acc_bits(Acb(1), p) == p - 4
        end
    end

    @testset "(a) escalation seeds survive a forced-degenerate ball" begin
        setprecision(Arb, 256) do
            # NaN argument c ⇒ every hypgeom_u evaluation yields a degenerate
            # ball ⇒ pre-fix: Int overflow → negative precision → crash
            # (and ceil(Int, NaN) in _u_loss_estimate before that).
            hp = Teukolsky.HUParams(Complex{Arb}(Arb(1), Arb(0)),
                                    Complex{Arb}(Arb(2), Arb(0)),
                                    Complex{Arb}(Arb(NaN), Arb(0)))
            H, acc = Teukolsky._hu_seed_acb(hp, 0, 240)
            @test acc ≥ 1                       # sanitized, no overflow
            @test !isfinite(H)                  # NaN propagates VISIBLY
            D, dacc = Teukolsky._dhu_seed_acb(hp, 0, 240, nothing)
            @test dacc ≥ 1
            @test !isfinite(D)
        end
    end

    @testset "(b) hypergeometric_U propagates NaN instead of throwing" begin
        # verified pre-fix repro: InexactError from round(Int, NaN)
        v = Teukolsky.hypergeometric_U(1.0 + 0im, NaN + 0im, 1.0 + 0im)
        @test isnan(real(v)) && isnan(imag(v))
        # Inf b and NaN a / z as well
        @test isnan(real(Teukolsky.hypergeometric_U(1.0 + 0im, Inf + 0im, 1.0 + 0im)))
        @test isnan(real(Teukolsky.hypergeometric_U(NaN + 0im, 2.0 + 0im, 1.0 + 0im)))
        @test isnan(real(Teukolsky.hypergeometric_U(1.0 + 0im, 2.0 + 0im, NaN + 0im)))
        # huge (Int-overflowing) real(b) must not throw either
        @test Teukolsky.hypergeometric_U(1.0 + 0im, 1e300 + 0im, 1.0 + 0im) isa Complex
        # sane inputs still work (vs Arb rigorous route)
        u = Teukolsky.hypergeometric_U(1.5 + 0.5im, 2.25 + 0im, 3.0 + 1.0im)
        uref = setprecision(Arb, 128) do
            ComplexF64(Complex{BigFloat}(Teukolsky.hypergeometric_U(
                Complex{Arb}(Arb("1.5"), Arb("0.5")),
                Complex{Arb}(Arb("2.25"), Arb(0)),
                Complex{Arb}(Arb(3), Arb(1)))))
        end
        @test abs(u - uref) / abs(uref) < 1e-10
    end
end
