# ============================================================
#  Issue R1 — Rin/dRin converge-or-error at moderate/large r
#  Issue R6 — decidable recurrence-instability guards on Arb
#
#  R1 background: the horizon-side 2F1 series terms GROW with n (peak
#  n ≈ 2.3·|x|, x = (r₊−r)/2κ) before the super-exponential decay of f_n
#  takes over, and the tail then CANCELS the peak (1.4e14× at r=30 for
#  s=-2 l=m=2 a=0.7 ω=0.8).  The old fixed-nmax loops exhausted n=80 with
#  no convergence check and returned silent garbage (rel-err 2.3e15 at
#  r=30, 256-bit; the Float64 path agreed with the wrong value to 2.6e-11).
#
#  Arbiters (algorithm-independent):
#   * The radial Teukolsky ODE residual
#       Δ R'' + 2(s+1)(r−1) R' + [(K²−2is(r−1)K)/Δ + 4isωr − λ] R = 0
#     with K = (r²+a²)ω − am, λ = p.λ — built here from exact integer/
#     parameter arithmetic only (no π, no same-code reuse).
#   * A deep reference at 2× precision (which is itself convergence-checked
#     and cancellation-certified by construction after this fix).
# ============================================================

using Test
using Teukolsky
using Arblib: Arb
import Arblib

# Relative ODE residual of a radial solution (R, R′) at r, with R″ from a
# central difference of the ANALYTIC derivative dR (step h).  Scale-free:
# residual is normalized by the magnitude sum of the operator terms.
function _teuk_ode_residual(p, R0, R1, dRfun, r, h)
    a, ω, s, m, λ = p.a, p.ω, p.s, p.m, p.λ
    R2 = (dRfun(r + h) - dRfun(r - h)) / (2h)
    Δ = r^2 - 2r + a^2
    K = (r^2 + a^2) * ω - a * m
    V = (K^2 - 2im * s * (r - 1) * K) / Δ + 4im * s * ω * r - λ
    res = Δ * R2 + 2 * (s + 1) * (r - 1) * R1 + V * R0
    scale = abs(Δ * R2) + abs(2 * (s + 1) * (r - 1) * R1) + abs(V * R0)
    return Float64(abs(res) / scale)
end

@testset "R1: Rin/dRin converge-or-error (horizon 2F1 series)" begin
    s, l, m = -2, 2, 2

    @testset "256-bit values at r = 15…30 vs 512-bit reference + ODE arbiter" begin
        # deep reference at 2× precision (independently converged + certified)
        refs = setprecision(BigFloat, 512) do
            ν, p = compute_nu(s, l, m, big"0.7", Complex{BigFloat}(big"0.8"))
            fn = compute_fn(p, ν)
            Dict(r => (Rin(p, ν, fn, BigFloat(r)), dRin(p, ν, fn, BigFloat(r)))
                 for r in (15, 20, 25, 30))
        end

        setprecision(BigFloat, 256) do
            ν, p = compute_nu(s, l, m, big"0.7", Complex{BigFloat}(big"0.8"))
            fn = compute_fn(p, ν)           # default nmax=80 — MUST auto-extend
            for r in (15, 20, 25, 30)
                v  = Rin(p, ν, fn, BigFloat(r))
                dv = dRin(p, ν, fn, BigFloat(r))
                # (a) value + derivative against the deep reference.  The old
                # code was wrong by 8.3 / 8e8 / 2.3e15 (r=20/25/30); the fixed
                # series is limited only by the certified cancellation floor
                # (measured 5e-62 at r=30, 256-bit) — gate well below the old
                # garbage and above the floor.
                @test abs(v - refs[r][1]) / abs(refs[r][1]) < 1e-45
                @test abs(dv - refs[r][2]) / abs(refs[r][2]) < 1e-45
                # (b) algorithm-independent ODE residual at the returned point.
                # h balances FD truncation (h²) against the certified
                # cancellation floor of dRin at r=30 (~1e-61): both land ≈1e-40.
                h = BigFloat(10)^(-20)
                res = _teuk_ode_residual(p, v, dv, rr -> dRin(p, ν, fn, rr),
                                         BigFloat(r), h)
                @test res < 1e-30
            end
            # the fn dict must have been extended in place beyond the callers nmax
            @test maximum(keys(fn)) > 80
        end
    end

    @testset "Float64: accurate where certified, honest error beyond" begin
        ν, p = compute_nu(s, l, m, 0.7, 0.8)
        fn = compute_fn(p, ν)
        ref = setprecision(BigFloat, 256) do
            νb, pb = compute_nu(s, l, m, big"0.7", Complex{BigFloat}(big"0.8"))
            fnb = compute_fn(pb, νb)
            Dict(r => Rin(pb, νb, fnb, BigFloat(r)) for r in (10, 12, 20))
        end

        # r ≤ 12: returns, and the value is genuinely accurate (the floor
        # certification eps·max|partial| ≤ √eps·|sum| held).
        for r in (10, 12)
            v = Rin(p, ν, fn, Float64(r))
            @test abs(v - ComplexF64(ref[r])) / abs(ref[r]) < 1e-7
        end

        # r = 20/30: the cancellation floor exceeds √eps — the old code
        # returned values wrong by 3.9e-3 (r=20) and 2.1e4 (r=30) RELATIVE;
        # now an honest error.
        @test_throws ErrorException Rin(p, ν, fn, 20.0)
        @test_throws ErrorException Rin(p, ν, fn, 30.0)
        @test_throws ErrorException dRin(p, ν, fn, 30.0)

        # explicitly loosened tol is honored — and the returned value really
        # is within that tolerance of the converged reference.
        v20 = Rin(p, ν, fn, 20.0; tol=1e-2)
        @test abs(v20 - ComplexF64(ref[20])) / abs(ref[20]) < 1e-2
    end

    @testset "truncated / exhausted fn errors instead of silent garbage" begin
        setprecision(BigFloat, 256) do
            ν, p = compute_nu(s, l, m, big"0.7", Complex{BigFloat}(big"0.8"))
            # aggressive |f_n|-based truncation zero-fills the tail; at r=30
            # the 2F1 term peak needs far more coefficients — Rin must refuse
            # (zero-tail termination with the last term above tolerance).
            fn_trunc = compute_fn(p, ν; nmax=80, tol=1e-30)
            if any(iszero, values(fn_trunc))
                @test_throws ErrorException Rin(p, ν, fn_trunc, big"30.0")
            end
        end
    end

    @testset "dRin consistency with numerical derivative of Rin (256-bit, r=25)" begin
        setprecision(BigFloat, 256) do
            ν, p = compute_nu(s, l, m, big"0.7", Complex{BigFloat}(big"0.8"))
            fn = compute_fn(p, ν)
            r = big"25.0"; h = BigFloat(10)^(-20)
            dnum = (Rin(p, ν, fn, r + h) - Rin(p, ν, fn, r - h)) / (2h)
            dana = dRin(p, ν, fn, r)
            @test abs(dnum - dana) / abs(dana) < 1e-30
        end
    end
end

@testset "R6: decidable instability guards (Arb backend)" begin
    @testset "old ball guard undecidable, new guard trips" begin
        setprecision(Arb, 256) do
            v = Arb(10); Arblib.add_error!(v, Arb(100))
            val = Complex{Arb}(v, Arb(0))          # ball containing zero
            t1 = Complex{Arb}(Arb(1000), Arb(0))
            t2 = -t1 + val
            # the pre-fix expression: undecidable ball comparison → false →
            # fallback dead (this is the bug, asserted so a regression to the
            # old expression is caught)
            @test (iszero(val) || max(abs(t1 / val), abs(t2 / val)) > 2.0) == false
            # the fixed guard decides on Float64 midpoints → trips
            @test Teukolsky._recur_guard_trips(val, t1, t2)
            # sane values do NOT trip
            g = Complex{Arb}(Arb(1), Arb(0))
            @test !Teukolsky._recur_guard_trips(g, g / 2, g / 2)
            # non-finite values trip conservatively
            @test Teukolsky._recur_guard_trips(complex(NaN, 0.0), complex(1.0, 0.0))
            @test Teukolsky._recur_guard_trips(complex(0.0, 0.0), complex(1.0, 0.0))
        end
    end

    @testset "BigFloat twin fallback fires ⟹ Arb fallback fires, values agree" begin
        s, l, m = -2, 2, 2
        cnt = Teukolsky._RECUR_GUARD_TRIP_COUNT
        # small ω: ν near integer ⇒ near-degenerate recurrence denominators
        vb, dvb, tb = setprecision(BigFloat, 256) do
            ν, p = compute_nu(s, l, m, big"0.0", Complex{BigFloat}(big"0.05"))
            fn = compute_fn(p, ν)
            c0 = cnt[]
            v = Rin(p, ν, fn, big"10.0"); dv = dRin(p, ν, fn, big"10.0")
            (ComplexF64(v), ComplexF64(dv), cnt[] - c0)
        end
        @test tb > 0                     # the BigFloat twin takes the fallback
        va, dva, ta = setprecision(Arb, 256) do
            aA = Arb(0); ωA = Complex{Arb}(Arb("0.05"), Arb(0))
            ν, p = Teukolsky._compute_nu_monodromy(s, l, m, aA, ωA)
            fn = compute_fn(p, ν)
            c0 = cnt[]
            v = Rin(p, ν, fn, Arb(10)); dv = dRin(p, ν, fn, Arb(10))
            (ComplexF64(Complex{BigFloat}(v)), ComplexF64(Complex{BigFloat}(dv)),
             cnt[] - c0)
        end
        @test ta > 0                     # …and so does the Arb path now
        @test abs(va - vb) / abs(vb) < 1e-12
        @test abs(dva - dvb) / abs(dvb) < 1e-12
    end
end
