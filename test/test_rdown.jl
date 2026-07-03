# ============================================================
#  Issue R10 — Rdown: certified-HU wiring, corrected asymptotics, coverage
#
#  * Rdown now routes its HU[n] evaluation through the same certified
#    _hu_dhu_evaluators machinery as Rup (certified escalated seeds + stable
#    outward march on the Arb/BigFloat backends).
#  * The docstring previously claimed Rdown ~ r^{-2s-1} e^{-iωr*}; the
#    MEASURED asymptotics (and the standard peeling of the spin-s Teukolsky
#    equation, ST Eq. (21)) are Rdown ~ r^{-1} e^{-iωr*} with UNIT ingoing
#    amplitude.  Both the power law and the unit amplitude are asserted.
#  * Value coverage: 2×-precision arbiter at moderate and large |ω| plus the
#    algorithm-independent Teukolsky ODE residual (finite differences of
#    Rdown itself — no same-code derivative involved).
# ============================================================

using Test
using Teukolsky

# tortoise coordinate (M = 1): r* = r + 2r₊/(r₊−r₋)·ln((r−r₊)/2) − 2r₋/(r₊−r₋)·ln((r−r₋)/2)
_rstar(p, r) = r + 2p.rp / (p.rp - p.rm) * log((r - p.rp) / 2) -
                   2p.rm / (p.rp - p.rm) * log((r - p.rm) / 2)

# relative ODE residual from 5-point finite differences of Rdown alone
function _rdown_ode_residual(p, ν, fn, r, h)
    a, ω, s, m, λ = p.a, p.ω, p.s, p.m, p.λ
    Rm2 = Rdown(p, ν, fn, r - 2h); Rm1 = Rdown(p, ν, fn, r - h)
    R0  = Rdown(p, ν, fn, r)
    Rp1 = Rdown(p, ν, fn, r + h); Rp2 = Rdown(p, ν, fn, r + 2h)
    R1 = (-Rp2 + 8Rp1 - 8Rm1 + Rm2) / (12h)          # O(h⁴)
    R2 = (-Rp2 + 16Rp1 - 30R0 + 16Rm1 - Rm2) / (12h^2)
    Δ = r^2 - 2r + a^2
    K = (r^2 + a^2) * ω - a * m
    V = (K^2 - 2im * s * (r - 1) * K) / Δ + 4im * s * ω * r - λ
    res = Δ * R2 + 2 * (s + 1) * (r - 1) * R1 + V * R0
    scale = abs(Δ * R2) + abs(2 * (s + 1) * (r - 1) * R1) + abs(V * R0)
    return Float64(abs(res) / scale)
end

@testset "R10: Rdown" begin
    s, l, m = -2, 2, 2

    @testset "value vs 2×-precision arbiter (moderate and large |ω|)" begin
        for (a_str, ω_str, rtol) in (("0.7", "0.8", 1e-50), ("0.7", "4.0", 1e-45))
            refs = setprecision(BigFloat, 512) do
                ν, p = compute_nu(s, l, m, parse(BigFloat, a_str),
                                  Complex{BigFloat}(parse(BigFloat, ω_str)))
                fn = compute_fn(p, ν)
                Dict(r => Rdown(p, ν, fn, BigFloat(r)) for r in (10, 40))
            end
            setprecision(BigFloat, 256) do
                ν, p = compute_nu(s, l, m, parse(BigFloat, a_str),
                                  Complex{BigFloat}(parse(BigFloat, ω_str)))
                fn = compute_fn(p, ν)
                for r in (10, 40)
                    v = Rdown(p, ν, fn, BigFloat(r))
                    @test abs(v - refs[r]) / abs(refs[r]) < rtol
                end
            end
        end
    end

    @testset "Teukolsky ODE residual (algorithm-independent arbiter)" begin
        setprecision(BigFloat, 256) do
            ν, p = compute_nu(s, l, m, big"0.7", Complex{BigFloat}(big"0.8"))
            fn = compute_fn(p, ν)
            for r in (big"10.0", big"40.0")
                res = _rdown_ode_residual(p, ν, fn, r, BigFloat(10)^(-15))
                @test res < 1e-30
            end
        end
    end

    @testset "asymptotics: Rdown ~ r^{-1} e^{-iωr*}, unit amplitude" begin
        setprecision(BigFloat, 256) do
            ν, p = compute_nu(s, l, m, big"0.7", Complex{BigFloat}(big"0.8"))
            fn = compute_fn(p, ν)
            ω = p.ω
            rs = (big"200.0", big"400.0", big"800.0")
            vals = [Rdown(p, ν, fn, r) for r in rs]
            # power law: local slope d log|Rdown| / d log r → −1 (NOT −2s−1 = 3,
            # the docstring's former claim, which is off by r⁴)
            sl1 = Float64(log(abs(vals[2]) / abs(vals[1])) / log(rs[2] / rs[1]))
            sl2 = Float64(log(abs(vals[3]) / abs(vals[2])) / log(rs[3] / rs[2]))
            @test abs(sl1 + 1) < 0.05
            @test abs(sl2 + 1) < 0.02
            @test abs(sl2 + 1) < abs(sl1 + 1)          # converging TO −1
            # unit ingoing amplitude: c(r) = Rdown·r·e^{+iωr*} → 1
            c = [v * r * exp(im * ω * _rstar(p, r)) for (v, r) in zip(vals, rs)]
            @test abs(c[3] - 1) < 0.01
            @test abs(c[3] - 1) < abs(c[2] - 1) < abs(c[1] - 1)
        end
    end

    @testset "Float64 backend returns finite (legacy evaluator path)" begin
        ν, p = compute_nu(s, l, m, 0.7, 0.8)
        fn = compute_fn(p, ν)
        ref = setprecision(BigFloat, 256) do
            νb, pb = compute_nu(s, l, m, big"0.7", Complex{BigFloat}(big"0.8"))
            fnb = compute_fn(pb, νb)
            Rdown(pb, νb, fnb, big"10.0")
        end
        v = Rdown(p, ν, fn, 10.0)
        @test isfinite(v)
        @test abs(v - ComplexF64(ref)) / abs(ref) < 1e-6
    end
end
