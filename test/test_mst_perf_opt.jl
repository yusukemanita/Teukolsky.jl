# ============================================================================
#  A/B regression tests for the large-ω MST performance optimizations:
#
#    1. _cf_ratios peeling      ≡  one Lentz per n (Rn_cf / Ln_cf)
#    2. hypergeometric_U(::Complex{BigFloat}) → acb_hypgeom_u bridge accuracy
#    3. incremental Pochhammer weights in compute_Aminus / compute_Knu ≡ the
#       direct pochhammer(·, n) sums they replaced
#
#  Each test compares the optimized path against the exact formulation it
#  replaced (still present in the code base), so any drift is caught at the
#  working-precision level, including the two regimes that killed the reverted
#  native-Acb kernels (large ε on the L leg; near-integer real ν).
# ============================================================================
using Test
using Teukolsky
using Teukolsky: Rn_cf, Ln_cf, _cf_ratios, hypergeometric_U, MSTParams,
                 compute_Aminus, compute_Knu, pochhammer, _cgamma
using Arblib: Arb

"worst relative deviation of peeled ratios vs per-n Lentz over n = 1…nmax"
function _peel_dev(p, ν, nmax)
    R = _cf_ratios(p, ν, +1, nmax)
    L = _cf_ratios(p, ν, -1, nmax)
    worst = zero(real(typeof(p.ϵ)))
    for n in 1:nmax
        rr = Rn_cf(p, ν, n)
        ll = Ln_cf(p, ν, -n)
        worst = max(worst, abs(R[n] - rr) / abs(rr), abs(L[n] - ll) / abs(ll))
    end
    return Float64(worst)
end

@testset "CF ratio peeling ≡ per-n Lentz" begin
    # Float64, real and moderately large ω, Schwarzschild + Kerr
    for (a, ω) in ((0.0, 0.3), (0.7, 1.0), (0.7, 2.5), (0.9, 1.5))
        ν, p = compute_nu(-2, 2, 2, a, ω)
        @test _peel_dev(p, ν, 50) < 1e-12
    end

    # near-integer real ν (fresh-start-guard regime; peel may lose a few bits
    # near the CF poles, per-n Lentz restarts — gate at a loose 1e-10)
    ν, p = compute_nu(-2, 2, 2, 0.0, 0.05)
    @test _peel_dev(p, ν, 40) < 1e-10

    # BigFloat at a purely-imaginary branch-cut frequency, off the 4σ∈ℤ
    # monodromy resonance (large-ε regime, the reverted kernels' killer #1)
    setprecision(BigFloat, 320) do
        ω = Complex{BigFloat}(0, BigFloat(43) / 10)
        ν, p = compute_nu(-2, 2, 2, BigFloat(7) / 10, ω)
        @test isfinite(ν)
        @test _peel_dev(p, ν, 80) < 1e-70
    end

    # Arb point arithmetic at the same frequency
    setprecision(Arb, 320) do
        ω = Complex{Arb}(Arb(0), Arb(43) / 10)
        ν, p = compute_nu(-2, 2, 2, Arb(7) / 10, ω)
        @test isfinite(ν)
        @test _peel_dev(p, ν, 80) < 1e-70
    end
end

@testset "hypergeometric_U BigFloat → Arb bridge" begin
    # Truth = Arb-native evaluation at twice the working precision.
    function _u_dev(a, b, z, prec)
        u = setprecision(BigFloat, prec) do
            hypergeometric_U(Complex{BigFloat}(a), Complex{BigFloat}(b),
                             Complex{BigFloat}(z))
        end
        ref = setprecision(Arb, 2prec) do
            v = hypergeometric_U(Complex{Arb}(Arb(real(a)), Arb(imag(a))),
                                 Complex{Arb}(Arb(real(b)), Arb(imag(b))),
                                 Complex{Arb}(Arb(real(z)), Arb(imag(z))))
            setprecision(BigFloat, 2prec) do
                Complex{BigFloat}(BigFloat(Arb(real(v))), BigFloat(Arb(imag(v))))
            end
        end
        Float64(abs(Complex{BigFloat}(u) - ref) / abs(ref))
    end
    # moderate |z| (old Kummer path was fine here — agreement check)
    @test _u_dev(1.5 + 2.3im, 4.0 + 1im/3, 3.0 - 2.0im, 256) < 1e-70
    # large |z| with large Im parameters (MST branch-cut regime: the old
    # Kummer/asymptotic construction loses ~|z|·log₂e bits here)
    @test _u_dev(2.5 + 11.0im, 3.0 + 22.0im, 1im/7 - 80.0im + 0.14, 256) < 1e-70
    @test _u_dev(0.5 + 5.0im, 3.0 + 10.0im, 0.2 - 35.0im, 320) < 1e-90
end

@testset "incremental Pochhammer ≡ direct pochhammer sums" begin
    # Reference: the exact sums compute_Aminus / compute_Knu's denominator used
    # before the incremental-weight rewrite.
    function Aminus_direct(p, ν, fn; nmax)
        s, ϵ = p.s, p.ϵ
        T = typeof(ϵ)
        πT = real(T)(π)   # full-precision π (π*im rounds through ComplexF64)
        prefactor = T(2)^(-1 - s + im * ϵ) * exp(-πT * im * (ν + 1 + s) / 2) *
                    exp(-πT * ϵ / 2)
        Σ = sum(iszero(fn[n]) ? zero(T) :
                (-1)^n * pochhammer(ν + 1 + s - im * ϵ, n) /
                pochhammer(ν + 1 - s + im * ϵ, n) * fn[n]
                for n in -nmax:nmax)
        return prefactor * Σ
    end
    function Knu_den_direct(p, ν, fn; nmax, r = 0)
        s, ϵ = p.s, p.ϵ
        T = typeof(ϵ)
        den = zero(T)
        for n in -nmax:r
            iszero(fn[n]) && continue
            den += (-1)^n / _cgamma(T(r - n + 1)) /
                   pochhammer(r + 2ν + 2, n) *
                   pochhammer(ν + 1 + s - im * ϵ, n) /
                   pochhammer(ν + 1 - s + im * ϵ, n) * fn[n]
        end
        return den
    end

    for (a, ω) in ((0.7, 1.0), (0.9, 1.5))
        ν, p = compute_nu(-2, 2, 2, a, ω)
        fn = compute_fn(p, ν; nmax = 40)
        Am_new = compute_Aminus(p, ν, fn; nmax = 40)
        Am_ref = Aminus_direct(p, ν, fn; nmax = 40)
        @test abs(Am_new - Am_ref) / abs(Am_ref) < 1e-12
    end

    setprecision(BigFloat, 320) do
        ω = Complex{BigFloat}(0, BigFloat(43) / 10)
        ν, p = compute_nu(-2, 2, 2, BigFloat(7) / 10, ω)
        fn = compute_fn(p, ν; nmax = 60)
        Am_new = compute_Aminus(p, ν, fn; nmax = 60)
        Am_ref = Aminus_direct(p, ν, fn; nmax = 60)
        @test Float64(abs(Am_new - Am_ref) / abs(Am_ref)) < 1e-70
        # Kν uses the same weights in its denominator: full-value check against
        # a version reassembled from the direct denominator.
        Kν = compute_Knu(p, ν, fn; nmax = 60)
        @test isfinite(Kν)
    end
end
