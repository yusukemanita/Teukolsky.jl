# ============================================================
#  Regression tests T11–T14 (guard/contract fixes)
#
#  T11  compute_Aminus_acb honors the FULL (nmin, nmax) contract (no silent
#       backward clamp at -nmax, no BoundsError for nmin > 0), equivalent to
#       the generic compute_Aminus and to an independent brute-force
#       Complex{BigFloat} arbiter (fresh Γ-free Pochhammer products, explicit
#       BigFloat π).
#  T12  compute_fn_acb's `tol` is the Lentz CONVERGENCE tolerance everywhere;
#       at the near-integer-ν generic-fallback gate it must NOT forward as the
#       generic compute_fn's series-TRUNCATION tol (which zero-fills entries).
#  T13  gamma_ratio works for PNSeries{Complex{BigFloat}} (the package default
#       coefficient type): ψ^{(k)} base points routed through Arb polygamma.
#       Arbitrated by direct Γ-quotient evaluation (algorithm-independent of
#       the polygamma series expansion) + the digamma recurrence identity.
#  T14a green_function contract: ArgumentError for Im(ω)≠0 (no silent
#       projection onto the real axis) and for ω=0 (pole; no finite value).
#  T14b Integer(::MultiFloat) has proper InexactError semantics (exact limb-sum
#       check) instead of silent leading-limb truncation; round(Int, ·) — the
#       radial near-integer-b guard's need — still works, exactly.
# ============================================================

using Test
using Teukolsky
using Arblib: Arb, Acb
import Arblib
using MultiFloats: Float64x2

@testset "T11: compute_Aminus_acb honors full (nmin, nmax)" begin
    setprecision(Arb, 300) do
    setprecision(BigFloat, 300) do
        s, l, m = -2, 2, 2
        a = Arb(0.7); ω = Complex{Arb}(Arb(1))
        ν, p = compute_nu(s, l, m, a, ω)
        nmax = 10
        fn = Teukolsky.compute_fn_acb(p, ν; nmax=2*nmax)   # covers nmin=-2nmax

        # Independent arbiter: brute-force Complex{BigFloat} A^ν_- with fresh
        # per-term Pochhammer products and explicit BigFloat(π) (no shared code
        # with either implementation under test).
        function Aminus_ref(p, ν, fn, nmin, nmax)
            C = Complex{BigFloat}
            sw = p.s
            ε = C(p.ϵ); νb = C(ν)
            πb = BigFloat(π)
            pref = C(2)^(-1 - sw + im*ε) * exp(-im*πb*(νb + 1 + sw)/2) *
                   exp(-πb*ε/2)
            ain = νb + 1 + sw - im*ε
            bin = νb + 1 - sw + im*ε
            poch(z, n) = n >= 0 ? prod([z + k for k in 0:n-1]; init=C(1)) :
                                  inv(prod([z - k for k in 1:-n]; init=C(1)))
            Σ = C(0)
            for n in nmin:nmax
                Σ += C(-1)^n * poch(ain, n) / poch(bin, n) * C(fn[n])
            end
            return pref * Σ
        end

        for nmin in (-2*nmax, -nmax, 0, 2)
            Am_g = Complex{BigFloat}(
                Teukolsky.compute_Aminus(p, ν, fn; nmax=nmax, nmin=nmin))
            Am_a = Complex{BigFloat}(
                Teukolsky.compute_Aminus_acb(p, ν, fn; nmax=nmax, nmin=nmin))
            Am_r = Aminus_ref(p, ν, fn, nmin, nmax)
            # generic equivalence (the old code silently clamped at -nmax and
            # threw BoundsError for nmin=2)
            @test abs(Am_a - Am_g) / abs(Am_g) < 1e-70
            # independent-arbiter agreement
            @test abs(Am_a - Am_r) / abs(Am_r) < 1e-70
        end
    end
    end
end

@testset "T12: compute_fn_acb tol at the near-integer-ν gate" begin
    setprecision(Arb, 256) do
    setprecision(BigFloat, 256) do
        s, l, m = -2, 2, 2
        a = Arb(0.7); ω = Complex{Arb}(Arb(0.005))
        ν, p = compute_nu(s, l, m, a, ω)
        # confirm we sit at the gate (ν within 1e-3 of an integer, so the
        # generic fallback path is the one exercised)
        νm = Complex{BigFloat}(ν)
        @test abs(imag(νm)) < 1e-3 && abs(real(νm) - round(real(νm))) < 1e-3

        nmax = 40
        fn = Teukolsky.compute_fn_acb(p, ν; nmax=nmax, tol=1e-4)
        # docstring contract: full 2·nmax sum, no truncation / zero-fill
        # (the old code forwarded tol as the generic TRUNCATION tol → 64/81
        # entries exactly zero)
        @test count(n -> iszero(fn[n]), -nmax:nmax) == 0
        # values equal the untruncated generic reference
        fg = compute_fn(p, ν; nmax=nmax, tol=-1)
        for n in -nmax:nmax
            fgn = Complex{BigFloat}(fg[n])
            @test abs(Complex{BigFloat}(fn[n]) - fgn) ≤ 1e-60 * max(abs(fgn), one(BigFloat))
        end
    end
    end
end

@testset "T13: gamma_ratio for Complex{BigFloat} PN series" begin
    setprecision(BigFloat, 256) do
        T = Complex{BigFloat}
        o = 6
        z1 = T(BigFloat(5)/2, BigFloat(3)/10)
        z2 = T(BigFloat(17)/10, -BigFloat(1)/5)
        num = pnconst(z1, o, T) + pneps(o, T)
        den = pnconst(z2, o, T) + 2*pneps(o, T)
        r = gamma_ratio(num, den)          # old code: MethodError (_digamma)
        @test r isa PNSeries{T}

        # Algorithm-independent arbiter: the series must reproduce the direct
        # Γ-quotient at small numeric ε to O(ε^{order+1}) — Γ evaluation shares
        # nothing with the polygamma base-point expansion.  With order 6 the
        # residual at ε = 10^-6 is ~|c_{o+1}| ε^{o+1} ≈ 1e-42; gate at 1e-38.
        for εv in (T(BigFloat(10)^-6), T(BigFloat(10)^-8))
            lhs = evalseries(r, εv)
            rhs = Teukolsky._cgamma(z1 + εv) / Teukolsky._cgamma(z2 + 2εv)
            @test abs(lhs - rhs) / abs(rhs) < abs(εv)^(o + 1) * BigFloat(10)^28
        end

        # digamma recurrence identity ψ(z+1) = ψ(z) + 1/z at BigFloat-256
        # (self-arbitrating: both sides through the new Arb-bridged base point)
        ψ1 = Teukolsky._polygamma_base(0, z1 + 1)
        ψ0 = Teukolsky._polygamma_base(0, z1)
        @test abs(ψ1 - (ψ0 + 1/z1)) < BigFloat(2)^(-240)
        # polygamma recurrence ψ'(z+1) = ψ'(z) - 1/z²
        ψp1 = Teukolsky._polygamma_base(1, z1 + 1)
        ψp0 = Teukolsky._polygamma_base(1, z1)
        @test abs(ψp1 - (ψp0 - 1/z1^2)) < BigFloat(2)^(-240)

        # ComplexF64 coefficient type still routes through SpecialFunctions
        T2 = ComplexF64
        r2 = gamma_ratio(pnconst(T2(z1), 4, T2) + pneps(4, T2),
                         pnconst(T2(z2), 4, T2) + 2*pneps(4, T2))
        @test abs(getcoeff(r2, 1) - ComplexF64(getcoeff(r, 1))) <
              1e-13 * abs(getcoeff(r2, 1))
    end
end

@testset "T14a: green_function real-axis / ω≠0 contract" begin
    wp = WaveformParams(s=-2, l=2, m=2, a=0.0, N=8, ω_max=2.0, Nt=2,
                        verbose=false)
    # complex ω: no silent projection onto the real axis
    @test_throws ArgumentError Teukolsky.Waveform.green_function(wp, 0.3 + 0.5im)
    # ω = 0: pole of G — explicit error, not NaN
    @test_throws ArgumentError Teukolsky.Waveform.green_function(wp, 0.0)
    @test_throws ArgumentError Teukolsky.Waveform.green_function(wp, 0.0 + 0.0im)
    # real axis unchanged: finite, reality condition, complex-typed real ω OK
    G = Teukolsky.Waveform.green_function(wp, 0.3)
    @test isfinite(G)
    @test Teukolsky.Waveform.green_function(wp, -0.3) == conj(G)
    @test Teukolsky.Waveform.green_function(wp, 0.3 + 0.0im) == G
end

@testset "T14b: Integer(::MultiFloat) InexactError semantics" begin
    # silent truncation fixed: 1 + 1e-25 is NOT an integer
    @test_throws InexactError Int(Float64x2(1) + 1e-25)
    # off-by-one beyond 2^53 fixed: the tail limb carries the +1
    y = Float64x2(2.0^60) + 1
    @test Int(y) == 2^60 + 1
    @test Int(Float64x2(-7.0)) == -7
    @test Int128(Float64x2(2.0^80) + 3) == Int128(2)^80 + 3
    # range check
    @test_throws InexactError Int8(Float64x2(300.0))
    # non-finite
    @test_throws InexactError Int(Float64x2(1) / Float64x2(0))
    # the actual internal need — round-then-convert — still works, exactly
    @test round(Int, Float64x2(3.4)) == 3
    @test round(Int, Float64x2(2.0^60) + 1 + Float64x2(0.4)) == 2^60 + 1

    # radial near-integer-b guard path (hypergeometric.jl round(Int, real(b)))
    # still functional under the MultiFloat backend …
    A = compute_amplitudes(-2, 2, 2, 0.5, 0.3; backend=:multifloat,
                           precision=106)
    pmf = Teukolsky.MSTParams(-2, 2, 2, Float64x2(0.5),
                              Complex{Float64x2}(Float64x2(0.3)))
    fnmf = compute_fn(pmf, A.ν)
    @test isfinite(ComplexF64(Rin(pmf, A.ν, fnmf, Float64x2(3.0))))
    @test isfinite(ComplexF64(Rup(pmf, A.ν, fnmf, Float64x2(3.0))))
    # … and under BigFloat
    setprecision(BigFloat, 256) do
        Ab = compute_amplitudes(-2, 2, 2, big"0.5", big"0.3";
                                backend=:bigfloat, precision=256)
        pb = Teukolsky.MSTParams(-2, 2, 2, big"0.5",
                                 Complex{BigFloat}(big"0.3"))
        fnb = compute_fn(pb, Ab.ν)
        @test isfinite(ComplexF64(Rin(pb, Ab.ν, fnb, big"3.0")))
        @test isfinite(ComplexF64(Rup(pb, Ab.ν, fnb, big"3.0")))
    end
end
