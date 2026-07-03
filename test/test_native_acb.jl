# Native-Acb kernels (M3) + Arb radial/CF robustness:
#   * f^ν_n and A^ν_± : native in-place Acb vs the generic Complex{Arb} path
#     (element-wise / value equivalence).  NOTE the reference history: the first
#     adversarial review "convicted" these kernels at large σ and near-integer ν,
#     but the truth arbiters (bottom-up CF evaluation + Miller minimal-solution
#     recurrence) later PROVED the reference itself was broken — the specialized
#     Complex{Arb} _lentz_cf stall exit (arb_compat.jl) returned a stale iterate
#     as converged for σ ≳ 13.3 — and the near-integer-ν deviation is intrinsic
#     1/δ conditioning shared by every fixed-precision kernel (incl. Miller).
#   * Arb-Lentz stall regression: Ln_cf on Complex{Arb} at σ=16 must match the
#     arbiter-certified truth (it was 40% wrong, silently, before the √tol gate).
#   * R^up : the generic recurrence with rigorous acb_hypgeom_u seeds vs a
#     700-bit reference (the old Kummer seed was ~1e14 off at σ=4).
# Uses ARBITRARY complex ν (both paths share it) — decoupled from compute_nu.
using Test
using Teukolsky
using Arblib: Arb, Acb

const _relC = (A, g) -> Float64(abs(Complex{BigFloat}(A) - Complex{BigFloat}(g)) /
                                max(abs(Complex{BigFloat}(g)), BigFloat("1e-300")))

@testset "native-Acb kernels (M3)" begin
    nmax = 40
    cases = ((-2,2,2,0.5, Complex(0.5, 0.30)),
             (-2,3,2,2.0, Complex(1.7,-0.40)),
             (-2,2,2,4.0, Complex(1.5,-0.47)),
             ( 2,2,2,1.0, Complex(0.8, 0.20)),
             (-1,3,1,1.5, Complex(2.1, 0.90)))
    setprecision(Arb, 300) do
        for (s,l,m,σ,ν0) in cases
            p = MSTParams(s, l, m, Arb(7)/10, Complex{Arb}(Arb(0), Arb(σ)))
            ν = Complex{Arb}(Arb(real(ν0)), Arb(imag(ν0)))
            fg = compute_fn(p, ν; nmax=nmax)
            fa = Teukolsky.compute_fn_acb(p, ν; nmax=nmax)
            @testset "fn s=$s l=$l σ=$σ" begin
                @test Set(keys(fa)) == Set(keys(fg))
                @test maximum(_relC(fa[n], fg[n]) for n in -nmax:nmax) < 1e-70
            end
            Apg = Teukolsky.compute_Aplus(p, ν, fg; nmax=nmax, nmin=-nmax)
            Amg = Teukolsky.compute_Aminus(p, ν, fg; nmax=nmax, nmin=-nmax)
            Apa = Teukolsky.compute_Aplus_acb(p, ν, fa; nmax=nmax)
            Ama = Teukolsky.compute_Aminus_acb(p, ν, fa; nmax=nmax)
            @testset "A± s=$s l=$l σ=$σ" begin
                @test _relC(Apa, Apg) < 1e-70
                @test _relC(Ama, Amg) < 1e-70
            end
        end
    end
end

@testset "A± prefactor truth arbiter (exact-π MPFR at prec+128)" begin
    # ALGORITHM-INDEPENDENT truth: the full A± values rebuilt in BigFloat/MPFR
    # at (working precision + 128) with EXPLICIT full-precision π, from the
    # same fn.  This arbiter CONVICTED the historical generic prefactors: the
    # old `exp(π*im*(ν+1-s)/2)` evaluated `π*im` first, promoting π through
    # Complex{Bool} to Float64 — a silent 1.2e-16 relative phase error at
    # every precision.  Both the native-Acb kernels and the fixed generic
    # path must now agree with the arbiter to ~eps of the working precision.
    prec = 320
    setprecision(Arb, prec) do
        for (s,l,m,σ,ν0) in ((-2,2,2,4.0, Complex(1.5,-0.47)),
                             ( 2,2,2,1.0, Complex(0.8, 0.20)))
            p = MSTParams(s, l, m, Arb(7)/10, Complex{Arb}(Arb(0), Arb(σ)))
            ν = Complex{Arb}(Arb(real(ν0)), Arb(imag(ν0)))
            nmax = 40
            fa = Teukolsky.compute_fn_acb(p, ν; nmax=nmax)
            Apa = Teukolsky.compute_Aplus_acb(p, ν, fa; nmax=nmax)
            Ama = Teukolsky.compute_Aminus_acb(p, ν, fa; nmax=nmax)
            Apg = Teukolsky.compute_Aplus(p, ν, fa; nmax=nmax, nmin=-nmax)
            Amg = Teukolsky.compute_Aminus(p, ν, fa; nmax=nmax, nmin=-nmax)
            Apr, Amr = setprecision(BigFloat, prec + 128) do
                C  = Complex{BigFloat}
                πb = BigFloat(π)
                ε  = C(BigFloat(real(p.ϵ)), BigFloat(imag(p.ϵ)))
                νb = C(BigFloat(real(ν)), BigFloat(imag(ν)))
                fb = Dict(n => C(BigFloat(real(fa[n])), BigFloat(imag(fa[n])))
                          for n in keys(fa))
                prefP = exp(-πb*ε/2) * exp(im*πb*(νb+1-s)/2) * C(2)^(-1+s-im*ε) *
                        Teukolsky._cgamma(νb + 1 - s + im*ε) /
                        Teukolsky._cgamma(νb + 1 + s - im*ε)
                ApR = prefP * sum(fb[n] for n in -nmax:nmax)
                prefM = C(2)^(-1-s+im*ε) * exp(-im*πb*(νb+1+s)/2) * exp(-πb*ε/2)
                aw = νb + 1 + s - im*ε; bw = νb + 1 - s + im*ε
                ΣM = fb[0]; w = C(1)
                for n in 1:nmax
                    w *= -(aw + (n-1)) / (bw + (n-1)); ΣM += w * fb[n]
                end
                w = C(1)
                for n in -1:-1:-nmax
                    w *= -(bw + n) / (aw + n); ΣM += w * fb[n]
                end
                (ApR, prefM * ΣM)
            end
            @testset "arbiter s=$s σ=$σ" begin
                # 1e-70 leaves ~54 orders of headroom below the 1.2e-16 bug
                # this arbiter convicts, while absorbing cancellation-driven
                # amplification of the 320-bit working eps (~2e-96).
                @test _relC(Apa, Apr) < 1e-70
                @test _relC(Ama, Amr) < 1e-70
                @test _relC(Apg, Apr) < 1e-70     # fixed generic path
                @test _relC(Amg, Amr) < 1e-70
            end
        end
    end
end

@testset "native-Acb A± hostile σ ∈ {8,12} at predictor bits" begin
    # Large-ε PIA regime at the precision-predictor bit counts; A± equivalence
    # native-Acb vs (fixed) generic on the SAME fn, plus a near-integer real ν
    # case (Pochhammer near-pole conditioning in the A− weights).
    for (σ, bits, nmax, ν0) in ((8.0,  640,  94, Complex(1.9, 1.3)),
                                (12.0, 896, 120, Complex(1.9, 2.2)),
                                (8.0,  640,  94, Complex(3.0 + 1e-20, 1e-25)))
        setprecision(Arb, bits) do
            p = MSTParams(-2, 2, 2, Arb(7)/10, Complex{Arb}(Arb(0), Arb(σ)))
            ν = Complex{Arb}(Arb(real(ν0)), Arb(imag(ν0)))
            fg = compute_fn(p, ν; nmax=nmax)
            Apg = Teukolsky.compute_Aplus(p, ν, fg; nmax=nmax, nmin=-nmax)
            Amg = Teukolsky.compute_Aminus(p, ν, fg; nmax=nmax, nmin=-nmax)
            Apa = Teukolsky.compute_Aplus_acb(p, ν, fg; nmax=nmax)
            Ama = Teukolsky.compute_Aminus_acb(p, ν, fg; nmax=nmax)
            @testset "A± σ=$σ bits=$bits ν≈$(ComplexF64(ν0))" begin
                @test _relC(Apa, Apg) < 1e-70
                @test _relC(Ama, Amg) < 1e-70
            end
        end
    end
end

@testset "internal Acb vector path ≡ public Dict path (M2 wiring)" begin
    # _compute_fn_acb_vec / _fn_dict_from_vec must be BIT-identical to the
    # public compute_fn_acb, and compute_mst_core_acb (which wires the vector
    # path) must return the same fn/Ap/Am as the public Dict-path kernels.
    bits = 320; nmax = 40
    setprecision(Arb, bits) do
        p = MSTParams(-2, 2, 2, Arb(7)/10, Complex{Arb}(Arb(0), Arb(43)/10))
        ν = Complex{Arb}(Arb(1884)/1000, Arb(503)/1000)
        fd = Teukolsky.compute_fn_acb(p, ν; nmax=nmax)
        fv = Teukolsky._compute_fn_acb_vec(p, ν; nmax=nmax)
        @test length(fv) == 2nmax + 1
        @test all(iszero(_relC(Complex{Arb}(fv[n + nmax + 1]), fd[n]))
                  for n in -nmax:nmax)
        core = compute_mst_core_acb(-2, 2, 2, 0.7, Complex(0.0, 4.3);
                                    ν=Complex{BigFloat}(BigFloat(real(ν)),
                                                        BigFloat(imag(ν))),
                                    nmax=nmax, precision=bits)
        @test core.fn isa Dict{Int,Complex{Arb}}
        # The core's A± window is converge-or-error (grows from the `nmax`
        # hint until the weighted tails pass the tol criterion), so wiring
        # parity is checked at the FINAL window the dict came out with.
        wmax = maximum(keys(core.fn))
        @test wmax >= nmax
        @test Set(keys(core.fn)) == Set(-wmax:wmax)
        Apd = Teukolsky.compute_Aplus_acb(core.p, core.ν, core.fn; nmax=wmax)
        Amd = Teukolsky.compute_Aminus_acb(core.p, core.ν, core.fn; nmax=wmax)
        @test iszero(_relC(core.Ap, Apd))
        @test iszero(_relC(core.Am, Amd))
    end
end

@testset "deep-IR near-integer-ν gate (Gpia T0b corner)" begin
    # At σ = 5e-4 (deep IR), ν(l′) → l′ within ~2e-7 and the backward peel
    # crosses CF poles; the native folded-coefficient peel lost ~7 digits there
    # (f₋₂/f₋₃ O(1) wrong for l′=5 — caught by the Gpia Wolfram cross-check).
    # The near-integer-ν gate now routes this corner through the generic ratio
    # peel; the full Arb chain must match BigFloat.
    for lp in (4, 5)
        vb = setprecision(BigFloat, 256) do
            a = BigFloat(7)/10; ω = Complex{BigFloat}(0, BigFloat(1)/2000)
            core = compute_mst_core(-2, lp, 2, a, ω; nmax=40)
            Complex{BigFloat}(qtilde_from_core(core) *
                Rup(core.p, core.ν, core.fn, BigFloat(10); nmax=40, ctrans=mst_ctrans(core)))
        end
        va = setprecision(Arb, 256) do
            a = Arb(7)/10; ω = Complex{Arb}(Arb(0), Arb(1)/2000)
            core = compute_mst_core(-2, lp, 2, a, ω; nmax=40)
            Complex{BigFloat}(qtilde_from_core(core) *
                Rup(core.p, core.ν, core.fn, Arb(10); nmax=40, ctrans=mst_ctrans(core)))
        end
        @test _relC(va, vb) < 1e-25
    end
end

@testset "Arb Lentz stall-exit regression (σ ≳ 13.3 backward CF)" begin
    # Truth is computed IN-TEST by the algorithm-independent bottom-up (tail-to-
    # head) evaluation of the same continued fraction — self-arbitrating, so the
    # test stays valid if upstream parameters (e.g. the λ branch) change.  Before
    # the conditioning-floor gate the specialized Complex{Arb} _lentz_cf returned
    # a stale early iterate here, silently flagged converged and ~40% wrong at
    # EVERY precision; the bottom-up value (cross-checked against the Miller
    # minimal-solution ratio to ~1e-114 during the fix campaign) is exact.
    function bottom_up_L(p, ν, K)
        t = zero(Complex{Arb})
        for j in K:-1:1
            aj = j == 1 ? -Teukolsky.αn(p, ν, -1) :
                          -Teukolsky.γn(p, ν, -1-j+2) * Teukolsky.αn(p, ν, -1-j+1)
            bj = Teukolsky.βn(p, ν, -1-j+1)
            t = aj / (bj + t)
        end
        return t
    end
    setprecision(Arb, 384) do
        p = MSTParams(-2,2,2, Arb(7)/10, Complex{Arb}(Arb(0), Arb(16)))
        ν = Complex{Arb}(Arb(49)/10, Arb(-35)/100)
        truth  = bottom_up_L(p, ν, 2000)
        truth2 = bottom_up_L(p, ν, 1000)
        @test _relC(truth2, truth) < 1e-80          # bottom-up depth-converged
        L = Teukolsky.Ln_cf(p, ν, -1; nmax=4000)
        @test _relC(L, truth) < 1e-90
        # Arb path must agree with the (always-correct) BigFloat generic path
        # across the former failure region.
        for σ in (13.5, 16.0, 20.0)
            pA = MSTParams(-2,2,2, Arb(7)/10, Complex{Arb}(Arb(0), Arb(σ)))
            LA = Teukolsky.Ln_cf(pA, ν, -1; nmax=4000)
            LB = setprecision(BigFloat, 384) do
                pB = MSTParams(-2,2,2, BigFloat(7)/10, Complex{BigFloat}(0, BigFloat(σ)))
                Teukolsky.Ln_cf(pB, Complex{BigFloat}(BigFloat(49)/10, BigFloat(-35)/100),
                                -1; nmax=4000)
            end
            @test _relC(LA, LB) < 1e-100
        end
    end
end

@testset "Arb Rup via rigorous acb_hypgeom_u" begin
    nmax = 60
    for (s,l,m,σ,ν0) in ((-2,2,2,0.5, Complex(0.5, 0.30)),
                         (-2,3,2,2.0, Complex(1.7,-0.40)),
                         (-2,2,2,4.0, Complex(1.5,-0.47)),   # old Kummer seed: ~1e14 off
                         ( 2,2,2,1.0, Complex(0.8, 0.20)),
                         (-1,3,1,1.5, Complex(2.1, 0.90)))
        ref_rup = setprecision(Arb, 700) do
            pr = MSTParams(s, l, m, Arb(7)/10, Complex{Arb}(Arb(0), Arb(σ)))
            νr = Complex{Arb}(Arb(real(ν0)), Arb(imag(ν0)))
            fr = compute_fn(pr, νr; nmax=nmax)
            ctr = Teukolsky._ctrans(pr, Teukolsky.compute_Aminus(pr, νr, fr; nmax=nmax))
            Complex{BigFloat}(Rup(pr, νr, fr, Arb(10); nmax=nmax, ctrans=ctr))
        end
        setprecision(Arb, 300) do
            p = MSTParams(s, l, m, Arb(7)/10, Complex{Arb}(Arb(0), Arb(σ)))
            ν = Complex{Arb}(Arb(real(ν0)), Arb(imag(ν0)))
            fg = compute_fn(p, ν; nmax=nmax)
            ct = Teukolsky._ctrans(p, Teukolsky.compute_Aminus(p, ν, fg; nmax=nmax))
            ra = Rup(p, ν, fg, Arb(10); nmax=nmax, ctrans=ct)
            @testset "Rup s=$s l=$l σ=$σ" begin
                @test isfinite(_relC(ra, ra))
                @test _relC(ra, ref_rup) < 1e-30
            end
        end
    end
end
