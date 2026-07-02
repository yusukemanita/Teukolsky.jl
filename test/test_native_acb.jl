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
            Apg = Teukolsky.compute_Aplus(p, ν, fg; nmax=nmax)
            Amg = Teukolsky.compute_Aminus(p, ν, fg; nmax=nmax)
            Apa = Teukolsky.compute_Aplus_acb(p, ν, fa; nmax=nmax)
            Ama = Teukolsky.compute_Aminus_acb(p, ν, fa; nmax=nmax)
            @testset "A± s=$s l=$l σ=$σ" begin
                @test _relC(Apa, Apg) < 1e-70
                @test _relC(Ama, Amg) < 1e-70
            end
        end
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
