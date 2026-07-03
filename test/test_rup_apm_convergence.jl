# ============================================================
#  Converge-or-error regressions: Rup/dRup series + A± windows (issue R1b)
#
#  The old fixed-nmax loops silently returned garbage on the positive
#  imaginary axis (PIA):
#
#    * Rup/dRup summed a fixed ±nmax window with an UNVERIFIED early break
#      and `get(fn, n, 0)` masking fn exhaustion — measured at s=-2 l=m=2
#      a=0.7: ω=10i, r=4 → 13% relative error at nmax=80; ω=16i, r=4 →
#      100% wrong at nmax=80, 11% at nmax=120.  Danger zone: small r +
#      large |ϵ| (the U-side terms only start decaying past
#      n ≈ (3.5–4.3)·|ẑ|).
#
#    * compute_Aplus/compute_Aminus summed a fixed ±nmax window with NO
#      stopping test at all.  On the PIA the A− weight grows polynomially
#      like n^(2|ϵ|+2s) (−2iϵ is real positive), so the weighted peak sits
#      far beyond fn's own decay: at the `suggest_mst_precision` settings
#      the truncation error was 1.1e-33 (ω=4.3i) … 2.6e-20 (ω=16i) — and it
#      enters EVERY Rup/q̃ value through A±/Ctrans, identically for Rup and
#      dRup and independent of r (how it was diagnosed).
#
#  Both now run through the _sum_mst_series!/_certify_mst_sum machinery
#  (adaptive fn extension, converge-or-error, cancellation-floor certified);
#  the native-Acb core grows its dense window until the weighted tails pass
#  the tol criterion.
#
#  Arbiter style (no pinned truth constants): independent-knob agreement —
#  window hints and working precision are varied independently; a silent
#  truncation reproduces the OLD wrong values, disagreeing across knobs.
# ============================================================

using Test, Teukolsky
using Arblib: Arb

_relCB(x, y) = Float64(abs(Complex{BigFloat}(x) - Complex{BigFloat}(y)) /
                       max(abs(Complex{BigFloat}(y)), BigFloat(1e-300)))

@testset "Rup/dRup converge-or-error on the PIA (old 13%–100% silent errors)" begin
    # The two measured failing points.  Window-hint independence: nmax=80
    # (old: garbage) must now agree with nmax=240 (old: converged) to far
    # below the old error levels.
    for (σnum, σden, r, bits) in ((10, 1, 4, 448), (16, 1, 4, 576))
        vals = map((80, 240)) do hint
            setprecision(Arb, bits) do
                a = Arb(7)/10
                ω = Complex{Arb}(Arb(0), Arb(σnum)/σden)
                core = compute_mst_core(-2, 5, 2, a, ω; nmax=hint)
                ct = mst_ctrans(core)
                ru = Rup(core.p, core.ν, core.fn, Arb(r); nmax=hint, ctrans=ct)
                dr = dRup(core.p, core.ν, core.fn, Arb(r); nmax=hint, ctrans=ct)
                (Complex{BigFloat}(ru), Complex{BigFloat}(dr))
            end
        end
        @testset "σ=$(σnum)i r=$r" begin
            @test _relCB(vals[1][1], vals[2][1]) < 1e-40
            @test _relCB(vals[1][2], vals[2][2]) < 1e-40
        end
    end
end

@testset "A± adaptive window ≡ wide exact window (old 1e-33…1e-20 truncation)" begin
    setprecision(Arb, 768) do
        a = Arb(7)/10
        ω = Complex{Arb}(Arb(0), Arb(10))
        ν, p = compute_nu(-2, 5, 2, a, ω; backend=:acb, precision=768)
        fn = Teukolsky.compute_fn(p, ν; nmax=900)
        # exact-window arbiter primitive at a window far beyond convergence
        Ap_wide = Teukolsky.compute_Aplus(p, ν, fn; nmax=800, nmin=-800)
        Am_wide = Teukolsky.compute_Aminus(p, ν, fn; nmax=800, nmin=-800)
        # adaptive from the (inadequate) suggested-scale hint
        fn2 = Teukolsky.compute_fn(p, ν; nmax=118)
        Ap_ad = Teukolsky.compute_Aplus(p, ν, fn2; nmax=118)
        Am_ad = Teukolsky.compute_Aminus(p, ν, fn2; nmax=118)
        @test _relCB(Ap_ad, Ap_wide) < 1e-100
        @test _relCB(Am_ad, Am_wide) < 1e-100
        # document the OLD bug: the fixed suggested window really was wrong
        Am_old = Teukolsky.compute_Aminus(p, ν, fn; nmax=118, nmin=-118)
        @test _relCB(Am_old, Am_wide) > 1e-50
    end
end

@testset "native-Acb core A± window ≡ generic adaptive" begin
    # compute_mst_core on Arb inputs routes to the native dense-vector core,
    # which must now grow its window to the same converged A± as the generic
    # adaptive path (independent implementations agreeing).
    setprecision(Arb, 448) do
        a = Arb(7)/10
        ω = Complex{Arb}(Arb(0), Arb(6))
        core = compute_mst_core(-2, 5, 2, a, ω; nmax=82)   # native path
        @test maximum(keys(core.fn)) > 82                  # window actually grew
        ν, p = compute_nu(-2, 5, 2, a, ω; backend=:acb, precision=448)
        fn = Teukolsky.compute_fn(p, ν; nmax=82)
        Apg = Teukolsky.compute_Aplus(p, ν, fn; nmax=82)   # generic adaptive
        Amg = Teukolsky.compute_Aminus(p, ν, fn; nmax=82)
        @test _relCB(core.Ap, Apg) < 1e-100
        @test _relCB(core.Am, Amg) < 1e-100
    end
end

@testset ">1074-bit precision no longer crashes (_swsh_lmax_auto underflow)" begin
    # Float64(eps(Arb)) underflows to 0.0 above 1074 bits; the old
    # round(Int, -log2(0.0)) crashed EVERY MSTParams construction at the
    # 1280/1536-bit ladder rungs suggest_mst_precision picks for |ω| ≳ 15.
    setprecision(Arb, 1280) do
        p = MSTParams(-2, 2, 2, Arb(7)/10, Complex{Arb}(Arb(0), Arb(16)))
        @test isfinite(Float64(real(p.λ)))
    end
end
