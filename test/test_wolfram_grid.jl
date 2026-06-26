using Test
using BHPtoolkit

# ============================================================
#  Quantitative cross-check vs Wolfram Teukolsky 1.1.1
#
#  Reference values generated with wolframscript driving the
#  BlackHolePerturbationToolkit `Teukolsky` paclet:
#    RenormalizedAngularMomentum[s,l,m,a,ω]      → ν
#    TeukolskyRadial[s,l,m,a,ω]["In"]["Amplitudes"]
#        "Incidence"  ↔ Binc   (transmission-normalized)
#        "Reflection" ↔ Bref
#  WorkingPrecision -> 20.  (script: scratchpad/teuk_ref.wls)
#
#  This grid spans ALL ν branches:
#    real (ν≈l), half-integer (Re ν=1/2, Im>0), integer (Re ν=0, Im<0),
#  Schwarzschild + Kerr, s=-2 and s=0, and ω up to 2 (the practical
#  accuracy wall — Wolfram itself returns ERR for s=-2 ℓ=m=2 at ω=4).
#
#  Tolerances are tiered by ω and reflect the *current* Float64-limited
#  accuracy of the radial/amplitude layer. They should be tightened once
#  the precision refactor lands (see the precision-status testset below).
# ============================================================

# (s, l, m, a, ω, ν_ref, Binc_ref, Bref_ref)
const GRID = [
 (-2,2,2,0.0,0.1, 1.97931545472084262855 + 0.0im,
   6522.40032021270115707 + 251268.283685574773347im,
   -10.8690064695185207688 - 12.7261943043535479431im),
 (-2,2,2,0.0,0.3, 1.77928054241991945364 + 0.0im,
   337.671613191770005238 + 74.9186104847653765643im,
   -1.75531801328576130076 - 0.379095976559026218244im),
 (-2,2,2,0.0,0.5, 0.5 + 0.361880615394167528135im,
   -41.1835801242498177529 - 30.0695301157412603826im,
   -0.255102728154579518494 + 0.0267077888166465868590im),
 (-2,2,2,0.0,1.0, 0.0 - 1.60855387765702277799im,
   -10.4717583152902486651 + 35.360179369060815744im,
   -0.00147365644971661629571 + 0.000629969473815147155157im),
 (-2,2,2,0.0,2.0, 0.0 - 3.68678902788934658280im,
   19.7445041542397398676 + 26.7424076824452656872im,
   -2.298297280502002335e-8 + 1.280587487184606403e-8im),
 (-2,2,2,0.9,0.5, 0.5 + 0.490707470232068739378im,
   9.36505614676274122128 + 1.87460985822846378270im,
   -0.0275061055735824011051 - 0.682984876895216536414im),
 (-2,2,2,0.9,1.0, 0.5 + 1.99432859529230076507im,
   1.45941512911047735561 + 0.780215840871144390090im,
   0.000672469089373538853447 + 0.00100217444584983504749im),
 (-2,2,2,0.9,2.0, 0.5 + 4.47338569273714503267im,
   3.106614196019256754 - 4.6896317913052762555im,
   5.0879411460165523564e-11 + 1.1353735651866067213e-10im),
 (-2,3,2,0.5,1.0, 0.5 + 0.909541366165471337003im,
   -12.3206708404279401835 - 9.1124961919642904808im,
   -0.0161129410315397992259 + 0.00430826460201253968734im),
 (0,0,0,0.0,0.5, 0.5 + 0.930590543501835523434im,
   1.93556376637579555357 + 0.503713892446760224495im,
   -0.00336935804846889252630 - 0.0111097514401901329797im),
 (0,2,0,0.0,1.0, 0.0 - 1.29164072953179318704im,
   -0.235077818284736838957 + 1.98613678255497632084im,
   -0.000672976651807130450582 + 0.000668406748319304754909im),
 (0,2,1,0.7,1.5, 0.5 + 2.42381418633661518963im,
   0.6292902707154643643 + 1.6017530315998837016im,
   -3.6966557352114213490e-7 - 6.1465060377182637924e-7im),
]

relerr(a, b) = abs(b) == 0 ? abs(a - b) : abs(a - b) / abs(b)

# ν is fixed only up to ν → -ν-1 (and complex conjugation across the cut);
# accept the closest representative.
function nu_dist(νj, νr)
    cands = (νr, -νr - 1, conj(νr), conj(-νr - 1))
    minimum(abs(νj - c) for c in cands)
end

@testset "vs Wolfram Teukolsky grid (all ν branches, Kerr, large ω)" begin
    for (s, l, m, a, ω, νr, Bincr, Brefr) in GRID
        # ω-tiered tolerances (current Float64-limited accuracy + margin)
        ν_tol, B_tol, rat_tol = ω ≤ 1.2 ? (1e-7, 1e-6, 1e-6) : (1e-6, 1e-4, 1e-3)
        @testset "s=$s l=$l m=$m a=$a ω=$ω" begin
            νj, _ = compute_nu(s, l, m, a, ω)
            amp = compute_amplitudes(s, l, m, a, ω)

            # ν matches Wolfram RenormalizedAngularMomentum (every branch)
            @test nu_dist(νj, νr) < ν_tol

            # Absolute amplitude match (same Transmission=1 normalization)
            @test relerr(amp.Binc, Bincr) < B_tol

            # Physical reflection coefficient Bref/Binc (drives G(ω))
            @test relerr(amp.Bref / amp.Binc, Brefr / Bincr) < rat_tol
        end
    end
end

# ============================================================
#  Precision status: the stack is type-generic and RUNS in BigFloat,
#  but the radial/λ/monodromy layers are still Float64-ACCURATE.
#  These guards lock in the current contract and will auto-flag
#  (Test reports "Unexpectedly Pass") once the precision refactor
#  makes BigFloat genuinely high-accuracy.
# ============================================================
@testset "precision status (BigFloat type-generic; accuracy gap documented)" begin
    s, l, m = -2, 2, 2

    # The BigFloat path must keep working (type-genericity is a real feature).
    amp_big = compute_amplitudes(s, l, m, big"0.0", Complex{BigFloat}(big"0.5"))
    @test amp_big.Binc isa Complex{BigFloat}
    ν_big, p_big = compute_nu(s, l, m, big"0.0", Complex{BigFloat}(big"0.3"))
    @test ν_big isa Complex{BigFloat}
    rin_big = Rin(p_big, ν_big, compute_fn(p_big, ν_big), BigFloat("10.0"))
    @test rin_big isa Complex{BigFloat}

    # KNOWN LIMITATION: BigFloat Rin is currently only ~1e-9 accurate
    # (radial layer downcasts to ComplexF64). Wolfram 20-digit reference
    # for s=-2,l=m=2,a=0,ω=0.3 at r=10:
    rin_ref = -129.59365769310846802183839606 + 1631.98664264768867986542190811im
    # @test_broken: expected to FAIL now; flips to "Unexpectedly Pass" after
    # the H2F1Params/HUParams generic-precision refactor.
    @test_broken relerr(rin_big, rin_ref) < 1e-15
end
