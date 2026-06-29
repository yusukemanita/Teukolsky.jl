# Hard @testset gate for the Arb backend of compute_nu (M1).
#
# Cross-checks backend=:arb against the BigFloat path (precision=256) for the 6
# branch-covering modes: real (|rc|<=1), half-integer (rc<-1), integer (rc>1)
# for both Schwarzschild (a=0) and Kerr (a=0.9).  Also asserts the branch
# classifier lands on the expected branch for each mode, and that ν respects the
# ν -> -ν-1 / conjugation symmetry used elsewhere.
#
# Observed agreement (256-bit, this machine): best 1.7e-73 (Schw real),
# worst 1.1e-49 (Kerr integer, s=0,l=2,m=0,ω=0.9).  Gate set at 1e-30 (30+ digit
# floor); a 1e-60 gate would false-fail because ball-radius widening on the
# integer/half branches at large ω costs digits.
using BHPtoolkit
using Test
using Arblib: Arb
const B = BHPtoolkit

# Symmetric ν distance (ν and -ν-1 and conjugates are physically equivalent).
nu_dist(a, b) = minimum(abs(a - c) for c in (b, -b - 1, conj(b), conj(-b - 1)))

# Branch classifier from rc = Re(cos(2πν)), computed in BigFloat.
function classify_branch(s, l, m, a, ω)
    setprecision(BigFloat, 256) do
        p  = B.MSTParams(s, l, m, BigFloat(a), Complex{BigFloat}(ω))
        c  = B._monodromy_adaptive(s, l, m, BigFloat(a), Complex{BigFloat}(ω), p.λ;
                                   R=BigFloat, nmax0=60)
        rc = real(c)
        (-1 ≤ rc ≤ 1) ? :real : (rc < -1 ? :half : :integer)
    end
end

# (s, l, m, a, ω, expected_branch)
const ARB_MODES = [
    (-2, 2, 2, 0.0, 0.3, :real),
    (-2, 2, 2, 0.0, 0.5, :half),
    (-2, 2, 2, 0.0, 1.0, :integer),
    (-2, 2, 2, 0.9, 0.1, :real),
    (-2, 2, 2, 0.9, 0.5, :half),
    ( 0, 2, 0, 0.9, 0.9, :integer),
]

const GATE = 1e-30

@testset "Arb backend ν solver (M1): Arb-vs-BigFloat, 6 branches" begin
    worst = 0.0
    for (s, l, m, a, ω, exp_branch) in ARB_MODES
        ν_arb, p_arb = compute_nu(s, l, m, a, ω; precision=256, backend=:arb)
        ν_bf,  _     = compute_nu(s, l, m, a, ω; precision=256)

        # backend=:arb returns native Arb types (M1: ν-solver-only).
        @test ν_arb isa Complex{Arb}
        @test p_arb isa MSTParams

        # branch classifier lands on the expected branch
        @test classify_branch(s, l, m, a, ω) === exp_branch

        d = setprecision(BigFloat, 256) do
            nu_dist(Complex{BigFloat}(ν_arb), Complex{BigFloat}(ν_bf))
        end
        df = Float64(d)
        worst = max(worst, df)
        @test df < GATE
    end
    @info "Arb-vs-BigFloat worst |ν_arb - ν_bf| (BigFloat metric)" worst
    @test worst < GATE
end
