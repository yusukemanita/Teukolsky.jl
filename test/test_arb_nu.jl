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

# ============================================================
#  M2: native-Acb in-place monodromy kernel (backend=:acb)
# ============================================================

@testset "Acb backend ν solver (M2): Acb-vs-BigFloat, 6 branches" begin
    worst = 0.0
    for (s, l, m, a, ω, exp_branch) in ARB_MODES
        ν_acb, p_acb = compute_nu(s, l, m, a, ω; precision=256, backend=:acb)
        ν_bf,  _     = compute_nu(s, l, m, a, ω; precision=256)

        # backend=:acb returns native Arb types (same plumbing as M1).
        @test ν_acb isa Complex{Arb}
        @test p_acb isa MSTParams

        # branch classifier lands on the expected branch
        @test classify_branch(s, l, m, a, ω) === exp_branch

        d = setprecision(BigFloat, 256) do
            nu_dist(Complex{BigFloat}(ν_acb), Complex{BigFloat}(ν_bf))
        end
        df = Float64(d)
        worst = max(worst, df)
        @test df < GATE
    end
    @info "Acb-vs-BigFloat worst |ν_acb - ν_bf| (BigFloat metric)" worst
    @test worst < GATE
end

@testset "Acb vs M1 Arb tie-test" begin
    # Both backends source λ from the same compute_lambda and run at Arb
    # precision, so they must agree well within the gate.  On the Schwarzschild
    # modes they are bit-identical (tie = 0); on the ill-conditioned Kerr modes
    # the small difference is dominated by M1's Complex{Arb} scalar setup (the
    # Acb kernel uses the de-risked BigFloat scalar block), so we gate at GATE
    # rather than over-tightening to a recurrence floor that would false-fail.
    worst_tie = 0.0
    for (s, l, m, a, ω, _exp) in ARB_MODES
        ν_acb, _ = compute_nu(s, l, m, a, ω; precision=256, backend=:acb)
        ν_arb, _ = compute_nu(s, l, m, a, ω; precision=256, backend=:arb)
        d = Float64(nu_dist(ν_acb, ν_arb))
        worst_tie = max(worst_tie, d)
        @test d < GATE
    end
    @info "Acb-vs-M1Arb worst tie |ν_acb - ν_arb|" worst_tie
    @test worst_tie < GATE
end

@testset "Acb kernel-fidelity (identical λ, identical nmax) >=48 digits" begin
    # Strongest check: at IDENTICAL λ_bf and IDENTICAL truncation, the in-place
    # Acb value must reproduce the BigFloat monodromy_cos2pi_nu to >=48 digits.
    # This is the test that fails loudly if any in-place `!` op drops its `prec=`.
    NMAX = 1200
    worst_fid = 0.0
    for (s, l, m, a, ω, _exp) in ARB_MODES
        λbf = setprecision(BigFloat, 256) do
            B.MSTParams(s, l, m, BigFloat(a), Complex{BigFloat}(ω)).λ
        end
        cbf = setprecision(BigFloat, 256) do
            B.monodromy_cos2pi_nu(s, l, m, BigFloat(a),
                Complex{BigFloat}(ω), λbf; nmax=NMAX)
        end
        cacb = setprecision(Arb, 256) do
            ctx = B._build_monodromy_ctx_acb(s, l, m, BigFloat(a),
                      Complex{BigFloat}(ω), λbf, NMAX)
            Complex{BigFloat}(B._monodromy_value_acb(ctx, NMAX))
        end
        rel = Float64(abs(cacb - cbf) / abs(cbf))
        worst_fid = max(worst_fid, rel)
        @test rel < 1e-48
    end
    @info "Acb kernel-fidelity worst rel error (identical λ,nmax)" worst_fid
end

@testset "Acb kernel buffer integrity" begin
    # (1) value at the same n twice must be identical (deterministic, no state
    #     leak across local value-scratch); (2) march-forward extend must equal a
    #     fresh build to a fresh, deeper truncation.
    for (s, l, m, a, ω) in ((-2, 2, 2, 0.0, 0.5), (0, 2, 0, 0.9, 0.9))
        setprecision(Arb, 256) do
            p   = B.MSTParams(s, l, m, Arb(a),
                      Complex{Arb}(Arb(real(complex(ω))), Arb(imag(complex(ω)))))
            λ   = p.λ
            ctx = B._build_monodromy_ctx_acb(s, l, m, Arb(a),
                      Complex{Arb}(Arb(real(complex(ω))), Arb(imag(complex(ω)))),
                      λ, 300)
            v1 = B._monodromy_value_acb(ctx, 300)
            v2 = B._monodromy_value_acb(ctx, 300)
            d_repeat = Float64(setprecision(BigFloat, 256) do
                abs(Complex{BigFloat}(v1) - Complex{BigFloat}(v2))
            end)
            @test d_repeat == 0.0

            # march-forward from 100 -> 300 must equal a fresh build to 300
            ctx2 = B._build_monodromy_ctx_acb(s, l, m, Arb(a),
                       Complex{Arb}(Arb(real(complex(ω))), Arb(imag(complex(ω)))),
                       λ, 100)
            B._extend_monodromy_ctx_acb!(ctx2, 300)
            v_ext = B._monodromy_value_acb(ctx2, 300)
            d_ext = Float64(setprecision(BigFloat, 256) do
                abs(Complex{BigFloat}(v_ext) - Complex{BigFloat}(v1))
            end)
            @test d_ext < 1e-65
        end
    end
end
