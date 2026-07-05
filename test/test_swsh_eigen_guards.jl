# ============================================================
#  test_swsh_eigen_guards.jl — the _swsh_eigen correctness gate
#
#  The spheroidal branch is SELECTED in Float64 then refined to working precision;
#  a wrong-but-consistent λ used to pass silently (every downstream MST guard
#  checks series self-consistency, not λ).  The gate added to `_swsh_eigen`
#  (identity + residual + gap post-refine; escalate-or-error branch match) must:
#    (A) NOT false-positive on verified-correct physics — incl. strong mixing at
#        large imaginary c=aω where the dominant mixing component is NOT at ℓ′=ℓ;
#    (B) leave the a=0 (c=0) exact fast path untouched;
#    (C) actually DISCRIMINATE a genuine eigenpair from a wrong/garbage one.
# ============================================================
using Test
using Teukolsky
using LinearAlgebra
using Arblib: Arb

@testset "swsh_eigen correctness guards" begin

    # ── (A) no false-positive on verified-correct physics ────────────────────
    @testset "no false-positive: a=0.7 PIA sweep, BigFloat & Arb" begin
        for σ in (1.0, 5.0, 10.0, 16.0, 22.0), l in (2, 3, 4), Rt in (BigFloat, Arb)
            setprecision(Rt, 1024) do
                a = Rt(7)/Rt(10); ω = Complex{Rt}(Rt(0), Rt(σ))
                λ = Teukolsky.compute_lambda(-2, l, 2, a, ω)   # must NOT throw
                @test isfinite(abs(ComplexF64(λ)))
            end
        end
    end

    @testset "strong mixing (σ=22, dominant≠ℓ) passes the identity guard" begin
        setprecision(Arb, 1536) do
            a = Arb(7)/Arb(10); ω = Complex{Arb}(Arb(0), Arb(22))
            ells, C = swsh_coefficients(-2, 2, 2, a, ω; l_max = 0)   # must NOT throw
            u = ComplexF64.(C); il = 2 - first(ells) + 1
            @test abs(u[il + 1]) > abs(u[il])          # dominant component is NOT at ℓ′=ℓ
        end
    end

    # ── (B) a=0 fast path untouched ──────────────────────────────────────────
    @testset "a=0 Schwarzschild fast path (c=0, no gate)" begin
        for l in (2, 3, 4)
            λ = Teukolsky.compute_lambda(-2, l, 2, 0.0, 1.0 + 0im)
            @test λ ≈ l*(l + 1) - (-2)*(-2 + 1)        # exact spherical A_lm
        end
    end

    # ── (C) the gate predicates discriminate correct vs wrong ────────────────
    #  Replicates the EXACT expressions used in the gate on synthetic eigenpairs.
    @testset "residual predicate: eigenpair passes, non-eigenpair fires" begin
        N = 8; M = rand(ComplexF64, N, N) .* 15; M += 40 * I   # non-normal, ‖M‖ ~ 40
        F = eigen(M); Mscale = maximum(abs, F.values)
        restol = 64 * N * eps(Float64) * Mscale
        v = F.vectors[:, 3]; v ./= norm(v); μ = F.values[3]
        @test sum(abs2, M*v .- μ.*v) ≤ restol^2                     # true eigenpair → PASS
        vr = rand(ComplexF64, N); vr ./= norm(vr); μr = vr' * (M * vr)
        @test sum(abs2, M*vr .- μr.*vr) > restol^2                  # random pair → FIRE
    end

    @testset "identity predicate: same branch passes, neighbour fires" begin
        N = 8; F = eigen(rand(ComplexF64, N, N))
        s1 = F.vectors[:, 1]; s1 ./= norm(s1)
        s2 = F.vectors[:, 2]; s2 ./= norm(s2)
        @test abs(dot(s1, s1)) ≥ 0.9                               # refined≈seed → PASS
        @test abs(dot(s1, s2)) < 0.9                               # hopped to neighbour → FIRE
    end

    @testset "gap predicate: coalescence fires, distinct-vector degeneracy passes" begin
        # Defective (Jordan block for λ=2): eigenvectors coalesce → cross≈1 → FIRE
        J = ComplexF64[2 1 0; 0 2 0; 0 0 5]; Fj = eigen(J)
        Mscale = maximum(abs, Fj.values); gaptol = 64 * 3 * eps(Float64) * Mscale
        idx = argmin(abs.(Fj.values .- 2))
        gap = minimum(abs(Fj.values[idx] - Fj.values[j]) for j in 1:3 if j != idx)
        vf = Fj.vectors[:, idx]; vf ./= norm(vf)
        cross = maximum(abs(dot(Fj.vectors[:, j], vf)) for j in 1:3 if j != idx)
        @test gap ≤ gaptol                                         # degenerate
        @test cross ≥ 0.7                                          # vectors coalesce → FIRE
        # Distinct eigenvectors, equal eigenvalues (diagonal) → PASSES (anti-over-erroring)
        D = ComplexF64[2 0 0; 0 2 0; 0 0 5]; Fd = eigen(D)
        vd = Fd.vectors[:, 1]; vd ./= norm(vd)
        crossd = maximum(abs(dot(Fd.vectors[:, j], vd)) for j in 1:3 if j != 1)
        @test crossd < 0.7                                         # orthogonal → PASS despite gap≈0
    end

    # ── end-to-end: the gate actually THROWS in situ (not just the expressions) ──
    #  Feed a coarse-precision `a` (64-bit ball) into a 1024-bit solve: the coarse
    #  ball radius propagates into MR and inflates the residual above restol, so
    #  the residual certificate must refuse (converge-or-error) — a real throw
    #  through _swsh_eigen, exercising the integrated gate, not a stand-in.
    @testset "in-situ throw: coarse-input a trips the residual certificate" begin
        a_coarse = setprecision(Arb, 64) do; Arb(7) / Arb(10); end
        setprecision(Arb, 1024) do
            ω = Complex{Arb}(Arb(0), Arb(16))
            @test_throws ErrorException Teukolsky.compute_lambda(-2, 2, 2, a_coarse, ω)
        end
        # control: same solve with a built at working precision must NOT throw
        setprecision(Arb, 1024) do
            a = Arb(7) / Arb(10); ω = Complex{Arb}(Arb(0), Arb(16))
            @test isfinite(abs(ComplexF64(Teukolsky.compute_lambda(-2, 2, 2, a, ω))))
        end
    end
end
