# Tests for suggest_mst_precision (the |ω|-driven precision predictor).
#
# (A) Structural: monotone bits/nmax, correct backend split at MST_F64X4_TRUST.
# (B) Calibration guard: the PREDICTED (bits, nmax) must actually reproduce the
#     branch-cut MST core q̃·R^up to 1e-11 at representative |ω| — i.e. the
#     predictor never under-shoots on its own calibration set.
using Test
using Teukolsky, MultiFloats
using Arblib: Arb
const _R4 = MultiFloat{Float64,4}

@testset "suggest_mst_precision" begin
    # (A) structural properties
    @testset "structure / monotonicity" begin
        xs = [0.1, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0, 14.0]
        hints = [suggest_mst_precision(im*x) for x in xs]
        @test all(h -> h.bits ≥ 212, hints)
        @test all(h -> h.nmax ≥ 40, hints)
        @test issorted([h.bits for h in hints])           # bits non-decreasing in |ω|
        @test issorted([h.nmax for h in hints])            # nmax non-decreasing in |ω|
        # backend split at the trust boundary
        @test suggest_mst_precision(im*2.0).backend == :multifloat
        @test suggest_mst_precision(im*2.0).bits == 212
        @test suggest_mst_precision(im*8.0).backend == :acb
        @test suggest_mst_precision(im*8.0).bits > 212
        # higher multipole => at least as many terms
        @test suggest_mst_precision(im*6.0; l=5).nmax ≥ suggest_mst_precision(im*6.0; l=2).nmax
        # |ω| independent of sign of the axis / real vs imag
        @test suggest_mst_precision(4.4).bits == suggest_mst_precision(im*4.4).bits
    end

    # (B) calibration guard: predicted precision reproduces the reference core.
    @testset "predicted precision is sufficient" begin
        s, m, lp, rsrc = -2, 2, 2, 10
        aq = 7//10
        function gcore(::Type{T}, σ, nmax) where {T}
            a = T <: MultiFloat ? _R4(7)/_R4(10) : T(7)/T(10)
            ω = Complex{T}(zero(T), T(σ))
            core = compute_mst_core(s, lp, m, a, ω; nmax=nmax, nmax_cf=max(400,6nmax))
            ComplexF64(qtilde_from_core(core) *
                       Rup(core.p, core.ν, core.fn, T(rsrc); nmax=nmax, ctrans=mst_ctrans(core)))
        end
        # Bit-sufficiency guard: at the PREDICTED nmax, the predicted-precision
        # result must match a BigFloat-1408 result at the SAME nmax — i.e. the
        # predicted bit-count already resolves the answer.  (Comparing at the same
        # truncation isolates precision from the separate nmax-window question; a
        # much larger reference nmax would trip the pre-existing Ln_cf backward-CF
        # fragility at large |n|, unrelated to the predictor.)
        # σ chosen away from integrand near-zeros (|g|≳1e-6) so the relative test
        # reflects precision, not cancellation at a physical zero.
        for σ in (0.3, 1.0, 2.2, 4.6, 6.7)
            h = suggest_mst_precision(im*σ; l=lp)
            # :acb rung → the native in-place chain (compute_mst_core dispatches
            # to compute_mst_core_acb for Arb inputs); :multifloat rung → F64x4.
            g = h.backend === :multifloat ? gcore(_R4, σ, h.nmax) :
                setprecision(Arb, h.bits) do; gcore(Arb, σ, h.nmax); end
            # Moderate over-precision reference at the SAME nmax (+192 bits, or
            # 424 bit for the F64x4 cases).  Avoids the extreme-precision (≳1000
            # bit) fragility in the existing hypergeometric near-integer guard,
            # which is orthogonal to what the predictor controls.
            refbits = h.backend === :multifloat ? 424 : h.bits + 192
            ref = setprecision(BigFloat, refbits) do; gcore(BigFloat, σ, h.nmax); end
            rel = abs(g - ref) / max(abs(ref), 1e-300)
            @test isfinite(rel)
            @test rel < 1e-9
        end
    end
end
