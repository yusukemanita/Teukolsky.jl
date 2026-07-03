# ============================================================
#  Certified HU/dHU evaluation for R^up (hypergeometric.jl rewrite)
#
#  TRUTH ARBITER (algorithm-independent of the evaluator under test):
#  per-n DIRECT Arblib.hypgeom_u! with the precision escalated until the
#  RIGOROUS result ball certifies rel_accuracy_bits ≥ target.  This is
#  self-validating — Arb's radii are rigorous error bounds, so the truth
#  is never trusted below the certified accuracy.  (Beware: at PIA
#  frequencies c = −2iẑ is real positive and acb_hypgeom_u loses
#  ~|c|·(1+cosarg c) bits INTERNALLY at every precision — a truth harness
#  at a fixed "prec+128" is silently garbage at large σ.  The escalation +
#  radius check makes that impossible here; this bit two earlier analysis
#  rounds in this repo.)
#
#  WHAT IS UNDER TEST (see "Certified HU / dHU evaluation" in
#  src/hypergeometric.jl):
#   * seeds HU[0], HU[1], dHU[0], dHU[1] certified by precision escalation;
#   * all other n marched OUTWARD by the 3-term recurrence in point
#     (radius-stripped) arithmetic — measured uniformly stable;
#   * ComplexF64 contamination shadow + certified refresh as safety net
#     (exercised here by the exact-integer-ν recurrence-denominator poles).
#
#  Grid includes the hostile regimes of the mandate: large-ε PIA
#  (ω = iσ, σ up to 16), EXACT resonances 4σ ∈ ℤ (σ = 12, 16 ARE resonant),
#  near-integer real ν, near-integer bU = 2ν+2 (integer/near-integer ν),
#  and complex angles θ ∈ {0°, 60°} at |ω| = 10.
# ============================================================
using Test
using Teukolsky
const TK = Teukolsky
using Arblib
using Arblib: Arb, Acb

# ---------- rigorous truth helpers ----------

# HU[n] = c^n U(n+aU, 2n+bU, c) certified to ≥ target bits (rigorous ball).
function _truth_hu(hp, n::Int, target::Int)
    c64 = TK._mid_c64(hp.c)
    p = target + 128 + ceil(Int, 1.2 * (abs(c64) + max(real(c64), 0.0)))
    for _ in 1:6
        aA = TK._acb_at(hp.aU, p); bA = TK._acb_at(hp.bU, p); cA = TK._acb_at(hp.c, p)
        U = Acb(0; prec=p)
        Arblib.hypgeom_u!(U, aA + n, bA + 2n, cA; prec=p)
        H = Acb(0; prec=p)
        Arblib.pow!(H, cA, n; prec=p)
        Arblib.mul!(H, H, U; prec=p)
        acc = Int(Arblib.rel_accuracy_bits(H))
        if acc >= target
            return setprecision(BigFloat, p) do
                Complex{BigFloat}(BigFloat(Arb(real(H))), BigFloat(Arb(imag(H))))
            end
        end
        p += (target - acc) + 256
    end
    error("truth escalation failed at n=$n")
end

# dHU[n] = -2i[(n/c)·HU[n] − c^n (aU+n) U(1+aU+n, 1+bU+2n, c)], both U's
# certified.  The second term is _truth_hu on the (aU+1, bU+1) shifted family.
function _truth_dhu(hp, n::Int, target::Int)
    htr = _truth_hu(hp, n, target)
    u2 = _truth_hu(TK.HUParams(hp.aU + 1, hp.bU + 1, hp.c), n, target)
    cB = Complex{BigFloat}(BigFloat(real(hp.c)), BigFloat(imag(hp.c)))
    aB = Complex{BigFloat}(BigFloat(real(hp.aU)), BigFloat(imag(hp.aU)))
    return -2im * ((n / cB) * htr - (aB + n) * u2)
end

_tobig(z::Complex{Arb}) = Complex{BigFloat}(BigFloat(real(z)), BigFloat(imag(z)))
_tobig(z::Complex{BigFloat}) = z
_relerr(v, t) = Float64(abs(v - t) / abs(t))

# physical ν(σ) at a=0.7, s=-2, l=m=2 (from compute_nu at 768 bits; the HU
# harness only needs representative values, exactness is irrelevant)
const NU_PHYS = Dict(
    0.5  => (1.5912566836, 0.0497788413),
    2.0  => (1.8896095676, 0.2724972484),
    5.0  => (1.5296978180, 0.6968265293),
    8.0  => (1.7548920315, 1.3742866899),
    12.0 => (1.9879422153, 2.2573173228),   # 4σ = 48 ∈ ℤ (resonant)
    16.0 => (1.8324177387, -3.1088703301))  # 4σ = 64 ∈ ℤ (resonant)

function _mk_hp(σre, σim, r, νt, prec)
    p = MSTParams(-2, 2, 2, Arb(7)/10, Complex{Arb}(Arb(σre), Arb(σim)))
    ν = Complex{Arb}(Arb(νt[1]), Arb(νt[2]))
    zhat = complex(p.ϵ * (Arb(r) - p.rm) / 2)
    return TK.HUParams(p, ν, zhat)
end

@testset "HU march vs rigorous truth — PIA grid (incl. resonant 4σ∈ℤ)" begin
    for σ in (0.5, 5.0, 8.0, 12.0, 16.0), r in (4.0, 10.0), prec in (256, 512)
        (prec == 512 && !(σ in (5.0, 16.0))) && continue
        setprecision(Arb, prec) do
            setprecision(BigFloat, prec + 64) do
                hp = _mk_hp(0.0, σ, r, NU_PHYS[σ], prec)
                get_hu, _ = TK._hu_dhu_evaluators(hp)
                worst = 0.0
                for n in vcat(0:40, -1:-1:-40)
                    worst = max(worst, _relerr(_tobig(get_hu(n)), _truth_hu(hp, n, prec + 32)))
                end
                @test worst <= 2.0^(-0.9 * prec)
            end
        end
    end
    # arbitrary complex ν (decoupled from compute_nu), σ resonant
    setprecision(Arb, 256) do
        setprecision(BigFloat, 320) do
            hp = _mk_hp(0.0, 12.0, 10.0, (1.5, -0.47), 256)
            get_hu, _ = TK._hu_dhu_evaluators(hp)
            worst = 0.0
            for n in vcat(0:40, -1:-1:-40)
                worst = max(worst, _relerr(_tobig(get_hu(n)), _truth_hu(hp, n, 288)))
            end
            @test worst <= 2.0^(-0.9 * 256)
        end
    end
end

@testset "HU march vs rigorous truth — complex angles θ" begin
    for θ in (0.0, 60.0), r in (4.0, 10.0)
        setprecision(Arb, 256) do
            setprecision(BigFloat, 320) do
                ω = 10.0 * cis(deg2rad(θ))
                hp = _mk_hp(real(ω), imag(ω), r, (1.7, 0.9), 256)
                get_hu, _ = TK._hu_dhu_evaluators(hp)
                worst = 0.0
                for n in vcat(0:40, -1:-1:-40)
                    worst = max(worst, _relerr(_tobig(get_hu(n)), _truth_hu(hp, n, 288)))
                end
                @test worst <= 2.0^(-0.9 * 256)
            end
        end
    end
end

@testset "dHU march vs rigorous truth" begin
    for σ in (2.0, 16.0)
        setprecision(Arb, 256) do
            setprecision(BigFloat, 320) do
                hp = _mk_hp(0.0, σ, 10.0, NU_PHYS[σ], 256)
                _, get_dhu = TK._hu_dhu_evaluators(hp)
                worst = 0.0
                for n in vcat(0:30, -1:-1:-30)
                    worst = max(worst, _relerr(_tobig(get_dhu(n)), _truth_dhu(hp, n, 304)))
                end
                @test worst <= 2.0^(-0.9 * 256)
            end
        end
    end
end

@testset "hazards: integer/near-integer ν (bU ∈ ℤ, recurrence denominator poles)" begin
    # ν = 2 exactly: bU = 6 ∈ ℤ (hypgeom_u limit branch in the seeds) AND the
    # hu_down denominator (bU-aU+n) = 0 EXACTLY at n = -1 for PIA σ=2 — the
    # marched value is Inf/NaN and the certified-refresh safety net MUST fire.
    # ν = 2 + 1e-25: same denominators at distance 1e-25 (catastrophic
    # single-step cancellation, q ≫ 2^24 trip).  ν = 1.5 + 1e-30: bU within
    # 2e-30 of 5 (near-integer-b seed regime).
    for (νt, σ) in (((2.0, 0.0), 2.0), ((2.0 + 1e-25, 0.0), 2.0), ((1.5 + 1e-30, 0.0), 8.0))
        setprecision(Arb, 256) do
            setprecision(BigFloat, 320) do
                hp = _mk_hp(0.0, σ, 10.0, νt, 256)
                get_hu, _ = TK._hu_dhu_evaluators(hp)
                worst = 0.0
                for n in vcat(0:30, -1:-1:-30)
                    worst = max(worst, _relerr(_tobig(get_hu(n)), _truth_hu(hp, n, 288)))
                end
                @test worst <= 2.0^(-0.9 * 256)
            end
        end
    end
end

# ---------- end-to-end Rup / dRup against an independent direct sum ----------
#
# Reference: Σ_n prefac · (-1)^n (aw)_n/(bw)_n · fn[n] · HU_truth[n] / ctrans
# evaluated in Complex{BigFloat} at (prec+192) from the SAME (p, ν, fn, ctrans)
# inputs, with HU_truth from the rigorous harness — independent of the
# radial_up marching/caching code under test.
function _rup_reference(p, ν, fn, ct, r, nmax, prec; deriv::Bool=false)
    setprecision(BigFloat, prec + 192) do
        ϵ = _tobig(p.ϵ); κ = _tobig(complex(p.κ)); τ = _tobig(complex(p.τ))
        s = p.s; rm = BigFloat(p.rm)
        νB = _tobig(ν)
        zhat = ϵ * (BigFloat(r) - rm) / 2
        zmek = zhat - ϵ*κ
        A = 2^νB * exp(-BigFloat(π)*ϵ) * exp(-im*BigFloat(π)*(νB + 1)) * exp(-im*BigFloat(π)*s)
        pow_z = zhat^(νB + im*(ϵ + τ)/2)
        pow_zmek = zmek^(-im*(ϵ + τ)/2 - s)
        exp_z = exp(im*zhat)
        prefac = A * exp_z * pow_z * pow_zmek
        α_z = νB + im*(ϵ + τ)/2
        β_zmek = -im*(ϵ + τ)/2 - s
        dprefac = A * (im*exp_z*pow_z*pow_zmek + exp_z*(α_z/zhat)*pow_z*pow_zmek +
                       exp_z*pow_z*(β_zmek/zmek)*pow_zmek) * (ϵ/2)
        hp = TK.HUParams(p, ν, complex(p.ϵ * (Arb(r) - p.rm) / 2))
        aw = νB + 1 + s - im*ϵ
        bw = νB + 1 - s + im*ϵ
        acc = zero(Complex{BigFloat})
        maxterm = zero(BigFloat)
        w = one(Complex{BigFloat})
        for n in 0:nmax
            n > 0 && (w = -w * (aw + (n-1)) / (bw + (n-1)))
            fB = _tobig(fn[n])
            iszero(fB) && continue
            h = _truth_hu(hp, n, prec + 64)
            t = deriv ? fB * w * (dprefac * h + prefac * (ϵ/2) * _truth_dhu(hp, n, prec + 64)) :
                        prefac * w * fB * h
            acc += t
            maxterm = max(maxterm, abs(t))
        end
        w = one(Complex{BigFloat})
        for n in -1:-1:-nmax
            w = -w * (bw + n) / (aw + n)
            fB = _tobig(fn[n])
            iszero(fB) && continue
            h = _truth_hu(hp, n, prec + 64)
            t = deriv ? fB * w * (dprefac * h + prefac * (ϵ/2) * _truth_dhu(hp, n, prec + 64)) :
                        prefac * w * fB * h
            acc += t
            maxterm = max(maxterm, abs(t))
        end
        # κ = max|term|/|Σ|: intrinsic cancellation of the MST radial sum
        # (≈ 1 at PIA frequencies, ≈ 2^78 at real ω = 10) — per-n HU errors
        # are amplified by up to κ REGARDLESS of how HU is evaluated.
        return acc / _tobig(ct), Float64(maxterm / abs(acc))
    end
end

@testset "end-to-end Rup/dRup vs independent direct-sum reference" begin
    for (σ, prec, nmax) in ((4.3, 320, 50), (8.0, 448, 50))
        setprecision(Arb, prec) do
            νt = σ == 4.3 ? (1.8843104008, 0.5030358956) : NU_PHYS[σ]
            p = MSTParams(-2, 2, 2, Arb(7)/10, Complex{Arb}(Arb(0), Arb(σ)))
            ν = Complex{Arb}(Arb(νt[1]), Arb(νt[2]))
            fn = compute_fn(p, ν; nmax=nmax)
            ct = TK._ctrans(p, TK.compute_Aminus(p, ν, fn; nmax=nmax))
            rv = Rup(p, ν, fn, Arb(10); nmax=nmax, ctrans=ct, tol=0)
            dv = dRup(p, ν, fn, Arb(10); nmax=nmax, ctrans=ct, tol=0)
            rref, κr = _rup_reference(p, ν, fn, ct, 10, nmax, prec)
            dref, κd = _rup_reference(p, ν, fn, ct, 10, nmax, prec; deriv=true)
            er = _relerr(_tobig(rv), rref)
            ed = _relerr(_tobig(dv), dref)
            @info "end-to-end σ=$σ prec=$prec" Rup_err=er dRup_err=ed κ_sum=κr
            @test er <= max(κr, 1.0) * 2.0^(-0.88 * prec + 8)
            @test ed <= max(κd, 1.0) * 2.0^(-0.88 * prec + 8)
        end
    end
    # real ω (θ=0), arbitrary ν
    setprecision(Arb, 256) do
        p = MSTParams(-2, 2, 2, Arb(7)/10, Complex{Arb}(Arb(10), Arb(0)))
        ν = Complex{Arb}(Arb(1.7), Arb(0.9))
        fn = compute_fn(p, ν; nmax=50)
        # exact-window A− (nmin passed): at real ω=10 / 256 bits the A− sum
        # cancels ~1e39, so the adaptive path correctly REFUSES to certify
        # √eps here — but ct divides both rv and rref identically below, so
        # its floor-limited accuracy cancels out of the comparison.
        ct = TK._ctrans(p, TK.compute_Aminus(p, ν, fn; nmax=50, nmin=-50))
        rv = Rup(p, ν, fn, Arb(10); nmax=50, ctrans=ct, tol=0)
        rref, κr = _rup_reference(p, ν, fn, ct, 10, 50, 256)
        er = _relerr(_tobig(rv), rref)
        @info "end-to-end real ω=10" Rup_err=er κ_sum=κr
        # at real ω the MST radial sum itself cancels ~κ (≈2^78 here); the
        # per-n certified-HU errors are amplified by up to κ intrinsically
        @test er <= max(κr, 1.0) * 2.0^(-0.88 * 256 + 8)
    end
end

@testset "BigFloat backend parity" begin
    # Complex{BigFloat} runs the same certified path (Acb-backed seeds); its
    # Rup must agree with the Complex{Arb} value at the same precision.
    for (σ, prec) in ((5.0, 320),)
        νt = NU_PHYS[σ]
        va = setprecision(Arb, prec) do
            p = MSTParams(-2, 2, 2, Arb(7)/10, Complex{Arb}(Arb(0), Arb(σ)))
            ν = Complex{Arb}(Arb(νt[1]), Arb(νt[2]))
            fn = compute_fn(p, ν; nmax=50)
            ct = TK._ctrans(p, TK.compute_Aminus(p, ν, fn; nmax=50))
            _tobig(Rup(p, ν, fn, Arb(10); nmax=50, ctrans=ct))
        end
        vb = setprecision(BigFloat, prec) do
            p = MSTParams(-2, 2, 2, BigFloat(7)/10, Complex{BigFloat}(BigFloat(0), BigFloat(σ)))
            ν = Complex{BigFloat}(BigFloat(νt[1]), BigFloat(νt[2]))
            fn = compute_fn(p, ν; nmax=50)
            ct = TK._ctrans(p, TK.compute_Aminus(p, ν, fn; nmax=50))
            Rup(p, ν, fn, BigFloat(10); nmax=50, ctrans=ct)
        end
        @test _relerr(va, Complex{BigFloat}(vb)) < 2.0^(-0.8 * prec)
    end
end

@testset "legacy Float64 path (generic backends unchanged)" begin
    # Float64 keeps the pre-rewrite guard scheme; sanity vs a 320-bit value.
    p64 = MSTParams(-2, 2, 2, 0.7, Complex(0.0, 0.5))
    ν64 = Complex(0.5, 0.30)
    fn64 = compute_fn(p64, ν64; nmax=40)
    r64 = Rup(p64, ν64, fn64, 10.0; nmax=40)
    rhi = setprecision(Arb, 320) do
        p = MSTParams(-2, 2, 2, Arb(7)/10, Complex{Arb}(Arb(0), Arb(1)/2))
        ν = Complex{Arb}(Arb(1)/2, Arb(3)/10)
        fn = compute_fn(p, ν; nmax=40)
        ComplexF64(_tobig(Rup(p, ν, fn, Arb(10); nmax=40)))
    end
    @test abs(r64 - rhi) / abs(rhi) < 1e-8
end
