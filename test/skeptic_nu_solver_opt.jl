# Skeptic verification of the optimized BigFloat ν monodromy solver.
# Tries to prove the OPTIMIZED ν is wrong or less accurate than before.
using BHPtoolkit
using Printf
const B = BHPtoolkit

# ----------------------------------------------------------------------
# Independent reference: the PRE-CHANGE monodromy_cos2pi_nu, extracted
# verbatim from git commit 7418dff^ (mathematically the brute force).
# Renamed ref_mono so it cannot accidentally call the optimized path.
# ----------------------------------------------------------------------
function ref_mono(s, _l, m, a, ω, λ; nmax::Int=3000)
    R = promote_type(typeof(float(real(a))), typeof(float(real(complex(ω)))))
    C = Complex{R}

    q  = R(a)
    ε  = 2 * C(ω)
    κ  = sqrt(C(1 - q^2))
    τ  = (ε - m*q) / κ

    γCH = 1 - s - im*ε - im*τ
    δCH = 1 + s + im*ε - im*τ
    εCH = 2im*ε*κ
    αε  = C(1 - s) + im*(ε - τ)
    qCH = s*(s+1) - ε^2 + im*(1-2s)*ε*κ + λ + im*τ + τ^2

    μ1C = αε - (γCH + δCH)
    μ2C = -αε

    a1 = zeros(C, nmax + 2)
    a2 = zeros(C, nmax + 2)
    a1[1] = one(C); a2[1] = one(C)

    for n in 1:nmax
        a1p = a1[n];  a1pp = n >= 2 ? a1[n-1] : zero(C)
        c2 = (αε - (n-1+δCH)) * (αε - (n-2+γCH+δCH)) * εCH / n
        c1 = (αε^2 + αε*(1-2n-γCH-δCH+εCH) +
              (n^2 - qCH + n*(-1+γCH+δCH-εCH) + εCH - δCH*εCH)) / n
        a1[n+1] = c2*a1pp - c1*a1p

        a2p = a2[n];  a2pp = n >= 2 ? a2[n-1] : zero(C)
        d2 = (αε + (n-2)) * (αε + (n-1-γCH)) * εCH / n
        d1 = (αε^2 + (n^2 - qCH + γCH + δCH - n*(1+γCH+δCH-εCH) - εCH) +
              αε*(-1+2n-γCH-δCH+εCH)) / n
        a2[n+1] = -d2*a2pp + d1*a2p
    end

    Poch_p = ones(C, nmax + 2)
    Poch_m = ones(C, nmax + 2)
    for i in 1:nmax
        Poch_p[i+1] = (-μ2C + μ1C + i - 1) * Poch_p[i]
        Poch_m[i+1] = ( μ2C - μ1C + i - 1) * Poch_m[i]
    end

    n    = nmax
    jmax = cld(n, 2)
    a1sum = B._cgamma(-μ2C + μ1C) * sum(a1[j+1] * Poch_p[n-j+1] for j in 0:jmax)
    a2sum = B._cgamma( μ2C - μ1C) * sum((-1)^j * a2[j+1] * Poch_m[n-j+1] for j in 0:jmax)

    return cos(π*(μ1C - μ2C)) + (2*R(π)^2 / (a1sum * a2sum)) * (-1)^(n-1) * a1[n+1] * a2[n+1]
end

# Replicate the EXACT branch formula from _compute_nu_monodromy so that we can
# turn a reference cos(2πν) into ν the same way the solver does.
function branch_nu(c2pn, l, ω, ::Type{R}) where {R}
    rc   = real(c2pn)
    twoπ = 2 * R(π)
    if imag(complex(ω)) != 0
        return R(l) - acos(complex(c2pn)) / twoπ
    elseif -1 ≤ rc ≤ 1
        return R(l) - acos(complex(rc)) / twoπ
    elseif rc < -1
        return Complex(R(1) / 2, +acosh(-rc) / twoπ)
    else
        return Complex(R(0), -acosh(rc) / twoπ)
    end
end

# Symmetric ν distance (ν and -ν-1 and conjugates are physically equivalent).
nu_dist(a, b) = minimum(abs(a - c) for c in (b, -b - 1, conj(b), conj(-b - 1)))

println("="^90)
println("PART (a): compute_nu vs brute-force ref_mono(nmax=3000), BigFloat-256, all four branches")
println("="^90)

cases = NamedTuple[]
for a in (0.0, 0.5, 0.9, -0.7), ω in (0.1, 0.3, 0.5, 1.0, 0.5 - 0.01im), l in (2, 3)
    push!(cases, (s=-2, l=l, m=2, a=a, ω=ω))
end
# add a couple of s=0 modes too
for ω in (0.3, 1.0, 0.5 - 0.01im)
    push!(cases, (s=0, l=2, m=0, a=0.0, ω=ω))
end

worst_c   = Ref(0.0)   # worst rel error of cos(2πν): adaptive vs ref(3000)
worst_nu  = Ref(0.0)   # worst rel error of ν: solver vs ref-branch
worst_wrap = Ref(0.0)  # worst rel error wrapper(3000) vs independent ref(3000) (should be ~0)
branch_seen = Dict("real"=>0, "half"=>0, "integer"=>0, "complex"=>0)

setprecision(BigFloat, 256) do
    tolwp = 1e-70   # ~ working precision for 256-bit BigFloat (~77 dec digits)
    for cs in cases
        s,l,m = cs.s, cs.l, cs.m
        aB = BigFloat(real(cs.a))
        ωB = Complex{BigFloat}(cs.ω)
        p  = B.MSTParams(s, l, m, aB, ωB)
        λ  = p.λ

        c_adapt = B._monodromy_adaptive(s, l, m, aB, ωB, λ; R=BigFloat, nmax0=60)
        c_ref   = ref_mono(s, l, m, aB, ωB, λ; nmax=3000)            # independent brute force
        c_wrap  = B.monodromy_cos2pi_nu(s, l, m, aB, ωB, λ; nmax=3000) # current wrapper @3000

        ec = abs(c_adapt - c_ref) / abs(c_ref)
        ew = abs(c_wrap  - c_ref) / max(abs(c_ref), eps(BigFloat))

        ν_solver, _ = B.compute_nu(s, l, m, aB, ωB; precision=256)
        ν_ref       = branch_nu(c_ref, l, ωB, BigFloat)
        eν = nu_dist(ν_solver, ν_ref) / max(abs(ν_ref), eps(BigFloat))

        # classify branch
        rc = real(c_ref)
        btype = imag(ωB) != 0 ? "complex" : (-1 ≤ rc ≤ 1 ? "real" : (rc < -1 ? "half" : "integer"))
        branch_seen[btype] += 1

        worst_c[]    = max(worst_c[], Float64(ec))
        worst_nu[]   = max(worst_nu[], Float64(eν))
        worst_wrap[] = max(worst_wrap[], Float64(ew))

        flag = (ec > tolwp || eν > tolwp) ? "  <-- REGRESSION?" : ""
        @printf("s%2d l%d m%d a%+.1f ω=%-9s [%-7s] ec=%.2e  eν=%.2e  ewrap=%.2e%s\n",
                s,l,m,Float64(real(cs.a)), string(cs.ω), btype,
                Float64(ec), Float64(eν), Float64(ew), flag)
    end
end
@printf("\nbranches exercised: %s\n", branch_seen)
@printf("worst rel err  cos(2πν) adaptive-vs-ref(3000): %.3e\n", worst_c[])
@printf("worst rel err  ν solver-vs-ref-branch        : %.3e\n", worst_nu[])
@printf("worst rel err  wrapper(3000)-vs-indep-ref     : %.3e  (must be ~0: confirms reference unchanged)\n", worst_wrap[])

println()
println("="^90)
println("PART (b): cross-check vs Mathematica reference test/wolfram_ref_hp.txt")
println("="^90)
pb(x) = parse(BigFloat, x)
pc(re, im) = (re == "ERR" || im == "ERR") ? nothing : Complex{BigFloat}(pb(re), pb(im))
REF = joinpath(@__DIR__, "wolfram_ref_hp.txt")
worst_wolfram = Ref(0.0)
setprecision(BigFloat, 256) do
    for ln in readlines(REF)
        isempty(strip(ln)) && continue
        f = split(strip(ln), ";")
        s = parse(Int,f[1]); l = parse(Int,f[2]); m = parse(Int,f[3])
        a = pb(f[4]); om = pb(f[5])
        νr = pc(f[8], f[9])
        νr === nothing && continue
        ω = Complex{BigFloat}(om)
        ν, _ = B.compute_nu(s, l, m, a, ω; precision=256)
        eν = nu_dist(ν, νr)
        worst_wolfram[] = max(worst_wolfram[], Float64(eν))
        @printf("s%2d l%d m%d a%.1f ω%.2f  |ν-ν_wolfram| = %.3e\n",
                s,l,m,Float64(a),Float64(om), Float64(eν))
    end
end
@printf("worst |ν - ν_wolfram| = %.3e (test gate was eν < 1e-13)\n", worst_wolfram[])

println()
println("="^90)
println("PART (c): edge ω (0.05 small, 2.0 large) — convergence & verify-and-extend trigger")
println("="^90)
setprecision(BigFloat, 256) do
    for (a, ω) in ((0.0, 0.05), (0.9, 0.05), (0.0, 2.0), (0.9, 2.0))
        s,l,m = -2,2,2
        aB = BigFloat(a); ωB = Complex{BigFloat}(ω)
        p  = B.MSTParams(s,l,m,aB,ωB); λ = p.λ
        c_adapt = B._monodromy_adaptive(s,l,m,aB,ωB,λ; R=BigFloat, nmax0=60)
        c_ref   = ref_mono(s,l,m,aB,ωB,λ; nmax=3000)
        ec = abs(c_adapt - c_ref)/abs(c_ref)
        @printf("a%.1f ω=%.2f  ec(adaptive vs ref3000)=%.3e %s\n",
                a, ω, Float64(ec), ec > 1e-68 ? " <-- UNDER-TRUNCATION?" : "")
    end

    println("\n-- correctness of _extend_monodromy_ctx! (march-forward must equal fresh build) --")
    for (a, ω) in ((0.0, 2.0), (0.9, 1.0))
        s,l,m=-2,2,2
        aB=BigFloat(a); ωB=Complex{BigFloat}(ω)
        p=B.MSTParams(s,l,m,aB,ωB); λ=p.λ
        ctx = B._build_monodromy_ctx(s,l,m,aB,ωB,λ, 100)
        v100 = B._monodromy_value(ctx, 100)
        B._extend_monodromy_ctx!(ctx, 800)         # march forward
        v800_ext = B._monodromy_value(ctx, 800)
        ctx_fresh = B._build_monodromy_ctx(s,l,m,aB,ωB,λ, 800)
        v800_fresh = B._monodromy_value(ctx_fresh, 800)
        # also value at 100 from the deep fresh build
        v100_fresh = B._monodromy_value(ctx_fresh, 100)
        d_ext   = abs(v800_ext - v800_fresh)
        d_mid   = abs(v100 - v100_fresh)
        @printf("a%.1f ω%.1f  |extend(800)-fresh(800)|=%.3e (bit-id=%s) ; |val@100 reused-vs-fresh|=%.3e (bit-id=%s)\n",
                a, ω, Float64(d_ext), v800_ext===v800_fresh,
                Float64(d_mid), v100===v100_fresh)
    end

    println("\n-- verify-and-extend ACTUALLY triggers when started under-truncated --")
    # Replicate the adaptive loop but FORCE a too-small starting nmax, count extensions.
    function adaptive_traced(s,l,m,a,ω,λ; R, nmax_start)
        tol = 16*eps(R); Δ = 128
        nmax = nmax_start
        ctx = B._build_monodromy_ctx(s,l,m,a,ω,λ, nmax)
        c = B._monodromy_value(ctx, nmax)
        next = 0
        for _ in 1:64
            nlo = max(60, nmax-Δ)
            clo = B._monodromy_value(ctx, nlo)
            abs(c-clo) ≤ tol*abs(c) && return c, nmax, next
            nmax ≥ 4000 && return c, nmax, next
            nmax = min(nmax+2Δ, 4000); next += 1
            B._extend_monodromy_ctx!(ctx, nmax)
            c = B._monodromy_value(ctx, nmax)
        end
        return c, nmax, next
    end
    for (a, ω) in ((0.0, 0.5), (0.0, 2.0), (0.9, 2.0))
        s,l,m=-2,2,2
        aB=BigFloat(a); ωB=Complex{BigFloat}(ω)
        p=B.MSTParams(s,l,m,aB,ωB); λ=p.λ
        c_trace, nfin, next = adaptive_traced(s,l,m,aB,ωB,λ; R=BigFloat, nmax_start=130)
        c_ref = ref_mono(s,l,m,aB,ωB,λ; nmax=3000)
        ec = abs(c_trace-c_ref)/abs(c_ref)
        @printf("a%.1f ω%.1f start=130 -> extensions=%d final_nmax=%d ; ec vs ref=%.3e\n",
                a, ω, next, nfin, Float64(ec))
    end
end

println()
println("="^90)
println("PART (d): Float64 path must be byte-for-byte unchanged (pre-change vs current)")
println("="^90)
worst_f64_bits_differ = Ref(false)
worst_f64_nu = Ref(0.0)
for (s,l,m,a,ω) in ((-2,2,2,0.0,0.1),(-2,2,2,0.0,0.5),(-2,2,2,0.0,1.0),
                    (-2,2,2,0.9,0.5),(-2,2,2,0.9,1.0),(-2,3,2,0.5,1.0),
                    (0,2,0,0.0,1.0),(0,2,1,0.7,1.5),(-2,2,2,0.0,0.3),
                    (-2,2,2,0.7,0.3))
    p   = B.MSTParams(s,l,m,a,Complex{Float64}(ω))
    λ   = p.λ
    c_new = B.monodromy_cos2pi_nu(s,l,m,a,Complex{Float64}(ω),λ; nmax=60)   # current
    c_old = ref_mono(s,l,m,a,Complex{Float64}(ω),λ; nmax=60)                # pre-change
    bits_id = (reinterpret(UInt64, real(c_new)) == reinterpret(UInt64, real(c_old))) &&
              (reinterpret(UInt64, imag(c_new)) == reinterpret(UInt64, imag(c_old)))
    bits_id || (worst_f64_bits_differ[] = true)
    # also the full ν via compute_nu (Float64 default) vs branch formula on c_old
    ν_solver, _ = B.compute_nu(s,l,m,a,ω)
    ν_old = branch_nu(c_old, l, Complex{Float64}(ω), Float64)
    dν = nu_dist(ν_solver, ν_old)
    worst_f64_nu[] = max(worst_f64_nu[], dν)
    @printf("s%2d l%d m%d a%.1f ω%.1f  cos2πν bit-identical=%s ; |ν_solver-ν_old(branch)|=%.3e\n",
            s,l,m,a,Float64(ω), bits_id, dν)
end
@printf("\nFloat64 cos(2πν) any bits differ: %s ; worst |ν_solver-ν_old|=%.3e\n",
        worst_f64_bits_differ[], worst_f64_nu[])

println()
println("#"^90)
@printf("SUMMARY  worst_c(adaptive vs ref3000)=%.3e  worst_ν=%.3e  worst_wolfram=%.3e\n",
        worst_c[], worst_nu[], worst_wolfram[])
@printf("         wrapper-vs-indep-ref=%.3e  Float64 bits differ=%s  Float64 worst ν=%.3e\n",
        worst_wrap[], worst_f64_bits_differ[], worst_f64_nu[])
println("#"^90)
