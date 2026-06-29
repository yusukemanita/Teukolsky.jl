# Clean benchmark of the BigFloat ν monodromy solver.
# Warmup once, then minimum of N @elapsed for compute_nu(-2,2,2,a,ω).
using BHPtoolkit
using Printf

const S, L, M = -2, 2, 2

# (label, a, ω)
const CASES = [
    ("Schw", 0.0, 0.5),
    ("Kerr", 0.9, 0.5),
]
const PRECS = [128, 256, 512]
const N = 12   # number of timed repetitions; report the minimum

function time_case(prec, a, ω; n=N)
    # warmup (compile) once
    compute_nu(S, L, M, a, ω; precision=prec)
    best = Inf
    for _ in 1:n
        t = @elapsed compute_nu(S, L, M, a, ω; precision=prec)
        best = min(best, t)
    end
    return best
end

# Arb backend timing (drop-in Complex{Arb}); same protocol (warmup once, min-of-n).
function time_case_arb(prec, a, ω; n=N)
    compute_nu(S, L, M, a, ω; precision=prec, backend=:arb)   # warmup
    best = Inf
    for _ in 1:n
        t = @elapsed compute_nu(S, L, M, a, ω; precision=prec, backend=:arb)
        best = min(best, t)
    end
    return best
end

println("# nu_solver benchmark  (compute_nu(-2,2,2,a,ω), method=Monodromy)")
println("# Julia $(VERSION), N=$N min-of-N @elapsed, warmup once")
println("precision  case   a     omega   min_seconds")
for prec in PRECS
    for (lbl, a, ω) in CASES
        t = time_case(prec, a, ω)
        println(rpad("BF$prec", 10), rpad(lbl, 6), rpad(a, 6), rpad(ω, 8),
                @sprintf("%.6f", t))
    end
end

# ------------------------------------------------------------------
# Arb-256 drop-in vs BigFloat-256: same protocol, plus ratio BF256/Arb256.
# Honest expectation: the Complex{Arb} drop-in still allocates + carries ball-
# tracking overhead, so it is only ~1.9-2.3x faster than BigFloat at 256-bit,
# NOT the ~20x target (that needs the M2 native-Acb in-place kernel).
# ------------------------------------------------------------------
println()
println("# Arb-256 drop-in vs BigFloat-256  (backend=:arb), min-of-N @elapsed")
println("case   a     omega   BF256_s    Arb256_s   ratio(BF/Arb)")
for (lbl, a, ω) in CASES
    tbf  = time_case(256, a, ω)
    tarb = time_case_arb(256, a, ω)
    println(rpad(lbl, 6), rpad(a, 6), rpad(ω, 8),
            @sprintf("%-11.6f", tbf), @sprintf("%-11.6f", tarb),
            @sprintf("%.2fx", tbf / tarb))
end

# ------------------------------------------------------------------
# Allocation / GC profile for ONE BF256 ν solve (Schwarzschild).
# ------------------------------------------------------------------
function alloc_profile(prec, a, ω)
    compute_nu(S, L, M, a, ω; precision=prec)          # warmup
    GC.gc()
    stats = Base.@timed compute_nu(S, L, M, a, ω; precision=prec)
    nalloc = Base.gc_alloc_count(stats.gcstats)
    return (bytes=stats.bytes, nalloc=nalloc, gctime=stats.gctime, time=stats.time)
end

# Arb-256 allocation profile: exposes Complex{Arb} per-op allocation + ball
# re-boxing overhead vs BigFloat (the reason M1 caps at ~2x; M2 removes it).
function alloc_profile_arb(prec, a, ω)
    compute_nu(S, L, M, a, ω; precision=prec, backend=:arb)   # warmup
    GC.gc()
    stats = Base.@timed compute_nu(S, L, M, a, ω; precision=prec, backend=:arb)
    nalloc = Base.gc_alloc_count(stats.gcstats)
    return (bytes=stats.bytes, nalloc=nalloc, gctime=stats.gctime, time=stats.time)
end

println()
println("# Allocation profile, ONE BF256 solve (Schwarzschild a=0, omega=0.5)")
p = alloc_profile(256, 0.0, 0.5)
@printf("BF256 Schw : %d allocations, %.1f MiB, gctime %.4f s (of %.4f s)\n",
        p.nalloc, p.bytes/2^20, p.gctime, p.time)
println("Kerr a=0.9:")
pk = alloc_profile(256, 0.9, 0.5)
@printf("BF256 Kerr : %d allocations, %.1f MiB, gctime %.4f s (of %.4f s)\n",
        pk.nalloc, pk.bytes/2^20, pk.gctime, pk.time)

println()
println("# Allocation profile, ONE Arb256 solve (backend=:arb)")
pa = alloc_profile_arb(256, 0.0, 0.5)
@printf("Arb256 Schw: %d allocations, %.1f MiB, gctime %.4f s (of %.4f s)\n",
        pa.nalloc, pa.bytes/2^20, pa.gctime, pa.time)
pak = alloc_profile_arb(256, 0.9, 0.5)
@printf("Arb256 Kerr: %d allocations, %.1f MiB, gctime %.4f s (of %.4f s)\n",
        pak.nalloc, pak.bytes/2^20, pak.gctime, pak.time)
