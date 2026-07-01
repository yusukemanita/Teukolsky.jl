# ============================================================================
#  Backend performance benchmark for the MST radial quantities.
#
#  Times the renormalized angular momentum ν (compute_nu), the asymptotic
#  amplitudes Binc/Bref (compute_amplitudes — one call yields both), and the
#  radial solutions Rin, Rup (one evaluation each, given a precomputed ν/fn),
#  at ω = 0.1 and ω = 10, across the selectable precision backends.
#
#  Schwarzschild (a=0), s=-2, l=m=2, nmax=80, radial eval at r=10.
#  Protocol: warm up once (compile), then report the MINIMUM of N @elapsed runs
#  (single process — no parallelism, so the timings are not skewed by CPU
#  contention).  Arbitrary-precision backends run at `PREC` = 256 bits.
#
#  Run:  julia --project=. scripts/bench_backends.jl
# ============================================================================
using Teukolsky
using Printf

const S, L, M, A = -2, 2, 2, 0.0
const NMAX   = 80
const R_EVAL = 10.0
const N      = 8
const PREC   = 256

# ν solver: all five backends (Arb vs Acb genuinely DIFFER here — Acb is the
# native-kernel fast path).  NOTE: MultiFloats caps at Float64x4, so
# `precision=256` for :multifloat actually yields ~212 bits (≈63 digits), not 256
# (≈77 digits) like the genuine BigFloat/Arb/Acb columns — labelled accordingly.
const NU_BACKENDS = [
    ("Float64",       :float64,    64),
    ("BigFloat-256",  :bigfloat,   PREC),
    ("MultiFloat-x4", :multifloat, PREC),   # 256 → Float64x4, ~212-bit
    ("Arb-256",       :arb,        PREC),
    ("Acb-256",       :acb,        PREC),
]
# Amplitudes & radial: :arb and :acb run the IDENTICAL Complex{Arb} code path
# (the native-Acb kernel only touches compute_nu), so they are measured once and
# reported as a single "Arb/Acb" row — avoiding a misleading pair of timings that
# differ only by GC/system noise.
const TYPE_BACKENDS = [
    ("Float64",       :float64,    64),
    ("BigFloat-256",  :bigfloat,   PREC),
    ("MultiFloat-x4", :multifloat, PREC),
    ("Arb/Acb-256",   :arb,        PREC),
]
const OMEGAS = [0.1, 10.0]

# warm up once (compile), collect garbage, then take the MINIMUM of N timed runs
# (the min is the GC-free best estimate; the explicit gc + N=8 stabilise it for
# the allocation-heavy arbitrary-precision backends).
function minelapsed(f; n::Int=N)
    f()
    GC.gc()
    best = Inf
    for _ in 1:n
        best = min(best, @elapsed f())
    end
    return best
end

# Robust timing: a backend/quantity that is unsupported reports NaN (→ "n/a")
# instead of aborting the whole sweep.
function safe(f)
    try
        return minelapsed(f)
    catch
        return NaN
    end
end

bench_nu(B, P, ω)  = safe(() -> compute_nu(S, L, M, A, ω; backend=B, precision=P))
bench_amp(B, P, ω) = safe(() -> compute_amplitudes(S, L, M, A, ω;
                                                    backend=B, precision=P, nmax=NMAX))

# Rin/Rup: time ONE evaluation given a precomputed (ν, p, fn) in the backend's
# native float type.  _with_backend converts (a, ω) to the working type (and, for
# :arb/:acb, runs inside setprecision(Arb, P)); :acb is identical to :arb here
# (the native-Acb kernel only accelerates compute_nu).
function bench_radial(B, P, ω)
    try
        return Teukolsky._with_backend(B, P, A, ω) do a_w, ω_w
            ν, p = compute_nu(S, L, M, a_w, ω_w)      # type-driven, native backend
            fn   = compute_fn(p, ν; nmax=NMAX)
            r    = real(typeof(ω_w))(R_EVAL)
            trin = safe(() -> Rin(p, ν, fn, r; nmax=NMAX))
            trup = safe(() -> Rup(p, ν, fn, r; nmax=NMAX))
            return (trin, trup)
        end
    catch
        return (NaN, NaN)
    end
end

fmt(t) = isnan(t) ? @sprintf("%-10s", "n/a") :
         t < 1e-3 ? @sprintf("%7.1f µs", t*1e6) :
         t < 1.0  ? @sprintf("%7.2f ms", t*1e3) :
                    @sprintf("%7.3f s ", t)

println("# Backend performance — Schwarzschild a=0, s=-2, l=m=2, nmax=$NMAX, r=$R_EVAL")
println("# Julia $(VERSION), min-of-$N @elapsed, warm-up + GC excluded, single process")
for ω in OMEGAS
    println()
    println("ω = $ω")
    println("  -- ν solver (compute_nu) --")
    println("  ", rpad("backend", 16), "min time")
    for (lbl, B, P) in NU_BACKENDS
        println("  ", rpad(lbl, 16), fmt(bench_nu(B, P, ω)))
    end
    println("  -- amplitudes & radial  (Binc/Bref via compute_amplitudes; Rin, Rup at r=$R_EVAL) --")
    println("  ", rpad("backend", 16), rpad("Binc/Bref", 16), rpad("Rin", 16), "Rup")
    for (lbl, B, P) in TYPE_BACKENDS
        tamp = bench_amp(B, P, ω)
        trin, trup = bench_radial(B, P, ω)
        println("  ", rpad(lbl, 16), rpad(fmt(tamp), 16), rpad(fmt(trin), 16), fmt(trup))
    end
end
