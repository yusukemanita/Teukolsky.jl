# ============================================================
#  Performance comparison of the three precision backends
#  (:float64, :bigfloat, :multifloat) for the MST solver.
#
#  Protocol: warmup once (compile), then report the MINIMUM of N timed
#  @elapsed runs — min-of-N suppresses GC/scheduler noise and is the standard
#  for microbenchmarks.  We time both public entry points:
#     compute_nu         (renormalized angular momentum ν, monodromy)
#     compute_amplitudes (full Binc/Bref/Btrans/Ctrans pipeline)
#
#  Backends are compared at MATCHED mantissa width so the comparison is
#  apples-to-apples:
#     Float64      ≈  53 bits
#     Float64xN    ≈  53·N bits   (N = 1..4, the MultiFloats width ceiling)
#     BigFloat(p)  =   p bits     (p ∈ {53,106,159,212})
# ============================================================
using BHPtoolkit
using Printf

const S, L, M = -2, 2, 2
const CASES = [("Schw", 0.0, 0.5), ("Kerr", 0.9, 0.5)]
const N = 12   # timed repetitions; report the minimum

# bit width  ↔  (BigFloat precision, MultiFloat limb count)
const LEVELS = [(53, 1), (106, 2), (159, 3), (212, 4)]

bestof(f; n=N) = (f(); m = Inf; for _ in 1:n; m = min(m, @elapsed f()); end; m)

# ── timing closures per backend ─────────────────────────────────────────────
t_f64_nu(a, ω)   = bestof(() -> compute_nu(S, L, M, a, ω; backend=:float64))
t_bf_nu(p, a, ω) = bestof(() -> compute_nu(S, L, M, a, ω; backend=:bigfloat,   precision=p))
t_mf_nu(p, a, ω) = bestof(() -> compute_nu(S, L, M, a, ω; backend=:multifloat, precision=p))

t_f64_amp(a, ω)   = bestof(() -> compute_amplitudes(S, L, M, a, ω; backend=:float64))
t_bf_amp(p, a, ω) = bestof(() -> compute_amplitudes(S, L, M, a, ω; backend=:bigfloat,   precision=p))
t_mf_amp(p, a, ω) = bestof(() -> compute_amplitudes(S, L, M, a, ω; backend=:multifloat, precision=p))

ms(x) = @sprintf("%9.3f", 1e3 * x)   # seconds → ms, right-aligned

function run_table(title, t_f64, t_bf, t_mf)
    println("\n# ", title)
    println("# times in milliseconds (min-of-$N).  speedup = BigFloat / MultiFloat at equal width.")
    println(rpad("case", 7), rpad("bits", 6), rpad("Float64", 11), rpad("BigFloat", 11),
            rpad("MultiFloat", 12), "speedup(BF/MF)")
    for (lbl, a, ω) in CASES
        tf = t_f64(a, ω)
        for (bits, _) in LEVELS
            tb = t_bf(bits, a, ω)
            tm = t_mf(bits, a, ω)
            f64col = bits == LEVELS[1][1] ? ms(tf) : rpad("-", 9)
            println(rpad(lbl, 7), rpad(bits, 6),
                    rpad(f64col, 11), rpad(ms(tb), 11), rpad(ms(tm), 12),
                    @sprintf("%.2fx", tb / tm))
        end
    end
end

println("# MST precision-backend performance comparison")
println("# Julia $(VERSION) | mode (s,l,m)=($S,$L,$M) | warmup once, min-of-$N @elapsed")

run_table("compute_nu  (renormalized angular momentum ν)", t_f64_nu, t_bf_nu, t_mf_nu)
run_table("compute_amplitudes  (full Binc/Bref/Btrans/Ctrans pipeline)", t_f64_amp, t_bf_amp, t_mf_amp)

# ── allocation profile, ONE solve, at 212 bits (x4) Kerr ────────────────────
function prof(f)
    f(); GC.gc()
    st = Base.@timed f()
    (bytes=st.bytes, nalloc=Base.gc_alloc_count(st.gcstats), time=st.time)
end
println("\n# Allocation profile — ONE compute_amplitudes solve, Kerr a=0.9 ω=0.5, 212-bit width")
for (lbl, f) in [
        ("Float64   ", () -> compute_amplitudes(S, L, M, 0.9, 0.5; backend=:float64)),
        ("BigFloat212", () -> compute_amplitudes(S, L, M, 0.9, 0.5; backend=:bigfloat,   precision=212)),
        ("Float64x4  ", () -> compute_amplitudes(S, L, M, 0.9, 0.5; backend=:multifloat, precision=212))]
    p = prof(f)
    @printf("%s : %8d allocations, %8.2f MiB, %.4f s\n", lbl, p.nalloc, p.bytes / 2^20, p.time)
end
