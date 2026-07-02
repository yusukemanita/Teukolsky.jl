# Benchmark + value dump for MST hot paths. Run with:
#   julia --project=<checkout> bench_mst.jl <label>
# Prints one line per (case, quantity): LABEL case quantity value_or_time

using Teukolsky
using Printf

const s, l, m, a = -2, 2, 2, 0.7
const r_eval = 10.0

function fmt(z)
    zc = ComplexF64(z)
    @sprintf "%.16e%+.16eim" real(zc) imag(zc)
end

function run_case(label, name, ω, bits, nmax)
    setprecision(BigFloat, bits) do
        ab = BigFloat(a)
        ωb = Complex{BigFloat}(ω)
        ν, p = compute_nu(s, l, m, ab, ωb)
        if !isfinite(ν)
            println("$label $name nu NONFINITE"); return
        end
        fn = compute_fn(p, ν; nmax=nmax)
        t_fn = @elapsed compute_fn(p, ν; nmax=nmax)
        Am = Teukolsky.compute_Aminus(p, ν, fn; nmax=nmax)
        t_am = @elapsed Teukolsky.compute_Aminus(p, ν, fn; nmax=nmax)
        Kν = Teukolsky.compute_Knu(p, ν, fn; nmax=nmax)
        t_k = @elapsed Teukolsky.compute_Knu(p, ν, fn; nmax=nmax)
        ct = Teukolsky._ctrans(p, Am)
        rup = Rup(p, ν, fn, BigFloat(r_eval); nmax=nmax, ctrans=ct)
        t_rup = @elapsed Rup(p, ν, fn, BigFloat(r_eval); nmax=nmax, ctrans=ct)
        rin = Rin(p, ν, fn, BigFloat(r_eval); nmax=nmax)
        t_rin = @elapsed Rin(p, ν, fn, BigFloat(r_eval); nmax=nmax)
        println("$label $name nu      $(fmt(ν))")
        println("$label $name Aminus  $(fmt(Am))")
        println("$label $name Knu     $(fmt(Kν))")
        println("$label $name Rup     $(fmt(rup))")
        println("$label $name Rin     $(fmt(rin))")
        @printf "%s %s t_fn %.4f t_Am %.4f t_Knu %.4f t_Rup %.4f t_Rin %.4f\n" label name t_fn t_am t_k t_rup t_rin
    end
end

function run_amp(label, name, ω, backend, bits, nmax)
    amp = compute_amplitudes(s, l, m, a, ω; backend=backend, precision=bits, nmax=nmax)
    t = @elapsed compute_amplitudes(s, l, m, a, ω; backend=backend, precision=bits, nmax=nmax)
    println("$label $name Binc    $(fmt(amp.Binc))")
    println("$label $name Bref    $(fmt(amp.Bref))")
    @printf "%s %s t_amp %.4f\n" label name t
end

label = ARGS[1]

# warmup / compile
run_case(label, "warmup", 0.5 + 0.0im, 128, 20)

run_case(label, "bf_w0.5_256",  0.5 + 0.0im, 256, 40)
run_case(label, "bf_s2.3_256",  2.3im,       256, 60)
run_case(label, "bf_s4.3_320",  4.3im,       320, 80)
run_case(label, "bf_s8.3_640",  8.3im,       640, 120)

run_amp(label, "amp_bf_s4.3_320", 4.3im, :bigfloat, 320, 80)
run_amp(label, "amp_arb_s4.3_320", 4.3im, :arb, 320, 80)
run_amp(label, "amp_mf_w2_212", 2.0 + 0.0im, :multifloat, 212, 60)

# Arb radial (native 2F1 target): Rin under Complex{Arb}
using Arblib
setprecision(Arb, 320) do
    ab = Arb(a); ωb = Complex{Arb}(Arb(0), Arb(43)/10)
    ν, p = compute_nu(s, l, m, ab, ωb)
    fn = compute_fn(p, ν; nmax=80)
    rin = Rin(p, ν, fn, Arb(r_eval); nmax=80)
    t = @elapsed Rin(p, ν, fn, Arb(r_eval); nmax=80)
    ct = Teukolsky._ctrans(p, Teukolsky.compute_Aminus(p, ν, fn; nmax=80))
    rup = Rup(p, ν, fn, Arb(r_eval); nmax=80, ctrans=ct)
    t2 = @elapsed Rup(p, ν, fn, Arb(r_eval); nmax=80, ctrans=ct)
    println("$label arb_s4_320 Rin     $(fmt(rin))")
    println("$label arb_s4_320 Rup     $(fmt(rup))")
    @printf "%s arb_s4_320 t_Rin %.4f t_Rup %.4f\n" label t t2
end

# ============================================================================
# A/B usage (perf/mst-large-omega vs its base 600ec3f):
#   julia --project=<checkout-at-base>   scripts/bench_mst_large_omega.jl BASE
#   julia --project=<checkout-at-branch> scripts/bench_mst_large_omega.jl NEW
# Measured on this machine (M-series, 2026-07): values agree to all 16 printed
# digits on every backend; timings
#   compute_fn      13.5–29.8×   (one Lentz per direction + CF peeling)
#   compute_Aminus  11–32×       (incremental Pochhammer)
#   Rup              5.6–23×     (BigFloat U → acb_hypgeom_u bridge; Arb fup)
#   compute_amplitudes end-to-end: 2.7× (BigFloat σ=4.3), 5.9× (Arb σ=4.3)
# NOTE: benchmark frequencies deliberately avoid the PIA monodromy resonance
# (ω = iσ with 4σ ∈ ℤ), where ν is NaN at this branch's base commit.
# ============================================================================
