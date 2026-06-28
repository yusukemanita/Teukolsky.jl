# ============================================================
#  SKEPTIC verification of change #2: BigFloat 2F1 flux fix (_h2f1_pfaff)
#  Adversarial checks (a)-(e). Prints machine-parseable RESULT lines.
# ============================================================
using BHPtoolkit
using HypergeometricFunctions: _₂F₁
using Printf

const BHP = BHPtoolkit

# ---- Instrument the fallback to count how often Pfaff engages ----
# Redefine _h2f1_robust inside the module to increment a counter on the
# DomainError catch branch. Body otherwise identical to source.
@eval BHP begin
    const _PFAFF_CALLS = Ref(0)
    @inline function _h2f1_robust(a, b, c, x)
        try
            return _₂F₁(a, b, c, x)
        catch e
            e isa DomainError || rethrow(e)
            _PFAFF_CALLS[] += 1
            return _h2f1_pfaff(a, b, c, x)
        end
    end
end
reset_pfaff!() = (BHP._PFAFF_CALLS[] = 0)
pfaff_calls()  = BHP._PFAFF_CALLS[]

worst = Dict{String,Float64}()
note(tag, v) = (worst[tag] = max(get(worst, tag, 0.0), v))

relerr(x, y) = abs(x - y) / max(abs(y), 1e-300)

println("="^70)
println("(a) BigFloat vs Float64 flux grid")
println("="^70)
fails_a = String[]
for (l, m) in [(2,2),(3,3),(2,1)], a in [0.0, 0.5, 0.9, -0.7], p in [6.0, 10.0, 20.0]
    tag = "l$l m$m a$a p$p"
    local md64, mdbf192, mdbf256
    ok = true
    try
        md64 = TeukolskyPointParticleMode(-2, l, m, a, p)
    catch e
        push!(fails_a, "$tag: F64 CRASH $(typeof(e))")
        continue
    end
    try
        mdbf192 = setprecision(() -> TeukolskyPointParticleMode(-2, l, m, big(a), big(p)), BigFloat, 192)
        mdbf256 = setprecision(() -> TeukolskyPointParticleMode(-2, l, m, big(a), big(p)), BigFloat, 256)
    catch e
        push!(fails_a, "$tag: BF CRASH $(typeof(e))")
        continue
    end
    for (lbl, mdbf) in (("192", mdbf192), ("256", mdbf256))
        fI = Float64(mdbf.EnergyFlux.Inf); fH = Float64(mdbf.EnergyFlux.Hor)
        if !isfinite(fI) || !isfinite(fH) || !(mdbf.EnergyFlux.Inf isa BigFloat)
            push!(fails_a, "$tag bf$lbl: NONFINITE Inf=$fI Hor=$fH"); ok = false; continue
        end
        eI = relerr(fI, md64.EnergyFlux.Inf)
        eH = relerr(fH, md64.EnergyFlux.Hor)
        note("a_Inf", eI); note("a_Hor", eH)
        if eI > 1e-10
            push!(fails_a, @sprintf("%s bf%s: Inf rel=%.2e (>1e-10)", tag, lbl, eI)); ok = false
        end
        if eH > 1e-7
            push!(fails_a, @sprintf("%s bf%s: Hor rel=%.2e (>1e-7)", tag, lbl, eH)); ok = false
        end
    end
    ok && @printf("  OK  %-22s  Inf=%.3e Hor=%.3e (worst rel so far Inf=%.1e Hor=%.1e)\n",
                  tag, md64.EnergyFlux.Inf, md64.EnergyFlux.Hor, get(worst,"a_Inf",0.0), get(worst,"a_Hor",0.0))
end
@printf("(a) worst Inf rel=%.3e  worst Hor rel=%.3e  fails=%d\n",
        get(worst,"a_Inf",0.0), get(worst,"a_Hor",0.0), length(fails_a))
for f in fails_a; println("    FAIL ", f); end

println("="^70)
println("(b) Wolfram flux_ref.txt at BigFloat (8 cases)")
println("="^70)
_pnum(t) = parse(Float64, replace(split(t, "`")[1], "*^" => "e"))
function _read_flux_cases(path)
    cases = Dict{String,Any}[]; cur = nothing
    for ln in readlines(path)
        st = strip(ln); (isempty(st) || startswith(st, "#")) && continue
        if startswith(st, "CASE")
            cur === nothing || push!(cases, cur)
            f = split(st)
            g(k) = _pnum(split(f[findfirst(x -> startswith(x, k), f)], "=")[2])
            cur = Dict{String,Any}("s"=>Int(g("s")),"l"=>Int(g("l")),"m"=>Int(g("m")),"a"=>g("a"),"p"=>g("p"))
        else
            f = split(st)
            vals = [_pnum(t) for t in f[2:end] if t != "+" && !occursin("I", t)]
            cur[f[1]] = length(vals) == 1 ? vals[1] : complex(vals[1], vals[2])
        end
    end
    cur === nothing || push!(cases, cur); return cases
end
fails_b = String[]
for c in _read_flux_cases(joinpath(@__DIR__, "flux_ref.txt"))
    s,l,m,a,p = c["s"],c["l"],c["m"],c["a"],c["p"]
    tag = "l$l m$m a$a p$p"
    mdbf = setprecision(() -> TeukolskyPointParticleMode(s, l, m, big(a), big(p); prograde=true), BigFloat, 192)
    eI = relerr(Float64(mdbf.EnergyFlux.Inf), real(c["EnergyFluxInf"]))
    eH = relerr(Float64(mdbf.EnergyFlux.Hor), real(c["EnergyFluxHor"]))
    note("b_Inf", eI); note("b_Hor", eH)
    bad = ""
    eI > 1e-7 && (bad *= @sprintf(" Inf=%.2e(>1e-7)", eI))
    eH > 5e-3 && (bad *= @sprintf(" Hor=%.2e(>5e-3)", eH))
    isempty(bad) ? @printf("  OK  %-22s Inf rel=%.2e Hor rel=%.2e\n", tag, eI, eH) :
                   push!(fails_b, tag*bad)
end
@printf("(b) worst Inf rel=%.3e  worst Hor rel=%.3e  fails=%d\n",
        get(worst,"b_Inf",0.0), get(worst,"b_Hor",0.0), length(fails_b))
for f in fails_b; println("    FAIL ", f); end

println("="^70)
println("(c) Pfaff vs _2F1 on WORKING cases (complex ν, ω=0.5)")
println("="^70)
# Build real H2F1Params from a mode with ω=0.5 (ν complex → _₂F₁ does NOT throw).
fails_c = String[]
function pfaff_accuracy(::Type{TR}, a_bh) where TR
    amp = BHP.compute_amplitudes(-2, 2, 2, TR(a_bh), Complex{TR}(0.5))
    ν = amp.ν
    p = BHP.MSTParams(-2, 2, 2, TR(a_bh), Complex{TR}(0.5))
    κ = p.κ; rp = p.rp
    wmax = 0.0
    for r in (6.0, 10.0, 30.0, 100.0)
        x = complex((rp - TR(r)) / (2κ))
        hp = BHP.H2F1Params(p, ν, x)
        for n in -8:2:20
            aF = n + hp.aF; bF = hp.bF - n; cF = hp.cF
            ref = _₂F₁(aF, bF, cF, hp.x)
            pf  = BHP._h2f1_pfaff(aF, bF, cF, hp.x)
            e = relerr(pf, ref)
            wmax = max(wmax, e)
        end
    end
    return wmax, eps(real(TR))
end
for (Tn, TR) in (("F64", Float64), ("BF192", BigFloat))
    w, ep = TR === BigFloat ? setprecision(() -> pfaff_accuracy(BigFloat, big(0.5)), BigFloat, 192) :
                              pfaff_accuracy(Float64, 0.5)
    note("c_$Tn", w)
    tol = TR === BigFloat ? 1e4*Float64(ep) : 1e-11  # allow a few digits for z→1 (r=100)
    status = w > tol ? "FAIL" : "OK"
    @printf("  %s  %-6s worst Pfaff-vs-2F1 rel=%.3e  (tol=%.1e, eps=%.1e)\n", status, Tn, w, tol, Float64(ep))
    w > tol && push!(fails_c, "$Tn rel=$w")
end

println("="^70)
println("(d) Non-interference: does the fallback engage in Float64?")
println("="^70)
# F64 flux grid
reset_pfaff!()
for (l,m) in [(2,2),(3,3),(2,1)], a in [0.0,0.5,0.9,-0.7], p in [6.0,10.0,20.0]
    TeukolskyPointParticleMode(-2, l, m, a, p)
end
f64_calls = pfaff_calls()
@printf("  Float64 flux grid: Pfaff fallback engaged %d times\n", f64_calls)
# BF flux grid
reset_pfaff!()
for (l,m) in [(2,2)], a in [0.0], p in [10.0]
    setprecision(() -> TeukolskyPointParticleMode(-2,l,m,big(a),big(p)), BigFloat, 192)
end
bf_calls = pfaff_calls()
@printf("  BigFloat (one mode a=0 p=10): Pfaff fallback engaged %d times\n", bf_calls)
# Direct DomainError probe of HGF in F64 vs BF for a representative small-ω mode.
function probe_domainerror(::Type{TR}, a_bh, pval) where TR
    orbit = KerrCircularOrbit(TR(a_bh), TR(pval))
    ω = 2*orbit.Ωφ
    amp = BHP.compute_amplitudes(-2, 2, 2, TR(a_bh), ω)
    ν = amp.ν; p = BHP.MSTParams(-2,2,2,TR(a_bh), ω)
    κ = p.κ; rp = p.rp
    x = complex((rp - TR(pval))/(2κ))
    hp = BHP.H2F1Params(p, ν, x)
    throws = 0; tot = 0
    for n in -10:20
        tot += 1
        try
            _₂F₁(n+hp.aF, hp.bF-n, hp.cF, hp.x)
        catch e
            e isa DomainError ? (throws += 1) : rethrow(e)
        end
    end
    return throws, tot, ν
end
tF, totF, νF = probe_domainerror(Float64, 0.0, 20.0)
tB, totB, νB = setprecision(() -> probe_domainerror(BigFloat, big(0.0), big(20.0)), BigFloat, 192)
@printf("  probe a=0 p=20: F64 ν=%s  DomainErrors %d/%d\n", string(νF), tF, totF)
@printf("  probe a=0 p=20: BF  ν=%s  DomainErrors %d/%d\n", string(ComplexF64(νB)), tB, totB)

println("="^70)
println("(e) Stress: large p (50,100) small ω at BigFloat")
println("="^70)
fails_e = String[]
for (l,m) in [(2,2),(2,1)], a in [0.0, 0.9], p in [50.0, 100.0]
    tag = "l$l m$m a$a p$p"
    local md64, mdbf
    try
        md64 = TeukolskyPointParticleMode(-2,l,m,a,p)
    catch e
        push!(fails_e, "$tag F64 CRASH $(typeof(e))"); continue
    end
    try
        mdbf = setprecision(() -> TeukolskyPointParticleMode(-2,l,m,big(a),big(p)), BigFloat, 256)
    catch e
        push!(fails_e, "$tag BF256 CRASH $(typeof(e))"); continue
    end
    fI = Float64(mdbf.EnergyFlux.Inf); fH = Float64(mdbf.EnergyFlux.Hor)
    finite = isfinite(fI) && isfinite(fH)
    eI = relerr(fI, md64.EnergyFlux.Inf); eH = relerr(fH, md64.EnergyFlux.Hor)
    note("e_Inf", eI); note("e_Hor", eH)
    bad = ""
    finite || (bad *= " NONFINITE")
    eI > 1e-9 && (bad *= @sprintf(" Inf=%.2e(>1e-9)", eI))
    eH > 1e-6 && (bad *= @sprintf(" Hor=%.2e(>1e-6)", eH))
    isempty(bad) ? @printf("  OK  %-18s Inf=%.3e (rel %.1e)  Hor=%.3e (rel %.1e)\n",
                            tag, md64.EnergyFlux.Inf, eI, md64.EnergyFlux.Hor, eH) :
                   push!(fails_e, tag*bad)
end
# Direct Pfaff convergence at z→1, 256-bit: compare capped(8000) vs long-ref(1024-bit,200000).
function pfaff_ref(a,b,c,x; kmax)
    z = x/(x-1); bb = c-b
    T = promote_type(typeof(complex(a)),typeof(complex(b)),typeof(complex(c)),typeof(complex(z)))
    term = one(T); s = term; tol = eps(real(T))
    for k in 0:kmax
        term *= (a+k)*(bb+k)/((c+k)*(k+1))*z
        s += term
        abs(term) ≤ tol*abs(s) && return (1-x)^(-a)*s, k, true
    end
    return (1-x)^(-a)*s, kmax, false
end
# Worst-case z near 1: p=100, a=0 → x=(2-100)/2=-49, z=0.98
let
    capped = setprecision(BigFloat, 256) do
        x = complex(big(-49.0)); a=complex(big(2.3)); b=complex(big(-1.7)); c=complex(big(0.5))
        BHP._h2f1_pfaff(a,b,c,x), pfaff_ref(a,b,c,x; kmax=7999)[3]
    end
    ref = setprecision(BigFloat, 1024) do
        x = complex(big(-49.0)); a=complex(big(2.3)); b=complex(big(-1.7)); c=complex(big(0.5))
        pfaff_ref(a,b,c,x; kmax=400000)[1]
    end
    val256, converged = capped
    e = relerr(val256, ref)
    note("e_pfaff_z", e)
    @printf("  Pfaff z=0.98 @256bit: converged-within-8000=%s  rel-vs-1024bit=%.3e\n",
            converged, e)
    !converged && push!(fails_e, "Pfaff z=0.98 did NOT converge within 8000 terms @256bit")
    e > 1e-60 && push!(fails_e, @sprintf("Pfaff z=0.98 @256bit accuracy loss rel=%.2e", e))
end
@printf("(e) worst Inf rel=%.3e  worst Hor rel=%.3e  fails=%d\n",
        get(worst,"e_Inf",0.0), get(worst,"e_Hor",0.0), length(fails_e))
for f in fails_e; println("    FAIL ", f); end

println("="^70)
println("SUMMARY")
println("="^70)
@printf("(a) grid:    worst Inf=%.2e Hor=%.2e  fails=%d\n", get(worst,"a_Inf",0.0), get(worst,"a_Hor",0.0), length(fails_a))
@printf("(b) wolfram: worst Inf=%.2e Hor=%.2e  fails=%d\n", get(worst,"b_Inf",0.0), get(worst,"b_Hor",0.0), length(fails_b))
@printf("(c) pfaff:   F64=%.2e BF192=%.2e  fails=%d\n", get(worst,"c_F64",0.0), get(worst,"c_BF192",0.0), length(fails_c))
@printf("(d) F64 fallback engagements=%d  BF engagements=%d  (F64 DE probe %d/%d, BF %d/%d)\n",
        f64_calls, bf_calls, tF, totF, tB, totB)
@printf("(e) stress:  worst Inf=%.2e Hor=%.2e pfaff_z=%.2e  fails=%d\n",
        get(worst,"e_Inf",0.0), get(worst,"e_Hor",0.0), get(worst,"e_pfaff_z",0.0), length(fails_e))
totalfails = length(fails_a)+length(fails_b)+length(fails_c)+length(fails_e)
@printf("TOTAL FAILS = %d\n", totalfails)
