# ============================================================
#  SKEPTIC verification of change #1: compute_Knu Pochhammer
#  optimization (incremental gamma-ratio product replacing six
#  fresh _cgamma per numerator term).
#
#  Strategy: try to PROVE IT WRONG.
#  (a) independence: re-implement the ORIGINAL numerator (six fresh
#      _cgamma per term) and compare full Knu to Teukolsky.compute_Knu
#      across a mode grid at BigFloat-256, for ν and -ν-1, r=0,1,2,
#      and BOTH compute_Knu and compute_Knu_mero.
#  (b) amplitudes vs Wolfram (wolfram_ref_hp.txt)
#  (c) self-consistency Binc: BF256 vs BF512
#  (d) Wronskian r-independence (radial untouched)
#  (e) edge: nonzero-r path + mero/nufixed references
# ============================================================
using Teukolsky
using Printf

const B = Teukolsky
const cg = B._cgamma
const poch = B.pochhammer

# ---- ORIGINAL numerator: six fresh Γ per term -----------------------------
# Σ_{n≥r} (-1)^n G(n) f_n,
#  G(n)=Γ(n+r+2ν+1)/Γ(n-r+1)·Γ(n+ν+1+s+iϵ)/Γ(n+ν+1-s-iϵ)·Γ(n+ν+1+iτ)/Γ(n+ν+1-iτ)
function orig_numerator(p, ν, fn; nmax::Int=80, r::Int=0)
    s, ϵ, τ = p.s, p.ϵ, p.τ
    T = typeof(ϵ)
    num_sum = zero(T)
    for n in r:nmax
        fn_n = get(fn, n, zero(T))
        iszero(fn_n) && continue
        G = cg(T(n + r) + 2ν + 1) / cg(T(n - r) + 1) *
            cg(T(n) + ν + 1 + s + im*ϵ) / cg(T(n) + ν + 1 - s - im*ϵ) *
            cg(T(n) + ν + 1 + im*τ) / cg(T(n) + ν + 1 - im*τ)
        num_sum += (-1)^n * G * fn_n
    end
    return num_sum
end

# Full ORIGINAL compute_Knu (numerator independent; den/prefac copied verbatim
# from the UNCHANGED parts so a Knu mismatch isolates the numerator change).
function orig_Knu(p, ν, fn; nmax::Int=80, r::Int=0, mero::Bool=false)
    s, ϵ, κ, τ = p.s, p.ϵ, p.κ, p.τ
    ϵp = p.ϵp
    T = typeof(ϵ)

    num_sum = orig_numerator(p, ν, fn; nmax=nmax, r=r)

    den_sum = zero(T)
    for n in -nmax:r
        fn_n = fn[n]
        iszero(fn_n) && continue
        term = (-1)^n / cg(T(r - n + 1)) /
               poch(r + 2ν + 2, n) *
               poch(ν + 1 + s - im*ϵ, n) /
               poch(ν + 1 - s + im*ϵ, n) *
               fn_n
        den_sum += term
    end

    expo = mero ? (s - r) : (s - ν - r)
    prefactor = exp(im*ϵ*κ) * (2*ϵ*κ)^(expo) * T(2)^(-s) /
                im^r *
                cg(T(1) - s - 2im*ϵp) *
                cg(T(r) + 2ν + 2) /
                (cg(T(r) + ν + 1 - s + im*ϵ) *
                 cg(T(r) + ν + 1 + im*τ) *
                 cg(T(r) + ν + 1 + s + im*ϵ))

    return prefactor * num_sum / den_sum
end

relerr(a, b) = abs(b) == 0 ? abs(a - b) : abs(a - b) / abs(b)

# ============================================================
println("="^70)
println("(a) INDEPENDENCE: orig (6 fresh Γ) vs current compute_Knu / _mero")
println("="^70)

worst_a = 0.0
worst_a_where = ""
ncmp_a = 0
setprecision(BigFloat, 256) do
    global worst_a, worst_a_where, ncmp_a
    for l in (2, 3), a in (big"0.0", big"0.5", big"0.9"), om in (big"0.1", big"0.3", big"0.5", big"1.0")
        s, m = -2, 2
        ω = Complex{BigFloat}(om)
        ν, p = compute_nu(s, l, m, a, ω)
        fn   = compute_fn(p, ν; nmax=80)
        fnn  = compute_fn(p, -ν - 1; nmax=80)
        for r in (0, 1, 2)
            # current (changed) code
            cur   = B.compute_Knu(p, ν, fn; nmax=80, r=r)
            org   = orig_Knu(p, ν, fn; nmax=80, r=r, mero=false)
            cur_n = B.compute_Knu(p, -ν - 1, fnn; nmax=80, r=r)
            org_n = orig_Knu(p, -ν - 1, fnn; nmax=80, r=r, mero=false)
            cur_m   = B.compute_Knu_mero(p, ν, fn; nmax=80, r=r)
            org_m   = orig_Knu(p, ν, fn; nmax=80, r=r, mero=true)
            cur_mn  = B.compute_Knu_mero(p, -ν - 1, fnn; nmax=80, r=r)
            org_mn  = orig_Knu(p, -ν - 1, fnn; nmax=80, r=r, mero=true)
            for (tag, c, o) in (("Knu",  cur,  org), ("Knu(-ν-1)", cur_n, org_n),
                                ("Kmero", cur_m, org_m), ("Kmero(-ν-1)", cur_mn, org_mn))
                e = relerr(c, o)
                ncmp_a += 1
                if e > worst_a
                    worst_a = e
                    worst_a_where = "l=$l a=$(Float64(a)) ω=$(Float64(om)) r=$r $tag"
                end
            end
        end
        @printf("l=%d a=%.1f ω=%.1f  running worst relerr = %.3e\n",
                l, Float64(a), Float64(om), worst_a)
    end
end
@printf("\n(a) comparisons=%d  WORST relerr = %.3e  at  %s\n",
        ncmp_a, worst_a, worst_a_where)

# ============================================================
println("\n" * "="^70)
println("(b) AMPLITUDES vs Wolfram wolfram_ref_hp.txt (Binc, Bref)")
println("="^70)
pb(s) = parse(BigFloat, s)
pc(re, im) = (re == "ERR" || im == "ERR") ? nothing : Complex{BigFloat}(pb(re), pb(im))
relerrN(a, b) = b === nothing ? NaN : (abs(b) == 0 ? abs(a - b) : abs(a - b) / abs(b))
REF = joinpath(@__DIR__, "wolfram_ref_hp.txt")

worst_b = 0.0; worst_b_where = ""
setprecision(BigFloat, 256) do
    global worst_b, worst_b_where
    for ln in readlines(REF)
        isempty(strip(ln)) && continue
        f = split(strip(ln), ";")
        s = parse(Int, f[1]); l = parse(Int, f[2]); m = parse(Int, f[3])
        a = pb(f[4]); om = pb(f[5])
        Bincr = pc(f[10], f[11]); Brefr = pc(f[12], f[13])
        ω = Complex{BigFloat}(om)
        amp = compute_amplitudes(s, l, m, a, ω)
        eB = relerrN(amp.Binc, Bincr)
        eR = Brefr === nothing ? NaN : relerrN(amp.Bref, Brefr)
        @printf("s%-2d l%d m%d a%.1f ω%.2f | Binc relerr=%.2e  Bref relerr=%.2e\n",
                s, l, m, Float64(a), Float64(om), eB, eR)
        if isfinite(eB) && eB > worst_b; worst_b = eB; worst_b_where = "Binc s=$s l=$l m=$m a=$(Float64(a)) ω=$(Float64(om))"; end
    end
end
@printf("\n(b) WORST Binc relerr vs Wolfram = %.3e  at  %s\n", worst_b, worst_b_where)

# ============================================================
println("\n" * "="^70)
println("(c) SELF-CONSISTENCY Binc: BF256 vs BF512")
println("="^70)
worst_c = 0.0; worst_c_where = ""
let
    global worst_c, worst_c_where
    bincat(s,l,m,a,om,prec) = setprecision(BigFloat, prec) do
        ω = Complex{BigFloat}(BigFloat(om))
        compute_amplitudes(s, l, m, BigFloat(a), ω).Binc
    end
    for (s,l,m,a,om) in [(-2,2,2,0.0,0.1),(-2,2,2,0.0,0.3),(-2,2,2,0.0,0.5),
                         (-2,2,2,0.5,0.5),(-2,2,2,0.9,0.5),(-2,2,2,0.9,1.0),
                         (-2,3,2,0.5,1.0),(-2,3,2,0.9,0.3)]
        lo = bincat(s,l,m,a,om,256); hi = bincat(s,l,m,a,om,512)
        e = abs(Complex{BigFloat}(lo) - Complex{BigFloat}(hi)) / abs(Complex{BigFloat}(hi))
        @printf("s%-2d l%d m%d a%.1f ω%.1f | Binc 256-vs-512 relerr=%.2e\n",
                s,l,m,Float64(a),Float64(om),e)
        if e > worst_c; worst_c = e; worst_c_where = "s=$s l=$l m=$m a=$a ω=$om"; end
    end
end
@printf("\n(c) WORST Binc 256-vs-512 relerr = %.3e  at  %s\n", worst_c, worst_c_where)

# ============================================================
println("\n" * "="^70)
println("(d) WRONSKIAN r-independence Δ^{s+1}(Rin dRup - dRin Rup)")
println("="^70)
worst_d = 0.0; worst_d_where = ""
setprecision(BigFloat, 256) do
    global worst_d, worst_d_where
    for (s,l,m,a,om) in [(-2,2,2,0.0,0.3),(-2,2,2,0.0,0.5),(-2,2,2,0.9,0.5),(-2,3,2,0.5,1.0)]
        ω = Complex{BigFloat}(BigFloat(om)); ν, p = compute_nu(s,l,m,BigFloat(a),ω)
        fn = compute_fn(p, ν)
        W(r) = (BigFloat(r)^2 - 2BigFloat(r) + BigFloat(a)^2)^(s+1) *
               (Rin(p,ν,fn,BigFloat(r))*dRup(p,ν,fn,BigFloat(r)) -
                dRin(p,ν,fn,BigFloat(r))*Rup(p,ν,fn,BigFloat(r)))
        e = abs(W(4) - W(8)) / abs(W(4))
        @printf("s%-2d l%d m%d a%.1f ω%.1f | Wronskian |ΔW|/|W|=%.2e\n",
                s,l,m,Float64(a),Float64(om),e)
        if e > worst_d; worst_d = e; worst_d_where = "s=$s l=$l m=$m a=$a ω=$om"; end
    end
end
@printf("\n(d) WORST Wronskian variation = %.3e  at  %s\n", worst_d, worst_d_where)

# ============================================================
println("\n" * "="^70)
println("(e) EDGE: mero/nufixed amplitudes vs references (nonzero-r path covered in (a))")
println("="^70)
worst_e = 0.0; worst_e_where = ""
setprecision(BigFloat, 256) do
    global worst_e, worst_e_where
    # (e1) nufixed path: current compute_Knu vs orig (6 fresh Γ) at fixed ν, r=0,1,2.
    # This is the same internal Knu the nufixed amplitudes call.
    for (s,l,m,a,om,νf) in [(-2,2,2,0.0,0.3,big"1.7"),(-2,2,2,0.9,0.5,big"1.2"),
                            (-2,3,2,0.5,1.0,big"3.2")]
        ω = Complex{BigFloat}(BigFloat(om))
        p = MSTParams(s,l,m,BigFloat(a),ω)
        ν = Complex{BigFloat}(νf + sqrt(eps(BigFloat)))
        fn = compute_fn(p, ν; nmax=80)
        for r in (0,1,2)
            e  = relerr(B.compute_Knu(p, ν, fn; nmax=80, r=r),
                        orig_Knu(p, ν, fn; nmax=80, r=r))
            em = relerr(B.compute_Knu_mero(p, ν, fn; nmax=80, r=r),
                        orig_Knu(p, ν, fn; nmax=80, r=r, mero=true))
            ee = max(e, em)
            @printf("nufixed s%-2d a%.1f ω%.1f r%d | Knu/Kmero orig-vs-cur relerr=%.2e\n",
                    s,Float64(a),Float64(om),r,ee)
            if ee > worst_e; worst_e = ee; worst_e_where = "nufixed s=$s a=$(Float64(a)) ω=$(Float64(om)) r=$r"; end
        end
    end
    # (e2) mero & nufixed amplitudes still high-precision (256-vs-512 self-conv).
    bincmero(s,l,m,a,om,prec) = setprecision(BigFloat, prec) do
        compute_amplitudes_mero(s,l,m,BigFloat(a),Complex{BigFloat}(BigFloat(om))).Binc
    end
    bincnuf(s,l,m,a,om,νf,prec) = setprecision(BigFloat, prec) do
        compute_amplitudes_nufixed(s,l,m,BigFloat(a),Complex{BigFloat}(BigFloat(om)),BigFloat(νf)).Binc
    end
    for (s,l,m,a,om) in [(-2,2,2,0.0,0.3),(-2,2,2,0.9,1.0),(-2,3,2,0.5,1.0)]
        e = abs(bincmero(s,l,m,a,om,256)-bincmero(s,l,m,a,om,512))/abs(bincmero(s,l,m,a,om,512))
        @printf("mero  s%-2d a%.1f ω%.1f | Binc 256-vs-512 relerr=%.2e\n",s,Float64(a),Float64(om),e)
        if e > worst_e; worst_e = e; worst_e_where = "mero selfconv s=$s a=$a ω=$om"; end
    end
    for (s,l,m,a,om,νf) in [(-2,2,2,0.0,0.3,1.7),(-2,2,2,0.9,0.5,1.2)]
        e = abs(bincnuf(s,l,m,a,om,νf,256)-bincnuf(s,l,m,a,om,νf,512))/abs(bincnuf(s,l,m,a,om,νf,512))
        @printf("nufix s%-2d a%.1f ω%.1f νf%.1f | Binc 256-vs-512 relerr=%.2e\n",s,Float64(a),Float64(om),νf,e)
        if e > worst_e; worst_e = e; worst_e_where = "nufixed selfconv s=$s a=$a ω=$om"; end
    end
end
@printf("\n(e) WORST edge discrepancy = %.3e  at  %s\n", worst_e, worst_e_where)

# ============================================================
println("\n" * "="^70)
println("SUMMARY")
println("="^70)
@printf("(a) independence orig-vs-current Knu : worst %.3e  [%s]\n", worst_a, worst_a_where)
@printf("(b) Binc vs Wolfram                  : worst %.3e  [%s]\n", worst_b, worst_b_where)
@printf("(c) Binc 256-vs-512                  : worst %.3e  [%s]\n", worst_c, worst_c_where)
@printf("(d) Wronskian r-independence         : worst %.3e  [%s]\n", worst_d, worst_d_where)
@printf("(e) edge (mero ratio / nufixed)      : worst %.3e  [%s]\n", worst_e, worst_e_where)
