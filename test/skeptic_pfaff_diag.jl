# Drill-down: which side (Pfaff vs _2F1) loses precision at high precision / z->1?
using BHPtoolkit
using HypergeometricFunctions: _₂F₁
using Printf
const BHP = BHPtoolkit
relerr(x,y) = abs(x-y)/max(abs(y),1e-300)

# Build the SAME params as part (c): mode s=-2,l=2,m=2,a=0.5,ω=0.5, complex ν.
function params_at(::Type{TR}) where TR
    amp = BHP.compute_amplitudes(-2,2,2,TR(0.5),Complex{TR}(0.5))
    p = BHP.MSTParams(-2,2,2,TR(0.5),Complex{TR}(0.5))
    return amp.ν, p
end

println("(c-diag) Pfaff vs _2F1 vs 512-bit reference, per orbit radius r")
println("  rel512 = error vs 512-bit independent reference (smaller => more accurate)")
# 512-bit reference values for ν, p (recompute at 512 for a faithful reference).
for r in (6.0, 10.0, 30.0, 100.0)
    # 192-bit working values
    ν192, p192 = setprecision(() -> params_at(BigFloat), BigFloat, 192)
    x192 = complex((p192.rp - BigFloat(r))/(2*p192.κ))
    hp192 = BHP.H2F1Params(p192, ν192, x192)
    # 512-bit reference (recompute params at 512 too)
    ν512, p512 = setprecision(() -> params_at(BigFloat), BigFloat, 512)
    worst_pf = 0.0; worst_2f = 0.0; nworst = 0
    for n in -8:2:20
        a192,b192,c192 = n+hp192.aF, hp192.bF-n, hp192.cF
        pf192 = BHP._h2f1_pfaff(a192,b192,c192, hp192.x)
        f2192 = _₂F₁(a192,b192,c192, hp192.x)
        # 512-bit reference using 512-bit params + same n
        ref = setprecision(BigFloat, 512) do
            x512 = complex((p512.rp - BigFloat(r))/(2*p512.κ))
            hp512 = BHP.H2F1Params(p512, ν512, x512)
            BHP._h2f1_pfaff(n+hp512.aF, hp512.bF-n, hp512.cF, hp512.x)
        end
        epf = relerr(pf192, ref); e2f = relerr(f2192, ref)
        if max(epf,e2f) > max(worst_pf,worst_2f); nworst = n; end
        worst_pf = max(worst_pf, epf); worst_2f = max(worst_2f, e2f)
    end
    @printf("  r=%-6.0f Pfaff rel512=%.2e   _2F1 rel512=%.2e   (worst at n=%d)\n",
            r, worst_pf, worst_2f, nworst)
end

println()
println("(e-diag) Pfaff term-count needed for full precision vs z (orbit radius)")
# How many terms to converge to eps at each precision, for x from r=6..200 (a=0).
function terms_needed(prec, x)
    setprecision(BigFloat, prec) do
        a=complex(big(2.3)); b=complex(big(-1.7)); c=complex(big(0.5)); xx=complex(BigFloat(x))
        z = xx/(xx-1); bb = c-b
        T = typeof(z); term=one(T); s=term; tol=eps(real(T))
        for k in 0:200000
            term *= (a+k)*(bb+k)/((c+k)*(k+1))*z
            s += term
            abs(term) ≤ tol*abs(s) && return k+1
        end
        return -1
    end
end
for (r, xval) in [(6.0,-2.0),(20.0,-9.0),(50.0,-24.0),(100.0,-49.0),(200.0,-99.0)]
    z = xval/(xval-1)
    k192 = terms_needed(192, xval)
    k256 = terms_needed(256, xval)
    flag192 = (k192 > 8000 || k192 < 0) ? "  CAP-EXCEEDED@192" : ""
    flag256 = (k256 > 8000 || k256 < 0) ? "  CAP-EXCEEDED@256" : ""
    @printf("  r=%-5.0f x=%-5.0f z=%.4f  terms@192=%d%s  terms@256=%d%s\n",
            r, xval, z, k192, flag192, k256, flag256)
end
println("  (source caps the Pfaff loop at 8000 terms: `for k in 0:7999`)")
