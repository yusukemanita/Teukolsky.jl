# ============================================================
#  Native-Acb asymptotic amplitudes A^ν_±  (M3)
#
#  In-place native-Acb clones of compute_Aplus / compute_Aminus
#  (amplitudes.jl, Eqs. 157-158).  The O(1) transcendental prefactors are built
#  once in Complex{BigFloat} (validated arithmetic, no perf benefit off the hot
#  loop) and converted to Acb; the n-sums run in preallocated Acb registers with
#  Arblib in-place `!` ops.  The A^ν_- Pochhammer ratio is marched incrementally
#  (one Acb mul/div per term) rather than re-evaluated.  `fn` is the Complex{Arb}
#  dict from `compute_fn_acb` (or the generic path); values are converted to Acb
#  per term.
# ============================================================

# Convert a Complex{Arb} to a fresh Acb at `prec`.
@inline _toacb(v::Complex{Arb}, prec::Int) = Acb(real(v), imag(v); prec=prec)

"""
    compute_Aplus_acb(p, ν, fn; nmax=80, nmin=-nmax) -> Complex{Arb}

Native-Acb A^ν_+ = prefactor · Σ_n f_n, with
prefactor = e^{-πε/2} e^{iπ(ν+1-s)/2} 2^{-1+s-iε} Γ(ν+1-s+iε)/Γ(ν+1+s-iε).
"""
function compute_Aplus_acb(p, ν, fn; nmax::Int=80, nmin::Int=-nmax)
    prec = precision(Arb); s = p.s
    pref = setprecision(BigFloat, prec) do
        C  = Complex{BigFloat}
        ε  = C(BigFloat(real(p.ϵ)), BigFloat(imag(p.ϵ)))
        νb = C(BigFloat(real(ν)), BigFloat(imag(ν)))
        exp(-π*ε/2) * exp(π*im*(νb + 1 - s)/2) * C(2)^(-1 + s - im*ε) *
            _cgamma(νb + 1 - s + im*ε) / _cgamma(νb + 1 + s - im*ε)
    end
    prefA = Acb(pref; prec=prec)
    acc = Acb(0)
    for n in nmin:nmax
        Arblib.add!(acc, acc, _toacb(fn[n], prec); prec=prec)
    end
    res = Acb(0); Arblib.mul!(res, prefA, acc; prec=prec)
    return Complex{Arb}(res)
end

"""
    compute_Aminus_acb(p, ν, fn; nmax=80, nmin=-nmax) -> Complex{Arb}

Native-Acb A^ν_- = prefactor · Σ_n (-1)^n P(n) f_n, with the Pochhammer ratio
P(n) = (ν+1+s-iε)_n / (ν+1-s+iε)_n marched incrementally, and
prefactor = 2^{-1-s+iε} e^{-iπ(ν+1+s)/2} e^{-πε/2}.
"""
function compute_Aminus_acb(p, ν, fn; nmax::Int=80, nmin::Int=-nmax)
    prec = precision(Arb); s = p.s
    pref, ain, bin = setprecision(BigFloat, prec) do
        C  = Complex{BigFloat}
        ε  = C(BigFloat(real(p.ϵ)), BigFloat(imag(p.ϵ)))
        νb = C(BigFloat(real(ν)), BigFloat(imag(ν)))
        pf = C(2)^(-1 - s + im*ε) * exp(-π*im*(νb + 1 + s)/2) * exp(-π*ε/2)
        (pf, νb + 1 + s - im*ε, νb + 1 - s + im*ε)          # ain, bin
    end
    prefA = Acb(pref; prec=prec)
    ainA  = Acb(ain; prec=prec); binA = Acb(bin; prec=prec)
    acc = Acb(0); tmp = Acb(0); tmp2 = Acb(0); term = Acb(0)

    # n = 0 term: (+1)·P(0)=1·f_0
    Arblib.add!(acc, acc, _toacb(fn[0], prec); prec=prec)

    # forward n = 1..nmax:  P(n) = P(n-1)·(ain+n-1)/(bin+n-1)
    Rf = Acb(1)
    for n in 1:nmax
        n > nmax && break
        Arblib.add!(tmp,  ainA, n - 1; prec=prec)
        Arblib.add!(tmp2, binA, n - 1; prec=prec)
        Arblib.div!(tmp, tmp, tmp2; prec=prec)
        Arblib.mul!(Rf, Rf, tmp; prec=prec)
        Arblib.mul!(term, Rf, _toacb(fn[n], prec); prec=prec)
        isodd(n) ? Arblib.sub!(acc, acc, term; prec=prec) :
                   Arblib.add!(acc, acc, term; prec=prec)
    end

    # backward n = -1..-nmax:  P(n) = P(n+1)·(bin+n)/(ain+n)
    Rb = Acb(1)
    for n in -1:-1:max(nmin, -nmax)
        Arblib.add!(tmp,  binA, n; prec=prec)
        Arblib.add!(tmp2, ainA, n; prec=prec)
        Arblib.div!(tmp, tmp, tmp2; prec=prec)
        Arblib.mul!(Rb, Rb, tmp; prec=prec)
        Arblib.mul!(term, Rb, _toacb(fn[n], prec); prec=prec)
        isodd(n) ? Arblib.sub!(acc, acc, term; prec=prec) :
                   Arblib.add!(acc, acc, term; prec=prec)
    end

    res = Acb(0); Arblib.mul!(res, prefA, acc; prec=prec)
    return Complex{Arb}(res)
end
