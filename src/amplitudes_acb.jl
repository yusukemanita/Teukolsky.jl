# ============================================================
#  Native-Acb asymptotic amplitudes A^ν_±  (M3)
#
#  In-place native-Acb clones of compute_Aplus / compute_Aminus
#  (amplitudes.jl, Eqs. 157-158).  The O(1) transcendental prefactors are built
#  natively in Acb (acb_hypgeom_gamma / exp / pow — the old Complex{BigFloat}
#  route cost ~4.3 ms PER Γ at 768 bits vs ~58 µs for acb_hypgeom_gamma); the
#  n-sums run in preallocated Acb registers with Arblib in-place `!` ops over a
#  DENSE Acb coefficient vector (fv[n + off] = f_n), so the hot loop does no
#  Dict lookups and no per-term Complex{Arb} → Acb boxing.  The A^ν_- Pochhammer
#  ratio is marched incrementally (one Acb mul/div per term) rather than
#  re-evaluated.
#
#  PUBLIC API is unchanged: compute_Aplus_acb / compute_Aminus_acb still accept
#  the Dict{Int,Complex{Arb}} `fn` from `compute_fn_acb` (or the generic path)
#  and convert it ONCE to the dense vector.  compute_mst_core_acb bypasses the
#  Dict entirely via `_compute_fn_acb_vec` + the internal `_Aplus_acb` /
#  `_Aminus_acb` vector kernels.
# ============================================================

# Convert a Complex{Arb} to a fresh Acb at `prec`.
@inline _toacb(v::Complex{Arb}, prec::Int) = Acb(real(v), imag(v); prec=prec)

# Dense Acb coefficient vector from an fn Dict over nmin:nmax (offset 1-nmin):
# out[n + (1-nmin)] = f_n.  One conversion per value, done once per A± call on
# the public Dict path; the core path never builds it (uses the fn vector).
_fn_vec_from_dict(fn, nmin::Int, nmax::Int, prec::Int) =
    Acb[_toacb(fn[n], prec) for n in nmin:nmax]

# --- A^ν_+ prefactor, natively in Acb -------------------------------------
#   pref = e^{-πε/2} e^{iπ(ν+1-s)/2} 2^{-1+s-iε} Γ(ν+1-s+iε)/Γ(ν+1+s-iε)
#        = exp( (π/2)(i(ν+1-s) − ε) ) · 2^{-1+s-iε} · Γ-ratio
# Written into `res`.
function _Aplus_pref_acb!(res::Acb, p, ν, prec::Int)
    s  = p.s
    εA = Acb(real(p.ϵ), imag(p.ϵ); prec=prec)
    νA = Acb(real(ν), imag(ν); prec=prec)
    πA = Arb(π; prec=prec)
    iε = Acb(0); t = Acb(0); g = Acb(0)
    Arblib.mul_onei!(iε, εA)                       # iε (exact)
    # res = Γ(ν+1-s+iε) / Γ(ν+1+s-iε)
    Arblib.add!(t, νA, 1 - s; prec=prec)
    Arblib.add!(t, t, iε; prec=prec)
    Arblib.hypgeom_gamma!(res, t; prec=prec)
    Arblib.add!(t, νA, 1 + s; prec=prec)
    Arblib.sub!(t, t, iε; prec=prec)
    Arblib.hypgeom_gamma!(g, t; prec=prec)
    Arblib.div!(res, res, g; prec=prec)
    # res *= exp( (π/2)·(i(ν+1-s) − ε) )
    Arblib.add!(t, νA, 1 - s; prec=prec)
    Arblib.mul_onei!(t, t)
    Arblib.sub!(t, t, εA; prec=prec)
    Arblib.mul!(t, t, πA; prec=prec)
    Arblib.mul_2exp!(t, t, -1)
    Arblib.exp!(t, t; prec=prec)
    Arblib.mul!(res, res, t; prec=prec)
    # res *= 2^{-1+s-iε}
    Arblib.neg!(t, iε)
    Arblib.add!(t, t, s - 1; prec=prec)
    Arblib.pow!(g, Acb(2; prec=prec), t; prec=prec)
    Arblib.mul!(res, res, g; prec=prec)
    return res
end

# Internal A^ν_+ over a dense Acb vector fv (fv[n + off] = f_n).  Returns a
# fresh Acb.  Assumes nmin ≤ 0 ≤ nmax and fv covers nmin:nmax at offset `off`.
function _Aplus_acb(p, ν, fv::Vector{Acb}, off::Int, nmin::Int, nmax::Int,
                    prec::Int)
    pref = Acb(0)
    _Aplus_pref_acb!(pref, p, ν, prec)
    acc = Acb(0)
    for n in nmin:nmax
        Arblib.add!(acc, acc, fv[n + off]; prec=prec)
    end
    Arblib.mul!(pref, pref, acc; prec=prec)
    return pref
end

"""
    compute_Aplus_acb(p, ν, fn; nmax=80, nmin=-nmax) -> Complex{Arb}

Native-Acb A^ν_+ = prefactor · Σ_n f_n, with
prefactor = e^{-πε/2} e^{iπ(ν+1-s)/2} 2^{-1+s-iε} Γ(ν+1-s+iε)/Γ(ν+1+s-iε)
built natively in Acb (acb_hypgeom_gamma).  `fn` is the Complex{Arb} dict from
`compute_fn_acb` (or the generic path), converted once to a dense Acb vector.
"""
function compute_Aplus_acb(p, ν, fn; nmax::Int=80, nmin::Int=-nmax)
    prec = precision(Arb)
    fv = _fn_vec_from_dict(fn, nmin, nmax, prec)
    return Complex{Arb}(_Aplus_acb(p, ν, fv, 1 - nmin, nmin, nmax, prec))
end

# --- A^ν_- prefactor + Pochhammer bases, natively in Acb -------------------
#   pref = 2^{-1-s+iε} e^{-iπ(ν+1+s)/2} e^{-πε/2}
#        = 2^{-1-s+iε} · exp( -(π/2)·(i(ν+1+s) + ε) )
#   ain  = ν+1+s-iε,  bin = ν+1-s+iε   (Pochhammer-ratio bases)
# Written into `res`, `ain`, `bin`.
function _Aminus_pref_acb!(res::Acb, ain::Acb, bin::Acb, p, ν, prec::Int)
    s  = p.s
    εA = Acb(real(p.ϵ), imag(p.ϵ); prec=prec)
    νA = Acb(real(ν), imag(ν); prec=prec)
    πA = Arb(π; prec=prec)
    iε = Acb(0); t = Acb(0)
    Arblib.mul_onei!(iε, εA)                       # iε (exact)
    # ain = ν+1+s-iε ; bin = ν+1-s+iε   (exact integer shifts + exact ±iε)
    Arblib.add!(ain, νA, 1 + s; prec=prec)
    Arblib.sub!(ain, ain, iε; prec=prec)
    Arblib.add!(bin, νA, 1 - s; prec=prec)
    Arblib.add!(bin, bin, iε; prec=prec)
    # res = 2^{-1-s+iε}
    Arblib.add!(t, iε, -1 - s; prec=prec)
    Arblib.pow!(res, Acb(2; prec=prec), t; prec=prec)
    # res *= exp( -(π/2)·(i(ν+1+s) + ε) )
    Arblib.add!(t, νA, 1 + s; prec=prec)
    Arblib.mul_onei!(t, t)
    Arblib.add!(t, t, εA; prec=prec)
    Arblib.mul!(t, t, πA; prec=prec)
    Arblib.mul_2exp!(t, t, -1)
    Arblib.neg!(t, t)
    Arblib.exp!(t, t; prec=prec)
    Arblib.mul!(res, res, t; prec=prec)
    return res
end

# Internal A^ν_- over a dense Acb vector fv (fv[n + off] = f_n).  The weight
# w(n) = (-1)^n (ain)_n/(bin)_n is marched incrementally from the n=0 anchor in
# both directions (one Acb mul + div per term).  Returns a fresh Acb.
function _Aminus_acb(p, ν, fv::Vector{Acb}, off::Int, nmin::Int, nmax::Int,
                     prec::Int)
    pref = Acb(0); ainA = Acb(0); binA = Acb(0)
    _Aminus_pref_acb!(pref, ainA, binA, p, ν, prec)
    acc = Acb(0); tmp = Acb(0); tmp2 = Acb(0); term = Acb(0)

    # n = 0 term: (+1)·P(0)=1·f_0
    Arblib.add!(acc, acc, fv[off]; prec=prec)

    # forward n = 1..nmax:  P(n) = P(n-1)·(ain+n-1)/(bin+n-1)
    Rf = Acb(1)
    for n in 1:nmax
        Arblib.add!(tmp,  ainA, n - 1; prec=prec)
        Arblib.add!(tmp2, binA, n - 1; prec=prec)
        Arblib.div!(tmp, tmp, tmp2; prec=prec)
        Arblib.mul!(Rf, Rf, tmp; prec=prec)
        Arblib.mul!(term, Rf, fv[n + off]; prec=prec)
        isodd(n) ? Arblib.sub!(acc, acc, term; prec=prec) :
                   Arblib.add!(acc, acc, term; prec=prec)
    end

    # backward n = -1..nmin:  P(n) = P(n+1)·(bin+n)/(ain+n)
    Rb = Acb(1)
    for n in -1:-1:max(nmin, -nmax)
        Arblib.add!(tmp,  binA, n; prec=prec)
        Arblib.add!(tmp2, ainA, n; prec=prec)
        Arblib.div!(tmp, tmp, tmp2; prec=prec)
        Arblib.mul!(Rb, Rb, tmp; prec=prec)
        Arblib.mul!(term, Rb, fv[n + off]; prec=prec)
        isodd(n) ? Arblib.sub!(acc, acc, term; prec=prec) :
                   Arblib.add!(acc, acc, term; prec=prec)
    end

    Arblib.mul!(pref, pref, acc; prec=prec)
    return pref
end

"""
    compute_Aminus_acb(p, ν, fn; nmax=80, nmin=-nmax) -> Complex{Arb}

Native-Acb A^ν_- = prefactor · Σ_n (-1)^n P(n) f_n, with the Pochhammer ratio
P(n) = (ν+1+s-iε)_n / (ν+1-s+iε)_n marched incrementally, and
prefactor = 2^{-1-s+iε} e^{-iπ(ν+1+s)/2} e^{-πε/2} built natively in Acb.
`fn` is the Complex{Arb} dict, converted once to a dense Acb vector.
"""
function compute_Aminus_acb(p, ν, fn; nmax::Int=80, nmin::Int=-nmax)
    prec = precision(Arb)
    fv = _fn_vec_from_dict(fn, nmin, nmax, prec)
    return Complex{Arb}(_Aminus_acb(p, ν, fv, 1 - nmin, nmin, nmax, prec))
end
