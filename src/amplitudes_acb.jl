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

# Float64 midpoint magnitude of an Acb — the decidable-guard convention
# (huge exponents saturate to Inf/0, which the callers treat conservatively).
@inline _mag_acb_f64(t::Acb) =
    abs(complex(Float64(Arblib.midref(Arblib.realref(t))),
                Float64(Arblib.midref(Arblib.imagref(t)))))

# Tail-criterion ratio for the window-adequacy check: q = |t|·2^tailbits/|acc|
# decided on the Float64 MIDPOINT of the Arb ratio (ratio-first — |t|/|acc|
# alone spans hundreds of digits at 1280 bits and would over/underflow
# Float64; after the 2^tailbits shift the decision boundary sits at q = 1, so
# Float64 saturation to Inf/0 still decides correctly).  Non-finite → not ok.
function _tail_q_acb(t::Acb, absacc::Arb, tailbits::Int, prec::Int)
    r = Arb(0)
    Arblib.abs!(r, t)
    Arblib.mul_2exp!(r, r, tailbits)
    Arblib.div!(r, r, absacc; prec=prec)
    return _mid_f64(r)
end

# Internal A^ν_+ over a dense Acb vector fv (fv[n + off] = f_n).  Any
# nmin ≤ nmax; fv must cover nmin:nmax at offset `off`.  Returns
# `(val::Acb, tail_ok::Bool, cancel::Float64)`:
#   tail_ok — the last two terms of BOTH legs satisfy |term| < 2^-tailbits·|Σ|
#             (the converge-or-error window-adequacy criterion; for A⁺ the
#             terms are the f_n themselves).  `tailbits ≤ 0` skips the check
#             (exact-window contract of the public wrapper) and reports true.
#   cancel  — max|partial| / |Σ| from Float64 midpoints (the cancellation
#             floor scale for _certify_mst_sum-style certification).
function _Aplus_acb(p, ν, fv::Vector{Acb}, off::Int, nmin::Int, nmax::Int,
                    prec::Int; tailbits::Int=-1)
    pref = Acb(0)
    _Aplus_pref_acb!(pref, p, ν, prec)
    acc = Acb(0)
    smax = 0.0
    for n in nmin:nmax
        Arblib.add!(acc, acc, fv[n + off]; prec=prec)
        smax = max(smax, _mag_acb_f64(acc), _mag_acb_f64(fv[n + off]))
    end
    tail_ok = true
    cancel = smax / _mag_acb_f64(acc)
    if tailbits > 0
        absacc = Arb(0)
        Arblib.abs!(absacc, acc)
        for k in (nmax, max(nmax - 1, nmin), nmin, min(nmin + 1, nmax))
            tail_ok &= _tail_q_acb(fv[k + off], absacc, tailbits, prec) < 1.0
        end
    end
    Arblib.mul!(pref, pref, acc; prec=prec)
    return pref, tail_ok, cancel
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
    val, _, _ = _Aplus_acb(p, ν, fv, 1 - nmin, nmin, nmax, prec)
    return Complex{Arb}(val)
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
# w(n) = (-1)^n (ain)_n/(bin)_n is marched incrementally from the anchor
# n0 = clamp(0, nmin, nmax) in both directions (one Acb mul + div per term),
# exactly mirroring the generic compute_Aminus (amplitudes.jl).  Honors the
# FULL nmin:nmax range (any nmin ≤ nmax, including nmin < -nmax and nmin > 0);
# fv must cover nmin:nmax at offset `off`.  Returns
# `(val::Acb, tail_ok::Bool, cancel::Float64)` — see _Aplus_acb; here the
# boundary terms are w(n)·f_n, reconstructed after each leg from the final
# marched weight (one extra mul/div per leg, no per-iteration cost).
function _Aminus_acb(p, ν, fv::Vector{Acb}, off::Int, nmin::Int, nmax::Int,
                     prec::Int; tailbits::Int=-1)
    pref = Acb(0); ainA = Acb(0); binA = Acb(0)
    _Aminus_pref_acb!(pref, ainA, binA, p, ν, prec)
    acc = Acb(0); tmp = Acb(0); tmp2 = Acb(0); term = Acb(0)
    smax = 0.0

    # anchor n0 = clamp(0, nmin, nmax):  P0 = (ain)_{n0} / (bin)_{n0}
    #   n0 > 0:  ∏_{k=0}^{n0-1} (ain+k)/(bin+k)
    #   n0 < 0:  ∏_{k=1}^{-n0}  (bin−k)/(ain−k)   ((z)_{-j} = 1/∏(z−k))
    n0 = clamp(0, nmin, nmax)
    P0 = Acb(1)
    for k in 0:n0-1
        Arblib.add!(tmp,  ainA, k; prec=prec)
        Arblib.add!(tmp2, binA, k; prec=prec)
        Arblib.div!(tmp, tmp, tmp2; prec=prec)
        Arblib.mul!(P0, P0, tmp; prec=prec)
    end
    for k in 1:-n0
        Arblib.sub!(tmp,  binA, k; prec=prec)
        Arblib.sub!(tmp2, ainA, k; prec=prec)
        Arblib.div!(tmp, tmp, tmp2; prec=prec)
        Arblib.mul!(P0, P0, tmp; prec=prec)
    end

    # anchor term: (-1)^{n0} P(n0) f_{n0}
    Arblib.mul!(term, P0, fv[n0 + off]; prec=prec)
    isodd(n0) ? Arblib.sub!(acc, acc, term; prec=prec) :
                Arblib.add!(acc, acc, term; prec=prec)
    smax = max(smax, _mag_acb_f64(term), _mag_acb_f64(acc))

    # forward n = n0+1..nmax:  P(n) = P(n-1)·(ain+n-1)/(bin+n-1)
    Rf = Acb(0)
    Arblib.set!(Rf, P0)
    for n in n0+1:nmax
        Arblib.add!(tmp,  ainA, n - 1; prec=prec)
        Arblib.add!(tmp2, binA, n - 1; prec=prec)
        Arblib.div!(tmp, tmp, tmp2; prec=prec)
        Arblib.mul!(Rf, Rf, tmp; prec=prec)
        Arblib.mul!(term, Rf, fv[n + off]; prec=prec)
        isodd(n) ? Arblib.sub!(acc, acc, term; prec=prec) :
                   Arblib.add!(acc, acc, term; prec=prec)
        smax = max(smax, _mag_acb_f64(term), _mag_acb_f64(acc))
    end
    # boundary terms of the forward leg: w(nmax)·f_nmax is still in `term`;
    # step Rf back once for w(nmax-1)·f_{nmax-1}
    tU1 = Acb(0); tU2 = Acb(0)
    if tailbits > 0 && nmax > n0
        Arblib.set!(tU1, term)
        if nmax - 1 > n0
            Arblib.add!(tmp,  binA, nmax - 1; prec=prec)
            Arblib.add!(tmp2, ainA, nmax - 1; prec=prec)
            Arblib.div!(tmp, tmp, tmp2; prec=prec)
            Arblib.mul!(tmp, tmp, Rf; prec=prec)
            Arblib.mul!(tU2, tmp, fv[nmax - 1 + off]; prec=prec)
        else
            Arblib.set!(tU2, tU1)
        end
    end

    # backward n = n0-1..nmin:  P(n) = P(n+1)·(bin+n)/(ain+n)
    Rb = Acb(0)
    Arblib.set!(Rb, P0)
    for n in n0-1:-1:nmin
        Arblib.add!(tmp,  binA, n; prec=prec)
        Arblib.add!(tmp2, ainA, n; prec=prec)
        Arblib.div!(tmp, tmp, tmp2; prec=prec)
        Arblib.mul!(Rb, Rb, tmp; prec=prec)
        Arblib.mul!(term, Rb, fv[n + off]; prec=prec)
        isodd(n) ? Arblib.sub!(acc, acc, term; prec=prec) :
                   Arblib.add!(acc, acc, term; prec=prec)
        smax = max(smax, _mag_acb_f64(term), _mag_acb_f64(acc))
    end

    tail_ok = true
    cancel = smax / _mag_acb_f64(acc)
    if tailbits > 0
        absacc = Arb(0)
        Arblib.abs!(absacc, acc)
        if nmax > n0
            tail_ok &= _tail_q_acb(tU1, absacc, tailbits, prec) < 1.0
            tail_ok &= _tail_q_acb(tU2, absacc, tailbits, prec) < 1.0
        end
        if nmin < n0
            # w(nmin)·f_nmin is still in `term`; step Rb forward once for
            # w(nmin+1)·f_{nmin+1}
            tail_ok &= _tail_q_acb(term, absacc, tailbits, prec) < 1.0
            if nmin + 1 < n0
                Arblib.add!(tmp,  ainA, nmin; prec=prec)
                Arblib.add!(tmp2, binA, nmin; prec=prec)
                Arblib.div!(tmp, tmp, tmp2; prec=prec)
                Arblib.mul!(tmp, tmp, Rb; prec=prec)
                Arblib.mul!(tmp, tmp, fv[nmin + 1 + off]; prec=prec)
                tail_ok &= _tail_q_acb(tmp, absacc, tailbits, prec) < 1.0
            end
        end
    end

    Arblib.mul!(pref, pref, acc; prec=prec)
    return pref, tail_ok, cancel
end

"""
    compute_Aminus_acb(p, ν, fn; nmax=80, nmin=-nmax) -> Complex{Arb}

Native-Acb A^ν_- = prefactor · Σ_{n=nmin}^{nmax} (-1)^n P(n) f_n, with the
Pochhammer ratio P(n) = (ν+1+s-iε)_n / (ν+1-s+iε)_n marched incrementally from
the anchor n₀ = clamp(0, nmin, nmax), and
prefactor = 2^{-1-s+iε} e^{-iπ(ν+1+s)/2} e^{-πε/2} built natively in Acb.
`fn` is the Complex{Arb} dict, converted once to a dense Acb vector; it must
contain every n in nmin:nmax.  The FULL documented (nmin, nmax) range is
summed, exactly like the generic `compute_Aminus` (any nmin ≤ nmax, including
nmin < -nmax and nmin > 0).
"""
function compute_Aminus_acb(p, ν, fn; nmax::Int=80, nmin::Int=-nmax)
    prec = precision(Arb)
    fv = _fn_vec_from_dict(fn, nmin, nmax, prec)
    val, _, _ = _Aminus_acb(p, ν, fv, 1 - nmin, nmin, nmax, prec)
    return Complex{Arb}(val)
end
