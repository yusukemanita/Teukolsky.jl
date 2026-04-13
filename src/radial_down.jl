# ============================================================
#  Rdown: Downgoing radial solution (at infinity, HypergeometricU-based)
#
#  Rdown = R^ν_+ / norm,  where
#    norm = A^ν_+ · ω^{-1} · exp(-i(ε ln ε - (1-κ)/2 · ε))
#
#  R^ν_+ is the MST Coulomb-wave expansion (Sasaki-Tagoshi eq. 159):
#
#  R^ν_+(r) = prefac(ẑ) × Σ_n i^n f^ν_n (2ẑ)^n Ψ(n+ν+1-s+iε, 2n+2ν+2; 2iẑ)
#
#  where ẑ = ε(r - r_-)/2
#
#  prefac = 2^ν e^{-πε} e^{iπ(ν+1-s)} Γ(ν+1-s+iε)/Γ(ν+1+s-iε)
#           × e^{-iẑ} ẑ^{ν+iε_+} (ẑ-εκ)^{-s-iε_+}
#
#  Compare with R^ν_- (= Rup, radial_up.jl):
#    - U argument: +2iẑ  (vs -2iẑ)
#    - U first param: ν+1-s+iε  (vs ν+1+s-iε)
#    - series coeff: i^n fn  (vs (-1)^n Poch/Poch fn)
#    - prefac sign of ẑ exponent: e^{-iẑ}  (vs e^{+iẑ})
# ============================================================

"""
    Rdown(p::MSTParams, ν, fn, r; nmax=40, tol=1e-14)

Compute the downgoing radial Teukolsky solution at Boyer-Lindquist radius r.
Rdown = R^ν_+ / norm, normalized so that at infinity:
    Rdown ~ r^{-2s-1} e^{-iωr*}
(pure ingoing wave at infinity).

norm = A^ν_+ · ω^{-1} · exp(-i(ε ln ε - (1-κ)/2 · ε))
"""
function Rdown(p::MSTParams, ν, fn, r; nmax::Int=40, tol::Float64=1e-14)
    ϵ, κ, τ, s = p.ϵ, p.κ, p.τ, p.s
    rm = p.rm
    zhat = complex(ϵ * (r - rm) / 2)

    # HUParams for R^ν_+: aU = ν+1-s+iε, bU = 2ν+2, c = +2iẑ
    hp = HUParams(ν + 1 - s + im*ϵ, 2ν + 2, 2im * zhat)

    ϵp = p.ϵp  # = (ε+τ)/2

    # Prefactor for R^ν_+
    prefac = 2^ν * exp(-π*ϵ) * exp(im*π*(ν + 1 - s)) *
             _cgamma(complex(ν + 1 - s + im*ϵ)) / _cgamma(complex(ν + 1 + s - im*ϵ)) *
             exp(-im*zhat) * zhat^(ν + im*ϵp) *
             (zhat - ϵ*κ)^(-s - im*ϵp)

    # HU cache with recurrence + fallback
    hu_cache = Dict{Int, ComplexF64}()

    function get_hu(n::Int)
        haskey(hu_cache, n) && return hu_cache[n]
        if n == 0 || n == 1
            val = hu_exact(hp, n)
        elseif n >= 2
            t1, t2 = hu_up(hp, n, get_hu(n-2), get_hu(n-1))
            val = t1 + t2
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val)) > 2.0
                val = hu_exact(hp, n)
            end
        else
            t1, t2 = hu_down(hp, n, get_hu(n+2), get_hu(n+1))
            val = t1 + t2
            if abs(val) > 0 && max(abs(t1/val), abs(t2/val)) > 2.0
                val = hu_exact(hp, n)
            end
        end
        hu_cache[n] = val
        return val
    end

    # Series coefficient: just fn
    # (the i^n from the image formula is already absorbed into
    #  hu_exact via c^n = (2iẑ)^n = i^n (2ẑ)^n)
    function fplus(n::Int)
        fn_n = get(fn, n, complex(0.0))
        return fn_n
    end

    # Sum bidirectionally
    result = complex(0.0)
    for n in 0:nmax
        fp = fplus(n)
        iszero(fp) && continue
        term = prefac * fp * get_hu(n)
        result += term
        n > 0 && abs(term) < tol * abs(result) + tol && break
    end

    res_down = complex(0.0)
    for n in -1:-1:-nmax
        fp = fplus(n)
        iszero(fp) && continue
        term = prefac * fp * get_hu(n)
        res_down += term
        abs(term) < tol * abs(res_down) + tol && break
    end

    Rnu_plus = result + res_down

    # Normalization: Rdown = R^ν_+ / norm
    # norm = A^ν_+ · ω^{-1} · exp(-i(ε ln ε - (1-κ)/2 · ε))
    Ap = compute_Aplus(p, ν, fn; nmax=nmax)
    ω_c = p.ω
    phase = exp(-im * (ϵ * log(ϵ) - (1 - κ) / 2 * ϵ))
    norm_val = Ap * ω_c^(-1) * phase

    return Rnu_plus / norm_val
end
