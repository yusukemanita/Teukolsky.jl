# ============================================================
#  PNSeries{T}: truncated (Laurent) power series in ε = 2ω
#
#  A lightweight commutative-ring type that lets the existing, Wolfram-validated
#  MST machinery (αn/βn/γn recurrence, continued fractions, the ν root finder)
#  run symbolically in the low-frequency parameter ε.  Evaluating the numeric
#  algorithm over this ring makes the post-Newtonian (PN) ε-expansion fall out
#  automatically — no separate symbolic solver is needed.
#
#  Representation:  series = Σ_k  c[k] · ε^k ,  with all terms of exponent
#  k > `order` dropped.  Negative exponents are permitted (Laurent), which is
#  required because the MST β-coefficient develops a 1/ε pole at the index
#  n = -l for Kerr (m ≠ 0) modes.  Only nonzero coefficients are stored.
# ============================================================

struct PNSeries{T}
    order::Int
    c::Dict{Int,T}
    function PNSeries{T}(order::Int, c::AbstractDict) where {T}
        d = Dict{Int,T}()
        for (k, v) in c
            (k <= order && !iszero(v)) && (d[k] = convert(T, v))
        end
        new{T}(order, d)
    end
end

coefftype(::PNSeries{T}) where {T} = T

"""
    getcoeff(s, k)

Coefficient of ε^k in the series `s` (zero if absent / above truncation).
"""
getcoeff(s::PNSeries{T}, k::Integer) where {T} = get(s.c, k, zero(T))

"""
    pneps(order, T=Complex{BigFloat})

The generator ε itself, as a `PNSeries{T}` truncated at ε^order.
"""
pneps(order::Int, ::Type{T}=Complex{BigFloat}) where {T} =
    PNSeries{T}(order, Dict(1 => one(T)))

"""
    pnconst(x, order, T)

The scalar `x` as a constant `PNSeries{T}` (coefficient of ε^0).
"""
pnconst(x, order::Int, ::Type{T}) where {T} =
    PNSeries{T}(order, Dict(0 => convert(T, x)))

Base.zero(s::PNSeries{T}) where {T} = PNSeries{T}(s.order, Dict{Int,T}())
Base.one(s::PNSeries{T}) where {T}  = pnconst(one(T), s.order, T)

# lowest stored exponent (the valuation); `nothing` for the zero series.
function _valuation(s::PNSeries)
    isempty(s.c) && return nothing
    minimum(keys(s.c))
end

# ------------------------------------------------------------
#  Arithmetic
# ------------------------------------------------------------

function Base.:+(a::PNSeries{T}, b::PNSeries{T}) where {T}
    o = min(a.order, b.order)
    d = Dict{Int,T}()
    for (k, v) in a.c; k <= o && (d[k] = get(d, k, zero(T)) + v); end
    for (k, v) in b.c; k <= o && (d[k] = get(d, k, zero(T)) + v); end
    PNSeries{T}(o, d)
end

Base.:-(a::PNSeries{T}) where {T} = PNSeries{T}(a.order, Dict(k => -v for (k, v) in a.c))
Base.:-(a::PNSeries{T}, b::PNSeries{T}) where {T} = a + (-b)

function Base.:*(a::PNSeries{T}, b::PNSeries{T}) where {T}
    o = min(a.order, b.order)
    d = Dict{Int,T}()
    for (ka, va) in a.c, (kb, vb) in b.c
        k = ka + kb
        k <= o && (d[k] = get(d, k, zero(T)) + va * vb)
    end
    PNSeries{T}(o, d)
end

# series long division (handles Laurent denominators / numerators)
function Base.:/(a::PNSeries{T}, b::PNSeries{T}) where {T}
    o  = min(a.order, b.order)
    vb = _valuation(b)
    vb === nothing && throw(DivideError())
    bv = b.c[vb]
    va = _valuation(a)
    va === nothing && return PNSeries{T}(o, Dict{Int,T}())
    d  = Dict{Int,T}()
    for k in (va - vb):o
        E = k + vb
        s = getcoeff(a, E)
        for (i, bi) in b.c
            i > vb || continue
            s -= bi * get(d, E - i, zero(T))
        end
        cval = s / bv
        !iszero(cval) && (d[k] = cval)
    end
    PNSeries{T}(o, d)
end

Base.inv(b::PNSeries{T}) where {T} = pnconst(one(T), b.order, T) / b

function Base.:^(a::PNSeries{T}, n::Integer) where {T}
    n == 0 && return pnconst(one(T), a.order, T)
    n < 0  && return inv(a)^(-n)
    r = a
    for _ in 2:n
        r = r * a
    end
    r
end
Base.literal_pow(::typeof(^), a::PNSeries, ::Val{p}) where {p} = a^p

# ------------------------------------------------------------
#  Scalar interoperability (promote a bare Number to a constant series)
# ------------------------------------------------------------

Base.:+(a::PNSeries{T}, x::Number) where {T} = a + pnconst(x, a.order, T)
Base.:+(x::Number, a::PNSeries) = a + x
Base.:-(a::PNSeries{T}, x::Number) where {T} = a - pnconst(x, a.order, T)
Base.:-(x::Number, a::PNSeries{T}) where {T} = pnconst(x, a.order, T) - a
Base.:*(a::PNSeries{T}, x::Number) where {T} =
    (xx = convert(T, x); PNSeries{T}(a.order, Dict(k => v * xx for (k, v) in a.c)))
Base.:*(x::Number, a::PNSeries) = a * x
Base.:/(a::PNSeries{T}, x::Number) where {T} =
    (xx = convert(T, x); PNSeries{T}(a.order, Dict(k => v / xx for (k, v) in a.c)))
Base.:/(x::Number, a::PNSeries{T}) where {T} = pnconst(x, a.order, T) / a

# ------------------------------------------------------------
#  Transcendental functions (expand about the constant term)
# ------------------------------------------------------------

function Base.sqrt(a::PNSeries{T}) where {T}
    f0 = getcoeff(a, 0)
    h  = a / f0 - one(a)                 # valuation ≥ 1
    acc  = one(a)
    term = one(a)
    coef = one(T)
    for k in 1:a.order
        coef *= (T(1) / 2 - (k - 1)) / k # binomial(1/2, k), incremental
        term  = term * h
        acc   = acc + coef * term
    end
    sqrt(f0) * acc
end

function Base.exp(a::PNSeries{T}) where {T}
    f0 = getcoeff(a, 0)
    h  = a - f0
    acc  = one(a)
    term = one(a)
    for k in 1:a.order
        term = term * h / k
        acc  = acc + term
    end
    exp(f0) * acc
end

function Base.log(a::PNSeries{T}) where {T}
    f0 = getcoeff(a, 0)
    h  = a / f0 - one(a)
    acc  = zero(a)
    term = one(a)
    for k in 1:a.order
        term = term * h
        acc  = acc + (iseven(k) ? -term / k : term / k)
    end
    log(f0) + acc
end

# ------------------------------------------------------------
#  Pochhammer and Gamma-ratio (series-valued)
# ------------------------------------------------------------

"""
    pochhammer(z::PNSeries, n)

Rising factorial (z)_n = z (z+1) ⋯ (z+n-1) over the series ring.
"""
function pochhammer(z::PNSeries{T}, n::Int) where {T}
    n == 0 && return one(z)
    if n > 0
        r = one(z)
        for k in 0:n-1
            r = r * (z + k)
        end
        return r
    else
        r = one(z)
        for k in 1:-n
            r = r / (z - k)
        end
        return r
    end
end

# ln Γ(z) expanded about z₀ = constant term:
#   lnΓ(z₀+δ) = lnΓ(z₀) + Σ_{k≥1} ψ^{(k-1)}(z₀) δ^k / k!
function _lngamma_series(z::PNSeries{T}) where {T}
    z0  = getcoeff(z, 0)
    δ   = z - z0
    acc = pnconst(loggamma(z0), z.order, T)
    fact = one(T)
    term = one(z)
    for k in 1:z.order
        fact *= k
        term  = term * δ
        ψ = k == 1 ? digamma(z0) : polygamma(k - 1, z0)
        acc = acc + (ψ / fact) * term
    end
    acc
end

"""
    gamma_ratio(num::PNSeries, den::PNSeries)

Γ(num)/Γ(den) as a truncated series, via the polygamma base-point expansion of
each lnΓ.  Both constant terms must avoid the poles of Γ (non-positive integers).
"""
gamma_ratio(num::PNSeries, den::PNSeries) =
    exp(_lngamma_series(num) - _lngamma_series(den))

# ------------------------------------------------------------
#  Evaluation
# ------------------------------------------------------------

"""
    evalseries(s, εval)

Evaluate the truncated series at a numeric value `εval` (Σ_k c[k] εval^k).
"""
function evalseries(s::PNSeries{T}, εval) where {T}
    S   = typeof(one(T) * one(εval))
    acc = zero(S)
    for (k, v) in s.c
        acc += v * εval^k
    end
    acc
end

# Drop all terms of exponent above `o`.
_truncate(s::PNSeries{T}, o::Int) where {T} =
    PNSeries{T}(o, Dict(k => v for (k, v) in s.c if k <= o))

function Base.show(io::IO, s::PNSeries{T}) where {T}
    print(io, "PNSeries{$T}(order=$(s.order)")
    for k in sort!(collect(keys(s.c)))
        print(io, ", ε^$k: ", s.c[k])
    end
    print(io, ")")
end
