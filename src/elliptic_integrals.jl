# ============================================================
#  Elliptic integrals & Jacobi elliptic functions
#
#  Shared, type-generic (Float64 / BigFloat / …) and BigFloat-safe
#  implementations used by the geodesic backends.
#
#  CONVENTION (matches Wolfram Mathematica and the KerrGeodesics paclet):
#  every routine here uses the PARAMETER  m = k²  (NOT the modulus k).
#  i.e.
#      ellK(m)        == EllipticK[m]
#      ellE(m)        == EllipticE[m]
#      ellF(φ,m)      == EllipticF[φ, m]
#      ellEinc(φ,m)   == EllipticE[φ, m]
#      ellPi(n,m)     == EllipticPi[n, m]          (complete)
#      ellPi(n,φ,m)   == EllipticPi[n, φ, m]       (incomplete)
#      jacobi_sn(u,m) == JacobiSN[u, m]
#      jacobi_am(u,m) == JacobiAmplitude[u, m]
#
#  The integrals are reduced to Carlson symmetric forms rf, rd, rj, rc
#  evaluated with the duplication algorithm (Carlson 1995 / Numerical
#  Recipes 3rd ed. §6.12), which converges to the working precision of the
#  real type used.
# ============================================================

# --- working-precision convergence threshold for the Carlson duplication ---
# The truncated symmetric series carries an O(δ⁶) error where δ is the largest
# remaining deviation from the running mean; stopping at δ < eps^(1/5) keeps the
# truncation error ≈ eps^(6/5) ≪ eps, i.e. full working precision.
@inline _carlson_tol(::Type{T}) where {T<:AbstractFloat} = eps(T)^(one(T) / 5)

# ------------------------------------------------------------
#  Carlson R_C(x, y)
# ------------------------------------------------------------
"""
    rc(x, y)

Carlson degenerate symmetric form
``R_C(x,y) = \\tfrac12 \\int_0^\\infty (t+x)^{-1/2}(t+y)^{-1}\\,dt``.
Type-generic and BigFloat-safe; handles `y < 0` (Cauchy principal value).
"""
function rc(x, y)
    T = float(promote_type(typeof(x), typeof(y)))
    return _rc(T(x), T(y))
end

function _rc(x::T, y::T) where {T<:AbstractFloat}
    tol = _carlson_tol(T)
    if y > 0
        xt = x; yt = y; w = one(T)
    else
        # y ≤ 0: Cauchy principal value via the standard shift.
        xt = x - y
        yt = -y
        w = sqrt(x) / sqrt(xt)
    end
    s = T(Inf)
    ave = one(T)
    while abs(s) > tol
        lam = 2 * sqrt(xt) * sqrt(yt) + yt
        xt = (xt + lam) / 4
        yt = (yt + lam) / 4
        ave = (xt + yt + yt) / 3
        s = (yt - ave) / ave
    end
    C1 = T(3 // 10); C2 = T(1 // 7); C3 = T(3 // 8); C4 = T(9 // 22)
    return w * (1 + s * s * (C1 + s * (C2 + s * (C3 + s * C4)))) / sqrt(ave)
end

# ------------------------------------------------------------
#  Carlson R_F(x, y, z)
# ------------------------------------------------------------
"""
    rf(x, y, z)

Carlson symmetric form of the first kind
``R_F(x,y,z) = \\tfrac12 \\int_0^\\infty [(t+x)(t+y)(t+z)]^{-1/2}\\,dt``.
"""
function rf(x, y, z)
    T = float(promote_type(typeof(x), typeof(y), typeof(z)))
    return _rf(T(x), T(y), T(z))
end

function _rf(x::T, y::T, z::T) where {T<:AbstractFloat}
    tol = _carlson_tol(T)
    delx = T(Inf); dely = T(Inf); delz = T(Inf); ave = one(T)
    while max(abs(delx), abs(dely), abs(delz)) > tol
        sx = sqrt(x); sy = sqrt(y); sz = sqrt(z)
        lam = sx * (sy + sz) + sy * sz
        x = (x + lam) / 4
        y = (y + lam) / 4
        z = (z + lam) / 4
        ave = (x + y + z) / 3
        delx = 1 - x / ave
        dely = 1 - y / ave
        delz = 1 - z / ave
    end
    e2 = delx * dely - delz * delz
    e3 = delx * dely * delz
    C1 = T(1 // 24); C2 = T(1 // 10); C3 = T(3 // 44); C4 = T(1 // 14)
    return (1 + (C1 * e2 - C2 - C3 * e3) * e2 + C4 * e3) / sqrt(ave)
end

# ------------------------------------------------------------
#  Carlson R_D(x, y, z)  ( = R_J(x,y,z,z) )
# ------------------------------------------------------------
"""
    rd(x, y, z)

Carlson symmetric form of the second kind
``R_D(x,y,z) = \\tfrac32 \\int_0^\\infty [(t+x)(t+y)]^{-1/2}(t+z)^{-3/2}\\,dt``.
"""
function rd(x, y, z)
    T = float(promote_type(typeof(x), typeof(y), typeof(z)))
    return _rd(T(x), T(y), T(z))
end

function _rd(x::T, y::T, z::T) where {T<:AbstractFloat}
    tol = _carlson_tol(T)
    sum = zero(T)
    fac = one(T)
    delx = T(Inf); dely = T(Inf); delz = T(Inf); ave = one(T)
    while max(abs(delx), abs(dely), abs(delz)) > tol
        sx = sqrt(x); sy = sqrt(y); sz = sqrt(z)
        lam = sx * (sy + sz) + sy * sz
        sum += fac / (sz * (z + lam))
        fac /= 4
        x = (x + lam) / 4
        y = (y + lam) / 4
        z = (z + lam) / 4
        ave = (x + y + 3 * z) / 5
        delx = (ave - x) / ave
        dely = (ave - y) / ave
        delz = (ave - z) / ave
    end
    ea = delx * dely
    eb = delz * delz
    ec = ea - eb
    ed = ea - 6 * eb
    ee = ed + ec + ec
    C1 = T(3 // 14); C2 = T(1 // 6); C3 = T(9 // 22)
    C4 = T(3 // 26); C5 = T(9 // 88); C6 = T(9 // 52)
    return 3 * sum + fac * (1 + ed * (-C1 + C5 * ed - C6 * delz * ee) +
                            delz * (C2 * ee + delz * (-C3 * ec + delz * C4 * ea))) /
                     (ave * sqrt(ave))
end

# ------------------------------------------------------------
#  Carlson R_J(x, y, z, p)
# ------------------------------------------------------------
"""
    rj(x, y, z, p)

Carlson symmetric form of the third kind
``R_J(x,y,z,p) = \\tfrac32 \\int_0^\\infty [(t+x)(t+y)(t+z)]^{-1/2}(t+p)^{-1}\\,dt``.
Handles `p < 0` (Cauchy principal value) via the Carlson/NR transformation.
"""
function rj(x, y, z, p)
    T = float(promote_type(typeof(x), typeof(y), typeof(z), typeof(p)))
    return _rj(T(x), T(y), T(z), T(p))
end

function _rj(x::T, y::T, z::T, p::T) where {T<:AbstractFloat}
    tol = _carlson_tol(T)
    if p > 0
        xt = x; yt = y; zt = z; pt = p
        a = zero(T); b = zero(T); rcx = zero(T); negp = false
    else
        # Negative p: Cauchy principal value (NR §6.12).
        xt = min(min(x, y), z)
        zt = max(max(x, y), z)
        yt = x + y + z - xt - zt
        a = 1 / (yt - p)
        b = a * (zt - yt) * (yt - xt)
        pt = yt + b
        rho = xt * zt / yt
        tau = p * pt / yt
        rcx = _rc(rho, tau)
        negp = true
    end
    sum = zero(T)
    fac = one(T)
    delx = T(Inf); dely = T(Inf); delz = T(Inf); delp = T(Inf); ave = one(T)
    while max(abs(delx), abs(dely), abs(delz), abs(delp)) > tol
        sx = sqrt(xt); sy = sqrt(yt); sz = sqrt(zt)
        lam = sx * (sy + sz) + sy * sz
        alpha = (pt * (sx + sy + sz) + sx * sy * sz)^2
        beta = pt * (pt + lam)^2
        sum += fac * _rc(alpha, beta)
        fac /= 4
        xt = (xt + lam) / 4
        yt = (yt + lam) / 4
        zt = (zt + lam) / 4
        pt = (pt + lam) / 4
        ave = (xt + yt + zt + 2 * pt) / 5
        delx = (ave - xt) / ave
        dely = (ave - yt) / ave
        delz = (ave - zt) / ave
        delp = (ave - pt) / ave
    end
    ea = delx * (dely + delz) + dely * delz
    eb = delx * dely * delz
    ec = delp * delp
    ed = ea - 3 * ec
    ee = eb + 2 * delp * (ea - ec)
    C1 = T(3 // 14); C2 = T(1 // 3); C3 = T(3 // 22); C4 = T(3 // 26)
    C5 = T(9 // 88); C6 = T(9 // 52); C7 = T(1 // 6); C8 = T(3 // 11)
    ans = 3 * sum + fac * (1 + ed * (-C1 + C5 * ed - C6 * ee) +
                           eb * (C7 + delp * (-C8 + delp * C4)) +
                           delp * ea * (C2 - delp * C3) - C2 * delp * ec) /
                    (ave * sqrt(ave))
    if negp
        ans = a * (b * ans + 3 * (rcx - _rf(xt, yt, zt)))
    end
    return ans
end

# ============================================================
#  Complete elliptic integrals  (parameter m = k²)
# ============================================================
"""
    ellK(m)

Complete elliptic integral of the first kind, `EllipticK[m]` (parameter `m=k²`).
"""
function ellK(m)
    T = float(typeof(m))
    mm = T(m)
    return _rf(zero(T), 1 - mm, one(T))
end

"""
    ellE(m)

Complete elliptic integral of the second kind, `EllipticE[m]` (parameter `m=k²`).
"""
function ellE(m)
    T = float(typeof(m))
    mm = T(m)
    return _rf(zero(T), 1 - mm, one(T)) - (mm / 3) * _rd(zero(T), 1 - mm, one(T))
end

"""
    ellPi(n, m)

Complete elliptic integral of the third kind, `EllipticPi[n, m]`
(characteristic `n`, parameter `m=k²`):
``\\int_0^{\\pi/2} d\\theta\\,[(1-n\\sin^2\\theta)\\sqrt{1-m\\sin^2\\theta}]^{-1}``.
"""
function ellPi(n, m)
    T = float(promote_type(typeof(n), typeof(m)))
    nn = T(n); mm = T(m)
    return _rf(zero(T), 1 - mm, one(T)) +
           (nn / 3) * _rj(zero(T), 1 - mm, one(T), 1 - nn)
end

# ============================================================
#  Incomplete elliptic integrals  (parameter m = k²)
#
#  The Carlson reductions below are valid for the principal amplitude
#  φ ∈ [-π/2, π/2].  For general φ we use the quasi-periodicity that
#  Mathematica's analytic continuation obeys,
#      F(φ+Nπ, m)    = F(φ, m)    + 2N K(m)
#      E(φ+Nπ, m)    = E(φ, m)    + 2N E(m)
#      Π(n,φ+Nπ, m)  = Π(n,φ, m)  + 2N Π(n, m),
#  with N chosen so the residual amplitude lies in [-π/2, π/2].
# ============================================================

# split φ = N·π + φ0 with φ0 ∈ [-π/2, π/2]
@inline function _amp_reduce(φ::T) where {T<:AbstractFloat}
    N = round(φ / T(π))
    return N, φ - N * T(π)
end

"""
    ellF(φ, m)

Incomplete elliptic integral of the first kind, `EllipticF[φ, m]`
(amplitude `φ`, parameter `m=k²`).
"""
function ellF(φ, m)
    T = float(promote_type(typeof(φ), typeof(m)))
    return _ellF(T(φ), T(m))
end

function _ellF(φ::T, m::T) where {T<:AbstractFloat}
    N, φ0 = _amp_reduce(φ)
    base = iszero(N) ? zero(T) : 2 * N * _rf(zero(T), 1 - m, one(T))
    s = sin(φ0); c = cos(φ0)
    s2 = s * s
    return base + s * _rf(c * c, 1 - m * s2, one(T))
end

"""
    ellEinc(φ, m)

Incomplete elliptic integral of the second kind, `EllipticE[φ, m]`
(amplitude `φ`, parameter `m=k²`).
"""
function ellEinc(φ, m)
    T = float(promote_type(typeof(φ), typeof(m)))
    return _ellEinc(T(φ), T(m))
end

function _ellEinc(φ::T, m::T) where {T<:AbstractFloat}
    N, φ0 = _amp_reduce(φ)
    base = iszero(N) ? zero(T) :
           2 * N * (_rf(zero(T), 1 - m, one(T)) - (m / 3) * _rd(zero(T), 1 - m, one(T)))
    s = sin(φ0); c = cos(φ0)
    s2 = s * s; c2 = c * c
    return base + s * _rf(c2, 1 - m * s2, one(T)) -
           (m / 3) * s * s2 * _rd(c2, 1 - m * s2, one(T))
end

"""
    ellPi(n, φ, m)

Incomplete elliptic integral of the third kind, `EllipticPi[n, φ, m]`
(characteristic `n`, amplitude `φ`, parameter `m=k²`):
``\\int_0^{\\varphi} d\\theta\\,[(1-n\\sin^2\\theta)\\sqrt{1-m\\sin^2\\theta}]^{-1}``.
"""
function ellPi(n, φ, m)
    T = float(promote_type(typeof(n), typeof(φ), typeof(m)))
    return _ellPiinc(T(n), T(φ), T(m))
end

function _ellPiinc(n::T, φ::T, m::T) where {T<:AbstractFloat}
    N, φ0 = _amp_reduce(φ)
    base = iszero(N) ? zero(T) :
           2 * N * (_rf(zero(T), 1 - m, one(T)) +
                    (n / 3) * _rj(zero(T), 1 - m, one(T), 1 - n))
    s = sin(φ0); c = cos(φ0)
    s2 = s * s; c2 = c * c
    return base + s * _rf(c2, 1 - m * s2, one(T)) +
           (n / 3) * s * s2 * _rj(c2, 1 - m * s2, one(T), 1 - n * s2)
end

# ============================================================
#  Jacobi elliptic functions  (parameter m = k²)
#
#  Descending Landen / AGM transformation (Abramowitz & Stegun 16.4),
#  BigFloat-safe.  Valid for m ∈ [0, 1).
# ============================================================

# core AGM descending Landen: returns the amplitude am(u, m)
function _jacobi_amplitude(u::T, m::T) where {T<:AbstractFloat}
    tol = eps(T)
    # Build the AGM sequences a_n, b_n, c_n.
    a = one(T)
    b = sqrt(1 - m)            # = k'
    c = sqrt(m)                # = k
    cs = T[c]
    as = T[a]
    n = 0
    # iterate until c_n is negligible (quadratic AGM convergence)
    maxiter = 8 * precision(T) + 64   # ample headroom for any precision
    while abs(c) > tol * abs(a) && n < maxiter
        a2 = (a + b) / 2
        b2 = sqrt(a * b)
        c2 = (a - b) / 2
        a = a2; b = b2; c = c2
        push!(as, a)
        push!(cs, c)
        n += 1
    end
    # φ_n = 2^n a_n u  (use the final index n)
    φ = ldexp(a, n) * u   # 2^n * a_n * u
    # descend: φ_{k-1} = ½ (φ_k + asin( (c_k / a_k) sinφ_k ))
    for k in n:-1:1
        φ = (φ + asin(clamp(cs[k+1] / as[k+1] * sin(φ), -one(T), one(T)))) / 2
    end
    return φ
end

"""
    jacobi_am(u, m)

Jacobi amplitude `JacobiAmplitude[u, m]` (parameter `m=k²`), BigFloat-safe.
"""
function jacobi_am(u, m)
    T = float(promote_type(typeof(u), typeof(m)))
    return _jacobi_amplitude(T(u), T(m))
end

"""
    jacobi_sn(u, m)

Jacobi elliptic function `JacobiSN[u, m] = sin(am(u, m))` (parameter `m=k²`),
BigFloat-safe.
"""
function jacobi_sn(u, m)
    T = float(promote_type(typeof(u), typeof(m)))
    return sin(_jacobi_amplitude(T(u), T(m)))
end
