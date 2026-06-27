# ============================================================
#  Special Kerr orbits
#
#  Innermost stable circular orbit (ISCO), photon-sphere radius,
#  innermost bound spherical orbit (IBSO), innermost stable spherical
#  orbit (ISSO) and the separatrix p_s(a,e,x).  Geometric units M = 1.
#
#  Closed forms from Bardeen–Press–Teukolsky ApJ 178 (1972), Teo GRG 35
#  (2003), and Stein & Warburton arXiv:1912.07609, transcribed from the
#  Wolfram KerrGeodesics paclet (KerrGeodesics`SpecialOrbits`).  Where no
#  closed form exists we solve the relevant polynomial with a
#  self-contained bracketed bisection (type-generic / BigFloat-capable).
#
#  All routines are type-generic: the working real type is recovered from
#  the inputs so Float64 and BigFloat both deliver full precision.
# ============================================================

# Promote (a, x) [and optionally e] to a common working float type.
@inline function _ax_R(a, x)
    R = float(promote_type(typeof(a), typeof(x)))
    return R, R(a), R(x)
end

# ------------------------------------------------------------
#  Self-contained bracketed bisection root finder (BigFloat-safe)
# ------------------------------------------------------------
function _bisect_root(f, lo::T, hi::T) where {T<:AbstractFloat}
    a = lo; b = hi
    fa = f(a); fb = f(b)
    iszero(fa) && return a
    iszero(fb) && return b
    if sign(fa) == sign(fb)
        throw(ArgumentError("root not bracketed: f($a)=$fa, f($b)=$fb"))
    end
    maxit = 8 * precision(T) + 200
    tol = eps(T)
    for _ in 1:maxit
        m = (a + b) / 2
        fm = f(m)
        if iszero(fm) || (b - a) <= 4 * tol * max(abs(m), one(T))
            return m
        end
        if sign(fm) == sign(fa)
            a = m; fa = fm
        else
            b = m; fb = fm
        end
    end
    return (a + b) / 2
end

# ------------------------------------------------------------
#  ISCO  (Bardeen–Press–Teukolsky, Eq. 2.21)
# ------------------------------------------------------------
"""
    kerr_geo_isco(a, x) -> r

Radius of the innermost stable circular orbit for an equatorial Kerr
geodesic (`x = ±1`).  `x = +1` prograde, `x = -1` retrograde.
"""
function kerr_geo_isco(a, x)
    R, a, x = _ax_R(a, x)
    iszero(a) && return R(6)
    Z1 = 1 + cbrt(1 - a^2) * (cbrt(1 + a) + cbrt(1 - a))
    Z2 = sqrt(3 * a^2 + Z1^2)
    return 3 + Z2 - _sgn(a * x) * sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
end

# ------------------------------------------------------------
#  Photon-sphere radius
# ------------------------------------------------------------
"""
    kerr_geo_photon_sphere_radius(a, x) -> r

Radius of the spherical photon orbit.  Closed forms for equatorial
(`x = ±1`, BPT 1972) and polar (`x = 0`, Teo 2003); otherwise a
bracketed numerical solve.
"""
function kerr_geo_photon_sphere_radius(a, x)
    R, a, x = _ax_R(a, x)
    iszero(a) && return R(3)
    if x == one(R)
        return 2 * (1 + cos((2 // 3) * acos(-a)))
    elseif x == -one(R)
        return 2 * (1 + cos((2 // 3) * acos(a)))
    elseif iszero(x)
        return 1 + 2 * sqrt(1 - a^2 / 3) *
               cos((1 // 3) * acos((1 - a^2) / (1 - a^2 / 3)^(R(3) / 2)))
    end
    # generic inclination: solve 1 - u0² - x² = 0 for r (Teo / KerrGeodesics)
    req = kerr_geo_photon_sphere_radius(a, _sgn(x))
    rpolar = kerr_geo_photon_sphere_radius(a, zero(R))
    function g(r)
        Φ = -((r^3 - 3 * r^2 + a^2 * r + a^2) / (a * (r - 1)))
        Qc = -((r^3 * (r^3 - 6 * r^2 + 9 * r - 4 * a^2)) / (a^2 * (r - 1)^2))
        u0Sq = ((a^2 - Qc - Φ^2) + sqrt((a^2 - Qc - Φ^2)^2 + 4 * a^2 * Qc)) / (2 * a^2)
        return 1 - u0Sq - x^2
    end
    lo, hi = minmax(req, rpolar)
    return _bisect_root(g, lo, hi)
end

# ------------------------------------------------------------
#  IBSO  (innermost bound spherical orbit)
# ------------------------------------------------------------
"""
    kerr_geo_ibso(a, x) -> r

Innermost bound spherical orbit radius.  Closed forms for Schwarzschild,
equatorial (BPT 1972) and polar; otherwise a bracketed solve of the
Stein–Warburton IBSO polynomial.
"""
function kerr_geo_ibso(a, x)
    R, a, x = _ax_R(a, x)
    iszero(a) && return R(4)
    if x == one(R)
        return 2 - a + 2 * sqrt(1 - a)
    elseif x == -one(R)
        return 2 + a + 2 * sqrt(1 + a)
    elseif iszero(x)
        δ = 27 * a^4 - 8 * a^6 + 3 * sqrt(R(3)) * sqrt(27 * a^8 - 16 * a^10)
        δ3 = cbrt(δ)
        s = 6 - 2 * a^2 + 4 * a^4 / δ3 + δ3
        return 1 + sqrt(12 - 4 * a^2 - (6 * sqrt(R(6)) * (-2 + a^2)) / sqrt(s) -
                        4 * a^4 / δ3 - δ3) / sqrt(R(6)) + sqrt(s) / sqrt(R(6))
    end
    # generic inclination: Stein–Warburton IBSO polynomial
    function ibsopoly(p)
        return (-4 + p)^2 * p^6 + a^8 * (-1 + x^2)^2 +
               2 * a^2 * p^5 * (-8 + 2 * p + 4 * x^2 - 3 * p * x^2) +
               2 * a^6 * p^2 * (2 - 5 * x^2 + 3 * x^4) +
               a^4 * p^3 * (-8 * (1 - 3 * x^2 + 2 * x^4) +
                            p * (6 - 14 * x^2 + 9 * x^4))
    end
    lo, hi = minmax(kerr_geo_ibso(a, one(R)), kerr_geo_ibso(a, zero(R)))
    return _bisect_root(ibsopoly, lo, hi)
end

# ------------------------------------------------------------
#  Separatrix p_s(a, e, x)
# ------------------------------------------------------------
"""
    kerr_geo_separatrix(a, e, x) -> p

Value of the semi-latus rectum on the separatrix between stable and
plunging bound geodesics.  Closed forms for Schwarzschild and the
extremal/parabolic limits; otherwise a bracketed solve of the
Stein–Warburton separatrix polynomials.
"""
function kerr_geo_separatrix(a, e, x)
    R = float(promote_type(typeof(a), typeof(e), typeof(x)))
    a = R(a); e = R(e); x = R(x)

    a < 0 && return kerr_geo_separatrix(-a, e, -x)
    iszero(a) && return 6 + 2 * e
    if a == one(R) && x == one(R)
        return 1 + e
    end
    if e == one(R)
        return 2 * kerr_geo_ibso(a, x)
    end

    # --- equatorial / polar polynomial sub-solvers ---
    SepEquat(p) = a^4 * (-3 - 2 * e + e^2)^2 + p^2 * (-6 - 2 * e + p)^2 -
                  2 * a^2 * (1 + e) * p * (14 + 2 * e^2 + 3 * p - e * p)
    SepPolar(p) = a^6 * (-1 + e)^2 * (1 + e)^4 + p^5 * (-6 - 2 * e + p) +
                  a^2 * p^3 * (-4 * (-1 + e) * (1 + e)^2 + (3 + e * (2 + 3 * e)) * p) -
                  a^4 * (1 + e)^2 * p *
                      (6 + 2 * e^3 + 2 * e * (-1 + p) - 3 * p - 3 * e^2 * (2 + p))

    pEquatPro() = _bisect_root(SepEquat, 1 + e, 6 + 2 * e)
    pEquatRet() = _bisect_root(SepEquat, 6 + 2 * e, 5 + e + 4 * sqrt(1 + e))
    pPolar()    = _bisect_root(SepPolar,
                               1 + sqrt(R(3)) + sqrt(3 + 2 * sqrt(R(3))), R(8))

    if x == one(R)
        return pEquatPro()
    elseif x == -one(R)
        return pEquatRet()
    elseif iszero(x)
        return pPolar()
    end

    # --- generic inclination: Stein–Warburton separatrix polynomial ---
    SepPoly(p) =
        -4 * (3 + e) * p^11 + p^12 +
        a^12 * (-1 + e)^4 * (1 + e)^8 * (-1 + x)^4 * (1 + x)^4 -
        4 * a^10 * (-3 + e) * (-1 + e)^3 * (1 + e)^7 * p * (-1 + x^2)^4 -
        4 * a^8 * (-1 + e) * (1 + e)^5 * p^3 * (-1 + x)^3 * (1 + x)^3 *
            (7 - 7 * x^2 - e^2 * (-13 + x^2) + e^3 * (-5 + x^2) + 7 * e * (-1 + x^2)) +
        8 * a^6 * (-1 + e) * (1 + e)^3 * p^5 * (-1 + x^2)^2 *
            (3 + e + 12 * x^2 + 4 * e * x^2 + e^3 * (-5 + 2 * x^2) + e^2 * (1 + 2 * x^2)) -
        8 * a^4 * (1 + e)^2 * p^7 * (-1 + x) * (1 + x) *
            (-3 + e + 15 * x^2 - 5 * e * x^2 + e^3 * (-5 + 3 * x^2) + e^2 * (-1 + 3 * x^2)) +
        4 * a^2 * p^9 * (-7 - 7 * e + e^3 * (-5 + 4 * x^2) + e^2 * (-13 + 12 * x^2)) +
        2 * a^8 * (-1 + e)^2 * (1 + e)^6 * p^2 * (-1 + x^2)^3 *
            (2 * (-3 + e)^2 * (-1 + x^2) +
             a^2 * (e^2 * (-3 + x^2) - 3 * (1 + x^2) + 2 * e * (1 + x^2))) -
        2 * p^10 * (-2 * (3 + e)^2 +
             a^2 * (-3 + 6 * x^2 + e^2 * (-3 + 2 * x^2) + e * (-2 + 4 * x^2))) +
        a^6 * (1 + e)^4 * p^4 * (-1 + x^2)^2 *
            (-16 * (-1 + e)^2 * (-3 - 2 * e + e^2) * (-1 + x^2) +
             a^2 * (15 + 6 * x^2 + 9 * x^4 + e^2 * (26 + 20 * x^2 - 2 * x^4) +
                    e^4 * (15 - 10 * x^2 + x^4) + 4 * e^3 * (-5 - 2 * x^2 + x^4) -
                    4 * e * (5 + 2 * x^2 + 3 * x^4))) -
        4 * a^4 * (1 + e)^2 * p^6 * (-1 + x) * (1 + x) *
            (-2 * (11 - 14 * e^2 + 3 * e^4) * (-1 + x^2) +
             a^2 * (5 - 5 * x^2 - 9 * x^4 + 4 * e^3 * x^2 * (-2 + x^2) +
                    e^4 * (5 - 5 * x^2 + x^4) + e^2 * (6 - 6 * x^2 + 4 * x^4))) +
        a^2 * p^8 *
            (-16 * (1 + e)^2 * (-3 + 2 * e + e^2) * (-1 + x^2) +
             a^2 * (15 - 36 * x^2 + 30 * x^4 + e^4 * (15 - 20 * x^2 + 6 * x^4) +
                    4 * e^3 * (5 - 12 * x^2 + 6 * x^4) + 4 * e * (5 - 12 * x^2 + 10 * x^4) +
                    e^2 * (26 - 72 * x^2 + 44 * x^4)))

    if x > 0
        return _bisect_root(SepPoly, pEquatPro(), pPolar())
    else
        return _bisect_root(SepPoly, pPolar(), R(12))
    end
end

# ------------------------------------------------------------
#  ISSO  (innermost stable spherical orbit)
# ------------------------------------------------------------
"""
    kerr_geo_isso(a, x) -> r

Innermost stable spherical orbit radius.  Equals the ISCO for
equatorial orbits (`|x| = 1`) and the `e = 0` separatrix otherwise.
"""
function kerr_geo_isso(a, x)
    R, a, x = _ax_R(a, x)
    if x^2 == one(R)
        return kerr_geo_isco(a, x)
    end
    return kerr_geo_separatrix(a, zero(R), x)
end
