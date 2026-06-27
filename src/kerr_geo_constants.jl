# ============================================================
#  Kerr geodesic constants of motion  (E, L, Q)
#
#  Energy E, axial angular momentum L, and Carter constant Q for a
#  bound Kerr geodesic specified by (a, p, e, x), with
#      x = cos(inclination),   r ∈ [p/(1+e), p/(1-e)],   M = 1.
#
#  Formulas transcribed from the Wolfram KerrGeodesics paclet
#  (KerrGeodesics`ConstantsOfMotion`, ConstantsOfMotion.m):
#    - Schwarzschild  (a = 0)             circular & eccentric
#    - Kerr equatorial (x² = 1)           circular & eccentric
#      (Glampedakis & Kennefick, PRD 66 (2002) 044002)
#    - Kerr polar      (x = 0)            spherical & eccentric
#    - Kerr spherical  (e = 0, generic x) Stoghianidis & Tsoubelis (1987)
#    - Kerr generic    (e > 0, generic x) Schmidt determinant method
#
#  Everything is type-generic and BigFloat-safe: the working real type
#  R = float(promote_type(...)) is recovered from the inputs and every
#  literal flows through R, so the routines deliver full precision at
#  Float64 and BigFloat alike.
# ============================================================

# Validate and promote the four orbital parameters to a common working float type.
# All geodesic entry points (constants, frequencies, orbit) route through this,
# so the bound-orbit domain is enforced in one place.
@inline function _orbit_R(a, p, e, x)
    (0 ≤ e < 1) || throw(ArgumentError(
        "bound Kerr geodesic requires 0 ≤ e < 1 (got e=$e); unbound/parabolic orbits are not supported"))
    abs(x) ≤ 1  || throw(ArgumentError("inclination cosine requires |x| ≤ 1 (got x=$x)"))
    abs(a) ≤ 1  || throw(ArgumentError("spin requires |a| ≤ 1 (got a=$a)"))
    p > 0       || throw(ArgumentError("semi-latus rectum must be positive (got p=$p)"))
    R = float(promote_type(typeof(a), typeof(p), typeof(e), typeof(x)))
    return R, R(a), R(p), R(e), R(x)
end

# ------------------------------------------------------------
#  Energy
# ------------------------------------------------------------
"""
    kerr_geo_energy(a, p, e, x) -> E

Orbital energy (per unit rest mass) of a bound Kerr geodesic with
semi-latus rectum `p`, eccentricity `e`, and inclination cosine
`x = cos(ι)`. Geometric units `M = 1`.
"""
function kerr_geo_energy(a, p, e, x)
    R, a, p, e, x = _orbit_R(a, p, e, x)

    # --- Schwarzschild (a = 0) ---
    if iszero(a)
        if iszero(e)
            return (p - 2) / sqrt((p - 3) * p)
        else
            return sqrt((-4 * e^2 + (p - 2)^2) / (p * (p - 3 - e^2)))
        end
    end

    # --- negative spin: (a, x) → (-a, -x) ---
    if a < 0
        return kerr_geo_energy(-a, p, e, -x)
    end

    # --- equatorial (x² = 1) ---
    if x^2 == one(R)
        if iszero(e)
            return ((p - 2) * sqrt(p) + a / x) /
                   sqrt(2 * (a / x) * p * sqrt(p) + (p - 3) * p^2)
        else
            rad = sqrt((a^6 * (e^2 - 1)^2 +
                        a^2 * (-4 * e^2 + (p - 2)^2) * p^2 +
                        2 * a^4 * p * (p - 2 + e^2 * (p + 2))) / (p^3 * x^2))
            num = a^2 * (1 + 3 * e^2 + p) +
                  p * (p - 3 - e^2 - 2 * x * rad)
            den = -4 * a^2 * (e^2 - 1)^2 + (3 + e^2 - p)^2 * p
            return sqrt(1 - (1 - e^2) * (1 + (e^2 - 1) * num / den) / p)
        end
    end

    # --- polar (x = 0) ---
    if iszero(x)
        if iszero(e)
            return sqrt((p * (a^2 - 2 * p + p^2)^2) /
                        ((a^2 + p^2) * (a^2 + a^2 * p - 3 * p^2 + p^3)))
        else
            return sqrt(-(p * (a^4 * (e^2 - 1)^2 +
                               (-4 * e^2 + (p - 2)^2) * p^2 +
                               2 * a^2 * p * (p - 2 + e^2 * (p + 2)))) /
                        (a^4 * (e^2 - 1)^2 * (e^2 - 1 - p) +
                         (3 + e^2 - p) * p^4 -
                         2 * a^2 * p^2 * (p - 1 - e^4 + e^2 * (p + 2))))
        end
    end

    # --- spherical (e = 0, generic x): Stoghianidis & Tsoubelis ---
    if iszero(e)
        xm = x^2 - 1                       # (-1 + x²)
        num = (p - 3) * (p - 2)^2 * p^5 -
              2 * a^5 * x * xm * sqrt(p^3 + a^2 * p * xm) +
              a^4 * p^2 * xm * (4 - 5 * p * xm + 3 * p^2 * xm) -
              a^6 * xm^2 * (x^2 + p^2 * xm - p * (1 + 2 * x^2)) +
              a^2 * p^3 * (4 - 4 * x^2 + p * (12 - 7 * x^2) -
                           3 * p^3 * xm + p^2 * (-13 + 10 * x^2)) +
              a * (-2 * p^4 * sqrt(p) * x * sqrt(p^2 + a^2 * xm) +
                   4 * p^3 * x * sqrt(p^3 + a^2 * p * xm)) +
              2 * a^3 * (2 * p * x * xm * sqrt(p^3 + a^2 * p * xm) -
                         x^3 * sqrt(p^7 + a^2 * p^5 * xm))
        den = (p^2 - a^2 * xm) *
              ((p - 3)^2 * p^4 -
               2 * a^2 * p^2 * (3 + 2 * p - 3 * x^2 + p^2 * xm) +
               a^4 * xm * (-1 + x^2 + p^2 * xm - 2 * p * (1 + x^2)))
        return sqrt(num / den)
    end

    # --- generic (e > 0, generic x): Schmidt determinant method ---
    r1 = p / (1 - e)
    r2 = p / (1 + e)
    zm2 = 1 - x^2
    Δ(r) = r^2 - 2 * r + a^2
    f(r) = r^4 + a^2 * (r * (r + 2) + zm2 * Δ(r))
    g(r) = 2 * a * r
    h(r) = r * (r - 2) + zm2 / (1 - zm2) * Δ(r)
    d(r) = (r^2 + a^2 * zm2) * Δ(r)

    κ = d(r1) * h(r2) - h(r1) * d(r2)
    ε = d(r1) * g(r2) - g(r1) * d(r2)
    ρ = f(r1) * h(r2) - h(r1) * f(r2)
    η = f(r1) * g(r2) - g(r1) * f(r2)
    σ = g(r1) * h(r2) - h(r1) * g(r2)

    return sqrt((κ * ρ + 2 * ε * σ -
                 2 * x * sqrt(σ * (σ * ε^2 + ρ * ε * κ - η * κ^2) / x^2)) /
                (ρ^2 + 4 * η * σ))
end

# ------------------------------------------------------------
#  Angular momentum
# ------------------------------------------------------------
"""
    kerr_geo_angular_momentum(a, p, e, x) -> L

Axial (z-component) angular momentum of a bound Kerr geodesic.
"""
function kerr_geo_angular_momentum(a, p, e, x)
    R, a, p, e, x = _orbit_R(a, p, e, x)

    # --- Schwarzschild (a = 0) ---
    if iszero(a)
        if iszero(e)
            return p * x / sqrt(p - 3)
        else
            return p * x / sqrt(p - 3 - e^2)
        end
    end

    # --- negative spin: L(a) = -L(-a, -x) ---
    if a < 0
        return -kerr_geo_angular_momentum(-a, p, e, -x)
    end

    # --- equatorial (x² = 1) ---
    if x^2 == one(R)
        if iszero(e)
            return ((a^2 + p^2) * x - 2 * a * sqrt(p)) /
                   (p^(R(3) / 4) * sqrt(x^2 * (p - 3) * sqrt(p) + 2 * a * x))
        else
            En = kerr_geo_energy(a, p, e, x)
            rad = sqrt((a^6 * (e^2 - 1)^2 +
                        a^2 * (-4 * e^2 + (p - 2)^2) * p^2 +
                        2 * a^4 * p * (p - 2 + e^2 * (p + 2))) / (x^2 * p^3))
            num = a^2 * (1 + 3 * e^2 + p) +
                  p * (p - 3 - e^2 - 2 * x * rad)
            den = (-4 * a^2 * (e^2 - 1)^2 + (3 + e^2 - p)^2 * p) * x^2
            return p * x * sqrt(num / den) + a * En
        end
    end

    # --- polar (x = 0): L vanishes ---
    if iszero(x)
        return zero(R)
    end

    # --- spherical-generic and generic: use the r1 turning point ---
    En = kerr_geo_energy(a, p, e, x)
    r1 = p / (1 - e)            # = p for the spherical (e = 0) case
    zm2 = 1 - x^2
    Δ = r1^2 - 2 * r1 + a^2
    f = r1^4 + a^2 * (r1 * (r1 + 2) + zm2 * Δ)
    g = 2 * a * r1
    h = r1 * (r1 - 2) + zm2 / (1 - zm2) * Δ
    d = (r1^2 + a^2 * zm2) * Δ
    return (-En * g + x * sqrt((-d * h + En^2 * (g^2 + f * h)) / x^2)) / h
end

# ------------------------------------------------------------
#  Carter constant
# ------------------------------------------------------------
"""
    kerr_geo_carter_constant(a, p, e, x) -> Q

Carter constant of a bound Kerr geodesic (Schmidt / KerrGeodesics
convention: `Q = 0` for equatorial orbits).
"""
function kerr_geo_carter_constant(a, p, e, x)
    R, a, p, e, x = _orbit_R(a, p, e, x)

    # --- Schwarzschild (a = 0) ---
    if iszero(a)
        if iszero(e)
            return -(p^2 * (x^2 - 1)) / (p - 3)
        else
            return (p^2 * (x^2 - 1)) / (3 + e^2 - p)
        end
    end

    # --- equatorial (x² = 1): Carter constant vanishes ---
    if x^2 == one(R)
        return zero(R)
    end

    # --- polar (x = 0) ---
    if iszero(x)
        if iszero(e)
            return (p^2 * (a^4 + 2 * a^2 * (p - 2) * p + p^4)) /
                   ((a^2 + p^2) * ((p - 3) * p^2 + a^2 * (1 + p)))
        else
            return -(p^2 * (a^4 * (e^2 - 1)^2 + p^4 +
                            2 * a^2 * p * (p - 2 + e^2 * (p + 2)))) /
                    (a^4 * (e^2 - 1)^2 * (e^2 - 1 - p) +
                     (3 + e^2 - p) * p^4 -
                     2 * a^2 * p^2 * (p - 1 - e^4 + e^2 * (p + 2)))
        end
    end

    # --- spherical-generic and generic ---
    En = kerr_geo_energy(a, p, e, x)
    L = kerr_geo_angular_momentum(a, p, e, x)
    zm2 = 1 - x^2
    return zm2 * (a^2 * (1 - En^2) + L^2 / (1 - zm2))
end

# ------------------------------------------------------------
#  Bundle
# ------------------------------------------------------------
"""
    kerr_geo_constants_of_motion(a, p, e, x) -> (E, L, Q)

Return the triple of constants of motion `(E, L, Q)` for a bound Kerr
geodesic specified by `(a, p, e, x)`.
"""
function kerr_geo_constants_of_motion(a, p, e, x)
    E = kerr_geo_energy(a, p, e, x)
    L = kerr_geo_angular_momentum(a, p, e, x)
    Q = kerr_geo_carter_constant(a, p, e, x)
    return (E, L, Q)
end
