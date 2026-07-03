# ============================================================
#  Kerr geodesic radial / polar roots and fundamental frequencies
#
#  For a bound Kerr geodesic specified by (a, p, e, x) with
#      x = cos(inclination),   r ∈ [p/(1+e), p/(1-e)],   M = 1,
#  this file provides
#
#    kerr_geo_radial_roots(a,p,e,x)  -> (r1,r2,r3,r4),  r1≥r2≥r3≥r4
#    kerr_geo_polar_roots(a,p,e,x)   -> (zp,zm)
#    kerr_geo_mino_frequencies(a,p,e,x)
#        -> (Upsilon_t, Upsilon_r, Upsilon_theta, Upsilon_phi)
#    kerr_geo_boyer_lindquist_frequencies(a,p,e,x)
#        -> (Omega_r, Omega_theta, Omega_phi)
#    kerr_geo_frequencies(a,p,e,x; time=:BoyerLindquist)
#
#  Closed forms from Fujita & Hikida, CQG 26 (2009) 135002
#  (arXiv:0906.1420), transcribed from the Wolfram KerrGeodesics paclet
#  (KerrGeodesics`OrbitalFrequencies`).  Everything is type-generic and
#  BigFloat-safe (working real type R recovered from the inputs).
#
#  NOTE: below the ISCO (e.g. retrograde unstable circular orbits) the
#  radial Mino frequency Υr and the BL frequency Ωr can be IMAGINARY.
#  We keep a Complex return value in that case rather than forcing real.
# ============================================================

@inline _sgn(x) = x < 0 ? -one(x) : one(x)

# ------------------------------------------------------------
#  Radial roots
# ------------------------------------------------------------
"""
    kerr_geo_radial_roots(a, p, e, x) -> (r1, r2, r3, r4)

Roots of the radial potential of a bound Kerr geodesic, ordered
`r1 ≥ r2 ≥ r3 ≥ r4`, with apastron `r1 = p/(1-e)` and periastron
`r2 = p/(1+e)`.  Uses the Fujita–Hikida sum/product relations.
"""
function kerr_geo_radial_roots(a, p, e, x)
    R, a, p, e, x = _orbit_R(a, p, e, x)
    En = kerr_geo_energy(a, p, e, x)
    Q  = kerr_geo_carter_constant(a, p, e, x)
    return _radial_roots(R, a, p, e, En, Q)
end

@inline function _radial_roots(R, a, p, e, En, Q)
    r1 = p / (1 - e)
    r2 = p / (1 + e)
    AplusB = 2 / (1 - En^2) - (r1 + r2)          # Fujita–Hikida Eq.(11)
    AB     = a^2 * Q / ((1 - En^2) * r1 * r2)
    r3 = (AplusB + sqrt(AplusB^2 - 4 * AB)) / 2
    r4 = AB / r3
    return (r1, r2, r3, r4)
end

# ------------------------------------------------------------
#  Polar roots
# ------------------------------------------------------------
"""
    kerr_geo_polar_roots(a, p, e, x) -> (zp, zm)

Polar roots `zm = √(1-x²)` and `zp = √(a²(1-E²) + L²/(1-zm²))`
(`zp = √Q` for polar orbits `x = 0`).  Convention of the KerrGeodesics
paclet: the polar equation is `(z²-zm²)(a²(1-E²)z² - zp²) = 0`.
"""
function kerr_geo_polar_roots(a, p, e, x)
    R, a, p, e, x = _orbit_R(a, p, e, x)
    En = kerr_geo_energy(a, p, e, x)
    L  = kerr_geo_angular_momentum(a, p, e, x)
    Q  = kerr_geo_carter_constant(a, p, e, x)
    return _polar_roots(R, a, x, En, L, Q)
end

@inline function _polar_roots(R, a, x, En, L, Q)
    zm = sqrt((1 - x) * (1 + x))           # √(1 - x²), exact-product form
    if iszero(x)
        zp = sqrt(Q)
    else
        # 1 - zm² = x² exactly; the old L²/(1 - zm²) recomputed 1 - (1 - x²)
        # and lost all precision for |x| ≲ 1e-8 (NaN at |x| ≲ 1e-9).
        zp = sqrt(a^2 * (1 - En^2) + (L / x)^2)
    end
    return (zp, zm)
end

# ------------------------------------------------------------
#  Mino-time fundamental frequencies
# ------------------------------------------------------------
"""
    kerr_geo_mino_frequencies(a, p, e, x)
        -> (Upsilon_t, Upsilon_r, Upsilon_theta, Upsilon_phi)

Mino-time fundamental frequencies (Fujita–Hikida closed forms).
`Upsilon_r` (and hence `Omega_r`) may be complex for unstable orbits
below the ISCO.
"""
function kerr_geo_mino_frequencies(a, p, e, x)
    R, a, p, e, x = _orbit_R(a, p, e, x)

    # ---------- Schwarzschild (a = 0) ----------
    if iszero(a)
        sg = _sgn(x)
        if iszero(e)
            Υr = sqrt((p - 6) * p / (p - 3))
            Υθ = p / sqrt(p - 3)
            Υφ = sg * p / sqrt(p - 3)
            Υt = sqrt(p^5 / (p - 3))
            return (Upsilon_t = Υt, Upsilon_r = Υr, Upsilon_theta = Υθ, Upsilon_phi = Υφ)
        else
            m  = 4 * e / (-6 + 2 * e + p)
            Kc = ellK(m); Ec = ellE(m)
            Π1 = ellPi((2 * e * (-4 + p)) / ((1 + e) * (-6 + 2 * e + p)), m)
            Π2 = ellPi((16 * e) / (12 + 8 * e - 4 * e^2 - 8 * p + p^2), m)
            Υr = sqrt(-(p * (-6 + 2 * e + p)) / (3 + e^2 - p)) * R(π) / (2 * Kc)
            Υθ = p / sqrt(-3 - e^2 + p)
            Υφ = sg * p / sqrt(-3 - e^2 + p)
            inner = -(((-4 + p) * p^2 * (-6 + 2 * e + p) * Ec) / (-1 + e^2)) +
                     (p^2 * (28 + 4 * e^2 - 12 * p + p^2) * Kc) / (-1 + e^2) -
                     (2 * (6 + 2 * e - p) * (3 + e^2 - p) * p^2 * Π1) / ((-1 + e) * (1 + e)^2) +
                     (4 * (-4 + p) * p * (2 * (1 + e) * Kc + (-6 - 2 * e + p) * Π1)) / (1 + e) +
                     2 * (-4 + p)^2 * ((-4 + p) * Kc - ((6 + 2 * e - p) * p * Π2) / (2 + 2 * e - p))
            Υt = (1 // 2) * sqrt((-4 * e^2 + (-2 + p)^2) / (p * (-3 - e^2 + p))) *
                 (8 + inner / ((-4 + p)^2 * Kc))
            return (Upsilon_t = Υt, Upsilon_r = Υr, Upsilon_theta = Υθ, Upsilon_phi = Υφ)
        end
    end

    # ---------- Kerr (a ≠ 0), generic Fujita–Hikida ----------
    # The extremal limit a = ±1 is NOT admitted: the horizons degenerate
    # (rout = rin = 1) and the Fujita–Hikida Υφr / Υtr closed forms contain
    # 1/√(1-a²) times a difference of Π-terms that vanishes in the limit —
    # a 0·∞ confluence, not a removable one-liner (the Π-terms must be
    # re-expanded around the double root).  The generic expressions used to
    # return Υt = Υφ = NaN silently; fail loudly instead.
    abs(a) == 1 && throw(DomainError(a,
        "kerr_geo_mino_frequencies: extremal spin |a| = 1 is not supported " *
        "(degenerate horizons make the Fujita–Hikida closed forms 0·∞); " *
        "use |a| < 1"))
    En = kerr_geo_energy(a, p, e, x)
    L  = kerr_geo_angular_momentum(a, p, e, x)
    Q  = kerr_geo_carter_constant(a, p, e, x)
    r1, r2, r3, r4 = _radial_roots(R, a, p, e, En, Q)
    zp, zm = _polar_roots(R, a, x, En, L, Q)

    rout = 1 + sqrt(1 - a^2)
    rin  = 1 - sqrt(1 - a^2)

    kr = (r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4)
    kθ = a^2 * (1 - En^2) * (zm / zp)^2

    Kr = ellK(kr); Er = ellE(kr)
    Kθ = ellK(kθ); Eθ = ellE(kθ)

    # ----- Υr (may be complex below ISCO) -----
    radr = (1 - En^2) * (r1 - r3) * (r2 - r4)
    Υr = radr < 0 ? R(π) * sqrt(complex(radr)) / (2 * Kr) :
                    R(π) * sqrt(radr) / (2 * Kr)

    # ----- Υθ -----
    Υθ = R(π) * zp / (2 * Kθ)

    # ----- helper ratios -----
    hr   = (r1 - r2) / (r1 - r3)
    hout = (r1 - r2) / (r1 - r3) * (r3 - rout) / (r2 - rout)
    hin  = (r1 - r2) / (r1 - r3) * (r3 - rin)  / (r2 - rin)

    Πhr   = ellPi(hr,   kr) / Kr
    Πhout = ellPi(hout, kr) / Kr
    Πhin  = ellPi(hin,  kr) / Kr

    # ----- Υφr -----
    Υφr = a / (2 * sqrt(1 - a^2)) *
          ((2 * En * rout - a * L) / (r3 - rout) *
               (1 - (r2 - r3) / (r2 - rout) * Πhout) -
           (2 * En * rin - a * L) / (r3 - rin) *
               (1 - (r2 - r3) / (r2 - rin) * Πhin))

    # ----- Υφθ -----
    if iszero(x)                              # polar orbit (L = 0)
        nin = r1 * r2 * (a^4 + r1^2 * r2^2 +
                         a^2 * ((-2 + r1) * r1 + (-2 + r2) * r2)) / 2
        din = a^4 * (2 + r1 + r2) +
              r1 * r2 * (r1^2 * (-2 + r2) + r1 * (-2 + r2) * r2 - 2 * r2^2) +
              a^2 * (r1^3 + r1^2 * r2 + r1 * (-4 + r2) * r2 + r2^3)
        karg = (a^2 * (a^4 + r1 * (r1 * (-2 + r2) - 2 * r2) * r2 +
                       a^2 * (r1^2 + r2^2))) /
               (r1 * r2 * (a^4 + r1^2 * r2^2 +
                           a^2 * ((-2 + r1) * r1 + (-2 + r2) * r2)))
        Υφθ = R(π) * sqrt(nin / din) / ellK(karg)
    else
        n = zm^2
        if n > max(kθ, one(R) / 2)
            # NEAR-POLAR CONDITIONING: ellPi(n, kθ) internally forms
            # 1 - n = 1 - (1 - x²), which cancels catastrophically as x → 0
            # (Π(1,·) = ∞ once zm² rounds to 1, |x| ≲ 1e-8 in Float64).
            # Use the complementary-characteristic identity
            # (Byrd & Friedman 413.01, valid for m < n < 1)
            #   Π(n, m) = K(m) + (π/2)·√(n/((1-n)(n-m))) - Π(m/n, m)
            # with 1 - n = x² EXACT and L/|x| smooth (L ∝ x near the pole).
            Υφθ = L * (one(R) - ellPi(kθ / n, kθ) / Kθ) +
                  (L / abs(x)) * (R(π) / 2) * sqrt(n / (n - kθ)) / Kθ
        else
            Υφθ = L * ellPi(n, kθ) / Kθ
        end
    end

    Υφ = Υφr + Υφθ

    # ----- Υtr -----
    Υtr = (a^2 + 4) * En +
          En * ((1 // 2) * (r3 * (r1 + r2 + r3) - r1 * r2 +
                            (r1 + r2 + r3 + r4) * (r2 - r3) * Πhr +
                            (r1 - r3) * (r2 - r4) * Er / Kr) +
                2 * (r3 + (r2 - r3) * Πhr) +
                (1 / sqrt(1 - a^2)) *
                    (((4 - a * L / En) * rout - 2 * a^2) / (r3 - rout) *
                         (1 - (r2 - r3) / (r2 - rout) * Πhout) -
                     ((4 - a * L / En) * rin - 2 * a^2) / (r3 - rin) *
                         (1 - (r2 - r3) / (r2 - rin) * Πhin)))

    # ----- Υtθ -----
    if x^2 == one(R)                          # equatorial
        Υtθ = -a^2 * En
    else
        Υtθ = (En * Q) / ((1 - En^2) * zm^2) * (1 - Eθ / Kθ) - a^2 * En
    end

    Υt = Υtr + Υtθ

    return (Upsilon_t = Υt, Upsilon_r = Υr, Upsilon_theta = Υθ, Upsilon_phi = Υφ)
end

# ------------------------------------------------------------
#  Boyer–Lindquist frequencies  Ω_i = Υ_i / Υ_t
# ------------------------------------------------------------
"""
    kerr_geo_boyer_lindquist_frequencies(a, p, e, x)
        -> (Omega_r, Omega_theta, Omega_phi)

Boyer–Lindquist (coordinate-time) fundamental frequencies
`Ω_i = Υ_i / Υ_t`.  `Omega_r` may be complex for unstable orbits.
"""
function kerr_geo_boyer_lindquist_frequencies(a, p, e, x)
    f = kerr_geo_mino_frequencies(a, p, e, x)
    Γ = f.Upsilon_t
    return (Omega_r = f.Upsilon_r / Γ,
            Omega_theta = f.Upsilon_theta / Γ,
            Omega_phi = f.Upsilon_phi / Γ)
end

# ------------------------------------------------------------
#  Generic dispatcher
# ------------------------------------------------------------
"""
    kerr_geo_frequencies(a, p, e, x; time = :BoyerLindquist)

Fundamental orbital frequencies.  `time = :Mino` returns the four
Mino-time frequencies `(Υt, Υr, Υθ, Υφ)`; `time = :BoyerLindquist`
(default) returns the three coordinate-time frequencies `(Ωr, Ωθ, Ωφ)`.
"""
function kerr_geo_frequencies(a, p, e, x; time::Symbol = :BoyerLindquist)
    if time === :Mino
        return kerr_geo_mino_frequencies(a, p, e, x)
    elseif time === :BoyerLindquist
        return kerr_geo_boyer_lindquist_frequencies(a, p, e, x)
    else
        throw(ArgumentError("time must be :Mino or :BoyerLindquist"))
    end
end
