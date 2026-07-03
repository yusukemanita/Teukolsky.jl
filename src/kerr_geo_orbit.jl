# ============================================================
#  Kerr geodesic orbit  (Mino-time parametrization)
#
#  KerrGeoOrbit(a, p, e, x; parametrization=:Mino, init_phases=(0,0,0,0))
#  builds a callable KerrGeoOrbitFunction; calling it at Mino time λ
#  returns the Boyer–Lindquist coordinates (t, r, θ, φ).
#
#  Closed-form trajectory from Fujita & Hikida, CQG 26 (2009) 135002
#  (arXiv:0906.1420), transcribed from the Wolfram KerrGeodesics paclet
#  (KerrGeodesics`KerrGeoOrbit`, generic Mino "Phases" construction):
#
#      r(λ) = r(q_r),         q_r = Υr λ + q_{r,0}
#      cosθ(λ) = z(q_θ),      q_θ = Υθ λ + q_{θ,0}
#      t(λ) = q_{t,0} + Υt λ + t_r(q_r) + t_θ(q_θ) − C_t
#      φ(λ) = q_{φ,0} + Υφ λ + φ_r(q_r) + φ_θ(q_θ) − C_φ
#
#  with r(q_r) and z(q_θ) given by Jacobi sn and the t_r, t_θ, φ_r, φ_θ
#  pieces by incomplete elliptic integrals.  Type-generic / BigFloat-safe.
# ============================================================

"""
    KerrGeoOrbitFunction

Callable object holding a Kerr bound-geodesic trajectory.  Calling it at
Mino time `λ` returns `(t, r, θ, φ)`.  Fields: `a, p, e, x`,
`constants = (E, L, Q)`, `frequencies` (Mino-time NamedTuple),
`radial_roots = (r1, r2, r3, r4)`, `polar_roots = (zp, zm)`,
`init_phases`, `parametrization`.
"""
struct KerrGeoOrbitFunction{R}
    a::R
    p::R
    e::R
    x::R
    constants::NTuple{3,R}
    frequencies::NamedTuple{(:Upsilon_t, :Upsilon_r, :Upsilon_theta, :Upsilon_phi)}
    radial_roots::NTuple{4,R}
    polar_roots::NTuple{2,R}
    init_phases::NTuple{4,R}
    parametrization::Symbol
    # precomputed series constants
    En::R
    L::R
    kr::R
    kθ::R
    Kr::R
    Kθ::R
    rp::R
    rm::R
    hr::R
    hp::R
    hm::R
    Ct::R
    Cφ::R
end

# ---- radial position r(q_r) ----
@inline function _rofqr(o::KerrGeoOrbitFunction{R}, qr) where {R}
    r1, r2, r3, r4 = o.radial_roots
    sn = jacobi_sn(o.Kr / R(π) * qr, o.kr)
    sn2 = sn * sn
    return (r3 * (r1 - r2) * sn2 - r2 * (r1 - r3)) /
           ((r1 - r2) * sn2 - (r1 - r3))
end

# ---- z(q_θ) = cosθ ----
# NOTE: every π here is materialized as R(π) at working precision; the old
# `(qz + π/2)` promoted Irrational/Int to a 53-bit Float64 constant and froze
# BigFloat trajectories at ~3e-15 relative accuracy.
@inline function _zofqz(o::KerrGeoOrbitFunction{R}, qz) where {R}
    _, zm = o.polar_roots
    return zm * jacobi_sn(o.Kθ * 2 / R(π) * (qz + R(π) / 2), o.kθ)
end

# ---- Δt_r(q_r) ----
function _trofqr(o::KerrGeoOrbitFunction{R}, qr) where {R}
    r1, r2, r3, r4 = o.radial_roots
    En = o.En; L = o.L; a = o.a
    kr = o.kr; rp = o.rp; rm = o.rm
    hr = o.hr; hp = o.hp; hm = o.hm
    ψr = jacobi_am(o.Kr / R(π) * qr, kr)
    qrπ = qr / R(π)
    Phr = ellPi(hr, kr) * qrπ - ellPi(hr, ψr, kr)
    Php = ellPi(hp, kr) * qrπ - ellPi(hp, ψr, kr)
    Phm = ellPi(hm, kr) * qrπ - ellPi(hm, ψr, kr)
    s = sin(ψr); c = cos(ψr)
    Eterm = ellE(kr) * qrπ - ellEinc(ψr, kr) +
            hr * (s * c * sqrt(1 - kr * s^2) / (1 - hr * s^2))
    return -En / sqrt((1 - En^2) * (r1 - r3) * (r2 - r4)) * (
        4 * (r2 - r3) * Phr
        - 4 * (r2 - r3) / (rp - rm) * (
            -(1 / ((r2 - rm) * (r3 - rm))) * (-2 * a^2 + rm * (4 - a * L / En)) * Phm
            + (1 / ((r2 - rp) * (r3 - rp))) * (-2 * a^2 + rp * (4 - a * L / En)) * Php)
        + (r2 - r3) * (r1 + r2 + r3 + r4) * Phr
        + (r1 - r3) * (r2 - r4) * Eterm)
end

# ---- Δφ_r(q_r) ----
function _phirofqr(o::KerrGeoOrbitFunction{R}, qr) where {R}
    r1, r2, r3, r4 = o.radial_roots
    En = o.En; L = o.L; a = o.a
    kr = o.kr; rp = o.rp; rm = o.rm
    hp = o.hp; hm = o.hm
    ψr = jacobi_am(o.Kr / R(π) * qr, kr)
    qrπ = qr / R(π)
    Php = ellPi(hp, kr) * qrπ - ellPi(hp, ψr, kr)
    Phm = ellPi(hm, kr) * qrπ - ellPi(hm, ψr, kr)
    return (2 * a * En * (
        -1 / ((r2 - rm) * (r3 - rm)) * (2 * rm - a * L / En) * (r2 - r3) * Phm
        + 1 / ((r2 - rp) * (r3 - rp)) * (2 * rp - a * L / En) * (r2 - r3) * Php)) /
           ((rp - rm) * sqrt((1 - En^2) * (r1 - r3) * (r2 - r4)))
end

# ---- Δt_θ(q_θ) ----
function _tθofqz(o::KerrGeoOrbitFunction{R}, qz) where {R}
    zp, zm = o.polar_roots
    En = o.En; kθ = o.kθ
    ψz = jacobi_am(o.Kθ * 2 / R(π) * (qz + R(π) / 2), kθ)
    return En * zp / (1 - En^2) *
           (ellE(kθ) * 2 * ((qz + R(π) / 2) / R(π)) - ellEinc(ψz, kθ))
end

# ---- Δφ_θ(q_θ) ----
function _phiθofqz(o::KerrGeoOrbitFunction{R}, qz) where {R}
    zp, zm = o.polar_roots
    L = o.L; kθ = o.kθ
    if iszero(L)
        # Polar orbit (x = 0, L = 0, zm = 1): the generic expression is a
        # 0 · ∞ indeterminate — ellPi(zm² = 1, kθ) = ∞ while L = 0 — but its
        # x → 0 limit is finite.  Reducing ψz = Nπ + ψ0 (|ψ0| ≤ π/2) via the
        # quasi-periodicity Π(n, ψ+Nπ, m) = Π(n, ψ, m) + 2N Π(n, m) gives
        #   Δφθ = -(L/zp) [(w - 2N) Π(zm², kθ) - Π(zm², ψ0, kθ)]
        # and, as x → 0,  (L/zp) Π(zm², kθ) → π/2  while
        # (L/zp) Π(zm², ψ0, kθ) → 0 for |ψ0| < π/2.  Hence exactly
        #   Δφθ = -(π/2) w + π N ,
        # a sawtooth whose π-jumps at the pole crossings are the geometric
        # azimuth flip of a trajectory passing over a pole.  (Verified
        # numerically: L·Π(zm²,kθ)/zp - π/2 ∝ x at 512-bit precision.)
        #
        # The crossings sit exactly at qz = jπ (where ψz = (2j+1)π/2), so we
        # index the branch N directly from qz — stable against rounding of
        # jacobi_am at the half-integer ties — and take the midpoint value at
        # an exact crossing, which is the pointwise x → 0⁺ limit there (the
        # sweep of π is centred on the crossing for every small x > 0).
        s = qz / R(π)
        j = floor(s)
        N = s == j ? j + one(R) / 2 : j + 1
        return -R(π) * s - R(π) / 2 + R(π) * N
    end
    ψz = jacobi_am(o.Kθ * 2 / R(π) * (qz + R(π) / 2), kθ)
    w = 2 * ((qz + R(π) / 2) / R(π))
    return -(L / zp) *
           (ellPi(zm^2, kθ) * w - ellPi(zm^2, ψz, kθ))
end

# ------------------------------------------------------------
#  Constructor
# ------------------------------------------------------------
"""
    KerrGeoOrbit(a, p, e, x; parametrization=:Mino, init_phases=(0,0,0,0))

Build a callable [`KerrGeoOrbitFunction`](@ref) for the bound Kerr
geodesic `(a, p, e, x)`.  `init_phases = (q_{t,0}, q_{r,0}, q_{θ,0},
q_{φ,0})`.  With the default zero phases the orbit starts at periastron
on the turning point `cosθ = zm`, with `t = φ = 0` at `λ = 0`.
"""
function KerrGeoOrbit(a, p, e, x; parametrization::Symbol = :Mino,
                      init_phases = (0, 0, 0, 0))
    parametrization === :Mino ||
        throw(ArgumentError("only parametrization = :Mino is supported"))
    R, a, p, e, x = _orbit_R(a, p, e, x)
    q = (R(init_phases[1]), R(init_phases[2]), R(init_phases[3]), R(init_phases[4]))

    E = kerr_geo_energy(a, p, e, x)
    L = kerr_geo_angular_momentum(a, p, e, x)
    Q = kerr_geo_carter_constant(a, p, e, x)
    freqs = kerr_geo_mino_frequencies(a, p, e, x)
    r1, r2, r3, r4 = _radial_roots(R, a, p, e, E, Q)
    zp, zm = _polar_roots(R, a, x, E, L, Q)

    rp = 1 + sqrt(1 - a^2)
    rm = 1 - sqrt(1 - a^2)
    kr = (r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4)
    kθ = a^2 * (1 - E^2) * (zm / zp)^2
    Kr = ellK(kr)
    Kθ = ellK(kθ)
    hr = (r1 - r2) / (r1 - r3)
    hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp))
    hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm))

    o = KerrGeoOrbitFunction{R}(a, p, e, x, (E, L, Q), freqs,
                                (r1, r2, r3, r4), (zp, zm), q, parametrization,
                                E, L, kr, kθ, Kr, Kθ, rp, rm, hr, hp, hm,
                                zero(R), zero(R))

    # normalization so that t = φ = 0 at λ = 0 when q_{t,0} = q_{φ,0} = 0
    Ct = _trofqr(o, q[2]) + _tθofqz(o, q[3])
    Cφ = _phirofqr(o, q[2]) + _phiθofqz(o, q[3])

    return KerrGeoOrbitFunction{R}(a, p, e, x, (E, L, Q), freqs,
                                   (r1, r2, r3, r4), (zp, zm), q, parametrization,
                                   E, L, kr, kθ, Kr, Kθ, rp, rm, hr, hp, hm,
                                   Ct, Cφ)
end

# ------------------------------------------------------------
#  Evaluation
# ------------------------------------------------------------
function (o::KerrGeoOrbitFunction{R})(λ) where {R}
    λ = R(λ)
    Υt = real(o.frequencies.Upsilon_t)
    Υr = real(o.frequencies.Upsilon_r)
    Υθ = real(o.frequencies.Upsilon_theta)
    Υφ = real(o.frequencies.Upsilon_phi)
    qt0, qr0, qz0, qφ0 = o.init_phases
    qr = Υr * λ + qr0
    qz = Υθ * λ + qz0
    t = qt0 + Υt * λ + _trofqr(o, qr) + _tθofqz(o, qz) - o.Ct
    r = _rofqr(o, qr)
    θ = acos(_zofqz(o, qz))
    φ = qφ0 + Υφ * λ + _phirofqr(o, qr) + _phiθofqz(o, qz) - o.Cφ
    return (t, r, θ, φ)
end

function Base.show(io::IO, o::KerrGeoOrbitFunction)
    print(io, "KerrGeoOrbitFunction(a=", o.a, ", p=", o.p,
          ", e=", o.e, ", x=", o.x, ")")
end
