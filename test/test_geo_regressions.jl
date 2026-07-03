using Test
using Teukolsky

# ============================================================
#  Regression tests for the geodesic/elliptic bug-fix batch:
#
#  G2   — Carlson duplication had no iteration cap: ellK(1) looped forever in
#         BigFloat (and returned NaN via underflow in Float64); jacobi_am
#         silently returned π/2 after maxiter for m = 1.
#  G3   — polar orbits (x = 0, L = 0) returned φ = NaN (0·∞ in the φθ piece).
#  G7a  — `(qz + π/2)` Float64-π contamination froze BigFloat trajectories
#         at ~3e-15.
#  G9   — near-polar catastrophic cancellation: 1 - zm² recomputed from
#         zm² = 1 - x² (zp rel-err 0.29 at x = 1e-8, NaN at x = 1e-9).
#  G14b — a = 1 admitted by the domain check but Υt, Υφ silently NaN.
#
#  All references are computed IN-TEST (self-arbitrating): exact limits,
#  independent quadrature, exact-algebra residual certification at 512-bit.
# ============================================================

@testset "G2: Carlson/elliptic m=1 and domain handling" begin
    # --- exact limits at m = 1 (previously: BigFloat infinite loop /
    #     Float64 NaN-by-underflow) ---
    @test Teukolsky.ellK(1.0) == Inf
    @test Teukolsky.ellE(1.0) == 1.0
    @test Teukolsky.ellPi(0.3, 1.0) == Inf
    setprecision(BigFloat, 256) do
        t0 = time()
        v = Teukolsky.ellK(big"1.0")
        @test v isa BigFloat && isinf(v) && v > 0
        @test time() - t0 < 5.0            # used to hang forever
        @test Teukolsky.ellE(big"1.0") == 1
        # near-singular argument still converges to the asymptotic law
        # K(m) = ln(16/(1-m))/2 + O((1-m)·ln(1-m))
        m = 1 - big"1e-60"
        @test isapprox(Teukolsky.ellK(m), log(16 / (1 - m)) / 2; rtol = big"1e-55")
    end

    # --- m > 1 is complex on the real line: loud DomainError, not NaN ---
    @test_throws DomainError Teukolsky.ellK(1.5)
    @test_throws DomainError Teukolsky.ellE(1.5)
    @test_throws DomainError Teukolsky.ellPi(0.3, 1.5)
    @test_throws DomainError Teukolsky.ellK(big"2.0")   # MPFR sqrt(neg) = silent NaN before

    # --- divergent Carlson inputs fail loudly (used to loop / underflow) ---
    @test_throws DomainError rf(0.0, 0.0, 1.0)
    @test_throws DomainError rd(0.0, 0.0, 1.0)
    @test_throws DomainError rd(1.0, 1.0, 0.0)
    @test_throws DomainError rc(1.0, 0.0)
    @test_throws DomainError rj(0.0, 0.0, 1.0, 1.0)
    @test_throws DomainError rj(1.0, 1.0, 1.0, 0.0)
    @test_throws DomainError rf(-1.0, 1.0, 1.0)         # negative argument

    # --- extreme-but-convergent inputs still converge under the cap ---
    @test isapprox(rf(1e-300, 2.0, 3.0), rf(1e-30, 2.0, 3.0); rtol = 1e-10)
    @test isfinite(rd(1e-280, 2.0, 3.0))

    # --- jacobi_am m = 1: silently returned π/2 (true gd(2) ≈ 1.3018) ---
    @test_throws DomainError jacobi_am(2.0, 1.0)
    @test_throws DomainError jacobi_sn(2.0, 1.0)
    @test_throws DomainError jacobi_am(2.0, 1.2)
    @test_throws DomainError jacobi_am(2.0, -0.1)
    # domain interior unaffected
    @test isapprox(jacobi_am(2.0, 0.999999), 2 * atan(tanh(1.0)); rtol = 1e-2)
end

@testset "G3: polar orbit (x = 0) φ trajectory" begin
    # --- repro: used to return φ = NaN ---
    orb = KerrGeoOrbit(0.7, 9.0, 0.0, 0.0)
    t, r, θ, φ = orb(5.0)
    @test isfinite(t) && isfinite(r) && isfinite(θ) && isfinite(φ)

    # --- arbiter 1 (a → 0 limit): Schwarzschild polar orbit is a great
    #     circle: φ is the staircase π·(⌊Υθλ/π⌋ + 1/2), constant between
    #     pole crossings with π jumps at them ---
    orb0 = KerrGeoOrbit(0.0, 9.0, 0.0, 0.0)
    Υθ0 = orb0.frequencies.Upsilon_theta
    for λ in (0.1, 0.5, 1.0, 2.0, 3.7, 5.0)
        expect = π * (floor(Υθ0 * λ / π) + 1 / 2)
        @test isapprox(orb0(λ)[4], expect; atol = 1e-12)
    end

    # --- arbiter 2 (exact ODE): for L = 0, dφ/dλ = aE((r²+a²)/Δ - 1)
    #     away from the poles; a spherical polar orbit has r = p so φ is
    #     rate·λ plus the π-jump staircase, and Υφ = rate + Υθ exactly ---
    a, p = 0.7, 9.0
    E = kerr_geo_energy(a, p, 0.0, 0.0)
    rate = a * E * ((p^2 + a^2) / (p^2 - 2p + a^2) - 1)
    Υθ = orb.frequencies.Upsilon_theta
    @test isapprox(orb.frequencies.Upsilon_phi, rate + Υθ; rtol = 1e-13)
    for λ in (0.3, 1.0, 2.0, 5.0, 8.3)
        expect = rate * λ + π * (floor(Υθ * λ / π) + 1 / 2)
        @test isapprox(orb(λ)[4], expect; rtol = 1e-12)
    end

    # --- arbiter 3 (independent quadrature): eccentric Kerr polar orbit,
    #     Simpson integration of dφ/dλ = aE((r(λ)²+a²)/Δ - 1) using the
    #     independently-validated r(λ), plus the π-jump staircase ---
    ae, pe, ee = 0.9, 10.0, 0.3
    orbe = KerrGeoOrbit(ae, pe, ee, 0.0)
    Ee = orbe.constants[1]
    Υθe = orbe.frequencies.Upsilon_theta
    integrand(λ) = (rr = orbe(λ)[2];
                    ae * Ee * ((rr^2 + ae^2) / (rr^2 - 2rr + ae^2) - 1))
    function simpson(f, lo, hi, n)
        n += isodd(n)
        h = (hi - lo) / n
        s = f(lo) + f(hi)
        for i in 1:n-1
            s += f(lo + i * h) * (isodd(i) ? 4 : 2)
        end
        return s * h / 3
    end
    for λ in (0.4, 1.0, 2.5, 6.0)
        φnum = simpson(integrand, 0.0, λ, 4000) +
               π * (floor(Υθe * λ / π) + 1 / 2)
        @test isapprox(orbe(λ)[4], φnum; atol = 1e-10)
    end

    # --- arbiter 4 (continuity): near-polar orbits converge to the x = 0
    #     trajectory linearly in x ---
    for (x, tol) in ((1e-3, 4e-3), (1e-5, 4e-5))
        orbx = KerrGeoOrbit(0.7, 9.0, 0.0, x)
        for λ in (0.3, 1.0, 2.0, 5.0)
            @test abs(orbx(λ)[4] - orb(λ)[4]) < tol
        end
    end
end

@testset "G7a: working-precision π in the trajectory pieces" begin
    # Independent rebuild of the θ-pieces with an explicitly-materialized
    # big(pi) at the SAME precision (a shared Float64-π/2 literal would
    # show up here as a ~1e-16 floor; the old code sat at ~3e-15).
    setprecision(BigFloat, 256) do
        orb = KerrGeoOrbit(big"0.8", big"11.0", big"0.3", big"0.6")
        λ = big"10.0"
        Pi = big(pi)
        qz = orb.frequencies.Upsilon_theta * λ            # qz0 = 0
        zp, zm = orb.polar_roots
        En = orb.En; kθ = orb.kθ; Kθ = orb.Kθ; L = orb.L
        u = Kθ * 2 / Pi * (qz + Pi / 2)
        ψz = jacobi_am(u, kθ)
        w = 2 * ((qz + Pi / 2) / Pi)
        tθ = En * zp / (1 - En^2) *
             (Teukolsky.ellE(kθ) * w - Teukolsky.ellEinc(ψz, kθ))
        φθ = -(L / zp) * (Teukolsky.ellPi(zm^2, kθ) * w -
                          Teukolsky.ellPi(zm^2, ψz, kθ))
        z = zm * jacobi_sn(u, kθ)
        @test abs(Teukolsky._tθofqz(orb, qz) - tθ) <= abs(tθ) * big"1e-70"
        @test abs(Teukolsky._phiθofqz(orb, qz) - φθ) <= abs(φθ) * big"1e-70"
        @test abs(Teukolsky._zofqz(orb, qz) - z) <= abs(z) * big"1e-70"
    end
    # Cross-precision agreement of the full trajectory (used to floor at
    # ~3e-15 regardless of precision).
    ref = setprecision(() -> KerrGeoOrbit(big"0.8", big"11.0", big"0.3",
                                          big"0.6")(big"10.0"), BigFloat, 320)
    got = setprecision(() -> KerrGeoOrbit(big"0.8", big"11.0", big"0.3",
                                          big"0.6")(big"10.0"), BigFloat, 256)
    for (g, rf_) in zip(got, ref)
        @test abs(g - rf_) <= abs(rf_) * big"1e-70"
    end
end

@testset "G9: near-polar conditioning of E, L, Q, zp, frequencies" begin
    # Reference: solve at 512-bit and CERTIFY it in-test with the exact
    # algebra (radial potential R(r1) = R(r2) = 0 and polar potential
    # Θ(zm) = 0) — algorithm-independent, no pinned constants.
    a, p, e = 0.9, 10.0, 0.3
    setprecision(BigFloat, 512) do
        for x in (1e-6, 1e-8, 1e-10)
            ab, pb, eb, xb = BigFloat(a), BigFloat(p), BigFloat(e), BigFloat(x)
            Eb = kerr_geo_energy(ab, pb, eb, xb)
            Lb = kerr_geo_angular_momentum(ab, pb, eb, xb)
            Qb = kerr_geo_carter_constant(ab, pb, eb, xb)
            Δb(r) = r^2 - 2r + ab^2
            Rr(r) = (Eb * (r^2 + ab^2) - ab * Lb)^2 -
                    Δb(r) * (r^2 + (Lb - ab * Eb)^2 + Qb)
            zmb = sqrt((1 - xb) * (1 + xb))
            Θz = Qb - zmb^2 * (Qb + ab^2 * (1 - Eb^2) + Lb^2) +
                 zmb^4 * ab^2 * (1 - Eb^2)
            # certification of the reference itself
            @test abs(Rr(pb / (1 - eb))) < big"1e-120"
            @test abs(Rr(pb / (1 + eb))) < big"1e-120"
            @test abs(Θz) < big"1e-120"

            # Float64 path vs certified reference (was rel-err 0.29 at
            # x = 1e-8 and NaN at x = 1e-9 for zp)
            E, L, Q = kerr_geo_constants_of_motion(a, p, e, x)
            zp, zm = kerr_geo_polar_roots(a, p, e, x)
            zpb = sqrt(ab^2 * (1 - Eb^2) + (Lb / xb)^2)
            @test abs(E - Float64(Eb)) <= 1e-13 * Float64(Eb)
            @test abs(L - Float64(Lb)) <= 1e-13 * abs(Float64(Lb))
            @test abs(Q - Float64(Qb)) <= 1e-13 * Float64(Qb)
            @test abs(zp - Float64(zpb)) <= 1e-13 * Float64(zpb)
            @test abs(zm - Float64(zmb)) <= 1e-13

            # frequencies finite and accurate at tiny x (Υφ used to lose
            # accuracy through ellPi(zm² → 1, kθ) and go Inf below x ≈ 1e-8)
            f = kerr_geo_mino_frequencies(a, p, e, x)
            fb = kerr_geo_mino_frequencies(ab, pb, eb, xb)
            for k in (:Upsilon_t, :Upsilon_r, :Upsilon_theta, :Upsilon_phi)
                @test isfinite(getproperty(f, k))
                @test abs(getproperty(f, k) - Float64(getproperty(fb, k))) <=
                      1e-12 * abs(Float64(getproperty(fb, k)))
            end
        end
    end

    # explicit former-NaN repro
    zp9, _ = kerr_geo_polar_roots(0.9, 10.0, 0.3, 1e-9)
    @test isfinite(zp9)

    # continuity into the exact polar branch x = 0
    E0, L0, Q0 = kerr_geo_constants_of_motion(a, p, e, 0.0)
    E1, L1, Q1 = kerr_geo_constants_of_motion(a, p, e, 1e-10)
    f0 = kerr_geo_mino_frequencies(a, p, e, 0.0)
    f1 = kerr_geo_mino_frequencies(a, p, e, 1e-10)
    @test abs(E1 - E0) <= 1e-9 * E0
    @test abs(L1 - L0) <= 1e-8
    @test abs(Q1 - Q0) <= 1e-9 * Q0
    @test abs(f1.Upsilon_phi - f0.Upsilon_phi) <= 1e-8 * abs(f0.Upsilon_phi)
    @test abs(f1.Upsilon_t - f0.Upsilon_t) <= 1e-9 * f0.Upsilon_t
    # near-odd symmetry of L in x survives the reformulation
    # (L(x) = c₁x + O(x²) for Kerr, so the sum is O(x²) ≈ 1e-16, not 0)
    @test abs(kerr_geo_angular_momentum(a, p, e, 1e-8) +
              kerr_geo_angular_momentum(a, p, e, -1e-8)) < 1e-14

    # moderate inclination unchanged at machine precision vs 512-bit
    setprecision(BigFloat, 512) do
        for x in (0.6, -0.4)
            Eb = kerr_geo_energy(big(a), big(p), big(e), big(x))
            Lb = kerr_geo_angular_momentum(big(a), big(p), big(e), big(x))
            @test abs(kerr_geo_energy(a, p, e, x) - Float64(Eb)) <= 1e-14
            @test abs(kerr_geo_angular_momentum(a, p, e, x) - Float64(Lb)) <=
                  1e-13 * abs(Float64(Lb))
        end
    end
end

@testset "G14b: extremal spin a = ±1 frequencies fail loudly" begin
    # used to return Υt = Υφ = NaN silently
    @test_throws DomainError kerr_geo_mino_frequencies(1.0, 8.0, 0.3, 0.7)
    @test_throws DomainError kerr_geo_boyer_lindquist_frequencies(1.0, 8.0, 0.3, 0.7)
    @test_throws DomainError kerr_geo_frequencies(1.0, 8.0, 0.3, 0.7)
    @test_throws DomainError KerrGeoOrbit(1.0, 8.0, 0.3, 0.7)
    @test_throws DomainError kerr_geo_mino_frequencies(-1.0, 8.0, 0.3, 0.7)
    # near-extremal spins remain finite
    f = kerr_geo_mino_frequencies(0.999, 8.0, 0.3, 0.7)
    @test all(isfinite, (f.Upsilon_t, f.Upsilon_theta, f.Upsilon_phi)) &&
          isfinite(f.Upsilon_r)
end
