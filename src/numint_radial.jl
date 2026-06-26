# ============================================================
#  NumericalIntegration radial backend  (Track B3)
#
#  Independent cross-check of the MST radial solutions: integrate the
#  homogeneous radial Teukolsky ODE directly with a self-contained, type-generic
#  adaptive Dormand–Prince RK (DP5), seeded from the MST asymptotics, and
#  reconstruct the same Transmission=1-normalized In/Up R(r).
#
#  Radial Teukolsky equation (M=1, this package's λ = p.λ convention, verified
#  by residual against the MST solution):
#     Δ R'' + (s+1)(2r-2) R' + V(r) R = 0,
#     V(r) = (K² - 2is(r-1)K)/Δ + 4isωr - λ,   K = (r²+a²)ω - ma,  Δ = r²-2r+a².
# ============================================================

# RHS of the first-order system u=(R, R'):  returns (R', R'').
@inline function _teuk_radial_rhs(r, R, Rp, s::Int, m::Int, a, ω, λ)
    Δ = r^2 - 2r + a^2
    K = (r^2 + a^2) * ω - m * a
    V = (K^2 - 2im * s * (r - 1) * K) / Δ + 4im * s * ω * r - λ
    Rpp = -((s + 1) * (2r - 2) * Rp + V * R) / Δ
    return (Rp, Rpp)
end

# Self-contained adaptive Dormand–Prince 5(4) integrator for the 2-state
# (R, R') from r0 to r1 (either direction). Type-generic / BigFloat-safe.
function _dp5_radial(s::Int, m::Int, a, ω, λ, R0, Rp0, r0, r1;
                     reltol, abstol, maxsteps::Int=200_000)
    RT = real(typeof(complex(R0)))
    # Butcher tableau (Rational → full precision in RT)
    c2,c3,c4,c5 = RT(1//5), RT(3//10), RT(4//5), RT(8//9)
    a21 = RT(1//5)
    a31,a32 = RT(3//40), RT(9//40)
    a41,a42,a43 = RT(44//45), RT(-56//15), RT(32//9)
    a51,a52,a53,a54 = RT(19372//6561), RT(-25360//2187), RT(64448//6561), RT(-212//729)
    a61,a62,a63,a64,a65 = RT(9017//3168), RT(-355//33), RT(46732//5247), RT(49//176), RT(-5103//18656)
    b1,b3,b4,b5,b6 = RT(35//384), RT(500//1113), RT(125//192), RT(-2187//6784), RT(11//84)
    e1,e3,e4,e5,e6,e7 = RT(71//57600), RT(-71//16695), RT(71//1920), RT(-17253//339200), RT(22//525), RT(-1//40)

    f(r, R, Rp) = _teuk_radial_rhs(r, R, Rp, s, m, a, ω, λ)
    R, Rp = complex(R0), complex(Rp0)
    rr = RT(r0); rend = RT(r1)
    dir = rend > rr ? one(RT) : -one(RT)
    h = dir * min(abs(rend - rr), max(abs(rend - rr) / 100, RT(1)/100))

    k1R, k1Rp = f(rr, R, Rp)          # FSAL: first stage
    reached = false
    for _ in 1:maxsteps
        if (dir > 0 ? rr ≥ rend : rr ≤ rend); reached = true; break; end
        # clamp final step to land exactly on rend
        if (dir > 0 && rr + h > rend) || (dir < 0 && rr + h < rend)
            h = rend - rr
        end
        # stages
        k2R, k2Rp = f(rr + c2*h, R + h*a21*k1R, Rp + h*a21*k1Rp)
        k3R, k3Rp = f(rr + c3*h, R + h*(a31*k1R + a32*k2R), Rp + h*(a31*k1Rp + a32*k2Rp))
        k4R, k4Rp = f(rr + c4*h, R + h*(a41*k1R + a42*k2R + a43*k3R),
                                 Rp + h*(a41*k1Rp + a42*k2Rp + a43*k3Rp))
        k5R, k5Rp = f(rr + c5*h, R + h*(a51*k1R + a52*k2R + a53*k3R + a54*k4R),
                                 Rp + h*(a51*k1Rp + a52*k2Rp + a53*k3Rp + a54*k4Rp))
        k6R, k6Rp = f(rr + h,    R + h*(a61*k1R + a62*k2R + a63*k3R + a64*k4R + a65*k5R),
                                 Rp + h*(a61*k1Rp + a62*k2Rp + a63*k3Rp + a64*k4Rp + a65*k5Rp))
        Rn  = R  + h*(b1*k1R  + b3*k3R  + b4*k4R  + b5*k5R  + b6*k6R)
        Rpn = Rp + h*(b1*k1Rp + b3*k3Rp + b4*k4Rp + b5*k5Rp + b6*k6Rp)
        k7R, k7Rp = f(rr + h, Rn, Rpn)    # FSAL
        # embedded error estimate
        errR  = h*(e1*k1R  + e3*k3R  + e4*k4R  + e5*k5R  + e6*k6R  + e7*k7R)
        errRp = h*(e1*k1Rp + e3*k3Rp + e4*k4Rp + e5*k5Rp + e6*k6Rp + e7*k7Rp)
        scR  = abstol + reltol * max(abs(R),  abs(Rn))
        scRp = abstol + reltol * max(abs(Rp), abs(Rpn))
        err  = sqrt((abs2(errR)/scR^2 + abs2(errRp)/scRp^2) / 2)

        if err ≤ 1
            rr += h
            R, Rp = Rn, Rpn
            k1R, k1Rp = k7R, k7Rp        # FSAL reuse
            fac = err == 0 ? RT(5) : RT(9)/10 * err^(-RT(1)/5)
            h *= clamp(fac, RT(1)/5, RT(5))
        else
            h *= clamp(RT(9)/10 * err^(-RT(1)/5), RT(1)/5, one(RT))
        end
    end
    reached || @warn "DP5: did not reach r=$(Float64(r1)) within $maxsteps steps " *
                     "(stopped at r=$(Float64(rr))); lower reltol expectations or raise maxsteps. " *
                     "DP5 is 5th-order — very deep precision needs a higher-order integrator."
    return R, Rp
end

# Default seed radii: In near the horizon, Up far out (∝ 1/|ω|).
_seed_in(p)  = p.rp + max(oftype(p.rp, 1)/2, oftype(p.rp, 1)/10)
_seed_up(p, ω) = max(oftype(real(ω), 50), oftype(real(ω), 30) / max(abs(ω), eps(real(typeof(real(ω))))))

"""
    NumericalIntegrationRadial(s, l, m, a, ω; nmax=80, reltol=..., abstol=..., r_seed_in, r_seed_up)

Independent ODE backend for the homogeneous radial Teukolsky solutions. Returns
a NamedTuple `(In, Up, ν, λ, amplitudes)` whose `In`/`Up` are callables
`R(r; deriv=0|1)`, seeded from the MST asymptotics and integrated with an
adaptive DP5 RK. Built to cross-check the MST `Rin`/`Rup` (same Transmission=1
normalization), not to replace them.
"""
function NumericalIntegrationRadial(s::Int, l::Int, m::Int, a, ω;
                                    nmax::Int=80,
                                    reltol::Real=-1, abstol::Real=-1,
                                    r_seed_in=nothing, r_seed_up=nothing)
    R = promote_type(typeof(float(real(a))), typeof(float(real(complex(ω)))))
    a_r = R(a); ω_c = Complex{R}(complex(ω))
    # DP5 is 5th-order: reltol≈1e-29 (BigFloat eps^¾) would need ~1e6 steps. Use a
    # feasible 1e-20 default at BigFloat (still far beyond Float64); the user can
    # lower it (and raise maxsteps) — a higher-order integrator is future work.
    rtol = reltol < 0 ? (R <: BigFloat ? R(1e-20) : R(1e-11)) : R(reltol)
    atol = abstol < 0 ? rtol : R(abstol)

    amp = compute_amplitudes(s, l, m, a_r, ω_c; nmax=nmax)
    ν   = amp.ν
    p   = MSTParams(s, l, m, a_r, ω_c)
    λ   = p.λ
    fn  = amp.fn
    ct  = amp.Ctrans
    rsi = r_seed_in  === nothing ? _seed_in(p)      : R(r_seed_in)
    rsu = r_seed_up  === nothing ? _seed_up(p, ω_c) : R(r_seed_up)

    function In(r; deriv::Int=0)
        deriv in (0,1) || throw(ArgumentError("deriv must be 0 or 1"))
        R0  = Rin(p, ν, fn, rsi; nmax=nmax)
        Rp0 = dRin(p, ν, fn, rsi; nmax=nmax)
        Rv, Rpv = _dp5_radial(s, m, a_r, ω_c, λ, R0, Rp0, rsi, R(r); reltol=rtol, abstol=atol)
        deriv == 0 ? Rv : Rpv
    end
    function Up(r; deriv::Int=0)
        deriv in (0,1) || throw(ArgumentError("deriv must be 0 or 1"))
        R0  = Rup(p, ν, fn, rsu; nmax=nmax, ctrans=ct)
        Rp0 = dRup(p, ν, fn, rsu; nmax=nmax, ctrans=ct)
        Rv, Rpv = _dp5_radial(s, m, a_r, ω_c, λ, R0, Rp0, rsu, R(r); reltol=rtol, abstol=atol)
        deriv == 0 ? Rv : Rpv
    end

    return (In=In, Up=Up, ν=ν, λ=λ, amplitudes=amp)
end
