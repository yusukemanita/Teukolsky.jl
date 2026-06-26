# ============================================================
#  Callable radial-solution object  (Track B1)
#
#  Mirrors Mathematica's TeukolskyRadial[s,l,m,a,ω]["In"/"Up"]:
#  a TeukolskyRadialFunction is callable at r (value or derivative) and carries
#  all mode data (s,l,m,a,ω,ν,λ,amplitudes) under string keys.
# ============================================================

"""
    TeukolskyRadialFunction

A homogeneous radial Teukolsky solution for one boundary condition (`:In` or
`:Up`). Callable:

    R = TeukolskyRadial(s,l,m,a,ω).In
    R(r)              # value
    R(r; deriv=1)     # dR/dr

Key access mirrors the Mathematica object: `R["ν"]`, `R["λ"]`, `R["Amplitudes"]`,
`R["BoundaryCondition"]`, `R["s"|"l"|"m"|"a"|"omega"]`.
"""
struct TeukolskyRadialFunction{R<:AbstractFloat}
    boundary::Symbol            # :In or :Up
    s::Int
    l::Int
    m::Int
    a::R
    ω::Complex{R}
    ν::Complex{R}
    λ::Complex{R}
    p::MSTParams{R}
    fn::Dict{Int,Complex{R}}
    amplitudes::NamedTuple
    nmax::Int
    prec::Int                   # BigFloat precision (bits); 0 for Float64
end

# Run a thunk at the stored BigFloat precision (no-op for Float64).
@inline _atprec(f, trf::TeukolskyRadialFunction) =
    trf.prec == 0 ? f() : setprecision(f, BigFloat, trf.prec)

function (trf::TeukolskyRadialFunction)(r; deriv::Int=0)
    deriv in (0, 1) || throw(ArgumentError("deriv must be 0 or 1 (got $deriv)"))
    _atprec(trf) do
        if trf.boundary === :In
            deriv == 0 ? Rin(trf.p, trf.ν, trf.fn, r; nmax=trf.nmax) :
                         dRin(trf.p, trf.ν, trf.fn, r; nmax=trf.nmax)
        else
            ct = trf.amplitudes.Ctrans
            deriv == 0 ? Rup(trf.p, trf.ν, trf.fn, r; nmax=trf.nmax, ctrans=ct) :
                         dRup(trf.p, trf.ν, trf.fn, r; nmax=trf.nmax, ctrans=ct)
        end
    end
end

function Base.getindex(trf::TeukolskyRadialFunction, key::AbstractString)
    k = lowercase(key)
    k == "boundarycondition" ? String(trf.boundary) :
    k == "s"                 ? trf.s :
    k == "l"                 ? trf.l :
    k == "m"                 ? trf.m :
    k == "a"                 ? trf.a :
    (k == "omega" || k == "ω")        ? trf.ω :
    (k == "nu"    || k == "ν")        ? trf.ν :
    (k == "lambda" || k == "λ")       ? trf.λ :
    (k == "amplitudes")               ? trf.amplitudes :
    throw(KeyError(key))
end

function Base.show(io::IO, trf::TeukolskyRadialFunction)
    print(io, "TeukolskyRadialFunction(\"$(trf.boundary)\"; s=$(trf.s), l=$(trf.l), ",
              "m=$(trf.m), a=$(trf.a), ω=$(trf.ω))")
end

function _validate_modes(s::Int, l::Int, m::Int, a)
    l ≥ abs(s)  || throw(ArgumentError("l ≥ |s| required (got l=$l, s=$s)"))
    abs(m) ≤ l  || throw(ArgumentError("|m| ≤ l required (got m=$m, l=$l)"))
    abs(a) < 1  || throw(ArgumentError("sub-extremal |a| < 1 required (got a=$a)"))
end

"""
    TeukolskyRadial(s, l, m, a, ω; nmax=80, l_max=20)

Construct the homogeneous radial Teukolsky solutions (MST method). Returns a
NamedTuple with callable `In` and `Up` [`TeukolskyRadialFunction`](@ref)s plus
shared mode data and the angular harmonic:

    tr = TeukolskyRadial(-2, 2, 2, 0.0, 0.5)
    tr.In(10.0); tr.Up(10.0; deriv=1)
    tr.ν; tr.λ; tr.amplitudes
    tr.S(θ)                       # spin-weighted spheroidal harmonic S_lm(θ)

Pass BigFloat `a`/`ω` (inside `setprecision`) for an arbitrary-precision solution.
"""
function TeukolskyRadial(s::Int, l::Int, m::Int, a, ω; nmax::Int=80, l_max::Int=20)
    _validate_modes(s, l, m, a)
    R    = promote_type(typeof(float(real(a))), typeof(float(real(complex(ω)))))
    a_r  = R(a)
    ω_c  = Complex{R}(complex(ω))
    prec = R <: BigFloat ? precision(a_r) : 0

    amp = compute_amplitudes(s, l, m, a_r, ω_c; nmax=nmax)
    ν   = amp.ν
    p   = MSTParams(s, l, m, a_r, ω_c; l_max=l_max)
    λ   = p.λ
    fn  = amp.fn

    mk(bc) = TeukolskyRadialFunction{R}(bc, s, l, m, a_r, ω_c, ν, λ, p, fn, amp, nmax, prec)
    S(θ; φ=0) = SpinWeightedSpheroidalHarmonicS(s, l, m, a_r, ω_c, θ; φ=φ, l_max=l_max)

    return (In=mk(:In), Up=mk(:Up), ν=ν, λ=λ, amplitudes=amp,
            s=s, l=l, m=m, a=a_r, ω=ω_c, S=S)
end
