# ============================================================
#  Arb compatibility shims (ADDITIVE — Float64/BigFloat untouched)
#
#  Base's generic Complex algorithm for sqrt / acos / acosh on Complex{T} relies
#  on `nextfloat(::T, ::Int)` (branch-cut / overflow guards), which is undefined
#  for T===Arb.  We route these three transcendentals through Arblib's native
#  complex type `Acb`, which implements them at full working precision with the
#  correct principal branch.
#
#  Each method below is strictly MORE SPECIFIC than Base's
#  `sqrt/acos/acosh(::Complex{T<:AbstractFloat})` (because Arb<:AbstractFloat),
#  so they fire ONLY for Complex{Arb} — the Float64 and BigFloat dispatch is
#  never re-selected and stays bit-identical.  Zero call-site edits required.
#
#  Type piracy note: these add methods for Base functions on the Arblib-owned
#  type Complex{Arb}.  Benign here (Arblib defines none of these for this
#  signature; they fire only for Complex{Arb}), and kept localised in this file.
# ============================================================

# Bridge: evaluate f on Acb, convert back to Complex{Arb}.
@inline function _acb_bridge(f, z::Complex{Arb})
    g = f(Acb(z))
    return Complex{Arb}(real(g), imag(g))
end

Base.sqrt(z::Complex{Arb})  = _acb_bridge(sqrt,  z)
Base.acosh(z::Complex{Arb}) = _acb_bridge(acosh, z)

# acos needs a NaN guard: Arblib's `acb_acos` returns NaN for |z| ≳ 1e40 at
# limited precision (the |cos 2πν| ≫ 1 regime reached at large |ω| + large
# Im ν), where Base's BigFloat algorithm is perfectly happy.  The ν path strips
# the argument to a point before calling acos, so falling back to a BigFloat
# evaluation at the working precision is exact-equivalent and branch-faithful
# (it IS Base.acos).  The common small-|z| case never trips the guard, so the
# tested low-ω paths are unchanged.
@inline _acb_has_nan(g::Acb) = isnan(Float64(real(g))) || isnan(Float64(imag(g)))
function Base.acos(z::Complex{Arb})
    g = acos(Acb(z))
    _acb_has_nan(g) || return Complex{Arb}(real(g), imag(g))
    setprecision(BigFloat, precision(Arb)) do
        w = acos(Complex{BigFloat}(z))
        Complex{Arb}(Arb(real(w)), Arb(imag(w)))
    end
end
# Base's generic log(::Complex{T}) calls ssqs → nextfloat(::T,::Int) for its
# scaling/overflow guard, which is undefined for T===Arb.  Route through Acb's
# native principal-branch complex log.  (exp/sin/cos use only real ops and need
# no bridge; complex `^` is exp(w·log(z)), so fixing log fixes those call sites.)
Base.log(z::Complex{Arb})   = _acb_bridge(log,   z)

# Arblib promotes mixed `Arb / Complex{Arb}` (and similar real-÷-complex ops) to
# its own `Acb`, where Base would keep `Complex{Arb}`.  Downstream MST code then
# calls `complex(::Acb)` (e.g. radial_in.jl `complex((rp-r)/(2κ))`) which Base
# has no method for.  Bridge it back into the generic Complex{Arb} stack so those
# call sites keep working without rewriting every mixed division.  (The native
# Acb monodromy kernel converts explicitly and never relies on this.)
Base.complex(z::Acb) = Complex{Arb}(real(z), imag(z))

# HypergeometricFunctions._₂F₁ (used by the radial Rin/Rup series) evaluates
# `loggamma` on Complex parameters; SpecialFunctions has no Complex{Arb} method,
# so the 2F1 connection formula dies with `_loggamma(::Complex{Arb})`.  Route
# through Acb's native log-Γ, mirroring `_cgamma` in utils.jl.  Fires only for
# Complex{Arb}; Float64/BigFloat keep SpecialFunctions' own implementation.
SpecialFunctions.loggamma(z::Complex{Arb}) =
    Complex{Arb}(SpecialFunctions.loggamma(Acb(z)))

# --- Strip the ball radius, returning the midpoint as an exact (0-radius) ball ---
#
# At large |ω| the monodromy closed form cos(2πν) loses ALL rigorous ball
# precision to catastrophic cancellation: its midpoint stays correct (it matches
# the BigFloat value to working precision), but the certified radius can exceed
# the midpoint magnitude (rel_accuracy_bits < 0).  `acos`/`acosh` of a ball that
# wide return NaN, killing the ν extraction at |ω|≳5.  Treating the converged
# value as a POINT estimate — the same midpoint philosophy as the convergence
# tests — recovers a finite, correct ν.  No-op for Float64/BigFloat (no radius);
# see the generic `_strip_radius(::Complex)=z` in nu_solver.jl.
function _strip_radius(z::Complex{Arb})
    re = Arb(real(z)); Arblib.zero!(Arblib.radref(re))
    im = Arb(imag(z)); Arblib.zero!(Arblib.radref(im))
    return Complex{Arb}(re, im)
end
function _strip_radius(z::Acb)
    re = Arb(real(z)); Arblib.zero!(Arblib.radref(re))
    im = Arb(imag(z)); Arblib.zero!(Arblib.radref(im))
    return Complex{Arb}(re, im)
end

# --- Modified-Lentz continued fraction for Complex{Arb} (point arithmetic) -----
#
# The generic `_lentz_cf` (recurrence.jl) is unusable on Complex{Arb} BALLS for
# ill-conditioned CFs (large |ω|): the recurrence's `inv(D)` / `aj/C` steps act
# on balls that straddle zero, so the result's MIDPOINT is corrupted from the
# first iterations on — the CF then "converges" (|Δ−1|→0) to a value that is
# ~100% wrong, INDEPENDENT of working precision.  BigFloat is immune because it
# carries no radius: its midpoint arithmetic is exact-to-precision.
#
# So we make Arb behave like BigFloat — POINT arithmetic — by stripping the ball
# radius of C, D and f every step (the midpoints are then computed exactly to the
# working precision, exactly as MPFR does).  Convergence is the same |Δ−1| < tol
# test as the generic path, but on the (0-radius) midpoint, so it is decidable
# and fires at the same place BigFloat would.  Accuracy then tracks BigFloat at
# the working precision on every mode and every |ω| (validated against the
# BigFloat amplitude path); buy more digits at large |ω| with a precision sweep.
#
# Strictly MORE SPECIFIC than `_lentz_cf(a,b,::Type{T<:Complex})`, so it fires
# only for Complex{Arb}; Float64/BigFloat keep the byte-identical generic path.
function _lentz_cf(a, b, ::Type{Complex{Arb}}; tol, maxiter::Int)
    T       = Complex{Arb}
    tiny    = T(eps(Arb)^2)
    one_arb = one(Arb)
    f = tiny; C = f; D = zero(T)
    best_f = f
    best_d = Arb(Inf)
    stalls = 0                 # consecutive iterations with no |Δ−1| improvement
    for j in 1:maxiter
        aj = a(j); bj = b(j)  # a, b are closures of the CF index j (see recurrence.jl)
        D = _strip_radius(bj + aj * D); iszero(D) && (D = tiny)
        D = _strip_radius(inv(D))
        C = _strip_radius(bj + aj / C); iszero(C) && (C = tiny)
        Δ = C * D
        f = _strip_radius(f * Δ)
        d = abs(_strip_radius(Δ) - one_arb)
        d < tol && return f, true                 # full convergence (well-conditioned)
        # Conditioning floor: at large |ω| the CF cannot reach the eps-scale tol
        # (it loses ~const bits to cancellation — buy them back with precision),
        # so |Δ−1| descends to a floor and plateaus.  Stop there instead of
        # churning every iteration to the cap; the point arithmetic keeps the
        # midpoint stable, so the best iterate IS the most accurate value.
        if d < best_d
            best_d = d; best_f = f; stalls = 0
        else
            stalls += 1
            stalls ≥ 24 && return best_f, true
        end
    end
    return best_f, false
end
