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
Base.acos(z::Complex{Arb})  = _acb_bridge(acos,  z)
Base.acosh(z::Complex{Arb}) = _acb_bridge(acosh, z)
