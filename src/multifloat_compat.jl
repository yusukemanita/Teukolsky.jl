# ============================================================
#  MultiFloats compatibility shims + precision-backend dispatch
#  (ADDITIVE — the Float64 / BigFloat / Arb paths are untouched)
#
#  MultiFloats.jl (Float64x1 … Float64x8 = MultiFloat{Float64,N}) gives native
#  +,−,*,/, sqrt, exp and real log at extended precision, and Base's generic
#  Complex algorithms (sqrt, exp, …) work on Complex{MultiFloat} out of the box
#  because `nextfloat`/`eps`/`precision` are all defined.  Three gaps remain for
#  the MST pipeline, patched here:
#
#   (1) Real transcendentals (sin, cos, tan, asin, …) throw a "not yet
#       implemented" stub.  We enable MultiFloats' own sanctioned BigFloat
#       fallback once, at module load, via `use_bigfloat_transcendentals()`.
#       This also defines the 1-argument `atan`.
#
#   (2) The 2-argument `atan(y, x)` is NOT covered by that fallback, yet
#       Base's `log`/`acos`/`acosh` on Complex{MultiFloat} need it (the
#       imaginary part is `atan2`).  We add it explicitly, routed through
#       BigFloat at the working precision.
#
#   (3) `gamma` has no MultiFloat method (SpecialFunctions).  We add a
#       `_cgamma(::Complex{MultiFloat})` method (strictly more specific than the
#       generic `_cgamma(::Complex{T<:AbstractFloat})` in utils.jl), routed
#       through BigFloat — exactly mirroring `_cgamma(::Complex{Arb})`.
#
#  Why BigFloat routing is fine: the MST hot loops (monodromy recurrence, the
#  f_n Lentz continued fraction, the amplitude/K_ν sums) are PURE arithmetic and
#  run natively in MultiFloat.  The transcendentals and Γ above appear only O(1)
#  times per call (prefactors / branch extraction), so the BigFloat detour costs
#  nothing measurable while keeping the bulk arithmetic in fast MultiFloat.
#
#  Type-piracy note: methods (2)–(3) add Base/Γ methods on the MultiFloats-owned
#  type Complex{MultiFloat}; benign (no existing method for these signatures) and
#  localised to this file, as with arb_compat.jl.
# ============================================================

using MultiFloats: MultiFloat

# ── (1) enable real transcendentals (called from BHPtoolkit.__init__) ────────
# `use_bigfloat_transcendentals()` evaluates Base method definitions, so it must
# run at load time (not precompile); BHPtoolkit's __init__ calls this.  Idempotent.
_enable_multifloat_transcendentals() = MultiFloats.use_bigfloat_transcendentals()

# ── (2) 2-argument atan (the one gap use_bigfloat_transcendentals misses) ────
@inline function Base.atan(y::MultiFloat{T,N}, x::MultiFloat{T,N}) where {T,N}
    p = precision(MultiFloat{T,N}) + 10
    return setprecision(BigFloat, p) do
        MultiFloat{T,N}(atan(BigFloat(y; precision=p), BigFloat(x; precision=p)))
    end
end

# ── (3) complex Γ at MultiFloat working precision (mirrors the Arb method) ────
function _cgamma(z::Complex{MultiFloat{T,N}}) where {T,N}
    MF = MultiFloat{T,N}
    p  = precision(MF) + 10
    return setprecision(BigFloat, p) do
        g = _cgamma(Complex{BigFloat}(BigFloat(real(z); precision=p),
                                      BigFloat(imag(z); precision=p)))
        Complex{MF}(MF(real(g)), MF(imag(g)))
    end
end

# ── (3b) Γ and log-Γ at MultiFloat precision (radial ₂F₁ connection formula) ──
# HypergeometricFunctions._₂F₁ (used by the radial Rin/Rup series) evaluates
# `gamma`/`loggamma` on BOTH real and complex parameters; SpecialFunctions has no
# MultiFloat method, so Rin dies with `_gamma`/`_loggamma(::(Complex){MultiFloat})`.
# Route each through BigFloat at the working precision (mirrors `_cgamma` above
# and the Arb bridges); these appear O(1) times per radial evaluation, so the
# BigFloat detour is negligible against the native-MultiFloat series arithmetic.
for G in (:gamma, :loggamma)
    @eval function SpecialFunctions.$G(x::MultiFloat{T,N}) where {T,N}
        MF = MultiFloat{T,N}; p = precision(MF) + 10
        return MF(setprecision(() -> SpecialFunctions.$G(BigFloat(x; precision=p)),
                               BigFloat, p))
    end
    @eval function SpecialFunctions.$G(z::Complex{MultiFloat{T,N}}) where {T,N}
        MF = MultiFloat{T,N}; p = precision(MF) + 10
        return setprecision(BigFloat, p) do
            g = SpecialFunctions.$G(Complex{BigFloat}(BigFloat(real(z); precision=p),
                                                      BigFloat(imag(z); precision=p)))
            Complex{MF}(MF(real(g)), MF(imag(g)))
        end
    end
end

# ── (4) Integer conversion of a MultiFloat ───────────────────────────────────
# MultiFloats provides round/ceil/floor/trunc (each returning a MultiFloat) but
# NOT the Integer conversion of the result, so `round(Int, ::MultiFloat)` — used
# in the radial hypergeometric path (the near-integer-b guard in
# hypergeometric.jl) — hits a missing `Int64(::MultiFloat)` and the radial Rin/Rup
# fail under the MultiFloat backend.  Route through the leading Float64 limb,
# which is exact for the small integer-valued arguments these call sites produce.
(::Type{I})(x::MultiFloat) where {I<:Integer} = I(Float64(x))

# ============================================================
#  Precision-backend dispatch  (Float64 / BigFloat / MultiFloat)
# ============================================================

# MultiFloats.jl ships specialised mul/sqr/div/sqrt kernels only for widths 1–4
# (Float64x1 … Float64x4); x5–x8 throw `no method matching mfmul/mfsqr`.  So the
# MultiFloat backend tops out at Float64x4 (~212 bits ≈ 63 decimal digits); for
# higher precision use the BigFloat backend.
const _MF_MAX_WIDTH = 4

"""
    _multifloat_type(precision) -> MultiFloat{Float64,N}

Working MultiFloat type for a requested bit precision: `N = clamp(⌈prec/53⌉,1,4)`
limbs (each Float64 limb ≈ 53 bits), so `precision=64` → `Float64x2` (~106 bits),
`128` → `Float64x3` (~159 bits), `≥160` → `Float64x4` (~212 bits, the maximum
width MultiFloats.jl provides arithmetic for; request more via the BigFloat
backend).
"""
function _multifloat_type(precision::Int)
    N = clamp(cld(precision, 53), 1, _MF_MAX_WIDTH)
    return MultiFloat{Float64, N}
end

"""
    _with_backend(f, backend, precision, a, ω)

Convert `(a, ω)` to the working float type selected by `backend` and call
`f(a_w, ω_w)`:

  - `:float64`    → `Float64` / `ComplexF64` (`precision` ignored)
  - `:bigfloat`   → `BigFloat` at `precision` bits (run inside `setprecision`)
  - `:multifloat` → `Float64xN`, `N` from [`_multifloat_type`](@ref)
  - `:arb`        → `Arb` ball arithmetic at `precision` bits (run inside `setprecision`)
  - `:acb`        → equivalent to `:arb` for amplitudes (`Complex{Arb}` pipeline)

This is the single dispatch point shared by `compute_nu` and `compute_amplitudes`.
"""
function _with_backend(f, backend::Symbol, precision::Int, a, ω)
    ωc = complex(ω)
    if backend === :float64
        return f(Float64(real(a)), ComplexF64(ωc))
    elseif backend === :bigfloat
        return setprecision(BigFloat, precision) do
            f(BigFloat(real(a)),
              Complex{BigFloat}(BigFloat(real(ωc)), BigFloat(imag(ωc))))
        end
    elseif backend === :multifloat
        R = _multifloat_type(precision)
        return f(R(real(a)), Complex{R}(R(real(ωc)), R(imag(ωc))))
    elseif backend === :arb || backend === :acb
        # Arb ball arithmetic at precision bits.  The amplitude/radial pipeline
        # runs entirely in Complex{Arb}, so :arb and :acb are EQUIVALENT here (the
        # native-Acb kernel only accelerates the standalone nu solver via
        # compute_nu(...; backend=:acb)); both are accepted for keyword parity.
        return setprecision(Arb, precision) do
            f(Arb(real(a)),
              Complex{Arb}(Arb(real(ωc)), Arb(imag(ωc))))
        end
    else
        error("_with_backend: unknown backend $(repr(backend)); " *
              "use :float64, :bigfloat, :multifloat, :arb, or :acb")
    end
end
