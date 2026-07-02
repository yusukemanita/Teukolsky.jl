# ============================================================
#  Precision predictor for the MST solve
#
#  At large frequency the MST three-term recurrence / continued fraction is
#  ill-conditioned: it needs BOTH more series terms (nmax) AND more mantissa bits.
#  An adaptive driver that blindly retries Float64x4 → BigFloat256 → BigFloat512 …
#  pays for every failed lower-precision attempt before the successful one.  This
#  helper predicts the working precision and truncation directly from |ω|, so the
#  driver can jump straight to (near) the right level and verify-and-escalate only
#  as a backstop.
#
#  CALIBRATION (see test/test_precision_hint.jl and the header of the Gpia driver):
#  measured on the branch-cut integrand  q̃·R^up  (s=-2, m=2, a=0.7, r'=10) over
#  |ω| ∈ [0.05, 10], comparing each (bits, nmax) against a BigFloat-1408 / nmax-240
#  reference to a relative tolerance 1e-12:
#
#     |ω| ≲ 2.4  :  Float64x4 (212 bit),  nmax 40
#     |ω| ≈ 3.9  :  Float64x4 still exact, nmax 60         (F64x4 valid to |ω|≈3.9)
#     |ω| ≈ 4.4  :  256 bit,  nmax 60
#     |ω| ≈ 5.1  :  320 bit
#     |ω| ≈ 6.7  :  384 bit,  nmax 80
#     |ω| ≈ 7.6  :  448 bit
#     |ω| ≈ 10   :  512 bit,  nmax 100
#
#  The fitted envelope below is deliberately a mild OVER-estimate (snapped up to a
#  discrete bit ladder) so a single solve usually suffices; it is a STARTING point,
#  never a correctness guarantee — always keep a finite-check + escalation backstop.
# ============================================================

# Discrete mantissa-bit ladder the predictor snaps up to (212 ≈ Float64x4).
const _MST_BIT_LADDER = (212, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536)

# |ω| below which Float64x4 (212 bit) is validated bit-faithful to BigFloat for
# the s=-2 branch-cut integrand.  A margin under the measured 3.9 breakdown so the
# fast native path is used only where it is provably clean.
const MST_F64X4_TRUST = 3.5

"""
    suggest_mst_precision(ω; l=2, margin=1.0)
      -> (backend::Symbol, bits::Int, nmax::Int)

Predict the working-precision backend, mantissa bit-count, and series truncation
`nmax` for ONE MST solve at frequency `ω`.  `backend` is `:multifloat`
(Float64x4, `bits==212`) for `|ω| ≤ MST_F64X4_TRUST`, else `:bigfloat` at the
predicted bit-count snapped up to [`_MST_BIT_LADDER`](@ref).  `l` bumps `nmax`
(higher multipoles need a few more terms); `margin` (>1 widens, <1 tightens)
scales the predicted bits before snapping.

This is calibrated for the spin-`s=-2` branch-cut integrand (see file header) and
is intended as the START of a verify-and-escalate ladder — pair it with a
finite/consistency check that escalates on failure; do not treat it as exact.
"""
function suggest_mst_precision(ω; l::Int=2, margin::Real=1.0)
    x = abs(complex(float(real(ω)), float(imag(complex(ω)))))

    # nmax(|ω|, l): smooth envelope over the measured need (40 up to |ω|≈2, then
    # ~9 terms per unit |ω|) plus a small per-multipole bump.  Deliberately smooth
    # — the sharp local nmax spikes in the calibration sit at near-zeros of the
    # integrand (magnitude ~1e-40, negligible in the σ-integral); the driver's
    # finite/escalation backstop covers those rare points.
    nmax = clamp(round(Int, 40 + 9 * max(0.0, x - 2.0)) + 2 * max(0, l - 2), 40, 120)

    if x ≤ MST_F64X4_TRUST
        return (backend = :multifloat, bits = 212, nmax = nmax)
    end

    # bits(|ω|): 212 at the F64x4 edge, ~72 bits per unit |ω| beyond, snapped UP
    # to the ladder (mild over-estimate → single solve usually converges).
    raw  = 212.0 + 72.0 * (x - MST_F64X4_TRUST)
    want = margin * raw
    bits = _MST_BIT_LADDER[end]
    for b in _MST_BIT_LADDER
        if b ≥ want && b > 212
            bits = b
            break
        end
    end
    return (backend = :bigfloat, bits = max(bits, 256), nmax = nmax)
end
