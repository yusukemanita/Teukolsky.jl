# ============================================================
#  MST Monodromy Coefficient K(ω)
#
#  K(ω) is defined by the connection formula (Leaver 1986 eq. 31):
#
#    R^up(ω e^{2πi}) = R^up(ω) − K(ω) R^down(ω)
#
#  It characterises the branch cut of G(ω) = B^ref/(2iω B^inc)
#  along the imaginary ω-axis, and controls the late-time power-law
#  tail of the Teukolsky waveform (Leaver 1986 eq. 36).
#
#  Formula (Leaver 1986 eq. 32, adapted to MST/Sasaki-Tagoshi):
#
#    K(ω) = (1 − e^{2πi(ε̃ − ν)}) × (2ε)^{−2iε̃} × N₊/N₋
#
#  where
#    ε = 2Mω = p.ϵ   (MST frequency parameter, M = 1 in this code)
#    ε̃ ≈ ε           (exact for Schwarzschild; Kerr: ε̃ = ε − isκ/2, TBD)
#    ν               (renormalized angular momentum from compute_nu)
#
#    N₊ = Σ_n fₙ × [Γ(n+ν+1+iε̃)/Γ(n+ν+1−iε̃)]^{−1/2} × e^{+iπ(n+ν)/2}
#    N₋ = Σ_n fₙ × [Γ(n+ν+1+iε̃)/Γ(n+ν+1−iε̃)]^{+1/2} × e^{−iπ(n+ν)/2}
#
#  References:
#    Leaver (1986) Phys. Rev. D 34, 384 — eqs. (31)-(36)
#    Mano, Suzuki, Takasugi (1996) PTP 96, 549
#    Sasaki & Tagoshi (2003) Living Rev. Rel. 6, 6 (journal version)
#    Casals & Ottewill (2022) Phys. Rev. D 106, 044030
# ============================================================

"""
    compute_monodromy_K(s, l, m, a, ω; nmax=40, nmax_cf=150)

Compute the MST monodromy coefficient K(ω) for the Teukolsky equation.

K(ω) encodes the analytic continuation R^up(ωe^{2πi}) = R^up(ω) - K(ω)R^down(ω),
and determines the branch-cut discontinuity of the retarded Green's function
G(ω) = B^ref / (2iω B^inc) along the imaginary ω-axis.

# Formula

    K(ω) = (1 − e^{2πi(ε̃−ν)}) × (2ε)^{−2iε̃} × N₊/N₋

where ε = p.ϵ = 2Mω (M=1 in this code), ε̃ ≈ ε (exact for Schwarzschild),
and N₊, N₋ are sums over the MST series coefficients fₙ^ν.

# Sanity checks

- K(ω → 0) ≈ −2πi ω e^{iℓπ} for Schwarzschild
- K = 0 when ν − ε̃ ∈ ℤ (monodromy factor vanishes at integer spacing)
- |K| ≫ 1 near QNM poles (where B^inc → 0)

# Returns

Named tuple `(K, ν, p)`:
- `K`: complex monodromy coefficient
- `ν`: renormalized angular momentum
- `p`: MSTParams struct
"""
function compute_monodromy_K(s::Int, l::Int, m::Int, a, ω;
                              nmax::Int=40, nmax_cf::Int=150)
    ν, p = compute_nu(s, l, m, a, ω; nmax_cf=nmax_cf)
    fn   = compute_fn(p, ν; nmax=nmax)

    ε  = p.ϵ   # = 2Mω  (M = 1)
    # Coulomb parameter ε̃:
    #   Schwarzschild (a=0): ε̃ = ε  (exact)
    #   Kerr general s:      ε̃ = ε − isκ/2  (from Sasaki-Tagoshi §3; NOT yet implemented)
    ε̃  = ε
    T  = typeof(ε)

    # ── Normalization sums N₊, N₋ ─────────────────────────────
    #
    #   N₊ = Σ_n fₙ exp(−g_n/2) exp(+iπ(n+ν)/2)
    #   N₋ = Σ_n fₙ exp(+g_n/2) exp(−iπ(n+ν)/2)
    #
    #   where g_n = log Γ(n+ν+1+iε̃) − log Γ(n+ν+1−iε̃)
    #
    # Use loggamma for numerical stability (avoids Gamma function overflow).

    N_plus  = zero(T)
    N_minus = zero(T)

    for n in -nmax:nmax
        fn_n = fn[n]
        iszero(fn_n) && continue

        arg_p = n + ν + 1 + im*ε̃
        arg_m = n + ν + 1 - im*ε̃
        g_half = (loggamma(arg_p) - loggamma(arg_m)) / 2

        e_phase = exp(im * π * (n + ν) / 2)

        N_plus  += fn_n * exp(-g_half) * e_phase
        N_minus += fn_n * exp(+g_half) / e_phase   # = exp(-g_half*conj) × exp(-iπ(n+ν)/2)
    end

    # ── Monodromy factor: 1 − e^{2πi(ε̃ − ν)} ────────────────
    mono_factor = 1 - exp(2π * im * (ε̃ - ν))

    # ── Branch-point factor: (2ε)^{−2iε̃} ─────────────────────
    # log(2ε) uses principal branch (Im(log) ∈ (−π, π]).
    # For ω near the negative imaginary axis (branch cut integration),
    # ε ≈ −2iσ and log(−2iσ) = log(2σ) − iπ/2, which is the correct
    # physical branch for the retarded Green's function.
    branch_factor = exp(-2im * ε̃ * log(2 * ε))

    # ── Assemble K ────────────────────────────────────────────
    K = mono_factor * branch_factor * N_plus / N_minus

    return (K=K, ν=ν, p=p)
end

# ============================================================
#  Branch-cut discontinuity via K
#
#  From Leaver (1986) eq. (34)-(36), the discontinuity of G across
#  the negative imaginary axis is related to K by:
#
#    ΔG_K(σ) = G_R(-iσ) − G_L(-iσ)
#            = K(-iσ) × R_in²(-iσ) / (B^inc(−iσ+δ) × W)
#
#  where W is the Wronskian (constant). This gives an alternative
#  to the direct two-sided evaluation used in branchcut_integral.jl.
#
#  NOTE: This is not yet implemented. The direct approach (ΔG = G_R − G_L)
#  is already working and numerically verified. K provides a complementary
#  analytic handle.
# ============================================================
