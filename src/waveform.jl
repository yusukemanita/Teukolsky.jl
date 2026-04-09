# ============================================================
#  Waveform module
#  Computes ψ₄(t) via direct frequency-domain integration:
#
#    ψ(t) = Δω/(2π) Σ_n G(ω_n) e^{-iω_n t}
#
#  G(ω) = B^ref / (2iω B^inc)  (retarded Green's function)
#
#  Half-integer frequency grid avoids ω = 0 singularity.
# ============================================================

module Waveform

using BHPtoolkit: compute_amplitudes

export WaveformParams, compute_waveform, green_function

# ============================================================
#  Parameter struct
# ============================================================

"""
    WaveformParams(; s, l, m, a, N, ω_max, t_ini, t_max, Nt)

Parameters for waveform computation.

# Fields
- `s`, `l`, `m`: spin weight and angular mode numbers
- `a`: Kerr spin parameter (0 ≤ a < 1)
- `N`: number of frequency grid points (default: 3000)
- `ω_max`: frequency cutoff in units of M⁻¹ (default: 6.0)
- `t_ini`: start time (default: -100.0 M)
- `t_max`: end time (default: 600.0 M)
- `Nt`: number of time points (default: 3000)
- `taper_frac`: fraction of ω_max over which a Planck-taper window rolls G(ω) to 0
               (default: 0.1).  Eliminates Gibbs pre-cursor and late-time noise floor.
- `verbose`: print progress (default: true)
"""
struct WaveformParams
    s::Int
    l::Int
    m::Int
    a::Float64
    N::Int
    ω_max::Float64
    t_ini::Float64
    t_max::Float64
    Nt::Int
    taper_frac::Float64
    verbose::Bool
end

function WaveformParams(;
    s::Int, l::Int, m::Int, a::Real,
    N::Int=3000, ω_max::Real=6.0,
    t_ini::Real=-100.0, t_max::Real=600.0, Nt::Int=3000,
    taper_frac::Real=0.1, verbose::Bool=true)

    WaveformParams(s, l, m, Float64(a), N, Float64(ω_max),
                   Float64(t_ini), Float64(t_max), Nt, Float64(taper_frac), verbose)
end

# ============================================================
#  Planck-taper window
# ============================================================

"""
    planck_taper(x)

Planck-taper kernel: C∞ ramp from 1 at x=0 to 0 at x=1.
Used to smoothly suppress G(ω) near ω_max, eliminating the
Gibbs pre-cursor and late-time noise floor that arise from
abrupt frequency truncation.
"""
@inline function planck_taper(x::Real)
    x ≤ 0.0 && return 1.0
    x ≥ 1.0 && return 0.0
    return 1 / (exp(1/x - 1/(1-x)) + 1)
end

"""
    taper_weight(ω, ω_max, frac)

Window weight for frequency ω: 1 for |ω| ≤ (1-frac)*ω_max,
then Planck-taper to 0 at |ω| = ω_max.
"""
@inline function taper_weight(ω::Real, ω_max::Real, frac::Real)
    frac ≤ 0.0 && return 1.0
    x = (abs(ω) / ω_max - (1 - frac)) / frac   # 0 at inner edge, 1 at ω_max
    return planck_taper(x)
end

# ============================================================
#  Green's function
# ============================================================

"""
    green_function(p::WaveformParams, ω) -> ComplexF64

Evaluate G(ω) = B^ref / (2iω B^inc).

For negative ω, enforces the reality condition G(-ω) = conj(G(ω)),
which makes the time-domain waveform ψ(t) real-valued.  This is the
appropriate convention for the gravitational-wave signal at a fixed
sky position (combining m and -m contributions).
"""
function green_function(p::WaveformParams, ω)
    s, l, m, a = p.s, p.l, p.m, p.a
    ω_pos = abs(real(ω))
    amp   = compute_amplitudes(s, l, m, a, ω_pos)
    G_pos = amp.Bref / (2im * ω_pos * amp.Binc)
    return real(ω) >= 0 ? G_pos : conj(G_pos)
end

# ============================================================
#  Main function
# ============================================================

"""
    compute_waveform(p::WaveformParams) -> (t_grid, ψ, GF, ω_grid)

Compute the time-domain waveform ψ₄(t) for the given parameters.

Returns:
- `t_grid`: time grid (StepRangeLen)
- `ψ`: complex waveform values (length Nt)
- `GF`: Green's function values on ω_grid (length N)
- `ω_grid`: half-integer frequency grid (length N)
"""
function compute_waveform(p::WaveformParams)
    s, l, m, a = p.s, p.l, p.m, p.a
    N, ω_max   = p.N, p.ω_max
    Δω = 2ω_max / N

    # Half-integer grid: ω_n = (n - N/2 + 1/2)Δω, avoids ω = 0
    ω_grid = [(n - N÷2 + 0.5) * Δω for n in 0:N-1]
    t_grid = range(p.t_ini, p.t_max; length=p.Nt)

    # ── Step 1: evaluate G(ω) on frequency grid ─────────────
    p.verbose && (println("Evaluating G(ω) on $N frequency points ..."); flush(stdout))

    # Evaluate G only at positive frequencies; negative half uses G(-ω) = conj(G(ω)).
    # The half-integer grid satisfies ω_grid[i] = -ω_grid[N+1-i], so we compute
    # the upper half (i = N÷2+1 : N) and mirror into the lower half.
    GF = Vector{ComplexF64}(undef, N)
    for i in (N÷2 + 1):N
        ω = ω_grid[i]
        w = taper_weight(ω, ω_max, p.taper_frac)
        if iszero(w)
            GF[i] = zero(ComplexF64)
        else
            amp    = compute_amplitudes(s, l, m, a, ω)
            GF[i]  = amp.Bref / (2im * ω * amp.Binc) * w
        end
        p.verbose && i % 200 == 0 && (print("."); flush(stdout))
    end
    for i in 1:(N÷2)
        GF[i] = conj(GF[N + 1 - i])   # G(-ω) = conj(G(ω))
    end
    p.verbose && println("\nDone.")

    # ── Step 2: direct integration over time ─────────────────
    p.verbose && (println("Integrating for $(p.Nt) time points ..."); flush(stdout))

    ψ      = Vector{ComplexF64}(undef, p.Nt)
    prefac = Δω / (2π)

    for (k, t) in enumerate(t_grid)
        s_val = zero(ComplexF64)
        @inbounds for n in 1:N
            s_val += GF[n] * exp(-im * ω_grid[n] * t)
        end
        ψ[k] = prefac * s_val
        p.verbose && k % 50 == 0 && (print("."); flush(stdout))
    end
    p.verbose && println("\nDone.")

    return t_grid, ψ, GF, ω_grid
end

end  # module Waveform
