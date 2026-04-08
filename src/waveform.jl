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
    verbose::Bool
end

function WaveformParams(;
    s::Int, l::Int, m::Int, a::Real,
    N::Int=3000, ω_max::Real=6.0,
    t_ini::Real=-100.0, t_max::Real=600.0, Nt::Int=3000,
    verbose::Bool=true)

    WaveformParams(s, l, m, Float64(a), N, Float64(ω_max),
                   Float64(t_ini), Float64(t_max), Nt, verbose)
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
    N, ω_max = p.N, p.ω_max
    Δω = 2ω_max / N

    # Half-integer grid: ω_n = (n - N/2 + 1/2)Δω, avoids ω = 0
    ω_grid = [(n - N÷2 + 0.5) * Δω for n in 0:N-1]
    t_grid = range(p.t_ini, p.t_max; length=p.Nt)

    # ── Step 1: evaluate G(ω) on frequency grid ─────────────
    p.verbose && (println("Evaluating G(ω) on $N frequency points ..."); flush(stdout))

    GF = Vector{ComplexF64}(undef, N)
    for (i, ω) in enumerate(ω_grid)
        GF[i] = green_function(p, ω)
        p.verbose && i % 200 == 0 && (print("."); flush(stdout))
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
