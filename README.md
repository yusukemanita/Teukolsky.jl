# Teukolsky.jl

A Julia implementation of the **Mano–Suzuki–Takasugi (MST)** formalism for the
homogeneous Teukolsky equation,
together with the angular sector, Kerr geodesics, and point-particle gravitational-wave
fluxes. It is a Julia port of the core functionality of the
Mathematica [`Teukolsky` paclet](https://bhptoolkit.org/Teukolsky/),
written to run at **arbitrary (BigFloat) precision**.

Convention: geometrized units `G = c = 1` and black-hole mass `M = 1`.
Following Sasaki & Tagoshi, *Living Rev. Rel.* **6** (2003) 6.

---

## Features

| Module | What it computes | Status |
|---|---|---|
| **MST core** | Renormalized angular momentum ν, MST coefficients `f_n`, asymptotic amplitudes `B^inc/B^ref/B^trans/C^trans`, homogeneous radial solutions `R_in`, `R_up`, `R_down` (+ derivatives) | Schwarzschild & Kerr, arbitrary precision |
| **Angular** | Spin-weighted spheroidal harmonics `S_lm(θ)` (+ θ-derivatives), eigenvalue `λ = A_lm`, spherical–spheroidal coupling | validated to ~10⁻¹⁵ vs Wolfram |
| **Radial object** | Callable `TeukolskyRadial` mirroring the Mathematica object API | — |
| **Numerical backend** | Independent radial solver by adaptive Dormand–Prince integration (cross-check) | — |
| **Geodesics** | Kerr constants of motion (E, L, Q), Mino & Boyer–Lindquist frequencies, orbit trajectories, special orbits (ISCO, photon sphere, IBSO, ISSO, separatrix), Carlson elliptic integrals + Jacobi functions | validated to ~10⁻¹³ vs Wolfram |
| **Fluxes** | Point-particle source convolution → `Z^∞`/`Z^H`, energy & angular-momentum fluxes at infinity and the horizon | s = −2 circular equatorial |
| **PN series** | Post-Newtonian (low-frequency) expansions of ν, `a_n`, λ | Schwarzschild, l ≥ 1 |

Everything is type-generic: pass `Float64` arguments for fast double-precision work,
or `BigFloat` arguments inside `setprecision(...)` for arbitrary precision.

---

## Installation

The package lives in this repository with a standard `Project.toml`.

```julia
julia> using Pkg
julia> Pkg.activate("/Users/yusuke/work/Teukolsky.jl")
julia> Pkg.instantiate()
julia> using Teukolsky
```

Dependencies: `HypergeometricFunctions`, `SpecialFunctions`, `LinearAlgebra`, `Printf`.

---

## Quick start

### Homogeneous radial solutions

```julia
using Teukolsky

# Teukolsky radial functions for s=-2, l=m=2, Schwarzschild, Mω = 0.5
tr = TeukolskyRadial(-2, 2, 2, 0.0, 0.5)

tr.In(10.0)             # R_in at r = 10
tr.Up(10.0; deriv=1)    # dR_up/dr at r = 10
tr.ν                    # renormalized angular momentum
tr.λ                    # spheroidal eigenvalue A_lm
tr.amplitudes           # (Binc, Bref, Btrans, Ctrans, ν, fn, ...)
tr.S(π/2)               # spin-weighted spheroidal harmonic S_lm(θ)
```

Low-level entry points are also exported:

```julia
ν, p = compute_nu(-2, 2, 2, 0.0, 0.5)         # MST renormalized angular momentum
amp  = compute_amplitudes(-2, 2, 2, 0.0, 0.5) # NamedTuple: Binc, Bref, Btrans, Ctrans, ν, fn, ...
fn   = compute_fn(p, ν)                        # MST coefficients f_n (Dict)
Rin(p, ν, fn, 10.0)                            # bare ingoing solution
Rup(p, ν, fn, 10.0)                            # upgoing solution (Rdown: third solution, enters q̃)
dRin(p, ν, fn, 10.0); dRup(p, ν, fn, 10.0)     # r-derivatives
```

### Branch-cut coefficients and the shared MST core

For Green's-function / branch-cut work (ω on the positive imaginary axis), one
core solve feeds everything — ν, `f_n`, and A^ν_± are computed ONCE and reused:

```julia
using Arblib: Arb

setprecision(Arb, 320) do
    ω    = Complex{Arb}(Arb(0), Arb(43)/10)            # ω = 4.3i on the branch cut
    core = compute_mst_core(-2, 2, 2, Arb(7)/10, ω)    # Arb inputs → native :acb chain
    qt   = qtilde_from_core(core)                      # branch-cut coefficient q̃(ω)
    q    = q_from_core(core)                           # branch-cut coefficient q(ω)
    ru   = Rup(core.p, core.ν, core.fn, Arb(10); ctrans=mst_ctrans(core))
end
```

One-call versions solve the core internally and return `(q, ν, p, Ap, Am)` /
`(qtilde, ν, p, Ap, Am)`:

```julia
setprecision(Arb, 320) do
    compute_q(-2, 2, 2, Arb(7)/10, im*Arb(43)/10)        # q(ω):  branch-cut strength of R^up
    compute_qtilde(-2, 2, 2, Arb(7)/10, im*Arb(43)/10)   # q̃(ω):  branch-cut strength of R^down
end
```

(Arb inputs route these through the native `:acb` chain too; `big"…"` inputs also
work but run the slower BigFloat path.)

### Selectable precision backends

`compute_nu` and `compute_amplitudes` accept a `backend` (`Symbol`) keyword together
with `precision` (`Int`, in bits).  Two backends cover all use cases:

| backend | working type | when to use |
|---------|--------------|-------------|
| `:float64` | `Float64` | fast double precision — small \|ω\| only |
| `:acb` | native in-place Acb kernels at `precision` bits | everything else |

**`:float64` breaks down at large \|ω\|.**  The MST series loses digits to
cancellation like ~10^(3.4·\|ω\|), so 53-bit doubles stop resolving the answer
beyond \|ω\| ≈ 1–2.  From there, use `:acb` at the bit-count suggested by
`suggest_mst_precision(ω)` — a measured envelope that jumps straight to the
precision the frequency needs.

(Two more backends exist but are not recommended.  `:bigfloat` works at any
precision — every kernel is type-generic — but MPFR heap-allocates every
operation, while the `:acb` chain runs zero-allocation in-place kernels
(ν monodromy, CF-peeled `f_n`, A±) plus FLINT's rigorous `acb_hypgeom_u`:
identical values, measured 2.8–7.6× faster end-to-end.  `:multifloat`
(Float64×4) has fast arithmetic but **no native special functions** — its Γ, U,
and monodromy all detour through BigFloat — and hard-caps at 212 bits, so we
don't use it.  `:acb` wins; use it.)

```julia
using Teukolsky
compute_nu(-2, 2, 2, 0.7, 4.3im; backend=:acb, precision=320)
amp = compute_amplitudes(-2, 2, 2, 0.7, 4.3im; backend=:acb, precision=320)

suggest_mst_precision(4.3im)   # → (backend = :acb, bits = 320, nmax = 61)
```

### Choosing precision automatically: `suggest_mst_precision`

At large \|ω\| the required precision is set by the physics (the ~10^(3.4·\|ω\|)
cancellation), not by taste.  `suggest_mst_precision(ω; l=2, margin=1.0)` returns
the measured envelope — **which backend, how many bits, how many series terms** —
so a driver jumps straight to a working configuration instead of failing upward
through a retry ladder:

```julia
suggest_mst_precision(0.5im)   # → (backend = :multifloat, bits = 212, nmax = 40)
suggest_mst_precision(10im)    # → (backend = :acb,        bits = 768, nmax = 112)
```

Consume it like this (for `backend = :acb`, pass `Arb` inputs inside
`setprecision` — `compute_mst_core` dispatches to the native in-place chain):

```julia
using Arblib: Arb

h = suggest_mst_precision(ω; l=l)
setprecision(Arb, h.bits) do
    ωA   = Complex{Arb}(Arb(real(ω)), Arb(imag(ω)))
    core = compute_mst_core(s, l, m, Arb(a), ωA; nmax=h.nmax)
    qt   = qtilde_from_core(core)                     # branch-cut coefficient q̃
    ru   = Rup(core.p, core.ν, core.fn, Arb(r); nmax=h.nmax, ctrans=mst_ctrans(core))
    qt * ru
end
```

Three rules of use:

1. **It is a starting point, not a guarantee** — keep a cheap verify-and-escalate
   backstop in production: check the result is finite, and on failure retry `:acb`
   one or two rungs up the bit ladder (more `:acb` bits is 2.8–7.6× cheaper than
   BigFloat at equal bits).  Keep ONE BigFloat attempt as the *last* resort — it
   runs the independent generic kernels, so it separates "needs more bits" from
   "implementation bug" (if BigFloat succeeds at the same bits where `:acb`
   failed, that is a bug report).  The predictor mildly over-provisions on
   purpose, so a single solve almost always suffices.
2. **Don't hand it fewer bits hoping for speed.**  Below the envelope you get
   finite *garbage*, not an error — that failure mode is exactly what this
   function exists to prevent.
3. **More delivered digits = more bits**, roughly 2^(bits − 3.4·\|ω\|·log₂10).
   Raise `margin` (or add bits) if you need extra digits at fixed ω.  `nmax` is
   calibrated for the s=−2 branch-cut use case; treat it as a floor for unusual
   regimes.

### Spin-weighted spheroidal harmonics

```julia
SpinWeightedSpheroidalEigenvalue(-2, 2, 2, 0.45)       # λ = A_lm, oblateness γ = aω
SpinWeightedSpheroidalHarmonicS(-2, 2, 2, 0.9, 0.5, π/3)          # S_lm(θ) for a, ω
SpinWeightedSpheroidalHarmonicS(-2, 2, 2, 0.9, 0.5, π/3; deriv=1) # dS/dθ

# lower level: eigenvalue, bare spin-weighted spherical harmonic, and the
# spherical–spheroidal mixing coefficients (S_lm = Σ_l′ C[l′]·ₛY_{l′m})
compute_lambda(-2, 2, 2, 0.9, 0.5)
sYlm(-2, 2, 2, π/3)
ells, C = swsh_coefficients(-2, 2, 2, 0.9, 0.5)
```

### Kerr geodesics

```julia
E, L, Q = kerr_geo_constants_of_motion(0.9, 10.0, 0.1, 1.0)  # a, p, e, x
kerr_geo_frequencies(0.9, 10.0, 0.0, 1.0)                    # (Omega_r, Omega_theta, Omega_phi)
kerr_geo_isco(0.9, 1.0)                                      # ISCO radius (prograde)

orbit = KerrGeoOrbit(0.9, 10.0, 0.1, 1.0)   # callable trajectory (t, r, θ, φ)(λ)
```

### Point-particle fluxes (s = −2, circular equatorial)

```julia
# l=m=2 mode, Schwarzschild, circular orbit at r = 10M
md = TeukolskyPointParticleMode(-2, 2, 2, 0.0, 10.0)

md.EnergyFlux.Inf        # energy flux to infinity  ≈ 2.684e-5
md.EnergyFlux.Hor        # energy flux down the horizon
md.Z.ZInf, md.Z.ZHor     # asymptotic amplitudes
md.AngularMomentumFlux   # (Inf, Hor) = (m/ω)·EnergyFlux

# Kerr: a < 0 with prograde=true is the retrograde convention.
TeukolskyPointParticleMode(-2, 2, 2, 0.9, 6.0)   # prograde horizon flux < 0 (superradiance)
```

### Post-Newtonian series (Schwarzschild, l ≥ 1)

```julia
nu_pn(-2, 2, 2, 0.0; order=4)      # low-frequency (PN) expansion of ν as a series in ε = 2Mω
lambda_pn(-2, 2, 2, 0.0; order=4)  # PN expansion of the eigenvalue λ
```

### Frequency-domain Green's function & time-domain waveform

```julia
wp = WaveformParams(s=-2, l=2, m=2, a=0.0, N=100, ω_max=6.0, Nt=64)

green_function(wp, 0.5)                  # retarded G(ω) at one frequency
t, ψ, GF, ωs = compute_waveform(wp)      # ψ(t) by inverse FFT of G on the ω-grid
```

`compute_waveform` samples `G(ω)` on an `N`-point grid up to `ω_max` (mirrored by
`G(−ω) = conj G(ω)`, so ψ is real to rounding) and returns the time grid, waveform,
Green's-function samples, and frequency grid.  Type-generic: `BigFloat` parameters
give a `Complex{BigFloat}` waveform.

### Independent numerical cross-check

```julia
ni = NumericalIntegrationRadial(-2, 2, 2, 0.0, 0.5)   # adaptive Dormand–Prince solver
ni.In(10.0)                                            # same conventions as TeukolskyRadial
```

An MST-free radial solver used to cross-validate `R_in`/`R_up` (accuracy capped at
~10⁻²⁰ by the DP5 order — a cross-check, not a deep-precision production path).

---

## How it works

The radial Teukolsky equation is solved with the MST method: the homogeneous
solution is written as a series of hypergeometric functions whose expansion
coefficients `f_n` satisfy a three-term recurrence

```
α_n f_{n+1} + β_n f_n + γ_n f_{n-1} = 0,
```

with the recurrence coefficients of Sasaki–Tagoshi Eq. (124). Requiring the
series to converge fixes the **renormalized angular momentum ν** as the root of a
continued-fraction (monodromy) condition; `f_n` is then the minimal solution,
evaluated by modified-Lentz continued fractions. Matching the two natural series
representations (`₂F₁` near the horizon, the Coulomb/`U` form near infinity)
gives the asymptotic amplitudes `B^inc`, `B^ref`, `B^trans`, `C^trans` and hence
the ingoing/upgoing solutions `R_in`, `R_up`.

The angular sector solves the spin-weighted spheroidal eigenvalue problem by
spherical-harmonic decomposition (refined by Rayleigh-quotient iteration), giving
both `λ = A_lm` and `S_lm(θ)`. For sourced problems, the point-particle Teukolsky
source is convolved against `R_in`, `R_up`, and `S_lm`, producing the asymptotic
amplitudes `Z^∞`/`Z^H` and the gravitational-wave fluxes.

See the references in `CLAUDE.md` (Sasaki & Tagoshi 2003; the Mathematica MST
source) for the full mathematical structure.

---

## Performance vs. the Mathematica Teukolsky paclet

Same quantities, same machine, both at ~**100-digit working precision**
(Julia `:acb` at 336 bits; paclet inputs at 100 digits), s=−2, l=m=2, a=0.7,
measured sequentially with warm-up excluded.  Operations: the incidence
amplitude `Binc` and the upgoing radial solution `Rup(10)`, at \|ω\| = 6 on the
real axis and at a complex angle — a frequency high enough to be demanding but
where both engines still return answers.

| quantity, ω | Julia `:acb` | `Teukolsky` paclet |
|---|---|---|
| `Binc`, ω = 6 | **0.02 s — 98 digits** | 0.31 s — 20 certified digits |
| `Rup(10)`, ω = 6 | **0.08 s — 93 digits** | 0.70 s — 6 certified digits |
| `Binc`, ω = 6·e^{iπ/3} | **0.06 s — 94 digits** | 0.34 s — 17 certified digits |
| `Rup(10)`, ω = 6·e^{iπ/3} | **0.02 s — 95 digits** | 0.54 s — 19 certified digits |

"digits" = decimal digits of agreement with a 700-bit reference (Julia) resp.
the paclet's own certified significance (its actual agreement with the reference
matches those certificates: e.g. 24 digits for `Binc` and 7 for `Rup` on the real
axis).  From the same 100 working digits, the `:acb` chain delivers 93–98 while
the paclet's internal cancellation leaves 6–20; pushing \|ω\| higher widens the
gap until the paclet fails outright (at ω = 10 it certifies ≤4.6 digits on the
real axis and returns `ComplexInfinity` at the complex angle, while `:acb` still
delivers 89–99 digits there).  Both engines face the same intrinsic
~10^(3.4·\|ω\|) conditioning — more digits are bought with more bits, which
`suggest_mst_precision` picks automatically.

Where both engines deliver full precision (moderate \|ω\|), the difference is
speed alone: the `:acb` chain — in-place ν monodromy kernel, one-anchor CF-ratio
peeling for `f_n`, in-place A±, rigorous `acb_hypgeom_u` radial seeds — measures
3–20× faster than the paclet and 2.8–7.6× faster than Julia's own BigFloat path
at equal bits, with values verified against algorithm-independent arbiters
(bottom-up CF evaluation, Miller recurrence, exact-π MPFR rebuilds).

Caveats: one machine, single thread; the paclet was run out of the box (careful
hand-tuning of its precision can push it further); decimal-digit ↔ bit matching
is nominal.

---

## Project layout

```
src/
  Teukolsky.jl         module + exports
  params.jl             MSTParams, spheroidal eigenvalue λ
  recurrence.jl         α_n / β_n / γ_n MST recurrence coefficients
  nu_solver.jl          renormalized angular momentum ν (monodromy / continued fraction)
  amplitudes.jl         asymptotic amplitudes B^inc/B^ref/B^trans/C^trans
  hypergeometric.jl     ₂F₁ / U building blocks (arbitrary precision)
  radial_in.jl          R_in (ingoing) + derivative
  radial_up.jl          R_up (upgoing) + derivative
  radial_down.jl        R_down
  spheroidal.jl         S_lm(θ), eigenvalue, θ-derivatives
  teukolsky_radial.jl   callable TeukolskyRadial object (B1)
  numint_radial.jl      independent DP5 numerical backend (B3)
  elliptic_integrals.jl Carlson R_F/R_D/R_J/R_C + elliptic K/E/F/Π + Jacobi sn/am (B4)
  kerr_geo_constants.jl Kerr E, L, Q
  kerr_geo_frequencies.jl  radial/polar roots, Mino & BL frequencies
  kerr_geo_orbit.jl     callable trajectory
  special_orbits.jl     ISCO, photon sphere, IBSO, ISSO, separatrix
  teukolsky_mode.jl     point-particle source convolution + fluxes (B5)
  pn_series.jl, pn.jl   post-Newtonian (low-frequency) series (B6)
  waveform.jl           time-domain waveform reconstruction
test/                   test suite + Wolfram reference data
```

---

## Status & limitations

The full MST stack (λ, ν, amplitudes, `R_in`/`R_up`/`R_down` and derivatives) runs
at genuine arbitrary precision for both Schwarzschild and Kerr, on the real axis,
at complex frequencies, and on the branch-cut (positive-imaginary) axis —
including the PIA resonances 4σ ∈ ℤ, where the monodromy formula degenerates and
is evaluated by a pole-free reformulation.  Every numerically hazardous kernel
(continued fractions, the spheroidal eigenvalue branch, the confluent-U seeds)
is validated against algorithm-independent arbiters (bottom-up CF evaluation,
Miller recurrence, radius-certified `acb_hypgeom_u`, exact-π MPFR rebuilds), and
the regression suite (~40 testsets) pins those arbiters, not stored outputs.

Known edges:

- **Large ω costs precision, by physics**: the MST series cancel like
  ~10^(3.4·\|ω\|), so the required bits grow ~72 per unit \|ω\|
  (`suggest_mst_precision` encodes the measured envelope; ω = 16 runs routinely
  at 1280 bits in well under a second per mode).
- **Spheroidal λ basis**: the spectral basis (`l_max = 20`) is validated to
  \|aω\| ≈ 10; far beyond that it should be enlarged.
- **PN series**: Schwarzschild only (a = 0, l ≥ 1); the Kerr case needs the SWSH
  eigenvalue `λ(c)` expansion in the Wolfram `c = aω` convention.
- **Fluxes**: circular equatorial orbits only; eccentric/generic orbits need the
  full Mino-time orbit integral.
- **Numerical backend**: DP5 (5th-order) caps deep BigFloat precision at ~10⁻²⁰;
  fine for cross-checking, not for deep-precision production runs.

---

## References

- M. Sasaki and H. Tagoshi, *Analytic Black Hole Perturbation Approach to
  Gravitational Radiation*, Living Rev. Rel. **6** (2003) 6.
- S. A. Teukolsky, *Perturbations of a rotating black hole*, ApJ **185** (1973) 635.
- [`Teukolsky` paclet](https://bhptoolkit.org/Teukolsky/) (Mathematica reference
  implementation).
