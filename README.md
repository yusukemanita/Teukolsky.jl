# Teukolsky.jl

A Julia implementation of the **Mano–Suzuki–Takasugi (MST)** formalism for the
homogeneous [Teukolsky equation](https://en.wikipedia.org/wiki/Teukolsky_equation),
together with the angular sector, Kerr geodesics, and point-particle gravitational-wave
fluxes. It is a Julia port of the core functionality of the
[Black Hole Perturbation Toolkit](https://bhptoolkit.org)'s Mathematica `Teukolsky`
package, written to run at **arbitrary (BigFloat) precision**.

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
```

### Arbitrary precision

```julia
setprecision(BigFloat, 256) do
    tr = TeukolskyRadial(-2, 2, 2, big"0.9", big"0.5")
    tr.In(big"10.0")    # full BigFloat-accurate radial value
end
```

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

(`BigFloat` also works at any precision — every kernel is type-generic — but MPFR
heap-allocates every operation, while the `:acb` chain runs zero-allocation
in-place kernels (ν monodromy, CF-peeled `f_n`, A±) plus FLINT's rigorous
`acb_hypgeom_u`: identical values, measured 2.8–7.6× faster end-to-end.
`:acb` wins; use it.)

```julia
using Teukolsky
compute_nu(-2, 2, 2, 0.7, 4.3im; backend=:acb, precision=320)
amp = compute_amplitudes(-2, 2, 2, 0.7, 4.3im; backend=:acb, precision=320)

suggest_mst_precision(4.3im)   # → (backend = :acb, bits = 320, nmax = 61)
```

### Spin-weighted spheroidal harmonics

```julia
SpinWeightedSpheroidalEigenvalue(-2, 2, 2, 0.45)       # λ = A_lm, oblateness γ = aω
SpinWeightedSpheroidalHarmonicS(-2, 2, 2, 0.9, 0.5, π/3)          # S_lm(θ) for a, ω
SpinWeightedSpheroidalHarmonicS(-2, 2, 2, 0.9, 0.5, π/3; deriv=1) # dS/dθ
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

## Validation

Every module is cross-checked against the Mathematica
[Black Hole Perturbation Toolkit](https://bhptoolkit.org) (`Teukolsky` 1.1.1,
`SpinWeightedSpheroidalHarmonics`, `KerrGeodesics`) via `wolframscript`:

- **ν / amplitudes** — match `RenormalizedAngularMomentum` on all branches;
- **spheroidal harmonics** — ~10⁻¹⁵ vs `SpinWeightedSpheroidalHarmonicS`;
- **geodesics** — ~10⁻¹³ vs `KerrGeodesics`;
- **fluxes** — energy flux to infinity ~10⁻⁹ vs `TeukolskyPointParticleMode`;
- **precision** — radial solutions self-converge to ~10⁻⁷⁴ at 512 bits, with the
  Wronskian constant to ~10⁻²¹–10⁻⁴⁴ (a reference-free precision metric).

Reference data is committed under `test/` (`*_ref.txt`) alongside the generators
(`*.wls`). Run the suite with:

```bash
julia --project -e 'using Pkg; Pkg.test()'
# or
julia --project test/runtests.jl
```

The suite has 18 testsets covering the Wolfram grid, the BigFloat precision gate,
Wronskian constancy, spheroidal harmonics, the numerical backend, elliptic
integrals, geodesics, PN series, and point-particle fluxes.

---

## Performance vs. the Mathematica Teukolsky paclet

Same quantities, same machine, both at ~**100-digit working precision**
(Julia `:acb` at 336 bits; paclet inputs at 100 digits), s=−2, l=m=2, a=0.7,
measured sequentially with warm-up excluded.  Operations: the incidence
amplitude `Binc` and the ingoing radial solution `Rin(10)`, at large \|ω\| on
the real axis and at a complex angle — the high-frequency / branch-cut regime
the MST engine is built for.

| quantity, ω | Julia `:acb` | `Teukolsky` paclet |
|---|---|---|
| `Binc`, ω = 10 | **0.03 s — 99 digits** | 0.27 s — 4.6 certified digits (~12 correct) |
| `Rin(10)`, ω = 10 | **0.08 s — 89 digits** | 0.32 s — 0 digits (magnitude off by 10⁵) |
| `Binc`, ω = 10·e^{iπ/3} | **0.06 s — 34 digits** | fails (`ComplexInfinity`) |
| `Rin(10)`, ω = 10·e^{iπ/3} | **0.02 s — 26 digits** | 0.38 s — 0 digits |

"digits" = decimal digits of agreement with a 700-bit reference (Julia) resp.
the paclet's own certified significance.  At the complex angle both engines face
the same intrinsic ~10^(3.4·\|ω\|) cancellation: the 34/26 digits delivered from
100 working digits ARE that conditioning, and more digits are bought with more
bits (`suggest_mst_precision` picks them automatically).  The paclet's adaptive
precision cannot recover past its inputs at this frequency — it self-reports
≤4.6 good digits on the real axis and fails outright at the complex angle
(`Power::infy` → `ComplexInfinity`).

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
at genuine arbitrary precision for both Schwarzschild and Kerr. Known edges:

- **PN series**: Schwarzschild only (a = 0, l ≥ 1); the Kerr case needs the SWSH
  eigenvalue `λ(c)` expansion in the Wolfram `c = aω` convention.
- **Fluxes**: circular equatorial orbits only; eccentric/generic orbits need the
  full Mino-time orbit integral.
- **Numerical backend**: DP5 (5th-order) caps deep BigFloat precision at ~10⁻²⁰;
  fine for cross-checking, not for deep-precision production runs.
- **Large ω**: ν degrades for `Mω ≳ 2` (`Mω = 4`, s=−2, l=m=2 fails even in the
  Mathematica package).

---

## References

- M. Sasaki and H. Tagoshi, *Analytic Black Hole Perturbation Approach to
  Gravitational Radiation*, Living Rev. Rel. **6** (2003) 6.
- S. A. Teukolsky, *Perturbations of a rotating black hole*, ApJ **185** (1973) 635.
- [Black Hole Perturbation Toolkit](https://bhptoolkit.org) (Mathematica reference
  implementation).
