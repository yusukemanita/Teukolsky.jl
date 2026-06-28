# BHPtoolkit.jl

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
julia> Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")
julia> Pkg.instantiate()
julia> using BHPtoolkit
```

Dependencies: `HypergeometricFunctions`, `SpecialFunctions`, `LinearAlgebra`, `Printf`.

---

## Quick start

### Homogeneous radial solutions

```julia
using BHPtoolkit

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

## Performance vs. the Mathematica Teukolsky package

Head-to-head timing of the *same* quantities (s=−2, l=m=2) against the Mathematica
`Teukolsky` paclet, on one machine, measured sequentially (one process at a time,
warm-up excluded). The short answer: **which is faster depends on precision.**

| Operation | Float64 | BigFloat-256 (≈77 digits) |
|---|---|---|
| ν (renormalized angular momentum) | **Julia ~90×** | Mathematica ~10× |
| radial solution construction | **Julia ~200×** | **≈ parity** |
| radial evaluation at a point | **Julia ~660×** | **Julia ~4.5×** |
| point-particle energy flux | **Julia ~96×** | both correct (Julia faster) |

**Float64.** Julia is overwhelmingly faster (90–660×) — though this is the least
fair comparison: Mathematica's machine-precision MST is a self-flagged degraded mode
(`RenormalizedAngularMomentum` warns it "only works reliably with arbitrary
precision"), and the gap is partly Mathematica's per-call interpreter overhead against
Julia's sub-100-µs calls.

**BigFloat.** This is where Mathematica's tuned arbitrary-precision kernel is strong.
The renormalized angular momentum ν still favours Mathematica (~10×). But radial
**construction** is now at parity at 256-bit and Julia *overtakes* it at 512-bit
(0.92 s vs 1.59 s, ~1.7×), and radial **evaluation** stays Julia-faster at every
precision (~4.5× at 256-bit). Both packages agree on the energy flux to 16 digits
(`2.684397739103742e-5` for l=m=2, a=0, p=10).

Two recent fixes drove the BigFloat numbers (see git history):
- a Pochhammer-recurrence rewrite of the `K_ν` matching coefficient cut BigFloat
  radial construction **1.76 s → 0.53 s at 256-bit** (3.3×; 4.9× at 512-bit), since
  it eliminates ~480 full-precision Γ evaluations per solve;
- a gamma-free Pfaff fallback in the ₂F₁ evaluator fixed a `DomainError` that
  previously crashed BigFloat point-particle fluxes (real-ν / low-frequency regime).

**Bottom line:** Julia for fast Float64 sweeps and for evaluating already-built
solutions; either package for high-precision construction (Julia leads at ≥512-bit,
Mathematica leads on ν). Caveats: BigFloat-bit ↔ decimal-digit matching is nominal,
Mathematica uses adaptive guard digits, single machine / single thread.

---

## Project layout

```
src/
  BHPtoolkit.jl         module + exports
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
