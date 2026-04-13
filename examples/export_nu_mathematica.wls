#!/usr/bin/env wolframscript
(* Export ν(ω) from Teukolsky package for comparison with BHPtoolkit.jl *)

Needs["Teukolsky`"]

s = -2; l = 2; m = 2; a = 0.9;

nω = 300;
ωmin = 0.05; ωmax = 6.0;
ωgrid = Table[ωmin + (ωmax - ωmin) (i - 1)/(nω - 1), {i, 1, nω}];

Print["Computing ν for ", nω, " frequencies..."];
results = Table[
  ω = ωgrid[[i]];
  λ = SpinWeightedSpheroidalEigenvalue[s, l, m, a ω];
  ν = RenormalizedAngularMomentum[s, l, m, N[a, 20], N[ω, 20], N[λ, 20]];
  If[i == 1 || Mod[i, 30] == 0, Print["  ω = ", ω, "  ν = ", N[ν]]];
  {ω, Re[N[ν]], Im[N[ν]]}
, {i, 1, nω}];

outfile = FileNameJoin[{DirectoryName[$InputFileName], "nu_mathematica.csv"}];
Export[outfile,
  Prepend[results, {"omega", "Re_nu", "Im_nu"}],
  "CSV"];
Print["Saved to: ", outfile];
