#!/usr/bin/env wolframscript
(*
  Compute ΔG⁺(σ) on the positive imaginary axis using the Teukolsky package.
  G(ω) = Bref / (2iω Binc)
  ΔG⁺(σ) = G(δ+iσ) - G(-δ+iσ)

  Uses arbitrary precision (wp=50) to avoid Float64 precision loss at large σ.
  Output: CSV with columns σ, Re[ΔG], Im[ΔG], Re[ν], Im[ν]
*)

Needs["Teukolsky`"];

wp = 50;
s = -2; l = 2; m = 2;
a = SetPrecision[9/10, wp];
delta = SetPrecision[10^-6, wp];

sigmaGrid = N[Exp[Range[Log[1/100], Log[30], (Log[30] - Log[1/100])/199]], wp];

outfile = FileNameJoin[{DirectoryName[$InputFileName], "DG_pos_wolfram.csv"}];
stream = OpenWrite[outfile];
WriteLine[stream, "sigma,ReDG,ImDG,Renu,Imnu"];

Do[
  sigma = SetPrecision[sigmaGrid[[i]], wp];
  omegaR = SetPrecision[delta + I sigma, wp];
  omegaL = SetPrecision[-delta + I sigma, wp];

  (* Right side: ω = δ + iσ *)
  solR = Quiet[TeukolskyRadial[s, l, m, a, omegaR,
    Method -> "MST", WorkingPrecision -> wp]];
  ampR = solR["In"]["Amplitudes"];
  BincR = ampR["Incidence"];
  BrefR = ampR["Reflection"];
  nuR = solR["In"]["RenormalizedAngularMomentum"];
  GR = BrefR / (2 I omegaR BincR);

  (* Left side: ω = -δ + iσ, use symmetry *)
  solL = Quiet[TeukolskyRadial[s, l, -m, a, Conjugate[omegaL],
    Method -> "MST", WorkingPrecision -> wp]];
  ampL = solL["In"]["Amplitudes"];
  BincL = ampL["Incidence"];
  BrefL = ampL["Reflection"];
  GL = Conjugate[BrefL] / (2 I omegaL Conjugate[BincL]);

  DG = GR - GL;

  WriteLine[stream, StringJoin[
    ToString[CForm[N[sigma, 20]]], ",",
    ToString[CForm[N[Re[DG], 20]]], ",",
    ToString[CForm[N[Im[DG], 20]]], ",",
    ToString[CForm[N[Re[nuR], 20]]], ",",
    ToString[CForm[N[Im[nuR], 20]]]
  ]];

  If[Mod[i, 20] == 0,
    Print["i=", i, "/", Length[sigmaGrid], "  sigma=", N[sigma,5],
      "  |DG|=", ScientificForm[Abs[DG] // N, 4],
      "  nu=", N[nuR, 6]]];
, {i, 1, Length[sigmaGrid]}];

Close[stream];
Print["Saved: ", outfile];
