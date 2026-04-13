#!/usr/bin/env wolframscript

(*
  Branch cut integral along the negative imaginary ω-axis.

    G(ω) = Bref / (2 i ω Binc)
    ψ_BC(t) = (1/2π) ∫₀^∞ ΔG(σ) e^{-σ t} dσ

  This script reproduces the same two-panel plot as
  examples/branchcut_integral.jl using the Mathematica Teukolsky package.

  Usage:
    wolframscript -file scripts/branchcut_integral.wls
    wolframscript -file scripts/branchcut_integral.wls --quick
*)

ClearAll["Global`*"];
Needs["Teukolsky`"];

repoRoot = ExpandFileName @ FileNameJoin[{DirectoryName[$InputFileName], ".."}];
figDir = FileNameJoin[{repoRoot, "figures"}];
If[! DirectoryQ[figDir], CreateDirectory[figDir, CreateIntermediateDirectories -> True]];

quickMode = MemberQ[Rest[$ScriptCommandLine], "--quick"];
If[quickMode, Print["Running in quick mode."]];

(* Parameters matching examples/branchcut_integral.jl *)
{s, l, m, a} = {-2, 2, 2, 0.9};
wp = If[quickMode, 40, 60];
delta = SetPrecision[10^-6, wp];
sigmaMax = SetPrecision[2.0, wp];
NSigma = If[quickMode, 40, 200];
NT = If[quickMode, 80, 500];
tIni = SetPrecision[10.0, wp];
tMax = SetPrecision[100.0, wp];

deltaSigma = N[sigmaMax/NSigma, wp];
sigmaGrid = N[Subdivide[deltaSigma, sigmaMax, NSigma - 1], wp];
tGrid = N[Subdivide[tIni, tMax, NT - 1], wp];

computeInMode[s_, l_, m_, a_, omega_, wp_] :=
  TeukolskyRadial[
    s, l, m,
    SetPrecision[a, wp],
    SetPrecision[omega, wp],
    Method -> "MST",
    WorkingPrecision -> wp
  ]["In"];

computeG[s_, l_, m_, a_, omega_, wp_] := Module[{trf, binc, bref},
  trf = computeInMode[s, l, m, a, omega, wp];
  binc = trf["Amplitudes"]["Incidence"];
  bref = trf["Amplitudes"]["Reflection"];
  bref/(2 I omega binc)
];

Print["Computing dG(sigma) on ", NSigma, " grid points..."];
deltaG = Table[
  Module[{sigma, omegaR, gR, omegaMirror, trfMirror, bincL, brefL, omegaL, gL, dg},
    sigma = sigmaGrid[[i]];

    (* Right side of the cut: ω_R = +δ - iσ *)
    omegaR = delta - I sigma;
    gR = computeG[s, l, m, a, omegaR, wp];

    (* Left side of the cut from the symmetry relation used in the Julia example *)
    omegaMirror = delta + I sigma;
    trfMirror = computeInMode[s, l, -m, a, omegaMirror, wp];
    bincL = Conjugate[trfMirror["Amplitudes"]["Incidence"]];
    brefL = Conjugate[trfMirror["Amplitudes"]["Reflection"]];
    omegaL = -delta - I sigma;
    gL = brefL/(2 I omegaL bincL);

    dg = gR - gL;
    If[Mod[i, Max[1, Floor[NSigma/10]]] == 0,
      Print[
        "  sigma=", ToString[NumberForm[N[sigma, 8], {6, 3}]],
        "  |dG|=", ToString[ScientificForm[N[Abs[dg], 8], 4]]
      ]
    ];
    dg
  ],
  {i, NSigma}
];

Print["Computing time-domain branch cut integral on ", NT, " time points..."];
psiBC = Table[
  If[Mod[k, Max[1, Floor[NT/10]]] == 0,
    Print["  t=", ToString[NumberForm[N[tGrid[[k]], 8], {7, 2}]]]
  ];
  (deltaSigma/(2 Pi)) Total[deltaG Exp[-sigmaGrid tGrid[[k]]]],
  {k, NT}
];

p1 = ListLinePlot[
  Transpose[{sigmaGrid, Abs[deltaG]}],
  Frame -> True,
  FrameLabel -> {"sigma", "|Delta G(sigma)|"},
  PlotStyle -> Directive[Blue, Thick],
  PlotRange -> All,
  ScalingFunctions -> {"Linear", "Log"},
  GridLines -> Automatic,
  ImageSize -> 700,
  PlotLabel -> Row[{"Branch cut discontinuity (s=", s, ", l=", l, ", m=", m, ", a=", a, ")"}]
];

p2 = ListLinePlot[
  Transpose[{tGrid, Abs[Re[psiBC]]}],
  Frame -> True,
  FrameLabel -> {"t [M]", "|Re[psi_BC(t)]|"},
  PlotStyle -> Directive[RGBColor[0.75, 0.1, 0.1], Thick],
  PlotRange -> All,
  ScalingFunctions -> {"Linear", "Log"},
  GridLines -> Automatic,
  ImageSize -> 700,
  PlotLabel -> "Branch cut contribution (power-law tail)"
];

fig = GraphicsGrid[{{p1}, {p2}}, Spacings -> {0.3, 0.3}, ImageSize -> 800];
outFile = FileNameJoin[{figDir, If[quickMode, "branchcut_integral_wolfram_quick.png", "branchcut_integral_wolfram.png"]}];
Export[outFile, fig, "PNG", ImageResolution -> 150];

Print["Saved plot to ", outFile];
outFile
