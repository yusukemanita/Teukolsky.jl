#!/usr/bin/env wolframscript

Needs["Teukolsky`"]

s = -2; l = 2; m = 2; a = 0.9;

(* 実軸上のω点 *)
omegas = Join[Range[0.05, 2.5, 0.05]];

lines = {"# Mathematica G(omega) = Bref/(2i*omega*Binc) on real axis",
         "# s=-2, l=2, m=2, a=0.9",
         "# columns: omega  Re(G)  Im(G)  |G|",
         ""};

Do[
  omega = N[omegas[[i]], 32];
  mode = TeukolskyRadial[s, l, m, SetPrecision[a, 32], SetPrecision[omega, 32],
                         Method -> "MST", WorkingPrecision -> 32];
  Binc = mode["In"]["Amplitudes"]["Incidence"];
  Bref = mode["In"]["Amplitudes"]["Reflection"];
  G = Bref / (2 I omega Binc);
  AppendTo[lines,
    ToString[N[omega,  16], InputForm] <> "  " <>
    ToString[N[Re[G], 16], InputForm] <> "  " <>
    ToString[N[Im[G], 16], InputForm] <> "  " <>
    ToString[N[Abs[G],16], InputForm]],
  {i, Length[omegas]}
];

outfile = "/Users/yusuke/work/BHPtoolkit.jl/test/green_function_mathematica.txt";
Export[outfile, StringJoin[Riffle[lines, "\n"]], "Text"];
Print["Done -> ", outfile];
