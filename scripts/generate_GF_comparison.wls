#!/usr/bin/env wolframscript
(* Compare G(ω) = Rin(r=10, ω) / (2 i ω Binc)
   s=-2, l=2, m=2, a=0
   ω = 0.01k + 0.001i,  k=1..200   (off-axis to avoid QNM poles)
   Outputs CSV: omega_re, omega_im, Re_G, Im_G, abs_G
*)

Needs["Teukolsky`"]

s = -2; l = 2; m = 2; a = 0; r = 10;
prec = MachinePrecision;

ωs = Table[0.01 k + 10^-3 I, {k, 1, 200}];
n  = Length[ωs];

outfile = FileNameJoin[{DirectoryName[$InputFileName], "GF_wolfram.csv"}];
stream  = OpenWrite[outfile];
WriteLine[stream, "omega_re,omega_im,Re_G,Im_G,abs_G"];

Do[
  ω = ωs[[i]];
  TR   = TeukolskyRadial[s, l, m, a, SetPrecision[ω, prec], Method -> "MST"]["In"];
  Binc = TR["Amplitudes"]["Incidence"];
  G    = TR[SetPrecision[r, prec]] / (2 I ω Binc);
  WriteLine[stream,
    ToString[CForm[N[Re[ω],  16]]] <> "," <>
    ToString[CForm[N[Im[ω],  16]]] <> "," <>
    ToString[CForm[N[Re[G],  16]]] <> "," <>
    ToString[CForm[N[Im[G],  16]]] <> "," <>
    ToString[CForm[N[Abs[G], 16]]]
  ];
  If[Mod[i, 20] == 0, Print["Done: ", i, "/", n]],
  {i, 1, n}
];

Close[stream];
Print["Saved: ", outfile];
