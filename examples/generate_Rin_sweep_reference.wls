#!/usr/bin/env wolframscript
(* Generate Rin(r=10) vs omega sweep for comparison with BHPtoolkit *)
(* s=-2, l=2, m=2, a=0.0, r=10M, omega in [0.05, 1.0] *)

Needs["Teukolsky`"];

s = -2; l = 2; m = 2; a = 0.0; r = 10.0;
nOmega = 80;
omegaGrid = Table[0.05 + (1.0 - 0.05) * (i - 1) / (nOmega - 1), {i, 1, nOmega}];

outputFile = FileNameJoin[{DirectoryName[$InputFileName], "Rin_sweep_reference.csv"}];
stream = OpenWrite[outputFile];
WriteLine[stream, "omega,Re_Rin,Im_Rin"];

Do[
  omega = omegaGrid[[i]];
  sol = TeukolskyRadial[s, l, m, a, N[omega, 30], Method -> "MST",
        BoundaryConditions -> "In"];
  Rin = sol[N[r, 30]];
  line = StringJoin[
    ToString[CForm[N[omega, 20]]], ",",
    ToString[CForm[N[Re[Rin], 20]]], ",",
    ToString[CForm[N[Im[Rin], 20]]]
  ];
  WriteLine[stream, line];
  If[Mod[i, 10] == 0, Print["Done: ", i, "/", nOmega]],
  {i, 1, nOmega}
];

Close[stream];
Print["Saved: ", outputFile];
