#!/usr/bin/env wolframscript

(* Generate reference values for Rup using the Teukolsky package *)
(* Rup is normalized so C^trans = 1 (unit transmission coefficient at infinity) *)

Needs["Teukolsky`"];

(* Test cases: {s, l, m, a, ω, r} — ω can be complex *)
cases = {
  (* Schwarzschild, real ω *)
  {-2, 2, 2, 0.0, 0.3, 6.0},
  {-2, 2, 2, 0.0, 0.3, 10.0},
  {-2, 2, 2, 0.0, 0.3, 20.0},
  {-2, 2, 2, 0.0, 0.5, 6.0},
  {-2, 2, 2, 0.0, 0.5, 10.0},
  {-2, 2, 1, 0.0, 0.3, 6.0},
  {-2, 3, 3, 0.0, 0.3, 6.0},
  (* Schwarzschild, complex ω *)
  {-2, 2, 2, 0.0, 0.3 - 0.1*I, 6.0},
  {-2, 2, 2, 0.0, 0.3 - 0.1*I, 10.0},
  {-2, 2, 2, 0.0, 0.5 + 0.2*I, 6.0},
  {-2, 2, 2, 0.0, 0.1 - 0.3*I, 6.0},
  (* Kerr, real ω *)
  {-2, 2, 2, 0.5, 0.3, 6.0},
  {-2, 2, 2, 0.5, 0.3, 10.0},
  {-2, 2, 2, 0.9, 0.3, 6.0},
  {-2, 2, 2, 0.9, 0.5, 6.0},
  {-2, 2, 1, 0.9, 0.3, 6.0},
  {-2, 3, 3, 0.5, 0.3, 6.0},
  (* Kerr, complex ω *)
  {-2, 2, 2, 0.5, 0.3 - 0.1*I, 6.0},
  {-2, 2, 2, 0.5, 0.3 - 0.1*I, 10.0},
  {-2, 2, 2, 0.9, 0.3 - 0.1*I, 6.0},
  {-2, 2, 2, 0.9, 0.5 + 0.2*I, 6.0},
  {-2, 2, 1, 0.9, 0.3 - 0.2*I, 6.0},
  {-2, 3, 3, 0.5, 0.3 - 0.1*I, 6.0}
};

outputFile = FileNameJoin[{DirectoryName[$InputFileName], "Rup_reference.csv"}];
stream = OpenWrite[outputFile];
WriteLine[stream, "s,l,m,a,Re_omega,Im_omega,r,Re_Rup,Im_Rup"];

Do[
  {s, l, m, a, omega, r} = c;
  sol = TeukolskyRadial[s, l, m, a, N[omega, 30], Method -> "MST",
        BoundaryConditions -> "Up"];
  Rup = sol[N[r, 30]];
  line = StringJoin[
    ToString[s], ",",
    ToString[l], ",",
    ToString[m], ",",
    ToString[CForm[N[a, 20]]], ",",
    ToString[CForm[N[Re[omega], 20]]], ",",
    ToString[CForm[N[Im[omega], 20]]], ",",
    ToString[CForm[N[r, 20]]], ",",
    ToString[CForm[Re[Rup]]], ",",
    ToString[CForm[Im[Rup]]]
  ];
  WriteLine[stream, line];
  Print[StringForm["s=``, l=``, m=``, a=``, \[Omega]=``, r=`` \[Rule] Rup = `` + ``i",
    s, l, m, a, omega, r, Re[Rup], Im[Rup]]],
  {c, cases}
];

Close[stream];
Print["Saved: ", outputFile];
