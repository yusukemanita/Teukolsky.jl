#!/usr/bin/env wolframscript

(* Generate reference values for Rin using the Teukolsky package *)
(* Rin is normalized so B^trans = 1 (unit transmission coefficient at horizon) *)
(* i.e., TeukolskyRadial[..., "In"] = MSTRadialIn / InTrans *)

Needs["Teukolsky`"];

(* Test cases: {s, l, m, a, omega, r} *)
(* r must be outside the horizon: r > r+ = 1 + Sqrt[1 - a^2] *)
cases = {
  (* Schwarzschild, real omega *)
  {-2, 2, 2, 0.0, 0.3, 6.0},
  {-2, 2, 2, 0.0, 0.3, 10.0},
  {-2, 2, 2, 0.0, 0.3, 20.0},
  {-2, 2, 2, 0.0, 0.5, 6.0},
  {-2, 2, 2, 0.0, 0.5, 10.0},
  {-2, 2, 1, 0.0, 0.3, 6.0},
  {-2, 3, 3, 0.0, 0.3, 6.0},
  (* Schwarzschild, complex omega *)
  {-2, 2, 2, 0.0, 0.3 - 0.1*I, 6.0},
  {-2, 2, 2, 0.0, 0.3 - 0.1*I, 10.0},
  {-2, 2, 2, 0.0, 0.5 + 0.2*I, 6.0},
  {-2, 2, 2, 0.0, 0.1 - 0.3*I, 6.0},
  (* Kerr, real omega *)
  {-2, 2, 2, 0.5, 0.3, 6.0},
  {-2, 2, 2, 0.5, 0.3, 10.0},
  {-2, 2, 2, 0.9, 0.3, 6.0},
  {-2, 2, 2, 0.9, 0.5, 6.0},
  {-2, 2, 1, 0.9, 0.3, 6.0},
  {-2, 3, 3, 0.5, 0.3, 6.0},
  (* Kerr, complex omega *)
  {-2, 2, 2, 0.5, 0.3 - 0.1*I, 6.0},
  {-2, 2, 2, 0.5, 0.3 - 0.1*I, 10.0},
  {-2, 2, 2, 0.9, 0.3 - 0.1*I, 6.0},
  {-2, 2, 2, 0.9, 0.5 + 0.2*I, 6.0},
  {-2, 2, 1, 0.9, 0.3 - 0.2*I, 6.0},
  {-2, 3, 3, 0.5, 0.3 - 0.1*I, 6.0}
};

outputFile = FileNameJoin[{DirectoryName[$InputFileName], "Rin_reference.csv"}];
stream = OpenWrite[outputFile];
WriteLine[stream, "s,l,m,a,Re_omega,Im_omega,r,Re_Rin,Im_Rin"];

Do[
  {s, l, m, a, omega, r} = c;
  sol = TeukolskyRadial[s, l, m, a, N[omega, 30], Method -> "MST",
        BoundaryConditions -> "In"];
  Rin = sol[N[r, 30]];
  line = StringJoin[
    ToString[s], ",",
    ToString[l], ",",
    ToString[m], ",",
    ToString[CForm[N[a, 20]]], ",",
    ToString[CForm[N[Re[omega], 20]]], ",",
    ToString[CForm[N[Im[omega], 20]]], ",",
    ToString[CForm[N[r, 20]]], ",",
    ToString[CForm[Re[Rin]]], ",",
    ToString[CForm[Im[Rin]]]
  ];
  WriteLine[stream, line];
  Print[StringForm["s=``, l=``, m=``, a=``, \[Omega]=``, r=`` \[Rule] Rin = `` + ``i",
    s, l, m, a, omega, r, Re[Rin], Im[Rin]]],
  {c, cases}
];

Close[stream];
Print["Saved: ", outputFile];
