#!/usr/bin/env wolframscript
(* Generate NumericalIntegration radial reference for B3 cross-check.
   Format per line (';'-delimited, 24-digit):
     s;l;m;a;omega;bc;r;Re(R);Im(R);Re(dR);Im(dR)
   bc: 0 = In, 1 = Up.  R, dR are the FULL TeukolskyRadial value/derivative
   with Transmission=1 normalization (default Amplitudes), via
   Method -> "NumericalIntegration". *)

Needs["Teukolsky`"];

(* Drop Mathematica's `prec precision tag, convert *^ scientific to e-form. *)
fmt1[x_] := StringReplace[
  ToString[N[x, 24], InputForm],
  {RegularExpression["`[0-9.]*"] -> "", "*^" -> "e"}];
fmt[x_] := fmt1[Re[x]] <> ";" <> fmt1[Im[x]];

cases = {
  {-2, 2, 2, 0,   0.5},
  {-2, 2, 2, 0.5, 0.5},
  { 0, 2, 0, 0.7, 0.3},
  {-2, 3, 2, 0.9, 0.5}
};

rsamples = {3, 6, 12, 30};

stream = OpenWrite["/Users/yusuke/work/BHPtoolkit.jl/test/numint_ref.txt"];

Do[
  Module[{s, l, m, a, om, wp, aP, omP, R, lines},
    {s, l, m, a, om} = c;
    wp = 32;
    aP  = SetPrecision[a, wp];
    omP = SetPrecision[om, wp];
    Do[
      Module[{bc, bcCode, Rf, r, val, dval},
        {bc, bcCode} = bcpair;
        Rf = TeukolskyRadial[s, l, m, aP, omP,
               Method -> "NumericalIntegration",
               "BoundaryConditions" -> bc,
               WorkingPrecision -> wp];
        Do[
          val  = Rf[SetPrecision[r, wp]];
          dval = Rf'[SetPrecision[r, wp]];
          WriteString[stream,
            ToString[s] <> ";" <> ToString[l] <> ";" <> ToString[m] <> ";" <>
            ToString[a] <> ";" <> ToString[om] <> ";" <> ToString[bcCode] <> ";" <>
            ToString[r] <> ";" <> fmt[val] <> ";" <> fmt[dval] <> "\n"];
        , {r, rsamples}];
      ],
    {bcpair, {{"In", 0}, {"Up", 1}}}];
  ],
{c, cases}];

Close[stream];
Print["done"];
