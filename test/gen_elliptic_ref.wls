#!/usr/bin/env wolframscript
(* Generate elliptic-integral / Jacobi reference values for the Carlson-form
   module cross-check.  All Wolfram functions use the PARAMETER m = k².

   Output (';'-delimited, 30-digit, written to test/elliptic_ref.txt):
     KE;m;EllipticK[m];EllipticE[m]
     F;phi;m;EllipticF[phi,m]
     EI;phi;m;EllipticE[phi,m]
     PIC;n;m;EllipticPi[n,m]
     PII;n;phi;m;EllipticPi[n,phi,m]
     SN;u;m;JacobiSN[u,m]
     AM;u;m;JacobiAmplitude[u,m]
*)

fmt[x_] := StringReplace[
  ToString[N[x, 30], InputForm],
  {RegularExpression["`[0-9.]*"] -> "", "*^" -> "e"}];

stream = OpenWrite["/Users/yusuke/work/Teukolsky.jl/test/elliptic_ref.txt"];

(* sample parameters: include m near 0 and near 1, negative n, phi inside and
   outside [-Pi/2, Pi/2] *)
ms   = {1/100, 1/4, 1/2, 3/4, 99/100, 999/1000};
phis = {1/5, 3/5, 1, Pi/2 - 1/10, 5/4, 2, 3};
ns   = {-3/2, -1/2, 1/4, 1/2, 3/4};
us   = {1/4, 1, 2, 7/2, 5, 8};

Do[ WriteString[stream,
      "KE;" <> fmt[m] <> ";" <> fmt[EllipticK[m]] <> ";" <> fmt[EllipticE[m]] <> "\n"],
    {m, ms}];

Do[ WriteString[stream,
      "F;" <> fmt[phi] <> ";" <> fmt[m] <> ";" <> fmt[EllipticF[phi, m]] <> "\n"],
    {m, ms}, {phi, phis}];

Do[ WriteString[stream,
      "EI;" <> fmt[phi] <> ";" <> fmt[m] <> ";" <> fmt[EllipticE[phi, m]] <> "\n"],
    {m, ms}, {phi, phis}];

Do[ WriteString[stream,
      "PIC;" <> fmt[n] <> ";" <> fmt[m] <> ";" <> fmt[EllipticPi[n, m]] <> "\n"],
    {m, ms}, {n, ns}];

Do[ WriteString[stream,
      "PII;" <> fmt[n] <> ";" <> fmt[phi] <> ";" <> fmt[m] <> ";" <>
        fmt[EllipticPi[n, phi, m]] <> "\n"],
    {m, ms}, {n, ns}, {phi, phis}];

Do[ WriteString[stream,
      "SN;" <> fmt[u] <> ";" <> fmt[m] <> ";" <> fmt[JacobiSN[u, m]] <> "\n"],
    {m, ms}, {u, us}];

Do[ WriteString[stream,
      "AM;" <> fmt[u] <> ";" <> fmt[m] <> ";" <> fmt[JacobiAmplitude[u, m]] <> "\n"],
    {m, ms}, {u, us}];

Close[stream];
Print["done"];
