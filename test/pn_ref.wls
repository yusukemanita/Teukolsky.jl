Needs["Teukolsky`"];
Needs["Teukolsky`PN`"];
nu  = Teukolsky`PN`Private`\[Nu];
aa  = Teukolsky`PN`Private`a;

str = OpenWrite["/tmp/pn_ref_raw.txt"];
emit[s_,l_,m_,ord_] := Block[{res},
  res = Block[{Print}, Teukolsky`PN`Private`KerrMSTSeries[s,l,m,ord]];
  WriteString[str, "==== MODE s=",s," l=",l," m=",m," ExpOrder=",ord," ====\n"];
  WriteString[str, "NU = ", ToString[Normal[res[nu]], InputForm], "\n"];
  Do[ If[KeyExistsQ[res, aa[n]],
       WriteString[str, "A[",n,"] = ", ToString[Normal[res[aa[n]]], InputForm], "\n"]],
     {n,-ord-2,ord}];
];
emit[-2,2,2,4];
emit[-2,2,0,4];
emit[0,0,0,4];
emit[-1,1,1,4];
Close[str];
Print["DONE"];
