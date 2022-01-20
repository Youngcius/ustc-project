(* ::Package:: *)

BeginPackage["ConstantPotential`"]
Clear[WaveOdd,WaveEven,EnergyOdd,EnergyEven,EnergySpectrum]
WaveOdd::usage="\:6c42\:89e3\:4e0d\:540c\:5bbd\:5ea6\:548c\:9ad8\:5ea6\:52bf\:9631\:7684\:6ce2\:51fd\:6570,\:5947\:5b87\:79f0"
WaveEven::usage="\:6c42\:89e3\:4e0d\:540c\:5bbd\:5ea6\:548c\:9ad8\:5ea6\:52bf\:9631\:7684\:6ce2\:51fd\:6570,\:5076\:5b87\:79f0"
EnergySpectrum::usage="\:6c42\:89e3\:80fd\:91cf\:8c31"
EnergyOdd::usage="\:5947\:5b87\:79f0\:7684\:80fd\:91cf\:8c31"
EnergyEven::usage="\:5076\:5b87\:79f0\:7684\:80fd\:91cf\:8c31"
(*\[HBar]=1.05*10^(-34);
x::usage="\:4f4d\:7f6e"
m::usage="\:7c92\:5b50\:8d28\:91cf"
height::usange="\:52bf\:9631\:9ad8\:5ea6"
width::usage="\:52bf\:9631\:5bbd\:5ea6"
order::usage="\:80fd\:7ea7\:6570"*)



Begin["`Private`"]
EnergyEven[height_,width_,m_]:=
	Module[{a,\[Xi],\[Eta],\[HBar],r,num,roots,energy},
		a=width/2;
		r=Sqrt[(2m*height*a^2)/\[HBar]^2];
		\[HBar]=1.05*10^(-34);
		
		solList=List[ToRules[N[Reduce[Rationalize[{Sqrt[r^2-\[Xi]^2]-\[Xi] Tan[\[Xi]]==0,\[Xi]>0,\[Xi]<r}],\[Xi],Reals]]]];
		roots=Table[\[Xi]/.solList[[i]],{i,1,Length[solList]}];
		energy=(roots^2 \[HBar]^2)/(2m*a^2)-height
	]
EnergyOdd[height_,width_,m_]:=
	Module[{a,\[Xi],\[Eta],\[HBar],r,num,roots,energy},
		a=width/2;
		r=Sqrt[(2m*height*a^2)/\[HBar]^2];
		\[HBar]=1.05*10^(-34);
		
		solList=List[ToRules[N[Reduce[Rationalize[{Sqrt[r^2-\[Xi]^2]+\[Xi] Cot[\[Xi]]==0,\[Xi]>0,\[Xi]<r}],\[Xi],Reals]]]];
		roots=Table[\[Xi]/.solList[[i]],{i,1,Length[solList]}];
		
		energy=(roots^2 \[HBar]^2)/(2m*a^2)-height
	]
EnergySpectrum[height_,width_,m_]:=Sort[Join[EnergyOdd[height,width,m],EnergyEven[height,width,m]]];

WaveEven[x_,height_,width_,m_,energy_]:=
	Module[{a,k,\[Beta],B,\[HBar],unit,\[Psi]},
		a=width/2;
		\[HBar]=1.05*10^(-34);
		k=Sqrt[2m(energy+height)]/\[HBar];
		\[Beta]=Sqrt[-2m*energy]/\[HBar];
		B=Exp[-\[Beta]*a]/Cos[k*a];
		unit=NIntegrate[Exp[\[Beta]*x]^2,{x,-10a,-a}]+NIntegrate[B^2*Cos[k*x]^2,{x,-a,a}]+NIntegrate[Exp[-\[Beta]*x]^2,{x,a,10a}];
		\[Psi]=\!\(\*
TagBox[GridBox[{
{"\[Piecewise]", GridBox[{
{
RowBox[{
RowBox[{"Exp", "[", 
RowBox[{"\[Beta]", "*", "x"}], "]"}], "/", 
SqrtBox["unit"]}], 
RowBox[{"x", "<", 
RowBox[{"-", "a"}]}]},
{
RowBox[{"B", "*", 
RowBox[{
RowBox[{"Cos", "[", 
RowBox[{"k", "*", "x"}], "]"}], "/", 
SqrtBox["unit"]}]}], 
RowBox[{
RowBox[{"Abs", "[", "x", "]"}], "<=", "a"}]},
{
RowBox[{
RowBox[{"Exp", "[", 
RowBox[{
RowBox[{"-", "\[Beta]"}], "*", "x"}], "]"}], "/", 
SqrtBox["unit"]}], 
RowBox[{"x", ">", "a"}]}
},
AllowedDimensions->{2, Automatic},
Editable->True,
GridBoxAlignment->{"Columns" -> {{Left}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}},
GridBoxItemSize->{"Columns" -> {{Automatic}}, "ColumnsIndexed" -> {}, "Rows" -> {{1.}}, "RowsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.84]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}},
Selectable->True]}
},
GridBoxAlignment->{"Columns" -> {{Left}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}},
GridBoxItemSize->{"Columns" -> {{Automatic}}, "ColumnsIndexed" -> {}, "Rows" -> {{1.}}, "RowsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.35]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}}],
"Piecewise",
DeleteWithContents->True,
Editable->False,
SelectWithContents->True,
Selectable->False,
StripWrapperBoxes->True]\)
	]
(*
WaveOdd[x_,height_,width_,m_,order_]:=
	Module[{a,k,\[Beta],\[HBar],B,energy,unit,\[Psi]},
		a=width/2;
		energy=EnergyOdd[height,width,m][[order]];
		\[HBar]=1.05*10^(-34);
		k=Sqrt[2m(energy+height)]/\[HBar];
		\[Beta]=Sqrt[-2m*energy]/\[HBar];
		B=Exp[-\[Beta]*a]/(-Sin[k*a]);
		unit=NIntegrate[Rational[Exp[\[Beta]*x]^2],{x,-10a,-a}]+NIntegrate[Rational[B^2*Sin[k*x]^2],{x,-a,a}]+NIntegrate[Rational[Exp[-\[Beta]*x]^2],{x,a,10a}];
		\[Psi]:=\[Piecewise]	Exp[\[Beta]*x]/Sqrt[unit]	x<-a
B*Sin[k*x]/Sqrt[unit]	Abs[x]<a
-Exp[-\[Beta]*x]/Sqrt[unit]	x>a


		

	]

*)
(*u[x_]:=WaveOdd[x,V0,a,\[Mu],1]*)


WaveOdd[x_,height_,width_,m_,energy_]:=
	Module[{a,k,\[Beta],\[HBar],B,unit,\[Psi]},
		a=width/2;
		\[HBar]=1.05*10^(-34);
		k=Sqrt[2m(energy+height)]/\[HBar];
		\[Beta]=Sqrt[-2m*energy]/\[HBar];
		B=Exp[-\[Beta]*a]/(-Sin[k*a]);
		unit=NIntegrate[Exp[\[Beta]*x]^2,{x,-10a,-a}]+NIntegrate[B^2*Sin[k*x]^2,{x,-a,a}]+NIntegrate[Exp[-\[Beta]*x]^2,{x,a,10a}];
		\[Psi]=\!\(\*
TagBox[GridBox[{
{"\[Piecewise]", GridBox[{
{
RowBox[{
RowBox[{"Exp", "[", 
RowBox[{"\[Beta]", "*", "x"}], "]"}], "/", 
SqrtBox["unit"]}], 
RowBox[{"x", "<", 
RowBox[{"-", "a"}]}]},
{
RowBox[{"B", "*", 
RowBox[{
RowBox[{"Sin", "[", 
RowBox[{"k", "*", "x"}], "]"}], "/", 
SqrtBox["unit"]}]}], 
RowBox[{
RowBox[{"Abs", "[", "x", "]"}], "<=", "a"}]},
{
RowBox[{
RowBox[{"-", 
RowBox[{"Exp", "[", 
RowBox[{
RowBox[{"-", "\[Beta]"}], "*", "x"}], "]"}]}], "/", 
SqrtBox["unit"]}], 
RowBox[{"x", ">", "a"}]}
},
AllowedDimensions->{2, Automatic},
Editable->True,
GridBoxAlignment->{"Columns" -> {{Left}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}},
GridBoxItemSize->{"Columns" -> {{Automatic}}, "ColumnsIndexed" -> {}, "Rows" -> {{1.}}, "RowsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.84]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}},
Selectable->True]}
},
GridBoxAlignment->{"Columns" -> {{Left}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}},
GridBoxItemSize->{"Columns" -> {{Automatic}}, "ColumnsIndexed" -> {}, "Rows" -> {{1.}}, "RowsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.35]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}}],
"Piecewise",
DeleteWithContents->True,
Editable->False,
SelectWithContents->True,
Selectable->False,
StripWrapperBoxes->True]\)
		

	]




End[]




EndPackage[]








