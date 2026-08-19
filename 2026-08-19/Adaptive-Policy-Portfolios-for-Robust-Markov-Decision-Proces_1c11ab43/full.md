# Adaptive Policy Portfolios for Robust Markov Decision Processes

Kasper Engelen<sup>1</sup>, Sebastian Junges<sup>2</sup>, Guillermo A. Pérez<sup>1</sup>, Marnix Suilen<sup>1</sup>

<sup>1</sup>U<sub>n</sub>i<sub>vers</sub>it<sub>y</sub> <sub>o</sub>f A<sub>n</sub>t<sub>werp,</sub> B<sub>e</sub>l<sub>g</sub>i<sub>um</sub>

<sup>2</sup>Radboud University, Nijmegen, The Netherlands

{kasper.engelen,guillermo.perez,marnix.suilen}@uantwerpen.be, sebastian.junges@ru.nl

## Abstract

R<sub>o</sub>b<sub>us</sub>t M<sub>ar</sub>k<sub>ov</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on processes op</sub>ti<sub>m</sub>i<sub>ze one po</sub>li<sub>cy aga</sub>i<sub>ns</sub>t <sub>a se</sub>t <sub>o</sub>f <sub>p</sub>l<sub>aus</sub>ibl<sub>e</sub> t<sub>rans</sub>iti<sub>on</sub> f<sub>unc</sub>ti<sub>ons.</sub> Thi<sub>s can</sub> b<sub>e conserva-</sub> ti<sub>ve</sub> <sub>w</sub>h<sub>en</sub> th<sub>e</sub> <sub>un</sub>k<sub>nown</sub> d<sub>ynam</sub>i<sub>cs</sub> <sub>are</sub> fi<sub>xe</sub>d <sub>an</sub>d b<sub>ecome</sub> <sub>par-</sub> tially identifiable after deployment. We study adaptive policy portfolios: finite sets of memoryless randomized policies synth<sub>es</sub>i<sub>ze</sub>d <sub>o</sub>fli<sub>ne an</sub>d <sub>pa</sub>i<sub>re</sub>d <sub>w</sub>ith <sub>a</sub> li<sub>g</sub>ht<sub>we</sub>i<sub>g</sub>ht <sub>on</sub>li<sub>ne se</sub>l<sub>ec</sub>t<sub>or.</sub> R<sub>o</sub>b<sub>us</sub>t <sub>regre</sub>t i<sub>s a na</sub>t<sub>ura</sub>l <sub>measure o</sub>f <sub>por</sub>tf<sub>o</sub>li<sub>o qua</sub>lit<sub>y:</sub> f<sub>or</sub> <sub>eac</sub>h <sub>p</sub>l<sub>aus</sub>ibl<sub>e env</sub>i<sub>ronmen</sub>t<sub>,</sub> it <sub>measures</sub> th<sub>e</sub> l<sub>oss o</sub>f th<sub>e</sub> b<sub>es</sub>t <sub>por</sub>tf<sub>o</sub>li<sub>o mem</sub>b<sub>er re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e po</sub>li<sub>cy</sub> th<sub>a</sub>t <sub>wou</sub>ld h<sub>ave</sub> b<sub>een</sub> <sub>op</sub>ti<sub>ma</sub>l h<sub>a</sub>d th<sub>a</sub>t <sub>env</sub>i<sub>ronmen</sub>t b<sub>een</sub> k<sub>nown.</sub> R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d <sub>regre</sub>t <sub>o</sub>b<sub>-</sub> jectives were studied b<sub>y</sub> Ghavamzadeh et al. (2016) with an <sub>emp</sub>h<sub>as</sub>i<sub>s on approx</sub>i<sub>ma</sub>ti<sub>ons an</sub>d <sub>re</sub>l<sub>axa</sub>ti<sub>ons</sub> f<sub>or sa</sub>f<sub>e po</sub>li<sub>cy</sub> i<sub>mprovemen</sub>t<sub>.</sub> W<sub>e g</sub>i<sub>ve a comp</sub>l<sub>ex</sub>it<sub>y-</sub>th<sub>eore</sub>ti<sub>c accoun</sub>t <sub>o</sub>f <sub>por</sub>t<sub>-</sub> f<sub>o</sub>li<sub>o cer</sub>tifi<sub>ca</sub>ti<sub>on an</sub>d <sub>syn</sub>th<sub>es</sub>i<sub>s.</sub> C<sub>er</sub>tif<sub>y</sub>i<sub>ng a g</sub>i<sub>ven por</sub>tf<sub>o</sub>li<sub>o</sub> is ∀R-com<sub>p</sub>lete alread<sub>y</sub> for deterministic <sub>p</sub>ortfolios in ac<sub>y</sub>clic (s, a)-rectangular RMDPs. Synthesizing a portfolio of unarybounded size is ∃∀R-com<sub>p</sub>lete for <sub>g</sub>eneral rational <sub>p</sub>ol<sub>y</sub>to<sub>p</sub>es, <sub>even</sub> <sub>w</sub>ith fi<sub>xe</sub>d di<sub>scoun</sub>t <sub>an</sub>d <sub>acyc</sub>li<sub>c</sub> d<sub>ynam</sub>i<sub>cs.</sub> Th<sub>e</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e-</sub> <sub>po</sub>li<sub>cy case</sub> i<sub>s a</sub>l<sub>rea</sub>d<sub>y</sub> h<sub>ar</sub>d<sub>,</sub> b<sub>o</sub>th <sub>com</sub>bi<sub>na</sub>t<sub>or</sub>i<sub>a</sub>ll<sub>y an</sub>d <sub>a</sub>l<sub>ge-</sub> b<sub>ra</sub>i<sub>ca</sub>ll<sub>y.</sub> Fi<sub>na</sub>ll<sub>y, we presen</sub>t <sub>an o</sub>fli<sub>ne por</sub>tf<sub>o</sub>li<sub>o cons</sub>t<sub>ruc</sub>ti<sub>on</sub> th<sub>a</sub>t i<sub>s</sub> <sub>amena</sub>bl<sub>e</sub> t<sub>o</sub> <sub>run</sub>ti<sub>me</sub> <sub>spec</sub>i<sub>a</sub>li<sub>za</sub>ti<sub>on.</sub>

## 1 Introduction

Markov decision rocesses (MDPs) are the standard formali<sub>sm</sub> f<sub>or sequen</sub>ti<sub>a</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>on-ma</sub>ki<sub>ng un</sub>d<sub>er uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y,</sub> b<sub>u</sub>t th<sub>ey requ</sub>i<sub>re prec</sub>i<sub>se</sub> k<sub>now</sub>l<sub>e</sub>d<sub>ge o</sub>f t<sub>rans</sub>iti<sub>on pro</sub>b<sub>a</sub>biliti<sub>es.</sub> Robust MDPs (RMDPs, for short) (Wiesemann, Kuhn, and Rustem 2013) relax this requirement b<sub>y</sub> o<sub>p</sub>timizin<sub>g</sub> a<sub>g</sub>ainst <sub>an uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y se</sub>t <sub>o</sub>f <sub>poss</sub>ibl<sub>e</sub> t<sub>rans</sub>iti<sub>on</sub> f<sub>unc</sub>ti<sub>ons.</sub> A <sub>ro</sub>b<sub>us</sub>t <sub>po</sub>li<sub>cy</sub> i<sub>s op</sub>ti<sub>m</sub>i<sub>ze</sub>d <sub>aga</sub>i<sub>ns</sub>t th<sub>e wors</sub>t t<sub>rans</sub>iti<sub>on</sub> d<sub>ynam</sub>i<sub>cs</sub> i<sub>n</sub> th<sub>a</sub>t <sub>se</sub>t<sub>.</sub> C<sub>onsequen</sub>tl<sub>y, ro</sub>b<sub>us</sub>t <sub>po</sub>li<sub>c</sub>i<sub>es can</sub> b<sub>e over</sub>l<sub>y conser-</sub> <sub>va</sub>ti<sub>ve</sub> b<sub>ecause</sub> th<sub>e</sub>i<sub>r</sub> b<sub>e</sub>h<sub>av</sub>i<sub>or</sub> i<sub>s</sub> d<sub>om</sub>i<sub>na</sub>t<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> h<sub>ar</sub>d<sub>es</sub>t <sub>env</sub>i<sub>ronmen</sub>t<sub>s, even w</sub>h<sub>en</sub> th<sub>ose env</sub>i<sub>ronmen</sub>t<sub>s are un</sub>lik<sub>e</sub>l<sub>y or</sub> <sub>qu</sub>i<sub>c</sub>kl<sub>y ru</sub>l<sub>e</sub>d <sub>ou</sub>t b<sub>y o</sub>b<sub>serva</sub>ti<sub>on.</sub>

Robust regret (Ahmed et al. 2013; Rigter, Lacerda, and Hawes 2021) ofers an alternative. Instead of maximizin<sub>g</sub> <sub>va</sub>l<sub>ue un</sub>d<sub>er</sub> th<sub>e wors</sub>t t<sub>rans</sub>iti<sub>on</sub> f<sub>unc</sub>ti<sub>on,</sub> it <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub> th<sub>e</sub> l<sub>arges</sub>t <sub>va</sub>l<sub>ue</sub> <sub>s</sub>h<sub>or</sub>tf<sub>a</sub>ll <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e</sub> <sub>po</sub>li<sub>cy</sub> th<sub>a</sub>t <sub>wou</sub>ld h<sub>ave</sub> b<sub>een op</sub>ti<sub>ma</sub>l h<sub>a</sub>d th<sub>e</sub> t<sub>rue</sub> t<sub>rans</sub>iti<sub>on</sub> f<sub>unc</sub>ti<sub>on</sub> b<sub>een</sub> k<sub>nown.</sub> Thi<sub>s</sub> <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> <sub>a</sub> <sub>po</sub>li<sub>cy</sub> <sub>w</sub>ith <sub>un</sub>if<sub>orm</sub>l<sub>y</sub> <sub>sma</sub>ll <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> l<sub>oss.</sub> It <sub>s</sub>till <sub>comm</sub>it<sub>s,</sub> h<sub>owever,</sub> t<sub>o</sub> <sub>one</sub> <sub>po</sub>li<sub>cy</sub> f<sub>or</sub> <sub>every</sub> <sub>poss</sub>ibl<sub>e</sub> t<sub>rans</sub>iti<sub>on</sub> f<sub>unc</sub>ti<sub>on</sub> <sub>an</sub>d <sub>canno</sub>t <sub>exp</sub>l<sub>o</sub>it <sub>ev</sub>id<sub>ence</sub> <sub>ga</sub>th<sub>ere</sub>d <sub>on</sub>li<sub>ne</sub> <sub>a</sub>b<sub>ou</sub>t <sub>w</sub>hi<sub>c</sub>h <sub>mo</sub>d<sub>e</sub>l <sub>governs</sub> th<sub>e env</sub>i<sub>ronmen</sub>t<sub>.</sub>

![](images/033677946919535c49eacc263311ed1f8a99cf41ee3c51b7306e2c47c36c79c0.jpg)  
Fi<sub>gure</sub> 1<sub>:</sub> P<sub>or</sub>tf<sub>o</sub>li<sub>o regre</sub>t i<sub>n</sub> th<sub>e ac</sub>t<sub>ua</sub>t<sub>or examp</sub>l<sub>e.</sub>

We propose adaptive policy portfolios. Ofline, we com-<sub>pu</sub>t<sub>e a se</sub>t <sub>o</sub>f <sub>po</sub>li<sub>c</sub>i<sub>es</sub> t<sub>a</sub>il<sub>ore</sub>d t<sub>o</sub> dif<sub>eren</sub>t <sub>reg</sub>i<sub>ons o</sub>f th<sub>e un-</sub> <sub>cer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y se</sub>t<sub>.</sub> O<sub>n</sub>li<sub>ne, accumu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>ev</sub>id<sub>ence</sub> i<sub>s use</sub>d t<sub>o se</sub>l<sub>ec</sub>t <sub>among</sub> th<sub>ese a</sub>l<sub>rea</sub>d<sub>y va</sub>lid<sub>a</sub>t<sub>e</sub>d <sub>po</sub>li<sub>c</sub>i<sub>es.</sub> Th<sub>e por</sub>tf<sub>o</sub>li<sub>o</sub> i<sub>s syn-</sub> th<sub>es</sub>i<sub>ze</sub>d b<sub>y m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng ro</sub>b<sub>us</sub>t <sub>regre</sub>t<sub>.</sub> W<sub>e</sub> f<sub>ocus on por</sub>tf<sub>o</sub>li<sub>os</sub> <sub>o</sub>f <sub>memory</sub>l<sub>ess</sub> <sub>po</sub>li<sub>c</sub>i<sub>es,</sub> th<sub>e</sub> <sub>s</sub>i<sub>mp</sub>l<sub>es</sub>t <sub>prac</sub>ti<sub>ca</sub>ll<sub>y</sub> i<sub>n</sub>t<sub>eres</sub>ti<sub>ng</sub> <sub>po</sub>li<sub>c</sub>i<sub>es</sub> f<sub>or</sub> <sub>sequen</sub>ti<sub>a</sub>l d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> <sub>ma</sub>ki<sub>ng.</sub>

A<sub>s a mo</sub>ti<sub>va</sub>ti<sub>ng examp</sub>l<sub>e, cons</sub>id<sub>er an ac</sub>t<sub>ua</sub>t<sub>or w</sub>h<sub>ose</sub> t<sub>rue</sub> <sub>s</sub>li<sub>p</sub> l<sub>eve</sub>l i<sub>s an un</sub>k<sub>nown parame</sub>t<sub>er</sub> $p \in [ 0 , 1 ]$ <sub>.</sub> Vi<sub>ewe</sub>d <sub>as a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e-s</sub>t<sub>a</sub>t<sub>e</sub> RMDP<sub>, eac</sub>h <sub>ac</sub>ti<sub>on</sub> i<sub>s a ca</sub>lib<sub>ra</sub>ti<sub>on</sub> $c \in C =$ $\{ 0 , ^ { \infty } \mathrm { { \bar { \alpha } } ^ { } \mathrm { { s } , } \{ \dots \mathrm { { } , 1 \} } } $ <sub>, an</sub>d th<sub>e un</sub>k<sub>nown</sub> $p$ <sub>se</sub>t<sub>s</sub> th<sub>e success pro</sub>b<sub>a</sub>bil<sub>-</sub> <sup>i</sup>t<sub>y</sub> $1 - ( p - \dot { c } ) ^ { 2 }$ <sub>o</sub>f <sub>reac</sub>hi<sub>ng an a</sub>b<sub>sor</sub>bi<sub>ng success s</sub>t<sub>a</sub>t<sub>e.</sub> Th<sub>e</sub> regret of a calibration is its excess squared distance to p over the closest calibration in C. A single robust-regret polic<sub>y</sub> <sub>c</sub>h<sub>ooses</sub> $c = 1 / 2$ <sub>an</sub>d h<sub>as wors</sub>t<sub>-case regre</sub>t $^ 1 / _ { 4 }$ <sub>.</sub> Th<sub>e por</sub>tf<sub>o-</sub> li<sub>o</sub> $\{ 0 , { } ^ { 1 } / 2 , { } 1 \}$ <sub>re</sub>d<sub>uces</sub> thi<sub>s</sub> t<sub>o</sub> $^ { 1 / 1 6 } \mathrm { ; }$ <sub>, w</sub>hil<sub>e</sub> $\{ 0 , { ^ { 1 } \mathrm { / 4 } } , { ^ { 1 } \mathrm { / 2 } } , { ^ { 3 } \mathrm { / 4 } } , 1 \}$ <sub>re</sub>d<sub>uces</sub> it t<sub>o</sub> $^ 1 / 6 4$ <sub>.</sub> I<sub>ncreas</sub>i<sub>ng</sub> th<sub>e por</sub>tf<sub>o</sub>li<sub>o s</sub>i<sub>ze prov</sub>id<sub>es a</sub> <sub>con</sub>t<sub>ro</sub>ll<sub>e</sub>d f<sub>orm o</sub>f <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on w</sub>hil<sub>e on</sub>li<sub>ne se</sub>l<sub>ec</sub>ti<sub>on rema</sub>i<sub>ns</sub> <sub>over a</sub> fi<sub>n</sub>it<sub>e se</sub>t<sub>.</sub> Fi<sub>gure</sub> 1 <sub>p</sub>l<sub>o</sub>t<sub>s</sub> th<sub>ese regre</sub>t <sub>curves.</sub>

I<sub>n</sub> thi<sub>s</sub> <sub>wor</sub>k<sub>,</sub> <sub>we</sub> <sub>s</sub>t<sub>u</sub>d<sub>y</sub> th<sub>e</sub> <sub>compu</sub>t<sub>a</sub>ti<sub>ona</sub>l <sub>comp</sub>l<sub>ex</sub>it<sub>y</sub> <sub>o</sub>f <sub>syn</sub>th<sub>es</sub>i<sub>z</sub>i<sub>ng m</sub>i<sub>n</sub>i<sub>ma</sub>l<sub>-regre</sub>t <sub>po</sub>li<sub>cy por</sub>tf<sub>o</sub>li<sub>os an</sub>d th<sub>e cer-</sub> tifi<sub>ca</sub>ti<sub>on an</sub>d <sub>compar</sub>i<sub>son su</sub>b<sub>pro</sub>bl<sub>ems</sub> th<sub>a</sub>t <sub>ar</sub>i<sub>se a</sub>l<sub>ong</sub> th<sub>e</sub> <sub>way.</sub> T<sub>a</sub>bl<sub>e</sub> 1 <sub>summar</sub>i<sub>zes</sub> <sub>our</sub> <sub>ma</sub>i<sub>n</sub> <sub>resu</sub>lt<sub>s:</sub> th<sub>e</sub> <sub>ro</sub>b<sub>us</sub>t<sub>-regre</sub>t <sub>pro</sub>bl<sub>ems are</sub> h<sub>ar</sub>d f<sub>or</sub> dif<sub>eren</sub>t l<sub>eve</sub>l<sub>s o</sub>f th<sub>e</sub> hi<sub>erarc</sub>h<sub>y o</sub>f the theor<sub>y</sub> of the reals (Schaefer and Stefankovic 2024). The coNP lower bound was stated by Ghavamzadeh, Petrik, and Chow (2016) under a <sub>p</sub>olic<sub>y</sub> restriction removed here. Our <sub>square-roo</sub>t<sub>-sum an</sub>d th<sub>eory-o</sub>f<sub>-</sub>th<sub>e-rea</sub>l<sub>s</sub> b<sub>oun</sub>d<sub>s are new, an</sub>d <sub>ne</sub>ith<sub>er</sub> th<sub>e</sub> B<sub>oo</sub>l<sub>ean nor</sub> th<sub>e square-roo</sub>t<sub>-sum</sub> b<sub>oun</sub>d i<sub>s</sub> k<sub>nown</sub> t<sub>o</sub> b<sub>e compara</sub>bl<sub>e w</sub>ith th<sub>e</sub> th<sub>eory-o</sub>f<sub>-</sub>th<sub>e-rea</sub>l<sub>s</sub> b<sub>oun</sub>d<sub>s.</sub> Thi<sub>s</sub> <sub>ru</sub>l<sub>es ou</sub>t <sub>e</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>genera</sub>l<sub>-purpose a</sub>l<sub>gor</sub>ith<sub>ms un</sub>d<sub>er s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>comp</sub>l<sub>ex</sub>it<sub>y assump</sub>ti<sub>ons.</sub> Th<sub>ese wors</sub>t<sub>-case</sub> b<sub>oun</sub>d<sub>s s</sub>till l<sub>eave</sub> <sub>room</sub> f<sub>or</sub> th<sub>e prac</sub>ti<sub>ca</sub>l <sub>o</sub>fli<sub>ne cons</sub>t<sub>ruc</sub>ti<sub>on</sub> d<sub>eve</sub>l<sub>ope</sub>d l<sub>a</sub>t<sub>er.</sub>

<table><tr><td>Task</td><td>Nonrect. uncertainty</td><td> $( s , a )$  -rectangular uncertainty</td><td>Hardness already holds for</td></tr><tr><td>Certify one given policy</td><td>∀R-complete</td><td>coNP-hard and coSQRS-hard; in ∀R</td><td>deterministic policies; acyclic models whose uncer- tain choices are two-Dirac, for coNP</td></tr><tr><td>Certify a given portfolio</td><td>∀R-complete</td><td>∀R-complete</td><td>deterministic portfolios with acyclic graphs and two- successor choices</td></tr><tr><td>Synthesize one random- ized policy</td><td>∀R-hard and in ∃∀R</td><td>NP-hard, coNP-hard, and SQRS⊥-hard; in ∃VR</td><td>two-Dirac models for the Boolean bounds</td></tr><tr><td>Synthesize at most k poli- cies</td><td>VR-complete</td><td>hard already for  $k = 1 ;$  exact complexity open</td><td>regret threshold 2, fixed discount, acyclic dynamics, two-successor uncertain choices</td></tr></table>

Table 1: Complexity of regret certification and synthesis. Certification is established by Theorems 1 to 4. Synthesis i <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>e</sub>d b<sub>y</sub> Th<sub>eorems</sub> 5 t<sub>o</sub> 9<sub>.</sub>

Th<sub>e</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s pers</sub>i<sub>s</sub>t <sub>even un</sub>d<sub>er s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>assump-</sub> ti<sub>ons</sub> th<sub>a</sub>t <sub>usua</sub>ll<sub>y s</sub>i<sub>mp</sub>lif<sub>y</sub> RMDP<sub>s.</sub> R<sub>ec</sub>t<sub>angu</sub>l<sub>ar</sub>it<sub>y</sub> i<sub>s</sub> th<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence assump</sub>ti<sub>on</sub> th<sub>a</sub>t t<sub>rans</sub>iti<sub>on uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y a</sub>t <sub>one</sub> <sub>s</sub>t<sub>a</sub>t<sub>e-ac</sub>ti<sub>on pa</sub>i<sub>r</sub> d<sub>oes no</sub>t <sub>cons</sub>t<sub>ra</sub>i<sub>n uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y a</sub>t <sub>ano</sub>th<sub>er,</sub> <sub>an</sub>d it <sub>common</sub>l<sub>y</sub> l<sub>owers</sub> <sub>comp</sub>l<sub>ex</sub>it<sub>y</sub> <sub>an</sub>d <sub>un</sub>d<sub>er</sub>li<sub>es</sub> <sub>muc</sub>h <sub>o</sub>f the RMDP literature (I<sub>y</sub>en<sub>g</sub>ar 2005; Wiesemann, Kuhn, and Rustem 2013; Suilen et al. 2024). Here it does not rescue <sub>p</sub>ortf<sub>o</sub>li<sub>o cer</sub>tifi<sub>ca</sub>ti<sub>on: a</sub>lth<sub>oug</sub>h <sub>rec</sub>t<sub>angu</sub>l<sub>ar</sub>it<sub>y removes</sub> th<sub>e rea</sub>l <sub>quan</sub>tifi<sub>er a</sub>lt<sub>erna</sub>ti<sub>on</sub> f<sub>rom compar</sub>i<sub>son</sub> b<sub>e</sub>t<sub>ween</sub> t<sub>wo acyc</sub>li<sub>c</sub> <sub>po</sub>li<sub>c</sub>i<sub>es,</sub> th<sub>e</sub> <sub>po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se</sub> <sub>max</sub>i<sub>mum</sub> <sub>over</sub> <sub>por</sub>tf<sub>o</sub>li<sub>o</sub> <sub>mem</sub>b<sub>ers</sub> <sub>re-</sub> stores it (Theorem 3). Consequentl<sub>y</sub>, checkin<sub>g</sub> a <sub>g</sub>iven <sub>p</sub>ortfolio is ∀R-com<sub>p</sub>lete alread<sub>y</sub> for deterministic <sub>p</sub>ortfolios in acyclic (s, a)-rectangular RMDPs. Choosing a portfolio und<sub>er a unary s</sub>i<sub>ze</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>a</sub>dd<sub>s one ex</sub>i<sub>s</sub>t<sub>en</sub>ti<sub>a</sub>l <sub>rea</sub>l bl<sub>oc</sub>k <sub>an</sub>d is ∃∀R-com<sub>p</sub>lete for <sub>g</sub>eneral rational <sub>p</sub>ol<sub>y</sub>to<sub>p</sub>ic uncertaint<sub>y</sub>. The reduction cou<sub>p</sub>les choices, so it does not establish ∃∀Rcom<sub>p</sub>leteness under rectan<sub>g</sub>ular uncertaint<sub>y</sub> (Theorem 9).

D<sub>esp</sub>it<sub>e</sub> th<sub>ese wors</sub>t<sub>-case</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s, we g</sub>i<sub>ve a prac</sub>ti<sub>ca</sub>l <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> th<sub>a</sub>t di<sub>scre</sub>ti<sub>zes</sub> th<sub>e uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> i<sub>n</sub>t<sub>o ce</sub>ll<sub>s, com-</sub> <sub>pu</sub>t<sub>es a can</sub>did<sub>a</sub>t<sub>e po</sub>li<sub>cy</sub> f<sub>or eac</sub>h <sub>ce</sub>ll<sub>, an</sub>d <sub>eva</sub>l<sub>ua</sub>t<sub>es every</sub> <sub>can</sub>did<sub>a</sub>t<sub>e aga</sub>i<sub>ns</sub>t <sub>a</sub>ll <sub>ce</sub>ll<sub>s.</sub> It th<sub>en c</sub>l<sub>us</sub>t<sub>ers</sub> th<sub>e resu</sub>lti<sub>ng re-</sub> <sub>gre</sub>t <sub>pro</sub>fil<sub>es</sub> t<sub>o</sub> th<sub>e</sub> d<sub>es</sub>i<sub>re</sub>d <sub>por</sub>tf<sub>o</sub>li<sub>o</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub> At <sub>run</sub>ti<sub>me,</sub> UCB (Cesa-Bianchi and Lu<sub>g</sub>osi 2006) selects amon<sub>g</sub> the <sub>p</sub>ortfolio <sub>mem</sub>b<sub>ers as o</sub>b<sub>serva</sub>ti<sub>ons accumu</sub>l<sub>a</sub>t<sub>e.</sub>

## Contributions. Our contributions are as follows.

1<sub>.</sub> W<sub>e</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> <sub>a</sub>d<sub>ap</sub>ti<sub>ve</sub> <sub>po</sub>li<sub>cy</sub> <sub>por</sub>tf<sub>o</sub>li<sub>os</sub> <sub>as</sub> <sub>a</sub> fi<sub>n</sub>it<sub>e</sub> <sub>an</sub>d <sub>cer</sub>tifi<sub>a</sub>bl<sub>e</sub> f<sub>orm o</sub>f <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> RMDP<sub>s.</sub>

2<sub>.</sub> W<sub>e</sub> <sub>prove</sub> th<sub>e</sub> t<sub>wo</sub> <sub>exac</sub>t <sub>por</sub>tf<sub>o</sub>li<sub>o</sub> <sub>c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>ons</sub> <sub>sum-</sub> <sub>mar</sub>i<sub>ze</sub>d i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 1<sub>: g</sub>i<sub>ven-por</sub>tf<sub>o</sub>li<sub>o cer</sub>tifi<sub>ca</sub>ti<sub>on</sub> i<sub>s</sub> ∀R-com<sub>p</sub>lete and bounded-<sub>p</sub>ortfolio s<sub>y</sub>nthesis is ∃∀Rcom<sub>p</sub>lete (Theorems 3 and 9).

3<sub>.</sub> W<sub>e</sub> id<sub>en</sub>tif<sub>y</sub> th<sub>e s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>sources o</sub>f h<sub>ar</sub>d<sub>ness: uncer</sub>t<sub>a</sub>i<sub>n</sub> t<sub>rans</sub>iti<sub>ons use</sub>d b<sub>y</sub> b<sub>o</sub>th <sub>po</sub>li<sub>c</sub>i<sub>es y</sub>i<sub>e</sub>ld B<sub>oo</sub>l<sub>ean</sub> h<sub>ar</sub>d<sub>ness,</sub> th<sub>e</sub>i<sub>r recurrence on cyc</sub>l<sub>es y</sub>i<sub>e</sub>ld<sub>s</sub> h<sub>ar</sub>d<sub>ness</sub> f<sub>rom sums o</sub>f <sub>square</sub> <sub>roo</sub>t<sub>s,</sub> <sub>an</sub>d <sub>c</sub>h<sub>oos</sub>i<sub>ng</sub> <sub>a</sub> <sub>po</sub>li<sub>cy</sub> <sub>a</sub>dd<sub>s</sub> f<sub>ur</sub>th<sub>er</sub> B<sub>oo</sub>l<sub>ean</sub> <sub>an</sub>d <sub>s</sub>i<sub>gne</sub>d<sub>-square-roo</sub>t<sub>-sum</sub> h<sub>ar</sub>d<sub>ness.</sub>

4<sub>.</sub> W<sub>e</sub> <sub>presen</sub>t <sub>an</sub> <sub>o</sub>fli<sub>ne</sub> <sub>por</sub>tf<sub>o</sub>li<sub>o</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> th<sub>a</sub>t i<sub>s</sub> <sub>amena</sub>bl<sub>e</sub> t<sub>o run</sub>ti<sub>me spec</sub>i<sub>a</sub>li<sub>za</sub>ti<sub>on.</sub>

Th<sub>e</sub> <sub>ma</sub>i<sub>n</sub> t<sub>ex</sub>t <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>pr</sub>i<sub>nc</sub>i<sub>pa</sub>l <sub>argumen</sub>t<sub>s.</sub> C<sub>omp</sub>l<sub>e</sub>t<sub>e</sub> <sub>proo</sub>f<sub>s</sub> an<sup>d</sup> su<sub>pp</sub>ort<sup>i</sup>n<sub>g</sub> com<sub>p</sub>ar<sup>i</sup>son resu<sup>l</sup>ts a<sub>pp</sub>ear <sup>i</sup>n t<sup>h</sup>e a<sub>pp</sub>en<sup>di</sup>x.

## Related Work

S<sub>evera</sub>l <sub>approac</sub>h<sub>es</sub> <sub>re</sub>d<sub>uce</sub> th<sub>e</sub> <sub>conserva</sub>ti<sub>veness</sub> <sub>o</sub>f <sub>wors</sub>t<sub>-</sub> <sub>case ro</sub>b<sub>us</sub>t <sub>po</sub>li<sub>c</sub>i<sub>es</sub> b<sub>y op</sub>ti<sub>m</sub>i<sub>z</sub>i<sub>ng over</sub> l<sub>ess conserva</sub>ti<sub>ve</sub> uncertaint<sub>y</sub> sets (Wiesemann, Kuhn, and Rustem 2013; Ben<sub>y</sub>amine et al. 2026). When multi<sub>p</sub>le o<sub>p</sub>timal robust <sub>p</sub>oli-<sub>c</sub>i<sub>es ex</sub>i<sub>s</sub>t<sub>,</sub> b<sub>es</sub>t<sub>-e</sub>f<sub>or</sub>t <sub>approac</sub>h<sub>es can se</sub>l<sub>ec</sub>t <sub>among</sub> th<sub>em w</sub>ith<sub>-</sub> out sacrificin<sub>g</sub> worst-case value (Abate et al. 2026).

A <sub>more</sub> <sub>a</sub>d<sub>ap</sub>ti<sub>ve</sub> <sub>response</sub> t<sub>o</sub> fi<sub>xe</sub>d b<sub>u</sub>t i<sub>n</sub>iti<sub>a</sub>ll<sub>y</sub> <sub>un-</sub> k<sub>nown</sub> t<sub>rans</sub>iti<sub>on</sub> d<sub>ynam</sub>i<sub>cs</sub> i<sub>s prov</sub>id<sub>e</sub>d b<sub>y</sub> B<sub>ayes-a</sub>d<sub>ap</sub>ti<sub>ve</sub> MDPs (Guez, Silver, and Da<sub>y</sub>an 2012), hidden-model MDPs (Chades et al. 2012), and hidden-<sub>p</sub>arameter MDPs (Doshi-Velez and Konidaris 2016). These models <sub>p</sub>lan <sub>over a</sub> b<sub>e</sub>li<sub>e</sub>f <sub>on</sub> th<sub>e un</sub>k<sub>nown</sub> d<sub>ynam</sub>i<sub>cs an</sub>d <sub>up</sub>d<sub>a</sub>t<sub>e</sub> it <sub>as ev</sub>i<sub>-</sub> d<sub>ence</sub> <sub>accumu</sub>l<sub>a</sub>t<sub>es.</sub> Thi<sub>s</sub> f<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> <sub>express</sub>i<sub>ve,</sub> b<sub>u</sub>t <sub>p</sub>l<sub>an-</sub> <sub>n</sub>i<sub>ng</sub> b<sub>ecomes</sub> i<sub>n</sub>t<sub>rac</sub>t<sub>a</sub>bl<sub>e ou</sub>t<sub>s</sub>id<sub>e narrow spec</sub>i<sub>a</sub>l <sub>cases.</sub> It<sub>s</sub> b<sub>e</sub>li<sub>e</sub>f <sub>represen</sub>t<sub>a</sub>ti<sub>on can</sub> b<sub>e con</sub>ti<sub>nuous or</sub> hi<sub>g</sub>h<sub>-</sub>di<sub>mens</sub>i<sub>ona</sub>l<sub>,</sub> <sub>an</sub>d <sub>on</sub>li<sub>ne</sub> b<sub>e</sub>li<sub>e</sub>f<sub>-space p</sub>l<sub>ann</sub>i<sub>ng can</sub> it<sub>se</sub>lf b<sub>e expens</sub>i<sub>ve.</sub>

Ad<sub>ap</sub>ti<sub>ve po</sub>li<sub>cy por</sub>tf<sub>o</sub>li<sub>os occupy an</sub> i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e po-</sub> <sub>s</sub>iti<sub>on.</sub> Th<sub>e</sub>i<sub>r expens</sub>i<sub>ve compu</sub>t<sub>a</sub>ti<sub>on</sub> i<sub>s per</sub>f<sub>orme</sub>d <sub>o</sub>fli<sub>ne,</sub> <sub>w</sub>hil<sub>e on</sub>li<sub>ne a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on re</sub>d<sub>uces</sub> t<sub>o se</sub>l<sub>ec</sub>ti<sub>ng among a</sub> fi<sub>xe</sub>d <sub>se</sub>t <sub>o</sub>f <sub>po</sub>li<sub>c</sub>i<sub>es.</sub> R<sub>egre</sub>t i<sub>s na</sub>t<sub>ura</sub>l i<sub>n</sub> thi<sub>s se</sub>tti<sub>ng</sub> b<sub>ecause</sub> th<sub>e</sub> <sub>por</sub>tf<sub>o</sub>li<sub>o</sub> <sub>s</sub>h<sub>ou</sub>ld <sub>con</sub>t<sub>a</sub>i<sub>n</sub> <sub>a</sub> <sub>po</sub>li<sub>cy</sub> th<sub>a</sub>t <sub>per</sub>f<sub>orms</sub> <sub>near-</sub> o<sub>p</sub>timall<sub>y</sub> for whichever environment turns out to be true (cf. Ghavamzadeh, Petrik, and Chow (2016)).

## 2 Problem Statement

For a finite set X, we write |X| for its cardinality, and ${ \mathcal { D } } ( X )$ for the set of discrete probability distributions over X, i.e., f<sub>unc</sub>ti<sub>ons</sub> $\mu \colon X \to [ 0 , { \bar { 1 } } ]$ <sub>w</sub>ith $\begin{array} { r } { \sum _ { x \in X } \mu ( x ) = 1 } \end{array}$ <sub>.</sub> V<sub>ec</sub>t<sub>o</sub>r<sub>s a</sub>nd <sub>ma</sub>t<sub>r</sub>i<sub>ces are</sub> b<sub>o</sub>ldf<sub>ace.</sub> F<sub>or</sub> $\pmb { x } \in \mathbb { R } ^ { n } , \pmb { x } ( i )$ denotes its i-th entr<sub>y</sub>. Throu<sub>g</sub>hout, S and A denote finite sets of states and <sub>ac</sub>ti<sub>ons</sub> <sub>o</sub>f <sub>a</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> <sub>process,</sub> <sub>an</sub>d <sub>we</sub> <sub>ca</sub>ll <sub>a</sub> <sub>pa</sub>i<sub>r</sub> $( s , a ) \in$ $S \times A$ a choice. We fix an arbitrary ordering of the triples $( s , a , s ^ { \prime } ) \in S \times A \times S$ , so that a vector u $\in \mathbb { R } ^ { \widecheck { S } \times A \times S }$ ass<sup>i</sup><sub>g</sub>ns <sub>a va</sub>l<sub>ue</sub> $\mathbf { \boldsymbol { u } } ( s , a , s ^ { \prime } )$ to each triple. We call u a transition vector if $\pmb { u } ( s , a , \cdot ) \in \mathcal { D } ( S )$ for every choice (s, a).

Definition 1 (RMDP). A robust Markov decision process (RMDP) is a tuple $( S , A , \mathcal { U } , R , s _ { \iota } , \gamma )$ , where $R \colon S \times { \mathsf { \bar { A } } } \to \mathbb { R }$ is the rewardfunction, $s _ { \iota } \in S$ is the initial state, $\gamma \in ( 0 , 1 )$

is a discount factor, and

$$
\mathcal { U } = \{ \pmb { u } \in \mathbb { R } ^ { S \times A \times S } \ | \ F \pmb { u } \leq \pmb { g } \}
$$

is a convex polytope of transition vectors, for $\pmb { F } \in \mathbb { R } ^ { m \times n }$ $\pmb { \mathscr { g } } \in \mathbb { R } ^ { m } ( n = | S | | \bar { A } | | \bar { S } | )$ ). The inequalities F u ≤ g include the constraints defining a transition vector, so every $\mathbf { \boldsymbol { u } } \in \mathcal { U }$ is a valid transitionfunction.

A <sub>po</sub>li<sub>cy maps pa</sub>th<sub>s</sub> t<sub>o</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons over ac</sub>ti<sub>ons,</sub> $\pi \colon ( { \bar { S A } } ) ^ { * } { \bar { S } } \  \ { \bar { \mathcal { D } } } ( { \bar { A } } )$ . A memoryless randomized policy $\pi \colon S \to D ( A )$ d<sub>epen</sub>d<sub>s on</sub>l<sub>y on</sub> th<sub>e curren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e.</sub> Th<sub>e se</sub>t <sub>o</sub>f <sub>suc</sub>h <sub>po</sub>li<sub>c</sub>i<sub>es</sub> i<sub>s</sub> $\dot { \Pi } ^ { \mathrm { M R } }$ . A memoryless deterministic policy $\pi \colon S  A$ f<sub>ur</sub>th<sub>er se</sub>l<sub>ec</sub>t<sub>s a s</sub>i<sub>ng</sub>l<sub>e ac</sub>ti<sub>on per s</sub>t<sub>a</sub>t<sub>e.</sub> Th<sub>e se</sub>t <sub>o</sub>f <sub>suc</sub>h <sub>po</sub>li<sub>c</sub>i<sub>es</sub> i<sub>s</sub> $\Pi ^ { \mathrm { M D } } \subseteq \Pi ^ { \mathrm { M R } }$ <sub>.</sub> U<sub>n</sub>l<sub>ess s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>o</sub>th<sub>erw</sub>i<sub>se,</sub> <sub>can</sub>did<sub>a</sub>t<sub>e po</sub>li<sub>c</sub>i<sub>es range over</sub> $\Pi ^ { \mathrm { M R } }$

E<sub>ac</sub>h $\textbf { \textit { u } } \in \textbf { \textit { u } }$ i<sub>n</sub>d<sub>uces a c</sub>l<sub>ass</sub>i<sub>ca</sub>l MDP $\begin{array} { r l } { M _ { u } } & { { } = } \end{array}$ $( S , A , P _ { \pmb { u } } , R , s _ { \iota } , \gamma )$ <sub>w</sub>ith $P _ { \bf u } ( s , a ) ( s ^ { \prime } ) = { \bf u } ( s , a , s ^ { \prime } )$ <sub>.</sub> F<sub>or a</sub> policy π, its value $V _ { \mathbf { \eta u } } ^ { \pi } \colon S \to \operatorname { \mathbb { R } }$ i<sub>n</sub> $M _ { u }$ i<sub>s</sub> th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d di<sub>s-</sub> <sub>coun</sub>t<sub>e</sub>d <sub>cumu</sub>l<sub>a</sub>ti<sub>ve rewar</sub>d<sub>,</sub>

$$
V _ { \pmb { u } } ^ { \pi } ( s ) = \mathbb { E } _ { \pi } \Big [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R ( s _ { t } , a _ { t } ) \ \Big | \ s _ { 0 } = s \Big ] ,
$$

<sub>an</sub>d $V _ { u } ^ { * } ( s ) = \operatorname* { s u p } _ { \pi } V _ { u } ^ { \pi } ( s )$ i<sub>s</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>va</sub>l<sub>ue.</sub> Thi<sub>s supre-</sub> <sub>mum</sub> i<sub>s a</sub>tt<sub>a</sub>i<sub>ne</sub>d b<sub>y a po</sub>li<sub>cy</sub> i<sub>n</sub> $\Pi ^ { \mathrm { M D } }$ <sub>an</sub>d $V _ { \pmb { u } } ^ { * }$ i<sub>s compu</sub>t<sub>a</sub>bl<sub>e</sub> in <sub>p</sub>ol<sub>y</sub>nomial time for fixed u (Puterman 1994).

Types of uncertainty. By default, U places no further <sub>s</sub>t<sub>ruc</sub>t<sub>ure on</sub> h<sub>ow uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> i<sub>n</sub>t<sub>erac</sub>t<sub>s across c</sub>h<sub>o</sub>i<sub>ces.</sub> W<sub>e ca</sub>ll this the general (non-rectangular, convex-polytopic) case. The s<sub>p</sub>ecial case where U decom<sub>p</sub>oses into inde<sub>p</sub>endent <sub>p</sub>er-<sub>c</sub>h<sub>o</sub>i<sub>ce uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y se</sub>t<sub>s</sub> i<sub>s</sub> $( s , a )$ -rectangular:

$$
\mathcal { U } = \bigcup _ { ( s , a ) \in S \times A } \mathcal { U } _ { ( s , a ) } , \qquad \mathcal { U } _ { ( s , a ) } \subseteq \mathbb { R } ^ { \{ s \} \times \{ a \} \times S } .
$$

W<sub>e ca</sub>ll $\mathcal { U } _ { ( s , a ) }$ th<sub>e uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y se</sub>t <sub>o</sub>f <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> $( s , a )$ <sub>, an</sub>d th<sub>e</sub> choice uncertain $\mathrm { i f } \mathcal { U } _ { ( s , a ) }$ i<sub>s no</sub>t <sub>a s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on.</sub> I<sub>n</sub>t<sub>u</sub>iti<sub>ve</sub>l<sub>y,</sub> $( s , a )$ <sub>rec</sub>t<sub>angu</sub>l<sub>ar</sub>it<sub>y</sub> i<sub>s an</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>ence assump</sub>ti<sub>on across c</sub>h<sub>o</sub>i<sub>ces.</sub> The <sub>g</sub>eneral case allows U to cou<sub>p</sub>le choices, whereas $( s , a )$ <sub>rec</sub>t<sub>angu</sub>l<sub>ar</sub>it<sub>y pro</sub>hibit<sub>s</sub> thi<sub>s.</sub>

Subclasses of RMDPs. An uncertain choice is twosuccessor if every distribution in its uncertainty set is supported on the same two states, and two-Dirac if that set is the li<sub>ne</sub> <sub>segmen</sub>t b<sub>e</sub>t<sub>ween</sub> t<sub>wo</sub> Di<sub>rac</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons.</sub> A<sub>n</sub> RMDP h<sub>as</sub> <sub>e</sub>ith<sub>er proper</sub>t<sub>y w</sub>h<sub>en a</sub>ll it<sub>s uncer</sub>t<sub>a</sub>i<sub>n c</sub>h<sub>o</sub>i<sub>ces</sub> d<sub>o.</sub> A<sub>n</sub> RMDP is acyclic when its possible-transition graph has no directed <sub>cyc</sub>l<sub>e</sub> <sub>excep</sub>t <sub>se</sub>lf<sub>-</sub>l<sub>oops</sub> <sub>a</sub>t <sub>a</sub>b<sub>sor</sub>bi<sub>ng</sub> fi<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es.</sub> A <sub>po</sub>li<sub>cy</sub> <sub>grap</sub>h i<sub>s acyc</sub>li<sub>c un</sub>d<sub>er</sub> th<sub>e ana</sub>l<sub>ogous res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>on</sub> t<sub>o c</sub>h<sub>o</sub>i<sub>ces</sub> <sub>use</sub>d b<sub>y</sub> th<sub>a</sub>t <sub>po</sub>li<sub>cy.</sub> C<sub>ons</sub>t<sub>ruc</sub>ti<sub>ons</sub> th<sub>a</sub>t d<sub>escr</sub>ib<sub>e</sub> <sub>on</sub>l<sub>y</sub> th<sub>e</sub>i<sub>r</sub> i<sub>n</sub>t<sub>en</sub>d<sub>e</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ces are comp</sub>l<sub>e</sub>t<sub>e</sub>d <sub>on every o</sub>th<sub>er s</sub>t<sub>a</sub>t<sub>e-ac</sub>ti<sub>on</sub> <sub>pa</sub>i<sub>r</sub> b<sub>y</sub> th<sub>e</sub> <sub>ru</sub>i<sub>nous-s</sub>i<sub>n</sub>k <sub>conven</sub>ti<sub>on</sub> <sub>o</sub>f S<sub>ec</sub>ti<sub>on</sub> A<sub>.</sub>1<sub>.</sub>

Robust Value & Regret. For a policy π, its robust value is

$$
V ^ { \pi , \mathrm { r o b } } ( s ) = \operatorname* { i n f } _ { { \pmb u } \in \mathcal { U } } V _ { \pmb u } ^ { \pi } ( s ) ,
$$

the worst-case value π can guarantee against any realization in U. The robust optimal value is $V ^ { \mathrm { r o b } } ( \breve { s } ) = \mathrm { s u p } _ { \pi } ^ { \cdot } V ^ { \pi , \mathrm { r o b } } ( s )$ and a policy attaining it is robust optimal. Robust value com-<sub>pares</sub> <sub>po</sub>li<sub>c</sub>i<sub>es</sub> i<sub>n</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> t<sub>erms,</sub> b<sub>u</sub>t it d<sub>oes</sub> <sub>no</sub>t di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h <sub>a</sub> <sub>po</sub>li<sub>cy</sub> <sub>per</sub>f<sub>orm</sub>i<sub>ng</sub> <sub>poor</sub>l<sub>y</sub> f<sub>rom</sub> <sub>a</sub> <sub>rea</sub>li<sub>za</sub>ti<sub>on</sub> th<sub>a</sub>t i<sub>s</sub> <sub>s</sub>i<sub>mp</sub>l<sub>y</sub> hard for every policy: even the best policy for a fixed u may <sub>a</sub>tt<sub>a</sub>i<sub>n</sub> <sub>a</sub> l<sub>ow</sub> <sub>va</sub>l<sub>ue</sub> th<sub>ere.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore</sub> t<sub>urn</sub> t<sub>o</sub> <sub>a</sub> <sub>compara</sub>ti<sub>ve</sub> <sub>no</sub>ti<sub>on,</sub> <sub>measur</sub>i<sub>ng</sub> <sub>a</sub> <sub>po</sub>li<sub>cy</sub>’<sub>s</sub> <sub>s</sub>h<sub>or</sub>tf<sub>a</sub>ll <sub>aga</sub>i<sub>ns</sub>t th<sub>e</sub> b<sub>es</sub>t <sub>po</sub>li<sub>cy</sub> f<sub>or eac</sub>h <sub>par</sub>ti<sub>cu</sub>l<sub>ar rea</sub>li<sub>za</sub>ti<sub>on, ra</sub>th<sub>er</sub> th<sub>an</sub> it<sub>s raw wors</sub>t<sub>-case</sub> <sub>va</sub>l<sub>ue.</sub> Th<sub>a</sub>t i<sub>s,</sub> f<sub>or</sub> $\pi \in \Pi ^ { \mathrm { M R } }$ , its robust regret (at s ) is

$$
\operatorname { R r e g } ( \pi ) = \operatorname* { s u p } _ { \boldsymbol { u } \in \mathcal { U } } \bigl ( V _ { \boldsymbol { u } } ^ { * } ( s _ { \iota } ) - V _ { \boldsymbol { u } } ^ { \pi } ( s _ { \iota } ) \bigr ) ,
$$

the largest shortfall of π from the policy optimal at u, over <sub>every rea</sub>li<sub>za</sub>ti<sub>on o</sub>f th<sub>e uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y.</sub> W<sub>e ca</sub>ll th<sub>e c</sub>h<sub>o</sub>i<sub>ce o</sub>f realization u nature’s move.

Definition 2 (Adaptive policy portfolio). An adaptive policy portfolio is a finite set $\dot { \boldsymbol { \Pi } } \subseteq \mathbf { \check { \Pi } } ^ { \mathrm { M R } }$ of candidate policies synthesized ofline and paired with a runtime mechanism that selects among its members as evidence accumulates.

R<sub>o</sub>b<sub>us</sub>t <sub>regre</sub>t lift<sub>s</sub> t<sub>o</sub> <sub>por</sub>tf<sub>o</sub>li<sub>os</sub> b<sub>y</sub> <sub>compar</sub>i<sub>ng</sub> <sub>aga</sub>i<sub>ns</sub>t th<sub>e</sub> best memberfor that realization:

$$
\mathrm { R r e g } ( \Pi ) = \operatorname* { s u p } _ { \pmb { u } \in \mathcal { U } } \Big ( V _ { \pmb { u } } ^ { * } \big ( s _ { \iota } \big ) - \operatorname* { m a x } _ { \pi \in \Pi } V _ { \pmb { u } } ^ { \pi } \big ( s _ { \iota } \big ) \Big ) .\tag{1}
$$

Thi<sub>s</sub> i<sub>s an o</sub>fli<sub>ne coverage guaran</sub>t<sub>ee:</sub> it id<sub>ea</sub>li<sub>zes away</sub> th<sub>e</sub> <sub>cos</sub>t <sub>o</sub>f id<sub>en</sub>tif<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> b<sub>es</sub>t <sub>mem</sub>b<sub>er an</sub>d <sub>assumes</sub> th<sub>a</sub>t <sub>mem-</sub> b<sub>er</sub> i<sub>s se</sub>l<sub>ec</sub>t<sub>e</sub>d<sub>.</sub> Th<sub>e on</sub>li<sub>ne</sub> id<sub>en</sub>tifi<sub>ca</sub>ti<sub>on cos</sub>t i<sub>s eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>separa</sub>t<sub>e</sub>l<sub>y</sub> i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 4<sub>.</sub> Thi<sub>s quan</sub>tit<sub>y canno</sub>t i<sub>n genera</sub>l b<sub>e</sub> <sub>recovere</sub>d f<sub>rom</sub> th<sub>e s</sub>i<sub>ng</sub>l<sub>e-po</sub>li<sub>cy regre</sub>t <sub>va</sub>l<sub>ues</sub> $\{ { \bar { \mathrm { R r e g } } } ( \pi )$ $\pi \in \Pi \}$ <sub>.</sub> P<sub>or</sub>tf<sub>o</sub>li<sub>o regre</sub>t <sub>eva</sub>l<sub>ua</sub>t<sub>es a</sub>ll <sub>mem</sub>b<sub>ers un</sub>d<sub>er</sub> th<sub>e</sub> <sub>same rea</sub>li<sub>za</sub>ti<sub>on</sub> b<sub>e</sub>f<sub>ore</sub> t<sub>a</sub>ki<sub>ng</sub> th<sub>e wors</sub>t <sub>case, w</sub>h<sub>ereas eac</sub>h <sub>s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on regre</sub>t h<sub>as a</sub>l<sub>rea</sub>d<sub>y</sub> t<sub>a</sub>k<sub>en</sub> it<sub>s own supremum.</sub> N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t th<sub>e</sub> b<sub>oun</sub>d $\begin{array} { r } { \operatorname { R r e g } ( \Pi ) \leq \operatorname* { m i n } _ { \pi \in \Pi } \operatorname { R r e g } ( \pi ) } \end{array}$ f<sub>o</sub>ll<sub>ows.</sub>

W<sub>e see</sub>k th<sub>e</sub> b<sub>es</sub>t <sub>guaran</sub>t<sub>ee ac</sub>hi<sub>eva</sub>bl<sub>e</sub> b<sub>y a por</sub>tf<sub>o</sub>li<sub>o o</sub>f b<sub>oun</sub>d<sub>e</sub>d <sub>s</sub>i<sub>ze</sub> $k \geq 1 { : }$

$$
\rho _ { k } = \operatorname* { i n f } _ { \Pi \subseteq \Pi ^ { \mathrm { M R } } : | \Pi | \leq k } { \mathrm { R r e g } } ( \Pi ) .
$$

Note $\begin{array} { r } { \rho _ { 1 } = \operatorname* { i n f } _ { \pi \in \Pi ^ { \mathrm { M R } } } \operatorname { R r e g } ( \pi ) \colon } \end{array}$ <sub>:</sub> <sub>m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>s</sub>i<sub>ng</sub>l<sub>e-po</sub>li<sub>cy</sub> <sub>regre</sub>t i<sub>s</sub> th<sub>e</sub> $k = 1$ <sub>case</sub> <sub>o</sub>f <sub>m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>por</sub>tf<sub>o</sub>li<sub>o</sub> <sub>regre</sub>t<sub>.</sub>

## 3 Computational Complexity

W<sub>e</sub> fi<sub>rs</sub>t <sub>reca</sub>ll th<sub>e requ</sub>i<sub>re</sub>d <sub>c</sub>l<sub>asses,</sub> th<sub>en</sub> t<sub>urn</sub> t<sub>o</sub> th<sub>e cer</sub>tifi<sub>ca-</sub> ti<sub>on an</sub>d <sub>syn</sub>th<sub>es</sub>i<sub>s o</sub>f <sub>por</sub>tf<sub>o</sub>li<sub>os.</sub> W<sub>e no</sub>t<sub>e</sub> th<sub>a</sub>t th<sub>e resu</sub>lt<sub>s y</sub>i<sub>e</sub>ld a rich landscape, but all studied problems lie in PSPACE.

## 3.1 Complexity Preliminaries

W<sub>e assume</sub> b<sub>as</sub>i<sub>c</sub> f<sub>am</sub>ili<sub>ar</sub>it<sub>y w</sub>ith <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>comp</sub>l<sub>ex</sub>it<sub>y</sub> classes such as NP, coNP, and PSPACE (Papadimitriou 1994; Arora and Barak 2009).

Input conventions. The rewards, γ, threshold t, polytope data F, g, and policy matrices in $\mathbb { Q } ^ { S \times A }$ <sub>are</sub> bi<sub>nary-enco</sub>d<sub>e</sub>d rationals. The portfolio budget k is encoded in unary.

Theory of the reals. We use the first-order theory of the <sub>rea</sub>l<sub>s over</sub> $( \mathbb { R } , 0 , 1 , + , \cdot , < )$ <sub>.</sub> I<sub>n</sub> thi<sub>s</sub> l<sub>og</sub>i<sub>c, we can as</sub>k <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>a</sub> <sub>sys</sub>t<sub>em</sub> <sub>o</sub>f <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>equa</sub>ti<sub>ons</sub> <sub>an</sub>d i<sub>nequa</sub>liti<sub>es</sub> h<sub>as</sub> <sub>a</sub> <sub>rea</sub>l <sub>so</sub>l<sub>u</sub>ti<sub>on, or</sub> h<sub>o</sub>ld<sub>s</sub> f<sub>or every rea</sub>l <sub>ass</sub>i<sub>gnmen</sub>t<sub>.</sub> W<sub>e cons</sub>id<sub>er</sub> th<sub>ree</sub> f<sub>ragmen</sub>t<sub>s, g</sub>i<sub>ven a quan</sub>tifi<sub>er-</sub>f<sub>ree</sub> f<sub>ormu</sub>l<sub>a</sub> $\varphi \colon$

• existential sentences $\exists x _ { 1 } , \dots , x _ { n } \ \varphi ( x _ { 1 } , \dots , x _ { n } ) ;$

• universal sentences $\forall x _ { 1 } , \ldots , x _ { n } \ \varphi ( x _ { 1 } , \ldots , x _ { n } )$

• existential-universal sentences

$$
\exists x _ { 1 } , \dots , x _ { n } \forall y _ { 1 } , \dots , y _ { m } \varphi ( x _ { 1 } , \dots , x _ { n } , y _ { 1 } , \dots , y _ { m } ) .
$$

Th<sub>e</sub> t<sub>ru</sub>th <sub>pro</sub>bl<sub>em o</sub>f <sub>eac</sub>h f<sub>ragmen</sub>t d<sub>e</sub>fi<sub>nes a comp</sub>l<sub>ex-</sub> it<sub>y</sub> class, denoted ∃R, ∀R, and ∃∀R res<sub>p</sub>ectivel<sub>y</sub> (Schaefer and Stefankovic 2024). These classes sit between ${ \mathsf { N P } }$ <sub>an</sub>d PSPACE (Canny 1988; Basu, Pollack, and Roy 2006), and $\forall \mathbb { R } = { \mathsf { c o } } \exists \mathbb { R }$ <sub>an</sub>d $\exists \mathbb { R } = { \mathsf { c o } } \forall \mathbb { R }$

Square-root-sum. We also consider variants of the wellknown square-root-sum (SQRS) decision problem and th<sub>e</sub>i<sub>r assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>comp</sub>l<sub>ex</sub>it<sub>y c</sub>l<sub>asses.</sub> F<sub>or</sub> th<sub>e var</sub>i<sub>an</sub>t<sub>s</sub> b<sub>e-</sub> l<sub>ow, assume</sub> fi<sub>n</sub>it<sub>e</sub> li<sub>s</sub>t<sub>s o</sub>f <sub>pos</sub>iti<sub>ve</sub> i<sub>n</sub>t<sub>egers</sub> $a _ { 1 } , \ldots , a _ { m }$ <sub>an</sub>d $b _ { 1 } , \ldots , b _ { n }$ <sub>, an</sub>d <sub>an</sub> i<sub>n</sub>t<sub>eger</sub> $k ,$ <sub>a</sub>ll <sub>enco</sub>d<sub>e</sub>d i<sub>n</sub> bi<sub>nary.</sub>

• SQRS. Decide whether $\textstyle \sum _ { i = 1 } ^ { m } { \sqrt { a _ { i } } } \leq k .$ . coSQRS is the <sub>comp</sub>l<sub>emen</sub>t <sub>c</sub>l<sub>ass.</sub>

${ \mathsf { S Q R S } } \pm$ . Given an operator ▷◁∈ $\{ \leq , \geq \}$ <sub>,</sub> d<sub>ec</sub>id<sub>e w</sub>h<sub>e</sub>th<sub>er</sub> $\begin{array} { r } { \sum _ { i = 1 } ^ { m } \sqrt { a _ { i } } \boxtimes \sum _ { j = 1 } ^ { n } \sqrt { b _ { j } } . } \end{array}$

Both SQRS and coSQRS reduce to ${ \mathsf { S Q R S } } _ { \pm }$ : SQRS uses $\begin{array} { r } { \boldsymbol { b } \ = \ \left( \boldsymbol { k } ^ { 2 } \right) } \end{array}$ <sub>;</sub> f<sub>or</sub> ${ \mathsf { c o S Q R S } } .$ <sub>,</sub> t<sub>es</sub>t <sub>equa</sub>lit<sub>y</sub> i<sub>n</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l ti<sub>me</sub> $( \sum _ { i } \sqrt { a _ { i } } \overset { \cdot } { \in } \mathbb { Q }$ if <sub>every</sub> $a _ { i }$ is a <sub>p</sub>erfect square), then use $\geq$

Polynomial hierarchy and common upper bound. The polynomial hierarchy PH is the union of the classes obt<sub>a</sub>i<sub>ne</sub>d b<sub>y a</sub>lt<sub>erna</sub>ti<sub>ng po</sub>l<sub>ynom</sub>i<sub>a</sub>ll<sub>y</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>ex</sub>i<sub>s</sub>t<sub>en</sub>ti<sub>a</sub>l <sub>an</sub>d <sub>un</sub>i<sub>versa</sub>l B<sub>oo</sub>l<sub>ean</sub> <sub>quan</sub>tifi<sub>ers</sub> i<sub>n</sub> f<sub>ron</sub>t <sub>o</sub>f <sub>a</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>-</sub>ti<sub>me</sub> <sub>pre</sub>di<sub>ca</sub>t<sub>e.</sub> I<sub>n</sub> <sub>par</sub>ti<sub>cu</sub>l<sub>ar,</sub> $\Sigma _ { 2 } ^ { p }$ b<sub>eg</sub>i<sub>ns</sub> <sub>w</sub>ith <sub>an</sub> <sub>ex</sub>i<sub>s</sub>t<sub>en</sub>ti<sub>a</sub>l bl<sub>oc</sub>k followed b<sub>y</sub> a universal block (Pa<sub>p</sub>adimitriou 1994; Arora and Barak 2009). Every level of PH, every fixed level of th<sub>e rea</sub>l hi<sub>erarc</sub>h<sub>y cons</sub>id<sub>ere</sub>d <sub>a</sub>b<sub>ove, an</sub>d $S \mathsf { Q } \mathsf { R } S _ { \pm }$ li<sub>e</sub> i<sub>n</sub> PSPACE (Canny 1988; Allender et al. 2009). Thus all classes used in this paper share a PSPACE upper bound.

## 3.2 Policy and Portfolio Comparison

Policy comparison. We call the following problem policy comparison: given two policies $\pi _ { 1 } , \pi _ { 2 } \in \Pi ^ { \mathrm { \breve { M R } } }$ <sub>an</sub>d <sub>a</sub> th<sub>res</sub>h<sub>-</sub> <sub>o</sub>ld $t ,$ d<sub>ec</sub>id<sub>e w</sub>h<sub>e</sub>th<sub>er</sub>

$$
\Delta _ { \mathscr { U } } ( \pi _ { 1 } , \pi _ { 2 } ) = \operatorname* { s u p } _ { \pmb { u } \in \mathscr { U } } \bigl ( V _ { \pmb { u } } ^ { \pi _ { 1 } } ( s _ { \iota } ) - V _ { \pmb { u } } ^ { \pi _ { 2 } } ( s _ { \iota } ) \bigr ) \leq t .
$$

P<sub>o</sub>li<sub>cy compar</sub>i<sub>son</sub> i<sub>s</sub> t<sub>rac</sub>t<sub>a</sub>bl<sub>e w</sub>h<sub>en</sub> $\pi _ { 1 }$ <sub>an</sub>d $\pi _ { 2 }$ <sub>s</sub>h<sub>are no un-</sub> certain choice (A<sub>pp</sub>endix B.1): in this case the robust value of <sub>eac</sub>h <sub>po</sub>li<sub>cy</sub> <sub>can</sub> b<sub>e</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>ll<sub>y</sub> <sub>an</sub>d thi<sub>s</sub> i<sub>s</sub> k<sub>nown</sub> to be feasible in <sub>p</sub>ol<sub>y</sub>nomial time (I<sub>y</sub>en<sub>g</sub>ar 2005; Suilen and Pérez 2026). Otherwise, it is coNP-hard<sup>1</sup> already in acyclic $( s , a )$ -rectangular RMDPs, and coNP-complete when no un-<sub>cer</sub>t<sub>a</sub>i<sub>n c</sub>h<sub>o</sub>i<sub>ce use</sub>d b<sub>y</sub> b<sub>o</sub>th <sub>po</sub>li<sub>c</sub>i<sub>es</sub> li<sub>es on a cyc</sub>l<sub>e o</sub>f <sub>e</sub>ith<sub>er</sub> <sub>p</sub>olic<sub>y</sub> <sub>g</sub>ra<sub>p</sub>h (A<sub>pp</sub>endices B.2 and B.3). In this c<sub>y</sub>cle-free <sub>reg</sub>i<sub>me</sub> th<sub>e</sub> <sub>supremum</sub> d<sub>e</sub>fi<sub>n</sub>i<sub>ng</sub> $\Delta _ { { \scriptscriptstyle M } }$ i<sub>s a</sub>tt<sub>a</sub>i<sub>ne</sub>d <sub>a</sub>t <sub>a ra</sub>ti<sub>ona</sub>l <sub>ver</sub>t<sub>ex</sub> <sub>o</sub>f th<sub>e</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> <sub>po</sub>l<sub>y</sub>t<sub>ope,</sub> <sub>so</sub> <sub>a</sub> <sub>w</sub>it<sub>ness</sub> <sub>rea</sub>li<sub>za</sub>ti<sub>on</sub> <sub>can</sub> b<sub>e wr</sub>itt<sub>en</sub> d<sub>own exac</sub>tl<sub>y.</sub>

Sketch ofcoNP-hardness. Reduce from UNSAT. For a $3 \mathrm { - }$ CNF f<sub>o</sub>rm<sub>u</sub>l<sub>a</sub> $\varphi ,$ th<sub>e</sub> RMDP k<sub>eeps</sub> t<sub>wo</sub> l<sub>oca</sub>l bit<sub>s</sub> <sub>per</sub> lit<sub>era</sub>l <sub>occurrence, one cer</sub>tif<sub>y</sub>i<sub>ng eac</sub>h <sub>va</sub>l<sub>ue</sub> f<sub>or</sub> th<sub>e</sub> bit<sub>s, an</sub>d l<sub>e</sub>t<sub>s</sub> <sub>na</sub>t<sub>ure</sub> fi<sub>x</sub> th<sub>em</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y</sub> th<sub>roug</sub>h <sub>separa</sub>t<sub>e</sub> t<sub>wo-</sub>Di<sub>rac</sub> <sub>c</sub>h<sub>o</sub>i<sub>ces.</sub> B<sub>o</sub>th <sub>po</sub>li<sub>c</sub>i<sub>es rea</sub>d th<sub>e same cop</sub>i<sub>es</sub> b<sub>u</sub>t <sub>as</sub>k dif<sub>eren</sub>t questions of them. The clause policy $\pi _ { C }$ <sup>acce</sup>p<sup>ts</sup> w<sup>hen</sup> <sup>e</sup>v<sup>er</sup>y clause has a locally true literal. The consistency policy π<sub>K</sub> <sub>un</sub>if<sub>orm</sub>l<sub>y se</sub>l<sub>ec</sub>t<sub>s a var</sub>i<sub>a</sub>bl<sub>e an</sub>d <sub>accep</sub>t<sub>s w</sub>h<sub>en a</sub>ll it<sub>s cop</sub>i<sub>es</sub> <sub>agree w</sub>ith <sub>one g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>va</sub>l<sub>ue.</sub> Th<sub>us</sub> $\pi _ { C }$ contributes either 0 or 1 to $\Delta _ { { \scriptscriptstyle M } }$ <sub>, w</sub>hil<sub>e</sub> $\pi _ { K }$ <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> th<sub>e</sub> f<sub>rac</sub>ti<sub>on o</sub>f <sub>var</sub>i<sub>a</sub>bl<sub>es w</sub>h<sub>ose</sub> <sub>au</sub>dit<sub>s pass.</sub> N<sub>a</sub>t<sub>ure a</sub>tt<sub>a</sub>i<sub>ns</sub> $2$ <sub>exac</sub>tl<sub>y w</sub>h<sub>en</sub> $\varphi$ i<sub>s sa</sub>ti<sub>s</sub>fi<sub>a</sub>bl<sub>e;</sub> <sub>o</sub>th<sub>erw</sub>i<sub>se</sub> th<sub>e</sub> dif<sub>erence</sub> i<sub>s a</sub>t <sub>mos</sub>t $\dot { 2 } - 1 / n$ □

E<sub>ven</sub> i<sub>n</sub> th<sub>e rec</sub>t<sub>angu</sub>l<sub>ar case, op</sub>ti<sub>ma</sub>lit<sub>y a</sub>t th<sub>e ver</sub>ti<sub>ces</sub> <sub>no</sub> l<sub>onger</sub> h<sub>o</sub>ld<sub>s once a s</sub>h<sub>are</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> li<sub>es on a cyc</sub>l<sub>e:</sub> th<sub>e</sub> <sub>supremum can</sub> th<sub>en</sub> b<sub>e a</sub>tt<sub>a</sub>i<sub>ne</sub>d i<sub>n</sub> th<sub>e</sub> i<sub>n</sub>t<sub>er</sub>i<sub>or, a</sub>t <sub>an</sub> i<sub>rra-</sub> ti<sub>ona</sub>l <sub>va</sub>l<sub>ue, a sum o</sub>f <sub>square roo</sub>t<sub>s, w</sub>hi<sub>c</sub>h i<sub>s exac</sub>tl<sub>y</sub> th<sub>e</sub> phenomenon coSQRS-hardness captures (see Appendix B.4 and the exam<sub>p</sub>le below). Membershi<sub>p</sub> in ∀R follows from <sub>a</sub> k<sub>nown</sub> <sub>a</sub>d<sub>ap</sub>t<sub>a</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> B<sub>e</sub>ll<sub>man</sub> <sub>equa</sub>ti<sub>ons</sub> f<sub>or</sub> th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> functions to the <sub>p</sub>arametric settin<sub>g</sub> (Jun<sub>g</sub>es et al. 2021).

Example 1. Consider this RMDP with discount $\gamma = 1 / 2 .$

![](images/bf009c3cc6e6b7f76046168d1465a17f8291d84cb0f116920e590e5dc5c70c0a.jpg)

Fix the realization $\mathbf { \boldsymbol { u } } \in \mathcal { U }$ corresponding to x. The value of g is 2. Under $\pi _ { 1 }$ , reaching c gives one zero-reward step back to $s _ { \iota }$ . Hence

$$
V _ { \pmb { u } } ^ { \pi _ { 1 } } ( s _ { \iota } ) = \frac { 1 } { 2 } \left( x \frac { 1 } { 2 } V _ { \pmb { u } } ^ { \pi _ { 1 } } ( s _ { \iota } ) + ( 1 - x ) 2 \right) ,
$$

hence $\begin{array} { r } { V _ { \pmb { u } } ^ { \pi _ { 1 } } ( s _ { \iota } ) = \frac { 1 - x } { 1 - x / 4 } } \end{array}$ . Under $\pi _ { 2 } ,$ , the state c has value zero, so $V _ { u } ^ { \pi _ { 2 } } ( s _ { \iota } ) = 1 - x$ . Thus $\begin{array} { r } { V _ { \pmb { u } } ^ { \pi _ { 1 } } ( s _ { \iota } ) - V _ { \pmb { u } } ^ { \pi _ { 2 } } ( s _ { \iota } ) = \frac { x ( 1 - x ) } { 4 - x } } \end{array}$ . This diference is zero at $x = 0$ and $x = 1 ,$ , but positive between them. Its maximum over $[ 0 , 1 ]$ is attained at $x ^ { * } = 4 - 2 { \sqrt { 3 } }$ Substitution gives $\Delta _ { \cal { U } } ( \pi _ { 1 } , \pi _ { 2 } ) = 7 - 4 \sqrt { 3 } .$ . The square root appears because $\pi _ { 1 }$ can revisit the same uncertain choice, so its value is a rational function of x whose maximum lies inside the interval.

Nonrectangularity makes things harder. Without rect-<sub>angu</sub>l<sub>ar</sub>it<sub>y, coup</sub>li<sub>ng</sub> l<sub>e</sub>t<sub>s one parame</sub>t<sub>er recur a</sub>t dif<sub>er-</sub> ent choices. Then com<sub>p</sub>arison is even ∀R-com<sub>p</sub>lete $( \mathsf { A p } \cdot$ <sub>p</sub>endix B.5). To obtain hardness, we encode <sub>p</sub>ol<sub>y</sub>nomial non-<sub>pos</sub>iti<sub>v</sub>it<sub>y</sub> <sub>an</sub>d b<sub>u</sub>ild <sub>an</sub> RMDP <sub>w</sub>h<sub>ose</sub> <sub>va</sub>l<sub>ue</sub> <sub>a</sub>t <sub>eac</sub>h <sub>rea</sub>li<sub>za</sub>ti<sub>on</sub> <sub>equa</sub>l<sub>s a g</sub>i<sub>ven po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>.</sub> A <sub>monom</sub>i<sub>a</sub>l $p _ { i _ { 1 } } \cdots p _ { i _ { d } }$ b<sub>ecomes</sub> a chain of d uncertain choices that survives with probability equal to the monomial. A certain splitter is a state whose <sub>so</sub>l<sub>e ac</sub>ti<sub>on</sub> d<sub>raws among</sub> th<sub>ese monom</sub>i<sub>a</sub>l b<sub>ranc</sub>h<sub>es w</sub>ith fi<sub>xe</sub>d <sub>ra</sub>ti<sub>ona</sub>l <sub>pro</sub>b<sub>a</sub>biliti<sub>es.</sub> O<sub>n</sub> <sub>eac</sub>h b<sub>ranc</sub>h<sub>,</sub> <sub>a</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>payo</sub>f <sub>can-</sub> <sub>ce</sub>l<sub>s</sub> it<sub>s cer</sub>t<sub>a</sub>i<sub>n sp</sub>litt<sub>er pro</sub>b<sub>a</sub>bilit<sub>y an</sub>d di<sub>scoun</sub>t<sub>, so</sub> th<sub>e</sub> b<sub>ranc</sub>h <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> it<sub>s s</sub>i<sub>gne</sub>d <sub>monom</sub>i<sub>a</sub>l<sub>.</sub> C<sub>ompar</sub>i<sub>ng</sub> thi<sub>s po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>eva</sub>l<sub>ua</sub>t<sub>or w</sub>ith <sub>a po</sub>li<sub>cy</sub> th<sub>a</sub>t <sub>goes</sub> t<sub>o a zero-rewar</sub>d <sub>s</sub>i<sub>n</sub>k d<sub>e-</sub> <sub>c</sub>id<sub>es</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>a</sub> <sub>g</sub>i<sub>ven</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l $f$ i<sub>s</sub> <sub>nonpos</sub>iti<sub>ve</sub> <sub>a</sub>t <sub>every</sub> <sub>p</sub>oint in the uncertaint<sub>y</sub> domain (A<sub>pp</sub>endix B.5).

Portfolio comparison. Portfolio comparison remains ∀R-<sub>com</sub> l<sub>e</sub>t<sub>e even un</sub>d<sub>er</sub> $( s , a )$ <sub>-rec</sub>t<sub>angu</sub>l<sub>ar uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y.</sub> Th<sub>e</sub> pointwise maximum max<sub>π∈Π</sub> $V _ { \pmb { u } } ^ { \pi }$ <sub>over</sub> th<sub>e</sub> <sub>exp</sub>li<sub>c</sub>it <sub>por</sub>tf<sub>o-</sub> li<sub>o, ra</sub>th<sub>er</sub> th<sub>an parame</sub>t<sub>er reuse or recurrence, supp</sub>li<sub>es</sub> th<sub>e</sub> real quantifier alternation (Section B.6).

## 3.3 Regret Certification

C<sub>er</sub>tifi<sub>ca</sub>ti<sub>on</sub> <sub>as</sub>k<sub>s</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub> <sub>a</sub> <sub>g</sub>i<sub>ven</sub> <sub>po</sub>li<sub>cy</sub> <sub>or</sub> <sub>por</sub>tf<sub>o</sub>li<sub>o</sub> <sub>mee</sub>t<sub>s</sub> <sup>a tar</sup>g<sup>et re</sup>g<sup>ret</sup> v<sup>al</sup>u<sup>e</sup>.

Problem 1 (Robust-regret certification). Given an RMDP, a policy $\pi \in \Pi ^ { \mathrm { M R } } ;$ , and a threshold $t \in \mathbb { Q }$ , decide whether $\mathrm { R r e g } ( \pi ) \leq t .$

Problem 2 (Portfolio-regret certification). Given an RMDP, aportfolio ${ \dot { \mathrm { ~ I ~ } } } \subseteq \Pi ^ { \mathrm { M R } }$ , and a threshold $t \in \mathbb { Q } ,$ decide whether $\mathrm { R r e g } ( \Pi ) \leq t .$

We show ∀R-membershi<sub>p</sub> for both <sub>p</sub>roblems:

Theorem 1 (Membership). Regret certification is in ∀R.

Th<sub>e</sub> id<sub>ea</sub> i<sub>s</sub> t<sub>o</sub> <sub>un</sub>i<sub>versa</sub>ll<sub>y</sub> <sub>quan</sub>tif<sub>y</sub> th<sub>e</sub> <sub>rea</sub>li<sub>za</sub>ti<sub>on,</sub> <sub>an</sub> <sub>op-</sub> ti<sub>ma</sub>l <sub>po</sub>li<sub>cy</sub> <sub>a</sub>t th<sub>a</sub>t <sub>rea</sub>li<sub>za</sub>ti<sub>on,</sub> <sub>an</sub>d <sub>one</sub> B<sub>e</sub>ll<sub>man</sub> <sub>sys</sub>t<sub>em</sub> <sub>per</sub> <sub>por</sub>tf<sub>o</sub>li<sub>o</sub> <sub>mem</sub>b<sub>er,</sub> <sub>compar</sub>i<sub>ng</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>va</sub>l<sub>ue</sub> <sub>aga</sub>i<sub>ns</sub>t th<sub>e</sub> b<sub>es</sub>t <sub>o</sub>f th<sub>em.</sub> S<sub>ee</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>

From comparison to regret. The obstacle to transferring <sub>compar</sub>i<sub>son</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> i<sub>s</sub> th<sub>a</sub>t it fi<sub>xes</sub> b<sub>o</sub>th <sub>po</sub>li<sub>c</sub>i<sub>es.</sub> R<sub>egre</sub>t <sub>measures</sub> <sub>a</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>aga</sub>i<sub>ns</sub>t th<sub>e</sub> b<sub>es</sub>t <sub>po</sub>li<sub>cy</sub> <sub>a</sub>t th<sub>e</sub> fi<sub>xe</sub>d realization. For a policy π and portfolio Π, write

$$
\Delta _ { \mathcal { U } } ( \pi , \Pi ) = \operatorname* { s u p } _ { \boldsymbol { u } \in \mathcal { U } } \left( V _ { \boldsymbol { u } } ^ { \pi } ( s _ { \iota } ) - \operatorname* { m a x } _ { \pi ^ { \prime } \in \Pi } V _ { \boldsymbol { u } } ^ { \pi ^ { \prime } } ( s _ { \iota } ) \right) .
$$

For a source RMDP N and reference policy $\pi _ { 0 } , ~ \widehat { N } ~ =$ $\mathrm { L i f t } ( N , \pi _ { 0 } )$ <sub>ma</sub>k<sub>es</sub> $\pi _ { 0 }$ <sub>op</sub>ti<sub>ma</sub>l <sub>a</sub>t <sub>every</sub> <sub>rea</sub>li<sub>za</sub>ti<sub>on</sub> $( \mathsf { A p - }$ <sub>p</sub>endix D). Each source choice becomes a ta<sub>g</sub> state that selects <sub>an ac</sub>ti<sub>on an</sub>d <sub>a se</sub>l<sub>ec</sub>t<sub>or s</sub>t<sub>a</sub>t<sub>e</sub> th<sub>a</sub>t <sub>carr</sub>i<sub>es</sub> it<sub>s</sub> t<sub>rans</sub>iti<sub>on uncer-</sub> taint<sub>y</sub>. For a source <sub>p</sub>ortfolio Π, let Πb be its ima<sub>g</sub>e under Lift. C<sub>er</sub>tifi<sub>ca</sub>ti<sub>on</sub> i<sub>n</sub> $\widehat { N }$ th<sub>en</sub> b<sub>ecomes</sub> <sub>compar</sub>i<sub>son</sub> <sub>aga</sub>i<sub>ns</sub>t $\pi _ { 0 } ,$ up t<sub>o a</sub> k<sub>nown cons</sub>t<sub>an</sub>t $\Lambda \colon { \mathrm { R r e g } } ( \widehat { \Pi } ) = \Lambda + \Delta _ { \mathcal { U } } ( \pi _ { 0 } , \Pi )$

Lower bounds. The following lower bounds combine com-<sub>par</sub>i<sub>son</sub> h<sub>ar</sub>d<sub>ness w</sub>ith th<sub>e</sub> lift <sub>cons</sub>t<sub>ruc</sub>ti<sub>on.</sub>

Theorem 2. Given-policy robust regret is coNP-hard and coSQRS-hard under $( s , a )$ -rectangular uncertainty. coNPhardness already holds when every uncertain choice is two-Dirac and the only other stochastic rows are certain uniform splitters.

Proofsketch. Apply Lift to $\mathrm { C m p } ( \varphi )$ <sub>an</sub>d t<sub>o</sub> th<sub>e s</sub>h<sub>are</sub>d<sub>-cyc</sub>l<sub>e</sub> <sub>square-roo</sub>t<sub>-sum cons</sub>t<sub>ruc</sub>ti<sub>on, respec</sub>ti<sub>ve</sub>l<sub>y.</sub> Th<sub>e</sub> t<sub>rans</sub>f<sub>orma-</sub> ti<sub>on</sub> <sub>conver</sub>t<sub>s</sub> <sub>eac</sub>h <sub>compar</sub>i<sub>son</sub> th<sub>res</sub>h<sub>o</sub>ld b<sub>y</sub> <sub>one</sub> k<sub>nown</sub> <sub>a</sub>d<sub>-</sub> diti<sub>ve cons</sub>t<sub>an</sub>t<sub>.</sub> S<sub>ee</sub> A<sub>ppen</sub>di<sub>x</sub> E<sub>.</sub>1<sub>.</sub> □

Th<sub>e</sub> <sub>po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se</sub> <sub>max</sub>i<sub>mum</sub> <sub>over</sub> <sub>mem</sub>b<sub>ers</sub> <sub>ma</sub>k<sub>es</sub> <sub>por</sub>tf<sub>o</sub>li<sub>o</sub> com<sub>p</sub>arison ∀R-hard even when rectan<sub>g</sub>ularit<sub>y</sub> removes the <sub>correspon</sub>di<sub>ng</sub> <sub>a</sub>lt<sub>erna</sub>ti<sub>on</sub> f<sub>or</sub> <sub>one</sub> <sub>acyc</sub>li<sub>c</sub> <sub>po</sub>li<sub>cy</sub> <sub>pa</sub>i<sub>r.</sub> Th<sub>e</sub> transformation Lift transfers this hardness to certification:

Theorem 3. Portfolio-regret certification is ∀R-hard, already for deterministic portfolios and acyclic $( s , a ) .$ rectangular RMDPs with two-successor uncertain choices.

Proof sketch. Apply Lift to the portfolio-comparison con-<sub>s</sub>t<sub>ruc</sub>ti<sub>on</sub> i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 3<sub>.</sub>2<sub>.</sub> S<sub>ee</sub> A<sub>ppen</sub>di<sub>x</sub> E<sub>.</sub>2<sub>.</sub> □

Theorem 4. Given-policy robust regret is ∀R-complete under general rational polytopic uncertainty, even for deterministic policies.

Proofsketch. Membership is the singleton-portfolio case of Theorem 1. For hardness, reduce from the ∀R-com<sub>p</sub>lete <sub>pro</sub>bl<sub>em o</sub>f d<sub>ec</sub>idi<sub>ng w</sub>h<sub>e</sub>th<sub>er a po</sub>l<sub>ynom</sub>i<sub>a</sub>l i<sub>s nonpos</sub>iti<sub>ve</sub> th<sub>roug</sub>h<sub>ou</sub>t <sub>a</sub> b<sub>oun</sub>d<sub>e</sub>d d<sub>oma</sub>i<sub>n,</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 3<sub>.</sub>2<sub>.</sub> At <sub>eac</sub>h <sub>rea</sub>li<sub>za</sub>ti<sub>on, we</sub> fi<sub>rs</sub>t f<sub>orce</sub> th<sub>e g</sub>i<sub>ven po</sub>li<sub>cy</sub> t<sub>o</sub> t<sub>a</sub>k<sub>e</sub> <sub>a zero-va</sub>l<sub>ue</sub>d b<sub>ranc</sub>h<sub>, w</sub>hil<sub>e an op</sub>ti<sub>ma</sub>l <sub>po</sub>li<sub>cy a</sub>t th<sub>a</sub>t <sub>rea</sub>l<sub>-</sub> i<sub>za</sub>ti<sub>on may c</sub>h<sub>oose e</sub>ith<sub>er</sub> th<sub>a</sub>t b<sub>ranc</sub>h <sub>or one eva</sub>l<sub>ua</sub>ti<sub>ng a</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l f<sub>unc</sub>ti<sub>on.</sub> U<sub>p</sub> t<sub>o</sub> th<sub>e</sub> fi<sub>xe</sub>d i<sub>n</sub>iti<sub>a</sub>l di<sub>scoun</sub>t<sub>,</sub> th<sub>e re-</sub> <sub>gre</sub>t i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> th<sub>e pos</sub>iti<sub>ve par</sub>t <sub>o</sub>f th<sub>e po</sub>l<sub>ynom</sub>i<sub>a</sub>l’<sub>s va</sub>l<sub>ue.</sub> C<sub>onsequen</sub>tl<sub>y,</sub> th<sub>e</sub> <sub>po</sub>li<sub>cy</sub> h<sub>as</sub> <sub>regre</sub>t <sub>a</sub>t <sub>mos</sub>t <sub>zero</sub> <sub>exac</sub>tl<sub>y</sub> <sub>w</sub>h<sub>en</sub> th<sub>e po</sub>l<sub>ynom</sub>i<sub>a</sub>l i<sub>s nonpos</sub>iti<sub>ve.</sub> S<sub>ee</sub> A<sub>ppen</sub>di<sub>x</sub> E<sub>.</sub>3<sub>.</sub> □

## 3.4 Minimal Robust Regret & Bounded Synthesis

C<sub>er</sub>tifi<sub>ca</sub>ti<sub>on</sub> fi<sub>xes</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e se</sub>t <sub>w</sub>h<sub>ereas</sub> i<sub>n</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za-</sub> ti<sub>on an</sub>d <sub>syn</sub>th<sub>es</sub>i<sub>s pro</sub>bl<sub>ems we nee</sub>d t<sub>o c</sub>h<sub>oose</sub> it<sub>.</sub>

Problem 3 (Minimal robust regret). Given an RMDP and a threshold $t \in \mathbb { Q } ,$ decide whether in $\mathrm { f } _ { \pi \in \Pi ^ { \mathrm { M R } } } \operatorname { R r e g } ( \pi ) \leq t .$

Problem 4 (Bounded portfolio synthesis). Given an RMDP, a budget k encoded in unary, and a threshold $t \in \mathbb { Q }$ , decide whether $\rho _ { k } \leq t .$

P<sub>or</sub>tf<sub>o</sub>li<sub>o regre</sub>t i<sub>s mono</sub>t<sub>one un</sub>d<sub>er</sub> i<sub>nc</sub>l<sub>us</sub>i<sub>on, an</sub>d $\mathrm { R r e g } ( \{ \pi \} ) = { \bar { \mathrm { R r e g } } } ( \pi )$ <sub>.</sub> Th<sub>us m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t i<sub>s</sub> th<sub>e</sub> <sub>spec</sub>i<sub>a</sub>l <sub>case o</sub>f b<sub>oun</sub>d<sub>e</sub>d <sub>por</sub>tf<sub>o</sub>li<sub>o syn</sub>th<sub>es</sub>i<sub>s w</sub>ith $\overset { \vartriangle } { \boldsymbol { k } } = 1$

Theorem 5 (Membership). Minimal robust regret and bounded portfolio synthesis belong to $\exists \forall \mathbb { R }$

F<sub>or s</sub>i<sub>ng</sub>l<sub>e-po</sub>li<sub>cy m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on, we ex</sub>i<sub>s</sub>t<sub>en</sub>ti<sub>a</sub>ll<sub>y quan</sub>tif<sub>y</sub> th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>po</sub>li<sub>cy</sub> t<sub>a</sub>bl<sub>e.</sub> F<sub>or</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>syn</sub>th<sub>es</sub>i<sub>s,</sub> <sub>we</sub> <sub>quan-</sub> tif<sub>y</sub> th<sub>e</sub> $k$ <sub>por</sub>tf<sub>o</sub>li<sub>o</sub> t<sub>a</sub>bl<sub>es.</sub> W<sub>e</sub> th<sub>en</sub> <sub>un</sub>i<sub>versa</sub>ll<sub>y</sub> <sub>quan</sub>tif<sub>y</sub> th<sub>e</sub> <sub>rea</sub>li<sub>za</sub>ti<sub>on,</sub> <sub>an</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>po</sub>li<sub>cy</sub> <sub>a</sub>t th<sub>a</sub>t <sub>rea</sub>li<sub>za</sub>ti<sub>on,</sub> <sub>an</sub>d th<sub>e</sub> <sub>cor-</sub> responding Bellman systems. Since k is encoded in unary, b<sub>o</sub>th f<sub>ormu</sub>l<sub>as</sub> h<sub>ave</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>s</sub>i<sub>ze.</sub> S<sub>ee</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>

Restricting the synthesized policy. Synthesis chooses its <sub>can</sub>did<sub>a</sub>t<sub>e,</sub> <sub>so</sub> <sub>a</sub> <sub>re</sub>d<sub>uc</sub>ti<sub>on</sub> <sub>mus</sub>t k<sub>eep</sub> th<sub>e</sub> <sub>syn</sub>th<sub>es</sub>i<sub>ze</sub>d <sub>po</sub>li<sub>cy</sub> i<sub>n</sub> th<sub>e ro</sub>l<sub>e ass</sub>i<sub>gne</sub>d b<sub>y</sub> it<sub>s enco</sub>di<sub>ng, e</sub>ith<sub>er nam</sub>i<sub>ng a va</sub>l<sub>ua</sub>ti<sub>on or</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> th<sub>e cons</sub>i<sub>s</sub>t<sub>ency po</sub>li<sub>cy, ra</sub>th<sub>er</sub> th<sub>an</sub> l<sub>e</sub>tti<sub>ng</sub> it <sub>escape</sub> t<sub>o an un</sub>i<sub>n</sub>t<sub>en</sub>d<sub>e</sub>d l<sub>ower-regre</sub>t <sub>po</sub>li<sub>cy.</sub> Th<sub>e po</sub>li<sub>cy-res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>on</sub> transformation Restrict makes disallowed actions so costly th<sub>a</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng over a</sub>ll <sub>po</sub>li<sub>c</sub>i<sub>es comes w</sub>ithi<sub>n any ra</sub>ti<sub>ona</sub>l $\varepsilon > 0$ of minimizin<sub>g</sub> over the allowed <sub>p</sub>olicies (A<sub>pp</sub>endix F).

## Combinatorial hardness under rectangular uncertainty.

Theorem 6. Minimal robust regret is NP-hard and coNPhard on $( s , a )$ )-rectangular RMDPs in which every uncertain choice is two-Dirac and the only other stochastic rows are certain uniform splitters.

Proofsketch. For coNP-hardness, lift $\mathrm { C m p } ( \varphi )$ <sub>w</sub>ith th<sub>e</sub> <sub>c</sub>l<sub>ause</sub> <sub>po</sub>li<sub>cy</sub> <sub>as</sub> <sub>re</sub>f<sub>erence</sub> <sub>an</sub>d <sub>res</sub>t<sub>r</sub>i<sub>c</sub>t th<sub>e</sub> <sub>syn</sub>th<sub>es</sub>i<sub>ze</sub>d <sub>po</sub>l<sub>-</sub> i<sub>cy</sub> t<sub>o</sub> th<sub>e</sub> lift<sub>e</sub>d <sub>cons</sub>i<sub>s</sub>t<sub>ency</sub> <sub>po</sub>li<sub>cy.</sub> Th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> th<sub>res</sub>h<sub>o</sub>ld separates satisfiable from unsatisfiable formulas. For NPh<sub>ar</sub>d<sub>ness,</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>re</sub>f<sub>erence</sub> fi<sub>n</sub>d<sub>s a</sub> f<sub>a</sub>l<sub>s</sub>ifi<sub>e</sub>d <sub>c</sub>l<sub>ause an</sub>d th<sub>e</sub> <sub>syn</sub>th<sub>es</sub>i<sub>ze</sub>d <sub>po</sub>li<sub>cy</sub> <sub>names</sub> <sub>a</sub> <sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>w</sub>h<sub>ose</sub> l<sub>oca</sub>l <sub>cop</sub>i<sub>es</sub> <sub>na-</sub> t<sub>ure can au</sub>dit<sub>.</sub> Th<sub>e mos</sub>t <sub>pro</sub>b<sub>a</sub>bl<sub>e ran</sub>d<sub>om</sub>i<sub>ze</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ces s</sub>till <sub>name</sub> <sub>a</sub> <sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>a</sub>t l<sub>eas</sub>t $2 ^ { - n }$ <sub>,</sub> <sub>w</sub>hi<sub>c</sub>h <sub>sup-</sub> plies the required gap. The transformations Lift and Restrict <sub>preserve</sub> b<sub>o</sub>th <sub>gaps.</sub> B<sub>o</sub>th <sub>re</sub>d<sub>uc</sub>ti<sub>ons re</sub>t<sub>a</sub>i<sub>n</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t t<sub>wo-</sub> Di<sub>rac c</sub>h<sub>o</sub>i<sub>ces.</sub> S<sub>ee</sub> A<sub>ppen</sub>di<sub>x</sub> F<sub>.</sub> □

Square-root-sum hardness under rectangular uncertainty. Beyond the combinatorial hardness, irrational re-<sub>gre</sub>t <sub>va</sub>l<sub>ues</sub> l<sub>e</sub>t <sub>us enco</sub>d<sub>e pro</sub>bl<sub>ems w</sub>ith <sub>s</sub>i<sub>gne</sub>d <sub>sums o</sub>f <sub>roo</sub>t<sub>s.</sub>

Theorem 7. Minimal robust regret is ${ \mathsf { S Q R S } } _ { \pm }$ -hard under (s, a)-rectangular uncertainty.

Proofsketch. The reduction uses local RMDPs in which the policy chooses a mixing probability x between two actions. F<sub>or su</sub>it<sub>a</sub>bl<sub>e ra</sub>ti<sub>ona</sub>l $A , B , q > 0$ <sub>,</sub> it<sub>s regre</sub>t i<sub>s</sub>

$$
\operatorname* { m a x } \left\{ { \frac { A ( 1 - x ) } { 1 - ( 1 - q ) x } } , { \frac { B x } { q + ( 1 - q ) x } } \right\} .
$$

As x increases the first term decreases while the second i<sub>ncreases, so</sub> th<sub>e</sub>i<sub>r max</sub>i<sub>mum</sub> i<sub>s m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ze</sub>d <sub>a</sub>t <sub>an</sub> i<sub>n</sub>t<sub>er</sub>i<sub>or</sub> b<sub>a</sub>l<sub>-</sub> <sub>ance po</sub>i<sub>n</sub>t<sub>, pro</sub>d<sub>uc</sub>i<sub>ng a square roo</sub>t<sub>.</sub> R<sub>ewar</sub>d <sub>s</sub>i<sub>gns rea</sub>li<sub>ze</sub> th<sub>e</sub> t<sub>wo</sub> f<sub>orms</sub> $D _ { b } - \sqrt { b }$ <sub>an</sub>d $\sqrt { b } - C _ { b }$ <sub>.</sub> A <sub>cer</sub>t<sub>a</sub>i<sub>n un</sub>if<sub>orm</sub> <sub>sp</sub>litt<sub>er a</sub>dd<sub>s</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>m</sub>i<sub>n</sub>i<sub>ma, w</sub>hi<sub>c</sub>h <sub>rea</sub>li<sub>zes a s</sub>i<sub>gne</sub>d <sub>sum.</sub> S<sub>ee</sub> A<sub>ppen</sub>di<sub>x</sub> H<sub>.</sub> □

Hardness under general polytopic uncertainty. Under <sub>genera</sub>l <sub>ra</sub>ti<sub>ona</sub>l <sub>po</sub>l<sub>y</sub>t<sub>op</sub>i<sub>c uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y, po</sub>li<sub>cy compar</sub>i<sub>son re-</sub> d<sub>uces</sub> t<sub>o</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e-po</sub>li<sub>cy</sub> <sub>regre</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on.</sub>

Theorem 8. Single-policy minimal robust regret is ∀R-hard for arbitrary rational polytopic uncertainty.

Proof sketch. Reduce from the ∀R-complete general-<sub>po</sub>l<sub>y</sub>t<sub>ope compar</sub>i<sub>son pro</sub>bl<sub>em</sub> d<sub>escr</sub>ib<sub>e</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 3<sub>.</sub>2<sub>.</sub> Th<sub>e</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> l<sub>eaves</sub> th<sub>e syn</sub>th<sub>es</sub>i<sub>ze</sub>d <sub>po</sub>li<sub>cy one</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on: a</sub> mixing probability x between forced copies of the two source <sub>po</sub>li<sub>c</sub>i<sub>es.</sub> T<sub>wo a</sub>b<sub>sor</sub>bi<sub>ng s</sub>t<sub>a</sub>t<sub>es o</sub>f <sub>va</sub>l<sub>ues</sub> $\pm Z$ <sub>anc</sub>h<sub>or</sub> th<sub>e</sub> b<sub>ranc</sub>h<sub>es so</sub> th<sub>a</sub>t <sub>on</sub>l<sub>y</sub> thi<sub>s m</sub>i<sub>x</sub>i<sub>ng pro</sub>b<sub>a</sub>bilit<sub>y ma</sub>tt<sub>ers.</sub> Th<sub>ey</sub> make its regret max $\{ ( 1 - x ) D , x \bar { E } \}$ , where E is a fixed positi<sub>ve ra</sub>ti<sub>ona</sub>l <sub>an</sub>d $D$ i<sub>s an a</sub>fi<sub>ne, s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> i<sub>ncreas</sub>i<sub>ng</sub> f<sub>unc</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e source compar</sub>i<sub>son va</sub>l<sub>ue</sub> $\Delta .$ <sub>.</sub> Th<sub>e m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng po</sub>li<sub>cy</sub> b<sub>a</sub>l<sub>ances</sub> th<sub>e</sub> t<sub>wo</sub> t<sub>erms, g</sub>i<sub>v</sub>i<sub>ng</sub> $D E / ( D + E )$ <sub>,</sub> <sub>aga</sub>i<sub>n</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> i<sub>ncreas</sub>i<sub>ng</sub> i<sub>n</sub> $\Delta ,$ <sub>, so</sub> th<sub>e compar</sub>i<sub>son</sub> th<sub>res</sub>h<sub>o</sub>ld t<sub>rans</sub>f<sub>ers</sub> b<sub>y a</sub> <sub>ra</sub>ti<sub>ona</sub>l t<sub>rans</sub>f<sub>orma</sub>ti<sub>on.</sub> S<sub>ee</sub> A<sub>ppen</sub>di<sub>x</sub> I<sub>.</sub> □

T<sub>oge</sub>th<sub>er w</sub>ith th<sub>e upper</sub> b<sub>oun</sub>d <sub>a</sub>b<sub>ove,</sub> thi<sub>s p</sub>l<sub>aces s</sub>i<sub>ng</sub>l<sub>e-</sub> <sub>p</sub>olic<sub>y</sub> minimal robust re<sub>g</sub>ret between ∀R-hardness and ∃∀R-<sub>mem</sub>b<sub>ers</sub>hi<sub>p</sub> <sub>un</sub>d<sub>er</sub> <sub>genera</sub>l <sub>ra</sub>ti<sub>ona</sub>l <sub>po</sub>l<sub>y</sub>t<sub>op</sub>i<sub>c</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y.</sub> Th<sub>e</sub> <sub>curren</sub>t b<sub>oun</sub>d<sub>s</sub> d<sub>o</sub> <sub>no</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>comp</sub>l<sub>e</sub>t<sub>eness.</sub>

R<sub>es</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>searc</sub>h t<sub>o</sub> <sub>memory</sub>l<sub>ess</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> <sub>po</sub>li<sub>-</sub> <sub>c</sub>i<sub>es on acyc</sub>li<sub>c</sub> $( s , a )$ <sub>-rec</sub>t<sub>angu</sub>l<sub>ar</sub> RMDP<sub>s ma</sub>k<sub>es m</sub>i<sub>n</sub>i<sub>ma</sub>l ro<sup>b</sup>ust re<sub>g</sub>ret $\Sigma _ { 2 } ^ { p }$ -com<sub>p</sub>lete (A<sub>pp</sub>endix G). This is a diferent <sub>pro</sub>bl<sub>em</sub> f<sub>rom</sub> P<sub>ro</sub>bl<sub>em</sub> 3<sub>, w</sub>hi<sub>c</sub>h <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes over</sub> $\Pi ^ { \mathrm { M R } }$ <sub>,</sub> <sub>an</sub>d <sub>ne</sub>ith<sub>er c</sub>l<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on</sub> i<sub>mp</sub>li<sub>es</sub> th<sub>e o</sub>th<sub>er.</sub> I<sub>n par</sub>ti<sub>cu</sub>l<sub>ar,</sub> th<sub>e</sub> $\Sigma _ { 2 ^ { - } } ^ { p }$ h<sub>ar</sub>d<sub>ness</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> l<sub>eaves</sub> th<sub>e</sub> <sub>syn</sub>th<sub>es</sub>i<sub>ze</sub>d <sub>po</sub>li<sub>cy</sub> <sub>a</sub> f<sub>ree</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>a</sub>t <sub>every</sub> <sub>ex</sub>i<sub>s</sub>t<sub>en</sub>ti<sub>a</sub>l <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> <sub>so</sub> <sub>a</sub> <sub>ran</sub>d<sub>om</sub>i<sub>ze</sub>d <sub>po</sub>li<sub>cy may m</sub>i<sub>x</sub> th<sub>ere an</sub>d <sub>un</sub>d<sub>ercu</sub>t <sub>every</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c can</sub>di<sub>-</sub> date. The NP-hardness of Theorem 6 thus rests on a separate <sub>cons</sub>t<sub>ruc</sub>ti<sub>on, w</sub>h<sub>ose</sub> $2 ^ { - n }$ <sub>gap ran</sub>d<sub>om</sub>i<sub>za</sub>ti<sub>on canno</sub>t <sub>c</sub>l<sub>ose.</sub>

Algorithm 1: Experimental pipeline   
Input: RMDP M, parameter box ${ \overline { { D , } } }$ <sub>por</sub>tf<sub>o</sub>li<sub>o s</sub>i<sub>ze</sub>   
$K ,$ <sub>an</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on see</sub>d<sub>s</sub>   
Construct: split D into cells and compute an optimal   
<sub>m</sub>id<sub>po</sub>i<sub>n</sub>t <sub>po</sub>li<sub>cy</sub> f<sub>or eac</sub>h <sub>ce</sub>ll<sub>;</sub>   
R<sub>o</sub>b<sub>us</sub>tl<sub>y eva</sub>l<sub>ua</sub>t<sub>e every can</sub>did<sub>a</sub>t<sub>e on every ce</sub>ll<sub>,</sub>   
<sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub> one re<sub>g</sub>ret <sub>p</sub>ro<sup>fil</sup>e <sub>p</sub>er <sub>p</sub>o<sup>li</sup>c<sub>y</sub>;   
Cl<sub>us</sub>t<sub>er</sub> th<sub>e pro</sub>fil<sub>es an</sub>d <sub>se</sub>l<sub>ec</sub>t th<sub>e po</sub>li<sub>cy neares</sub>t <sub>eac</sub>h   
of the K centers;   
Evaluate: for 3 seeded samples of $D ,$ <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> <sub>m</sub>i<sub>n</sub>   
<sub>regre</sub>t i<sub>n por</sub>tf<sub>o</sub>li<sub>o, max</sub>i<sub>m</sub>i<sub>ze</sub>d <sub>over samp</sub>l<sub>es;</sub>   
Deploy: treat members as bandit arms and run UCB   
i<sub>n eac</sub>h fi<sub>xe</sub>d <sub>samp</sub>l<sub>e</sub>d <sub>env</sub>i<sub>ronmen</sub>t<sub>;</sub>

Bounded portfolio synthesis. For bounded portfolios, the <sub>genera</sub>l <sub>upper</sub> b<sub>oun</sub>d i<sub>s</sub> ti<sub>g</sub>ht<sub>.</sub>

Theorem 9. Bounded portfolio synthesis is ∃∀R-complete under general rationalpolytopic uncertainty, with k encoded in unary. Hardness holds alreadyfor regret threshold two, the fixed discount $\begin{array} { r } { \gamma = \frac { 1 } { 2 } } \end{array}$ , RMDPs acyclic apart from absorbing final states, and uncertain choices with two successors.

Proofsketch. Membership is the encoding above. For hard-<sub>ness,</sub> <sub>norma</sub>li<sub>ze</sub> $\exists x \forall y \ : \ { F } ( x , y ) \ \geq \ 0$ i<sub>n</sub>t<sub>o</sub> t<sub>es</sub>t<sub>s</sub> $g _ { i } ( x , \eta )$ afine in x, and use one portfolio member per test. A case di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> l<sub>e</sub>t<sub>s na</sub>t<sub>ure au</sub>dit <sub>ro</sub>l<sub>es, w</sub>it<sub>ness agreemen</sub>t<sub>, an</sub>d t<sub>es</sub>t<sub>s</sub> th<sub>roug</sub>h fi<sub>xe</sub>d<sub>-rewar</sub>d <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>eva</sub>l<sub>ua</sub>t<sub>ors.</sub> Th<sub>e regre</sub>t threshold is met exactly when some x passes every test for <sup>ever</sup>y $\eta .$ S<sub>ee</sub> A<sub>ppen</sub>di<sub>x</sub> J<sub>.</sub> □

Th<sub>e equa</sub>lit<sub>y</sub> ti<sub>es coup</sub>l<sub>e c</sub>h<sub>o</sub>i<sub>ces a</sub>t dif<sub>eren</sub>t <sub>s</sub>t<sub>a</sub>t<sub>e-ac</sub>ti<sub>on</sub> <sub>pa</sub>i<sub>rs,</sub> <sub>so</sub> th<sub>e</sub> <sub>re</sub>d<sub>uc</sub>ti<sub>on</sub> d<sub>oes</sub> <sub>no</sub>t <sub>se</sub>ttl<sub>e</sub> th<sub>e</sub> <sub>rec</sub>t<sub>angu</sub>l<sub>ar</sub> <sub>case.</sub> U<sub>n</sub>d<sub>er rec</sub>t<sub>angu</sub>l<sub>ar uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y,</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>syn</sub>th<sub>es</sub>i<sub>s</sub> i<sub>s</sub> h<sub>ar</sub>d <sub>a</sub>l<sub>-</sub> <sub>rea</sub>d<sub>y</sub> f<sub>or</sub> th<sub>e</sub> fi<sub>xe</sub>d b<sub>u</sub>d<sub>ge</sub>t $k = 1$

## 4 Portfolio Construction and Evaluation

W<sub>e s</sub>h<sub>ow</sub> th<sub>a</sub>t <sub>even a s</sub>i<sub>mp</sub>l<sub>e o</sub>fli<sub>ne p</sub>i<sub>pe</sub>li<sub>ne y</sub>i<sub>e</sub>ld<sub>s por</sub>tf<sub>o-</sub> li<sub>os w</sub>ith <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y</sub> l<sub>ower emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>regre</sub>t<sub>,</sub> d<sub>esp</sub>it<sub>e</sub> th<sub>e</sub> i<sub>n</sub>t<sub>rac</sub>t<sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f <sub>por</sub>tf<sub>o</sub>li<sub>o</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>on.</sub> T<sub>o</sub> thi<sub>s</sub> <sub>en</sub>d<sub>,</sub> <sub>we</sub> i<sub>m-</sub> <sub>p</sub>l<sub>emen</sub>t<sub>e</sub>d th<sub>e</sub> th<sub>ree-p</sub>h<sub>ase pro</sub>t<sub>o</sub>t<sub>ype summar</sub>i<sub>ze</sub>d i<sub>n</sub> Al<sub>-</sub> <sub>gor</sub>ith<sub>m</sub> 1<sub>.</sub> T<sub>wo</sub> <sub>researc</sub>h <sub>ques</sub>ti<sub>ons</sub> <sub>gu</sub>id<sub>e</sub> th<sub>e</sub> <sub>exper</sub>i<sub>men</sub>t<sub>s:</sub> <sub>w</sub>h<sub>e</sub>th<sub>er por</sub>tf<sub>o</sub>li<sub>os re</sub>d<sub>uce emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t <sub>as</sub> th<sub>e</sub> b<sub>u</sub>d<sub>-</sub> get grows (RQ1), and whether the best member can be identified online in a fixed but unknown environment (RQ2). All d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> <sub>on</sub> th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s,</sub> <sub>exper</sub>i<sub>men</sub>t <sub>pro</sub>t<sub>oco</sub>l<sub>,</sub> <sub>an</sub>d <sub>resu</sub>lt<sub>s</sub> <sub>can</sub> b<sub>e</sub> f<sub>oun</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> K<sub>.</sub>

Benchmarks. We introduce two new benchmarks: datacenter climate control and UAV control. The datacenter b<sub>enc</sub>h<sub>mar</sub>k <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> t<sub>wo</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub> <sub>parame</sub>t<sub>ers,</sub> <sub>coo</sub>li<sub>ng</sub> <sub>e</sub>f<sub>-</sub> f<sub>ec</sub>ti<sub>veness</sub> <sub>an</sub>d <sub>am</sub>bi<sub>en</sub>t <sub>pressure,</sub> b<sub>o</sub>th <sub>rang</sub>i<sub>ng</sub> <sub>over</sub> <sub>a</sub> b<sub>ox</sub> d<sub>oma</sub>i<sub>n.</sub> Th<sub>e</sub> <sub>rewar</sub>d<sub>s</sub> <sub>com</sub>bi<sub>ne</sub> <sub>a</sub> <sub>per-ac</sub>ti<sub>on</sub> <sub>energy</sub> <sub>cos</sub>t <sub>w</sub>ith <sub>pena</sub>lti<sub>es ac</sub>ti<sub>va</sub>ti<sub>ng near</sub> th<sub>e</sub> t<sub>empera</sub>t<sub>ure,</sub> h<sub>um</sub>idit<sub>y, an</sub>d <sub>queue</sub> <sub>ce</sub>ili<sub>ngs,</sub> <sub>g</sub>i<sub>v</sub>i<sub>ng</sub> $\bar { V } _ { \mathrm { m a x } } = 2 0 8 0$ <sub>.</sub> Th<sub>e</sub> UAV b<sub>enc</sub>h<sub>mar</sub>k i<sub>s</sub> <sub>a</sub> <sub>p</sub>l<sub>ann</sub>i<sub>ng</sub> <sub>pro</sub>bl<sub>em</sub> th<sub>roug</sub>h <sub>a</sub> th<sub>ree-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>gr</sub>id <sub>un-</sub> d<sub>er</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub> <sub>w</sub>i<sub>n</sub>d i<sub>n</sub>t<sub>ens</sub>it<sub>y</sub> $p$ <sub>an</sub>d <sub>ac</sub>t<sub>ua</sub>t<sub>or-</sub>d<sub>rop</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $q ,$ <sub>aga</sub>i<sub>n</sub> <sub>over</sub> <sub>a</sub> b<sub>ox</sub> d<sub>oma</sub>i<sub>n.</sub> R<sub>eac</sub>hi<sub>ng</sub> th<sub>e</sub> <sub>goa</sub>l i<sub>s</sub> th<sub>e</sub> <sub>on</sub>l<sub>y</sub> <sub>rewar</sub>d<sub>e</sub>d <sub>even</sub>t<sub>;</sub> th<sub>us,</sub> th<sub>e va</sub>l<sub>ue</sub> i<sub>s</sub> th<sub>e</sub> di<sub>scoun</sub>t<sub>e</sub>d l<sub>an</sub>di<sub>ng</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y,</sub> <sub>an</sub>d $V _ { \mathrm { m a x } } ~ = ~ 1 0 0$ <sub>.</sub> W<sub>e cons</sub>id<sub>er</sub> th<sub>ree</sub> dif<sub>eren</sub>t instances of this benchmark, uav-small, uav-medium, <sub>an</sub>d $\mathtt { u a v - l a r g e }$ <sub>.</sub> E<sub>very gr</sub>id <sub>pos</sub>iti<sub>on o</sub>f<sub>ers seven ac</sub>ti<sub>ons,</sub> so even uav-small admits $\dot { 7 } ^ { 9 5 }$ <sub>memory</sub>l<sub>ess</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> <sub>po</sub>li<sub>c</sub>i<sub>es, ru</sub>li<sub>ng ou</sub>t <sub>ex</sub>h<sub>aus</sub>ti<sub>ve searc</sub>h<sub>.</sub>

Portfolio construction. The benchmarks are nonrectan-<sub>gu</sub>l<sub>ar</sub> RMDP<sub>s</sub> <sub>w</sub>h<sub>ose</sub> t<sub>rans</sub>iti<sub>on</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es</sub> <sub>are</sub> <sub>a</sub>fi<sub>ne</sub> i<sub>n</sub> <sub>a</sub> parameter vector ranging over a box D. Hence, every valu-<sub>a</sub>ti<sub>on</sub> $\theta \in D$ i<sub>n</sub>d<sub>uces a rea</sub>li<sub>za</sub>ti<sub>on</sub> $\mathbf { \pmb { u } } \in { \mathcal { U } }$ <sub>.</sub> Di<sub>scre</sub>ti<sub>z</sub>i<sub>ng eac</sub>h parameter into 10 bins gives a set $\mathcal { C }$ of 100 cells c for our t<sub>wo-parame</sub>t<sub>er</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s,</sub> <sub>an</sub>d th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>po</sub>li<sub>cy</sub> <sub>a</sub>t <sub>eac</sub>h cell midpoint mid(c) becomes a candidate $\pi \in \Pi ( { \bar { C } } )$ i<sub>n</sub> th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e se</sub>t<sub>.</sub> R<sub>o</sub>b<sub>us</sub>t <sub>po</sub>li<sub>c eva</sub>l<sub>ua</sub>ti<sub>on</sub> th<sub>en ass</sub>i<sub>gns eac</sub>h candidate π a midpoint-based approximate regret vector $L _ { \pi }$ <sub>w</sub>ith <sub>one en</sub>t<sub>ry per ce</sub>ll<sub>:</sub> $L _ { \pi , c } = \dot { V } _ { \mathrm { m i d } ( c ) } ^ { * } ( s _ { \iota } ) - \operatorname* { i n f } _ { \theta \in c } V _ { \theta } ^ { \pi } ( s _ { \iota } )$ W<sub>e</sub> th<sub>en use</sub> K<sub>-means over</sub> th<sub>ese vec</sub>t<sub>ors</sub> t<sub>o se</sub>l<sub>ec</sub>t th<sub>e po</sub>li<sub>cy</sub> <sub>neares</sub>t <sub>eac</sub>h <sub>cen</sub>t<sub>er,</sub> <sub>resu</sub>lti<sub>ng</sub> i<sub>n</sub> <sub>a</sub> <sub>se</sub>t $\Pi _ { K } \subseteq \Pi ( { \mathcal { C } } )$ <sub>o</sub>f <sub>po</sub>li<sub>-</sub> <sub>c</sub>i<sub>es.</sub> Cl<sub>us</sub>t<sub>er</sub>i<sub>ng</sub> i<sub>s a compu</sub>t<sub>a</sub>ti<sub>ona</sub>ll<sub>y c</sub>h<sub>eap approx</sub>i<sub>ma</sub>ti<sub>on</sub> f<sub>or o</sub>bt<sub>a</sub>i<sub>n</sub>i<sub>ng por</sub>tf<sub>o</sub>li<sub>os o</sub>f d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c po</sub>li<sub>c</sub>i<sub>es.</sub>

Portfolio evaluation. A uniform parameter sample $\widehat { D } \subset D$ <sub>w</sub>ith $| \widehat { D } | = 1 0 0 0$ i<sub>s</sub> d<sub>rawn</sub> t<sub>o es</sub>ti<sub>ma</sub>t<sub>e por</sub>tf<sub>o</sub>li<sub>o regre</sub>t b<sub>y</sub>

$$
\widehat { \mathrm { R r e g } } _ { \widehat { D } } ( \Pi ) = \operatorname* { m a x } _ { \theta \in \widehat { D } } \Big ( V _ { \theta } ^ { \ast } \big ( s _ { \iota } \big ) - \operatorname* { m a x } _ { \pi \in \Pi } V _ { \theta } ^ { \pi } \big ( s _ { \iota } \big ) \Big ) ,
$$

<sub>w</sub>hi<sub>c</sub>h <sub>measures coverage w</sub>ith<sub>ou</sub>t <sub>c</sub>h<sub>arg</sub>i<sub>ng</sub> f<sub>or on</sub>li<sub>ne</sub> id<sub>en-</sub> tifi<sub>ca</sub>ti<sub>on an</sub>d <sub>un</sub>d<sub>er-approx</sub>i<sub>ma</sub>t<sub>es</sub> th<sub>e</sub> t<sub>rue regre</sub>t<sub>, s</sub>i<sub>nce</sub> it maximizes over finitel<sub>y</sub> man<sub>y</sub> sam<sub>p</sub>les (cf. Equation (1)). For de<sub>p</sub>lo<sub>y</sub>ment, UCB (Cesa-Bianchi and Lu<sub>g</sub>osi 2006) treats <sub>por</sub>tf<sub>o</sub>li<sub>o mem</sub>b<sub>ers as</sub> b<sub>an</sub>dit <sub>arms w</sub>h<sub>ose re</sub>t<sub>urns come</sub> f<sub>rom</sub> fixed-length trajectories in a fixed but unknown environment. W<sub>e use</sub> it <sub>w</sub>ith <sub>error</sub> $\varepsilon ~ = ~ 0 . 0 0 1$ (as a fraction of the return bound), confidence $\delta = 0 . 1$ <sub>, an</sub>d <sub>emp</sub>i<sub>r</sub>i<sub>ca</sub>l<sub>-</sub>B<sub>erns</sub>t<sub>e</sub>i<sub>n</sub> bounds (Mnih, Sze<sub>p</sub>esvári, and Audibert 2008) to s<sub>p</sub>eed u<sub>p</sub> <sub>convergence.</sub> F<sub>or</sub> th<sub>e</sub> UCB<sub>-</sub>b<sub>ase</sub>d d<sub>ep</sub>l<sub>oymen</sub>t<sub>,</sub> <sub>we</sub> <sub>un</sub>if<sub>orm</sub>l<sub>y</sub> draw 30 valuations and run UCB once per valuation with a trajectory of length $H = 1 0 0$ <sub>samp</sub>l<sub>e</sub>d <sub>on eac</sub>h it<sub>era</sub>ti<sub>on.</sub> W<sub>e p</sub>l<sub>o</sub>t<sub>, per</sub> it<sub>era</sub>ti<sub>on,</sub> th<sub>e</sub> f<sub>rac</sub>ti<sub>on o</sub>f UCB <sub>runs w</sub>h<sub>ere</sub> th<sub>e</sub> fi<sub>na</sub>l <sub>recommen</sub>d<sub>e</sub>d <sub>arm</sub> i<sub>s</sub> th<sub>e</sub> b<sub>es</sub>t <sub>por</sub>tf<sub>o</sub>li<sub>o mem</sub>b<sub>er an</sub>d th<sub>e</sub> dif<sub>erence</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> b<sub>es</sub>t <sub>an</sub>d <sub>recommen</sub>d<sub>e</sub>d <sub>po</sub>li<sub>cy,</sub> normalized between 0 (best) and 1 (worst).

Setup. The Python prototype uses Stormvogel (Volk et al. 2026), robust value iteration (I<sub>y</sub>en<sub>g</sub>ar 2005), and scikit-learn K<sub>-</sub>m<sub>ea</sub>n<sub>s.</sub> E<sub>xpe</sub>rim<sub>e</sub>nt<sub>s we</sub>r<sub>e</sub> r<sub>u</sub>n <sub>o</sub>n <sub>a</sub> 2022 M<sub>ac</sub>B<sub>oo</sub>k Pr<sub>o</sub> M1 <sub>un</sub>d<sub>er</sub> <sub>mac</sub>OS T<sub>a</sub>h<sub>oe</sub> 26<sub>.</sub>5<sub>.</sub>2<sub>,</sub> <sub>us</sub>i<sub>ng</sub> 6 th<sub>rea</sub>d<sub>s</sub> f<sub>or</sub> <sub>ro</sub>b<sub>us</sub>t <sub>po</sub>li<sub>cy</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on, regre</sub>t <sub>approx</sub>i<sub>ma</sub>ti<sub>on, an</sub>d UCB <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub> All <sub>exper</sub>i<sub>men</sub>t<sub>s</sub> <sub>are</sub> d<sub>one</sub> f<sub>or</sub> th<sub>ree</sub> dif<sub>eren</sub>t <sub>see</sub>d<sub>s.</sub> F<sub>or</sub> th<sub>e</sub> <sub>p</sub>l<sub>o</sub>t<sub>s</sub> in Figure 2, we merged all 90 samples across the three seeds.

RQ1: Portfolios reduce robust regret. Table 2 shows empirical robust regret decreasing with K on every benchmark, <sub>w</sub>ith th<sub>e</sub> l<sub>arges</sub>t <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> d<sub>rop</sub> <sub>a</sub>t $K = 2 ,$ , and further gains as K i<sub>ncreases.</sub> Th<sub>e</sub> $K = 1$ <sub>row</sub> i<sub>s a s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on ra</sub>th<sub>er</sub> th<sub>an an a</sub>d<sub>ap-</sub> ti<sub>ve</sub> <sub>por</sub>tf<sub>o</sub>li<sub>o.</sub> S<sub>ee</sub>d<sub>-</sub>l<sub>eve</sub>l <sub>va</sub>l<sub>ues</sub> <sub>an</sub>d K<sub>-means</sub> i<sub>ner</sub>ti<sub>a</sub> <sub>appear</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 5 i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> K<sub>.</sub> F<sub>or</sub> <sub>re</sub>f<sub>erence,</sub> th<sub>e</sub> <sub>m</sub>i<sub>n</sub>i<sub>-max</sub> <sub>regre</sub>t $\mathrm { m i n } _ { \pi \in \Pi ( C ) }$ max<sub>c∈C</sub> $L _ { \pi , c }$ is 0.039, 0.118, 0.146, and 41.502 by column. Policy portfolios beat it for all K on the three l<sub>arges</sub>t b<sub>enc</sub>h<sub>mar</sub>k<sub>s.</sub>

<table><tr><td>K</td><td>UAV-S UAV-M (98) (358)</td><td>UAV-L (1813)</td><td>datacenter (330)</td></tr><tr><td>1</td><td>0.053 0.091</td><td>0.126</td><td>14.28</td></tr><tr><td>2</td><td>0.032 0.019</td><td>0.017</td><td>3.86</td></tr><tr><td>3</td><td>0.018 0.010</td><td>0.016</td><td>2.82</td></tr><tr><td>5</td><td>0.002 0.003</td><td>0.015</td><td>2.79</td></tr><tr><td>7</td><td>0.002 0.003</td><td>0.008</td><td>1.79</td></tr><tr><td>10</td><td>0.002</td><td>0.003 0.003</td><td>0.80</td></tr><tr><td>Construction time</td><td>43.8s</td><td>384.1s 2357.8s</td><td>2732.7s</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 2<sub>:</sub> E<sub>mp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t<sub>, average</sub>d <sub>over</sub> th<sub>ree see</sub>d<sub>s.</sub> P<sub>aren</sub>th<sub>eses g</sub>i<sub>ve</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>s</sub>t<sub>a</sub>t<sub>es.</sub> B<sub>o</sub>tt<sub>om row</sub> i<sub>n</sub>di<sub>ca</sub>t<sub>es</sub> <sub>wa</sub>ll<sub>-c</sub>l<sub>oc</sub>k ti<sub>me.</sub>

![](images/02e61d8b697e07cd24119179cd7af2b852d049d7b75da868f2a3d30f340cab44.jpg)

![](images/42bcd752664e28de478cdcbd3d445cea5780f0bd2829033ed58118420d75daf9.jpg)  
(a) Runs with the correct arm (b) Re<sub>g</sub>ret of recommended arm. <sub>recommen</sub>d<sub>e</sub>d<sub>.</sub>  
Fi<sub>gu</sub>r<sub>e</sub> 2<sub>:</sub> R<sub>epo</sub>rt<sub>e</sub>d UCB d<sub>ep</sub>l<sub>oy</sub>m<sub>e</sub>nt r<sub>esu</sub>lt<sub>s</sub> <sub>o</sub>n th<sub>e</sub> d<sub>a</sub>t<sub>ace</sub>nt<sub>e</sub>r across all tested values of K.

RQ2: Portfolio members can be identified online. At <sub>eac</sub>h it<sub>era</sub>ti<sub>on o</sub>f th<sub>e</sub> UCB <sub>a</sub>l<sub>gor</sub>ith<sub>m, we measure w</sub>hi<sub>c</sub>h arm was <sub>p</sub>ulled (maximal UCB) and which arm was recommended (maximal LCB). Fi<sub>g</sub>ure 2a <sub>g</sub>ives the fraction of UCB <sub>runs recommen</sub>di<sub>ng</sub> th<sub>e</sub> b<sub>es</sub>t <sub>por</sub>tf<sub>o</sub>li<sub>o po</sub>li<sub>cy a</sub>t <sub>eac</sub>h it<sub>era</sub>ti<sub>on.</sub> Id<sub>en</sub>tifi<sub>ca</sub>ti<sub>on s</sub>l<sub>ows as</sub> th<sub>e por</sub>tf<sub>o</sub>li<sub>o grows: a</sub>ft<sub>er</sub> $1 0 ^ { 4 }$ it<sub>era</sub>ti<sub>ons, a</sub>b<sub>ou</sub>t h<sub>a</sub>lf th<sub>e runs recover</sub> th<sub>e</sub> b<sub>es</sub>t <sub>mem</sub>b<sub>er a</sub>t $K = 1 0$ <sub>.</sub> Si<sub>nce</sub> thi<sub>s</sub> i<sub>gnores</sub> h<sub>ow we</sub>ll th<sub>e rema</sub>i<sub>n</sub>i<sub>ng mem</sub>b<sub>ers</sub> <sub>per</sub>f<sub>orm,</sub> Fi<sub>gure</sub> 2b i<sub>ns</sub>t<sub>ea</sub>d <sub>repor</sub>t<sub>s</sub> th<sub>e regre</sub>t <sub>o</sub>f th<sub>e recom-</sub> <sub>men</sub>d<sub>e</sub>d <sub>po</sub>li<sub>cy re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e</sub> b<sub>es</sub>t <sub>por</sub>tf<sub>o</sub>li<sub>o po</sub>li<sub>cy, an</sub>d th<sub>ere</sub> th<sub>e va</sub>l<sub>ue</sub> dif<sub>erence a</sub>ft<sub>er</sub> $1 0 ^ { 4 }$ it<sub>era</sub>ti<sub>ons</sub> i<sub>s m</sub>i<sub>n</sub>i<sub>ma</sub>l<sub>.</sub>

## 5 Conclusion

Ad<sub>ap</sub>ti<sub>ve po</sub>li<sub>cy por</sub>tf<sub>o</sub>li<sub>os s</sub>it b<sub>e</sub>t<sub>ween comm</sub>itti<sub>ng</sub> t<sub>o a s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>ro</sub>b<sub>us</sub>t <sub>po</sub>li<sub>cy an</sub>d <sub>p</sub>l<sub>ann</sub>i<sub>ng over a</sub> f<sub>u</sub>ll b<sub>e</sub>li<sub>e</sub>f <sub>s</sub>t<sub>a</sub>t<sub>e, o</sub>f<sub>er</sub>i<sub>ng</sub> fi<sub>-</sub> nite and certifiable ada<sub>p</sub>tation. Certification is ∀R-com<sub>p</sub>lete and bounded s<sub>y</sub>nthesis ∃∀R-com<sub>p</sub>lete under rational <sub>p</sub>ol<sub>y</sub>- t<sub>op</sub>i<sub>c uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y; cer</sub>tifi<sub>ca</sub>ti<sub>on s</sub>t<sub>ays</sub> h<sub>ar</sub>d <sub>un</sub>d<sub>er rec</sub>t<sub>angu</sub>l<sub>ar-</sub> it<sub>y, s</sub>i<sub>nce</sub> th<sub>e po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se max</sub>i<sub>mum over mem</sub>b<sub>ers re</sub>i<sub>n</sub>t<sub>ro</sub>d<sub>uces</sub> th<sub>e quan</sub>tifi<sub>er a</sub>lt<sub>erna</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>rec</sub>t<sub>angu</sub>l<sub>ar</sub>it<sub>y e</sub>li<sub>m</sub>i<sub>na</sub>t<sub>es.</sub> O<sub>ur</sub> <sub>pro</sub>t<sub>o</sub>t<sub>ype emp</sub>i<sub>r</sub>i<sub>ca</sub>ll<sub>y s</sub>h<sub>ows por</sub>tf<sub>o</sub>li<sub>os w</sub>h<sub>ose regre</sub>t f<sub>a</sub>ll<sub>s</sub> <sub>s</sub>h<sub>arp</sub>l<sub>y w</sub>ith th<sub>e</sub> fi<sub>rs</sub>t f<sub>ew mem</sub>b<sub>ers, a</sub>t <sub>an</sub> id<sub>en</sub>tifi<sub>ca</sub>ti<sub>on cos</sub>t that grows with K. Bounded synthesis under rectangular uncertaint<sub>y</sub>, and the <sub>g</sub>a<sub>p</sub> between ∀R-hardness and ∃∀R-<sub>mem</sub>b<sub>ers</sub>hi<sub>p</sub> f<sub>or s</sub>i<sub>ng</sub>l<sub>e-po</sub>li<sub>cy m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t<sub>, rema</sub>i<sub>n</sub> <sup>o</sup>p<sup>en</sup>.

## References

Ab<sub>a</sub>t<sub>e,</sub> A<sub>.;</sub> B<sub>a</sub>di<sub>ngs,</sub> T<sub>.;</sub> Gi<sub>acomo,</sub> G<sub>.</sub> D<sub>.; an</sub>d F<sub>a</sub>bi<sub>ano,</sub> F<sub>.</sub> 2026<sub>.</sub> B<sub>es</sub>t<sub>-</sub>Ef<sub>or</sub>t P<sub>o</sub>li<sub>c</sub>i<sub>es</sub> f<sub>or</sub> R<sub>o</sub>b<sub>us</sub>t M<sub>ar</sub>k<sub>ov</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> P<sub>ro-</sub> cesses. In AAAI, 36120–36128. AAAI Press.

Ah<sub>me</sub>d<sub>,</sub> A<sub>.;</sub> V<sub>ara</sub>k<sub>an</sub>th<sub>am,</sub> P<sub>.;</sub> Ad<sub>u</sub>l<sub>yasa</sub>k<sub>,</sub> Y<sub>.; an</sub>d J<sub>a</sub>ill<sub>e</sub>t<sub>,</sub> P<sub>.</sub> 2013<sub>.</sub> R<sub>egre</sub>t b<sub>ase</sub>d R<sub>o</sub>b<sub>us</sub>t S<sub>o</sub>l<sub>u</sub>ti<sub>ons</sub> f<sub>or</sub> U<sub>ncer</sub>t<sub>a</sub>i<sub>n</sub> M<sub>ar</sub>k<sub>ov</sub> Decision Processes. In NIPS, 881–889.

Allender, E.; Bürgisser, P.; Kjeldgaard-Pedersen, J.; and Milt<sub>e</sub>r<sub>se</sub>n<sub>,</sub> P<sub>.</sub> B<sub>.</sub> 2009<sub>.</sub> On th<sub>e</sub> C<sub>o</sub>m<sub>p</sub>l<sub>ex</sub>it<sub>y</sub> <sub>o</sub>f N<sub>u</sub>m<sub>e</sub>ri<sub>ca</sub>l An<sub>a</sub>l<sub>ys</sub>i<sub>s.</sub> SIAM J. Comput., 38(5): 1987–2006.

Arora, S.; and Barak, B. 2009. Computational Complexity - A Modern Approach. Cambridge University Press.

Basu, S.; Pollack, R.; and Roy, M.-F. 2006. Algorithms in real algebraic geometry. Springer.

B<sub>e</sub>n<sub>ya</sub>min<sub>e,</sub> A<sub>.;</sub> Gr<sub>a</sub>nd<sub>-</sub>Clém<sub>e</sub>nt<sub>,</sub> J<sub>.;</sub> P<sub>e</sub>trik<sub>,</sub> M<sub>.;</sub> J<sub>o</sub>rd<sub>a</sub>n<sub>,</sub> M<sub>.</sub> I<sub>.;</sub> <sub>an</sub>d D<sub>urmus,</sub> A<sub>.</sub> 2026<sub>.</sub> D<sub>ynam</sub>i<sub>c</sub> P<sub>rogramm</sub>i<sub>ng</sub> f<sub>or</sub> E<sub>p</sub>i<sub>s-</sub> temic Uncertainty in Markov Decision Processes. CoRR, <sub>a</sub>b<sub>s</sub>/2602<sub>.</sub>03381<sub>.</sub>

C<sub>a</sub>nn<sub>y,</sub> J<sub>.</sub> F<sub>.</sub> 1988<sub>.</sub> S<sub>o</sub>m<sub>e</sub> Al<sub>ge</sub>br<sub>a</sub>i<sub>c a</sub>nd G<sub>eo</sub>m<sub>e</sub>tri<sub>c</sub> C<sub>o</sub>m<sub>pu-</sub> tations in PSPACE. In STOC, 460–467. ACM.

Cesa-Bianchi, N.; and Lugosi, G. 2006. Prediction, learning, and games. Cambridge University Press. ISBN 978-0-521- 84108<sub>-</sub>5<sub>.</sub>

Ch<sub>a</sub>d<sub>es,</sub> I<sub>.;</sub> C<sub>arwar</sub>di<sub>ne,</sub> J<sub>.;</sub> M<sub>ar</sub>ti<sub>n,</sub> T<sub>.</sub> G<sub>.;</sub> Ni<sub>co</sub>l<sub>,</sub> S<sub>.;</sub> S<sub>a</sub>bb<sub>a</sub>di<sub>n,</sub> R<sub>.; an</sub>d B<sub>u</sub>f<sub>e</sub>t<sub>,</sub> O<sub>.</sub> 2012<sub>.</sub> MOMDP<sub>s:</sub> A S<sub>o</sub>l<sub>u</sub>ti<sub>on</sub> f<sub>or</sub> M<sub>o</sub>d<sub>e</sub>lli<sub>ng</sub> Adaptive Management Problems. In AAAI, 267–273. AAAI P<sub>ress.</sub>

D<sub>os</sub>hi<sub>-</sub>V<sub>e</sub>l<sub>ez,</sub> F<sub>.; an</sub>d K<sub>on</sub>id<sub>ar</sub>i<sub>s,</sub> G<sub>.</sub> D<sub>.</sub> 2016<sub>.</sub> Hidd<sub>en</sub> P<sub>aram-</sub> <sub>e</sub>t<sub>e</sub>r M<sub>a</sub>rk<sub>ov</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>o</sub>n Pr<sub>ocesses:</sub> A S<sub>e</sub>mi<sub>pa</sub>r<sub>a</sub>m<sub>e</sub>tri<sub>c</sub> R<sub>eg</sub>r<sub>es-</sub> <sub>s</sub>i<sub>on</sub> A<sub>pproac</sub>h f<sub>or</sub> Di<sub>scover</sub>i<sub>ng</sub> L<sub>a</sub>t<sub>en</sub>t T<sub>as</sub>k P<sub>arame</sub>t<sub>r</sub>i<sub>za</sub>ti<sub>ons.</sub> In IJCAI, 1432–1440. IJCAI/AAAI Press.

Gh<sub>avamza</sub>d<sub>e</sub>h<sub>,</sub> M<sub>.;</sub> P<sub>e</sub>t<sub>r</sub>ik<sub>,</sub> M<sub>.; an</sub>d Ch<sub>ow,</sub> Y<sub>.</sub> 2016<sub>.</sub> S<sub>a</sub>f<sub>e</sub> P<sub>o</sub>li<sub>cy</sub> Im<sub>p</sub>r<sub>ove</sub>m<sub>e</sub>nt b<sub>y</sub> Minimi<sub>z</sub>in<sub>g</sub> R<sub>o</sub>b<sub>us</sub>t B<sub>ase</sub>lin<sub>e</sub> R<sub>eg</sub>r<sub>e</sub>t<sub>.</sub> In NIPS, 2298–2306.

G<sub>uez,</sub> A<sub>.;</sub> Sil<sub>ver,</sub> D<sub>.; an</sub>d D<sub>ayan,</sub> P<sub>.</sub> 2012<sub>.</sub> Efi<sub>c</sub>i<sub>en</sub>t B<sub>ayes-</sub>Ad<sub>ap</sub>ti<sub>ve</sub> R<sub>e</sub>inf<sub>o</sub>r<sub>ce</sub>m<sub>e</sub>nt L<sub>ea</sub>rnin<sub>g us</sub>in<sub>g</sub> S<sub>a</sub>m<sub>p</sub>l<sub>e-</sub> Based Search. In NIPS, 1034–1042.

Iyengar, G. N. 2005. Robust Dynamic Programming. Math. Oper. Res., 30(2): 257–280.

J<sub>unges,</sub> S<sub>.;</sub> K<sub>a</sub>t<sub>oen,</sub> J<sub>.;</sub> Pé<sub>rez,</sub> G<sub>.</sub> A<sub>.; an</sub>d Wi<sub>n</sub>kl<sub>er,</sub> T<sub>.</sub> 2021<sub>.</sub> Th<sub>e</sub> <sub>comp</sub>l<sub>ex</sub>it<sub>y</sub> <sub>o</sub>f <sub>reac</sub>h<sub>a</sub>bilit<sub>y</sub> i<sub>n</sub> <sub>parame</sub>t<sub>r</sub>i<sub>c</sub> M<sub>ar</sub>k<sub>ov</sub> d<sub>ec</sub>i<sub>-</sub> sion processes. J. Comput. Syst. Sci., 119: 183–210.

Mnih<sub>,</sub> V<sub>.;</sub> S<sub>zepesv</sub>ári<sub>,</sub> C<sub>.; a</sub>nd A<sub>u</sub>dib<sub>e</sub>rt<sub>,</sub> J<sub>.-</sub>Y<sub>.</sub> 2008<sub>.</sub> Em<sub>-</sub> pirical Bernstein stopping. In Proceedings of the 25th International Conference on Machine Learning, ICML ’08, 672<sub>–</sub>679<sub>.</sub> N<sub>ew</sub> Y<sub>or</sub>k<sub>,</sub> NY<sub>,</sub> USA<sub>:</sub> A<sub>ssoc</sub>i<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> C<sub>ompu</sub>ti<sub>ng</sub> M<sub>ac</sub>hin<sub>e</sub>r<sub>y.</sub> ISBN 9781605582054<sub>.</sub>

Papadimitriou, C. H. 1994. Computational complexity. Addi<sub>son-</sub>W<sub>es</sub>l<sub>ey.</sub>

Puterman, M. L. 1994. Markov Decision Processes: Discrete Stochastic Dynamic Programming. Wiley Series in P<sub>ro</sub>b<sub>a</sub>bilit<sub>y an</sub>d St<sub>a</sub>ti<sub>s</sub>ti<sub>cs.</sub> Wil<sub>ey.</sub>

Ri<sub>g</sub>t<sub>e</sub>r<sub>,</sub> M<sub>.;</sub> L<sub>ace</sub>rd<sub>a,</sub> B<sub>.; a</sub>nd H<sub>awes,</sub> N<sub>.</sub> 2021<sub>.</sub> Minim<sub>ax</sub> R<sub>e-</sub> <sub>gre</sub>t O<sub>p</sub>ti<sub>m</sub>i<sub>sa</sub>ti<sub>on</sub> f<sub>or</sub> R<sub>o</sub>b<sub>us</sub>t Pl<sub>ann</sub>i<sub>ng</sub> i<sub>n</sub> U<sub>ncer</sub>t<sub>a</sub>i<sub>n</sub> M<sub>ar</sub>k<sub>ov</sub> Decision Processes. In AAAI, 11930–11938. AAAI Press.

S<sub>c</sub>h<sub>ae</sub>f<sub>er,</sub> M<sub>.; an</sub>d St<sub>e</sub>f<sub>an</sub>k<sub>ov</sub>i<sub>c,</sub> D<sub>.</sub> 2024<sub>.</sub> B<sub>eyon</sub>d th<sub>e</sub> E<sub>x</sub>i<sub>s-</sub> tential Theory of the Reals. Theory Comput. Syst., 68(2): 195<sub>–</sub>226<sub>.</sub>

S<sub>u</sub>il<sub>en,</sub> M<sub>.;</sub> B<sub>a</sub>di<sub>ngs,</sub> T<sub>.;</sub> B<sub>ovy,</sub> E<sub>.</sub> M<sub>.;</sub> P<sub>ar</sub>k<sub>er,</sub> D<sub>.; an</sub>d J<sub>ansen,</sub> N<sub>.</sub> 2024<sub>.</sub> R<sub>o</sub>b<sub>us</sub>t M<sub>ar</sub>k<sub>ov</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> P<sub>rocesses:</sub> A Pl<sub>ace</sub> Wh<sub>ere</sub> AI and Formal Methods Meet. In Principles of Verification (3), volume 15262 of Lecture Notes in Computer Science, 126<sub>–</sub>154<sub>.</sub> S<sub>pr</sub>i<sub>nger.</sub>

S<sub>u</sub>il<sub>en,</sub> M<sub>.; an</sub>d Pé<sub>rez,</sub> G<sub>.</sub> A<sub>.</sub> 2026<sub>.</sub> O<sub>n</sub> th<sub>e</sub> C<sub>omp</sub>l<sub>ex</sub>it<sub>y o</sub>f R<sub>o-</sub> b<sub>us</sub>t M<sub>ar</sub>k<sub>ov</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>on</sub> P<sub>rocesses an</sub>d Bi<sub>s</sub>i<sub>mu</sub>l<sub>a</sub>ti<sub>on</sub> M<sub>e</sub>t<sub>r</sub>i<sub>cs.</sub> In CONCUR. To appear; arXiv:2604.26748.

Volk, M.; Heck, L.; Jun<sub>g</sub>es, S.; Katoen, J.; and Quatmann, T<sub>.</sub> 2026<sub>.</sub> P<sub>ro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> M<sub>o</sub>d<sub>e</sub>l Ch<sub>ec</sub>ki<sub>ng</sub> T<sub>a</sub>k<sub>en</sub> b<sub>y</sub> St<sub>orm:</sub> A T<sub>u</sub>t<sub>or</sub>i<sub>a</sub>l <sub>on</sub> th<sub>e</sub> P<sub>ro</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> M<sub>o</sub>d<sub>e</sub>l Ch<sub>ec</sub>k<sub>er</sub> St<sub>orm.</sub> I<sub>n</sub> FM, volume 16557 of Lecture Notes in Computer Science, 524<sub>–</sub>549<sub>.</sub> S<sub>pr</sub>i<sub>nger.</sub>

Wi<sub>esemann,</sub> W<sub>.;</sub> K<sub>u</sub>h<sub>n,</sub> D<sub>.; an</sub>d R<sub>us</sub>t<sub>em,</sub> B<sub>.</sub> 2013<sub>.</sub> R<sub>o</sub>b<sub>us</sub>t Markov Decision Processes. Math. Oper. Res., 38(1): 153– 183<sub>.</sub>

## Appendix Contents

A Conventions and primitives 10   
B On Policy and Portfolio Comparison 12   
C Proofs of Theorems 1 and 5: Membership for Certification and Synthesis 21   
D Exact Transfer from Policy Comparison to Robust Regret 21   
E Proofs of Theorems 2 to 4: Transferring Comparison Bounds to Certification 24   
F Proof of Theorem 6: Combinatorial Minimal-Regret Hardness 25   
G Deterministic Minimal-Regret Complexity 29   
H Proof of Theorem 7: Signed Square-Root-Sum Hardness 32   
I Proof of Theorem 8: General-Polytope Minimal-Regret Hardness 36   
J Proof of Theorem 9: Bounded Portfolio Synthesis 38   
K Experimental setup 44   
L Experimental data 48

## A Conventions and primitives

Thi<sub>s a en</sub>di<sub>x co</sub>ll<sub>ec</sub>t<sub>s</sub> th<sub>e conven</sub>ti<sub>ons an</sub>d b<sub>u</sub>ildi<sub>n</sub> bl<sub>oc</sub>k<sub>s</sub> th<sub>a</sub>t th<sub>e</sub> l<sub>a</sub>t<sub>er re</sub>d<sub>uc</sub>ti<sub>ons s</sub>h<sub>are.</sub> Th<sub>ree rou s</sub> f<sub>o</sub>ll<sub>ow.</sub> Fi<sub>rs</sub>t<sub>,</sub> t<sub>wo</sub> <sub>numer</sub>i<sub>ca</sub>l <sub>conven</sub>ti<sub>ons:</sub> <sub>a</sub> di<sub>scoun</sub>t $\gamma _ { 0 }$ <sub>c</sub>h<sub>osen</sub> <sub>so</sub> th<sub>a</sub>t th<sub>e</sub> <sub>compar</sub>i<sub>son-</sub>t<sub>o-regre</sub>t lift <sub>composes</sub> <sub>exac</sub>tl<sub>y,</sub> <sub>an</sub>d <sub>one</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>a</sub>b<sub>ou</sub>t it th<sub>a</sub>t th<sub>e norma</sub>li<sub>ze</sub>d <sub>square-roo</sub>t <sub>ga</sub>d<sub>ge</sub>t <sub>nee</sub>d<sub>s.</sub> S<sub>econ</sub>d<sub>, re</sub>d<sub>uc</sub>ti<sub>on pr</sub>i<sub>m</sub>iti<sub>ves:</sub> h<sub>ow a s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>s g</sub>i<sub>ven a prescr</sub>ib<sub>e</sub>d <sub>va</sub>l<sub>ue,</sub> h<sub>ow</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y en</sub>t<sub>ers a s</sub>i<sub>ng</sub>l<sub>e c</sub>h<sub>o</sub>i<sub>ce,</sub> th<sub>e</sub> t<sub>wo-</sub>Di<sub>rac ga</sub>d<sub>ge</sub>t<sub>s</sub> b<sub>y w</sub>hi<sub>c</sub>h <sub>na</sub>t<sub>ure enco</sub>d<sub>es a</sub> B<sub>oo</sub>l<sub>ean ass</sub>i<sub>gnmen</sub>t <sub>an</sub>d <sub>a ver</sub>ifi<sub>er rea</sub>d<sub>s</sub> it b<sub>ac</sub>k<sub>, an</sub>d <sub>ru</sub>i<sub>nous-s</sub>i<sub>n</sub>k <sub>comp</sub>l<sub>e</sub>ti<sub>on, w</sub>hi<sub>c</sub>h <sub>ma</sub>k<sub>es every c</sub>h<sub>o</sub>i<sub>ce we</sub> did <sub>no</sub>t d<sub>escr</sub>ib<sub>e so cos</sub>tl<sub>y</sub> th<sub>a</sub>t <sub>no po</sub>li<sub>cy ga</sub>i<sub>ns</sub> b<sub>y</sub> t<sub>a</sub>ki<sub>ng</sub> it<sub>.</sub> Thi<sub>r</sub>d<sub>,</sub> <sub>enco</sub>di<sub>ngs:</sub> <sub>a</sub> d<sub>egree-</sub>f<sub>our</sub> <sub>norma</sub>l f<sub>orm</sub> th<sub>a</sub>t t<sub>urns</sub> <sub>a</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t i<sub>n</sub>t<sub>o</sub> <sub>sma</sub>ll <sub>res</sub>id<sub>ua</sub>l<sub>s,</sub> <sub>an</sub>d th<sub>e</sub> B<sub>e</sub>ll<sub>man</sub> f<sub>ormu</sub>l<sub>as</sub> b<sub>e</sub>hi<sub>n</sub>d th<sub>e mem</sub>b<sub>ers</sub>hi<sub>p proo</sub>f<sub>s.</sub> A <sub>rea</sub>d<sub>er may s</sub>ki<sub>p a</sub>h<sub>ea</sub>d <sub>an</sub>d <sub>re</sub>t<sub>urn</sub> h<sub>ere w</sub>h<sub>en a</sub> l<sub>a</sub>t<sub>er cons</sub>t<sub>ruc</sub>ti<sub>on c</sub>it<sub>es one o</sub>f th<sub>ese</sub> b<sub>y name.</sub>

W<sub>e</sub> fi<sub>rs</sub>t fi<sub>x</sub> t<sub>wo</sub> <sub>conven</sub>ti<sub>ons</sub> <sub>use</sub>d th<sub>roug</sub>h<sub>ou</sub>t th<sub>e</sub> <sub>appen</sub>di<sub>x.</sub> S<sub>e</sub>t

$$
\beta = \frac { 1 9 } { 2 0 } , \qquad \gamma _ { 0 } = \beta ^ { 2 } = \frac { 3 6 1 } { 4 0 0 } .
$$

Th<sub>e</sub> id<sub>en</sub>tit<sub>y</sub> $\beta ^ { 2 } = \gamma _ { 0 }$ i<sub>s</sub> <sub>nee</sub>d<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> <sub>exac</sub>t <sub>compar</sub>i<sub>son-</sub>t<sub>o-regre</sub>t lift<sub>,</sub> <sub>w</sub>hil<sub>e</sub>

$$
\left( \frac { 1 + \gamma _ { 0 } + \gamma _ { 0 } ^ { 2 } } { 1 + \gamma _ { 0 } } \right) ^ { 2 } > 2
$$

i<sub>s</sub> th<sub>e numer</sub>i<sub>ca</sub>l i<sub>nequa</sub>lit<sub>y use</sub>d b<sub>y</sub> th<sub>e norma</sub>li<sub>ze</sub>d <sub>square-roo</sub>t <sub>ga</sub>d<sub>ge</sub>t<sub>.</sub> All <sub>appen</sub>di<sub>x cons</sub>t<sub>ruc</sub>ti<sub>ons use</sub> $\gamma _ { 0 }$ <sub>un</sub>l<sub>ess s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>o</sub>th<sub>erw</sub>i<sub>se.</sub> Th<sub>e</sub> b<sub>oun</sub>d<sub>e</sub>d<sub>-syn</sub>th<sub>es</sub>i<sub>s re</sub>d<sub>uc</sub>ti<sub>on uses</sub> $\begin{array} { r } { \gamma = \frac { 1 } { 2 } } \end{array}$ b<sub>ecause</sub> it d<sub>oes no</sub>t i<sub>nvo</sub>k<sub>e</sub> th<sub>e</sub> lift <sub>an</sub>d h<sub>ence</sub> d<sub>oes no</sub>t <sub>requ</sub>i<sub>re</sub> $\beta ^ { 2 } = \gamma _ { 0 }$

## A.1 Reduction primitives

A state is an absorbing final when every described action has a Dirac self-loop. A terminal with payof c is an absorbing final <sub>w</sub>h<sub>ose</sub> d<sub>escr</sub>ib<sub>e</sub>d <sub>ac</sub>ti<sub>on</sub> h<sub>as rewar</sub>d $( 1 - \gamma ) c$ , so its value is exactly c. Every action not separately described at such a terminal is assi ned the same reward and Dirac self-loo . Reachin such a terminal after d transitions with robabilit w contributes $w \gamma ^ { d } c$ t<sub>o</sub> th<sub>e va</sub>l<sub>ue a</sub>t th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e.</sub>

F<sub>or a c</sub>h<sub>o</sub>i<sub>ce</sub> $( s , a )$ <sub>,</sub> <sub>ca</sub>ll th<sub>e</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $\mathbf { \Omega } _ { \pmb { \mathscr { L } } } ( s , a , \cdot )$ its row; the choice is uncertain exactl when its row is not fixed. A selector is a state whose described action has a row carrying one uncertainty coordinate. A splitter is a state whose described action has a fixed row, inde endent of both the realization and the olic ; such a row is certain. A s litter is uniform when its row i uniform over its successors. We reserve ⊥ for an absorbin<sub>g</sub> zero-reward sink, that is, the terminal with <sub>p</sub>a<sub>y</sub>of zero, and ⊥ for a <sub>ru</sub>i<sub>nous s</sub>i<sub>n</sub>k<sub>.</sub>

Definition 3 (Local-bit verifier primitives). For an occurrence o and Boolean value $c _ { \mathrm { * } }$ a local-bit selector ${ { q } _ { o , c } }$ has outcomes $q _ { o , c } ^ { 0 } , q _ { o , c } ^ { 1 }$ and row

$$
\{ ( 1 - p ) \delta _ { q _ { o , c } ^ { 0 } } + p \delta _ { q _ { o , c } ^ { 1 } } : p \in [ 0 , 1 ] \} .\tag{2}
$$

Its vertices encode the bit $b _ { o , c } = p \in \{ 0 , 1 \}$ . A pair testfor an occurrence with designated satisfying value val(o) reads $q _ { o , \mathrm { v a l } ( o ) }$ and then $q _ { o , 1 - \mathrm { v a l } ( o ) }$ . Outcomes $( 1 , 0 )$ certify local truth and $( 0 , 1 )$ certify localfalsity. An audit committed to value c visits ${ { q } _ { o , c } }$ for the relevant occurrences in afixed order, advancing on outcome one and rejecting on outcome zero.

Lemma 10 (Acceptance-diference verifier). Suppose reference and candidate verifier paths are padded to depth H, all nonaccepting payofs are zero, and their accepting terminals have payofs $\gamma ^ { - H }$ and $- \gamma ^ { - \hat { H } }$ . Then their value diference is the sum oftheir acceptance probabilities.

Proof. Every accepting run contributes $\gamma ^ { H } \gamma ^ { - H } = 1$ to the corresponding signed value, while every rejecting run contributes zero. □

A construction’s ruinous-sink completion adds a fresh absorbing state $\perp _ { \mathrm { r } }$ <sub>an</sub>d <sub>a ra</sub>ti<sub>ona</sub>l $Z > 0$ <sub>.</sub> E<sub>very ac</sub>ti<sub>on a</sub>t $\perp _ { \mathrm { r } }$ h<sub>as</sub> <sub>rewar</sub>d $- ( 1 - \gamma ) Z$ <sub>an</sub>d th<sub>e s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on row</sub> $\{ \delta _ { \perp _ { \mathrm { r } } } \}$ , so $V ( \bot _ { \mathrm { r } } ) = - Z$ <sub>.</sub> E<sub>very o</sub>th<sub>erw</sub>i<sub>se un</sub>d<sub>escr</sub>ib<sub>e</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce a</sub>t <sub>a non</sub>t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> h<sub>as rewar</sub>d <sub>zero an</sub>d th<sub>e s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on row</sub> $\{ \delta _ { \perp _ { \mathrm { r } } } \}$ <sub>.</sub> W<sub>r</sub>it<sub>e</sub> $\begin{array} { r } { \dot { V } ^ { \mathrm { b d } ^ { \prime } } = \operatorname* { m a x } _ { s , a } | \dot { R ( s , a ) } | / ( 1 - \gamma ) } \end{array}$ f<sub>or</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d b<sub>oun</sub>d <sub>on va</sub>l<sub>ues,</sub> th<sub>e</sub> maximum ranging over the choices described before completion. Unless stated otherwise, Z is fixed after those rewards, has <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>enco</sub>di<sub>ng</sub> l<sub>eng</sub>th<sub>, an</sub>d <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\gamma Z > V ^ { \mathrm { b d } } + 1$ <sub>.</sub> W<sub>e</sub> i<sub>nvo</sub>k<sub>e</sub> thi<sub>s conven</sub>ti<sub>on</sub> b<sub>y say</sub>i<sub>ng</sub> th<sub>a</sub>t th<sub>e rema</sub>i<sub>n</sub>i<sub>ng c</sub>h<sub>o</sub>i<sub>ces use</sub> <sub>ru</sub>i<sub>nous-s</sub>i<sub>n</sub>k <sub>comp</sub>l<sub>e</sub>ti<sub>on.</sub> Th<sub>e a</sub>dd<sub>e</sub>d <sub>s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>s an a</sub>b<sub>sor</sub>bi<sub>ng</sub> fi<sub>na</sub>l <sub>an</sub>d <sub>a</sub>ll <sub>a</sub>dd<sub>e</sub>d <sub>rows are s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>ons, so</sub> th<sub>e comp</sub>l<sub>e</sub>ti<sub>on preserves</sub> acyclicity, rectangularity, and every two-successor or two-Dirac restriction on uncertain choices. Call a choice ruinous when its <sub>row</sub> i<sub>s</sub> $\{ \bar { \delta } _ { \perp _ { \mathrm { r } } } \}$ <sub>an</sub>d it<sub>s rewar</sub>d i<sub>s zero w</sub>h<sub>e</sub>th<sub>er</sub> th<sub>e com</sub> l<sub>e</sub>ti<sub>on su</sub> li<sub>e</sub>d it <sub>or a</sub> l<sub>a</sub>t<sub>er</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> d<sub>escr</sub>ib<sub>e</sub>d it <sub>ex</sub> li<sub>c</sub>itl <sub>.</sub> C<sub>a</sub>ll <sub>a</sub> policy compliant when it plays a nonruinous action at every state, and write π¯ for the policy obtained from π by conditioning <sub>on</sub> th<sub>e nonru</sub>i<sub>nous ac</sub>ti<sub>ons a</sub>t <sub>every s</sub>t<sub>a</sub>t<sub>e, us</sub>i<sub>ng an ar</sub>bit<sub>rary nonru</sub>i<sub>nous ac</sub>ti<sub>on w</sub>h<sub>ere</sub> th<sub>a</sub>t <sub>con</sub>diti<sub>on</sub>i<sub>ng</sub> h<sub>as zero mass.</sub>

Lemma 11 (Ruinous dominance). Suppose $| V _ { u } ^ { \bar { \rho } } ( s ) | \leq M$ for every compliant policy $\bar { \rho } ,$ every state s other than $\perp _ { \mathrm { r } } ,$ , and every realization $u ,$ and suppose $\gamma Z > M + 1$ . Then $\dot { V _ { u } ^ { \pi } } ( s ) \le V _ { u } ^ { \bar { \pi } } ( s )$ for every stationary randomized policy π, every state, and every realization.

Proof. Both policies have value $- Z { \mathrm { ~ a t ~ } } \bot _ { \mathrm { r } } ,$ so fix any other state s. The action value of a nonruinous choice $( s , a )$ <sub>un</sub>d<sub>er</sub> $V _ { u } ^ { \bar { \pi } }$ is the value at s of the compliant policy that plays a once and follows π¯ afterwards, hence at least −M. A ruinous choice has <sub>ac</sub>ti<sub>on va</sub>l<sub>ue</sub> $\gamma V ( \perp _ { \mathrm { r } } ) = - \bar { \gamma } Z < - M - 1 .$ <sub>, s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y sma</sub>ll<sub>er.</sub> W<sub>r</sub>iti<sub>ng</sub> $T _ { \pi }$ for the Bellman operator of π at $u ,$ <sub>eva</sub>l<sub>ua</sub>ti<sub>ng</sub> b<sub>o</sub>th <sub>po</sub>li<sub>c</sub>i<sub>es a</sub>t $V _ { u } ^ { \bar { \pi } }$ th<sub>ere</sub>f<sub>ore g</sub>i<sub>ves</sub> $T _ { \pi } V _ { u } ^ { \bar { \pi } } \leq T _ { \bar { \pi } } V _ { u } ^ { \bar { \pi } } = \dot { V } _ { u } ^ { \bar { \pi } }$ , since π¯ redistributes $\pi ^ { \prime } s$ <sub>ru</sub>i<sub>nous mass on</sub>t<sub>o s</sub>t<sub>r</sub>i<sub>c</sub>tl l<sub>arger ac</sub>ti<sub>on va</sub>l<sub>ues.</sub> M<sub>ono</sub>t<sub>on</sub>i<sub>c</sub>it<sub>y an</sub>d <sub>con</sub>t<sub>rac</sub>ti<sub>on o</sub>f $T _ { \pi }$ <sub>g</sub><sup>i</sup>ve $V _ { u } ^ { \pi } \leq V _ { u } ^ { \bar { \pi } }$ □

U<sub>n</sub>d<sub>er</sub> th<sub>e</sub> d<sub>e</sub>f<sub>au</sub>lt <sub>ru</sub>l<sub>e,</sub> $M = V ^ { \mathrm { b d } }$ b<sub>y</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d b<sub>oun</sub>d <sub>on</sub> di<sub>scoun</sub>t<sub>e</sub>d <sub>va</sub>l<sub>ues, so</sub> th<sub>e</sub> h<sub>ypo</sub>th<sub>es</sub>i<sub>s</sub> h<sub>o</sub>ld<sub>s.</sub> A <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> th<sub>a</sub>t fi<sub>xes</sub> it<sub>s own cons</sub>t<sub>an</sub>t i<sub>ns</sub>t<sub>ea</sub>d<sub>, suc</sub>h <sub>as</sub> $Z _ { \mathrm { r } }$ i<sub>n</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 19<sub>, ex</sub>hibit<sub>s a</sub> b<sub>oun</sub>d <sub>on</sub> it<sub>s own comp</sub>li<sub>an</sub>t <sub>va</sub>l<sub>ues an</sub>d <sub>c</sub>h<sub>ec</sub>k<sub>s</sub> th<sub>e</sub> i<sub>nequa</sub>lit<sub>y aga</sub>i<sub>ns</sub>t th<sub>a</sub>t<sub>.</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 12 i<sub>s</sub> th<sub>e excep</sub>ti<sub>on:</sub> it <sub>c</sub>h<sub>ooses</sub> $Z _ { \varepsilon }$ t<sub>o mee</sub>t <sub>a quan</sub>tit<sub>a</sub>ti<sub>ve requ</sub>i<sub>remen</sub>t <sub>o</sub>f it<sub>s own an</sub>d <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h<sub>es</sub> d<sub>om</sub>i<sub>nance</sub> di<sub>rec</sub>tl<sub>y</sub> i<sub>n</sub> th<sub>e proo</sub>f <sub>o</sub>f L<sub>emma</sub> 32<sub>.</sub> V<sub>a</sub>l<sub>ue</sub> id<sub>en</sub>titi<sub>es</sub> b<sub>e</sub>l<sub>ow are s</sub>t<sub>a</sub>t<sub>e</sub>d f<sub>or comp</sub>li<sub>an</sub>t <sub>po</sub>li<sub>c</sub>i<sub>es an</sub>d h<sub>o</sub>ld <sub>as upper</sub> b<sub>oun</sub>d<sub>s</sub> i<sub>n genera</sub>l b<sub>y</sub> L<sub>emma</sub> 11<sub>.</sub>

W<sub>e use</sub> f<sub>unc</sub>ti<sub>on-</sub>lik<sub>e no</sub>t<sub>a</sub>ti<sub>on</sub> f<sub>or</sub> th<sub>e</sub> f<sub>our</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>ons</sub> th<sub>a</sub>t <sub>recur</sub> b<sub>e</sub>l<sub>ow.</sub> F<sub>or a</sub> 3<sub>-</sub>CNF f<sub>ormu</sub>l<sub>a</sub> $\varphi$ on n variables, $\mathrm { C m p } ( \varphi )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> B<sub>oo</sub>l<sub>ean compar</sub>i<sub>son</sub> RMDP <sub>o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> $^ { 6 , }$ t<sub>oge</sub>th<sub>er w</sub>ith it<sub>s c</sub>l<sub>ause an</sub>d <sub>cons</sub>i<sub>s</sub>t<sub>ency po</sub>li<sub>c</sub>i<sub>es an</sub>d th<sub>res</sub>h<sub>o</sub>ld $\begin{array} { r } { 2 { \bar { \bf \Delta } } - \frac { { \bar { \bf \Delta } } { \bar { \bf \Delta } } } { 2 n } . } \end{array}$ For a source RMDP N and reference policy $\pi _ { 0 } , \mathrm { L i f t } ( N , \pi _ { 0 } )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e exac</sub>t <sub>re re</sub>t lift <sub>o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 10<sub>, w</sub>ith it<sub>s a</sub>ll<sub>owe</sub>d <sub>ac</sub>ti<sub>on</sub> f<sub>am</sub>il<sub>y spec</sub>ifi<sub>e</sub>d b<sub>y con</sub>t<sub>ex</sub>t<sub>.</sub> $\mathrm { R e s t r i c t } ( N , A _ { \mathrm { a l l o w } } , \varepsilon )$ d<sub>eno</sub>t<sub>es</sub> th<sub>e po</sub>li<sub>cy-res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>on</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 12<sub>.</sub> Fi<sub>na</sub>ll<sub>y,</sub> $\mathrm { R u i n } ( N , Z )$ d<sub>eno</sub>t<sub>es ru</sub>i<sub>nous-s</sub>i<sub>n</sub>k <sub>comp</sub>l<sub>e</sub>ti<sub>on o</sub>f $N$ <sub>w</sub>ith <sub>cons</sub>t<sub>an</sub>t $Z .$

All <sub>cons</sub>t<sub>ruc</sub>ti<sub>ons</sub> b<sub>e</sub>l<sub>ow run</sub> i<sub>n po</sub>l<sub>ynom</sub>i<sub>a</sub>l ti<sub>me, an</sub>d <sub>a</sub>ll th<sub>e ra</sub>ti<sub>ona</sub>l <sub>cons</sub>t<sub>an</sub>t<sub>s a</sub>b<sub>ove</sub> h<sub>ave po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>enco</sub>di<sub>ng</sub> l<sub>eng</sub>th<sub>.</sub> R<sub>u</sub>i<sub>nous-s</sub>i<sub>n</sub>k <sub>comp</sub>l<sub>e</sub>ti<sub>on a</sub>dd<sub>s one s</sub>t<sub>a</sub>t<sub>e an</sub>d $O ( | S | | A | )$ <sub>s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on rows, so</sub> it <sub>preserves po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>s</sub>i<sub>ze.</sub> W<sub>e men</sub>ti<sub>on s</sub>i<sub>ze</sub> b<sub>oun</sub>d<sub>s</sub> <sub>on</sub>l<sub>y w</sub>h<sub>ere</sub> th<sub>ey are no</sub>t i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub> f<sub>rom</sub> th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on.</sub>

## A.2 Degree-Four Residual Normal Form

The bounded-domain equivalence of Schaefer and Stefankovic (2024, Pro<sub>p</sub>osition 2.13) and the de<sub>g</sub>ree-four normal form quoted i<sub>n</sub> th<sub>e</sub>i<sub>r proo</sub>f <sub>o</sub>f L<sub>emma</sub> 2<sub>.</sub>8 <sub>a</sub>ll<sub>ow</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>rea</sub>l <sub>sen</sub>t<sub>ences</sub> t<sub>o use an exp</sub>li<sub>c</sub>itl<sub>y represen</sub>t<sub>e</sub>d <sub>ra</sub>ti<sub>ona</sub>l <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>o</sub>f d<sub>egree a</sub>t <sub>mos</sub>t f<sub>our</sub> <sub>over</sub> $[ 0 , 1 ] ^ { N }$ <sub>.</sub> W<sub>r</sub>it<sub>e</sub> <sub>suc</sub>h <sub>a</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>as</sub>

$$
F ( w ) = \sum _ { \nu = 1 } ^ { N _ { \mathrm { m o n } } } c _ { \nu } \prod _ { j = 1 } ^ { d _ { \nu } } w _ { \nu , j } , \qquad d _ { \nu } \leq 4 , \qquad B = \operatorname* { m a x } \left\{ 1 , \sum _ { \nu = 1 } ^ { N _ { \mathrm { m o n } } } | c _ { \nu } | \right\} .
$$

Th<sub>us</sub> $| F | \le B$ on the box, and B has polynomial encoding length.

Definition 4 (Degree-four residual normal form). For every occurrence $w _ { \nu , j } ,$ introduce a copy $z _ { \nu , j } ~ \in ~ [ 0 , 1 ]$ and residual $h _ { \nu , j } ^ { \mathrm { c p } } = z _ { \nu , j } - w _ { \nu , j }$ . Compute each monomial with auxiliary coordinates in $[ 0 , \dot { 1 } ] .$

$f o r d _ { \nu } = 0 ,$ , set $p _ { \nu } = 1$ , and for $d _ { \nu } = 1 $ , set $p _ { \nu } = z _ { \nu , 1 } ,$

• for d<sub>ν</sub> = 2, use $h _ { \nu } = p _ { \nu } - z _ { \nu , 1 } z _ { \nu , 2 } ;$

• for d<sub>ν</sub> = 3, use t<sub>ν</sub> − z<sub>ν,1</sub>z<sub>ν,2</sub> and $p _ { \nu } - t _ { \nu } z _ { \nu , 3 } ;$

• for d<sub>ν</sub> = 4, use t<sub>ν,12</sub> − z<sub>ν,1</sub>z<sub>ν,2</sub>, t<sub>ν,34</sub> − z<sub>ν,3</sub>z<sub>ν,4</sub>, and $p _ { \nu } - t _ { \nu , 1 2 } t _ { \nu , 3 4 }$

These copy and product residuals lie in $[ - 1 , 1 ]$ , are afine or afine plus one product ofdistinct variables, and have a common zero exactly at correct copies and products. The decoded polynomial is $\widehat { F } = \textstyle \sum _ { \nu } c _ { \nu } p _ { \nu }$ . When an explicit output is needed, introduce $\bar { o } \in [ 0 , 1 ] , p u t o = B ( 2 \bar { o } - 1 )$ , and append $h ^ { \mathrm { o u t } } = ( o - \widehat { F } ) / ( 2 B )$ ; this residual also lies in $[ - 1 , 1 ]$

Lemma 12 (Degree-four residual error). For any assignment to the inputs and auxiliary coordinates, let δ be the largest absolute copy or product residual. Then

$$
\begin{array} { r } { | p _ { \nu } - \prod _ { j } w _ { \nu , j } | \leq 7 \delta \quad \ a n d \quad | \widehat { F } - F | \leq 7 B \delta . } \end{array}
$$

If the output residual is present and $\delta ^ { \prime }$ also includes $| h ^ { \mathrm { o u t } } |$ , then $| o - F | \le 9 B \delta ^ { \prime }$

Proof. For $a , b , a ^ { \prime } , b ^ { \prime } \in [ 0 , 1 ] , | a b - a ^ { \prime } b ^ { \prime } | \leq | a - a ^ { \prime } | + | b - b ^ { \prime } | ,$ , because a $\ - \ a ^ { \prime } b ^ { \prime } = b ( a - a ^ { \prime } ) + a ^ { \prime } ( b - b ^ { \prime } )$ <sub>.</sub> Th<sub>e errors</sub> i<sub>n</sub> de<sub>g</sub>rees zero and one are zero and at most δ. For de<sub>g</sub>ree two the error is at most $\delta + \delta + \delta = 3 \delta$ <sub>.</sub> F<sub>or</sub> d<sub>egree</sub> th<sub>ree</sub> it i<sub>s</sub> <sub>a</sub>t <sub>mos</sub>t $\delta + 3 \delta + \delta = 5 \delta$ <sub>, an</sub>d f<sub>or</sub> d<sub>e ree</sub> f<sub>our</sub> it i<sub>s a</sub>t <sub>mos</sub>t $\delta + 3 \delta + 3 \delta = 7 \delta$ <sub>.</sub> Th<sub>ere</sub>f<sub>ore</sub> $\begin{array} { r } { | \widehat { F } - F | \le 7 \delta \sum _ { \nu } | c _ { \nu } | \le 7 B \delta } \end{array}$ <sub>.</sub> With th<sub>e</sub> <sub>ou</sub>t<sub>pu</sub>t <sub>res</sub>id<sub>ua</sub>l<sub>,</sub> $| o - \widehat { F } | = 2 B | h ^ { \mathrm { o u t } } | \leq 2 B \delta ^ { \prime }$ <sub>, an</sub>d th<sub>e</sub> t<sub>r</sub>i<sub>ang</sub>l<sub>e</sub> i<sub>nequa</sub>lit<sub>y g</sub>i<sub>ves</sub> th<sub>e</sub> l<sub>as</sub>t <sub>c</sub>l<sub>a</sub>i<sub>m.</sub> □

## A.3 Encoding primitives

For an uncertainty vector u, policy table $\sigma ,$ <sub>va</sub>l<sub>ues</sub> $v _ { s }$ <sub>, an</sub>d <sub>ac</sub>ti<sub>on va</sub>l<sub>ues</sub> $q _ { s , a }$ <sub>,</sub> d<sub>e</sub>fi<sub>ne</sub>

$$
\mathrm { B e l l } ( \sigma , u , v , q ) = \bigwedge _ { s } \left( v _ { s } = \sum _ { a } \sigma _ { s , a } q _ { s , a } \right) \wedge \bigwedge _ { s , a } \left( q _ { s , a } = R ( s , a ) + \gamma \sum _ { s ^ { \prime } } u ( s , a , s ^ { \prime } ) v _ { s ^ { \prime } } \right) .
$$

L<sub>e</sub>t $\Phi _ { \mathcal { U } } ( { \boldsymbol { u } } )$ b<sub>e</sub> th<sub>e</sub> <sub>ra</sub>ti<sub>ona</sub>l li<sub>near</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> <sub>po</sub>l<sub>y</sub>t<sub>ope,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>nonnega</sub>ti<sub>v</sub>it<sub>y</sub> <sub>an</sub>d <sub>norma</sub>li<sub>za</sub>ti<sub>on,</sub> <sub>an</sub>d l<sub>e</sub>t

$$
\mathrm { P o l i c y } ( \sigma ) = \bigwedge _ { s } \left( \sum _ { a } \sigma _ { s , a } = 1 \right) \wedge \bigwedge _ { s , a } \sigma _ { s , a } \geq 0 .
$$

Lemma 13 (Universal Bellman encoding). For an explicit family $\Pi = \{ \pi _ { 1 } , \ldots , \pi _ { r } \}$ and a universally quantified policy τ , the assertion

$$
V _ { u } ^ { \tau } ( s _ { \iota } ) - \operatorname* { m a x } _ { i \leq r } V _ { u } ^ { \pi _ { i } } ( s _ { \iota } ) \leq t \quad f o r e \nu e r y u \in \mathcal { U }
$$

has a polynomial-size universalformula over the reals. The policy τ may instead befixed, and prefixing an existential blockfor r valid policy tables gives an existential-universalformula when r is unary-bounded.

Proof. Use the universal implication

$$
\forall \tau , u , v ^ { \tau } , q ^ { \tau } , ( v ^ { i } , q ^ { i } ) _ { i = 1 } ^ { r } : \ \left( \mathrm { P o l i c y } ( \tau ) \wedge \Phi _ { u } ( u ) \wedge \mathrm { B e l l } ( \tau , u , v ^ { \tau } , q ^ { \tau } ) \wedge \bigwedge _ { i = 1 } ^ { r } \mathrm { B e l l } ( \pi _ { i } , u , v ^ { i } , q ^ { i } ) \right) \Longrightarrow \stackrel { \gamma } { \underset { i = 1 } { \overset { r } { \prod } } } ( v _ { s _ { i } } ^ { \tau } - v _ { s _ { i } } ^ { i } \leq t ) .
$$

Di<sub>scoun</sub>ti<sub>n ma</sub>k<sub>es eac</sub>h <sub>va</sub>lid B<sub>e</sub>ll<sub>man s s</sub>t<sub>em un</sub>i <sub>ue.</sub> I<sub>nva</sub>lid <sub>o</sub>li<sub>c , rea</sub>li<sub>za</sub>ti<sub>on, or</sub> B<sub>e</sub>ll<sub>man ass</sub>i <sub>nmen</sub>t<sub>s</sub> f<sub>a</sub>l<sub>s</sub>if th<sub>e an</sub>t<sub>ece</sub>d<sub>en</sub>t<sub>.</sub> Fixing τ removes its policy variables, while existentially quantified family members require their simplex constraints outside th<sub>e un</sub>i<sub>versa</sub>l i<sub>mp</sub>li<sub>ca</sub>ti<sub>on.</sub> C<sub>ompac</sub>t<sub>ness o</sub>f th<sub>e po</sub>li<sub>cy s</sub>i<sub>mp</sub>l<sub>exes an</sub>d <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y po</sub>l<sub>y</sub>t<sub>ope ensures</sub> th<sub>a</sub>t th<sub>e re</sub>l<sub>evan</sub>t <sub>ex</sub>t<sub>rema are</sub> <sub>a</sub>tt<sub>a</sub>i<sub>ne</sub>d<sub>.</sub> □

## B On Policy and Portfolio Comparison

W<sub>e</sub> <sub>co</sub>ll<sub>ec</sub>t th<sub>e</sub> <sub>compar</sub>i<sub>son</sub> <sub>pro</sub>bl<sub>ems</sub> <sub>an</sub>d <sub>cons</sub>t<sub>ruc</sub>ti<sub>ons</sub> <sub>use</sub>d b<sub>y</sub> th<sub>e</sub> <sub>regre</sub>t l<sub>ower</sub> b<sub>oun</sub>d<sub>s.</sub>

## B.1 Comparison Preliminaries

Problem 5 (Robust policy comparison). Given an RMDP M, policies $\pi _ { 1 } , \pi _ { 2 } \in \Pi ^ { \mathrm { M R } }$ , and a rational threshold t, decide whether

$$
\Delta _ { \mathscr { U } } ( \pi _ { 1 } , \pi _ { 2 } ) = \operatorname* { s u p } _ { \pmb { u } \in \mathscr { U } } \bigl ( V _ { \pmb { u } } ^ { \pi _ { 1 } } ( s _ { \iota } ) - V _ { \pmb { u } } ^ { \pi _ { 2 } } ( s _ { \iota } ) \bigr ) \leq t .
$$

A <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> $( s , a )$ is used by $\pi$ if s is reachable with positive probability under some realization and $\pi ( s , a ) > 0$ . It is shared <sup>b</sup><sub>y</sub> $\pi _ { 1 }$ <sub>an</sub>d $\pi _ { 2 }$ if b<sub>o</sub>th <sub>use</sub> it<sub>.</sub> Th<sub>e se</sub>t <sub>o</sub>f <sub>s</sub>h<sub>are</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ces</sub> i<sub>s</sub> d<sub>eno</sub>t<sub>e</sub>d $\mathrm { S h } ( \pi _ { 1 } , \pi _ { 2 } )$ . Call s absorbing under π when every choice <sub>use</sub>d b<sub>y</sub> $\pi$ at s has row $\{ \delta _ { s } \}$ <sub>.</sub> L<sub>e</sub>t $G _ { \pi }$ <sub>con</sub>t<sub>a</sub>i<sub>n</sub> $s \to s ^ { \prime }$ when a choice used by π at s can reach $s ^ { \prime }$ <sub>un</sub>d<sub>er some rea</sub>li<sub>za</sub>ti<sub>on, excep</sub>t that the self-loop at a state absorbing under π is omitted. This exception matches the convention for RMDP acyclicity, which lik<sub>ew</sub>i<sub>se perm</sub>it<sub>s se</sub>lf<sub>-</sub>l<sub>oops on</sub>l<sub>y a</sub>t <sub>a</sub>b<sub>sor</sub>bi<sub>ng</sub> fi<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es.</sub> S<sub>e</sub>lf<sub>-</sub>l<sub>oops a</sub>t <sub>o</sub>th<sub>er s</sub>t<sub>a</sub>t<sub>es rema</sub>i<sub>n one-e</sub>d<sub>ge cyc</sub>l<sub>es.</sub> A <sub>use</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> $( s , a )$ is cycle-free under π if s lies on no directed c cle of $G _ { \pi }$ <sub>.</sub> W<sub>e wr</sub>it<sub>e</sub> $\operatorname { C F } ( \pi )$ f<sub>or</sub> th<sub>ese c</sub>h<sub>o</sub>i<sub>ces.</sub>

Definition 5 (Cycle-free on shared choices). The pair $( \pi _ { 1 } , \pi _ { 2 } )$ is cycle-free on shared choices $i f \mathrm { S h } ( \pi _ { 1 } , \pi _ { 2 } ) \subseteq \mathrm { C F } ( \pi _ { 1 } ) \cap \mathrm { C F } ( \pi _ { 2 } )$

$$
\pi _ { 1 } , \pi _ { 2 }
$$

$$
\in { \mathcal { U } } ,
$$

$$
( P )
$$

$$
D _ { \pi _ { 1 } , \pi _ { 2 } } ( \pmb { u } ) : = V _ { \pmb { u } } ^ { \pi _ { 1 } } ( s _ { \iota } ) - V _ { \pmb { u } } ^ { \pi _ { 2 } } ( s _ { \iota } )
$$

$$
\begin{array} { r } { \Delta _ { \mathcal { U } } ( \pi _ { 1 } , \pi _ { 2 } ) = \operatorname* { s u p } _ { { \pmb u } \in \mathcal { U } } D _ { \pi _ { 1 } , \pi _ { 2 } } ( \pmb u ) } \end{array}
$$

$$
( s , a )
$$

$$
P ,
$$

$$
\begin{array} { r } { \left( \mathcal { U } \right) : = \prod _ { s , a } { \mathrm { v e r t } } ( \mathcal { U } _ { s , a } ) } \end{array}
$$

Separated choices. If $\pi _ { 1 }$ <sub>an</sub>d $\pi _ { 2 }$ <sub>s</sub>h<sub>are</sub> <sub>no</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce,</sub> <sub>rec</sub>t<sub>angu</sub>l<sub>ar</sub>it<sub>y</sub> l<sub>e</sub>t<sub>s</sub> <sub>na</sub>t<sub>ure</sub> <sub>op</sub>ti<sub>m</sub>i<sub>ze</sub> th<sub>e</sub>i<sub>r</sub> <sub>use</sub>d <sub>rows</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y,</sub> so

$$
\Delta _ { { \mathcal U } } ( \pi _ { 1 } , \pi _ { 2 } ) = \operatorname* { s u p } _ { u _ { 1 } } V _ { u _ { 1 } } ^ { \pi _ { 1 } } ( s _ { \iota } ) - \operatorname* { i n f } _ { u _ { 2 } } V _ { u _ { 2 } } ^ { \pi _ { 2 } } ( s _ { \iota } ) .
$$

Thi<sub>s</sub> i<sub>s a</sub> di<sub>rec</sub>t <sub>consequence o</sub>f th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>s</sub>ti<sub>c an</sub>d <sub>ro</sub>b<sub>us</sub>t fi<sub>xe</sub>d<sub>-po</sub>li<sub>cy</sub> li<sub>near programs an</sub>d i<sub>s compu</sub>t<sub>a</sub>bl<sub>e</sub> i<sub>n po</sub>l<sub>ynom</sub>i<sub>a</sub>l time (I<sub>y</sub>en<sub>g</sub>ar 2005; Suilen and Pérez 2026).

## B.2 Combinatorial Hardness of Robust Policy Comparison

Th<sub>e re</sub>d<sub>uc</sub>ti<sub>on ma</sub>k<sub>es one</sub> l<sub>oca</sub>l <sub>copy o</sub>f <sub>every var</sub>i<sub>a</sub>bl<sub>e</sub> i<sub>n every c</sub>l<sub>ause.</sub> O<sub>ne po</sub>li<sub>cy scans</sub> th<sub>e c</sub>l<sub>auses, an</sub>d th<sub>e o</sub>th<sub>er au</sub>dit<sub>s a</sub> <sub>un</sub>if<sub>orm</sub>l<sub>y se</sub>l<sub>ec</sub>t<sub>e</sub>d <sub>var</sub>i<sub>a</sub>bl<sub>e aga</sub>i<sub>ns</sub>t <sub>one va</sub>l<sub>ua</sub>ti<sub>on.</sub>

Theorem 14 (Shared-choice Boolean hardness). Robust policy comparison is coNP-hard even for deterministic policies in acyclic $( s , a )$ -rectangular RMDPs in which every uncertain choice is a two-Dirac segment and the only other stochastic row is a certain uniform splitter.

Fi<sub>x a</sub> 3<sub>-</sub>CNF f<sub>o</sub>rm<sub>u</sub>l<sub>a</sub> $\varphi = { \textstyle \bigwedge } _ { i = 1 } ^ { m } C _ { i }$ <sub>over var</sub>i<sub>a</sub>bl<sub>es</sub> $x _ { 1 } , \ldots , x _ { n }$ <sub>.</sub> W<sub>e may remove</sub> t<sub>au</sub>t<sub>o</sub>l<sub>og</sub>i<sub>ca</sub>l <sub>c</sub>l<sub>auses, repea</sub>t<sub>e</sub>d lit<sub>era</sub>l<sub>s, an</sub>d <sub>var</sub>i<sub>a</sub>bl<sub>es w</sub>ith <sub>no occurrence;</sub> th<sub>e cons</sub>t<sub>an</sub>t <sub>case</sub> i<sub>s</sub> d<sub>ec</sub>id<sub>e</sub>d di<sub>rec</sub>tl<sub>y, so assume</sub> $n \geq 1$ <sub>.</sub> F<sub>or a</sub> lit<sub>era</sub>l <sub>occurrence</sub> $^ { O , }$ let var(o) be its <sub>var</sub>i<sub>a</sub>bl<sub>e,</sub> l<sub>e</sub>t $\mathrm { v a l } ( o ) \in \{ 0 , 1 \}$ b<sub>e</sub> th<sub>e va</sub>l<sub>ue</sub> th<sub>a</sub>t <sub>ma</sub>k<sub>es</sub> it<sub>s</sub> lit<sub>era</sub>l t<sub>rue, an</sub>d <sub>or</sub>d<sub>er</sub> th<sub>e occurrences</sub> $O _ { x }$ of each variable x by clause <sub>an</sub>d th<sub>en</sub> b<sub>y pos</sub>iti<sub>on w</sub>ithi<sub>n</sub> th<sub>e c</sub>l<sub>ause.</sub>

Definition (Clause-local encoding). For every occurrence o, introduce two local bits $b _ { o , 1 }$ <sub>1</sub> and $b _ { o , 0 }$ , indexed by the values they certify. The equation $b _ { o , c } = 1$ says that occurrence o is locally consistent with value c. Occurrence o is locall true when $b _ { o , \mathrm { v a l } ( o ) } = 1$ and $b _ { o , 1 - \mathrm { v a l } ( o ) } = 0$ . Write $\Phi _ { C }$ for the clause condition

$$
\Phi _ { C } : = \bigwedge _ { i = 1 } ^ { m } \left( \bigvee _ { o \in C _ { i } } \left( b _ { o , \mathrm { v a l } ( o ) } \wedge \neg b _ { o , \mathrm { 1 - v a l } ( o ) } \right) \right) .\tag{3}
$$

The local bits of diferent clauses are disjoint.

For every original variable x, introduce a global bit $X _ { x } .$ . For each variable write

$$
\Phi _ { K } ^ { x } : = \bigwedge _ { o \in O _ { x } } b _ { o , X _ { x } } , \qquad \Phi _ { K } : = \bigwedge _ { x } \Phi _ { K } ^ { x } .\tag{4}
$$

Lemma (Clause-local equivalence). Formula φ is satisfiable if its global and local bits have an assignment satisfying $\Phi _ { C } \wedge \Phi _ { K }$

Proof. If a valuation v satisfies φ, set $X _ { x } = v ( x )$ <sub>an</sub>d <sub>se</sub>t $b _ { o , c } = 1$ <sub>exac</sub>tl<sub>y w</sub>h<sub>en</sub> $c = v ( \mathrm { v a r } ( o ) )$ . <sup>E</sup>ver<sub>y</sub> $\Phi _ { K } ^ { x }$ h<sub>o</sub>ld<sub>s.</sub> E<sub>very</sub> <sub>c</sub>l<sub>ause</sub> has an occurrence o with $\mathrm { v a l } ( o ) = v ( \mathrm { v a r } ( o ) )$ <sub>, so</sub> th<sub>a</sub>t <sub>occurrence</sub> h<sub>as</sub> th<sub>e</sub> l<sub>oca</sub>ll<sub>y</sub> t<sub>rue pa</sub>tt<sub>ern an</sub>d $\Phi _ { C }$ h<sub>o</sub>ld<sub>s.</sub>

<sup>Con</sup>v<sup>ersel</sup>y, <sup>s</sup>upp<sup>ose</sup> $\Phi _ { C } \wedge \Phi _ { K }$ h<sub>o</sub>ld<sub>s an</sub>d d<sub>e</sub>fi<sub>ne</sub> $v ( x ) : = X _ { x }$ . Each clause has a locally true occurrence o with $b _ { o , 1 - \mathrm { v a l } ( o ) } = 0$ Si<sub>nce</sub> $\Phi _ { K } ^ { \mathrm { v a r } ( o ) }$ f<sub>orces</sub> $b _ { o , X _ { \mathrm { v a r } ( o ) } } = 1$ <sub>, we</sub> h<sub>ave</sub> $X _ { \mathrm { v a r } ( o ) } ~ = ~ \mathrm { v a l } ( o )$ . Thus the literal at o is true under $v ,$ <sub>an</sub>d <sub>every c</sub>l<sub>ause</sub> i<sub>s</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>e</sub>d<sub>.</sub> □

$\Phi _ { K }$ d<sub>oes no</sub>t f<sub>orce an occurrence</sub>’<sub>s</sub> t<sub>wo</sub> bit<sub>s</sub> t<sub>o</sub> b<sub>e comp</sub>l<sub>emen</sub>t<sub>ary.</sub> Th<sub>e pa</sub>i<sub>r</sub> t<sub>es</sub>t i<sub>s par</sub>t <sub>o</sub>f $\Phi _ { C }$

Example (A clause-local encoding). Consider

$$
\varphi = ( x _ { 1 } \vee \neg x _ { 2 } \vee x _ { 3 } ) \wedge ( \neg x _ { 1 } \vee x _ { 2 } \vee x _ { 3 } ) .
$$

Take the valuation:

$$
x _ { 1 } = x _ { 2 } = 1 , x _ { 3 } = 0 ,
$$

so

$$
X _ { x _ { 1 } } = X _ { x _ { 2 } } = 1 a n d X _ { x _ { 3 } } = 0 .
$$

Following the encoding in the proof above, set $b _ { o , c } = 1$ exactly when $c = X _ { \operatorname { v a r } ( o ) } ,$ , for every occurrence o.

<table><tr><td>Clause</td><td>literal</td><td>var(o)</td><td>val(o)</td><td> $X _ { \mathrm { v a r } ( o ) }$ </td><td> $\left( \boldsymbol { b } _ { o , 1 } , \boldsymbol { b } _ { o , 0 } \right)$ </td><td>locally true?</td></tr><tr><td rowspan="3"> $C _ { 1 }$ </td><td> $x _ { 1 }$ </td><td>x1</td><td>1</td><td>1</td><td>(1,0)</td><td>yes</td></tr><tr><td> $\neg x _ { 2 }$ </td><td>x2</td><td>0</td><td>1</td><td>(1,0)</td><td>no</td></tr><tr><td> $x _ { 3 }$ </td><td>x3</td><td>1</td><td>0</td><td>(0,1)</td><td>no</td></tr><tr><td rowspan="3"> $C _ { 2 }$ </td><td> $\neg x _ { 1 }$ </td><td>x1</td><td>0</td><td>1</td><td>(1,0)</td><td>no</td></tr><tr><td> $x _ { 2 }$ </td><td>x2</td><td>1</td><td>1</td><td>(1,0)</td><td>yes</td></tr><tr><td> $x _ { 3 }$ </td><td>x3</td><td>1</td><td>0</td><td>(0, 1)</td><td>no</td></tr></table>

A literal is locally true exactly when $b _ { o , \mathrm { v a l } ( o ) } = 1 _ { \mathrm { : } }$ , i.e., when va $. ( o ) = X _ { \operatorname { v a r } ( o ) } .$ : this holds for $x _ { 1 }$ in $C _ { 1 }$ and $x _ { 2 }$ in $C _ { 2 }$ , matching the fact that these are the literals true under the valuation. Using the labels $O _ { 1 } , \ldots , O _ { 6 }$ , the clause and consistency conditions instantiate to

$$
\Phi _ { C } = \left[ \left( b _ { o _ { 1 } , 1 } \wedge - b _ { o _ { 1 } , 0 } \right) \vee \left( b _ { o _ { 2 } , 0 } \wedge - b _ { o _ { 2 } , 1 } \right) \vee \left( b _ { o _ { 3 } , 1 } \wedge - b _ { o _ { 3 } , 0 } \right) \right] \wedge \left[ \left( b _ { o _ { 4 } , 0 } \wedge - b _ { o _ { 4 } , 1 } \right) \vee \left( b _ { o _ { 5 } , 1 } \wedge - b _ { o _ { 5 } , 0 } \right) \vee \left( b _ { o _ { 6 } , 1 } \wedge - b _ { o _ { 6 } , 0 } \right) \right] ,
$$

$$
\Phi _ { K } ^ { x _ { 1 } } = b _ { o _ { 1 } , 1 } \wedge b _ { o _ { 4 } , 1 } , \qquad \Phi _ { K } ^ { x _ { 2 } } = b _ { o _ { 2 } , 1 } \wedge b _ { o _ { 5 } , 1 } , \qquad \Phi _ { K } ^ { x _ { 3 } } = b _ { o _ { 3 } , 0 } \wedge b _ { o _ { 6 } , 0 } , \qquad \Phi _ { K } = \Phi _ { K } ^ { x _ { 1 } } \wedge \Phi _ { K } ^ { x _ { 2 } } \wedge \Phi _ { K } ^ { x _ { 3 } } .
$$

Substituting the values from the table: in $\Phi _ { C } \mathbf { \ ' } _ { s }$ first bracket, o<sub>1</sub> gives $( 1 \land 1 ) = 1$ while $o _ { 2 } , o _ { 3 } \ g i \nu e \ 0 ,$ so the bracket is 1; in the second, o<sub>5</sub> gives $( 1 \land \dot { 1 } ) = 1$ while $O _ { 4 } , O _ { 6 }$ give 0, so that bracket is also 1; hence $\Phi _ { C } = 1$ . Each $\Phi _ { K } ^ { x }$ is a conjunction of two matching bits $( e . g . \Phi _ { K } ^ { x _ { 1 } } = 1 \wedge 1 = 1 )$ , so $\Phi _ { K } = 1$ as well.

Th<sub>e re</sub>d<sub>uc</sub>ti<sub>on rea</sub>li<sub>zes</sub> $\Phi _ { C }$ <sub>as a c</sub>l<sub>ause scan an</sub>d <sub>eac</sub>h $\Phi _ { K } ^ { x }$ <sub>as one au</sub>dit b<sub>ranc</sub>h i<sub>n</sub> th<sub>e same</sub> RMDP<sub>.</sub> N<sub>a</sub>t<sub>ure se</sub>l<sub>ec</sub>t<sub>s</sub> th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>an</sub>d l<sub>oca</sub>l bit<sub>s.</sub> Th<sub>e</sub> t<sub>wo pa</sub>th<sub>s s</sub>h<sub>are every</sub> l<sub>oca</sub>l<sub>-</sub>bit <sub>c</sub>h<sub>o</sub>i<sub>ce, so</sub> th<sub>ey eva</sub>l<sub>ua</sub>t<sub>e</sub> th<sub>e same ass</sub>i<sub>gnmen</sub>t<sub>.</sub>

Definition 6 (Boolean comparison RMDP). For a 3-CNF formula $\varphi = { \textstyle \bigwedge } _ { i = 1 } ^ { m } C _ { i }$ over variables $x _ { 1 } , \ldots , x _ { n }$ with $n \geq 1$ , no tautological clause, no repeated literal within a clause, and no variable absent from every clause, the Boolean comparison RMDP is $M _ { \varphi } = ( S , A , \mathcal { U } , R , s _ { \iota } , \gamma _ { 0 } )$ , with thefollowing components.

• S is the disjoint union of the following groups.

– The initial state $s _ { \iota } .$

– Clause-scan states: a control state $f _ { i , j }$ for each literal occurrence $o = ( i , j )$ , together with the terminals acc and $\mathrm { r e j } _ { C } .$

– Audit states: global selectors q<sub>X</sub> with outcomes $q _ { X _ { x } } ^ { 0 } , q _ { X _ { x } } ^ { 1 }$ , controls $k _ { x , c , \ell }$ that inspect the ℓth occurrence of x after committing to $c \in \{ 0 , 1 \}$ , together with the terminals acc<sub>K</sub> and $\mathrm { r e j } _ { K }$

– Shared local-bit selectors:for each occurrence o and $c \in \{ 0 , 1 \}$ , the selector ofDefinition 3, used by both policies.

– Padding states $S _ { \mathrm { p a d } }$ , reward-free,fixed once the transitions below are defined.

– A ruinous sink $\perp _ { \mathrm { r } } .$

![](images/aa585d0a57f0bf9c7bf32784df82073c956c5b9676f6d3493d9591f74a699c57.jpg)  
Fi<sub>gure</sub> 3<sub>:</sub> Th<sub>e s</sub>h<sub>are</sub>d f<sub>ragmen</sub>t f<sub>or a pos</sub>iti<sub>ve occurrence</sub> $o = ( i , j )$ <sub>an</sub>d th<sub>e au</sub>dit b<sub>ranc</sub>h $X _ { x } = 1$ <sub>.</sub> Th<sub>e c</sub>l<sub>ause scan</sub> t<sub>es</sub>t<sub>s</sub> th<sub>e</sub> <sub>p</sub>a<sup>i</sup>r $\left( \boldsymbol { b } _ { o , 1 } , \boldsymbol { b } _ { o , 0 } \right)$ <sub>, w</sub>h<sub>ereas</sub> thi<sub>s au</sub>dit b<sub>ranc</sub>h <sub>nee</sub>d<sub>s on</sub>l<sub>y</sub> $b _ { o , 1 }$ <sub>.</sub> Th<sub>e</sub> b<sub>ranc</sub>h $X _ { x } = 0$ <sub>ana</sub>l<sub>o ous</sub>l <sub>au</sub>dit<sub>s</sub> ${ { q } _ { o , 0 } }$ <sub>.</sub> S<sub>e</sub>l<sub>ec</sub>t<sub>or arrows are</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n; arrows</sub> f<sub>rom ou</sub>t<sub>come s</sub>t<sub>a</sub>t<sub>es are</sub> th<sub>e cer</sub>t<sub>a</sub>i<sub>n, ro</sub>l<sub>e-spec</sub>ifi<sub>c ac</sub>ti<sub>ons o</sub>f th<sub>e</sub> t<sub>wo po</sub>li<sub>c</sub>i<sub>es.</sub>

• At $\boldsymbol { s } _ { t } ,$ , action $a _ { C }$ enters $f _ { 1 , 1 } ,$ , whereas action $a _ { K }$ has the fixed uniform transition to $q _ { X _ { x } } , x \in \{ x _ { 1 } , \ldots , x _ { n } \}$ . Every selector has one action. Outcome $q _ { X _ { x } } ^ { c }$ leads to $k _ { x , c , 1 }$ , each $f _ { i , j }$ leads to the first local selector for its occurrence, and each $k _ { x , c , \ell }$ leads to the corresponding ${ q _ { o , c } } .$ . From a shared outcome $q _ { o , c } ^ { b } ,$ , the clause-scan action and audit action have their respective deterministic continuations, ending in ac $\mathrm { c }_ { C } / \mathrm { r e j } _ { C }$ or acc $\kappa / \mathrm { r e j } _ { K }$ as described below. All other control states have the indicated unique deterministic continuation. Fix H at least as long as the longest ofthese paths. $S _ { \mathrm { p a d } }$ consists offresh states inserted along every shorter path so it also reaches its terminal after exactly H transitions, each with a single reward-free action continuing toward that terminal. This common depth is necessary because a shared accepting terminal can be reached at realization-dependent depths, so one terminal payof cannot otherwise cancel every discount factor.

• U is the product of the two-Dirac segments in Equation (2), one for every global bit $X _ { x }$ and local bit $( o , c )$ . All remaining described choices are singletons, including the certain uniform splitter at $\boldsymbol { s } _ { \iota }$ . Its vertices are exactly the Boolean realizations $p _ { z } \in \{ 0 , 1 \}$ .

• acc is the terminal with payof $\gamma _ { 0 } ^ { - H }$ and acc the terminal with $p a y o f f - \gamma _ { 0 } ^ { - H }$ , while every other described reward, including at $\mathrm { r e j } _ { C } , \mathrm { r e j } _ { K }$ and at padding states, is zero.

• All remaining choices use ruinous-sink completion.

• The initial state is $\boldsymbol { s } _ { \iota }$ and the discount is γ<sub>0</sub>.

Fi<sub>x a rea</sub>li<sub>za</sub>ti<sub>on</sub> $u \in \mathcal { U }$ <sub>an</sub>d <sub>a s</sub>t<sub>a</sub>ti<sub>onary po</sub>li<sub>cy</sub> $\pi .$ Since π prescribes one action at every state and u fixes one outcome at every two-Dirac selector, the only randomness left in π’s run under u comes from any fixed-probability transitions π itself uses, such as the certain uniform splitter. Because the graph is ac<sub>y</sub>clic and S is finite, this run reaches a terminal with probabilit<sub>y</sub> one. Say π accepts under u with probability equal to the chance of reaching an accepting terminal; $\mathrm { i f } \ \pi$ <sub>never uses suc</sub>h <sub>a</sub> t<sub>rans</sub>iti<sub>on</sub> <sub>– as</sub> i<sub>s</sub> th<sub>e case</sub> f<sub>or</sub> $\pi _ { C } -$ this probability is always 0 or 1, and we speak of π accepting or rejecting outright.

The clause policy. Policy $\pi _ { C }$ <sub>processes c</sub>l<sub>auses an</sub>d th<sub>e</sub>i<sub>r</sub> lit<sub>era</sub>l<sub>s</sub> i<sub>n or</sub>d<sub>er.</sub> At <sub>occurrence</sub> $o = ( i , j )$ <sub>,</sub> it <sub>uses</sub> th<sub>e pa</sub>i<sub>r</sub> t<sub>es</sub>t <sub>o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> $^ { 3 , }$ <sub>accep</sub>ti<sub>ng</sub> th<sub>e</sub> l<sub>oca</sub>ll<sub>y</sub> t<sub>rue ou</sub>t<sub>come pa</sub>i<sub>r.</sub> A t<sub>rue</sub> lit<sub>era</sub>l <sub>a</sub>d<sub>vances</sub> t<sub>o</sub> th<sub>e nex</sub>t <sub>c</sub>l<sub>ause, a</sub> f<sub>a</sub>l<sub>se</sub> lit<sub>era</sub>l <sub>a</sub>d<sub>vances</sub> t<sub>o</sub> th<sub>e</sub> next literal, and an entirely false clause rejects. Satisfying the last clause accepts. Thus $\pi _ { C }$ <sub>c</sub>h<sub>ec</sub>k<sub>s exac</sub>tl<sub>y</sub> $\Phi _ { C }$

The consistency policy. Policy $\pi _ { K }$ first takes the certain uniform splitter, which selects a variable x. It reads $X _ { x }$ <sub>an</sub>d th<sub>ere</sub>b<sub>y</sub> <sub>comm</sub>it<sub>s</sub> t<sub>o</sub> $c \in \{ 0 , 1 \}$ <sub>.</sub> It th<sub>en uses</sub> th<sub>e au</sub>dit <sub>c</sub>h<sub>a</sub>i<sub>n o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 3 <sub>over</sub> $O _ { x }$ in clause-major order. Conditioned on the certain uniform splitter selecting x, $\pi _ { K }$ th<sub>ere</sub>f<sub>ore c</sub>h<sub>ec</sub>k<sub>s exac</sub>tl<sub>y</sub> $\Phi _ { K } ^ { x }$

Lemma (Unambiguous prescriptions). Both policies above are stationary deterministic policies.

Proof. The clause scan encounters each $f _ { i , j }$ <sub>once an</sub>d <sub>reac</sub>h<sub>es</sub> th<sub>e</sub> t<sub>wo se</sub>l<sub>ec</sub>t<sub>ors o</sub>f <sub>an occurrence</sub> i<sub>n</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>or</sub>d<sub>er prescr</sub>ib<sub>e</sub>d <sub>a</sub>b<sub>ove.</sub> A<sub>n</sub> <sub>au</sub>dit b<sub>ranc</sub>h <sub>comm</sub>it<sub>s</sub> <sub>a</sub>t $q _ { X _ { a } } ^ { c }$ b<sub>e</sub>f<sub>ore</sub> <sub>reac</sub>hi<sub>ng</sub> <sub>any</sub> $k _ { x , c , \ell }$ <sub>.</sub> Th<sub>us eac</sub>h <sub>po</sub>li<sub>cy prescr</sub>ib<sub>es a s</sub>i<sub>ng</sub>l<sub>e con</sub>ti<sub>nua</sub>ti<sub>on a</sub>t <sub>every</sub> <sub>ou</sub>t<sub>come s</sub>t<sub>a</sub>t<sub>e</sub> it <sub>can reac</sub>h<sub>.</sub> At <sub>a s</sub>h<sub>are</sub>d <sub>ou</sub>t<sub>come</sub> $q _ { o , c } ^ { b } ,$ th<sub>e</sub> t<sub>wo ro</sub>l<sub>es use</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>ac</sub>ti<sub>ons, so</sub> th<sub>e</sub>i<sub>r con</sub>ti<sub>nua</sub>ti<sub>ons nee</sub>d <sub>no</sub>t <sub>agree.</sub> C<sub>omp</sub>l<sub>e</sub>t<sub>e eac</sub>h <sub>po</sub>li<sub>cy a</sub>t it<sub>s unreac</sub>h<sub>a</sub>bl<sub>e s</sub>t<sub>a</sub>t<sub>es w</sub>ith <sub>a</sub> fi<sub>xe</sub>d d<sub>e</sub>f<sub>au</sub>lt <sub>ac</sub>ti<sub>on, us</sub>i<sub>ng ru</sub>i<sub>nous-s</sub>i<sub>n</sub>k <sub>comp</sub>l<sub>e</sub>ti<sub>on w</sub>h<sub>en no ro</sub>l<sub>e ac</sub>ti<sub>on</sub> <sub>was</sub> d<sub>escr</sub>ib<sub>e</sub>d th<sub>ere.</sub> □

B<sub>y</sub> L<sub>emma</sub> 10<sub>,</sub> th<sub>e</sub>i<sub>r va</sub>l<sub>ue</sub> dif<sub>erence</sub> i<sub>s</sub> th<sub>e sum o</sub>f th<sub>e</sub>i<sub>r accep</sub>t<sub>ance pro</sub>b<sub>a</sub>biliti<sub>es.</sub>

Th<sub>e</sub> <sub>s</sub>h<sub>are</sub>d f<sub>ragmen</sub>t f<sub>or</sub> <sub>a</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>occurrence</sub> i<sub>s</sub> <sub>s</sub>h<sub>own</sub> i<sub>n</sub> Fi<sub>gure</sub> 3<sub>.</sub> Th<sub>e</sub> <sub>au</sub>dit <sub>pa</sub>th <sub>s</sub>h<sub>own</sub> i<sub>s</sub> th<sub>e</sub> b<sub>ranc</sub>h th<sub>a</sub>t <sub>comm</sub>itt<sub>e</sub>d t<sub>o</sub> $X _ { x } = 1$

Lemma (Policy semantics). At a vertex u of the uncertainty polytope, $\pi _ { C }$ accepts under u exactly when $\Phi _ { C }$ holds under u. Conditioned on the certain uniform splitter selecting x, π<sub>K</sub> accepts under u exactly when $\Phi _ { K } ^ { x }$ holds under u; unconditionally, this makes $\pi _ { K } \dot { s }$ probability of accepting under u equal $t o \ { \frac { 1 } { n } } | \{ x : \Phi _ { K } ^ { x }$ holds under $\mathbf { \delta } _ { \mathbf { \pmb { u } } } \}$ |, the fraction of variables whose audit would pass. Moreover,

$$
D _ { \pi _ { C } , \pi _ { K } } ( \pmb { u } ) = \mathbf { 1 } _ { \left\{ \pi _ { C } ~ a c c e p t s ~ u n d e r ~ \pmb { u } \right\} } + \mathrm { P r } [ \pi _ { K } ~ a c c e p t s ~ u n d e r ~ \pmb { u } ] .
$$

Proof. At a vertex, each transition in (2) selects one Boolean value. The clause path implements (3), and the audit branch for x implements its conjunct in (4). Padding places every final state at depth H, so the accepting rewards contribute 1 and −1 to the two policy values. The certain uniform splitter averages the audit contribution over the n variables, and rejecting paths <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e zero.</sub> □

Example. For the satisfying assignment in the encoding table, π<sub>C</sub> confirms $x _ { 1 }$ i<sub>n</sub> $C _ { 1 }$ <sub>an</sub>d $x _ { 2 }$ i<sub>n</sub> $C _ { 2 }$ <sub>, an</sub>d <sub>every var</sub>i<sub>a</sub>bl<sub>e au</sub>dit passes, giving diference 2. Now take the unsatisfying valuation $x _ { 1 } = 0 , x _ { 2 } = 1 , x _ { 3 } = 0$ <sub>an</sub>d it<sub>s canon</sub>i<sub>ca</sub>l l<sub>oca</sub>l <sub>pa</sub>i<sub>rs.</sub> Th<sub>e</sub> clause scan rejects $C _ { 1 }$ <sub>, a</sub>lth<sub>oug</sub>h <sub>every au</sub>dit <sub>passes.</sub> If <sub>on</sub>l<sub>y</sub> th<sub>e copy o</sub>f $x _ { 1 }$ i<sub>n</sub> $C _ { 1 }$ is changed from (0, 1) to (1, 0), the clause scan <sub>accep</sub>t<sub>s,</sub> b<sub>u</sub>t th<sub>e</sub> $x _ { 1 }$ audit, committed to 0, reads $b _ { o , 0 } = 0$ and rejects. The malformed pairs (0, 0) and (1, 1) cannot make a literal l<sub>oca</sub>ll<sub>y</sub> t<sub>rue</sub> b<sub>ecause</sub> th<sub>e c</sub>l<sub>ause scan</sub> t<sub>es</sub>t<sub>s</sub> b<sub>o</sub>th bit<sub>s.</sub>

Lemma 15. There is a vertex under which π<sub>C</sub> accepts and every variable audit passes ifand only $i f \varphi$ is satisfiable.

Proof. This is the clause-local equivalence, together with the policy semantics established above.

□

Th<sub>e uncer</sub>t<sub>a</sub>i<sub>n c</sub>h<sub>o</sub>i<sub>ces are</sub> i<sub>n</sub>d<sub>e en</sub>d<sub>en</sub>t li<sub>ne se men</sub>t<sub>s</sub> b<sub>e</sub>t<sub>ween</sub> t<sub>wo</sub> Di<sub>rac</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons.</sub> H<sub>ence</sub> th<sub>e</sub> RMDP i<sub>s</sub> $( s , a )$ -rectan<sub>g</sub>u<sup>l</sup>ar <sub>an</sub>d <sub>every uncer</sub>t<sub>a</sub>i<sub>n c</sub>h<sub>o</sub>i<sub>ce</sub> i<sub>s</sub> t<sub>wo-</sub>Di<sub>rac.</sub> Th<sub>e on</sub>l<sub>y o</sub>th<sub>er s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c row</sub> i<sub>s</sub> th<sub>e cer</sub>t<sub>a</sub>i<sub>n un</sub>if<sub>orm sp</sub>litt<sub>er.</sub> Aft<sub>er</sub> d<sub>e</sub>l<sub>e</sub>ti<sub>ng</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>se</sub>lf<sub>-</sub>l<sub>oops,</sub> th<sub>e</sub> f<sub>u</sub>ll t<sub>rans</sub>iti<sub>on grap</sub>h i<sub>s a</sub> DAG<sub>: a</sub>ft<sub>er</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e, or</sub>d<sub>er</sub> th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l<sub>-se</sub>l<sub>ec</sub>t<sub>or</sub> bl<sub>oc</sub>k<sub>s</sub> fi<sub>rs</sub>t<sub>,</sub> th<sub>en</sub> th<sub>e occurrence</sub> blocks in clause-major order, placing within each occurrence the selector for val(o) before the other selector, and finally the <sub>pa</sub>ddi<sub>ng an</sub>d t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es.</sub> E<sub>very con</sub>t<sub>ro</sub>l <sub>an</sub>d <sub>ou</sub>t<sub>come s</sub>t<sub>a</sub>t<sub>e can</sub> b<sub>e p</sub>l<sub>ace</sub>d i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y</sub> b<sub>e</sub>f<sub>ore or a</sub>ft<sub>er</sub> it<sub>s assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>se</sub>l<sub>ec</sub>t<sub>or.</sub> I<sub>n par</sub>ti<sub>cu</sub>l<sub>ar, no run un</sub>d<sub>er e</sub>ith<sub>er po</sub>li<sub>cy uses</sub> th<sub>e same uncer</sub>t<sub>a</sub>i<sub>n c</sub>h<sub>o</sub>i<sub>ce</sub> t<sub>w</sub>i<sub>ce, an</sub>d <sub>every</sub> t<sub>erm</sub>i<sub>na</sub>l th<sub>a</sub>t b<sub>o</sub>th <sub>po</sub>li<sub>c</sub>i<sub>es reac</sub>h i<sub>s</sub> <sub>a</sub>b<sub>sor</sub>bi<sub>n un</sub>d<sub>er eac</sub>h <sub>o</sub>f th<sub>em so ever s</sub>h<sub>are</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> i<sub>s c c</sub>l<sub>e-</sub>f<sub>ree un</sub>d<sub>er</sub> b<sub>o</sub>th <sub>o</sub>li<sub>c</sub>i<sub>es</sub> i<sub>n</sub> th<sub>e sense o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 5<sub>.</sub> B L<sub>emma</sub> 17 $\pi _ { C }$ <sub>an</sub>d $\pi _ { K } { \bf \ ' } _ { \bf S }$ <sub>va</sub>l<sub>ue</sub> dif<sub>erence a</sub>tt<sub>a</sub>i<sub>ns</sub> it<sub>s max</sub>i<sub>mum a</sub>t <sub>a</sub> t<sub>up</sub>l<sub>e o</sub>f <sub>c</sub>h<sub>o</sub>i<sub>ce ver</sub>ti<sub>ces.</sub> B<sub>y</sub> thi<sub>s an</sub>d th<sub>e</sub> t<sub>wo</sub> l<sub>emmas a</sub>b<sub>ove,</sub>

$$
\varphi \in \mathrm { S A T } \Longrightarrow \Delta _ { \cal U } ( \pi _ { \cal C } , \pi _ { \cal K } ) = 2 ,
$$

$$
\varphi \in { \mathrm { U N S A T } } \Longrightarrow \Delta _ { { \boldsymbol { u } } } ( \pi _ { C } , \pi _ { K } ) \leq 2 - { \frac { 1 } { n } } .
$$

I<sub>n</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>case, na</sub>t<sub>ure c</sub>h<sub>ooses</sub> th<sub>e canon</sub>i<sub>ca</sub>l <sub>ver</sub>t<sub>ex o</sub>f <sub>a sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng va</sub>l<sub>ua</sub>ti<sub>on.</sub> I<sub>n</sub> th<sub>e secon</sub>d<sub>, a ver</sub>t<sub>ex sa</sub>ti<sub>s</sub>f<sub>y</sub>i<sub>ng</sub> $\Phi _ { C }$ h<sub>as a</sub>t l<sub>eas</sub>t <sub>one</sub> f<sub>a</sub>ili<sub>n var</sub>i<sub>a</sub>bl<sub>e au</sub>dit b L<sub>emma</sub> 15<sub>, an</sub>d th<sub>ere</sub>f<sub>ore</sub> h<sub>as</sub> dif<sub>erence a</sub>t <sub>mos</sub>t $1 + ( n - 1 ) / n ;$ <sub>a ver</sub>t<sub>ex v</sub>i<sub>o</sub>l<sub>a</sub>ti<sub>ng</sub> $\Phi _ { C }$ h<sub>as</sub> dif<sub>erence</sub> at most 1. Vertex suficiency shows that an interior realization cannot do better.

ProofofTheorem 14. Use threshold $\textstyle 2 - { \frac { 1 } { 2 n } }$ <sub>.</sub> Th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\varphi \in { \mathrm { U N S A T } } \iff \Delta _ { \mathcal { U } } ( \pi _ { C } , \pi _ { K } ) \leq 2 - \frac { 1 } { 2 n } .
$$

I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub>

$$
2 - \frac { 1 } { n } < 2 - \frac { 1 } { 2 n } < 2 .
$$

Th<sub>e</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> h<sub>as</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>s</sub>i<sub>ze,</sub> $\gamma _ { 0 } ^ { - H } = ( 4 0 0 / 3 6 1 ) ^ { H }$ h<sub>as</sub> $O ( H )$ bit<sub>s, an</sub>d <sub>a</sub>ll <sub>uncer</sub>t<sub>a</sub>i<sub>n c</sub>h<sub>o</sub>i<sub>ces are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t t<sub>wo-</sub>Di<sub>rac</sub> <sub>c</sub>h<sub>o</sub>i<sub>ces.</sub> Thi<sub>s</sub> i<sub>s a po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>re</sub>d<sub>uc</sub>ti<sub>on</sub> f<sub>rom</sub> UNSAT<sub>.</sub> □

## B.3 Vertex-Extremal Membership

Theorem 16 (Vertex-extremal membership). Robust policy comparison is in coNP for $( s , a )$ -rectangular polytopes when both policies are cycle-free on shared choices (Definition 5).

Definition 7 (Vertex-extremal pair). The pair $( \pi _ { 1 } , \pi _ { 2 } )$ is vertex-extremal $i f$

$$
\operatorname* { s u p } _ { { \pmb u } \in \mathcal { U } } D _ { \pi _ { 1 } , \pi _ { 2 } } ( { \pmb u } ) = \operatorname* { m a x } _ { { \pmb v } \in \mathrm { V e r t } ( \mathcal { U } ) } D _ { \pi _ { 1 } , \pi _ { 2 } } ( { \pmb v } ) .
$$

Lemma 17 (Cycle-freeness on shared choices implies vertex-extremality). $I f \left( \pi _ { 1 } , \pi _ { 2 } \right)$ is cycle-free on shared choices $( D e f i n i -$ tion 5), then it is vertex-extremal (Definition 7).

Th<sub>e</sub> <sub>ma</sub>i<sub>n-</sub>b<sub>o</sub>d<sub>y</sub> <sub>examp</sub>l<sub>e</sub> E<sub>xamp</sub>l<sub>e</sub> 1 <sub>s</sub>h<sub>ows</sub> <sub>w</sub>h<sub>y</sub> th<sub>e</sub> h<sub>ypo</sub>th<sub>es</sub>i<sub>s</sub> i<sub>s</sub> <sub>necessary:</sub> <sub>one</sub> <sub>s</sub>h<sub>are</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>on</sub> <sub>a</sub> <sub>po</sub>li<sub>cy</sub> <sub>cyc</sub>l<sub>e</sub> <sub>can</sub> <sub>crea</sub>t<sub>e</sub> <sub>an</sub> i<sub>rra</sub>ti<sub>ona</sub>l i<sub>n</sub>t<sub>er</sub>i<sub>or max</sub>i<sub>mum, so en</sub>d<sub>po</sub>i<sub>n</sub>t <sub>eva</sub>l<sub>ua</sub>ti<sub>on</sub> i<sub>s no</sub> l<sub>onger soun</sub>d<sub>.</sub> W<sub>e use</sub> t<sub>wo s</sub>t<sub>an</sub>d<sub>ar</sub>d f<sub>ac</sub>t<sub>s.</sub> A <sub>separa</sub>t<sub>e</sub>l<sub>y a</sub>fi<sub>ne</sub> f<sub>unc</sub>ti<sub>on</sub> <sub>on a pro</sub>d<sub>uc</sub>t <sub>o</sub>f <sub>po</sub>l<sub>y</sub>t<sub>opes</sub> i<sub>s ex</sub>t<sub>rem</sub>i<sub>ze</sub>d <sub>a</sub>t <sub>a</sub> t<sub>up</sub>l<sub>e o</sub>f <sub>ver</sub>ti<sub>ces</sub> b<sub>y op</sub>ti<sub>m</sub>i<sub>z</sub>i<sub>ng one</sub> bl<sub>oc</sub>k <sub>a</sub>t <sub>a</sub> ti<sub>me.</sub> A fi<sub>n</sub>it<sub>e</sub> li<sub>near-</sub>f<sub>rac</sub>ti<sub>ona</sub>l f<sub>unc</sub>ti<sub>on</sub> <sub>on a po</sub>l<sub>y</sub>t<sub>ope</sub> i<sub>s ex</sub>t<sub>rem</sub>i<sub>ze</sub>d <sub>a</sub>t <sub>a ver</sub>t<sub>ex</sub> b<sub>ecause</sub> it<sub>s va</sub>l<sub>ue on a segmen</sub>t li<sub>es</sub> b<sub>e</sub>t<sub>ween</sub> it<sub>s en</sub>d<sub>po</sub>i<sub>n</sub>t <sub>va</sub>l<sub>ues.</sub>

ProofofLemma 17. Fix a single block $\mathcal { U } _ { s _ { 0 } , a _ { 0 } }$ <sub>,</sub> h<sub>o</sub>ld <sub>every o</sub>th<sub>er</sub> bl<sub>oc</sub>k fi<sub>xe</sub>d<sub>, an</sub>d <sub>cons</sub>id<sub>er</sub> t<sub>wo cases accor</sub>di<sub>ng</sub> t<sub>o w</sub>h<sub>e</sub>th<sub>er</sub> $( s _ { 0 } , a _ { 0 } ) \in \mathrm { S h } ( \pi _ { 1 } , \pi _ { 2 } )$ <sub>.</sub> Th<sub>en</sub> it<sub>era</sub>t<sub>e over</sub> th<sub>e</sub> bl<sub>oc</sub>k<sub>s.</sub>

$( s _ { 0 } , a _ { 0 } ) \not \in \mathrm { S h } ( \pi _ { 1 } , \pi _ { 2 } )$ , <sup>sa</sup>y <sup>onl</sup>y $\pi _ { 1 }$ <sub>uses</sub> it<sub>.</sub> Th<sub>en</sub> $V _ { \pmb { u } } ^ { \pi _ { 2 } } ( s _ { t } )$ i<sub>s cons</sub>t<sub>an</sub>t i<sub>n</sub> thi<sub>s</sub> bl<sub>oc</sub>k<sub>, an</sub>d $D _ { \pi _ { 1 } , \pi _ { 2 } }$ dif<sub>ers</sub> f<sub>rom</sub> $V _ { \pmb { u } } ^ { \pi _ { 1 } } ( s _ { \iota } )$ <sup>b</sup><sub>y</sub> a <sub>cons</sub>t<sub>an</sub>t<sub>.</sub> B<sub>y</sub> C<sub>ramer</sub>’<sub>s ru</sub>l<sub>e app</sub>li<sub>e</sub>d t<sub>o</sub> th<sub>e one equa</sub>ti<sub>on o</sub>f $\pi _ { 1 } \mathrm { ^ { * } s }$ B<sub>e</sub>ll<sub>man sys</sub>t<sub>em</sub> th<sub>a</sub>t thi<sub>s</sub> bl<sub>oc</sub>k <sub>en</sub>t<sub>ers,</sub> $V _ { \pmb { u } } ^ { \pi _ { 1 } }$ i<sub>s</sub> li<sub>near-</sub>f<sub>rac</sub>ti<sub>ona</sub>l i<sub>n</sub> thi<sub>s</sub> bl<sub>oc</sub>k<sub>, a ra</sub>ti<sub>o o</sub>f t<sub>wo</sub> f<sub>unc</sub>ti<sub>ons eac</sub>h <sub>a</sub>fi<sub>ne</sub> i<sub>n</sub> it <sub>regar</sub>dl<sub>ess o</sub>f <sub>cyc</sub>l<sub>es e</sub>l<sub>sew</sub>h<sub>ere</sub> i<sub>n</sub> th<sub>e grap</sub>h<sub>, so</sub> th<sub>e s</sub>t<sub>an</sub>d<sub>ar</sub>d f<sub>ac</sub>t <sub>a</sub>b<sub>ove</sub> <sub>ma</sub>k<sub>es</sub> it<sub>, an</sub>d h<sub>ence</sub> $D _ { \pi _ { 1 } , \pi _ { 2 } }$ <sub>, ex</sub>t<sub>rema</sub>l <sub>a</sub>t <sub>a ver</sub>t<sub>ex.</sub>

![](images/ec8f0b239377c0a08d0c31082599f29e30f579bcdc74babced56fa081bbbf6ae.jpg)  
Fi<sub>gure</sub> 4<sub>:</sub> Th<sub>e</sub> <sub>s</sub>h<sub>are</sub>d<sub>-cyc</sub>l<sub>e</sub> <sub>ga</sub>d<sub>ge</sub>t <sub>rea</sub>li<sub>z</sub>i<sub>ng</sub> <sub>one</sub> <sub>norma</sub>li<sub>ze</sub>d <sub>square-roo</sub>t t<sub>erm.</sub>

$( s _ { 0 } , a _ { 0 } ) \in \mathrm { S h } ( \pi _ { 1 } , \pi _ { 2 } )$ (hence, b<sub>y</sub> h<sub>yp</sub>othesis, c<sub>y</sub>cle-free under both $\pi _ { 1 }$ <sub>an</sub>d $\pi _ { 2 } )$ <sub>.</sub> If $s _ { 0 }$ i<sub>s a</sub>b<sub>sor</sub>bi<sub>ng un</sub>d<sub>er one o</sub>f th<sub>e</sub> t<sub>wo</sub> <sub>po</sub>li<sub>c</sub>i<sub>es,</sub> th<sub>en</sub> th<sub>a</sub>t <sub>po</sub>li<sub>cy uses</sub> $( s _ { 0 } , a _ { 0 } )$ <sub>an</sub>d th<sub>e c</sub>h<sub>o</sub>i<sub>ce</sub> h<sub>as row</sub> $\{ \delta _ { s _ { 0 } } \}$ <sub>, so</sub> th<sub>e</sub> bl<sub>oc</sub>k i<sub>s</sub> th<sub>e s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on</sub> $\{ \delta _ { s _ { 0 } } \}$ <sub>an</sub>d $D _ { \pi _ { 1 } , \pi _ { 2 } }$ i<sub>s</sub> <sub>cons</sub>t<sub>an</sub>t i<sub>n</sub> it<sub>.</sub> Oth<sub>erw</sub>i<sub>se no se</sub>lf<sub>-</sub>l<sub>oop was om</sub>itt<sub>e</sub>d <sub>a</sub>t $s _ { 0 } ,$ <sub>so cyc</sub>l<sub>e-</sub>f<sub>reeness</sub> i<sub>s</sub> th<sub>e</sub> lit<sub>era</sub>l <sub>grap</sub>h <sub>con</sub>diti<sub>on.</sub> U<sub>n</sub>d<sub>er e</sub>ith<sub>er po</sub>li<sub>cy,</sub> <sub>a run</sub> th<sub>en uses</sub> th<sub>e c</sub>h<sub>o</sub>i<sub>ce</sub> $( s _ { 0 } , a _ { 0 } )$ <sub>a</sub>t <sub>mos</sub>t <sub>once.</sub> Th<sub>e pro</sub>b<sub>a</sub>bilit<sub>y o</sub>f <sub>reac</sub>hi<sub>ng</sub> $s _ { 0 }$ i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>o</sub>f it<sub>s ou</sub>t<sub>go</sub>i<sub>ng row, an</sub>d th<sub>e con</sub>ti<sub>nua</sub>ti<sub>on va</sub>l<sub>ue a</sub>ft<sub>er</sub> l<sub>eav</sub>i<sub>ng</sub> $s _ { 0 }$ <sub>canno</sub>t d<sub>epen</sub>d <sub>on</sub> th<sub>a</sub>t <sub>row</sub> b<sub>ecause</sub> th<sub>e run never re</sub>t<sub>urns.</sub> C<sub>on</sub>diti<sub>on</sub>i<sub>ng on reac</sub>hi<sub>ng</sub> <sub>an</sub>d <sub>se</sub>l<sub>ec</sub>ti<sub>ng</sub> $( s _ { 0 } , a _ { 0 } )$ th<sub>ere</sub>f<sub>ore ma</sub>k<sub>es eac</sub>h <sub>po</sub>li<sub>cy va</sub>l<sub>ue a</sub>fi<sub>ne</sub> i<sub>n</sub> th<sub>e</sub> f<sub>ree</sub> bl<sub>oc</sub>k<sub>.</sub> Th<sub>e</sub>i<sub>r</sub> dif<sub>erence</sub> i<sub>s a</sub>fi<sub>ne as we</sub>ll <sub>an</sub>d i <sub>ex</sub>t<sub>rem</sub>i<sub>ze</sub>d <sub>a</sub>t <sub>a ver</sub>t<sub>ex.</sub>

E<sub>very</sub> bl<sub>oc</sub>k f<sub>a</sub>ll<sub>s</sub> <sub>un</sub>d<sub>er</sub> <sub>one</sub> <sub>o</sub>f th<sub>e</sub> t<sub>wo</sub> <sub>cases,</sub> <sub>so</sub> $D _ { \pi _ { 1 } , \pi _ { 2 } }$ i<sub>s</sub> <sub>ex</sub>t<sub>rema</sub>l <sub>a</sub>t <sub>a</sub> <sub>ver</sub>t<sub>ex</sub> i<sub>n</sub> <sub>every</sub> bl<sub>oc</sub>k<sub>,</sub> <sub>g</sub>i<sub>v</sub>i<sub>ng</sub> <sub>ver</sub>t<sub>ex-</sub>t<sub>up</sub>l<sub>e</sub> <sub>ex</sub>t<sub>rema</sub>lit<sub>y.</sub>

ProofofTheorem 16. By Lemma 17, the pair is vertex-extremal. For the strict complement, guess a vertex tuple v with $D _ { \pi _ { 1 } , \pi _ { 2 } } ( \pmb { v } ) > t .$ <sub>.</sub> S<sub>uc</sub>h <sub>a</sub> t<sub>up</sub>l<sub>e</sub> h<sub>as po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>enco</sub>di<sub>ng</sub> l<sub>eng</sub>th b<sub>ecause eac</sub>h <sub>componen</sub>t <sub>ver</sub>t<sub>ex so</sub>l<sub>ves a</sub> f<sub>u</sub>ll<sub>-ran</sub>k <sub>su</sub>b<sub>sys</sub>t<sub>em o</sub>f ti<sub>g</sub>ht <sub>ra</sub>ti<sub>ona</sub>l <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s.</sub> Th<sub>e ver</sub>ifi<sub>er c</sub>h<sub>ec</sub>k<sub>s</sub> $v \in { \mathcal { U } } ,$ <sub>so</sub>l<sub>ves</sub> th<sub>e</sub> t<sub>wo ra</sub>ti<sub>ona</sub>l B<sub>e</sub>ll<sub>man sys</sub>t<sub>ems, an</sub>d <sub>compares</sub> th<sub>e</sub>i<sub>r</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>va</sub>l<sub>ues</sub> in ol nomial time. Thus the com lement is in NP. □

## B.4 Shared Cycles and Square-Root-Sum Hardness

Theorem 18 (Shared-cycle algebraic hardness). Robust policy comparison is coSQRS-hardfor deterministic policies on $( s , a ) .$ rectangular RMDPs in which every uncertain choice is two-Dirac and the only other stochastic rows are certain uniform splitters.

W<sub>e cons</sub>t<sub>ruc</sub>t <sub>a ga</sub>d<sub>ge</sub>t <sub>w</sub>ith <sub>a s</sub>h<sub>are</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce on a cyc</sub>l<sub>e, w</sub>hi<sub>c</sub>h <sub>crea</sub>t<sub>es one norma</sub>li<sub>ze</sub>d <sub>square-roo</sub>t t<sub>erm.</sub> A<sub>n exac</sub>t d<sub>er</sub>i<sub>va</sub>ti<sub>ve</sub> <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> l<sub>oca</sub>t<sub>es</sub> it<sub>s</sub> i<sub>n</sub>t<sub>er</sub>i<sub>or max</sub>i<sub>mum, an</sub>d <sub>a cer</sub>t<sub>a</sub>i<sub>n sp</sub>litt<sub>er com</sub>bi<sub>nes</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>tl<sub>y c</sub>h<sub>osen</sub> l<sub>oca</sub>l <sub>max</sub>i<sub>ma</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> t<sub>arge</sub>t <sub>sum o</sub>f <sub>square roo</sub>t<sub>s.</sub> Th<sub>e</sub> d<sub>e</sub>li<sub>ca</sub>t<sub>e po</sub>i<sub>n</sub>t i<sub>s s</sub>i<sub>gn con</sub>t<sub>ro</sub>l <sub>a</sub>t th<sub>e cr</sub>iti<sub>ca</sub>l <sub>po</sub>i<sub>n</sub>t<sub>.</sub> All <sub>c</sub>h<sub>o</sub>i<sub>ces s</sub>t<sub>ay</sub> $( s , a )$ <sub>-rec</sub>t<sub>angu</sub>l<sub>ar an</sub>d <sub>use</sub> th<sub>e</sub> <sub>appen</sub>di<sub>x-w</sub>id<sub>e</sub> di<sub>scoun</sub>t $\gamma _ { 0 }$ <sub>requ</sub>i<sub>re</sub>d b<sub>y</sub> th<sub>e exac</sub>t lift<sub>.</sub>

Fi<sub>x</sub>

$$
b \geq 2 , \qquad m = \lceil { \sqrt { b } } \rceil .
$$

W<sub>e</sub> r<sub>ou</sub>nd ${ \sqrt { b } } \mathbf { u p }$ t<sub>o</sub> th<sub>e neares</sub>t i<sub>n</sub>t<sub>eger</sub> $m ,$ <sub>, so</sub> th<sub>a</sub>t $b \leq m ^ { 2 }$ (needed below), while m sta<sub>y</sub>s close enou<sub>g</sub>h to $\sqrt { b }$ f<sub>or</sub> th<sub>e</sub> fi<sub>na</sub>l id<sub>en</sub>tit<sub>y</sub> t<sub>o come ou</sub>t <sub>exac</sub>tl<sub>y r</sub>i<sub>g</sub>ht<sub>.</sub> Th<sub>e</sub> l<sub>oca</sub>l <sub>ga</sub>d<sub>ge</sub>t i<sub>n</sub> Fi<sub>gure</sub> 4 h<sub>as en</sub>t<sub>ry s</sub>t<sub>a</sub>t<sub>e</sub> $s ,$ i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e s</sub>t<sub>a</sub>t<sub>es</sub> $s _ { 1 } , s _ { 2 }$ <sub>, an</sub>d <sub>a zero-rewar</sub>d <sub>s</sub>i<sub>n</sub>k<sub>.</sub> P<sub>o</sub>li<sub>cy</sub> $\pi _ { u }$ chooses u at s and moves to $s _ { 1 }$ <sub>w</sub>ith <sub>rewar</sub>d $r _ { u } .$ <sub>.</sub> P<sub>o</sub>li<sub>cy</sub> $\pi _ { v }$ chooses v and moves to $s _ { 2 }$ di<sub>rec</sub>tl<sub>y,</sub> <sub>w</sub>ith <sub>rewar</sub>d <sub>zero.</sub> Th<sub>e</sub> f<sub>orce</sub>d e<sup>d</sup><sub>g</sub>e $s _ { 1 } \to s _ { 2 }$ h<sub>as rewar</sub>d <sub>zero:</sub> it<sub>s on</sub>l<sub>y purpose</sub> i<sub>s</sub> t<sub>o g</sub>i<sub>ve</sub> $\pi _ { u }$ <sub>one ex</sub>t<sub>ra s</sub>t<sub>ep</sub> b<sub>e</sub>f<sub>ore reac</sub>hi<sub>ng</sub> $s _ { 2 } .$ , so that its eventual return to s i<sub>s</sub> di<sub>scoun</sub>t<sub>e</sub>d <sub>one ex</sub>t<sub>ra</sub> f<sub>ac</sub>t<sub>or o</sub>f $\gamma _ { 0 }$ com<sub>p</sub>are<sup>d</sup> to $\pi _ { v } \mathrm { ^ { * } s . }$ Thi<sub>s asymme</sub>t<sub>ry</sub> i<sub>n cyc</sub>l<sub>e</sub> l<sub>eng</sub>th i<sub>s essen</sub>ti<sub>a</sub>l<sub>: w</sub>ith<sub>ou</sub>t it<sub>,</sub> $Q _ { u } - Q _ { \ i }$ b<sub>e</sub>l<sub>ow</sub> <sub>wou</sub>ld b<sub>e a ra</sub>ti<sub>o o</sub>f t<sub>wo a</sub>fi<sub>ne</sub> f<sub>unc</sub>ti<sub>ons, w</sub>hi<sub>c</sub>h i<sub>s mono</sub>t<sub>one an</sub>d h<sub>as no</sub> i<sub>n</sub>t<sub>er</sub>i<sub>or max</sub>i<sub>mum.</sub> Th<sub>e same mec</sub>h<sub>an</sub>i<sub>sm appears</sub> i<sub>n</sub> E<sub>xamp</sub>l<sub>e</sub> 1<sub>, w</sub>h<sub>ere one po</sub>li<sub>cy c</sub>l<sub>oses</sub> th<sub>e cyc</sub>l<sub>e an</sub>d th<sub>e o</sub>th<sub>er ex</sub>it<sub>s</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e</sub>l<sub>y.</sub> $\mathrm { A t } \ s _ { 2 } ,$ th<sub>e</sub> <sub>un</sub>i<sub>que</sub> <sub>s</sub>h<sub>are</sub>d <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>re</sub>t<sub>urns</sub> to s with probability $p$ <sub>an</sub>d <sub>en</sub>t<sub>ers</sub> th<sub>e s</sub>i<sub>n</sub>k <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $1 - p ,$ , p<sup>a</sup>y<sup>in</sup>g <sup>re</sup>w<sup>ard</sup> $c _ { v }$ <sub>e</sub>ith<sub>er</sub> <sub>way.</sub> O<sub>ur</sub> <sub>goa</sub>l i<sub>s</sub> t<sub>o</sub> <sub>c</sub>h<sub>oose</sub> $U , V , c _ { v } , r _ { u }$ <sub>so</sub> th<sub>a</sub>t $\begin{array} { r } { \operatorname* { s u p } _ { p \in [ 0 , 1 ] } \bigl ( Q _ { u } ( p ) - Q _ { v } ( p ) \bigr ) } \end{array}$  takes the form $C _ { b } - \sqrt { b }$ <sub>,</sub> f<sub>or a cons</sub>t<sub>an</sub>t $C _ { b }$ depending only on b. Lemma 19 establishes this. S<sub>e</sub>t

$$
\begin{array} { c } { { U = \displaystyle \frac { m ( 1 - \gamma _ { 0 } ) } { 2 \gamma _ { 0 } } , } } \\ { { V = \displaystyle \frac { b ( 1 - \gamma _ { 0 } ) } { 2 m } , } } \\ { { c _ { v } = \displaystyle \frac { V } { \gamma _ { 0 } } , } } \\ { { r _ { u } = U - \gamma _ { 0 } V = \displaystyle \frac { ( 1 - \gamma _ { 0 } ) ( m ^ { 2 } - \gamma _ { 0 } ^ { 2 } b ) } { 2 \gamma _ { 0 } m } . } } \end{array}
$$

All <sub>cons</sub>t<sub>an</sub>t<sub>s are nonnega</sub>ti<sub>ve:</sub> $U , V , c _ { v } \ > \ 0$ <sub>,</sub> <sub>w</sub>hil<sub>e</sub> $m ^ { 2 } \ge b > \gamma _ { 0 } ^ { 2 } b$ <sub>g</sub><sup>i</sup>ves $r _ { u } ~ > ~ 0$ <sub>.</sub> Th<sub>e</sub> B<sub>e</sub>ll<sub>man</sub> <sub>equa</sub>ti<sub>ons</sub> <sub>g</sub>i<sub>ve,</sub> <sub>wr</sub>iti<sub>ng</sub> $Q _ { i } ( p ) : = V _ { u } ^ { \pi _ { i } } ( s )$ f<sub>or</sub> th<sub>e va</sub>l<sub>ue o</sub>f $\pi _ { i }$ <sub>a</sub>t th<sub>e en</sub>t<sub>ry s</sub>t<sub>a</sub>t<sub>e as a</sub> f<sub>unc</sub>ti<sub>on o</sub>f th<sub>e s</sub>h<sub>are</sub>d <sub>re</sub>t<sub>urn pro</sub>b<sub>a</sub>bilit<sub>y</sub> $p ,$

$$
Q _ { u } ( p ) = r _ { u } + \gamma _ { 0 } ^ { 2 } c _ { v } + \gamma _ { 0 } ^ { 3 } p Q _ { u } ( p ) = \frac { U } { 1 - \gamma _ { 0 } ^ { 3 } p } ,
$$

$$
Q _ { v } ( p ) = \gamma _ { 0 } c _ { v } + \gamma _ { 0 } ^ { 2 } p Q _ { v } ( p ) = \frac { V } { 1 - \gamma _ { 0 } ^ { 2 } p } .
$$

Lemma 19 (Rational-fraction maximum). With the constants above,

$$
\operatorname* { s u p } _ { p \in [ 0 , 1 ] } \bigl ( Q _ { u } ( p ) - Q _ { v } ( p ) \bigr ) = C _ { b } - \sqrt { b } , \qquad C _ { b } = \frac { m } { 2 \gamma _ { 0 } } + \frac { \gamma _ { 0 } b } { 2 m } .
$$

Proof. Put $x = 1 - \gamma _ { 0 } ^ { 2 } p$ <sub>.</sub> Th<sub>en</sub> $x \in [ 1 - \gamma _ { 0 } ^ { 2 } , 1 ]$ <sub>an</sub>d

$$
f ( x ) = \frac { U } { 1 - \gamma _ { 0 } + \gamma _ { 0 } x } - \frac { V } { x } .
$$

Aft<sub>er</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>y</sub>i<sub>ng</sub> b<sub>y</sub> th<sub>e</sub> <sub>pos</sub>iti<sub>ve</sub> d<sub>enom</sub>i<sub>na</sub>t<sub>ors,</sub> th<sub>e</sub> <sub>s</sub>i<sub>gn</sub> <sub>o</sub>f $f ^ { \prime } ( x )$ i<sub>s</sub> th<sub>e</sub> <sub>s</sub>i<sub>gn</sub> <sub>o</sub>f

$$
\sqrt { V } ( 1 - \gamma _ { 0 } + \gamma _ { 0 } x ) - \sqrt { \gamma _ { 0 } U } x .
$$

Si<sub>nce</sub> $V \leq \gamma _ { 0 } U ,$ <sub>,</sub> th<sub>e a</sub>fi<sub>ne s</sub>i<sub>gn</sub> f<sub>ac</sub>t<sub>or</sub> h<sub>as nega</sub>ti<sub>ve s</sub>l<sub>ope.</sub> It i<sub>s pos</sub>iti<sub>ve</sub> b<sub>e</sub>f<sub>ore</sub> it<sub>s un</sub>i<sub>que zero an</sub>d <sub>nega</sub>ti<sub>ve a</sub>ft<sub>er</sub> it<sub>, so</sub> th<sub>e zero</sub> i<sub>s</sub> <sub>a max</sub>i<sub>mum.</sub> It li<sub>es</sub> i<sub>n</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>prov</sub>id<sub>e</sub>d

$$
V \leq \gamma _ { 0 } U \quad \mathrm { a n d } \quad \gamma _ { 0 } U \leq h _ { \gamma _ { 0 } } ^ { 2 } V , \qquad h _ { \gamma _ { 0 } } = \frac { 1 - \gamma _ { 0 } ^ { 3 } } { 1 - \gamma _ { 0 } ^ { 2 } } = \frac { 4 3 4 7 2 1 } { 3 0 4 4 0 0 } .
$$

Th<sub>e</sub> fi<sub>rs</sub>t i<sub>nequa</sub>lit<sub>y</sub> i<sub>s</sub> $b \leq m ^ { 2 }$ <sub>.</sub> Th<sub>e secon</sub>d f<sub>o</sub>ll<sub>ows</sub> f<sub>rom</sub> $m ^ { 2 } / b \leq 2$ <sub>an</sub>d th<sub>e exac</sub>t <sub>ra</sub>ti<sub>ona</sub>l <sub>compar</sub>i<sub>son</sub> $h _ { \gamma _ { 0 } } ^ { 2 } > 2$ <sub>.</sub> At th<sub>e cr</sub>iti<sub>ca</sub>l <sub>po</sub>i<sub>n</sub>t<sub>,</sub> di<sub>rec</sub>t <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>on y</sub>i<sub>e</sub>ld<sub>s</sub>

$$
\operatorname* { m a x } f = { \frac { ( { \sqrt { U } } - { \sqrt { \gamma _ { 0 } V } } ) ^ { 2 } } { 1 - \gamma _ { 0 } } } = { \frac { m } { 2 \gamma _ { 0 } } } + { \frac { \gamma _ { 0 } b } { 2 m } } - { \sqrt { b } } .
$$

Thi<sub>s</sub> <sub>va</sub>l<sub>ue</sub> i<sub>s</sub> <sub>nonnega</sub>ti<sub>ve</sub> b<sub>y</sub> th<sub>e</sub> <sub>ar</sub>ith<sub>me</sub>ti<sub>c-geome</sub>t<sub>r</sub>i<sub>c</sub> <sub>mean</sub> i<sub>nequa</sub>lit<sub>y</sub> <sub>app</sub>li<sub>e</sub>d t<sub>o</sub> $m / \gamma _ { 0 }$ <sub>an</sub>d $\gamma _ { 0 } b / m$

Combining independent copies. For inputs $b _ { 1 } , \ldots , b _ { n }$ , take disjoint copies of the gadget. A fresh certain uniform splitter <sub>en</sub>t<sub>ers eac</sub>h <sub>copy w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $1 / n .$ <sub>.</sub> S<sub>ca</sub>li<sub>ng a</sub>ll <sub>rewar</sub>d<sub>s</sub> i<sub>n every copy</sub> b<sub>y</sub> $n / \gamma _ { 0 }$ <sub>cance</sub>l<sub>s</sub> th<sub>e sp</sub>litt<sub>er pro</sub>b<sub>a</sub>bilit<sub>y an</sub>d fi<sub>rs</sub>t di<sub>scoun</sub>t<sub>.</sub> Aft<sub>er</sub> thi<sub>s sca</sub>li<sub>ng, every rema</sub>i<sub>n</sub>i<sub>ng c</sub>h<sub>o</sub>i<sub>ce uses ru</sub>i<sub>nous-s</sub>i<sub>n</sub>k <sub>comp</sub>l<sub>e</sub>ti<sub>on.</sub> R<sub>ec</sub>t<sub>angu</sub>l<sub>ar</sub>it<sub>y ma</sub>k<sub>es</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>parame</sub>t<sub>ers</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>, so</sub>

$$
\Delta _ { { \mathcal U } } ( \pi _ { u } , \pi _ { v } ) = \sum _ { i = 1 } ^ { n } C _ { b _ { i } } - \sum _ { i = 1 } ^ { n } { \sqrt { b _ { i } } } .
$$

ProofofTheorem 18. Reduce from the coSQRS variant asking whether $\textstyle \sum _ { i } { \sqrt { b _ { i } } } \geq k$ <sub>.</sub> S<sub>e</sub>t

$$
t = \sum _ { i } C _ { b _ { i } } - k .
$$

Th<sub>en</sub> $\Delta _ { \mathcal { U } } ( \pi _ { u } , \pi _ { v } ) \le t$ exactl<sub>y</sub> when the coSQRS instance is positive. Ever<sub>y</sub> uncertain choice is two-Dirac.

## B.5 Real-Hierarchy Completeness of Policy Comparison

Theorem 20 (General policy comparison). Robust policy comparison is $\forall \mathbb { R } .$ -complete for deterministic policies under general rational polytopic uncertainty.

Lemma 21 (∀R membership for policy comparison). Robust policy comparison is in $\forall \mathbb { R } .$ for deterministic policies under general rational polytopic uncertainty.

Proof. Apply Lemma 13 with τ fixed to $\pi _ { 1 }$ <sub>an</sub>d th<sub>e s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on</sub> f<sub>am</sub>il<sub>y</sub> $\{ \pi _ { 2 } \}$

Definition 8 (Polynomial-evaluation RMDP). Fix an explicit sparse polynomial

$$
f ( \boldsymbol { p } ) = \sum _ { \ell = 1 } ^ { N } c _ { \ell } \prod _ { j = 1 } ^ { d _ { \ell } } p _ { i _ { \ell , j } } , \qquad \boldsymbol { p } \in [ 0 , 1 ] ^ { m } ,
$$

and a rational discount $\gamma \in ( 0 , 1 )$ . The component $\operatorname { P o l y } _ { \gamma } ( f )$ is defined as follows.

• A reward-zero certain uniform splitter at $s _ { f }$ enters branch ℓ with probability $1 / N .$ . That branch has states $b _ { \ell } ^ { 0 } , \ldots , b _ { \ell } ^ { d _ { \ell } }$ and a common zero-reward sink ⊥.

$A t \ b _ { \ell } ^ { j - 1 }$ , the unique action continues to $b _ { \ell } ^ { j }$ with probability $u _ { \ell , j }$ and enters ⊥ otherwise. State $b _ { \ell } ^ { d _ { \ell } }$ is absorbing with value $r _ { \ell } = N \gamma ^ { - ( d _ { \ell } + 1 ) } c _ { \ell } .$

![](images/23f4436a634cb378e8ab0f8b3c3e63c87a2dcb6d659924af56c3dc6ff4c5bef9.jpg)  
Fi<sub>g</sub>ure 5: One branch of the <sub>p</sub>ol<sub>y</sub>nomial-evaluation RMDP. E<sub>q</sub>ualities in U tie re<sub>p</sub>eated <sub>p</sub>arameter occurrences across branches.

• Each $u _ { \ell , j }$ is a coordinate of a two-successor row in [0, 1]. Occurrences representing the same logical coordinate may be tied by rational linear equalities.

• At each state the action described above is intended, every other described reward is zero, and $R ( b _ { \ell } ^ { d _ { \ell } } ) = ( 1 - \gamma ) r _ { \ell }$

• All remaining choices use ruinous-sink completion.

Forpolicy comparison, choose a representative $u _ { i } ^ { \mathrm { r e p } }$ for everyparameter and impose $u _ { \ell , j } = u _ { i } ^ { \mathrm { r e p } }$ whenever $i _ { \ell , j } = i ;$ occurrences of $1 - p _ { i } ,$ when present, instead satisfy $u _ { \ell , j } = 1 - u _ { i } ^ { \mathrm { r e p } }$ . Together with stochasticity and box constraints, these equalities define U and the resulting RMDP $M _ { f } = ( S , A , \bar { \mathcal { U } } , R , s _ { f } , \dot { \gamma } )$ . Repeated indices in a monomial represent powers and are realized by distinct rows tied to the same representative

Fi<sub>gure</sub> 5 <sub>s</sub>h<sub>ows one monom</sub>i<sub>a</sub>l b<sub>ranc</sub>h <sub>o</sub>f thi<sub>s cons</sub>t<sub>ruc</sub>ti<sub>on.</sub>

Lemma 22 (Polynomial evaluation). For every realization of the occurrence rows, the compliant value of $\operatorname { P o l y } _ { \gamma } ( f )$ is $\textstyle \sum _ { \ell } c _ { \ell } \prod _ { j } u _ { \ell , j }$ . In particular, when the representative coordinates equal $p ,$ the unique compliant policy of $M _ { f }$ has value $f ( p ) a t s _ { f }$ . The component is acyclic apart from absorbing terminals, every uncertain row has two successors, and its only other stochastic row is a certain uniform splitter. It has size $O ( N$ max d ) and rewards ofpolynomial encoding length. Moreover, U is one rational polytope ofpolynomial description size.

Proof. Branch ℓ is reached with probability $1 / N _ { : }$ <sub>,</sub> <sub>surv</sub>i<sub>ves</sub> <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $\Pi _ { j } u _ { \ell , j }$ <sub>, an</sub>d <sub>reac</sub>h<sub>es</sub> it<sub>s</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>a</sub>ft<sub>er</sub> $d _ { \ell } + 1$ t<sub>rans</sub>iti<sub>ons.</sub> It<sub>s</sub> di<sub>scoun</sub>t<sub>e</sub>d <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub>

$$
\frac { 1 } { N } \gamma ^ { d _ { \ell } + 1 } \left( N \gamma ^ { - ( d _ { \ell } + 1 ) } c _ { \ell } \right) \prod _ { j } u _ { \ell , j } ,
$$

<sub>an</sub>d <sub>summ</sub>i<sub>ng g</sub>i<sub>ves</sub> th<sub>e</sub> fi<sub>rs</sub>t id<sub>en</sub>tit<sub>y.</sub> Th<sub>e represen</sub>t<sub>a</sub>ti<sub>ve equa</sub>liti<sub>es</sub> th<sub>en g</sub>i<sub>ve</sub> $f ( \boldsymbol p )$ <sub>.</sub> E<sub>ac</sub>h <sub>occurrence con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es one row.</sub> E<sub>very</sub> b<sub>ranc</sub>h i<sub>s</sub> <sub>a</sub> di<sub>rec</sub>t<sub>e</sub>d <sub>c</sub>h<sub>a</sub>i<sub>n,</sub> <sub>an</sub>d th<sub>e</sub> <sub>payo</sub>f <sub>exponen</sub>t i<sub>s</sub> <sub>a</sub>t <sub>mos</sub>t th<sub>e</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l i<sub>npu</sub>t <sub>s</sub>i<sub>ze.</sub> Th<sub>e</sub> t<sub>y</sub>i<sub>ng,</sub> <sub>comp</sub>l<sub>emen</sub>t<sub>,</sub> <sub>s</sub>t<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub>it<sub>y,</sub> <sub>an</sub>d i<sub>n</sub>t<sub>erva</sub>l <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s are ra</sub>ti<sub>ona</sub>l li<sub>near cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s, so</sub> th<sub>e</sub>i<sub>r</sub> i<sub>n</sub>t<sub>ersec</sub>ti<sub>on</sub> i<sub>s a po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>-s</sub>i<sub>ze ra</sub>ti<sub>ona</sub>l <sub>po</sub>l<sub>y</sub>t<sub>ope.</sub> □

Lemma 23 (∀R hardness for policy comparison). Robustpolicy comparison is ∀R-hardfor deterministic policies under general rational polytopic uncertainty.

Proof. By the strict degree-six normal form of Schaefer and Stefankovic (2024, Lemma 2.8) and their bounded-open equivalence (Schaefer and Stefankovic 2024 Pro osition 2.12) $\exists p \in [ 0 , 1 ] ^ { m } : f ( p ) > 0$ is ∃R-com<sub>p</sub>lete for an ex<sub>p</sub>licitl<sub>y</sub> re<sub>p</sub>resented d<sub>egree-s</sub>i<sub>x po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> th<sub>e a</sub>fi<sub>ne</sub> i<sub>mage o</sub>f th<sub>e source</sub> b<sub>ox</sub> i<sub>s</sub> $( 0 , 1 ) ^ { m }$ <sub>,</sub> <sub>an</sub>d <sub>s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>pos</sub>iti<sub>v</sub>it<sub>y</sub> th<sub>ere</sub> i<sub>s</sub> <sub>equ</sub>i<sub>va</sub>l<sub>en</sub>t t<sub>o</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>pos</sub>iti<sub>v</sub>it<sub>y on</sub> it<sub>s c</sub>l<sub>osure</sub> b<sub>y con</sub>ti<sub>nu</sub>it<sub>y.</sub> It<sub>s comp</sub>l<sub>emen</sub>t i<sub>s</sub> $\forall p : f ( p ) \leq 0 .$ C<sub>o</sub>n<sub>s</sub>tr<sub>uc</sub>t $M _ { f }$ b<sub>y</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 8<sub>, a</sub>dd <sub>a</sub> f<sub>res</sub>h i<sub>n</sub>iti<sub>a</sub>l <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> b<sub>e</sub>t<sub>ween</sub> $s _ { f }$ <sub>an</sub>d <sub>a zero-rewar</sub>d <sub>s</sub>i<sub>n</sub>k<sub>, an</sub>d l<sub>e</sub>t d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c po</sub>li<sub>c</sub>i<sub>es</sub> $\pi _ { f } , \pi _ { \perp }$ <sub>c</sub>h<sub>oose</sub> th<sub>e</sub> t<sub>wo ac</sub>ti<sub>ons.</sub> B<sub>y</sub> L<sub>emma</sub> 22<sub>,</sub>

$$
\Delta _ { \mathcal { U } } ( \pi _ { f } , \pi _ { \perp } ) \le 0 \iff \forall p \in [ 0 , 1 ] ^ { m } : f ( p ) \le 0 .
$$

Thi<sub>s</sub> <sub>proves</sub> h<sub>ar</sub>d<sub>ness.</sub>

ProofofTheorem 20. Immediate from Lemmas 21 and 23.

## B.6 Portfolio Comparison and the Shared-Selector Evaluator

Problem 6 (Portfolio comparison). Given an RMDP M, a policy $\pi _ { 0 } \in \Pi ^ { \mathrm { M R } }$ , an input portfolio $\Pi \subset \Pi ^ { \mathrm { M R } }$ , and rational t, decide whether

$$
\Delta _ { \mathscr { U } } ( \pi _ { 0 } , \Pi ) = \operatorname* { s u p } _ { \pm \mathscr { U } } \biggl ( V _ { \pmb { u } } ^ { \pi _ { 0 } } ( s _ { \iota } ) - \operatorname* { m a x } _ { \pi \in \Pi } V _ { \pmb { u } } ^ { \pi } ( s _ { \iota } ) \biggr ) \leq t .
$$

Theorem 24. Portfolio comparison is ∀R-complete, withportfolio sizepart ofthe input. Hardness already holdsfor deterministic policies and acyclic $( s , a )$ -rectangular RMDPs with two-successor uncertain choices.

Lemma 25 (Membership). Portfolio comparison is in ∀R.

Proof. Apply Lemma 13 with τ fixed to $\pi _ { 0 }$ and the explicit family Π.

Problem 7 (Strict elementary feasibility). Given rational expressions $g _ { 1 } , \ldots , g _ { r }$ over variables x $\in [ 0 , 1 ] ^ { n }$ , each afine or afine plus one bilinear term, with no variable occurring twice in the same expression, decide whether

$$
g _ { 1 } ( x ) > 0 , \ldots , g _ { r } ( x ) > 0
$$

has a solution $x \in [ 0 , 1 ] ^ { n }$

Lemma 26. Problem 7 is ∃R-complete.

Proof. Membership is immediate. For hardness, use the bounded degree-four equality normal form underlying Definition 4. It is ∃R-com<sub>p</sub>lete to decide

$$
\exists y \in [ - 1 , 1 ] ^ { n } : \quad f ( y ) = 0 ,
$$

<sub>an</sub>d <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> $y _ { i } = 2 x _ { i } - 1$ <sub>c</sub>h<sub>anges</sub> th<sub>e</sub> d<sub>oma</sub>i<sub>n</sub> t<sub>o</sub> $[ 0 , 1 ] ^ { n }$ <sub>.</sub> A<sub>pp</sub>l<sub>y</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 4 <sub>w</sub>ith<sub>ou</sub>t it<sub>s</sub> <sub>op</sub>ti<sub>ona</sub>l <sub>ou</sub>t<sub>pu</sub>t <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>an</sub>d <sub>appen</sub>d th<sub>e a</sub>fi<sub>ne res</sub>id<sub>ua</sub>l ${ \widehat { f } } / B .$ <sub>, w</sub>hi<sub>c</sub>h <sub>se</sub>t<sub>s</sub> th<sub>e</sub> d<sub>eco</sub>d<sub>e</sub>d <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l t<sub>o zero.</sub> D<sub>eno</sub>t<sub>e a</sub>ll <sub>res</sub>id<sub>ua</sub>l<sub>s</sub> b<sub>y</sub> $h _ { 1 } , \ldots , h _ { t }$ <sub>.</sub> If th<sub>ey</sub> h<sub>ave no</sub> <sub>common zero on</sub> th<sub>e compac</sub>t b<sub>ox,</sub> th<sub>en</sub>

$$
\eta = \operatorname* { m i n } _ { x } \operatorname* { m a x } _ { j } | h _ { j } ( x ) | > 0 .
$$

Cl<sub>ear</sub> d<sub>enom</sub>i<sub>na</sub>t<sub>ors</sub> <sub>an</sub>d <sub>pu</sub>t $\begin{array} { r } { F = \sum _ { j } h _ { j } ^ { 2 } } \end{array}$ <sub>.</sub> Thi<sub>s</sub> i<sub>s a nonnega</sub>ti<sub>ve</sub> i<sub>n</sub>t<sub>eger po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>o</sub>f d<sub>egree a</sub>t <sub>mos</sub>t f<sub>our, w</sub>ith <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l coeficient bit length. If there is no common zero, apply the efective Łojasiewicz bound of Schaefer and Stefankovic (2024,

Theorem 2.2) on the com<sub>p</sub>act box to $F$ <sub>an</sub>d th<sub>e</sub> <sub>cons</sub>t<sub>an</sub>t <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>one.</sub> It <sub>g</sub>i<sub>ves</sub> <sub>a</sub> l<sub>ower</sub> b<sub>oun</sub>d $2 ^ { - \ell D ^ { c N ^ { 2 } } }$ i<sub>n</sub> t<sub>erms o</sub>f th<sub>e num</sub>b<sub>er</sub> $N$ <sub>o</sub>f <sub>var</sub>i<sub>a</sub>bl<sub>es,</sub> d<sub>egree</sub> $\dot { D } \leq 4 ,$ and coeficient length ℓ. Consequentl<sub>y</sub> an explicit pol<sub>y</sub>nomial $p$ i<sub>n</sub> th<sub>e</sub> i<sub>npu</sub>t l<sub>eng</sub>th <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\eta > 2 ^ { - 2 ^ { \varepsilon } }$ <sub>.</sub> I<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> $a _ { 0 } , b _ { 0 } , \dotsc , a _ { p } , b _ { p }$ <sub>w</sub>ith

$$
0 < a _ { 0 } < \frac { 1 } { 2 } , \qquad 0 < b _ { 0 } < \frac { 1 } { 2 } ,
$$

$$
0 < a _ { j + 1 } < \frac { a _ { j } b _ { j } } { 4 } , \qquad 0 < b _ { j + 1 } < \frac { a _ { j } b _ { j } } { 4 } \quad ( j < p ) .
$$

Th<sub>e</sub> <sub>expo</sub>n<sub>e</sub>nt r<sub>ecu</sub>rr<sub>e</sub>n<sub>ce</sub> <sub>g</sub>i<sub>ves</sub> $a _ { p } < 2 ^ { - 2 ^ { p } }$ <sub>.</sub> R<sub>ep</sub>l<sub>ace</sub> $h _ { j } = 0$ <sup>b</sup><sub>y</sub>

$$
a _ { p } + h _ { j } > 0 , \qquad a _ { p } - h _ { j } > 0 .
$$

A<sub>n exac</sub>t <sub>norma</sub>l<sub>-</sub>f<sub>orm so</sub>l<sub>u</sub>ti<sub>on sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>sys</sub>t<sub>em.</sub> C<sub>onverse</sub>l<sub>y, a s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>so</sub>l<sub>u</sub>ti<sub>on wou</sub>ld h<sub>ave</sub> $| h _ { j } | < a _ { p } < \eta$ <sup>f</sup>or ever<sub>y</sub> $j ,$ <sub>con</sub>t<sub>ra</sub>di<sub>c</sub>ti<sub>ng</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f $\eta$ <sub>w</sub>h<sub>en</sub> th<sub>e res</sub>id<sub>ua</sub>l <sub>sys</sub>t<sub>em</sub> i<sub>s</sub> i<sub>n</sub>f<sub>eas</sub>ibl<sub>e.</sub> E<sub>ac</sub>h <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t i<sub>s a</sub>fi<sub>ne or a</sub>fi<sub>ne p</sub>l<sub>us one pro</sub>d<sub>uc</sub>t <sub>o</sub>f di<sub>s</sub>ti<sub>nc</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>es.</sub> I<sub>npu</sub>t <sub>cop</sub>i<sub>es ensure</sub> th<sub>a</sub>t <sub>an</sub> i<sub>npu</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>e occurs on</sub>l<sub>y</sub> i<sub>n a</sub>fi<sub>ne copy res</sub>id<sub>ua</sub>l<sub>s.</sub> □

Lemma 27 (Hardness). Portfolio comparison is ∀R-hard already for deterministic policies and acyclic $( s , a )$ -rectangular RMDPs with two-successor uncertain choices.

Definition 9 (Shared-selector evaluator). Fix a strict elementary feasibility instance $g _ { 1 } , \ldots , g _ { r }$ over $x \in [ 0 , 1 ] ^ { n }$ (Problem 7) and a discount $\gamma .$ . For each i, treat $- g _ { i }$ as its constant term plus a list of monomials, each either linear or bilinear in two distinct variables, and order the variables of every monomial by increasing index. The shared-selector evaluator is the RMDP $M = ( S , A , \mathcal { U } , R , s _ { \iota } , \gamma )$ , where

$S = \{ s _ { 0 } , \bot \} \cup \{ q _ { j } , q _ { j } ^ { 0 } , q _ { j } ^ { 1 } : 1 \leq j \leq n \} \cup \{ \tau _ { i , \ell } \}$ : the initial state $s _ { 0 } ;$ afresh absorbing zero-reward state ⊥; one selector $q _ { j }$ and its two successors $q _ { j } ^ { 0 } , q _ { j } ^ { 1 }$ per variable $x _ { j } ,$ , shared by every branch that mentions $x _ { j } , \cdot$ and one terminal $\tau _ { i , \ell } p e r$ branch $\ell$ $o f { - } g _ { i }$ (its constant term counts as one branch);

$A = \{ a _ { 0 } , a _ { 1 } , \ldots , a _ { r } \} \cup \{ d _ { i , j , b } \} .$ : at s , action $a _ { 0 }$ leads deterministically to ⊥, and action $a _ { i }$ enters a rational certain splitter with one branch per term $o f - g _ { i } ; a t q _ { j } ^ { b } ;$ , the decoder action $d _ { i , j , b }$ is enabledfor every policy i whose current branch visits $x _ { j } ,$ it leads to ⊥ when $b = 0 ,$ and otherwise to the next selector or to the branch’s terminal $i f x _ { j }$ was its last variable;

• U is (s, a)-rectangular: the certain splitter at a and every decoder are deterministic; the one uncertain choice at q has $a _ { i }$ $q _ { j }$ distribution $( 1 - x _ { j } ) \delta _ { q _ { i } ^ { 0 } } + x _ { j } \delta _ { q _ { i } ^ { 1 } }$ , the same distribution regardless ofwhich branch or which $g _ { i }$ is passing through;

• branch ℓ, selected with splitter probability w $> 0$ and reaching its terminal $\tau _ { i , \ell }$ after d transitions, has terminal payof $w ^ { - 1 } \gamma ^ { - d }$ times its coeficient (including its sign); every other described choice pays reward $0 ,$ and no common branch depth is needed because each payofcancels its own depth;

• all remaining choices use ruinous-sink completion;

![](images/02fcbd7453d08d542748ee0635409e27f8ac960ae158315aa467ab85f2f148e0.jpg)  
Fi<sub>gure</sub> 6<sub>:</sub> Th<sub>e s</sub>h<sub>are</sub>d<sub>-se</sub>l<sub>ec</sub>t<sub>or eva</sub>l<sub>ua</sub>t<sub>or o</sub>f E<sub>xamp</sub>l<sub>e</sub> 2<sub>.</sub> E<sub>ac</sub>h b<sub>ranc</sub>h <sub>payo</sub>f <sub>cance</sub>l<sub>s</sub> th<sub>a</sub>t b<sub>ranc</sub>h’<sub>s own pro</sub>b<sub>a</sub>bilit<sub>y an</sub>d d<sub>ep</sub>th<sub>.</sub> E<sub>ac</sub>h <sub>a</sub>b<sub>sor</sub>bi<sub>ng</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>uses</sub> th<sub>e payo</sub>f <sub>conven</sub>ti<sub>on o</sub>f A<sub>ppen</sub>di<sub>x</sub> A<sub>.</sub>

$\boldsymbol { s } _ { t } = \boldsymbol { s } _ { 0 } .$

Lemma 28 (Evaluator policies). In the shared-selector evaluator of Definition 9, the following are well-defined deterministic (stationary) policies.

$\pi _ { 0 }$ chooses a<sub>0</sub> at $s _ { 0 }$ and a fixed ruinous-completed default choice at every other nonterminal state, which is unreachable under π<sub>0</sub>.

• For each $i \in \{ 1 , \ldots , r \}$ $\pi _ { i }$ chooses $a _ { i }$ at $s _ { 0 } ;$ and for every variable $x _ { j }$ occurring in $g _ { i }$ and every outcome $b \in \{ 0 , 1 \}$ $\pi _ { i }$ chooses $d _ { i , j , b } a t q _ { j } ^ { b } .$

This assignment is well-defined because a stationary policy fixes a single action per state and no variable occurs twice in one expression $g _ { i } .$

ProofofLemma 28. No prescription is repeated within one policy because no variable occurs twice in one expression. Pre <sub>scr</sub>i<sub>p</sub>ti<sub>ons ma</sub>d<sub>e</sub> b<sub>y</sub> dif<sub>eren</sub>t <sub>po</sub>li<sub>c</sub>i<sub>es nee</sub>d <sub>no</sub>t <sub>agree.</sub> □

Example ${ \displaystyle \mathbf { 2 } \left( \mathbf { A } \right. }$ worked evaluator). Take $\begin{array} { r } { n = 2 , r = 2 , \gamma = \frac { 1 } { 2 } , g _ { 1 } ( x ) = x _ { 1 } - \frac { 1 } { 2 } , } \end{array}$ , and $\begin{array} { r } { g _ { 2 } ( x ) = x _ { 1 } x _ { 2 } - \frac { 1 } { 4 } } \end{array}$ , so that $\begin{array} { r } { - g _ { 1 } ( x ) = \frac { 1 } { 2 } - x _ { 1 } } \end{array}$ and $\begin{array} { r } { - g _ { 2 } ( x ) = \frac { 1 } { 4 } - x _ { 1 } x _ { 2 } } \end{array}$ . Figure 6 shows the resulting RMDP. By Lemma 28, $\pi _ { 0 }$ (always $a _ { 0 } ) , \pi _ { 1 }$ (choosing $a _ { 1 }$ at $s _ { 0 }$ and $d _ { 1 , 1 , 1 }$ at $q _ { 1 } ^ { 1 } )$ , and π<sub>2</sub> (choosing a<sub>2</sub> at s<sub>0</sub>, $d _ { 2 , 1 , 1 } a t q _ { 1 } ^ { 1 }$ , and $d _ { 2 , 2 , 1 }$ at $q _ { 2 } ^ { 1 } )$ are all well-defined.

Each constant branch reaches $\tau _ { i , 0 }$ in one step withprobability 1/2. The monomial branch ofπ follows $s _ { 0 }  q _ { 1 }  q _ { 1 } ^ { 1 }  \tau _ { 1 , 1 } ,$ with total branch probability $x _ { 1 } / 2$ and depth three. That of π<sub>2</sub> follows $s _ { 0 }  q _ { 1 }  q _ { 1 } ^ { 1 }  q _ { 2 }  q _ { 2 } ^ { 1 }  \tau _ { 2 , 1 } ,$ , with probability $x _ { 1 } x _ { 2 } / 2$ and depthfive. The policies share the selector $q _ { 1 }$ and diverge only through their decoder actions at $q _ { 1 } ^ { 1 }$

The monomial branches reach their terminals after depths three and five, while each constant branch has depth one. With $\begin{array} { r } { w = \frac { 1 } { 2 } a n d \gamma = \frac { 1 } { 2 } , } \end{array}$ , the rule $\boldsymbol { R } = \boldsymbol { w } ^ { - 1 } \gamma ^ { - d } \times $ (coeficient) applies at each branch’s own depth and gives

$$
R ( \tau _ { 1 , 0 } ) = 2 , \qquad R ( \tau _ { 1 , 1 } ) = - 1 6 , \qquad R ( \tau _ { 2 , 0 } ) = 1 , \qquad R ( \tau _ { 2 , 1 } ) = - 6 4 .
$$

The first policy consequently has value

$$
\begin{array} { r } { V _ { x } ^ { \pi _ { 1 } } ( s _ { 0 } ) = \gamma \cdot \frac { 1 } { 2 } \cdot 2 + \gamma ^ { 3 } \cdot \frac { 1 } { 2 } x _ { 1 } ( - 1 6 ) = \frac { 1 } { 2 } - x _ { 1 } , } \end{array}
$$

and similarly

$$
\begin{array} { r } { V _ { x } ^ { \pi _ { 2 } } ( s _ { 0 } ) = \gamma \cdot \frac { 1 } { 2 } \cdot 1 + \gamma ^ { 5 } \cdot \frac { 1 } { 2 } x _ { 1 } x _ { 2 } ( - 6 4 ) = \frac { 1 } { 4 } - x _ { 1 } x _ { 2 } , } \end{array}
$$

as required $b y - g _ { 1 }$ and $- g _ { 2 } .$ . Since π<sub>0</sub> always reaches ⊥, $V _ { x } ^ { \pi _ { 0 } } ( s _ { 0 } ) = 0 f o r$ every x. Hence

$$
\Delta _ { \mathcal { U } } ( \pi _ { 0 } , \{ \pi _ { 1 } , \pi _ { 2 } \} ) = \operatorname* { s u p } _ { x \in [ 0 , 1 ] ^ { 2 } } \operatorname* { m i n } \bigl ( g _ { 1 } ( x ) , g _ { 2 } ( x ) \bigr ) ,
$$

which is maximized at $x = ( 1 , 1 )$ , giving min $\begin{array} { r } { ( \frac { 1 } { 2 } , \frac { 3 } { 4 } ) = \frac { 1 } { 2 } } \end{array}$

ProofofLemma 27. We reduce from strict elementary feasibility (Lemma 26) using the shared-selector evaluator of Definition 9 <sub>a</sub>t di<sub>scoun</sub>t $\gamma = \gamma _ { 0 }$ <sub>.</sub> It<sub>s po</sub>li<sub>c</sub>i<sub>es</sub> $\pi _ { 0 } , \pi _ { 1 } , \ldots , \pi _ { r }$ <sub>are we</sub>ll<sub>-</sub>d<sub>e</sub>fi<sub>ne</sub>d b<sub>y</sub> L<sub>emma</sub> 28<sub>.</sub> E<sub>very uncer</sub>t<sub>a</sub>i<sub>n c</sub>h<sub>o</sub>i<sub>ce</sub> i<sub>s a</sub>t <sub>a se</sub>l<sub>ec</sub>t<sub>or</sub> $q _ { j }$ <sub>an</sub>d h<sub>as</sub> t<sub>wo successors, an</sub>d th<sub>e se</sub>l<sub>ec</sub>t<sub>ors</sub>’ <sub>c</sub>h<sub>o</sub>i<sub>ces are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>, so</sub> th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on</sub> i<sub>s</sub> $( s , a )$ <sub>-rec</sub>t<sub>angu</sub>l<sub>ar w</sub>ith t<sub>wo-successor</sub> <sub>c</sub>h<sub>o</sub>i<sub>ces.</sub> O<sub>r</sub>d<sub>er</sub>i<sub>ng every monom</sub>i<sub>a</sub>l’<sub>s se</sub>l<sub>ec</sub>t<sub>ors</sub> b<sub>y</sub> i<sub>ncreas</sub>i<sub>ng var</sub>i<sub>a</sub>bl<sub>e</sub> i<sub>n</sub>d<sub>ex ma</sub>k<sub>es</sub> th<sub>e</sub> f<sub>u</sub>ll t<sub>rans</sub>iti<sub>on grap</sub>h <sub>acyc</sub>li<sub>c apar</sub>t f<sub>rom</sub> it<sub>s</sub> <sub>a</sub>b<sub>sor</sub>bi<sub>ng</sub> fi<sub>na</sub>l<sub>s.</sub> Th<sub>ere are po</sub>l<sub>ynom</sub>i<sub>a</sub>ll<sub>y many se</sub>l<sub>ec</sub>t<sub>ors,</sub> d<sub>eco</sub>d<sub>ers, an</sub>d t<sub>erm</sub>i<sub>na</sub>l<sub>s, an</sub>d <sub>every</sub> b<sub>ranc</sub>h d<sub>ep</sub>th i<sub>s po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>, so</sub> th<sub>e</sub> terminal <sub>p</sub>a<sub>y</sub>ofs have <sub>p</sub>ol<sub>y</sub>nomial bit len<sub>g</sub>th. B<sub>y</sub> construction, π reaches onl<sub>y</sub> ⊥, so $V _ { x } ^ { \pi _ { 0 } } ( s _ { \iota } ) = 0$ . Policy π realizes exactly the di<sub>scoun</sub>t<sub>e</sub>d <sub>sum o</sub>f $- g _ { i }$ ’<sub>s</sub> t<sub>erms, so</sub> $V _ { x } ^ { \pi _ { i } } ( { \bar { s } } _ { \iota } ) = { \dot { - } } g _ { i } ( x )$ <sub>.</sub> H<sub>ence, w</sub>ith $\Pi = \{ \pi _ { 1 } , \ldots , \pi _ { r } \}$

$$
\Delta _ { \mathscr { U } } ( \pi _ { 0 } , \Pi ) = \operatorname* { s u p } _ { x \in [ 0 , 1 ] ^ { n } } \Bigl ( 0 - \operatorname* { m a x } _ { i } \bigl ( - g _ { i } ( x ) \bigr ) \Bigr ) = \operatorname* { s u p } _ { x \in [ 0 , 1 ] ^ { n } } \operatorname* { m i n } _ { i } g _ { i } ( x ) .
$$

Th<sub>e r</sub>i<sub>g</sub>ht <sub>s</sub>id<sub>e</sub> i<sub>s pos</sub>iti<sub>ve exac</sub>tl <sub>w</sub>h<sub>en</sub> th<sub>e s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>s s</sub>t<sub>em</sub> i<sub>s</sub> f<sub>eas</sub>ibl<sub>e an</sub>d i<sub>s a</sub>t <sub>mos</sub>t <sub>zero o</sub>th<sub>erw</sub>i<sub>se.</sub> H<sub>ence</sub> it<sub>s upper-</sub>th<sub>res</sub>h<sub>o</sub>ld lan<sub>g</sub>ua<sub>g</sub>e at threshold zero is the com<sub>p</sub>lement of an ∃R-com<sub>p</sub>lete <sub>p</sub>roblem, <sub>p</sub>rovin<sub>g</sub> hardness. □

Proof of Theorem 24. Immediate from Lemmas 25 and 27.

## C Proofs of Theorems 1 and 5: Membership for Certification and Synthesis

B<sub>e</sub>ll<sub>man</sub> <sub>equa</sub>ti<sub>ons</sub> <sub>enco</sub>d<sub>e</sub> <sub>exp</sub>li<sub>c</sub>it <sub>or</sub> <sub>ex</sub>i<sub>s</sub>t<sub>en</sub>ti<sub>a</sub>ll<sub>y</sub> <sub>quan</sub>tifi<sub>e</sub>d <sub>por</sub>tf<sub>o</sub>li<sub>o</sub> <sub>mem</sub>b<sub>ers</sub> t<sub>oge</sub>th<sub>er</sub> <sub>w</sub>ith <sub>a</sub> <sub>un</sub>i<sub>versa</sub>ll<sub>y</sub> <sub>quan</sub>tifi<sub>e</sub>d <sub>po</sub>li<sub>cy.</sub> Theorem 1 (Membership). Regret certification is in ∀R.

Proof. Apply Lemma 13 to the explicit input portfolio. Quantifying all memoryless randomized policies is sound because every realized discounted MDP has an optimal deterministic policy, so the encoded assertion is exactly Rreg(Π) ≤ t. The given-policy <sub>pro</sub>bl<sub>em</sub> i<sub>s</sub> th<sub>e case</sub> $r = 1$ □

Theorem 5 (Membership). Minimal robust regret and bounded portfolio synthesis belong to ∃∀R.

Proof. Existentially quantify the k policy tables before the universal encoding of Lemma 13. Their policy-simplex constraints remain outside the universal implication, and unary k keeps the formula polynomial. The single-policy problem is the case $k = 1$

The resulting formula asserts that some portfolio of at most k members has regret at most t, whereas $\rho _ { k }$ i<sub>s an</sub> i<sub>n</sub>fi<sub>mum, so</sub> th<sub>e</sub> t<sub>wo agree on</sub>l<sub>y w</sub>h<sub>en</sub> th<sub>a</sub>t i<sub>n</sub>fi<sub>mum</sub> i<sub>s a</sub>tt<sub>a</sub>i<sub>ne</sub>d<sub>.</sub> It i<sub>s:</sub> th<sub>e</sub> i<sub>npu</sub>t <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y se</sub>t i<sub>s a ra</sub>ti<sub>ona</sub>l <sub>po</sub>l<sub>y</sub>t<sub>ope,</sub> h<sub>ence compac</sub>t<sub>, an</sub>d L<sub>e</sub>mm<sub>a</sub> 29 <sub>app</sub>li<sub>es.</sub> □

Lemma 29 (Portfolio attainment). In a finite discounted RMDP with compact U, the infimum

$$
\rho _ { r } = \operatorname* { i n f } _ { | \Pi | \leq r } { \mathrm { R r e g } ( \Pi ) }
$$

is attained.

Proof. Represent a portfolio by an ordered r-tuple, repeating members when the set is smaller. This does not change its pointwise maximum, so the two infima agree. Three facts give the claim. The space of ordered r-tuples of memoryless randomized policie is a finite <sub>p</sub>roduct of sim<sub>p</sub>lexes, hence com<sub>p</sub>act, and U is com<sub>p</sub>act b<sub>y</sub> h<sub>yp</sub>othesis. For a fixed discount, the Bellman s<sub>y</sub>stem of a policy has a unique solution that depends continuously on the policy probabilities and the realization jointly, and $V _ { u } ^ { * } ( s _ { \iota } )$ i<sub>s</sub> th<sub>e</sub> maximum of finitely many such solutions, one per memoryless deterministic policy, hence also jointly continuous. Therefore

$$
f ( \pi , u ) = V _ { u } ^ { * } ( s _ { \iota } ) - \operatorname* { m a x } _ { i } V _ { u } ^ { \pi _ { i } } \big ( s _ { \iota } \big )
$$

is jointly continuous, $\pi \mapsto \operatorname* { m a x } _ { u \in \mathcal { U } } f ( \pi , u )$ i<sub>s</sub> <sub>con</sub>ti<sub>nuous,</sub> <sub>an</sub>d it <sub>a</sub>tt<sub>a</sub>i<sub>ns</sub> it<sub>s</sub> <sub>m</sub>i<sub>n</sub>i<sub>mum</sub> <sub>on</sub> th<sub>e</sub> <sub>compac</sub>t <sub>po</sub>li<sub>cy-</sub>t<sub>up</sub>l<sub>e</sub> <sub>space.</sub>

N<sub>o</sub> <sub>acyc</sub>li<sub>c</sub>it<sub>y</sub> i<sub>s</sub> <sub>nee</sub>d<sub>e</sub>d<sub>,</sub> <sub>so</sub> th<sub>e</sub> l<sub>emma</sub> <sub>app</sub>li<sub>es</sub> t<sub>o</sub> <sub>every</sub> i<sub>npu</sub>t <sub>o</sub>f P<sub>ro</sub>bl<sub>em</sub> 4 <sub>an</sub>d t<sub>o</sub> th<sub>e</sub> <sub>syn</sub>th<sub>es</sub>i<sub>s</sub> RMDP <sub>o</sub>f A<sub>ppen</sub>di<sub>x</sub> J <sub>a</sub>lik<sub>e.</sub>

## D Exact Transfer from Policy Comparison to Robust Regret

R<sub>egre</sub>t <sub>uses</sub> <sub>a</sub> <sub>rea</sub>li<sub>za</sub>ti<sub>on-</sub>d<sub>epen</sub>d<sub>en</sub>t b<sub>es</sub>t <sub>response,</sub> <sub>w</sub>h<sub>ereas</sub> <sub>compar</sub>i<sub>son</sub> fi<sub>xes</sub> b<sub>o</sub>th <sub>po</sub>li<sub>c</sub>i<sub>es.</sub> Th<sub>e</sub> lift b<sub>e</sub>l<sub>ow</sub> <sub>ma</sub>k<sub>es</sub> <sub>one</sub> fi<sub>xe</sub>d <sub>re</sub>f<sub>erence op</sub>ti<sub>ma</sub>l <sub>a</sub>t <sub>every rea</sub>li<sub>za</sub>ti<sub>on:</sub> it f<sub>ac</sub>t<sub>ors eac</sub>h <sub>source c</sub>h<sub>o</sub>i<sub>ce</sub> th<sub>roug</sub>h <sub>a separa</sub>t<sub>e rou</sub>ti<sub>ng s</sub>t<sub>a</sub>t<sub>e an</sub>d <sub>g</sub>i<sub>ves on</sub>l<sub>y</sub> th<sub>e re</sub>f<sub>erence</sub> <sub>an a</sub>dditi<sub>ona</sub>l b<sub>onus.</sub> W<sub>e s</sub>t<sub>a</sub>t<sub>e</sub> it f<sub>or an a</sub>ll<sub>owe</sub>d<sub>-ac</sub>ti<sub>on</sub> f<sub>am</sub>il<sub>y</sub> $A _ { \mathrm { a l l o w } }$ <sub>.</sub> F<sub>or</sub> <sub>a</sub> fi<sub>n</sub>it<sub>e</sub> <sub>por</sub>tf<sub>o</sub>li<sub>o,</sub> t<sub>a</sub>k<sub>e</sub> th<sub>e</sub> <sub>un</sub>i<sub>on</sub> <sub>o</sub>f it<sub>s</sub> <sub>suppor</sub>t<sub>s.</sub>

Definition 10 (Lifted RMDP). Fix an $( s , a )$ -rectangular source RMDP $M = ( S , A , \mathcal { U } , R , s _ { \iota } , \gamma _ { 0 } )$ , a deterministic reference policy $\pi _ { 0 } ,$ , and a nonempty allowed-action family $A _ { \mathrm { a l l o w } } ( s ) \subseteq A$ at each $s \in S ,$ , not necessarily containing $\pi _ { 0 } ( s )$ . Let ${ \dot { F } } \subseteq S$ be its absorbing final states, and let $c _ { f }$ be the payof of $f \in F$ . Fix

$$
V _ { \mathrm { m a x } } = \frac { \operatorname* { m a x } _ { s , a \in A _ { \mathrm { a l l o w } } ( s ) \cup \{ \pi _ { 0 } ( s ) \} } \left| R ( s , a ) \right| } { 1 - \gamma _ { 0 } } , \qquad K = 2 V _ { \mathrm { m a x } } + 1 , \qquad \Lambda = \frac { K } { 1 - \gamma _ { 0 } } .
$$

Choose a rational Z of polynomial encoding length such that $\beta Z > \Lambda + V _ { \mathrm { m a x } } .$ . The lifted RMDP is the tuple $\widehat { M } =$ $( \widehat { S } , \widehat { A } , \widehat { \mathcal { U } } , \widehat { R } , \widehat { s _ { \iota } } , \beta )$ , where

$\bullet \widehat { S } = \{ x _ { s } : s \in S \} \cup \{ y _ { s , a } : s \in S \setminus F , a \in A _ { \mathrm { a l l o w } } ( s ) \cup \{ \pi _ { 0 } ( s ) \} \} \cup \{ \bot _ { \mathrm { r } } \}$ . There is one tag state $x _ { s }$ per source state, one <sub>se</sub>l<sub>ec</sub>t<sub>or s</sub>t<sub>a</sub>t<sub>e</sub> $y _ { s , a } f o r$ every allowed or reference action at a nonfinal source state, and one ruinous sink.

![](images/d1ca6d40c97194cb77c5179b94185ecb39a9c4c0e49c7d40af4ab70c095d0217.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> 7<sub>:</sub> Th<sub>e</sub> <sub>sou</sub>r<sub>ce</sub> <sub>a</sub>nd it<sub>s</sub> lift<sub>e</sub>d RMDP in E<sub>xa</sub>m<sub>p</sub>l<sub>e</sub> 3<sub>.</sub>

${ \widehat { A } } = A \cup$ {bonus, route} adds twofresh action symbols to the source action set.

• Ub is (s, a)-rectangular. At a nonfinal tag state, an allowed action a sends $x _ { s } t o y _ { s , a }$ and the bonus action sends it to $y _ { s , \pi _ { 0 } ( s ) }$ The route action at $y _ { s , a }$ carries exactly the original uncertainty set $\mathcal { U } _ { ( s , a ) }$ , redirectedfrom successors s<sup>′</sup> to tag states $x _ { s ^ { \prime } } . A t$ afinal tag x<sub>f</sub>, every allowed action self-loops, as does the bonus action.

• At a nonfinal tag, $\widehat { R } ( x _ { s } , a ) = R ( s , a )$ for $a \in A _ { \mathrm { a l l o w } } ( s )$ and $\widehat { R } ( x _ { s } , \mathrm { b o n u s } ) = R ( s , \pi _ { 0 } ( s ) ) + K$ . Every route action has reward zero. At a final tag x<sub>f</sub>, every allowed action has reward $( 1 - \beta ) c _ { f } ,$ , while the bonus action has reward $( 1 - \beta ) ( c _ { f } + \Lambda )$

• Every action at ⊥ has reward $- ( 1 - \beta ) Z$ and self-loops. Every action at $x _ { f }$ not already described also self-loops with reward $- ( 1 - \beta ) Z .$ . All remaining choices use ruinous-sink completion with this Z.

$$
\widehat { s _ { \iota } } = x _ { s _ { \iota } }
$$

Write $\widehat { \pi } _ { 0 }$ for the policy always choosing the bonus action. The lift πb of any memoryless randomized policy π supported on $A _ { \mathrm { a l l o w } }$ uses the same action distribution as π at each tag state and the route action at every selector state. For afinite portfolio $\Pi = \{ \pi _ { 1 } , \ldots , \pi _ { r } \}$ , write $\widehat { \Pi } = \{ \widehat { \pi } _ { 1 } , \ldots , \widehat { \pi } _ { r } \}$

Example 3 (A worked lift). Take a source RMDP with states $S = \{ s _ { 0 } , s _ { 1 } \} , s _ { \iota } = s _ { 0 } ,$ , actions $A = \{ a , b , \mathrm { s t o p } \}$ , and discount $\gamma _ { 0 } . A t s _ { 0 } ,$ , action a moves deterministically to $s _ { 1 }$ with reward $R ( s _ { 0 } , a ) = 0 .$ . Action b is uncertain, moving to $s _ { 1 }$ with probability p and back to $s _ { 0 }$ with probability $1 - p f o r$ any $p \in [ 0 , 1 ] ,$ , with reward $\begin{array} { r } { R ( s _ { 0 } , b ) = \frac { 1 - \gamma _ { 0 } } { 2 } } \end{array}$ regardless of outcome. $A t ~ s _ { 1 }$ , the only action stop self-loops deterministically with reward $R ( s _ { 1 } , \mathrm { s t o p } ) = 1 - \gamma _ { 0 } ,$ so $\tilde { V ( s _ { 1 } ) } \doteq 1$ (Figure 7a). Fix reference $\pi _ { 0 } ( s _ { 0 } ) = a , \pi _ { 0 } ( s _ { 1 } ) =$ stop and a single candidate $\pi _ { 1 } ( s _ { 0 } ) = b , \pi _ { 1 } ( s _ { 1 } ) = \mathrm { s t o p }$ , so $\Pi = \left\{ \pi _ { 1 } \right\}$ and $A _ { \mathrm { a l l o w } } ( s _ { 0 } ) = \{ b \}$ $\boldsymbol { A } _ { \mathrm { a l l o w } } ( s _ { 1 } ) = \{ \mathrm { s t o p } \}$ . Note $A _ { \mathrm { a l l o w } } ( s _ { 0 } )$ does not contain $\pi _ { 0 } ( s _ { 0 } ) = a ,$ while $A _ { \mathrm { a l l o w } } ( s _ { 1 } )$ happens to coincide with $\pi _ { 0 } ( s _ { 1 } )$ . The Bellman equations give

$$
V _ { u } ^ { \pi _ { 0 } } ( s _ { 0 } ) = \gamma _ { 0 } , \qquad V _ { u } ^ { \pi _ { 1 } } ( s _ { 0 } ) = \frac { \frac { 1 - \gamma _ { 0 } } { 2 } + \gamma _ { 0 } p } { ( 1 - \gamma _ { 0 } ) + \gamma _ { 0 } p } ,
$$

and the second value is increasing in p. Hence

$$
\begin{array} { r } { \Delta _ { \mathcal { U } } ( \pi _ { 0 } , \{ \pi _ { 1 } \} ) = \gamma _ { 0 } - \frac { 1 } { 2 } = \frac { 1 6 1 } { 4 0 0 } . } \end{array}
$$

Definition 10 gives tag states $x _ { s _ { 0 } } , x _ { s _ { 1 } }$ , selectors $y _ { s _ { 0 } , a } , y _ { s _ { 0 } , b } ,$ a ruinous sink, and $\beta = 1 9 / 2 0$ (Figure 7b). Because $s _ { 1 }$ is an absorbing final of payof one, the allowed and bonus actions at $\boldsymbol { x } _ { s _ { 1 } }$ self-loop with rewards $1 - \beta$ and $( 1 - \beta ) ( 1 + \Lambda )$ . The selector $y _ { s _ { 0 } , b }$ inherits the source intervalfor p. The relevant rewards are $\begin{array} { r } { R ( s _ { 0 } , a ) = 0 , R ( s _ { 0 } , b ) = \frac { 1 - \gamma _ { 0 } } { 2 } , R ( s _ { 1 } , \mathrm { s t o p } ) = 1 - \gamma _ { 0 } , } \end{array}$ so Definition 10 gives

$$
V _ { \mathrm { m a x } } = \frac { 1 - \gamma _ { 0 } } { 1 - \gamma _ { 0 } } = 1 , \qquad K = 2 V _ { \mathrm { m a x } } + 1 = 3 , \qquad \Lambda = \frac { K } { 1 - \gamma _ { 0 } } = \frac { 4 0 0 } { 1 3 } ,
$$

and $\begin{array} { r } { \mathrm { R r e g } ( \{ \widehat \pi _ { 1 } \} ) = \Lambda + \Delta _ { \mathcal { U } } ( \pi _ { 0 } , \{ \pi _ { 1 } \} ) = \frac { 4 0 0 } { 1 3 } + \frac { 1 6 1 } { 4 0 0 } . } \end{array}$

Th<sub>e</sub> <sub>marg</sub>i<sub>n</sub> i<sub>n</sub> $K = 2 V _ { \mathrm { m a x } } + 1$ i<sub>s w</sub>h<sub>a</sub>t <sub>ma</sub>k<sub>es</sub> th<sub>e</sub> b<sub>onus ac</sub>ti<sub>on un</sub>i<sub>que</sub>l<sub>y op</sub>ti<sub>ma</sub>l <sub>a</sub>t <sub>every</sub> t<sub>ag s</sub>t<sub>a</sub>t<sub>e.</sub> L<sub>emma</sub> 30 f<sub>orma</sub>li<sub>zes</sub> thi <sub>p</sub>o<sup>i</sup>nt.

Lemma 30 (Exact comparison-to-regret lift). Consider the construction of Definition 10 for a deterministic reference $\pi _ { 0 }$ and an allowed-action family $A _ { \mathrm { a l l o w } }$ . For every memoryless randomized policy π supported on this family and every realization $u ,$ its lift satisfies

$$
V _ { \widehat { u } } ^ { \widehat { \pi } } ( x _ { s } ) = V _ { u } ^ { \pi } ( s ) \qquad ( s \in S ) .
$$

This construction preserves $( s , a )$ -rectangularity. It preserves determinism when the lifted policy is deterministic. It preserves acyclicity of the RMDP and of policy graphs, with every source absorbing final remaining an absorbing final. Each non-singleton source choice is copied unchanged to one selector state. In particular, two-successor and two-Dirac restrictions on uncertain

choices are preserved. The bonus policy $\widehat { \pi } _ { 0 }$ is pointwise optimal in every realization, and the bonus action is the unique optimal action at every tag state. Consequently,for everyfinite portfolio Π ofpolicies supported on $A _ { \mathrm { a l l o w } }$

$$
\operatorname { R r e g } ( \widehat { \Pi } ) = \Lambda + \Delta _ { \mathcal { U } } ( \pi _ { 0 } , \Pi ) .
$$

ProofofLemma $3 0 .$ W<sub>e</sub> fi<sub>rs</sub>t <sub>s</sub>h<sub>ow</sub> th<sub>a</sub>t th<sub>e</sub> b<sub>onus po</sub>li<sub>cy</sub> i<sub>s op</sub>ti<sub>ma</sub>l i<sub>n</sub> th<sub>e</sub> t<sub>arge</sub>t RMDP <sub>a</sub>t <sub>every rea</sub>li<sub>za</sub>ti<sub>on,</sub> t<sub>oge</sub>th<sub>er w</sub>ith it<sub>s</sub> value. Fix a realization u of the source RMDP, inducing a realization u of the target RMDP via the redirected transitions of D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 10<sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> $W ( s ) : = \Lambda + V _ { u } ^ { \pi _ { 0 } } ( s )$ <sup>f</sup>or ever<sub>y</sub> source state $s ,$ <sub>an</sub>d <sub>rea</sub>d it <sub>as a can</sub>did<sub>a</sub>t<sub>e va</sub>l<sub>ue</sub> f<sub>or</sub> th<sub>e</sub> t<sub>ag s</sub>t<sub>a</sub>t<sub>e</sub> $x _ { s }$ <sub>.</sub> F<sub>or a</sub> <sub>non</sub>fi<sub>na</sub>l <sub>source s</sub>t<sub>a</sub>t<sub>e,</sub> $\beta ^ { 2 } = \gamma _ { 0 }$ <sub>means</sub> th<sub>a</sub>t <sub>go</sub>i<sub>ng</sub> f<sub>rom</sub> $x _ { s }$ th<sub>roug</sub>h it<sub>s se</sub>l<sub>ec</sub>t<sub>or</sub> t<sub>o</sub> th<sub>e nex</sub>t t<sub>ag s</sub>t<sub>a</sub>t<sub>e con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> th<sub>e source</sub> di<sub>scoun</sub>t <sub>overa</sub>ll<sub>.</sub> W<sub>e may</sub> th<sub>ere</sub>f<sub>ore compare</sub> $W$ <sub>aga</sub>i<sub>ns</sub>t thi<sub>s</sub> t<sub>wo-s</sub>t<sub>ep</sub> B<sub>e</sub>ll<sub>man</sub> b<sub>ac</sub>k<sub>up</sub> di<sub>rec</sub>tl<sub>y.</sub> F<sub>or</sub> th<sub>e</sub> b<sub>onus ac</sub>ti<sub>on a</sub>t $x _ { s }$ <sub>,</sub> th<sub>e</sub> b<sub>ac</sub>k<sub>up</sub> i<sub>s</sub>

$$
R ( s , \pi _ { 0 } ( s ) ) + K + \gamma _ { 0 } \sum _ { s ^ { \prime } } u ( s , \pi _ { 0 } ( s ) , s ^ { \prime } ) W ( s ^ { \prime } ) .
$$

S<sub>u</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>ng</sub> $\pi _ { 0 } \mathrm { ^ { \circ } s }$ <sub>own</sub> B<sub>e</sub>ll<sub>man</sub> <sub>equa</sub>ti<sub>on</sub> $\begin{array} { r } { V _ { u } ^ { \pi _ { 0 } } ( s ) = R ( s , \pi _ { 0 } ( s ) ) + \gamma _ { 0 } \sum _ { s ^ { \prime } } u ( s , \pi _ { 0 } ( s ) , s ^ { \prime } ) V _ { u } ^ { \pi _ { 0 } } ( s ^ { \prime } ) } \end{array}$ f<sub>or</sub> $R ( s , \pi _ { 0 } ( s ) )$ <sub>,</sub> thi<sub>s</sub> <sub>s</sub>i<sub>mp</sub>lifi<sub>es</sub> to

$$
\begin{array} { r } { V _ { u } ^ { \pi _ { 0 } } ( s ) + K + \gamma _ { 0 } \Lambda , } \end{array}
$$

<sub>w</sub>hi<sub>c</sub>h <sub>equa</sub>l<sub>s</sub> $W ( s ) = \Lambda + V _ { u } ^ { \pi _ { 0 } } ( s )$ <sub>exac</sub>tl<sub>y,</sub> <sub>s</sub>i<sub>nce</sub> $K = \Lambda ( 1 - \gamma _ { 0 } )$ b<sub>y</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f $\Lambda .$ F<sub>or any a</sub>ll<sub>owe</sub>d <sub>ac</sub>ti<sub>on</sub> $a \in A _ { \mathrm { a l l o w } } ( s )$ <sub>,</sub> th<sub>e</sub> <sub>same</sub> t<sub>wo-s</sub>t<sub>ep</sub> b<sub>ac</sub>k<sub>up</sub> i<sub>s</sub>

$$
R ( s , a ) + \gamma _ { 0 } \sum _ { s ^ { \prime } } u ( s , a , s ^ { \prime } ) W ( s ^ { \prime } ) = \underbrace { R ( s , a ) + \gamma _ { 0 } \sum _ { s ^ { \prime } } u ( s , a , s ^ { \prime } ) V _ { u } ^ { \pi _ { 0 } } ( s ^ { \prime } ) - V _ { u } ^ { \pi _ { 0 } } ( s ) } _ { \mathrm { a t m o s t 2 } V _ { \mathrm { m a x } } \mathrm { i n a b s o l u t e v a l u c } } + V _ { u } ^ { \pi _ { 0 } } ( s ) + \gamma _ { 0 } \Lambda .
$$

Th<sub>e</sub> b<sub>rac</sub>k<sub>e</sub>t<sub>e</sub>d t<sub>erm</sub> i<sub>s a</sub>t <sub>mos</sub>t $2 V _ { \mathrm { m a x } }$ b<sub>ecause</sub> th<sub>e rewar</sub>d i<sub>s a</sub>t <sub>mos</sub>t $( 1 - \gamma _ { 0 } ) V _ { \mathrm { m a x } }$ <sub>an</sub>d b<sub>o</sub>th <sub>va</sub>l<sub>ue</sub> t<sub>erms are a</sub>t <sub>mos</sub>t $V _ { \mathrm { m a x } }$ i<sub>n</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e va</sub>l<sub>ue.</sub> C<sub>om</sub>bi<sub>ne</sub>d <sub>w</sub>ith $( 1 - \gamma _ { 0 } ) \Lambda = K = 2 V _ { \operatorname* { m a x } } + 1$ <sub>,</sub> th<sub>e w</sub>h<sub>o</sub>l<sub>e</sub> b<sub>ac</sub>k<sub>up</sub> i<sub>s a</sub>t <sub>mos</sub>t $W ( s ) - 1 < W ( s )$ <sub>.</sub> At <sub>an a</sub>b<sub>sor</sub>bi<sub>ng</sub> fi<sub>na</sub>l $f$ o<sup>f</sup> <sub>p</sub>a<sub>y</sub>o<sup>f</sup> $c _ { f }$ <sub>,</sub> th<sub>e</sub> b<sub>onus</sub> b<sub>ac</sub>k<sub>up</sub> i<sub>s</sub>

$$
( 1 - \beta ) ( c _ { f } + \Lambda ) + \beta W ( f ) = W ( f ) ,
$$

<sub>w</sub>hil<sub>e</sub> <sub>every</sub> <sub>a</sub>ll<sub>owe</sub>d<sub>-ac</sub>ti<sub>on</sub> b<sub>ac</sub>k<sub>up</sub> i<sub>s</sub>

$$
( 1 - \beta ) c _ { f } + \beta W ( f ) = c _ { f } + \beta \Lambda = W ( f ) - ( 1 - \beta ) \Lambda < W ( f ) .
$$

E<sub>very o</sub>th<sub>er ac</sub>ti<sub>on a</sub>t $x _ { f }$ h<sub>as va</sub>l<sub>ue</sub> b<sub>e</sub>l<sub>ow</sub> $W ( f )$ b<sub>ecause</sub> it<sub>s se</sub>lf<sub>-</sub>l<sub>oop payo</sub>f i<sub>s</sub> $- Z ,$ <sub>, an</sub>d <sub>every un</sub>d<sub>escr</sub>ib<sub>e</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce a</sub>t <sub>a non</sub>fi<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>e</sub> h<sub>as</sub> b<sub>ac</sub>k<sub>up</sub> $- \beta Z \overset { \cdot } { < } W ( s )$ b<sub>y</sub> th<sub>e c</sub>h<sub>o</sub>i<sub>ce o</sub>f $Z .$ <sub>.</sub> At <sub>se</sub>l<sub>ec</sub>t<sub>or s</sub>t<sub>a</sub>t<sub>es,</sub> th<sub>e rou</sub>t<sub>e ac</sub>ti<sub>on</sub> i<sub>s</sub> lik<sub>ew</sub>i<sub>se</sub> b<sub>e</sub>tt<sub>er</sub> th<sub>an en</sub>t<sub>er</sub>i<sub>ng</sub> th<sub>e ru</sub>i<sub>nous</sub> <sub>s</sub>i<sub>n</sub>k<sub>.</sub> Th<sub>us</sub> $W$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e</sub> B<sub>e</sub>ll<sub>man op</sub>ti<sub>ma</sub>lit<sub>y equa</sub>ti<sub>on a</sub>t <sub>every</sub> t<sub>ag s</sub>t<sub>a</sub>t<sub>e, w</sub>ith th<sub>e</sub> b<sub>onus ac</sub>ti<sub>on as</sub> th<sub>e un</sub>i<sub>que max</sub>i<sub>m</sub>i<sub>zer.</sub> Si<sub>nce</sub> th<sub>e</sub> di<sub>scoun</sub>t<sub>e</sub>d B<sub>e</sub>ll<sub>man op</sub>ti<sub>ma</sub>lit<sub>y equa</sub>ti<sub>on</sub> h<sub>as a un</sub>i<sub>que so</sub>l<sub>u</sub>ti<sub>on,</sub> $W$ i<sub>s exac</sub>tl<sub>y</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>on on</sub> t<sub>ag s</sub>t<sub>a</sub>t<sub>es an</sub>d th<sub>e</sub> b<sub>onus po</sub>li<sub>cy</sub> i<sub>s po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se op</sub>ti<sub>ma</sub>l<sub>.</sub> I<sub>n par</sub>ti<sub>cu</sub>l<sub>ar,</sub>

$$
V _ { \widehat { u } } ^ { \widehat { \pi } _ { 0 } } ( x _ { s } ) = \Lambda + V _ { u } ^ { \pi _ { 0 } } ( s )
$$

for every source state s.

For any policy π supported on $A _ { \mathrm { a l l o w } }$ , <sup>it</sup>s <sup>lift</sup> πb rep<sup>l</sup>ays $\pi ^ { \prime } \boldsymbol { \mathrm { s } }$ B<sub>e</sub>ll<sub>man</sub> <sub>equa</sub>ti<sub>on</sub> <sub>over</sub> t<sub>wo</sub> lift<sub>e</sub>d <sub>s</sub>t<sub>eps</sub> <sub>a</sub>t <sub>non</sub>fi<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es</sub> <sub>an</sub>d <sub>uses</sub> th<sub>e payo</sub>f<sub>-preserv</sub>i<sub>ng se</sub>lf<sub>-</sub>l<sub>oop a</sub>t fi<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es.</sub> Di<sub>rec</sub>t <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>on us</sub>i<sub>ng</sub> $\beta ^ { \dot { 2 } } = \gamma _ { 0 }$ th<sub>ere</sub>f<sub>ore</sub> <sub>g</sub>i<sub>ves</sub>

$$
V _ { \widehat { u } } ^ { \widehat { \pi } } ( x _ { s } ) = V _ { u } ^ { \pi } ( s )
$$

for every s. No optimality claim is needed here, only that $\widehat { \pi } \mathrm { { ^ { s } } }$ <sub>va</sub>l<sub>ue</sub> <sub>ma</sub>t<sub>c</sub>h<sub>es</sub> $\pi ^ { \prime } \boldsymbol { \mathrm { s } }$ <sub>un</sub>d<sub>er</sub> th<sub>e</sub> <sub>same</sub> di<sub>scoun</sub>t<sub>-ma</sub>t<sub>c</sub>hi<sub>ng</sub> <sub>su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>on</sub> <sub>use</sub>d <sub>a</sub>b<sub>ove.</sub>

C<sub>om</sub>bi<sub>n</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>wo</sub> id<sub>en</sub>titi<sub>es</sub> <sub>a</sub>t $\boldsymbol { s } _ { t } .$

$$
\begin{array} { r l } & { \mathrm { R r e g } ( \widehat { \Pi } ) = \underset { \widehat { u } } { \operatorname* { s u p } } \Big ( V _ { \widehat { u } } ^ { * } \big ( \widehat { s _ { \iota } } \big ) - \underset { 1 \leq i \leq r } { \operatorname* { m a x } } V _ { \widehat { u } } ^ { \widehat { \pi } _ { i } } \big ( \widehat { s _ { \iota } } \big ) \Big ) } \\ & { \quad \quad \quad \quad = \underset { u } { \operatorname* { s u p } } \Big ( \Lambda + V _ { u } ^ { \pi _ { 0 } } \big ( s _ { \iota } \big ) - \underset { 1 \leq i \leq r } { \operatorname* { m a x } } V _ { u } ^ { \pi _ { i } } \big ( s _ { \iota } \big ) \Big ) } \\ & { \quad \quad \quad = \Lambda + \Delta _ { \mathcal { U } } \big ( \pi _ { 0 } , \Pi \big ) , } \end{array}
$$

<sub>us</sub>i<sub>ng po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se op</sub>ti<sub>ma</sub>lit<sub>y o</sub>f th<sub>e</sub> b<sub>onus po</sub>li<sub>cy, so</sub> $V _ { \widehat { u } } ^ { * } ( \widehat { s } _ { \iota } ) = V _ { \widehat { u } } ^ { \widehat { \pi } _ { 0 } } ( \widehat { s } _ { \iota } )$ <sub>.</sub> E<sub>very uncer</sub>t<sub>a</sub>i<sub>n source c</sub>h<sub>o</sub>i<sub>ce occurs a</sub>t <sub>one se</sub>l<sub>ec</sub>t<sub>or</sub> <sub>s</sub>t<sub>a</sub>t<sub>e, so s</sub>h<sub>are</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ces are no</sub>t <sub>cop</sub>i<sub>e</sub>d<sub>.</sub> P<sub>ro</sub>d<sub>uc</sub>t<sub>s o</sub>f <sub>source c</sub>h<sub>o</sub>i<sub>ce uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y se</sub>t<sub>s rema</sub>i<sub>n pro</sub>d<sub>uc</sub>t<sub>s, w</sub>hi<sub>c</sub>h <sub>preserves</sub> $( s , a ) -$ <sub>rec</sub>t<sub>angu</sub>l<sub>ar</sub>it<sub>y.</sub> R<sub>ep</sub>l<sub>ac</sub>i<sub>ng eac</sub>h <sub>non</sub>fi<sub>na</sub>l <sub>source e</sub>d<sub>ge</sub> b<sub>y</sub> it<sub>s</sub> t<sub>wo-e</sub>d<sub>ge</sub> t<sub>ag-se</sub>l<sub>ec</sub>t<sub>or pa</sub>th <sub>preserves a</sub> t<sub>opo</sub>l<sub>og</sub>i<sub>ca</sub>l <sub>or</sub>d<sub>er, w</sub>hil<sub>e eac</sub>h fi<sub>na</sub>l t<sub>ag an</sub>d th<sub>e ru</sub>i<sub>nous s</sub>i<sub>n</sub>k <sub>rema</sub>i<sub>ns a</sub>b<sub>sor</sub>bi<sub>ng.</sub> F<sub>or po</sub>li<sub>cy grap</sub>h<sub>s</sub> th<sub>e reac</sub>h<sub>a</sub>bilit<sub>y qua</sub>lifi<sub>er</sub> i<sub>n</sub> th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f <sub>a use</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> i<sub>s</sub> l<sub>oa</sub>d<sub>-</sub>b<sub>ear</sub>i<sub>ng</sub> <sub>ra</sub>th<sub>er</sub> th<sub>an</sub> i<sub>nc</sub>id<sub>en</sub>t<sub>a</sub>l<sub>:</sub> <sub>rou</sub>t<sub>e</sub> <sub>ac</sub>ti<sub>ons</sub> <sub>are</sub> i<sub>ns</sub>t<sub>a</sub>ll<sub>e</sub>d <sub>a</sub>t <sub>every</sub> <sub>se</sub>l<sub>ec</sub>t<sub>or</sub> <sub>s</sub>t<sub>a</sub>t<sub>e,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> <sub>se</sub>l<sub>ec</sub>t<sub>ors</sub> f<sub>or</sub> <sub>source</sub> <sub>ac</sub>ti<sub>ons</sub> th<sub>e</sub> lift<sub>e</sub>d <sub>o</sub>li<sub>c never c</sub>h<sub>ooses, an</sub>d th<sub>ose s</sub>t<sub>a</sub>t<sub>es are unreac</sub>h<sub>a</sub>bl<sub>e un</sub>d<sub>er</sub> it<sub>.</sub> R<sub>ea</sub>di<sub>n</sub> $G _ { \widehat { \pi } }$ <sub>as con</sub>t<sub>a</sub>i<sub>n</sub>i<sub>n ever</sub> t<sub>rans</sub>iti<sub>on</sub> th<sub>e o</sub>li<sub>c</sub> <sub>g</sub>i<sub>ves</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y,</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>on</sub>l<sub>y</sub> it<sub>s</sub> <sub>use</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ces,</sub> <sub>wou</sub>ld th<sub>ere</sub>f<sub>ore</sub> <sub>a</sub>dd <sub>e</sub>d<sub>ges</sub> th<sub>a</sub>t <sub>no</sub> <sub>run</sub> t<sub>raverses</sub> <sub>an</sub>d b<sub>rea</sub>k th<sub>e</sub> <sub>c</sub>l<sub>a</sub>i<sub>m.</sub> Thi<sub>s proves</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>acyc</sub>li<sub>c</sub>it<sub>y preserva</sub>ti<sub>on</sub> f<sub>or</sub> th<sub>e</sub> f<sub>u</sub>ll RMDP <sub>an</sub>d f<sub>or</sub> it<sub>s po</sub>li<sub>cy grap</sub>h<sub>s.</sub> □

# E Proofs of Theorems 2 to 4: Transferring Comparison Bounds to Certification

## E.1 Rectangular Given-Policy Hardness

Th<sub>e</sub> <sub>exac</sub>t lift t<sub>rans</sub>f<sub>ers</sub> th<sub>e</sub> B<sub>oo</sub>l<sub>ean</sub> <sub>an</sub>d <sub>square-roo</sub>t<sub>-sum</sub> <sub>compar</sub>i<sub>son</sub> b<sub>oun</sub>d<sub>s.</sub>

Theorem 2. Given-policy robust regret is coNP-hard and coSQRS-hard under $( s , a )$ -rectangular uncertainty. coNP-hardness already holds when every uncertain choice is two-Dirac and the only other stochastic rows are certain uniform splitters.

Proof. For the coNP lower bound, start with $\mathrm { C m p } ( \varphi )$ <sup>and a</sup>pp<sup>l</sup>y $\mathrm { L i f t } ( \mathrm { C m p } ( \varphi ) , \pi _ { C } )$ <sub>w</sub>ith <sub>s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on por</sub>tf<sub>o</sub>li<sub>o</sub> $\{ \pi _ { K } \}$ <sub>.</sub> Th<sub>e source</sub> <sub>cons</sub>i<sub>s</sub>t<sub>s o</sub>f d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c po</sub>li<sub>c</sub>i<sub>es</sub> $\pi _ { C } , \pi _ { K }$ i<sub>n an acyc</sub>li<sub>c</sub> $( s , a )$ <sub>-rec</sub>t<sub>angu</sub>l<sub>ar</sub> RMDP <sub>w</sub>ith i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t t<sub>wo-</sub>Di<sub>rac uncer</sub>t<sub>a</sub>i<sub>n c</sub>h<sub>o</sub>i<sub>ces,</sub> <sub>one cer</sub>t<sub>a</sub>i<sub>n un</sub>if<sub>orm sp</sub>litt<sub>er, an</sub>d di<sub>scoun</sub>t $\gamma _ { 0 }$ <sub>, an</sub>d it <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub>

$$
\varphi \in \mathrm { U N S A T } \quad \Longleftrightarrow \quad \Delta { u } ( \pi _ { C } , \pi _ { K } ) \leq 2 - { \frac { 1 } { 2 n } } .
$$

F<sub>or</sub> th<sub>e</sub> lift<sub>e</sub>d <sub>can</sub>did<sub>a</sub>t<sub>e</sub> $\widehat { \pi } _ { K }$ th<sub>e</sub> l<sub>emma g</sub>i<sub>ves</sub>

$$
\operatorname { R r e g } ( \widehat { \pi } _ { K } ) = \Lambda + \Delta _ { { \mathcal { U } } } ( \pi _ { C } , \pi _ { K } ) .
$$

Th<sub>us, w</sub>ith th<sub>e ra</sub>ti<sub>ona</sub>l t<sub>arge</sub>t th<sub>res</sub>h<sub>o</sub>ld $\begin{array} { r } { \widehat { t } = \Lambda + 2 - \frac { 1 } { 2 n } . } \end{array}$ <sub>,</sub> th<sub>e</sub> lift<sub>e</sub>d <sub>regre</sub>t i<sub>ns</sub>t<sub>ance</sub> i<sub>s</sub> <sub>a</sub> <sub>yes-</sub>i<sub>ns</sub>t<sub>ance</sub> <sub>exac</sub>tl<sub>y</sub> <sub>w</sub>h<sub>en</sub> $\varphi \in \mathrm { U N S A T }$ The structural restrictions follow from Lemma 30, proving coNP-hardness.

For the coSQRS lower bound, use the deterministic policies $\pi _ { u } , \pi _ { v }$ and threshold t produced by Theorem 18. That construction <sub>uses</sub> th<sub>e</sub> <sub>same</sub> di<sub>scoun</sub>t $\gamma _ { 0 } .$ <sub>,</sub> i<sub>s</sub> $( s , a )$ <sub>-rec</sub>t<sub>angu</sub>l<sub>ar,</sub> <sub>an</sub>d h<sub>as</sub> t<sub>wo-</sub>Di<sub>rac</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub> <sub>c</sub>h<sub>o</sub>i<sub>ces.</sub> A<sub>pp</sub>l<sub>y</sub> $\mathrm { L i f t } \dot { (} N , \pi _ { u } )$ <sub>w</sub>ith <sub>s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on can</sub>did<sub>a</sub>t<sub>e</sub> $\{ \pi _ { v } \}$ <sub>.</sub> Th<sub>en</sub>

$$
\Delta _ { \mathcal { U } } ( \pi _ { u } , \pi _ { v } ) \leq t \quad \Longleftrightarrow \quad \operatorname { R r e g } ( \widehat { \pi } _ { v } ) \leq \Lambda + t .
$$

Hence given-polic<sub>y</sub> robust regret is coSQRS-hard. Together the two reductions prove the theorem.

## E.2 Rectangular Portfolio Certification

Th<sub>e</sub> f<sub>am</sub>il<sub>y</sub> f<sub>orm o</sub>f th<sub>e exac</sub>t lift t<sub>rans</sub>f<sub>ers</sub> th<sub>e rec</sub>t<sub>angu</sub>l<sub>ar por</sub>tf<sub>o</sub>li<sub>o-compar</sub>i<sub>son</sub> b<sub>oun</sub>d<sub>.</sub>

Theorem 3. Portfolio-regret certification is ∀R-hard, alreadyfor deterministicportfolios and acyclic $( s , a )$ -rectangular RMDPs with two-successor uncertain choices.

Proof. The evaluator above was instantiated at $\gamma _ { 0 }$ <sub>, so app</sub>l<sub>y</sub> L<sub>emma</sub> 30 t<sub>o</sub> it<sub>s re</sub>f<sub>erence po</sub>li<sub>cy an</sub>d <sub>exp</sub>li<sub>c</sub>it <sub>por</sub>tf<sub>o</sub>li<sub>o.</sub> Th<sub>e exac</sub>t id<sub>en</sub>tit<sub>y</sub>

$$
\operatorname { R r e g } ( \widehat { \Pi } ) = \Lambda + \Delta _ { \mathcal { U } } ( \pi _ { 0 } , \Pi )
$$

translates the zero com<sub>p</sub>arison threshold to the rational re<sub>g</sub>ret threshold Λ. The structural conclusions are those of Lemma 30. Therefore <sub>p</sub>ortfolio-re<sub>g</sub>ret certification is ∀R-hard under all restrictions stated in the theorem. □

## E.3 General-Polytope Regret Certification

M<sub>em</sub>b<sub>ers</sub>hi<sub>p</sub> i<sub>s</sub> th<sub>e</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on</sub> <sub>case</sub> <sub>o</sub>f Th<sub>eorem</sub> 1<sub>.</sub> F<sub>or</sub> h<sub>ar</sub>d<sub>ness,</sub> <sub>a</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> b<sub>e</sub>t<sub>ween</sub> th<sub>e</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>eva</sub>l<sub>ua</sub>t<sub>or</sub> <sub>an</sub>d <sub>zero</sub> <sub>ma</sub>k<sub>es</sub> <sub>regre</sub>t <sub>compu</sub>t<sub>e</sub> th<sub>e</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l’<sub>s</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>par</sub>t<sub>.</sub>

Theorem 4. Given-policy robust regret is ∀R-complete under general rational polytopic uncertainty, even for deterministic policies.

Th<sub>e mem</sub>b<sub>ers</sub>hi<sub>p</sub> h<sub>a</sub>lf <sub>o</sub>f Th<sub>eorem</sub> 4 i<sub>s</sub> th<sub>e s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on-</sub>f<sub>am</sub>il<sub>y case o</sub>f L<sub>emma</sub> 13<sub>.</sub>

## Polynomial-evaluation hardness.

Definition 11 (Positive-part gadget). Given $M _ { f }$ from Definition 8, add a fresh initial state $s ^ { + }$ with two reward-zero actions: $a _ { f }$ enters $M _ { f }$ at $s _ { f } ,$ , and $a _ { \perp }$ enters a zero-reward sink. Let π<sub>⊥</sub> choose $a _ { \perp }$ at $s ^ { + }$ . All choices left undescribed after enlarging the global action set use the same ruinous-sink completion as $M _ { f }$

Lemma 31 (Positive-part regret). The fixed policy in Definition 11 satisfies

$$
\operatorname { R r e g } ( \pi _ { \perp } ) = \operatorname* { s u p } _ { p \in [ 0 , 1 ] ^ { m } } \operatorname* { m a x } \{ \gamma _ { 0 } f ( p ) , 0 \} .
$$

Consequently $\mathrm { R r e g } ( \pi _ { \perp } ) \le 0$ exactly when $\forall p \in [ 0 , 1 ] ^ { m } : f ( p ) \leq 0 .$

Proof. At realization $p ,$ th<sub>e</sub> t<sub>wo</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>ac</sub>ti<sub>ons</sub> h<sub>ave va</sub>l<sub>ues</sub> $\gamma _ { 0 } f ( \boldsymbol { p } )$ <sub>an</sub>d <sub>zero</sub> b<sub>y</sub> L<sub>emma</sub> 22<sub>.</sub> Th<sub>e</sub>i<sub>r pos</sub>iti<sub>ve par</sub>t i<sub>s prec</sub>i<sub>se</sub>l<sub>y</sub> th<sub>e</sub> <sub>s</sub>h<sub>or</sub>tf<sub>a</sub>ll <sub>o</sub>f $\pi _ { \perp }$ <sub>.</sub> T<sub>a</sub>ki<sub>ng</sub> th<sub>e supremum proves</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y.</sub> R<sub>egre</sub>t i<sub>s nonnega</sub>ti<sub>ve, so</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>zero</sub> i<sub>s</sub> ti<sub>g</sub>ht<sub>.</sub> □

E<sub>xamp</sub>l<sub>e</sub> 4 ill<sub>us</sub>t<sub>ra</sub>t<sub>es</sub> th<sub>e zero-</sub>th<sub>res</sub>h<sub>o</sub>ld <sub>case.</sub>

Example 4. For $f ( p ) = p - p ^ { 2 }$ , the unique compliant policy inside $M _ { f }$ has value $p - p ^ { 2 }$ and

$$
\mathrm { R r e g } ( \pi _ { \perp } ) = \gamma _ { 0 } \operatorname* { m a x } _ { p \in [ 0 , 1 ] } ( p - p ^ { 2 } ) = \gamma _ { 0 } / 4 > 0 .
$$

Thus this instance is a no-instance at threshold zero.

Proof of Theorem 4. Apply Lemma 31 to the bounded universal-polynomial instances used in Lemma 23. Together with the <sub>mem</sub>b<sub>ers</sub>hi<sub>p enco</sub>di<sub>ng a</sub>b<sub>ove,</sub> thi<sub>s proves</sub> th<sub>e</sub> th<sub>eorem.</sub> □

## F Proof of Theorem 6: Combinatorial Minimal-Regret Hardness

W<sub>e prove</sub> th<sub>e</sub> B<sub>oo</sub>l<sub>ean</sub> l<sub>ower</sub> b<sub>oun</sub>d<sub>s</sub> f<sub>or m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t<sub>.</sub>

Theorem 6. Minimal robust regret is NP-hard and coNP-hard on $( s , a )$ -rectangular RMDPs in which every uncertain choice is two-Dirac and the only other stochastic rows are certain uniform splitters.

B<sub>o</sub>th <sub>re</sub>d<sub>uc</sub>ti<sub>ons</sub> fi<sub>rs</sub>t <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub> th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> t<sub>o</sub> <sub>a</sub>ll<sub>owe</sub>d <sub>ac</sub>ti<sub>ons.</sub> L<sub>emma</sub> 32 <sub>ma</sub>k<sub>es</sub> di<sub>sa</sub>ll<sub>owe</sub>d <sub>ac</sub>ti<sub>ons</sub> <sub>ru</sub>i<sub>nous</sub> <sub>an</sub>d th<sub>ere</sub>b<sub>y</sub> transfers this restricted minimum to unrestricted synthesis within an arbitrarily small ε.

## F.1 Policy Restriction

Fix an RMDP N, nonempty allowed action sets, and rational $\varepsilon > 0$ <sub>.</sub> Th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on</sub> b<sub>e</sub>l<sub>ow</sub> t<sub>urns res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d <sub>syn</sub>th<sub>es</sub>i<sub>s</sub> i<sub>n</sub>t<sub>o</sub> <sub>or</sub>di<sub>nary unres</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d <sub>syn</sub>th<sub>es</sub>i<sub>s: every</sub> di<sub>sa</sub>ll<sub>owe</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> i<sub>s</sub> h<sub>an</sub>d<sub>e</sub>d t<sub>o</sub> th<sub>e a</sub>d<sub>versary as an ex</sub>t<sub>ra op</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>can co</sub>ll<sub>apse s</sub>t<sub>ra</sub>i<sub>g</sub>ht i<sub>n</sub>t<sub>o a s</sub>t<sub>a</sub>t<sub>e so</sub> b<sub>a</sub>d th<sub>a</sub>t <sub>no m</sub>i<sub>n-regre</sub>t <sub>po</sub>li<sub>cy w</sub>ill <sub>ever c</sub>h<sub>oose</sub> t<sub>o use</sub> it<sub>.</sub> F<sub>or</sub> $\mathcal { P } \in \{ \Pi ^ { \mathrm { \check { M R } } } , \Pi ^ { \mathrm { M D } } \}$ <sub>, wr</sub>it<sub>e</sub> $\mathcal { P } _ { \mathrm { a l l o w } }$ f<sub>or</sub> th<sub>e po</sub>li<sub>c</sub>i<sub>es</sub> i<sub>n</sub> $\mathcal { P }$ <sub>suppor</sub>t<sub>e</sub>d <sub>on</sub> th<sub>e a</sub>ll<sub>owe</sub>d <sub>ac</sub>ti<sub>ons.</sub>

Definition 12 (ε-restricted RMDP). Let $N = ( S , A , \mathcal { U } , R , s _ { \iota } , \gamma )$ be an RMDP and let $A _ { \mathrm { a l l o w } } ( s ) \subseteq A$ be a nonempty set of allowed actions for every $s \in S .$ . Suppose every disallowed choice has a fixed transition distribution, i.e., $\mathcal { U } _ { ( s , a ) } = \bar { \{ p _ { s , a } \} } \bar { f o r }$ every $s \in S$ and $a \in A \setminus A _ { \mathrm { a l l o w } } ( s )$ . Choices supplied by ruinous-sink completion satisfy this hypothesis. The construction proceeds in two steps.

First it unfolds the absorbing finals of $N ,$ so that no state both self-loops and can leave. Add afresh zero-reward absorbing sink $\perp _ { 0 } ,$ , every action ofwhich has reward zero and row $\{ \delta _ { \perp _ { 0 } } \}$ , and let every action be allowed there. At every absorbing final $f$ ofN, replace the reward $R ( f , a )$ by $R ( f , a ) / ( 1 - \gamma )$ and the row by $\{ \delta _ { \perp _ { 0 } } \}$ , keeping $A _ { \mathrm { a l l o w } } ( f )$ as it is. An action a at f had value $R ( { \bar { f } } , a ) / ( 1 - \gamma )$ before the step, since it self-looped forever, and it collects exactly that reward once and then nothing after the step. The unfolding therefore leaves the value of every choice at $f$ unchanged. Every state therefore keeps its value under every stationary randomized policy and every realization, so the optimal value and every robust regret are unchanged as well. The unfolding adds no uncertain choice, preserves rectangularity and acyclicity, and leaves $\perp _ { 0 }$ as the only absorbing final. It raises the largest reward magnitude to at most the largest value magnitude of N, hence multiplies $V _ { \mathrm { m a x } }$ below by at most $1 / ( 1 - \gamma )$ and keeps every constant ofpolynomial encoding length. Write N for the unfolded RMDPfrom here on.

Second, let $\varepsilon > 0$ be rational, and set

$$
V _ { \mathrm { m a x } } = \operatorname* { m a x } \left\{ 1 , \frac { \operatorname* { m a x } _ { s , a } \left| R ( s , a ) \right| } { 1 - \gamma } \right\} , \qquad Z _ { \varepsilon } = \frac { ( 2 - \gamma ) V _ { \mathrm { m a x } } + 4 V _ { \mathrm { m a x } } ^ { 2 } / \varepsilon } { \gamma } .
$$

The ε-restricted RMDP is the tuple $N _ { \varepsilon } = ( S _ { \varepsilon } , A _ { \varepsilon } , \mathcal { U } _ { \varepsilon } , R _ { \varepsilon } , s _ { \iota \varepsilon } , \gamma _ { \varepsilon } ) ,$ , where:

$S _ { \varepsilon } = S \cup \{ \bot _ { \mathrm { r } } \}$ , adding onefresh ruinous sink $\perp _ { \mathrm { r } } ;$

$A _ { \varepsilon } = A , \gamma _ { \varepsilon } = \gamma$ , and $s _ { \iota \varepsilon } = s _ { \iota } ;$

$R _ { \varepsilon } ( s , a ) = R ( s , a )$ for every $s \in S$ and $a \in A ,$ while every action at $\perp$ <sub>r</sub> has reward $- ( 1 - \gamma ) Z _ { \varepsilon }$

$\mathcal { U } _ { \varepsilon } ( s , a ) = \mathcal { U } ( s , a )$ for every allowed choice, $a \in A _ { \mathrm { a l l o w } } ( s ) ;$

$\mathcal { U } _ { \varepsilon } ( s , a ) = \big \{ \theta p _ { s , a } + ( 1 - \theta ) \delta _ { \perp _ { \mathrm { r } } } : \theta \in [ 0 , 1 ] \big \}$ for every disallowed choice, $a \in A \setminus A _ { \mathrm { a l l o w } } ( s ) ;$

• every action at $\perp _ { \mathrm { r } }$ has row $\{ \delta _ { \perp _ { \mathrm { r } } } \}$

Th<sub>e se</sub>lf<sub>-</sub>l<sub>oop rewar</sub>d <sub>a</sub>t $\perp _ { \mathrm { r } }$ <sub>g</sub>i<sub>ves</sub> it <sub>va</sub>l<sub>ue</sub> $- Z _ { \varepsilon }$ <sub>un</sub>d<sub>er every rea</sub>li<sub>za</sub>ti<sub>on:</sub> $V ( \perp _ { \mathrm { r } } ) = - ( 1 - \gamma ) Z _ { \varepsilon } + \gamma V ( \perp _ { \mathrm { r } } )$ f<sub>orces</sub> $V ( \bot _ { \mathrm { r } } ) = - Z _ { \varepsilon }$ At th<sub>e en</sub>d<sub>po</sub>i<sub>n</sub>t $\theta = 1 , \mathcal { U } _ { \varepsilon } ( s , a )$ <sub>re</sub>d<sub>uces</sub> t<sub>o</sub> $\{ p _ { s , a } \}$ , exact<sup>l</sup><sub>y</sub> $N \mathbf { \bar { s } }$ <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>c</sub>h<sub>o</sub>i<sub>ce.</sub> $\mathbf { A } \mathbf { t } \theta = 0$ <sub>,</sub> it <sub>co</sub>ll<sub>apses</sub> t<sub>o</sub> $\{ \delta _ { \perp _ { \mathrm { r } } } \}$ <sub>, rou</sub>ti<sub>ng s</sub>t<sub>ra</sub>i<sub>g</sub>ht into the sink. The unfolding is what makes the interpolation safe. When N is acyclic its only self-loops sit at absorbing finals, so <sub>a</sub>ft<sub>er</sub> th<sub>e un</sub>f<sub>o</sub>ldi<sub>ng</sub> th<sub>e on</sub>l<sub>y se</sub>lf<sub>-</sub>l<sub>oop</sub>i<sub>ng s</sub>t<sub>a</sub>t<sub>e</sub> i<sub>s</sub> $\perp _ { 0 } ,$ <sub>w</sub>h<sub>ere no</sub>thi<sub>ng</sub> i<sub>s</sub> di<sub>sa</sub>ll<sub>owe</sub>d<sub>.</sub> E<sub>very</sub> i<sub>n</sub>t<sub>erpo</sub>l<sub>a</sub>t<sub>e</sub>d <sub>row</sub> th<sub>ere</sub>f<sub>ore</sub> li<sub>es</sub> b<sub>e</sub>t<sub>ween</sub> t<sub>wo</sub> di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ons</sub> th<sub>a</sub>t b<sub>o</sub>th l<sub>eave</sub> th<sub>e</sub>i<sub>r s</sub>t<sub>a</sub>t<sub>e, an</sub>d <sub>no s</sub>t<sub>a</sub>t<sub>e acqu</sub>i<sub>res a se</sub>lf<sub>-</sub>l<sub>oop</sub> it did <sub>no</sub>t <sub>a</sub>l<sub>rea</sub>d<sub>y</sub> h<sub>ave.</sub>

Lemma 32 (Policy restriction). For either $\mathcal { P } = \Pi ^ { \mathrm { M R } } o r \mathcal { P } = \Pi ^ { \mathrm { M D } }$

$$
\operatorname* { i n f } _ { \rho \in \mathcal { P } } \operatorname { R r e g } ^ { N _ { \varepsilon } } ( \rho ) \leq \operatorname* { i n f } _ { \pi \in \mathcal { P } _ { \mathrm { a l l o w } } } \operatorname { R r e g } ^ { N } ( \pi ) ,
$$

$$
\operatorname* { i n f } _ { \rho \in \mathcal { P } } \operatorname { R r e g } ^ { N _ { \varepsilon } } ( \rho ) \geq \operatorname* { i n f } _ { \pi \in \mathcal { P } _ { \mathrm { a l l o w } } } \operatorname { R r e g } ^ { N } ( \pi ) - \varepsilon .
$$

![](images/c3b7e87664cfd21b72f078a5e1f64850ce6495382afc69670ff87ff9f9f7b616.jpg)  
Fi<sub>gure</sub> 8<sub>:</sub> P<sub>o</sub>li<sub>cy</sub> <sub>res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>on</sub> b<sub>e</sub>f<sub>ore</sub> <sub>an</sub>d <sub>a</sub>ft<sub>er</sub> th<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on.</sub> Th<sub>e</sub> <sub>a</sub>ll<sub>owe</sub>d <sub>ac</sub>ti<sub>on</sub> <sub>rema</sub>i<sub>ns</sub> <sub>unc</sub>h<sub>ange</sub>d<sub>,</sub> <sub>w</sub>hil<sub>e</sub> <sub>na</sub>t<sub>ure</sub> <sub>can</sub> <sub>rou</sub>t<sub>e</sub> th<sub>e</sub> di<sub>sa</sub>ll<sub>owe</sub>d <sub>ac</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e ru</sub>i<sub>nous s</sub>i<sub>n</sub>k<sub>.</sub>

E<sub>xamp</sub>l<sub>e</sub> 5 ill<sub>us</sub>t<sub>ra</sub>t<sub>es</sub> th<sub>e</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> <sub>on</sub> t<sub>wo</sub> <sub>s</sub>t<sub>a</sub>t<sub>es.</sub>

Example 5 (A two-state restriction). Let the source RMDP have states s and $^ { g , }$ with s initial and g an absorbing zero-payof terminal. $A t \ s ,$ both the allowed action a and the disallowed action b move deterministically to $^ { g , }$ and $A _ { \mathrm { a l l o w } } ( s ) = \{ a \}$ . In Restrict $( N , A _ { \mathrm { a l l o w } } , \varepsilon )$ , the unfolding first sends g to the fresh zero sink $\perp _ { 0 }$ in one step, leaving its value at zero. Action a is then unchanged, while b reaches g with probability θ and $\perp _ { \mathrm { r } }$ with probability $1 - \theta ,$ as shown in Figure 8. The allowed policy reproduces the restricted source optimum, and Lemma 32 places the unrestricted optimum in the intervalfrom that value minu ε to that value. Thus the transformation introduces at most the promised ε gap, which is zero in this symmetric example because choosing a has regret zero.

ProofofLemma 32. The unfolding step changes no state’s value under any stationary policy or realization, so it changes neither $\mathrm { R r e g } ^ { N }$ nor the set of allowed policies, and N below denotes the unfolded RMDP. Fix a source realization u and let $W = V _ { u , \theta \equiv 1 } ^ { * }$ i<sub>n</sub> $N _ { \varepsilon } . \mathrm { A t }$ th<sub>e a</sub>ll<sub>-norma</sub>l <sub>rea</sub>li<sub>za</sub>ti<sub>on,</sub> th<sub>e or</sub>i i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es re ro</sub>d<sub>uce</sub> $N _ { u } , V ( \bot _ { \mathrm { r } } ) = - Z _ { \varepsilon }$ <sub>, an</sub>d <sub>no s</sub>i<sub>n</sub>k<sub>-rou</sub>t<sub>e</sub>d <sub>ac</sub>ti<sub>on</sub> i<sub>mproves on an</sub> <sub>or</sub>i<sub>g</sub>i<sub>na</sub>l <sub>ac</sub>ti<sub>on.</sub> Th<sub>us</sub> $W ( s ) = V _ { u } ^ { * } ( s )$ f<sub>or</sub> $s \in S$ <sub>an</sub>d $W ( \bar { \perp _ { \mathrm { r } } } ) = - Z _ { \varepsilon }$ <sub>.</sub> F<sub>or</sub> <sub>a</sub> di<sub>sa</sub>ll<sub>owe</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>an</sub>d <sub>ar</sub>bit<sub>rary</sub> $\theta \in [ 0 , 1 ]$ <sub>,</sub> it<sub>s</sub> b<sub>ac</sub>k<sub>up</sub> at W is

$$
R ( s , a ) + \gamma \left( \theta \sum _ { s ^ { \prime } } p _ { s , a } ( s ^ { \prime } ) W ( s ^ { \prime } ) + ( 1 - \theta ) ( - Z _ { \varepsilon } ) \right) \leq R ( s , a ) + \gamma \sum _ { s ^ { \prime } } p _ { s , a } ( s ^ { \prime } ) W ( s ^ { \prime } ) ,
$$

b<sub>ecause</sub> $- Z _ { \varepsilon } \leq - V _ { \mathrm { m a x } } \leq \mathrm { m i n } _ { s \in S } W ( s )$ <sub>.</sub> All<sub>owe</sub>d b<sub>ac</sub>k<sub>ups</sub> <sub>are</sub> <sub>unc</sub>h<sub>ange</sub>d<sub>,</sub> <sub>so</sub> th<sub>e</sub> <sub>op</sub>ti<sub>ma</sub>l B<sub>e</sub>ll<sub>man</sub> <sub>opera</sub>t<sub>or</sub> <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $T _ { u , \theta } W \leq$ W. Monotonicity and contraction give $\dot { V } _ { u . \theta } ^ { * } \le W$ <sub>.</sub> A<sub>n a</sub>ll<sub>owe</sub>d <sub>po</sub>li<sub>cy never uses a mo</sub>difi<sub>e</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce,</sub> h<sub>ence</sub> it<sub>s va</sub>l<sub>ue</sub> i<sub>s</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t of θ and agrees with its value in $N _ { u }$ <sub>.</sub> It<sub>s</sub> <sub>regre</sub>t i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>max</sub>i<sub>m</sub>i<sub>ze</sub>d <sub>a</sub>t $\theta \equiv 1$ <sub>, an</sub>d <sub>every a</sub>ll<sub>owe</sub>d <sub>po</sub>li<sub>cy</sub> h<sub>as</sub> th<sub>e same ro</sub>b<sub>us</sub>t re<sub>g</sub>ret <sup>i</sup>n $\bar { N }$ <sub>an</sub>d $N _ { \varepsilon }$ <sub>.</sub> Thi<sub>s</sub> <sub>proves</sub> th<sub>e</sub> <sub>upper</sub> i<sub>nequa</sub>lit<sub>y.</sub>

Fix an arbitrary policy ρ and condition it on allowed actions at every state, using an arbitrary allowed action if the conditioning <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> i<sub>s zero.</sub> C<sub>a</sub>ll th<sub>e resu</sub>lt ${ \bar { \rho } } .$ Th<sub>e con</sub>diti<sub>on</sub>i<sub>ng</sub> i<sub>s</sub> t<sub>o</sub>t<sub>a</sub>l b<sub>ecause</sub> $A _ { \mathrm { a l l o w } } ( s )$ <sup>is nonem</sup>p<sup>t</sup>y <sup>at e</sup>v<sup>er</sup>y <sup>state</sup>. <sup>It</sup> p<sup>reser</sup>v<sup>es</sup> <sub>mem</sub>b<sub>ers</sub>hi<sub>p</sub> i<sub>n</sub> $\Pi ^ { \mathrm { M R } }$ <sub>, an</sub>d it <sub>preserves</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>sm w</sub>h<sub>en</sub> $\mathcal { P } = \Pi ^ { \mathrm { M D } }$

Th<sub>e we</sub>i<sub>g</sub>ht b<sub>e</sub>l<sub>ow</sub> d<sub>epen</sub>d<sub>s on</sub> th<sub>e rea</sub>li<sub>za</sub>ti<sub>on, so</sub> fi<sub>x a source rea</sub>li<sub>za</sub>ti<sub>on</sub> $u \in \mathcal { U }$ <sub>an</sub>d <sub>argue po</sub>i<sub>n</sub>t<sub>w</sub>i<sub>se</sub> i<sub>n</sub> it<sub>,</sub> t<sub>a</sub>ki<sub>ng</sub> th<sub>e supremum</sub> over u only at the end. Couple the two trajectories by reusing the same transition draw and, while an allowed action is selected, the same normalized action draw. The two trajectories then agree until $\rho$ first chooses a nonallowed action at time τ. Put

$$
d = \mathbb { E } [ \gamma ^ { \tau } \mathbf { 1 } _ { \{ \tau < \infty \} } ] .
$$

At the joint realization $( u , \theta \equiv 1 )$ <sub>, every ac</sub>ti<sub>on va</sub>l<sub>ue,</sub> i<sub>nc</sub>l<sub>u</sub>di<sub>ng</sub> it<sub>s</sub> i<sub>mme</sub>di<sub>a</sub>t<sub>e rewar</sub>d <sub>an</sub>d di<sub>scoun</sub>t<sub>e</sub>d <sub>con</sub>ti<sub>nua</sub>ti<sub>on,</sub> li<sub>es</sub> i<sub>n</sub> $[ - V _ { \mathrm { m a x } } , V _ { \mathrm { m a x } } ]$ <sub>.</sub> C<sub>on</sub>diti<sub>ona</sub>l <sub>on</sub> th<sub>e</sub> fi<sub>rs</sub>t dif<sub>er</sub>i<sub>ng c</sub>h<sub>o</sub>i<sub>ce,</sub> $\rho$ th<sub>ere</sub>f<sub>ore ga</sub>i<sub>ns a</sub>t <sub>mos</sub>t $2 V _ { \mathrm { m a x } }$ over ${ \bar { \rho } } ,$ so

$$
V _ { u , \theta \equiv 1 } ^ { \rho } ( s _ { \iota } ) \leq V _ { u } ^ { \bar { \rho } } ( s _ { \iota } ) + 2 V _ { \operatorname* { m a x } } d , \qquad \mathrm { R r e g } ^ { N _ { c } } ( \rho ) \geq V _ { u } ^ { * } ( s _ { \iota } ) - V _ { u } ^ { \bar { \rho } } ( s _ { \iota } ) - 2 V _ { \operatorname* { m a x } } d .
$$

At the joint realization extending the same u with $\theta = 0$ <sub>a</sub>t <sub>every</sub> <sub>nona</sub>ll<sub>owe</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ce,</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>suc</sub>h <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> i<sub>ncurs</sub> l<sub>oss</sub> <sub>a</sub>t l<sub>eas</sub>t

$$
\gamma Z _ { \varepsilon } - ( 2 - \gamma ) V _ { \mathrm { m a x } } = \frac { 4 V _ { \mathrm { m a x } } ^ { 2 } } { \varepsilon } ,
$$

<sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub>

$$
\mathrm { R r e g } ^ { N _ { \varepsilon } } ( \rho ) \geq \frac { 4 V _ { \mathrm { m a x } } ^ { 2 } } { \varepsilon } d .
$$

Both endpoints are valid joint realizations extending this same $u ,$ b<sub>ecause</sub> th<sub>e mo</sub>difi<sub>e</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ces are</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t<sub>.</sub> $\mathrm { I f } 2 V _ { \mathrm { m a x } } d \leq \varepsilon .$ the first bound loses at most ε. Otherwise the second exceeds $2 V _ { \mathrm { m a x } }$ <sub>an</sub>d h<sub>ence</sub> d<sub>om</sub>i<sub>na</sub>t<sub>es every poss</sub>ibl<sub>e regre</sub>t i<sub>n</sub> $N$ <sub>.</sub> Eith<sub>er</sub> <sup>wa</sup>y

$$
\mathrm { R r e g } ^ { N _ { \varepsilon } } ( \rho ) \geq V _ { u } ^ { * } ( s _ { \iota } ) - V _ { u } ^ { \bar { \rho } } ( s _ { \iota } ) - \varepsilon .
$$

The left side does not depend on u, so taking the supremum over $u \in \mathcal { U }$ <sub>g</sub><sup>i</sup>ves

$$
\mathrm { R r e g } ^ { N _ { \varepsilon } } ( \rho ) \ge \mathrm { R r e g } ^ { N } ( \bar { \rho } ) - \varepsilon \ge \operatorname* { i n f } _ { \pi \in \mathcal { P } _ { \mathrm { a l l o w } } } \mathrm { R r e g } ^ { N } ( \pi ) - \varepsilon .
$$

T<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> i<sub>n</sub>fi<sub>mum</sub> <sub>over</sub> $\rho$ proves the lower inequality. For B-bit rational inputs $V _ { \mathrm { m a x } }$ <sub>an</sub>d $\varepsilon ,$ th<sub>e</sub> f<sub>ormu</sub>l<sub>a a</sub>b<sub>ove</sub> i<sub>ves</sub> $Z _ { \varepsilon }$ polynomial bit length in B and the model size. Each modified choice is independent, so rectangularity is preserved, and a Dirac $p _ { s , a }$ <sub>y</sub>i<sub>e</sub>ld<sub>s a</sub> t<sub>wo-</sub>Di<sub>rac segmen</sub>t<sub>.</sub> A<sub>cyc</sub>li<sub>c</sub>it<sub>y</sub> i<sub>s preserve</sub>d <sub>as we</sub>ll<sub>: a</sub>ft<sub>er</sub> th<sub>e un</sub>f<sub>o</sub>ldi<sub>ng,</sub> $\perp _ { 0 }$ is the only self-looping state of N and <sub>no</sub>thi<sub>ng</sub> i<sub>s</sub> di<sub>sa</sub>ll<sub>owe</sub>d th<sub>ere, so every</sub> i<sub>n</sub>t<sub>erpo</sub>l<sub>a</sub>t<sub>e</sub>d <sub>row</sub> l<sub>eaves</sub> it<sub>s s</sub>t<sub>a</sub>t<sub>e, an</sub>d th<sub>e</sub> t<sub>wo a</sub>dd<sub>e</sub>d <sub>s</sub>i<sub>n</sub>k<sub>s are a</sub>b<sub>sor</sub>bi<sub>ng</sub> fi<sub>na</sub>l<sub>s.</sub> □

## F.2 coNP-Hardness

A<sub>pp</sub>l<sub>y</sub> the exact lift (Definition 10) to the com<sub>p</sub>arison instance of Theorem 14, which builds from $\varphi$ <sub>a</sub> B<sub>oo</sub>l<sub>ean</sub> <sub>compar</sub>i<sub>son</sub> RMDP <sub>w</sub>ith <sub>a</sub> <sub>c</sub>l<sub>ause</sub> <sub>po</sub>li<sub>cy</sub> $\pi _ { C }$ <sub>an</sub>d <sub>a</sub> <sub>cons</sub>i<sub>s</sub>t<sub>ency</sub> <sub>po</sub>li<sub>cy</sub> $\pi _ { K }$ <sub>suc</sub>h th<sub>a</sub>t

$$
\Delta _ { { \boldsymbol u } } ( \pi _ { C } , \pi _ { K } ) \leq 2 - { \frac { 1 } { n } } { \mathrm { i f ~ } } \varphi \in \mathrm { U N S A T } , \qquad \Delta _ { { \boldsymbol u } } ( \pi _ { C } , \pi _ { K } ) = 2 { \mathrm { i f ~ } } \varphi \in \mathrm { S A T } .
$$

T<sub>a</sub>k<sub>e</sub> $\pi _ { 0 } = \pi _ { C }$ <sub>as</sub> th<sub>e</sub> lift’<sub>s</sub> <sub>re</sub>f<sub>erence</sub> <sub>po</sub>li<sub>cy.</sub> N<sub>o</sub> <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> <sub>proper</sub>t<sub>y</sub> <sub>o</sub>f $\pi _ { C }$ i<sub>s</sub> <sub>nee</sub>d<sub>e</sub>d<sub>,</sub> <sub>on</sub>l<sub>y</sub> th<sub>a</sub>t it i<sub>s</sub> th<sub>e</sub> fi<sub>xe</sub>d <sub>po</sub>li<sub>cy</sub> th<sub>e</sub> lift <sub>compares aga</sub>i<sub>ns</sub>t<sub>.</sub> R<sub>es</sub>t<sub>r</sub>i<sub>c</sub>t th<sub>e can</sub>did<sub>a</sub>t<sub>es a</sub>d<sub>m</sub>itt<sub>e</sub>d b<sub>y</sub> th<sub>e</sub> lift t<sub>o</sub> $\pi _ { K }$ <sup>alone: at e</sup>v<sup>er</sup>y <sup>ta</sup>g <sup>state</sup> $x _ { s } .$ , set $A _ { \mathrm { a l l o w } } ( x _ { s } ) = \{ \pi _ { K } ( s ) \}$ while every selector state has only its allowed route action. Recall from Definition 10 that a tag state’s available actions are $A _ { \mathrm { a l l o w } } ( s ) \cup .$ {bonus}, never the reference’s own action directly. The action $\pi _ { C } ( s )$ is reachable only through bonus and $y _ { s , \pi _ { C } ( s ) }$ Th<sub>us po</sub>li<sub>cy res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>on</sub> k<sub>eeps a can</sub>did<sub>a</sub>t<sub>e</sub> f<sub>rom</sub> t<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> b<sub>onus rou</sub>t<sub>e mean</sub>t <sub>on</sub>l<sub>y</sub> f<sub>or</sub> th<sub>e re</sub>f<sub>erence.</sub>

B<sub>ecause</sub> $A _ { \mathrm { a l l o w } }$ i<sub>s</sub> <sub>a</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>on</sub> <sub>a</sub>t <sub>every</sub> t<sub>ag</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>an</sub>d <sub>se</sub>l<sub>ec</sub>t<sub>or</sub> <sub>s</sub>t<sub>a</sub>t<sub>es</sub> <sub>o</sub>f<sub>er</sub> <sub>no</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>a</sub>t <sub>a</sub>ll<sub>,</sub> $\Pi _ { \mathrm { a l l o w } } ^ { \mathrm { M R } }$ <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> <sub>exac</sub>tl<sub>y</sub> <sub>one</sub> <sub>po</sub>li<sub>cy,</sub> th<sub>e</sub> lift $\widehat { \pi } _ { K }$ <sub>o</sub>f $\pi _ { K }$ . So there is no actual minimization left to do: writing N for this lifted RMDP,

$$
\operatorname * { i n f } _ { \pi \in \Pi _ { \mathrm { { a l l o w } } } ^ { \mathrm { M R } } } \mathrm { R r e g } ^ { N } ( \pi ) = \mathrm { R r e g } ^ { N } ( \widehat { \pi } _ { K } ) \overset { L e m m a ~ 3 0 } { = } \Lambda + \Delta _ { { \mathcal U } } ( \pi _ { C } , \pi _ { K } ) ,
$$

which is exactl<sub>y</sub> Theorem 14’s <sub>g</sub>a<sub>p</sub> <sub>p</sub>lus the lift’s constant Λ. The certification <sub>g</sub>a<sub>p</sub> transfers verbatim, with no se<sub>p</sub>arate <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on argumen</sub>t <sub>nee</sub>d<sub>e</sub>d<sub>.</sub>

Th<sub>e comp</sub>l<sub>e</sub>t<sub>e c</sub>h<sub>a</sub>i<sub>ne</sub>d <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> i<sub>s</sub>

$$
\begin{array} { r } { N _ { \varepsilon } = \mathrm { R e s t r i c t } \big ( \mathrm { L i f t } ( \mathrm { C m p } ( \varphi ) , \pi _ { C } ) , A _ { \mathrm { a l l o w } } , \frac { 1 } { 4 n } \big ) . } \end{array}
$$

F<sub>rom</sub> h<sub>ere on wr</sub>it<sub>e</sub> $\mathrm { R r e g } ( \pi )$ f<sub>or</sub> $\mathrm { R r e g } ^ { N _ { \varepsilon } } ( \pi )$ <sub>, ma</sub>t<sub>c</sub>hi<sub>ng</sub> Th<sub>eorem</sub> 6’<sub>s no</sub>t<sub>a</sub>ti<sub>on.</sub> L<sub>emma</sub> 32 <sub>w</sub>ith $\mathcal { P } = \Pi ^ { \mathrm { M R } }$ th<sub>en conver</sub>t<sub>s</sub> th<sub>e</sub> t<sub>wo</sub> <sub>cases a</sub>b<sub>ove</sub> i<sub>n</sub>t<sub>o</sub>

$$
\varphi \in \mathrm { U N S A T } \Longrightarrow \operatorname* { i n f } _ { \pi } \mathrm { R r e g } ( \pi ) \leq \operatorname* { i n f } _ { \pi \in \Pi _ { \mathrm { a l l o w } } ^ { \mathrm { M R } } } \mathrm { R r e g } ^ { N } ( \pi ) \leq \Lambda + 2 - \frac { 1 } { n } ,
$$

$$
\varphi \in \mathrm { S A T } \Longrightarrow \operatorname* { i n f } _ { \pi } \mathrm { R r e g } ( \pi ) \geq \operatorname* { i n f } _ { \pi \in \Pi _ { \mathrm { a l l o w } } ^ { \mathrm { M R } } } \mathrm { R r e g } ^ { N } ( \pi ) - \frac { 1 } { 4 n } = \Lambda + 2 - \frac { 1 } { 4 n } ,
$$

usin<sub>g</sub> the lemma’s u<sub>pp</sub>er inequalit<sub>y</sub> (no slack) in the first line and its lower inequalit<sub>y</sub> (losin<sub>g</sub> $\begin{array} { r } { \varepsilon = \frac { 1 } { 4 n } ) } \end{array}$ i<sub>n</sub> th<sub>e</sub> <sub>secon</sub>d<sub>.</sub> Si<sub>nce</sub>

$$
\Lambda + 2 - \frac { 1 } { n } < \Lambda + 2 - \frac { 1 } { 2 n } < \Lambda + 2 - \frac { 1 } { 4 n } ,
$$

th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld $\textstyle \Lambda + 2 - { \frac { 1 } { 2 n } }$ gives the coNP-hard part of Theorem 6.

## F.3 NP-Hardness

The NP reduction swaps the roles of the two policies. The fixed reference now looks for a falsified clause, while the synthesized <sub>po</sub>li<sub>cy names a va</sub>l<sub>ua</sub>ti<sub>on w</sub>h<sub>ose</sub> l<sub>oca</sub>l <sub>occurrences na</sub>t<sub>ure can au</sub>dit<sub>.</sub>

Definition 13 (Falsification and valuation verifier RMDP). For a 3-CNF formula $\varphi = { \textstyle \bigwedge } _ { i = 1 } ^ { m } C _ { i }$ over variables $x _ { 1 } , \ldots , x _ { n } ,$ , the verifier RMDP is

$$
N _ { \varphi } = ( S _ { \varphi } , A _ { \varphi } , { \mathcal { U } } _ { \varphi } , R _ { \varphi } , s _ { \iota } , \gamma _ { 0 } ) ,
$$

with the following components.

$S _ { \varphi } = \{ s _ { \iota } \} \cup S _ { \mathrm { s c a n } } \cup S _ { \mathrm { a u d } } \cup S _ { \mathrm { s e l } } \cup \{ \perp _ { \mathrm { r } } \}$ , consisting of four verifier groups and a ruinous sink:

$S _ { \mathrm { s c a n } } ,$ visited only by the falsification (scanner) policy: $S _ { F } = \{ f _ { i , j } : i \in [ m ] , j \in \{ 1 , 2 , 3 \} \}$ , for clause $C _ { i }$ and literal $j ,$ together with acc , rej , and scanner-side padding states,fixed once the transitions below are defined.

$S _ { \mathrm { a u d } }$ , visited only by the valuation (audit) policy: {v : x occurs in $\varphi \}$ and, for every x, $S _ { x , T } = \{ s _ { x , T , \ell } \} _ { \ell = 1 } ^ { | O _ { x } | } , S _ { x , F } =$ $\{ s _ { x , F , \ell } \} _ { \ell = 1 } ^ { | O _ { x } | }$ , where x is assigned b and occurrence $\ell o f O _ { x }$ is audited (occurrences of x, fixed order), together with ac $\dot { } _ { ; K }$ $\mathrm { r e j } _ { K } ,$ , and audit-side padding states

$\ - \ S _ { \mathrm { s e l } } = \{ q _ { o , t } , q _ { o , t } ^ { 0 } , q _ { o , t } ^ { 1 } : \epsilon$ an occurrence, $t \in \{ T , F \} \}$ , the local-bit selectors of Definition 3, with T, F in place of 1, 0. These are the only states either policy’s transitions can lead intofrom the other’s territory.

Let val(o) = Tfor a positive occurrence and val(o) = Ffor a negative occurrence, and write ¬ val(o)for the other Boolean value.

• s has actions • s<sub>ι</sub> has actions $a _ { F } ^ { \mathrm { i n i t } }$ , leading into , leading into $S _ { \mathrm { s c a n } }$ at at $f _ { 1 , 1 }$ , and , and $a _ { K } ^ { \mathrm { i n i t } }$ , leading into , leading into $S _ { \mathrm { a u d } }$ at at $v _ { x _ { 1 } }$ .

Within $S _ { \mathrm { s c a n } } { \cdot }$ every $q _ { o , t } \in S _ { \mathrm { s e l } }$ has a single action, whose outcome $q _ { o , t } ^ { 0 }$ or $q _ { o , t } ^ { 1 }$ is governed by $\mathcal { U } _ { \varphi }$ below. Each selector outcome ofers a scanner action $a _ { F }$ and an audit action $a _ { K }$ with the role-specific continuations described next. For $o = ( i , j )$ action a at $f _ { i , j }$ uses the pair test ofDefinition $^ { 3 , }$ and its locallyfalse outcome advances to $f _ { i , j + 1 }$ or acc when $j = 3 . A l l$ other outcomes abandon the current clause, continuing to $f _ { i + 1 , 1 } ,$ or to rej $_ { F } \dot { \imath } f \dot { \imath } = m$   
Within $S _ { \mathrm { a u d } } \colon e \nu e r y v _ { x }$ ofers actions T, F leading to $s _ { x , T , 1 } , s _ { x , F , 1 }$ . The controls $s _ { x , b , \ell }$ implement the audit chain ofDefinition 3; after its last occurrence, the chain continues to the next variable or to acc .

Fix H at least as long as the longest path in either graph just described. The scanner-side and audit-side padding states each consist of fresh states inserted along every shorter path in their own graph so it also reaches its terminal after exactly H transitions, each with a single reward-free action continuing toward that terminal.

$\mathcal { U } _ { \varphi }$ is the product of the local-bit segments in Equation (2), with T, F in place of 1, 0. All remaining described choices are singletons.

• acc<sub>F</sub> is the terminal with payof $\gamma _ { 0 } ^ { - H }$ and acc<sub>K</sub> the terminal with payof $- \gamma _ { 0 } ^ { - H }$ , while every other described reward, including at rej and $\mathrm { r e j } _ { K } ,$ , is zero.

• All remaining choices use ruinous-sink completion.

• The initial state is $\boldsymbol { s } _ { \iota }$ and the discount is the appendix-wide $\gamma _ { 0 } .$

Variables with no occurrence are removed.

Si<sub>nce</sub> $\mathcal { U } _ { \varphi }$ i<sub>s</sub> <sub>a</sub> <sub>pro</sub>d<sub>uc</sub>t <sub>o</sub>f t<sub>wo-</sub>Di<sub>rac</sub> <sub>segmen</sub>t<sub>s,</sub> it<sub>s</sub> <sub>ver</sub>ti<sub>ces</sub> <sub>are</sub> <sub>exac</sub>tl<sub>y</sub> th<sub>e</sub> B<sub>oo</sub>l<sub>ean</sub> <sub>ass</sub>i<sub>gnmen</sub>t<sub>s</sub> t<sub>o</sub> th<sub>e</sub> l<sub>oca</sub>l bit<sub>s</sub> $\{ b _ { o , t } \}$ <sub>.</sub> Fi<sub>gu</sub>r<sub>e</sub> 9 <sub>an</sub>d <sub>examp</sub>l<sub>e</sub> 6 ill<sub>us</sub>t<sub>ra</sub>t<sub>e</sub> th<sub>e ver</sub>ifi<sub>er</sub> RMDP <sub>an</sub>d th<sub>e</sub> t<sub>wo po</sub>li<sub>cy pa</sub>th<sub>s</sub> th<sub>roug</sub>h it<sub>s s</sub>h<sub>are</sub>d <sub>se</sub>l<sub>ec</sub>t<sub>ors.</sub>

Definition 14 (Falsification and valuation policies). For the verifier RMDP of Definition 13, define

$$
\begin{array} { r l r } & { \pi _ { F } ( s _ { \iota } ) = a _ { F } ^ { \mathrm { i n i t } } , } & { \pi _ { F } ( s ) = a _ { F } ( s ) \qquad ( s \in S _ { F } ) , } \\ & { \pi _ { F } ( q _ { o , t } ^ { b } ) = a _ { F } ( q _ { o , t } ^ { b } ) , } \\ & { K _ { \sigma } ( s _ { \iota } ) = a _ { K } ^ { \mathrm { i n i t } } , } & { K _ { \sigma } ( v _ { x } ) ( b ) = \sigma _ { x } ( b ) , } \\ & { K _ { \sigma } ( s ) = a _ { x , b } ( s ) \qquad } & { ( s \in S _ { x , b } ) , } \\ & { K _ { \sigma } ( q _ { o , t } ^ { b } ) = a _ { K } ( q _ { o , t } ^ { b } ) . } \end{array}
$$

On decision states outside their reachable verifier graphs,fix default actions whose otherwise undescribed choices use ruinous sink completion. For a valuation α, the deterministic policy $K _ { \alpha }$ is obtainedfrom $\sigma _ { x } ( T ) = \alpha ( x )$ and $\sigma _ { x } ( F ) = 1 - \alpha ( x )$ . At a Boolean realization, a vertex of $\mathrm { \Delta } \mathcal { U } _ { \varphi } ,$ , every selector sends its unique action to a fixed outcome, so a deterministic policy follows a single path. We say the policy accepts that realization when this path reaches its accepting terminal, acc<sub>F</sub> for π<sub>F</sub> or acc<sub>K</sub> for $K _ { \alpha } .$ . Thus π accepts exactly when some clause is locallyfalse, while $K _ { \sigma }$ audits the value selected at each $v _ { x } .$

Th<sub>e</sub> <sub>common</sub> d<sub>ep</sub>th <sub>an</sub>d t<sub>erm</sub>i<sub>na</sub>l <sub>payo</sub>f<sub>s</sub> <sub>sa</sub>ti<sub>s</sub>f<sub>y</sub> L<sub>emma</sub> 10<sub>,</sub> <sub>so</sub> th<sub>e</sub> <sub>compar</sub>i<sub>son</sub> <sub>va</sub>l<sub>ue</sub> i<sub>s</sub> th<sub>e</sub> <sub>sum</sub> <sub>o</sub>f th<sub>e</sub> t<sub>wo</sub> <sub>accep</sub>t<sub>ance</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es.</sub>

Example 6. For $\varphi = ( x _ { 1 } \vee \neg x _ { 2 } \vee x _ { 3 } ) \wedge ( \neg x _ { 1 } \vee x _ { 2 } \vee x _ { 3 } )$ , the valuation $( x _ { 1 } , x _ { 2 } , x _ { 3 } ) = ( 1 , 1 , 0 )$ satisfies $\varphi ,$ so an accepting valuation audit precludes falsification. By contrast, (0, 1, 0) falsifies the first clause, and its canonical local pairs make both verifiers accept.

Lemma 33 (Verifier separation). Consider the verifier RMDP andpolicies ofDefinitions 13 and 14. At every Boolean realization at which $\pi _ { F }$ and $K _ { \alpha }$ both accept, α falsifies a clause of φ. Conversely, if α falsifies a clause of φ, then there exists a Boolean realization at which both accept. In particular, ifα satisfies φ, the two acceptance events are disjoint at every realization.

Proof. Fix a Boolean realization and suppose $K _ { \alpha }$ accepts at it. Then every variable set true by α has $b _ { o , T } = 1$ <sub>a</sub>t <sub>eac</sub>h <sub>occurrence,</sub> <sub>an</sub>d <sub>every var</sub>i<sub>a</sub>bl<sub>e se</sub>t f<sub>a</sub>l<sub>se</sub> h<sub>as</sub> $b _ { o , F } = 1$ . In every clause satisfied by α, a true positive literal therefore cannot have the false air (0, 1), and a true ne ative literal cannot have (1, 0). Hence no clause satisfied b α is locall false in all three ositions. This also covers malformed pairs (0, 0) and (1, 1). Since π accepts only by finding a clause that is locally false in all three positions, its acceptance at the same realization exhibits a clause that α falsifies. The same argument read contrapositively gives the disjointness claim for satisfying α.

For the converse, let α falsify a clause and let nature use the canonical pair (1, 0) for true variables and $( 0 , 1 )$ f<sub>or</sub> f<sub>a</sub>l<sub>se</sub> <sub>var</sub>i<sub>a</sub>bl<sub>es.</sub> At th<sub>a</sub>t <sub>rea</sub>li<sub>za</sub>ti<sub>on</sub> $K _ { \alpha }$ <sub>accep</sub>t<sub>s</sub> <sub>a</sub>ll <sub>o</sub>f it<sub>s</sub> <sub>au</sub>dit<sub>s</sub> <sub>an</sub>d <sub>every</sub> lit<sub>era</sub>l <sub>o</sub>f th<sub>e</sub> f<sub>a</sub>l<sub>s</sub>ifi<sub>e</sub>d <sub>c</sub>l<sub>ause</sub> i<sub>s</sub> l<sub>oca</sub>ll<sub>y</sub> f<sub>a</sub>l<sub>se,</sub> <sub>so</sub> $\pi _ { F }$ <sup>acce</sup>p<sup>ts</sup> <sup>as</sup> <sub>we</sub>ll<sub>.</sub> O<sub>n</sub>l<sub>y ex</sub>i<sub>s</sub>t<sub>ence</sub> i<sub>s c</sub>l<sub>a</sub>i<sub>me</sub>d h<sub>ere, s</sub>i<sub>nce ano</sub>th<sub>er rea</sub>li<sub>za</sub>ti<sub>on may ma</sub>k<sub>e some au</sub>dit f<sub>a</sub>il<sub>.</sub> □

![](images/21bd2561429cbe326fef7972825ee8986b2442bd129ddda70759d02cb25eb2c5.jpg)  
Fi<sub>gure</sub> 9<sub>:</sub> Th<sub>e ver</sub>ifi<sub>er</sub> RMDP <sub>o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 13 f<sub>or</sub> $\varphi = ( x _ { 1 } \vee \neg x _ { 2 } \vee x _ { 3 } ) \wedge ( \neg x _ { 1 } \vee x _ { 2 } \vee x _ { 3 } )$ <sub>, s</sub>h<sub>ow</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>our s</sub>t<sub>a</sub>t<sub>e groups.</sub> F<sub>or eac</sub>h lit<sub>era</sub>l <sub>o</sub>f $C _ { 1 }$ , the scanner first reads the value that would make it true; only outcome 0 followed by outcome 1 at the opposite-value selector confirms that it is locally false. The displayed 0, 1 paths advance through the three literals and then <sub>accep</sub>t<sub>; every o</sub>th<sub>er ou</sub>t<sub>come a</sub>b<sub>an</sub>d<sub>ons</sub> $C _ { 1 }$ <sub>.</sub> Th<sub>e au</sub>dit<sub>or po</sub>li<sub>cy</sub> $\bar { K } _ { \alpha }$ <sub>wa</sub>lk<sub>s</sub> $v _ { x _ { 1 } } , v _ { x _ { 2 } } , v _ { x _ { 3 } }$ <sub>an</sub>d<sub>,</sub> h<sub>av</sub>i<sub>ng</sub> <sub>ass</sub>i<sub>gne</sub>d <sub>a</sub> <sub>va</sub>l<sub>ue</sub> t<sub>o</sub> <sub>eac</sub>h <sub>var</sub>i<sub>a</sub>bl<sub>e, c</sub>h<sub>ec</sub>k<sub>s</sub> it<sub>s occurrences.</sub> D<sub>o</sub>tt<sub>e</sub>d <sub>e</sub>d<sub>ges summar</sub>i<sub>ze au</sub>dit <sub>rea</sub>d<sub>s o</sub>f th<sub>e s</sub>h<sub>are</sub>d <sub>se</sub>l<sub>ec</sub>t<sub>ors:</sub> th<sub>e scanner an</sub>d th<sub>e au</sub>dit<sub>or o</sub>f $x _ { 1 } = F$ <sub>rea</sub>d th<sub>e same</sub> $q _ { ( 1 , 1 ) , F } ,$ coupling their acceptances. The remaining occurrences, both clauses, and the depth-H padding f<sub>o</sub>ll<sub>ow</sub> th<sub>e same pa</sub>tt<sub>ern.</sub>

F<sub>or a ran</sub>d<sub>om</sub>i<sub>ze</sub>d <sub>va</sub>l<sub>ua</sub>ti<sub>on po</sub>li<sub>cy, pu</sub>t

$$
G _ { \varphi } ( \sigma ) = \operatorname* { s u p } _ { u } \bigl ( V _ { u } ^ { \pi _ { F } } ( s _ { \iota } ) - V _ { u } ^ { K _ { \sigma } } ( s _ { \iota } ) \bigr ) .
$$

<sup>A</sup>t a vertex, acce<sub>p</sub>tance <sup>b</sup><sub>y</sub> $\pi _ { F }$ <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> <sub>one</sub> <sub>an</sub>d <sub>accep</sub>t<sub>ance</sub> b<sub>y</sub> $K _ { \sigma }$ <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>es</sub> <sub>ano</sub>th<sub>er</sub> <sub>one</sub> t<sub>o</sub> thi<sub>s</sub> dif<sub>erence.</sub>

Lemma 34 (Verifier gap). For the verifier RMDP andpolicies ofDefinitions 13 and 14, thefollowing gap holds. $I f \varphi$ is satisfiable, some deterministic valuation policy has $G _ { \varphi } \leq 1 . \ : I f \varphi$ is unsatisfiable, every randomized valuation policy has $G _ { \varphi } \geq 1 + 2 ^ { - n }$

Proof. For a satisfying α, Lemma 33 bounds the diference by one at every vertex. Both policy graphs are acyclic and visit each l<sub>oca</sub>l <sub>se</sub>l<sub>ec</sub>t<sub>or a</sub>t <sub>mos</sub>t <sub>once, so</sub> L<sub>emma</sub> 17 <sub>ex</sub>t<sub>en</sub>d<sub>s</sub> th<sub>e</sub> b<sub>oun</sub>d t<sub>o</sub> th<sub>e en</sub>ti<sub>re pro</sub>d<sub>uc</sub>t <sub>po</sub>l<sub>y</sub>t<sub>ope.</sub>

Now fix σ for an unsatisfiable formula. At each of the n variable states, choose a most probable action. These actions form a valuation α. Since every such state is visited exactly once, $K _ { \sigma }$ f<sub>o</sub>ll<sub>ows</sub> th<sub>a</sub>t <sub>comp</sub>l<sub>e</sub>t<sub>e va</sub>l<sub>ua</sub>ti<sub>on w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y a</sub>t l<sub>eas</sub>t $\delta = 2 ^ { - n }$ U<sub>se</sub> th<sub>e canon</sub>i<sub>ca</sub>l <sub>rea</sub>li<sub>za</sub>ti<sub>on</sub> f<sub>or</sub> $\alpha .$ <sub>.</sub> Th<sub>e</sub> f<sub>ormu</sub>l<sub>a</sub> h<sub>as a c</sub>l<sub>ause</sub> f<sub>a</sub>l<sub>s</sub>ifi<sub>e</sub>d b<sub>y</sub> $\alpha , { \bf s o } \ \pi _ { F }$ <sub>accep</sub>t<sub>s w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y one an</sub>d $K _ { \sigma }$ <sub>w</sub>ith <sub>p</sub>robabilit<sub>y</sub> at least δ. Thus $G _ { \varphi } ( \sigma ) \geq 1 + \delta$ □

W<sub>e now</sub> t<sub>rans</sub>f<sub>er</sub> th<sub>e ver</sub>ifi<sub>er gap</sub> t<sub>o m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t <sub>w</sub>hil<sub>e preven</sub>ti<sub>ng</sub> th<sub>e syn</sub>th<sub>es</sub>i<sub>ze</sub>d <sub>po</sub>li<sub>cy</sub> f<sub>rom</sub> l<sub>eav</sub>i<sub>ng</sub> th<sub>e va</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>g</sub>r<sub>ap</sub>h<sub>.</sub> A<sub>pp</sub>l<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> 30 <sub>w</sub>ith $\pi _ { F }$ <sub>as re</sub>f<sub>erence.</sub> At <sub>var</sub>i<sub>a</sub>bl<sub>e s</sub>t<sub>a</sub>t<sub>es a</sub>ll<sub>ow</sub> $\{ T , { \bar { F } } \}$ <sub>, a</sub>t <sub>ver</sub>ifi<sub>er s</sub>t<sub>a</sub>t<sub>es a</sub>ll<sub>ow</sub> th<sub>e un</sub>i<sub>que ac</sub>ti<sub>on o</sub>f $K _ { \sigma } .$ <sub>an</sub>d <sub>a</sub>t <sub>source s</sub>t<sub>a</sub>t<sub>es use</sub>d <sub>on</sub>l<sub>y</sub> b<sub>y</sub> $\pi _ { F }$ <sub>a</sub>ll<sub>ow</sub> th<sub>e</sub>i<sub>r un</sub>i<sub>que ac</sub>ti<sub>on.</sub> Th<sub>e</sub> lift<sub>e</sub>d <sub>a</sub>ll<sub>owe</sub>d <sub>po</sub>li<sub>c</sub>i<sub>es are prec</sub>i<sub>se</sub>l<sub>y</sub> th<sub>e va</sub>l<sub>ua</sub>ti<sub>on po</sub>li<sub>c</sub>i<sub>es</sub> <sub>an</sub>d

$$
\mathrm { R r e g } ^ { N } ( K _ { \sigma } ) = \Lambda + G _ { \varphi } ( \sigma ) .
$$

Equivalently, the target is Restrict $( \mathrm { L i f t } ( N _ { \varphi } , \pi _ { F } ) , A _ { \mathrm { a l l o w } } , 2 ^ { - n } / 4 )$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub> $N _ { \varphi }$ i<sub>s</sub> th<sub>e ver</sub>ifi<sub>er</sub> RMDP <sub>o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 13<sub>.</sub> Th<sub>e</sub> di<sub>sa</sub>ll<sub>owe</sub>d b<sub>onus c</sub>h<sub>o</sub>i<sub>ces are s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>ons, so</sub> L<sub>emma</sub> 32 <sub>app</sub>li<sub>es w</sub>ith $\mathcal { P } = \Pi ^ { \mathrm { M R } }$ <sub>.</sub> Th<sub>e</sub> <sub>gap</sub> <sub>o</sub>f L<sub>e</sub>mm<sub>a</sub> 34 b<sub>eco</sub>m<sub>es</sub>

$$
\varphi \in \mathrm { S A T } \Longrightarrow \operatorname* { i n f } _ { \pi } \mathrm { R r e g } ( \pi ) \leq \Lambda + 1 , \qquad \varphi \in \mathrm { U N S A T } \Longrightarrow \operatorname* { i n f } _ { \pi } \mathrm { R r e g } ( \pi ) \geq \Lambda + 1 + \frac { 3 } { 4 } 2 ^ { - n } .
$$

Th<sub>res</sub>h<sub>o</sub>ld $\Lambda + 1 + 2 ^ { - n - 1 }$ proves NP-hardness.

Proof. The construction in the coNP subsection is a polynomial reduction from UNSAT, and the construction in the NP <sub>su</sub>b<sub>sec</sub>ti<sub>on</sub> i<sub>s a o</sub>l <sub>nom</sub>i<sub>a</sub>l <sub>re</sub>d<sub>uc</sub>ti<sub>on</sub> f<sub>rom</sub> SAT<sub>.</sub> Th<sub>e</sub>i<sub>r s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ons</sub> f<sub>o</sub>ll<sub>ow</sub> f<sub>rom</sub> th<sub>e source ver</sub>ifi<sub>ers,</sub> L<sub>emma</sub> 30<sub>, an</sub>d L<sub>emma</sub> 32<sub>.</sub> Th<sub>e ra</sub>ti<sub>ona</sub>l <sub>gaps a</sub>b<sub>ove</sub> l<sub>eave a va</sub>lid <sub>separa</sub>ti<sub>ng</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>a</sub>ft<sub>er</sub> th<sub>e res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>on</sub> l<sub>oss.</sub> Mi<sub>n</sub>i<sub>ma</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> both NP-hard and coNP-hard under the restrictions stated in the theorem. □

## G Deterministic Minimal-Regret Complexity

W<sub>e</sub> fi<sub>rs</sub>t <sub>es</sub>t<sub>a</sub>bli<sub>s</sub>h <sub>mem</sub>b<sub>ers</sub>hi<sub>p an</sub>d th<sub>en g</sub>i<sub>ve</sub> th<sub>e ma</sub>t<sub>c</sub>hi<sub>ng</sub> h<sub>ar</sub>d<sub>ness cons</sub>t<sub>ruc</sub>ti<sub>on.</sub>

## G.1 Acyclic Deterministic Membership

Lemma 35. Minimal robust regret over memoryless deterministic policies is in $\Sigma _ { 2 } ^ { p }$ on acyclic $( s , a )$ -rectangular polytopic RMDPs.

Proof. Fix an acyclic RMDP. For any uncertainty realization, ordinary finite discounted-MDP optimality gives

$$
V _ { u } ^ { * } ( s ) = \operatorname* { m a x } _ { \pi ^ { \prime } \in \Pi ^ { \mathrm { M D } } } V _ { u } ^ { \pi ^ { \prime } } ( s ) .
$$

I<sub>n</sub>d<sub>ee</sub>d<sub>, process</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on s</sub>t<sub>a</sub>t<sub>es</sub> i<sub>n reverse</sub> t<sub>opo</sub>l<sub>og</sub>i<sub>ca</sub>l <sub>or</sub>d<sub>er an</sub>d <sub>se</sub>l<sub>ec</sub>t <sub>a max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng ac</sub>ti<sub>on.</sub> N<sub>o</sub> hi<sub>s</sub>t<sub>ory-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>or ran</sub>d<sub>om</sub>i<sub>ze</sub>d <sub>po</sub>li<sub>cy</sub> <sub>can</sub> i<sub>mprove</sub> th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> b<sub>ac</sub>k<sub>war</sub>d<sub>-</sub>i<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> <sub>va</sub>l<sub>ue.</sub>

For every deterministic candidate π,

$$
\begin{array} { r } {  { \mathrm { R r e g } } ( \pi ) = \underset { u } { \operatorname* { s u p } } \ \underset { \pi ^ { \prime } \in \Pi ^ { \mathrm { M D } } } { \operatorname* { m a x } } \left( V _ { u } ^ { \pi ^ { \prime } } ( s _ { \iota } ) - V _ { u } ^ { \pi } ( s _ { \iota } ) \right) } \\ { = \underset { \pi ^ { \prime } \in \Pi ^ { \mathrm { M D } } } { \operatorname* { m a x } } \ \underset { u } { \operatorname* { s u p } } \left( V _ { u } ^ { \pi ^ { \prime } } ( s _ { \iota } ) - V _ { u } ^ { \pi } ( s _ { \iota } ) \right) . } \end{array}
$$

A<sub>cyc</sub>li<sub>c</sub>it<sub>y ma</sub>k<sub>es every po</sub>li<sub>cy pa</sub>i<sub>r cyc</sub>l<sub>e-</sub>f<sub>ree on</sub> it<sub>s s</sub>h<sub>are</sub>d <sub>c</sub>h<sub>o</sub>i<sub>ces,</sub> th<sub>e on</sub>l<sub>y se</sub>lf<sub>-</sub>l<sub>oops</sub> b<sub>e</sub>i<sub>ng</sub> th<sub>ose a</sub>t <sub>a</sub>b<sub>sor</sub>bi<sub>ng</sub> fi<sub>na</sub>l<sub>s, w</sub>hi<sub>c</sub>h D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 5 <sub>om</sub>it<sub>s.</sub> B<sub>y</sub> L<sub>emma</sub> 17<sub>,</sub> th<sub>e rema</sub>i<sub>n</sub>i<sub>ng supremum</sub> i<sub>s a</sub>tt<sub>a</sub>i<sub>ne</sub>d <sub>a</sub>t <sub>one ver</sub>t<sub>ex o</sub>f <sub>every c</sub>h<sub>o</sub>i<sub>ce uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y se</sub>t<sub>.</sub> Th<sub>ere</sub>f<sub>ore</sub>

$$
{ \mathrm { R r e g } } ( \pi ) \leq t \iff \forall \pi ^ { \prime } \in \Pi ^ { \mathrm { M D } } \forall v \in \prod _ { s , a } \mathrm { v e r t } ( \mathcal { U } _ { s , a } ) : V _ { v } ^ { \pi ^ { \prime } } ( s _ { \iota } ) - V _ { v } ^ { \pi } ( s _ { \iota } ) \leq t .
$$

The existential certificate is the action table of π. The universal certificate contains the action table of $\pi ^ { \prime }$ <sub>an</sub>d<sub>,</sub> f<sub>or every c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y se</sub>t<sub>, a ver</sub>t<sub>ex represen</sub>t<sub>e</sub>d b<sub>y</sub> li<sub>near</sub>l<sub>y</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t ti<sub>g</sub>ht i<sub>npu</sub>t f<sub>ace</sub>t<sub>s p</sub>l<sub>us</sub> it<sub>s ra</sub>ti<sub>ona</sub>l <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub> C<sub>ramer</sub>’<sub>s ru</sub>l<sub>e g</sub>i<sub>ves</sub> <sub>o</sub>l <sub>nom</sub>i<sub>a</sub>l bit l<sub>en</sub> th<sub>.</sub> Th<sub>e re</sub>di<sub>ca</sub>t<sub>e ver</sub>ifi<sub>es</sub> th<sub>e</sub> f<sub>ace</sub>t<sub>s an</sub>d <sub>eva</sub>l<sub>ua</sub>t<sub>es</sub> b<sub>o</sub>th <sub>o</sub>li<sub>c</sub>i<sub>es</sub> b <sub>exac</sub>t b<sub>ac</sub>k<sub>war</sub>d i<sub>n</sub>d<sub>uc</sub>ti<sub>on</sub> i<sub>n o</sub>l <sub>nom</sub>i<sub>a</sub> ti<sub>me.</sub> Th<sub>us</sub> th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on pro</sub>bl<sub>em</sub> h<sub>as an ex</sub>i<sub>s</sub>t<sub>en</sub>ti<sub>a</sub>l <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>cer</sub>tifi<sub>ca</sub>t<sub>e</sub> f<sub>o</sub>ll<sub>owe</sub>d b<sub>y a un</sub>i<sub>versa</sub>l <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>cer</sub>tifi<sub>ca</sub>t<sub>e w</sub>ith <sub>a</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>-</sub>ti<sub>me pre</sub>di<sub>ca</sub>t<sub>e, p</sub>l<sub>ac</sub>i<sub>ng</sub> it i<sub>n</sub> $\Sigma _ { 2 } ^ { p }$ □

## G.2 Σ<sup>p</sup>-Hardness over Deterministic Policies

R<sub>e</sub>d<sub>uce</sub> f<sub>rom</sub> th<sub>e</sub> <sub>canon</sub>i<sub>ca</sub>l $\Sigma _ { 2 } ^ { p }$ -com<sub>p</sub>lete <sub>p</sub>roblem (Pa<sub>p</sub>adimitriou 1994) of decidin<sub>g</sub> whether

$$
\exists x \in \{ 0 , 1 \} ^ { n } \forall y \in \{ 0 , 1 \} ^ { m } : \quad \psi ( x , y ) ,
$$

where ψ is a 3-DNF.

Definition 15 (Universal falsification reference). Write $\textstyle \psi = \bigvee _ { i = 1 } ^ { r } D _ { i }$ , where every $D _ { i }$ is a conjunction ofthree literals. Decide constant cases directly, remove universal variables with no occurrence, and order every variable’s occurrences by term and then position. Ifnecessary, add one unused existential variable so that $n \geq 1$ . For an occurrence $^ { O , }$ let $\mathrm { v a l } ( o ) \in \{ T , F \}$ be the value making its literal true. The universal falsification RMDP is

$$
N _ { \psi } = ( S _ { \psi } , A _ { \psi } , \mathcal { U } _ { \psi } , R _ { \psi } , s _ { \iota } , \gamma _ { 0 } ) ,
$$

with the following components.

$S _ { \psi }$ contains the initial state and one candidate branch-entry state per variable,

$$
\{ r _ { 1 } ^ { K } , \ldots , r _ { n } ^ { K } \} \cup \{ r _ { 1 } ^ { Y } , \ldots , r _ { m } ^ { Y } \} .
$$

It contains a global selector $q _ { Y _ { j } }$ with outcomes $q _ { Y _ { j } } ^ { T } , q _ { Y _ { j } } ^ { F }$ for every universal variable, and the shared local-bit selector of Definition 3for every occurrence o and $c \in \{ T , F \}$

• The reference-side controls are the term-scan states $f _ { i , h } .$ The candidate-side controls are the decision states ${ \boldsymbol { v } } _ { x _ { i } }$ , the existential audit states $k _ { i , c , \ell } ^ { x } ,$ , and the universal audit states $k _ { j , c , \ell } ^ { y } .$ . There are also accepting, rejecting, padding, and zero-reward terminal states, together with the ruinous sink supplied below.

• At $s _ { \iota } ,$ , action $a _ { F } ^ { \mathrm { i n i t } }$ leads deterministically to the scanner entry $f _ { 1 , 1 }$ , and action $a _ { K } ^ { \mathrm { i n i t } }$ has the certain uniform distribution over the n + m branch-entry states.

• State $r _ { j } ^ { Y }$ leads to $q _ { Y _ { j } }$ . Outcome $q _ { Y _ { i } } ^ { c }$ leads to $k _ { j , c , 1 } ^ { y }$ , which implements the audit chain of Definition 3 for the occurrences of $y _ { j }$ in term-major order. A universal branch therefore ofers the candidate no choice, the committed value being nature’s.

• State $r _ { i } ^ { K }$ leads to $v _ { x _ { i } } . A t ~ v _ { x _ { i } }$ , actions T and F enter the corresponding chain $k _ { i , c , 1 } ^ { x }$ . That control implements the same audit chainfor the occurrences ofx<sub>i</sub> in term-major order. $H x _ { i }$ has no occurrence, both actions at $v _ { x } ,$ accept immediately.

• At occurrence $o = \left( i , h \right)$ , the term scanner uses the pair test ofDefinition 3; its locallyfalse outcome advances to $f _ { i + 1 , 1 } ,$ , or accepts $i f i = r .$ . Every other outcome advances to $f _ { i , h + 1 }$ , or rejects $i f h = 3$

• Every selector has one described action. At a shared local outcome, one described action takes the scanner continuation and another takes the audit continuation. Since the two auditfamilies lie behind $a _ { K } ^ { \mathrm { i n i t } }$ and the scanner behind $a _ { F } ^ { \mathrm { i n i t } }$ , a policy that plays one of those initial actions needs only one of the two continuations, and each continuation is determined by the outcome state itself. All paths are padded so that acceptance or rejection occurs after a common depth H.

$\mathcal { U } _ { \psi }$ is the product of the local-bit segments in Equation (2) and analogous segments for the global selectors $q _ { Y _ { j } }$ . The initial splitter o $f a _ { K } ^ { \mathrm { i n i t } }$ is certain and uniform, and all remaining described choices are singletons.

• Before ruinous-sink completion, $R _ { \psi }$ is zero except at padded acceptance terminals. Every reference-side accepting terminal has $p a y o f f \gamma _ { 0 } ^ { - H }$ , every candidate-side accepting terminal has payof $- \gamma _ { 0 } ^ { - H }$ , and every rejecting terminal has payofzero.

• All remaining choices use ruinous-sink completion.

• The initial state is $\boldsymbol { s } _ { \iota }$ and the discount is the appendix-wide $\gamma _ { 0 } .$

Definition 16 (Universal falsification policies). For the RMDP ofDefinition 15, thefixed deterministic reference $\pi _ { F }$ takes $a _ { F } ^ { \mathrm { i n i t } }$ and then uses the pair-test scanner action throughout. The synthesized deterministic policy $K _ { \alpha }$ takes $a _ { K } ^ { \mathrm { i n i t } } .$ , chooses $\alpha ( x _ { i } )$ at every $v _ { x _ { i } , }$ , and follows the audit action for the committed value at every audit state, that value being $\alpha ( x _ { i } )$ on an existential branch and nature’s selection at $q _ { Y _ { j } }$ on a universal one. Each policy therefore carries one role: the reference scans and the candidate audits. At states outside a policy’s reachable graph, fix default actions whose undescribed choices use ruinous-sink completion.

B<sub>y</sub> L<sub>emma</sub> 10<sub>,</sub> th<sub>e va</sub>l<sub>ue</sub> dif<sub>erence</sub> i<sub>s</sub> th<sub>e sum o</sub>f th<sub>e re</sub>f<sub>erence an</sub>d <sub>can</sub>did<sub>a</sub>t<sub>e accep</sub>t<sub>ance pro</sub>b<sub>a</sub>biliti<sub>es.</sub>

Lemma (Unambiguous verifier policies). The policies in Definition 16 are stationary and deterministic, and no run of either policy visits a local selector twice.

Proof. The candidate’s certain uniform splitter chooses one branch. An audit commits to one value before visiting one variable’s occurrences in term-major order, while the term scanner visits the two selectors of each occurrence in their fixed pair-test order. E<sub>very occurrence</sub> b<sub>e</sub>l<sub>ongs</sub> t<sub>o one var</sub>i<sub>a</sub>bl<sub>e, so</sub> th<sub>e au</sub>dit <sub>reac</sub>hi<sub>ng a g</sub>i<sub>ven se</sub>l<sub>ec</sub>t<sub>or</sub> i<sub>s un</sub>i<sub>que, an</sub>d th<sub>e scanner reac</sub>h<sub>es eac</sub>h <sub>se</sub>l<sub>ec</sub>t<sub>or</sub> <sub>once.</sub> Si<sub>nce every au</sub>dit b<sub>e</sub>l<sub>ongs</sub> t<sub>o</sub> $K _ { \alpha }$ an<sup>d</sup> ever<sub>y</sub> scan to $\pi _ { F }$ <sub>, no s</sub>t<sub>a</sub>t<sub>e as</sub>k<sub>s e</sub>ith<sub>er po</sub>li<sub>cy</sub> t<sub>o</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h th<sub>e</sub> b<sub>ranc</sub>h b<sub>y w</sub>hi<sub>c</sub>h it <sub>was reac</sub>h<sub>e</sub>d<sub>, an</sub>d <sub>one ac</sub>ti<sub>on per s</sub>t<sub>a</sub>t<sub>e su</sub>fi<sub>ces.</sub> □

I<sub>gnor</sub>i<sub>ng</sub> fi<sub>na</sub>l <sub>se</sub>lf<sub>-</sub>l<sub>oops,</sub> th<sub>e</sub> f<sub>u</sub>ll RMDP i<sub>s</sub> <sub>acyc</sub>li<sub>c.</sub> A t<sub>opo</sub>l<sub>og</sub>i<sub>ca</sub>l <sub>or</sub>d<sub>er</sub> <sub>p</sub>l<sub>aces</sub> $\boldsymbol { s } _ { \iota }$ fi<sub>rs</sub>t<sub>,</sub> th<sub>en</sub> th<sub>e</sub> b<sub>ranc</sub>h<sub>-en</sub>t<sub>ry</sub> <sub>s</sub>t<sub>a</sub>t<sub>es,</sub> th<sub>e</sub> <sub>g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>se</sub>l<sub>ec</sub>t<sub>ors</sub> $Y _ { 1 } , \dots , { \bar { Y } } _ { m }$ <sub>,</sub> th<sub>e</sub> d<sub>ec</sub>i<sub>s</sub>i<sub>on s</sub>t<sub>a</sub>t<sub>es</sub> $v _ { x _ { 1 } } , \ldots , v _ { x _ { n } }$ , the occurrence selectors in term-major order with their associated <sub>con</sub>t<sub>ro</sub>l<sub>s an</sub>d <sub>ou</sub>t<sub>comes, an</sub>d fi<sub>na</sub>ll<sub>y</sub> th<sub>e pa</sub>ddi<sub>ng,</sub> t<sub>erm</sub>i<sub>na</sub>l<sub>, an</sub>d <sub>ru</sub>i<sub>nous-s</sub>i<sub>n</sub>k <sub>s</sub>t<sub>a</sub>t<sub>es.</sub>

For a valuation α of $x ,$ <sub>wr</sub>it<sub>e</sub>

$$
G _ { \psi } ( \alpha ) = \operatorname* { s u p } _ { u } \bigl ( V _ { u } ^ { \pi _ { F } } ( s _ { \iota } ) - V _ { u } ^ { K _ { \alpha } } ( s _ { \iota } ) \bigr ) .
$$

Lemma 36 (DNF dichotomy). Let $M = n + m$ . For every α, $i f \forall y : \psi ( \alpha , y )$ then $\begin{array} { r } { G _ { \psi } ( \alpha ) \le 2 - \frac { 1 } { M } } \end{array}$ , while $i f \exists y : \lnot \psi ( \alpha , y )$ then $G _ { \psi } ( \alpha ) = 2$

Proof. At a Boolean vertex, let $A _ { j }$ b<sub>e</sub> th<sub>e even</sub>t th<sub>a</sub>t th<sub>e</sub> $y _ { j }$ <sub>au</sub>dit <sub>passes,</sub> l<sub>e</sub>t $B$ b<sub>e</sub> th<sub>e even</sub>t th<sub>a</sub>t th<sub>e</sub> t<sub>erm scanner passes, an</sub>d l<sub>e</sub>t $C _ { i }$ b<sub>e</sub> th<sub>e even</sub>t th<sub>a</sub>t th<sub>e</sub> $x _ { i }$ <sub>au</sub>dit <sub>aga</sub>i<sub>ns</sub>t $\alpha ( x _ { i } )$ <sub>passes.</sub> Th<sub>e re</sub>f<sub>erence reac</sub>h<sub>es</sub> it<sub>s s</sub>i<sub>ng</sub>l<sub>e scanner</sub> b<sub>ranc</sub>h <sub>w</sub>ith <sub>pro</sub>b<sub>a</sub>bilit<sub>y one,</sub> <sub>w</sub>hil<sub>e</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e</sub>’<sub>s sp</sub>litt<sub>er g</sub>i<sub>ves eac</sub>h <sub>o</sub>f th<sub>e</sub> $n + m$ <sub>au</sub>dit b<sub>ranc</sub>h<sub>es we</sub>i<sub>g</sub>ht $\textstyle { \frac { 1 } { n + m } }$ <sub>, so</sub> th<sub>e pa</sub>ddi<sub>ng an</sub>d t<sub>erm</sub>i<sub>na</sub>l <sub>rewar</sub>d<sub>s g</sub>i<sub>ve</sub>

$$
{ \cal D } _ { \pi _ { F } , K _ { \alpha } } = \mathbf { 1 } _ { \{ B \} } + { \frac { | \{ j : A _ { j } \} | + | \{ i : C _ { i } \} | } { n + m } } .
$$

$\mathrm { I f } \exists y : \lnot \psi ( \alpha , y )$ <sub>,</sub> th<sub>e canon</sub>i<sub>ca</sub>l l<sub>oca</sub>l <sub>pa</sub>i<sub>rs</sub> f<sub>or</sub> $( \alpha , y )$ <sub>an</sub>d th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>c</sub>h<sub>o</sub>i<sub>ces</sub> $Y _ { j } = y _ { j }$ <sub>ma</sub>k<sub>e every even</sub>t h<sub>o</sub>ld<sub>.</sub> Thi<sub>s g</sub>i<sub>ves va</sub>l<sub>ue</sub> t<sub>wo,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>a</sub>l<sub>so</sub> th<sub>e</sub> l<sub>arges</sub>t <sub>poss</sub>ibl<sub>e</sub> <sub>va</sub>l<sub>ue.</sub>

<sup>No</sup>w <sup>s</sup>upp<sup>ose</sup> $\forall y : \psi ( \alpha , y )$ <sub>.</sub> All <sub>even</sub>t<sub>s canno</sub>t h<sub>o</sub>ld <sub>a</sub>t <sub>once.</sub> I<sub>n</sub>d<sub>ee</sub>d<sub>,</sub> if <sub>every au</sub>dit <sub>passes, se</sub>t $y _ { j } = Y _ { j }$ <sub>.</sub> Wh<sub>enever</sub> th<sub>e pa</sub>i<sub>r</sub> t<sub>es</sub>t d<sub>ec</sub>l<sub>ares an occurrence</sub> l<sub>oca</sub>ll<sub>y</sub> f<sub>a</sub>l<sub>se,</sub> it<sub>s va</sub>l<sub>ue-cer</sub>tif<sub>y</sub>i<sub>ng</sub> bit i<sub>s zero, w</sub>hil<sub>e</sub> th<sub>e pass</sub>i<sub>ng au</sub>dit f<sub>orces</sub> th<sub>e</sub> bit i<sub>n</sub>d<sub>exe</sub>d b<sub>y</sub> th<sub>e</sub> <sub>comm</sub>itt<sub>e</sub>d <sub>var</sub>i<sub>a</sub>bl<sub>e va</sub>l<sub>ue</sub> t<sub>o one.</sub> Th<sub>e comm</sub>itt<sub>e</sub>d <sub>va</sub>l<sub>ue</sub> th<sub>ere</sub>f<sub>ore ma</sub>k<sub>es</sub> th<sub>a</sub>t lit<sub>era</sub>l f<sub>a</sub>l<sub>se.</sub> E<sub>ven</sub>t $B$ <sub>wou</sub>ld th<sub>en</sub> i<sub>ve a</sub> f<sub>a</sub>l<sub>se</sub> lit<sub>era</sub> in <sub>eve</sub>r<sub>y</sub> DNF t<sub>e</sub>rm<sub>,</sub> <sub>co</sub>ntr<sub>a</sub>di<sub>c</sub>tin<sub>g</sub> $\psi ( \alpha , y )$ <sub>.</sub> At l<sub>eas</sub>t <sub>one even</sub>t f<sub>a</sub>il<sub>s, an</sub>d it<sub>s we</sub>i<sub>g</sub>ht i<sub>s a</sub>t l<sub>eas</sub>t

$$
\operatorname* { m i n } \left\{ 1 , \frac { 1 } { n + m } \right\} = \frac { 1 } { M } .
$$

Th<sub>us every ver</sub>t<sub>ex</sub> h<sub>as va</sub>l<sub>ue a</sub>t <sub>mos</sub>t $\textstyle { 2 - { \frac { 1 } { M } } }$ <sub>.</sub> N<sub>o run repea</sub>t<sub>s an uncer</sub>t<sub>a</sub>i<sub>n c</sub>h<sub>o</sub>i<sub>ce, so</sub> L<sub>emma</sub> 17 <sub>ex</sub>t<sub>en</sub>d<sub>s</sub> th<sub>e</sub> b<sub>oun</sub>d t<sub>o</sub> th<sub>e</sub> f<sub>u</sub>ll p<sup>rod</sup>u<sup>ct</sup> p<sup>ol</sup>y<sup>to</sup>p<sup>e</sup>. □

Lemma 37. Minimal robust regret over $\Pi ^ { \mathrm { M D } } i s \Sigma _ { 2 } ^ { p }$ -hard on acyclic $( s , a )$ -rectangular RMDPs in which every uncertain choice is two-Dirac and the only other stochastic rows are certain uniform splitters.

Proof. Set $M = n + m$ <sub>.</sub> B<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> 36<sub>, a yes-</sub>in<sub>s</sub>t<sub>a</sub>n<sub>ce</sub> h<sub>as so</sub>m<sub>e</sub> $K _ { \alpha }$ <sub>w</sub>ith $\begin{array} { r } { G _ { \psi } ( \alpha ) \le 2 - \frac { 1 } { M } } \end{array}$ <sub>, w</sub>hil<sub>e every</sub> $K _ { \alpha }$ i<sub>n a no-</sub>i<sub>ns</sub>t<sub>ance</sub> h<sub>as</sub> $G _ { \psi } ( \alpha ) = 2$ <sub>.</sub> A<sub>pp</sub>l<sub>y</sub> th<sub>e exac</sub>t lift <sub>w</sub>ith <sub>re</sub>f<sub>erence</sub> $\pi _ { F }$ <sub>.</sub> At <sub>ex</sub>i<sub>s</sub>t<sub>en</sub>ti<sub>a</sub>l <sub>var</sub>i<sub>a</sub>bl<sub>e s</sub>t<sub>a</sub>t<sub>es a</sub>ll<sub>ow</sub> $\{ T , F \}$ <sub>.</sub> At <sub>every o</sub>th<sub>er s</sub>t<sub>a</sub>t<sub>e a</sub>ll<sub>ow</sub> th<sub>e un</sub>i<sub>que ac</sub>ti<sub>on use</sub>d b<sub>y</sub> th<sub>e syn</sub>th<sub>es</sub>i<sub>ze</sub>d <sub>ver</sub>ifi<sub>er, an</sub>d <sub>a</sub>t i<sub>n</sub>t<sub>erna</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es use</sub>d <sub>on</sub>l<sub>y</sub> b<sub>y</sub> $\pi _ { F }$ t<sub>a</sub>k<sub>e</sub> $\bar { A } _ { \mathrm { a l l o w } } ( s ) = \{ \pi _ { F } ( s ) \}$ <sub>.</sub> Th<sub>us</sub> <sub>every a</sub>ll<sub>owe</sub>d <sub>se</sub>t i<sub>s nonemp</sub>t<sub>y.</sub> At <sub>a s</sub>h<sub>are</sub>d l<sub>oca</sub>l <sub>ou</sub>t<sub>come</sub> thi<sub>s a</sub>ll<sub>ows</sub> th<sub>e au</sub>dit <sub>ac</sub>ti<sub>on an</sub>d <sub>no</sub>t th<sub>e scanner ac</sub>ti<sub>on, w</sub>hi<sub>c</sub>h <sub>cos</sub>t<sub>s</sub> th<sub>e re</sub>f<sub>erence no</sub>thi<sub>ng:</sub> b<sub>y</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 10 th<sub>e</sub> lift <sub>rep</sub>l<sub>ays</sub> $\pi _ { F }$ th<sub>roug</sub>h th<sub>e</sub> b<sub>onus ac</sub>ti<sub>on ra</sub>th<sub>er</sub> th<sub>an</sub> th<sub>roug</sub>h $A _ { \mathrm { a l l o w } }$ <sub>.</sub> Th<sub>e a</sub>ll<sub>owe</sub>d d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c po</sub>li<sub>c</sub>i<sub>es</sub> th<sub>ere</sub>f<sub>ore</sub> h<sub>ave m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>regre</sub>t <sub>a</sub>t <sub>mos</sub>t $\Lambda + \textstyle { \overline { { 2 } } } - { \frac { 1 } { M } }$ i<sub>n yes-</sub>i<sub>ns</sub>t<sub>ances an</sub>d <sub>exac</sub>tl<sub>y</sub> $\Lambda + 2$ i<sub>n no-</sub>i<sub>ns</sub>t<sub>ances.</sub>

Th<sub>e</sub> <sub>compose</sub>d t<sub>arge</sub>t i<sub>s</sub>

$$
\mathrm { R e s t r i c t } \left( \mathrm { L i f t } ( N _ { \psi } , \pi _ { F } ) , A _ { \mathrm { a l l o w } } , \frac { 1 } { 4 M } \right) .
$$

A<sub>pp</sub>l<sub>y</sub> L<sub>emma</sub> 32 <sub>w</sub>ith $\mathcal { P } = \Pi ^ { \mathrm { M D } }$ <sub>.</sub> Th<sub>e</sub> di<sub>sa</sub>ll<sub>owe</sub>d b<sub>onus c</sub>h<sub>o</sub>i<sub>ces are s</sub>i<sub>ng</sub>l<sub>e</sub>t<sub>ons.</sub> I<sub>n yes-</sub>i<sub>ns</sub>t<sub>ances</sub> th<sub>e unres</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d <sub>m</sub>i<sub>n</sub>i<sub>mum</sub> i<sub>s</sub> at most $\textstyle \Lambda + 2 - { \frac { 1 } { M } }$ <sub>, w</sub>hil<sub>e</sub> i<sub>n no-</sub>i<sub>ns</sub>t<sub>ances</sub> it i<sub>s a</sub>t l<sub>eas</sub>t $\Lambda + 2 - \textstyle { \frac { 1 } { 4 M } }$ <sub>.</sub> Si<sub>nce</sub>

$$
\Lambda + 2 - \frac { 1 } { M } < \Lambda + 2 - \frac { 1 } { 2 M } < \Lambda + 2 - \frac { 1 } { 4 M } ,
$$

th<sub>res</sub>h<sub>o</sub>ld $\textstyle \Lambda + 2 - { \frac { 1 } { 2 M } }$ <sub>separa</sub>t<sub>es</sub> th<sub>e cases.</sub> Th<sub>e s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ons</sub> f<sub>o</sub>ll<sub>ow</sub> f<sub>rom</sub> L<sub>emmas</sub> 30 <sub>an</sub>d 32<sub>, w</sub>h<sub>ose un</sub>f<sub>o</sub>ldi<sub>ng s</sub>t<sub>ep</sub> i<sub>s</sub> <sub>w</sub>h<sub>a</sub>t k<sub>eeps</sub> th<sub>e</sub> <sub>compose</sub>d t<sub>arge</sub>t <sub>acyc</sub>li<sub>c.</sub> □

Theorem 38. Minimal robust regret over memoryless deterministic policies is $\Sigma _ { 2 } ^ { p }$ -complete on acyclic $( s , a )$ -rectangular polytopic RMDPs. Hardness already holds when every uncertain choice is two-Dirac and the only other stochastic rows are certain uniform splitters.

Proof of Theorem 38. Membership is Lemma 35. Hardness is Lemma 37.

## H Proof of Theorem 7: Signed Square-Root-Sum Hardness

Th<sub>e</sub> <sub>source</sub> <sub>pro</sub>bl<sub>em</sub> ${ \mathsf { S Q R S } } _ { \pm }$ <sub>g</sub>i<sub>ves</sub> t<sub>wo</sub> li<sub>s</sub>t<sub>s</sub> <sub>o</sub>f <sub>pos</sub>iti<sub>ve</sub> i<sub>n</sub>t<sub>egers</sub> <sub>an</sub>d <sub>as</sub>k<sub>s,</sub> i<sub>n</sub> it<sub>s</sub> <sub>non-s</sub>t<sub>r</sub>i<sub>c</sub>t f<sub>orwar</sub>d di<sub>rec</sub>ti<sub>on,</sub> <sub>w</sub>h<sub>e</sub>th<sub>er</sub>

$$
\sum _ { i } { \sqrt { a _ { i } } } \leq \sum _ { j } { \sqrt { b _ { j } } } .
$$

Th<sub>e re</sub>d<sub>uc</sub>ti<sub>on represen</sub>t<sub>s eac</sub>h <sub>square roo</sub>t b<sub>y a one-s</sub>t<sub>a</sub>t<sub>e ga</sub>d<sub>ge</sub>t <sub>w</sub>h<sub>ose regre</sub>t b<sub>a</sub>l<sub>ances a</sub> d<sub>ecreas</sub>i<sub>ng</sub> t<sub>erm aga</sub>i<sub>ns</sub>t <sub>an</sub> i<sub>ncreas</sub>i<sub>ng</sub> t<sub>erm</sub> i<sub>n</sub> th<sub>e po</sub>li<sub>cy</sub>’<sub>s m</sub>i<sub>x</sub>i<sub>ng pro</sub>b<sub>a</sub>bilit<sub>y.</sub> Th<sub>e op</sub>ti<sub>mum</sub> i<sub>s</sub> th<sub>ere</sub>f<sub>ore</sub> th<sub>e so</sub>l<sub>u</sub>ti<sub>on o</sub>f <sub>a qua</sub>d<sub>ra</sub>ti<sub>c equa</sub>ti<sub>on an</sub>d i<sub>s genera</sub>ll<sub>y</sub> i<sub>rra</sub>ti<sub>ona</sub>l<sub>.</sub> Theorem 7. Minimal robust regret is ${ \mathsf { S Q R S } } _ { \pm }$ -hard under $( s , a )$ -rectangular uncertainty.

## H.1 One-State Balancing

![](images/f96b67534e33cbf71d7f1a6db69e0ea84ffaa7fcd6bc9f0aa371dd43ecd82f37.jpg)  
Fi<sub>gure</sub> 10<sub>:</sub> Th<sub>e</sub> <sub>one-s</sub>t<sub>a</sub>t<sub>e</sub> b<sub>a</sub>l<sub>anc</sub>i<sub>ng</sub> <sub>ga</sub>d<sub>ge</sub>t h<sub>as</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>c</sub>h<sub>o</sub>i<sub>ces</sub> $p _ { L } , p _ { R } \in [ 0 , h ]$ <sub>an</sub>d <sub>a zero-rewar</sub>d <sub>s</sub>i<sub>n</sub>k<sub>.</sub>

Definition 17 (One-state balancing gadget). Fix $q \in ( 0 , 1 )$ and $\gamma \in ( 1 - q , 1 )$ , and put $h = ( 1 - q ) / \gamma$ . The one-state balancing RMDP shown in Figure 10 is

$$
G = ( S , A , \mathcal { U } , R , s , \gamma ) ,
$$

with the following components.

$S = \{ s , \bot \}$ , where ⊥ is a zero-reward sink.

• The global action set is $\{ L , R \} . A t \perp$ , both actions have a zero-reward Dirac self-loop.

• U is the product of the independent uncertainty sets

$$
\mathcal { U } _ { ( s , L ) } = \left\{ p _ { L } \delta _ { s } + ( 1 - p _ { L } ) \delta _ { \perp } : p _ { L } \in [ 0 , h ] \right\} , \qquad \mathcal { U } _ { ( s , R ) } = \left\{ p _ { R } \delta _ { s } + ( 1 - p _ { R } ) \delta _ { \perp } : p _ { R } \in [ 0 , h ] \right\} .
$$

• R assigns rewards $R _ { L }$ and $R _ { R }$ to actions L and R at s, respectively, and assigns zero reward at ⊥.

• The initial state is s and the discount is the local parameter γ.

The condition $\gamma > 1 - q$ is exactly $h < 1$ , so these are valid transition probabilities. Write $d _ { L } = 1 - \gamma p _ { L }$ and $d _ { R } = 1 - \gamma p _ { R }$ Both range over [q, 1].

F<sub>or</sub> th<sub>e ga</sub>d<sub>ge</sub>t <sub>o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 17<sub>,</sub> th<sub>e</sub> t<sub>wo parame</sub>t<sub>r</sub>i<sub>za</sub>ti<sub>ons</sub> b<sub>e</sub>l<sub>ow c</sub>h<sub>oose</sub> $( R _ { L } , R _ { R } )$ <sub>so</sub> th<sub>a</sub>t it<sub>s m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t <sub>rea</sub>li<sub>zes</sub> th<sub>e</sub> <sub>s</sub>i<sub>gne</sub>d <sub>square-roo</sub>t f<sub>orms</sub> $m _ { + }$ <sub>an</sub>d $m _ { - }$ <sub>summe</sub>d b<sub>y</sub> th<sub>e</sub> ${ \mathsf { S Q R S } } _ { \pm }$ <sub>re</sub>d<sub>uc</sub>ti<sub>on.</sub> F<sub>or</sub> $A , B > 0$ , pu<sup>t</sup> $D = A + B$ . The positive-reward parametrization is

$$
r _ { L } = \frac { q ( A + q B ) } { 1 - q ^ { 2 } } , \qquad r _ { R } = \frac { q ( B + q A ) } { 1 - q ^ { 2 } } , \qquad ( R _ { L } , R _ { R } ) = ( r _ { L } , r _ { R } ) ,
$$

<sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
m _ { + } ( A , B ) = \frac { D - \sqrt { D ^ { 2 } - 4 ( 1 - q ^ { 2 } ) A B } } { 2 ( 1 - q ^ { 2 } ) } .
$$

The negative-reward parametrization is

$$
\ell = \frac { B + q A } { 1 - q ^ { 2 } } , \qquad r = \frac { A + q B } { 1 - q ^ { 2 } } , \qquad ( R _ { L } , R _ { R } ) = ( - \ell , - r ) ,
$$

<sub>an</sub>d d<sub>e</sub>fi<sub>ne</sub>

$$
m _ { - } ( A , B ) = \frac { \sqrt { q ^ { 2 } D ^ { 2 } + 4 ( 1 - q ^ { 2 } ) A B } - q D } { 2 ( 1 - q ^ { 2 } ) } .
$$

Lemma 39 (One-state balancing gadgets). The positive-reward parametrization has minimal robust regret $m _ { + } ( A , B )$ . If $q ^ { 2 } \geq 1 / 2 ,$ , the negative-reward parametrization has minimal robust regret $m _ { - } ( A , B )$

Proof. A memoryless randomized policy in this gadget is determined by one number $x \in [ 0 , 1 ]$ , the probability of playing L at $s .$

Fi<sub>rs</sub>t<sub>, cons</sub>id<sub>er</sub> th<sub>e pos</sub>iti<sub>ve-rewar</sub>d f<sub>orm.</sub> F<sub>or</sub> fi<sub>xe</sub>d $d _ { L } , d _ { R } \in [ q , 1 ]$

$$
V _ { L } = \frac { r _ { L } } { d _ { L } } , \qquad V _ { R } = \frac { r _ { R } } { d _ { R } } , \qquad V _ { x } = \frac { x r _ { L } + ( 1 - x ) r _ { R } } { x d _ { L } + ( 1 - x ) d _ { R } } .
$$

A di<sub>rec</sub>t <sub>ca</sub>l<sub>cu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>g</sub>i<sub>ves</sub>

$$
V _ { L } - V _ { x } = \frac { ( 1 - x ) ( r _ { L } d _ { R } - r _ { R } d _ { L } ) } { d _ { L } ( x d _ { L } + ( 1 - x ) d _ { R } ) }
$$

<sub>an</sub>d

$$
V _ { R } - V _ { x } = \frac { x ( r _ { R } d _ { L } - r _ { L } d _ { R } ) } { d _ { R } ( x d _ { L } + ( 1 - x ) d _ { R } ) } .
$$

O<sub>n</sub> th<sub>e</sub> <sub>reg</sub>i<sub>on</sub> $V _ { L } \ \ge \ V _ { R }$ <sub>,</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>express</sub>i<sub>on</sub> i<sub>s</sub> d<sub>ecreas</sub>i<sub>ng</sub> i<sub>n</sub> $d _ { L }$ <sub>an</sub>d i<sub>ncreas</sub>i<sub>ng</sub> i<sub>n</sub> $d _ { R } ,$ <sub>, so</sub> it<sub>s max</sub>i<sub>mum</sub> i<sub>s a</sub>tt<sub>a</sub>i<sub>ne</sub>d <sub>a</sub>t $d _ { L } = q , d _ { R } = 1$ <sub>.</sub> Si<sub>nce</sub>

$$
\frac { r _ { L } } { q } - r _ { R } = A ,
$$

thi<sub>s max</sub>i<sub>mum</sub> i<sub>s</sub>

$$
{ \frac { A ( 1 - x ) } { 1 - ( 1 - q ) x } } .
$$

Si<sub>m</sub>il<sub>ar</sub>l<sub>y,</sub> <sub>on</sub> th<sub>e</sub> <sub>reg</sub>i<sub>on</sub> $V _ { R } \geq V _ { L }$ <sub>,</sub> th<sub>e</sub> <sub>secon</sub>d <sub>express</sub>i<sub>on</sub> i<sub>s</sub> <sub>max</sub>i<sub>m</sub>i<sub>ze</sub>d <sub>a</sub>t $d _ { L } = 1 , d _ { R } = q$ <sub>.</sub> Si<sub>nce</sub>

$$
\frac { r _ { R } } { q } - r _ { L } = B ,
$$

thi<sub>s</sub> <sub>max</sub>i<sub>mum</sub> i<sub>s</sub>

$$
{ \frac { B x } { q + ( 1 - q ) x } } .
$$

Th<sub>us</sub>

$$
\operatorname { R r e g } ( x ) = \operatorname* { m a x } \left\{ { \frac { A ( 1 - x ) } { 1 - ( 1 - q ) x } } , { \frac { B x } { q + ( 1 - q ) x } } \right\} .
$$

The first term is strictl decreasin in x and the second is strictl increasin in x. Hence the minimum is attained when the two t<sub>erms are equa</sub>l<sub>.</sub> If th<sub>e common va</sub>l<sub>ue</sub> i<sub>s</sub> $y ,$ th<sub>en</sub>

$$
( 1 - q ^ { 2 } ) y ^ { 2 } - D y + A B = 0 , \qquad D = A + B .
$$

Th<sub>e sma</sub>ll<sub>er roo</sub>t <sub>g</sub>i<sub>ves</sub> th<sub>e c</sub>l<sub>a</sub>i<sub>me</sub>d <sub>va</sub>l<sub>ue.</sub>

F<sub>or</sub> th<sub>e</sub> <sub>nega</sub>ti<sub>ve-rewar</sub>d f<sub>orm</sub> th<sub>e</sub> <sub>same</sub> t<sub>wo-reg</sub>i<sub>on</sub> <sub>sp</sub>lit <sub>app</sub>li<sub>es,</sub> <sub>now</sub> <sub>w</sub>ith $V _ { L } = - \ell / d _ { L }$ <sub>an</sub>d $V _ { R } = - r / d _ { R }$ <sub>.</sub> Th<sub>e con</sub>diti<sub>ons</sub> $A , B > 0$ <sub>are equ</sub>i<sub>va</sub>l<sub>en</sub>t t<sub>o</sub>

$$
r - q \ell = A > 0 , \ell - q r = B > 0 ,
$$

<sub>an</sub>d i<sub>mp</sub>l<sub>y</sub> $q < r / \ell < 1 / q$ <sub>.</sub> F<sub>or</sub> $V _ { L } - V _ { x } , \mathrm { s e t } z = d _ { R } / d _ { L } \in [ q , 1 / q ] . \mathrm { O n } z \in [ q , 1 ]$ <sub>,</sub> <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> th<sub>e</sub> f<sub>eas</sub>ibl<sub>e</sub> $d _ { L }$ <sub>g</sub><sup>i</sup>ves $\frac { ( 1 - x ) z ( r - \ell z ) } { q ( x + ( 1 - x ) z ) }$ <sub>w</sub>h<sub>ose</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>ve</sub> h<sub>as numera</sub>t<sub>or</sub> $x ( r - 2 \ell z ) - ( 1 - x ) \ell z ^ { 2 } \leq 0$ b<sub>ecause</sub> $z \geq q , r / \ell < 1 / q ,$ <sub>an</sub>d $q ^ { 2 } \geq 1 / 2 . \mathrm { O n } z \in \overline { { [ 1 , 1 / q ] } }$ <sub>,</sub> th<sub>e</sub> corres<sub>p</sub>on<sup>di</sup>n<sub>g</sub> ex<sub>p</sub>ress<sup>i</sup>on $\frac { ( 1 - x ) ( r - \ell z ) } { q ( x + ( 1 - x ) z ) }$ i<sub>s s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> d<sub>ecreas</sub>i<sub>ng.</sub> Th<sub>us</sub> th<sub>e max</sub>i<sub>mum occurs a</sub>t $z = q$ , name<sup>l</sup><sub>y</sub> $d _ { L } = 1 , d _ { R } = q .$ <sub>,</sub> <sub>an</sub>d e<sub>q</sub>ua<sup>l</sup>s $A ( 1 - x ) / ( q + ( 1 - q ) \dot { x } )$ ). By symmetry the other region contributes $B x / ( 1 - ( 1 - q ) x )$ <sub>.</sub> H<sub>ence</sub>

$$
\operatorname { R r e g } ( x ) = \operatorname* { m a x } \left\{ { \frac { A ( 1 - x ) } { q + ( 1 - q ) x } } , { \frac { B x } { 1 - ( 1 - q ) x } } \right\} .
$$

B<sub>a</sub>l<sub>anc</sub>i<sub>ng</sub> th<sub>e</sub> d<sub>ecreas</sub>i<sub>ng</sub> <sub>an</sub>d i<sub>ncreas</sub>i<sub>ng</sub> t<sub>erms</sub> <sub>g</sub>i<sub>ves</sub> $( 1 - q ^ { 2 } ) y ^ { 2 } + q D y - A B = 0$ <sub>.</sub> It<sub>s</sub> <sub>pos</sub>iti<sub>ve</sub> <sub>roo</sub>t i<sub>s</sub> $m _ { - } ( A , B )$

## H.2 Rational Square-Root Gadgets

Lemma 40 (Local gadget for a negative square-root coeficient). For every integer $b \geq 2 ,$ , one can compute in polynomial time rational numbers $q _ { b } , D _ { b } , A _ { b } , B _ { b }$ such that, for any $\gamma \in ( 1 - q _ { b } , 1 )$ , the positive-reward balancing gadget ofLemma 39, after a rational reward scaling, has minimal robust regret of

$$
D _ { b } - { \sqrt { b } } .
$$

Proof. Choose a rational number $\eta \in ( \sqrt { b } - 1 , \sqrt { b - 1 } )$ <sub>.</sub> Th<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l h<sub>as</sub> l<sub>eng</sub>th <sub>grea</sub>t<sub>er</sub> th<sub>an</sub> $1 / 2$ f<sub>or</sub> $b \geq 2 .$ , so scann<sup>i</sup>n<sub>g</sub> a constant-denominator dyadic grid finds such an η. Both endpoint tests use exact rational square comparisons. Thus η has $O ( \log b )$ bit<sub>s an</sub>d i<sub>s</sub> f<sub>oun</sub>d i<sub>n po</sub>l<sub>ynom</sub>i<sub>a</sub>l ti<sub>me.</sub> D<sub>e</sub>fi<sub>ne</sub>

$$
q _ { b } = \frac { b - 1 - \eta ^ { 2 } } { 2 \eta } , N _ { b } = \frac { \eta + \frac { b - 1 } { \eta } } { 2 } , D _ { b } = \frac { N _ { b } } { q _ { b } } .
$$

The upper bound on η gives $q _ { b } > 0$ <sub>, w</sub>hil<sub>e</sub> $\eta > \sqrt { b } - 1$ i<sub>s</sub> <sub>equ</sub>i<sub>va</sub>l<sub>en</sub>t t<sub>o</sub> $q _ { b } < 1$ <sub>.</sub> Di<sub>rec</sub>t <sub>expans</sub>i<sub>on</sub> <sub>g</sub>i<sub>ves</sub> $N _ { b } ^ { 2 } - q _ { b } ^ { 2 } = b - 1$ <sub>.</sub> H<sub>ence</sub> $N _ { b } > q _ { b } ,$ so $D _ { b } > 1$ <sub>.</sub> H<sub>ence,</sub> $\dot { q } _ { b } ^ { 2 } D _ { b } ^ { 2 } + 1 - \dot { q } _ { b } ^ { 2 } = b$

No<sub>w</sub> <sub>s</sub>et

$$
A _ { b } = \frac { D _ { b } + 1 } { 2 } , \qquad B _ { b } = \frac { D _ { b } - 1 } { 2 } .
$$

Th<sub>en</sub> $A _ { b } , B _ { b } > 0 , A _ { b } + B _ { b } = D _ { b }$ <sub>,</sub> <sub>an</sub>d

$$
\begin{array} { r l } & { D _ { b } ^ { 2 } - 4 ( 1 - q _ { b } ^ { 2 } ) A _ { b } B _ { b } = D _ { b } ^ { 2 } - ( 1 - q _ { b } ^ { 2 } ) ( D _ { b } ^ { 2 } - 1 ) } \\ & { \phantom { D _ { b } ^ { 2 } - } = q _ { b } ^ { 2 } D _ { b } ^ { 2 } + 1 - q _ { b } ^ { 2 } } \\ & { \phantom { D _ { b } ^ { 2 } - } = b . } \end{array}
$$

B<sub>y</sub> L<sub>emma</sub> 39<sub>,</sub> th<sub>e</sub> <sub>unsca</sub>l<sub>e</sub>d <sub>pos</sub>iti<sub>ve-rewar</sub>d <sub>ga</sub>d<sub>ge</sub>t h<sub>as</sub> <sub>m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t

$$
{ \frac { D _ { b } - { \sqrt { b } } } { 2 ( 1 - q _ { b } ^ { 2 } ) } } .
$$

S<sub>ca</sub>li<sub>ng</sub> <sub>a</sub>ll <sub>rewar</sub>d<sub>s</sub> b<sub>y</sub> $2 ( 1 - q _ { b } ^ { 2 } )$ <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>va</sub>l<sub>ue</sub> $D _ { b } - { \sqrt { b } } .$ <sub>.</sub> M<sub>oreover</sub> $D _ { b } ^ { 2 } = 1 + ( b - 1 ) / q _ { b } ^ { 2 } > b ,$ <sub>, so</sub> thi<sub>s</sub> l<sub>oca</sub>l <sub>regre</sub>t i<sub>s pos</sub>iti<sub>ve.</sub> All <sub>ra</sub>ti<sub>ona</sub>l <sub>opera</sub>ti<sub>ons</sub> <sub>preserve</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l bit l<sub>eng</sub>th<sub>.</sub> □

E<sub>xamp</sub>l<sub>e</sub> 7 i<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>t<sub>es</sub> th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on a</sub>t $b = 2$

Example 7 (The gadget for $b = 2 )$ . Take $\eta = 1 / 2$ . Then $q _ { 2 } = 3 / 4 , N _ { 2 } = 5 / 4 ,$ , and $D _ { 2 } = 5 / 3$ , with $N _ { 2 } ^ { 2 } - q _ { 2 } ^ { 2 } = 1 = b - 1$ After the prescribed reward scaling, the local minimal regret is $5 / 3 - { \sqrt { 2 } } .$

Lemma 41 (Local gadget for a positive square-root coeficient). For every integer $b \geq 2 ,$ , one can compute in polynomial time a rational constant $C _ { b }$ and a negative-reward $( s , a )$ -rectangular gadget such that,for any $\gamma \in \left( \textstyle { \frac { 1 } { 5 } } , 1 \right)$ , its minimal robust regret is

$$
{ \sqrt { b } } - C _ { b } .
$$

Proof. Choose a rational number m such that

$$
\frac { 2 b } { 3 } < m ^ { 2 } < b .
$$

Th<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l $( { \sqrt { 2 b / 3 } } , { \sqrt { b } } )$ h<sub>as w</sub>idth <sub>a</sub>t l<sub>eas</sub>t ${ \sqrt { 2 } } ( 1 - { \sqrt { 2 / 3 } } )$ <sub>.</sub> S<sub>cann</sub>i<sub>ng</sub> d<sub>ya</sub>di<sub>cs</sub> <sub>o</sub>f <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l bit l<sub>eng</sub>th <sub>w</sub>ith <sub>ra</sub>ti<sub>ona</sub>l <sub>square</sub> comparisons therefore finds m in polynomial time. Define

$$
D = { \frac { m + { \frac { b } { m } } } { 2 } } , \qquad c = { \frac { { \frac { b } { m } } - m } { 2 } } .
$$

Th<sub>en</sub> $D , c \in \mathbb { Q } , c > 0$ <sub>,</sub> <sub>an</sub>d $D ^ { 2 } - c ^ { 2 } = b$ <sub>.</sub> M<sub>oreover,</sub>

$$
\frac { c } { D } = \frac { b - m ^ { 2 } } { b + m ^ { 2 } } < \frac { 1 } { 5 } .
$$

S<sub>e</sub>t

$$
q = \frac { 4 } { 5 } , \quad E = \frac { 5 } { 3 } c , \quad A = \frac { D + E } { 2 } , \quad B = \frac { D - E } { 2 } .
$$

Si<sub>nce</sub> $\begin{array} { r } { \frac { E } { D } < \frac { 1 } { 3 } } \end{array}$ , we have A, $B > 0$ <sub>.</sub> Al<sub>so</sub> $A + B = D$ <sub>,</sub> <sub>an</sub>d b<sub>ecause</sub> $\textstyle 1 - q ^ { 2 } = { \frac { 9 } { 2 5 } }$ <sub>,</sub> <sub>we</sub> h<sub>ave</sub>

$$
\begin{array} { r l } { q ^ { 2 } D ^ { 2 } + 4 ( 1 - q ^ { 2 } ) A B = q ^ { 2 } D ^ { 2 } + ( 1 - q ^ { 2 } ) ( D ^ { 2 } - E ^ { 2 } ) } \\ { = D ^ { 2 } - ( 1 - q ^ { 2 } ) E ^ { 2 } } \\ { = D ^ { 2 } - c ^ { 2 } } \\ { = b . } \end{array}
$$

Fi<sub>na</sub>ll<sub>y,</sub> $\frac { c } { D } < \frac { 1 } { 5 }$ i<sub>mp</sub>li<sub>es</sub> $\begin{array} { r } { D ^ { 2 } = b + c ^ { 2 } < \frac { 2 5 b } { \mathcal { D } 4 } } \end{array}$ <sub>,</sub> <sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> $q ^ { 2 } D ^ { 2 } < b .$ Th<sub>us</sub> $q D < { \sqrt { b } } .$

N<sub>ow use</sub> th<sub>e nega</sub>ti<sub>ve-rewar</sub>d f<sub>orm o</sub>f L<sub>emma</sub> 39 <sub>w</sub>ith th<sub>ese</sub> $q , A , B , D$ <sub>.</sub> It<sub>s</sub> <sub>unsca</sub>l<sub>e</sub>d <sub>m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t i<sub>s</sub>

$$
{ \frac { { \sqrt { b } } - q D } { 2 ( 1 - q ^ { 2 } ) } } .
$$

S<sub>ca</sub>li<sub>ng a</sub>ll <sub>rewar</sub>d<sub>s</sub> b<sub>y</sub> $2 ( 1 - q ^ { 2 } )$ <sub>g</sub>i<sub>ves</sub> <sub>m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t ${ \sqrt { b } } - q D$ <sub>.</sub> S<sub>e</sub>t $C _ { b } = q D$ <sub>.</sub> Th<sub>e</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>t i<sub>nequa</sub>lit<sub>y</sub> $q D < { \sqrt { b } }$ <sub>p</sub>rove<sup>d</sup> <sub>a</sub>b<sub>ove</sub> <sub>ma</sub>k<sub>es</sub> thi<sub>s</sub> l<sub>oca</sub>l <sub>regre</sub>t <sub>pos</sub>iti<sub>ve,</sub> <sub>an</sub>d <sub>a</sub>ll <sub>parame</sub>t<sub>ers</sub> h<sub>ave</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l bi<sub>nary</sub> l<sub>eng</sub>th<sub>.</sub> □

![](images/56df10d73344e34e48425996cd029e90b28f779e6f2b23f062bc4d1f3cf44ab9.jpg)  
Fi<sub>gure</sub> 11<sub>:</sub> Additi<sub>ve</sub> <sub>sp</sub>litt<sub>er</sub> f<sub>or</sub> <sub>m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>regre</sub>t<sub>.</sub>

## H.3 Additive Composition

Definition 18 (Additive s litter). Let $H _ { 1 } , \ldots , H _ { n }$ be disjoint $( s , a )$ -rectangular gadgets with common discount $\gamma ,$ entry states $s _ { i } ,$ local regrets R $\cdot \mathrm { e g } _ { H _ { i } }$ , and mutually independent uncertainty sets. Add a zero-reward initial certain uniform splitter $s _ { \mathrm { i n } }$ whose only action enters each $s _ { i }$ with probability $1 / n ,$ and scale every reward in every $H _ { i }$ by $n / \gamma$ (Figure 11). After scaling, every remaining choice uses ruinous-sink completion. Denote the resulting RMDP, whose uncertainty set is the product of the local sets, by H.

Lemma 42 (Additive composition for minimal regret). $\begin{array} { r } { I f m _ { i } = \operatorname* { i n f } _ { \pi _ { i } \in \Pi ^ { \mathrm { M R } } } } \end{array}$ $\mathrm { R r e g } _ { H _ { i } } ( \pi _ { i } )$ , then the additive splitter of Definition 18 satisfies

$$
\operatorname* { i n f } _ { \pi \in \Pi ^ { \mathrm { M R } } } \mathrm { R r e g } _ { H } ( \pi ) = \sum _ { i = 1 } ^ { n } m _ { i } .
$$

Proof. For a global policy π, let $\pi _ { i }$ b<sub>e</sub> it<sub>s</sub> <sub>res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>on</sub> t<sub>o</sub> $H _ { i } .$ <sup>For</sup> <sup>an</sup>y <sup>com</sup>p<sup>etin</sup>g p<sup>olic</sup>y $\pi ^ { * }$ an<sup>d</sup> an<sub>y</sub> rectan<sub>g</sub>u<sup>l</sup>ar uncerta<sup>i</sup>nt<sub>y</sub> <sub>rea</sub>li<sub>za</sub>ti<sub>on</sub> $\pmb { u } = ( \pmb { u } _ { 1 } , \dots , \pmb { u } _ { n } )$ <sub>,</sub> th<sub>e</sub> B<sub>e</sub>ll<sub>man equa</sub>ti<sub>on a</sub>t th<sub>e cer</sub>t<sub>a</sub>i<sub>n un</sub>if<sub>orm sp</sub>litt<sub>er</sub> h<sub>as va</sub>l<sub>ue</sub>

$$
V _ { u } ^ { \pi ^ { * } } ( s _ { \mathrm { i n } } ) - V _ { u } ^ { \pi } ( s _ { \mathrm { i n } } ) = \gamma \sum _ { i = 1 } ^ { n } \frac { 1 } { n } \frac { n } { \gamma } \left( V _ { u _ { i } } ^ { \pi _ { i } ^ { * } } ( s _ { i } ) - V _ { u _ { i } } ^ { \pi _ { i } } ( s _ { i } ) \right) = \sum _ { i = 1 } ^ { n } \left( V _ { u _ { i } } ^ { \pi _ { i } ^ { * } } ( s _ { i } ) - V _ { u _ { i } } ^ { \pi _ { i } } ( s _ { i } ) \right) .
$$

The local uncertainty sets are independent and the competing policy can be chosen independently inside the disjoint gadgets. H<sub>ence</sub>

$$
\operatorname { R r e g } _ { H } ( \pi ) = \sum _ { i = 1 } ^ { n } \operatorname { R r e g } _ { H _ { i } } ( \pi _ { i } ) .
$$

T<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> i<sub>n</sub>fi<sub>mum over</sub> th<sub>e pro</sub>d<sub>uc</sub>t <sub>o</sub>f th<sub>e</sub> l<sub>oca</sub>l <sub>po</sub>li<sub>cy s</sub>i<sub>mp</sub>l<sub>exes proves</sub> th<sub>e c</sub>l<sub>a</sub>i<sub>m.</sub> B<sub>y</sub> L<sub>emma</sub> 11<sub>, every ru</sub>i<sub>nous c</sub>h<sub>o</sub>i<sub>ce</sub> i<sub>s</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y worse</sub> th<sub>an a</sub> d<sub>escr</sub>ib<sub>e</sub>d l<sub>oca</sub>l <sub>one, so a</sub>ddi<sub>ng</sub> th<sub>e g</sub>l<sub>o</sub>b<sub>a</sub>l <sub>ac</sub>ti<sub>on a</sub>l<sub>p</sub>h<sub>a</sub>b<sub>e</sub>t <sub>crea</sub>t<sub>es no</sub> f<sub>ur</sub>th<sub>er m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zer or compara</sub>t<sub>or.</sub> Th<sub>e</sub> construction preserves (s, a)-rectangularity. □

## H.4 Proof of Theorem 7

ProofofTheorem 7. Reduce from the non-strict ≤ direction of ${ \mathsf { S Q R S } } _ { \pm }$ <sub>.</sub> Gi<sub>ven</sub> t<sub>wo</sub> li<sub>s</sub>t<sub>s</sub> $a _ { 1 } , \ldots , a _ { m }$ <sub>an</sub>d $b _ { 1 } , \ldots , b _ { n }$ <sub>,</sub> we construct i<sub>n po</sub>l<sub>ynom</sub>i<sub>a</sub>l ti<sub>me an</sub> $( s , a )$ )-rectangular RMDP M and a rational threshold t such that

$$
\operatorname* { i n f } _ { \pi \in \Pi ^ { \mathrm { M R } } } \mathrm { R r e g } _ { M } ( \pi ) \leq t
$$

if <sub>an</sub>d <sub>on</sub>l<sub>y</sub> if

$$
\sum _ { i = 1 } ^ { m } { \sqrt { a _ { i } } } \leq \sum _ { j = 1 } ^ { n } { \sqrt { b _ { j } } } .
$$

Th<sub>e reverse</sub>d <sub>non-s</sub>t<sub>r</sub>i<sub>c</sub>t <sub>compar</sub>i<sub>son uses</sub> th<sub>e same cons</sub>t<sub>ruc</sub>ti<sub>on a</sub>ft<sub>er swapp</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>wo</sub> li<sub>s</sub>t<sub>s.</sub> L<sub>e</sub>t $I = \{ i \mid a _ { i } \geq 2 \} , J = \{ j \mid$ $b _ { j } \geq 2 \}$ <sub>, an</sub>d l<sub>e</sub>t $r _ { 0 } = | \{ i \mid { \bar { a } } _ { i } = 1 \} | - | \{ j \mid b _ { j } = 1 \}$ b<sub>e</sub> th<sub>e ra</sub>ti<sub>ona</sub>l <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on o</sub>f <sub>un</sub>it <sub>roo</sub>t<sub>s.</sub> If $I \cup J = \emptyset$ , w<sup>e</sup> <sup>o</sup>u<sup>t</sup>pu<sup>t</sup> <sup>a</sup> <sub>s</sub>i<sub>ng</sub>l<sub>e-s</sub>t<sub>a</sub>t<sub>e zero-rewar</sub>d RMDP <sub>w</sub>ith th<sub>res</sub>h<sub>o</sub>ld $- r _ { 0 }$ <sub>, w</sub>hi<sub>c</sub>h i<sub>s a yes-</sub>i<sub>ns</sub>t<sub>ance exac</sub>tl<sub>y w</sub>h<sub>en</sub> $r _ { 0 } \le 0$

I<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng,</sub> <sub>assume</sub> th<sub>a</sub>t <sub>a</sub>t l<sub>eas</sub>t <sub>one</sub> <sub>non-un</sub>it <sub>roo</sub>t i<sub>s</sub> <sub>presen</sub>t<sub>.</sub> P<sub>u</sub>t $N = | \boldsymbol { I } | + | \boldsymbol { J } |$ f<sub>or</sub> th<sub>e num</sub>b<sub>er o</sub>f l<sub>oca</sub>l <sub>ga</sub>d<sub>ge</sub>t<sub>s.</sub> F<sub>o</sub> <sup>ever</sup>y $i \in I .$ <sub>,</sub> <sub>app</sub>l<sub>y</sub> L<sub>emma</sub> 41<sub>.</sub> Thi<sub>s</sub> <sub>g</sub>i<sub>ves</sub> <sub>a</sub> l<sub>oca</sub>l <sub>ga</sub>d<sub>ge</sub>t <sub>w</sub>ith <sub>m</sub>i<sub>n</sub>i<sub>ma</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t

$$
\sqrt { a _ { i } } - C _ { i } ^ { + }
$$

f<sub>or a ra</sub>ti<sub>ona</sub>l <sub>cons</sub>t<sub>an</sub>t $C _ { i } ^ { + }$ . <sup>F</sup>or ever<sub>y</sub> $j \in J ,$ <sub>,</sub> <sub>app</sub>l<sub>y</sub> L<sub>emma</sub> 40 t<sub>o</sub> <sub>o</sub>bt<sub>a</sub>i<sub>n</sub> <sub>ra</sub>ti<sub>ona</sub>l <sub>parame</sub>t<sub>ers</sub> $q _ { j } , D _ { j } ^ { - } , A _ { j } ^ { - } , B _ { j } ^ { - }$ <sub>.</sub> Aft<sub>er</sub> th<sub>e rewar</sub>d scaling from that lemma, gadget j has local minimal robust regret

$$
{ \cal D } _ { j } ^ { - } - \sqrt { b _ { j } } .
$$

Ch<sub>oose one common</sub> di<sub>scoun</sub>t f<sub>ac</sub>t<sub>or</sub>

$$
q _ { \operatorname* { m i n } } = \operatorname* { m i n } \left( \left\{ { \frac { 4 } { 5 } } \right\} \cup \{ q _ { j } \mid j \in J \} \right) , \qquad \gamma = 1 - { \frac { q _ { \operatorname* { m i n } } } { 2 } } .
$$

Th<sub>en</sub> $\begin{array} { r } { \gamma \in ( \frac { 1 } { 5 } , 1 ) } \end{array}$ <sub>, so</sub> th<sub>e pos</sub>iti<sub>ve-coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>ga</sub>d<sub>ge</sub>t<sub>s are va</sub>lid<sub>, an</sub>d $\gamma > 1 - q _ { j }$ <sup>f</sup>or ever<sub>y</sub> $j \in J ,$ <sub>,</sub> <sub>so</sub> th<sub>e</sub> <sub>nega</sub>ti<sub>ve-coe</sub>fi<sub>c</sub>i<sub>en</sub>t <sub>ga</sub>d<sub>ge</sub>t<sub>s</sub> <sub>are</sub> <sub>va</sub>lid <sub>as</sub> <sub>we</sub>ll<sub>.</sub> C<sub>ompose</sub> <sub>a</sub>ll l<sub>oca</sub>l <sub>ga</sub>d<sub>ge</sub>t<sub>s</sub> <sub>us</sub>i<sub>ng</sub> L<sub>emma</sub> 42<sub>.</sub> Th<sub>e</sub> <sub>resu</sub>lti<sub>ng</sub> RMDP h<sub>as</sub>

$$
\begin{array} { l } { \underset { \pi \in \Pi ^ { \mathrm { M R } } } { \operatorname* { i n f } } \mathrm { R r e g } ( \pi ) = \displaystyle \sum _ { i \in I } ( \sqrt { a _ { i } } - C _ { i } ^ { + } ) + \displaystyle \sum _ { j \in J } ( D _ { j } ^ { - } - \sqrt { b _ { j } } ) } \\ { = C + \displaystyle \sum _ { i \in I } \sqrt { a _ { i } } - \displaystyle \sum _ { j \in J } \sqrt { b _ { j } } , } \end{array}
$$

<sub>w</sub>h<sub>ere</sub>

$$
C = \sum _ { j \in J } D _ { j } ^ { - } - \sum _ { i \in I } C _ { i } ^ { + } .
$$

S<sub>e</sub>t th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld $t = C - r _ { 0 }$ <sub>.</sub> Th<sub>en</sub>

$$
\begin{array} { r c l } { { \displaystyle \operatorname* { i n f } _ { \pi \in \Pi ^ { \mathrm { M R } } } \mathrm { R r e g } ( \pi ) \leq t \iff \sum _ { i \in I } \sqrt { a _ { i } } - \sum _ { j \in J } \sqrt { b _ { j } } \leq - r _ { 0 } } }  \\ { { \iff \displaystyle \sum _ { i = 1 } ^ { m } \sqrt { a _ { i } } \leq \sum _ { j = 1 } ^ { n } \sqrt { b _ { j } } . } } \end{array}
$$

Th<sub>e</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> <sub>se</sub>t i<sub>s</sub> <sub>a</sub> <sub>pro</sub>d<sub>uc</sub>t <sub>o</sub>f <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> <sub>se</sub>t<sub>s,</sub> <sub>so</sub> th<sub>e</sub> <sub>re</sub>d<sub>uc</sub>ti<sub>on</sub> <sub>preserves</sub> $( s , a )$ -rectan<sub>g</sub>u<sup>l</sup>ar<sup>i</sup>t<sub>y</sub>.

## I Proof of Theorem 8: General-Polytope Minimal-Regret Hardness

W<sub>e</sub> fi<sub>rs</sub>t <sub>prove</sub> th<sub>e ex</sub>i<sub>s</sub>t<sub>en</sub>ti<sub>a</sub>l<sub>-un</sub>i<sub>versa</sub>l <sub>upper</sub> b<sub>oun</sub>d<sub>,</sub> th<sub>en use</sub> t<sub>wo-ac</sub>ti<sub>on anc</sub>h<sub>or</sub>i<sub>ng</sub> t<sub>o</sub> t<sub>urn genera</sub>l <sub>po</sub>li<sub>cy compar</sub>i<sub>son</sub> i<sub>n</sub>t<sub>o a</sub> strictly monotone minimal-regret objective.

Theorem 8. Single-policy minimal robust regret is ∀R-hard for arbitrary rational polytopic uncertainty.

## I.1 Existential-Universal Membership

Th<sub>e</sub> <sub>mem</sub>b<sub>ers</sub>hi<sub>p</sub> <sub>c</sub>l<sub>a</sub>i<sub>m</sub> i<sub>s</sub> <sub>prove</sub>d <sub>w</sub>ith it<sub>s</sub> <sub>cer</sub>tifi<sub>ca</sub>ti<sub>on</sub> <sub>coun</sub>t<sub>erpar</sub>t i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>

## I.2 Two-Action Anchoring

St<sub>ar</sub>t <sub>w</sub>ith <sub>a</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c</sub> <sub>compar</sub>i<sub>son</sub> i<sub>ns</sub>t<sub>ance</sub> $( M , \pi _ { 1 } , \pi _ { 2 } , t )$ (Problem 5) and write

$$
\Delta = \operatorname* { s u p } _ { p } \bigl ( V _ { p } ^ { \pi _ { 1 } } ( s _ { \iota } ) - V _ { p } ^ { \pi _ { 2 } } ( s _ { \iota } ) \bigr ) , \qquad V _ { \mathrm { m a x } } = \frac { \operatorname* { m a x } _ { s , a } | R ( s , a ) | } { 1 - \gamma } .
$$

Ch<sub>oose</sub>

$$
L = 2 \gamma V _ { \mathrm { m a x } } + | \gamma t | + 1 , \qquad Z = L / \gamma + 2 V _ { \mathrm { m a x } } + 1 .
$$

E<sub>very comp</sub>li<sub>an</sub>t <sub>va</sub>l<sub>ue o</sub>f th<sub>e cons</sub>t<sub>ruc</sub>ti<sub>on</sub> b<sub>e</sub>l<sub>ow</sub> i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d i<sub>n a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e va</sub>l<sub>ue</sub> b<sub>y</sub> $M _ { \widehat { M } } = L + \gamma ( Z + V _ { \operatorname* { m a x } } ) + Z ,$ <sub>, s</sub>i<sub>nce</sub> th<sub>e cop</sub>i<sub>es</sub> <sub>con</sub>t<sub>r</sub>ib<sub>u</sub>t<sub>e a</sub>t <sub>mos</sub>t $V _ { \mathrm { m a x } }$ <sub>,</sub> th<sub>e anc</sub>h<sub>ors</sub> $\pm Z _ { i }$ , and the initial action at most L on top of those. Choose a rational $Z _ { \mathrm { r } }$ <sub>o</sub>f <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>enco</sub>di<sub>ng</sub> l<sub>eng</sub>th <sub>suc</sub>h th<sub>a</sub>t

$$
\gamma Z _ { \mathrm { r } } > M _ { \widehat { M } } + 1 ,
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> th<sub>e</sub> h<sub>ypo</sub>th<sub>es</sub>i<sub>s o</sub>f L<sub>emma</sub> 11<sub>.</sub>

![](images/4b0d188fa5d303b8c8b0fafeeed074fb9c775a6fb21f2751225fe1fa01365210.jpg)  
Fi<sub>gure</sub> 12<sub>:</sub> Th<sub>e</sub> t<sub>wo-ac</sub>ti<sub>on anc</sub>h<sub>or</sub>i<sub>ng</sub> RMDP<sub>.</sub> Th<sub>e</sub> t<sub>wo cop</sub>i<sub>es s</sub>h<sub>are</sub> th<sub>e rea</sub>li<sub>za</sub>ti<sub>on</sub> $p ,$ <sub>w</sub>h<sub>ereas</sub> $\lambda _ { \alpha }$ <sub>an</sub>d $\lambda _ { \beta }$ <sub>are</sub> f<sub>res</sub>h <sub>coor</sub>di<sub>na</sub>t<sub>es.</sub>

Definition 19 (Two-action anchoring RMDP). Given $( M , \pi _ { 1 } , \pi _ { 2 } , t )$ , the two-action anchoring RMDP is

$$
\widehat { M } = ( \widehat { S } , \widehat { A } , \widehat { \mathcal { U } } , \widehat { R } , \widehat { s } , \gamma ) ,
$$

with the following components.

• $\widehat { S }$ contains a fresh initial state ${ \widehat { s } } ,$ two disjointforced copies of M, absorbing states z<sub>−</sub> and $z _ { + }$ of values −Z and $Z ,$ and a ruinous sink ⊥<sub>r</sub> of value $- Z _ { \mathrm { r } }$

• At each state in copy i, the action prescribed by $\pi _ { i }$ is the described action. The initial state ofers actions α and $\beta .$ Every action $a t z _ { - }$ and $z _ { + }$ self-loops with the reward realizing its displayed payof.

• Ub ties corresponding transition coordinates of the forced copies to one realization p of U. It also contains mutually independent coordinates $\lambda _ { \alpha } , \lambda _ { \beta } \in [ 0 , 1 ]$ that are unconstrained by the equalities tying the copies. Action α enters the first copy with probability $\lambda _ { \alpha }$ and z otherwise. Action $\beta$ enters the second copy with probability $\lambda _ { \beta }$ and $z _ { + }$ otherwise.

• Rewards inside the forced copies agree with M. Action α pays L, action β pays zero, and the self-loop rewards at z<sub>−</sub> and $z _ { + }$ are $- ( 1 - \gamma ) Z$ and $( 1 - \gamma ) \bar { Z } ,$ , respectively. Every action at $\perp _ { \mathrm { r } }$ self-loops with reward $- ( 1 - \gamma ) Z _ { \mathrm { r } }$ . All remaining choices use ruinous-sink completion with $\bar { Z } _ { \mathrm { r } }$

• The initial state is sb and the discount is γ.

Figure 12 illustrates the construction.

Lemma 43 (Anchored action values). At a realization $\left( p , \lambda _ { \alpha } , \lambda _ { \beta } \right)$ of Definition 19, the values of α and β at sb are

$$
A = L + \gamma \big ( \lambda _ { \alpha } V _ { p } ^ { \pi _ { 1 } } ( s _ { \iota } ) - ( 1 - \lambda _ { \alpha } ) Z \big ) , \qquad B = \gamma \big ( \lambda _ { \beta } V _ { p } ^ { \pi _ { 2 } } ( s _ { \iota } ) + ( 1 - \lambda _ { \beta } ) Z \big ) .
$$

Proof. The choice of Z makes every ruinous-completed action strictly worse than the prescribed continuation at a copied state, f<sub>or</sub> b<sub>o</sub>th th<sub>e compara</sub>t<sub>or an</sub>d th<sub>e can</sub>did<sub>a</sub>t<sub>e.</sub> Th<sub>us</sub> th<sub>e</sub> t<sub>wo cop</sub>i<sub>es</sub> h<sub>ave va</sub>l<sub>ues</sub> $V _ { p } ^ { \pi _ { 1 } } ( s _ { \iota } )$ <sub>an</sub>d $V _ { p } ^ { \pi _ { 2 } } ( s _ { \iota } )$ <sub>.</sub> O<sub>ne-s</sub>t<sub>ep con</sub>diti<sub>on</sub>i<sub>ng on</sub> th<sub>e</sub> t<sub>wo successors o</sub>f <sub>eac</sub>h i<sub>n</sub>iti<sub>a</sub>l <sub>ac</sub>ti<sub>on g</sub>i<sub>ves</sub> th<sub>e</sub> f<sub>ormu</sub>l<sub>as.</sub> □

S<sub>e</sub>t $D = L + \gamma \Delta$ <sub>an</sub>d $E = 2 \gamma Z - L$

Lemma 44 (Two-action anchoring). The constants D and E are positive. A policy choosing α with probability x has robust regret max $\{ ( 1 - x ) D , x E \}$

Proof. By Lemma 43, the action-value diference is afine in each fresh coordinate, so its extrema occur at their four endpoin <sub>pa</sub>ir<sub>s.</sub> F<sub>o</sub>r $A - B$ th<sub>ese va</sub>l<sub>ues are</sub>

$$
\begin{array} { r } { \frac { \big ( \lambda _ { \alpha } , \lambda _ { \beta } \big ) } { ( 1 , 1 ) } \left. \begin{array} { c } { A - B } \\ { L + \gamma \big ( V _ { p } ^ { \pi _ { 1 } } - V _ { p } ^ { \pi _ { 2 } } \big ) } \\ { L + \gamma \big ( V _ { p } ^ { \pi _ { 1 } } - Z \big ) } \\ { L - \gamma \big ( Z + V _ { p } ^ { \pi _ { 2 } } \big ) } \\ { \big ( 0 , 0 \big ) } \end{array} \right. } \end{array}
$$

Th<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>on o</sub>f $Z$ <sub>an</sub>d $| V _ { p } ^ { \pi _ { i } } | \le V _ { \mathrm { m a x } }$ <sub>ma</sub>k<sub>e</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>en</sub>d<sub>po</sub>i<sub>n</sub>t <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> l<sub>arges</sub>t <sub>a</sub>ft<sub>er</sub> <sub>op</sub>ti<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> <sub>over</sub> $p ,$ <sub>so</sub> it<sub>s</sub> <sub>supremum</sub> i<sub>s</sub> $D .$ F<sub>or</sub> $B - A$ th<sub>e</sub> f<sub>our va</sub>l<sub>ues are respec</sub>ti<sub>ve</sub>l<sub>y</sub>

$$
\gamma ( V _ { p } ^ { \pi _ { 2 } } - V _ { p } ^ { \pi _ { 1 } } ) - L , \quad \gamma Z - L - \gamma V _ { p } ^ { \pi _ { 1 } } , \quad \gamma ( Z + V _ { p } ^ { \pi _ { 2 } } ) - L , \quad 2 \gamma Z - L .
$$

Th<sub>e</sub> l<sub>as</sub>t i<sub>s s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> l<sub>arges</sub>t <sub>an</sub>d <sub>equa</sub>l<sub>s</sub> $E > 0 .$ <sub>.</sub> M<sub>oreover,</sub> $\gamma \Delta \ge - 2 \gamma V _ { \mathrm { m a x } }$ <sub>, an</sub>d h<sub>ence</sub> $D \geq | \gamma t | + 1 > 0$ <sub>.</sub> At <sub>any rea</sub>li<sub>za</sub>ti<sub>on</sub> th<sub>e</sub> loss of the x-mixture is $( 1 - x ) ( A - B )$ ) when $A \geq B$ <sub>an</sub>d $x ( B - A )$ <sub>o</sub>th<sub>erw</sub>i<sub>se.</sub> T<sub>a</sub>ki<sub>ng</sub> th<sub>e</sub> <sub>supremum,</sub> <sub>us</sub>i<sub>ng</sub> $D , E > 0 .$ , <sub>g</sub><sup>i</sup>ves th<sub>e c</sub>l<sub>a</sub>i<sub>me</sub>d <sub>max</sub>i<sub>mum.</sub> □

Lemma 44 covers the policies that play α or $\beta$ <sub>an</sub>d th<sub>en</sub> th<sub>e</sub> <sub>prescr</sub>ib<sub>e</sub>d <sub>con</sub>ti<sub>nua</sub>ti<sub>on,</sub> <sub>w</sub>h<sub>ereas</sub> th<sub>e</sub> i<sub>n</sub>fi<sub>mum</sub> <sub>ranges</sub> <sub>over</sub> <sub>every</sub> <sub>s</sub>t<sub>a</sub>ti<sub>onary po</sub>li<sub>cy o</sub>f th<sub>e comp</sub>l<sub>e</sub>t<sub>e</sub>d RMDP<sub>, so</sub> th<sub>e</sub> t<sub>wo mus</sub>t fi<sub>rs</sub>t b<sub>e</sub> id<sub>en</sub>tifi<sub>e</sub>d<sub>.</sub> F<sub>or an ar</sub>bit<sub>rary po</sub>li<sub>cy</sub> $\pi ,$ , its compliant projection π¯ h<sub>as va</sub>l<sub>ue a</sub>t l<sub>eas</sub>t $\pi ^ { \prime } \mathbf { s }$ <sub>a</sub>t <sub>every rea</sub>li<sub>za</sub>ti<sub>on</sub> b<sub>y</sub> L<sub>emma</sub> 11<sub>,</sub> h<sub>ence regre</sub>t <sub>a</sub>t <sub>mos</sub>t $\pi ^ { \prime } \mathbf { s } .$ . Each copied state leaves π¯ only the prescribed action, so π¯ is determined by the single probability $x = \bar { \pi } ( \widehat { s } , \alpha )$ <sub>, an</sub>d th<sub>e</sub> t<sub>wo</sub> i<sub>n</sub>fi<sub>ma co</sub>i<sub>nc</sub>id<sub>e.</sub> B<sub>a</sub>l<sub>anc</sub>i<sub>ng</sub> th<sub>e</sub> t<sub>wo</sub> b<sub>ranc</sub>h<sub>es</sub> th<sub>ere</sub>f<sub>ore g</sub>i<sub>ves m</sub>i<sub>n</sub>i<sub>mum</sub> $\tilde { D E } / ( D + \check { E } )$ <sub>.</sub> L<sub>e</sub>t

$$
T = L + \gamma t , \qquad t ^ { \prime } = { \frac { E T } { E + T } } .
$$

Si<sub>nce</sub> $z \mapsto E z / ( E + z )$ i<sub>s s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> i<sub>ncreas</sub>i<sub>ng</sub> f<sub>or</sub> $z > 0 ,$

$$
\operatorname* { i n f } _ { \pi } \operatorname { R r e g } ( \pi ) \leq t ^ { \prime } \iff \Delta \leq t .
$$

E<sub>xamp</sub>l<sub>e</sub> 8 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>rema</sub>i<sub>ns</sub> <sub>ra</sub>ti<sub>ona</sub>l f<sub>or</sub> <sub>an</sub> i<sub>rra</sub>ti<sub>ona</sub>l <sub>compar</sub>i<sub>son</sub> <sub>gap.</sub>

Example 8 (Anchoring an irrational comparison gap). For the RMDP of Example $I , \gamma = 1 / 2 , V _ { \mathrm { m a x } } = 2 ,$ , and $\Delta = 7 - 4 { \sqrt { 3 } } .$ Taking $t = 0$ gives $L = 3 , Z = 1 1 , D = 1 3 / 2 - 2 { \sqrt { 3 } } , E = 8 , T = 3 ,$ , and $t ^ { \prime } = 2 4 / 1 1$ . Thus the anchoring constants and threshold remain rational even though the balanced regret $D E / ( D + E )$ depends on the irrational source gap.

ProofofTheorem 8. The equivalence above is a reduction from Theorem 20’s ∀R-hard comparison problem.

## J Proof of Theorem 9: Bounded Portfolio Synthesis

Th<sub>e re</sub>d<sub>uc</sub>ti<sub>on rea</sub>d<sub>s</sub> th<sub>e</sub> t<sub>wo quan</sub>tifi<sub>ers o</sub>f <sub>a</sub> b<sub>oun</sub>d<sub>e</sub>d <sub>rea</sub>l <sub>sen</sub>t<sub>ence as</sub> th<sub>e</sub> t<sub>wo p</sub>l<sub>ayers</sub> i<sub>n syn</sub>th<sub>es</sub>i<sub>s.</sub> Th<sub>e por</sub>tf<sub>o</sub>li<sub>o p</sub>l<sub>ays</sub> th<sub>e</sub> <sub>ex</sub>i<sub>s</sub>t<sub>en</sub>ti<sub>a</sub>l <sub>w</sub>it<sub>ness an</sub>d <sub>na</sub>t<sub>ure p</sub>l<sub>ays</sub> th<sub>e un</sub>i<sub>versa</sub>l bl<sub>oc</sub>k<sub>.</sub> Th<sub>ree o</sub>b<sub>s</sub>t<sub>ac</sub>l<sub>es</sub> h<sub>ave</sub> t<sub>o</sub> b<sub>e</sub> h<sub>an</sub>dl<sub>e</sub>d<sub>: a rewar</sub>d <sub>may no</sub>t d<sub>epen</sub>d <sub>on a</sub> <sub>po</sub>li<sub>cy or a rea</sub>li<sub>za</sub>ti<sub>on, a s</sub>t<sub>a</sub>ti<sub>onary po</sub>li<sub>cy</sub> fi<sub>xes one con</sub>ti<sub>nua</sub>ti<sub>on per s</sub>t<sub>a</sub>t<sub>e, an</sub>d dif<sub>eren</sub>t <sub>mem</sub>b<sub>ers mus</sub>t <sub>agree on</sub> th<sub>e w</sub>it<sub>ness.</sub> Th<sub>e</sub> <sub>cons</sub>t<sub>ruc</sub>ti<sub>on</sub> b<sub>e</sub>l<sub>ow reso</sub>l<sub>ves a</sub>ll th<sub>ree</sub> b<sub>y norma</sub>li<sub>z</sub>i<sub>ng</sub> th<sub>e ma</sub>t<sub>r</sub>i<sub>x</sub> i<sub>n</sub>t<sub>o</sub> t<sub>es</sub>t<sub>s</sub> th<sub>a</sub>t <sub>are a</sub>fi<sub>ne</sub> i<sub>n</sub> th<sub>e w</sub>it<sub>ness, an</sub>d b<sub>y ma</sub>ki<sub>ng na</sub>t<sub>ure</sub>’<sub>s</sub> <sub>c</sub>h<sub>o</sub>i<sub>ce o</sub>f <sub>au</sub>dit <sub>a coe</sub>fi<sub>c</sub>i<sub>en</sub>t i<sub>ns</sub>id<sub>e one po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>per mem</sub>b<sub>er ra</sub>th<sub>er</sub> th<sub>an a</sub> b<sub>ranc</sub>h i<sub>n</sub> th<sub>e mo</sub>d<sub>e</sub>l<sub>.</sub>

Theorem 9. Bounded portfolio synthesis is ∃∀R-complete under general rational polytopic uncertainty, with k encoded in unary. Hardness holds already for regret threshold two, the fixed discount $\begin{array} { r } { \gamma = \frac { 1 } { 2 } } \end{array}$ , RMDPs acyclic apart from absorbing final states, and uncertain choices with two successors.

## J.1 Existential-Universal Membership

M<sub>em</sub>b<sub>ers</sub>hi<sub>p</sub> i<sub>s</sub> <sub>prove</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> C<sub>.</sub>

## J.2 A Policy-Afine Max Normal Form

Usin<sub>g</sub> the bounded de<sub>g</sub>ree-four normal form of Definition 4, it is ∃∀R-com<sub>p</sub>lete to decide

$$
\exists x \in [ 0 , 1 ] ^ { n } \forall y \in [ 0 , 1 ] ^ { m } : \quad F ( x , y ) \geq 0
$$

<sub>a</sub>ft<sub>er</sub> th<sub>e a</sub>fi<sub>ne su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>on</sub> th<sub>a</sub>t <sub>moves</sub> th<sub>e source</sub> b<sub>ox</sub> f<sub>rom</sub> $[ - 1 , 1 ] \ \mathrm { t o } \ [ 0 , 1 ]$ <sub>.</sub> W<sub>r</sub>it<sub>e</sub>

$$
F ( x , y ) = \sum _ { \nu = 1 } ^ { N _ { \mathrm { m o n } } } c _ { \nu } \prod _ { j = 1 } ^ { d _ { \nu } } w _ { \nu , j } , \qquad d _ { \nu } \leq 4 ,
$$

w<sup>h</sup>ere ever<sub>y</sub> $w _ { \nu , j }$ i<sub>s</sub> <sub>a</sub> <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> <sub>o</sub>f $x \cup y$ , an<sup>d</sup> <sub>p</sub>ut

$$
B = \operatorname* { m a x } \left\{ 1 , \sum _ { \nu = 1 } ^ { N _ { \mathrm { m o n } } } | c _ { \nu } | \right\} .
$$

Th<sub>us</sub> $| F | \le B$ on the box, and B has polynomial encoding length. Apply Definition 4 with its optional output coordinate to the <sup>i</sup>n<sub>p</sub>ut vector $( x , y )$ . Let η consist of y and every copy, product, and output coordinate, and list all residuals as $h _ { 1 } , \ldots , h _ { s }$ <sub>.</sub> Onl<sub>y</sub> copy residuals for occurrences of x contain an existential coordinate; each does so afinely and contains exactly one.

Definition 20 (Elementary tests). For every residual $h _ { j }$ and every $\sigma \in \{ - 1 , 1 \}$ , put

$$
g _ { j , \sigma } ( x , \eta ) = \frac { o + \sigma 9 B h _ { j } } { 1 0 B } ,
$$

and enumerate the resulting $r = 2 s$ expressions as $g _ { 1 } , \ldots , g _ { r } .$

Lemma 45 (Elementary-test shape). Each test of Definition 20 satisfies:

$I . ~ g _ { i } ( x , \eta ) \in [ - 1 , 1 ]$ on the box;

2. $\begin{array} { r } { g _ { i } ( x , \eta ) = c _ { i } ( \eta ) + \sum _ { \ell } d _ { i , \ell } x _ { \ell } , } \end{array}$ , where at most one $d _ { i , \ell }$ is nonzero and that coeficient $i s \pm \frac { 9 } { 1 0 }$ ; and

## 3. $c _ { i }$ has degree at most two in η.

Proof. The first claim follows from $| o | \le B$ <sub>an</sub>d $| h _ { j } | \leq 1$ <sub>.</sub> O<sub>n</sub>l<sub>y copy res</sub>id<sub>ua</sub>l<sub>s con</sub>t<sub>a</sub>i<sub>n an ex</sub>i<sub>s</sub>t<sub>en</sub>ti<sub>a</sub>l <sub>var</sub>i<sub>a</sub>bl<sub>e, eac</sub>h <sub>con</sub>t<sub>a</sub>i<sub>ns</sub> exactl<sub>y</sub> one with coeficient −1, so its two tests have coeficient $\mp \frac { 9 } { 1 0 }$ . Every other residual and o are free of x. Copy and output residuals are afine, product residuals are quadratic in η, and o is afine in o¯. □

Lemma 46 (Policy-afine max form).

$$
\exists x \forall y : F ( x , y ) \geq 0 \quad \Longleftrightarrow \quad \exists x \forall \eta : \operatorname* { m a x } _ { i \in [ r ] } g _ { i } ( x , \eta ) \geq 0 .
$$

Proof. For the forward direction, fix a witness x and any $\eta ,$ retain its y-part, and let $\delta = \operatorname* { m a x } _ { j } | h _ { j } |$ . Choose j with $| h _ { j } | = \delta$ <sub>an</sub>d σ with $\sigma h _ { j } = \delta$ <sub>.</sub> B<sub>y</sub> L<sub>emma</sub> 12<sub>,</sub>

$$
o \ge F ( x , y ) - 9 B \delta \ge - 9 B \delta ,
$$

<sub>an</sub>d th<sub>ere</sub>f<sub>ore</sub> $g _ { j , \sigma } \geq 0$ <sub>.</sub> Thi<sub>s a</sub>l<sub>so covers</sub> $\delta = 0$

Conversely, if x is not a witness, choose y with $F ( x , y ) < 0$ <sub>.</sub> Gi<sub>ve every copy an</sub>d <sub>pro</sub>d<sub>uc</sub>t <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> it<sub>s correc</sub>t <sub>va</sub>l<sub>ue an</sub>d <sub>se</sub>t

$$
\bar { o } = \frac { F ( x , y ) + B } { 2 B } \in [ 0 , 1 ] .
$$

Th<sub>en</sub> $o = F ( x , y )$ <sub>an</sub>d <sub>every res</sub>id<sub>ua</sub>l <sub>van</sub>i<sub>s</sub>h<sub>es, so every</sub> t<sub>es</sub>t <sub>equa</sub>l<sub>s</sub> $F ( x , y ) / ( 1 0 B ) < 0$

E<sub>xamp</sub>l<sub>e</sub> 9 <sub>ma</sub>k<sub>es a</sub>ll <sub>cons</sub>t<sub>an</sub>t<sub>s</sub> i<sub>n</sub> th<sub>e norma</sub>li<sub>za</sub>ti<sub>on exp</sub>li<sub>c</sub>it<sub>.</sub>

Example 9 (A policy-afine normal form). Let $n = m = 1$ and $\begin{array} { r } { F ( x , y ) = x - \frac { 1 } { 2 } y , } \end{array}$ whose sentence is true exactly for $\begin{array} { r } { x \ge \frac { 1 } { 2 } . } \end{array}$ Here $\begin{array} { r } { N _ { \mathrm { m o n } } = 2 , c _ { 1 } = 1 , c _ { 2 } = - \frac { 1 } { 2 } } \end{array}$ , and $\begin{array} { r } { B = \frac { 3 } { 2 } } \end{array}$ . The copies are $z _ { 1 } , z _ { 2 } ,$ , with $p _ { 1 } = z _ { 1 } , p _ { 2 } = z _ { 2 }$ and no product residuals:

$$
h _ { 1 } = z _ { 1 } - x , \qquad h _ { 2 } = z _ { 2 } - y , \qquad h _ { 3 } = \frac { 1 } { 3 } \left( o - z _ { 1 } + \frac { 1 } { 2 } z _ { 2 } \right) , \qquad o = 3 \bar { o } - \frac { 3 } { 2 } .
$$

Thus $\begin{array} { r } { s = 3 , r = 6 , 9 B = \frac { 2 7 } { 2 } } \end{array}$ , and $1 0 B = 1 5$ . Only $g _ { 1 , \pm }$ mention x, with coeficients $\mp \frac { 9 } { 1 0 }$ . For $\begin{array} { r } { x = \frac { 1 } { 3 } } \end{array}$ , the reverse construction takes $\begin{array} { r } { y = 1 , z _ { 1 } = \frac { 1 } { 3 } , z _ { 2 } = 1 } \end{array}$ , and $\begin{array} { r } { \bar { o } = \frac { 4 } { 9 } . } \end{array}$ . Every residual then vanishes and every test equals $( x - { \textstyle \frac { 1 } { 2 } } ) / 1 5 < 0 .$

## J.3 Exact Evaluators

B<sub>o</sub>th <sub>componen</sub>t<sub>s use</sub> th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l<sub>-payo</sub>f <sub>conven</sub>ti<sub>on o</sub>f A<sub>ppen</sub>di<sub>x</sub> A<sub>.</sub> Th<sub>roug</sub>h<sub>ou</sub>t thi<sub>s sec</sub>ti<sub>on,</sub> $\begin{array} { r } { \gamma = \frac { 1 } { 2 } } \end{array}$ <sub>.</sub> F<sub>o</sub>r <sub>a spa</sub>r<sub>se po</sub>l<sub>y</sub>n<sub>o</sub>mi<sub>a</sub>l $P ,$ <sub>, wr</sub>it<sub>e</sub> ${ \bar { \mathrm { P o l y } } } ( P )$ f<sub>or</sub> th<sub>e componen</sub>t <sub>o</sub>f D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 8 <sub>a</sub>t thi<sub>s</sub> di<sub>scoun</sub>t<sub>, w</sub>ith <sub>repea</sub>t<sub>e</sub>d l<sub>og</sub>i<sub>ca</sub>l <sub>coor</sub>di<sub>na</sub>t<sub>es</sub> ti<sub>e</sub>d l<sub>a</sub>t<sub>er</sub> b<sub>y</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 24<sub>,</sub> <sub>an</sub>d <sub>wr</sub>it<sub>e</sub> $e _ { P }$ f<sub>or</sub> it<sub>s en</sub>t<sub>ry.</sub> L<sub>emma</sub> 22 <sub>g</sub>i<sub>ves</sub> it<sub>s exac</sub>t <sub>comp</sub>li<sub>an</sub>t <sub>va</sub>l<sub>ue an</sub>d it<sub>s s</sub>t<sub>ruc</sub>t<sub>ura</sub>l b<sub>oun</sub>d<sub>s.</sub>

Definition 21 (Afine-policy evaluator). For

$$
G ( x , u ) = P _ { 0 } ( u ) + \sum _ { \ell = 1 } ^ { n } x _ { \ell } P _ { \ell } ( u ) ,
$$

with every $P _ { \ell }$ explicit, sparse, and ofconstant degree, the component $\operatorname { A f f } ( G )$ has:

• an entry e<sub>G</sub>; coordinate states $s _ { 1 } , \ldots , s _ { n } ,$ , each with actions on and of; a zero-payof terminal zr; and copies of

$$
\mathrm { P o l y } \left( \frac { n + 1 } { \gamma } P _ { 0 } \right) \quad a n d \quad \mathrm { P o l y } \left( \frac { n + 1 } { \gamma ^ { 2 } } P _ { \ell } \right) \quad ( \ell \in [ n ] ) ;
$$

• at $e _ { G } ,$ one action with the certain uniform splitter over the $n + 1$ successors $e _ { P _ { 0 } } , s _ { 1 } , \ldots , s _ { n } ;$ at $s _ { \ell } ,$ a reward-zero singleton transition from on to $e _ { P _ { \ell } }$ and from of to zr;

• the rewards and completion of its Poly copies; every other choice of this component uses ruinous-sink completion.

Lemma 47 (Afine-policy evaluation). For every realization u and every compliant $\pi \in \Pi ^ { \mathrm { M R } }$

$$
V _ { u } ^ { \pi } ( e _ { G } ) = P _ { 0 } ( u ) + \sum _ { \ell = 1 } ^ { n } x _ { \ell } P _ { \ell } ( u ) , \qquad x _ { \ell } = \pi ( s _ { \ell } , \mathsf { o n } ) .
$$

Proof. The constant branch contributes

$$
\frac { 1 } { n + 1 } \gamma \frac { n + 1 } { \gamma } P _ { 0 } ( u ) = P _ { 0 } ( u ) .
$$

At $s _ { \ell } ,$

$$
V _ { u } ^ { \pi } ( s _ { \ell } ) = x _ { \ell } \gamma \frac { n + 1 } { \gamma ^ { 2 } } P _ { \ell } ( u ) + ( 1 - x _ { \ell } ) \gamma \cdot 0 = x _ { \ell } \frac { n + 1 } { \gamma } P _ { \ell } ( u ) ,
$$

so branch ℓ contributes $\begin{array} { r } { \frac { \gamma } { n + 1 } x _ { \ell } \frac { n + 1 } { \gamma } P _ { \ell } ( u ) = x _ { \ell } P _ { \ell } ( u ) } \end{array}$ <sub>.</sub> S<sub>umm</sub>i<sub>ng</sub> <sub>proves</sub> th<sub>e</sub> id<sub>en</sub>tit<sub>y.</sub>

![](images/93c37a9b4759c5c99db7bace2f2443aabf81bdc6d99e5d5cbeaeddd992fe06bd.jpg)  
Fi<sub>gure</sub> 13<sub>:</sub> Th<sub>e</sub> <sub>syn</sub>th<sub>es</sub>i<sub>s</sub> RMDP <sub>an</sub>d it<sub>s</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>or</sub> <sub>componen</sub>t<sub>s.</sub> Th<sub>e</sub> <sub>on</sub>l<sub>y</sub> <sub>po</sub>li<sub>cy</sub> <sub>c</sub>h<sub>o</sub>i<sub>ces</sub> <sub>are</sub> <sub>a</sub>t $\boldsymbol { s } _ { \iota }$ <sub>an</sub>d th<sub>e</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>es;</sub> <sub>every</sub> <sub>rewar</sub>d <sub>s</sub>h<sub>own</sub> i<sub>s a</sub> fi<sub>xe</sub>d <sub>ra</sub>ti<sub>ona</sub>l<sub>.</sub>

Lemma 48 (One predecessor per coordinate). In $\operatorname { A f f } ( G )$ every coordinate state $s _ { \ell }$ has exactly one predecessor, namely $e _ { G } ,$ and each ofits two actions has onefixed continuation. Thus a stationary policy’s behavior at $s _ { \ell }$ has a single interpretation, and no continuation depends on the contextfrom which $s _ { \ell }$ was reached.

Proof. The certain uniform splitter at $_ { e _ { G } }$ i<sub>s</sub> th<sub>e</sub> <sub>on</sub>l<sub>y</sub> t<sub>rans</sub>iti<sub>on</sub> i<sub>n</sub>t<sub>o</sub> $s _ { \ell } ,$ while on and of always lead to $e _ { P _ { \ell } }$ and zr, respectively.

E<sub>very occurrence ge</sub>t<sub>s a</sub> f<sub>res</sub>h <sub>row, an</sub>d th<sub>e on</sub>l<sub>y po</sub>li<sub>cy c</sub>h<sub>o</sub>i<sub>ces are a</sub>t th<sub>e coor</sub>di<sub>na</sub>t<sub>e s</sub>t<sub>a</sub>t<sub>es.</sub> Th<sub>us</sub> th<sub>e eva</sub>l<sub>ua</sub>t<sub>or</sub> h<sub>as ne</sub>ith<sub>er</sub> <sub>s</sub>h<sub>are</sub>d <sub>se</sub>l<sub>ec</sub>t<sub>or</sub> d<sub>eco</sub>d<sub>er ac</sub>ti<sub>ons nor con</sub>t<sub>ex</sub>t<sub>-</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>con</sub>ti<sub>nua</sub>ti<sub>ons.</sub> Fi<sub>gure</sub> 13 <sub>s</sub>h<sub>ows</sub> th<sub>e nes</sub>t<sub>e</sub>d <sub>eva</sub>l<sub>ua</sub>t<sub>or s</sub>t<sub>ruc</sub>t<sub>ure.</sub>

## J.4 Roles, Cases, and Scores

Definition 22 (Portfolio case scores). Let r be the number of tests and also the portfolio budget. Put

$$
\begin{array} { r } { \mathcal { H } = \{ \mathrm { e v } \} \cup \{ ( i , \ell , \right. ) , ( i , \ell , \left. ) : 2 \leq i \leq r , \ell \in [ n ] \} \cup \{ ( i , t ) : i \in [ r ] , t \in [ 3 r + 1 ] \} . } \end{array}
$$

Nature’s coordinates are a case distribution $\lambda \in { \mathcal { D } } ( { \mathcal { H } } )$ , the evaluation vector $\eta ,$ and one coordinate $q _ { h } \in [ 0 , 1 ]$ for every oriented equality case. Write $\xi _ { h }$ for the coordinates local to case $h .$ For role i, define the pure-case score $A _ { h , i } ( x _ { i } , \xi _ { h } )$ asfollows.

• In the evaluation case, $A _ { \mathrm { e v } , i } ( x _ { i } , \eta ) = g _ { i } ( x _ { i } , \eta )$

• For $h = ( i , \ell ,  ) ,$

$$
A _ { h , 1 } = x _ { 1 , \ell } - q _ { h } , \quad \quad A _ { h , i } = q _ { h } - x _ { i , \ell } ,
$$

and $A _ { h , j } = - 1 f o r j \notin \{ 1 , i \}$

• For $h = ( i , \ell ,  ) ,$

$$
A _ { h , 1 } = q _ { h } - x _ { 1 , \ell } , \qquad A _ { h , i } = x _ { i , \ell } - q _ { h } ,
$$

and $A _ { h , j } = - 1 f o r j \notin \{ 1 , i \} .$

• For $h = ( i , t ) , s e t A _ { h , i } = 0 a n d A _ { h , j } = - 1 f o r j \neq i .$

Lemma 49 (Pure-score shape). Every pure-case score of Definition 22 lies in $[ - 1 , 1 ]$ and has the form

$$
A _ { h , i } ( x _ { i } , \xi _ { h } ) = c _ { h , i } ( \xi _ { h } ) + \sum _ { \ell } d _ { h , i , \ell } x _ { i , \ell } ,
$$

where $c _ { h , i }$ has degree at most two in nature’s coordinates and every $d _ { h , i , \ell }$ is rational.

Proof. The evaluation scores have the claimed range and shape by Lemma 45. Equality scores are diferences of two numbers in [0, 1], with $c _ { h , \cdot } = \mp q _ { h }$ <sub>an</sub>d $d _ { h , \cdot } = \pm 1$ <sub>.</sub> F<sub>o</sub>r<sub>c</sub>in<sub>g sco</sub>r<sub>es a</sub>r<sub>e co</sub>n<sub>s</sub>t<sub>a</sub>nt<sub>s.</sub> □

Define the mixed score of role i by

$$
W _ { i } = \sum _ { h \in \mathcal { H } } \lambda _ { h } A _ { h , i } ( x _ { i } , \xi _ { h } ) + 1 - \sum _ { h \in \mathcal { H } } \lambda _ { h } ^ { 2 } .\tag{5}
$$

Lemma 50 (Mixed-score polynomial). The score $W _ { i }$ has the form

$$
W _ { i } = P _ { i , 0 } ( \lambda , \eta , q ) + \sum _ { \ell } x _ { i , \ell } P _ { i , \ell } ( \lambda ) ,
$$

where

$$
P _ { i , 0 } = 1 - \sum _ { h } \lambda _ { h } ^ { 2 } + \sum _ { h } \lambda _ { h } c _ { h , i } ( \xi _ { h } ) , \qquad P _ { i , \ell } = \sum _ { h } \lambda _ { h } d _ { h , i , \ell } .
$$

The first polynomial has degree at most three, the second has degree one, and both have polynomially many monomials. Consequently, Lemma 47 applies to $W _ { i }$

Proof. Substitute Lemma 49 into (5) and collect coeficients. The largest degree is $1 + 2 ,$ <sub>,</sub> f<sub>rom</sub> $\lambda _ { h }$ <sub>mu</sub>lti<sub>p</sub>l<sub>y</sub>i<sub>ng</sub> <sub>a</sub> <sub>qua</sub>d<sub>ra</sub>ti<sub>c</sub> $c _ { h , i } .$ Th<sub>e num</sub>b<sub>er o</sub>f <sub>monom</sub>i<sub>a</sub>l<sub>s</sub> i<sub>s</sub> b<sub>oun</sub>d<sub>e</sub>d b<sub>y</sub> th<sub>e sum o</sub>f th<sub>e exp</sub>li<sub>c</sub>it <sub>monom</sub>i<sub>a</sub>l <sub>coun</sub>t<sub>s o</sub>f th<sub>e pure scores p</sub>l<sub>us</sub> $\dot { 1 } + | \mathcal { H } |$ 口

Th<sub>e cases are coor</sub>di<sub>na</sub>t<sub>es o</sub>f <sub>na</sub>t<sub>ure</sub> i<sub>ns</sub>id<sub>e</sub> th<sub>ese po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>s, no</sub>t b<sub>ranc</sub>h<sub>es</sub> i<sub>n</sub> th<sub>e mo</sub>d<sub>e</sub>l<sub>.</sub> Thi<sub>s</sub> l<sub>e</sub>t<sub>s one eva</sub>l<sub>ua</sub>t<sub>or compu</sub>t<sub>e a</sub>ll <sub>o</sub>f $W _ { i }$ <sub>w</sub>hil<sub>e</sub> <sub>every</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>re</sub>t<sub>a</sub>i<sub>ns</sub> th<sub>e</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> <sub>pre</sub>d<sub>ecessor</sub> <sub>guaran</sub>t<sub>ee</sub>d b<sub>y</sub> L<sub>emma</sub> 48<sub>.</sub>

## J.5 The Synthesis RMDP

Definition 23 (Portfolio-synthesis RMDP). Let $M _ { \mathrm { s y n } } = ( S , A , \mathcal { U } , R , s _ { \iota } , \frac { 1 } { 2 } )$ be defined as follows.

• Its states are $s _ { \iota } ;$ for every role $i \in [ r ]$ , a copy $E _ { i } o f \mathrm { A f f } \left( \gamma ^ { - 1 } W _ { i } \right)$ with coordinate states $s _ { i , 1 } , \ldots , s _ { i , n } ; f o r$ every $h \in \mathcal { H } ,$ a copy $U _ { h } o f \mathrm { P o l y } ( \gamma ^ { - 1 } ( 3 \lambda _ { h } - 1 ) ) ;$ ; and a ruinous sink.

• At $s _ { \iota } ,$ , the role action $a _ { i }$ has reward zero and its singleton row enters $E _ { i } .$ . The auxiliary action $b _ { h }$ has reward zero and its singleton row enters $U _ { h } .$ . All other described transitions are those ofthe two componentfamilies.

• The uncertainty set is given in Definition 24.

• Rewards are those of the components. Every choice still undescribed after the global action set has been installed uses ruinous-sink completion.

• The initial state is $\boldsymbol { s } _ { \iota }$ and $\begin{array} { r } { \gamma = \frac { 1 } { 2 } . } \end{array}$

Definition 24 (Portfolio uncertainty). Every occurrence of a logical nature coordinate inside a component is represented by a fresh two-successor row. For every $\lambda _ { h } ,$ every coordinate of η, and every $q _ { h } ,$ , designate one occurrence as its representative and impose a rational linear equality setting every other occurrence’s success probability equal to that representative. On the λ representatives add $\lambda _ { h } \geq 0$ and $\dot { \Sigma _ { h } } \lambda _ { h } = 1$ , together with the usual stochasticity constraints on all rows.

Lemma 51 (Uncertainty shape). The set U is a nonempty rational polytope. Every uncertain row has two successors, and the only other stochastic rows are the certain uniform splitters at component entries. The uncertainty is nonrectangular only through the tying equalities and the simplex constraint.

Proof. Choose any λ in the simplex and arbitrary values in [0, 1] for the other representatives, then propagate them through th<sub>e</sub> t<sub>y</sub>i<sub>ng equa</sub>liti<sub>es, w</sub>hi<sub>c</sub>h <sub>w</sub>it<sub>nesses nonemp</sub>ti<sub>ness.</sub> It<sub>s</sub> h<sub>a</sub>lf<sub>-space</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>on cons</sub>i<sub>s</sub>t<sub>s o</sub>f <sub>exac</sub>tl<sub>y</sub> f<sub>our</sub> f<sub>am</sub>ili<sub>es, a</sub>ll li<sub>s</sub>t<sub>e</sub>d i<sub>n</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>on</sub> 24<sub>: one equa</sub>lit<sub>y per nonrepresen</sub>t<sub>a</sub>ti<sub>ve occurrence</sub> t<sub>y</sub>i<sub>ng</sub> it t<sub>o</sub> it<sub>s represen</sub>t<sub>a</sub>ti<sub>ve,</sub> th<sub>e</sub> b<sub>oun</sub>d<sub>s</sub> $\lambda _ { h } \geq 0$ <sub>,</sub> th<sub>e equa</sub>lit<sub>y</sub> $\textstyle \sum _ { h } \lambda _ { h } = 1$ <sub>, an</sub>d <sub>nonnega</sub>ti<sub>v</sub>it<sub>y w</sub>ith <sub>norma</sub>li<sub>za</sub>ti<sub>on on every row.</sub> E<sub>very coe</sub>fi<sub>c</sub>i<sub>en</sub>t i<sub>s</sub> $0 ~ \mathrm { o r } \pm 1$ <sub>, an</sub>d th<sub>e num</sub>b<sub>er o</sub>f <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> i<sub>s</sub> li<sub>near</sub> i<sub>n</sub> th<sub>e num</sub>b<sub>er o</sub>f <sub>rows, so</sub> th<sub>e</sub> d<sub>escr</sub>i<sub>p</sub>ti<sub>on</sub> h<sub>as po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>enco</sub>di<sub>ng</sub> l<sub>eng</sub>th <sub>an</sub>d th<sub>e re</sub>d<sub>uc</sub>ti<sub>on can wr</sub>it<sub>e</sub> it d<sub>own.</sub> Th<sub>e</sub> <sub>rema</sub>i<sub>n</sub>i<sub>ng c</sub>l<sub>a</sub>i<sub>ms</sub> f<sub>o</sub>ll<sub>ow</sub> f<sub>rom</sub> D<sub>e</sub>fi<sub>n</sub>iti<sub>ons</sub> 8<sub>,</sub> 21 <sub>an</sub>d 24<sub>.</sub> □

## J.6 Action Values and the Optimal-Value Bound

Lemma 52 (Role-action values). In the RMDP of Definition 23, for every realization and every compliant $\pi \in \Pi ^ { \mathrm { M R } }$

$$
Q _ { u } ^ { \pi } ( s _ { \iota } , a _ { i } ) = W _ { i } ( x _ { i } , \lambda , \eta , q ) , \qquad x _ { i , \ell } = \pi ( s _ { i , \ell } , \mathsf { o n } ) .
$$

Proof. The initial action makes one reward-zero transition into $E _ { i }$ <sub>,</sub> <sub>w</sub>h<sub>ose</sub> <sub>va</sub>l<sub>ue</sub> i<sub>s</sub> $\gamma ^ { - 1 } W _ { i }$ b<sub>y</sub> L<sub>emmas</sub> 47 <sub>an</sub>d 50<sub>.</sub>

Lemma 53 (Auxiliary-action values). For every realization and every compliant policy,

$$
Q _ { u } ( s _ { \iota } , b _ { h } ) = 3 \lambda _ { h } - 1 .
$$

At a pure case $\lambda = \delta _ { h ^ { * } }$ , this value is two for $h = h ^ { * }$ and −1 otherwise.

Proof. Apply Lemma 22 to $\cdot \gamma ^ { - 1 } ( 3 \lambda _ { h } - 1 )$ <sub>an</sub>d i<sub>nc</sub>l<sub>u</sub>d<sub>e</sub> th<sub>e</sub> i<sub>n</sub>iti<sub>a</sub>l di<sub>scoun</sub>t<sub>.</sub>

Lemma 54 (Optimal-value bound). For every realization, $V _ { u } ^ { * } ( s _ { \iota } ) \leq 2 ,$ , with equality whenever $\lambda = \delta _ { h }$ is a pure case.

Proof. The only policy choices at described states are at $s _ { \iota }$ <sub>an</sub>d <sub>a</sub>t <sub>coor</sub>di<sub>na</sub>t<sub>e s</sub>t<sub>a</sub>t<sub>es.</sub> Th<sub>e ac</sub>ti<sub>on va</sub>l<sub>ues are</sub> th<sub>ose o</sub>f L<sub>emmas</sub> 52 <sub>an</sub>d 53<sub>.</sub> Si<sub>nce every pure score</sub> i<sub>s a</sub>t <sub>mos</sub>t <sub>one,</sub>

$$
W _ { i } \leq \sum _ { h } \lambda _ { h } + 1 - \sum _ { h } \lambda _ { h } ^ { 2 } = 2 - \sum _ { h } \lambda _ { h } ^ { 2 } \leq 2 ,
$$

<sub>an</sub>d $3 \lambda _ { h } - 1 \le 2$ <sub>.</sub> B L<sub>emma</sub> 11<sub>,</sub> th<sub>ese u er</sub> b<sub>oun</sub>d<sub>s ex</sub>t<sub>en</sub>d f<sub>rom com</sub> li<sub>an</sub>t <sub>o</sub>li<sub>c</sub>i<sub>es</sub> t<sub>o a</sub>ll <sub>o</sub>li<sub>c</sub>i<sub>es.</sub> A <sub>ran</sub>d<sub>om</sub>i<sub>ze</sub>d <sub>ac</sub>ti<sub>on a</sub>t $s _ { \iota }$ <sub>g</sub>i<sub>ves a convex com</sub>bi<sub>na</sub>ti<sub>on o</sub>f th<sub>ese va</sub>l<sub>ues, so</sub> it <sub>canno</sub>t <sub>excee</sub>d t<sub>wo.</sub> $\mathrm { A t } \lambda = \delta _ { h }$ <sub>, ac</sub>ti<sub>on</sub> $b _ { h }$ <sub>a</sub>tt<sub>a</sub>i<sub>ns</sub> t<sub>wo.</sub> □

Lemma 55 (Combining pure cases). For fixed coordinate vectors $x _ { 1 } , \ldots , x _ { r } ,$ , the following are equivalent:

1. for every pure case h and every valuation $o f \xi _ { h } ,$ , some role has $A _ { h , i } \geq 0 ,$

2. for every $\lambda \in { \mathcal { D } } ( { \mathcal { H } } )$ and every valuation ofall coordinates, some role has $W _ { i } \geq 0 .$

Proof. The second claim implies the first by taking $\lambda = \delta _ { h }$ <sub>,</sub> <sub>w</sub>h<sub>en</sub> $W _ { i } = A _ { h , i }$ <sub>.</sub> F<sub>or</sub> th<sub>e converse,</sub> fi<sub>x a rea</sub>li<sub>za</sub>ti<sub>on an</sub>d t<sub>a</sub>k<sub>e</sub> $h ^ { * } \in \arg \operatorname* { m a x } _ { h } \lambda _ { h }$ . <sup>A</sup><sub>pp</sub><sup>l</sup><sub>y</sub> t<sup>h</sup>e <sub>p</sub>ure-case <sup>h</sup><sub>yp</sub>ot<sup>h</sup>es<sup>i</sup>s at $h ^ { * }$ <sub>w</sub>ith th<sub>e rea</sub>li<sub>za</sub>ti<sub>on</sub>’<sub>s own</sub> $\xi _ { h } .$ ∗ to obtain a role i with $A _ { h ^ { * } , i } \geq 0$ <sub>.</sub> All other <sub>p</sub>ure scores are at least −1, so

$$
\sum _ { h } { \lambda _ { h } A _ { h , i } } \ge - ( 1 - { \lambda _ { h ^ { * } } } ) .
$$

M<sub>oreover,</sub>

$$
\sum _ { h } \lambda _ { h } ^ { 2 } \leq \lambda _ { h ^ { * } } \sum _ { h } \lambda _ { h } = \lambda _ { h ^ { * } } .
$$

Substitution in (5) <sub>g</sub>ives

$$
W _ { i } \geq - ( 1 - \lambda _ { h ^ { * } } ) + 1 - \lambda _ { h ^ { * } } = 0 .
$$

E<sub>ac</sub>h <sub>or</sub>i<sub>en</sub>t<sub>e</sub>d <sub>equa</sub>lit<sub>y case nee</sub>d<sub>s</sub> it<sub>s own coor</sub>di<sub>na</sub>t<sub>e</sub> $q _ { h } \colon$ th<sub>e ro</sub>l<sub>e se</sub>l<sub>ec</sub>t<sub>e</sub>d f<sub>or</sub> $h ^ { * }$ d<sub>epen</sub>d<sub>s on</sub> $\xi _ { h ^ { * } }$ <sub>, w</sub>hi<sub>c</sub>h <sub>mus</sub>t <sub>rema</sub>i<sub>n</sub> f<sub>ree o</sub>f <sub>every o</sub>th<sub>er case</sub>’<sub>s coor</sub>di<sub>na</sub>t<sub>es.</sub> E<sub>xamp</sub>l<sub>e</sub> 10 <sub>s</sub>h<sub>ows</sub> th<sub>a</sub>t th<sub>e correc</sub>ti<sub>on</sub> b<sub>oun</sub>d <sub>can</sub> b<sub>e</sub> ti<sub>g</sub>ht<sub>.</sub>

Example 10 (The mixing correction). $I f \lambda$ is uniform on two cases, then $\begin{array} { r } { 1 - \sum _ { h } \lambda _ { h } ^ { 2 } = \frac { 1 } { 2 } } \end{array}$ . If a role has score zero at one of those cases and −1 at the other, its mixed score is

$$
{ \frac { 1 } { 2 } } \cdot 0 + { \frac { 1 } { 2 } } \cdot ( - 1 ) + { \frac { 1 } { 2 } } = 0 .
$$

Thus the combining bound is tight.

## J.7 Correctness

Lemma 56 (Forward direction). If the source sentence holds with witness $x ,$ then the portfolio $\Pi = \{ \pi _ { 1 } , \ldots , \pi _ { r } \}$ defined by

$$
\pi _ { i } ( s _ { \iota } ) = a _ { i } , \qquad \pi _ { i } ( s _ { j , \ell } , \mathsf { o n } ) = x _ { \ell } \quad f o r e \nu e r y j , \ell
$$

and completed compliantly at every other state satisfies $\mathrm { R r e g } ( \Pi ) \leq 2 .$

Proof. Every $\pi _ { i }$ i<sub>s s</sub>t<sub>a</sub>ti<sub>onary an</sub>d $V _ { u } ^ { \pi _ { i } } ( s _ { \iota } ) = W _ { i } ( x )$ b<sub>y</sub> L<sub>e</sub>mm<sub>a</sub> 52<sub>.</sub> W<sub>e</sub> <sub>ve</sub>rif<sub>y</sub> <sub>pu</sub>r<sub>e-case</sub> <sub>cove</sub>r<sub>age.</sub> $\mathbf { A t \thinspace e v . }$ <sub>,</sub> L<sub>emma</sub> 46 <sub>g</sub>i<sub>ves</sub> max<sub>i</sub> $g _ { i } ( x , \eta ) \geq 0$ for every η. At an oriented equality case, the two nonconstant scores have maximum

$$
\operatorname* { m a x } \{ x _ { \ell } - q _ { h } , q _ { h } - x _ { \ell } \} = | x _ { \ell } - q _ { h } | \geq 0 .
$$

At <sub>a</sub> f<sub>orc</sub>i<sub>ng case</sub> $( i , t )$ , role i scores zero. By Lemma 55, every realization has some $W _ { i } \geq 0$ <sub>.</sub> C<sub>om</sub>bi<sub>n</sub>i<sub>ng</sub> thi<sub>s w</sub>ith L<sub>emma</sub> 54 b<sub>oun</sub>d<sub>s regre</sub>t b<sub>y</sub> t<sub>wo.</sub> □

L<sub>emma</sub> 29 i<sub>s nee</sub>d<sub>e</sub>d h<sub>ere</sub> b<sub>ecause</sub> th<sub>e reverse</sub> di<sub>rec</sub>ti<sub>on</sub> t<sub>urns</sub> $\rho _ { r } \leq 2$ into a concrete <sub>p</sub>ortfolio. Its h<sub>yp</sub>otheses hold: U is a <sub>compac</sub>t <sub>ra</sub>ti<sub>ona</sub>l <sub>po</sub>l<sub>y</sub>t<sub>ope</sub> b<sub>y</sub> L<sub>emma</sub> 51<sub>.</sub> Th<sub>e o</sub>th<sub>er syn</sub>th<sub>es</sub>i<sub>s re</sub>d<sub>uc</sub>ti<sub>ons compu</sub>t<sub>e</sub> th<sub>e</sub>i<sub>r</sub> i<sub>n</sub>fi<sub>mum</sub> i<sub>n c</sub>l<sub>ose</sub>d f<sub>orm.</sub>

Lemma $^ { 5 7 }$ (Role forcing). If Π has at most r members and $\mathrm { R r e g } ( \Pi ) \leq 2 ,$ , then it has exactly r members, exactly one member chooses each $a _ { i }$ with probability one, and no member chooses an auxiliary action at $s _ { \iota }$

Proof. At a pure case, Lemma 54 gives $V _ { u } ^ { * } ( s _ { \iota } ) = 2$ <sub>.</sub> Th<sub>us</sub> th<sub>e regre</sub>t b<sub>oun</sub>d i<sub>mp</sub>li<sub>es</sub>

$$
\operatorname* { m a x } _ { \pi \in \Pi } V _ { u } ^ { \pi } ( s _ { \iota } ) \geq 0
$$

<sub>a</sub>t <sub>every pure case an</sub>d <sub>every va</sub>l<sub>ua</sub>ti<sub>on o</sub>f it<sub>s coor</sub>di<sub>na</sub>t<sub>es.</sub>

A <sub>mem</sub>b<sub>er nee</sub>d <sub>no</sub>t b<sub>e comp</sub>li<sub>an</sub>t<sub>:</sub> b<sub>es</sub>id<sub>es</sub> it<sub>s</sub> i<sub>n</sub>iti<sub>a</sub>l <sub>c</sub>h<sub>o</sub>i<sub>ce</sub> it <sub>may p</sub>l<sub>ay a g</sub>l<sub>o</sub>b<sub>a</sub>ll<sub>y</sub> i<sub>ns</sub>t<sub>a</sub>ll<sub>e</sub>d <sub>ac</sub>ti<sub>on</sub> i<sub>ns</sub>id<sub>e a componen</sub>t<sub>, w</sub>h<sub>ere</sub> the action values below do not apply. Replace each member π by ${ \bar { \pi } } ,$ <sub>w</sub>hi<sub>c</sub>h L<sub>emma</sub> 11 <sub>s</sub>h<sub>ows</sub> h<sub>as</sub> <sub>va</sub>l<sub>ue</sub> <sub>a</sub>t l<sub>eas</sub>t π’s $\pi ^ { \prime } \boldsymbol { \mathrm { s } }$ <sup>at</sup> <sup>ever</sup>y state and realization, so the displayed bound is only strengthened. The projection keeps the initial distribution over $\{ a _ { i } , b _ { h } \}$ <sub>an</sub>d <sub>renorma</sub>li<sub>zes a</sub>t <sub>eac</sub>h <sub>coor</sub>di<sub>na</sub>t<sub>e s</sub>t<sub>a</sub>t<sub>e, w</sub>hi<sub>c</sub>h l<sub>eaves</sub> $\pi ( s _ { i , \ell } , \mathsf { o n } )$ in [0, 1], so π¯ still encodes a witness and the count below is <sub>una</sub>f<sub>ec</sub>t<sub>e</sub>d<sub>.</sub>

Fix a role i and a forcing case $h = ( i , t ) . \mathrm { A t } \lambda = \delta _ { h }$ <sub>, ac</sub>ti<sub>on</sub> $a _ { i }$ has value zero, ever<sub>y</sub> other role action has value −1, action $b _ { h }$ has value two, ever<sub>y</sub> other auxiliar<sub>y</sub> action has value −1, and ever<sub>y</sub> com<sub>p</sub>leted action has value at most −1. For a <sub>p</sub>olic<sub>y</sub> with i<sub>n</sub>iti<sub>a</sub>l di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> $p ,$ <sub>co</sub>ll<sub>ec</sub>t th<sub>e mass ou</sub>t<sub>s</sub>id<sub>e</sub> $\{ a _ { i } , b _ { h } \}$ t<sub>o o</sub>bt<sub>a</sub>i<sub>n</sub>

$$
V _ { u } ^ { \pi } ( s _ { \iota } ) \leq p ( a _ { i } ) + 3 p ( b _ { h } ) - 1 .
$$

<sup>S</sup>upp<sup>ose</sup> $p ( a _ { i } ) < 1$ an<sup>d</sup> <sub>p</sub>ut $\delta _ { \pi } = 1 - p ( a _ { i } ) > 0$ . Covering h forces $p ( b _ { h } ) \ge \delta _ { \pi } / 3$ <sub>.</sub> Th<sub>e</sub> <sub>ac</sub>ti<sub>ons</sub> $b _ { ( i , t ) }$ <sub>are</sub> di<sub>s</sub>ti<sub>nc</sub>t <sub>an</sub>d <sub>ou</sub>t<sub>s</sub>id<sub>e</sub> $a _ { i } .$ so

$$
\sum _ { t = 1 } ^ { 3 r + 1 } p ( b _ { ( i , t ) } ) \leq \delta _ { \pi } .
$$

Thi<sub>s po</sub>li<sub>cy can</sub> th<sub>ere</sub>f<sub>ore cover a</sub>t <sub>mos</sub>t th<sub>ree o</sub>f th<sub>e</sub> $3 r + 1$ forcing cases for role i. If every portfolio member had $p ( a _ { i } ) < 1$ <sub>,</sub> th<sub>e</sub> at most r members would cover at most 3r such cases, a contradiction. Thus some member has $p ( a _ { i } ) = 1$

N<sub>o</sub> <sub>po</sub>li<sub>cy</sub> <sub>can</sub> <sub>pu</sub>t <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>one</sub> <sub>on</sub> b<sub>o</sub>th $a _ { i }$ <sub>an</sub>d $a _ { j }$ f<sub>or</sub> $i \neq j .$ . The r roles therefore require r distinct members, exhausting th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub> E<sub>very mem</sub>b<sub>er</sub> i<sub>s pure on</sub> it<sub>s ro</sub>l<sub>e ac</sub>ti<sub>on a</sub>t $s _ { \iota }$ <sub>,</sub> l<sub>eav</sub>i<sub>ng none on an aux</sub>ili<sub>ary ac</sub>ti<sub>on.</sub> □

Lemma 58 (Equality cases). Under the hypotheses ofLemma $^ { 5 7 , }$ let $x _ { i }$ be the vector encoded by the member assigned to role i. Then $x _ { 1 } = \cdot \cdot \cdot = x _ { r }$

Proof. Fix $i \geq 2$ and ℓ. At pure case $( i , \ell , \to )$ , covera<sub>g</sub>e <sup>f</sup>or a <sub>g</sub><sup>i</sup>ven $q \in [ 0 , 1 ]$ i<sub>s</sub>

$$
\operatorname* { m a x } \{ x _ { 1 , \ell } - q , q - x _ { i , \ell } , - 1 \} \geq 0 ,
$$

<sub>equ</sub>i<sub>va</sub>l<sub>en</sub>tl<sub>y</sub>

$$
\operatorname* { m a x } \{ x _ { 1 , \ell } - q , q - x _ { i , \ell } \} \geq 0 .
$$

If $x _ { 1 , \ell } < x _ { i , \ell } ,$ th<sub>e</sub> l<sub>ega</sub>l <sub>m</sub>id<sub>po</sub>i<sub>n</sub>t $\begin{array} { r } { q = \frac { 1 } { 2 } ( x _ { 1 , \ell } + x _ { i , \ell } ) } \end{array}$ <sub>ma</sub>k<sub>es</sub> b<sub>o</sub>th t<sub>erms nega</sub>ti<sub>ve.</sub> C<sub>onverse</sub>l<sub>y,</sub> if $x _ { 1 , \ell } \geq x _ { i , \ell } .$ , every q satisfies $q \leq x _ { 1 , \ell } \mathrm { o r } q \geq x _ { i , \ell } ,$ so one term is nonnegative. Coverage for every q is therefore equivalent to $x _ { 1 , \ell } \geq x _ { i , \ell } .$ Th<sub>e reverse or</sub>i<sub>en</sub>t<sub>e</sub>d case <sub>g</sub><sup>i</sup>ves $x _ { i , \ell } \geq x _ { 1 , \ell }$ □

Lemma 59 (Evaluation case). Under the same hypotheses, with x the common vector,

$$
\operatorname* { m a x } _ { i } { g _ { i } ( x , \eta ) \geq 0 } \qquad f o r e \nu e r y \eta .
$$

Proof. At the pure case ev, role i has value $A _ { \mathrm { e v } , i } ( x _ { i } , \eta ) = g _ { i } ( x _ { i } , \eta )$ . Pure-case coverage gives max<sub>i</sub> $g _ { i } ( x _ { i } , \eta ) \geq 0$ . <sup>A</sup>pp<sup>l</sup>y L<sub>e</sub>mm<sub>a</sub> 58<sub>.</sub> □

ProofofTheorem 9. Membership is the existential-universal encoding in the first subsection. For hardness, Lemma 56 gives $\rho _ { r } \leq 2$ <sub>w</sub>h<sub>enever</sub> th<sub>e source sen</sub>t<sub>ence</sub> h<sub>o</sub>ld<sub>s.</sub> C<sub>onverse</sub>l <sub>,</sub> if $\rho _ { r } \leq 2 ,$ , then Lemma 29 supplies a portfolio of at most r members <sub>w</sub>ith <sub>re re</sub>t <sub>a</sub>t <sub>mos</sub>t t<sub>wo.</sub> L<sub>emmas</sub> 57 t<sub>o</sub> 59<sub>,</sub> f<sub>o</sub>ll<sub>owe</sub>d b L<sub>emma</sub> 46<sub>,</sub> i<sub>ves</sub> th<sub>e source sen</sub>t<sub>ence.</sub> Th<sub>us</sub>

$$
\rho _ { r } \leq 2 \quad \Longleftrightarrow \quad \exists x \forall y \colon F ( x , y ) \geq 0 .
$$

Th<sub>e re</sub>d<sub>uc</sub>ti<sub>on</sub> i<sub>s po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>an</sub>d <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e s</sub>t<sub>a</sub>t<sub>e</sub>d <sub>res</sub>t<sub>r</sub>i<sub>c</sub>ti<sub>ons</sub> b<sub>y</sub> L<sub>emmas</sub> 60 <sub>an</sub>d 61<sub>.</sub>

E<sub>xamp</sub>l<sub>e</sub> 11 ill<sub>us</sub>t<sub>ra</sub>t<sub>es</sub> b<sub>o</sub>th <sub>equa</sub>lit<sub>y au</sub>diti<sub>ng an</sub>d <sub>ro</sub>l<sub>e</sub> f<sub>orc</sub>i<sub>ng.</sub>

Example 11 (A two-role synthesis instance). Take $r = 2$ and $n = 1$ . Then $3 r + 1 = 7$ and $| \mathcal { H } | = 1 + 2 + 1 4 = 1 7 .$ . At the pure case $( { \bar { 2 } } , 1 ,  )$ with $\textstyle q = { \frac { 2 } { 5 } }$ , role one scores $\textstyle { x _ { 1 , 1 } - { \frac { 2 } { 5 } } }$ and role two scores $\begin{array} { r } { \frac { 2 } { 5 } - x _ { 2 , 1 } . { F o r } x _ { 1 , 1 } = \frac { 3 } { 1 0 } } \end{array}$ and $\begin{array} { r } { x _ { 2 , 1 } = \frac { 1 } { 2 } } \end{array}$ , both are ${ \dot { - } } { \frac { 1 } { 1 0 } } .$ No member is nonnegative, and the regret is $2 + { \frac { 1 } { 1 0 } }$ , exposing $x _ { 1 , 1 } < x _ { 2 , 1 }$ . If both coordinates are $\scriptstyle { \frac { 1 } { 2 } } ;$ , role one scores $\frac { 1 } { 1 0 }$ and covers the case.

At forcing case (1, 3), a member with $p ( a _ { 1 } ) = 1$ scores zero. A member with $\begin{array} { r } { p ( a _ { 1 } ) = p ( b _ { ( 1 , 3 ) } ) = \frac { 1 } { 2 } } \end{array}$ scores

$$
3 \cdot { \frac { 1 } { 2 } } + { \frac { 1 } { 2 } } - 1 = 1
$$

and also covers that case, but its remaining mass $\textstyle { \frac { 1 } { 2 } }$ can place the required $\frac { 1 } { 6 }$ on at most three of role one’s seven forcing actions.

## J.8 Size and Structural Restrictions

Lemma 60 (Synthesis size). The reduction runs in polynomial time.

Proof. There is one co variable er occurrence, at most three roduct variables er monomial, and one out ut coordinate. Thus s and $r = 2 s$ <sub>are po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>, an</sub>d

$$
| \mathcal { H } | = 1 + 2 ( r - 1 ) n + r ( 3 r + 1 ) .
$$

<sup>E</sup>ver<sub>y</sub> $W _ { i }$ has degree at most three and polynomially many monomials. Each Poly component has one branch per monomial and at most three factor rows per branch. The budget r is written in unary and is polynomial. Every constant comes from rational arithmetic on input coeficients, B, polynomially bounded integers, and powers $2 ^ { e _ { \nu } + 1 }$ <sub>,</sub> <sub>so</sub> <sub>every</sub> <sub>enco</sub>di<sub>ng</sub> l<sub>eng</sub>th i<sub>s</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l<sub>.</sub> □

Lemma 61 (Synthesis structure). The RMDP $M _ { \mathrm { s y n } }$ has fixed discount ${ \frac { 1 } { 2 } } ,$ , threshold two, and is acyclic apart from absorbing terminals. Every uncertain row has two successors, and the only other stochastic rows are certain uniform splitters. Every reward is afixed rational independent ofthe policy and realization. Nonrectangularity arises onlyfrom the tying equalities and simplex constraint ofDefinition 24.

Proof. Every described path goes from $s _ { \iota }$ t<sub>o a componen</sub>t <sub>en</sub>t<sub>ry,</sub> th<sub>roug</sub>h <sub>a cer</sub>t<sub>a</sub>i<sub>n un</sub>if<sub>orm sp</sub>litt<sub>er, an op</sub>ti<sub>ona</sub>l <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e an</sub>d <sub>a c</sub>h<sub>a</sub>i<sub>n o</sub>f f<sub>ac</sub>t<sub>or se</sub>l<sub>ec</sub>t<sub>ors, an</sub>d th<sub>en</sub> t<sub>o a</sub> t<sub>erm</sub>i<sub>na</sub>l<sub>.</sub> N<sub>o</sub> t<sub>rans</sub>iti<sub>on re</sub>t<sub>urns</sub> t<sub>o an ear</sub>li<sub>er s</sub>t<sub>a</sub>t<sub>e, an</sub>d th<sub>e on</sub>l<sub>y se</sub>lf<sub>-</sub>l<sub>oops are</sub> <sub>a</sub>t t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>t<sub>a</sub>t<sub>es an</sub>d <sub>name</sub>d <sub>ru</sub>i<sub>nous s</sub>i<sub>n</sub>k<sub>s.</sub> Th<sub>e</sub> f<sub>ac</sub>t<sub>or se</sub>l<sub>ec</sub>t<sub>ors are prec</sub>i<sub>se</sub>l<sub>y</sub> th<sub>e</sub> f<sub>res</sub>h t<sub>wo-successor uncer</sub>t<sub>a</sub>i<sub>n rows.</sub> P<sub>o</sub>li<sub>cy</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es en</sub>t<sub>er on</sub>l<sub>y</sub> th<sub>roug</sub>h <sub>coor</sub>di<sub>na</sub>t<sub>e-s</sub>t<sub>a</sub>t<sub>e ac</sub>ti<sub>ons, an</sub>d <sub>rea</sub>li<sub>za</sub>ti<sub>on coor</sub>di<sub>na</sub>t<sub>es on</sub>l<sub>y</sub> th<sub>roug</sub>h f<sub>ac</sub>t<sub>or rows; a</sub>ll t<sub>erm</sub>i<sub>na</sub>l <sub>payo</sub>f<sub>s an</sub>d <sub>rewar</sub>d<sub>s were</sub> fi<sub>xe</sub>d b<sub>e</sub>f<sub>ore</sub> th<sub>e rea</sub>li<sub>za</sub>ti<sub>on an</sub>d <sub>po</sub>li<sub>cy are c</sub>h<sub>osen.</sub> Th<sub>e rema</sub>i<sub>n</sub>i<sub>ng c</sub>l<sub>a</sub>i<sub>ms</sub> f<sub>o</sub>ll<sub>ow</sub> f<sub>rom</sub> L<sub>emma</sub> 51<sub>.</sub>

F<sub>o</sub>ldi<sub>ng</sub> th<sub>e</sub> <sub>cases</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> <sub>coe</sub>fi<sub>c</sub>i<sub>en</sub>t<sub>s</sub> <sub>o</sub>f <sub>one</sub> <sub>po</sub>l<sub>ynom</sub>i<sub>a</sub>l <sub>per</sub> <sub>ro</sub>l<sub>e</sub> <sub>ensures</sub> th<sub>a</sub>t <sub>eac</sub>h <sub>ro</sub>l<sub>e</sub>’<sub>s</sub> <sub>en</sub>ti<sub>re</sub> <sub>m</sub>i<sub>xe</sub>d <sub>score</sub> i<sub>s</sub> <sub>compu</sub>t<sub>e</sub>d b<sub>y</sub> <sub>one</sub> <sub>eva</sub>l<sub>ua</sub>t<sub>or</sub> <sub>an</sub>d <sub>every</sub> <sub>coor</sub>di<sub>na</sub>t<sub>e</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> h<sub>as</sub> <sub>one</sub> <sub>pre</sub>d<sub>ecessor.</sub>

## K Experimental setup

Thi<sub>s</sub> <sub>appen</sub>di<sub>x</sub> <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> d<sub>e</sub>t<sub>a</sub>il<sub>s</sub> b<sub>e</sub>hi<sub>n</sub>d S<sub>ec</sub>ti<sub>on</sub> 4<sub>.</sub> W<sub>e</sub> fi<sub>rs</sub>t fi<sub>x</sub> th<sub>e</sub> <sub>c</sub>l<sub>ass</sub> <sub>o</sub>f <sub>mo</sub>d<sub>e</sub>l<sub>s</sub> th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> b<sub>e</sub>l<sub>ong</sub> t<sub>o</sub> <sub>an</sub>d th<sub>e</sub> <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y</sub> <sub>se</sub>t <sub>eac</sub>h <sub>one</sub> i<sub>n</sub>d<sub>uces,</sub> th<sub>en</sub> d<sub>escr</sub>ib<sub>e</sub> th<sub>e o</sub>fli<sub>ne cons</sub>t<sub>ruc</sub>ti<sub>on o</sub>f <sub>a por</sub>tf<sub>o</sub>li<sub>o, an</sub>d th<sub>en</sub> th<sub>e</sub> t<sub>wo eva</sub>l<sub>ua</sub>ti<sub>ons answer</sub>i<sub>ng</sub> th<sub>e researc</sub>h uestions: whether ortfolios reduce em irical robust re ret as the bud et rows (RQ1), and whether the best member can be identified online in a fixed but unknown environment (RQ2). Al orithm 2 summarizes the whole rocedure. Detailed d<sub>escr</sub>i<sub>p</sub>ti<sub>ons</sub> <sub>o</sub>f th<sub>e</sub> t<sub>wo</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> <sub>c</sub>l<sub>ose</sub> th<sub>e</sub> <sub>appen</sub>di<sub>x,</sub> <sub>an</sub>d th<sub>e</sub> t<sub>a</sub>bl<sub>es</sub> <sub>an</sub>d fi<sub>gures</sub> th<sub>emse</sub>l<sub>ves</sub> <sub>appear</sub> i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> L<sub>.</sub>

## K.1 Parametric MDPs and their uncertainty sets

A parametric MDP (pMDP) is a tuple $( S , A , P , R , s _ { \iota } , \gamma , X )$ in which X is a finite set of parameters and every transition <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> $P ( s , a , s ^ { \prime } )$ i<sub>s a po</sub>l<sub>ynom</sub>i<sub>a</sub>l i<sub>n</sub> $\mathbb { Q } [ X ] . \mathbf { A }$ <sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> $\theta \colon X  \mathbb { R }$ is well defined when it turns every choice row into a probability distribution, and graph preserving when no transition polynomial that is not identically zero evaluates to zero under it. Both benchmarks below are afine pMDPs: every transition polynomial has degree one, and the admissible valuations form a com<sub>p</sub>act <sup>b</sup>ox $D \subseteq \mathbb { R } ^ { X }$ <sub>on w</sub>hi<sub>c</sub>h <sub>we</sub>ll<sub>-</sub>d<sub>e</sub>fi<sub>ne</sub>d<sub>ness</sub> h<sub>o</sub>ld<sub>s</sub> b<sub>y cons</sub>t<sub>ruc</sub>ti<sub>on, so no c</sub>l<sub>amp</sub>i<sub>ng or renorma</sub>li<sub>za</sub>ti<sub>on</sub> i<sub>s ever nee</sub>d<sub>e</sub>d<sub>.</sub>

S<sub>uc</sub>h <sub>a</sub> b<sub>enc</sub>h<sub>mar</sub>k i<sub>s an</sub> RMDP i<sub>n</sub> th<sub>e sense o</sub>f S<sub>ec</sub>ti<sub>on</sub> 2<sub>: eac</sub>h $\theta \in D$ i<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>t<sub>es a</sub> t<sub>rans</sub>iti<sub>on vec</sub>t<sub>or</sub> $\mathbf { \Delta } \mathbf { u } _ { \theta }$ <sub>, an</sub>d

$$
\mathcal { U } _ { D } = \{ \boldsymbol { u } _ { \theta } : \boldsymbol { \theta } \in D \}
$$

is the induced uncertainty set. This set is not $( s , a )$ <sub>-rec</sub>t<sub>angu</sub>l<sub>ar.</sub> A <sub>s</sub>i<sub>ng</sub>l<sub>e parame</sub>t<sub>er occurs</sub> i<sub>n</sub> th<sub>e rows o</sub>f <sub>many</sub> dif<sub>eren</sub>t <sub>c</sub>h<sub>o</sub>i<sub>ces</sub> — the wind intensity p governs every grid cell of the UAV benchmark, and the cooling efectiveness $p _ { c }$ <sub>every s</sub>t<sub>a</sub>t<sub>e o</sub>f th<sub>e</sub> d<sub>a</sub>t<sub>acen</sub>t<sub>er</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>— so</sub> fi<sub>x</sub>i<sub>ng na</sub>t<sub>ure</sub>’<sub>s move a</sub>t <sub>one c</sub>h<sub>o</sub>i<sub>ce</sub> fi<sub>xes</sub> it <sub>everyw</sub>h<sub>ere e</sub>l<sub>se.</sub>

Robust <sub>p</sub>olic<sub>y</sub> evaluation, however, is <sub>p</sub>erformed b<sub>y</sub> robust value iteration (I<sub>y</sub>en<sub>g</sub>ar 2005), which requires a rectan<sub>g</sub>ular <sub>uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y se</sub>t<sub>.</sub> W<sub>e</sub> th<sub>ere</sub>f<sub>ore re</sub>l<sub>ax eac</sub>h <sub>parame</sub>t<sub>er reg</sub>i<sub>on</sub> $c \subseteq D$ t<sub>o</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l MDP <sub>w</sub>h<sub>ose per-c</sub>h<sub>o</sub>i<sub>ce uncer</sub>t<sub>a</sub>i<sub>n</sub>t<sub>y se</sub>t<sub>s are</sub> th<sub>e</sub> coordinatewise ranges of the transition polynomials over c. Writing $\mathcal { U } _ { c } ^ { \boxed { \Pi } }$ f<sub>or</sub> th<sub>e resu</sub>lti<sub>ng rec</sub>t<sub>angu</sub>l<sub>ar se</sub>t<sub>,</sub> thi<sub>s re</sub>l<sub>axa</sub>ti<sub>on on</sub>l<sub>y</sub> <sub>ever a</sub>dd<sub>s</sub> t<sub>rans</sub>iti<sub>on vec</sub>t<sub>ors,</sub>

$$
\mathcal { U } _ { c } = \{ \boldsymbol { \mathscr { u } } _ { \boldsymbol { \theta } } : \boldsymbol { \theta } \in c \} \subseteq \mathcal { U } _ { c } ^ { \boxed { \mathrm { Z } } } ,
$$

b<sub>ecause</sub> it d<sub>rops</sub> th<sub>e equa</sub>liti<sub>es</sub> t<sub>y</sub>i<sub>ng repea</sub>t<sub>e</sub>d <sub>occurrences o</sub>f <sub>a parame</sub>t<sub>er</sub> t<sub>o a common va</sub>l<sub>ue.</sub>

## K.2 Ofline portfolio construction

Candidates. We split the parameter box D into a uniform grid of cells by dividing each parameter’s interval into 10 equal-<sub>w</sub>idth bi<sub>ns,</sub> <sub>so</sub> <sub>a</sub> t<sub>wo-parame</sub>t<sub>er</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>y</sub>i<sub>e</sub>ld<sub>s</sub> $| \mathcal { C } | = 1 0 ^ { 2 } \overset { \cdot } { = } 1 0 0 \mathrm { c e l l s }$ <sub>.</sub> F<sub>or</sub> <sub>every</sub> <sub>ce</sub>ll $c \in { \mathcal { C } }$ <sub>we</sub> i<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>t<sub>e</sub> th<sub>e</sub> <sub>p</sub>MDP <sub>a</sub>t th<sub>e</sub> cell midpoint mid(c) and solve the resulting ordinary MDP by value iteration at tolerance $1 0 ^ { - 8 }$ <sub>, w</sub>hi<sub>c</sub>h <sub>g</sub>i<sub>ves an op</sub>ti<sub>ma</sub>l <sub>po</sub>li<sub>cy</sub> $\pi _ { \mathrm { m i d } ( c ) } ^ { * }$ <sub>an</sub>d it<sub>s</sub> <sub>va</sub>l<sub>ue</sub> $V _ { \mathrm { m i d } ( c ) } ^ { * } ( s _ { \iota } )$ . The candidate set Π(C) collects one such memoryless deterministic policy per cell.

Scoring candidates. Every candidate is then evaluated robustly against every cell. For a candidate π and a cell $c ^ { \prime } ,$ <sub>, ro</sub>b<sub>us</sub>t <sub>va</sub>l<sub>ue</sub> it<sub>era</sub>ti<sub>on on</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>re</sub>l<sub>axa</sub>ti<sub>on</sub> $\mathcal { U } _ { c ^ { \prime } } ^ { \boxed { \Pi } }$ returns

$$
\underline { { V } } _ { c ^ { \prime } } ^ { \pi } = \operatorname* { i n f } _ { \pmb { u } \in \mathcal { U } _ { c ^ { \prime } } ^ { \bigstar } } V _ { \pmb { u } } ^ { \pi } ( s _ { \iota } ) ,
$$

<sub>an</sub>d th<sub>e</sub> <sub>score</sub> <sub>recor</sub>d<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> l<sub>oss</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> i<sub>s</sub> th<sub>e</sub> <sub>m</sub>id<sub>po</sub>i<sub>n</sub>t<sub>-anc</sub>h<sub>ore</sub>d dif<sub>erence</sub>

$$
\begin{array} { r } { L _ { \pi , c ^ { \prime } } = V _ { \mathrm { m i d } ( c ^ { \prime } ) } ^ { * } ( s _ { \iota } ) - \underline { { V } } _ { c ^ { \prime } } ^ { \pi } . } \end{array}\tag{6}
$$

<sup>T</sup>wo a<sub>pp</sub>rox<sup>i</sup>mat<sup>i</sup>ons se<sub>p</sub>arate $L _ { \pi , c ^ { \prime } }$ f<sub>rom</sub> th<sub>e exac</sub>t <sub>ce</sub>ll<sub>w</sub>i<sub>se ro</sub>b<sub>us</sub>t <sub>regre</sub>t ${ \mathrm { s u p } } _ { \theta \in c ^ { \prime } } \left( V _ { \theta } ^ { * } ( s _ { \iota } ) - V _ { \theta } ^ { \pi } ( s _ { \iota } ) \right)$ <sub>.</sub> Fi<sub>rs</sub>t<sub>,</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>va</sub>l<sub>ue</sub> i<sub>s</sub> taken at the single point mid $\left( c ^ { \prime } \right)$ <sub>ra</sub>th<sub>er</sub> th<sub>an a</sub>t th<sub>e max</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> $\theta ,$ <sub>w</sub>hi<sub>c</sub>h <sub>may err</sub> i<sub>n e</sub>ith<sub>er</sub> di<sub>rec</sub>ti<sub>on.</sub> S<sub>econ</sub>d<sub>,</sub> b<sub>y</sub> th<sub>e</sub> i<sub>nc</sub>l<sub>us</sub>i<sub>on</sub> <sub>a</sub>b<sub>ove,</sub> th<sub>e</sub> i<sub>n</sub>fi<sub>mum over</sub> $\mathcal { U } _ { c ^ { \prime } } ^ { \boxed { \Pi } }$ i<sub>s a</sub>t <sub>mos</sub>t th<sub>e</sub> i<sub>n</sub>fi<sub>mum over</sub> $\mathcal { U } _ { c ^ { \prime } }$ <sub>, so</sub> thi<sub>s</sub> t<sub>erm a</sub>l<sub>one ma</sub>k<sub>es</sub> $L _ { \pi , c ^ { \prime } }$ <sub>an over-es</sub>ti<sub>ma</sub>t<sub>e o</sub>f th<sub>e</sub> l<sub>oss</sub> it <sub>s</sub>t<sub>an</sub>d<sub>s</sub> i<sub>n</sub> f<sub>or.</sub> Th<sub>e</sub> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $L$ i<sub>s</sub> <sub>a</sub> <sub>c</sub>l<sub>us</sub>t<sub>er</sub>i<sub>ng</sub> <sub>score,</sub> <sub>no</sub>t <sub>a</sub> <sub>cer</sub>tifi<sub>ca</sub>t<sub>e;</sub> th<sub>e</sub> <sub>guaran</sub>t<sub>ees</sub> <sub>repor</sub>t<sub>e</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 4 <sub>come</sub> f<sub>rom</sub> th<sub>e</sub> <sub>samp</sub>l<sub>e</sub>d <sub>eva</sub>l<sub>ua</sub>ti<sub>on o</sub>f th<sub>e nex</sub>t <sub>su</sub>b<sub>sec</sub>ti<sub>on, w</sub>hi<sub>c</sub>h <sub>uses no re</sub>l<sub>axa</sub>ti<sub>on a</sub>t <sub>a</sub>ll<sub>.</sub>

Selecting the portfolio. Each candidate π thus carries a profile $L _ { \pi } \in \mathbb { R } ^ { c }$ <sub>, one en</sub>t<sub>ry per ce</sub>ll<sub>,</sub> d<sub>escr</sub>ibi<sub>ng w</sub>h<sub>ere</sub> i<sub>n</sub> th<sub>e parame</sub>t<sub>er</sub> s ace it does well. For a bud et K we run K-means on these rofiles and kee , from each of the K clusters, the candidate whose <sub>pro</sub>fil<sub>e</sub> i<sub>s neares</sub>t th<sub>a</sub>t <sub>c</sub>l<sub>us</sub>t<sub>er</sub>’<sub>s cen</sub>t<sub>er, y</sub>i<sub>e</sub>ldi<sub>ng a por</sub>tf<sub>o</sub>li<sub>o</sub> $\bar { \Pi _ { K } } \subseteq \Pi ( { \mathcal { C } } )$ <sub>o</sub>f $K$ <sub>memory</sub>l<sub>ess</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c po</sub>li<sub>c</sub>i<sub>es.</sub> Cl<sub>us</sub>t<sub>er</sub>i<sub>ng</sub> i<sub>s</sub> <sub>a c</sub>h<sub>eap s</sub>t<sub>an</sub>d<sub>-</sub>i<sub>n</sub> f<sub>or</sub> th<sub>e</sub> i<sub>n</sub>t<sub>rac</sub>t<sub>a</sub>bl<sub>e s n</sub>th<sub>es</sub>i<sub>s pro</sub>bl<sub>em o</sub>f P<sub>ro</sub>bl<sub>em</sub> 4<sub>:</sub> it <sub>never</sub> l<sub>eaves</sub> th<sub>e can</sub>did<sub>a</sub>t<sub>e se</sub>t<sub>, an</sub>d it <sub>op</sub>ti<sub>m</sub>i<sub>zes pro</sub>fil<sub>e</sub> <sub>s</sub>i<sub>m</sub>il<sub>ar</sub>it<sub>y ra</sub>th<sub>er</sub> th<sub>an por</sub>tf<sub>o</sub>li<sub>o regre</sub>t<sub>.</sub>

A single-policy reference point. The same loss matrix also yields an exact reference for $K = 1$ <sub>.</sub> Mi<sub>n</sub>i<sub>m</sub>i<sub>z</sub>i<sub>ng eac</sub>h <sub>can</sub>did<sub>a</sub>t<sub>e</sub>’<sub>s</sub> <sub>wors</sub>t <sub>ce</sub>ll<sub>,</sub>

$$
\operatorname* { m i n } _ { \pi \in \Pi ( { \mathcal { C } } ) } \operatorname* { m a x } _ { c \in { \mathcal { C } } } L _ { \pi , c } ,
$$

is the best worst-case score attainable by any single member of the candidate set, computed directly from L without sampling <sub>or c</sub>l<sub>us</sub>t<sub>er</sub>i<sub>ng.</sub> It i<sub>s</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>-max regre</sub>t fi<sub>gure quo</sub>t<sub>e</sub>d i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 4<sub>.</sub>

Implementation. The prototype is written in Python. Models are built and instantiated with Stormvogel (Volk et al. 2026) <sub>on</sub> t<sub>op o</sub>f St<sub>orm, w</sub>hi<sub>c</sub>h <sub>a</sub>l<sub>so prov</sub>id<sub>es</sub> th<sub>e parame</sub>t<sub>er reg</sub>i<sub>ons an</sub>d th<sub>e convers</sub>i<sub>on o</sub>f <sub>a reg</sub>i<sub>on</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> i<sub>n</sub>t<sub>erva</sub>l MDP <sub>use</sub>d f<sub>or</sub> th<sub>e</sub> <sub>re</sub>l<sub>axa</sub>ti<sub>on a</sub>b<sub>ove.</sub> V<sub>a</sub>l<sub>ue</sub> it<sub>era</sub>ti<sub>on, ro</sub>b<sub>us</sub>t <sub>va</sub>l<sub>ue</sub> it<sub>era</sub>ti<sub>on, an</sub>d fi<sub>xe</sub>d<sub>-po</sub>li<sub>cy eva</sub>l<sub>ua</sub>ti<sub>on are our own</sub> N<sub>um</sub>P<sub>y</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>ons,</sub> i<sub>n</sub> d<sub>ense an</sub>d <sub>sparse var</sub>i<sub>an</sub>t<sub>s;</sub> th<sub>e</sub> l<sub>arger</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s use</sub> th<sub>e sparse pa</sub>th th<sub>roug</sub>h<sub>ou</sub>t<sub>.</sub> Cl<sub>us</sub>t<sub>er</sub>i<sub>ng uses sc</sub>ikit<sub>-</sub>l<sub>earn</sub>’<sub>s</sub> K<sub>-means w</sub>ith <sub>au</sub>t<sub>oma</sub>ti<sub>c res</sub>t<sub>ar</sub>t <sub>se</sub>l<sub>ec</sub>ti<sub>on an</sub>d th<sub>e see</sub>d <sub>as</sub> it<sub>s ran</sub>d<sub>om s</sub>t<sub>a</sub>t<sub>e.</sub> E<sub>ac</sub>h <sub>s</sub>t<sub>age o</sub>f th<sub>e p</sub>i<sub>pe</sub>li<sub>ne cac</sub>h<sub>es</sub> it<sub>s ou</sub>t<sub>pu</sub>t i<sub>n</sub> HDF5<sub>, so a s</sub>t<sub>age can</sub> b<sub>e</sub> <sub>resume</sub>d <sub>or</sub> <sub>re-run</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>repea</sub>ti<sub>ng</sub> it<sub>s</sub> <sub>pre</sub>d<sub>ecessors.</sub>

## K.3 RQ1: measuring portfolio regret

The portfolios are scored against valuations drawn directly from D, with no cells and no rectangular relaxation involved. For a <sub>see</sub>d <sub>we</sub> d<sub>raw</sub> $\widehat { D } \subset D$ <sub>un</sub>if<sub>orm</sub>l<sub>y a</sub>t <sub>ran</sub>d<sub>om w</sub>ith $| \widehat { D } | = 1 0 0 0$ , an<sup>d</sup> at ever<sub>y</sub> $\theta \in \widehat { D }$ <sub>we</sub> in<sub>s</sub>t<sub>a</sub>nti<sub>a</sub>t<sub>e</sub> th<sub>e p</sub>MDP <sub>a</sub>nd <sub>co</sub>m<sub>pu</sub>t<sub>e</sub> t<sub>wo</sub> <sub>exac</sub>t <sub>quan</sub>titi<sub>es:</sub> th<sub>e op</sub>ti<sub>ma</sub>l <sub>va</sub>l<sub>ue</sub> $V _ { \theta } ^ { * } ( s _ { \iota } )$ b<sub>y va</sub>l<sub>ue</sub> it<sub>era</sub>ti<sub>on, an</sub>d th<sub>e va</sub>l<sub>ue</sub> $V _ { \theta } ^ { \pi } ( s _ { \iota } )$ <sub>o</sub>f <sub>eac</sub>h <sub>por</sub>tf<sub>o</sub>li<sub>o mem</sub>b<sub>er</sub> b<sub>y</sub> fi<sub>xe</sub>d<sub>-po</sub>li<sub>cy</sub> <sub>eva</sub>l<sub>ua</sub>ti<sub>on.</sub> Th<sub>e repor</sub>t<sub>e</sub>d <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> i<sub>s</sub>

$$
\widehat { \mathrm { R r e g } } _ { \widehat { D } } ( \Pi ) = \operatorname* { m a x } _ { \theta \in \widehat { D } } \Bigl ( V _ { \theta } ^ { \ast } ( s _ { \iota } ) - \operatorname* { m a x } _ { \pi \in \Pi } V _ { \theta } ^ { \pi } ( s _ { \iota } ) \Bigr ) .
$$

Each valuation is thus charged the shortfall of the best member at that valuation, matching the ofline coverage quantity of (1) and char<sub>g</sub>in<sub>g</sub> nothin<sub>g</sub> for identif<sub>y</sub>in<sub>g</sub> that member online; RQ2 measures the identification cost se<sub>p</sub>aratel<sub>y</sub>. Because the su<sub>p</sub>remum in (1) is re<sub>p</sub>laced b<sub>y</sub> a maximum over finitel<sub>y</sub> man<sub>y</sub> sam<sub>p</sub>les, $\widehat { \mathrm { R r e g } } _ { \widehat { D } } ( \Pi )$ under-approximates Rreg(Π). The whole construction-and-evaluation procedure is repeated for the three seeds 0, 1, 2, which reseed both the K-means initialization and th<sub>e samp</sub>l<sub>e</sub> $\widehat { D } \smash { \mathrm { : } }$ <sub>;</sub> T<sub>a</sub>bl<sub>e</sub> 5 <sub>repor</sub>t<sub>s</sub> th<sub>e</sub> i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>see</sub>d<sub>s an</sub>d T<sub>a</sub>bl<sub>e</sub> 2 th<sub>e</sub>i<sub>r mean.</sub>

## K.4 RQ2: online portfolio selection

Setting. A fixed but hidden valuation θ defines a generative model $G _ { \theta }$ from which trajectories of the instantiated MDP can b<sub>e samp</sub>l<sub>e</sub>d <sub>s</sub>t<sub>ar</sub>ti<sub>ng a</sub>t $s _ { \iota }$ . The portfolio members are treated as the arms of a bandit: pulling arm π rolls out one trajectory of h<sub>or</sub>i<sub>zon</sub> $H = 1 0 0$ under π and returns its discounted return. Nothing but these sampled returns is observed, neither θ nor an<sub>y</sub> <sub>va</sub>l<sub>ue</sub> f<sub>unc</sub>ti<sub>on, so a run mus</sub>t di<sub>scr</sub>i<sub>m</sub>i<sub>na</sub>t<sub>e</sub> b<sub>e</sub>t<sub>ween mem</sub>b<sub>ers</sub> f<sub>rom ro</sub>ll<sub>ou</sub>t<sub>s a</sub>l<sub>one.</sub> R<sub>e</sub>t<sub>urns</sub> li<sub>e</sub> i<sub>n</sub> $[ - B _ { H } , B _ { H } ]$ f<sub>or</sub>

$$
B _ { H } = R _ { \mathrm { m a x } } \frac { 1 - \gamma ^ { H } } { 1 - \gamma } ,
$$

<sub>w</sub>ith $R _ { \mathrm { m a x } }$ th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k’<sub>s</sub> l<sub>arges</sub>t <sub>rewar</sub>d <sub>magn</sub>it<sub>u</sub>d<sub>e.</sub>

The selection rule. We use UCB (Cesa-Bianchi and Lugosi 2006) in its best-arm-identification form. After one warm-up pull of every arm, round t computes for each arm i still in contention a confidence radius: the empirical-Bernstein radius (Mnih, Sze<sub>p</sub>esvári, and Audibert 2008), whose leadin<sub>g</sub> term shrinks with the em<sub>p</sub>irical variance rather than with the worst-case ran<sub>g</sub>e; <sub>on</sub> th<sub>ese</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub> th<sub>e re</sub>t<sub>urns are</sub> f<sub>ar</sub> l<sub>ess var</sub>i<sub>a</sub>bl<sub>e</sub> th<sub>an</sub> $B _ { H }$ <sub>su es</sub>t<sub>s, so</sub> it <sub>conver es cons</sub>id<sub>era</sub>bl f<sub>as</sub>t<sub>er</sub> th<sub>an a</sub> H<sub>oe</sub>fdi<sub>n</sub> <sub>ra</sub>di<sub>us.</sub> W<sub>r</sub>iti<sub>ng</sub> $i _ { \mathrm { s a f e } }$ f<sub>or</sub> th<sub>e</sub> <sub>arm</sub> <sub>o</sub>f <sub>grea</sub>t<sub>es</sub>t l<sub>ower</sub> <sub>con</sub>fid<sub>ence</sub> b<sub>oun</sub>d<sub>,</sub> th<sub>e</sub> <sub>run</sub> <sub>s</sub>t<sub>ops</sub> <sub>as</sub> <sub>soon</sub> <sub>as</sub>

$$
\mathrm { l c b } ( i _ { \mathrm { s a f e } } ) \geq \operatorname* { m a x } _ { j \neq i _ { \mathrm { s a f e } } } \mathrm { u c b } ( j ) - \varepsilon _ { \mathrm { a b s } } ,
$$

th<sub>a</sub>t i<sub>s, as soon as no o</sub>th<sub>er arm can</sub> b<sub>e more</sub> th<sub>an</sub> $\varepsilon _ { \mathrm { a b s } }$ b<sub>e</sub>tt<sub>er.</sub> Oth<sub>erw</sub>i<sub>se every arm w</sub>h<sub>ose upper</sub> b<sub>oun</sub>d h<sub>as</sub> f<sub>a</sub>ll<sub>en</sub> b<sub>e</sub>l<sub>ow</sub> $\mathrm { l c b } ( i _ { \mathrm { s a f e } } ) - \varepsilon _ { \mathrm { a b s } }$ i<sub>s e</sub>li<sub>m</sub>i<sub>na</sub>t<sub>e</sub>d<sub>, an</sub>d th<sub>e arm o</sub>f <sub>grea</sub>t<sub>es</sub>t <sub>upper con</sub>fid<sub>ence</sub> b<sub>oun</sub>d i<sub>s pu</sub>ll<sub>e</sub>d <sub>nex</sub>t<sub>.</sub> A <sub>run</sub> th<sub>ere</sub>f<sub>ore</sub> h<sub>as no</sub> fi<sub>xe</sub>d iteration budget: it stops when its own stopping test fires, and the number of pulls varies from run to run and grows with K.

W<sub>e use con</sub>fid<sub>ence</sub> $\delta = 0 . 1$ <sub>an</sub>d t<sub>o</sub>l<sub>erance</sub> $\varepsilon = 1 0 ^ { - 3 }$ <sub>,</sub> th<sub>e</sub> l<sub>a</sub>tt<sub>er expresse</sub>d <sub>as a</sub> f<sub>rac</sub>ti<sub>on o</sub>f th<sub>e re</sub>t<sub>urn</sub> b<sub>oun</sub>d<sub>, so</sub> th<sub>e a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e</sub> t<sub>o</sub>l<sub>erance</sub> i<sub>s</sub> $\varepsilon _ { \mathrm { a b s } } = \varepsilon \cdot B _ { H } = 1 0 ^ { - 3 } B _ { H }$ . Since the empirical-Bernstein radius does the work of separating the arms, ε acts only <sub>as a cap</sub> th<sub>a</sub>t k<sub>eeps near-</sub>ti<sub>es</sub> f<sub>rom runn</sub>i<sub>ng</sub> i<sub>n</sub>d<sub>e</sub>fi<sub>n</sub>it<sub>e</sub>l<sub>y, an</sub>d i<sub>s</sub> k<sub>ep</sub>t <sub>sma</sub>ll<sub>.</sub>

What is measured. At each iteration, a run has both a pulled arm, the exploratory one of greatest upper confidence bound, and a recommended arm, the one of greatest lower confidence bound — the member it would deploy if forced to stop right then. W<sub>e repor</sub>t th<sub>e recommen</sub>d<sub>e</sub>d <sub>arm,</sub> th<sub>e pu</sub>ll<sub>e</sub>d <sub>arm</sub> b<sub>e</sub>i<sub>ng exp</sub>l<sub>ora</sub>t<sub>ory</sub> b<sub>y</sub> d<sub>es</sub>i<sub>gn.</sub>

A recommendation is scored against the best portfolio member at the hidden valuation, that is, arg $\operatorname* { m a x } _ { \pi \in \Pi } V _ { \theta } ^ { \pi } ( s _ { \iota } )$ com<sub>p</sub>ute<sup>d</sup> <sub>exac</sub>tl<sub>y</sub> b<sub>y</sub> fi<sub>xe</sub>d<sub>-po</sub>li<sub>cy eva</sub>l<sub>ua</sub>ti<sub>on, an</sub>d <sub>no</sub>t <sub>aga</sub>i<sub>ns</sub>t th<sub>e unres</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d <sub>op</sub>ti<sub>mum</sub> $V _ { \theta } ^ { * }$ <sub>.</sub> Th<sub>e ques</sub>ti<sub>on</sub> i<sub>s w</sub>h<sub>e</sub>th<sub>er on</sub>li<sub>ne se</sub>l<sub>ec</sub>ti<sub>on</sub> fi<sub>n</sub>d th<sub>e</sub> b<sub>es</sub>t <sub>o</sub>li<sub>c</sub> th<sub>e or</sub>tf<sub>o</sub>li<sub>o ac</sub>t<sub>ua</sub>ll <sub>con</sub>t<sub>a</sub>i<sub>ns.</sub> T<sub>wo uan</sub>titi<sub>es are recor</sub>d<sub>e</sub>d <sub>er</sub> it<sub>era</sub>ti<sub>on.</sub> Th<sub>e</sub> fi<sub>rs</sub>t i<sub>s</sub> th<sub>e</sub> f<sub>rac</sub>ti<sub>on o</sub>f <sub>runs w</sub>h<sub>ose</sub> <sub>recommen</sub>d<sub>a</sub>ti<sub>on</sub> i<sub>s</sub> th<sub>a</sub>t b<sub>es</sub>t <sub>mem</sub>b<sub>er.</sub> Th<sub>e secon</sub>d i<sub>s</sub> th<sub>e s</sub>h<sub>or</sub>tf<sub>a</sub>ll <sub>o</sub>f th<sub>e recommen</sub>d<sub>e</sub>d <sub>mem</sub>b<sub>er, norma</sub>li<sub>ze</sub>d b th<sub>e sprea</sub>d <sub>o</sub>f th<sub>e</sub> <sub>por</sub>tf<sub>o</sub>li<sub>o a</sub>t th<sub>a</sub>t <sub>va</sub>l<sub>ua</sub>ti<sub>on,</sub>

$$
L _ { \theta } ^ { \pi } = \frac { V _ { \theta } ^ { \mathrm { b e s t } } ( \iota ) - V _ { \theta } ^ { \pi } ( \iota ) } { V _ { \theta } ^ { \mathrm { b e s t } } ( \iota ) - V _ { \theta } ^ { \mathrm { w o r s t } } ( \iota ) } ,
$$

so that recommendin the best member scores 0 and recommendin the worst scores 1 (with $L _ { \theta } ^ { \pi } = 0$ b<sub>y conven</sub>ti<sub>on w</sub>h<sub>en every</sub> member ties). The second measure is the more informative of the two: the first counts a near-tie between two almost equall<sub>y</sub> <sub>goo</sub>d <sub>mem</sub>b<sub>ers as an ou</sub>t<sub>r</sub>i<sub>g</sub>ht f<sub>a</sub>il<sub>ure, w</sub>h<sub>ereas</sub> th<sub>e secon</sub>d <sub>c</sub>h<sub>arges on</sub>l<sub>y w</sub>h<sub>a</sub>t th<sub>a</sub>t <sub>con</sub>f<sub>us</sub>i<sub>on ac</sub>t<sub>ua</sub>ll<sub>y cos</sub>t<sub>s.</sub>

Sampling and aggregation. For each portfolio size K and each seed, we draw 30 valuations uniformly from D and run the selection rule once er valuation, all from one seeded random stream. Poolin the three seeds ives the 90 runs behind each curve. Because runs stop at diferent iterations, a curve at iteration t averages over runs still active at t together with runs that h<sub>ave a</sub>l<sub>rea</sub>d<sub>y comm</sub>itt<sub>e</sub>d<sub>, eac</sub>h <sub>o</sub>f th<sub>e</sub> l<sub>a</sub>tt<sub>er con</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>ng</sub> it<sub>s own</sub> d<sub>ep</sub>l<sub>oye</sub>d <sub>ou</sub>t<sub>come</sub> f<sub>rom</sub> it<sub>s s</sub>t<sub>opp</sub>i<sub>ng</sub> ti<sub>me onwar</sub>d<sub>.</sub> C<sub>urves are</sub> d<sub>rawn on a</sub> l<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> it<sub>era</sub>ti<sub>on ax</sub>i<sub>s w</sub>ith <sub>va</sub>l<sub>ues</sub> bi<sub>nne</sub>d <sub>accor</sub>di<sub>ng</sub>l<sub>y, an</sub>d <sub>are s</sub>h<sub>own w</sub>ith <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d<sub>-error</sub> b<sub>an</sub>d<sub>s.</sub>

## K.5 Datacenter benchmark

This benchmark models a controller regulating a server room’s temperature and humidity while working of a job queue, <sub>uncer</sub>t<sub>a</sub>i<sub>n a</sub>b<sub>ou</sub>t h<sub>ow s</sub>t<sub>rong</sub>l<sub>y</sub> it<sub>s ac</sub>ti<sub>ons an</sub>d th<sub>e am</sub>bi<sub>en</sub>t <sub>env</sub>i<sub>ronmen</sub>t <sub>ac</sub>t<sub>ua</sub>ll<sub>y move</sub> th<sub>e sys</sub>t<sub>em.</sub>

State space. A state $s = ( T , H , L ) \in S$ <sub>recor</sub>d<sub>s a</sub> di<sub>scre</sub>ti<sub>ze</sub>d t<sub>empera</sub>t<sub>ure</sub> l<sub>eve</sub>l $T \in \{ 0 , \ldots , 1 0 \}$ <sub>, a</sub> h<sub>um</sub>idit<sub>y</sub> l<sub>eve</sub>l $H \in$ $\{ 0 , \ldots , 5 \}$ <sub>,</sub> <sub>an</sub>d <sub>a</sub> <sub>queue</sub> l<sub>eng</sub>th $L \in \{ 0 , \dots , 4 \} , \mathrm { s o } | S | = 1 1 \cdot 6 \cdot 5 = 3 3 0$ <sub>.</sub> E<sub>p</sub>i<sub>so</sub>d<sub>es s</sub>t<sub>ar</sub>t <sub>a</sub>t $s _ { 0 } = ( 5 , 2 , 2 )$ <sub>, a com</sub>f<sub>or</sub>t<sub>a</sub>bl<sub>e m</sub>iddl<sub>e</sub> <sup>of e</sup>v<sup>er</sup>y <sup>ran</sup>g<sup>e</sup>.

Action space. At every state, the controller picks one of five actions,

$$
A = \left\{ \mathrm { c o o L \mathrm { - } H I G H , C O O L \mathrm { - } L O W , H O L D , H E A T , D E H U M I D I F Y } \right\} .
$$

Th<sub>ey</sub> t<sub>ra</sub>d<sub>e</sub> <sub>energy</sub> <sub>aga</sub>i<sub>ns</sub>t <sub>com</sub>f<sub>or</sub>t<sub>:</sub> <sub>cool-high</sub> <sub>pus</sub>h<sub>es</sub> h<sub>ar</sub>d<sub>es</sub>t <sub>on</sub> t<sub>empera</sub>t<sub>ure</sub> <sub>an</sub>d <sub>cos</sub>t<sub>s</sub> th<sub>e</sub> <sub>mos</sub>t<sub>,</sub> <sub>cool-low</sub> i<sub>s</sub> it<sub>s</sub> <sub>c</sub>h<sub>eaper</sub> <sub>an</sub>d <sub>wea</sub>k<sub>er coun</sub>t<sub>erpar</sub>t<sub>, hold spen</sub>d<sub>s a</sub>l<sub>mos</sub>t <sub>no</sub>thi<sub>ng an</sub>d l<sub>e</sub>t<sub>s</sub> th<sub>e room</sub> d<sub>r</sub>ift<sub>, heat moves</sub> t<sub>empera</sub>t<sub>ure</sub> th<sub>e o</sub>th<sub>er way, an</sub>d <sub>dehumidify</sub> t<sub>arge</sub>t<sub>s</sub> h<sub>um</sub>idit<sub>y</sub> i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f t<sub>empera</sub>t<sub>ure.</sub>

Parameter space. Two parameters are unknown, $\theta = ( p _ { c } , p _ { e } ) \in \Theta = [ 0 . 5 , 0 . 9 ] \times [ 0 . 1 , 0 . 8 ]$ . The coolin efectiveness p is $p _ { c }$ h<sub>ow re</sub>li<sub>a</sub>bl<sub>y an ac</sub>ti<sub>on moves</sub> th<sub>e room as</sub> i<sub>n</sub>t<sub>en</sub>d<sub>e</sub>d<sub>, an</sub>d th<sub>e env</sub>i<sub>ronmen</sub>t<sub>a</sub>l <sub>pressure</sub> $p _ { e }$ i<sub>s</sub> h<sub>ow s</sub>t<sub>rong</sub>l<sub>y wor</sub>kl<sub>oa</sub>d <sub>an</sub>d <sub>am</sub>bi<sub>en</sub>t <sub>con</sub>diti<sub>ons us</sub>h b<sub>ac</sub>k<sub>.</sub> A hi h $p _ { c }$ makes the controller’s actions de endable a hi h p means the room heats humidifies and $p _ { e }$ <sub>accumu</sub>l<sub>a</sub>t<sub>es wor</sub>k <sub>regar</sub>dl<sub>ess o</sub>f <sub>w</sub>h<sub>a</sub>t th<sub>e con</sub>t<sub>ro</sub>ll<sub>er</sub> d<sub>oes.</sub>

Transitions. Every state-action pair has four successor branches, split evenly between a control outcome governed by $p _ { c }$ <sub>an</sub>d an environment outcome governed by $p _ { e } \colon$

$$
\begin{array} { r l } & { { \mathbb P } _ { \theta } ( s ^ { \prime } \mid s , a ) \in \big \{ \frac { p _ { c } } { 2 } , \frac { 1 - p _ { c } } { 2 } , \frac { p _ { c } } { 2 } , \frac { 1 - p _ { c } } { 2 } \big \} , \qquad \frac { p _ { c } } { 2 } + \frac { 1 - p _ { c } } { 2 } + \frac { p _ { c } } { 2 } + \frac { 1 - p _ { c } } { 2 } = 1 , } \end{array}
$$

<sub>so</sub> th<sub>e</sub> f<sub>our</sub> b<sub>ranc</sub>h<sub>es sum</sub> t<sub>o one a</sub>t <sub>every</sub> $\theta$ and ever<sub>y</sub> transition probabilit<sub>y</sub> is afine in θ. Writing $[ x ] _ { l o } ^ { h i } : = \operatorname* { m i n } ( h i , \operatorname* { m a x } ( l o , x ) )$ f<sub>or c</sub>l<sub>amp</sub>i<sub>ng a coor</sub>di<sub>na</sub>t<sub>e</sub> t<sub>o</sub> it<sub>s range, an</sub>d <sub>a</sub>bb<sub>rev</sub>i<sub>a</sub>ti<sub>ng</sub> $T ^ { \pm } : \stackrel { \cdot \cdot } { = } [ T \pm 1 ] _ { 0 } ^ { 1 0 } , H ^ { \pm } : = [ H \pm 1 ] _ { 0 } ^ { 5 } , L ^ { \bar { \pm } } : \stackrel { \cdot \cdot \cdot \cdot } { = } [ L \pm 1 ] _ { 0 } ^ { 4 } ,$ T<sub>a</sub>bl<sub>e</sub> 3 <sub>g</sub>i<sub>ves</sub> th<sub>e</sub> <sub>successor reac</sub>h<sub>e</sub>d b<sub>y eac</sub>h b<sub>ranc</sub>h<sub>.</sub> At <sub>a</sub> b<sub>oun</sub>d<sub>ary s</sub>t<sub>a</sub>t<sub>e,</sub> t<sub>wo</sub> b<sub>ranc</sub>h<sub>es can</sub> l<sub>an</sub>d <sub>on</sub> th<sub>e same successor,</sub> i<sub>n w</sub>hi<sub>c</sub>h <sub>case</sub> th<sub>e</sub>i<sub>r</sub> <sub>pro</sub>b<sub>a</sub>biliti<sub>es are a</sub>dd<sub>e</sub>d<sub>.</sub>

<table><tr><td>Action</td><td> $p _ { c } / 2$ </td><td> $( 1 - p _ { c } ) / 2$ </td><td> $p _ { e } / 2$ </td><td> $( 1 - p _ { e } ) / 2$ </td></tr><tr><td>COOL-HIGH</td><td> $( T ^ { - } , H , L )$ </td><td> $( T ^ { + } , H , L )$ </td><td> $( T , H , L ^ { - } )$ </td><td> $( T , H ^ { + } , L ^ { + } )$ </td></tr><tr><td>COOL-LOW</td><td> $( T ^ { - } , H , L )$ </td><td> $( T , H , L )$ </td><td> $( T , H , L ^ { - } )$ </td><td> $\check { ( } T ^ { + } , H , L ^ { + } \check { ) }$ </td></tr><tr><td>HOLD</td><td> $( T , H , L )$ </td><td> $( \dot { T } ^ { + } , H , \dot { L } )$ </td><td> $( \dot { T } , H ^ { - } , L ^ { - } )$ </td><td> $( \dot { T } ^ { + } , H ^ { + } , L ^ { + } )$ </td></tr><tr><td>HEAT</td><td> $( T ^ { + } , H , \dot { L } )$ </td><td> $( T ^ { - } , H , L )$ </td><td> $( T , H , L ^ { - } )$ </td><td> $\dot { ( } T ^ { + } , H ^ { + } , L ^ { + } \dot { ) }$ </td></tr><tr><td>DEHUMIDIFY</td><td> $( T , H ^ { - } , L )$ </td><td> $( T , H ^ { + } , L )$ </td><td> $( \dot { T } ^ { - } , H , L ^ { - } )$ </td><td> $\dot { ( } T ^ { + } , H ^ { + } , L ^ { + } \dot { ) }$ </td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 3<sub>:</sub> S<sub>uccessor s</sub>t<sub>a</sub>t<sub>e reac</sub>h<sub>e</sub>d b<sub>y eac</sub>h <sub>o</sub>f th<sub>e</sub> f<sub>our</sub> b<sub>ranc</sub>h<sub>es o</sub>f $\mathbb { P } _ { \theta } ( \cdot \mid s , a )$ <sub>,</sub> f<sub>or</sub> $s = ( T , H , L )$ <sub>,</sub> i<sub>n</sub> th<sub>e</sub> d<sub>a</sub>t<sub>acen</sub>t<sub>er</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>.</sub>

Rewards and discount. The reward is deterministic and independent of θ:

$$
r ( s , a ) = r _ { E } ( a ) + r _ { T } ( T ) + r _ { H } ( H ) + r _ { L } ( L ) ,
$$

com<sup>bi</sup>n<sup>i</sup>n<sub>g</sub> a <sub>p</sub>er-act<sup>i</sup>on ener<sub>gy</sub> cost

$$
r _ { E } ( \mathrm { c o o L - H G H } ) = - 4 , \quad r _ { E } ( \mathrm { c o o L - L o w } ) = - 2 , \quad r _ { E } ( \mathrm { \mathrm { H o L D } } ) = - 1 , \quad r _ { E } ( \mathrm { \mathrm { H E A T } } ) = - 2 , \quad r _ { E } ( \mathrm { p E H U M I P F Y } ) = - 3 ,
$$

<sub>w</sub>ith <sub>pena</sub>lti<sub>es</sub> th<sub>a</sub>t <sub>ac</sub>ti<sub>va</sub>t<sub>e</sub> <sub>as</sub> <sub>eac</sub>h di<sub>mens</sub>i<sub>on</sub> <sub>approac</sub>h<sub>es</sub> it<sub>s</sub> <sub>ce</sub>ili<sub>ng:</sub>

$$
r _ { T } ( T ) = \left\{ \begin{array} { l l } { 0 } & { T \leq 7 } \\ { - 2 0 } & { T \in \{ 8 , 9 \} } \\ { - 5 0 } & { T = 1 0 } \end{array} \right. \quad \quad r _ { H } ( H ) = \left\{ \begin{array} { l l } { 0 } & { H \leq 3 } \\ { - 1 5 } & { H = 4 } \\ { - 4 0 } & { H = 5 } \end{array} \right. \quad \quad r _ { L } ( L ) = \left\{ \begin{array} { l l } { 0 } & { L \neq 4 } \\ { - 1 0 } & { L = 4 } \end{array} \right.
$$

Th<sub>e</sub> <sub>con</sub>t<sub>ro</sub>ll<sub>er</sub> th<sub>ere</sub>f<sub>ore</sub> <sub>pays</sub> <sub>con</sub>ti<sub>nuous</sub>l<sub>y</sub> f<sub>or</sub> <sub>energy</sub> <sub>an</sub>d <sub>ca</sub>t<sub>as</sub>t<sub>rop</sub>hi<sub>ca</sub>ll<sub>y</sub> f<sub>or</sub> l<sub>e</sub>tti<sub>ng</sub> <sub>any</sub> di<sub>mens</sub>i<sub>on</sub> <sub>reac</sub>h it<sub>s</sub> <sub>ce</sub>ili<sub>ng,</sub> <sub>an</sub>d th<sub>e parame</sub>t<sub>ers</sub> d<sub>ec</sub>id<sub>e</sub> h<sub>ow expens</sub>i<sub>ve</sub> it i<sub>s</sub> t<sub>o</sub> k<sub>eep away</sub> f<sub>rom</sub> th<sub>ose ce</sub>ili<sub>ngs.</sub> Th<sub>e</sub> l<sub>arges</sub>t <sub>rewar</sub>d <sub>magn</sub>it<sub>u</sub>d<sub>e</sub> i<sub>s</sub> $R _ { \mathrm { m a x } } ~ =$ $\mathrm { m a x } _ { s , a } | r ( s , a ) | \ = \ 1 0 4$ <sub>, a</sub>tt<sub>a</sub>i<sub>ne</sub>d <sub>a</sub>t $T = 1 0 , H = 5 , \dot { L } = 4$ <sub>un</sub>d<sub>er cool-high, an</sub>d th<sub>e</sub> di<sub>scoun</sub>t i<sub>s</sub> $\gamma \ : = \ : 0 . 9 5$ , <sub>g</sub><sup>i</sup>v<sup>i</sup>n<sub>g</sub> $V _ { \mathrm { m a x } } ~ =$ $R _ { \mathrm { { m a x } } } / ( 1 - \gamma ) = 2 0 8 0$

## K.6 UAV benchmark

Thi<sub>s</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>mo</sub>d<sub>e</sub>l<sub>s a</sub> UAV fl<sub>y</sub>i<sub>ng</sub> th<sub>roug</sub>h <sub>a c</sub>l<sub>u</sub>tt<sub>ere</sub>d th<sub>ree-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>gr</sub>id f<sub>rom a</sub> fi<sub>xe</sub>d <sub>s</sub>t<sub>ar</sub>t t<sub>o a</sub> l<sub>an</sub>di<sub>ng pa</sub>d<sub>, un</sub>d<sub>er w</sub>i<sub>n</sub>d d<sub>r</sub>ift <sub>an</sub>d <sub>ac</sub>t<sub>ua</sub>t<sub>or-</sub>f<sub>a</sub>il<sub>ure r</sub>i<sub>s</sub>k<sub>.</sub>

State space. A grid state $s = ( x , y , z )$ <sup>ran</sup>g<sup>es o</sup>v<sup>er</sup> $\{ 0 , \ldots , L _ { x } { - } 1 \} \times \{ 0 , \ldots , L _ { y } { - } 1 \} \times \{ 0 , \ldots , L _ { z } { - } 1 \}$ , where x is the direction of travel, y the lateral axis the wind blows across, and z the altitude with $z = 0$ <sub>a</sub>t <sub>groun</sub>d l<sub>eve</sub>l<sub>.</sub> Th<sub>ree</sub> f<sub>ur</sub>th<sub>er s</sub>t<sub>a</sub>t<sub>es comp</sub>l<sub>e</sub>t<sub>e</sub> th<sub>e</sub> <sub>space: an a</sub>b<sub>sor</sub>bi<sub>ng</sub> C<sub>rash s</sub>t<sub>a</sub>t<sub>e, a</sub> G<sub>oal s</sub>t<sub>a</sub>t<sub>e, an</sub>d th<sub>e</sub> t<sub>erm</sub>i<sub>na</sub>l <sub>s</sub>i<sub>n</sub>k <sub>done</sub> th<sub>a</sub>t G<sub>oal moves</sub> t<sub>o.</sub> Th<sub>e s</sub>t<sub>ar</sub>t i<sub>s</sub> $s _ { 0 } = ( 0 , \lfloor L _ { y } / 2 \rfloor , 1 )$

Action space. In every grid cell, the UAV chooses one of seven actions,

$$
\begin{array} { r } { \mathcal { A } = \{ \mathrm { E } , \mathrm { W } , \mathrm { N } , \mathrm { S } , \mathrm { U P } , \mathrm { D O W N } , \mathrm { H O V E R } \} , } \end{array}
$$

th<sub>e</sub> <sub>s</sub>i<sub>x</sub> <sub>un</sub>it <sub>moves</sub> <sub>a</sub>l<sub>ong</sub> $\pm x , \pm y , \pm z$ t<sub>o e</sub>th<sub>er w</sub>ith <sub>a no-o .</sub> E <sub>a</sub>d<sub>vances</sub> t<sub>owar</sub>d th<sub>e oa</sub>l N <sub>an</sub>d S <sub>s</sub>t<sub>eer</sub> l<sub>a</sub>t<sub>era</sub>ll UP <sub>an</sub>d DOWN t<sub>ra</sub>d<sub>e a</sub>ltit<sub>u</sub>d<sub>e aga</sub>i<sub>ns</sub>t <sub>exposure</sub> t<sub>o</sub> th<sub>e</sub> t<sub>wo</sub> h<sub>azar</sub>d<sub>s, an</sub>d HOVER h<sub>o</sub>ld<sub>s pos</sub>iti<sub>on w</sub>hil<sub>e</sub> th<sub>e</sub> di<sub>s</sub>t<sub>ur</sub>b<sub>ances s</sub>till <sub>ac</sub>t<sub>.</sub> G<sub>oal o</sub>f<sub>ers</sub> th<sub>e</sub> <sub>s</sub>i<sub>ng</sub>l<sub>e</sub> f<sub>orce</sub>d <sub>ac</sub>ti<sub>on collect</sub> l<sub>ea</sub>di<sub>ng</sub> t<sub>o done, an</sub>d C<sub>rash an</sub>d <sub>done o</sub>f<sub>er on</sub>l<sub>y a se</sub>lf<sub>-</sub>l<sub>oop.</sub>

Parameter space. Two parameters are unknown, $\theta = ( p , q ) \in \Theta = [ 0 , p _ { \operatorname* { m a x } } ] \times [ 0 , q _ { \operatorname* { m a x } } ]$ . The wind intensity p is the chance that a step is overridden by a one-cell lateral drift, and the actuator-drop probability q is the probability that it is overridden by l<sub>os</sub>i<sub>ng a</sub> l<sub>eve</sub>l <sub>o</sub>f <sub>a</sub>ltit<sub>u</sub>d<sub>e.</sub> Th<sub>e genera</sub>t<sub>or requ</sub>i<sub>res</sub> $p _ { \operatorname* { m a x } } + q _ { \operatorname* { m a x } } < 1 , \mathrm { s o } ~ 1 - p - q > 0$ <sub>a</sub>t <sub>every a</sub>d<sub>m</sub>i<sub>ss</sub>ibl<sub>e va</sub>l<sub>ua</sub>ti<sub>on an</sub>d <sub>every</sub> b<sub>ranc</sub>h b<sub>e</sub>l<sub>ow</sub> i<sub>s</sub> <sub>a</sub> <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>w</sub>ith <sub>no</sub> <sub>c</sub>l<sub>amp</sub>i<sub>ng</sub> <sub>or</sub> <sub>renorma</sub>li<sub>za</sub>ti<sub>on.</sub> Th<sub>e</sub> t<sub>wo</sub> <sub>parame</sub>t<sub>ers</sub> <sub>pena</sub>li<sub>ze</sub> <sub>oppos</sub>it<sub>e</sub> <sub>rou</sub>t<sub>es,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>w</sub>h<sub>a</sub>t <sub>ma</sub>k<sub>es</sub> th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k i<sub>n</sub>t<sub>eres</sub>ti<sub>ng:</sub> <sub>w</sub>i<sub>n</sub>d h<sub>ur</sub>t<sub>s</sub> th<sub>e</sub> l<sub>ow</sub> <sub>corr</sub>id<sub>or,</sub> <sub>a</sub>ltit<sub>u</sub>d<sub>e</sub> d<sub>rops</sub> h<sub>ur</sub>t th<sub>e</sub> hi<sub>g</sub>h <sub>cross</sub>i<sub>ng.</sub>

Transitions. Writing clip(·) for componentwise clamping into the grid bounds, every grid action has the same three-branch f<sub>orm</sub>

$$
\mathbb { P } _ { \theta } \left( s ^ { \prime } \mid s , a \right) = \left( 1 - p - q \right) \mathbf { 1 } \left[ s ^ { \prime } = \mathrm { r e d i r e c t } ( s _ { \mathrm { i n t } } ) \right] + p \mathbf { 1 } \left[ s ^ { \prime } = \mathrm { r e d i r e c t } ( s _ { \mathrm { w i n d } } ) \right] + q \mathbf { 1 } \left[ s ^ { \prime } = \mathrm { r e d i r e c t } ( s _ { \mathrm { d r o p } } ) \right] ,
$$

<sub>w</sub>h<sub>ere</sub> $s _ { \mathrm { i n t } } = \mathrm { c l i p } ( s + \mathrm { m o v e } ( a ) )$ is the intended move, the only branch that depends on a; $s _ { \mathrm { w i n d } } = \mathrm { c l i p } ( s + ( 0 , 1 , 0 ) )$ i<sub>s</sub> th<sub>e</sub> l<sub>a</sub>t<sub>era</sub>l d<sub>r</sub>ift<sub>; an</sub>d $s _ { \mathrm { d r o p } } = ( x , y , z - 1 ) { \mathrm { ~ i f ~ } } z > 0 ,$ , else s itself, a ground-level skid. Coincident branches are summed, as in the datacenter benchmark. The ma redirect $( x , y , z )$ sends a candidate cell to Crash if it lies in the obstacle set O, to Goal if it lies in the tar<sub>g</sub>et re<sub>g</sub>ion T, and to the <sub>g</sub>rid state $( x , y , z )$ <sub>o</sub>th<sub>erw</sub>i<sub>se.</sub>

Th<sub>e</sub> l<sub>ayou</sub>t i<sub>s proce</sub>d<sub>ura</sub>l i<sub>n</sub> th<sub>e gr</sub>id <sub>ex</sub>t<sub>en</sub>t<sub>s.</sub> With $y _ { c } = \lfloor L _ { y } / 2 \rfloor , w _ { x } = \operatorname* { m a x } ( 1 , \lfloor L _ { x } / 6 \rfloor ) , x _ { w } = \lfloor L _ { x } / 2 \rfloor - \lfloor w _ { x } / 2 \rfloor$ $x _ { p } = \lfloor 3 \bar { L } _ { x } / 4 \rfloor$ <sub>, an</sub>d $x _ { c } = L _ { x } - \mathrm { m a x } ( 2 , \lfloor L _ { x } / 8 \rfloor )$ <sub>,</sub> th<sub>e o</sub>b<sub>s</sub>t<sub>ac</sub>l<sub>e se</sub>t $\mathcal { O } \stackrel { = } { = } \mathcal { O } _ { \mathrm { s l a b } } \cup \mathcal { O } _ { \mathrm { p i l l a r } } \cup \mathcal { O } _ { \mathrm { c a n o p y } }$ <sub>cons</sub>i<sub>s</sub>t<sub>s o</sub>f <sub>a wa</sub>ll <sub>spann</sub>i<sub>ng</sub> $x ^ { \prime } \in \left[ x _ { w } , x _ { w } \mathrm { + } w _ { x } \right)$ <sub>a</sub>t <sub>every a</sub>ltit<sub>u</sub>d<sub>e</sub> $z \leq L _ { z } { - } \bar { 2 }$ <sub>excep</sub>t <sub>a</sub> t<sub>wo-</sub>l<sub>eve</sub>l <sub>corr</sub>id<sub>or on</sub> th<sub>e cen</sub>t<sub>er</sub>li<sub>ne</sub> $\hat { ( \boldsymbol { y } ) } = \boldsymbol { y } _ { c } , z \le 1 )$ <sub>a</sub> t<sub>wo-w</sub>id<sub>e</sub> <sub>p</sub>ill<sub>ar</sub> <sub>a</sub>t $x = x _ { p } , y \in \{ y _ { c } , y _ { c } + 1 \} , z \leq L _ { z } - 2 ;$ <sub>; an</sub>d <sub>a canopy a</sub>t th<sub>e</sub> t<sub>op a</sub>ltit<sub>u</sub>d<sub>e</sub> $z = L _ { z } - 1$ <sub>res</sub>t<sub>r</sub>i<sub>c</sub>t<sub>e</sub>d t<sub>o</sub> $x \geq x _ { c } .$ <sub>.</sub> Th<sub>e</sub> t<sub>a</sub>r<sub>ge</sub>t re<sub>g</sub><sup>i</sup>on $\mathcal T = \{ ( \dot { L } _ { x } - 1 , \dot { y _ { , } } 0 ) : y \in \{ y _ { c } - 1 , y _ { c } , y _ { c } + 1 \} \}$ i<sub>s a</sub> th<sub>ree-ce</sub>ll l<sub>an</sub>di<sub>ng pa</sub>d <sub>on</sub> th<sub>e</sub> f<sub>ar wa</sub>ll<sub>.</sub> Th<sub>e genera</sub>t<sub>or c</sub>h<sub>ec</sub>k<sub>s</sub> th<sub>a</sub>t th<sub>e</sub> <sub>s</sub>t<sub>ar</sub>t i<sub>s no</sub>t <sub>an o</sub>b<sub>s</sub>t<sub>ac</sub>l<sub>e an</sub>d th<sub>a</sub>t $\tau \cap \mathcal { O } = \emptyset$ <sub>.</sub> T<sub>oge</sub>th<sub>er</sub> th<sub>e s</sub>l<sub>a</sub>b<sub>, passa</sub>bl<sub>e on</sub>l<sub>y</sub> th<sub>roug</sub>h <sub>a corr</sub>id<sub>or expose</sub>d t<sub>o</sub> th<sub>e w</sub>i<sub>n</sub>d<sub>, an</sub>d th<sub>e</sub> canopy, blocking the high route just before the goal, force a trade-of between a short low-altitude route hurt by p and a longer high-altitude route hurt by q, so the optimal route switches with $( p , q )$

Rewards and discount. The reward is deterministic and parameter-independent: $r ( s , a ) = 1$ for (Goal, collect) and 0 for <sub>every o</sub>th<sub>er reac</sub>h<sub>a</sub>bl<sub>e s</sub>t<sub>a</sub>t<sub>e-ac</sub>ti<sub>on pa</sub>i<sub>r, so</sub> $R _ { \operatorname* { m a x } } = 1$ . The optimal discounted value at s is therefore exactly the discounted <sub>pro</sub>b<sub>a</sub>bilit<sub>y</sub> <sub>o</sub>f <sub>even</sub>t<sub>ua</sub>ll<sub>y</sub> l<sub>an</sub>di<sub>ng.</sub> Th<sub>e</sub> di<sub>scoun</sub>t i<sub>s</sub> $\gamma = 0 . 9 9$ , kept close to 1 because routes take on the order of $L _ { x }$ ste<sub>p</sub>s an<sup>d</sup> a <sub>sma</sub>ll<sub>er</sub> di<sub>scoun</sub>t <sub>wou</sub>ld <sub>was</sub>h <sub>ou</sub>t th<sub>e reac</sub>h<sub>a</sub>bilit<sub>y s</sub>i<sub>gna</sub>l <sub>over</sub> th<sub>a</sub>t h<sub>or</sub>i<sub>zon;</sub> thi<sub>s g</sub>i<sub>ves</sub> $V _ { \mathrm { m a x } } = R _ { \mathrm { m a x } } / ( 1 - \gamma ) = 1 0 0$

Scalability. The grid extents $\left( L _ { x } , L _ { y } , L _ { z } \right)$ are free parameters of the generator, subject to $L _ { x } \geq 8 , L _ { z } \geq 3$ <sub>, an</sub>d <sub>o</sub>dd $L _ { y } \geq 5$ <sub>so</sub> th<sub>a</sub>t th<sub>e cen</sub>t<sub>er</sub>li<sub>ne</sub> $y _ { c }$ i<sub>s a s</sub>i<sub>ng</sub>l<sub>e co</sub>l<sub>umn.</sub> E<sub>very o</sub>b<sub>s</sub>t<sub>ac</sub>l<sub>e an</sub>d <sub>goa</sub>l f<sub>ormu</sub>l<sub>a a</sub>b<sub>ove</sub> d<sub>epen</sub>d<sub>s on</sub>l<sub>y on</sub> $\left( L _ { x } , L _ { y } , L _ { z } \right)$ <sub>,</sub> <sub>so</sub> th<sub>e</sub> l<sub>ow-versus-</sub>hi<sub>g</sub>h t<sub>ra</sub>d<sub>e-o</sub>f <sub>surv</sub>i<sub>ves a</sub>t <sub>every s</sub>i<sub>ze, an</sub>d

$$
| S | = 3 + \left| \{ ( x , y , z ) \in \mathrm { g r i d } : ( x , y , z ) \notin \mathcal { O } \cup \mathcal { T } \} \right| = O ( L _ { x } L _ { y } L _ { z } ) .
$$

Th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>comes</sub> i<sub>n</sub> th<sub>ree s</sub>i<sub>zes,</sub> li<sub>s</sub>t<sub>e</sub>d i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 4<sub>, an</sub>d <sub>a</sub>ll <sub>s</sub>h<sub>are</sub> $( p _ { \mathrm { m a x } } , q _ { \mathrm { m a x } } , \gamma ) = ( 0 . 2 5 , 0 . 2 0 , 0 . 9 9 )$ <sub>an</sub>d th<sub>e</sub> l<sub>ayou</sub>t <sub>a</sub>b<sub>ove,</sub> dif<sub>er</sub>i<sub>ng</sub> <sub>on</sub>l<sub>y</sub> i<sub>n</sub> $( L _ { x } , L _ { y } , L _ { z } )$

<table><tr><td>Name</td><td> $\left( L _ { x } , L _ { y } , L _ { z } \right)$ </td><td>|S|</td></tr><tr><td>uav-small</td><td>(8, 5, 3)</td><td>98</td></tr><tr><td>uav-medium</td><td>(12,9,4)</td><td>358</td></tr><tr><td>uav-large</td><td>(24, 15, 6)</td><td>1,813</td></tr></table>

Table 4: Sizes of the uav benchmark famil<sub>y</sub> used in the ex<sub>p</sub>eriments.

E<sub>very gr</sub>id <sub>ce</sub>ll <sub>o</sub>f<sub>ers</sub> th<sub>e same seven ac</sub>ti<sub>ons, so</sub> th<sub>e</sub> f<sub>am</sub>il<sub>y a</sub>d<sub>m</sub>it<sub>s</sub> $\gamma ^ { | { \cal S } | - 3 }$ <sub>memory</sub>l<sub>ess</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>s</sub>ti<sub>c po</sub>li<sub>c</sub>i<sub>es.</sub> E<sub>ven</sub> uav-small therefore rules out exhaustive search over <sub>p</sub>olicies.

## L Experimental data

Regret results (RQ1). Table 5 ives the seed-level results summarized in Table 2. For each ortfolio size K and each b<sub>enc</sub>h<sub>mar</sub>k<sub>,</sub> it <sub>repor</sub>t<sub>s</sub> t<sub>wo quan</sub>titi<sub>es.</sub> Th<sub>e</sub> “<sub>max-regre</sub>t” <sub>co</sub>l<sub>umn</sub> i<sub>s</sub> $\widehat { \mathrm { R r e g } _ { \widehat { D } } } ( \Pi _ { K } )$ , the largest portfolio regret over that seed’s 1000 <sub>samp</sub>l<sub>e</sub>d <sub>va</sub>l<sub>ua</sub>ti<sub>ons;</sub> th<sub>ese are</sub> th<sub>e num</sub>b<sub>ers average</sub>d i<sub>n</sub>t<sub>o</sub> T<sub>a</sub>bl<sub>e</sub> 2<sub>, an</sub>d th<sub>ey s</sub>h<sub>ou</sub>ld b<sub>e rea</sub>d <sub>aga</sub>i<sub>ns</sub>t th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k’<sub>s</sub> $V _ { \mathrm { m a x } } .$ li<sub>s</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> h<sub>ea</sub>d<sub>er, s</sub>i<sub>nce</sub> th<sub>e</sub> d<sub>a</sub>t<sub>acen</sub>t<sub>er va</sub>l<sub>ues</sub> li<sub>ve on a sca</sub>l<sub>e over</sub> t<sub>wen</sub>t<sub>y</sub> ti<sub>mes</sub> l<sub>arger</sub> th<sub>an</sub> th<sub>e</sub> UAV <sub>ones.</sub> Th<sub>e</sub> “i<sub>ner</sub>ti<sub>a</sub>” <sub>co</sub>l<sub>umn</sub> i<sub>s</sub> the K-means objective, the within-cluster sum of squared distances from each profile to its assigned center. It measures how tightly the loss profiles cluster, not how well the portfolio performs, and is included only as a diagnostic: it falls steeply with K <sub>on</sub> <sub>every</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>,</sub> <sub>con</sub>fi<sub>rm</sub>i<sub>ng</sub> th<sub>a</sub>t th<sub>e</sub> <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>pro</sub>fil<sub>es</sub> d<sub>o</sub> <sub>separa</sub>t<sub>e</sub> i<sub>n</sub>t<sub>o</sub> di<sub>s</sub>ti<sub>ngu</sub>i<sub>s</sub>h<sub>a</sub>bl<sub>e</sub> <sub>groups.</sub>

Online selection results (RQ2). Figure 2 in the main text shows the two datacenter measures. Figures 14 and 15 show the same pair of measures for all four benchmarks, one benchmark per row, with all values of K overlaid within each panel. The left column is the fraction of the 90 ooled runs whose recommended member is the best ortfolio member at their hidden valuation; th<sub>e r</sub>i ht <sub>co</sub>l<sub>umn</sub> i<sub>s</sub> th<sub>a</sub>t <sub>recommen</sub>d<sub>a</sub>ti<sub>on</sub>’<sub>s norma</sub>li<sub>ze</sub>d <sub>s</sub>h<sub>or</sub>tf<sub>a</sub>ll $L _ { \theta } ^ { \pi }$ , which is 0 when the best member is recommended and 1 <sub>w</sub>h<sub>en</sub> th<sub>e wors</sub>t i<sub>s.</sub> R<sub>ea</sub>di<sub>ng</sub> th<sub>e</sub> t<sub>wo co</sub>l<sub>umns</sub> t<sub>oge</sub>th<sub>er</sub> i<sub>s w</sub>h<sub>a</sub>t th<sub>e</sub> di<sub>scuss</sub>i<sub>on</sub> i<sub>n</sub> S<sub>ec</sub>ti<sub>on</sub> 4 <sub>re</sub>li<sub>es on:</sub> th<sub>e</sub> l<sub>e</sub>ft <sub>co</sub>l<sub>umn</sub> d<sub>egra</sub>d<sub>es</sub> visibl as K rows, since more members mean more near-ties to resolve, while the ri ht column sta s low throu hout, showin th<sub>a</sub>t th<sub>e mem</sub>b<sub>ers con</sub>f<sub>use</sub>d f<sub>or one ano</sub>th<sub>er are c</sub>l<sub>ose</sub> i<sub>n va</sub>l<sub>ue an</sub>d th<sub>e cos</sub>t <sub>o</sub>f th<sub>e con</sub>f<sub>us</sub>i<sub>on</sub> i<sub>s sma</sub>ll<sub>.</sub>

<table><tr><td rowspan="5"></td><td rowspan="5"></td><td colspan="2">uav-small</td><td colspan="2">uav-medium</td><td colspan="2">uav-large</td><td colspan="2">data-center-climate</td></tr><tr><td colspan="2">|S| = 98</td><td colspan="2"> $\vert { S } \vert = 3 5 8$ </td><td colspan="2"> $\vert S \vert = 1 8 1 3$ </td><td colspan="2"> $| S | = 3 3 0$ </td></tr><tr><td colspan="2"> $V _ { \mathrm { m a x } } ^ { \cdot } = 1 0 0$ </td><td colspan="2"> $\dot { V } _ { \mathrm { m a x } } = 1 0 0$ </td><td colspan="2"> $\dot { V } _ { \mathrm { m a x } } = 1 0 0$ </td><td colspan="2"> $V _ { \mathrm { m a x } } = 2 0 8 0$ </td></tr><tr><td>max-regret</td><td>inertia</td><td>max-regret</td><td>inertia</td><td>max-regret</td><td>inertia</td><td>max-regret</td><td>inertia</td></tr><tr><td>1</td><td>0</td><td>0.053</td><td>23.239</td><td>0.091</td><td>42.668</td><td>0.125</td><td>59.498</td><td>14.649</td><td>1313595.187</td></tr><tr><td>1</td><td>1</td><td>0.053</td><td>23.239</td><td>0.091</td><td>42.668</td><td>0.124</td><td>59.498</td><td>13.404</td><td>1313595.187</td></tr><tr><td>1</td><td>2</td><td>0.053</td><td>23.239</td><td>0.091</td><td>42.668</td><td>0.129</td><td>59.498</td><td>14.771</td><td>1313595.187</td></tr><tr><td>1</td><td>avg</td><td>0.053</td><td>23.239</td><td>0.091</td><td>42.668</td><td>0.126</td><td>59.498</td><td>14.275</td><td>1313595.187</td></tr><tr><td>2</td><td>0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>4.127</td><td>215193.617</td></tr><tr><td>2</td><td>1</td><td>0.032 0.032</td><td>1.935 1.935</td><td>0.019 0.019</td><td>3.974 3.974</td><td>0.016 0.016</td><td>1.629 1.629</td><td>3.413</td><td>220731.474</td></tr><tr><td>2</td><td>2</td><td>0.032</td><td>1.935</td><td>0.020</td><td>3.974</td><td>0.018</td><td>1.629</td><td>4.053</td><td>215193.617</td></tr><tr><td>2</td><td>avg</td><td>0.032</td><td>1.935</td><td>0.019</td><td>3.974</td><td>0.017</td><td>1.629</td><td>3.864</td><td>217039.569</td></tr><tr><td>3</td><td>0</td><td>0.014</td><td>0.697</td><td>0.005</td><td>1.951</td><td>0.016</td><td>0.638</td><td>2.813</td><td>72737.797</td></tr><tr><td>3</td><td>1</td><td>0.027</td><td>1.157</td><td>0.005</td><td>1.951</td><td>0.014</td><td>1.317</td><td>2.836</td><td>72737.797</td></tr><tr><td>3</td><td>2</td><td>0.014</td><td>0.697</td><td>0.020</td><td>2.152</td><td>0.018</td><td>0.638</td><td>2.809</td><td>72737.797</td></tr><tr><td>3</td><td>avg</td><td>0.018</td><td>0.850</td><td>0.010</td><td>2.018</td><td>0.016</td><td>0.865</td><td>2.820</td><td>72737.797</td></tr><tr><td>5</td><td>0</td><td>0.002</td><td>0.094</td><td></td><td></td><td></td><td></td><td>2.809</td><td></td></tr><tr><td>5</td><td>1</td><td>0.002</td><td>0.093</td><td>0.003 0.003</td><td>0.193 0.183</td><td>0.016 0.014</td><td>0.107 0.107</td><td>2.836</td><td>37639.989 40156.574</td></tr><tr><td>5</td><td>2</td><td>0.002</td><td>0.093</td><td>0.003</td><td>0.183</td><td>0.015</td><td>0.107</td><td>2.730</td><td>39166.131</td></tr><tr><td>5</td><td>avg</td><td>0.002</td><td>0.093</td><td>0.003</td><td>0.186</td><td>0.015</td><td>0.107</td><td>2.792</td><td>38987.564</td></tr><tr><td>7</td><td>0</td><td>0.002</td><td>0.015</td><td>0.003</td><td>0.032</td><td>0.004</td><td>0.044</td><td>2.809</td><td>19076.922</td></tr><tr><td>7</td><td>1</td><td>0.002</td><td>0.012</td><td>0.003</td><td>0.032</td><td>0.003</td><td>0.044</td><td>1.170</td><td>20262.995</td></tr><tr><td>7</td><td>2</td><td>0.002</td><td>0.012</td><td>0.003</td><td>0.032</td><td>0.015</td><td>0.048</td><td>1.392</td><td>19462.184</td></tr><tr><td>7</td><td>avg</td><td>0.002</td><td>0.013</td><td>0.003</td><td>0.032</td><td>0.008</td><td>0.045</td><td>1.790</td><td>19600.700</td></tr><tr><td>10</td><td>0</td><td>0.002</td><td>0.002</td><td>0.003</td><td>0.007</td><td>0.004</td><td>0.010</td><td>0.817</td><td>8950.955</td></tr><tr><td>10</td><td>1</td><td>0.002</td><td>0.002</td><td>0.003</td><td>0.007</td><td>0.003</td><td>0.011</td><td>0.824</td><td>9239.430</td></tr><tr><td>10</td><td>2</td><td>0.002</td><td>0.002</td><td>0.003</td><td>0.009</td><td>0.002</td><td>0.010</td><td>0.759</td><td>8610.714</td></tr><tr><td>10</td><td>avg</td><td>0.002</td><td>0.002</td><td>0.003</td><td>0.008</td><td>0.003</td><td>0.011</td><td>0.800</td><td>8933.700</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 5: Full re<sub>g</sub>ret results (RQ1). For each <sub>p</sub>ortfolio size K, rows 0–2 re<sub>p</sub>ort the three clusterin<sub>g</sub> seeds and “av<sub>g</sub>” their arithmetic <sub>mean.</sub> “<sub>max-regre</sub>t” i<sub>s</sub> th<sub>e</sub> l<sub>arges</sub>t <sub>samp</sub>l<sub>e</sub>d <sub>por</sub>tf<sub>o</sub>li<sub>o regre</sub>t<sub>.</sub> “I<sub>ner</sub>ti<sub>a</sub>” i<sub>s</sub> th<sub>e</sub> K<sub>-means w</sub>ithi<sub>n-c</sub>l<sub>us</sub>t<sub>er sum o</sub>f <sub>square</sub>d di<sub>s</sub>t<sub>ances</sub> b<sub>e</sub>t<sub>ween</sub> d<sub>a</sub>t<sub>a</sub> <sub>po</sub>i<sub>n</sub>t<sub>s</sub> <sub>an</sub>d th<sub>e</sub> <sub>c</sub>l<sub>us</sub>t<sub>er</sub> <sub>cen</sub>t<sub>ro</sub>id<sub>,</sub> <sub>no</sub>t <sub>a</sub> <sub>regre</sub>t <sub>measure.</sub>

![](images/652646d265641179ab32c7ce7630cb16af63c7d803a581395a263aab6b49b419.jpg)  
Fi<sub>g</sub>ure 14: Re<sub>p</sub>orted UCB de<sub>p</sub>lo<sub>y</sub>ment results across all tested values of K, for the datacenter and uav-small benchmark (RQ2). E<sub>ac</sub>h <sub>row correspon</sub>d<sub>s</sub> t<sub>o a</sub> b<sub>enc</sub>h<sub>mar</sub>k<sub>; co</sub>l<sub>umns s</sub>h<sub>ow,</sub> l<sub>e</sub>ft t<sub>o r</sub>i<sub>g</sub>ht<sub>,</sub> th<sub>e repor</sub>t<sub>e</sub>d <sub>correc</sub>t<sub>-arm ra</sub>t<sub>e</sub> f<sub>or</sub> th<sub>e recommen</sub>d<sub>e</sub>d <sub>arm an</sub>d th<sub>e regre</sub>t <sub>o</sub>f th<sub>e recommen</sub>d<sub>e</sub>d <sub>arm.</sub>

![](images/c37b75982c58e56e3b80fa20b27a2baf3399a50eb2778aa0de139bd7b0633dc9.jpg)  
Figure 15: Reported UCB deployment results across all tested values of K, for the UAV-medium and UAV-large benchmarks (RQ2). Each row corres<sub>p</sub>onds to a benchmark; columns show, left to ri<sub>g</sub>ht, the re<sub>p</sub>orted correct-arm rate for the recommended <sub>arm an</sub>d th<sub>e regre</sub>t <sub>o</sub>f th<sub>e recommen</sub>d<sub>e</sub>d <sub>arm.</sub>

Algorithm 2: Paired portfolio synthesis and evaluation protocol   
Input: benchmark pMDP M, domain D, portfolio budgets K, seeds S, empirical robust regret sample size n, UCB   
sample size m, UCB parameters (H, ε, δ)   
foreach K ∈ K do   
// Offline candidate construction.   
Cells ← DiscretizeDomain(D);   
Candidates ← ComputeCandidatePolicies(M, Cells);   
// Offline robust candidate evaluation.   
Scores ← RobustPolicyEvaluation(M, Cells, Candidates);   
// Offline mini-max regret evaluation.   
Com<sub>p</sub>uteMinimaxRe<sub>g</sub>ret(Scores, Cells, Candidates);   
foreach s ∈ S do   
// Offline construction of portfolio.   
Π ← ReduceAndCluster(Candidates, Scores, K, s);   
// Offline empirical robust regret evaluation of portfolio.   
Db ← SampleDomain(D, n, s);   
foreach θ ∈ Db do   
V<sup>∗</sup> ← OptimalValue(M, θ);   
V<sub>Π</sub> ← max<sub>π∈Π</sub> EvaluatePolicy(M, θ, π);   
r ← V<sup>∗</sup> − V ;   
Store(K, s, θ, r);   
M<sub>ax</sub>i<sub>m</sub>i<sub>se emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t <sub>w</sub>ithi<sub>n one see</sub>d<sub>;</sub>   
// Online UCB-based deployment.   
Db ← SampleDomain(D, m, s);   
foreach θ ∈ Db do   
Create generative model G<sub>θ</sub>;   
z ← RunUCB(G<sub>θ</sub>, Π, H, ε, δ);   
Store(K, s, θ, z);   
A<sub>verage emp</sub>i<sub>r</sub>i<sub>ca</sub>l <sub>ro</sub>b<sub>us</sub>t <sub>regre</sub>t <sub>over see</sub>d<sub>s;</sub>   
P<sub>oo</sub>l UCB <sub>sa</sub>m<sub>p</sub>l<sub>es ac</sub>r<sub>oss see</sub>d<sub>s;</sub>   
GenerateReport();