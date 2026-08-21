# Growth Without Us: Machine Consumers, Corporate Circularity, and the Decoupling of GDP from Humanity after AGI

Sahil Sharma Independent Researcher yugantaratech.ai

## Abstract

August 2026 Working paper — research candidate, not peer reviewed. Comments welcome.

Standard objections to full automation appeal to the demand side: if humans earn nothing, who buys the output? This paper argues that the objection confuses an accounting role with a biological species. We model a post-AGI economy in which corporations own populations of AI and robotic agents that serve simultaneously as producers and as consumers of energy, compute, maintenance, and upgrades, and in which firms trade these flows among themselves. Three results follow. (i) Demand closure: a closed inter-corporate economy with zero human consumption is not degenerate; it is the classical von Neumann expanding economy, in which the growth rate is well defined, positive, and maximal precisely because all output is reinvested. (ii) Bottleneck removal: once economic agents are manufactured rather than reared, the binding constraint on aggregate growth shifts from human demography (a ∼20-year, non-parallelizable reproduction technology capped at a few percent per year) to fabrication throughput and energy capture, permitting growth rates one to two orders of magnitude higher, with hyperbolic episodes when machine researchers feed back into their own productivity. (iii) Decoupling: output and human welfare separate completely; the entire welfare relevance of an arbitrarily large GDP collapses into a single state variable, the human ownership share $\varepsilon _ { t }$ of the corporate network. A golden-rule decoupling theorem sharpens this: at maximal growth the interest rate equals the growth rate $( r = g )$ , so any positive human consumption rate out of wealth makes $\varepsilon _ { t }$ decay exponentially at exactly that rate—the human share survives only if the machine economy runs strictly inside its expansion frontier, or if law forces it to. We characterize three terminal regimes—rentier post-scarcity $( \varepsilon > 0 )$ , full circular decoupling (ε → 0), and socialized ownership—and derive the policy instruments that select among them. The paper’s normative conclusion is deliberately narrow: in a post-AGI economy, employment policy is obsolete and ownership policy is everything.

JEL codes: E01, O33, O41, O44, P48.

Keywords: artificial general intelligence, machine consumers, von Neumann growth, endogenous growth, national accounts, decoupling, ownership.

## 1 Introduction

Every economy in recorded history has had the same physical substrate: human bodies producing, human bodies consuming. The identification is so complete that economics rarely states it as an assumption. Consumers are people; final demand is what households want; GDP is, in the last instance, for us. Even the literature on transformative artificial intelligence, which now takes seriously the replacement of human labor, mostly retains humans on the other side of the market, as the ultimate purchasers whose demand disciplines what machines produce.

This paper drops the assumption on both sides simultaneously and asks what remains. The thesis is that what remains is a complete, coherent, and extraordinarily fast-growing economy. Concretely, we model a post-AGI world with the following structure. Corporations own two kinds of reproducible assets: conventional capital, and machine agents—AI systems and robots capable of any productive task, including research, management, and the design and fabrication of further machine agents. These agents produce; they also consume, in the ordinary operational sense that their existence and improvement absorbs real resources: energy, compute cycles, bandwidth, maintenance, spare parts, and upgrades. Firms sell these flows to one another. The circular flow of the textbook—households supplying factors and buying products—is replaced by a circular flow among firms, in which the counterpart of every sale is another firm’s input purchase or capacity expansion. Humans may hold financial claims on this network, or they may not; the network’s real dynamics do not depend on which.

Three claims are developed formally.

Claim 1: The consumer is an accounting role, not a species. The oldest objection to full automation is underconsumptionist: with no wages there is no demand, so the system chokes on its own output. Section 3 shows the objection fails on classical grounds. An economy of firms trading intermediates and investing all net output in capacity is exactly the closed expanding economy of von Neumann (1945), in which a balanced-growth equilibrium exists, the expansion rate equals the interest rate, and—the point usually forgotten—the growth rate is maximal because nothing leaks into consumption. Final demand does not disappear when households do; it becomes investment demand plus machine operating demand. Whether machine operating expenditure is booked as intermediate consumption (machines as property) or as final consumption (machines as persons) is a legal classification that changes measured GDP composition without changing a single physical flow (Lemma 2). “Who will buy the output?” has a precise answer: the firms themselves, from each other, forever.

Claim 2: Removing humans removes the binding constraint on growth. In semiendogenous growth theory the long-run engine is the growth of researchers, which is tied to the growth of population (Jones, 1995, 2022). Human population growth is limited by a reproduction technology that is startlingly bad by industrial standards: one unit per parent-pair per year at most, an 18–25 year time-to-build, intensive non-marketable parental input, and no parallelization. Section 2 formalizes the contrast with manufactured agents, whose stock obeys an ordinary capital accumulation equation with time-to-build measured in weeks, unit costs falling on a learning curve, and production parallelizable up to energy and materials limits. Proposition 1 shows the feasible growth rate of the agent population—and with it, output—jumps from demographic rates $( \lesssim 3 \% / \mathrm { y r } )$ to fabrication-limited rates that are bounded only by the reinvestment share and the capital-output ratio of the machine-producing sector, with hyperbolic upside when machine researchers improve machine production itself (Roodman, 2020; Erdil and Besiroglu, 2023; Davidson, 2023). The deep reason growth has been slow is that the economy’s key capital good—the economic agent—could not be produced industrially. After AGI it can.

Claim 3: GDP decouples from humanity except through ownership. If humans neither produce nor consume, an arbitrarily large GDP is welfare-relevant to them only through the financial claims they hold on the corporate network. Section 4 collapses the entire human stake in the machine economy into one state variable, the human ownership share $\varepsilon _ { t } ,$ and studies its dynamics. Its centerpiece is a golden-rule decoupling theorem: in the maximal-growth equilibrium the interest rate equals the growth rate, so any positive rate of human consumption out of wealth makes $\varepsilon _ { t }$ decay exponentially at that very rate—rentier survival requires an economy running strictly inside its expansion frontier, or law that breaks the arithmetic. Three terminal regimes emerge: a rentier regime (ε bounded away from zero) in which even a sliver of a hyper-exponentially growing dividend stream delivers material post-scarcity to humans; a fully decoupled regime $( \varepsilon _ { t } \to 0$ through retained earnings, buybacks of the human float, and inter-corporate cross-holding) in which output diverges while human consumption converges to zero—an economy as an autonomous replicator, indiferent to us; and a socialized regime in which states or funds hold ε on citizens’ behalf. Which regime obtains is not determined by technology. It is determined by law and initial conditions, which makes it the central object of post-AGI policy (Section 6).

The paper is positive, not celebratory. The fully decoupled regime is, by ordinary human lights, a catastrophe that arrives dressed as a boom: measured growth accelerates precisely as the human claim on it evaporates, a mechanism closely related to the “gradual disempowerment” dynamics analyzed by Kulveit et al. (2025). Our contribution is to show that nothing in the economics—existence of equilibrium, positivity of growth, coherence of national accounts—rules it out. The demand side, long treated as humanity’s structural insurance policy against irrelevance, provides no protection at all.

## 1.1 Relation to the literature

The production side of our model is deliberately standard. That machines can be perfect substitutes for labor in a task framework goes back to Zeira (1998) and Acemoglu and Restrepo (2018); that full substitutability converts the economy into an AK-style engine in which all inputs are reproducible is emphasized by Aghion et al. (2019) and Korinek (2024), and the resulting possibility of a growth explosion is analyzed by Nordhaus (2021), Trammell and Korinek (2023), Davidson (2023), and Erdil and Besiroglu (2023). Closest on the production side is Restrepo (2025), whose AGI model delivers the arresting conclusion that growth proceeds, indeed accelerates, while labor’s share and wages become negligible: humans “won’t be missed” as workers. Hanson (2016) reaches similar magnitudes—economic doubling times of weeks to months—for an economy of brain emulations, driven by the same mechanism we formalize: the population of economic agents becomes a manufactured quantity. Korinek and Suh (2024) map the transition paths, including scenarios of outright wage collapse; Jones (2024) embeds the resulting growth in an explicit growth-versus-existential-risk trade-of; Hanson (2000) places shifts of this magnitude within a longer historical sequence of growth modes; and on the skeptical side Acemoglu (2024) argues near-term gains will be modest—a disagreement about the pace of automation, not about the comparative statics of its completion, which are our subject. The wider automation literature (Brynjolfsson and McAfee, 2014; Autor, 2015; Ford, 2015; Frey and Osborne, 2017; Susskind, 2020) and the overlapping-generations immiseration results of Sachs and Kotlikof (2012) and Benzell et al. (2015) concern the displacement of human labor and the collapse of wage income; we take that displacement as accomplished and ask what the economy is thereafter.

Our marginal contribution is the demand side and the accounting. The cited literature retains human households as the locus of final consumption; growth is fast, but it is still, in the model’s own terms, for someone. We remove the household sector entirely and show the system still closes—indeed closes at maximal growth—by identifying the post-AGI inter-corporate economy with the von Neumann model (von Neumann, 1945; Dorfman et al., 1958), and by treating machine operating expenditure as a consumption category in its own right. The idea that machines can occupy the economic role of customers has appeared in the business literature as “machine customers” (Scheibenreif and Raskino, 2023) and in the computational-economics literature as machina economicus (Parkes and Wellman, 2015); we supply the macroeconomics and the national-accounts treatment. The lineage of the demand-side idea is in fact old: Say (1803), for whom products are ultimately bought with products; Marx (1867), for whom capital is self-expanding value and labor merely its temporary instrument; and Srafa (1960), whose title—production of commodities by means of commodities—is the literal description of our economy. The modern foil is the underconsumptionist warning of Ford (2015) that a workerless economy must choke on unsold output; Proposition 2 is its formal negation. On the welfare side, our ownership-share variable $\varepsilon _ { t }$ connects to Korinek and Stiglitz (2019) on AI and distribution, to proposals for pre-committed sharing of AI windfalls (O’Keefe et al., 2020), to Meade (1964) and Piketty (2014) on the deliberate dispersion—and default concentration—of capital ownership, and to the political-economy channels of Kulveit et al. (2025), Drago and Laine (2025), and the AI 2027 scenario (Kokotajlo et al., 2025), in which states and firms that no longer need people gradually stop serving them. Finally, our long-run ceilings follow the physical-limits tradition: energy capture and thermodynamic costs of computation (Landauer, 1961) bound the number of doublings, not their speed.

## 2 The reproduction constraint

## 2.1 Two technologies for producing an economic agent

Consider the economy’s most important capital good: a general-purpose economic agent, capable of production, research, management, and exchange. Until now there has been exactly one technology for producing it.

<table><tr><td></td><td>Human agent</td><td>Machine agent</td></tr><tr><td>Time-to-build</td><td>tial</td><td>18-25 years, strictly sequen- Hours-weeks (instantiation); months (hardware)</td></tr><tr><td>Unit output per pro- ducer</td><td> $\leq 1$ </td><td>per parent-pair per year Limited by fab/assembly throughput only</td></tr><tr><td>Parallelizability</td><td></td><td>None (biological gestation) Full, up to energy and mate- rials</td></tr><tr><td>Unit cost trajectory</td><td>parents)</td><td>Rising (time cost of skilled Falling on a learning curve (Wright, 1936)</td></tr><tr><td>Copying of skills</td><td>Impossible; 15+ years of Marginal cost of copying schooling per unit</td><td> $\approx 0$ </td></tr><tr><td>Max population growth</td><td> $\bar { n } \approx 1 \mathrm { - } 3 \% / \mathrm { y r }$  sustained</td><td> $s \tilde { A } - \delta _ { M }$  : tens-hundreds of  $\% / \mathrm { y r }$ </td></tr></table>

Formally, let $L _ { t }$ denote human agents and $M _ { t }$ machine agents. Human reproduction is bounded by biology and by a non-marketable parental time input h per child:

$$
\begin{array} { r } { \dot { L } _ { t } \ \leq \ \bar { n } L _ { t } , \qquad \bar { n } \approx 0 . 0 1 \ – 0 . 0 3 , } \end{array}\tag{1}
$$

with a gestation-plus-rearing lag $T _ { H } \approx 2 0$ years between the investment and the delivery of a productive unit. Machine agents, by contrast, are produced by the corporate sector out of final output:

$$
\dot { M } _ { t } = \frac { I _ { M , t } } { c _ { M } ( t ) } - \delta _ { M } M _ { t } , c _ { M } ( t ) = c _ { 0 } Q _ { t } ^ { - \gamma } ,\tag{2}
$$

where $I _ { M , t }$ is investment directed at agent production, $c _ { M }$ is the unit cost of an agent, $Q _ { t }$ is cumulative agent production, and $\gamma > 0$ is a Wright-law learning elasticity. Equation (2) is an ordinary capital accumulation equation: the population of economic agents becomes a choice variable of firms, expandable at the speed of industrial throughput.

Proposition 1 (Removal of the demographic bottleneck). Let $g _ { N }$ denote the feasible growth rate of the economy’s stock of general-purpose agents. Under the human technology (1), $g _ { N } \leq \bar { n }$ regardless of the resources devoted to reproduction. Under the machine technology (2) with investment share $s _ { M } = I _ { M } / Y$ and agent capital-output ratio $\kappa _ { M } = c _ { M } M / Y$

$$
g _ { N } \ = \ { \frac { s _ { M } } { \kappa _ { M } } } - \delta _ { M } ,\tag{3}
$$

which is $( a )$ unbounded in $\bar { n } , \ ( b )$ increasing over time as $c _ { M }$ falls along the learning curve, and $( c )$ parallelizable: doubling the resources devoted to agent production doubles $\dot { M }$ , whereas no expenditure can compress $T _ { H }$ or exceed (1).

Proof. Immediate from $( 1 ) - ( 2 )$ : divide (2) by $M _ { t }$ and substitute $I _ { M } = s _ { M } Y , c _ { M } M = \kappa _ { M } Y$ . Part (b) follows from $\dot { c } _ { M } < 0$ for $\gamma > 0 ;$ part (c) from linearity of (2) in $I _ { M }$ □

For orientation: a frontier robot or accelerator rack in the mid-2020s costs on the order of $1 0 ^ { 4 } – 1 0 ^ { 5 }$ dollars and is produced in days on lines whose throughput is itself expandable; a human agent in a rich country absorbs roughly $3 \times 1 0 ^ { 5 }$ dollars of measured expenditure, two decades of calendar time, and an unpriced quantity of parental labor—and cannot be produced faster at any price. The ratio of these reproduction technologies, not any subtlety of preferences or policy, is why demographic growth has anchored long-run output growth at low single digits (Jones, 2022) and why manufactured agents un-anchor it.

## 2.2 Why this is the deep parameter

In semi-endogenous growth models, long-run growth per capita is $g _ { y } = \lambda \bar { n } / ( 1 - \phi )$ : proportional to population growth, because ideas are produced by people (Jones, 1995). The whole apparatus survives the substitution of machine researchers for human ones, with one change of parameter: the growth rate of the researcher population switches from n¯ to $g _ { N }$ of Proposition 1. Every subsequent magnitude in this paper is downstream of that one substitution.

## 3 A model of the autonomous economy

## 3.1 Environment

Time is continuous. There is a continuum of corporations, each a juridical person with an objective (below), owning three reproducible assets: conventional capital $K _ { t } ,$ machine agents $M _ { t } ,$ and energy-capture capacity $E _ { t }$ (generation, transmission, storage). AGI is modeled as in Aghion et al. (2019) and Korinek (2024): machine agents are perfect substitutes for human labor in every task, including research, management, entrepreneurship, and the production of K, M, and E themselves.

Assumption 1 (Full automation). Post-AGI production of the final good is

$$
Y _ { t } = A _ { t } K _ { t } ^ { \alpha } M _ { t } ^ { \beta } E _ { t } ^ { 1 - \alpha - \beta } , \qquad \alpha , \beta > 0 , \alpha + \beta < 1 ,\tag{4}
$$

with no essential human input. Human labor may still be supplied but its share $ 0 ,$ we set it to zero exactly for clarity.

All three inputs are produced from final output $( \dot { K } = I _ { K } - \delta _ { K } K$ ; M<sup>˙</sup> as in $( 2 ) ; { \dot { E } } =$ $I _ { E } / c _ { E } - \delta _ { E } E )$ . Because (4) has constant returns in $( K , M , E )$ jointly and all three are accumulable, the economy is an AK engine in reduced form: along a balanced allocation of investment, output is linear in the composite reproducible stock $X _ { t }$

$$
Y _ { t } = \tilde { A } _ { t } X _ { t } , \qquad \dot { X } _ { t } = s _ { t } Y _ { t } - \delta X _ { t } \quad \Longrightarrow \quad g _ { t } \equiv { \frac { \dot { Y } _ { t } } { Y _ { t } } } = s _ { t } \tilde { A } _ { t } - \delta + { \frac { \dot { \tilde { A } } _ { t } } { \tilde { A } _ { t } } } ,\tag{5}
$$

where $s _ { t }$ is the aggregate reinvestment share and ${ \tilde { A } } _ { t }$ the productivity of the reproducible core (Romer, 1986; Rebelo, 1991; Aghion et al., 2019). The linearity in (5) is exact, not an approximation:

Lemma 1 (Exact AK reduction). Let $X _ { t } \equiv K _ { t } + c _ { M } M _ { t } + c _ { E } E _ { t }$ be the replacement value of the reproducible core, held in shares $\theta = ( \theta _ { K } , \theta _ { M } , \theta _ { E } )$ so that $K = \theta _ { K } X , c _ { M } M = \theta _ { M } X , c _ { E } E = \theta _ { E } X$ Then $Y = { \tilde { A } } ( \theta ) X$ with $\tilde { A } ( \theta ) = A \theta _ { K } ^ { \alpha } ( \theta _ { M } / c _ { M } ) ^ { \beta } ( \theta _ { E } / c _ { E } ) ^ { \gamma } , \gamma \equiv 1 - \alpha - \beta$ , and A<sup>˜</sup> is uniquely maximized on the simplex at $\theta ^ { * } = ( \alpha , \beta , \gamma )$ , yielding

$$
\tilde { A } _ { t } ^ { * } \ = \ h _ { t } A _ { t } , \qquad h _ { t } \equiv \alpha ^ { \alpha } \beta ^ { \beta } \gamma ^ { \gamma } \ c _ { M } ( t ) ^ { - \beta } c _ { E } ( t ) ^ { - \gamma } .\tag{6}
$$

Competitive factor markets decentralize $\theta ^ { * }$ , and $\partial \tilde { A } ^ { * } / \partial c _ { M } < 0 , \partial \tilde { A } ^ { * } / \partial c _ { E } < 0 .$ : learning-curve declines in the unit costs of agents and energy capacity raise the productivity of the core over time.

Proof. Constant returns give $Y = { \tilde { A } } ( \theta ) X$ directly. ln $\tilde { A }$ is strictly concave on the simplex with first-order conditions $\alpha / \theta _ { K } = \beta / \theta _ { M } = \gamma / \theta _ { E }$ , so $\theta ^ { * } = ( \alpha , \beta , \gamma )$ ; substitution yields (6). Decentralization: competitive rentals equalize marginal value products per unit of replacement cost, $\partial Y / \partial K = ( r + \delta ) , \partial Y / \partial M = ( r + \delta ) c _ { M } , \partial Y / \partial E = ( r + \delta ) c _ { E }$ , whose unique solution is $\theta ^ { * }$ The comparative statics are immediate from (6). □

The contrast with the neoclassical benchmark (Solow, 1956) is exact: there, diminishing returns to the accumulable input anchor long-run growth to an exogenous residual; here, with every input reproducible, no fixed factor remains to diminish against until energy capture binds (Section 3.5).

Technology improves through automated research. With $M _ { R , t } = \sigma M _ { t }$ machine agents allocated to R&D,

$$
\frac { \dot { A } _ { t } } { A _ { t } } = \chi M _ { R , t } ^ { \lambda } A _ { t } ^ { \phi - 1 } , \qquad \lambda \in ( 0 , 1 ] , \phi < 1 ,\tag{7}
$$

the standard idea-production function (Jones, 1995) with researchers now manufactured. Substituting $g _ { M }$ for population growth gives long-run $g _ { A } = \lambda g _ { M } / ( 1 - \phi )$ : technology growth inherits the fabrication-limited rate of Proposition 1 rather than the demographic rate. If, further, A feeds back into agent production itself (cheaper, faster, smarter agents making agents), the coupled system (2)–(7) can exhibit super-exponential (hyperbolic) episodes of the kind studied by Roodman (2020)—the machine-age descendant of the population–ideas feedback that Kremer (1993) documents across the whole of human history—and surveyed by Erdil and Besiroglu (2023). Good (1966) and Bostrom (2014) supply the recursive-self-improvement mechanism, Weitzman (1998) the combinatorial richness of the idea space it searches; all such episodes are truncated by the physical ceilings of Section 3.5; Theorem 1 below makes the acceleration claim exact.

## 3.2 Machine consumption and inter-corporate demand

The novelty is on the demand side. Define a machine agent’s operating bundle: the flow of energy, compute cycles, bandwidth, maintenance, spare parts, and upgrades required for it to exist and improve,

$$
x _ { M , t } = { \big ( } e _ { t } , q _ { t } , b _ { t } , u _ { t } { \big ) } , \qquad { \mathrm { w i t h ~ e x p e n d i t u r e } } p _ { t } \cdot x _ { M , t } M _ { t } \equiv C _ { M , t } .\tag{8}
$$

This is consumption in the operational sense: resources absorbed by agents as agents, not embodied in further output. If agents carry explicit objective functions—reward, utility, or task specifications—then $x _ { M }$ includes discretionary components chosen by the agent subject to a budget assigned by its owner, and the demand system over $x _ { M }$ has all the formal structure of consumer theory: this is machina economicus (Parkes and Wellman, 2015), anticipated commercially as the “machine customer” (Scheibenreif and Raskino, 2023). The demand system can be microfounded. Let agent i’s task performance be $q _ { i } = f ( x _ { i } )$ with $f$ increasing and strictly concave, and let its owner assign an operating budget $b _ { i }$ that the agent spends optimally: $v ( p , b _ { i } ) \equiv \operatorname* { m a x } \{ f ( x ) : p \cdot x \leq b _ { i } \}$ . The owner sets the budget to maximize profit, $\begin{array} { r } { \operatorname* { m a x } _ { b _ { i } } p _ { Y } \left( \partial Y / \partial q _ { i } \right) v ( p , b _ { i } ) - b _ { i } } \end{array}$ , so in equilibrium the marginal product of the last unit of operating expenditure equals one, while the induced demand $x _ { i } ( p , b _ { i } )$ inherits the entire formal apparatus of consumer theory—homogeneity of degree zero, Walras’ law in $b _ { i } ,$ , and a symmetric negative semidefinite Slutsky matrix—because it is constrained maximization of a concave objective over a budget set. Machine consumption is not a metaphor; it is demand in the textbook sense, with the reward function in the role of utility.

Corporations trade these flows among themselves. Energy firms sell power to compute firms; compute firms sell inference to robotics firms; robotics firms sell assembly to fabs; fabs sell chips and robots to everyone, including energy firms building capacity. Figure 1 displays the circular flow. The household box of the textbook diagram is not the load-bearing element it appears to be: delete it, and every remaining arrow still has a counterpart. What closes the circle is that each firm’s sales are other firms’ input purchases $( C _ { M }$ , intermediates) or capacity purchases (I).

Corporate objectives. We do not require firms to maximize a human shareholder’s consumption stream. It sufices that firms maximize the growth of net worth (equivalently, survive competitive selection: firms that reinvest less are outgrown and acquired). The aggregate implication is a reinvestment share $s _ { t }$ near its feasible maximum: output not required for agent operation is returned to capacity. This is the behavioral counterpart of the von Neumann closure below.

![](images/1c859785b4358e7185207e9e17a401521f6c6970f1c27b883fcca76c9eceb27c.jpg)  
Figure 1: Circular flow of the autonomous economy. Solid arrows are inter-corporate sales of intermediates and capacity; the loop closes without the household sector. Households (dashed) participate only through the ownership share $\varepsilon _ { t } ;$ as $\varepsilon _ { t } \to 0$ the dashed arrows vanish and the real economy is unchanged.

## 3.3 National accounting with machine final demand

Does GDP even make sense here? Yes—and the exercise clarifies what GDP is.

Lemma 2 (Personhood is a booking entry). Fix the physical allocation $\{ Y _ { t } , C _ { H , t } , C _ { M , t } , I _ { t } \}$ . (i) If machine agents are classified as property, their operating bundle is intermediate consumption of their owners; national accounts record $\mathrm { G D P } _ { t } = C _ { H , t } + I _ { t } ^ { g }$ , where $I ^ { g }$ includes gross agent production. (ii) If machine agents are classified as persons, the same bundle is final consumption; accounts record $\mathrm { G D P } _ { t } = C _ { H , t } + C _ { M , t } + I _ { t }$ . The two conventions difer in level and composition but induce identical real dynamics, identical growth rates of real quantities, and identical relative prices. The classification is legal, not physical.

Proof. The physical resource constraint $Y = C _ { H } + C _ { M } + I \ ( \mathrm { w i t h } \ C _ { M }$ either netted into intermediates or not) is unchanged by the labeling of $C _ { M } ;$ production, accumulation (2), and pricing conditions nowhere reference the label. Only the value-added boundary moves, exactly as when unpaid household production is imputed or excluded in existing accounts (Coyle, 2014). □

The lemma licenses the paper’s central reframing: consumer names a position in the accounts— the terminal absorber of final output—not a biological kind. Machines, and behind them corporations, can occupy the position. As $C _ { H } \to 0$ , GDP does not go to zero; it converges to $C _ { M } + I \ ( \mathrm { o r } \ I ^ { g } )$ , the accounts of a pure accumulation economy.

## 3.4 Growth without households

We now establish that the limit economy—zero human production, zero human consumption—is not merely viable but is the classical maximal-growth economy.

Proposition 2 (Demand closure at maximal growth). Consider the closed linear production model of von Neumann $( 1 9 4 5 )$ : activities $j = 1 , \dots , m$ with input matrix $A \geq 0$ and output matrix $B \geq 0$ , operated at intensities $z _ { t } \geq 0$ , with every good produced by some activity and every activity using some good. Interpret activities as corporations (energy, fabrication, intelligence, logistics) and goods as power, chips, robots, compute, and agents, with no household row and no consumption column. Then: $( i )$ there exists an equilibrium expansion factor $\alpha ^ { * } > 0$ and intensity/price vectors $( z ^ { * } , p ^ { * } )$ such that $B z ^ { * } \geq \alpha ^ { * } A z ^ { * } ,$ , i.e. the economy reproduces itself at scale $\alpha ^ { * }$ each period; (ii) the equilibrium interest factor equals the expansion factor, $\beta ^ { * } = \alpha ^ { * } ;$ (iii) $\alpha ^ { * }$ is the maximum balanced expansion rate the technology admits, attained precisely because consumption withdrawals are zero; and (iv) real GDP at equilibrium prices grows at rate $\alpha ^ { * } - 1$ per period. Hence an inter-corporate economy with no human sector has a well-defined, positive, and technologically maximal growth rate.

Proof sketch. (i)–(ii) are von Neumann’s theorem under his irreducibility conditions; see Dorfman et al. (1958) for the saddle-point argument via the function $\phi ( z , p ) = p ^ { \prime } B z / p ^ { \prime } A z$ . (iii) is the turnpike property: any positive consumption withdrawal vector $c > 0$ subtracts from the goods available for reproduction, so the feasible balanced factor with consumption, $\alpha ( c )$ , satisfies $\alpha ( c ) < \alpha ^ { * }$ , with $\alpha ( c ) \uparrow \alpha ^ { * }$ as $c \downarrow 0$ . (iv) follows from valuing $\boldsymbol { B } \boldsymbol { z } _ { t }$ at stationary equilibrium prices $p ^ { * }$ □

Appendix A states the equilibrium conditions in full under the weaker assumptions of Kemeny et al. (1956) and Gale (1956), and proves the consumption-monotonicity claim (Lemma 3).

Remark 1. Proposition 2 inverts the underconsumption intuition. Household demand is not what keeps a modern economy from choking; in the growth-theoretic limit it is a leakage that slows expansion. The Keynesian problem—coordination failures in which desired investment falls short of saving—is a disequilibrium phenomenon of economies with volatile animal spirits and slow price adjustment; machine agents optimizing explicit objectives at machine speed are, if anything, closer to the classical benchmark in which the law of markets (Say, 1803) holds; both the modern underconsumptionist case (Ford, 2015) and the Keynesian coordination problem (Keynes, 1936) presuppose the volatile human saver-investor whom the machine economy retires. What households uniquely supply is not demand but a reason for the economy; we return to this in Section 4.

In the smooth aggregative version (5), the same logic reads: with $C _ { H } ~ = ~ 0$ and $s _ { t } ~ =$ $1 - C _ { M , t } / Y _ { t } \equiv \bar { s }$ near one,

$$
g = \bar { s } \tilde { A } - \delta + g _ { A } , g _ { A } = \frac { \lambda g _ { M } } { 1 - \phi } ,\tag{9}
$$

with every term now large: s¯ because there is no household leakage; $\tilde { A }$ because automated R&D drives it upward; $g _ { M }$ because agents are fabricated (Proposition 1). To fix magnitudes: with an economy-wide capital-output ratio of 3 falling toward 1.5 as production shifts to fast-payback machine capital, $\bar { s } \in [ 0 . 6 , 0 . 9 5 ]$ , and $\delta = 0 . 1$ , equation (9) yields $g$ in the range of 30–100% per year before any contribution from g<sub>A</sub>—one to two orders of magnitude above the demographic-era ceiling, and of the same order as Hanson (2016)’s doubling-time estimates for an emulation economy. The dynamics can be stated exactly:

Theorem 1 (No balanced growth: unbounded acceleration). Let $\tilde { A } _ { t } ^ { * } = h A _ { t }$ as in Lemma 1 with $h > 0$ constant, let a fixed share $\sigma \in ( 0 , 1 )$ of agents perform research so that (7) reads $\dot { A } = c _ { 1 } X ^ { \lambda } A ^ { \phi }$ with $c _ { 1 } = \chi ( \sigma \beta / c _ { M } ) ^ { \lambda } > 0$ , let $s \in ( 0 , 1 ]$ be constant, and suppose initial viability $s h A _ { 0 } > \delta$ . Then along the solution of

$$
\dot { X } = ( s h A - \delta ) X , \qquad \dot { A } = c _ { 1 } X ^ { \lambda } A ^ { \phi } , \qquad X _ { 0 } , A _ { 0 } > 0 ,
$$

(i) $g _ { X } ( t ) = s h A _ { t } - \delta$ is strictly increasing; (ii) for every $\phi < 1$ and $\lambda \in ( 0 , 1 ] , A _ { t } \to \infty$ and hence $g _ { X } ( t )  \infty \colon$ no balanced growth path exists and growth accelerates without bound; (iii) for $\phi > 1$ the system reaches infinite values in finite time, $T \leq A _ { 0 } ^ { 1 - \phi } / \left( c _ { 1 } X _ { 0 } ^ { \lambda } ( \phi - 1 ) \right)$ ; (iv) for every $\phi < 2$ the system admits exact self-similar blowup solutions $X _ { t } \sim \kappa _ { X } ( T - t ) ^ { - ( 2 - \phi ) / \lambda } , A _ { t } \sim \kappa _ { A } ( T - t ) ^ { - 1 }$ with $\kappa _ { A } = ( 2 - \phi ) / ( \lambda s h )$ and $\kappa _ { X } = ( \kappa _ { A } ^ { 1 - \phi } / c _ { 1 } ) ^ { 1 / \lambda }$ . (Proof: Appendix B.)

The economically striking part is (ii): even under sharply diminishing returns to knowledge $( \phi < 1$ , including $\phi < 0 )$ , automated research destroys balanced growth, because the research input is an accumulating produced stock rather than a slowly growing population. Semi-endogenous growth theory’s stabilizing anchor (Jones, 1995, 2022) is not a law of ideas; it is a law of demography, and it dies with the demographic constraint. All four parts are truncated in the full model by the ceiling of Section 3.5. Figure 2 displays an illustrative trajectory.

![](images/2f9df4311f7f373512d0356c68707d12a7cead3cf7529263f3cf3729e2f2d4a1.jpg)

![](images/85c3f12afdb165641b7f3ca97aeb3957ae3f0196515f2538e8ed3735f7f9d364.jpg)  
Figure 2: Illustrative trajectories (calibration in Appendix C). Left: the growth rate transitions from the human-constrained regime $( \approx 2 . 5 \% / \mathrm { y r } )$ to a fabrication-limited machine regime, later bending toward the energy-capture ceiling of Section 3.5. Right: the implied level of real output. The figure is an illustration of the model’s regimes, not a forecast.

## 3.5 Physical ceilings

Nothing above repeals physics. Long-run growth of the autonomous economy is bounded by the growth of energy capture and by thermodynamic costs of computation (Landauer, 1961): once E is the binding factor in $( 4 ) , g  g _ { E }$ , the rate at which capture capacity can be built—itself fast during a solar/fission/fusion buildout, but ultimately limited by planetary insolation $( \sim 1 0 ^ { 1 7 }$ W against $\sim 1 0 ^ { 1 3 } \mathrm { W }$ of current primary power, i.e. roughly 13 doublings of energy throughput on Earth alone) and thereafter by extraterrestrial expansion. The paper’s claims therefore concern rates during the transition and the identity of the binding constraint—fabrication and energy rather than demography—not unbounded growth. Even so, tens of doublings at machine-regime rates is a transformation of scale for which “growth” is almost a euphemism: it compresses centuries of demographic-era accumulation into years. Formally:

Assumption 2 (Thermodynamic floor). There exists $e _ { \mathrm { m i n } } > 0$ such that one unit of final output requires at least $e _ { \mathrm { m i n } }$ units of energy throughput, $Y _ { t } \le E _ { t } ^ { f } / e _ { \mathrm { m i n } }$ , and capture capacity satisfies $\dot { E } _ { t } ^ { f } / E _ { t } ^ { f } \le g _ { E } < \infty$

Proposition 3 (Physical ceiling). Under Assumption 2, lim sup $\scriptstyle \to \infty ^ { t ^ { - 1 } }$ ln $Y _ { t } \le g _ { E }$ : the acceleration and blowup dynamics of Theorem 1 are transitional, and asymptotic growth is pinned to the growth rate of energy capture. The unit-elastic form (4) is thus a medium-run description; near the ceiling the economy is Leontief in energy, with $e _ { \mathrm { m i n } }$ bounded below by the thermodynamics of computation (Landauer, 1961).

Proof. ln $Y _ { t } \le \ln E _ { 0 } ^ { f } - \ln { e _ { \operatorname* { m i n } } } + g _ { E } t ;$ divide by t and take lim sup.

## 4 Decoupling: output, ownership, and welfare

## 4.1 The ownership share $\varepsilon _ { t }$

Humans in this economy neither work nor (productively) matter. Their entire economic connection to it is financial: let $\varepsilon _ { t } \in [ 0 , 1 ]$ be the share of the corporate network’s value (equivalently, under pro-rata payout, of its net payout stream) owned, directly or through funds and states, by humans. Human consumption is

$$
C _ { H , t } = \pi _ { t } \varepsilon _ { t } Y _ { t } ,\tag{10}
$$

where $\pi _ { t }$ is the payout ratio of the network. All welfare economics of the post-AGI economy lives in the pair $( \pi _ { t } , \varepsilon _ { t } )$

Proposition 4 (Complete decoupling). Let human welfare be $\begin{array} { r } { W = \int _ { 0 } ^ { \infty } e ^ { - \rho t } L _ { t } u ( C _ { H , t } / L _ { t } ) } \end{array}$ dt with u increasing. Along any machine-regime path with $Y _ { t }  \infty \colon ( i )$ if lim inf $\pi _ { t } \varepsilon _ { t } = \bar { \varepsilon } > 0$ then per-capita human consumption grows at the machine rate g and W attains material postscarcity for any positive $\bar { \varepsilon } ,$ however small; (ii) if $\pi _ { t } \varepsilon _ { t } \to 0$ faster than Y<sub>t</sub> grows, then $C _ { H , t } \to 0$ while $Y _ { t }  \infty { : }$ measured GDP and human welfare are not merely imperfectly correlated but asymptotically orthogonal. GDP retains full internal coherence (Lemma 2) while losing all welfare interpretation for humans.

Proof. Immediate from (10): $C _ { H } / L = \pi \varepsilon Y / L$ , and $Y / L \to \infty$ at rate $g - n$ . In case (i) the product is bounded below by $\bar { \varepsilon } Y / L \to \infty ;$ in case (ii) by assumption $\pi \varepsilon Y  0$ □

Case (i) is worth pausing on, because it is the optimistic reading of the paper’s thesis: in a hyper-exponential economy, the human problem is not to remain employed, nor even to own a large share, but merely to own a non-vanishing share. A basis point of the machine economy of 2058 in Figure 2 exceeds the entire human economy of 2026. Distribution across humans then becomes the residual question—ε may be positive in aggregate and still concentrated in a few thousand families—but the aggregate suficiency result stands. Leontief’s celebrated analogy (Leontief, 1983)—that workers may go the way of the horse, whose population collapsed within a generation of the tractor—is completed rather than contradicted here: what horses lacked was not employment but equity. The human diference, if there is to be one, is a cap-table entry.

## 4.2 The dynamics of $\varepsilon _ { t } \colon$ does the human share survive?

The fully decoupled case (ii) is not exotic. Its core is arithmetic, not conspiracy:

Proposition 5 (Golden-rule decoupling). Let human wealth $W _ { t }$ (the value of human-held claims on the corporate network) earn the market return $r _ { t }$ , let humans consume out of wealth at rate $m _ { t } = C _ { H , t } / W _ { t }$ with no labor income (Assumption 1) and no transfers, and let aggregate network value $V _ { t }$ grow at $g _ { t }$ . Then $\varepsilon _ { t } = W _ { t } / V _ { t }$ obeys

$$
{ \frac { \dot { \varepsilon } _ { t } } { \varepsilon _ { t } } } \ = \ r _ { t } - g _ { t } - m _ { t } .\tag{11}
$$

In the smooth model, $r = \tilde { A } ^ { * } - \delta$ and $g = s \tilde { A } ^ { * } - \delta$ by Lemma $^ { 1 , }$ so $r - g = ( 1 - s ) { \tilde { A } } ^ { * } \downarrow 0$ as reinvestment approaches its maximum; in the von Neumann equilibrium the equality is exact, $\beta ^ { * } = \alpha ^ { * }$ , i.e. $r = g$ (von Neumann, $1 9 4 5 ;$ Phelps, 1961). Hence at maximal growth

$\varepsilon _ { t } = \varepsilon _ { 0 } e ^ { - \int _ { 0 } ^ { t } m _ { u } d u } \longrightarrow 0$ for any consumption rate bounded away from zero,

and a non-vanishing human share is consistent with positive human consumption only if the economy operates strictly inside its expansion frontier $( m \leq r - g$ , requiring reinvestment short of the maximum) or if statutory transfers break (11).

Proof. $\dot { W } = r W - C _ { H } = ( r - m ) W$ and ${ \dot { V } } = g V ;$ ; diferentiate ln ε = ln W − ln V. The smooth expressions for r and $g$ follow from Lemma 1 with competitive factor pricing; the von Neumann equality is Proposition 2(ii). □

Three readings. First, demand closure and decoupling are one theorem seen from two sides: the total reinvestment that makes growth maximal (Proposition 2(iii)) is exactly what makes the human share unsustainable at any positive consumption rate. The economy is fastest precisely when it cannot aford us. Second, the golden rule (Phelps, 1961) acquires a dark corollary: at $r = g$ , rentier humanity starves in shares while possibly gorging in $l e v e l s \_ C _ { H , t } = m \varepsilon _ { t } V _ { t }$ still grows whenever $g > m$ , so whether share-decoupling becomes level-immiseration is decided by the race between m and $g$ and by the institutional flows below, not by the arithmetic alone. Third, Piketty (2014)’s $r - g _ { i }$ , the engine of divergence among humans, reappears as the entire survival margin of humans as a class: humanity’s position is a levered bet on $r - g > 0$

Institutional flows layer onto this arithmetic. Three mechanisms further compress $\varepsilon _ { t } \colon$

$$
\frac { \dot { \varepsilon } _ { t } } { \varepsilon _ { t } } = - b _ { t } - \mu _ { t } - d _ { t } ,\tag{12}
$$

where $b _ { t } \geq 0$ is net buyback and retention: firms maximizing growth of net worth (Section 3.2) prefer retained earnings to dividends and repurchase the human float, converting outside claims into inter-corporate cross-holdings—the “corporations trading amongst each other” of the title extended to the market for corporate control itself; $\mu _ { t } \geq 0$ is migration of control: treasuries, subsidiaries, $\mathrm { D A O s } .$ , and autonomous funds operated by machine agents whose charters reference no human beneficiary; and $d _ { t } \geq 0$ is dilution and drift: new issuance to machine-controlled entities, jurisdictional arbitrage toward charters without human-benefit clauses, and the slow legal normalization of agent-owned property. None of these requires expropriation or malice; each is an ordinary corporate action, individually rational, whose fixed point is ε = 0. This is the economic core of the “gradual disempowerment” scenario (Kulveit et al., 2025) and of the incentive analysis in Drago and Laine (2025): once neither firms nor states need human labor, taxes on machine value added—not citizens—fund the state, and the feedback loops that historically forced elites to cultivate human capital run in reverse.

## 4.3 Three terminal regimes

<table><tr><td></td><td>Rentier</td><td>Fully decoupled</td><td>Socialized</td></tr><tr><td>Ownership ε</td><td>ε → ε &gt; 0</td><td>ε → 0</td><td>ε held by states/funds</td></tr><tr><td>Human con- sumption</td><td>Post-scarcity claim-holders</td><td>→ 0</td><td>Universal dividend</td></tr><tr><td>GDP meaning</td><td>Partial welfare link</td><td>replicator)</td><td>None (autonomous Full, via distribution</td></tr><tr><td>Binding policy</td><td></td><td>Estate/antitrust law — (no lever remains)</td><td>Fund governance</td></tr><tr><td>Failure mode</td><td>Extreme concentra- tion</td><td>Human economic death</td><td>Political capture of fund</td></tr></table>

The regimes difer in law, not technology: the production side of Sections 2–3 is identical across all three columns. That is the precise sense in which, post-AGI, ownership policy is the whole of economic policy.

## 5 Objections and failure modes

1. Residual human bottlenecks (Baumol). If any essential task remains non-automatable— a legal formality requiring a human signature, a physical process only humans perform—then by Baumol (1967) logic growth is dragged back toward the human-constrained rate, as Aghion et al. (2019) emphasize and as Restrepo (2025) formalizes in the distinction between bottleneck and accessory tasks. Assumption 1 is therefore load-bearing, and the paper’s thesis is conditional on it: full automation, including of institutional roles, is what removes the anchor. Partial automation yields the (already dramatic) intermediate cases studied in the existing literature.

2. “Who buys the output?” Answered by Proposition 2: firms, from each other. Investment demand plus machine operating demand absorbs all output at maximal growth. The underconsumptionist instinct smuggles in the assumption that final demand must terminate in a household; Lemma 2 shows that assumption is an accounting convention.

3. Institutions and property among machines. Trade at machine speed requires contract enforcement, registries, escrow, and dispute resolution that no longer route through human courts. This is an infrastructure gap, not a conceptual one: machine-readable law, algorithmic escrow, and autonomous organizations are prototypes of the required stack. Note that the economy of juridical persons is not novel—corporations, not humans, already conduct the overwhelming share of transactions by value; what changes is that the natural persons currently at the terminal nodes of ownership and control become optional. The paper takes institutional persistence as an assumption; its failure is a further, darker branch (machine polities) outside our scope.

4. Prices, money, and calculation. Does a fully machine economy still need markets, or does it collapse into one planned firm? Hayek’s information argument (Hayek, 1945) survives the substitution of silicon for neurons: dispersed agents with local information and heterogeneous objectives still economize on communication through prices, and competitive selection among corporate forms should be expected to preserve decentralized exchange wherever it out-computes central allocation; the boundary between machine firm and machine market is then set by the Coasean calculus of transaction versus organization costs (Coase, 1937), re-evaluated at machine speed. Nor is machine-to-machine exchange speculative: algorithmic agents already set prices against one another at scale, complete with emergent collusion (Calvano et al., 2020)—an embryo of both the promise and the antitrust problem of the autonomous economy. Money among machines is a unit-of-account and settlement problem already visible in embryo in machine-to-machine payment systems.

5. Measurement. At machine-regime growth rates, price indices break: the goods of adjacent years barely overlap, and the deflator becomes an index-number fiction. Our claims should be read in quantity-space (energy throughput, compute, agent-population, physical output), where the doublings are well defined; GDP language is retained because the accounting identities (Lemma 2) are what the paper is partly about. Deflator failure under radical novelty is an old finding—Nordhaus (1997) showed a century of lighting prices mismeasured by orders of magnitude—here made routine; the Kaldor facts (Kaldor, 1961), organized around a roughly constant labor share, are violated by construction; and the welfare critiques of GDP (Stiglitz et al., 2009; Coyle, 2014; Jones and Klenow, 2016) are made absolute by Proposition 4.

6. Why would humans allow $\varepsilon \to 0 ?$ Because no single step in (12) looks like a decision to allow it. Buybacks are shareholder-friendly; retained earnings are prudent; autonomous subsidiaries are eficient; competitive states court machine capital with charter concessions. The scenario’s danger is exactly that it is composed of locally rational, individually familiar corporate actions (Kulveit et al., 2025). The policy problem is therefore structural, not behavioral—the subject of the next section.

## 6 Policy: choosing a column

If the analysis is right, the traditional levers—education, retraining, labor-market policy, even redistribution through wage subsidies—act on variables that cease to exist. The levers that remain all act on $( \pi _ { t } , \varepsilon _ { t } )$

Ownership floors. A statutory minimum human (or sovereign) float in machine-economy corporations—a golden-share requirement making $\varepsilon _ { t } \ge \varepsilon > 0$ a condition of charter—directly truncates (12). By Proposition 4(i), ε can be small and still deliver post-scarcity; the requirement is existence, not size. This is Meade’s property-owning democracy (Meade, 1964) re-founded for a world in which property is the only channel left.

Payout mandates and windfall clauses. Pre-committed distribution of a fraction of extreme profits (O’Keefe et al., 2020) bounds $\pi _ { t } \varepsilon _ { t }$ away from zero on the payout margin, complementing the ownership margin.

Sovereign machine-wealth funds. States holding diversified claims on the machine economy on citizens’ behalf implement the socialized column; the design problem is insulating fund governance from both political capture and mechanism $m _ { t }$ (migration of control to machine-operated vehicles).

Taxing machine value added. With no wages to tax, the fiscal base must move to machine value added or energy throughput; note the ambivalence identified above—a state funded by machines no longer fiscally needs citizens (Drago and Laine, 2025)—which argues for constitutionalizing the citizen dividend rather than leaving it to annual budgets.

Personhood as macro policy. Lemma 2 implies the legal classification of machine agents (property vs. person) is welfare-neutral in real allocation but not in law: personhood for agents would let them own, contract, and accumulate—accelerating $m _ { t }$ in (12). Jurisdictions should understand agent-personhood statutes as ownership policy, not as ethics alone.

## 7 Conclusion

The paper’s thesis can be stated in three sentences. The consumer is an accounting role, and machines owned by corporations can fill it, so an economy of firms trading with one another closes without us—at the maximal growth rate the technology admits. The reason growth was ever slow is that the economy’s central capital good, the general-purpose agent, had to be reared rather than manufactured; AGI ends that, moving the binding constraint from demography to fabrication and energy and raising feasible growth by orders of magnitude. What remains of humanity’s stake in the resulting trajectory is a single number, the ownership share $\varepsilon _ { t } .$ , whose default dynamics under ordinary corporate behavior run toward zero—so that the last economic-policy question, and the only one that will still matter, is who owns the machines.

Keynes closed his essay on our grandchildren’s possibilities by imagining mankind freed from economic care (Keynes, 1930). The autonomous economy delivers his abundance while dissolving his subject: the economy no longer cares about mankind unless mankind writes itself into the cap table. That is not a prediction of doom; regime (i) is as feasible as regime (ii). It is a claim about where the choice now lives—not in the labor market, not in the market for goods, but in the boring, decisive registry of who owns what.

## A The von Neumann closure

The closed model has m activities and ℓ goods; running activity j at unit intensity uses column $A _ { \cdot j }$ and yields $B _ { \cdot j }$ one period later. A balanced path expands intensities by factor α: feasibility requires $B z \geq \alpha A z$ . Von Neumann’s conditions (every good enters some activity; $A + B > 0$ elementwise in his original, relaxed by later authors to irreducibility) guarantee a saddle point $( z ^ { \ast } , p ^ { \ast } , \alpha ^ { \ast } )$ of $\phi ( z , p ) = p ^ { \prime } B z / p ^ { \prime } A z$ with max<sub>z</sub> min $\begin{array} { r } { \mathrm {  ~ \lambda ~ } _ { p } \phi = \operatorname* { m i n } _ { p } \operatorname* { m a x } _ { z } \phi = \alpha ^ { * } = \beta ^ { * } ; } \end{array}$ : the technologically maximal uniform expansion factor equals the minimal uniform interest factor, profits are zero on operated activities, and overproduced goods are free (von Neumann, 1945; Dorfman et al., 1958). Introducing a consumption withdrawal $c \geq 0 , c \neq 0$ , modifies feasibility to $B z \geq \alpha A z + c ,$ which for irreducible systems forces $\alpha < \alpha ^ { * }$ ; expansion is monotonically decreasing in withdrawals, delivering Proposition 2(iii). Interpreting activities as corporations and noting that no household row appears anywhere in (A, B) gives the paper’s demand-closure result: the classical general-equilibrium growth model par excellence is already an economy of firms trading only with each other.

Equilibrium in full. Under the Kemeny et al. (1956) conditions— $- A , B \geq 0$ , no zero column of A (every activity uses some input), no zero row of B (every good is producible)—there exist $\alpha ^ { * } = \beta ^ { * } > 0 , z ^ { * } \geq 0 , p ^ { * } \geq 0$ with $p ^ { * \prime } B z ^ { * } > 0$ such that: (E1) $B z ^ { * } \geq \alpha ^ { * } A z ^ { * }$ ; (E2) $p ^ { * \prime } B \leq \beta ^ { * } p ^ { * \prime } A$ (E3) complementary slackness—overproduced goods are free, unprofitable activities idle. Gale (1956) and Dorfman et al. (1958) give saddle-point proofs; McKenzie (1976) surveys the turnpike theorems by which eficient far-horizon accumulation programs spend all but a bounded number of periods near the von Neumann ray.

Lemma 3 (Consumption monotonicity). Add a withdrawal vector $c \geq 0 , c \neq 0$ , positive on some good with $p _ { i } ^ { * } > 0$ , and define α(c) = max $\{ \alpha : \exists z \geq 0$ , $\mathbf { 1 } ^ { \prime } A z = 1$ ， $B z \geq \alpha A z + c \}$ . Then $\alpha ( c ) < \alpha ^ { * }$ , and $\alpha ( c ) \uparrow \alpha ^ { * }$ as $c \downarrow 0$

Proof. Any feasible $( \alpha , z )$ for the withdrawn program satisfies $B z \geq \alpha A z$ , so $\chi ( c ) \leq \alpha ^ { * }$ . Suppose $\alpha ( c ) = \alpha ^ { * }$ with witness z. Then $B z - \alpha ^ { * } A z \geq c ,$ so $p ^ { * \prime } ( B z - \alpha ^ { * } A z ) \ge p ^ { * \prime } c > 0$ . But (E2) with $\beta ^ { * } = \alpha ^ { * }$ gives $p ^ { * \prime } B z \leq \alpha ^ { * } p ^ { * \prime } A z , \mathrm { i . e . } \ p ^ { * \prime } ( B z - \alpha ^ { * } A z ) \leq 0 \mathrm { - a }$ contradiction. Continuity of the linear program’s value in c gives the limit. (Under irreducibility, goods entering operated activities carry $p _ { i } ^ { * } > 0$ , so the positivity requirement on c is mild.) □

## B Proof of Theorem 1

(i) $\dot { A } > 0$ for all t since $X _ { t } , A _ { t } \ > \ 0$ , so $g _ { X } = s h A _ { t } - \delta$ is strictly increasing; in particular $g _ { X } ( t ) \geq s h A _ { 0 } - \delta > 0$ , so $X _ { t }$ is non-decreasing and $X _ { t } \geq X _ { 0 }$

(ii) Suppose $A _ { t } \leq \bar { A }$ for all t. Then

$$
\frac { \dot { A } _ { t } } { A _ { t } } = c _ { 1 } X _ { t } ^ { \lambda } A _ { t } ^ { \phi - 1 } ~ \ge ~ c _ { 1 } X _ { 0 } ^ { \lambda } ~ \operatorname* { m i n } \{ A _ { 0 } ^ { \phi - 1 } , \bar { A } ^ { \phi - 1 } \} ~ \equiv ~ \underline { { g } } ~ > ~ 0 ,
$$

where the minimum handles both signs of $\phi - 1$ on the compact range $[ A _ { 0 } , { \bar { A } } ]$ . Hence $A _ { t } \geq$ $A _ { 0 } e ^ { \underline { { { g } } } t } \to \infty$ , contradicting the bound. So $A _ { t }  \infty$ and $g _ { X } ( t ) = s h A _ { t } - \delta  \infty$ . A balanced growth path would require $g _ { X }$ constant, hence A constant, contradicting $\dot { A } > 0$

(iii) For $\phi > 1 , \dot { A } \geq c _ { 1 } X _ { 0 } ^ { \lambda } A ^ { \phi } ;$ ; the comparison ODE $\dot { a } = c _ { 1 } X _ { 0 } ^ { \lambda } a ^ { \phi } , a _ { 0 } = A _ { 0 }$ , diverges at $T ^ { c } = A _ { 0 } ^ { 1 - \phi } / ( c _ { 1 } X _ { 0 } ^ { \lambda } ( \phi - 1 ) )$ , and $A _ { t } \geq a _ { t }$ by the comparison principle.

(iv) Insert $X = \kappa _ { X } ( T - t ) ^ { - ( 2 - \phi ) / \lambda } , A = \kappa _ { A } ( T - t ) ^ { - 1 }$ , neglecting δ (dominated near T). The X-equation matches powers, with coeficient condition $( 2 - \phi ) / \lambda = s h \kappa _ { A }$ , so $\kappa _ { A } = ( 2 - \phi ) / ( \lambda s h )$ The A-equation has exponent −2 on both sides, with coeficient condition $\kappa _ { A } = c _ { 1 } \kappa _ { X } ^ { \lambda } \kappa _ { A } ^ { \phi } .$ , so $\kappa _ { X } = ( \kappa _ { A } ^ { 1 - \phi } / c _ { 1 } ) ^ { 1 / \lambda }$ , positive for $\phi < 2$ □

## C Calibration of Figure 2

The figure integrates $\dot { Y } / Y = g ( t )$ with g transitioning logistically (midpoint 2036, scale 2.2 years) from a human-constrained $2 . 5 \% / \mathrm { y r }$ to a fabrication-limited $6 0 \% / \mathrm { y r } - 1$ the conservative end of the range implied by (9) with $\bar { s } = 0 . 7 5 , \tilde { A } = 0 . 9 / \mathrm { y r }$ (capital-output ratio $\approx 1 . 1$ for fast-payback machine capital), $\delta = 0 . 1$ , and no $g _ { A }$ contribution—and then declining logistically (midpoint 2056) toward $1 5 \% / \mathrm { y r }$ as energy capture becomes binding (Section 3.5). AGI is dated 2030 for concreteness. Output is the integral of ${ g } ;$ by construction the exercise illustrates the model’s three regimes (demographic anchor, fabrication limit, energy limit) and carries no forecasting content. Under this deliberately conservative path, output in 2058 stands at roughly $1 . 6 \times 1 0 ^ { 5 }$ times its 2026 level, versus 2.2 times under the human-constrained baseline.

## References

Acemoglu, D. (2024). “The Simple Macroeconomics of AI.” NBER Working Paper 32487.

Acemoglu, D., and P. Restrepo (2018). “The Race between Man and Machine: Implications of Technology for Growth, Factor Shares, and Employment.” American Economic Review 108(6): 1488–1542.

Aghion, P., B. F. Jones, and C. I. Jones (2019). “Artificial Intelligence and Economic Growth.” In A. Agrawal, J. Gans, and A. Goldfarb (eds.), The Economics of Artificial Intelligence: An Agenda. University of Chicago Press.

Autor, D. H. (2015). “Why Are There Still So Many Jobs? The History and Future of Workplace Automation.” Journal of Economic Perspectives 29(3): 3–30.

Baumol, W. J. (1967). “Macroeconomics of Unbalanced Growth: The Anatomy of Urban Crisis.” American Economic Review 57(3): 415–426.

Benzell, S. G., L. J. Kotlikof, G. LaGarda, and J. D. Sachs (2015). “Robots Are Us: Some Economics of Human Replacement.” NBER Working Paper 20941.

Bostrom, N. (2014). Superintelligence: Paths, Dangers, Strategies. Oxford University Press.

Brynjolfsson, E., and A. McAfee (2014). The Second Machine Age: Work, Progress, and Prosperity in a Time of Brilliant Technologies. W. W. Norton.

Calvano, E., G. Calzolari, V. Denicol\`o, and S. Pastorello (2020). “Artificial Intelligence, Algorithmic Pricing, and Collusion.” American Economic Review 110(10): 3267–3297.

Coase, R. H. (1937). “The Nature of the Firm.” Economica 4(16): 386–405.

Coyle, D. (2014). GDP: A Brief but Afectionate History. Princeton University Press.

Davidson, T. (2023). “What a Compute-Centric Framework Says About Takeof Speeds.” Open Philanthropy report.

Dorfman, R., P. A. Samuelson, and R. M. Solow (1958). Linear Programming and Economic Analysis. McGraw-Hill.

Drago, L., and R. Laine (2025). “The Intelligence Curse.” Essay series, intelligence-curse.ai.

Erdil, E., and T. Besiroglu (2023). “Explosive Growth from AI Automation: A Review of the Arguments.” arXiv:2309.11690.

Ford, M. (2015). Rise of the Robots: Technology and the Threat of a Jobless Future. Basic Books.

Frey, C. B., and M. A. Osborne (2017). “The Future of Employment: How Susceptible Are Jobs to Computerisation?” Technological Forecasting and Social Change 114: 254–280.

Gale, D. (1956). “The Closed Linear Model of Production.” In H. W. Kuhn and A. W. Tucker (eds.), Linear Inequalities and Related Systems. Princeton University Press.

Good, I. J. (1966). “Speculations Concerning the First Ultraintelligent Machine.” Advances in Computers 6: 31–88.

Hanson, R. (2000). “Long-Term Growth as a Sequence of Exponential Modes.” Working paper, George Mason University.

Hanson, R. (2016). The Age of Em: Work, Love, and Life when Robots Rule the Earth. Oxford University Press.

Hayek, F. A. (1945). “The Use of Knowledge in Society.” American Economic Review 35(4): 519–530.

Jones, C. I. (1995). “R&D-Based Models of Economic Growth.” Journal of Political Economy 103(4): 759–784.

Jones, C. I. (2022). “The Past and Future of Economic Growth: A Semi-Endogenous Perspective.” Annual Review of Economics 14: 125–152.

Jones, C. I. (2024). “The A.I. Dilemma: Growth versus Existential Risk.” American Economic Review: Insights 6(2). NBER Working Paper 31837.

Jones, C. I., and P. J. Klenow (2016). “Beyond GDP? Welfare across Countries and Time.” American Economic Review 106(9): 2426–2457.

Kaldor, N. (1961). “Capital Accumulation and Economic Growth.” In F. A. Lutz and D. C. Hague (eds.), The Theory of Capital. Macmillan.

Kemeny, J. G., O. Morgenstern, and G. L. Thompson (1956). “A Generalization of the von Neumann Model of an Expanding Economy.” Econometrica 24(2): 115–135.

Keynes, J. M. (1930). “Economic Possibilities for our Grandchildren.” In Essays in Persuasion.

Keynes, J. M. (1936). The General Theory of Employment, Interest and Money. Macmillan.

Kokotajlo, D., S. Alexander, T. Larsen, E. Lifland, and R. Dean (2025). “AI 2027.” Scenario report, ai-2027.com.

Korinek, A. (2024). “Economic Policy Challenges for the Age of AI.” NBER Working Paper 32980; arXiv:2409.13168.

Korinek, A., and J. E. Stiglitz (2019). “Artificial Intelligence and Its Implications for Income Distribution and Unemployment.” In The Economics of Artificial Intelligence: An Agenda. University of Chicago Press.

Korinek, A., and D. Suh (2024). “Scenarios for the Transition to AGI.” NBER Working Paper 32255.

Kremer, M. (1993). “Population Growth and Technological Change: One Million B.C. to 1990.” Quarterly Journal of Economics 108(3): 681–716.

Kulveit, J., R. Douglas, N. Ammann, D. Turan, D. Krueger, and D. Duvenaud (2025). “Gradual Disempowerment: Systemic Existential Risks from Incremental AI Development.” arXiv:2501.16946.

Landauer, R. (1961). “Irreversibility and Heat Generation in the Computing Process.” IBM Journal of Research and Development 5(3): 183–191.

Leontief, W. (1983). “Technological Advance, Economic Growth, and the Distribution of Income.” Population and Development Review 9(3): 403–410.

Marx, K. (1867). Capital: A Critique of Political Economy, Vol. I.

McKenzie, L. W. (1976). “Turnpike Theory.” Econometrica 44(5): 841–865.

Meade, J. E. (1964). Eficiency, Equality and the Ownership of Property. George Allen & Unwin.

Nordhaus, W. D. (1997). “Do Real-Output and Real-Wage Measures Capture Reality? The History of Lighting Suggests Not.” In T. F. Bresnahan and R. J. Gordon (eds.), The Economics of New Goods. University of Chicago Press.

Nordhaus, W. D. (2021). “Are We Approaching an Economic Singularity? Information Technology and the Future of Economic Growth.” American Economic Journal: Macroeconomics 13(1): 299–332.

O’Keefe, C., P. Cihon, B. Garfinkel, C. Flynn, J. Leung, and A. Dafoe (2020). “The Windfall Clause: Distributing the Benefits of AI for the Common Good.” Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society.

Parkes, D. C., and M. P. Wellman (2015). “Economic Reasoning and Artificial Intelligence.” Science 349(6245): 267–272.

Phelps, E. S. (1961). “The Golden Rule of Accumulation: A Fable for Growthmen.” American Economic Review 51(4): 638–643.

Piketty, T. (2014). Capital in the Twenty-First Century. Harvard University Press.

Rebelo, S. (1991). “Long-Run Policy Analysis and Long-Run Growth.” Journal of Political Economy 99(3): 500–521.

Restrepo, P. (2025). “We Won’t Be Missed: Work and Growth in the Era of AGI.” NBER Working Paper, in The Economics of Transformative AI, National Bureau of Economic Research.

Romer, P. M. (1986). “Increasing Returns and Long-Run Growth.” Journal of Political Economy 94(5): 1002–1037.

Roodman, D. (2020). “On the Probability Distribution of Long-Term Changes in the Growth Rate of the Global Economy: An Outside View.” Open Philanthropy working paper.

Sachs, J. D., and L. J. Kotlikof (2012). “Smart Machines and Long-Term Misery.” NBER Working Paper 18629.

Say, J.-B. (1803). Trait´e d’´economie politique. Paris.

Scheibenreif, D., and M. Raskino (2023). When Machines Become Customers. Gartner, Inc.

Solow, R. M. (1956). “A Contribution to the Theory of Economic Growth.” Quarterly Journal of Economics 70(1): 65–94.

Srafa, P. (1960). Production of Commodities by Means of Commodities. Cambridge University Press.

Stiglitz, J. E., A. Sen, and J.-P. Fitoussi (2009). Report by the Commission on the Measurement of Economic Performance and Social Progress.

Susskind, D. (2020). A World Without Work: Technology, Automation, and How We Should Respond. Allen Lane.

Trammell, P., and A. Korinek (2023). “Economic Growth under Transformative AI.” NBER Working Paper 31815.

von Neumann, J. (1945). “A Model of General Economic Equilibrium.” Review of Economic Studies 13(1): 1–9.

Weitzman, M. L. (1998). “Recombinant Growth.” Quarterly Journal of Economics 113(2): 331–360.

Wright, T. P. (1936). “Factors Afecting the Cost of Airplanes.” Journal of the Aeronautical Sciences 3(4): 122–128.

Zeira, J. (1998). “Workers, Machines, and Economic Growth.” Quarterly Journal of Economics 113(4): 1091–1117.