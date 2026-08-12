# Expert-Guided g-computation with Large Language Models for Estimating Causal Efects on Timings: Applications to Hospital Quality Improvement

Patrick Vossler<sup>1</sup>, Jialin Ouyang<sup>1</sup>, F. Richard Guo<sup>2</sup>, Anran Huang<sup>1</sup>,

Ali Shojaie<sup>3</sup>, Lucas Zier<sup>1</sup>, Fan Xia<sup>1∗</sup>, Jean Feng<sup>1∗†</sup>

<sup>1</sup>University of California, San Francisco, <sup>2</sup>University of Michigan, Ann Arbor <sup>3</sup>University of Washington, Seattle ∗ Co-senior authors

<sup>†</sup> Corresponding author: jean.feng@ucsf.edu, 550 16th Street, San Francisco, CA, 94158

## Summary

Hospital quality improvement (QI) programs are routinely faced with multiple candidate interventions to optimize hospital flow, but existing methods are limited in their ability to estimate and rank the causal efects of such interventions. This work focuses on one of the most standard hospital metrics—the average length of stay (LOS)—and its associated causal estimand—the average time saved. To characterize this causal efect, qualitative approaches rely on expert judgment to map complex patient trajectories, making them susceptible to cognitive biases; conversely, quantitative approaches rely on data-driven models, which fail when interventions are entirely hypothetical with no historical data and/or complex causal mechanisms that cannot be learned through data and instead require clinical reasoning. Here we propose expert-guided g-computation or egg-computation, which builds on the complementary strengths of qualitative and quantitative approaches and connects Gantt charts commonly used to map patient trajectories with the causal DAG literature. We first introduce a causal model over Gantt charts and establish causal identification using a variant of g-computation that only seeks expert input for components unidentifiable from data. To make egg-computation practical, we develop a Large Language Model (LLM)-assisted pipeline that reliably scales up expert reasoning. Simulations show that egg-computation improves accuracy compared to conventional causal inference methods in complex settings where patients have diverse causal structures and intervention mechanisms. In a study of eleven candidate QI interventions at an urban safety-net hospital, the LLM pipeline generated graphs and time-saving estimates that were highly concordant with those of human experts. Beyond healthcare, egg-computation is a broadly applicable framework for estimating average time saved for candidate interventions whose causal mechanisms can be represented using Gantt charts.

Key words:

Causal inference, g-computation, Large language models, Gantt charts

## 1. Introduction

Hospital Quality Improvement (QI) programs seek to improve the healthcare delivery process and key hospital performance metrics like length of stay (LOS). Numerous QI interventions have been proposed in the literature, including better triaging of imaging studies and medical procedures (Xu et al., 2026; Hurlen et al., 2010), balancing workload across a week (Vidal-Carreras et al., 2022), and using Artificial Intelligence (AI)-based clinical decision support tools (Levin et al., 2021; Wornow et al., 2023). Recently, Large Language Models (LLMs) have even been used to generate a bank of candidate QI initiatives by identifying delays/bottlenecks most commonly mentioned in hospital records (Vossler et al., 2026). Yet QI programs can only implement a handful of interventions—often just one or two—and must therefore prioritize by expected impact, which ultimately requires answering a causal question. In this work, our focus is on characterizing the causal efect on LOS, i.e., average time saved, as LOS is among the most widely-used metrics of hospital eficiency, is linked to patient outcomes, and factors into reimbursement (Rojas-García et al., 2018; Ghosh et al., 2023; Liang et al., 2026; The Press Association, 2014).

To anticipate how a QI intervention would afect LOS, QI teams commonly rely on semistructured qualitative analyses such as staf interviews, manual chart review, and Lean healthcare methods (Catalyst, 2018; D’Andreamatteo et al., 2015). A central tool in these qualitative analyses is the classical Gantt chart (or, equivalently, the value stream map) (Rother and Shook, 1999; Vidal-Carreras et al., 2022), where a patient’s hospital journey is broken down into tasks and their dependencies. For example, consider the following early discharge-planning QI intervention based on findings in Vossler et al. (2026):

Any patient with complex discharge needs, as identified on the first day of their hospital stay, will be evaluated for discharge planning and meet with social work or case management within their first 24-48 hours. If the patient agrees, appropriate discharge resources will be contacted within 24 hours after meeting with the patient. Exclusion criteria include patients who cannot meet because of medical or legal reasons.

By creating a Gantt chart of the patient’s journey, hospital QI teams can use their domain expertise to reason about how a described intervention would likely shift event timings and the overall LOS. While experts may be accurate at estimating how an intervention shifts the events it directly targets (e.g., the time when the patient meets with social work or case management), its likely impact on more downstream events is harder to predict, as those event timings depend on interactions among multiple preceding events and cannot be determined through simple fixed rules. For example, an earlier social work consultation may not substantially accelerate discharge if the information needed to determine the appropriate disposition is not yet available. Similarly, moving a consultation from Monday to the preceding Friday may leave LOS unchanged if the anticipated discharge institution (e.g., a skilled nursing facility) does not process admissions over the weekend.

Given the limitations of qualitative analyses, QI teams often conduct complementary quantitative analyses. The most common approach is to fit a regression model; however, such models generally estimate association not causation, and traditionally rely solely on tabular features, which alone may be insuficient for eliminating confounding bias (Steverson et al., 2023; Zeng et al., 2022). While a formal causal framework is desired, developing such a framework is non-trivial: (i) Understanding a QI intervention and its efects requires complex clinical reasoning, and much of the relevant information resides in unstructured clinical notes rather than the tabular data. (ii) The commonly assumed positivity assumption is dificult to satisfy as proposed QI interventions are, by definition, new policies that the hospital has never implemented and no patients have been exposed to (Westreich and Cole, 2010; Petersen et al., 2012). (iii) Because QI interventions often cover a broad and heterogeneous patient population, constructing a unified causal diagram that encompasses the all patient trajectories is a near-impossible task, as even the set of hospital events documented in clinical notes is open-ended. Furthermore, even if such a causal diagram existed, the available data is limited for accurately estimating the conditional relationships in the graph.

![](images/6dd7534e8d2e5878cd4960a085f519d88c9dd29fd5a8b0ef40f6d1374904e568.jpg)  
Fig. 1. Overview of expert-guided g-computation. To estimate the efect of a candidate Quality Improvement (QI) intervention on the average length of stay (LOS) among eligible hospitalizations, the expert, or an aligned LLM, reads unstructured texts describing the candidate intervention and patient records (left) and uses expert reasoning to extract patient-specific Gantt charts, which we prove can be endowed with probabilities and causal semantics (middle). Through a combination of expert reasoning and statistical modeling, we simulate the counterfactual charts under the candidate intervention (right) and estimate the target estimand “Average Time Saved” $\tau = \mathbb { E } [ Y ( 0 ) - Y ( 1 ) ]$ where $Y ( 1 )$ and $Y ( 0 )$ are the LOS with or without intervention, respectively. Dashed outlines mark events whose timings are shifted in the counterfactual trajectory.

To resolve these technical gaps, this work combines qualitative and quantitative approaches to conduct what we refer to as “expert-guided causal inference:” expert judgment is elicited solely in areas where experts are accurate but data-driven modeling is not, while statistical models are employed where data-driven models are accurate but experts may not be (Figure 1). The final causal contrast reflects the likely efect of an intervention based on both the data and expert opinion and, for settings where the expert is accurate, we establish identifiability of the true causal efect. Proving this result requires carefully translating Gantt charts into directed acyclic graphs (DAGs) endowed with formal probabilistic and causal semantics for event processes (Section 2). Given the resulting DAGs, we then introduce a variant of g-computation (Robins, 1986) that we refer to as expert-guided g-computation or egg-computation, where expert manipulation of the Gantt chart is combined with statistical sampling of downstream events to generate counterfactual patient trajectories (Section 3). Because a fully human-powered approach is likely prohibitively time- and resource-intensive, we developed an LLM-assisted pipeline that supplies the expert reasoning needed to conduct egg-computation at scale (Section 3.2). This LLM pipeline can be easily adapted to new clinical settings or interventions, as it only depends on prompting rather than model retraining or fine-tuning.

To demonstrate the eficacy of the proposed egg-computation framework, we conduct extensive empirical evaluation. We begin with comparing egg-computation with conventional causal inference methods in simulation studies, which show that egg-computation is more accurate when the causal structure or intervention mechanism varies substantially across patients. Then in a real-world study of eleven candidate QI interventions at an urban safety-net hospital, we evaluate the use of LLMs to scale human-expert-guided causal inference. We find that LLM extractions are highly concordant with human experts and even as accurate as outcome regression models, leading to highly concordant time saving estimates. Clinical experts also found that LLM-generated reasoning traces provide useful insights into how well an intervention works and how it might be better designed. Finally, while our focus is on estimating the efects of hospital QI interventions on patient trajectories, the methodology is broadly applicable to settings where Gantt charts are suitable for describing task dependencies and process timings (Wilson, 2003; Clark et al., 1922) and the Related Work section in the Appendix places this contribution in the broader literature.

## 2. Causal models for Gantt charts

A Gantt chart (see Figure 1) is a collection of horizontal bars corresponding to events, where the horizontal axis corresponds to time. Directed edges between bars represent dependencies between events, which are often categorized into start-start, finish-start, start-finish, and finish-finish relationships (Project Management Institute, 2000). If we ignore the time axis and divide an event into a pair of (start, end) events with the end depending on the start, a Gantt chart can be drawn as a Directed Acyclic Graph (DAG), where each node represents the time of an event and directed edges describe the dependency between event times. To represent the start and completion of a multi-step process, a Gantt chart must include source and sink nodes, which in our setting correspond to hospital admission and discharge/death, respectively.

While a Gantt chart corresponds to a DAG—a representation commonly used for causal models of various kinds (Wright, 1934; Pearl, 1995; Robins, 1986)—this correspondence alone does not endow it with the causal semantics needed to address the inferential question of interest. Rather, formalizing the causal model associated with a Gantt chart requires clarifying what edges mean exactly in this context, how an intervention changes the event process, and which features of that process are assumed to remain unchanged. To this end, we introduce a causal model in terms of event waiting times that enables identification of intervention efects within subpopulations defined by X.

Notation. We use upper-case letters $( \mathrm { e . g . } , X , Y , Z )$ for random variables or vectors and use lower-case $( \mathrm { e . g . } , x , y , z )$ for values. We use Roman capital letters $( \mathrm { e . g . , C , D ) }$ for sets, which can appear as subscripts for indexing random vectors. For a node k in a DAG, we use the standard graphical notation: $\operatorname* { P a } ( k ) : = \{ j : j \to k \}$ for parents, $\operatorname { C h } ( k ) : = \{ j : j  k \}$ for children, $\operatorname { D e } ( k ) : = \{ k \} \cup \{ j : k \to \cdots \to j \}$ for descendants, and $\mathrm { N d } ( k ) : = \mathrm { D e } ^ { c } ( k )$ for non-descendants. We use $\cdot { = } _ { d } \cdot$ to denote equality in distribution.

## 2.1 Causal model over waiting times

Consider a random patient with baseline covariates $X \in { \mathcal { X } }$ . The patient’s Gantt chart, when ignoring the specific event times, can be described as a DAG over a finite set of events, which we denote as $\mathrm { D } = \{ 0 , 1 , \ldots , K \}$ ; see Fig. 2 for an example. Given the DAG, an event k with $\operatorname { P a } ( k ) \neq \varnothing$ can only occur $a f t e r$ all the events in $\operatorname { P a } ( k )$ have occurred. We assume the labels are topologically ordered so that $j \in \operatorname { P a } ( k )$ implies $j < k$ . For simplicity, we will suppose both the baseline covariates X and the set of events D are conditioned $o n ,$ thereby fixing the DAG associated with the Gantt chart. In other words, the random patient is considered drawn from the subpopulation with the given X and D.

![](images/ff3c267784282cce88610e6d371b624b3fc4c0e115a73a5870b95c23334be6b5.jpg)  
Fig. 2. An example inpatient trajectory drawn as a Gantt chart with its causal DAG. Each row is one event, placed at its event time in hours from admission, and gray arrows are the DAG edges; events are labeled $k = 0 , \ldots , K$ in topological order, with $K = 7 .$ . Node A is not an event: it denotes a binary intervention that directly afects the waiting times of the transthoracic echocardiogram (TTE) acquisition and report (dashed arrows). Bar colors give each event’s relation to $A \colon k \in \{ 0 , 1 , 2 , 3 \}$ are non-descendants of $A ( \equiv ) , k \in \{ 4 , 5 \}$ are children of $A ~ ( = \overline { { 1 } } )$ , and $k \in \{ 6 , 7 \}$ are descendants of A that are not children of $A ~ ( = \overline { { 1 } } )$ . Event ‘Admission’ defines the start of wall time $T _ { 0 } = W _ { 0 } = 0 ;$ ‘Discharge’ defines the outcome of interest $Y : = T _ { 7 }$ . The observed data is generated under $A = 0$

For each event $k ,$ let $T _ { k }$ be the wall time at which k occurs. Let $W _ { k }$ be the waiting time for its occurrence, namely the time taken for it to occur since the occurrence of all of its preceding events. The wall time and waiting time are related through

$$
T _ { k } = \left\{ \begin{array} { l l } { \operatorname* { m a x } _ { j \in \mathrm { P a } ( k ) } T _ { j } + W _ { k } , } & { \quad \mathrm { P a } ( k ) \neq \emptyset } \\ { W _ { k } , } & { \quad \mathrm { P a } ( k ) = \emptyset } \end{array} \right. .\tag{2.1}
$$

Given $W _ { k }$ for each $k ,$ one can use (2.1) to recursively compute $T _ { 0 } , T _ { 1 } , \dots , T _ { K }$ ; conversely, given $T _ { k }$ for each k, (2.1) directly recovers $W _ { 0 } , W _ { 1 } , \dots , W _ { K }$ . We let $T _ { k }$ and $W _ { k }$ take values in $[ 0 , + \infty ]$ where $T _ { k } = + \infty$ means the event $k$ never occurs. If $T _ { k } = + \infty$ for some event $k .$ , the ensuing events also never occur, i.e., $T _ { j } = + \infty$ for all $j \in \mathrm { D e } ( k )$ . We designate two special events: $k = 0$ as the source event (‘Admission’) with wall time $T _ { 0 } = W _ { 0 } : = 0$ and $k = K$ as the sink event (‘Discharge’), which defines the outcome of interest $Y : = T _ { K }$ . All events are required to be an ancestor of event K, i.e., $k  \cdots  K$ for each $k \neq K ;$ otherwise we can discard such events without loss of generality.

In our event-time setting, the DAG model posits the local Markov property in terms of conditional independence

$$
W _ { k } \perp \perp T _ { \mathrm { N d } ( k ) } \mid T _ { \mathrm { P a } ( k ) } , X , \mathrm { D } \quad k = 0 , 1 , \ldots , K .\tag{2.2}
$$

This property should be diferentiated from that of a standard Bayesian network model. By (2.1), wall times obey the ordering $T _ { k } \leqslant T _ { j }$ for all $j \in \mathrm { D e } ( k )$ ; therefore, (2.2) cannot be restated as $T _ { k } \perp \perp T _ { \mathrm { N d } ( k ) } \mid T _ { \mathrm { P a } ( k ) } , X , \mathrm { D }$ nor $W _ { k } \perp \perp W _ { \mathrm { N d } ( k ) } \mid W _ { \mathrm { P a } ( k ) } , X .$ , D in general.

A QI policy is defined using a free-text description along with inclusion and exclusion criteria, like the early discharge intervention example described in Section 1. To represent the causal impact of such a policy, the Gantt DAG is endowed with an additional non-event node A that represents the binary intervention: $A = 0$ represents the current policy under which our data is observed, while $A = 1$ represents a new QI policy, which we do not observe but want to infer. We refer to the direct children of an intervention, denoted $\operatorname { C h } ( A )$ , as intervention targets, whose waiting times are directly modified by the new policy. We assume A is exogenous, represented by $\operatorname { P a } ( A ) = \varnothing$ in the augmented Gantt chart. Every event k is associated with the potential waiting times $W _ { k } ( 0 ) , W _ { k } ( 1 ) \in [ 0 , + \infty ]$ which correspond to potential wall times $T _ { k } ( 0 ) , T _ { k } ( 1 ) \in [ 0 , + \infty ]$ through relationships defined in a similar manner to (2.1), i.e.,

$$
T _ { k } ( a ) = \left\{ \begin{array} { l l } { \operatorname* { m a x } _ { j \in \mathrm { P a } ( k ) } T _ { j } ( a ) + W _ { k } ( a ) , \quad } & { \mathrm { P a } ( k ) \neq \emptyset } \\ { W _ { k } ( a ) , \quad } & { \mathrm { P a } ( k ) = \emptyset } \end{array} , \quad \right. \quad a = 0 , 1 ,\tag{2.3}
$$

where again we let $W _ { 0 } ( 0 ) = W _ { 0 } ( 1 ) = 0$ . Implicit in our notation is the simplifying assumption of no interference or SUTVA (Rubin, 1980). For example, if A = 1 changes the CT schedule, we assume each patient is afected separately, even though in reality they may not be, due to resource constraints; see also Section 5 for more discussion.

$Y ( a ) : = T _ { K } ( a )$ for $a = 0 ,$ 1 is our potential outcome of interest measuring the total wall time from source to sink. Thus, the individual treatment efect is $Y ( 0 ) - Y ( 1 )$ , namely the time saved by the new QI policy. As we will see in Section 3, the individual treatment efect is not identified. Instead, we are interested in inferring the population-level average treatment efect

$$
\tau : = \mathbb { E } [ Y ( 0 ) - Y ( 1 ) ] = \mathbb { E } \left\{ \mathbb { E } [ Y ( 0 ) - Y ( 1 ) \mid X , \mathrm { D } ] \right\} ,\tag{2.4}
$$

where the outer expectation is averaging over both X and D.

Following the single-world/FFRCISTG causal semantics (Robins, 1986; Richardson and Robins, 2013) associated with a DAG, we associate our Gantt chart with a causal model defined by (2.3) along with the following assumptions. Recall that $T _ { k }$ and $W _ { k }$ are observed only under the current policy $A = 0$ . This difers from common clinical trial and observational study settings, where one observes both A = 0 and A = 1 arms.

Assumption 1 (Consistency) For $k = 0 , 1 , \ldots , K .$ , we have $T _ { k } = T _ { k } ( 0 )$ and $W _ { k } = W _ { k } ( 0 )$

Assumption 2 (Exogenous intervention) For $a = 0 , 1$ , we have $W _ { 0 } ( a ) , W _ { 1 } ( a ) , \ldots , W _ { K } ( a ) \perp \perp A \mid X , \mathrm { D }$

Assumption 3 (Exclusion restriction) For $k = 0 , 1 , \ldots , K$

1. if $k \in \operatorname { N d } ( A )$ , then $W _ { k } ( 1 ) = W _ { k } ( 0 )$

2. if $k \in \operatorname { D e } ( A ) \setminus \operatorname { C h } ( A )$ , then $W _ { k } ( 0 ) \mid T _ { \mathrm { P a } ( k ) } ( 0 ) , X , \mathrm { D } = _ { d } W _ { k } ( 1 ) \mid T _ { \mathrm { P a } ( k ) } ( 1 ) , X , \mathrm { D }$

Assumption 4 (Local Markov property) For $k = 0 , 1 , \ldots , K$ , we have $W _ { k } ( a ) \perp \perp T _ { \mathrm { N d } ( k ) } ( a )$ $T _ { \mathrm { P a } ( k ) } ( a )$ , X, D for $a = 0 , 1$ , where we let A be excluded from $\operatorname { N d } ( k )$ and $\operatorname { P a } ( k )$

Under Assumption 1, we have $Y ( 0 ) = T _ { K }$ and hence $\tau = \mathbb { E } T _ { K } - \mathbb { E } \left\{ \mathbb { E } [ T _ { K } ( 1 ) \mid X , \mathrm { D } ] \right\}$ by (2.4), where $T _ { K }$ is observed. Therefore, we can estimate τ if $T _ { K } ( 1 )$ can be imputed based on observed data under $A = 0$ . By the assumptions above, the distribution of the counterfactual timings can be factorized as

$$
p ( T _ { \mathrm { D } } ( 1 ) \mid X , \mathrm { D } ) = \prod _ { k = 0 } ^ { K } p ( T _ { k } ( 1 ) \mid T _ { \mathrm { P a } ( k ) } ( 1 ) , X , \mathrm { D } )\tag{2.5}
$$

so Section 3 will discuss how to obtain each factor in (2.5) so to recursively impute $T _ { \mathrm { D } } ( 1 )$

Remark 2.1 For ease of exposition, our presentation has treated the event set D in the Gantt chart as fixed and our notation has implicitly assumed that $\mathrm { D } ( 0 ) = \mathrm { D } ( 1 )$ for each patient. While this can be overly restrictive, the following causal identification results still hold if we relax this assumption to distributional equivalence, i.e., D(0) | $X = _ { d } \operatorname { D } ( 1 ) \mid X$ , with Assumptions 1 to 4 adjusted accordingly. Such an assumption is plausible primarily for operational interventions in healthcare, such as speeding up the timing of an imaging or consult order or increasing the availability of ancillary services over the weekend. In contrast, interventions that drastically change a patient’s trajectory $( \mathrm { e . g . }$ , swapping the treatment from chemotherapy to surgery) would not satisfy this assumption and should not be analyzed through this framework.

## 3. Expert-Guided g-computation<sub>,</sub> plus LLMs

Having established patient-specific Markovian DAGs as valid causal models for Gantt charts, we now further leverage this connection between Gantt charts and causal DAGs. In particular, we note that expert-manipulation of timestamps in a Gantt chart shares many similarities to using g-computation to simulate counterfactual node values on causal DAGs (Robins, 1986). Merging these ideas, we introduce “expert-guided g-computation” (egg-computation): we can identify the causal estimand $\tau$ defined in (2.4) through a combination of data-driven modeling and expert input, where the latter supplies information about the interventional distribution that cannot be learned from observational data.

## 3.1 Assumptions and Identifiability

Because $\mathbb { E } Y ( 0 )$ can be identified by taking a simple average of the observed outcomes Y , this section focuses on identifying $\mathbb { E } Y ( 1 )$ . Our general strategy is to simulate the counterfactual outcome Y (1) by recursively imputing the intermediate waiting times $W _ { k } ( 1 )$ for $k \in \mathrm { D }$ and iteratively applying definition (2.3). This begins with the expert supplying accurate patient-specific DAGs.

Assumption 5 (Expert DAG specification) Given the patient’s baseline covariates X and the observed event sequence, the Gantt DAG posed by the expert is correctly specified. That is, Pa(k) for each event k and the intervention targets Ch(A) are correctly specified.

Although this assumption is untestable, we believe it is plausible for many operational interventions, because clinical notes combined with tabular data often contain suficiently rich detail for experts to understand the causal relationships between diferent events (Vossler et al., 2026).<sup>∗</sup> For instance, tabular data may provide a sequence of events and timestamps (e.g., “10:30 AM MRI completed, 11:45 AM IV antibiotics started”) while clinical notes often describe how these events are linked (e.g., “starting IV antibiotics for the deep-tissue infection seen on MRI”). In contrast, this assumption may be less plausible if the intervention changes complex treatment regimens. Thus, prior to conducting egg-computation, one should assess whether Assumption 5 can be (approximately) satisfied.

Given the DAG, the next step is to determine how best to impute the waiting times. Since we have access to both observational data and an expert, we should choose the optimal information source for diferent nodes, which we divide into three mutually exclusive types:

Type I: Events that are not descendants of intervention A, i.e., $k \in \operatorname { N d } ( A )$ . This is the simplest option. By Assumption 3, waiting times of Type I nodes can be directly copied from the factual timeline, i.e., $W _ { k } ( 1 ) = W _ { k }$ . As these nodes are completely unafected by the intervention (as all ancestors are also unafected), we also have $T _ { k } ( 1 ) = T _ { k }$ for all Type I nodes.

Type II: Events that are descendants of A but not direct children, i.e., $k \in \mathrm { D e } ( A ) \backslash \mathrm { C h } ( A )$ Assuming the conditional distribution of the observed waiting times is correctly learned, i.e., $W _ { k } \mid T _ { \mathrm { P a } ( k ) } , X , \mathrm { D }$ , the model can be used to directly impute $W _ { k } ( 1 )$ since these two distributions are equivalent by Assumption 3. We state this modeling assumption below and discuss estimation in a later section.

Assumption 6 (Timing Model) We have a Type II timing model that equals the true conditional distribution $p \left( W _ { k } \mid T _ { \mathrm { P a } ( k ) } , X , \mathrm { D } \right)$ for all $k \in \operatorname { D e } ( A ) \setminus \operatorname { C h } ( A )$

Type III: Events that are direct children of intervention A, i.e., $k \in \operatorname { C h } ( A )$ The waiting times directly afected by a new QI policy are never observed, so we must rely on expert guidance. To achieve identification, we must assume our expert is accurate at imputing, as formalized below:

Assumption 7 (Expert Counterfactual Simulation) The expert-specified distribution $p _ { \mathrm { e x p e r t } }$ equals the counterfactual waiting time distribution $p \left( W _ { k } ( 1 ) \mid T _ { \mathrm { P a } ( k ) } ( 1 ) , X , \mathrm { D } \right)$ for all $k \in \operatorname { C h } ( A )$

In practice, this assumption need not hold exactly. If the intervention’s true direct efects are unknown, the expert can hypothesize a set of functions that vary the size of the intervention’s direct efects and weight the resulting estimates against their prior belief.

Now that counterfactual timestamps can be simulated for all three types of nodes along the patient’s DAG in topological order, we summarize the entire pipeline from observed Gantt charts to sampling counterfactual outcomes in Box :

## Box : Expert-guided g-computation

1. DAG Construction: The expert first determines whether the intervention, based on its inclusion and exclusion criteria, applies to the hospitalization. If not, the intervention node A has no children and the counterfactual trajectory equals the observed one. Otherwise, the expert analyzes the observed event sequence under $A = 0$ and constructs the causal DAG per Assumption 5. Each node is classified as Type I, II, or III based on its relationship to the intervention.

2. Expert-guided Simulation: Following the topological order of the DAG, generate counterfactual event times t<sup>′</sup> for each patient under $A = 1$

• For Type I nodes: copy observed times directly from factual sequence

• For Type II nodes: predict timing using models trained on observational data, conditioned on the (now-modified) parent timings

• For Type III nodes: sample from the expert-supplied distribution $p _ { \mathrm { e x p e r t } }$

Output: Take the average diference between the factual outcome $Y _ { i }$ and the imputed counterfactual outcome $Y _ { i } ^ { \prime }$ , i.e.,

$$
\hat { \tau } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( Y _ { i } - Y _ { i } ^ { \prime } \right)\tag{3.6}
$$

The following theorem gives the guarantee of egg-computation; see Appendix A for its proof.

Theorem 3.1 (Correctness of egg-computation) Suppose Assumptions 1 to 7 hold. Then, $Y _ { i } ^ { \prime }$ generated by the egg-computation is identically distributed as the counterfactual time $Y _ { i } ( 1 )$ Consequently, τˆ in (3.6) is an unbiased estimator of the average treatment efect $\tau$ in (2.4).

## 3.2 Scaling up with LLMs

Compared to classical causal inference frameworks, egg-computation requires substantially more expert input. In some sense, the “amount” of input needed for egg-computation scales with the number of observations: based on the intervention description and each patient’s clinical chart, the expert must (i) decide whether and when the intervention applies, (ii) construct a DAG and (iii) assist with event time simulation. While this allows egg-computation to conduct expert-guided causal reasoning in substantially more complex settings, a key question is how one can feasibly

supply this expert input in practice.

With the major advancements in reasoning abilities of LLMs, it is now practically feasible to scale egg-computation. Below, we discuss how LLMs are prompted to supply expert input for the two steps of egg-computation: eligibility determination and DAG construction in Step 1 and expert-guided simulation in Step 2. A more in-depth discussion of the pipeline’s implementation is included in Appendix C, and all prompts will be available in an open-source software package.

3.2.1 Step 1 Step 1 of egg-computation determines whether the intervention applies to a given hospitalization at all and, if so, how to construct the DAG. Determining eligibility is a relatively simple information extraction and reasoning task (Agrawal et al., 2022; Singhal et al., 2023; Kanjee et al., 2023), so an LLM needs only minimal prompt tuning to accurately assess whether a patient is eligible for an intervention. However, DAG construction is harder. A single zero-shot LLM prompt has only been shown to accurately extract small causal DAGs (⩽ 5 nodes) in well-understood settings (Kıcıman et al., 2024), but this strategy is unlikely to yield valid, let alone accurate, DAGs when both the number of DAGs and the number of nodes per DAG are much larger.

Substantial engineering is required to translate manual, and often highly iterative, approaches of DAG construction into a set of LLM prompts with reliable performance. At a high-level, our LLM pipeline is composed of five steps (Figure 3), where each step is one LLM call that refines the set of nodes and/or edges in the working DAG. The first step prompts the LLM to reason about the patient’s clinical course and draft the causal structure; the second and third steps formalize this draft as a Gantt chart by deciding the nodes and then the edges, respectively; and the final two steps construct the DAG in a structured format. This multi-step process was designed to eficiently spend the LLM’s reasoning tokens by prioritizing diferent aspects of DAG construction. Early steps in the pipeline prioritized clinical and causal reasoning and later steps prioritized structuring the output to match the expected encoding and double-checking the DAG structure. Throughout, a key design principle when crafting the LLM prompts was to use formal statistical language, as this precision encourages the LLM to use similarly rigorous reasoning.

![](images/024ca1cc420f27d1cdb627e6138bd2cc10cf1217154a60d850c1ffb89a33cc3a.jpg)  
Fig. 3. Five-step LLM pipeline for constructing patient DAGs. Each numbered box corresponds to each step’s LLM call and shows an example of the expected output. Given an unstructured patient chart, the LLM conducts free-form clinical reasoning (step 1), creates a Gantt chart (step 2 creates nodes, step 3 creates edges), and finally encodes the graph into a parseable format (step 4 creates nodes, step 5 creates edges). LLM outputs are required to contain quotes from the patient chart, so that LLM reasoning is transparent and auditable. The pipeline also includes checks to ensure validity of the final graph.

Central to this pipeline is designing step-by-step instructions to the LLM for deciding which nodes to add and when there were enough. The primary goal is to include enough nodes for adequate confounder adjustment so that conditional exchangeability in Assumption 5 is satisfied. To do this, prompts were based on the iterative graph expansion procedure of Guo and Zhao (2026) and the disjunctive criterion (VanderWeele, 2019), which starts from a graph containing only the exposure and outcome and expands it until causal identification can be achieved. To help the LLM find these confounders, the prompts instruct it to walk backwards from the intervention targets (Type III nodes) and from the sink node, enumerating the events encountered along these paths. A secondary goal was to include enough nodes in the DAG to improve the precision of timing estimates. For this, the prompts encouraged including mediating events along the traced paths, as these shorten the gaps between consecutive nodes, and shorter gaps are easier to model because they have higher data support and lower variability (For instance, given the causal pathway “initiation of IV antibiotics” → “end IV antibiotics” → “social worker visit” → “order a wheelchair”, there might be very few patients for whom we observe the initial and final event but likely many patients for whom we observe one or more consecutive pairs from this causal pathway). Finally, because each additional node increases the LLM costs, prompts were designed to limit overly large DAGs. The final prompts led to DAGs with an average of 14 nodes and 21 edges, which was manageable in terms of cost and query time.

The final pipeline used GPT-5-mini for generating the patient DAGs. For prompt tuning, we used a small training dataset composed of a synthetic test patient and three real patients. Outputs from the LLM pipeline were verified through a custom HTML interface that visualized the extracted graphs, the LLM’s reasoning, and the raw patient timeline. We developed a PHIcompliant annotation interface as well to evaluate agreement between our human experts and the LLM (see Appendix I for details).

3.2.2 Step 2 Step 2 of egg-computation needs to sample timings for Type II and III nodes. Below, we discuss how timing models for these nodes can be scaled up using LLMs while ensuring the timing estimates are aligned with the expert and real-world data distributions.

Type II nodes. To simulate timings of Type II nodes given their parent nodes, we consider fitting a pooled model that takes as input descriptions of the parent events, timings of the parent events, and a description of the child event. The training data for such a pooled model is obtained by constructing the observed Gantt charts for patients in the training dataset and collecting their

Type II nodes. Pooling across patients is needed because a single hospitalization only contributes one draw from each waiting-time factor. Since exact repeats of a conditioning event essentially never recur, the pooled model relies on a similarity assumption: events with similar descriptions, parent timings, and patient context follow similar timing distributions (see Remark A.3 in the Appendix for the formal pooling assumption and its connection to Theorem 3.1).

We consider three ways to fit a model to this pooled dataset. One option is to extract predefined tabular features from the free-form event descriptors and then train corresponding tabular models (e.g., random forests). Another option is to train models that take in the original event descriptors directly, such as ridge regression with sentence embeddings as features. A third option, based on in-context learning (ICL) (Brown et al., 2020), prompts an LLM to use its reasoning abilities and prior knowledge to predict the most likely timing, given the timings observed in similar training examples. We recommend choosing among these options by cross-validation on the pooled dataset. In the experiments, we compare all three options under leave-one-out cross validation, with GPT-5 as the ICL option.

Type III nodes. There are generally two options for making an LLM output timestamp distributions that are aligned with the expert distribution p<sub>expert</sub> at a Type III node. One option is to fine-tune the LLM or train a calibration model (e.g., using isotonic regression) if experts have annotated a suficiently large number of samples, but this is infeasible for many hospital QI teams. The other option, which we take here, is to specify the intervention in as much detail as possible such that the unknown expert-supplied distribution is precisely defined. As demonstrated in prior work, ambiguity in the prompt is indeed one of the major drivers of misalignment between human and LLM outputs (Tamkin et al., 2023; Subramonyam et al., 2024; Kothari et al., 2026). Consequently, each intervention includes details on the motivation for the intervention, what the intervention is, examples of what the expected timings are under the intervention, and inclusion/exclusion criteria (see example prompts in Section D.1 of the Appendix). After the initial prompt is created based on the detailed intervention, an iterative prompt-tuning procedure on a small set of patient DAGs can be used to further align the LLM-sampled event times with those from the human expert. In practice, we suggest conducting additional sensitivity analyses that vary the direct timing impact of the intervention, either by perturbing the prompt or applying varying shrinkage factors on the LLM-anticipated efect.

## 4. Experiments

Having established the theoretical properties of egg-computation, we now study its empirical performance. First, we study egg-computation assuming access to an oracle expert, using simulation studies to understand how it compares to classical causal inference methods. Then on real-world patient timelines, we apply egg-computation with frontier LLMs to estimate the efects of candidate QI interventions on LOS and evaluate concordance of the LLM pipeline with human experts.

4.1 Simulation studies comparing egg-computation to classical tabular causal inference methods We have made two claims regarding the advantages of egg-computation over classical causal inference methods thus far. First, egg-computation difers qualitatively in that it applies even in settings with limited or no data support, because it can additionally draw on expert input. Second, egg-computation has a quantitative advantage in that its estimates can be more accurate through more faithful modeling of causal dependencies, in part because the method analyzes both structured and unstructured data. While the former point is evident, the latter point has yet to be quantified and requires empirical study.

We present simulation results comparing egg-computation to existing methods for the following hypothetical clinical scenario (Figure 4). After patient admission at time $T _ { \mathrm { A d m } } = 0$ , this hypothetical hospital orders a diagnostic CT study at time $T _ { O _ { \mathrm { C T } } }$ and generates a report at time $T _ { R _ { \mathrm { C T } } } ;$ ; the same is done for an MRI study, with corresponding timestamps $T _ { O _ { \mathrm { M R I } } }$ and $T _ { R _ { \mathrm { M R I } } }$ . Depending on results from one or both of the studies, a final decision regarding the patient’s treatment will be made at some time $T _ { \mathrm { D e c i s i o n } }$

![](images/18cff2d4850bbf61cd80a43c1d9acbd7413c26b066f29cd9eafff9acc81efd28.jpg)  
Fig. 4. Causal graph generation scheme for the simulation study. Each patient’s causal graph has all solid edges, only one of the dashed edges, and one or both of the dotted edges.

Our goal is to understand the efect of a candidate intervention on the time to this final decision. As shown in Figure 4, the causal dependencies between these event timings vary between patients due to patient baseline factors, in that (i) the CT report may require the MRI report coming out first or vice versa and (ii) the final decision may depend on one or both of these imaging reports. For this simulation study, we generate $n = 4 { , } 0 0 0$ patients whose edges are uniformly sampled at random; as such, half of the patients have the CT → MRI dependency while the other half have the MRI → CT dependency. We study two candidate interventions, one that caps the time from CT order to report generation at 12 hours and another that caps the time to MRI report generation in the same fashion. Each patient’s sampled DAG is rendered as a synthetic clinical note that records a summary of the true causal structure.

We compare estimation methods that vary in the DAGs used—either a single shared DAG   
(Single-DAG) versus extracted subject-specific DAGs (DAG-extractor)—and how event timings   
are imputed—either by reusing the observed wait times as commonly done in qualitative analyses   
of Gantt charts (Gantt-timing) or by fitting a regression model for wait times (OLS-timing,   
kNN-timing) (see Appendix E for details). Performance is measured in terms of estimation bias. Unsurprisingly, Table 1 shows that DAG-extractor + Gantt-timing achieves zero bias, since

Table 1. Simulation results. The signed bias (hours) of the estimated average time saved, ${ \hat { \tau } } - \tau ,$ under each intervention, reported as mean ± standard error over the 50 replications. A positive value overstates the speed-up the intervention delivers and a negative value understates it. Each method pairs a DAG (patient-specific DAG-extractor versus a shared Single-DAG with an assumed report ordering) with a timing update (Gantt-timing reuses the factual time diferences exactly; OLS-timing and kNN-timing fit models for the timing gaps). DAG-extractor + Gantt-timing is the oracle configuration in this simulation, since the patient-specific DAG is correct and exact propagation matches the data-generating process. The Priority-CT and Priority-MRI columns are the two interventions.
<table><tr><td rowspan="2">Method</td><td colspan="2">ATE Bias (h)</td></tr><tr><td>Priority-CT</td><td>Priority-MRI</td></tr><tr><td> $\Delta A G \mathrm { - e x t r a c t o r \ + \ G a n t t \mathrm { - } t i m i n g \ }$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td> $\Delta A \mathsf { G - e x t r a c t o r \hbar + \hbar 0 L S \mathrm { - } t i m i n g }$ </td><td> $- 0 . 3 5 \pm 0 . 0 2$ </td><td> $- 0 . 2 2 \pm 0 . 0 2$ </td></tr><tr><td> $\Delta A \mathsf { G - e x t r a c t o r \Delta + \Delta \ k N N \mathrm { - t i m i n g } }$ </td><td> $2 . 7 5 \pm 0 . 0 1$ </td><td> $2 . 1 1 \pm 0 . 0 1$ </td></tr><tr><td>Single-DAG + Gantt-timing (CT→MRI)</td><td> $1 3 . 0 1 \pm 0 . 0 6$ </td><td> $- 2 . 5 6 \pm 0 . 0 7$ </td></tr><tr><td>Single-DAG + Gantt-timing (MRI→CT)</td><td> $- 5 . 3 7 \pm 0 . 0 4$ </td><td> $1 4 . 1 9 \pm 0 . 0 7$ </td></tr><tr><td> $\mathrm { \ S i n g l e - D A G ~ + ~ 0 L S - t i m i n g ~ ( C T \to M R I ) }$ </td><td> $1 0 . 7 3 \pm 0 . 0 8$ </td><td> $- 3 . 0 5 \pm 0 . 1 6$ </td></tr><tr><td> $\mathrm { S i n g l e - D A G ~ + ~ 0 L S - t i m i n g ~ ( M R I \to C T ) }$ </td><td> $7 . 0 6 \pm 0 . 1 1$ </td><td> $5 . 4 7 \pm 0 . 1 6$ </td></tr></table>

DAG-extractor recovers the true patient-specific structure and Gantt-timing uses the true timing model. The worst performing methods are those that rely on Single-DAG, where the shared DAG is always wrong for a majority of the cohort; as a result, the estimation bias can reach up to 14 hours. In contrast, imperfect timing models with the patient-specific DAG extractor added comparatively little bias: at most 0.4 hours for OLS-timing and 2.8 hours for kNN-timing. Thus, this simulation demonstrates that when patient-specific causal structure is heterogeneous and latent, methods that impose a single population-level DAG incur large and unpredictable biases. Egg-computation can eliminate structural misspecification and achieve unbiased estimation when it can accurately recover the subject-specific DAG auxiliary data such as clinical notes.

## 4.2 Real-world Evaluation of Candidate QI interventions

Building on prior LLM-based analyses of major bottlenecks and delays at the Zuckerberg San Francisco General Hospital (ZSFG) (Vossler et al., 2026), we analyze eleven candidate QI interventions that the hospital is considering in response. Described in more detail in Appendix G, candidate interventions range from accelerating completion of high-priority orders, advancing the initiation of certain processes, and increasing weekend services. Here we present quantitative and qualitative evaluations of the egg-computation pipeline on this real-world dataset.

Using the same data pipeline as Vossler et al. (2026), we extracted 2,193 inpatient timelines composed of clinical notes, clinical orders and events, and laboratory/test results for adult hospitalizations from the five most common diagnoses. Applying the LLM-based egg-computation pipeline, we first screened each hospitalization for whether it was eligible for an intervention and then estimated the causal efect on average LOS for the filtered population. To control computation costs, egg-computation pipeline was run for at most 30 eligible hospitalizations per diagnosis group per intervention, resulting in a total of 1,531 LLM-generated DAGs. The experiment’s LLM API cost totaled \$490; a cost breakdown is provided in Appendix H.

4.2.1 Expert agreement with LLM-constructed DAGs For LLMs to successfully scale up eggcomputation, human experts should generally agree with DAG generations from the LLM pipeline. To evaluate this, five experts from the hospital’s QI team (PV, JO, AH, LZ, JF) were asked to evaluate LLM-generated DAGs for two candidate interventions: early discharge planning and disposition-critical imaging studies. We evaluated concordance between DAG edges extracted by the LLM pipeline versus those indicated by expert annotators, to assess the reliability of using LLMs for scaling up human experts.

Twelve hospitalizations were annotated per intervention, so a total of 24 intervention-hospitalization pairs were annotated, where 8 were double-annotated to measure inter-annotator agreement. Given that patient timelines often spanned multiple days and averaged roughly 16,000 words, we designed a structured annotation workflow and interface that would balance time/resource constraints against potential biases in the annotation process. Our final annotation workflow is based on Vossler et al. (2026): we asked annotators to check which extracted nodes should be direct children

Table 2. Concordance between the LLM-generated and annotator-corrected DAGs, with 95% confidence intervals. For intervention targets, we report the share of events the annotator judged non-targets that the LLM also left unflagged (specificity) and the share of annotator-endorsed targets the LLM endorsed (recall). For input edges to nodes downstream of intervention, we report the proportion of LLM-proposed edges that were endorsed (precision) and the proportion of annotator-endorsed edges present in the LLM’s graph (recall). The final row reports the mean diference (hours) in the estimated average time saved when egg-computation was rerun on LLM-generated versus the annotator-corrected DAGs.
<table><tr><td colspan="3">Early discharge planning Disposition-critical imaging</td></tr><tr><td>Intervention-target specificity (%)</td><td>96.5 [91.9, 100.0]</td><td>96.5 [91.4, 100.0]</td></tr><tr><td>Intervention-target recall (%)</td><td>88.9 [79.2, 100.0]</td><td>95.7 [85.2, 100.0]</td></tr><tr><td>Edge precision (%)</td><td>91.8 [85.6, 97.7]</td><td>94.2 [89.2, 100.0]</td></tr><tr><td>Edge recall (%)</td><td>80.2 [65.6, 96.4]</td><td>86.0 [73.5, 100.0]</td></tr><tr><td>Mean difference in est. time saved (h)</td><td>-2.6 [-8.3, 0.8]</td><td>8.1 [0.0, 24.2]</td></tr></table>

of the intervention A (i.e., check which nodes are Type III) and which edges should be included for nodes downstream of the intervention (i.e., check input edges to all Type II nodes); see the full protocol in Appendix I. A total of 476 node verdicts and 243 edge verdicts were collected.

The annotation study revealed high concordance between the nodes and edges generated by the LLM pipeline and the human annotators (Table 2). Across both interventions, specificity and precision were consistently high. The annotators mostly agreed with the structure the LLM included. Recall was also high but generally lower than specificity and precision. This is the expected, as the LLM is prompted to only include nodes and edges it can confidently justify from the timeline (Section 3.2.1). Inter-annotator agreement was also high (Table 7 of the Appendix), with an average agreement rate of over 80%. Most importantly, we re-ran egg-computation on LLM-generated versus annotator-corrected DAGs (Table 2, final row). The mean diference in the estimated average time saved was −2.6 hours for early discharge planning and 8.1 hours for disposition-critical imaging, with 19 of the 24 pairs yielding identical estimates, demonstrating how LLM-based estimates are indeed quite close to human-based estimates.

4.2.2 Accuracy of timing models for downstream events For egg-computation to produce accurate causal estimates, the timing models for downstream events should approximate the true conditional distributions $p \left( W _ { k } \mid T _ { \mathrm { P a } ( k ) } , X , D \right)$ of Type II nodes. Under leave-one-out cross-validation at the hospitalization level, we evaluated the timing models of Section 3.2.2 on all 4,790 downstream node timings across the 736 hospitalizations described above, pooled over the five diagnosis groups and the eleven candidate interventions. The comparison includes two baselines (predict zero delay and predict the median delay among event pairs of the same type), three regression models (one of which includes a sentence embeddings), and a simple ICL solution with GPT-5 that requires no feature engineering nor model fitting. Experiment details are in Appendix J.

As shown in Table 3, ICL attains the lowest mean absolute error (24.1 hours against 27.2 for the closest baseline). The performance of ICL is significantly better than all five baselines for eight of the eleven interventions; the performance diferences are not statistically significant for the remaining three (Table 4, final column). The latter three cases should be taken into context: the structure of their wait time distributions are sharply zero-inflated and have 48–71% of their wait times under one hour. As such, the trivial baseline of predicting zero can perform very well for these special cases. The fact that the ICL method outperformed most methods despite its simplicity mirrors recent results finding that few-shot LLM prompting can be highly competitive with trained tabular models (Hegselmann et al., 2022), particularly in settings with highly complex data and small sample sizes. Interestingly, closer analysis revealed that ICL was better at predicting longer waiting times than regression models (Appendix J), as longer wait times tend to be noisier and have less data support.

Table 3. Wait time prediction accuracy for Type II nodes (4,790 downstream node timings from 736 hospitalizations). Mean Absolute Error (MAE) are shown in hours, along with Relative MAE compared to that of the zero-gap baseline. Values below one improve on predicting zero delay for every node. The final column gives each baseline’s excess MAE over ICL, with 95% intervals from a cluster bootstrap over hospitalizations. Per-intervention and per-delay-stratum breakdowns are given in Tables 4 and 5.
<table><tr><td>Method</td><td>MAE</td><td>Rel. MAE ∆MAE vs. ICL [95% CI]</td></tr><tr><td>Zero gap</td><td>28.30 1.00</td><td>+4.18 [3.57, 4.77]</td></tr><tr><td>Pair-type median</td><td>27.23</td><td>0.96 +3.11 [2.57, 3.67]</td></tr><tr><td>Ridge</td><td>34.30 1.21</td><td>+10.18 [9.37, 10.99]</td></tr><tr><td>Random forest</td><td>36.58 1.29</td><td>+12.46 [11.18, 13.90]</td></tr><tr><td>Embedding + ridge</td><td>33.79 1.19</td><td>+9.67 [8.57, 10.70]</td></tr><tr><td>ICL (GPT-5)</td><td>24.12 0.85</td><td></td></tr></table>

![](images/f87518bee2228dc93645c58a3a7f12b8e26b43e3f85a28eca9688be3baca2e8e.jpg)  
Fig. 5. Prevalence-based versus causal prioritization of the eleven candidate interventions. Eligibility rate is the share of hospitalizations that meet an intervention’s inclusion criteria. Expected time saved is the causal efect estimate from egg-computation. Interventions are ordered by expected impact.

4.2.3 Counterfactual Gantt charts and estimated rankings from egg-computation Having evaluated the accuracy of diferent components of our LLM-powered egg-computation pipeline, we now apply the final pipeline to obtain expert-informed causal efect estimates for all eleven candidate interventions. To illustrate results from egg-computation, Figure 6 shows an analysis of a randomly sampled hospitalization. Here, an ischemic stroke patient is waiting on a transthoracic echocardiogram (TTE), a common requirement prior to discharge. The TTE was ordered eight hours after admission but the imaging procedure was not completed until hour 72 of the patient’s stay. The clinician’s disposition review followed within an hour of the TTE report. Under the disposition-critical imaging priority intervention, egg-computation shifts the TTE completion and report roughly two days earlier, copies wait times for Type I nodes, and re-predicts timings for events downstream of the intervention targets. The imputed LOS is consequently 42 hours shorter.

Using egg-computation, we obtain estimated rankings for the candidate interventions by their expected impact, i.e., the eligibility rate multiplied by the average time savings (Figure 5, Table 10). Appendix M provides sensitivity analyses of the intervention rankings. To understand the face validity of these ranking estimates, we asked four clinicians to review these results through a structured interface (Figure 7, protocol in Appendix L).

![](images/ef274103c37ad58d4b5c8313834a78b98634bb32310aef16447ffd3f63f328ef.jpg)  
Fig. 6. Observed and counterfactual trajectories for one hospitalization (ischemic stroke and disposition-critical imaging priority). Each row is one event in the patient’s Gantt chart, placed at its event time in hours from admission, and gray arrows are the patient’s DAG edges. Bar colors give each event’s relation to the intervention: children of A, i.e., the intervention targets ( ), non-descendants of A ( ), and descendants of A that are not children of A ( ). In the counterfactual panel, dashed outlines mark events whose timing changed and are labeled with the shift in hours.

Ranking by expected impact, “Consult response escalation” ranks first at 18.9 hours per screened hospitalization, followed by “Disposition-critical imaging priority” at 15.6. While CI for their estimated impacts are wide, the resulting shortlist is stable, with the same three interventions occupying the top three positions in 89% of the bootstrap resamples (Appendix M). Critically, these causal rankings difer from the simpler option of ranking interventions solely based on the percent of patients who are eligible. “Disposition-critical imaging priority” is ranked as having the most eligible patients, while “Consult response escalation” is estimated to have a higher causal impact. When clinicians were asked to order the top three interventions based on their experiences at the hospital (same across eligibility and egg-computation rankings), we found the egg-computation estimates to align more with clinical intuition: three of the four clinicians ranked “Consult response escalation” above “Disposition-critical imaging priority” (the pair on which the two rankings disagree) and two exactly reproduced the egg-computation ranking, while none reproduced the eligibility ranking. Using the estimated causal efect also swaps the positions of “Urgent IR procedure block” (a protected block for urgent interventional-radiology procedures) and “Antimicrobial stewardship” (which encourages earlier conversion from intravenous to oral antibiotics), moving the former up two positions relative to the eligibility ranking. Its estimated impact is more than double that of “Antimicrobial stewardship”, even though its eligibility rate is less than half. When asked about this, clinicians agreed egg-computation’s ordering was more plausible, as a prolonged IV antibiotic course usually reflects a deliberate clinical decision and oral antibiotics would not be given substantially earlier through the implementation of an automatic IV-to-oral notification system.

Egg-computation results also revealed additional insights into intervention design. For instance, while the third-ranked intervention of early discharge planning had a comparably high eligibility rate (69%, versus 80% for the second-ranked intervention), its causal estimate was much lower (around two-thirds of the second highest causal efect). This estimate was lower than the literature would suggest, as other hospitals have reported that early discharge planning substantially accelerated their workflows (Wassef et al., 2018). Investigating the egg-computation results further, clinicians agreed with the LLM’s analysis that early discharge planning was often blocked by patient needs (e.g., medical needs) that a social worker or case manager could not resolve; in fact, clinicians felt that the LLM was often still too lenient in assuming the social worker or case manager could resolve patient needs early. Based on the results, clinicians hypothesized that realizing the purported efectiveness of this intervention likely required much deeper organizational changes than those originally described in the intervention’s specification, perhaps involving additional hospital staf (e.g., nursing) (Bechir and Bechir, 2025; Bryant-Lukosius et al., 2015).

![](images/bb7af198f6fe6d67c51417d004458e9592c866b114d098800dd6fa041ccd4dc2.jpg)  
Fig. 7. LLM summary of egg-computation output of a candidate intervention provided for clinician review. For each intervention, reviewers were shown the intervention description, the eligibility rate and estimated savings, the distribution of estimated savings across eligible hospitalizations, and an LLM-generated summary, based on all the hospitalizations, of when the intervention did and did not save time. An expandable section provided patient-level examples comparing each observed timeline to its counterfactual. Reviewers were then asked whether the analyses were clinically plausible and whether they would modify the intervention’s design.

Overall, clinicians agreed that the egg-computation analysis was helpful, and they singled out the patient-level counterfactual analyses like those in Figure 6 as particularly helpful. Being able to easily review a hospitalization, what could have happened under the intervention, and how many hours would have been saved made it easier to judge when an intervention would or would not be applicable. By providing the counterfactual Gantt chart visualizations, clinicians could readily audit plausibility of the overall procedure, gain trust its causal reasoning, and provide feedback on how the LLM-assisted system could be further improved.

## 5. Discussion

This work introduces a causal framework for estimating the average time saved by a candidate intervention that has not yet been implemented; this causal question commonly arises in hospital QI but in other contexts as well. The key insight underlying the proposal is that expert knowledge can be combined with data-driven modeling to obtain accurate causal estimates if we carefully elicit information from the right sources: experts should supply information in areas where they are accurate but data support is limited, while data should supply information in areas where modeling is accurate but experts are not. Using the language of Gantt charts as the bridge between expert reasoning and formal causal reasoning, we formalize a model for Gantt charts with both probabilistic and causal semantics and then develop expert-guided g-computation, or egg. Finally, we develop a LLM-assisted implementation to scale expert reasoning required by the framework. Simulations demonstrate improved efect estimation relative to conventional causal inference methods, while the empirical study shows that the LLM-assisted pipeline is a practical tool whose outputs are largely concordant with those of human experts.

The current formulation suggests several natural extensions. In particular, operational interventions often rely on shared resources, so accelerating one patient’s care may afect another patient’s trajectory. Our current estimand should therefore either be interpreted as the efect for patients in the same population strata defined by X keeping the same or efect of the intervention assuming unlimited resources for the proposed intervention and thus an upper-bound. One solution is to apply egg-computation jointly to batches of patients instead, with each patient’s access to an intervention or resource determined by concurrent demand from other patients. Formal inferential procedures are also needed to construct confidence intervals that account for sampling variability and uncertainty in expert-supplied inputs. More extensive calibration with expert judgment will also further strengthen its performance.

To our knowledge, this is the first framework to bring a causal perspective to Gantt-based process improvement, providing an estimation strategy that combines expert knowledge with observed data and an LLM-assisted approach for scaling its implementation. More broadly, eggcomputation can be adapted to other time-ordered processes with task dependencies represented by Gantt charts, such as production scheduling in manufacturing, software development planning, and human resource management. This broad applicability positions egg-computation to become a useful framework for evaluating and guiding operational improvements across complex systems.

## References

Monica Agrawal, Stefan Hegselmann, Hunter Lang, Yoon Kim, and David Sontag. Large language models are few-shot clinical information extractors. Empirical Methods in Natural Language Processing, pages 1998–2022, December 2022.

Gati Aher, Rosa I Arriaga, and Adam Tauman Kalai. Using large language models to simulate multiple humans and replicate human subject studies. arXiv [cs.CL], August 2022.

Alessandro Antonucci, Gregorio Piqué, and Marco Zafalon. Zero-shot causal graph extrapolation from text via LLMs. AAAI Workshop XAI4Sci: Explainable machine learning for sciences, 2024.

Lisa P Argyle, Ethan C Busby, Nancy Fulda, Joshua R Gubler, Christopher Rytting, and David Wingate. Out of one, many: Using language models to simulate human samples. Polit. Anal., 31(3):337–351, July 2023.

Andrew D. Auerbach, Sunil Kripalani, Eduard E. Vasilevskis, Neil Sehgal, Peter K. Lindenauer, Joshua P. Metlay, Grant Fletcher, Gregory W. Ruhnke, Scott A. Flanders, Christopher Kim,

Mark V. Williams, Larissa Thomas, Vernon Giang, Shoshana J. Herzig, Kanan Patel, W. John Boscardin, Edmondo J. Robinson, and Jefrey L. Schnipper. Preventability and causes of readmissions in a national cohort of general medicine patients. JAMA Internal Medicine, 176 (4):484–493, 2016. doi: 10.1001/jamainternmed.2015.7863.

J Banks. Discrete Event System Simulation. Pearson Prentice Hall, 2005.

George Bechir and Angelina Bechir. Start with the end: Early hospital discharge planning as a day-one priority. Cureus, 17(6):e87019, June 2025.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jefrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.

Denise Bryant-Lukosius, Nancy Carter, Kim Reid, Faith Donald, Ruth Martin-Misener, Kelley Kilpatrick, Patricia Harbman, Sharon Kaasalainen, Deborah Marshall, Renee Charbonneau-Smith, and Alba DiCenso. The clinical efectiveness and cost-efectiveness of clinical nurse specialist-led hospital to home transitional care: a systematic review: CNS-led transitional care. J. Eval. Clin. Pract., 21(5):763–781, October 2015.

Nejm Catalyst. What is lean healthcare. NEJM Catalyst, April 2018.

D V Cicchetti and A R Feinstein. High agreement but low kappa: II. resolving the paradoxes. J. Clin. Epidemiol., 43(6):551–558, 1990.

Carlos Cinelli and Chad Hazlett. Making sense of sensitivity: Extending omitted variable bias. Journal of the Royal Statistical Society Series B: Statistical Methodology, 82(1):39–67, 2020. doi: 10.1111/rssb.12348.

W Clark, W N Polakov, and F W Trabold. The Gantt chart: A working tool of management. Ronald Press Company, 1922.

David C. Classen, Roger Resar, Frances Grifin, Frank Federico, Terri Frankel, Nancy Kimmel, John C. Whittington, Allan Frankel, Andrew Seger, and Brent C. James. ‘Global Trigger Tool shows that adverse events in hospitals may be ten times greater than previously measured. Health Afairs, 30(4):581–589, 2011. doi: 10.1377/hlthaff.2011.0190.

Leonardo Cotta, Nikita Dhawan, Rahul Krishnan, Chris Maddison, and Karen Ullrich. End-to-end causal efect estimation from unstructured natural language data. In A Globerson, L Mackey, D Belgrave, A Fan, U Paquet, J Tomczak, and C Zhang, editors, Advances in Neural Information Processing Systems 37, volume 37, pages 77165–77199, San Diego, California, USA, 2024. Neural Information Processing Systems Foundation, Inc. (NeurIPS).

Antonio D’Andreamatteo, Luca Ianni, Federico Lega, and Massimo Sargiacomo. Lean in healthcare: A comprehensive review. Health Policy, 119(9):1197–1209, September 2015.

Vanessa Didelez. Graphical models for marked point processes based on local independence: Graphical models for marked point processes. J. R. Stat. Soc. Series B Stat. Methodol., 70(1): 245–264, February 2008.

A R Feinstein and D V Cicchetti. High agreement but low kappa: I. the problems of two paradoxes. J. Clin. Epidemiol., 43(6):543–549, 1990.

Gaël Gendron, Jože M Rožanec, Michael Witbrock, and Gillian Dobbie. Causal cartographer: From mapping to reasoning over counterfactual worlds. arXiv [cs.AI], May 2025.

Arnab K Ghosh, Said Ibrahim, Jennifer Lee, Martin F Shapiro, and Jessica Ancker. Comparing hospital length of stay risk-adjustment models in US value-based physician payments. Qual. Manag. Health Care, 32(1):22–29, 2023.

Linda Green. Queueing analysis in healthcare. In International Series in Operations Research & Management Science, International Series in Operations Research & Management Science, pages 281–307. Springer US, Boston, MA, 2006.

Frances A. Grifin and Roger K. Resar. IHI Global Trigger Tool for measuring adverse events. IHI innovation series white paper, Institute for Healthcare Improvement, Cambridge, MA, 2009. 2nd edition.

F Richard Guo and Qingyuan Zhao. Confounder selection via iterative graph expansion. Ann. Stat., 54(1):516–541, February 2026.

Kilem Li Gwet. Computing inter-rater reliability and its variance in the presence of high agreement. Br. J. Math. Stat. Psychol., 61(Pt 1):29–48, May 2008.

Andrew F Hayes and Klaus Krippendorf. Answering the call for a standard reliability measure for coding data. Commun. Methods Meas., 1(1):77–89, April 2007.

Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag. TabLLM: Few-shot classification of tabular data with large language models. arXiv [cs.CL], October 2022.

Petter Hurlen, Truls Østbye, Arne S. Borthne, and Pål Gulbrandsen. Does improved access to diagnostic imaging results reduce hospital length of stay? A retrospective study. BMC Health Services Research, 10:262, 2010. doi: 10.1186/1472-6963-10-262.

Zahir Kanjee, Byron Crowe, and Adam Rodman. Accuracy of a generative artificial intelligence model in a complex diagnostic challenge. JAMA, 330(1):78–80, 2023. doi: 10.1001/jama.2023. 8288.

Hossein Keshavarz, George Michailidis, and Y Y Atchade. Sequential change-point detection in high-dimensional gaussian graphical models. J. Mach. Learn. Res., 21, 2020.

Avni Kothari, Patrick Vossler, Jean Digitale, Mohammad Forouzannia, Elise Rosenberg, Michele Lee, Jennee Bryant, Melanie Molina, James Marks, Lucas Zier, and Jean Feng. When the domain expert has no time and the LLM developer has no clinical expertise: Real-world lessons from LLM co-design in a safety-net hospital. Proc. Conf. AAAI Artif. Intell., 2026.

Emre Kıcıman, Robert Ness, Amit Sharma, and Chenhao Tan. Causal reasoning and large language models: Opening a new frontier for causality. transactions of machine learning research, 2024.

Scott Levin, Sean Barnes, Matthew Toerper, Arnaud Debraine, Anthony DeAngelo, Eric Hamrock, Jeremiah Hinson, Erik Hoyer, Trushar Dungarani, and Eric Howell. Machine-learning-based hospital discharge predictions can support multidisciplinary rounds and decrease hospital length-of-stay. BMJ Innovations, 7(2):414–421, 2021. doi: 10.1136/bmjinnov-2020-000420.

Lan Liang, Brian Moore, and Anita Soni. National inpatient hospital costs: the most expensive conditions by payer, 2022. Technical report, Agency for Healthcare Research and Quality, 2026.

Chenxi Liu, Yongqiang Chen, Tongliang Liu, Mingming Gong, James Cheng, Bo Han, and Kun Zhang. Discovery of the hidden world with large language models. Adv. Neural Inf. Process. Syst., February 2024.

Xiaoyu Liu, Paiheng Xu, Junda Wu, Jiaxin Yuan, Yifan Yang, Yuhang Zhou, Fuxiao Liu, Tianrui Guan, Haoliang Wang, Tong Yu, Julian McAuley, Wei Ai, and Furong Huang. Large language models and causal inference in collaboration: A comprehensive survey. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 7668–7684, Stroudsburg, PA, USA, 2025. Association for Computational Linguistics.

Søren Wengel Mogensen and Niels Richard Hansen. Markov equivalence of marginalized local independence graphs. The Annals of Statistics, 48(1):539–559, 2020.

Anthony O’Hagan, Caitlin E. Buck, Alireza Daneshkhah, J. Richard Eiser, Paul H. Garthwaite, David J. Jenkinson, Jeremy E. Oakley, and Tim Rakow. Uncertain Judgements: Eliciting Experts’ Probabilities. John Wiley & Sons, Chichester, UK, 2006. ISBN 978-0470029992.

Judea Pearl. Causal diagrams for empirical research. Biometrika, 82(4):669–688, 1995.

Maya L. Petersen, Kristin E. Porter, Susan Gruber, Yue Wang, and Mark J. van der Laan. Diagnosing and responding to violations in the positivity assumption. Statistical Methods in Medical Research, 21(1):31–54, 2012. doi: 10.1177/0962280210386207.

Project Management Institute. A guide to the project management body of knowledge (pmbok guide). Project Management Institute, 2000.

Thomas S. Richardson and James M. Robins. Single world intervention graphs (SWIGs): A unification of the counterfactual and graphical approaches to causality. Technical Report 128, Center for the Statistics and the Social Sciences, University of Washington Series, 2013.

James Robins. A new approach to causal inference in mortality studies with a sustained exposure period—application to control of the healthy worker survivor efect. Mathematical Modelling, 7 (9):1393–1512, January 1986.

Antonio Rojas-García, Simon Turner, Elena Pizzo, Emma Hudson, James Thomas, and Rosalind Raine. Impact and experiences of delayed discharge: A mixed-studies systematic review. Health Expect., 21(1):41–56, February 2018.

Mike Rother and John Shook. Learning to See: Value Stream Mapping to Add Value and Eliminate Muda. Lean Enterprise Institute, Brookline, MA, 1999. ISBN 978-0966784305.

Kjetil Røysland, Pål C Ryalen, Mari Nygård, and Vanessa Didelez. Graphical criteria for the identification of marginal causal efects in continuous-time survival and event-history analyses. Journal of the Royal Statistical Society Series B: Statistical Methodology, 87(1):74–97, 2025.

Donald B Rubin. Randomization analysis of experimental data: The fisher randomization test comment. Journal of the American statistical association, 75(371):591–593, 1980.

John F Shortle, James M Thompson, Donald Gross, and Carl M Harris. Fundamentals of queueing theory. Wiley Series in Probability and Statistics. John Wiley & Sons, Nashville, TN, January 2018.

Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, Perry Payne, Martin Seneviratne, Paul Gamble, Chris Kelly, Abubakr Babiker, Nathanael Schärli, Aakanksha Chowdhery, Philip Mansfield, Dina Demner-Fushman, Blaise Agüera Y Arcas, Dale Webster, Greg S Corrado, Yossi Matias, Katherine Chou, Juraj Gottweis, Nenad Tomasev, Yun Liu, Alvin Rajkomar, Joelle Barral, Christopher Semturs, Alan Karthikesalingam, and Vivek Natarajan. Large language models encode clinical knowledge. Nature, 620(7972):172–180, August 2023.

Alexandra B Steverson, Paul J Marano, Caren Chen, Yifei Ma, Rachel J Stern, Jean Feng, Efstathios D Gennatas, James D Marks, Matthew S Durstenfeld, Jonathan D Davis, Priscilla Y Hsue, and Lucas S Zier. Predictors of all-cause thirty-day readmissions in patients with heart failure at an urban safety net hospital: The importance of social determinants of health. American Journal of Medicine Open, page 100060, September 2023.

Hari Subramonyam, Roy Pea, Christopher Pondoc, Maneesh Agrawala, and Colleen Seifert. Bridging the gulf of envisioning: Cognitive challenges in prompt based interactions with LLMs. In Proceedings of the CHI Conference on Human Factors in Computing Systems, volume 31, pages 1–19, New York, NY, USA, May 2024. ACM.

Alex Tamkin, Kunal Handa, Avash Shrestha, and Noah Goodman. Task ambiguity in humans and language models. In International Conference on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2212.10711.

The Press Association. Hospital discharge delays ’cost NHS £100m’. https://www.hsj.co.uk/fi nance-and-efficiency/hospital-discharge-delays-cost-nhs-100m/5068992.article, March 2014. Accessed: 2026-7-21.

Tyler J VanderWeele. Principles of confounder selection. Eur. J. Epidemiol., 34(3):211–219, March 2019.

Pilar I Vidal-Carreras, Julio J Garcia-Sabater, and Juan A Marin-Garcia. Applying value stream mapping to improve the delivery of patient care in the oncology day hospital. Int. J. Environ. Res. Public Health, 19(7):4265, April 2022.

Patrick Vossler, Jean Feng, Venkat Sivaraman, Robert Gallo, Hemal Kanzaria, Dana Freiser, Christopher Ross, Amy Ou, James Marks, Susan Ehrlich, Christopher Peabody, and Lucas Zier. From fuzzy to formal: Scaling hospital quality improvement with AI. arXiv [cs.AI], April 2026.

Yoav Wald, Mark Goldstein, Yonathan Efroni, Wouter A C van Amsterdam, and Rajesh Ranganath. Time after time: Deep-Q efect estimation for interventions on when and what to do. In The Thirteenth International Conference on Learning Representations, October 2024.

Maggy Wassef, Marc-Olivier Trépanier, Julie Mayrand, Martine Habra, and Sylvie Beauchamp. Efectiveness of discharge planning and transitional care interventions in reducing hospital readmissions for the elderly. Technical report, Montréal, 2018.

Daniel Westreich and Stephen R Cole. Invited commentary: positivity in practice. Am. J. Epidemiol., 171(6):674–7; discussion 678–81, March 2010.

James M Wilson. Gantt charts: A centenary appreciation. Eur. J. Oper. Res., 149(2):430–437, September 2003.

Michael Wornow, Elsie Gyang Ross, Alison Callahan, and Nigam H Shah. APLUS: A python library for usefulness simulations of machine learning models in healthcare. J. Biomed. Inform., 139(104319):104319, March 2023.

Sewall Wright. The method of path coeficients. The Annals of Mathematical Statistics, 5(3): 161–215, 1934.

Obert Xu, Keith Cross, Louis P. Riccelli, Steven McGaughey, Jack Marshall, Kenneth DeVane, Skyler Kieran, and Beech Burns. Breaking the bottleneck: a quality improvement initiative to reduce overnight CT turnaround times in the emergency department. Emergency Radiology, 33 (2):291–303, 2026. doi: 10.1007/s10140-026-02449-3.

Jiaming Zeng, Michael F. Gensheimer, Daniel L. Rubin, Susan Athey, and Ross D. Shachter. Uncovering interpretable potential confounders in electronic medical records. Nature Communications, 13:1014, 2022. doi: 10.1038/s41467-022-28546-8.

## APPENDIX

## A. Proofs for egg-computation

The following lemmas, established for Type I and II nodes respectively, are used to prove Theorem 3.1. In this Appendix, we use the shorthand $[ k ] : = \{ 0 , \ldots , k \}$ with $[ - 1 ] : = \varnothing$

Lemma A.1 Under Assumptions 1 to 4, for $k \in \operatorname { N d } ( A )$ , we have $T _ { k } ( 1 ) = T _ { k }$

Proof of Theorem A.1. Observe that $\operatorname { N d } ( A )$ is ancestral in the sense that $j \in \operatorname { N d } ( A )$ and $l \in \mathrm { P a } ( j )$ imply $l \in \operatorname { N d } ( A )$ . In light of this, the result follows from the first case of Assumption 3 and $\mathrm { E q . ~ ( 2 . 3 ) }$ □

Lemma A.2 Under Assumptions 1 to 4, for $k \in \mathrm { D e } ( A ) \backslash \mathrm { C h } ( A )$ , we have the equality in conditional distributions:

$$
p \left( W _ { k } ( \mathbf { 1 } ) = w _ { k } \mid T _ { [ k - 1 ] } ( 1 ) = t _ { [ k - 1 ] } , X , \mathbf { D } \right) = p \left( W _ { k } = w _ { k } \mid T _ { \mathrm { P a } ( k ) } = t _ { \mathrm { P a } ( k ) } , X , \mathbf { D } \right) ,
$$

Proof of Theorem A.2. For any $w _ { k }$ and t, we have

$$
p \left( W _ { k } ( 1 ) = w _ { k } \mid T _ { [ k - 1 ] } ( 1 ) = t _ { [ k - 1 ] } \right)
$$

$$
\mathrm { ( b y ~ A s s u m p t i o n ~ 4 ) } = p \left( W _ { k } ( 1 ) = w _ { k } \mid T _ { \mathrm { P a } ( k ) } ( 1 ) = t _ { \mathrm { P a } ( k ) } \right)
$$

(by 2nd case of Assumption 3) = p  W<sub>k</sub>(0) = w<sub>k</sub> | T<sub>Pa(k)</sub>(0) = t<sub>Pa(k)</sub>

$$
\mathrm { ( b y ~ A s s u m p t i o n ~ 2 ) } = p \left( W _ { k } ( 0 ) = w _ { k } \mid T _ { \mathrm { P a } ( k ) } ( 0 ) = t _ { \mathrm { P a } ( k ) } , A = 0 \right)
$$

(by Assumption 1) = p  W<sub>k</sub> = w<sub>k</sub> | T<sub>Pa(k)</sub> = t<sub>Pa(k)</sub> .

Remark A.3 (Pooling assumption for the Type II timing models) Theorem A.2 reduces the counterfactual waiting-time distribution of a Type II node to the factual factor $p \left( W _ { k } \mid T _ { \operatorname { P a } ( k ) } \right)$ , but estimating this factor from data raises two issues that the pooled model of Section 3.2.2 is designed to address. First, a single hospitalization contributes only one draw from each factor, so any estimator must pool information across patients. We therefore assume that hospitalizations whose child event and parent configuration match are drawn from the same waiting-time distribution, which makes the factors identifiable from a population where each configuration occurs. Exact configuration matches are essentially non-existent in practice, since event descriptions are free text, sets of parent nodes difer from patient to patient, and parent timings are continuous. As a result, the pooled model uses information from hospitalizations that are similar but not identical, assuming that events with similar descriptions, parent timings, and patient context follow similar timing distributions. Whereas tabular and embedding models fix a notion of similarity in advance through their features, an LLM can weigh similarity through the clinical context of the descriptions themselves, which motivates the in-context learning option of Section 3.2.2. Theorem 3.1 requires the Type II timing models to equal the true conditional distributions, and in our implementation that condition holds to the extent this similarity assumption does. Second, pooling changes what the model must condition on. The per-patient statements of Section 2.1 hold conditionally on the baseline covariates X, so Theorem A.2 suppresses X from the notation: the DAG is constructed from the patient’s own chart, Type I node timings are copied from the same trajectory, and Type III node distributions are specified for the intervention acting on that patient, so none of these steps combines information across patients and each conditions on X implicitly. A model pooled across patients loses this implicit conditioning, and because the waiting time between parent and child events may depend on patient characteristics, it must condition on X explicitly, i.e., estimate $p \left( W _ { k } \mid T _ { \operatorname { P a } ( k ) } , X \right)$

Lemma A.4 (g-computation) For $k = 0 , 1 , \ldots , K$ , let $T _ { k } ^ { \prime } ( 1 )$ be iteratively computed as follows: 1. if $k \in \operatorname { N d } ( A )$ , let $T _ { k } ^ { \prime } ( 1 ) = T _ { k }$ ;

2. if $k \in \operatorname { C h } ( A )$ , draw $W _ { k } ^ { \prime } ( 1 ) \sim p \left( W _ { k } ( 1 ) \mid T _ { \mathrm { P a } ( k ) } ( 1 ) = T _ { \mathrm { P a } ( k ) } ^ { \prime } ( 1 ) , X , \mathrm { D } \right)$ and compute $T _ { k } ^ { \prime } ( 1 )$ via (2.3);

3. if $k \in \operatorname { D e } ( A ) \setminus \operatorname { C h } ( A )$ , draw $W _ { k } ^ { \prime } ( 1 ) \sim p \left( W _ { k } \mid T _ { \mathrm { P a } ( k ) } = T _ { \mathrm { P a } ( k ) } ^ { \prime } ( 1 ) , X , \mathrm { D } \right)$ and compute $T _ { k } ^ { \prime } ( 1 )$ via (2.3).

Then, we have $T _ { K } ^ { \prime } ( 1 ) = _ { d } Y ( 1 )$

Proof of Theorem A.4. By marginalization, it sufices to show

$$
( T _ { 0 } ^ { \prime } ( 1 ) , T _ { 1 } ^ { \prime } ( 1 ) , \ldots , T _ { K } ^ { \prime } ( 1 ) ) \mid X , \mathrm { D } = _ { d } ( T _ { 0 } ( 1 ) , T _ { 1 } ( 1 ) , \ldots , T _ { K } ( 1 ) ) \mid X , \mathrm { D } .
$$

This identity follows from applying Theorems A.1 and A.2 and Assumption 4 to each corresponding term in the factorization

$$
{ \begin{array} { r l } & { p ( T _ { 0 } ( 1 ) , T _ { 1 } ( 1 ) , \dots , T _ { K } ( 1 ) \mid X , \mathbf { D } ) = p \left( T _ { \mathrm { N d } ( A ) } ( 1 ) \mid X , \mathbf { D } \right) } \\ & { \qquad \times \prod _ { k \in \operatorname { C h } ( A ) } p \left( T _ { k } ( 1 ) \mid T _ { [ k - 1 ] } ( 1 ) \mid X , \mathbf { D } \right) \prod _ { k \in \operatorname { D e } ( A ) \setminus \operatorname { C h } ( A ) } p \left( T _ { k } ( 1 ) \mid T _ { [ k - 1 ] } ( 1 ) \mid X , \mathbf { D } \right) } \end{array} }\tag{A.1}
$$

We are ready to prove our main result. Proof of Theorem 3.1. Given the additional Assumptions 5 to 7 about the expert’s specification, the first statement follows from Theorem A.4. Then, by

linearity of expectation, we have

$$
\begin{array} { r } { \mathbb { E } \hat { \tau } = \mathbb { E } Y - \mathbb { E } T _ { K } ^ { \prime } ( 1 ) = \mathbb { E } Y ( 0 ) - \mathbb { E } Y ( 1 ) = \tau , } \end{array}
$$

where we used the fact that $Y = Y ( 0 )$ by Assumption 1.

## B. Related work

Causal Inference Methods and their Assumptions. Causal inference methods for longitudinal data allow treatment efects to be estimated over long-ranging time sequences, such as using g-computation methods for discrete time grids (Robins, 1986) and Local Independence Graphs for continuous time (Didelez, 2008; Mogensen and Hansen, 2020; Røysland et al., 2025; Wald et al., 2024). Nevertheless, existing methods largely rely on the core assumptions of conditional exchangeability, consistency, and positivity. Furthermore, existing methods typically assume the intervention’s efect is encapsulated by a simple formula/function. However, complex interventions, such as those described via text, may not have such simple descriptors. Finally, existing methods typically assume a common causal graph, whereas the causal graph may not be constant over time (Keshavarz et al., 2020).

Role of Experts in Causal Inference. In traditional causal inference methods, experts play the role of specifying the causal structure, as causal structure cannot be determined from data alone. To this end, various guidelines have emerged regarding what variables to include in the causal graph (VanderWeele, 2019; Guo and Zhao, 2026). Nevertheless, traditional applications rely on human judgment, which makes these methods dificult to scale. Beyond specifying causal structure, experts are also relied upon to supply quantitative judgments, such as eliciting efect magnitudes as probability distributions (O’Hagan et al., 2006) and bounding the plausible strength of unmeasured confounding in sensitivity analyses (Cinelli and Hazlett, 2020).

LLMs for Causal Discovery and Mimicking Expert Reasoning. Given the technical and practical dificulties of causal DAG construction, there is now increasing interest and evidence that LLMs can assist in conducting causal reasoning (Liu et al., 2025). Within causal inference/discovery, prior works have demonstrated that LLMs can accurately infer pairwise causal relationships simply through variable names (Kıcıman et al., 2024; Liu et al., 2024). LLMs have also achieved good results when inferring larger multi-node causal graphs from scientific/medical abstracts and news articles (e.g., through iterated pairwise queries) (Antonucci et al., 2024; Gendron et al., 2025). Most importantly, with proper calibration, LLMs can accurately mimic human reasoning (Argyle et al., 2023; Aher et al., 2022), which motivates our approach in using LLMs to conduct expert-guided causal reasoning. We do not claim to perform exact causal reasoning; because the graph itself is a reflection of expert judgment, this mirrors standard practice in causal inference. To our knowledge, the closest work on using LLMs to conduct end-to-end causal inference is (Cotta et al., 2024), which considers a diferent setting where the goal is to scrape internet data to estimate the causal efect by (i) extracting the outcome, treatment, and confounders and (ii) running classical causal inference methods such as IPW.

Informatics Approaches to Healthcare Flow Modeling. Parallel to the statistical literature, informatics and operations research communities have long examined hospital flow through process-improvement frameworks such as Lean methodology (Catalyst, 2018; D’Andreamatteo et al., 2015), Gantt charts (Wilson, 2003; Clark et al., 1922), discrete-event simulation (DES) (Banks, 2005), and queueing theory (Shortle et al., 2018; Green, 2006). Nevertheless, existing methods lack rigorous causal identification guarantees and are not designed to leverage unstructured data to improve the causal reasoning. This work illustrates how the operational practice

## REFERENCES

of modeling with Gantt charts can be combined with formal causal reasoning methods (e.g., g-computation variants) to produce causal estimates with explicit identification assumptions.

## C. The LLM pipeline for DAG construction

This section describes what each of the pipeline’s five steps instructs the LLM to produce, how the graphs are kept compact, and the checks applied to the output. Each admission’s notes, orders, and results are merged into a single timeline with repeated note text deduplicated; these timelines have a median length of roughly 48,000 tokens. The first step reads this full patient timeline and the operational intervention and, in free-form clinical prose, identifies the events that determine length of stay. It enumerates the intervention targets (Type III nodes) that the intervention directly re-times, the sink-gating events that must complete before discharge can occur, and, tracing backward from both toward admission, the intermediate events and candidate confounders that lie along those causal paths. The confounder search is carried out in this first step, following the iterative graph-expansion view of Guo and Zhao (2026) and the disjunctive cause criterion of VanderWeele (2019), so a pre-exposure variable is retained when it is a cause of an intervention target, a discharge-blocking event, or both. The second and third steps turn this reasoning into the Gantt chart’s fixed set of events and then the timing dependencies among those events. The second step finalizes the event set by deduplicating and filtering the first step’s events, giving each event a start and an end timestamp read from the chart, and tagging which events the intervention directly afects. The third step holds the event set fixed and records, for every event, which other events’ timings determine when it starts and, for events with nonzero duration, when it ends. The fourth and fifth steps re-express the same content in the structured graph format that the downstream code consumes, again splitting the work into nodes first and edges second. The fourth step emits the graph’s nodes with their clinical category, event type, and timestamps, while the fifth step emits the directed edges, each carrying a one-sentence causal rationale and a verbatim supporting quote from the timeline. Consistent with the design principle stated in the main text, the earlier steps are prompted to spend their reasoning on clinical and causal judgment and the later steps to spend it on faithful transcription into the required format. Algorithm 1 summarizes the recipe the prompts encode.

No step sets an explicit target for the number of nodes or edges; the reported average of roughly 14 nodes and 21 edges follows from two instructions rather than a fixed quota. The first is a granularity target applied during backward tracing: the first step is told to “aim for a granularity such that all causal links span no more than” a set number of hours (twenty-four in our runs), with any longer link “split further by adding more detailed event breakdowns.” This adds nodes where consecutive events are far apart in time, the gaps that are hardest to model and have the least data support. The second is the stopping rule implied by the disjunctive cause criterion, under which backward expansion halts once every retained pre-exposure variable is a cause of an intervention target or a sink-gating event, bounding the confounder set rather than requiring an enumeration of all baseline covariates. The second step then removes near-duplicate events and drops any event that does not help determine the timing of another event, trimming the graph before it is formalized in the structured steps.

Once the structured node and edge steps have run, each graph passes through a set of deterministic checks before it is accepted. The extracted timestamps are compared against the source timeline, and any timestamp with no charted event within an hour is flagged for later review. The node and edge text is then parsed into a graph, and a parse failure or a structural problem triggers one automatic repair attempt in which the ofending output is returned to the model together with the specific errors found and a request to fix only those. Structural validation checks for cycles using a topological sort over the split start and end endpoints of each event and confirms that every event has a directed path to the discharge node. Small temporal inconsistencies in which a parent’s timestamp slightly postdates its child’s are corrected automatically within a few-minute tolerance, while larger violations are recorded as issues. Events that have no cause or no downstream efect are pruned, with the exception of the reserved admission, baseline, and discharge nodes and the intervention targets, which are never dropped. Because the baseline nodes are connected to every non-baseline event automatically during post-processing, the LLM is instructed to omit those edges from its output. Graphs that still fail structural validation after the single repair attempt are kept and marked rather than discarded; the sensitivity of the timing results to these graphs is reported in Section J.

<table><tr><td>Algorithm 1 The DAG-construction recipe encoded in the pipeline&#x27;s prompts. The recipe is realized as a fixed sequence of five model calls rather than a literal node-by-node loop; the first call performs the full backward trace.</td></tr><tr><td>1: Seed the graph with the sink event (discharge/death in this setting); the DAG describes the hospitalization as observed, and the intervention node A is attached automatically after</td></tr><tr><td>construction. 2: Add the intervention targets, the events whose timing the intervention directly changes, and</td></tr><tr><td>the sink-gating events (the direct parents of the sink). 3: Trace backward from each sink-gating event, adding the intermediate events that determine</td></tr><tr><td>its timing. 4: Trace backward from each intervention target in the same fashion.</td></tr><tr><td>5: Following the disjunctive cause criterion, retain the confounders: pre-exposure variables that cause an intervention target, a sink-gating event, or both, including a baseline patient-</td></tr><tr><td>characteristics node. 6: Record dependencies that link parallel care pathways, where an event is gated by an event on a different pathway; these become edges in the later steps.</td></tr><tr><td>7: Finalize the DAG over the baseline, admission, intervention-target, intermediate, confounder, sink-gating, and sink nodes, with edges defined accordingly.</td></tr></table>

## D. Prompt templates

This section reproduces the two timing-query templates referenced in Section 3.2.2. Curly-brace fields mark runtime substitutions, and the remaining pipeline prompts are available in the accompanying software package.

## D.1 Type III node timing query

This prompt elicits the revised timestamp of an intervention target (a Type III node) under the proposed intervention. At runtime, intervention\_text receives the intervention’s label, description, and recommended change from its specification; patient\_summary a short summary of the hospitalization; parent\_text the target’s parent events with their current timestamps; and child\_text the target event itself, without a timestamp. GPT-5-mini returns a short rationale and a single revised timestamp, to which the capping rule of Appendix M is applied.

## # TASK BACKGROUND

## REFERENCES

As a member of the hospital’s Quality Improvement team, you have suggested   
implementing the following operational intervention to reduce overall LOS in   
the hospital:   
II Il "I   
{intervention\_text}   
II II I

Your task now is to look at a snippet of the patient’s encounter that is relevant to this intervention and determine the anticipated timestamp of the event of interest, given recent events for this patient that are causally relevant for determining the timestamp of this event in question.

\# PATIENT INFO Patient summary: {patient\_summary}

What has happened recently for this patient that are relevant: {parent\_text}

The event that we need you to determine the timestamp for: {child\_text}

Consider:

1. Your planned intervention as described and how it would apply to this specific patient encounter

2. The parent node timestamps (the new timestamp cannot be earlier than any parent)

Output your reasoning and the new timestamp in JSON format:

"reasoning": <your reasoning about how the intervention shifts this event>, "new\_timestamp": "<YYYY-MM-DD HH:MM>" }

## D.2 Type II node timing query

This prompt elicits the delay, in hours, between a downstream event (a Type II node) and the latest of its parent events. At runtime, training\_csv receives fifteen worked examples from other hospitalizations, matched on parent and child event type where possible and completed at random, each giving the patient summary, the parent and child event descriptors, and the observed delay. The patient\_summary, parent\_text, and child\_text fields describe the query itself, and GPT-5 returns a step-by-step rationale and a single delay.

## # TASK BACKGROUND

You are predicting the gap time between nodes in a hospital DAG that represents the dependencies within an inpatient encounter, which will help us reason about overall LOS.

## # WHAT IS A DAG?

```markdown
# NODE TYPES
```

The DAG represents causal relationships between clinical events for a hospital patient. DAG nodes are instantaneous clinical events and edges represent causal dependencies.

\- start or end nodes of a process: indicates start/end of a clinical event

\- action: Instantaneous decision/order (start=end) (e.g. PO ordered)

\- start or end nodes of a waiting period: indicates start/end of a waiting period, potentially modifiable by intervention

## # GAP TIME

Defined as the gap time between the child node and the maximum time among parent nodes (i.e. time of latest event)

# TRAINING EXAMPLES   
Here are examples from similar patient encounters:   
{training\_csv}

## # YOUR TASK

Predict target\_gap\_hours for this edge:   
{patient\_summary}parent\_text: {parent\_text}   
child\_text: {child\_text}

Think step by step about the clinical process, what the likely last parent event is and what it represents. Use your clinical judgment on what is the most likely length, along with looking at the provided examples of typical durations at this hospital.

Output your reasoning and final prediction in hours.

Format your answer in JSON format:   
{   
"reasoning": <your reasoning>,   
"prediction": <number in hours>   
}

## E. Simulation details

Estimation methods. We compare a set of estimation methods that difer along two axes: the causal structure they assume, either the patient’s own DAG or a single shared DAG fit to the whole cohort, and how they determine the Type II timings downstream of the intervention, either by exact event propagation or by a fitted model. The methods built on the patient-specific DAG use the true latent structure recorded in the note, namely which reports the decision waits on and which report’s timing depends on the other, an idealization that isolates the value of the patient-specific structure from the separate task of recovering the label from the note. We describe each method in turn.

## REFERENCES

(i) Oracle: Computes the counterfactual mechanically from the true data-generating structure by clipping the targeted delay at the cap. This serves as the ground truth against which bias is measured and does not appear as a row in Table 1.

(ii) DAG-extractor + Gantt-timing: Egg-computation using each patient’s correct DAG, propagating the capped report time through the exact Gantt update that reuses the factual time diferences (Gantt-timing, Section 4.1). With the correct DAG this method recovers the truth, so it is the oracle configuration and an upper bound on what a perfect Type II step can achieve.

(iii) DAG-extractor + OLS-timing, DAG-extractor + kNN-timing: Use the same correct DAG, but the Type II timings are estimated rather than propagated exactly. The OLS variant fits linear models for the downstream report and the decision, conditioned on which reports the patient’s decision waits on and which report’s timing depends on the other; the kNN variant estimates the counterfactual delay using a k-nearest-neighbor imputation from patients whose targeted delay already falls below the cap, mirroring the in-context retrieval the LLM Type II step performs. We select k per intervention by leave-one-out cross validation on the observations already below the cap. Per-patient error and ATE bias are insensitive to the choice of k (Appendix F) because the kNN variant’s residual error arises from estimating the mean of the below-cap observations rather than from the neighborhood size.

(iv) Single-DAG + Gantt-timing: The same exact Gantt update used by DAG-extractor + Gantt-timing, applied to a single shared DAG (every patient is assumed to wait on both reports with one common dependence direction). Because the propagation of events has no estimation error, the two Gantt-timing methods share an identical, noiseless Type II step and difer only in the assumed DAG, so any error results from the wrong DAG structure. We report both orientations, CT→MRI and MRI→CT.

(v) Single-DAG + OLS-timing: Fits a pooled g-computation model that assumes the same common DAG structure with a fitted linear Type II step (ordinary least squares for the downstream report and the decision). We report both dependence direction orientations, (CT→MRI) and (MRI→CT).

We report each quantity averaged over 50 independent simulation replications, with its standard error across replications.

## F. Sensitivity of DAG-extractor + kNN-timing to the neighborhood size k

The kNN-timing model imputes the counterfactual delay of each patient whose delay is capped from the k nearest neighbors in baseline acuity, a baseline severity covariate in the simulation, among patients whose targeted delay already falls below the cap, so the neighborhood size k is a tuning choice that could in principle drive its error. We select k per intervention by leave-one-out cross-validation among the observations with delays below the cap, predicting each donor’s own delay from its k nearest neighbors and minimizing held-out MAE. This is the only model-selection signal available without the counterfactual, and it mirrors how a practitioner would set the number of in-context exemplars in the LLM Type II step.

Figure 8 sweeps k from a single neighbor to several hundred and shows that both the per-patient MAE and the ATE bias are flat to within 0.1 hours across the entire grid, for both interventions, and far below the best-performing Single-DAG method throughout. This is because every donor observation lies below the 12-hour cap, so the kNN variant imputes a post-intervention delay below the cap regardless of the number of neighbors used. Its residual error therefore arises from conditioning on observations below the cap, not from the choice of k.

![](images/1a6641e7806dbad4e8c2ef3d29ceecece977b9e952415fca3b496cdec4be2f71.jpg)

![](images/9953175b05e92991130a10fe049376d7c2c7132cef0cfe1b08d82073384901ba.jpg)  
Fig. 8. DAG-extractor + kNN-timing error as a function of the neighborhood size $k ,$ averaged over the 50 replications (bands are ±1 standard error). Left: per-patient MAE, with the dotted lines marking the lowest-MAE Single-DAG method for each intervention. Right: signed bias of the estimated average time saved The dashed vertical lines are the cross-validation-selected k. Both panels are flat in $k ,$ and the MAE stays far below the Single-DAG floor throughout, so the kNN variant’s error reflects its estimation strategy rather than the choice of k.

## G. Definitions of the candidate interventions

The eleven candidates were drawn from a prior LLM-assisted analysis of the most frequent bottlenecks and delays at the hospital (Vossler et al., 2026), and each was specified as an operational rule together with the inclusion and exclusion criteria that determine which hospitalizations are eligible. Each candidate is operational in the sense that it changes when already-indicated work is completed, not what care the patient receives. Each entry below reproduces the intervention’s operational rule verbatim from its specification, followed by a summary of its eligibility scope; the complete specifications given to the LLM for eligibility determination and Type III timing are provided in the accompanying software package.

Consult response escalation. Rule: For a prioritized ancillary consult (SLP, social work, PT, or OT) that gates discharge-relevant work, pending consult age is tracked on the flow navigator and escalated at 12 hours without a first note. As such, order-to-first-evaluation should generally not exceed 12 hours, which encompasses acknowledgment of the order, the service reaching the patient, and the initial evaluation note.

The current median order-to-first-evaluation is about 1.5 days for the slowest of these services. A hospitalization is eligible when such an ancillary consult is ordered and its evaluation gates functional clearance, the placement decision, or discharge planning. Physician and medical-team consults (for example cardiology, neurology, or surgery) are not in scope, and a hospitalization is excluded when the patient could not be evaluated within the window for medical reasons, a clinical prerequisite had to occur first (such as extubation before a swallow evaluation), or the entire window falls on a weekend or holiday when the service does not staf evaluations.

Disposition-critical imaging priority. Rule: If a diagnostic imaging study (TTE, TEE, CT with contrast, MRI, or ultrasound) is needed for deciding patient disposition, the study is prioritized so that the study will be completed the same calendar day if the study is ordered before 3pm and will be completed by noon the following day if ordered after 3pm. As such, order-to-completion for a flagged study should generally never exceed 24 hours, which encompasses the time of acknowledgment of the order, transportation of the patient, capturing the image, and the final report on the imaging study.

A hospitalization is eligible when one of these modalities is ordered and its result gates the disposition decision. This is one of the two interventions used in the annotation study, and its full definition as shown to annotators is reproduced in Appendix I.

Early discharge planning. Rule: Starting at the 1 day mark (24 hours), all patients will be evaluated for discharge planning. Discharge planning will then proceed within the following 24 hours, and patients with complex discharge needs relating to housing, post-acute facility placement (SNF, acute rehab, LTACH), hospice, durable medical equipment, transportation, home health, IHSS, social-benefit applications (e.g., Medi-Cal), and/or substance use will meet with social work or case management within the 24 hour (i.e., 24–48 hours) time period. Qualifying complex patients are: (1) unhoused patients or patients living in a shelter; (2) patients who came directly from a SNF or another facility/program; (3) patients who had home services; (4) patients who, based on the first 24 hours, will need to be set up with home services, a SNF, acute rehab, a long-term acute care facility (LTACH), or a hospice referral. If the patient agrees, the social work/case manager will contact appropriate discharge resources within 24 hours after meeting with the patient.

A hospitalization is eligible when a complex-needs placement or discharge-services referral occurs and the need was identifiable within the first 24 hours of admission. This is the second annotationstudy intervention, and its full definition as shown to annotators is reproduced in Appendix I.

Discharge before noon. Rule: At 4 PM each day, evening discharge rounds identify patients expected to discharge the following day; for those patients, the team enters the discharge order and prescriptions and completes readiness tasks that evening. As such, on the discharge day itself, the discharge order should generally be in by 11 AM and the patient should leave before noon—the intervention moves the discharge earlier within the day, not to a diferent day.

The current median discharge time is near 4 PM. A hospitalization is eligible when the patient left on a routine discharge and was documented by 4 PM the prior day as likely to discharge the next day. It is excluded when the discharge still depended on a result or consult that could not be anticipated that evening, when new instability is documented overnight, or when the departure time was set by the receiving side rather than by discharge-task readiness.

Weekend discharge coordinator. Rule: Discharge coordinators staf Saturday and Sunday from 8 AM to 4 PM, rounding on patients flagged at Friday interdisciplinary rounds as likely to discharge within 48 hours and completing the coordination tasks that would otherwise wait for Monday. As such, a patient who reaches medical readiness between Friday afternoon and Sunday, and whose remaining barriers are coordination tasks, should generally be discharged the same or next day rather than held to Monday.

A hospitalization is excluded when a clinical barrier remains, when the discharge requires a weekday-only service such as a pending consult or placement decision that coordinator coverage cannot resolve, or when the departure timing is set by the receiving facility.

Urgent IR procedure block. Rule: An IR triage physician reviews pending inpatient requests by 10 AM and directs urgent flow-related cases—active infection requiring drainage, an obstructed system with sepsis, vascular access for time-sensitive therapy—into 2 protected daily slots. As such, request-to-completion for a qualifying urgent IR procedure should generally not exceed 24 hours. The rule bounds the procedure, not the finalized report.

The current median request-to-completion is near one day, with the slowest tenth near six days. Elective and outpatient cases are out of scope, and a hospitalization is excluded when the patient was not medically stable for the procedure, a clinical prerequisite had to occur first (such as anticoagulation reversal), or consent could not be obtained within the window.

Same-day echo completion. Rule: A discharge-prioritized TTE ordered before 11 AM is completed the same calendar day in a reserved slot; TEEs get add-on spots, and any TEE not completed within 24 hours of order is escalated to cardiology and anesthesia leadership—anesthesia scheduling delays are within the escalation’s reach. As such, order-to-completion for a discharge-prioritized TTE or TEE, including the report, should generally not exceed 24 hours.

A hospitalization is eligible when a TTE or TEE on the discharge critical path is ordered, for example for suspected endocarditis, new heart failure, or preoperative cardiac clearance. It is excluded when the patient was not medically stable for the study, a clinical prerequisite had to occur first (such as NPO status for a TEE), or the 24-hour window falls entirely on a weekend or holiday when the reserved slots are unstafed.

Antimicrobial stewardship. Rule: When a patient with sepsis on IV antibiotics meets oral conversion criteria—afebrile >24h, tolerating PO, clinically improving, and culture susceptibilities showing an oral option—an automated alert fires and a pre-authorized pharmacist protocol converts standard regimens without waiting for rounds. As such, the time from meeting conversion criteria to the IV-to-PO conversion order should generally not exceed 12 hours.

The current median time from meeting criteria to conversion is about one day. A hospitalization is excluded when the infection requires sustained IV therapy (for example endocarditis, bacteremia requiring prolonged IV, or a central-nervous-system infection), when the patient cannot tolerate oral intake, or when no oral-susceptible organism is available.

Weekend SNF intake. Rule: Partner SNFs accept weekend admissions with intake by 2 PM; case management completes referral packets by Thursday for patients anticipated to be SNF-ready over the weekend, partner facilities confirm bed availability by Friday noon, and transportation is pre-arranged. As such, a patient with a SNF disposition identified by Thursday who reaches medical readiness between Friday and Sunday should generally transfer the same or next day rather than waiting until Monday.

A hospitalization is excluded when the SNF need arose too late to anticipate, when the patient requires acute rehabilitation or a long-term acute care hospital rather than a SNF, or when the patient needs services no partner facility provides.

Weekend PT/OT coverage. Rule: PT and OT therapists staf Saturday and Sunday from at least 8 AM to 2 PM, working from the Friday-rounds flag list and prioritizing patients who could discharge the same day if cleared. As such, a patient pending only PT/OT clearance over the weekend should generally be evaluated by Sunday noon rather than waiting for Monday stafing. A hospitalization is excluded when a clinical barrier beyond therapy clearance remains, when the patient’s needs require inpatient-rehabilitation intensity rather than a clearance evaluation, or when the patient cannot participate in the evaluation at any stafed time.

Weekend diagnostic slots. Rule: Inpatient studies whose result gates discharge get reserved weekend capacity—at least 4 echo and 6 CT/MRI slots per weekend day, scheduled first-come from Friday noon, with anything unfinished by end of Saturday escalated to the on-call radiologist or cardiologist for Sunday morning. As such, a discharge-critical study should generally be completed, including the report, within 24 hours of its weekend clock start—the order time for studies ordered after Friday noon, and Friday noon for studies ordered earlier and still pending then—rather than held to Monday.

A hospitalization is excluded when the patient was not medically stable for the study, when the study requires a weekday-only resource or specialized stafing the weekend lacks (such as anesthesia support for a TEE), or when a clinical prerequisite had to occur first.

## H. Computational cost of the LLM pipeline

Token usage was logged for every LLM call in the real-data pipeline; the figures below price those tokens at OpenAI list rates (GPT-5: \$1.25 per million input tokens and \$10 per million output tokens; GPT-5-mini: \$0.25 and \$2, respectively). The full pipeline consumed 562 million input tokens and 90 million output tokens, for a total cost of approximately \$560 at standard rates, or roughly \$490 after prompt-caching discounts. By stage, screening all 2,193 hospitalizations for eligibility against the eleven interventions cost \$29, constructing the 1,531 DAGs cost \$215 (about 14 cents per DAG), the Type III timing draws cost \$15, and the Type II ICL predictions (both the model evaluation and the counterfactual propagation) cost \$305. The non-LLM timing baselines and the final averaging step incur no API cost.

## I. Annotation study protocol and metric definitions

Interface and review protocol. Annotators reviewed the LLM-generated DAGs through a web based annotation interface that displays the events extracted for a hospitalization, the intervention description, and the intervention-relevant portion of the graph. Each annotation task is one annotator’s review of one intervention-hospitalization pair and proceeds in a fixed order. First, the annotator confirms or rejects each event the LLM flagged as an intervention target and reviews the full list of extracted events to flag any targets the LLM missed. Second, for each event reachable from the corrected intervention targets, the annotator approves or deletes each edge the LLM proposed into that event and reviews earlier events ofered as candidate missing parents, adding any edges the LLM omitted. Annotators corrected the LLM’s graph rather than drawing their own from scratch, so all rates measure agreement under LLM-assisted review. Figure 9 shows the interface, rendered on the synthetic test patient used during interface development.

The next two paragraphs reproduce the written guidance the annotators received. Its vocabulary difers from the paper’s in three places: the guide says “encounter” for hospitalization, “QI intervention” for candidate intervention, and “directly afected event” for intervention target (Type III node). References to “the first tab” point to the portions of the guide document reproduced under the conventions and intervention definitions below.

Annotator instructions. The remainder of this paragraph reproduces, verbatim, the written guidance given to the annotators, comprising the general annotation conventions and the step-by step annotation guide for the interface.

The DAG indicates how event timings causally depend on one another. There are two general tasks:

![](images/7507bee41ba10d4cef5b8cf27076c0dbc7fa1e287669f56b31a58b72cc93f3c9.jpg)  
Fig. 9. The annotation interface, rendered on the synthetic test patient used for interface development; no real patient data is shown. The left panel poses one verdict question at a time together with the LLM’s stated reasoning and its supporting timeline quotes. The center panel shows the extracted DAG on a department-by-time canvas. The right panel pins the intervention’s rule and inclusion/exclusion criteria above the patient timeline.

• Would this event’s timing be directly afected by the QI intervention? That is, would implementing the QI intervention directly change an event’s timing?

• Is this event’s timing directly dependent on a prior event’s timing? That is, would a change in event A’s timing directly cause event B’s timing to change?

## Annotation conventions:

• A direct connection between events A and B means there is not a diferent event/node in the graph that would fully explain away the dependency between the timings of events A and B.

• Note that an intervention can still afect an event’s timing if the event already happens very fast in the given patient, as the intervention may directly speed up this event’s timing for a diferent patient.

– Answer “no” only when the mechanism doesn’t apply or an exclusion applies

• New events cannot be added to the DAG, you can only:

– change which events count as directly afected by the intervention

## REFERENCES

– add or remove connections between the displayed events

Each encounter is shown as a graph of events extracted from the patient’s chart, things like orders, results, consults, transfers, discharges, and any other events the LLM extracts from the notes. An arrow between two events means the LLM believes the later event’s timing directly depends on the earlier event’s timing: a change in the earlier event’s timing would directly cause the later event’s timing to change. In addition to the graph you have access to the patient timeline with events and the notes that the LLM saw when constructing the graph.

Your job is to check the model’s constructed graph for the intervention shown in the first tab of this doc and in the annotation interface. The tool goes through questions one at a time and you answer yes or no. Your reasoning should be based on what the chart documents and the question is always “would this intervention, as written, directly afect this event for this patient?”

## Task 1: which events would the intervention afect directly?

The interface first asks you to review nodes in the DAG to determine if the LLM correctly marked events that are directly afected by the intervention.

• “Directly afected” means the intervention itself acts on that event’s timing. Events that would move only because an earlier event moved don’t count; the model handles those downstream changes.

• An event that was already fast still gets a yes if the mechanism applies to it (see the conventions on the first tab).

• When you answer no, add a few words on why: which exclusion applies, or why the mechanism doesn’t act on this event.

After the LLM-flagged events, the interface asks whether the intervention would directly afect any other events in the graph. If there are any directly-afected events that were missed, you can add them now.

## Task 2: which event timings does this event timing depend on?

Next you review the edges of the DAG to determine if event timing dependencies are correctly determined by the LLM.

• Keep an edge to the parent event if the event truly had to wait for it. Had the parent happened earlier, this event could have happened sooner too

• Remove the edge if there is no causal dependence between them

• Add a missing edge if an earlier event in the graph blocked this one and isn’t listed.

• New events cannot be added. You can only add or remove connections between the events shown in the DAG

• Note that the final (sink) node in the DAG is always “Discharge/Death”, which indicates the end of the hospital stay where the patient was either discharged or died. Which of the two should be evident from the patient timeline and LLM extractions.

## When something looks of

• If an event looks incorrect (mislabeled, incorrect time, not supported by the notes), describe the problem in the “no” note

• If the intervention doesn’t apply to this patient at all, answer no for each flagged event and explain once in the first note.

Intervention definitions shown to annotators. The two interventions used in the annotation study were presented to annotators with the following definitions, reproduced here verbatim.

## Disposition-critical imaging priority

Rule: If a diagnostic imaging study (TTE, TEE, CT with contrast, MRI, or ultrasound) is needed for deciding patient disposition, the study is prioritized so that the study will be completed the same calendar day if the study is ordered before 3pm and will be completed by noon the following day if ordered after 3pm. As such, order-to-completion for a flagged study should generally never exceed 24 hours, which encompasses the time of acknowledgment of the order, transportation of the patient, capturing the image, and the final report on the imaging study.

• Imaging studies that are one of the listed modalities

• Its result is needed for the disposition decision, e.g., the team is waiting on it for critical diagnostic work-ups, to decide whether or not a patient can be discharged, or where they can be discharged to.

## Exclusion Criteria:

• The patient is not medically stable for transport or scanning

– Hemodynamic instability, somnolence, active resuscitation

• Study blocked by anesthesia availability or study prerequisites (e.g. NPO status) and not scheduling queue

• Blocking clinical prerequisites that need to happen first, such as completion of the treatments, required follow-up intervals, or medical improvements.

• Documented patient factors that prevent the study from happening sooner (refusal, agitation, non-cooperation, patient in another procedure, documented allergies to contrasts, etc.)

## Early Discharge planning

Rule: Starting at the 1 day mark (24 hours), all patients will be evaluated for discharge planning. Discharge planning will then proceed within the following 24 hours and patients with complex discharge needs relating to housing, post-acute facility placement (SNF, acute rehab, LTACH), hospice, durable medical equipment, transportation, home health, IHSS, social-benefit applications (e.g., Medi-Cal), and/or substance use will meet with social work or case management within the 24 hour (i.e., 24–48 hours) time period. Qualifying complex patients are:

• Unhoused/live in a shelter

• Patients who came directly from a SNF/facilities/programs

• Patients who had home services

• Patients who will need to be set up with home services or a SNF or acute rehab or Long-term acute care facility (LTACH) or hospice referral based on the first 24 hours

If the patient agrees, the social work/case manager will contact appropriate discharge resources within 24 hours after meeting with the patient.   
Inclusion Criteria:

## REFERENCES

• A post-acute placement referral happens during the encounter.

• The placement need was identifiable within 24 hours of admission

## Exclusion Criteria:

• The placement need came up later in the encounter (e.g., a new stroke, deterioration, or new functional decline) so identification does not occur within the first 24 hours

• The patient has been discharged or dies in the hospital within 24 hours

• Meeting with the social work/case manager within the 24–48 hour window is not possible because of medical reasons or legal reasons, e.g., patient is not responding and no family members available, guardianship/conservatorship proceedings

• Patients who cannot be medically assessed for disposition placement (e.g., needs PT/OT/SLP)

• Meeting with the social work/case manager within the 24–48 hour window is not possible because it’s a holiday/weekend

• Discharge needs related to medications (e.g., discharge medications and medication reconciliation) are not accelerated through this intervention.

Human-versus-LLM metrics. Intervention-target specificity is the number of events the annotator judged non-targets that the LLM also left unflagged, divided by the number of events the annotator judged non-targets. Intervention-target recall is the number of annotator-endorsed targets the LLM flagged, divided by the number of annotator-endorsed targets. Both metrics are computed over every event extracted for the hospitalization, not only the events the LLM flagged. Edge precision is the number of LLM-proposed edges the annotator kept, divided by the number of LLM-proposed edges the annotator explicitly reviewed. Edge recall is the number of annotatorendorsed edges present in the LLM’s graph, divided by the number of annotator-endorsed edges. As a broader secondary denominator, we also compute an edge specificity that counts every unselected candidate parent as a negative judgment; this yields 97.9% pooled across interventions. Point estimates weight each intervention-hospitalization pair by the inverse of its annotation multiplicity, so a double-annotated pair contributes total weight one. Confidence intervals are computed by a cluster bootstrap over hospital stays, where the same hospitalization appearing under both interventions resamples jointly, and are conditional on the five annotators. Edge additions concentrate in individual annotators, so edge recall is sensitive to the panel; recomputing it with any one annotator removed moves it between roughly 79% and 92%.

Estimate-agreement setup. For the comparison of estimated time saved on annotator-corrected versus LLM-generated graphs (final row of Table 2), we re-ran egg-computation on each graph while holding the timing models fixed. The LLM’s Type III timing predictions are applied to each graph’s own intervention targets and capped at the observed times, since operational interventions remove delays rather than create them. The Type II step is deterministic, with each downstream event occurring as soon as its last parent does. Under this setup, any diference in the estimated time saved traces to the graphs alone rather than to timing-model noise.

Inter-annotator metrics. Inter-annotator agreement is computed on the eight double-annotated pairs. Edge units are the edges both annotators explicitly reviewed, coded keep versus delete; one-sided additions and diferences in reviewed scope are not measured. Node units are the intervention-target status of every event in the hospitalization. We report raw percent agreement, Krippendorf’s α, Gwet’s $\mathrm { A C _ { 1 } }$ , and within-class agreement (Gwet, 2008; Hayes and Krippendorf, 2007). For a rating class c with $n _ { c }$ concordant pairs and $d _ { c }$ discordant pairs involving a c rating, within-class agreement is $2 n _ { c } / ( 2 n _ { c } + d _ { c } )$ , the share of c ratings matched by the other annotator (Feinstein and Cicchetti, 1990; Cicchetti and Feinstein, 1990). We use the nominal versions of both chance-corrected coeficients because annotator pairings varied across tasks, so no stable raterone-versus-rater-two assignment exists for Cohen’s κ. Under the keep-skewed rating distribution, α and $\mathrm { A C _ { 1 } }$ diverge because their chance models difer, and the within-class rates are what localize the disagreement to deletions. All inter-annotator statistics are descriptive rather than inferential, as they pool eight pairs covering seven distinct hospitalizations and condition on this annotator panel.

## J. Type II timing evaluation details

Protocol. Each method predicts the delay, in hours, between a downstream event (Type II node) and the last of its parent events to complete, given the parent events with their timestamps, the event’s own description without its timestamp, and a patient summary. Under leave-onehospitalization-out cross-validation, every method trains on the delays of all other hospitalizations, pooled across the five diagnosis groups within the same intervention, and predicts the held-out hospitalization’s delays. The ICL model receives fifteen worked examples per query, drawn first from training delays whose parent and child event types match the query and topped up at random, and returns a numeric delay with a short rationale (the prompt template is given in Appendix D.2); the same GPT-5 configuration is used for every intervention. The zero baseline predicts no delay; pair-type median predicts the training median among delays with the same parent and child event types; ridge and random forest use event-type indicators and temporal features (hour of day, day of week, time since admission); embedding + ridge adds sentence embeddings of the same text the ICL model reads, so the strongest regression baseline has the same information access.

Cleaning descriptions to remove timing-revealing language. The event descriptions and patient summaries the models read are scrubbed of timing-revealing language (clock times, calendar dates, weekday names, and spelled-out durations) so the target delay cannot leak into the inputs; parent timestamps are retained because they are legitimate inputs. An automated audit scans every model-visible field for the same patterns and reports zero flagged rows across all 4,790 evaluated nodes.

Per-intervention accuracy. Table 4 breaks the pooled comparison of Table 3 out by intervention. ICL attains the lowest MAE for nine of the eleven interventions, with every paired interval clear of zero for eight. The exceptions concentrate where delays are near-immediate: discharge before noon (median delay 1.3 hours) is the one intervention where the zero baseline is numerically better, and for early discharge planning and the weekend discharge coordinator the closest baseline is statistically indistinguishable from ICL.

Accuracy by delay stratum. Table 5 reports MAE and median AE within strata of the observed delay. The zero baseline is nearly exact on the 40% of delays under one hour and remains strong to six hours, while the regression models hedge toward mid-range predictions that lose badly there. ICL concedes little on the short strata, is clearly best between six hours and one day, and trails

Table 4. Per-intervention Type II timing accuracy (hours). ∆MAE is the paired diference in MAE between a baseline and ICL (baseline minus ICL, so positive values favor ICL), shown only for each intervention’s closest baseline, i.e., the competitor with the smallest ∆MAE. The final column counts how many of the five baseline comparisons have 95% intervals lying entirely above zero, so 5/5 means ICL beats every baseline with an interval clear of zero.
<table><tr><td>Intervention</td><td>Nodes</td><td>Hosp.</td><td>ICL MAE</td><td>ICL Med. AE</td><td>Closest baseline</td><td>∆MAE [95% CI]</td><td>CIs &gt; 0</td></tr><tr><td>Discharge before noon</td><td>206</td><td>141</td><td>6.2</td><td>1.3</td><td>Zero gap</td><td>-0.7 [-2.0, 0.5]</td><td>2/5</td></tr><tr><td>Weekend diagnostic slots</td><td>453</td><td>131</td><td>11.8</td><td>2.5</td><td>Pair-type median</td><td>+1.3 [0.6, 2.0]</td><td>5/5</td></tr><tr><td>Weekend discharge coordinator</td><td>253</td><td>148</td><td>11.9</td><td>1.6</td><td>Zero gap</td><td>+1.1 [-0.8, 3.3]</td><td>3/5</td></tr><tr><td>Antimicrobial stewardship</td><td>238</td><td>109</td><td>18.6</td><td>2.6</td><td>Zero gap</td><td>+2.8 [1.2, 4.7]</td><td>5/5</td></tr><tr><td>Weekend PT/OT coverage</td><td>515</td><td>145</td><td>21.5</td><td>2.0</td><td>Pair-type median</td><td>+2.1 [1.2, 3.1]</td><td>5/5</td></tr><tr><td>Same-day echo completion</td><td>514</td><td>147</td><td>21.6</td><td>2.9</td><td>Pair-type median</td><td>+2.8 [1.6, 4.3]</td><td>5/5</td></tr><tr><td>Weekend SNF intake</td><td>237</td><td>118</td><td>22.5</td><td>2.0</td><td>Zero gap</td><td>+2.7 [0.7, 4.7]</td><td>5/5</td></tr><tr><td>Disposition-critical imaging priority</td><td>711</td><td>149</td><td>23.3</td><td>3.9</td><td>Pair-type median</td><td>+3.3 [2.0, 4.5]</td><td>5/5</td></tr><tr><td>Early discharge planning</td><td>382</td><td>150</td><td>24.4</td><td>3.3</td><td>Pair-type median</td><td>-0.2 [-2.2, 1.6]</td><td>3/5</td></tr><tr><td>Consult response escalation</td><td>582 699</td><td>150</td><td>34.5 40.0</td><td>4.0</td><td>Pair-type median</td><td>+5.4 [3.4, 7.3]</td><td>5/5</td></tr><tr><td>Urgent IR procedure block</td><td></td><td>120</td><td></td><td>14.9</td><td>Pair-type median</td><td>+6.6 [4.8, 8.6]</td><td>5/5</td></tr></table>

only the regression models’ small edge in the extreme tail. Within-stratum MAE is dominated by a small share of large errors, so the median gives the typical error; for example, ICL’s 4.2-hour MAE on the shortest stratum reflects a median error of 1.5 hours alongside the 12% of nodes missed by more than 10 hours.

Table 5. MAE (median AE in parentheses, both in hours) by observed-delay stratum, pooled across interventions.
<table><tr><td>Delay stratum</td><td>Nodes</td><td>Zero gap</td><td>Pair-type median</td><td>Ridge</td><td>Random forest</td><td>Embedding + ridge</td><td>ICL (GPT-5)</td></tr><tr><td>≤1h</td><td>1904</td><td>0.1 (0.0)</td><td>4.6 (1.8)</td><td>24.0 (20.6)</td><td>23.8 (13.9)</td><td>21.5 (13.6)</td><td>4.2 (1.5)</td></tr><tr><td>1-6h</td><td>914</td><td>2.9 (2.6)</td><td>4.7 (2.0)</td><td>22.3 (19.0)</td><td>23.6 (12.9)</td><td>21.1 (11.7)</td><td>4.3 (1.3)</td></tr><tr><td>6-24h</td><td>715</td><td>16.0 (17.7)</td><td>11.9 (11.3)</td><td>18.5 (13.5)</td><td>26.1 (15.7)</td><td>23.8 (17.9)</td><td>10.0 (5.9)</td></tr><tr><td>&gt;24h</td><td>1257</td><td>96.5 (62.6)</td><td>86.7 (50.8)</td><td>67.6 (34.0)</td><td>71.3 (37.3)</td><td>67.3 (35.9)</td><td>76.8 (44.5)</td></tr></table>

Structural validity sensitivity. About 10% of the downstream node timings (509 of 4,790) come from DAGs that fail an automated structural check (temporal inconsistencies, cycles, or events with no path to discharge); these timings remain in the primary analysis. Table 6 recomputes the pooled comparison excluding them: every MAE moves by less than half an hour and every interval conclusion is unchanged.

## K. Additional Results

Inter-rater evaluation results. Table 7 reports agreement between annotators on the double annotated intervention-hospitalization pairs. Seven of the eight pairs produced identical sets of intervention targets, and raw agreement on the keep-versus-delete decision for commonly reviewed edges was similarly high. Because roughly 89% of the individual edge ratings are in the “keep” group, the two chance-corrected coeficients diverge under this skew (Krippendorf’s $\alpha = 0 . 1 1$ , Gwet’s $\mathrm { A C _ { 1 } } = 0 . 7 8 )$ (Feinstein and Cicchetti, 1990; Cicchetti and Feinstein, 1990), so we follow (Cicchetti and Feinstein, 1990) and report agreement within each rating class, i.e., how often a rating of a given class was matched by the other annotator. Annotators matched 90% of keep ratings but only 20% of deletions, and seven of the eight discordant edges concern the same judgment of whether an event that precedes discharge actually gates the discharge decision or merely happens before it. Disagreements over intervention targets likewise reflect difering interpretations of an eligibility criterion; annotators applied diferent thresholds for when a patient’s discharge needs are identifiable within the first day of the stay, which determines whether early discharge planning applies to the patient at all. Both are due to diferences in interpretation of the intervention specification rather than disagreements about the DAG generated by the LLM.

Table 6. Structural-validity sensitivity for the pooled Type II timing comparison.
<table><tr><td>Population</td><td>Method</td><td>MAE</td><td>Med. AE</td><td>∆MAE vs. ICL [95% CI]</td></tr><tr><td rowspan="6">All DAGs (4790 nodes, 736 hosp.)</td><td>Zero gap</td><td>28.30</td><td>2.74</td><td>+4.18 [3.57, 4.77]</td></tr><tr><td>Pair-type median</td><td>27.23</td><td>4.63</td><td>+3.11 [2.57, 3.67]</td></tr><tr><td>Ridge</td><td>34.30</td><td>20.92</td><td>+10.18 [9.37, 10.99]</td></tr><tr><td>Random forest</td><td>36.58</td><td>17.73</td><td>+12.46 [11.18, 13.90]</td></tr><tr><td>Embedding + ridge</td><td>33.79</td><td>18.90</td><td>+9.67 [8.57, 10.70]</td></tr><tr><td>ICL (GPT-5)</td><td>24.12</td><td>3.00</td><td></td></tr><tr><td rowspan="6">Structurally valid only (4281 nodes, 701 hosp.)</td><td>Zero gap</td><td>28.57</td><td>2.85</td><td>+4.23 [3.60, 4.89]</td></tr><tr><td>Pair-type median</td><td>27.27</td><td>4.63</td><td>+2.94 [2.38, 3.50]</td></tr><tr><td>Ridge</td><td>34.22</td><td>20.55</td><td>+9.89 [9.04, 10.75]</td></tr><tr><td>Random forest</td><td>36.59</td><td>17.73</td><td>+12.25 [10.76, 13.90]</td></tr><tr><td>Embedding + ridge</td><td>33.87</td><td>18.51</td><td>+9.53 [8.43, 10.70]</td></tr><tr><td>ICL (GPT-5)</td><td>24.34</td><td>3.00</td><td></td></tr></table>

Table 7. Inter-annotator agreement on the double-annotated hospitalizations. Edge units are the edges both annotators explicitly reviewed; node units are the intervention-target status of every event in the hospitalization.
<table><tr><td></td><td>n</td><td>Raw agreement</td><td>Krippendorff&#x27;s α</td><td>Gwet&#x27;s AC1</td><td>Within-class agreement</td></tr><tr><td>Edges (keep vs. delete)</td><td>45</td><td>82.2%</td><td>0.11</td><td>0.78</td><td>keep 90.0%, delete 20.0%</td></tr><tr><td>Nodes (intervention target vs. not)</td><td>116</td><td>97.4%</td><td>0.89</td><td>0.97</td><td>non-target 98.5%, target 90.3%</td></tr></table>

Table 8. Estimated average time saved per eligible hospitalization (hours), by intervention, from eggcomputation with the ICL timing model, for only the eligible interventions. Intervals are 95% bootstrap intervals over hospitalizations. Reference variants and per-diagnosis breakdowns are given in Appendix M.
<table><tr><td>Intervention</td><td>n</td><td>Mean savings (h)</td><td>Median (h)</td></tr><tr><td>Consult response escalation</td><td>150</td><td>26.6 [17.0, 37.8]</td><td>0.9</td></tr><tr><td>Disposition-critical imaging priority</td><td>150</td><td>19.6 [12.8, 27.8]</td><td>0.0</td></tr><tr><td>Early discharge planning</td><td>150</td><td>15.7 [7.2, 25.6]</td><td>0.0</td></tr><tr><td>Discharge before noon</td><td>150</td><td>14.6 [9.2, 20.8]</td><td>1.0</td></tr><tr><td>Weekend discharge coordinator</td><td>150</td><td>19.6 [10.9, 30.9]</td><td>0.6</td></tr><tr><td>Urgent IR procedure block</td><td>120</td><td>49.9 [27.0, 78.7]</td><td>0.0</td></tr><tr><td>Same-day echo completion</td><td>150</td><td>22.9 [11.7, 37.0]</td><td>0.0</td></tr><tr><td>Antimicrobial stewardship</td><td>110</td><td>10.6 [4.2, 17.9]</td><td>0.0</td></tr><tr><td>Weekend SNF intake</td><td>123</td><td>21.3 [12.7, 32.0]</td><td>1.3</td></tr><tr><td>Weekend PT/OT coverage</td><td>145</td><td>11.2 [5.4, 18.9]</td><td>0.0</td></tr><tr><td>Weekend diagnostic slots</td><td>133</td><td>13.8 [6.7, 21.7]</td><td>0.0</td></tr></table>

## REFERENCES

## L. Clinician review of the estimated rankings

Reviewer panel. Four clinicians from the hospital’s QI team reviewed the estimates, two of them authors of this work (AH, LZ) and two of them not. Reviewers worked independently and did not see one another’s responses.

Interface and review protocol. Reviewers completed three tasks asynchronously through a self contained web page. The page put the tasks in a fixed order and withheld our estimates until the first was finished.

Task 1: ranking the shortlist. The three interventions our estimates ranked highest by expected impact were shown in an order that carried no information about our estimated ranking. Each was shown with a one-sentence description of the operational rule and the current delay it targets. The remaining eight interventions were also included for reference. Reviewers were asked to rank the three, with the instruction “Before seeing the order our estimates put them in, rank the three yourself: 1st = most total inpatient time saved.” Only after all three were ranked did the page reveal the full ranking of eleven candidates, reporting for each the eligibility rate, the estimated time saved per eligible hospitalization, and the expected impact, together with a slopegraph pairing the eligibility and expected-impact rankings and a scatter of the two quantities whose product is the expected impact. Reviewers were then asked which of the estimated rankings surprised them and which matched their experience. Table 9 gives the orderings from the reviewers.

Task 2: reviewing example hospitalizations. Each reviewer was assigned three interventions: the top two interventions, consult response escalation and disposition-critical imaging priority, and one random intervention. For each assigned intervention, the page showed the eligibility rate, the estimated time saved per eligible hospitalization, the expected impact, the share of eligible hospitalizations with no estimated benefit, and the median saving among those that benefit (see Figure 7). It also showed the eligibility rate within each diagnosis group and a histogram of estimated savings across eligible hospitalizations. Below these summaries the page gave an LLM-written account of the mechanism, separating the hospitalizations in which the intervention shortens the stay from those in which discharge remains gated by some other event, together with the number of hospitalizations of each kind. Four example hospitalizations followed, each rendered as the observed and counterfactual timings of every event in its DAG, with the intervention targets marked and each event’s incoming edges shown on hover. Reviewers were asked two questions of each assigned intervention:

• “Are the generated analyses clinically plausible and helpful in interpreting the likely impact of this QI intervention? Please provide detailed feedback since this will help us improve the interface and understand how well the current pipeline is working.”

• “Based on the generated analyses, would you consider modifying the intervention further to improve its eficacy? If so, please explain how you would change it and what in the generated analyses has prompted you to consider this modification.”

For each intervention we drew two hospitalizations at random from those whose stay the estimate shortens and two at random from those whose estimated saving is within one hour of zero. The examples are therefore balanced between the two outcomes by construction and do not reflect the share of eligible hospitalizations that benefit, which the accompanying histogram reports separately.

Task 3: general comment. Reviewers were asked two closing questions: “Do you have general feedback on the QI intervention analysis pipeline? How can the analysis be improved further?” and “Are there other data sources that you think would be important to add to our pipeline (we currently analyze only orders, notes, results, and their timestamps)?”

Table 9. Orderings of the three highest-ranked interventions returned in Task 1, each produced before the reviewer saw any estimate, against the ordering by estimated expected impact.
<table><tr><td></td><td>First</td><td>Second</td><td>Third</td></tr><tr><td>R1</td><td>Consult response escalation</td><td>Disposition-critical imaging</td><td>Early discharge planning</td></tr><tr><td>R2</td><td>Consult response escalation</td><td>Disposition-critical imaging</td><td>Early discharge planning</td></tr><tr><td>R3</td><td>Early discharge planning</td><td>Disposition-critical imaging</td><td>Consult response escalation</td></tr><tr><td>R4</td><td>Early discharge planning</td><td>Consult response escalation</td><td>Disposition-critical imaging</td></tr><tr><td>Estimated</td><td>Consult response escalation</td><td>Disposition-critical imaging</td><td>Early discharge planning</td></tr><tr><td>Eligibility</td><td>Disposition-critical imaging</td><td>Consult response escalation</td><td>Early discharge planning</td></tr></table>

## M. Time-saved estimation details

Table 10. Prevalence-based versus causal prioritization of the eleven candidate interventions. Eligibility rate is the share of hospitalizations that meet an intervention’s inclusion criteria. Expected time saved is the causal efect estimate from egg-computation. Interventions are ordered by expected impact. Brackets give 95% percentile bootstrap intervals, resampling the screened cohort within each diagnosis group and the eligible hospitalizations within each intervention.
<table><tr><td rowspan="3">Intervention</td><td colspan="3">Eligibility rate</td><td colspan="3">Expected impact</td></tr><tr><td>% [95% CI]</td><td></td><td>rank</td><td>h/screened [95% CI]</td><td></td><td>rank</td></tr><tr><td></td><td>71 [69, 73]</td><td>2</td><td></td><td></td><td></td></tr><tr><td>Consult response escalation</td><td></td><td></td><td>1</td><td></td><td>18.9 [11.4, 26.9] 15.6 [10.2, 22.1]</td><td>1 2</td></tr><tr><td>Disposition-critical imaging priority</td><td></td><td>80 [78, 81] 69 [67, 70]</td><td>3</td><td></td><td>10.8 [5.0, 17.5]</td><td>3</td></tr><tr><td>Early discharge planning Discharge before noon</td><td></td><td>42 [40, 44]</td><td>4</td><td></td><td>6.2 [3.9, 8.6]</td><td>4</td></tr><tr><td>Weekend discharge coordinator</td><td></td><td>26 [24, 28]</td><td>5</td><td></td><td>5.0 [2.7, 8.0]</td><td>5</td></tr><tr><td>Urgent IR procedure block</td><td></td><td>10 [8, 11]</td><td>8</td><td></td><td>4.8 [2.5, 7.4]</td><td>6</td></tr><tr><td>Same-day echo completion</td><td></td><td>17 [15, 18]</td><td>7</td><td></td><td>3.9 [2.0, 6.5]</td><td>7</td></tr><tr><td>Antimicrobial stewardship</td><td></td><td>21 [20, 22]</td><td>6</td><td></td><td>2.2 [0.9, 3.7]</td><td>8</td></tr><tr><td>Weekend SNF intake</td><td></td><td>8 [7, 9]</td><td>10</td><td></td><td>1.7 [1.0, 2.6]</td><td>9</td></tr><tr><td>Weekend PT/OT coverage</td><td></td><td>9 [8, 10]</td><td>9</td><td></td><td>1.0 [0.5, 1.7]</td><td>10</td></tr><tr><td>Weekend diagnostic slots</td><td></td><td>7 [6, 8]</td><td>11</td><td></td><td>0.9 [0.4, 1.6]</td><td>11</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## Results.

Rank stability. Table 11 quantifies which ranking statements survive resampling. Each of 2,000 bootstrap resamples redraws the screened cohort within each diagnosis group (jointly across interventions, preserving the correlation of eligibility within a hospitalization) and the eligible hospitalizations within each intervention, then recomputes both rankings. The top of the expected-

## REFERENCES

impact ranking is stable: consult response escalation is first in 72% of resamples and dispositioncritical imaging priority in 24%, and the same three interventions occupy the top three in 89% of resamples, while the bottom three are drawn almost entirely from the weekend interventions.

Table 11. Rank stability across 2,000 bootstrap resamples: the probability each intervention lands in the top three of each ranking, is first by expected impact, or is in the bottom three by expected impact.
<table><tr><td rowspan="2">Intervention</td><td colspan="2">Top 3 probability</td><td rowspan="2">P(rank 1, expected impact)</td><td rowspan="2">P(bottom 3, expected impact)</td></tr><tr><td>Prevalence</td><td>Expected impact</td></tr><tr><td>Consult response escalation</td><td>1.00</td><td>1.00</td><td>0.72</td><td>0.00</td></tr><tr><td>Disposition-critical imaging priority</td><td>1.00</td><td>1.00</td><td>0.24</td><td>0.00</td></tr><tr><td>Early discharge planning</td><td>1.00</td><td>0.89</td><td>0.03</td><td>0.00</td></tr><tr><td>Discharge before noon</td><td>0.00</td><td>0.07</td><td>0.00</td><td>0.00</td></tr><tr><td>Weekend discharge coordinator</td><td>0.00</td><td>0.03</td><td>0.00</td><td>0.00</td></tr><tr><td>Urgent IR procedure block</td><td>0.00</td><td>0.02</td><td>0.00</td><td>0.00</td></tr><tr><td>Same-day echo completion</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.01</td></tr><tr><td>Antimicrobial stewardship</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.28</td></tr><tr><td>Weekend SNF intake</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.72</td></tr><tr><td>Weekend PT/OT coverage</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.99</td></tr><tr><td>Weekend diagnostic slots</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td></tr></table>

Table 12. Primary estimate of average time saved (hours) by intervention and diagnosis group; cell entries are mean (number of eligible hospitalizations in the analysis subsample).
<table><tr><td>Intervention</td><td>A41</td><td>I63</td><td>L03</td><td></td><td>S06</td><td>T40</td><td>Pooled</td></tr><tr><td>Consult response escalation</td><td>29.5 (30)</td><td>52.1 (30)</td><td>12.4 (30)</td><td></td><td>21.1 (30)</td><td>17.9 (30)</td><td>26.6 (150)</td></tr><tr><td>Disposition-critical imaging priority</td><td>28.0 (30)</td><td>6.8 (30)</td><td>24.5 (30)</td><td></td><td>26.7 (30)</td><td>11.9 (30)</td><td>19.6 (150)</td></tr><tr><td>Early discharge planning</td><td>19.6 (30)</td><td>37.7 (30)</td><td>9.1 (30)</td><td></td><td>3.0 (30)</td><td>9.4 (30)</td><td>15.7 (150)</td></tr><tr><td>Discharge before noon</td><td>14.8 (30)</td><td>24.3 (30)</td><td>7.5 (30)</td><td></td><td>5.8 (30)</td><td>20.5 (30)</td><td>14.6 (150)</td></tr><tr><td>Weekend discharge coordinator</td><td>28.1 (30)</td><td>12.0 (30)</td><td>9.4 (30)</td><td></td><td>41.1 (30)</td><td>7.5 (30)</td><td>19.6 (150)</td></tr><tr><td>Urgent IR procedure block</td><td>61.9 (30)</td><td>82.0 (30)</td><td>80.0 (7)</td><td></td><td>22.9 (30)</td><td>18.4 (23)</td><td>49.9 (120)</td></tr><tr><td>Same-day echo completion</td><td>29.7 (30)</td><td>22.0 (30)</td><td>11.5 (30)</td><td></td><td>46.9 (30)</td><td>4.3 (30)</td><td>22.9 (150)</td></tr><tr><td>Antimicrobial stewardship</td><td>15.1 (30)</td><td>5.7 (12)</td><td>4.5 (30)</td><td></td><td>9.7 (8)</td><td>14.4 (30)</td><td>10.6 (110)</td></tr><tr><td>Weekend SNF intake</td><td>15.3 (30)</td><td>30.2 (30)</td><td>5.6 (11)</td><td></td><td>17.8 (30)</td><td>29.8 (22)</td><td>21.3 (123)</td></tr><tr><td>Weekend PT/OT coverage</td><td>10.0 (30)</td><td>3.3 (30)</td><td>7.0 (25)</td><td></td><td>21.3 (30)</td><td>13.8 (30)</td><td>11.2 (145)</td></tr><tr><td>Weekend diagnostic slots</td><td>28.2 (30)</td><td>16.8 (30)</td><td>14.8 (30)</td><td></td><td>-1.1 (16)</td><td>2.1 (27)</td><td>13.8 (133)</td></tr></table>

## Acknowledgments

The PROSPECT lab (PV, JO, LZ, JF) thanks Zuckerberg Priscilla Chan quality improvement fund via the San Francisco General Foundation for funding this project. We thank Robert Gallo and Nina Singh for helping review the final analyses and providing clinical feedback.

## Ethics Statement

This study was conducted as a quality improvement initiative at Zuckerberg San Francisco General Hospital and Trauma Center. This study was approved by the University of California, San Francisco Institutional Review Board (protocol 22-36613) as well as Zuckerberg San Francisco General Hospital. Individual informed consent was waived, given the retrospective nature of the analysis and the use of deidentified electronic health record data. All data were stored and

## REFERENCES

analyzed within institutional systems in compliance with HIPAA regulations.