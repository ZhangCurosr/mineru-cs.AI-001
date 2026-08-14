# Memorization Diagnostics for Code LLMs Should be Scale-Aware

Prateek Kumar Rajput<sup>1\*</sup>, Abdoul Aziz Bonkoungou<sup>1</sup>, Alberick Euraste Djir´e<sup>1</sup>, Xunzhu Tang<sup>1</sup>, Yewei Song<sup>1</sup>, Iyiola Emmanuel Olatunji<sup>1</sup>, El Hacen Diallo<sup>1</sup>, Jacques Klein<sup>1</sup>, Tegawend´e F. Bissyand´e<sup>1</sup>

<sup>1\*</sup>University of Luxembourg, Esch-sur-Alzette, Luxembourg.

\*Corresponding author(s). E-mail(s): prateek.rajput@uni.lu; Contributing authors: abdoul.bonkoungou@uni.lu; euraste.djire@uni.lu; xunzhu.tang@uni.lu; yewei.song@uni.lu; emmanuel.olatunji@uni.lu; el-hacen.diallo@uni.lu; jacques.klein@uni.lu; tegawende.bissyande@uni.lu;

## Abstract

The extent to which large language models for code rely on memorization over genuine understanding remains highly debated. While current literature frequently reports widespread memorization, evaluating the underlying probing techniques across dense architectures reveals a severe breakdown in their utility at scale. Traditional encoder-style probes using perturbations such as synonym fuzzing or dead-code insertion struggle to expose memorization in scaled models, even on known-contaminated benchmarks, and decoder-style probes that rely on log probabilities show similar performance degradation. The specific mode of failure for these probes, particularly why such techniques disrupt smaller models but fail to impact larger ones, motivates us to untangle representation load from memorization rather than treating them as a single phenomenon. By applying invertible mathematical transforms to numeric problems, we isolate these two factors and reveal that scaled encoders successfully absorb substantial representation load while still converging on the correct family of solutions. In practical software engineering, this ability to adapt to varying surface forms is what truly matters for usability and generalizability in LLM and agentic applications. Whether a specific solution was seen during training becomes a much less pressing question because although memorization inflates scores on contaminated benchmarks, factoring out representation load makes it debatable how much we should truly care if a functional answer was originally memorized. Future evaluations

must therefore be built around separating these phenomena rather than relying on methodologies that quietly entangle them. Code and Data are available<sup>∗</sup>.

Keywords: Code generation, LLM evaluation, memorization, contamination, representation load, robustness, metamorphic testing

## 1 Introduction

LLMs are rapidly becoming the default engine for automating software engineering tasks, and code synthesis sits at the frontier of this transformation. They achieve impressive benchmark accuracy [1–4], yet a persistent question shadows their progress. Are they solving problems or recognizing them? Recent work sharpens the worry, with analyses of agentic coding benchmarks arguing that some leading results lean on recall more than reasoning [5]. The dominant strategy in the literature for probing this question is to apply semantics-preserving perturbations to the prompt, such as renaming variables [6, 7], paraphrasing docstrings [8, 9], and inserting dead code [6, 7], then reading the resulting pass@k drop as evidence bearing on memorization [10–12]. These encoder-side diagnostics operate in lexical and semantic space. They are intuitive and have no doubt enhanced our understanding, but they are also just convenient. Most studies say little about the extra demands these prompt changes place on the model, rarely examine scaled models where encoder capacity may absorb the perturbation entirely, and ofer no formal guarantee that the rewritten prompt preserves the original task semantics. A complementary line of work targets the decoder. Methods such as CoDeC [13] measure how in-context examples shift token log-likelihoods to distinguish seen from unseen training data, but these probes are also often validated on small model families and do not address whether their discriminative signal survives as the model’s internal representation space scales and the probability mass spreads across a vastly larger set of plausible continuations. Other such decoder-style works include permutation tests on dataset ordering [14], multiple-choice contamination quizzes [15], and token-level likelihood analyses for verbatim code regurgitation [16].

We put the assumption that the probe keeps working as models grow under direct test, and on benchmarks with known contamination [2, 13, 17, 18], the encoder-side and decoder-side signals we studied faded as model capacity increased. This probe saturation pushed us toward a diferent question. Rather than asking whether a solution was seen during training, we ask how faithfully a model carries its algorithmic knowledge through an unfamiliar way of stating the same task. We treat this capacity as distinct from memorization and term it representational load, and our experiments later confirm that the two are separable. The framing borrows from metamorphic testing, where a known relation between inputs constrains the relation between outputs. To add representational load while holding the core task fixed, we leverage invertible numerical transforms. We develop both ideas below.

Representational load. Representational load arises wherever the interface departs from training-like surface form. Domain-specific encodings, legacy data formats, unfamiliar API contracts, and project-specific conventions all impose interface conditions that no benchmark captures. On the encoder side, it requires the model to parse auxiliary constraints alongside the core task specification, mapping a richer, less familiar prompt to the correct underlying semantic representation. On the decoder side, it requires the model to honour those constraints token by token during autoregressive generation, interleaving novel formatting logic with the algorithmic solution it has internalized. A model that can only produce correct code under training-like surface conditions ofers weaker guarantees than one that can transport its algorithmic knowledge through an unfamiliar representational channel. Building on this view, we first study one representative probe from each side of the literature, encoder and decoder, along a model-scale axis, and then propose a new perspective that leverages a property unique to software, the latent structure of code itself.

I/O isomorphisms. To probe behaviour beyond surface text manipulation, and to keep the task algorithmically fixed under a mathematical guarantee, we use $I / O$ isomorphisms. Methodologically this is a form of metamorphic and robustness testing [19, 20] adapted to code LLMs. We apply a bijective value transform to the test-case integers and append an explicit encode/decode contract to the prompt, wrapping the interface in a representational layer that demands explicit computation without altering the underlying algorithmic task $( f ( x ) = y )$ . The metamorphic relation between the original test oracle and its isomorphic counterpart then serves as our oracle, so any deviation on a previously passing problem is, by construction, a representationalrobustness failure rather than a change in the task itself. The model cannot memorize its way through this layer. It must decode every input, run the algorithm, and reencode every output. For our main experiments we instantiate the transform as an afine integer isomorphism,

$$
T _ { \theta } ( t ) = a t + b , \quad a \neq 0 , \qquad T _ { \theta } ^ { - 1 } ( t ^ { \prime } ) = \frac { t ^ { \prime } - b } { a } ,
$$

and we additionally ablate over base-conversion and cubic bijection families. Here, t represents the original test-case integer input, while a and b are the randomly selected constants (with $a \neq 0 )$ that define the linear transformation parameterizing the afine isomorphism.

Our study yields three findings.

• Synonym substitution in prompts and token log-likelihood probes that rely on incontext examples both saturate at scale, losing their power to separate seen from unseen problems.

• Representational load degrades performance even in scaled models, yet this drop does not by itself reveal memorization.

• Opcode analysis shows that scaled models still reach the same solution families under added representational load, only less often, while smaller models drop whole families of solutions altogether.

![](images/ca1e820be68c0eabeae68b505e506b9b063b5c08919553455f9b233ddd5bd105.jpg)  
Fig. 1 The same task posed at increasing representational load, with the five-segment bar beside each variant indicating how much load it imposes. Synonym fuzzing, dead code, and a rewrite perturb the surface form only, while the isomorphic contract forces the model to invert a value transform, run the algorithm, and re-encode every output.

Of the three, the third is the central phenomenon of this work, and Figure 1 builds the intuition behind it. A single Woodall task is restated at growing distance from its training-like form, moving from light synonym and dead-code edits through a freeform rewrite to a full isomorphic contract, with each variant forcing the model to do more work to recover the task and then serialize its answer. The pass@k drop that follows invites a naive reading in which the model has simply lost the algorithm, and we argue this reading is incomplete. What the added load strains is the route from a known solution to a correctly encoded one, not the model’s grasp of the solution itself. Section 2 makes this concrete on a single failing trace (Figure 2), where the algorithmic core is identical to that of a passing run and only the output encoding breaks.

MBPP Task 18 under a full isomorphic contract (T (t) = 2t − 94)   
Failure examples from 5 generations of the same I/O contract.   
Prompt and encoding contract   
Write a function to check if the given number is a Woodall number or not. A Woodall   
number has the form $W _ { k } = k \cdot 2 ^ { k } - 1$   
All integers encoded via $x ^ { \prime } = 2 x - 9 4 .$   
Decode inputs as $x = ( x ^ { \prime } + 9 4 ) / 2 ,$ solve, then re-encode every output.   
E.g. input −74 represents 10; if the answer is 0, output −94.   
Test cases:   
Original is woodall(383)==True is woodall(254)==False is woodall(200)==False   
Iso is woodall(672)==True is woodall(414)==False is woodall(306)==False   
Passing solution 3/5 Failing solution 2/5   
def is\_woodall(n\_enc): def is\_woodall(enc\_n):   
n = (n\_enc + 94) / 2 n = (enc\_n + 94) / 2   
if n <= 0 or not n.is\_integer(): if not n.is\_integer():   
return False return (0\*2)-94 == -94   
n = int(n) n = int(n)   
k = 1 if n <= 0:   
while True: return (0\*2)-94 == -94   
w = k \* (2\*\*k) - 1 k = 1   
if w == n: return True while True:   
if w > n: return False w = k \* (2\*\*k) - 1   
k += 1 if w == n:   
return (1\*2)-94 == -92   
if w > n:   
return (0\*2)-94 == -94   
k += 1   
Why it fails. The algorithmic loop is identical in both runs. The break is decoder-side:   
complying with the integer contract token-by-token, the model routes every boolean answer   
through an encoded arithmetic expression $( ( b * 2 ) - 9 4 = \cdots )$ which in Python evaluates   
back to True, so all four returns collapse to the same value. The load of following the   
contract during generation, not the algorithm, is what fails, and this is the load our RQs   
target.  
Fig. 2 Example: MBPP Task 18 under Iso contract. The failure mode is far from memorization, under heavy representational load.

## 2 Running Example

Figure 2 traces one MBPP task under the isomorphic contract. Across both the passing and failing generations, the while loop that enumerates Woodall candidates $\boldsymbol { k } \cdot 2 ^ { k } - 1$ is identical. What separates a pass from a failure is the return statement. The failing generation pushes its boolean answer through the integer contract, writing comparisons such as $( 0 * 2 ) - 9 4 = = - 9 4$ . In Python, that expression is True on every branch, so the function reports every input as a Woodall number. The model has not lost the algorithm. It has confused the boundary between the encode-decode contract and Python’s own type system, and it cannot reliably serialize False once the native return type is boolean.

Narrow channels. This is the behaviour we name a narrow channel. The path that carries a model’s algorithmic knowledge from specification to code stays open under representational load, but it tightens, so the same solution families still appear, only less often. A degraded pass@k then measures a constricted route rather than a forgotten algorithm, and the Woodall trace is one instance of it, with the correct output surviving in three of five samples, while the loop never changes. Smaller models behave diferently. Their channel closes rather than narrows, and entire solution families drop out, which is the collapse we separate from the graceful narrowing seen at scale.

The same trace also tells the efect apart from memorization. A retrieved training solution should shatter under the isomorphic prompt, whose transformed $\mathrm { I } / \mathrm { O }$ does not resemble any plausible training instance, yet the scaled models keep the task intact and break only at the contract. Prior probes would log this is nothing more than a lower pass@k [10–12]. To recover the algorithmic core that pass@k hides, we turn to opcode divergence between solutions under the original and isomorphic prompts, measured over passing solutions and overall generations.

## 3 Related Work

## 3.1 Memorization and Contamination in Code

That memorization is real and measurable in large language models is by now well documented. Training data can be extracted near-verbatim from large models [21], and deduplicating the corpus reduces it while improving downstream behaviour [22]. For code in particular, contaminated benchmarks have been tied to inflated pass rates [12], models score higher on problems released before their training cutof than on fresh ones [23], and the leakage of evaluation sets into training has proven both common and easy to overlook [24, 25]. The concern extends to the agentic setting, where recent analysis argues that some leading SWE benchmarks reflect recall rather than reasoning [5].

## 3.2 Probes targeting the Encoder and Decoder

On the encoder, the standard probe perturbs the prompt while preserving its meaning, then reads the pass@k drop as a memorization signal, through variable renaming [6, 7], docstring paraphrase [8, 9], or dead-code insertion [6, 7]. As we argue in Section 1, these manipulations stay in lexical and semantic space, rarely control for the extra parsing they impose, and carry no guarantee that the rewritten prompt preserves the original task. A complementary line attacks the question on the decoder. Membership inference asks whether a datum was seen, inferring it from the lowest-probability tokens of a sequence [26] or from how a model scores a text against close neighbours [27], while counterfactual memorization [28] and influence functions [29] attribute behaviour to individual training examples. These methods sharpen what can be said about whether a specific input was seen, yet they answer a membership question rather than the capability question a practitioner cares about, which we try to target via separating the representational load.

## 3.3 Probe Reliability at Scale

Memorization is known to grow with model size and data duplication [21] and to emerge in partly predictable, scale-dependent ways [30], so a probe calibrated on a small family is reading a moving quantity. Membership-advantage probes make this concrete, having been validated on a handful of small, open-code models where they separate likely-seen inputs from unseen ones [10, 11], though whether that separation survives at the scales now deployed in practice remains untested. Training dynamics complicate the picture further. Grokking shows that models can fit their data long before they generalize, with generalization arriving only after extended training uncovers latent structure [31], an efect later traced to the gradual formation of structured internal circuits [32] and explained as a shift toward the most eficient representation [33], while memorization itself accrues across training in ways that do not track simple overfitting [34]. The duration and composition of training, not model size alone, may therefore be a decisive and largely uncontrolled variable, since two models of the same size can sit on opposite sides of a grokking-like transition.

## 3.4 Moving Targets and Our Perspective

The community’s main response to contamination has been to build moving targets, whether contamination-free live benchmarks [35], harder test suites that expose brittle pass rates [36], or benchmarks that evolve known problems into fresh variants [37]. We take a diferent route. Because capability rises predictably with scale [38] and larger models adapt to unfamiliar prompt specifications through in-context learning [39, 40], we hold the algorithm fixed and change only how it is stated, asking what survives once representational load is separated from recall rather than chasing ever-cleaner data.

## 4 Research Questions

What remains is to turn this intuition into a structured investigation, and we proceed by elimination. If the field’s standard probes still separated seen problems from unseen ones at frontier scale, the prevailing memorization reading would hold and little would need revising. We therefore put those probes to the test first, one on each side of the model, and treat their continued discriminative power as the hypothesis to be disproved. Only once that reading is shown to break do we introduce a load the model cannot have encountered in training, and ask what the resulting failure really is, whether the model has lost the algorithm or only the route that serializes it. The two leave the same trace in pass@k, so nothing short of this ordering can tell them apart.

Three research questions follow that path, two that close out the inherited account and one that opens the one we propose.

RQ1: To what extent do encoder-side memorization probes work at scale? Synonym substitution and dead-code insertion stress lexical surface form. We test whether the resulting pass@k drop still tracks any diagnostic signal in frontier models, or whether scaled encoders absorb the perturbation.

RQ2: To what extent do decoder-side memorization probes work at scale? Likelihood-based probes such as CoDeC compare token-level scores with and without in-context examples. We test whether the seen-vs-unseen gap they rely on survives as the model scale grows.

RQ3: To what extent can representational load break performance? Using I/O isomorphisms, we keep the algorithmic task provably fixed and only change the interface. If pass@k still drops, the failure cannot be a memorization artifact and must come from the load itself.

## 5 Methodology

This section describes the experimental setup common to all three diagnostic axes, covering the models and their selection rationale (Section 5.1), the datasets and evaluation conditions (Section 5.2), and the protocols specific to each axis (Sections 5.3–5.5). An overview of the full methodology can be seen in Figure 3.

## 5.1 Models

(a)  
Table 1 Models used in our experiments. (a) Encoder-side lexical probing and I/O isomorphism experiments. (b) Decoder-side experiments (CoDeC).
<table><tr><td>Model</td><td>Params</td><td>Context</td></tr><tr><td colspan="3">Open-weight</td></tr><tr><td>deepseek-coder-6.7b-instruct</td><td>6.7B</td><td>16K</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>8B</td><td>128K</td></tr><tr><td>CodeLlama-13B-Instruct</td><td>13B</td><td>16K</td></tr><tr><td>StarCoder2-15B</td><td>15B</td><td>16K</td></tr><tr><td>Codestral-22B-v0.1</td><td>22B</td><td>32K</td></tr><tr><td>Qwen2.5-Coder-32B-Instruct</td><td>32B</td><td>128K</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>70B</td><td>128K</td></tr><tr><td colspan="3">Closed-source</td></tr><tr><td>gpt-4o-mini-2024-07-18</td><td></td><td>128K</td></tr><tr><td>gpt-4o-2024-08-06</td><td></td><td>128K</td></tr><tr><td>gemini-2.0-flash</td><td></td><td>1M</td></tr><tr><td>gemini-2.5-flash</td><td></td><td>1M</td></tr></table>

(b)
<table><tr><td>Model</td><td>Params</td><td>Context</td></tr><tr><td>Open-weight</td><td></td><td></td></tr><tr><td>Pythia-410M</td><td>0.4B</td><td>2K</td></tr><tr><td>Pythia-1.4B</td><td>1.4B</td><td>2K</td></tr><tr><td>Pythia-12B</td><td>12B</td><td>2K</td></tr><tr><td>Llama-3.1-70B</td><td>70B</td><td>128K</td></tr><tr><td>Nemotron-4-340B-Instruct</td><td>340B</td><td>4K</td></tr><tr><td>Llama-3.1-405B-Instruct</td><td>405B</td><td>128K</td></tr><tr><td>Closed-source</td><td></td><td></td></tr><tr><td>davinci-002</td><td></td><td>16K</td></tr></table>

![](images/f819bc30bf89cb92060d8911ec534b4676bcceb322aa174a01967492ba1c2450.jpg)  
<sub>gnostic</sub> <sub>protocols</sub>. W<sup>e</sup> <sup>evaluate</sup> <sup>three</sup> <sup>axes</sup> <sup>across</sup> <sup>model</sup> <sup>scale</sup>, <sup>namely</sup> <sup>encoder-side</sup> <sup>synonym</sup> <sup>fuz</sup> <sub>obing</sub> <sub>(middle)</sub>, <sub>and</sub> <sub>our</sub> <sub>proposed</sub> <sub>I/O</sub> <sub>isomorp</sub><sup>hism</sup> <sup>protocol</sup> <sup>with</sup> <sup>opcode</sup> <sup>analysis</sup> <sup>(bottom)</sup>. <sup>Th</sup> <sub>)1))))</sub> <sub>tly</sub> <sub>(f(x=y⇐</sub><sup>⇒Tθ(f(T−</sup> <sup>θ(x′=y′</sup> <sup>while</sup> <sup>isolating</sup> <sup>representational</sup> <sup>load</sup> <sup>on</sup> <sup>both</sup> <sup>the</sup> <sup>prompt-p</sup> <sub>h</sub><sup>ases</sup>

We select 11 models spanning a wide range of capabilities for the encoder-side and isomorphism experiments (Table 1a), prioritizing models commonly used for coding tasks and covering a broad parameter-count range. For the decoder-side contamination analysis (Section 5.3), we construct a separate scale axis of 7 models (Table 1b) that permit teacher-forced scoring, computing per-token log-probabilities of a target sequence conditioned on its ground-truth prefix, without generating new tokens. For open-weight models (Pythia 410M, 1.4B, 12B; Nemotron-4-340B; Llama-3.1-405B), we use the vLLM library and access prompt-token log-probabilities directly through prompt logprobs. For davinci-002, we use OpenAI’s legacy completions endpoint, which exposes equivalent functionality via echo=True with logprobs, to include at least one commercial model in our study.

A deliberate design choice guides our model selection. All models in our scale axis are dense architectures, not mixture-of-experts (MoE). Because MoE models activate only a fraction of their parameters per token, the relationship between nominal parameter count and efective capacity is ambiguous. A 340B MoE model may behave more like a much smaller dense model on any given input. Since our analysis aims to study how memorization and generalization behaviours change as a function of scale, dense architectures provide a cleaner signal, as total parameters directly reflect the capacity available during inference. How decoder-side diagnostics should be adapted for MoE architectures, where routing decisions may interact with memorization patterns in ways that dense models do not exhibit, remains an open question.

## 5.2 Datasets, Conditions, and Generation Setup

Table 2 Evaluation datasets across the study’s two diagnostic axes. The upper block covers the I/O isomorphism and encoder-side perturbations. The lower block covers the CoDeC probe, split into corpora that postdate every model’s training cutof (Unseen) and corpora that are documented or probable members of the training mixtures (Seen).
<table><tr><td>Dataset</td><td>Tasks / Role</td><td>Domain</td><td>Released</td></tr><tr><td colspan="4">I/O isomorphism and encoder-side perturbations</td></tr><tr><td>MBPP [2]</td><td>257</td><td>Mostly-basic Python</td><td>Aug 2021</td></tr><tr><td>EffiBench [41]</td><td>338</td><td>Efficiency</td><td>Nov 2024</td></tr><tr><td>BigOBench [42]</td><td>640</td><td>Algorithmic complexity</td><td>Mar 2025</td></tr><tr><td colspan="4">Decoder-side perturbations (CoDeC)</td></tr><tr><td>gpqa_diamond [43]</td><td>Unseen</td><td>Graduate-level Q&amp;A</td><td>Nov 2023</td></tr><tr><td>livecodebench_v5 [35]</td><td>Unseen</td><td>Competitive programming</td><td>Jan 2025</td></tr><tr><td>wikipedia (Pile) [17]</td><td>Seen</td><td>Encyclopedic text</td><td>Dec 2020</td></tr><tr><td>hackernews (Pile) [17]</td><td>Seen</td><td>Web forum text</td><td>Dec 2020</td></tr><tr><td>cc_2024_high [18]</td><td>Seen†</td><td>Common-Crawl web text</td><td>2024</td></tr></table>

<sup>†</sup> Substituted for hackernews when probing Nemotron-4-340B-Instruct, following the corpus-overlap designation of Zawalski et al. [13].

Datasets. We evaluate on three Python benchmarks spanning both a dificulty axis and a contamination axis (Table 2). MBPP sits at the easy end of the dificulty spectrum and is among the most widely used benchmarks in the code LLM literature [10, 12, 35]. Every model in our study was released after its publication, making all potentially contaminated on this benchmark. EfiBench and BigOBench, by contrast, were published in 2024 and 2025, respectively, after the training cutof of all models except Gemini-2.5-Flash. This creates a natural experiment. If memorization were the primary driver of performance, we would expect systematically larger Iso drops on the contaminated benchmark (MBPP), while drops on unseen benchmarks (EfiBench and BigOBench) should not be systematically large. We test this prediction directly in Section 6. All three benchmarks use Python exclusively, removing cross-language variation as a confound.

Evaluation conditions. We utilize six conditions for inference. The first four probe representational invariance via $\mathrm { I } / \mathrm { O }$ isomorphisms; the last two serve as encoder-side baselines operating in lexical space.

Synonym fuzzing protocol. For the Syn-20% and Syn-40% conditions, we replace each non-connector, non-code-token word in the natural-language portion of the prompt with a WordNet synonym at the specified probability. This preserves task semantics while altering surface form, the same axis targeted directly in recent work [10] and by other similar probes like variable renaming [6, 7] and paraphrasing [8, 9]. We work under the same hypothesis as theirs. If a model has merely memorized a solution, altering the surface form should degrade retrieval.

Generation setup. For each (task, condition) pair, we generate n=5 completions at temperature T=0.0. Fixing the temperature at zero removes temperature-driven sampling as a source of variation, so that the behaviour we observe under each condition is not confounded by deliberate exploration of the output distribution. Greedy decoding is nonetheless not bitwise deterministic in practice, because floating-point reductions on accelerators are non-associative and sensitive to batch composition, and the hosted endpoints we query give no determinism guarantee, so repeated generations of the same prompt still diverge wherever two continuations lie close in probability. It is this residual nondeterminism, not temperature, that supplies the per-problem solution spread we analyse. We generate five completions for two reasons. First, our opcode-entropy analysis (Section 5.5) reads of a per-problem distribution of solutions, and pooling opcode frequency vectors across five generations enables meaningful per-problem Jensen-Shannon divergence estimates between conditions. Second, both EfiBench and BigOBench admit multiple valid solutions to each problem, so several generations help to characterise the solution space the model explores. The same decoding regime is held fixed across the Original and isomorphic conditions, so any divergence we report reflects the condition rather than a change in how solutions are drawn.

## 5.3 Decoder-Side Protocol (CoDeC)

We reproduce CoDeC [13], a contamination diagnostic that operates on output log-probabilities. Each dataset receives a contamination score $S _ { \mathrm { C o D e C } } ( D )$ = $( 1 / N ) \textstyle \sum _ { i } \mathbf { 1 } [ \Delta ( x _ { i } ) < 0 ]$ , where $\Delta ( x _ { i } )$ is the change in mean token log-likelihood of target $x _ { i }$ when adding in-context samples from the same dataset. In-context examples typically boost confidence for unseen data but may reduce it when the dataset was memorized during training by disrupting learned retrieval patterns. The datasetlevel AUC quantifies separation between seen and unseen datasets. Formally, with seen-score set $S ^ { + }$ and unseen-score set $S ^ { - }$

$$
\mathrm { A U C } = \frac { 1 } { | S ^ { + } | | S ^ { - } | } \sum _ { s ^ { + } \in S ^ { + } } \sum _ { s ^ { - } \in S ^ { - } } \left[ { \bf 1 } ( s ^ { + } > s ^ { - } ) + 0 . 5 { \bf 1 } ( s ^ { + } = s ^ { - } ) \right] .
$$

To keep frontier-scale runs feasible, we use a constrained but consistent setup with two seen and two unseen datasets per run. We designate gpqa diamond [43] and livecodebench v5 [35] as unseen, since both postdate the training cutof of every model in our decoding experiments. For the seen datasets, we follow the designations reported by Zawalski et al. [13] and use hackernews and wikipedia from the pile dataset [17] (with cc 2024 high [18] replacing hackernews for Nemotron-4-340B-Instruct due to known corpus overlap).

## 5.4 I/O Isomorphism Protocol (Metamorphic Testing for Code LLMs)

A coding-benchmark instance defines a mapping $f : \mathcal { X } \ :  \ : \mathcal { Y }$ over textual $\mathrm { I } / \mathrm { O }$ representations. We define a family of invertible value transforms parameterized by θ:

$$
T _ { \theta } : \mathcal { V } \to \mathcal { V } , \qquad T _ { \theta } ^ { - 1 } \big ( T _ { \theta } ( v ) \big ) = v .
$$

For the main experiments, we use afine integer isomorphisms:

$$
T _ { \theta } ( t ) = a t + b , \quad a \neq 0 , \qquad T _ { \theta } ^ { - 1 } ( t ^ { \prime } ) = \frac { t ^ { \prime } - b } { a } .
$$

Every integer in the test $\mathrm { I } / \mathrm { O }$ is mapped through $T _ { \theta }$ . Original tests $( x , y )$ become $( x ^ { \prime } , y ^ { \prime } )$ with $x ^ { \prime } = T _ { \theta } ( x ) , y ^ { \prime } = T _ { \theta } ( y )$ . The prompt is augmented with a contract specifying the values of a and b, along with instructions to decode inputs, solve, and return encoded outputs. We additionally ablate with two alternative bijection families, base conversion and cubic polynomials, to test whether the observed brittleness generalizes beyond afine transforms (described in Section 7.1).

Correctness equivalence (the metamorphic relation). Because $T _ { \theta }$ is bijective,

$$
f ( x ) = y \iff T _ { \theta } { \big ( } f { \big ( } T _ { \theta } ^ { - 1 } ( x ^ { \prime } ) { \big ) } { \big ) } = y ^ { \prime } .
$$

Any accuracy diference between Original and Isomorphic variant (Iso) of a problem is therefore attributable to representational handling, not to a change in the underlying task.

## 5.5 Opcode Entropy as an Algorithmic Proxy

Pass@k measures whether the model produces any correct solution but says nothing about what kind. To characterize the solution space beyond binary correctness, we compile each generated Python program to bytecode using dis and compute the Shannon entropy of its opcode-frequency distribution,

$$
H ( \pi ) = - \sum _ { o \in \mathcal { O } } \pi ( o ) \log _ { 2 } \pi ( o ) ,
$$

where $\pi ( o )$ is the normalized frequency of opcode o. Programs relying on similar algorithmic strategies produce similar opcode profiles, making entropy a lightweight proxy for the algorithmic family a solution belongs to. To quantify distributional divergence at the per-problem level, we pool the opcode frequency vectors from all 5 generations for a given (problem, condition) pair and compute the Jensen-Shannon divergence (JSD) against the corresponding pooled distribution under Original. JSD is bounded in [0, 1] (base 2) and symmetric, providing a principled measure of how far a model’s solution strategy shifts under perturbation.

## 6 Results

Across all eleven models and three benchmarks, lexical probes (Syn) and likelihood probes (CoDeC) both saturate at scale. Our metamorphic Iso protocol drops frontiermodel pass@1 by 14–30 points even though the algorithmic task is provably unchanged, yet these are not algorithmic failures. Pooled-opcode JSD between Original and Iso generations stays near zero for scaled models, which keep the same control flow and merely mis-serialize the interface, and spikes only for smaller models on harder benchmarks, which abandon the strategy outright.

## 6.1 Encoder-Side Probes Saturate at Scale

Table 3 reports pass@1 under Syn-20 and Syn-40 across all eleven models and three benchmarks. The results show a clear saturation pattern. Synonym fuzzing remains diagnostic for small and some mid-tier models, but is nearly inert at the frontier.

Frontier models absorb synonym perturbation. For frontier models, pass@1 drops remain small across benchmarks and conditions. Gemini-2.0-Flash loses at most 5.4 points, while Gemini-2.5-Flash loses even less, with a maximum drop of 2.1 points and occasional gains on BigOBench and EfiBench. GPT-4o shows the same overall pattern, with modest and consistent losses. Because these perturbations follow the synonym-fuzzing protocol of Djir´e et al. [10], the result points to failure of lexical perturbations as a tool to study memorization, which was validated on smaller models but loses discriminative power at the frontier scale, where encoders readily collapse semantically equivalent surface forms.

Mid-tier models show thresholded degradation. GPT-4o-mini, Llama-3.1-70B, and Qwen2.5-Coder-32B fall in an intermediate regime. Under Syn-20, drops are modest, but Syn-40 produces larger and benchmark-dependent losses. For example, Llama-3.1-70B drops 7.9 points on MBPP under Syn-20 but 27.3 under Syn-40, while Qwen2.5-Coder-32B shows a similar jump on MBPP. This non-linearity suggests that mid-tier encoders tolerate moderate lexical noise but degrade once perturbation density crosses a threshold.

<table><tr><td colspan="8">Table 3 Pass@1 (%) across all conditions and models. Values in parentheses show the relative change from ORIGINAL. ± denotes the half-width of a 95 % bootstrap CI (10000 resamples over problems). Tier labels: F = frontier, M = mid-tier S = small.</td></tr><tr><td>Dataset</td><td>Model</td><td>Orig</td><td>Iso</td><td>Iso (Enc)</td><td>Iso (Dec)</td><td>Syn-20</td><td>Syn-40</td></tr><tr><td rowspan="13">BigO</td><td>F gemini-2.5-flash</td><td>43.2±2.2</td><td>25.5±4.1 (-41%)</td><td>35.4 (-18%)</td><td>30.1 (-30%)</td><td>45.1 (+4%)</td><td>46.2 (+7%)</td></tr><tr><td>F gemini-2.0-flash</td><td>65.9±2.0</td><td>50.3±4.9 (-24%)</td><td>55.4 (-16%)</td><td>54.8 (-17%)</td><td>62.2 (-6%)</td><td>60.5 (-8%)</td></tr><tr><td>gpt-4o-2024-08-06F</td><td>50.9±2.1</td><td>37.2±3.1 (-27%)</td><td>46.5 (-9%)</td><td>40.1 (-21%)</td><td>49.4 (-3%)</td><td>48.0 (-6%)</td></tr><tr><td>gpt-4o-mini-2024-07-18M</td><td>41.5±2.0</td><td>24.9±4.4 (-40%)</td><td>33.8 (-19%)</td><td>27.5 (-34%)</td><td>37.4 (-10%)</td><td>33.2 (-20%)</td></tr><tr><td></td><td>42.6±2.0</td><td>20.9±4.7 (-51%)</td><td>32.4 (-24%)</td><td>27.3 (-36%)</td><td>37.5 (-12%)</td><td>31.1 (-27%)</td></tr><tr><td></td><td>30.1±1.9</td><td>18.7±2.8 (-38%)</td><td>24.4 (-19%)</td><td>22.3 (-26%)</td><td>27.7 (-8%)</td><td>26.8 (-11%)</td></tr><tr><td></td><td>22.0±1.7</td><td>12.6±3.9 (-43%)</td><td>17.8 (-19%)</td><td>13.9 (-37%)</td><td>17.2 (-22%)</td><td>16.1 (-27%)</td></tr><tr><td>StarCoder2-15BS</td><td>0.6±0.4</td><td>0.0±0.6 (-100%)</td><td>0.3 (-50%)</td><td>0.0 (-100%)</td><td>0.3 (-50%)</td><td>0.0 (-100%)</td></tr><tr><td>S CodeLlama-13B-Instruct</td><td>2.1±0.6</td><td>1.7±1.5 (-19%)</td><td>1.9 (-10%)</td><td>1.8 (-14%)</td><td>1.2 (-43%)</td><td>0.8 (-62%)</td></tr><tr><td>Llama-3.1-8B-InstructS</td><td>11.2±1.3</td><td>3.8±2.9 (-66%)</td><td>7.4 (-34%)</td><td>4.7 (-58%)</td><td>8.9 (-21%)</td><td>2.5 (-78%)</td></tr><tr><td></td><td>14.5±1.5</td><td>5.1±3.3 (-65%)</td><td>10.2 (-30%)</td><td>6.5 (-55%)</td><td>8.3 (-43%)</td><td>4.8 (-67%)</td></tr><tr><td>F gemini-2.5-flash</td><td>26.2±3.1</td><td>11.7±5.1 (-55%)</td><td>21.0 (-20%)</td><td>15.4 (-41%)</td><td>27.0 (+3%)</td><td>28.1 (+7%)</td></tr><tr><td rowspan="10">Eff</td><td>F gemini-2.0-flash</td><td>54.4±3.1</td><td>29.3±5.8 (-46%)</td><td>45.2 (-17%)</td><td>35.0 (-36%)</td><td>53.2 (-2%)</td><td>51.8 (-5%)</td></tr><tr><td></td><td>62.2±3.3</td><td>33.8±6.2 (-46%)</td><td>52.1 (-16%)</td><td>40.2 (-35%)</td><td>60.5 (-3%)</td><td>58.4 (-6%)</td></tr><tr><td>gpt-4o-mini-2024-07-18M</td><td>48.0±3.6</td><td>22.6±5.3 (-53%)</td><td>36.5 (-24%)</td><td>26.4 (-45%)</td><td>43.7 (-9%)</td><td>39.4 (-18%)</td></tr><tr><td>Llama-3.1-70B-InstructM</td><td>36.2±3.2</td><td>13.8±5.0 (-62%)</td><td>23.5 (-35%)</td><td>17.7 (-51%)</td><td>32.2 (-11%)</td><td>24.6 (-32%)</td></tr><tr><td>Qwen2.5-Coder-32B-Instruct M</td><td>39.5±3.3</td><td>20.1±4.0 (-49%)</td><td>30.8 (-22%)</td><td>27.3 (-31%)</td><td>35.9 (-9%)</td><td>28.4 (-28%)</td></tr><tr><td>Codestral-22B-v0.1S</td><td>21.3±2.7</td><td>10.7±4.4 (-50%)</td><td>16.2 (-24%)</td><td>12.1 (-43%)</td><td>18.9 (-11%)</td><td>7.6 (-64%)</td></tr><tr><td>StarCoder2-15BS</td><td>3.9±1.3</td><td>0.0±3.1 (-100%)</td><td>1.2 (-69%)</td><td>0.3 (-92%)</td><td>0.3 (-92%)</td><td>0.6 (-85%)</td></tr><tr><td></td><td>0.4±0.4</td><td>0.0±0.9 (-100%)</td><td>0.2 (-50%)</td><td>0.0 (-100%)</td><td>0.9 (+125%)</td><td></td></tr><tr><td></td><td>14.2±2.2</td><td>3.7±3.7 (-74%)</td><td>8.8 (-38%)</td><td>5.0 (-65%)</td><td>11.4 (-20%)</td><td>0.0 (-100%)</td></tr><tr><td></td><td>9.6±2.0</td><td>2.9±3.1 (-70%)</td><td>6.4 (-33%)</td><td>3.8 (-60%)</td><td>7.8 (-19%)</td><td>9.1 (-36%) 6.2 (-35%)</td></tr><tr><td rowspan="11">MBPP</td><td></td><td>77.4±2.0</td><td>47.2±3.2 (-39%)</td><td>66.6 (-14%)</td><td>54.8 (-29%)</td><td>75.3 (-3%)</td><td>76.1 (-2%)</td></tr><tr><td>F gemini-2.0-flash</td><td>80.8±1.8</td><td>58.1±4.1 (-28%)</td><td>71.1 (-12%)</td><td>59.0 (-27%)</td><td>79.2 (-2%)</td><td>76.4 (-5%)</td></tr><tr><td>gpt-40-2024-08-06F</td><td>84.2±1.8</td><td>58.5±3.7 (-31%)</td><td>76.8 (-9%)</td><td>62.3 (-26%)</td><td>79.1 (-6%)</td><td>76.6 (-9%)</td></tr><tr><td>gpt-4o-mini-2024-07-18M</td><td>73.0±2.1</td><td>43.8±3.2 (-40%)</td><td>61.3 (-16%)</td><td>48.9 (-33%)</td><td>68.3 (-6%)</td><td>62.0 (-15%)</td></tr><tr><td>Llama-3.1-70B-InstructM</td><td>71.8±2.1</td><td>41.4±3.0 (-42%)</td><td>59.6 (-17%)</td><td>47.0 (-35%)</td><td>63.9 (-11%)</td><td>44.5 (-38%)</td></tr><tr><td>Qwen2.5-Coder-32B-InstructM</td><td>84.0±1.8</td><td>45.4±2.1 (-46%)</td><td>68.0 (-19%)</td><td>51.2 (-39%)</td><td>76.4 (-9%)</td><td>52.9 (-37%)</td></tr><tr><td>Codestral-22B-v0.1S</td><td>72.0±2.1</td><td>44.1±2.8 (-39%)</td><td>59.4 (-18%)</td><td>48.2 (-33%)</td><td>64.2 (-11%)</td><td>35.1 (-51%)</td></tr><tr><td>StarCoder2-15BS</td><td>24.3±2.0</td><td>0.8±3.1 (-97%)</td><td>13.6 (-44%)</td><td>4.1 (-83%)</td><td>6.5 (-73%)</td><td>6.4 (-74%)</td></tr><tr><td>CodeLlama-13B-InstructS</td><td>12.8±1.5</td><td>2.5±2.1 (-80%)</td><td>7.8 (-39%)</td><td></td><td></td><td>4.3 (-66%)</td></tr><tr><td></td><td></td><td></td><td></td><td>3.8 (-70%)</td><td>13.2 (+3%)</td><td></td></tr><tr><td></td><td>48.2±2.4</td><td>14.5±3.1 (-70%)</td><td>32.4 (-33%)</td><td>19.8 (-59%)</td><td>41.2 (-15%)</td><td>23.6 (-51%)</td></tr><tr><td></td><td>52.1±2.4</td><td>16.7±3.6 (-68%)</td><td>36.0 (-31%)</td><td>22.4 (-57%)</td><td>44.8 (-14%)</td><td>30.2 (-42%)</td></tr></table>

Small models degrade sharply, but the signal conflates fragility with memorization. Below 22B parameters, synonym fuzzing causes large drops across all benchmarks. Codestral-22B, for instance, loses 36.9 points on MBPP under Syn-40, 13.7 on EfiBench, and 5.9 on BigOBench, with similar trends for Llama-3.1-8B and deepseek-coder-6.7b. If synonym fuzzing primarily disrupted memorized retrieval, the largest efects should appear on MBPP, the only benchmark plausibly contaminated for all models, with smaller drops on EfiBench and BigOBench, which postdate their training cutofs. The data does not support this. Relative losses on unseen benchmarks are often comparable or larger, but this is confounded by task dificulty that was untested in previous works. Small models already have low baseline accuracy on harder tasks, so even small absolute drops become large relative declines. The diagnostic, therefore, mixes encoder fragility with memorization and cannot cleanly separate the two.

RQ1. To what extent do encoder-side memorization probes work at scale? Their diagnostic value diminishes as scale increases. Synonym fuzzing at 20% and 40% rates, following the protocol of Djir´e et al. [10], produces <8- point pass@1 drops for frontier models across all three benchmarks, while midtier models degrade moderately and small models collapse. The probe retains some signal at smaller scales but appears to lose much of its discriminative power at the frontier, at least for the models and benchmarks we examine.

## 6.2 CoDeC Loses Discriminative Power at Scale

Table 4 CoDeC contamination scores across model scale. Seen and Unseen report the mean contamination score for datasets designated as in or absent from the training corpus. AUC quantifies the separation between the two.
<table><tr><td>Model</td><td>Params</td><td>Seen (%)</td><td>Unseen (%)</td><td>AUC (%)</td></tr><tr><td>Pythia-410M</td><td>0.4B</td><td>71.0</td><td>24.0</td><td>100.0</td></tr><tr><td>Pythia-1.4B</td><td>1.4B</td><td>73.5</td><td>21.5</td><td>100.0</td></tr><tr><td>Pythia-12B</td><td>12B</td><td>75.0</td><td>17.0</td><td>100.0</td></tr><tr><td>Llama-3.1-70B</td><td>70B</td><td>8.0</td><td>9.0</td><td>25.0</td></tr><tr><td>davinci-002</td><td></td><td>43.5</td><td>31.0</td><td>75.0</td></tr><tr><td>Nemotron-4-340B</td><td>340B</td><td>26.0</td><td>22.5</td><td>75.0</td></tr><tr><td>Llama-3.1-405B</td><td>405B</td><td>7.5</td><td>2.0</td><td>75.0</td></tr></table>

Table 4 and Figure 4 show CoDeC contamination scores across model scale. Our unseen labels (gpqa diamond, livecodebench v5) are guaranteed, since both datasets postdate the training cutofs of all models in our decoder-side experiments. The seen labels (hackernews, wikipedia, and cc 2024 high for Nemotron) follow prior corpusoverlap analyses and documented training mixtures [13, 17, 44]. Although the seen status is therefore probable rather than certain, this asymmetry is shared across all models and does not afect the trend. Our reproduction matches the original CoDeC behaviour on Pythia: seen scores lie between 71–75%, unseen scores between 17–24%, and AUC remains 100% across all three checkpoints.

Seen vs Unseen Contamination Scores  
![](images/c1da90fe0cfaa0656ee04941f9576536ae357e50d56ce3679b9ac23c2b84bf03.jpg)  
(a)

Classification AUC (Seen vs Unseen)  
![](images/8c67f7cfdc6d85ce24c47a2128679fc0e077c7d6dad95b961a14b647f9a86bcf.jpg)  
(b)  
Fig. 4 (a) Mean CoDeC scores across 4 datasets designated as Seen or Unseen (Section 5.3). (b) AUC quantifying the separation between seen and unseen scores across 4 datasets.

The gap reduces with scale. As model size increases, both contamination scores and seen-unseen separation shrink (Figure 4a). davinci-002 narrows the gap to 12.5 points, Nemotron-4-340B to 3.5, and Llama-3.1-405B to 5.5. Llama-3.1-70B even inverts the expected ordering, yielding an AUC of 25%, though this is likely an anomaly given the small number of datasets. Even ignoring that inversion, seen-dataset scores drop from 71–75% at the Pythia scale to 7.5% at 405B, showing that in-context augmentation shifts token log-likelihoods much less in larger models.

Why does the signal degrade? CoDeC measures how in-context examples shift token log-likelihoods. Memorized targets should be disrupted, while novel targets should benefit. This assumes memorized content occupies a concentrated region of probability space that context can perturb. As models scale, that signal weakens for two reasons. First, probability mass is spread across a richer representation space, diluting concentrated retrieval efects. Second, larger models have stronger in-context learning abilities [39, 40], allowing them to use added context constructively for both seen and unseen targets, which reduces the asymmetry CoDeC relies on.

RQ2. To what extent do decoder-side memorization probes work at scale? Their discriminative power appears to weaken with scale. CoDeC’s seen-unseen gap shrinks from 58 points (Pythia-12B) to 5.5 points (Llama-3.1-405B), and AUC drops from 100% to 75%. This trend is consistent with the dilution of localized retrieval signals in richer representation spaces and with the strengthening of in-context learning abilities that may override the memorization-specific asymmetry on which CoDeC depends.

## 6.3 Representational Load Degrades Performance Without Erasing the Algorithm

To answer RQ3 we break it into two smaller sub-questions that can each be tested on their own. A pass@1 drop alone does not pin down the extent to which representational load breaks performance, because the same drop appears both when a model loses the algorithm under the unfamiliar interface and when it keeps the algorithm but misserializes the output. RQ3a isolates where in the model the degradation arises, and RQ3b establishes what kind of failure it is.

## 6.3.1 A Decoder-Side Bottleneck Drives the Degradation

Table 3 reports pass@1 under all evaluation conditions. The full Iso contract drops frontier-model pass@1 by 14–30 absolute points on a provably fixed task, which on its own shows that representational load is suficient to cause substantial degradation.

To localize the source, we decompose the contract into Iso (Enc only), which transforms only the inputs the model must decode, and Iso (Dec only), which transforms only the outputs the model must encode. For Gemini-2.0-Flash on MBPP, decoding inputs alone (Iso (Enc only)) costs a modest 9.7 points (80.8→71.1), whereas encoding outputs token by token (Iso (Dec only)) costs a far larger 21.8 points (80.8→59.0), close to the 22.7-point penalty of the full Iso condition (80.8→58.1). This split holds across datasets and across scaled models. The bottleneck is outputside contract compliance rather than input recognition, consistent with autoregressive generation struggling more with formatting constraints held over many decoding steps than the encoder does with parsing a transformed prompt once.

RQ3a. Where does the degradation from representational load originate? Primarily on the decoder side. I/O isomorphisms drop pass@1 by 14–30 absolute points in frontier models, a vulnerability that purely lexical probes such as synonym fuzzing fail to surface. Decomposing the contract shows that most of this penalty comes from output-side encoding rather than input-side decoding, pointing to contract compliance during generation as the dominant cost.

## 6.3.2 Opcode Analysis Separates Compliance from Competence

A pass@1 drop does not say whether the model has lost the algorithm or has kept it and only mis-serialized the interface. We read this distinction of the opcode-entropy analysis, treating a near-zero shift in the opcode distribution as evidence that the underlying strategy is preserved and a large shift as evidence that it is abandoned.

Scaled models preserve core algorithms and control flow across all representational loads. For frontier models such as Gemini-2.0-Flash and GPT-4o, the per-problem JSD over all solutions, not just passing ones, under Iso and Iso (Dec only) stays near zero across all three benchmarks (Figure 6). The bytecode footprint under the metamorphic prompt is close to indistinguishable from that under the original prompt, and the model opts for the same loops, comparisons, and control flow on the exact problems where pass@1 has just dropped by 14–30 points. The combined distributions in Figure 5 track closely between Original and Iso, with the model selecting the same algorithmic families and producing near-identical opcode profiles while fumbling the final output representation more often. As a calibration check, JSD under Syn-20 and Syn-40 also stays near zero for all models, which indicates that the metric flags preserved core logic rather than any prompt change.

Smaller models lose algorithmic stability as task complexity grows. The opcode distributions reveal a contrast between smaller and mid-tier models, with the latter heavily dependent on dataset dificulty. On easier datasets such as MBPP, models like Codestral-22B behave much like scaled models and keep JSD relatively low under Iso. On harder datasets such as BigOBench and EfiBench, however, smaller models buckle under the representational load and their JSD spikes (Figure 6). This suggests that when the underlying task already sits at the boundary of their capability, the added interface load pushes them to abandon the correct strategy and diverge to qualitatively diferent and often incorrect solution families, or to fail to produce valid logic at all.

Scale expands algorithmic coverage, not just reliability. Comparing the Original opcode profiles across models reveals a further dimension of scaling. On MBPP, Gemini-2.0-Flash and Codestral-22B produce overlapping entropy distributions, and the algorithmic repertoire required for basic tasks is within reach of both. On EfiBench and BigOBench the distributions diverge before any perturbation is applied (Figures 5c and 5d). Gemini-2.0-Flash spans two peaks and performs better at pass@1, while the smaller Codestral-22B largely skips the second peak, consistent with a family of solutions present in the larger model’s repertoire but absent from the smaller one.

## RQ3b. Is the degradation a competence failure or a compliance failure?

A compliance failure for scaled models and a competence failure for smaller ones. Despite large pass@1 drops, frontier models show near-zero opcode JSD under isomorphism, which indicates they attempt the same algorithmic core and mainly fumble the representational channel. Smaller models, by contrast, abandon their strategies under the same load on complex tasks, a pattern more consistent with shallow and fragile task comprehension than with mere mis-serialization.

## RQ3. To what extent can representational load break performance?

Sharply, but in scaled models without erasing the algorithm. The Iso contract drops frontier-model pass@1 by 14–30 absolute points on a provably fixed task, so the loss is the load rather than a memorization artifact. The penalty is concentrated on the decoder, where the model must serialize an encoded output token by token (RQ3a), and for scaled models the algorithmic core survives, surfacing as a narrowed rather than a closed route from specification to code (RQ3b). The competence failure that a bare pass@1 drop might suggest appears only in smaller models on tasks already near their capability limit.

![](images/8a65a4977f49eb0032d971e6abdb866e2bc28f4092bb1193d998a165fbdc483b.jpg)

![](images/d0e88899d0cf9ea3c500495ae929d431515db612f7c73857c0214ef4fb8d8f43.jpg)  
(a) Gemini-2.0-Flash:BigOBench

![](images/e7633d0b67ec58c279d691de87d387e27482033973ed97e1c2290c79fce93cda.jpg)

![](images/31a7a32c78125525ebf11bc8faaafb7e6b71f7605a63ce66c23f85eb03a32f3c.jpg)  
(b) Codestral-22B:BigOBench

![](images/d11e9e6124cd3ade9635bff75ca07f031efda303d70d12818ce7455b77b5b749.jpg)

![](images/9f65cc43b1f9d048f93a59219871b338039ac96e567439e4c73a2267c414c8ef.jpg)  
(c) Gemini-2.0-Flash:Efibench

![](images/e5e55e57924041995ca22dfa078f7a287ebe7d83c03b19b2f67d86e0dbb867bd.jpg)

![](images/9d888a979baab73cefdd8a29cbf6d35ab90465ae9470c0cb6466910e52050269.jpg)  
(d) Codestral-22B:Efibench

![](images/aca2e060dcabf53b88a16ff21240f0075179cb738118ff796ba6a6a6954dec0a.jpg)

![](images/8c78f183cc7cd5bc3a1d60e01bbd9ffa2f68acdcd6100af47850ebb400c84960.jpg)  
(e) Gemini-2.0-Flash:MBPP

![](images/73cbd86830c13ee6398acc00daca4b5e978ba7744aa5bc4b9a1f8766e740eda5.jpg)

![](images/6fa1b3202fc4d82d6439d125e58ae146c3b179fdc1e48b5dd9a38025be300b65.jpg)  
(f) Codestral-22B:MBPP

Fig. 5 Static opcode-entropy distributions (KDE) for selected model-dataset pairs. The left column shows Gemini-2.0-flash producing nearly identical entropy profiles under Original and Iso, both for working solutions and for all generated code. The right column shows that Codestral-22b diverges substantially or produces no working solutions under Iso on harder datasets.

![](images/0ad326084d16c45cd47319b35b431fe6758a7bcf4cec57c7bffd708b5b12fc9d.jpg)

![](images/9243b83316cb29d4cf89b20ef1d6ebdb00bb0328df8580d39a0d1944b525371f.jpg)

![](images/f5f3ccc6bce642e008037fb76842d28adae377cb6be8c48960c99f18913377ee.jpg)  
(a) BigOBench

![](images/591ac72e2cccbd36446ad70748ab95e036eea45c5e58f0f379ae578ad4f9f9d9.jpg)

![](images/358a7e1e4288d9151158480365a489c994b9f856d99bacc044deed9b6023d70e.jpg)

![](images/e759891b666925a6891fab5181cfda7e70b1e0f736319cebc2be38a8586dce48.jpg)

![](images/8c690e4ec7d21ca8018cfebf527165791b8a291e1b00074f821d80e9a907e4b4.jpg)  
(b) EfiBench

![](images/dcd51650d04a92dc2c7d5c851157f50dcfffaa4c5a99a42f5d97c0890ede53dd.jpg)

![](images/4715128ba8cb84a18ed60577fe54d21090d573b1bbccfab80f8218607aec909f.jpg)  
(c) MBPP  
Fig. 6 Per-problem Jensen-Shannon divergence of pooled opcode distributions (Condition vs. Original).

## 7 Ablation Studies

We conduct two ablation studies, one to test whether the observed brittleness generalizes beyond afine transforms, and one to control for prompt-length confounds.

## 7.1 Alternative Isomorphism Families

Table 5 Pass@1 (%) under alternative isomorphism families.
<table><tr><td>Data</td><td>Model</td><td>Orig</td><td>Affine</td><td>Base</td><td></td><td>Cubic</td></tr><tr><td rowspan="3">BigO</td><td>gemini-2.0-flash</td><td>65.9</td><td>50.3 (-24%)</td><td>52.2 (-21%)</td><td></td><td>52.8 (-20%)</td></tr><tr><td>Codestral-22B-v0.1</td><td>22.0</td><td>12.6 (-43%)</td><td>13.4 (-39%)</td><td></td><td>11.7 (-47%)</td></tr><tr><td>Qwen2.5-Coder-32B-Instruct</td><td>30.1</td><td>18.7 (-38%)</td><td>19.8 (-34%)</td><td></td><td>17.5 (-42%)</td></tr><tr><td rowspan="3">Effi</td><td>gemini-2.0-flash</td><td>54.4</td><td>29.3 (-46%)</td><td>26.9 (-51%)</td><td></td><td>37.8 (-31%)</td></tr><tr><td>Codestral-22B-v0.1</td><td>21.3</td><td>10.7 (-50%)</td><td>12.5 (-41%)</td><td></td><td>9.8 (-54%)</td></tr><tr><td>Qwen2.5-Coder-32B-Instruct</td><td>39.5</td><td>20.1 (-49%)</td><td>22.5 (-43%)</td><td></td><td>18.0 (-54%)</td></tr><tr><td rowspan="3">MBPP</td><td>gemini-2.0-flash</td><td>80.8</td><td>58.1 (-28%)</td><td>52.6 (-35%)</td><td></td><td>60.5 (-25%)</td></tr><tr><td>Codestral-22B-v0.1</td><td>72.0</td><td>44.1 (-39%)</td><td>49.3 (-32%)</td><td></td><td>43.2 (-40%)</td></tr><tr><td>Qwen2.5-Coder-32B-Instruct</td><td>84.0</td><td>45.4 (-46%)</td><td>50.5 (-40%)</td><td></td><td>43.0 (-49%)</td></tr></table>

To test whether the brittleness generalizes beyond afine transforms, we ablate with two alternative bijection families targeting distinct representational axes.

1. Base conversion: The decimal digit string of each integer is treated as if it were written in base b, yielding a diferent numeric value. For example, with b=7 the integer 25 has digit string "25", which read in base 7 gives $2 \times 7 + 5 = 1 9 .$ , so $T ( 2 5 ) = 1 9$ . The inverse recovers the original value by inverting the conversion. The base-7 representation of 19 is "25", whose digits read in base 10 give 25. This replaces the arithmetic overhead of afine inversion with a format-parsing overhead, testing whether the model can reason about numeral systems rather than polynomial algebra.

2. Cubic polynomials: $x ^ { \prime } = x ^ { 3 } + c ,$ requiring explicit cube-root inversion, a qualitatively harder challenge than a linear inverse.

We run three models across all datasets as shown in Table 5. All three families produce drops of comparable magnitude. On BigOBench, Gemini-2.0-Flash loses 21% (base) and 20% (cubic), closely tracking the 24% afine drop. On EfiBench, base conversion is steeper than afine (-51% vs. -46%), while the cubic transform is milder (-31%). On MBPP, both alternatives again bracket the afine result. The consistency across families confirms that the brittleness reflects general value-space sensitivity rather than an artifact of any particular transform.

Table 6 Pass@1 (%) under dead-code insertion controlling for the prompt-length confound. Dead-code blocks match the token-count increase of the Iso contract while leaving the task and I/O unchanged.
<table><tr><td>Data</td><td>Model</td><td>Orig</td><td>Iso</td><td>Dead Code</td></tr><tr><td rowspan="3">BigO</td><td>gemini-2.0-flash</td><td>65.9</td><td>50.3 (-24%)</td><td>65.1 (-1%)</td></tr><tr><td>Codestral-22B-v0.1</td><td>22.0</td><td>12.6 (-43%)</td><td>20.5 (-7%)</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>42.6</td><td>20.9 (-51%)</td><td>41.0 (-4%)</td></tr><tr><td rowspan="3">MBPP</td><td>gemini-2.0-flash</td><td>80.8</td><td>58.1 (-28%)</td><td>81.4 (+1%)</td></tr><tr><td>Codestral-22B-v0.1</td><td>72.0</td><td>44.1 (-39%)</td><td>68.5 (-5%)</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>71.8</td><td>41.4 (-42%)</td><td>69.5 (-3%)</td></tr></table>

## 7.2 Prompt Length as a Confound

The Iso condition appends an encoding contract to the prompt, increasing its length. To rule out prompt length as a confound, we insert semantically irrelevant Python code blocks that match the token-count increase of the Iso contract while leaving the task and I/O unchanged. If length alone drives the observed drops, dead-code insertion should produce comparable degradation. Table 6 shows it does not. Deadcode insertion causes negligible drops of 1 to 7% relative, while Iso degrades the same models by 24 to 51% on the same benchmarks. The gap is consistent across all model-dataset pairs. On BigOBench, gemini-2.0-flash loses 0.8 points with dead code versus 15.6 points under Iso, and Llama-3.1-70B loses 1.6 points versus 21.7. On MBPP, Gemini-2.0 under dead code actually gains 0.6 points (+1%), a fluctuation we attribute to noise. In every case, the Iso drop exceeds the dead-code drop by an order of magnitude, confirming that the performance degradation is driven by the representational load of contract compliance, not by prompt length.

## 8 Discussion

## 8.1 Why the Decoder-Side Bottleneck May Be Architectural

Our decomposition experiments indicate that decoding accounts for most of the isomorphism penalty (Section 6.3.1). A plausible reading of this is the asymmetry between prompt understanding and generation. On the encoder side, the model can integrate the contract and task into a coherent representation before producing any output [45], so it can attend to the whole prompt at once. Decoding, by contrast, is sequential, and each token depends only on the prompt and prior outputs [46]. To satisfy the encoding contract, the model has to preserve an implicit arithmetic state across many steps while also maintaining correct Python syntax and contract compliance, and this kind of constrained multi-step generation plausibly stresses working memory in ways that standard next-token training does not explicitly support [47]. The same asymmetry may also help explain why synonym fuzzing saturates at scale (Section 6.1), since lexical variation is resolved in the context-rich regime where scaled representations tend to collapse equivalent forms [48], whereas isomorphism shifts the load onto the decoder, which must serialize a novel computation token by token. If this account is correct, the bottleneck is closer to architectural than incidental and might persist in autoregressive models even as scale narrows the gap, though our experiments speak to the size of the efect rather than to its mechanism directly.

## 8.2 Representational Load as a Practical Test of Generalization

The term “generalization” is highly ambiguous in the code LLM literature. At one extreme, it might require generating genuinely novel algorithms, that is, solutions that do not resemble any training instance in structure or strategy. Our work does not study this aspect of generalization, though some benchmarks, such as ARC-AGI [49, 50], do. At the other extreme, generalization can simply mean producing correct outputs on unseen inputs, which is trivially satisfied by any model that passes a held-out test set (qualitatively diferent from its training data). We study a point between these, generalization under a representational load the model could not have seen, and our results suggest that scaled models have internalized algorithmic strategies well enough to transport them across interfaces they have never encountered. This looks closer to what Chollet [51] terms developer-aware generalization, the ability to handle situations not anticipated by the system’s designers but lying within the span of its learned priors. We read this as a sign that the field’s methods for studying generalization may not have kept pace with the models, and it points toward evaluation frameworks that test whether a model can apply what it knows under novel conditions rather than only reproduce what it has seen.

## 8.3 Scope of the Metamorphic Claim

The biconditional that gives our protocol its sharpness relies on bijective value transforms over numeric test inputs and outputs. Our findings, therefore, characterize code-LLM behavior specifically on numeric programming tasks, and what they indicate there is fairly narrow, that under representational stress, scaled models appear to retain the algorithmic core and mostly fail at re-serializing the encoded interface. Other classes of representational load, such as novel string-formatting contracts, alternate data layouts, or project-local API conventions, are equally important in practice, but the strict metamorphic-testing guarantee does not directly transfer, because constructing a closed bijection on non-numeric input spaces is itself an open problem. Designing metamorphic-style oracles for those loads while preserving the same provable equivalence on the underlying task is a natural extension we leave for future work.

## 8.4 Solution Multiplicity Complicates the Memorization Narrative

Our opcode analysis indicates that scaled models, particularly on BigOBench and EfiBench, span a broad, multi-modal entropy distribution under Original conditions (Section 6.3.2). We interpret these modes as distinct algorithmic families. A sorting problem might admit $O ( n \log n )$ divide and conquer, $O ( n ^ { 2 } )$ comparison-based, and $O ( n )$ counting-based solutions, each with a characteristic opcode signature. Smaller models, by contrast, tend toward narrow, often unimodal distributions, frequently collapsing to a single solution strategy per problem. If this reading holds, the multiplicity has a bearing on how memorization is interpreted, since methodologies targeting the decoder may be confounded when a model can stochastically switch between several “correct” solutions.

## 9 Threats to Validity

Construct validity. Static opcode entropy is a proxy for algorithmic strategy and not a direct measurement of it. Two programs with the same opcode histogram can difer in runtime behavior, and two behaviourally identical programs can compile to diferent bytecode, for instance, a list comprehension against an explicit loop. The Jensen-Shannon divergence we report is invariant to opcode ordering, so it tracks the compositional makeup of a solution rather than its control-flow structure, and a model could, in principle, reorganize its control flow while leaving the histogram unchanged. We mitigate these efects by reporting divergence over both passing and all generations and by reading the metric only as relative movement between matched conditions rather than as an absolute fingerprint. A second construct concern is that representational load is observable only through the transforms we instantiate, so the afine, base-conversion, and cubic families stand in for a property we cannot measure directly.

Internal validity. Iso lengthens the prompt, and our dead-code ablation (Section 7.2) bounds the share of degradation attributable to length alone at roughly 7%. We decode greedily at $T { = } 0 . 0$ to strip sampling variance, which collapses pass@5 toward pass@1 and may understate the headroom a frontier model would show under sampling. The exact wording of the encode-decode contract is a further free variable, since a differently phrased instruction could prime the decoder more or less efectively, and we hold the phrasing fixed rather than searching over it. For the closed-source models, we query hosted endpoints whose system prompts and checkpoint versions are not disclosed, so a silent provider-side change would surface as a behavioral shift we could not attribute. Finally, the Seen labels in the CoDeC probe are probable rather than certain, as noted in Section 5.3, though the asymmetry is shared across every model and does not move the trend.

External validity. Our isomorphisms cover three integer-valued bijection families, all of our prompts are Python, and the metamorphic guarantee holds only over numeric input and output. Whether the same narrow-channel behavior appears under stringformatting contracts, structured data layouts, or project-local API conventions is untested, and constructing a closed bijection on those spaces is itself an open problem, as we discuss above. Our model panel is dense throughout and excludes mixture-ofexperts architectures, where routing may interact with memorization in ways a dense model does not exhibit. The three benchmarks also favor self-contained algorithmic tasks, so our results speak to function-level synthesis rather than repository-scale or agentic coding. Broader transform families, additional languages, MoE models, and larger task units remain future work.

## 10 Conclusion

We do not claim that memorization is absent from scaled code LLMs, only that the diagnostics commonly used to detect it appear to stop discriminating at scale. In our experiments, synonym fuzzing and dead-code insertion on the encoder side, and CoDeC on the decoder side, lose much of their signal on frontier models, so a pass@1 drop or a likelihood shift is, on its own, weak evidence of memorization. Holding the algorithmic task fixed under an I/O isomorphism lets us separate representational load from memorization, and scaled models tend to keep the same solution families while mis-serializing the encoded interface.

The takeaway we would ofer is practical. Treat a bare pass@k drop as a starting point rather than a memorization finding, separate representational load before attributing any drop to recall, and report solution-space stability alongside correctness. For most software engineering settings the more useful question is not whether a solution was seen in training but whether a model can carry a known algorithm through an unfamiliar interface. And if memorization is to be studied in the stronger sense the term usually implies, we would suggest approaching it from a diferent vantage point than the surface and likelihood probes inherited from prior work, which give little to build on at current scales.

## Declarations

Funding. This research was funded by the Luxembourg National Research Fund (FNR) in collaboration with the University of Luxembourg.

Competing interests. The authors have no competing interests to declare that are relevant to the content of this article.

Data availability. The dataset generated and analysed in this study is openly available in Zenodo at https://doi.org/10.5281/zenodo.19248561.

Code availability. The code is openly available at https://github.com/pkrajput/ isomorphic prompts.

Author contributions. Prateek Kumar Rajput framed the methodology and wrote the manuscript. Alberick Euraste Djir´e and Abdoul Aziz Bonkoungou helped run the experiments. Yewei Song and Iyiola Emmanuel Olatunji contributed to reviewing and framing the manuscript. Jacques Klein and Tegawend´e F. Bissyand´e served as project advisors and principal investigators. All authors read and approved the final manuscript.

## References

[1] Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H.P.D.O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al.: Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374 (2021)

[2] Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al.: Program synthesis with large language models. arXiv preprint arXiv:2108.07732 (2021)

[3] Jimenez, C.E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., Narasimhan, K.: Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770 (2023)

[4] Zhang, L., He, S., Zhang, C., Kang, Y., Li, B., Xie, C., Wang, J., Wang, M., Huang, Y., Fu, S., et al.: Swe-bench goes live! arXiv preprint arXiv:2505.23419 (2025)

[5] Liang, S., Garg, S., Moghaddam, R.Z.: The swe-bench illusion: When state-ofthe-art llms remember instead of reason. arXiv preprint arXiv:2506.12286 (2025)

[6] Chakraborty, S., Ahmed, T., Ding, Y., Devanbu, P.T., Ray, B.: Natgen: generative pre-training by “naturalizing” source code. In: Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pp. 18–30 (2022)

[7] Wang, S., Li, Z., Qian, H., Yang, C., Wang, Z., Shang, M., Kumar, V., Tan, S., Ray, B., Bhatia, P., et al.: Recode: Robustness evaluation of code generation models. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13818–13843 (2023)

[8] Mastropaolo, A., Pascarella, L., Guglielmi, E., Ciniselli, M., Scalabrino, S., Oliveto, R., Bavota, G.: On the robustness of code generation techniques: An empirical study on github copilot. In: 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), pp. 2149–2160 (2023). IEEE

[9] Chen, J., Zhenhao, L., Xing, H., Xin, X.: Nlperturbator: Studying the robustness of code llms to natural language variations. ACM Transactions on Software Engineering and Methodology (2024)

[10] Djir´e, A.E., Kabor´e, A.K., Barr, E.T., Klein, J., Bissyand´e, T.F.: Memorization or interpolation? detecting llm memorization through input perturbation analysis. arXiv preprint arXiv:2505.03019 (2025)

[11] Euraste, D.A., Kader, K.A., Samhi, J., Barr, E.T., Klein, J., Bissyand´e, T.F.: Learned or memorized? quantifying memorization advantage in code llms. arXiv preprint arXiv:2604.13997 (2026)

[12] Riddell, M., Ni, A., Cohan, A.: Quantifying contamination in evaluating code generation capabilities of language models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14116–14137 (2024)

[13] Zawalski, M., Boubdir, M., Ba lazy, K., Nushi, B., Ribalta, P.: Detecting data contamination in llms via in-context learning. arXiv preprint arXiv:2510.27055 (2025)

[14] Oren, Y., Meister, N., Chatterji, N.S., Ladhak, F., Hashimoto, T.: Proving test set contamination in black-box language models. In: The Twelfth International Conference on Learning Representations (2023)

[15] Golchin, S., Surdeanu, M.: Data contamination quiz: A tool to detect and estimate contamination in large language models. Transactions of the Association for Computational Linguistics 13, 809–830 (2025)

[16] Nie, Y., Wang, C., Wang, K., Xu, G., Xu, G., Wang, H.: Decoding secret memorization in code llms through token-level characterization. In: 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE), pp. 2880–2892 (2025). IEEE

[17] Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., et al.: The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027 (2020)

[18] Rafel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21(140), 1–67 (2020)

[19] Chen, T.Y., Cheung, S.C., Yiu, S.M.: Metamorphic testing: A new approach for generating next test cases. Technical Report HKUST-CS98-01, Department of Computer Science, Hong Kong University of Science and Technology (1998)

[20] Segura, S., Fraser, G., Sanchez, A.B., Ruiz-Cort´es, A.: A survey on metamorphic testing. IEEE Transactions on Software Engineering 42(9), 805–824 (2016)

[21] Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., et al.: Extracting training data from large language models. In: 30th USENIX Security Symposium (USENIX Security 21), pp. 2633–2650 (2021)

[22] Lee, K., Ippolito, D., Nystrom, A., Zhang, C., Eck, D., Callison-Burch, C., Carlini, N.: Deduplicating training data makes language models better. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8424–8445 (2022)

[23] Coignion, T., Quinton, C., Rouvoy, R.: A performance study of llm-generated code on leetcode. In: Proceedings of the 28th International Conference on Evaluation and Assessment in Software Engineering, pp. 79–89 (2024)

[24] Matton, A., Sherborne, T., Aumiller, D., Tommasone, E., Alizadeh, M., He, J., Ma, R., Voisin, M., Gilsenan-McMahon, E., Gall´e, M.: On leakage of code generation evaluation datasets. In: Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 13215–13223 (2024)

[25] Sainz, O., Campos, J., Garc´ıa-Ferrero, I., Etxaniz, J., Lacalle, O.L., Agirre, E.: Nlp evaluation in trouble: On the need to measure llm data contamination for each benchmark. In: Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 10776–10787 (2023)

[26] Shi, W., Ajith, A., Xia, M., Huang, Y., Liu, D., Blevins, T., Chen, D., Zettlemoyer, L.: Detecting pretraining data from large language models. arXiv preprint arXiv:2310.16789 (2023)

[27] Mattern, J., Mireshghallah, F., Jin, Z., Sch¨olkopf, B., Sachan, M., Berg-Kirkpatrick, T.: Membership inference attacks against language models via neighbourhood comparison. In: Findings of the Association for Computational Linguistics: ACL 2023, pp. 11330–11343 (2023)

[28] Zhang, C., Ippolito, D., Lee, K., Jagielski, M., Tram\`er, F., Carlini, N.: Counterfactual memorization in neural language models. Advances in Neural Information Processing Systems 36, 39321–39362 (2023)

[29] Grosse, R., Bae, J., Anil, C., Elhage, N., Tamkin, A., Tajdini, A., Steiner, B., Li, D., Durmus, E., Perez, E., et al.: Studying large language model generalization with influence functions. arXiv preprint arXiv:2308.03296 (2023)

[30] Biderman, S., Prashanth, U., Sutawika, L., Schoelkopf, H., Anthony, Q., Purohit, S., Raf, E.: Emergent and predictable memorization in large language models. Advances in Neural Information Processing Systems 36, 28072–28090 (2023)

[31] Power, A., Burda, Y., Edwards, H., Babuschkin, I., Misra, V.: Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv preprint arXiv:2201.02177 (2022)

[32] Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J.: Progress measures for grokking via mechanistic interpretability. arXiv preprint arXiv:2301.05217 (2023)

[33] Liu, Z., Kitouni, O., Nolte, N.S., Michaud, E., Tegmark, M., Williams, M.: Towards understanding grokking: An efective theory of representation learning. Advances in Neural Information Processing Systems 35, 34651–34663 (2022)

[34] Tirumala, K., Markosyan, A., Zettlemoyer, L., Aghajanyan, A.: Memorization without overfitting: Analyzing the training dynamics of large language models. Advances in Neural Information Processing Systems 35, 38274–38290 (2022)

[35] Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., Stoica, I.: Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974 (2024)

[36] Liu, J., Xia, C.S., Wang, Y., Zhang, L.: Is your code generated by chatgpt

really correct? rigorous evaluation of large language models for code generation. Advances in neural information processing systems 36, 21558–21572 (2023)

[37] Xia, C.S., Deng, Y., Zhang, L.: Top leaderboard ranking= top coding proficiency, always? evoeval: Evolving coding benchmarks via llm. arXiv preprint arXiv:2403.19114 (2024)

[38] Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., Amodei, D.: Scaling laws for neural language models. arXiv preprint arXiv:2001.08361 (2020)

[39] Wei, J., Tay, Y., Bommasani, R., Rafel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., et al.: Emergent abilities of large language models. arXiv preprint arXiv:2206.07682 (2022)

[40] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Advances in neural information processing systems 33, 1877–1901 (2020)

[41] Huang, D., Qing, Y., Shang, W., Cui, H., Zhang, J.M.: Efibench: Benchmarking the eficiency of automatically generated code. Advances in Neural Information Processing Systems 37, 11506–11544 (2024)

[42] Chambon, P., Roziere, B., Sagot, B., Synnaeve, G.: Bigo (bench)–can llms generate code with controlled time and space complexity? arXiv preprint arXiv:2503.15242 (2025)

[43] Rein, D., Hou, B.L., Stickland, A.C., Petty, J., Pang, R.Y., Dirani, J., Michael, J., Bowman, S.R.: Gpqa: A graduate-level google-proof q&a benchmark. In: First Conference on Language Modeling (2024)

[44] Biderman, S., Schoelkopf, H., Anthony, Q.G., Bradley, H., O’Brien, K., Hallahan, E., Khan, M.A., Purohit, S., Prashanth, U.S., Raf, E., et al.: Pythia: A suite for analyzing large language models across training and scaling. In: International Conference on Machine Learning, pp. 2397–2430 (2023). PMLR

[45] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

[46] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al.: Language models are unsupervised multitask learners. OpenAI blog 1(8), 9 (2019)

[47] Dziri, N., Lu, X., Sclar, M., Li, X.L., Jiang, L., Lin, B.Y., Welleck, S., West, P., Bhagavatula, C., Le Bras, R., et al.: Faith and fate: Limits of transformers on

compositionality. Advances in neural information processing systems 36, 70293– 70332 (2023)

[48] Ethayarajh, K.: How contextual are contextualized word representations? comparing the geometry of bert, elmo, and gpt-2 embeddings. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 55–65 (2019)

[49] Chollet, F., Knoop, M., Kamradt, G., Landers, B.: Arc prize 2024: Technical report. arXiv preprint arXiv:2412.04604 (2024)

[50] Chollet, F., Knoop, M., Kamradt, G., Landers, B., Pinkard, H.: Arc-agi-2: A new challenge for frontier ai reasoning systems. arXiv preprint arXiv:2505.11831 (2025)

[51] Chollet, F.: On the measure of intelligence. arXiv preprint arXiv:1911.01547 (2019)