# Do LLMs Know a Good Hypothesis When They See One? Logit-Based Energy Scoring Outperforms Prompted LLM-as-Judge for Scientific Hypothesis Ranking

Swati Rajwal<sup>†∗</sup>, Sanjay Das<sup>†∗</sup>, Tirthankar Ghosal<sup>†</sup>

<sup>†</sup> Oak Ridge National Laboratory, {rajwals,dass3,ghosalt}@ornl.gov <sup>∗</sup>These authors contributed equally to this work.

Abstract—Large language models (LLMs) are increasingly used for scientific hypothesis generation. However, evaluating generated hypotheses remains a challenge for trustworthy AIenabled scientific workflows. Existing approaches often use LLMs as judges or rely on semantic similarity, which can favor familiar ideas over novel ones. We propose a logitbased energy scoring method that evaluates hypotheses using a language model’s intrinsic confidence rather than comparative judgment. We benchmarked seven language models on 1,323 papers across 12 disciplines. Each paper was paired with its hypothesis and fifteen incorrect alternatives. Intrinsic scoring reached 33.0% Hit@1 pooled across both scorers, compared with 16.6% for prompted listwise ranking. The strongest configuration, a 1-billion-parameter model using logit-based energy scoring, reached 53.1%, though this was the maximum across 14 model-by-scorer combinations selected post hoc. Overall, intrinsic model confidence shows potential for scientific hypothesis evaluation. This study also motivates future research on confidencebased methods for trustworthy AI-enabled scientific discovery.

Index Terms—Large Language Models, Scientific Hypothesis Ranking, Energy-Based Scoring, Scientific Discovery.

## I. INTRODUCTION

Recent advances in large language models (LLMs) have transformed the landscape of scientific research assistance, enabling unprecedented capabilities in hypothesis generation, literature analysis, and even autonomous discovery workflows. Pioneering systems, such as Google’s AI Co-Scientist or Sakana AI, exemplify this technological trend, illustrating how LLMs can propose plausible hypotheses either in collaboration with human experts or as independent agents within scientific pipelines. These developments open new possibilities for accelerating discovery, fostering interdisciplinarity, and automating hypothesis-driven exploration in both established and emerging fields.

A central step in the scientific process is the formulation of a hypothesis that is both consistent with prior literature and capable of explaining or predicting a phenomenon of interest.

As LLMs are increasingly proposed as tools for accelerating scientific discovery (from literature-grounded ideation to automated experimental design) it becomes important to ask a more basic question: given the background and research question that motivate a study, can a language model recognize which of several candidate hypotheses is the one that domain experts actually pursued? This question is distinct from, and arguably prerequisite to, the harder task of generating novel hypotheses: before a model can be trusted to propose new scientific ideas, it should first be able to discriminate correct hypotheses from superficially plausible but incorrect ones.

The evaluation of LLM-generated scientific hypotheses currently relies primarily on LLM-as-a-judge frameworks, which depend on explicit verbal feedback and semantic similarity metrics, or manual human expert verification. Both paradigms pose significant constraints: automated judges frequently suffer from position bias, verbosity bias, and semantic alignment tendencies that favor conventional ideas [1]–[3] while discarding genuinely novel hypotheses, whereas expert human review is inherently unscalable, time-consuming, and resource-intensive.

## Key Finding

A model’s own likelihood over a candidate hypothesis is a stronger signal of scientific correctness than explicitly prompting a proprietary model to rank candidates. Nearly every open-weight LLM we tested beats the prompted baseline. Which intrinsic scorer is better, however, is model-dependent: the raw-logit energy criterion helps some models substantially and fails for others, and understanding why is an open question.

To overcome these limitations, we propose Logit-based Energy Scoring, an alternative model-intrinsic evaluation framework. By leveraging output logit probabilities as a direct proxy for internal model confidence, our approach calculates “energy scores” to reliably identify the most plausible candidate hypothesis without relying on verbalized judgment. Extensive benchmarking across a curated dataset reveals that this mechanism scales effectively from 1B-parameter models up to 20B systems and top proprietary baselines. Notably, lightweight open-source models such as Llama 3.2 1B match or surpass a high-capacity proprietary baseline under either intrinsic criterion, with the raw target-logit energy variant producing the largest single gain. We emphasize that the energy criterion is not uniformly superior to likelihood scoring: its benefit is concentrated in a subset of models, and characterizing that subset is a contribution of this work rather than a caveat to it. Ultimately, this work shows the potential of intrinsic confidence metrics for scientific discovery and offers a path toward democratizing advanced hypothesis evaluation via open-source models.

![](images/da7a339843e2731be91212ff92af3fb15ca783320cbe2362a8577aa3445e2a03.jpg)  
Fig. 1. Overview of the study. Given a background and research question, a candidate hypothesis is scored by a language model’s internal confidence rather than by prompted comparison. 53% is the best of 14 model-by-scorer combinations.

## Our contributions are as follows:

1) This work introduces a novel, fully automatic evaluation metric for LLM-generated scientific hypotheses based on logit-derived energy scoring, thus harnessing the internal confidence distributions of language models, in contrast to conventional judge-style approaches relying on externalized judgments or semantic similarity measures.

2) The paper presents the first systematic comparison of a range of open-source LLMs and a state-of-the-art closed commercial LLM, evaluating their capabilities to discern correct scientific hypotheses, thereby highlighting performance disparities between model architectures, model scale, and evaluation paradigms.

3) Our analysis demonstrates that intrinsic model confidence provides a more reliable signal of scientific hypothesis validity than explicit LLM-judge prompting.

The rest of the article is organized as follows. Section II provides the background on LLM-based scientific hypothesis generation and evaluations. Section III describes the proposed techniques. Section IV delineates the experimental setup and Section V showcases the results. Lastly, the limitations and future works are discussed in Section VI and the article is concluded in Section VII.

## II. BACKGROUND AND RELATED WORK

## A. LLM-based Scientific Hypothesis Generation

A growing body of work deploys LLMs as active participants in the scientific discovery process rather than as passive writing assistants. Google’s AI co-scientist system orchestrates multiple LLM agents in a generate-debate-rank loop, producing biomedical hypotheses that were subsequently evaluated favorably by domain experts on a small number of case studies [4]. Sakana AI’s AI Scientist pursues fully automated, closed-loop research: an LLM ideates a project, writes and executes code to test it, and drafts the resulting paper [5]. Related systems target materials discovery, retrosynthesis planning, and automated theorem or conjecture generation. Across this literature, hypothesis generation itself is treated as largely solved in the sense that LLMs reliably produce fluent, domain-appropriate candidate hypotheses; the harder and less standardized problem, which motivates this paper, is deciding which of many plausible-sounding candidates are actually correct or promising.

## B. Evaluation via LLM-as-Judge and Semantic Similarity

The two evaluation strategies in current practice are (i) LLM-as-judge protocols, in which a (typically large, closed) model is prompted to compare a candidate hypothesis to a reference or to other candidates and produce a score, ranking, or preference [6]; and (ii) embedding-based semantic similarity, in which a candidate hypothesis and a reference hypothesis are each encoded and compared via cosine similarity or a related metric [7]. Both strategies have been widely adopted because they are inexpensive, automatable, and correlate reasonably well with human judgment in aggre gate benchmarks composed largely of previously documented findings. However, both strategies presuppose that correctness and similarity-to-reference are tightly coupled. This coupling is a much weaker assumption for open-ended or forwardlooking scientific hypotheses, where a correct hypothesis is, by definition, not already documented in a form the evaluator has seen, and where surface-level plausibility can be a poor proxy for mechanistic correctness.

## C. Confidence and Likelihood as Evaluation Signals

A separate line of work outside the scientific-discovery literature has studied a language model’s own token-level probabilities as a signal for downstream selection tasks, including re-ranking of generated text, factuality estimation, and out-of-distribution or hallucination detection [8]. Energybased models (EBMs) formalize an unnormalized notion of compatibility between an input and a candidate output via a scalar energy function, with lower energy corresponding to higher compatibility, and have been used to reinterpret the output layer of a classifier or language model without requiring the softmax normalization used to obtain a proper probability distribution [9]–[11]. This literature motivates our use of raw, pre-softmax logit magnitude as a complementary signal to normalized log-likelihood: because normalization is computed over the full vocabulary at each position, it can compress or distort the very confidence signal that distinguishes a wellsupported hypothesis continuation from a merely fluent one, particularly in smaller or less calibrated models.

## III. METHODOLOGY

In this section, we present our approach to investigating whether logit-based models can reliably evaluate scientific hypotheses without relying on comparative judging. Rather than exposing the model to surface fluency or stylistic biases through comparative selection, we measure its per-token confidence given the background context. A model assigns high probability to a well-supported hypothesis while registering high surprise (score) for a less plausible one, reframing hypothesis evaluation as an absolute assessment of intrinsic model predictability.

## A. Problem Setup: Confidence as a Plausibility Signal

For a given paper $p ,$ let $b _ { p }$ denote its background survey, $q _ { p }$ its research question, and $\mathbf { \mathcal { H } } _ { p } = \{ h _ { p } ^ { ( 1 ) } , \dots , h _ { p } ^ { ( 1 6 ) } \}$ a set of sixteen candidate hypotheses, exactly one of which, $h _ { p } ^ { \star } ,$ is the gold hypothesis, with the remaining fifteen serving as plausible but incorrect distractors. Rather than presenting $\mathcal { H } _ { p }$ jointly and requesting a single selection, we evaluate each candidate independently: for every $h _ { p } ^ { ( i ) } \in \mathcal { H } _ { p } ,$ an open-weight language model is conditioned on $( b _ { p } , q _ { p } )$ , and its next-token predictions are compared against the tokens of $h _ { p } ^ { ( i ) }$ . Because all candidates for a given paper are scored under an identical background and research question, differences in score are attributable to the hypothesis text alone, and the candidate assigned the lowest predictive surprisal is treated as the model’s top choice. Hypothesis evaluation is thereby reduced to a ranking problem requiring a single forward pass per candidate, with no explicit comparative judgment step.

## B. Prompt Construction and Hypothesis-Span Isolation

Each candidate hypothesis is scored using a fixed prompt template that places the paper’s background and research question first, followed by the candidate hypothesis:

Prompt Template   
Background:   
[background survey]   
Research Question:   
[research question]   
Hypothesis:   
[candidate hypothesis text]

The full prompt is tokenized and processed in a single forward pass, producing next-token logits at every position. Only positions corresponding to the hypothesis span are used in scoring: the background and research question are identical across all sixteen candidates for a given paper, so including their token-level predictions would add shared noise without aiding discrimination among candidates. The hypothesis span is isolated by separately tokenizing the fixed prefix (everything up to and including the literal string “Hypothesis:”) and using its token length as an offset into the full tokenized sequence. Token positions at or beyond this offset constitute the hypothesis span H and contribute to the score; positions preceding it are discarded. This ensures that the resulting score reflects the model’s assessment of the hypothesis text alone, with the conditioning context held fixed.

## C. Logit-Based Energy Scores

Let $x = ( x _ { 1 } , \ldots , x _ { n } ) $ denote the tokenized prompt, with the hypothesis span occupying the final positions $t \in { \mathcal { H } } =$ $\{ s , \ldots , n \}$ , where s is the offset determined in Section III-B. Let $z _ { t } ~ \in ~ \mathbb { R } ^ { | V | }$ denote the model’s output logit vector at position $t \mathrm { ~ - ~ } 1 \mathrm { ~ } ( \mathrm { i . e . }$ ., the logits used to predict token $x _ { t } )$ where V is the model’s vocabulary. From $z _ { t }$ we derive two complementary token-level scores, one normalized and one unnormalized, and aggregate each over the hypothesis span into a single scalar per candidate.

1) Softmax Negative Log-Likelihood (NLL) Energy Score: The first score is the standard per-token cross-entropy between the model’s predictive distribution and the token observed at that position:

$$
\ell _ { t } = - \log \operatorname { s o f t m a x } ( z _ { t } ) [ x _ { t } ] = - \log { \frac { \exp ( z _ { t } [ x _ { t } ] ) } { \sum _ { v \in V } \exp ( z _ { t } [ v ] ) } } .\tag{1}
$$

$\ell _ { t }$ is small when the model’s normalized probability distribution concentrates mass on the observed token $x _ { t }$ , and large otherwise. Because $\ell _ { t }$ is computed after softmax normalization, it reflects the model’s calibrated probability estimate for the hypothesis text; a lower NLL indicates that the model’s predictive distribution assigns higher likelihood to the hypothesis given the context, which we interpret as evidence of plausibility.

2) Raw Target-Logit Energy Score: The second score follows the general framing of unnormalized log-probabilities as an energy function over the vocabulary. For each position in the hypothesis span, the energy is defined as the negated raw logit assigned to the observed token, prior to softmax normalization:

$$
E _ { t } ~ = ~ - z _ { t } [ x _ { t } ] .\tag{2}
$$

Unlike $\ell _ { t } , \ E _ { t }$ is not renormalized against the remainder of the vocabulary distribution and is therefore directly sensitive to the scale and calibration of the model’s output logits: two hypotheses with identical softmax probability can receive different energies if the underlying logit distribution differs in sharpness. This score is computed in addition to NLL to test whether ranking conclusions are an artifact of softmax normalization or hold under a complementary, unnormalized notion of confidence. For brevity, we refer to the softmax negative log-likelihood score as NLL and the raw target-logit energy score as Raw throughout the remainder of the paper, including in all tables and figures. Finally, note that the two criteria differ only in the partition term, which measures the model’s total confidence mass at that position independent of the observed token. NLL divides it out; Raw retains it.

```latex
Algorithm 1 Logit-Based Energy Scoring for One Paper
Require: background $b _ { p } ,$ research question $q _ { p } ,$ candidate set
$\mathsf { \bar { \mathcal { H } } } _ { p } = \{ h _ { p } ^ { ( 1 ) } , \dots , h _ { p } ^ { ( 1 6 ) } \}$ , model M
Ensure: rank $r _ { p }$ of the gold hypothesis under $S _ { \mathrm { N L L } }$ and $S _ { R a w }$
1: for each candidate $\boldsymbol { h } \in \mathcal { H } _ { p }$ do
prefix ← FORMATPROMPT(b<sub>p</sub>, q<sub>p</sub>) ▷ everything
through “Hypothesis:”
3: s ← |TOKENIZE(prefix)| ▷ hypothesis-span offset
4: x ← TOKENIZE(prefix ∥ h)
5: ${ \mathcal { H } } \gets \{ s , \ldots , | x | \}$ ▷ token positions belonging to h
6: $\{ \boldsymbol { z } _ { t } \} _ { t = 1 } ^ { | x | } \gets \mathrm { F o R w A R D } ( M , x )$ ▷ single forward pass,
teacher forcing
7: $\begin{array} { r } { S _ { \mathrm { N L L } } ( h ) \bar {  } \frac { 1 } { | \mathcal { H } | } \sum _ { t \in \mathcal { H } } } \end{array}$ − log softmax(z<sub>t</sub>)[x<sub>t</sub>]
8: $\begin{array} { r } { S _ { R a w } ( h ) \gets \frac { 1 } { | \mathcal { H } | } \sum _ { t \in \mathcal { H } } - z _ { t } [ x _ { t } ] } \end{array}$
9: end for
10: Rank $\mathcal { H } _ { p }$ ascending by $S _ { \mathrm { { N L L } } }$ (resp. $S _ { R a w } )$
11: $r _ { p } \gets$ rank position of the gold hypothesis $h _ { p } ^ { \star }$
12: return $r _ { p }$
```

3) Sequence-Level Aggregation: Both token-level scores are aggregated over the hypothesis span by averaging, yielding one scalar per candidate hypothesis h:

$$
S _ { \mathrm { N L L } } ( h ) \ = \ \frac { 1 } { | \mathcal { H } | } \sum _ { t \in \mathcal { H } } \ell _ { t } , \qquad S _ { R a w } ( h ) \ = \ \frac { 1 } { | \mathcal { H } | } \sum _ { t \in \mathcal { H } } E _ { t } .\tag{3}
$$

(We additionally report the unnormalized sum over H for both scores, to check sensitivity to hypothesis length; the mean in Eq. (3) is our primary score.) Both $S _ { \mathrm { { N L L } } }$ and $S _ { R a w }$ are loweris-better: for a given paper $p ,$ we rank all sixteen candidates in $\mathcal { H } _ { p }$ in ascending order of score and record the rank $r _ { p }$ of the gold hypothesis $h _ { p } ^ { \star }$ as the outcome of interest for that paper. A model that reliably assigns the gold hypothesis the lowest energy achieves $r _ { p } = 1 ;$ our top-line metric across the benchmark is the fraction of papers for which $r _ { p } = 1$

## D. Scoring Pipeline

Algorithm 1 summarizes the procedure applied to every (paper, candidate) pair. The prefix is tokenized separately from the full sequence (lines 3–5) to locate the hypothesis span, ensuring that the shared background and research question never contribute to the score for any candidate.

Scoring a candidate requires only a single forward pass, with no sampling and no additional model calls; the procedure is therefore deterministic given the model weights, and the cost of evaluating a paper scales linearly in the number of candidates (i.e., 16).

## E. Proprietary LLM-as-Judge

As a point of comparison, we evaluate a proprietary, instruction-tuned model accessible only through an API, for which token-level logits are not exposed and the logit-based scoring pipeline described above cannot be applied. For this baseline, we construct a single zero-shot prompt for each paper as shown below. We restrict the prompted model to identifying its top 5 candidates rather than producing a full ordering of all 16, reasoning that selecting the most plausible subset is a simpler and more tractable task for the model than exhaustively ranking the entire pool. This baseline represents standard practice in using a large commercial model as an explicit judge and enables a direct comparison between comparative judgment and intrinsic per-candidate confidence as signals for hypothesis evaluation. The following prompt template is used for proprietary LLMs.

![](images/2777a695ecf305c5b27f6254ec5bcb9f87722a891bc3f29f4fd4d0aa76233612.jpg)  
Fig. 2. Distribution of papers across disciplines in ResearchBench [12].

Prompt Template   
You are a scientific research assistant. Your task is to evaluate a set of   
hypotheses for a given research question and background, then identify   
the top 5 most plausible and well-supported hypotheses.   
## Background   
[background survey]   
## Research Question   
[research question]   
## Hypotheses   
[indexed list of all 16 candidate hypotheses]   
## Instructions   
- Carefully read each hypothesis.   
- Rank all 16 hypotheses from most to least plausible given the   
background and research question.   
- Return ONLY a JSON object in this exact format, with no extra text:   
{”ranking”: [rank-1 index, rank-2 index, rank-3 index, rank-4 index,   
rank-5 index]}   
The indices must be integers from 0 to 15. Return only the top 5.

## IV. EXPERIMENTAL SETUP

## A. Dataset

We used ResearchBench [12] which is constructed from 1,323 papers across 12 disciplines (Figure 2). Each instance in this dataset corresponds to a single published paper and consists of: (i) a background survey, a passage summarizing the prior literature and context motivating the study; (ii) a research question posed by the paper; and (iii) a pool of 16 candidate hypotheses of which exactly one is the gold hypothesis. We build directly on this decomposition and formulate hypothesis identification as a within-paper ranking task. Given the background survey and research question, each LLM in our paper produces a complete ranking of the candidate hypotheses. A system’s performance is determined by the extent to which it ranks the gold hypothesis above the candidate hypotheses.

TABLE I  
MODELS EVALUATED WITH KNOWLEDGE CUTOFF DATE, PARAMETER COUNTS, RELEASE DATES, AND LICENSE TERMS.
<table><tr><td>Model</td><td>Cutoff Date</td><td>Params</td><td>Release Date (License)</td></tr><tr><td>Llama 3.2 1B [13]</td><td>Dec 2023</td><td>1B</td><td>Sep 2024 (CL)</td></tr><tr><td>Llama 3.2 3B [14]</td><td>Dec 2023</td><td>3B</td><td>Sep 2024 (CL)</td></tr><tr><td>Gemma 2 2B [15]</td><td>April 2024</td><td>2B</td><td>Jun 2024 (ToU)</td></tr><tr><td>Gemma 4 12B [16]</td><td>Jan 2025</td><td>12B</td><td>Jun 2026 (Apache 2.0)</td></tr><tr><td>Mistral 7B [17]</td><td>Sep 2023</td><td>7B</td><td>Sep 2023 (Apache 2.0)</td></tr><tr><td>Phi-4 [18]</td><td>June 2024</td><td>14B</td><td>Dec 2024 (MIT)</td></tr><tr><td>GPT OSS 20B [19]</td><td>June 2024</td><td>20B</td><td>Aug 2025 (Apache 2.0)</td></tr><tr><td>GPT-5 [20]</td><td>Azure API</td><td>N/A</td><td>Aug 2025 (Service Terms)</td></tr></table>

## B. Compute Setup

We report results for 8 language models, ranging from roughly one to twenty billion parameters (Table I). Each model is scored under both the NLL and Raw energy criteria described in Section III-C. GPT-5 (proprietary) was evaluated under the zero-shot listwise prompting protocol. All results below are computed over the same pool of 1,323 papers. Open-weight models were run on a shared compute node with NVIDIA A100-SXM4 GPUs (80 GB memory each), two AMD EPYC 7742 64-core processors, and 2 TB of system memory, running Ubuntu 22.04.5. We used Python 3.12.13, PyTorch 2.6.0 with CUDA 12.4, and Transformers 5.12.1. The proprietary model was accessed through the Azure API and used no local GPUs.

## C. Evaluation Metrics

For every paper and every LLM, we obtain the rank $r _ { p }$ of the gold hypothesis within the full sixteen-candidate pool. All metrics below are computed over the resulting distribution of gold-hypothesis ranks across the P = 1,323 papers in the benchmark.

• Hit@k. The fraction of papers for which the gold hypothesis is ranked at or above position k, for $k \in \{ 1 , 2 , 3 , 5 \}$

$$
{ \mathrm { H i t @ } } k = { \frac { 1 } { P } } \sum _ { p = 1 } ^ { P } \mathcal { H } [ r _ { p } \leq k ]\tag{4}
$$

Hit@1 corresponds to the gold hypothesis being identified as the single most plausible candidate.

• Mean Reciprocal Rank (MRR). The mean, across papers, of the reciprocal of the gold hypothesis’s rank:

$$
\mathrm { M R R } = \frac { 1 } { P } \sum _ { p = 1 } ^ { P } \frac { 1 } { r _ { p } }\tag{5}
$$

![](images/136ac1d0acf9199ef790c71592a183a24e00da4212a44a3e2c683d2ec094d3ad.jpg)  
Fig. 3. Hit@k by scoring paradigm and open-weight model. Each panel shows Hit@k (gold hypothesis ranked in the top k of 16 candidates) for $k \in \{ 1 , 2 , 3 , 5 \}$ , across N = 1,323 papers. Bars compare NLL and energybased scoring per model; the dashed line marks the proprietary model under zero-shot prompted ranking. All panels share the same y-axis scale.

## V. RESULTS

## A. Overall Hypothesis-Identification Performance

Table II reports Hit@k and MRR for every model under every scoring paradigm, aggregated across all papers and disciplines. Reported values are means of per-hypothesis, with 95% percentile confidence intervals from 2,000 bootstrap resamples.

Two patterns stand out. First, the raw target-logit energy criterion is not uniformly better or worse than NLL: it substantially improves ranking performance for the three small open-weight models (Llama 3.2 1B, Llama 3.2 3B, Mistral 7B) relative to NLL, but degrades performance for Gemma 2 2B and leaves GPT OSS 20B, Gemma 4 12B, and Phi-4 roughly comparable to their NLL-based scores. Gemma 2 applies tanh-based soft-capping to its final logits [15], compressing precisely the raw logit magnitude that $S _ { R a w }$ reads while leaving the normalized quantity $S _ { N L L }$ reads intact. Gemma 4 12B, which does not use the same capping, recovers to 0.263 under the Raw criterion. We offer this explanation as a mechanistic conjecture.

The two criteria are indistinguishable (e.g., Hit@1 0.3315 vs. 0.3276; MRR 0.4742 vs. 0.4622), a gap well inside the ±0.026 margin of a single Hit@1 estimate at $\Nu = 1 , 3 2 3$ . The variance across models is far larger than the variance across scorers. The highest single cell is Llama 3.2 1B under the Raw criterion (Hit@1 = 0.5314 [0.504, 0.558]), but as the maximum of 14 model-by-scorer combinations chosen after seeing the results, this value should be read as an upper bound on what a well-matched pairing can achieve rather than as an unbiased estimate of that pairing’s performance.

Second, GPT-5 under zero-shot listwise prompting performs markedly worse than nearly every open-weight likelihoodbased configuration: its Hit@1 of 0.1663 and Hit@5 of 0.4172 fall below every NLL-scored model and every Raw-scored model except Gemma 2 2B under the energy criterion. In other words, asking a large proprietary model to explicitly rank all 16 candidate hypotheses in a single pass is, in aggregate, a less reliable indicator of the true hypothesis than measuring how probable a much smaller open-weight model already finds the hypothesis text to be, without any explicit ranking instruction.

TABLE II  
OVERALL HYPOTHESIS-RANKING PERFORMANCE, AGGREGATED ACROSS ALL 1,323 PAPERS. GPT-5 REPORTS TOP-5 RANKS AND MRR IS OMITTED (SEE SECTION V-C).
<table><tr><td rowspan=1 colspan=11>Model          Hit@1                    Hit@2             Hit@3             Hit@5              MRR</td></tr><tr><td rowspan=1 colspan=11>Softmax NLL Energy Score</td></tr><tr><td rowspan=2 colspan=1>GPT OSS 20B</td><td rowspan=2 colspan=2>0.278[0.254, 0.303]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.398[0</td><td rowspan=1 colspan=1>.372, 0.423]</td><td rowspan=1 colspan=1>0.482[0</td><td rowspan=1 colspan=1>.457, 0.509]</td><td rowspan=1 colspan=2>0.614[0.589, 0.640]</td><td rowspan=1 colspan=1>0.437[0</td><td rowspan=1 colspan=1>.418, 0.458]</td></tr><tr><td rowspan=1 colspan=1>Gemma 2 2B</td><td rowspan=1 colspan=1>0.352</td><td rowspan=1 colspan=1>[0.328, 0.378]</td><td rowspan=1 colspan=1>0.463</td><td rowspan=1 colspan=1>[0.437, 0.489]</td><td rowspan=1 colspan=1>0.537</td><td rowspan=1 colspan=1>[0.510, 0.565]</td><td rowspan=1 colspan=2>0.640[0.615, 0.666]</td><td rowspan=1 colspan=1>0.493</td><td rowspan=1 colspan=1>[0.474, 0.515]</td></tr><tr><td rowspan=1 colspan=1>Gemma 4 12B</td><td rowspan=1 colspan=1>0.346</td><td rowspan=1 colspan=1>[0.321, 0.371]</td><td rowspan=1 colspan=1>0.459</td><td rowspan=1 colspan=1>[0.432, 0.485]</td><td rowspan=1 colspan=1>0.521</td><td rowspan=1 colspan=1>[0.494, 0.548]</td><td rowspan=1 colspan=2>0.636[0.610, 0.661]</td><td rowspan=1 colspan=1>0.487</td><td rowspan=1 colspan=1>[0.466, 0.508]</td></tr><tr><td rowspan=1 colspan=1>Llama 3.2 1B</td><td rowspan=1 colspan=1>0.322</td><td rowspan=1 colspan=1>[0.299, 0.347]</td><td rowspan=1 colspan=1>0.429[</td><td rowspan=1 colspan=1>0.402, 0.456]</td><td rowspan=1 colspan=1>0.494</td><td rowspan=1 colspan=1>[0.467, 0.521]</td><td rowspan=1 colspan=2>0.587[0.561, 0.613]</td><td rowspan=1 colspan=1>0.460</td><td rowspan=1 colspan=1>[0.439, 0.481]</td></tr><tr><td rowspan=1 colspan=1>Llama 3.2 3B</td><td rowspan=1 colspan=1>0.368</td><td rowspan=1 colspan=1>[0.342, 0.395]</td><td rowspan=1 colspan=1>0.475</td><td rowspan=1 colspan=1>[0.448, 0.501]</td><td rowspan=1 colspan=1>0.546</td><td rowspan=1 colspan=1>[0.520, 0.574]</td><td rowspan=1 colspan=1>0.649</td><td rowspan=1 colspan=1>[0.624, 0.675]</td><td rowspan=1 colspan=1>0.506</td><td rowspan=1 colspan=1>[0.486, 0.528]</td></tr><tr><td rowspan=1 colspan=1>Mistral 7B</td><td rowspan=1 colspan=1>0.327</td><td rowspan=1 colspan=1>[0.302, 0.353]</td><td rowspan=1 colspan=1>0.426</td><td rowspan=1 colspan=1>[0.401, 0.454]</td><td rowspan=1 colspan=1>0.492</td><td rowspan=1 colspan=1>[0.466, 0.520]</td><td rowspan=1 colspan=1>0.602</td><td rowspan=1 colspan=1>[0.577, 0.628]</td><td rowspan=1 colspan=1>0.464</td><td rowspan=1 colspan=1>[0.444, 0.486]</td></tr><tr><td rowspan=1 colspan=1>Phi-4</td><td rowspan=1 colspan=1>0.327</td><td rowspan=1 colspan=1>[0.302, 0.352]</td><td rowspan=1 colspan=1>0.439</td><td rowspan=1 colspan=1>[0.414, 0.466]</td><td rowspan=1 colspan=1>0.506</td><td rowspan=1 colspan=1>[0.479, 0.533]</td><td rowspan=1 colspan=1>0.621</td><td rowspan=1 colspan=1>[0.596, 0.646]</td><td rowspan=1 colspan=1>0.471</td><td rowspan=1 colspan=1>[0.451, 0.492]</td></tr><tr><td rowspan=1 colspan=1>All models</td><td rowspan=1 colspan=1>0.331</td><td rowspan=1 colspan=1>[0.312, 0.352]</td><td rowspan=1 colspan=1>0.441[</td><td rowspan=1 colspan=1>0.420, 0.463]</td><td rowspan=1 colspan=1>0.511</td><td rowspan=1 colspan=1>[0.489, 0.533]</td><td rowspan=1 colspan=1>0.621[</td><td rowspan=1 colspan=1>0.601, 0.642]</td><td rowspan=1 colspan=1>0.474</td><td rowspan=1 colspan=1>[0.457, 0.492]</td></tr><tr><td rowspan=1 colspan=11>Raw Target Logit Energy Score</td></tr><tr><td rowspan=2 colspan=1>GPT OSS 20B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.249</td><td rowspan=1 colspan=1>[0.227, 0.272]</td><td rowspan=1 colspan=1>0.367[</td><td rowspan=1 colspan=1>0.342, 0.394]</td><td rowspan=1 colspan=1>0.435</td><td rowspan=1 colspan=1>[0.409, 0.462]</td><td rowspan=1 colspan=1>0.550</td><td rowspan=1 colspan=1>[0.522, 0.577]</td><td rowspan=1 colspan=1>0.403</td><td rowspan=1 colspan=1>[0.384, 0.423]</td></tr><tr><td rowspan=1 colspan=1>Gemma 2 2B</td><td rowspan=1 colspan=1>0.135</td><td rowspan=1 colspan=1>[0.116, 0.153]</td><td rowspan=1 colspan=1>0.192[</td><td rowspan=1 colspan=1>0.172, 0.212]</td><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>[0.228, 0.272]</td><td rowspan=1 colspan=1>0.336[</td><td rowspan=1 colspan=1>0.312, 0.361]</td><td rowspan=1 colspan=1>0.260</td><td rowspan=1 colspan=1>[0.244, 0.276]</td></tr><tr><td rowspan=1 colspan=1>Gemma 4 12B</td><td rowspan=1 colspan=1>0.263</td><td rowspan=1 colspan=1>[0.239, 0.286]</td><td rowspan=1 colspan=1>0.357[</td><td rowspan=1 colspan=1>0.331, 0.380]</td><td rowspan=1 colspan=1>0.420</td><td rowspan=1 colspan=1>[0.393, 0.446]</td><td rowspan=1 colspan=1>0.531</td><td rowspan=1 colspan=1>[0.504, 0.557]</td><td rowspan=1 colspan=1>0.402</td><td rowspan=1 colspan=1>[0.382, 0.422]</td></tr><tr><td rowspan=1 colspan=1>Llama 3.2 1B</td><td rowspan=1 colspan=1>0.531</td><td rowspan=1 colspan=1>[0.505, 0.558]</td><td rowspan=1 colspan=1>0.620</td><td rowspan=1 colspan=1>[0.594, 0.646]</td><td rowspan=1 colspan=1>0.679</td><td rowspan=1 colspan=1>[0.654, 0.704]</td><td rowspan=1 colspan=2>0.774[0.751, 0.797]</td><td rowspan=1 colspan=1>0.641</td><td rowspan=1 colspan=1>[0.620, 0.662]</td></tr><tr><td rowspan=1 colspan=1>Llama 3.2 3B</td><td rowspan=1 colspan=1>0.458</td><td rowspan=1 colspan=1>[0.432, 0.485]</td><td rowspan=1 colspan=1>0.568</td><td rowspan=1 colspan=1>[0.544, 0.594]</td><td rowspan=1 colspan=1>0.639[</td><td rowspan=1 colspan=1>0.614, 0.664]</td><td rowspan=1 colspan=1>0.732[</td><td rowspan=1 colspan=1>0.709, 0.757]</td><td rowspan=1 colspan=1>0.586</td><td rowspan=1 colspan=1>[0.566, 0.608]</td></tr><tr><td rowspan=1 colspan=1>Mistral 7B</td><td rowspan=1 colspan=1>0.411</td><td rowspan=1 colspan=1>[0.386, 0.439]</td><td rowspan=1 colspan=1>0.506</td><td rowspan=1 colspan=1>[0.480, 0.534]</td><td rowspan=1 colspan=1>0.576</td><td rowspan=1 colspan=1>[0.549, 0.603]</td><td rowspan=1 colspan=1>0.676[</td><td rowspan=1 colspan=1>0.651, 0.702]</td><td rowspan=1 colspan=1>0.538</td><td rowspan=1 colspan=1>[0.517, 0.560]</td></tr><tr><td rowspan=1 colspan=1>Phi-4</td><td rowspan=1 colspan=1>0.246[</td><td rowspan=1 colspan=1>0.224, 0.271]</td><td rowspan=1 colspan=1>0.367</td><td rowspan=1 colspan=1>[0.341, 0.393]</td><td rowspan=1 colspan=1>0.447</td><td rowspan=1 colspan=1>[0.422, 0.474]</td><td rowspan=1 colspan=1>0.559[</td><td rowspan=1 colspan=1>0.534, 0.586]</td><td rowspan=1 colspan=1>0.405</td><td rowspan=1 colspan=1>[0.386, 0.425]</td></tr><tr><td rowspan=1 colspan=1>All models</td><td rowspan=1 colspan=1>0.328</td><td rowspan=1 colspan=1>[0.314, 0.342]</td><td rowspan=1 colspan=1>0.425</td><td rowspan=1 colspan=1>[0.411, 0.441]</td><td rowspan=1 colspan=1>0.492</td><td rowspan=1 colspan=1>[0.477, 0.507]</td><td rowspan=1 colspan=2>0.594[0.580, 0.608]</td><td rowspan=1 colspan=1>0.462</td><td rowspan=1 colspan=1>[0.450, 0.475]</td></tr><tr><td rowspan=1 colspan=11>Prompted</td></tr><tr><td rowspan=1 colspan=11>GPT-5          0.166 [0.146, 0.186]  0.232[0.209, 0.255]  0.280[0.254, 0.304]  0.417[0.389, 0.443]</td></tr></table>

We highlight that ResearchBench [12] used published papers (in 2024 or later). Hence, memorization is a natural concern. However, in our experiments, the strongest results come from Llama 3.2 1B and 3B, whose training data ends in December 2023, well before these papers were published. While the models with later cutoffs perform worse. This does not rule out memorization, but the pattern runs opposite to what it would predict.

## B. Per-Discipline Performance

Table III breaks down Hit@1 and Hit@5 by discipline for each of the three scoring paradigms (NLL and Raw pooled across all seven open-weight models, and GPT-5 alone). The best-performing paradigm for each discipline and metric is shown in bold.

Likelihood-based scoring (NLL or Raw) yields the best Hit@1 and Hit@5 in the large majority of disciplines. GPT-5 is nominally best in Physics and Material Science, though per-discipline N (approximately 110 to 125) makes these differences statistically indistinguishable (Table III). We note this pattern as a hypothesis for future work rather than a finding.

TABLE III  
HIT@1 AND HIT@5 BY DISCIPLINE.
<table><tr><td rowspan=1 colspan=7>Hit@1                   Hit@5Discipline           NLL   Raw   GPT-5  NLL   Raw   GPT-5</td></tr><tr><td rowspan=1 colspan=1>Astronomy</td><td rowspan=1 colspan=1>0.308</td><td rowspan=1 colspan=1>0.326</td><td rowspan=1 colspan=1>0.195</td><td rowspan=1 colspan=1>0.678</td><td rowspan=1 colspan=1>0.610</td><td rowspan=1 colspan=1>0.488</td></tr><tr><td rowspan=1 colspan=1>Biology</td><td rowspan=1 colspan=1>0.379</td><td rowspan=1 colspan=1>0.382</td><td rowspan=1 colspan=1>0.243</td><td rowspan=1 colspan=1>0.626</td><td rowspan=1 colspan=1>0.628</td><td rowspan=1 colspan=1>0.449</td></tr><tr><td rowspan=1 colspan=1>Business</td><td rowspan=1 colspan=1>0.439</td><td rowspan=1 colspan=1>0.427</td><td rowspan=1 colspan=1>0.130</td><td rowspan=1 colspan=1>0.742</td><td rowspan=1 colspan=1>0.702</td><td rowspan=1 colspan=1>0.370</td></tr><tr><td rowspan=1 colspan=1>Cell Biology</td><td rowspan=1 colspan=1>0.376</td><td rowspan=1 colspan=1>0.371</td><td rowspan=1 colspan=1>0.180</td><td rowspan=1 colspan=1>0.666</td><td rowspan=1 colspan=1>0.650</td><td rowspan=1 colspan=1>0.400</td></tr><tr><td rowspan=1 colspan=1>Chemistry</td><td rowspan=1 colspan=1>0.345</td><td rowspan=1 colspan=1>0.332</td><td rowspan=1 colspan=1>0.065</td><td rowspan=1 colspan=1>0.622</td><td rowspan=1 colspan=1>0.569</td><td rowspan=1 colspan=1>0.315</td></tr><tr><td rowspan=1 colspan=1>Earth Science</td><td rowspan=1 colspan=1>0.421</td><td rowspan=1 colspan=1>0.413</td><td rowspan=1 colspan=1>0.046</td><td rowspan=1 colspan=1>0.720</td><td rowspan=1 colspan=1>0.664</td><td rowspan=1 colspan=1>0.185</td></tr><tr><td rowspan=1 colspan=1>Energy Science</td><td rowspan=1 colspan=1>0.264</td><td rowspan=1 colspan=1>0.228</td><td rowspan=1 colspan=1>0.246</td><td rowspan=1 colspan=1>0.524</td><td rowspan=1 colspan=1>0.493</td><td rowspan=1 colspan=1>0.553</td></tr><tr><td rowspan=1 colspan=1>Environmental Science</td><td rowspan=1 colspan=1>0.381</td><td rowspan=1 colspan=1>0.394</td><td rowspan=1 colspan=1>0.081</td><td rowspan=1 colspan=1>0.740</td><td rowspan=1 colspan=1>0.678</td><td rowspan=1 colspan=1>0.324</td></tr><tr><td rowspan=1 colspan=1>Law</td><td rowspan=1 colspan=1>0.298</td><td rowspan=1 colspan=1>0.328</td><td rowspan=1 colspan=1>0.190</td><td rowspan=1 colspan=1>0.568</td><td rowspan=1 colspan=1>0.603</td><td rowspan=1 colspan=1>0.368</td></tr><tr><td rowspan=1 colspan=1>Material Science</td><td rowspan=1 colspan=1>0.188</td><td rowspan=1 colspan=1>0.196</td><td rowspan=1 colspan=1>0.212</td><td rowspan=1 colspan=1>0.435</td><td rowspan=1 colspan=1>0.441</td><td rowspan=1 colspan=1>0.575</td></tr><tr><td rowspan=1 colspan=1>Math</td><td rowspan=1 colspan=1>0.371</td><td rowspan=1 colspan=1>0.368</td><td rowspan=1 colspan=1>0.176</td><td rowspan=1 colspan=1>0.646</td><td rowspan=1 colspan=1>0.633</td><td rowspan=1 colspan=1>0.422</td></tr><tr><td rowspan=1 colspan=1>Physics</td><td rowspan=1 colspan=1>0.210</td><td rowspan=1 colspan=1>0.184</td><td rowspan=1 colspan=1>0.224</td><td rowspan=1 colspan=1>0.507</td><td rowspan=1 colspan=1>0.475</td><td rowspan=1 colspan=1>0.544</td></tr></table>

## C. Pairwise Comparison of Correct & Competing Hypotheses

Table IV reports how reliably each method separates a paper’s actual hypothesis from the alternatives it is presented alongside. For every paper we take the correct hypothesis and compare it against each competing one in turn, and record the share of these comparisons in which the correct hypothesis is placed higher; a method that ordered hypotheses at random would score 50%. Likelihood-based scoring is stable across models, clustering between 69% and 74% regardless of scale, which suggests the signal it captures is a general property of the models rather than an artifact of any one of them. Raw target-logit energy-based scoring is more variable but reaches the strongest results overall, with Llama 3.2 1B at 82.2% and Llama 3.2 3B at 79.1% outperforming every likelihood-based configuration; the exception is Gemma 2 2B at 45.8%, which is indistinguishable from chance and indicates that the Raw signal is not informative for that model. Listwise prompting of GPT-5 performs worst at 37.4%, falling below what random ordering would achieve.

TABLE IV  
PERCENTAGE OF HEAD-TO-HEAD (PAIRWISE) COMPARISONS IN WHICH THE CORRECT HYPOTHESIS OUTRANKS A CANDIDATE.
<table><tr><td>Model</td><td>Gold beats candidate</td></tr><tr><td>Softmax NLL Energy Score</td><td></td></tr><tr><td>GPT OSS 20B</td><td>70.9%</td></tr><tr><td>Gemma  $2 ~ 2 \mathrm { B }$ </td><td>73.2%</td></tr><tr><td>Gemma 4 12B</td><td>72.7%</td></tr><tr><td>Llama  $3 . 2 \ 1 \mathrm { B }$ </td><td>68.8%</td></tr><tr><td>Llama  $3 . 2 \ 3 \mathrm { B }$ </td><td>74.3%</td></tr><tr><td>Mistral 7B</td><td>69.8%</td></tr><tr><td>Phi-4</td><td>71.7%</td></tr><tr><td>Raw Target Logit Energy Score</td><td></td></tr><tr><td>GPT OSS 20B</td><td>66.7%</td></tr><tr><td>Gemma 2 2B</td><td>45.8%</td></tr><tr><td>Gemma 4 12B</td><td>63.7%</td></tr><tr><td>Llama 3.2 1B</td><td>82.2%</td></tr><tr><td>Llama 3.2 3B</td><td>79.1%</td></tr><tr><td>Mistral 7B</td><td>74.7%</td></tr><tr><td>Phi-4</td><td>67.3%</td></tr><tr><td>GPT-5</td><td>37.4%</td></tr></table>

## D. Cross-Model Agreement

Across the seven open-weight models, all seven agree in ranking the gold hypothesis first for 167 of 1,323 papers (12.6%) under NLL scoring, but only 3 of 1,323 papers (0.2%) under Raw scoring. This indicates that while NLLbased rankings are of similar aggregate quality to (slightly ahead in seven of twelve disciplines and slightly behind in the remainder) raw target logit-based rankings, they are considerably more consistent with one another across models, whereas Raw-based rankings vary more sharply from model to model despite comparable average performance.

One further point is needed to correctly interpret Hit@k for the prompted model. Because GPT-5 is asked only to produce an explicit top-5 ranking, all candidates excluded from the top 5 are, by construction, placed in a single unranked tied group; under the fixed tie-breaking convention used to place these candidates on a numerical scale, the gold hypothesis, when excluded from the top 5, is always assigned the same fixed rank rather than a range of ranks reflecting a graded preference. Consistent with this, the empirical rank distribution for GPT-5 has non-zero mass only at ranks 1 through 5 (16.6%, 6.6%, 4.8%, 5.7%, and 8.0% of papers, respectively) and at the single terminal rank, which accounts for the remaining 58.3% of papers. For this reason, Table II reports only Hit@k for $k \leq$

5 for GPT-5 and omits MRR, which would conflate “ranked just outside the top $5 ^ { \circ }$ with “ranked as the least plausible candidate of all $1 6 . { } ^ { , }$ Hit@5 for GPT-5 is therefore equivalent to its raw top-5 inclusion rate (41.7%, 552 of 1,323 papers) and is the most directly interpretable summary statistic for the prompt-based paradigm.

## VI. DISCUSSION

Across all three scoring paradigms tested, a consistent picture emerges. Likelihood-based scoring of open-weight models, whether measured through NLL or through the Raw criterion, substantially outperforms zero-shot listwise prompting of a proprietary flagship model at identifying the correct scientific hypothesis from a pool of sixteen candidates. The strongest single configuration in our results is not the largest or most capable model: it is a one-billion-parameter open-weight model scored under the Raw criterion, which outperforms nearly every other model, scoring function, and paradigm we tested, including a substantially larger proprietary model that was explicitly instructed to rank the candidates. At the same time, the advantage of likelihood-based scoring is not uniform across disciplines: prompted ranking is nominally best in Physics and Material Science and competitive in Energy Science. With roughly 110 to 125 papers per discipline, however, these gaps are within sampling noise, and we report the pattern as a hypothesis worth testing at larger per-discipline N rather than as a finding. And while Raw target-logit energy scoring achieves the best raw performance of any configuration, it is also considerably less consistent across models than NLLbased scoring, with far fewer papers on which every model agrees that the gold hypothesis is the top choice.

## A. Mechanistic Signal May Outperform Explicit Prompting

The most interesting result of this study is the qualitative contrast it points to: a mechanistic signal, obtained without ever asking a model to reason about or articulate a preference, can rival or exceed the performance of directly instructing a much larger model to do exactly that. NLL and Raw energy scores are computed purely from the model’s own nexttoken distribution over the hypothesis text, conditioned on the background and research question; the model is never told that a ranking task is taking place, never asked to compare candidates against one another, and never required to produce a well-formed structured response. Prompted ranking, by contrast, asks the model to explicitly read all sixteen candidates, reason about their relative merit, and report a preference in a specific format. This is a strictly harder and more indirect pathway to the same judgment: it depends on the model’s ability to hold sixteen candidates in context simultaneously, to follow the ranking instruction faithfully, and to externalize a judgment that may or may not track its own internal estimate of plausibility.

Our results align with prior work showing that a model’s stated judgments do not always reflect its internal probabilities. Similar gaps have been found for bias: a model’s outputs and internal reasoning can show different bias signals [21]. If the same is true here, then for tasks like ours, choosing between a specific hypothesis and plausible alternatives, it may be better to use the model’s likelihoods directly rather than asking it to explain its own uncertainty. We believe this is the key novel aspect of our results: unlike prior work on this benchmark, which compares prompted models, we compare models against their own internal likelihoods.

The fact that the smallest model in our study performs best overall is worth noting, although we are cautious about drawing strong conclusions from it. This suggests that a model’s usefulness for this task does not simply increase with size. Other factors may matter, such as how the model was trained, what data it saw, or how confidently it assigns probabilities to different answers. We did not measure these factors directly, so we cannot say which one explains the result. We see this as an open question for future work.

## B. Limitations

This work is an initial step rather than a definitive evaluation of energy-based scoring, prompted ranking, or their relative strengths. Several limitations should be considered when interpreting the results. Our benchmark is built from published hypotheses and the authors curated the candidate hypothesis after the given paper was published. As a result, the task is closer to identifying a known hypothesis than evaluating truly open-ended scientific reasoning. In addition, gold hypotheses were written by domain experts and may simply be more fluent than the distractors. This raises the possibility that likelihoodbased methods capture writing quality as well as scientific plausibility, which our current design cannot fully separate. Relatedly, the low cross-model agreement under energy scoring leaves open whether the models share a common signal tracking scientific correctness or whether each one is picking up different surface cues that happen to favor the gold hypothesis. Our evaluation is also limited in scope. We tested prompted ranking with only one proprietary model and one zero-shot prompting strategy, so different prompting methods or models could produce different results. Likewise, the open-weight models differ in size, training data, tokenizer, and alignment, making them useful for comparison but not for isolating the cause of performance differences. These limitations mean our findings should be viewed as evidence from one experimental setting rather than a general conclusion about the strengths of energy-based scoring or prompted ranking.

## C. Future Work

Our next goal is to evaluate hypothesis ranking on genuinely novel scientific hypotheses. We plan to collaborate with experts across scientific fields to build a benchmark of open research questions paired with unpublished candidate hypotheses [22,23]. Experts will independently rank these hypotheses based on scientific judgment, enabling comparison with energy-based and prompt-based methods. Their rankings will also allow us to measure inter-annotator agreement and derive a robust reference ranking. This setting requires a different evaluation protocol: because open research questions lack a single agreed-upon answer, evaluation should focus on agreement between model and expert rankings (e.g., Kendall’s τ or Spearman’s $\rho )$ rather than retrieval metrics like accuracy/Hit@k.

## VII. CONCLUSION

We show that scoring candidate hypotheses by a language model’s own likelihood, particularly an energy-based criterion, matches or exceeds explicitly prompting a much larger proprietary model to rank them in aggregate, with perdiscipline differences too small at our sample sizes to support paradigm-specific claims. This suggests mechanistic scoring signals deserve more attention relative to prompting-based approaches. Because our benchmark is built from alreadypublished hypotheses with known ground truth, we cannot rule out that likelihood-based scores partly reflect textual familiarity rather than scientific reasoning. We therefore propose evaluating these methods against domain-expert rankings of genuinely novel hypotheses as the critical next step.

## ACKNOWLEDGMENTS

This work was primarily supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research and Office of Basic Energy Sciences, Scientific Discovery through Advanced Computing (SciDAC) program under the FORUM-AI project. Swati Rajwal gratefully acknowledges support through the Graduate Research at Oak Ridge National Laboratory (GRO) program. Sanjay Das is sponsored by the Office of the Laboratory Director, Oak Ridge National Laboratory’s Operational Excellence Initiatives, which is supported by the United States Department of Energy (DOE)’s Office of Science under Contract No. DE-AC05-00OR22725.

## DATA AND CODE AVAILABILITY

The data used in this study are publicly available from a previously published study [12]. The code and associated artifacts will be released upon acceptance.

## REFERENCES

[1] L. Shi, C. Ma, W. Liang, X. Diao, W. Ma, and S. Vosoughi, “Judging the judges: A systematic study of position bias in LLM-as-a-judge,” in Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics. Mumbai, India: The Asian Federation of Natural Language Processing and The Association for Computational Linguistics, Dec. 2025, pp. 292–314. [Online]. Available: https://aclanthology.org/2025.ijcnlp-long.18/

[2] K. Saito, A. Wachi, K. Wataoka, and Y. Akimoto, “Verbosity bias in preference labeling by large language models,” 2023. [Online]. Available: https://arxiv.org/abs/2310.10076

[3] K. Wataoka, T. Takahashi, and R. Ri, “Self-preference bias in llm-asa-judge,” 2025. [Online]. Available: https://arxiv.org/abs/2410.21819

[4] J. Gottweis, W.-H. Weng, A. Daryin, T. Tu, P. Sirkovic, A. Myaskovsky, G. Glowaty, F. Weissenberger, A. Orlandi, D. Popovici et al., “Accelerating scientific discovery with co-scientist,” Nature, pp. 1–3, 2026. [Online]. Available: https://doi.org/10.1038/s41586-026-10644-y

[5] C. Lu, C. Lu, R. T. Lange, Y. Yamada, S. Hu, J. Foerster, D. Ha, and J. Clune, “Towards end-to-end automation of ai research,” Nature, vol. 651, no. 8107, pp. 914–919, 2026. [Online]. Available: https://doi.org/10.1038/s41586-026-10265-5

[6] L. Zheng, W.-L. Chiang, Y. Sheng et al., “Judging llm-asa-judge with mt-bench and chatbot arena,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 46 595–46 623. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets and Benchmarks.pdf

[7] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan, Eds. Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 3982–3992. [Online]. Available: https://aclanthology.org/D19-1410/

[8] S. Kadavath, T. Conerly, A. Askell et al., “Language models (mostly) know what they know,” 2022. [Online]. Available: https: //arxiv.org/abs/2207.05221

[9] Y. LeCun, S. Chopra, R. Hadsell, M. Ranzato, and F. J. Huang, “A tutorial on energy-based learning,” in Predicting Structured Data, G. Bakır, T. Hofmann, B. Scholkopf, A. Smola, and¨ B. Taskar, Eds. MIT Press, 2006. [Online]. Available: https: //cs.nyu.edu/<sup>∼</sup>sumit/publications/assets/ebmtutorial.pdf

[10] W. Grathwohl, K.-C. Wang, J.-H. Jacobsen, D. Duvenaud, M. Norouzi, and K. Swersky, “Your classifier is secretly an energy based model and you should treat it like one,” in International Conference on Learning Representations, 2020. [Online]. Available: https: //openreview.net/forum?id=Hkxzx0NtDB

[11] W. Liu, X. Wang, J. D. Owens, and Y. Li, “Energy-based out-ofdistribution detection,” in Proceedings of the 34th International Conference on Neural Information Processing Systems, ser. NIPS ’20. Red Hook, NY, USA: Curran Associates Inc., 2020.

[12] Y. Liu, Z. Yang, T. Xie, J. Ni, B. Gao, Y. Li, S. Tang, W. Ouyang, E. Cambria, and D. Zhou, “Researchbench: Benchmarking llms in scientific discovery via inspiration-based task decomposition,” in Findings of the Association for Computational Linguistics:

ACL 2026, 2026, pp. 13 187–13 207. [Online]. Available: https: //doi.org/10.18653/v1/2026.findings-acl.644

[13] Meta, “Llama-3.2-1b-instruct,” https://huggingface.co/meta-llama/ Llama-3.2-1B-Instruct, 2024, model card, accessed August 3, 2026.

[14] Meta, “meta-llama/llama-3.2-3b,” https://huggingface.co/meta-llama/ Llama-3.2-3B, 2024, model card, accessed August 3, 2026.

[15] G. Team, M. Riviere, S. Pathak et al., “Gemma 2: Improving open language models at a practical size,” 2024. [Online]. Available: https://arxiv.org/abs/2408.00118

[16] G. Team, S. E. Abd, V. Aggarwal et al., “Gemma 4 technical report,” 2026. [Online]. Available: https://arxiv.org/abs/2607.02770

[17] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed, “Mistral 7b,” 2023. [Online]. Available: https://arxiv.org/abs/2310.06825

[18] M. Abdin, J. Aneja, H. Behl et al., “Phi-4 technical report,” 2024. [Online]. Available: https://arxiv.org/abs/2412.08905

[19] OpenAI, :, S. Agarwal, L. Ahmad et al., “gpt-oss-120b & gpt-oss-20b model card,” 2025. [Online]. Available: https://arxiv.org/abs/2508.10925

[20] A. Singh, A. Fry, A. Perelman et al., “Openai gpt-5 system card,” 2026. [Online]. Available: https://arxiv.org/abs/2601.03267

[21] S. Rajwal, S. Garg, R. Abdel-Salam, and A. Zayed, “Do biased models have biased thoughts?” 2025. [Online]. Available: https: //arxiv.org/abs/2508.06671

[22] C. Si, D. Yang, and T. Hashimoto, “Can llms generate novel research ideas? a large-scale human study with 100+ nlp researchers,” in International Conference on Learning Representations, Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, Eds., vol. 2025, 2025, pp. 94 003– 94 092. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/ 2025/file/ea94957d81b1c1caf87ef5319fa6b467-Paper-Conference.pdf

[23] C. Si, T. Hashimoto, and D. Yang, “The ideation-execution gap: Execution outcomes of LLM-generated versus human research ideas,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=Fllp8l6Puy