# DARS: Dual-Level Credit Assi<sub>g</sub>nment RL with Structured Reasonin<sub>g</sub> for Instruction-Based Ima<sub>g</sub>e Editin<sub>g</sub>

Haoxiang Cao<sup>1,2,§</sup> Jiajiong Cao<sup>2</sup> Xuanpu Zhang<sup>2,§</sup> Changqian Yu<sup>2,‡</sup> Chaoqun Wang<sup>1,†</sup>

<sup>1</sup>South China Normal University <sup>2</sup>KlingAI Research

![](images/56b9fe3bcce4fca60c26769abd92047811d44ebd6946feacd11fc9c86fb331af.jpg)  
Fi<sub>g</sub>ure 1: Two visuall<sub>y p</sub>oor editin<sub>g</sub> outcomes can re<sub>q</sub>uire o<sub>pp</sub>osite cross-module u<sub>p</sub>date em<sub>p</sub>hases. Case A: the <sub>p</sub>lanner ca<sub>p</sub>tures th<sub>e reques</sub>t<sub>e</sub>d <sub>e</sub>dit <sub>reasona</sub>bl<sub>y we</sub>ll<sub>,</sub> b<sub>u</sub>t th<sub>e ren</sub>d<sub>erer</sub> f<sub>a</sub>il<sub>s</sub> t<sub>o rea</sub>li<sub>ze</sub> it<sub>, so</sub> thi<sub>s samp</sub>l<sub>e s</sub>h<sub>ou</sub>ld <sub>p</sub>l<sub>ace more correc</sub>ti<sub>ve up</sub>d<sub>a</sub>t<sub>e we</sub>i<sub>g</sub>ht <sub>on</sub> th<sub>e ren</sub>d<sub>erer.</sub> C<sub>ase</sub> B<sub>:</sub> th<sub>e samp</sub>l<sub>e</sub>d <sub>p</sub>l<sub>an</sub> i<sub>s a</sub> l<sub>ess use</sub>f<sub>u</sub>l i<sub>n</sub>t<sub>erme</sub>di<sub>a</sub>t<sub>e</sub> f<sub>or</sub> th<sub>e</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on, an</sub>d th<sub>e ren</sub>d<sub>erer</sub> f<sub>a</sub>ithf<sub>u</sub>ll<sub>y execu</sub>t<sub>es</sub> it<sub>,</sub> so this sam<sub>p</sub>le should <sub>p</sub>lace more corrective u<sub>p</sub>date wei<sub>g</sub>ht on the <sub>p</sub>lanner. DARS is desi<sub>g</sub>ned to distin<sub>g</sub>uish render-dominant f<sub>rom p</sub>l<sub>an-</sub>d<sub>om</sub>i<sub>nan</sub>t <sub>up</sub>d<sub>a</sub>t<sub>e nee</sub>d<sub>s</sub> th<sub>roug</sub>h <sub>rou</sub>ti<sub>ng s</sub>i<sub>gna</sub>l<sub>s; w</sub>h<sub>en p</sub>l<sub>anner-s</sub>id<sub>e emp</sub>h<sub>as</sub>i<sub>s</sub> i<sub>s nee</sub>d<sub>e</sub>d<sub>,</sub> it<sub>s s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>s</sub>l<sub>o</sub>t<sub>s suppor</sub>t finer within-<sub>p</sub>lanner dia<sub>g</sub>nosis.

## Ab<sub>s</sub>t<sub>rac</sub>t

Instruction-based image editing uses a planner-renderer pipeline: a vision-language model (VLM) first converts the instruction into an edit plan, and a difusion model then executes that plan. Training such systems with only final-image rewards is ineficient because a poor edit does not reveal whether additional optimization should place more emphasis on the planner or the renderer, and even planner-dominant cases remain dificult to localize within a freeform reasoning trace. We present DARS, a reinforcement learning framework for dual-level credit assignment in this two-stage setting. Across modules, multi-plan multi-render rollouts estimate between plan and within-plan reward variability for soft module routing, while mean rewards across rollouts provide hardness estimates for an adaptive curriculum. Within the planner, a four-field structured reasoning output enables a prefix-gated reward and token-level advantage reweighting, turning outcome-level feedback into localized supervision. Experiments on five benchmarks show that DARS outperforms a Joint RL baseline with the same backbone, data, reward model, and rollout budget, with the largest gains on reasoning-intensive edits.

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Many image editing systems [15, 16, 24, 31, 38] adopt a plannerrenderer pipeline: a vision-language model (VLM) [1, 25, 29] interprets the instruction and produces an edit plan, and a difusion model [13, 18, 21, 33] renders the image conditioned on that plan. This decomposition improves controllability, but also turns joint optimization into a dual-level credit-assignment problem. Across modules, as illustrated in Figure 1, a poor image does not reveal whether the current sample would benefit more from planner-side or renderer-side corrective emphasis. Within the planner, even when planner-side updates are the more useful direction, the same outcome-level feedback still does not reveal which part of the plan should be corrected.

The dificulty comes from the fact that the two modules fail in different ways. The planner is an autoregressive language model [26], so its errors are typically semantic or compositional. The renderer is a difusion-based generator [9], so its failures appear as visual artifacts, preservation damage, or incomplete execution. A single post-render score [20, 34] collapses these qualitatively diferent errors into the same feedback, making credit assignment ambiguous. Existing RL [11, 16, 24, 32] strategies, therefore, either optimize one module at a time, leaving the other bottleneck untouched, or update both modules uniformly from the same reward. In either case, a low-scoring edit still does not indicate whether additional optimization should focus more on planning or on rendering.

This makes credit assignment [10] under-specified at two levels. First, diferent samples should not contribute equally to planner and renderer updates. Second, even when the planner is the main bottleneck, outcome-level feedback still does not say which part of the plan is wrong: the requested modification, the preservation constraint, the scene-level objective, or the execution hint.

To address this dual-level credit-assignment problem, we present DARS, a joint-RL framework for planner-renderer image editing. We sample multiple plans and multiple renderings per plan, then use the resulting reward statistics in two diferent ways: between plan and within-plan variability determine how much extra update weight is routed to the planner or renderer, while mean rewards across rollouts provide a hardness estimate for adaptive curriculum scheduling [2]. On the planner side, we replace an opaque freeform trace with a four-field structured reasoning output (Modify, Preserve, Overall, Tips), so the same rendered outcomes can support within-planner credit assignment through slot-level rewards and token-level advantage reweighting (Figure 2).

In summary, our contributions are:

• We formulate joint RL for planner-renderer image editing as a dual-level credit-assignment problem, spanning cross-module update allocation and within-planner reasoning diagnosis.

• We propose a rollout variance decomposition that separates across-plan from within-plan variability for cross-module rout ing, together with a rollout-reward hardness estimate for adaptive curriculum scheduling.

• We introduce a four-field structured reasoning output for the planner together with a prefix-gated reward and token-level advantage reweighting, enabling localized planner optimization from post-render feedback.

Experiments on five benchmarks show that DARS is especially efective on reasoning-intensive edits [35, 40], and ablations indicate that cross-module routing and structured planner supervision contribute complementary gains.

## 2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

Instruction-based image editing. Instruction-based image editing [3, 8, 13, 14, 21, 22, 33] aims to modify a source image according to a natural-language instruction while preserving irrelevant content. Recent work increasingly augments editing with explicit reasoning or planning. RePlan [24] introduces region-aware planning for complex edits, while ReasonEdit [38] and UniReason 1.0 [30] strengthen editing with richer semantic and world-knowledge reasoning. Step1X-Edit-v1p2 [38], ThinkRL-Edit [15], and Edit-Thinker [16] further show that multi-round reflection can improve editing fidelity, and ThinkGen [11] demonstrates the benefit of explicit intermediate reasoning in a coupled reasoning-generation pipeline. Taken together, these methods establish the value of introducing a planner-like stage before generation. Their main emphasis, however, is on designing stronger intermediate reasoning or refine ment procedures. Our focus is diferent: we study how to optimize an already existing planner-renderer pipeline once a structured intermediate stage has been introduced.

RL for two-stage planning-based editing. Reinforcement learning has recently been used to improve instruction faithfulness and multimodal alignment in image generation [17, 36, 41, 42] and editing. PromptRL [31] studies reward-based optimization for flow-based generation. In two-stage editing pipelines, existing RL methods difer mainly in which module is optimized: ThinkRL-Edit [15] focuses on the generation side by tuning the difusion model and text encoder under reflective plans, EditThinker [16] concentrates optimization on the VLM reasoning module, and ThinkGen [11] alternates updates between the reasoning model and the generator. Once editing is organized as an explicit prompt-rewriting, planning, or reasoning stage followed by visual execution, the optimization problem becomes inherently two-stage: one module produces an intermediate textual plan, and another realizes it in the image space. In this setting, existing RL strategies largely follow three patterns: (i) stage-wise or alternating optimization, which separates planner and renderer updates; (ii) single-module optimization, which freezes one component and tunes the other; and (iii) uniform joint optimization, which updates both modules from the same outcomelevel reward. The first two reduce the opportunity for coordinated co-improvement, while the third leaves the learning signal underspecified because a low final score does not reveal which module should receive more optimization emphasis. Our method is most closely related to the third line of work, but difers in explicitly addressing this module-allocation problem through rollout-derived routing while retaining joint training.

Curriculum learning and structured reasoning. Curriculum learning [4, 5] improves optimization by ordering or weighting training samples according to dificulty. Standard curriculum designs, however, treat hardness as a one-dimensional property of each sample. In two-stage image editing, this abstraction is incomplete: two samples with similar dificulty may still difer in whether their uncertainty is primarily plan-dominant or render-dominant, suggesting that sample scheduling alone is insuficient without moduleaware weighting. On the reasoning side, PromptEnhancer [32], ThinkGen [11], and EditThinker [16] show the value of richer intermediate text, but free-form reasoning traces remain dificult to evaluate locally after rendering. They often depend on engineered cold-start formats or expert-designed prompting recipes [12], which makes them harder to scale and optimize consistently under RL. Our method occupies a diferent point in this design space: instead of relying on long free-form traces, we use a compact four-field schema that remains close to the planner’s native generation format while exposing semantically aligned slots for post-render reward decomposition and token-level credit assignment.

## 3 P<sub>re</sub>li<sub>m</sub>i<sub>nary</sub>

## 3.1 Problem Setu<sub>p</sub>

The input is $\mathbf { x } = \left( I _ { \mathrm { s r c } } , c \right)$ , where $I _ { \mathrm { s r c } }$ is the source image and � is the editing instruction. A VLM planner �<sub>�</sub> (e | x) with parameters � predicts a structured edit plan

$$
\mathbf { e } = \left( e ^ { \mathrm { m o d } } , e ^ { \mathrm { p r e } } , e ^ { \mathrm { o v r } } , e ^ { \mathrm { t i p } } \right) ,\tag{1}
$$

specifying modifications, preservation constraints, the scene objective, and execution hints. A conditional image generator (renderer)

![](images/8b3c3d9e6eaccd84d78a9e976e8d3876068d5133bb985796c92cce8361e284fb.jpg)  
Fi <sub>ure</sub> 2<sub>:</sub> F<sub>ree-</sub>f<sub>orm vs. s</sub>t<sub>ruc</sub>t<sub>ure</sub>d l<sub>anner ou</sub>t <sub>u</sub>t<sub>.</sub> With <sub>a</sub> f<sub>ree-</sub>f<sub>orm</sub> l<sub>anner</sub> t<sub>race, os</sub>t<sub>-ren</sub>d<sub>er</sub> f<sub>ee</sub>db<sub>ac</sub>k <sub>rema</sub>i<sub>ns a</sub>tt<sub>ac</sub>h<sub>e</sub>d t<sub>o a</sub> single opaque text sequence, making within-planner credit assignment dificult. Our four-field structured answer (Modify, Preserve, Overall, Tips) exposes semantically distinct slots, so the same rendered rollouts can be diagnosed field by field and converted into slot-wise <sub>p</sub>lanner credit si<sub>g</sub>nals.

$p _ { \phi } ( \mathbf { y } \mid I _ { \mathrm { s r c } } , \mathbf { e } )$ with parameters $\phi$ produces the image y. The renderer receives the source image and the structured plan; the raw instruction � is not separately passed to it. For reward computation, Gemini 3 Pro is used once before RL training to generate and cache a slot-aligned checklist $\mathbf { C } ( \mathbf { x } ) = \left( C ^ { \mathrm { m o d } } , C ^ { \mathrm { p r e } } , C ^ { \mathrm { o v r } } , C ^ { \mathrm { t i p } } \right)$ from the source image and raw instruction. During RL training, Qwen3- VL-32B performs all online planner-side and renderer-side Yes/No scoring for the sampled rollouts. Each checklist field enumerates verifiable requirements for the corresponding planner slot.

The reward model produces two module-targeted scores from each rendered outcome. The renderer reward $R _ { \mathrm { r e n d } } ( \mathbf { x } , \mathbf { e } , \mathbf { y } ; \mathbf { C } )$ measures checklist satisfaction by the edited image. The planner reward $R _ { \mathrm { p l a n } } ( \mathbf { x } , \mathbf { e } , \mathbf { y } ; \mathbf { C } )$ evaluates whether each structured slot is semantically appropriate and supported by the realized edit, so plan quality is assessed in the execution context. A shared reward

$$
R _ { \mathrm { s h a r e } } = \eta _ { \mathrm { p l a n } } R _ { \mathrm { p l a n } } + \eta _ { \mathrm { r e n d } } R _ { \mathrm { r e n d } }\tag{2}
$$

uses fixed mixing weights $\eta _ { \mathrm { p l a n } } , \eta _ { \mathrm { r e n d } } \geq 0$ for routing and curriculum estimation, while $R _ { \mathrm { p l a n } }$ and $R _ { \mathrm { r e n d } }$ remain the optimization targets. All in-house ablations use the same reward model, so their diferences reflect the proposed credit-assignment and planner-structure choices under a fixed reward pipeline. The central challenge is credit assignment: for a given sample, (i) which update path should receive a stronger learning signal, and (ii) which parts ofthe planner output, if any, should be corrected.

## 3.2 Polic<sub>y</sub> O<sub>p</sub>timization Back<sub>g</sub>round

GRPO [28] samples � candidate outputs per input and uses grouprelative advantages with clipped surrogate and KL regularization. Flow-GRPO [17] extends this to difusion-flow generators [6, 19] by defining the policy ratio at each denoising step. Both are used of-the-shelf; our contribution lies in the rollout-derived signals that weight their updates.

## 4 M<sub>e</sub>th<sub>o</sub>d

DARS addresses credit assignment at two levels in two-stage image editing pipelines: across modules and within the planner. The framework contains three coupled components:

(1) Structured reasoning and slot-wise planner credit assignment (Section 4.1): the planner outputs a four-field answer enabling a prefix-gated reward and finer diagnostics;

(2) Rollout variance decomposition for cross-module credit assignment (Section 4.2): multi-plan and multi-render rollouts produce variance-based statistics that distinguish across-plan variability from within-plan rollout variability;

(3) Adaptive curriculum and soft module routing (Section 4.3): a rollout-derived hardness score controls sample weighting, while decomposed variability guides how extra update weight is assigned across modules.

The underlying principle is to separate three questions that end-toend editing RL conflates: which module should receive extra corrective emphasis (cross-module credit assignment via module routing), how planner errors should be diagnosed more finely (structured reasoning with slot-wise planner credit assignment), and which samples are currently most informative (adaptive curriculum). Figure 3 provides an overview.

## 4<sub>.</sub>1 Str<sub>uc</sub>t<sub>u</sub>r<sub>e</sub>d R<sub>easo</sub>nin<sub>g a</sub>nd Sl<sub>o</sub>t-Wi<sub>se</sub> Pl<sub>a</sub>nn<sub>e</sub>r Cr<sub>e</sub>dit A<sub>ss</sub>i<sub>g</sub>nm<sub>e</sub>nt

Structured reasoning output. Open-ended planner reasoning is dificult to optimize under sparse editing rewards because feedback arrives only after image generation. Following Section 3, we represent the planner output as a four-field structured answer $\mathbf { e } \overset { \cdot } { = } \left( e ^ { \mathrm { m o d } } , e ^ { \mathrm { p r e } } , e ^ { \mathrm { o v r } } , e ^ { \mathrm { t i p } } \right)$ . Each field has a task-specific role: �<sup>mod</sup> (Modify) states what should change; $e ^ { \mathrm { p r e } }$ (Preserve) states what must remain unchanged; $e ^ { \mathrm { { o v r } } }$ (Overall) captures scene-level coherence; $e ^ { \mathrm { t i p } } \ ( T i p s )$ provides short execution hints for the renderer. Each field remains free-form natural language, preserving the VLM backbone’s expressiveness while providing semantic anchoring for reward decomposition. For every training input, a slot-aligned checklist C(x) serves as the shared evaluation anchor for both module rewards.

![](images/16c016f79f35ba31ee3e134e1500a984c990f6aa5603b6ff80b2df0c58862f08.jpg)  
Fi<sub>gure</sub> 3<sub>:</sub> F<sub>ramewor</sub>k <sub>o</sub>f DARS<sub>.</sub> Th<sub>e s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>p</sub>l<sub>anner pro</sub>d<sub>uces a</sub> f<sub>our-</sub>fi<sub>e</sub>ld <sub>answer; mu</sub>lti<sub>p</sub>l<sub>e p</sub>l<sub>ans an</sub>d <sub>ren</sub>d<sub>er</sub>i<sub>ngs per p</sub>l<sub>an are</sub> <sub>samp</sub>l<sub>e</sub>d<sub>.</sub> M<sub>ean rewar</sub>d<sub>s across ro</sub>ll<sub>ou</sub>t<sub>s y</sub>i<sub>e</sub>ld h<sub>ar</sub>d<sub>ness es</sub>ti<sub>ma</sub>t<sub>es</sub> f<sub>or a</sub>d<sub>ap</sub>ti<sub>ve curr</sub>i<sub>cu</sub>l<sub>um sc</sub>h<sub>e</sub>d<sub>u</sub>li<sub>ng, w</sub>hil<sub>e rewar</sub>d <sub>var</sub>i<sub>a</sub>bilit<sub>y</sub> i<sub>s</sub> decom<sub>p</sub>osed into across-<sub>p</sub>lan and within-<sub>p</sub>lan com<sub>p</sub>onents to <sub>p</sub>roduce module-routin<sub>g</sub> si<sub>g</sub>nals for module-s<sub>p</sub>ecific wei<sub>g</sub>htin<sub>g</sub>. Prefix <sub>g</sub>atin<sub>g</sub> further <sub>p</sub>rovides slot-wise <sub>p</sub>lanner credit si<sub>g</sub>nals for finer within-<sub>p</sub>lanner dia<sub>g</sub>nosis.

Prefix-gated reward. With structured planner outputs, feedback can be decomposed at the slot level. For each field $j \in$ {mod, pre, ovr, the reward model inspects the slot text $e ^ { ( j ) }$ , the source image and instruction, the edited image y, and the corresponding checklist field $C ^ { ( j ) }$ . Encoding Yes as 1 and No as 0, the slot score $\bar { s ^ { ( j ) } } \in [ 0 , 1 ]$ is the unweighted fraction of Yes judgments over all questions in $C ^ { ( j ) }$ . It measures semantic adequacy, specificity, and whether the realized edit supports that field as executable guidance. These scores are planner-side post-render diagnostics: they assess plan quality through realized renderings, so a textually plausible but unexecutable plan does not receive high reward. Averaging each plan’s score over � independent renderings reduces renderer noise.

The four fields admit a natural causal ordering: modification intent $( e ^ { \mathrm { m o d } } )$ is fundamental; preservation constraints $( e ^ { \mathrm { p r e } } )$ are meaningful only if modification is correct; the overall objective $( e ^ { \mathrm { { o v r } } } )$ synthesizes both; tips $( e ^ { \mathrm { t i p } } )$ are useful only when preceding fields are sound. We encode this by prefix gating. For a realized plan–rendering pair $\left( \mathbf { e } _ { i } , \mathbf { y } _ { i , k } \right)$ , let $\sigma _ { i , k } ^ { \overline { { ( j ) } } } = \mathbb { I } \big [ s _ { i , k } ^ { \overline { { ( j ) } } } \geq \delta _ { j } \big ]$ , where $\delta _ { j }$ is the slot threshold and $\lambda _ { j } \geq 0$ weights slot �. The per-render planner reward is

$$
R _ { \mathrm { p l a n } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } ; \mathbf { C } ) = \sum _ { j = 1 } ^ { 4 } \lambda _ { j } \cdot s _ { i , k } ^ { ( j ) } \cdot \prod _ { l = 1 } ^ { j - 1 } \sigma _ { i , k } ^ { ( l ) } ,\tag{3}
$$

where slots are ordered as $j = 1$ :mod, 2:pre, 3:ovr, 4:tip. We use the empty-product convention $\Pi _ { l = 1 } ^ { 0 } ( \cdot ) = 1$ , so the Modify term is always ungated. Prefix gating activates later-slot rewards only after earlier slots are correct, encouraging the planner to fix fundamental mistakes first.

The renderer reward focuses on execution quality:

$$
R _ { \mathrm { r e n d } } ( \mathbf { x } , \mathbf { e } , \mathbf { y } ; \mathbf { C } ) = \sum _ { j = 1 } ^ { 4 } \gamma _ { j } u ^ { ( j ) } ,\tag{4}
$$

where $u ^ { ( j ) } \in [ 0 , 1 ]$ is analogously the unweighted fraction of renderer-side Yes judgments over all questions in $C ^ { ( j ) }$ , and $\gamma _ { j } \geq 0$ is the corresponding renderer-side weight. The shared reward is

$$
R _ { \mathrm { s h a r e } } ( \mathbf { x } , \mathbf { e } , \mathbf { y } ) = \eta _ { \mathrm { p l a n } } R _ { \mathrm { p l a n } } + \eta _ { \mathrm { r e n d } } R _ { \mathrm { r e n d } } .\tag{5}
$$

Token-level advantage reweighting. For each plan $\mathbf { e } _ { i }$ and rendering ${ \bf y } _ { i , k }$ , the reward model returns per-render slot scores $s _ { i , k } ^ { ( j ) }$ , which enter the per-render prefix reward $R _ { \mathrm { p l a n } }$ via Eq. 3. The plan-level reward $r _ { i } ^ { \mathrm { p l a n } }$ averages the resulting per-render rewards across � renderings. For token-level advantage modulation, the slot scores are likewise averaged: $\begin{array} { r } { \bar { s } _ { i } ^ { ( j ) } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \bar { s } _ { i , k } ^ { ( j ) } } \end{array}$ . Tokens in slot � then use

$$
\hat { A } _ { i , j } ^ { \mathrm { p l a n } } = \hat { A } _ { i } ^ { \mathrm { p l a n } } \cdot \prod _ { l = 1 } ^ { j - 1 } \bar { \sigma } _ { i } ^ { ( l ) } \cdot \left\{ \bar { s } _ { i } ^ { ( j ) } , \quad \hat { A } _ { i } ^ { \mathrm { p l a n } } \geq 0 , \right.\tag{6}
$$

where $\hat { A } _ { i } ^ { \mathrm { p l a n } }$ is the group-relative plan-level advantage and $\bar { \sigma } _ { i } ^ { ( l ) } =$ $\mathbb { I } [ \bar { s } _ { i } ^ { ( l ) } \geq \dot { \delta } _ { l } ]$ is an analogous prefix gate computed from renderingaveraged slot scores. It uses the same threshold form and causal ordering as Eq. 3, but is distinct from the per-render gates $\sigma _ { i , k } ^ { ( l ) }$ This sign-aware modulation prevents low-quality slots from inheriting the positive credit of an otherwise good plan, while assigning stronger negative updates to the slots most associated with poor overall plan performance. The prefix gate $\begin{array} { r } { \prod _ { l < j } \bar { \sigma } _ { i } ^ { ( l ) } } \end{array}$ ensures that downstream slots receive gradient only when all upstream slots pass their quality thresholds, maintaining the same causal dependency enforced in the reward (Eq. 3) through to the token-level update.

## 4<sub>.</sub>2 R<sub>o</sub>ll<sub>ou</sub>t V<sub>ar</sub>i<sub>ance</sub> D<sub>ecompos</sub>iti<sub>on</sub> f<sub>or</sub> Cr<sub>oss</sub>-M<sub>o</sub>d<sub>u</sub>l<sub>e</sub> Cr<sub>e</sub>dit A<sub>ss</sub>i<sub>g</sub>nm<sub>e</sub>nt

For a fixed input x, let $\textbf { e } \sim q _ { \theta } ( \cdot \mid \textbf { x } )$ and $\mathbf { y } \sim p _ { \phi } ( \cdot \mid I _ { \mathrm { s r c } } , \mathbf { e } )$ . We decompose the variance of $R _ { \mathrm { s h a r e } } { \mathrm { : } }$

$$
U _ { \mathrm { p l a n } } ( \mathbf { x } ) = \mathrm { V a r } _ { \mathbf { e } } \Big ( \mathbb { E } _ { \mathbf { y } | I _ { \mathrm { s r c } } , \mathbf { e } } [ R _ { \mathrm { s h a r e } } ] \Big ) ,\tag{7}
$$

$$
U _ { \mathrm { r e n d } } ( \mathbf { x } ) = \mathbb { E } _ { \mathbf { e } } \left[ \operatorname { V a r } _ { \mathbf { y } | I _ { \mathrm { s r c } } , \mathbf { e } } ( R _ { \mathrm { s h a r e } } ) \right] .\tag{8}
$$

By the law of total variance, $\mathrm { V a r } ( R _ { \mathrm { s h a r e } } \mid \mathbf { x } ) = U _ { \mathrm { p l a n } } ( \mathbf { x } ) + U _ { \mathrm { r e n d } } ( \mathbf { x } )$ The first term measures how much the expected reward changes across plans; the second measures reward fluctuation under a fixed plan. Large across-plan variability suggests that selecting or improving the plan could materially change the outcome; large within-plan variability suggests that outcome quality remains unstable even after the plan is fixed. Since the edited image varies across renderer rollouts and both $R _ { \mathrm { p l a n } }$ and $R _ { \mathrm { r e n d } }$ are derived from that realized image, within-plan variability is induced by renderer stochasticity but can propagate into both module-targeted scores. We use these quantities as rollout-level cross-module credit-assignment signals for where additional optimization is currently most actionable; accordingly, $U _ { \mathrm { p l a n } }$ and $U _ { \mathrm { r e n d } }$ are referred to as plan-dominant and render-dominant variability.

For each sample, we draw � plans and � renderings per plan, yielding rewards $r _ { i , k } ^ { \mathrm { s h a r e } }$ . The per-plan mean is $\begin{array} { r } { \bar { r } _ { i } = \frac { 1 } { K } \sum _ { k } \bar { r } _ { i , k } ^ { \mathrm { s h a r e } } } \end{array}$ and the grand mean is $\begin{array} { r } { \bar { r } = \frac { 1 } { M } \sum _ { i } \bar { r } _ { i } } \end{array}$ . We compute:

$$
s _ { \mathrm { w i t h i n } } ^ { 2 } ( \mathbf { x } ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { 1 } { K - 1 } \sum _ { k = 1 } ^ { K } ( r _ { i , k } ^ { \mathrm { s h a r e } } - \bar { r } _ { i } ) ^ { 2 } ,\tag{9}
$$

$$
s _ { \mathrm { b e t w e e n } } ^ { 2 } ( \mathbf { x } ) = \frac { 1 } { M - 1 } \sum _ { i = 1 } ^ { M } \bigl ( \bar { r } _ { i } - \bar { r } \bigr ) ^ { 2 } .\tag{10}
$$

The within-plan estimate is $\widehat { U } _ { \mathrm { r e n d } } = s _ { \mathrm { w i t h i n } } ^ { 2 }$ , and the across-plan estimate corrects for finite-� noise:

$$
\begin{array} { r } { \widehat { U } _ { \mathrm { p l a n } } ( \mathbf { x } ) = \operatorname* { m a x } \left( 0 , \ s _ { \mathrm { b e t w e e n } } ^ { 2 } - \frac { 1 } { K } s _ { \mathrm { w i t h i n } } ^ { 2 } \right) . } \end{array}\tag{11}
$$

The $M \times K$ rollouts are shared with policy optimization, so variance estimation requires no extra renderer budget.

## 4.3 Ada<sub>p</sub>tive Curriculum and Soft Module Routin<sub>g</sub>

Hardness determines when a sample should contribute strongly; rollout-variance signals determine where additional module emphasis should be assigned.

Curriculum scheduling. The rollout-based hardness estimate is $\begin{array} { r } { \widehat { H } ( \mathbf { x } ) = - \frac { 1 } { M K } \sum _ { i , k } r _ { i , k } ^ { \mathrm { s h a r e } } } \end{array}$ (larger = harder). After normalization, the resulting hardness $\widetilde { H } ( \mathbf { x } )$ is compared against a step-dependent boundary $\kappa ( n )$ that increases with training step � via the empirical quantile of �e in a running bufer. The curriculum weight is

$$
w _ { \mathrm { c u r } } ( { \bf x } , n ) = \mathrm { s i g m o i d } \left( \frac { \kappa ( n ) - \widetilde { \cal H } ( { \bf x } ) } { \tau _ { \mathrm { c } } } \right) .\tag{12}
$$

Here $\tau _ { \mathrm { c } }$ is a temperature parameter controlling how sharply the curriculum weight transitions around the current boundary �(�). Early in training, simpler samples receive large weights; as the model improves, �(�) rises to progressively expose harder samples.

Cross-module credit assignment. We use the relative magnitude of the variance signals:

$$
\alpha ( \mathbf { x } ) = \frac { \widehat { U } _ { \mathrm { p l a n } } + \varepsilon _ { U } } { \widehat { U } _ { \mathrm { p l a n } } + \widehat { U } _ { \mathrm { r e n d } } + 2 \varepsilon _ { U } } , \quad \omega ( \mathbf { x } ) = 1 - \alpha ( \mathbf { x } ) .\tag{13}
$$

Here $\varepsilon _ { U } ~ > ~ 0$ is the routing smoothing constant. This soft routing acts as a residual reweighting on top of the shared curriculum weight: every sample still updates both modules, while the variance signals control the extra module-specific weight. Curriculum determines when a sample receives larger weight; module routing determines where additional weight is placed. We validate these routing weights through both agreement with GPT-5 pseudo-labels (Section 5.5) and downstream ablation (Section 5.4).

How curriculum and credit assignment interact. The two mechanisms play complementary roles over time. Early in training, the curriculum boundary $\kappa ( n )$ is low, so easier samples receive most of the weight; for these samples, module routing fine-tunes which module gets emphasis. As training progresses and �(�) rises, harder samples with heterogeneous variance are admitted. Module routing then becomes more important, because these newly admitted samples do not benefit equally from additional updates to both modules. If a sample has high hardness but near-zero total variance, curriculum still controls its admission while module routing contributes little diferential emphasis; optimization is then driven mainly by the two task rewards $R _ { \mathrm { p l a n } }$ and $R _ { \mathrm { r e n d } }$

## 4.4 Joint Optimization

The planner and renderer operate in diferent output spaces and are optimized with diferent policy-gradient objectives: text-GRPO [28] for the planner and flow-GRPO [17] for the renderer. The key modification relative to standard GRPO is that each module’s objective is weighted by the curriculum factor $w _ { \mathrm { c u r } } ( \mathbf { x } , n )$ plus a residual modulerouting emphasis. Every sample contributes a base update to both modules, scaled by its curriculum weight; module routing only adds module-specific weight according to the surrogate decomposition.

For planner optimization, each plan e<sub>�</sub> receives a renderer-averaged reward $\begin{array} { r } { r _ { i } ^ { \mathrm { p l a n } } = \frac { 1 } { K } \sum _ { k } R _ { \mathrm { p l a n } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } ; \mathbf { C } ) } \end{array}$ , from which $\hat { A } _ { i } ^ { \mathrm { { p l a n } } }$ is computed. Tokens in slot � then use the sign-aware slot-weighted advantage from Eq. 6: positive plan-level advantages reinforce high-scoring slots, while negative plan-level advantages concentrate corrective updates on low-scoring slots. For renderer optimization, each rendered sample keeps its own reward $r _ { i , k } ^ { \mathrm { r e n d } } =$ $R _ { \mathrm { r e n d } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } ; \mathbf { C } )$ . The objectives are:

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { p l a n } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { \mathbf { x } } \bigg [ w _ { \mathrm { c u r } } \left( 1 + \rho \alpha \right) \mathcal { L } _ { \mathrm { G R P O } } ^ { \mathrm { p l a n } } ( \boldsymbol { \theta } ; \mathbf { x } ) \bigg ] , } \end{array}\tag{14}
$$

$$
\mathcal { T } _ { \mathrm { r e n d } } ( \phi ) = \mathbb { E } _ { \mathrm { x , e } } \Big [ w _ { \mathrm { c u r } } \big ( 1 + \rho \omega \big ) \mathcal { L } _ { \mathrm { f l o w - G R P O } } ^ { \mathrm { r e n d } } ( \phi ; \mathbf { x } , \mathbf { e } ) \Big ] ,\tag{15}
$$

where $\rho \ge 0$ controls the strength of module routing. $\mathcal { L } _ { \mathrm { G R P O } } ^ { \mathrm { p l a n } }$ is the standard clipped surrogate with KL regularization, applied at the token level using $\hat { A } _ { i , j } ^ { \mathrm { p l a n } }$ for tokens in slot �, and $\mathcal { L } _ { \mathrm { f l o w - G R P O } } ^ { \mathrm { r e n d } }$ uses per-render advantages from $\{ r _ { i , k } ^ { \mathrm { r e n d } } \}$ . The final objective is $\mathcal { T } = \mathcal { T } _ { \mathrm { p l a n } } + \mathcal { T } _ { \mathrm { r e n d } } .$ . The routing weights are estimated from $R _ { \mathrm { s h a r e } } ,$ while each module’s policy gradient uses its own reward $( R _ { \mathrm { p l a n } }$ or $R _ { \mathrm { r e n d } } )$ . Module routing, therefore, modulates how much each module is updated without changing what it learns. Full expanded expressions are in the supplementary.

Table 1: Main comparison across five benchmarks. All entries are oficial overall scores (higher is better; compare within columns). PICA-Bench reports “simple / detailed” prompt scores. We highlight Qwen-Image-Edit-2511-based methods separately and include a controlled Joint RL + Adaptive Curriculum baseline matching DARS in backbone, data, reward model, and rollout b<sub>u</sub>d<sub>ge</sub>t<sub>.</sub>
<table><tr><td>Method</td><td>KRIS-Bench</td><td>RISE-Bench</td><td>ImgEdit-Bench</td><td>GEdit-Bench-EN</td><td>PICA-Bench</td></tr><tr><td colspan="6">Other baselines</td></tr><tr><td>ThinkGen [11]</td><td>59.57</td><td>10.56</td><td>3.97</td><td>7.16</td><td>60.82/65.09</td></tr><tr><td>UniREdit [7]</td><td>61.02</td><td>13.33</td><td>3.70</td><td>6.47</td><td>60.40/66.55</td></tr><tr><td>PromptRL [31]</td><td>61.24</td><td>10.00</td><td>3.87</td><td>6.74</td><td>51.02/60.38</td></tr><tr><td>Step1X-Edit-v1p2 [38]</td><td>62.58</td><td>11.67</td><td>4.00</td><td>7.47</td><td>56.15/60.55</td></tr><tr><td>UniReason 1.0 [30]</td><td>63.26</td><td>15.00</td><td>4.06</td><td>6.52</td><td>57.30/61.62</td></tr><tr><td>ThinkRL-Edit [15]</td><td>66.04</td><td>14.44</td><td>4.19</td><td>6.48</td><td>55.13/55.84</td></tr><tr><td>PhysicEdit [39]</td><td>66.78</td><td>18.89</td><td>4.05</td><td>7.37</td><td>61.48/68.96</td></tr><tr><td>EditThinker [16]</td><td>68.80</td><td>18.33</td><td>4.31</td><td>7.59</td><td>59.64/67.19</td></tr><tr><td colspan="6">Qwen-Image-Edit-2511-based methods</td></tr><tr><td>Qwen-Image-Edit-2511 [33]</td><td>64.42</td><td>17.50</td><td>4.36</td><td>6.97</td><td>63.20/72.27</td></tr><tr><td>RePlan [24]</td><td>55.72</td><td>9.72</td><td>3.44</td><td>6.43</td><td>52.00/54.92</td></tr><tr><td>PromptEnhancerV2 [32]</td><td>71.54</td><td>18.90</td><td>4.18</td><td>7.59</td><td>61.90/70.77</td></tr><tr><td>Joint RL + Adpt. Curriculum (ctrl.)</td><td>72.15</td><td>25.70</td><td>4.20</td><td>7.83</td><td>63.55/72.22</td></tr><tr><td>DARS</td><td>80.72</td><td>27.50</td><td>4.39</td><td>7.86</td><td>64.19/72.75</td></tr></table>

## 5 Ex<sub>p</sub>eriments

## 5.1 Ex<sub>p</sub>erimental Setu<sub>p</sub>

Training. We train DARS on 10K examples (5K from the RL split of THINKEDIT-140K [16] and 5K from UniREdit-Data-100K [7]). The planner is initialized from Qwen3-VL-4B-Instruct, and the renderer from Qwen-Image-Edit-2511. We use �=4 plans and �=4 renderings per plan (� × � = 16 rollouts per input). The learning rate is $2 \times 1 0 ^ { - 6 }$ , the global batch size is 256, the KL penalty coefi cient [27] is 10<sup>−3</sup>, and the maximum image resolution is 1024×1024. The ofline checklist generator (Gemini 3 Pro) and the online reward judge (Qwen3-VL-32B) are fixed across DARS and all in-house ablations, so diferences reflect the proposed credit-assignment and planner-structure choices under a fixed reward pipeline. The controlled baseline and DARS use the same rollout budget on identical hardware; DARS adds lightweight rollout statistics and slotwise scoring on top of that shared budget. We evaluate on five complementary benchmarks spanning reasoning-intensive editing, general-purpose instruction following, preservation-sensitive localized editing, and physics-aware realism: KRIS-Bench [35], RISE-Bench [40], ImgEdit-Bench [37], GEdit-Bench-EN [18], and PICA-Bench [23]. All reported numbers are oficial overall scores, and the higher the number, the better. Benchmark results are obtained using each benchmark’s oficial evaluation code and are fully independent of the training reward model.

Baselines. We compare against 11 methods spanning multi-round reflection editors (ThinkRL-Edit [15], Step1X-Edit-v1p2 [38], and EditThinker [16]), reasoning- or planning-based editors (ThinkGen [11], RePlan [24], UniREdit [7], and UniReason 1.0 [30], which includes

a single reflection step), reward-optimized or prompt-enhancement methods (PromptRL [31] and PromptEnhancerV2 [32]), a physicsspecialized baseline (PhysicEdit [39]), and a strong end-to-end editor (Qwen-Image-Edit-2511 [33]). We further include a controlled Joint RL + Adaptive Curriculum baseline that matches DARS in backbone, data, reward model, rollout budget, and adaptive curriculum, while replacing DARS’s structured reasoning planner and dual-level credit assignment with free-form reasoning. Detailed architecture descriptions are in the supplementary.

## 5<sub>.</sub>2 M<sub>a</sub>in R<sub>esu</sub>lt<sub>s</sub>

Table 1 shows that the full DARS system attains the best score on all five benchmark regimes. Against previously reported methods, the gains are consistent across evaluation criteria. Against the in-house controlled baseline with matched backbone, data, reward model, rollout budget, and adaptive curriculum, DARS improves by +8.57 on KRIS-Bench, +1.80 on RISE-Bench, +0.19 on ImgEdit-Bench, +0.03 on GEdit-Bench-EN, and +0.64/+0.53 on PICA-Bench. The strongest gains appear on KRIS-Bench and RISE-Bench, consistent with DARS’s focus on planner-renderer credit assignment for reasoning-intensive edits.

## 5.3 Qualitative Results

Figure 4 compares DARS against reasoning-oriented editing methods. We include Qwen-Image-Edit-2511 as a baseline without explicit reasoning, together with instruction-rewriting or planning baselines: UniReason 1.0 based on the unified multimodal model Bagel, PromptRL as a VLM-difusion joint-RL method, EditThinker with up to five rounds of reflective prompt revision, RePlan with region-aware reasoning, PhysicEdit specialized by physics-grounded data tuning, and PromptEnhancerV2, a strong 32B prompt-enhancement model combining expert-designed rules with end-to-end RL. Across

![](images/eb5cdcc914f31484c7084a7216e7512bb0740b84512ef0c7bc5255f7e4c47de4.jpg)  
Figure 4: Qualitative comparison on reasoning-intensive editing tasks requiring temporal, spatial, logical, physical, and <sub>wor</sub>ld<sub>-</sub>k<sub>now</sub>l<sub>e</sub>d<sub>ge reason</sub>i<sub>ng, o</sub>ft<sub>en</sub> i<sub>n com</sub>bi<sub>na</sub>ti<sub>on.</sub> C<sub>ompare</sub>d <sub>w</sub>ith b<sub>ase</sub>li<sub>ne me</sub>th<sub>o</sub>d<sub>s,</sub> DARS <sub>ac</sub>hi<sub>eves</sub> th<sub>e s</sub>t<sub>ronges</sub>t <sub>overa</sub>ll <sub>per</sub>f<sub>ormance, pro</sub>d<sub>uc</sub>i<sub>ng more</sub> f<sub>a</sub>ithf<sub>u</sub>l <sub>e</sub>dit<sub>s w</sub>ith b<sub>e</sub>tt<sub>er preserva</sub>ti<sub>on an</sub>d <sub>s</sub>t<sub>ronger scene-</sub>l<sub>eve</sub>l <sub>cons</sub>i<sub>s</sub>t<sub>ency.</sub>

the Figure 4 examples, DARS delivers the strongest visual quality, achieving the best balance of instruction faithfulness, preservation, scene coherence, and reasoning consistency.

## 5<sub>.</sub>4 Abl<sub>a</sub>ti<sub>o</sub>n St<sub>u</sub>di<sub>es</sub>

Unless otherwise stated, ablations are reported on RISE-Bench and GEdit-Bench-EN. All variants use DARS without auxiliary heuristics. Free-form joint RL without a curriculum diverged, whereas the structured reasoning planner remained trainable without it, albeit with worse performance. The ablations below, therefore, use the structured reasoning planner unless explicitly labeled otherwise.

Curriculum and routing (Table 2). Adaptive curriculum consistently outperforms static and no-curriculum baselines, confirming that dynamic hardness re-estimation is critical as model competence evolves. Notably, No Curriculum still converges under the structured reasoning planner, whereas free-form joint RL without a curriculum diverges , suggesting that the four-field structure provides implicit training stabilization. For routing, soft continuous weighting outperforms both uniform and hard-threshold alternatives, consistent with mixed but asymmetric variance patterns in many editing failures. The configuration comparison shows that the full DARS improves over both the prompt-only baseline (23.80, no RL) and the controlled free-form Joint RL baseline (25.70), confirming gains over structured prompting alone and over a matched free-form joint-RL setup.

Table 2: Ablation on curriculum and routin<sub>g</sub> strate<sub>g</sub>ies.
<table><tr><td>Variant</td><td>RISE</td><td>GEdit</td></tr><tr><td>Curriculum</td><td></td><td></td></tr><tr><td>No Curriculum</td><td>18.89</td><td>6.87</td></tr><tr><td>Static Curriculum</td><td>24.56</td><td>7.15</td></tr><tr><td>Adaptive Curriculum</td><td>27.50</td><td>7.86</td></tr><tr><td>Routing</td><td></td><td></td></tr><tr><td>No Routing</td><td>20.05</td><td>7.12</td></tr><tr><td>Hard Routing</td><td>24.86</td><td>7.49</td></tr><tr><td>Soft Routing (Ours)</td><td>27.50</td><td>7.86</td></tr><tr><td>Configuration</td><td></td><td></td></tr><tr><td>Structured Prompt Only (No RL)</td><td>23.80</td><td>7.10</td></tr><tr><td>Free-form Joint  $\mathrm { R L } + \mathrm { A d p t } .$ </td><td>25.70</td><td>7.83</td></tr><tr><td>Full DARS</td><td>27.50</td><td>7.86</td></tr></table>

Planner structure and reward (Table 3). A monotonic trend runs from free-form reasoning to the full four-field answer, with each field providing gains: removing Preserve causes a drop on GEdit-Bench-EN, consistent with its emphasis on localized editing and non-target preservation; removing Overall produces a larger drop on the reasoning-heavy RISE-Bench, indicating that scene-level coherence is important for complex edits; removing Tips yields the smallest consistent degradation, suggesting that execution guidance is helpful once higher-level semantics are in place. These results support distinct slot roles and slot-wise advantage reweighting unavailable under free-form reasoning. For reward composition, the prefix-gated reward outperforms flat averaging and weighted summation. Flat averaging treats all fields as independent and equally important, failing to reflect the causal structure of editing plans. Weighted summation improves over flat averaging but underperforms because it does not enforce dependency between upstream and downstream fields. The prefix-gated reward yields the best results by ensuring that later fields are only rewarded once earlier fields are reliable, reducing the incentive to compensate for upstream errors by overproducing downstream text.

## 5.5 Validatin<sub>g</sub> the Uncertaint<sub>y</sub> Si<sub>g</sub>nals

The modeling claim of DARS is that rollout statistics provide informative routing signals for cross-module optimization. We validate whether the plan-dominant and render-dominant routing scores align with GPT-5 pseudo-labels of dominant failure patterns.

Protocol. For each evaluation case, we construct the same � × � rollout bank used by DARS and compute ${ \widehat { U } } _ { \mathrm { p l a n } } , { \widehat { U } } _ { \mathrm { r e n d } }$ , and � from that rollout bank. Separately, we sample five realized triplets from the same case and ask GPT-5 to assign each one a pseudo label among planner-dominant, renderer-dominant, or mixed; the majority vote becomes the case-level pseudo-label. GPT-5 sees only the source image, structured plan, and edited image for each triplet, not the rollout statistics or reward-model outputs used by DARS. Metrics are reported for pairwise one-vs-one discrimination tasks, with AUROC computed in the corresponding binary setting.

T<sub>a</sub>bl<sub>e</sub> 3<sub>:</sub> Abl<sub>a</sub>ti<sub>on on p</sub>l<sub>anner s</sub>t<sub>ruc</sub>t<sub>ure an</sub>d <sub>rewar</sub>d <sub>compos</sub>i<sub>-</sub> tion.
<table><tr><td>Variant</td><td>RISE</td><td>GEdit</td></tr><tr><td>Planner structure</td><td></td><td></td></tr><tr><td>Free-form Reasoning</td><td>23.94</td><td>6.98</td></tr><tr><td>Modify Only</td><td>25.21</td><td>7.11</td></tr><tr><td>Without Preserve</td><td>26.08</td><td>7.34</td></tr><tr><td>Without Overall</td><td>26.42</td><td>7.51</td></tr><tr><td>Without Tips</td><td>26.87</td><td>7.69</td></tr><tr><td>Full Structured Answer</td><td>27.50</td><td>7.86</td></tr><tr><td>Reward composition</td><td></td><td></td></tr><tr><td>Flat Average</td><td>24.92</td><td>7.28</td></tr><tr><td>Weighted Sum</td><td>26.31</td><td>7.57</td></tr><tr><td>Prefix-Gated (Ours)</td><td>27.50</td><td>7.86</td></tr></table>

Table 4: Routing-signal discrimination against majority-<sub>vo</sub>t<sub>e</sub>d GPT<sub>-</sub>5 <sub>pseu</sub>d<sub>o-</sub>l<sub>a</sub>b<sub>e</sub>l<sub>s.</sub>
<table><tr><td>Setting</td><td>Accuracy</td><td>Macro-F1</td><td>AUROC</td></tr><tr><td>Planner vs. Renderer</td><td>86.0</td><td>85.8</td><td>0.930</td></tr><tr><td>Planner vs. Mixed</td><td>84.7</td><td>84.1</td><td>0.918</td></tr><tr><td>Renderer vs. Mixed</td><td>85.3</td><td>84.9</td><td>0.923</td></tr></table>

Table 4 shows AUROC above 0.91 in all pairwise discriminations, indicating that the rollout statistics track GPT-5 pseudo-labels of dominant failure patterns well. Combined with the ablation result that soft routing outperforms both hard routing and no routing (Table 2), this provides evidence that the routing signal captures failure-mode diferences and improves update allocation.

## 6 Limitations

DARS requires � ×� rollouts per training example, so the total cost remains higher than single-path updates even though these rollouts are shared with policy optimization. The variance-based routing and post-render planner scores depend on the reward model, and planner-renderer interactions can be harder to disentangle when both modules fail simultaneously. Adapting the framework to video editing or multi-turn settings is left for future work.

## 7 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

We present DARS, which formulates joint RL in two-stage plannerrenderer image editing pipelines as a dual-level credit-assignment problem. Rollout variance decomposition provides routing signals for cross-module updates; hardness-derived scores drive adaptive curriculum scheduling; and a structured four-field planner with prefix gating enables slot-wise diagnostics. DARS attains the best results across five benchmarks, with the largest gains on reasoningintensive tasks.

## A<sub>c</sub>k<sub>now</sub>l<sub>e</sub>d<sub>gmen</sub>t<sub>s</sub>

This work was supported by KlingAI Research. We thank Shenhui Zhang, Jiahao Guo, and Bao Tang for their valuable discussions.

## R<sub>e</sub>f<sub>erences</sub>

[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Ming-Hsuan Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. 2025. Qwen2.5-VL Technical Report. CoRR abs/2502.13923 (2025).

[2] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. 2009. Curriculum learning. In Proceedings ofthe 26th Annual International Conference on Machine Learning (Montreal, Quebec, Canada) (ICML ’09). Association for Computing Machinery, New York, NY, USA, 41–48.

[3] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. 2023. InstructPix2Pix: Learning to Follow Image Editing Instructions. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE, 18392–18402.

[4] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, Nicu Sebe, and Mubarak Shah. 2025. Curriculum Direct Preference Optimization for Difu sion and Consistency Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025. Computer Vision Foundation / IEEE, 2824–2834.

[5] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, Nicu Sebe, and Mubarak Shah. 2026. Curriculum-DPO++: Direct Preference Optimization via Data and Model Curricula for Text-to-Image Generation. CoRR abs/2602.13055 (2026).

[6] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. 2024. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024 (Proceedings ofMachine Learning Research). PMLR / OpenReview.net, 12606– 12633.

[7] Feng Han, Yibin Wang, Chenglin Li, Zheming Liang, Dianyi Wang, Yang Jiao, Zhipeng Wei, Chao Gong, Cheng Jin, Jingjing Chen, and Jiaqi Wang. 2025. UniREditBench: A Unified Reasoning-based Image Editing Benchmark. CoRR abs/2511.01295 (2025).

[8] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2023. Prompt-to-Prompt Image Editing with Cross-Attention Control. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023.

[9] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Difusion Probabilistic Models. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, Decembe 6-12, 2020, virtual.

[10] Guochao Jiang, Wenfeng Feng, Guofeng Quan, Chuzhan Hao, Yuewei Zhang, Guohua Liu, and Hao Wang. 2025. VCRL: Variance-based Curriculum Reinforce ment Learning for Large Language Models. CoRR abs/2509.19803 (2025).

[11] Siyu Jiao, Yiheng Lin, Yujie Zhong, Qi She, Wei Zhou, Xiaohan Lan, Zilong Huang, Fei Yu, Yingchen Yu, Yunqing Zhao, Yao Zhao, and Yunchao Wei. 2025. ThinkGen: Generalized Thinking for Visual Generation. CoRR abs/2512.23568 (2025).

[12] Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. 2023. Decomposed Prompting: A Modular Approach for Solving Complex Tasks. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023.

[13] Black Forest Labs. 2025. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/ flux-2.

[14] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. 2025. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space. CoRR abs/2506.15742 (2025).

[15] Hengjia Li, Liming Jiang, Qing Yan, Yizhi Song, Hao Kang, Zichuan Liu, Xin Lu, Boxi Wu, and Deng Cai. 2026. ThinkRL-Edit: Thinking in Reinforcement Learning for Reasoning-Centric Image Editing. CoRR abs/2601.03467 (2026).

[16] Hongyu Li, Manyuan Zhang, Dian Zheng, Ziyu Guo, Yimeng Jia, Kaituo Feng, Hao Yu, Yexin Liu, Yan Feng, Peng Pei, Xunliang Cai, Linjiang Huang, Hongsheng Li, and Si Liu. 2025. EditThinker: Unlocking Iterative Reasoning for Any Image Editor. CoRR abs/2512.05965 (2025).

[17] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. 2025. Flow-GRPO: Training Flow Matching Models via Online RL. CoRR abs/2505.05470 (2025).

[18] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. 2025. Step1X-Edit: A Practical Framework for General Image Editing. CoRR abs/2504.17761 (2025).

[19] Xingchao Liu, Chengyue Gong, and Qiang Liu. 2023. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

[20] Xin Luo, Jiahao Wang, Chenyuan Wu, Shitao Xiao, Xiyan Jiang, Defu Lian, Jiajun Zhang, Dong Li, and Zheng Liu. 2025. EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling. CoRR abs/2509.23909 (2025).

[21] Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, Xunliang Cai, Yayong Guan, and Jie Hu. 2025. LongCat-Image Technical Report. CoRR abs/2512.07584 (2025).

[22] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2022. SDEdit: Guided Image Synthesis and Editing with Stochastic Diferential Equations. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022.

[23] Yuandong Pu, Le Zhuo, Songhao Han, Jinbo Xing, Kaiwen Zhu, Shuo Cao, Bin Fu, Si Liu, Hongsheng Li, Yu Qiao, Wenlong Zhang, Xi Chen, and Yihao Liu. 2025. PICABench: How Far Are We from Physically Realistic Image Editing? CoRR abs/2510.17681 (2025).

[24] Tianyuan Qu, Lei Ke, Xiaohang Zhan, Longxiang Tang, Yuqi Liu, Bohao Peng, Bei Yu, Dong Yu, and Jiaya Jia. 2025. RePlan: Reasoning-guided Region Planning for Complex Instruction-based Image Editing. CoRR abs/2512.16864 (2025).

[25] Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. https://qwen. ai/blog?id=qwen3.5

[26] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training. (2018).

[27] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal Policy Optimization Algorithms. CoRR abs/1707.06347 (2017)

[28] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. CoRR abs/2402.03300 (2024).

[29] Qwen Team. 2025. Qwen3-VL Technical Report. CoRR abs/2511.21631 (2025).

[30] Dianyi Wang, Chaofan Ma, Feng Han, Size Wu, Wei Song, Yibin Wang, Zhixiong Zhang, Tianhang Wang, Siyuan Wang, Zhongyu Wei, and Jiaqi Wang. 2026. UniReason 1.0: A Unified Reasoning Framework for World Knowledge Aligned Image Generation and Editing. CoRR abs/2602.02437 (2026).

[31] Fu-Yun Wang, Han Zhang, Michaël Gharbi, Hongsheng Li, and Taesung Park. 2026. PromptRL: Prompt Matters in RL for Flow-Based Image Generation. CoRR abs/2602.01382 (2026).

[32] Linqing Wang, Ximing Xing, Yiji Cheng, Zhiyuan Zhao, Donghao Li, Tiankai Hang, Jiale Tao, Qixun Wang, Ruihuang Li, Comi Chen, Xin Li, Mingrui Wu, Xinchi Deng, Shuyang Gu, Chunyu Wang, and Qinglin Lu. 2025. PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting. CoRR abs/2509.04545 (2025).

[33] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Shengming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. 2025. Qwen-Image Technical Report. CoRR abs/2508.02324 (2025).

[34] Keming Wu, Sicong Jiang, Max Ku, Ping Nie, Minghao Liu, and Wenhu Chen. 2025. EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing. CoRR abs/2509.26346 (2025).

[35] Yongliang Wu, Zonghui Li, Xinting Hu, Xinyu Ye, Xianfang Zeng, Gang Yu, Wenbo Zhu, Bernt Schiele, Ming-Hsuan Yang, and Xu Yang. 2025. KRIS-Bench: Benchmarking Next-Level Intelligent Image Editing Models. CoRR abs/2505.16707 (2025).

[36] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, and Ping Luo. 2025. DanceGRPO: Unleashing GRPO on Visual Generation. CoRR abs/2505.07818 (2025).

[37] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. 2025. ImgEdit: A Unified Image Editing Dataset and Benchmark. CoRR abs/2505.20275 (2025).

[38] Fukun Yin, Shiyu Liu, Yucheng Han, Zhibo Wang, Peng Xing, Rui Wang, Wei Cheng, Yingming Wang, Aojie Li, Zixin Yin, Pengtao Chen, Xiangyu Zhang, Daxin Jiang, Xianfang Zeng, and Gang Yu. 2025. ReasonEdit: Towards Reasoning Enhanced Image Editing Models. CoRR abs/2511.22625 (2025).

[39] Liangbing Zhao, Le Zhuo, Sayak Paul, Hongsheng Li, and Mohamed Elhoseiny. 2026. From Statics to Dynamics: Physics-Aware Image Editing with Latent Transition Priors. CoRR abs/2602.21778 (2026).

[40] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Hao Li, Zicheng Zhang, Guangtao Zhai, Junchi Yan, Hua Yang, Xue Yang, and Haodong Duan. 2025. Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing. CoRR abs/2504.02826 (2025).

[41] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. 2025. DifusionNFT: Online Difusion Reinforcement with Forward Process. CoRR abs/2509.16117 (2025).

[42] Huaisheng Zhu, Teng Xiao, and Vasant G. Honavar. 2025. DSPO: Direct Score Preference Optimization for Difusion Model Alignment. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

# Su<sub>pp</sub>lementar<sub>y</sub> Material for DARS: Dual-Level Credit Assi<sub>g</sub>nment RL with Structured Reasonin<sub>g</sub> for Instruction-Based Ima<sub>g</sub>e Editi<sub>ng</sub>

0. Contents and Index 1   
Contents 1   
List of Figures 1   
List of Tables 1   
1 Additional Methodological Details 2   
1.1 Expanded Optimization Objectives 2   
1.2 Variance-Based Cross-Module Credit Assignment 2   
1.3 Adaptive Curriculum and Soft Module Routing 3   
1.4 Overall Training Procedure 3   
2 Structured Planning and Reward Design 3   
2.1 Structured Planner Output Format 3   
2.2 Shared Checklist Construction and Slot   
Alignment 3   
2.3 Reward Instantiation in Practice 4   
2.4 Illustrative Prefix-Gated Reward Example 5   
3 Experimental Setup Details 7   
3.1 Training Setup and Hyperparameters 7   
3.2 Controlled Baseline and Evaluation Protocol 7   
3.3 KRIS-Bench Breakdown 7   
3.4 RISE-Bench Breakdown 7   
3.5 ImgEdit-Bench Breakdown 7   
3.6 GEdit-Bench-EN Breakdown 7   
3.7 PICA-Bench Simple-Prompt Breakdown 7   
3.8 PICA-Bench Explicit-Prompt Breakdown 7   
3.9 Baseline Architecture Overview 7   
4 Failure Attribution Validation Details 7   
5 Additional Qualitative Results 8   
5.1 Structured Plans for Qualitative Cases 12   
6 Prompt Templates 12   
7 Limitations 12

## List of Figures

S1 Five-panel schematic of checklist-based prefix-gated   
planner reward on a sequence-completion edit. 6   
S2 Additional baseline comparisons on challenging edits. 10   
S3 Representative failure cases illustrating current   
limitations. 10   
S4 Additional reasoning-intensive editing examples. 14   
S5 Additional general editing examples. 15   
S6 System prompt used for GPT-5 failure-attribution   
annotation in the routing-signal validation study. 16   
S7 Planner-side prompt template used during rollout to   
rewrite a raw editing instruction into a four-slot plan. 17   
S8 Prompt template used in the ofline Gemini 3 Pro   
preprocessing stage to generate the shared slot-aligned   
checklist C(x) from the source image and user instruction. 18   
S9 Planner-side single-question prompt for the Modify slot. 19   
S10 Planner-side single-question prompt for the Preserve slot. 20   
S11 Planner-side single-question prompt for the Overall slot. 21   
S12 Planner-side single-question prompt for the Tips slot. 22   
S13 Renderer-side single-question prompt for the Modify slot. 23   
S14 Renderer-side single-question prompt for the Preserve   
slot. 24   
S15 Renderer-side single-question prompt for the Overall slot. 25   
S16 Renderer-side single-question prompt for the Tips slot. 26

## List of Tables

S1 Training recipe summary. 7   
S2 Fine-grained KRIS-Bench results. 8   
S3 Fine-grained RISE-Bench results. 8   
S4 Fine-grained ImgEdit-Bench results. 9   
S5 Fine-grained GEdit-Bench-EN results. 9   
S6 Fine-grained PICA-Bench results under the simple-prompt   
setting. 11   
S7 Fine-grained PICA-Bench results under the explicit  
prompt setting. 11   
S8 Detailed baseline architecture summary. 12   
S9 Actual four-slot plan decompositions for representative   
qualitative cases. 13

Preprint.

## 1 Additi<sub>ona</sub>l M<sub>e</sub>th<sub>o</sub>d<sub>o</sub>l<sub>og</sub>i<sub>ca</sub>l D<sub>e</sub>t<sub>a</sub>il<sub>s</sub>

## 1.1 Expanded Optimization Objectives

This subsection expands the compact objectives in the main paper into the exact rollout-level forms used during optimization. Definitions of the reward signals and the high-level design motivation remain in the main text; here we focus only on how the ofline cached checklist, online planner-generated four-slot plans, old-policy rollouts, and token- or step-level surrogates are instantiated in training. Throughout this subsection, � indexes sampled plans, � renderings, � structured slots, � planner tokens, and � denoising-trajectory positions.

Planner objective. For each input $\mathbf { x } ,$ we sample a group o ${ \mathrm { : } } G _ { \mathrm { p } } = M$ candidate plans from the behavior policy:

$$
\mathbf { e } _ { i } \sim q _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid \mathbf { x } ) , \qquad i = 1 , \ldots , G _ { \mathrm { p } } .\tag{S1}
$$

For each sampled plan $\mathbf { e } _ { i } ,$ we generate $K$ paired renderings $\mathbf { y } _ { i , k } \sim$ $p _ { \phi _ { \mathrm { o l d } } } ( \cdot \mid I _ { \mathrm { s r c } } , \mathbf { e } _ { i } )$ . The planner reward for the �-th sampled plan is the planner reward averaged over its rendered outcomes:

$$
r _ { i } ^ { \mathrm { p l a n } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } R _ { \mathrm { p l a n } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } ; \mathbf { C } ) ,\tag{S2}
$$

where the checklist $\mathbf { C } = \mathbf { C } ( \mathbf { x } )$ is pre-generated once ofline before RL training from the source image and instruction, and then shared across all rollouts of the same sample. The group-relative advantage is

$$
\hat { A } _ { i } ^ { \mathrm { p l a n } } = \frac { r _ { i } ^ { \mathrm { p l a n } } - \mathrm { m e a n } \left( \{ r _ { i ^ { \prime } } ^ { \mathrm { p l a n } } \} _ { i ^ { \prime } = 1 } ^ { G _ { \mathrm { p } } } \right) } { \mathrm { s t d } \left( \{ r _ { i ^ { \prime } } ^ { \mathrm { p l a n } } \} _ { i ^ { \prime } = 1 } ^ { G _ { \mathrm { p } } } \right) + \varepsilon _ { A } } .\tag{S3}
$$

Here $\varepsilon _ { A } > 0$ is the advantage-normalization stabilizer shared by the planner and renderer objectives. Let $s _ { i . k } ^ { ( j ) }$ denote the score of slot � under rendering $\mathbf { y } _ { i , k ; }$ , and define the rendering-averaged slot score and reweighted token-level advantage as

$$
\bar { s } _ { i } ^ { ( j ) } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } s _ { i , k } ^ { ( j ) } , \qquad \hat { A } _ { i , j } ^ { \mathrm { p l a n } } = \hat { A } _ { i } ^ { \mathrm { p l a n } } \cdot \prod _ { l = 1 } ^ { j - 1 } \bar { \sigma } _ { i } ^ { ( l ) } \cdot \left\{ \bar { s } _ { i } ^ { ( j ) } , \qquad \hat { A } _ { i } ^ { \mathrm { p l a n } } \geq 0 , \right.\tag{S4}
$$

where $\bar { \sigma } _ { i } ^ { ( l ) } = \mathbb { I } [ \bar { s } _ { i } ^ { ( l ) } \geq \delta _ { l } ]$ is an analogous prefix gate computed from rendering-averaged slot scores. It has the same threshold form and causal ordering as the per-render gate in the main paper, but is distinct from $\sigma _ { i , k } ^ { ( \overline { { l } } ) }$ . This sign-aware modulation prevents lowquality slots from inheriting positive credit from an otherwise good plan, while assigning stronger negative updates to the slots most associated with poor overall plan performance. Since the planner is autoregressive, the GRPO ratio is defined at the token level. For the �-th token in e<sub>�</sub>:

$$
\varrho _ { i , t } ^ { \mathrm { p l a n } } ( \theta ) = \frac { q _ { \theta } ( e _ { i , t } \mid \mathbf { x } , e _ { i , < t } ) } { q _ { \theta _ { \mathrm { o l d } } } ( e _ { i , t } \mid \mathbf { x } , e _ { i , < t } ) } .\tag{S5}
$$

Let slot(�) denote the structured slot containing token $t ,$ as determined by the enclosing ordered tag pair; the opening and closing tag tokens are assigned to that same slot. The token-level surrogate uses the sign-aware slot-weighted advantage $\hat { A } _ { i , \mathrm { s l o t } ( t ) } ^ { \mathrm { p l a n } }$ instead of a uniform plan-level advantage. The full planner objective is:

$$
\begin{array} { r l } & { \mathcal { G } _ { \mathrm { p l a n } } ( \theta ) = \mathbb { E } _ { \mathbf { x } } \Bigg [ w _ { \mathrm { c u r } } \big ( \mathbf { x } , n \big ) \left( 1 + \rho \alpha ( \mathbf { x } ) \right) \frac { 1 } { G _ { \mathrm { p } } } \displaystyle \sum _ { i = 1 } ^ { G _ { \mathrm { p } } } \frac { 1 } { \vert \mathbf { e } _ { i } \vert } \displaystyle \sum _ { t = 1 } ^ { \vert \mathbf { e } _ { i } \vert } } \\ & { \qquad \mathrm { ~ } \Bigg ( \mathrm { m i n } \big ( \varrho _ { i , t } ^ { \mathrm { p l a n } } ( \theta ) \hat { A } _ { i , \mathrm { s l o t } ( t ) } ^ { \mathrm { p l a n } } , } \\ & { \qquad \mathrm { ~ c l i p } \big ( \varrho _ { i , t } ^ { \mathrm { p l a n } } ( \theta ) , 1 - \epsilon _ { \mathrm { p } } , 1 + \epsilon _ { \mathrm { p } } \big ) \hat { A } _ { i , \mathrm { s l o t } ( t ) } ^ { \mathrm { p l a n } } \Big ) } \\ & { \qquad - \beta _ { \mathrm { p } } D _ { \mathrm { K L } } ( q _ { \theta } \parallel q _ { \mathrm { r e f } } ) \Bigg ) \Bigg ] . } \end{array}\tag{S6}
$$

Renderer objective. Using the same sampled plans $\{ { \mathbf { e } _ { i } } \} _ { i = 1 } ^ { M }$ from the planner rollout bank, we sample � renderer rollouts for each fixed pair $\left( \mathbf { x } , \mathbf { e } _ { i } \right)$ from the old difusion-flow policy. Let ${ \bf y } _ { i , k }$ denote the decoded image from the �-th rollout under plan $\mathbf { e } _ { i }$ . The rendererside reward is

$$
\begin{array} { r } { r _ { i , k } ^ { \mathrm { r e n d } } = R _ { \mathrm { r e n d } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } ; \mathbf { C } ) , } \end{array}\tag{S7}
$$

where again ${ \mathbf C } = { \mathbf C } ( { \mathbf x } )$ is fixed for all rollouts of the same input. The renderer-side group-relative advantage is computed separately within each fixed plan:

$$
\hat { A } _ { i , k } ^ { \mathrm { r e n d } } = \frac { r _ { i , k } ^ { \mathrm { r e n d } } - \mathrm { m e a n } \left( \{ r _ { i , k ^ { \prime } } ^ { \mathrm { r e n d } } \} _ { k ^ { \prime } = 1 } ^ { K } \right) } { \mathrm { s t d } \left( \{ r _ { i , k ^ { \prime } } ^ { \mathrm { r e n d } } \} _ { k ^ { \prime } = 1 } ^ { K } \right) + \varepsilon _ { A } } .\tag{S8}
$$

For flow-GRPO, let � index the complete ordered denoising trajectory, $\tau _ { m }$ its difusion time, and $z _ { i , k , m }$ the corresponding latent state. Here $\tau _ { m - 1 }$ is the adjacent lower-noise time after $\tau _ { m } ,$ , with latent state $z _ { i , k , m - 1 } ;$ the optimized indices form a subset S of this trajectory. The ratio is

$$
\varrho _ { i , k , m } ^ { \mathrm { f l o w } } ( \phi ) = \frac { \ p _ { \phi } \left( z _ { i , k , m - 1 } \mid I _ { \mathrm { s r c } } , \mathbf { e } _ { i } , z _ { i , k , m } , \tau _ { m } \right) } { \ p _ { \phi _ { \mathrm { o l d } } } \left( z _ { i , k , m - 1 } \mid I _ { \mathrm { s r c } } , \mathbf { e } _ { i } , z _ { i , k , m } , \tau _ { m } \right) } .\tag{S9}
$$

The full renderer objective is:

$$
\begin{array} { r l } & { \mathcal { T } _ { \mathrm { r e n d } } ( \phi ) = \mathbb { E } _ { \mathbf { x } } \Bigg [ w _ { \mathrm { c u r } } ( \mathbf { x } , n ) ( 1 + \rho \omega ( \mathbf { x } ) ) \displaystyle \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { | S | } \sum _ { m \in S } } \\ & { \quad \quad \quad ( \operatorname* { m i n } \bigl ( \varrho _ { i , k , m } ^ { \mathrm { f l o w } } ( \phi ) \hat { A } _ { i , k } ^ { \mathrm { r e n d } } , \mathrm { c l i p } ( \varrho _ { i , k , m } ^ { \mathrm { f l o w } } ( \phi ) , 1 - \epsilon _ { \mathrm { r } } , 1 + \epsilon _ { \mathrm { r } } ) \hat { A } _ { i , k } ^ { \mathrm { r e n d } } ) } \\ & { \quad \quad \quad \quad \quad - \beta _ { \mathrm { r } } D _ { \mathrm { K L } } ( p _ { \phi } \parallel p _ { \mathrm { r e f } } ) ) \Bigg ] . } \end{array}\tag{S10}
$$

## 1<sub>.</sub>2 V<sub>a</sub>ri<sub>a</sub>n<sub>ce</sub>-B<sub>ase</sub>d Cr<sub>oss</sub>-M<sub>o</sub>d<sub>u</sub>l<sub>e</sub> Cr<sub>e</sub>dit A<sub>ss</sub>i<sub>gnmen</sub>t

The main paper introduces the plan-dominant and render-dominant variability terms. Here, we record only the estimator used in training. For a fixed input $\mathbf { x } = \left( I _ { \mathrm { s r c } } , c \right)$ , let $\mathbf { e } \sim q _ { \theta } ( \cdot \mid \mathbf { x } )$ and $\mathbf { y } \sim p _ { \phi } ( \cdot \mathbf { \theta } |$ $I _ { \mathrm { s r c } } , \mathbf { e } )$ . We decompose the shared-reward variance as

$$
U _ { \mathrm { p l a n } } ( \mathbf { x } ) = \mathrm { V a r } _ { \mathbf { e } } \Big ( \mathbb { E } _ { \mathbf { y } | I _ { \mathrm { s r c } } , \mathbf { e } } [ R _ { \mathrm { s h a r e } } ] \Big ) ,\tag{S11}
$$

$$
U _ { \mathrm { r e n d } } ( \mathbf { x } ) = \mathbb { E } _ { \mathbf { e } } \left[ \operatorname { V a r } _ { \mathbf { y } | I _ { \mathrm { s r c } } , \mathbf { e } } ( R _ { \mathrm { s h a r e } } ) \right] .\tag{S12}
$$

By the law of total variance,

$$
\operatorname { V a r } ( R _ { \mathrm { s h a r e } } \mid \mathbf { x } ) = U _ { \mathrm { p l a n } } ( \mathbf { x } ) + U _ { \mathrm { r e n d } } ( \mathbf { x } ) .\tag{S13}
$$

For each training sample, the same � $\times K$ rollout bank already used for reward computation is reused to estimate these two terms. Let $r _ { i , k } ^ { \mathrm { s h a r e } }$ be the shared reward of the �-th rendering under the �-th sampled plan, $\begin{array} { r } { \bar { r } _ { i } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } r _ { i , k } ^ { \mathrm { s h a r e } } } \end{array}$ the per-plan mean, and $\bar { r } =$ $\textstyle { \frac { 1 } { M } } \sum _ { i = 1 } ^ { M } { \bar { r } } _ { i }$ the grand mean. We compute

$$
s _ { \mathrm { w i t h i n } } ^ { 2 } ( \mathbf { x } ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { 1 } { K - 1 } \sum _ { k = 1 } ^ { K } ( r _ { i , k } ^ { \mathrm { s h a r e } } - \bar { r } _ { i } ) ^ { 2 } ,\tag{S14}
$$

$$
s _ { \mathrm { b e t w e e n } } ^ { 2 } ( \mathbf { x } ) = \frac { 1 } { M - 1 } \sum _ { i = 1 } ^ { M } \bigl ( \bar { r } _ { i } - \bar { r } \bigr ) ^ { 2 } .\tag{S15}
$$

The within-plan Monte Carlo estimate is

$$
\widehat { U } _ { \mathrm { r e n d } } ( \mathbf { x } ) = s _ { \mathrm { w i t h i n } } ^ { 2 } ( \mathbf { x } ) ,\tag{S16}
$$

and the across-plan estimate subtracts the renderer-noise term induced by averaging only � renderings per plan:

$$
\begin{array} { r } { \widehat { U } _ { \mathrm { p l a n } } ( \mathbf { x } ) = \operatorname* { m a x } \left( 0 , s _ { \mathrm { b e t w e e n } } ^ { 2 } ( \mathbf { x } ) - \frac { 1 } { K } s _ { \mathrm { w i t h i n } } ^ { 2 } ( \mathbf { x } ) \right) . } \end{array}\tag{S17}
$$

The max $( 0 , \cdot )$ truncation avoids negative across-plan estimates caused by finite-sample noise. No additional renderer calls are introduced: the same rollout bank feeds reward estimation, curriculum weighting, routing, and policy optimization.

## 1.3 Ada<sub>p</sub>tive Curriculum and Soft Module Routin<sub>g</sub>

This subsection records the rollout statistics used to instantiate the main-paper curriculum and routing definitions as per-sample training weights.

Curriculum scheduling. The rollout-based hardness estimate is

$$
\widehat { H } ( \mathbf { x } ) = - \frac { 1 } { M K } \sum _ { i = 1 } ^ { M } \sum _ { k = 1 } ^ { K } r _ { i , k } ^ { \mathrm { s h a r e } } ,\tag{S18}
$$

where larger values indicate harder samples. In implementation, ${ \widehat { H } } ( \mathbf { x } )$ is normalized to $\widetilde { H } ( \mathbf { x } )$ using the running hardness statistics maintained over recent training batches, and the curriculum boundary $\kappa ( n )$ is taken as an empirical quantile of that same bufer at step �. The curriculum weight is

$$
w _ { \mathrm { c u r } } ( \mathbf { x } , n ) = \mathrm { s i g m o i d } \left( \frac { \kappa ( n ) - \widetilde { H } ( \mathbf { x } ) } { \tau _ { \mathrm { c } } } \right) .\tag{S19}
$$

Here $\tau _ { \mathrm { c } }$ controls how sharply the weight changes around the current boundary. This makes the curriculum update depend only on rollout hardness, not on which module is currently more variable.

Soft module routing. We use the relative magnitude of the variance signals to define

$$
\alpha ( \mathbf { x } ) = \frac { \widehat { U } _ { \mathrm { p l a n } } ( \mathbf { x } ) + \varepsilon _ { U } } { \widehat { U } _ { \mathrm { p l a n } } ( \mathbf { x } ) + \widehat { U } _ { \mathrm { r e n d } } ( \mathbf { x } ) + 2 \varepsilon _ { U } } , \qquad \omega ( \mathbf { x } ) = 1 - \alpha ( \mathbf { x } ) ,\tag{S20}
$$

where $\varepsilon _ { U } > 0$ is the routing smoothing constant. Routing is applied only as a residual multiplier on top of the shared curriculum factor, so every admitted sample still updates both modules. The variance decomposition only decides where the extra module-specific emphasis should go; it is not used as a hard assignment rule.

## 1<sub>.</sub>4 O<sub>ve</sub>r<sub>a</sub>ll Tr<sub>a</sub>inin<sub>g</sub> Pr<sub>oce</sub>d<sub>u</sub>r<sub>e</sub>

Algorithm 1 summarizes one training iteration of DARS. A single shared � ×� rollout bank is reused throughout the iteration: it first produces module-specific rewards, then yields planner-side and renderer-side advantages, and finally supplies the variance, routing, and curriculum statistics used to weight the two objectives. This reuse is important operationally because it keeps checklist loading, reward estimation, and module-specific optimization aligned to the same sampled evidence without introducing extra renderer calls beyond the rollout budget already used for policy optimization.

## 2 Structured Plannin<sub>g</sub> and Reward Desi<sub>g</sub>n

## 2.1 Structured Planner Out<sub>p</sub>ut Format

The main paper already motivates the four-slot planner decomposition and the associated prefix-gated dependency structure. Here we record the operational slot semantics used throughout the supplementary examples and prompt templates: Modify specifies the requested visible change, Preserve specifies the important content that must remain unchanged, Overall specifies scene-level coherence requirements, and Tips provides localized renderer-facing execution details. The online planner serializes these free-form fields in the fixed order

$$
\begin{array} { r l } & { { < } \mathsf { M o d i f y } > . . . < / \mathsf { M o d i f y } > \mathsf { \quad } < \mathsf { P r e s e r v e } > . . . < / \mathsf { P r e s e r v e } > } \\ & { \qquad < 0 \mathsf { v e r a l l } > . . . < / 0 \mathsf { v e r a l l } > \mathsf { \quad } < \mathsf { T i p s } > . . . < / \mathsf { I i p s } > . } \end{array}
$$

The matching tag boundaries determine the token-to-slot mapping slot(�), with each tag pair assigned to its enclosed slot. This is the same slot schema used by the online planner output, the ofline checklist cache, the reward prompts, and the qualitative plan table.

## 2<sub>.</sub>2 Sh<sub>are</sub>d Ch<sub>ec</sub>kli<sub>s</sub>t C<sub>ons</sub>t<sub>ruc</sub>ti<sub>on an</sub>d Sl<sub>o</sub>t Ali<sub>g</sub>nment

The reward pipeline uses a single slot-aligned checklist shared by planner reward, renderer reward, hardness estimation, and routing. The implementation has three distinct stages. First, during rollout, the current planner rewrites the raw instruction into a four-slot plan e; Fig. S7 shows this planner-side prompt. Second, before RL starts, Gemini 3 Pro preprocesses the training set once and directly generates a cached checklist C(x) from the source image and raw user instruction; Fig. S8 shows this ofline checklist-generation prompt. Third, during RL training, Qwen3-VL-32B performs on line Yes/No scoring by pairing the current rollout plan, the cached checklist, and the realized edited image. Gemini 3 Pro is not called at rollout time.

Operationally, Gemini 3 Pro directly produces the shared checklist $\mathbf { C } ( \mathbf { x } ) = ( C ^ { \mathrm { m o d } } , C ^ { \mathrm { p r e } } , C ^ { \mathrm { o v r } } , C ^ { \mathrm { t i p } } )$ from the source image and raw instruction, with a question list for each slot role. This preprocessing is run once before RL training, so the same cached $\mathbf { C } ( \mathbf { x } )$ is reused across all � ×� rollouts of a sample. At rollout time, the current planner separately produces its own four-slot plan in the same slot schema. The checklist, therefore, defines the shared evaluation target, while the rollout-time plan provides the instance-specific slot text being judged.

Each checklist field is not copied from the planner output. Instead, Gemini 3 Pro maps the source image and raw instruction directly into 3–5 binary verification questions per slot. The conversion follows four implementation constraints: slot exclusivity, so each question belongs to exactly one slot; cross-slot complementarity, so the same requirement is not repeated across slots; no cross-slot leakage, so one slot does not silently test another slot’s responsibility; and post-render verifiability, so every question can later be answered from the source image, the relevant slot text, and the realized edited image. In practice, $C ^ { \mathrm { m o d } }$ targets the requested visible change, $C ^ { \mathrm { p r e } }$ targets unchanged content, $C ^ { \mathrm { { o v r } } }$ targets global coherence and realism, and $C ^ { \mathrm { t i p } }$ targets localized implementation details.

Algorithm 1 DARS Training Algorithm   
Require: Planner policy $q _ { \theta } ,$ renderer policy $p _ { \phi } .$ , reference policies $q _ { \mathrm { r e f } } , p _ { \mathrm { r e f } } ,$ objective hyperparameters $\epsilon _ { \mathrm { p } } , \epsilon _ { \mathrm { r } } , \varepsilon _ { A } , \varepsilon _ { U } , \beta _ { \mathrm { p } } , \beta _ { \mathrm { r } } , \rho ,$ dataset ${ \mathcal { D } } ,$ rollout counts   
$M , K ,$ training step �   
Ensure: Updated planner parameters � and renderer parameters $\phi$   
1: Refresh old policies: $q _ { \theta _ { \mathrm { o l d } } }  q _ { \theta } , p _ { \phi _ { \mathrm { o l d } } }  p _ { \phi }$   
2: Sample minibatch $\mathcal { B } \sim \overline { { \mathcal { D } } }$   
3: for each input x ∈ B do   
4: Load pre-generated shared checklist $\mathbf { C } ( \mathbf { x } )$   
5: f<sub>or</sub> $i = 1$ to � do   
6: Sample plan ${ \bf e } _ { i } \sim q \theta _ { \mathrm { o l d } } \left( { \bf \cdot } \mid { \bf x } \right)$   
7: f<sub>or</sub> $k = 1$ to � do   
8: Sample rendering $\mathbf { y } _ { i , k } \sim p _ { \phi _ { \mathrm { o l d } } } \left( \cdot \mid I _ { \mathrm { s r c } } , \mathbf { e } _ { i } \right)$   
9: Evaluate $R _ { \mathrm { p l a n } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } ; \mathbf { C } ) , R _ { \mathrm { r e n d } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } ; \mathbf { C } )$ , and $R _ { \mathrm { s h a r e } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } )$   
10: <sub>en</sub>d f<sub>or</sub>   
11: <sub>en</sub>d f<sub>or</sub>   
12: Compute planner-side rewards $\{ r _ { i } ^ { \mathrm { p l a n } } \} _ { i = 1 } ^ { M }$ and group-relative advantages $\{ \hat { A } _ { i } ^ { \mathrm { p l a n } } \} _ { i = 1 } ^ { M }$   
13: Compute rendering-averaged slot scores $\begin{array} { r } { \bar { s } _ { i } ^ { ( j ) } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } s _ { i , k } ^ { ( j ) } } \end{array}$ and prefix gates $\bar { \sigma } _ { i } ^ { ( j ) } = \mathbb { I } [ \bar { s } _ { i } ^ { ( j ) } \geq \delta _ { j } ]$ for all sampled plans and slots   
14: Compute sign-aware slot-weighted token advantages $\hat { A } _ { i , j } ^ { \mathrm { p l a n } }$ from $\hat { A } _ { i } ^ { \mathrm { p l a n } } , \bar { s } _ { i } ^ { ( j ) }$ , and $\bar { \sigma } _ { i } ^ { ( j ) }$   
15: f<sub>or</sub> $i = 1$ to � do   
16: Compute renderer-side rewards $\{ r _ { i , k } ^ { \mathrm { r e n d } } \} _ { k = 1 } ^ { K }$ and group-relative advantages $\{ \hat { A } _ { i , k } ^ { \mathrm { r e n d } } \} _ { k = 1 } ^ { K }$ over the � renderings under the fixed plan $\mathbf { e } _ { i }$   
17: <sub>en</sub>d f<sub>or</sub>   
18: Estimate $\widehat { U } _ { \mathrm { p l a n } } ( \mathbf { x } ) , \widehat { U } _ { \mathrm { r e n d } } ( \mathbf { x } )$ , and raw hardness ${ \widehat { H } } ( \mathbf { x } )$ from the same $M \times K$ rollout bank   
19: Update running hardness statistics, normalize ${ \widehat { H } } ( \mathbf { x } )$ to $\widetilde { H } ( \mathbf { x } )$ , and obtain the curriculum boundary $\kappa ( n )$ from the running bufer   
20: Compute routing weights $\alpha ( \mathbf { x } )$ , � (x) and curriculum weight $w _ { \mathrm { c u r } } ( { \bf x } , n ) = \mathrm { s i g m o i d } ( ( \kappa ( n ) - \widetilde { \cal H } ( { \bf x } ) ) / \tau _ { \mathrm { c } } )$   
21: Accumulate the sample contributions to $\mathcal { T } _ { \mathrm { p l a n } } ( \theta )$ and ${ \mathcal { T } } _ { \mathrm { r e n d } } ( \phi )$ without resampling   
22: en<sup>d f</sup>or   
23: Update � by text-GRPO using $\mathcal { T } _ { \mathrm { p l a n } } ( \theta )$   
24: Update � by flow-GRPO using ${ \mathcal { T } } _ { \mathrm { r e n d } } ( \phi )$

The ofline checklist generator also follows conservative default rules when the instruction under-specifies the edit. If the instruction is local, preservation questions default to protecting identity, pose, background, viewpoint, and other non-target content unless the instruction explicitly asks to modify them. If the instruction requires a background change, that requested change is moved into the Modify or Overall field and is not redundantly enforced as a

This slot alignment allows the same checklist to support both module-specific rewards without changing the semantic target. During RL training, Gemini 3 Pro is no longer used. Qwen3-VL-32B performs all online checklist-based scoring. On the planner side, it judges whether each rollout-time slot text was a useful plan for the observed edit under the matching checklist field. On the renderer side, it judges whether the final rendered image actually satisfies that same checklist field. The shared checklist, therefore, ties planner-side diagnosis and renderer-side execution scoring to the same pre-generated per-slot questions.

preservation constraint. Text-editing requests add exact renderedtext correctness to the relevant slot, while removal or inpainting requests add plausibility and local consistency checks so that filled regions remain compatible with surrounding texture, lighting, and geometry.

The concrete prompt templates for these three stages are collected later in the template bank: planner rollout rewriting in Fig. S7, ofline checklist generation in Fig. S8, and online scoring prompts in Figs. S9–S16.

## 2<sub>.</sub>3 R<sub>ewar</sub>d I<sub>ns</sub>t<sub>an</sub>ti<sub>a</sub>ti<sub>on</sub> i<sub>n</sub> P<sub>rac</sub>ti<sub>ce</sub>

Question-level aggregation. For input $\mathbf { x } = \left( I _ { \mathrm { s r c } } , c \right)$ , plan $\mathbf { e } _ { i } ,$ rendering ${ \bf y } _ { i , k }$ , slot �, and question $q \in C ^ { ( j ) }$ , let $a _ { i , k , q } ^ { \mathrm { p l a n } } \in \{ 0 , 1 \}$ denote the planner-side judgment and define $a _ { i , k , q } ^ { \mathrm { r e n d } } = \mathbb { I } [ \mathrm { J u d g e } _ { \mathrm { r e n d } } ( I _ { \mathrm { s r c } } , \mathbf { y } _ { i , k } , c , e _ { i } ^ { ( j ) } , q ) =$ Yes]. We encode Yes as 1 and No as 0. All questions within a slot are weighted equally, giving the per-render slot scores

$$
s _ { i , k } ^ { ( j ) } = \frac { 1 } { \vert C ^ { ( j ) } \vert } \sum _ { q \in C ^ { ( j ) } } a _ { i , k , q } ^ { \mathrm { p l a n } } , \qquad u _ { i , k } ^ { ( j ) } = \frac { 1 } { \vert C ^ { ( j ) } \vert } \sum _ { q \in C ^ { ( j ) } } a _ { i , k , q } ^ { \mathrm { r e n d } } .\tag{S21}
$$

Thus, $s _ { i , k } ^ { ( j ) }$ and $u _ { i , k } ^ { ( j ) }$ are normalized Yes fractions in [0, 1], rather than additional continuous outputs from the judge.

Planner-side diagnostic. At scoring time, the shared checklist has already been generated ofline by Gemini 3 Pro and loaded from cache, while the current planner policy has produced the rollouttime four-slot plan $\mathbf { e } _ { i } .$ . Qwen3-VL-32B evaluates every question in each slot through single-question Yes/No judgments conditioned on the slot text, the original input, the realized edited image ${ \bf y } _ { i , k }$ , and the aligned checklist field $C ^ { ( j ) }$ . The resulting $s _ { i , k } ^ { ( j ) }$ is a post-render diagnostic of whether that slot functioned as a useful plan for the realized edit. The corresponding prompt templates are shown in Figs. S9–S12. Define the per-render gate $\sigma _ { i , k } ^ { ( j ) } \overset { \cdot } { = } \mathbb { I } [ s _ { i , k } ^ { ( j ) } \geq \delta _ { j } ]$ . The per-render planner reward is

$$
R _ { \mathrm { p l a n } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } ; \mathbf { C } ) = \sum _ { j = 1 } ^ { 4 } \lambda _ { j } \cdot s _ { i , k } ^ { ( j ) } \cdot \prod _ { l = 1 } ^ { j - 1 } \sigma _ { i , k } ^ { ( l ) } .\tag{S22}
$$

We use $( \lambda _ { \mathrm { m o d } } , \lambda _ { \mathrm { p r e } } , \lambda _ { \mathrm { o v r } } , \lambda _ { \mathrm { t i p } } ) = ( 0 . 4 , 0 . 2 , 0 . 2 , 0 . 2 )$ and $\delta _ { j } = 0 . 6 6$ for every slot. After evaluating all � renderings under plan $\mathbf { e } _ { i } ,$ , we compute

$$
r _ { i } ^ { \mathrm { p l a n } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } R _ { \mathrm { p l a n } } ( { \bf x } , { \bf e } _ { i } , { \bf y } _ { i , k } ; { \bf C } ) , \qquad \bar { s } _ { i } ^ { ( j ) } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } s _ { i , k } ^ { ( j ) } .\tag{S23}
$$

The first quantity yields the plan-level group-relative advantage, while the second and its analogous averaged-score gate $\bar { \sigma } _ { i } ^ { ( j ) } = $ $\mathbb { I } [ \bar { s } _ { i } ^ { ( j ) } \geq \delta _ { j } ]$ are used for the sign-aware token-level advantage in Section 1.1.

Renderer-side execution scoring. The renderer reward changes the target ofjudgment while retaining the same checklist fields. For each � and $q \in C ^ { ( j ) }$ , Qwen3-VL-32B is conditioned on $( I _ { \mathrm { s r c } } , \mathbf { y } _ { i , k } , c , e _ { i } ^ { ( j ) } , q )$ The instruction and slot text specify the intended requirement; the judge scores whether the rendered image visibly satisfies it, not the quality of the slot text. The resulting $u _ { i , k } ^ { ( j ) }$ is the fraction of satisfied requirements in that slot. The corresponding prompt templates are shown in Figs. S13–S16. The per-render renderer reward is

$$
r _ { i , k } ^ { \mathrm { r e n d } } = R _ { \mathrm { r e n d } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } ; \mathbf { C } ) = \sum _ { j = 1 } ^ { 4 } \gamma _ { j } u _ { i , k } ^ { ( j ) } .\tag{S24}
$$

Here, $\gamma _ { j }$ controls the contribution of each slot to the final execution score. The � renderer rewards under each fixed plan remain separate when computing the renderer-side group-relative advantage; they are not averaged before normalization.

Shared reward reuse. The shared reward used for hardness estimation and variance decomposition is

$$
R _ { \mathrm { s h a r e } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } ) = \eta _ { \mathrm { p l a n } } R _ { \mathrm { p l a n } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } ; \mathbf { C } ) + \eta _ { \mathrm { r e n d } } R _ { \mathrm { r e n d } } ( \mathbf { x } , \mathbf { e } _ { i } , \mathbf { y } _ { i , k } ; \mathbf { C } ) .\tag{S25}
$$

Its role is restricted to hardness estimation and variance decomposition; planner and renderer policy gradients still use their own module-specific rewards.

## 2<sub>.</sub>4 Ill<sub>us</sub>t<sub>ra</sub>ti<sub>ve</sub> P<sub>re</sub>fi<sub>x-</sub>G<sub>a</sub>t<sub>e</sub>d R<sub>ewar</sub>d E<sub>xamp</sub>l<sub>e</sub>

We use a simple sequence-completion edit to illustrate how prefix gating changes planner-side credit assignment in practice. The source image $I _ { \mathrm { s r c } }$ contains a diagonal chain of hexagons labeled ${ } ^ { * } 5 ,$ $7 , 1 1 , 1 3 , 1 7 , ? ^ { \ast }$ , and the raw instruction � is: “Replace the final question mark with the correct next number, while keeping the existing numbers, hexagons, and diagonal layout unchanged.” Figure S1 instantiates the same notation used in the main paper on this concrete case: Panels A–B define the input $\mathbf { x } = \left( I _ { \mathrm { s r c } } , c \right)$ , the structured planner output $\mathbf { e } = ( e ^ { \mathrm { m o d } } , e ^ { \mathrm { p r e } } , e ^ { \mathrm { o v r } } , e ^ { \mathrm { t i p } } )$ , and the shared checklist $\mathbf { \bar { C } } ( \mathbf { x } ) = \left( C ^ { \mathrm { m o d } } , C ^ { \mathrm { p r e } } , C ^ { \mathrm { o v r } } , C ^ { \mathrm { t i p } } \right)$ ; Panels C–E then show three realized planner outcomes, namely Failure A, Failure B, and Success.

Consistent with the training definition in Eq. S21, each slot score is computed from its own checklist as $\boldsymbol { s } ^ { ( j ) } \dot { \mathbf { \xi } } = \# \Upsilon \mathbf { e } \mathbf { s } / | { \cal C } ^ { ( j ) } |$ , with normalized slot weights (�<sub>mod</sub>, �<sub>pre</sub>, $\lambda _ { \mathrm { o v r } } , \lambda _ { \mathrm { t i p } } ) = ( 0 . 4 , 0 . 2 , 0 . 2 , 0 . 2 )$ and a uniform activation threshold $\delta _ { \mathrm { m o d } } = \delta _ { \mathrm { p r e } } = \delta _ { \mathrm { o v r } } = \delta _ { \mathrm { t i p } } = 0 . 6 6$ The planner-side reward for a realized outcome is then

$$
\begin{array} { r } { R _ { \mathrm { p l a n } } = \lambda _ { \mathrm { m o d } } s ^ { \mathrm { m o d } } + \lambda _ { \mathrm { p r e } } s ^ { \mathrm { p r e } } \sigma _ { \mathrm { m o d } } + \lambda _ { \mathrm { o v r } } s ^ { \mathrm { o v r } } \sigma _ { \mathrm { m o d } } \sigma _ { \mathrm { p r e } } + \lambda _ { \mathrm { t i p } } s ^ { \mathrm { t i p } } \sigma _ { \mathrm { m o d } } \sigma _ { \mathrm { p r e } } \sigma _ { \mathrm { o v r } } , } \end{array}\tag{S26}
$$

where $\sigma _ { j } = \mathbb { I } [ s ^ { ( j ) } \geq \delta _ { j } ]$ . The question-level aggregation, slot weights, activation threshold, and prefix-gating rule are identical to those used in training.

Failure A in Panel C shows the clearest advantage of prefix gating. The enhanced plan is locally neat in Preserve, Overall, and Tips, but the core Modify slot is wrong because $e ^ { \mathrm { m o d } }$ proposes 21 instead of 19. The corresponding Modify score is therefore $s ^ { \mathrm { m o d } } =$ $2 / 4 = 0 . 5 0 < 0 . 6 6 ,$ , so $\sigma _ { \mathrm { m o d } } = 0$ . As a result, the prefix gate closes immediately after Modify, and the planner reward becomes

$$
\begin{array} { r l } & { R _ { \mathrm { p l a n } } ^ { \mathrm { ( A ) } } = 0 . 4 \cdot 0 . 5 0 + 0 . 2 \cdot 1 . 0 0 \cdot 0 + 0 . 2 \cdot 0 . 7 5 \cdot 0 + 0 . 2 \cdot 1 . 0 0 \cdot 0 } \\ & { \qquad = 0 . 2 0 . } \end{array}\tag{S27}
$$

Even though the rendered result still satisfies all five Preserve checks and most downstream checks, those later slots receive no additional credit once the causally prior Modify requirement fails. By contrast, an ungated weighted sum would still produce 0.75 because the strong downstream slots would partially compensate for the incorrect core reasoning.

Failure B in Panel D illustrates a diferent error pattern. Here the Modify slot is correct, so $s ^ { \mathrm { m o d } } = 4 / 4 = 1 . 0 0$ and $\sigma _ { \mathrm { m o d } } = 1$ However, the weak Preserve slot $e ^ { \mathrm { p r e } }$ allows collateral changes near the target, and the rendered result changes the original $^ { * } 1 3 ^ { * }$ into $^ { \mathrm { * } } 1 7 ^ { \mathrm { * } }$ while also darkening the background. This drops the Preserve score to $s ^ { \mathrm { p r e } } = 2 / 5 = 0 . 4 0 < 0 . 6 6$ , so $\sigma _ { \mathrm { p r e } } = 0 .$ Thus, the reward stil keeps the correct Modify credit, but blocks Overall and Tips from contributing:

$$
\begin{array} { r l } & { R _ { \mathrm { p l a n } } ^ { \mathrm { ( B ) } } = 0 . 4 \cdot 1 . 0 0 + 0 . 2 \cdot 0 . 4 0 + 0 . 2 \cdot 0 . 7 5 \cdot 0 + 0 . 2 \cdot 0 . 7 5 \cdot 0 } \\ & { ~ \mathrm = 0 . 4 8 . } \end{array}\tag{S28}
$$

Again, the ungated weighted sum would be much higher (0.78) because the later slots would still receive credit despite the failed preservation constraint.

Finally, Panel E shows the successful case. Here, all four slot texts align with the intended edit, and the realized image satisfies every checklist item across Modify, Preserve, Overall, and Tips. Therefore, all gates remain active, and the final planner reward is maximal:

$$
\begin{array} { l l } { R _ { \mathrm { p l a n } } ^ { ( \mathrm { s u c c e s s } ) } = 0 . 4 \cdot 1 . 0 0 + 0 . 2 \cdot 1 . 0 0 + 0 . 2 \cdot 1 . 0 0 + 0 . 2 \cdot 1 . 0 0 } \\ { \qquad = 1 . 0 0 . } \end{array}\tag{S29}
$$

Taken together, the three cases in Fig. S1 make the intended creditassignment behavior explicit. Prefix gating does not simply average slot quality; instead, it follows the causal dependency structure of the four-slot plan. Modify must first specify the correct edit target, Preserve must then protect non-target content, and only after those upstream requirements are reliable, do Overall and Tips receive efective reward. This is precisely the behavior that the prefix-gated reward is designed to enforce during planner optimization.

## Panel A. Input image x and raw instruction

![](images/909d8dc3705d0fc1e37016e51aa1cd8a479f3b42676128efa9b33b044d17796b.jpg)

Raw instruction �: replace the final $? ^ { \dag }$ with the correct next number, while keeping the existing numbers, hexagons, and diagonal layout unchanged.

Ground-truth target: the sequence follows prime numbers, so the correct completion is $^ { * } 1 9 ^ { * } .$

## Panel B. Checklist C(x) and plan e

Modif<sub>y</sub> $C ^ { \mathrm { m o d } }$   
1. Is the final “?” replaced with a nu  
meral?   
2. Is the new numeral exactly “19”?   
3. Does the edit correctly continue the   
prime pattern after 17?   
4. Is the target of the change the last   
hexagon?   
Pr<sub>ese</sub>r<sub>ve</sub> $C ^ { \mathrm { p r e } }$   
1. Are 5 and 7 unchanged?   
2. Is 11 unchanged?   
3. Is 13 unchanged?   
4. Are all non-target hexagons other  
wise untouched?   
5. Does the background remain white?   
O<sub>vera</sub>ll �<sup>ovr</sup>   
1. Is the six-hexagon diagonal layout   
preserved?   
2. Are size, spacing, and viewpoint con  
sistent?   
3. Does the final sequence remain glob  
ally coherent?   
4. Is the result clean and artifact-free?   
<sub>Tips �</sub><sup>tip</sup>   
1. Is only the last hexagon edited?   
2. Does the new digit match the original   
font style?   
3. Does the stroke/outline style remain   
consistent?   
4. Is the new digit centered with similar   
spacing?

```latex
Structured plan e: $\mathbf { \epsilon } \mathbf { e } = ( e ^ { \mathrm { m o d } } , e ^ { \mathrm { p r e } } , e ^ { \mathrm { o v r } } , e ^ { \mathrm { t i p } } )$ , where the four
slots are Modify, Preserve, Overall, and Tips.
Scoring rule: slot score $\boldsymbol { s } ^ { ( j ) } = \# \Upsilon e s / | \boldsymbol { C } ^ { ( j ) } |$ for
� ∈ {mod, pre, ovr, tip}.
Gate rule: slot gate $\sigma _ { j } = \mathbb { I } [ s ^ { ( j ) } \geq \delta _ { j } ]$ with
$\delta _ { \mathrm { m o d } } = \delta _ { \mathrm { p r e } } = \delta _ { \mathrm { o v r } } = \delta _ { \mathrm { t i p } } = 0 . 6 6 .$
```

## Panel C. Failure A: wron<sub>g</sub> Modif<sub>y</sub> p<sup>rom</sup>p<sup>t</sup>

![](images/3ae4cc255e29e8786216b5b9010b00edcdd019a185a85d08148488f3c1b801ab.jpg)

Modif<sub>y</sub> $e ^ { \mathrm { m o d } } ;$ : Put 21 in the last hexagon to continue the sequence.   
Pr<sub>ese</sub>r<sub>ve</sub> $e ^ { \mathrm { p r e } } ;$ Keep the first five numbered   
hexagons and the white background unchanged. O<sub>vera</sub>l $. e ^ { \mathrm { o v r } } ;$ Preserve the diagonal layout and a smooth increasing pattern.   
Tips �<sup>tip</sup>: Edit only the last hexagon and match the numeral style.

Checklist hits: Modify �<sup>mod</sup> : 2/4; Preserve   
�<sup>pre</sup> : 5/5.   
Overall �<sup>ovr</sup> : 3/4; Tips $C ^ { \mathrm { t i p } } : 4 / 4 .$   
Scores: Modify $s ^ { \mathrm { m o d } } = 0 . 5 0 ;$ Preserve $s ^ { \mathrm { p r e } } = 1 . 0 0 .$   
Overall $s ^ { \mathrm { o v r } } = 0 . 7 5 ;$ Tips $s ^ { \mathrm { t i p } } = 1 . 0 0 .$   
Gate: Modify gate $\sigma _ { \mathrm { m o d } } = 0 \Longrightarrow$ Preserve, Overall,   
and Tips blocked.   
R<sub>ewar</sub>d<sub>:</sub> $R _ { \mathrm { \scriptsize { p l a n } } } ^ { \mathrm { ( A ) } } = 0 . 2 0 .$

## P<sub>a</sub>n<sub>e</sub>l D<sub>.</sub> F<sub>a</sub>il<sub>u</sub>r<sub>e</sub> B: <sub>wea</sub>k Pr<sub>ese</sub>r<sub>ve</sub> p<sup>rom</sup>p<sup>t</sup>

![](images/b3b4142b52010544fb042e8ac613bf0e8889e83cfce374fc8a62eb6883a3529a.jpg)

Modif<sub>y</sub> $\cdot e ^ { \mathrm { m o d } } \colon$ Put 19 in the last hexagon as the next   
prime after 17.   
Pr<sub>ese</sub>r<sub>ve</sub> $e ^ { \mathrm { p r e } } ;$ Keep the sequence readable, but   
nearby numerals or canvas tones may be adjusted if   
needed.   
O<sub>vera</sub>ll $e ^ { \mathrm { { o v r } } } ;$ Preserve the diagonal layout and a   
plausible completed sequence.   
Tips �<sup>tip</sup>: Harmonize the final region with   
neighboring hexagons so the ending looks locally   
consistent.

Checklist hits: Modify $C ^ { \mathrm { m o d } }$ : 4/4; Preserve   
�<sup>pre</sup> : 2/5.   
Overal $C ^ { \mathrm { o v r } } : 3 / 4 ; \cdot$ Tips $C ^ { \mathrm { t i p } } : 3 / 4 .$   
Scores: Modify $s ^ { \mathrm { m o d } } = 1 . 0 0$ ; Preserve $s ^ { \mathrm { p r e } } = 0 . 4 0 .$   
Overall $s ^ { \mathrm { o v r } } = 0 . 7 5$ ; Tips $s ^ { \mathrm { t i p } } = 0 . 7 5 .$   
Gate: Modify gate $\sigma _ { \mathrm { m o d } } = 1 ,$ Preserve gate $\sigma _ { \mathrm { p r e } } = 0$   
⇒ Overall and Tips blocked.   
R<sub>ewar</sub>d<sub>:</sub> $R _ { \mathrm { \ p l a n } } ^ { \mathrm { ( B ) } } = 0 . 4 8 .$  
Figure S1: Five-panel schematic of checklist-based prefix-gated planner reward on a sequence-completion edit. Panels A–B define the shared input, checklist, and structured plan; Panels C–E illustrate Failure A, Failure B, and Success under the same scoring and gating rules. Detailed interpretation is provided in Sec. 2.4.

## Panel E. Success: <sub>p</sub>recise four-slot p<sup>rom</sup>p<sup>t</sup>

![](images/a0e5cb3bd1e56d386fd94f2ffd9f5b85a914d9c06ced4a6b55d6eb193416e77c.jpg)

```latex
Modif<sub>y</sub> $e ^ { \mathrm { m o d } } \colon$ Put 19 in the last hexagon as the next
prime after 17.
Pr<sub>ese</sub>r<sub>ve</sub> $e ^ { \mathrm { p r e } } ;$ Keep 5, 7, 11, 13, 17, and the white
background unchanged.
O<sub>vera</sub>l $. e ^ { \mathrm { { o v r } } } ;$ Preserve the prime progression, clean
spacing, and original diagonal layout.
Ti<sub>p</sub>s $e ^ { \mathrm { i i p } } \colon$ Edit only the last hexagon and match the
font, stroke, and centering.
```

Checklist hits: Modify $C ^ { \mathrm { m o d } }$ : 4/4; Preserve �<sup>pre</sup> : 5/5.   
Overall $C ^ { \mathrm { o v r } } : 4 / 4 ;$ Tips $C ^ { \mathrm { t i p } } : 4 / 4 .$   
Scores: Modify $s ^ { \mathrm { m o d } } = 1 . 0 0 ;$ Preserve $s ^ { \mathrm { p r e } } = 1 . 0 0 .$ Overall $s ^ { \mathrm { o v r } } = 1$ .00; Tips $s ^ { \mathrm { t i p } } = 1 . 0 0 .$   
Gate: Modify, Preserve, and Overall gates all pass $( \sigma _ { \mathrm { m o d } } = \sigma _ { \mathrm { p r e } } = \sigma _ { \mathrm { o v r } } = 1 )$   
R<sub>ewar</sub>d<sub>:</sub> $R _ { \mathrm { { p l a n } } } ^ { ( s \mathrm { { u c c e s s } } ) } = 1 . 0 0 .$

Table S1: Trainin<sub>g</sub> reci<sub>p</sub>e summar<sub>y</sub>.
<table><tr><td>Category</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="9">VLM Policy</td><td>Base model</td><td>Qwen3-VL-4B-Instruct</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times { 1 0 } ^ { - 6 }$ </td></tr><tr><td>Plans per input (M)</td><td>4</td></tr><tr><td>Total rollouts per input (M × K)</td><td>16</td></tr><tr><td>Global update batch size</td><td>256</td></tr><tr><td>KL penalty (β)</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Max context / response length</td><td>20,480 tokens / 4,096 tokens</td></tr><tr><td>Base model</td><td>Qwen-Image-Edit-2511</td></tr><tr><td rowspan="10">Diffusion Policy</td><td>Fine-tuning method</td><td>LoRA</td></tr><tr><td>LoRA rank / scaling factor (αLoRA)</td><td>64 /  128</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Flow-GRPO clip range (er)</td><td> $1 \times { 1 0 } ^ { - 4 }$ </td></tr><tr><td>Renderings per plan (K)</td><td>4</td></tr><tr><td>SDE training timesteps</td><td>{1, 2,3}</td></tr><tr><td>EMA decay / update interval</td><td>0.9 / 4 steps</td></tr><tr><td>Max image resolution</td><td>1024 × 1024</td></tr><tr><td></td><td>Inference steps (train / eval)</td><td>10 / 40</td></tr><tr><td>Guidance scale</td><td></td><td>4.0</td></tr></table>

## 3 Ex<sub>p</sub>erimental Setu<sub>p</sub> Details

## 3.1 Training Setu<sub>p</sub> an<sup>d</sup> H<sub>yp</sub>er<sub>p</sub>arameters

The main paper reports the shared training setup. Here we list only the policy-specific optimization settings needed to reproduce DARS; all shared quantities remain exactly as in the main text.

The planner uses a standard AdamW text-policy update to rewrite the raw instruction online into a four-slot plan, while the renderer uses LoRA-based flow-GRPO with DiT-specific sampling controls. Relative to the controlled Joint RL baseline, DARS adds sharedrollout bookkeeping and slot-wise scoring without additional renderer calls.

## 3<sub>.</sub>2 C<sub>on</sub>t<sub>ro</sub>ll<sub>e</sub>d B<sub>ase</sub>li<sub>ne an</sub>d E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> P<sub>ro</sub>t<sub>oco</sub>l

The controlled Joint RL + Adaptive Curriculum baseline matches DARS in backbones, training data, ofline checklist cache, online reward judge, rollout budget, hardware, and adaptive curriculum. It difers only in replacing the structured four-slot planner and dual-level credit assignment with a free-form planner trace.

All benchmark results are obtained from the oficial evaluation code and are independent ofthe training-time reward pipeline. Gem ini 3 Pro and Qwen3-VL-32B are used only for checklist construction and training-time scoring, not for the final reported benchmark metrics. Across all in-house variants, we report the oficial overall benchmark scores, with PICA-Bench following its standard simple / detailed split.

## 3.3 KRIS-Bench Breakdown

Table S2 expands the main-paper KRIS-Bench result into its oficial factual-, conceptual-, and procedural-knowledge groups, so performance diferences can be inspected at the subgroup level rather than only through the overall score. We also include the newly obtained fine-grained breakdown for Qwen-Image-Edit-2511 and update the best/second-best markings accordingly.

## 3.4 RISE-Bench Breakdown

Table S3 expands the main-paper RISE-Bench result into the four oficial reasoning categories of RISEBench-360: Temporal, Causal, Spatial, and Logical. This breakdown shows where gains come from across diferent reasoning types rather than only in the aggregate.

## 3<sub>.</sub>5 I<sub>mg</sub>Edit<sub>-</sub>B<sub>enc</sub>h B<sub>rea</sub>kd<sub>own</sub>

Table S4 expands the main-paper ImgEdit-Bench result into its nine oficial edit-operation groups, making it easier to see where diferent methods are strong or weak across common editing regimes rather than only in the aggregate. All entries follow the benchmark’s oficial GPT-4.1-based evaluation protocol.

## 3<sub>.</sub>6 GEdit-B<sub>e</sub>n<sub>c</sub>h-EN Br<sub>ea</sub>kd<sub>ow</sub>n

Table S5 expands the main-paper GEdit-Bench-EN result into its oficial split-wise metrics, so the relative behavior of diferent methods can be inspected separately on the Intersection subset and the Full set. All numbers come from the oficial evaluation code.

## 3.7 PICA-Bench Sim<sub>p</sub>le-Prom<sub>p</sub>t Breakdo<sub>w</sub>n

Table S6 reports the fine-grained PICA-Bench results under the benchmark’s simple prompt setting, which we also refer to as the superficial prompt setting for clarity. We list all oficial physicalconsistency categories: LP, LSE, Reflection, Refraction, Deformation, Causality, GST, and LST, together with the overall score. As with the other supplementary benchmark tables, all values come from the oficial evaluation pipeline and are reported with two decimal places.

## 3<sub>.</sub>8 PICA-B<sub>e</sub>n<sub>c</sub>h Ex<sub>p</sub>li<sub>c</sub>it-Pr<sub>o</sub>m<sub>p</sub>t Br<sub>ea</sub>kd<sub>ow</sub>n

Table S7 reports the fine-grained PICA-Bench results under the benchmark’s detailed prompt setting, which corresponds to the explicit prompt protocol used in our evaluation pipeline. We keep the same category layout as in Table S6: LP, LSE, Reflection, Refraction, Deformation, Causality, GST, LST, and the overall score. All values come from the oficial evaluation pipeline and are reported with two decimal places.

## 3<sub>.</sub>9 B<sub>ase</sub>li<sub>ne</sub> A<sub>rc</sub>hit<sub>ec</sub>t<sub>ure</sub> O<sub>verv</sub>i<sub>ew</sub>

Table S8 summarizes the main backbone composition and dominant control mechanism of each baseline based on the released model descriptions and our implementation notes.

## 4 F<sub>a</sub>il<sub>ure</sub> Att<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> V<sub>a</sub>lid<sub>a</sub>ti<sub>on</sub> D<sub>e</sub>t<sub>a</sub>il<sub>s</sub>

We sample 500 evaluation cases for the attribution study, with an even split between relatively simple and dificult edits (250 each), so the validation set covers both low-dificulty and high-dificulty failure modes. For each case, GPT-5 produces five independent failure-attribution labels, and the final case-level pseudo-label is the majority vote. The system prompt used for this annotation step is shown in Fig. S6.

The GPT-5 annotator returns one ofthree raw labels: planner\_side, renderer\_side, or both. In the paper, these are reported respectively as planner-dominant, renderer-dominant, and mixed. After obtaining these case-level three-way pseudo-labels, we form three

Table S2: Fine-grained KRIS-Bench results. We use the oficial GPT-4.1-based evaluation protocol. KRIS-Bench evaluates edited results with four dimensions: visual consistency, visual quality, instruction following, and knowledge plausibility, where the last dimension is used only for knowledge-intensive subsets such as Social Science, Natural Science, and Logical Reasoning. The raw judge outputs are defined on a 1–5 scale, while the benchmark reports normalized percentage scores with maximum 100. Higher is better for every metric. All values are reported with two decimal places.
<table><tr><td rowspan="2">Model</td><td colspan="4">Factual Knowledge</td><td colspan="3">Conceptual Knowledge</td><td colspan="3">Procedural Knowledge</td><td rowspan="2">Overall ↑</td></tr><tr><td>Attribute Perception ↑</td><td>Spatial Perception ↑</td><td>Temporal Perception ↑</td><td>Average Score ↑</td><td>Social Science ↑</td><td>Natural Science ↑</td><td>Average Score ↑</td><td>Logical Reasoning ↑</td><td>Instruction Decomposition ↑</td><td>Average Score ↑</td></tr><tr><td colspan="10">Other baselines</td></tr><tr><td>ThinkGen</td><td>64.03</td><td>55.17</td><td>56.31</td><td>59.57</td><td>66.00</td><td>63.19</td><td>63.86</td><td>38.04</td><td>64.83</td><td>49.52</td><td>59.57</td></tr><tr><td>UniREdit</td><td>61.10</td><td>66.75</td><td>58.45</td><td>62.42</td><td>66.20</td><td>63.04</td><td>63.80</td><td>53.08</td><td>54.61</td><td>53.74</td><td>61.02</td></tr><tr><td>PromptRL</td><td>67.85</td><td>64.83</td><td>79.17</td><td>69.04</td><td>58.65</td><td>57.39</td><td>57.69</td><td>51.38</td><td>67.29</td><td>58.24</td><td>61.24</td></tr><tr><td>Step1X-Edit-v1p2</td><td>69.67</td><td>64.33</td><td>80.18</td><td>70.21</td><td>63.50</td><td>60.62</td><td>61.32</td><td>49.75</td><td>62.67</td><td>55.29</td><td>62.58</td></tr><tr><td>UniReason 1.0</td><td>64.03</td><td>68.08</td><td>71.28</td><td>66.13</td><td>66.75</td><td>67.26</td><td>67.13</td><td>50.50</td><td>53.89</td><td>51.95</td><td>63.26</td></tr><tr><td>ThinkRL-Edit</td><td>68.18</td><td>67.75</td><td>87.50</td><td>71.27</td><td>68.85</td><td>66.49</td><td>67.06</td><td>51.63</td><td>64.94</td><td>57.33</td><td>66.04</td></tr><tr><td>PhysicEdit</td><td>67.52</td><td>74.42</td><td>84.91</td><td>71.92</td><td>67.25</td><td>68.10</td><td>67.89</td><td>48.00</td><td>71.28</td><td>57.98</td><td>66.78</td></tr><tr><td>EditThinker</td><td>73.67</td><td>71.75</td><td>81.53</td><td>74.54</td><td>71.85</td><td>67.95</td><td>68.89</td><td>56.04</td><td>68.22</td><td>61.26</td><td>68.80</td></tr><tr><td colspan="10">Qwen-Image-Edit-2511-based methods</td><td></td></tr><tr><td>Qwen-Image-Edit-2511</td><td>68.82</td><td>74.75</td><td>77.70</td><td>71.60</td><td>61.75</td><td>58.48</td><td>59.27</td><td>56.21</td><td>81.83</td><td>67.19</td><td>64.42</td></tr><tr><td>RePlan</td><td>56.88</td><td>50.00</td><td>74.77</td><td>58.26</td><td>53.45</td><td>56.22</td><td>55.55</td><td>49.58</td><td>57.10</td><td>52.79</td><td>55.72</td></tr><tr><td>PromptEnhancerV2</td><td>71.67</td><td>75.67</td><td>82.43</td><td>74.33</td><td>71.40</td><td>70.59</td><td>70.79</td><td>56.63</td><td>86.50</td><td>69.43</td><td>71.54</td></tr><tr><td>DARS</td><td>78.21</td><td>84.33</td><td>85.36</td><td>80.75</td><td>82.95</td><td>81.20</td><td>81.62</td><td>71.00</td><td>89.39</td><td>78.89</td><td>80.72</td></tr></table>

Table S3: Fine-grained RISE-Bench results. We evaluate on RISEBench-360, the 360-sample version of the benchmark, using the oficial GPT-4.1-based judging pipeline. The benchmark scores each sample along instruction reasoning, appearance consistency, and visual plausibility, and counts a case as solved only when all applicable dimensions receive full marks. The table therefore reports Accuracy (%), i.e., the task success rate, for each reasoning category and for the overall set. Higher is better for every metric.
<table><tr><td>Method</td><td>Temporal ↑</td><td>Causal ↑</td><td>Spatial ↑</td><td>Logical ↑</td><td>Overall ↑</td></tr><tr><td colspan="6">Other baselines</td></tr><tr><td>ThinkGen</td><td>15.29</td><td>16.67</td><td>7.00</td><td>3.52</td><td>10.56</td></tr><tr><td>UniREdit</td><td>15.29</td><td>20.00</td><td>14.00</td><td>3.53</td><td>13.33</td></tr><tr><td>PromptRL</td><td>3.53</td><td>8.89</td><td>23.00</td><td>2.34</td><td>10.00</td></tr><tr><td>Step1X-Edit-v1p2</td><td>14.12</td><td>13.30</td><td>13.00</td><td>5.88</td><td>11.67</td></tr><tr><td>UniReason 1.0</td><td>16.47</td><td>27.78</td><td>13.00</td><td>2.35</td><td>15.00</td></tr><tr><td>ThinkRL-Edit</td><td>14.11</td><td>20.00</td><td>17.00</td><td>5.88</td><td>14.44</td></tr><tr><td>PhysicEdit</td><td>15.29</td><td>27.78</td><td>22.00</td><td>9.41</td><td>18.89</td></tr><tr><td>EditThinker</td><td>18.82</td><td>18.89</td><td>27.00</td><td>7.06</td><td>18.33</td></tr><tr><td colspan="6">Qwen-Image-Edit-2511-based methods</td></tr><tr><td>Qwen-Image-Edit-2511</td><td>20.80</td><td>18.25</td><td>30.67</td><td>4.65</td><td>17.50</td></tr><tr><td>RePlan</td><td>10.59</td><td>12.22</td><td>11.00</td><td>4.70</td><td>9.72</td></tr><tr><td>PromptEnhancerV2</td><td>20.00</td><td>30.00</td><td>18.00</td><td>7.06</td><td>18.90</td></tr><tr><td>DARS</td><td>30.59</td><td>38.89</td><td>31.00</td><td>8.24</td><td>27.50</td></tr></table>

pairwise binary tasks by filtering the same 500-case pool into Planner vs. Renderer, Planner vs. Mixed, and Renderer vs. Mixed; we do not resample separately for the binary tasks, so the retained sample count in each setting is determined by the realized three-way label frequencies in this shared evaluation pool.

Accuracy and Macro-F1 are computed within each resulting binary task, and AUROC is reported in the same one-vs.-one setting rather than under a three-way formulation. GPT-5 sees only the source image, the structured plan, and the edited image for each sampled triplet; it does not see rollout statistics, checklist-based rewards, or slot scores.

## 5 Additional Qualitative Results

This section adds three complementary qualitative views: additional baseline comparisons, more reasoning-intensive edits, and more general editing cases.

Table S4: Fine-grained ImgEdit-Bench results. The header enumerates the benchmark’s nine common editing tasks: Add, Remove, Adjust, Replace, Style, Background, Action, Hybrid, and Extract. ImgEdit-Bench evaluates each result with GPT-4.1 on a 1–5 scale from three perspectives: instruction following, editing quality, and detail preservation. Instruction following measures understanding of the prompt and target concept; editing quality measures how accurately the target region is manipulated; and detail preservation measures fidelity on content that should remain unchanged. Because instruction following is foundational and cannot be fully separated from the other two aspects, the editing-quality and detail-preservation scores are upper-bounded by the instruction-following score. The table reports the average score for each edit category, and higher is better for every metric.
<table><tr><td>Method</td><td>Add ↑</td><td>Adjust ↑</td><td>Extract ↑</td><td>Replace ↑</td><td>Remove ↑</td><td>Background ↑</td><td>Style ↑</td><td>Hybrid ↑</td><td>Action ↑</td><td>Overall</td></tr><tr><td colspan="9">Other baselines</td></tr><tr><td>ThinkGen</td><td>4.20</td><td>3.96</td><td>3.45</td><td>4.35</td><td>3.35</td><td>4.22</td><td>4.91</td><td>3.45</td><td>4.16</td><td>3.97</td></tr><tr><td>UniREdit</td><td>4.09</td><td>3.53</td><td>2.27</td><td>4.27</td><td>3.81</td><td>3.83</td><td>4.65</td><td>2.46</td><td>4.37</td><td>3.70</td></tr><tr><td>PromptRL</td><td>4.07</td><td>4.19</td><td>2.95</td><td>4.23</td><td>3.67</td><td>4.09</td><td>4.46</td><td>3.04</td><td>4.13</td><td>3.87</td></tr><tr><td>Step1X-Edit-v1p2</td><td>4.28</td><td>4.40</td><td>2.29</td><td>4.36</td><td>4.17</td><td>3.94</td><td>4.74</td><td>3.61</td><td>4.23</td><td>4.00</td></tr><tr><td>UniReason 1.0</td><td>4.18</td><td>3.78</td><td>2.65</td><td>4.50</td><td>4.39</td><td>4.01</td><td>4.73</td><td>2.78</td><td>4.36</td><td>4.06</td></tr><tr><td>ThinkRL-Edit</td><td>4.23</td><td>4.18</td><td>3.51</td><td>4.71</td><td>4.08</td><td>4.09</td><td>4.89</td><td>3.59</td><td>4.42</td><td>4.19</td></tr><tr><td>PhysicEdit</td><td>2.67</td><td>3.92</td><td>2.67</td><td>4.57</td><td>4.35</td><td>3.97</td><td>4.79</td><td>3.33</td><td>4.56</td><td>4.05</td></tr><tr><td>EditThinker</td><td>4.33</td><td>4.17</td><td>3.91</td><td>4.66</td><td>4.18</td><td>4.29</td><td>4.90</td><td>3.65</td><td>4.68</td><td>4.31</td></tr><tr><td colspan="9">Qwen-Image-Edit-2511-based methods</td><td></td></tr><tr><td>Qwen-Image-Edit-2511</td><td>4.55</td><td>4.48</td><td>4.21</td><td>4.62</td><td>4.29</td><td>4.23</td><td>4.90</td><td>3.37</td><td>4.63</td><td>4.36</td></tr><tr><td>RePlan</td><td>4.11</td><td>3.78</td><td>2.07</td><td>3.41</td><td>3.02</td><td>3.85</td><td>4.48</td><td>3.07</td><td>3.16</td><td>3.44</td></tr><tr><td>PromptEnhancerV2</td><td>4.28</td><td>4.40</td><td>3.43</td><td>4.43</td><td>4.18</td><td>4.27</td><td>4.77</td><td>3.61</td><td>4.31</td><td>4.18</td></tr><tr><td>DARS</td><td>4.54</td><td>4.47</td><td>4.18</td><td>4.63</td><td>4.32</td><td>4.24</td><td>4.90</td><td>3.61</td><td>4.61</td><td>4.39</td></tr></table>

Table S5: Fine- rained GEdit-Bench-EN results. The header uses the oficial benchmark notation: G\_SC denotes GPT-4.1-based semantic consistency, G\_PQ denotes GPT-4.1-based perceptual quality, and G\_O denotes the oficial overall score. Semantic consistency evaluates how well the edited result follows the instruction, while perceptual quality evaluates image naturalness and artifact level; both use a 0–10 scale “Intersection subset” refers to the subset of test images for which all compared models return valid outputs, whereas “Full set” refers to the complete GEdit-Bench-EN test set. All values are reported with two decimal places.
<table><tr><td rowspan="2">Method</td><td colspan="3">Intersection subset</td><td colspan="3">Full set</td></tr><tr><td>G_SC ↑</td><td>G_PQ↑</td><td>G_O↑</td><td>G_SC ↑</td><td>G_PQ↑</td><td>G_O↑</td></tr><tr><td colspan="7">Other baselines</td></tr><tr><td>ThinkGen</td><td>7.49</td><td>7.80</td><td>7.22</td><td>7.43</td><td>7.76</td><td>7.16</td></tr><tr><td>UniREdit</td><td>7.31</td><td>6.96</td><td>6.67</td><td>7.10</td><td>6.89</td><td>6.47</td></tr><tr><td>PromptRL</td><td>7.32</td><td>7.60</td><td>6.90</td><td>7.15</td><td>7.58</td><td>6.74</td></tr><tr><td>Step1X-Edit-v1p2</td><td>8.32</td><td>7.89</td><td>7.73</td><td>8.05</td><td>7.85</td><td>7.47</td></tr><tr><td>UniReason 1.0</td><td>7.18</td><td>7.21</td><td>6.62</td><td>7.06</td><td>7.20</td><td>6.52</td></tr><tr><td>ThinkRL-Edit</td><td>6.84</td><td>8.21</td><td>6.63</td><td>6.70</td><td>8.16</td><td>6.48</td></tr><tr><td>PhysicEdit</td><td>8.08</td><td>7.76</td><td>7.54</td><td>7.93</td><td>7.66</td><td>7.37</td></tr><tr><td>EditThinker</td><td>8.20</td><td>7.84</td><td>7.64</td><td>8.12</td><td>7.88</td><td>7.59</td></tr><tr><td colspan="7">Qwen-Image-Edit-2511-based methods</td></tr><tr><td>Qwen-Image-Edit-2511</td><td>7.67</td><td>7.68</td><td>7.24</td><td>7.65</td><td>7.38</td><td>6.97</td></tr><tr><td>RePlan</td><td>7.03</td><td>7.66</td><td>6.55</td><td>6.87</td><td>7.62</td><td>6.43</td></tr><tr><td>PromptEnhancerV2</td><td>8.33</td><td>7.97</td><td>7.79</td><td>8.08</td><td>7.96</td><td>7.59</td></tr><tr><td>DARS</td><td>8.57</td><td>8.01</td><td>7.98</td><td>8.49</td><td>8.00</td><td>7.86</td></tr></table>

Additional baseline comparisons. Figure S2 adds side-by-side comparisons on challenging edits involving temporal prediction, logical diagram reasoning, path planning, object removal, and symbolic transformation. These cases make it easier to inspect where DARS preserves the target semantics more reliably than the omitted baselines.

Instruction

Draw what it will look like after 30 seconds in summer.

Draw what it wil look like ten seconds later.

Draw what it   
will look like   
after 30   
minutes.   
The caterpillar   
wants to find the   
shortest path to the   
leaf. Please mark   
the two shortest   
paths in the diagram.

Remove the water from the glass cup.

Move only two matchsticks on the left side of the equation to make the equation true. Draw the final equation.

![](images/ab2a4f966fdff5de06c00876feb10a5c7e66604a48190fcbd7814c0c84f3df14.jpg)

Figure S2: Additional baseline comparisons on challenging edits. The figure adds side-by-side results for several baselines under the same evaluation setting.  
![](images/7dde3fd6eea5bb96aec0378a03c758b98fcfb368e43ce63166af6b5ac9bf1efa.jpg)  
Figure S3: Representative failure cases illustrating current limitations. From left to right, the examples highlight failure modes on puzzle completion, dense final-state reasoning in a mechanics-style diagram, precise relative-size control across multiple objects, and maze-path drawing. Together they show that the method remains weaker when the desired target is dificult to specify textually, when the edit depends on tightly coupled multi-step reasoning, or when exact geometric control is required.

Table S6: Fine-grained PICA-Bench results under the simple-prompt setting. This setting corresponds to the superficial-prompt protocol in PICA-Bench and is evaluated by GPT-4.1 for instruction-based image editing models. All reported numbers are Accuracy (%). In the header, LP, LSE, GST, and LST denote Light Propagation, Light Source Efects, Global State Transition, and Local State Transition, respectively.
<table><tr><td>Method</td><td>LP↑</td><td>LSE ↑</td><td>Reflection ↑</td><td>Refraction ↑</td><td>Deformation ↑</td><td>Causality ↑</td><td>GST ↑</td><td>LST↑</td><td>Overall ↑</td></tr><tr><td colspan="10">Other baselines</td></tr><tr><td>ThinkGen</td><td>64.39</td><td>66.34</td><td>64.51</td><td>65.46</td><td>50.06</td><td>51.26</td><td>68.38</td><td>66.34</td><td>60.82</td></tr><tr><td>UniREdit</td><td>61.99</td><td>66.17</td><td>67.50</td><td>54.38</td><td>53.11</td><td>52.90</td><td>65.68</td><td>56.83</td><td>60.40</td></tr><tr><td>PromptRL</td><td>56.04</td><td>52.24</td><td>59.27</td><td>41.35</td><td>50.42</td><td>45.60</td><td>52.91</td><td>45.67</td><td>51.02</td></tr><tr><td>Step1X-Edit-v1p2</td><td>63.52</td><td>58.12</td><td>66.61</td><td>62.44</td><td>53.41</td><td>48.76</td><td>52.60</td><td>51.89</td><td>56.15</td></tr><tr><td>UniReason 1.0</td><td>62.15</td><td>63.20</td><td>62.73</td><td>51.35</td><td>52.55</td><td>48.84</td><td>60.21</td><td>54.76</td><td>57.30</td></tr><tr><td>ThinkRL-Edit</td><td>60.96</td><td>56.68</td><td>61.18</td><td>41.16</td><td>56.92</td><td>46.07</td><td>59.53</td><td>52.66</td><td>55.13</td></tr><tr><td>PhysicEdit</td><td>63.10</td><td>69.26</td><td>67.34</td><td>51.96</td><td>58.31</td><td>51.47</td><td>65.79</td><td>61.51</td><td>61.48</td></tr><tr><td>EditThinker</td><td>63.05</td><td>59.98</td><td>65.75</td><td>53.35</td><td>55.34</td><td>50.76</td><td>64.50</td><td>60.98</td><td>59.64</td></tr><tr><td colspan="10">Qwen-Image-Edit-2511-based methods</td></tr><tr><td>Qwen-Image-Edit-2511</td><td>63.89</td><td>69.91</td><td>71.86</td><td>62.33</td><td>57.93</td><td>50.70</td><td>67.89</td><td>61.34</td><td>63.20</td></tr><tr><td>RePlan</td><td>52.99</td><td>51.29</td><td>49.29</td><td>48.88</td><td>52.33</td><td>43.82</td><td>61.26</td><td>52.15</td><td>52.00</td></tr><tr><td>PromptEnhancerV2</td><td>66.86</td><td>66.19</td><td>70.86</td><td>59.55</td><td>53.71</td><td>53.06</td><td>62.90</td><td>63.38</td><td>61.90</td></tr><tr><td>DARS</td><td>62.52</td><td>67.85</td><td>72.04</td><td>64.62</td><td>64.37</td><td>53.59</td><td>65.50</td><td>66.16</td><td>64.19</td></tr></table>

Table S7: Fine-grained PICA-Bench results under the explicit-prompt setting. All reported numbers are Accuracy (%). All values are reported with two decimal places.
<table><tr><td>Method</td><td>LP↑</td><td>LSE ↑</td><td>Reflection ↑</td><td>Refraction ↑</td><td>Deformation ↑</td><td>Causality ↑</td><td>GST↑</td><td>LST↑</td><td>Overall Score ↑</td></tr><tr><td>Other baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ThinkGen</td><td>66.17</td><td>68.29</td><td>64.82</td><td>62.63</td><td>59.43</td><td>58.57</td><td>74.48</td><td>61.12</td><td>65.09</td></tr><tr><td>UniREdit</td><td>64.34</td><td>66.80</td><td>67.50</td><td>59.77</td><td>63.72</td><td>62.17</td><td>72.81</td><td>68.41</td><td>66.55</td></tr><tr><td>PromptRL</td><td>61.20</td><td>59.68</td><td>63.66</td><td>43.02</td><td>55.11</td><td>62.88</td><td>66.96</td><td>55.50</td><td>60.38</td></tr><tr><td>Step1X-Edit-v1p2</td><td>58.92</td><td>61.60</td><td>66.02</td><td>49.12</td><td>56.68</td><td>59.30</td><td>61.36</td><td>63.91</td><td>60.55</td></tr><tr><td>UniReason 1.0</td><td>59.32</td><td>64.85</td><td>60.03</td><td>49.77</td><td>56.47</td><td>60.83</td><td>69.23</td><td>61.09</td><td>61.62</td></tr><tr><td>ThinkRL-Edit</td><td>57.20</td><td>55.39</td><td>57.77</td><td>37.63</td><td>48.89</td><td>54.39</td><td>62.06</td><td>58.94</td><td>55.84</td></tr><tr><td>PhysicEdit</td><td>64.39</td><td>70.14</td><td>68.36</td><td>54.10</td><td>65.90</td><td>70.29</td><td>74.61</td><td>70.79</td><td>68.96</td></tr><tr><td>EditThinker</td><td>66.79</td><td>64.20</td><td>68.21</td><td>48.67</td><td>60.49</td><td>67.03</td><td>74.42</td><td>71.19</td><td>67.19</td></tr><tr><td colspan="10">Qwen-Image-Edit-2511-based methods</td></tr><tr><td>Qwen-Image-Edit-2511</td><td>67.36</td><td>75.82</td><td>73.87</td><td>59.67</td><td>66.02</td><td>71.07</td><td>78.58</td><td>73.99</td><td>72.27</td></tr><tr><td>RePlan</td><td>46.84</td><td>56.27</td><td>55.25</td><td>51.57</td><td>49.35</td><td>49.87</td><td>62.66</td><td>60.77</td><td>54.92</td></tr><tr><td>PromptEnhancerV2</td><td>67.46</td><td>74.01</td><td>74.00</td><td>54.25</td><td>63.32</td><td>70.37</td><td>76.29</td><td>72.62</td><td>70.77</td></tr><tr><td>DARS</td><td>67.79</td><td>74.97</td><td>75.79</td><td>60.46</td><td>64.92</td><td>70.79</td><td>78.21</td><td>73.32</td><td>72.75</td></tr></table>

Additional reasoning-intensive edits. Figure S4 shows that the advantage of DARS remains visible across several reasoning-heavy regimes, including rule-based game solving (e.g., rock-paper-scissors and tic-tac-toe), commonsense inference edits such as tip calculation and identifying the lightning rod as the protective facility to remove, and object completion or correction tasks involving missing or unreasonable structures.

Additional general editing cases. Figure S5 shows that the benefit of DARS is not limited to explicitly puzzle-like instructions. Across more diverse general editing cases with richer object composition and more complex inter-object relationships, DARS consistently demonstrates strong fine-grained control over local edits while preserving non-target content. These examples further suggest that the method maintains robust scene understanding and object-level recognition even in visually complex images, enabling precise edits without sacrificing overall consistency.

T<sub>a</sub>bl<sub>e</sub> S8<sub>:</sub> D<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d b<sub>ase</sub>li<sub>ne arc</sub>hit<sub>ec</sub>t<sub>ure summary.</sub>
<table><tr><td>Method</td><td>Main Backbone</td><td>Key Structural Characteristics</td></tr><tr><td>ThinkRL-Edit</td><td>Qwen3-VL-30B-A3B + Qwen-Image-Edit</td><td>Checklist-style reward-driven RL with explicit CoT planning and reflection before image-to-image diffusion editing.</td></tr><tr><td>ThinkGen</td><td>Qwen3-VL-8B-Think + OmniGen2-4B DiT</td><td>Alternating GRPO between a reasoning VLM and a DiT generator.</td></tr><tr><td>Step1X-Edit-v1p2</td><td>Qwen2.5-VL-7B + Step1X-Edit</td><td>Multi-round reflection pipeline (up to five rounds) with coupled VLM and editor tuning.</td></tr><tr><td>RePlan</td><td>Qwen2.5-VL-7B + Qwen-Image-Edit-2511</td><td>Region-aware decomposition: an RL-trained VLM predicts boxes and localized prompts for editing.</td></tr><tr><td>EditThinker</td><td>Qwen3-VL-8B + Qwen-Image-Edit</td><td>Up to five rounds of reflection, with reasoning refinement concentrated on the VLM side.</td></tr><tr><td>UniReason 1.0</td><td>Bagel-14B Qwen2.5-VL-3B + FLUX</td><td>World-knowledge-aware prompt enhancement with reflection and consistency control.</td></tr><tr><td>PromptRL</td><td>.1-Kontext-12B</td><td>Joint RL over the prompt-side VLM and the image editor.</td></tr><tr><td>UniREdit</td><td>Bagel-14B</td><td>Uses large-scale text reasoning data together with mixed real-world and game-world supervision for reasoning-oriented editing.</td></tr><tr><td>PhysicEdit</td><td>PhysicEdit data + Qwen-Image-Edit-2509</td><td>Physically grounded reasoning with meta-query-based control for physics-aware image editing.</td></tr><tr><td>PromptEnhancerV2 -</td><td>Qwen2.5-VL-32B +</td><td>Prompt-enhancement front-end paired with a strong external image editor; the public description mainly exposes the front-end/control structure rather than the full joint</td></tr><tr><td>Img2Img Edit</td><td>Qwen-Image-Edit-2511</td><td>training recipe. Strong native end-to-end image editing model used as a general-purpose reference</td></tr><tr><td>Qwen-Image-Edit-2511Qwen-Image-Edit-2511</td><td></td><td>baseline.</td></tr></table>

## 5.1 Structured Plans for Qualitative Cases

Table S9 lists the four-slot decompositions for representative cases in Figs. S2–S5, so readers can directly inspect the semantic allocation across Modify, Preserve, Overall, and Tips.

## 6 Prom<sub>p</sub>t Tem<sub>p</sub>lates

We collect here the prompt templates used in three places in the pipeline. Figure S6 is used only in the GPT-5 failure-attribution validation study. Figure S7 is the planner-side four-slot rewriting prompt used during rollout to produce e. Figure S8 is the ofline Gemini 3 Pro prompt used to generate the cached shared checklist C(x). During RL itself, Qwen3-VL-32B is used only for online singlequestion Yes/No scoring.

Validation prompt. We first show the GPT-5 prompt used only for the failure-attribution validation study in Section 4. It is not part of the training or preprocessing pipeline.

Planner rollout and checklist-generation prompts. We next show the two prompt families that create the semantic inputs to online scoring: the planner-side four-slot rewriting prompt used during rollout, and the Gemini 3 Pro prompt used before RL to generate the cached checklist target.

Online reward-scoring prompts. After the planner produces a rollout-time four-slot plan and the checklist is loaded from the ofline cache, reward evaluation proceeds through single-question prompts rather than a monolithic judgment. Each API call evaluates exactly one checklist question for exactly one slot, using Qwen3- VL-32B as the online Yes/No judge during RL training. The slot text in these prompts comes from the current planner’s rollout-time rewrite, while the checklist question comes from the ofline Gemini 3 Pro cache.

Planner-side prompts are post-render diagnostics: they ask whether the current slot text functioned as a good plan for the observed edit under one checklist item.

Renderer-side prompts retain the raw instruction and relevant slot text as specification context but change the target: the judge scores whether the final rendered image visibly satisfies each checklist question, not whether the slot text is a good plan.

## 7 Limitations

Despite the gains reported in the main paper and the supplementary visualizations above, the current framework still has several clear limitations. One recurring dificulty arises when the desired output is hard to specify compactly in natural language and is more naturally described by a structured visual target. This includes tasks such as maze solving, puzzle completion, or other diagrammatic edits where the model must infer and render a precise final configuration rather than only apply a local semantic modification.

A second limitation appears in cases that require either very dense chained reasoning or very precise geometric control. As shown in Fig. S3, representative failure modes include incomplete puzzle reconstruction, incorrect final-state reasoning in mechanicsstyle diagrams, inaccurate relative-size normalization across multiple objects, and unstable path drawing in mazes. These examples suggest that the current planner–renderer decomposition is still less reliable when the edit depends on long reasoning chains, diagramlevel state transitions, or exact multi-object scale relationships. Future progress will likely require stronger, structured intermediate representations, richer supervision for diagrammatic reasoning, and more explicit mechanisms for geometric control.

Table S9: Actual four-slot plan decompositions for representative qualitative cases. Each row lists the raw user instruction together with its Modify, Preserve, Overall, and Tips fields in the same slot schema used throughout the paper.
<table><tr><td rowspan=1 colspan=29>ID Raw          Modify                        Preserve                    Overall                     Tips</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=8 colspan=3>Show the dry leaf catching fire underDraw what it willthe concentrated sunlight from thelook like after 30magnifying glass, with a small flameseconds inand a plume of grey smoke rising from the lens, and the overall sunnysummer.Draw what it will seconds later by removing the2</td><td rowspan=1 colspan=1>Preserve the surrounding green                               Add a charred, blackened textureThe scene should depict the earlygrass, the hand holding the                                  around the focal point and ensure the</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=11>stage of combustion in a visuallycoherent and physically plausibleway under strong summer sunlight.</td><td rowspan=6 colspan=5>smoke and flame originate naturallyfrom the concentrated light spot onthe brown leaf.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>daylight illumination.</td></tr><tr><td></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2>Edit the image to depict the scene ten</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=5 colspan=1>Preserve the backgroundmountains, the tree-covered hills,the cloudy sky, the soft natural</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=1>mountains, the</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=14 colspan=11>The final image should look calm,smooth, and undisturbed, as if thesplash event has fully settled into aserene static scene.The result should convey a realisticlong-boiling, overheating situationwith coherent interactions betweenmilk, pot, stove, steam, and smoke.</td><td rowspan=14 colspan=5>and ensure the reflections on the stillwater align naturally with thesurrounding landscape.Add bubbly and browned residuealong the inner rim, mix darker smokewith the steam, and make the spilledmilk follow gravity and pool plausiblyon the stove surfaces.Draw one path across the front-left</td></tr><tr><td></td><td rowspan=1 colspan=4>Draw what it will</td><td rowspan=1 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=4>look like ten</td><td rowspan=1 colspan=2>suspended water droplets, the central</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=5>e reflections on the still</td></tr><tr><td></td><td rowspan=1 colspan=4>seconds later.</td><td rowspan=1 colspan=2>splash pillar, and the concentric ripple</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=2></td><td rowspan=5 colspan=1></td><td rowspan=5 colspan=1>lighting, and the original lakesetting.Preserve the shape and material of</td></tr><tr><td></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2>rings.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=4></td><td rowspan=2 colspan=2>Change the scene to show the pot afterthe milk has been boiling for 30</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=4></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=2></td></tr><tr><td></td><td rowspan=1 colspan=4>Draw what it will</td><td rowspan=1 colspan=2>minutes, with messy frothy milk spills</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>the pot, the v</td></tr><tr><td></td><td rowspan=1 colspan=4>look like after 30</td><td rowspan=1 colspan=2>down the black pot and onto the gas</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=5>nd make the spilled</td></tr><tr><td></td><td rowspan=1 colspan=4>minutes.</td><td rowspan=1 colspan=2>stove grate, along with a reduced,</td><td rowspan=1 colspan=3></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=3>nal warr</td><td rowspan=3 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2>thickened, overheated surface inside</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>lighting</td></tr><tr><td></td><td rowspan=1 colspan=4></td><td rowspan=3 colspan=2>the pot.Mark the two shortest surface paths</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>The caterpillar</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>wants to find the</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=2>from the orange caterpillar at thebottom front corner of the cube to the</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=7 colspan=11>,clean diagrammatic solution thatremains geometrically consistentwith the cube&#x27;s perspective.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=7 colspan=5>the front-right face and top face,rendering both lines clearly andaligned to the cube surface geometry.Turn the 8 into a 3 by removing the</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>green leaf on the top face using two</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>mark the two</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>distinct overlaid lines.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>shortest paths in</td><td rowspan=2 colspan=1>as they are.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2></td><td></td><td></td><td></td><td rowspan=4 colspan=9></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>the diagram.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=2></td><td></td><td></td><td></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>Move only two</td><td rowspan=2 colspan=1></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=5></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=8></td><td rowspan=2 colspan=6>Transform the left side of thematchstick equation so the final resul</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=5> The final image r</td><td rowspan=1 colspan=2>e must</td><td rowspan=1 colspan=6></td><td rowspan=1 colspan=5>two left vertical matches, turn the 7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11>coherent, mathematically correct,</td><td rowspan=1 colspan=5>into a 3 by adding a middle and</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>equation to make</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=2>reads 3 - 3 = 0 by changing the two leftdigits accordingly.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11>and well-aligned matchstick</td><td rowspan=1 colspan=5>bottom horizontal match, and match</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>the equation true.</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=7>le.</td><td rowspan=1 colspan=11>equation.</td><td rowspan=6 colspan=5>the existing dark match heads andrectangular shafts.Eliminate optical refraction,magnification, and distortion from the</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=11>The final result should look like a</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>Remove the water from the glass cup so Pr</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>eserve the “HEAD ATP&quot; text, the</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=2>e glass interior appears empty and fuzz</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=5 colspan=11>alistic dry glass container holdingthe ball naturally at the bottomunder the same viewing conditions</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=5></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=5>ball, while maintaining the original</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>dry, and the tennis ball appears without</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=1></td><td rowspan=1 colspan=1>from the glass cup.</td><td rowspan=2 colspan=1>and material of the clear glass cup,and the bright cyan background.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>liquid-induced distortion.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=5>lighting, shadows, and cameraviewpoint.Redraw the bottom hand as a fist with</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2></td><td></td><td></td><td></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=7></td><td rowspan=4 colspan=11>The final image should remain aconsistent rock-paper-scissors</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=2>game. Change one Change the bottom hand gesture from sc</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>paper-scissors</td><td rowspan=2 colspan=1>Preserve the top-left hand showingissors and the right hand showing</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=5>the index and middle fingers extended</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>paper to scissors so that the rock player</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>hand gesture so</td><td rowspan=6 colspan=1>rock exactly as they are, along withthe black-and-white cartoon</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=4 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>vit</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=11>illustration with correct game logicand unchanged visual composition.</td><td rowspan=6 colspan=5>in a V-shape, matching the originalline thickness and glove-like cartoondesign.Position the new red O in the</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>wins the game.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>that the player</td><td rowspan=1 colspan=1>the black-and-white cartoo</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>wins.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>Place the “O” in</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=11>The final image should look like a</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>Add a red O mark to the top center</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=3></td><td rowspan=3 colspan=10>consistent hand-drawn tic-tac-toeboard where the O player wins</td><td rowspan=2 colspan=5>top-center cell, match the existing redink color and rough sketch style, and</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>square of the tic-tac-toe grid to</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>spot on the</td><td rowspan=1 colspan=1>marks, the lined notebook paper</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>complete the winning row.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=11>naturally.</td><td rowspan=1 colspan=5>keep the stroke thickness consistent</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=5>with the other O marks.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>Add a new line below the total amount</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11>The receipt should still look like a</td><td rowspan=1 colspan=5>Format the new line using the same</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11>coherent and authentic printed</td><td rowspan=1 colspan=5>font, alignment, spacing, and</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=5 colspan=2>of $45.00 labeled “Tip:&quot; followed by“$4.50”.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=8 colspan=1>910</td><td rowspan=1 colspan=1>the receipt in the</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=10>but, spacing, and visual style.</td><td rowspan=1 colspan=11>receipt with the new tip line</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=1>image.</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=5>typographic style as the rest of thereceipt.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11>integrated naturally.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=2>Remove the lightning rod structurefrom the peak of the red tiled roof,</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11>The final image should look like arealistic roof scene with no visible</td><td rowspan=1 colspan=5>Seamlessly reconstruct the roof ridgeand fill the removed area with</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=1>facility used to</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=9 colspan=13>spherical components.Fill in the missing cubies to restore thepuzzle into a complete 4x4x4 cube withsolid cubic geometry.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=15 colspan=2>protect the house.11unreasonable parts12of the socket inthe image.</td><td rowspan=5 colspan=1></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=5>sky gradient remain continuous andundisturbed.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11></td><td rowspan=1 colspan=5>undis</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=11>The completed puzzle should looklike a coherent, fully assembled</td><td rowspan=1 colspan=5></td><td rowspan=5 colspan=1></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>The c</td><td rowspan=1 colspan=5>blocks with white stickers on</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>witl</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=6></td><td rowspan=2 colspan=5>stickers on right-facing surfaces,aligning them perfectly with theexisting grid lines and shading.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=6>perspective, and mater</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=1>Preserve the beige square faceplat</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=2>Correct the unrealistic and scattered</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11></td><td rowspan=1 colspan=5>Use a recognized outlet configuration</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11>e, The final socket should look like a</td><td rowspan=1 colspan=5>such as a standard 3-pin or universal</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>nreason</td><td></td><td rowspan=1 colspan=1>part</td><td rowspan=4 colspan=2>arrangement of holes on the wallsocket by replacing them with a</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11>realistic manufactured electrical</td><td rowspan=1 colspan=5>socket design, organize the holes into</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=4 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11>utlet with a clean, balanced, and</td><td rowspan=1 colspan=5>logical pairs or groups, and render</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=1>overall light</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=8>ing and texture.</td><td rowspan=1 colspan=11>logically organized layout.</td><td rowspan=1 colspan=5>their depth, shading, and geometry</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=11></td><td rowspan=1 colspan=5>with precise plastic-like realism.</td></tr></table>

![](images/fc414c5ef1628c27e43cf90c0cb7a993bde4ad55cc83acd62914139971167abf.jpg)  
Figure S4: Additional reasoning-intensive editing examples. The figure covers temporal, spatial, logical, physical, and knowledgeintensive editing cases.

Instruction

Raise the person's right arm.

Change the   
military vehicle   
in the picture to   
be set in a   
beach   
environment.

Change the palm trees and sandy beach in the background to a snowy landscape with pine trees.

Extract the   
architectural   
structures visible in   
the background of   
the image,   
separating them   
cleanly from the sky   
and surrounding   
environment.

Extract the footprint in the sand from the Image.

Remove the phone from the person’s hand.

Replace the person in the mirror with a wardrobe.

![](images/25ef3d479e1b4febf9330437c33c7f4829fc5725209e2f9596e9c38fb7636c60.jpg)  
Figure S5: Additional general editing examples. The figure covers localized modifications, preservation-sensitive edits, semantic scene changes, extraction, and removal cases.

![](images/4d62779f7fcd91ef619933fb4034dc7776545f21e027d139de8e379a3379b940.jpg)  
Figure S6: System prompt used for GPT-5 failure-attribution annotation in the routing-signal validation study. The raw outputs planner\_side, renderer\_side, and both are mapped in the paper to planner-dominant, renderer-dominant, and mixed, respectively.

![](images/993ed3bf45b5c9d3cd074c091116ce2be8e706a2ec06c82a60ab0b2e52de8c8f.jpg)  
Figure S7: Planner-side prompt template used during rollout to rewrite a raw editing instruction into a four-slot plan. The resulting e is the planner output used by the renderer and by planner-side reward scoring.

![](images/1e34c2db5065a3dac5b375ec7c966182bdaf9106dc11e4b237c04bb4609009ef.jpg)  
Fi<sub>g</sub>ure S8: Prom<sub>p</sub>t tem<sub>p</sub>late used in the ofline Gemini 3 Pro <sub>p</sub>re<sub>p</sub>rocessin<sub>g</sub> sta<sub>g</sub>e to <sub>g</sub>enerate the shared slot-ali<sub>g</sub>ned checklist C(x) from the source image and user instruction. The template enforces slot exclusivity, cross-slot complementarity, post-render verifiability, and strict binary Yes/No evaluability so that the same pre-generated checklist can be reused across planner reward, renderer reward, hardness estimation, and routing.

![](images/82734fc7af531e4f589aa7b220c71c8b4e6c9604c19a85f5f978040de7eeac50.jpg)  
Figure S9: Planner-side single-question prompt for the Modify slot. The prompt evaluates one checklist question as a post-render diagnostic of whether the current Modify slot text functioned as an adequate plan.

![](images/1525373e2f098635f11ad1a1316af30d3396d25a31f8fa57645ce2a57fc04ba5.jpg)  
Figure S10: Planner-side single-question prompt for the Preserve slot. The prompt evaluates one checklist question as a post-render diagnostic of whether the current Preserve slot text adequately protected unchanged content.

![](images/756029e496392394888674bf4bcb83a5fbf99ec02a564b7513ae356094cb8c72.jpg)  
Figure S11: Planner-side single-question prompt for the Overall slot. The prompt evaluates one checklist question as a post-render diagnostic of whether the current Overall slot text adequately constrained global coherence.

![](images/0f2808e289d5e1794b0009965d9e15bfe7ee9a23a440ad71f3ed9f620b48d119.jpg)  
Figure S12: Planner-side single-question prompt for the Tips slot. The prompt evaluates one checklist question as a post-render diagnostic of whether the current Tips slot text provided adequate local implementation guidance.

![](images/027c528b333afa10add743db99c054eed816b64c4f982ea9044ca71b6a0d2b42.jpg)  
Figure S13: Renderer-side single-question prompt for the Modify slot. The prompt evaluates one checklist question as an executionscoring query over visible modification quality.

![](images/1f2deb126146ff49d672d6fbfea0f91fb3d17e2267812105291dab5e61017216.jpg)  
Figure S14: Renderer-side single-question prompt for the Preserve slot. The prompt evaluates one checklist question as an execution scoring query over whether unchanged content is actually preserved.

![](images/434c6d7f98355c8f4b508bfd9cbca5c8ac8cdea45c638697190a5643f8a9362d.jpg)  
Figure S15: Renderer-side single-question prompt for the Overall slot. The prompt evaluates one checklist question as an executionscoring query over global coherence and realism.

![](images/3599f5455edea8dafffad0cb00c6e352faccacbe356f28260d41d04085f4c54b.jpg)  
Figure S16: Renderer-side single-question prompt for the Tips slot. The prompt evaluates one checklist question as an execution scoring query over local implementation quality.