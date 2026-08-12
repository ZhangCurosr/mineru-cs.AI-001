# Continuous Interaction Difusion: A Difusion-Native Runtime for Asynchronous Tool-Augmented Reasoning

Yuhang Cao

Nanjing University

caoyuhang@smail.nju.edu.cn

August 2026

## Abstract

Large language models increasingly rely on external tools to access up-to-date information, perform computation, and interact with the outside world. For autoregressive models, tool use naturally fits the generation process: the model emits a tool call, waits for the result, and then continues generating. Difusion language models (dLLMs), however, reason by repeatedly refining many parts of their output in parallel, making this stop-andresume interaction pattern unnecessarily restrictive. It can force tool decisions before the model’s reasoning has stabilized, delay useful observations until a discrete call finishes, and introduce redundant refinement and tool execution, potentially hurting both task accuracy and inference eficiency.

We introduce Continuous Interaction Difusion (CID), a difusion-native model–runtime architecture that integrates tool interaction into iterative denoising. CID separates a model-read-only fact channel, a thought channel represented by a Typed Cognitive Tensor, and a display channel. Information needs can emerge before a textual or JSON call is fully serialized, allowing perceptual bindings to launch external reads while denoising continues. Returned results are projected into the evolving thought state and can revise earlier cognition and display regions. Persistent bindings reuse static results without repeated external execution and refresh changing sources when needed. CID is designed to expose evidence earlier, overlap tool latency with model computation, reduce duplicate external work, and preserve useful computation after new evidence arrives. We formalize the architecture, runtime, and training objectives, and define an evaluation protocol for task quality and end-to-end eficiency. This first paper focuses on read-only tools and makes no empirical performance claims.

## 1 Introduction

Language models increasingly rely on external tools for information and capabilities that are dificult or impossible to obtain from model parameters alone. A model may search the Web for recent facts, read a file, query a database, call a calculator, or inspect the state of a software environment. In current systems, these interactions are usually organized as a sequence of explicit rounds: the model produces some reasoning, emits a structured tool request, waits for the external system to return a result, appends that result to its context, and then resumes generation. ReAct-style agents make this reasoning–acting alternation explicit <sup>17</sup>, while Toolformer-style training teaches models to insert API calls directly into token sequences <sup>15</sup>. This interface has become a natural default because it closely matches the generation process of autoregressive language models.

An autoregressive language model generates text from left to right. At each step it predicts the next token conditioned on tokens that have already been produced. Once a token has been emitted, the standard decoding process treats it as fixed and moves forward. Tool use therefore fits naturally into the same timeline: generate a call, receive an observation, place the observation after the call, and continue generating later tokens. The model’s internal computation, visible text, and external observations are all ordered along one causal sequence.

Difusion language models (dLLMs) use a diferent generation process. Instead of committing to one new token at a time, a dLLM starts from a partially unknown or corrupted sequence and repeatedly refines it through denoising steps. Masked difusion models can fill multiple unresolved positions in parallel, while decoding variants can additionally re-mask or reopen low-confidence positions <sup>1;8;13;19</sup>. Generation is therefore less naturally described as a single left-to-right timeline. At an intermediate denoising step, many parts of the output may remain uncertain while other parts are already relatively stable, and additional computation can revise unresolved or explicitly reopened content.

This diference becomes important once external tools are introduced. Suppose a search result arrives while a dLLM is still refining its answer. The new evidence may afect a hypothesis formed several denoising steps earlier, change which information is still needed, invalidate part of a draft, and alter several output regions at once. A conventional call–wait–observe interface can still be imposed on a dLLM, but it exposes only a small part of the model’s revisable state to the external environment. Current dLLM agents largely retain this interface. DLLM-Searcher, for example, decodes a toolcall region early and continues reasoning while search is in flight, improving overlap while preserving an explicit serialized call boundary <sup>20</sup>. More broadly, direct use of current dLLMs in conventional agent workflows shows weaknesses in symbolic precision and temporal feedback handling <sup>12</sup>. These observations motivate a tool interface designed around the dynamics of difusion generation itself.

We propose Continuous Interaction Difusion (CID), a difusion-native architecture in which model cognition, external perception, and user-visible generation evolve concurrently. The central abstraction is a persistent perceptual binding. When the model continues to need some information, the runtime maintains a binding between that need and an external source. If the source is static, the runtime can reuse the cached result while repeatedly re-projecting its meaning into the evolving thought state. If the source changes, the runtime can refresh or stream new observations. The model therefore does not have to repeatedly serialize the same request merely to keep the corresponding information cognitively active.

CID separates the system state into three coupled channels. The fact channel contains information controlled outside the generative state, such as user-pinned constraints or authoritative external values; the model may read these facts but cannot overwrite them. The thought channel contains hypotheses, plans, unresolved information needs, tool intentions, interpretations, uncertainty, and intermediate conclusions. We represent it as a Typed Cognitive Tensor (TCT), a continuous locally difused structure augmented with role distributions, symbolic anchors, source links, uncertainty, and per-cell noise levels. The display channel is the evolving token canvas that converges to the user-visible response. These channels can change together while obeying different representations and write permissions.

This paper formalizes the architecture and makes the design precise enough to implement and falsify. Its contributions are:

• We formulate the mismatch between turn-based tool protocols and continuously revisable difusion generation, and define continuous tool interaction as a model–runtime problem.

• We introduce a three-channel architecture with an externally controlled fact channel, a typed continuous thought channel, and a discrete display channel.

• We define persistent perceptual bindings that distinguish repeated cognitive refresh from repeated external execution, supporting both static and changing information sources.

• We specify an asynchronous runtime, local difusion clocks, model-side intent interface, training objectives, and a concrete evaluation protocol for read-only tools.

The remainder of the paper reviews the necessary background and derives the design requirements in Section 2, formalizes CID in Section 3, specifies an empirical evaluation protocol in Section 4, relates the architecture to prior work in Section 5, and discusses limitations in Section 6.

## 2 Background and Design Requirements

CID connects ideas from language-model generation, tool-augmented agents, difusion models, and continuous latent reasoning. This section introduces the parts of those areas needed to understand the design. We focus on how information enters the model over time, which parts of the model state can still be revised, and how external observations are represented.

## 2.1 Autoregressive Generation and Turn-Based Tool Use

Most deployed language models are autoregressive (AR). Given an input context x and an output sequence $y =$ $\left( y _ { 1 } , \dots , y _ { L } \right)$ , an AR model factorizes generation as

$$
p ( y \mid x ) = \prod _ { \ell = 1 } ^ { L } p ( y _ { \ell } \mid x , y _ { < \ell } ) .\tag{1}
$$

At decoding step $\ell ,$ the model predicts the next token from the fixed prefix $y _ { < \ell }$ . Standard decoding therefore has a simple temporal structure: earlier tokens become context for later tokens, and generation advances from left to right. Although implementations may cache hidden states or speculate over future tokens, the modelfacing abstraction remains a growing prefix.

Conventional tool use inherits this structure. Let q be a user request, $r _ { k }$ a textual reasoning segment, a<sub>k</sub> a serialized action such as a function call, and $o _ { k }$ the resulting observation. A typical trajectory is

$$
q  r _ { 1 }  a _ { 1 }  o _ { 1 }  r _ { 2 }  a _ { 2 }  o _ { 2 }  y .\tag{2}
$$

For example, a model answering a question about a document may first generate a request such as search $( { \tt q u e r y = . . . } )$ , stop or yield control while the request is executed, receive passages as new context, and then continue its answer. ReAct explicitly alternates reasoning and actions <sup>17</sup>; Toolformer represents API calls inside token sequences and trains the model to decide when they are useful<sup>15</sup>.

This call–observe pattern is efective because an external result can simply be appended after everything the AR model has already generated. The model then conditions future tokens on the new observation. The same interface also creates a strong commitment boundary: information that arrives later naturally influences future generation, while revising already committed reasoning or output requires an additional mechanism such as regeneration, editing, or a new conversational turn.

Tool requests are also usually serialized. Before execution, the system needs a suficiently concrete action containing a tool identity and arguments, often expressed as JSON or another schema-constrained structure. An internal sense that “more evidence about this entity is needed” does not by itself initiate $1 / \mathrm { O } ;$ ; it must first become an executable request. This distinction between an emerging information need and a fully specified call is central to CID.

## 2.2 Asynchronous Tool Interaction

A tool call need not block the entire software runtime. Event-driven agents can launch external work asynchronously, continue other computation, and handle results when they arrive <sup>3</sup>. Speculative tool calling can start likely requests before the surrounding generation is complete, and streaming systems can overlap model computation with incoming observations <sup>7</sup>. These techniques reduce idle wall-clock time.

However, asynchronous execution alone does not change the model’s representation of interaction. In a typical AR agent, a completed result still becomes a later event in the token history. The runtime may be concurrent while the model-facing state remains an ordered sequence of linguistic commitments. Likewise, if the model needs the same source throughout a long generation, persistence is usually implemented through repeated calls, cached prompt content, or runtime-specific bookkeeping rather than through an explicit model–source relation.

It is therefore useful to separate two questions. The first is when external work executes: synchronously or asynchronously. The second is how the model represents an ongoing need for external information. CID primarily changes the second interface and then uses asynchronous execution to support it.

## 2.3 Difusion Language Models

Difusion language models (dLLMs) generate with iterative refinement instead of strict next-token prediction. The exact formulation varies across models, but masked difusion provides a useful concrete example. Starting from a clean token sequence $x _ { 0 } .$ , a forward corruption process progressively replaces tokens with a mask symbol or otherwise removes information, producing increasingly corrupted states $x _ { t }$ . A learned reverse process receives a corrupted state and predicts cleaner token assignments. At generation time, the model starts from a highly masked sequence and applies multiple denoising updates until a complete sequence is obtained.

This process changes the geometry of generation. At an intermediate step, the sequence can contain a mixture of stable tokens, uncertain tokens, and unresolved masked positions. A denoising update may predict several positions in parallel. Some methods can also remask or reopen low-confidence positions, allowing earlier guesses to be reconsidered <sup>8</sup>. LLaDA showed that masked difusion can scale to instruction-following language modeling <sup>13</sup>; Block Difusion combines blockwise autoregressive structure with difusion-style generation <sup>1</sup>; Dream explores difusion language modeling with flexible decoding order<sup>19</sup>. Planned Difusion further illustrates how a coarse plan can coexist with parallel refinement of diferent output spans <sup>9</sup>.

The important property for this paper is revisabil-$_ { i t y }$ . Consider a partially denoised answer with several already readable claims and several unresolved regions. If new evidence arrives, a difusion process can use that evidence to change multiple unresolved positions during later denoising steps and, when the decoder supports reopening, revisit previously predicted positions. The computation is therefore naturally described as refinement of a current global state, rather than extension of an immutable prefix.

This does not imply that every dLLM automatically supports arbitrary editing, perfect parallelism, or eficient tool use. Diferent dLLM architectures impose diferent masking schedules, block structures, and decoding constraints. CID relies only on the broader capability that some regions of the generative state can remain revisable across iterative updates. The architecture then asks how external information should enter such a state while it is still evolving.

## 2.4 Why the Conventional Tool Interface Underuses Difusion

Suppose a model is drafting an answer while a search request is running. Before the search result arrives, it may have formed a tentative hypothesis, reserved an output span for evidence, and identified a second question whose relevance depends on the first result. When the result arrives, three kinds of change may be useful simultaneously: revise the hypothesis, update the remaining information needs, and modify an earlier part of the displayed answer.

A turn-based interface represents the same event as a new observation appended after the call. This representation is suficient for left-to-right continuation, but it does not directly express which earlier cognitive regions the observation should reopen or which ongoing information need the observation satisfies. Current dLLM agents often retain such explicit call regions. DLLM-Searcher, for example, prioritizes decoding a search-call region and lets other reasoning continue while search is in flight <sup>20</sup>. This improves overlap, yet the external interaction still begins from a serialized call and returns through a discrete observation boundary.

![](images/e87bdb78275a0d39a24064fc67b98bd035489e8e2c5c505c1dfa709fe9bc849b.jpg)  
Figure 1: Interaction timelines of turn-based autoregressive tool use and CID.

Existing evaluations also indicate that simply replacing an AR backbone with a dLLM does not automatically produce a strong agent. Current dLLMs can struggle with exact tool schemas and temporally ordered feedback in conventional workflows <sup>12</sup>. A difusion-native agent interface therefore needs to preserve precise symbolic grounding while giving the runtime access to information needs and revisable state before they become ordinary text.

Figure 1 summarizes the mismatch. Beyond asynchronous I/O, CID exposes a need before textual serialization is complete and allows returned evidence to alter multiple still-revisable regions of the same trajectory.

## 2.5 Continuous Latent Reasoning

A related line of work questions whether every intermediate reasoning state must be expressed as naturallanguage tokens. Textual chains of thought are convenient because they are readable and can be processed by the same language model, but they force intermediate computation through a discrete linguistic representation. Ambiguous hypotheses, multiple possible continuations, or partially formed plans must all be serialized into words before the next reasoning step can use them.

Continuous latent reasoning keeps some intermediate computation in vector-valued model states. Coconut, for example, feeds continuous hidden states back into an autoregressive model as reasoning states and reports that these states can encode multiple possible continuations <sup>6</sup>. LaDiR applies latent difusion to structured thought representations so that reasoning can be revised more holistically<sup>10</sup>. Such work motivates a continuous thought space in CID.

For tool interaction, however, an unrestricted latent tensor is not enough. The runtime must know whether a latent region represents a hypothesis, an unresolved information need, a source-derived percept, or a stable conclusion. It must also retain exact objects such as file paths, numbers, entity identifiers, source versions, and tool-schema fields. CID therefore combines continuous semantic content with typed roles, symbolic anchors, source links, uncertainty, and local difusion state.

## 2.6 Design Requirements

The preceding background yields six requirements for a difusion-native interaction architecture.

Externally Controlled Facts. Some values must remain outside the model’s write authority, including userpinned constraints, exact source values, and runtimemaintained state, while thought and display remain revisable. Such values can still change when their external source changes. Read-only therefore describes the model’s permission over the value, not whether the value is constant over time.

Precise Symbolic Grounding. Continuous representations must remain connected to exact external objects. Paths, numbers, entity identifiers, tool schemas, source versions, and provenance should be preserved through symbolic anchors and explicit source links instead of relying on latent similarity alone.

Early, Non-Linguistic Intent. A coarse information need should become usable before a complete tool name, query string, or JSON object has converged. Tool identity has been shown to be linearly readable from internal activations in autoregressive models <sup>16</sup>, providing evidence that at least source or tool choice can become readable before serialized emission. CID treats the richer typed need interface, including partial arguments and dependencies, as a capability to train and evaluate rather than an established consequence of that result.

Persistent Perception. An information need may remain relevant across many model updates. The system should represent this persistence explicitly. For a static source, the same external value can remain active without repeated I/O. For a changing source, the binding can request refreshed values or consume a stream.

Table 1: How CID realizes the six design requirements in Section 2.6.
<table><tr><td>Design Requirement</td><td>CID Mechanism</td><td>Effect</td></tr><tr><td>Externally Controlled Facts</td><td>Three Coupled Channels</td><td>Externally controlled values remain outside the model&#x27;s write authority while thought and display remain revisable.</td></tr><tr><td>Precise Symbolic Grounding</td><td>Typed Cognitive Tensor</td><td>Symbolic anchors and source links keep exact external objects connected to continuous representations.</td></tr><tr><td>Early, Non-Linguistic Intent</td><td>Latent Information Needs</td><td>An information need becomes usable before a complete tool name, query string, or JSON object has converged.</td></tr><tr><td>Persistent Perception</td><td>Persistent Perceptual Bindings</td><td>The same external value can remain active without repeated  $\mathrm { I / O ; }$  changing sources can be refreshed or streamed.</td></tr><tr><td>Joint Revision</td><td>Perceptual Assimilation</td><td>Conflicting evidence can reopen relevant regions while unrelated, well-supported content remains stable.</td></tr><tr><td>Continuous Evolution</td><td>Asynchronous Runtime</td><td>Reasoning and display generation continue while read-only external operations are in flight.</td></tr></table>

![](images/11d843ac9c4ecd63ea63d3533a560a3b38071d6f5442e98c27b02080523779ac.jpg)  
Figure 2: Overview of the CID model–runtime architecture.

Joint Revision. New observations should be able to afect several thought and display regions in one evolving state. The system needs selective reopening so conflicting evidence can make relevant regions revisable while unrelated, well-supported content remains stable.

Continuous Evolution. Reasoning and display generation should continue while read-only external operations are in flight. When a result arrives, it should condition the current trajectory directly, without requiring the system to restart reasoning as a separate model turn.

## 3 CID Design

CID is designed around two practical goals: improve task quality by allowing external evidence to correct uncertain or stale model state as soon as it becomes available, and improve end-to-end eficiency by overlapping external work with useful denoising while avoiding duplicate tool execution and unnecessary recomputation. The architecture follows from three corresponding requirements. First, exact externally supplied information must remain stable even while model cognition is revised. Second, an information need should become actionable before the model has fully serialized a textual or JSON tool call, so external latency can start earlier. Third, once a result has arrived, the model should be able to keep using or reinterpret it without repeatedly executing the same external operation. The components below implement these requirements jointly rather than treating tool use as a separate pause between generation rounds. Table 1 maps the six design requirements in Section 2.6 to the corresponding CID mechanisms and summarizes their intended efects.

Given a task prompt $P ,$ at difusion update s, CID maintains

$$
\mathcal { S } _ { s } = ( F _ { s } , T _ { s } , Y _ { s } , B _ { s } , \mathcal { T } _ { s } ) ,\tag{3}
$$

where $F _ { s }$ is the fact channel, $T _ { s }$ the thought channel, $Y _ { s }$ the display channel, $B _ { s }$ the active perceptual bindings, and $\mathcal { T } _ { s }$ the in-flight external jobs. The original prompt $P$ is fixed token-level conditioning across the trajectory rather than a fourth mutable channel; explicitly protected prompt content may additionally be represented in $F _ { s }$ . The model updates $T _ { s }$ and $Y _ { s }$ , while the environment and runtime update $F _ { s } , B _ { s }$ , and $\mathcal { I } _ { s }$ under explicit permissions.

This separation makes ownership explicit. Externally controlled values need not be regenerated and therefore cannot silently drift during denoising, while cognition and display remain revisable. Keeping bindings and in-flight jobs in the same system state also lets external work advance without stopping the model trajectory.

Running Example. Consider a user who asks the model to compare two recently released systems and cite exact latency numbers from their documentation. At the start of generation, the model can already organize parts of the answer that do not depend on retrieval: it may identify the comparison dimensions, reserve display regions for evidence, and keep the numerical claims unresolved. During the same denoising process, a region of $T _ { s }$ can begin to represent a need for the latency specification of the first system. This need can become useful to the runtime before the model has produced a complete textual request such as a search query or JSON call.

Once the intent adapter assigns suficient probability to a registered documentation source and enough arguments are bound, the runtime creates a perceptual binding and launches the read asynchronously. Model denoising continues while the read is in flight. The thought state can refine the comparison plan, and the display state can develop stable material whose correctness does not depend on the missing number. Cells that depend on the unavailable evidence remain uncertain rather than being forced to guess a value merely because generation has advanced.

Suppose the documentation then returns an exact latency of 37 ms. The runtime stores the returned value together with its provenance and version information. Policy may promote the exact value to the fact channel when it must be preserved verbatim. The percept encoder simultaneously constructs a projection conditioned on the current thought and display states. A numerical claim cell can become grounded in 37 ms, a tentative hypothesis that assumed a lower latency can reopen, and an already drafted comparison sentence can be revised during subsequent display denoising. These changes can occur within the same evolving trajectory because the observation conditions the current state rather than starting a separate reasoning turn.

If the answer continues to rely on the same specification, the binding stays active. Later denoising updates can re-project the cached 37 ms value without reading the documentation again, preserving its cognitive influence while the surrounding argument changes. If the source is versioned and a newer document appears, the same binding can request an external refresh and assimilate the new value. When the information need becomes inactive and its dependent output is stable, the runtime retires the binding. This example separates four events that conventional tool interfaces often collapse into one textual call: the emergence of a need, binding to a source, arrival of an external value, and continued

cognitive use of that value.

## 3.1 Three Coupled Channels

Fact Channel. $F _ { s }$ contains information whose value is controlled outside the generative state. The model can attend to it but cannot overwrite it. Importantly, this does not imply temporal immutability:

$$
F _ { s + 1 } = { \mathrm { E x t e r n a l U p d a t e } } ( F _ { s } , e _ { s + 1 } ) .\tag{4}
$$

A clock, system load, changing document, or userupdated constraint may therefore change between difusion steps. The channel may contain user-pinned statements, exact values that must be quoted, authoritative tool results, and provenance metadata. Not every tool result must be promoted to $F _ { s } ;$ ephemeral or low-value observations may remain in the runtime cache and enter cognition only through a temporary projection.

This channel provides grounding while leaving reasoning to the mutable cognitive state. Separating protected values from mutable cognition prevents exact numbers, constraints, and provenance from being altered by later denoising, while external updates can still replace them when the underlying source genuinely changes. This directly targets factual and copying errors without freezing the rest of the model state.

Each fact item is represented as

$$
f _ { j } = ( v _ { j } , \kappa _ { j } , t _ { j } , \nu _ { j } , p _ { j } ) ,\tag{5}
$$

where $v _ { j }$ is the value, $\kappa _ { j }$ its source type, $t _ { j }$ a timestamp, $\nu _ { j }$ a version or freshness marker, and $p _ { j }$ provenance. Fact-channel policy is an explicit runtime decision governing whether a tool result is promoted.

Thought Channel. The thought channel is a structured cognitive field $T _ { s } = ( c _ { s , 1 } , \ldots , c _ { s , N } )$ , with semantic matrix $\bar { H } _ { s } \in \mathbb { R } ^ { N \times d }$ whose row i is $h _ { s , i }$ . It contains reasoning, hypotheses, information needs, tool intentions, percept interpretations, plans, constraints, and conclusions. Unlike text scratchpads or untyped hidden caches, it exposes structured mutable state through a stable model–runtime contract.

Making these states explicit gives the runtime something more useful than a partially generated string to act on. In particular, it can detect an information need, track which regions depend on it, and preserve uncertainty while a result is still in flight. This enables earlier tool execution and more selective revision after the result arrives.

Display Channel. $Y _ { s }$ is a length-L sequence over V ∪ {mask} that converges to the user-visible output. It may be revised as $T _ { s }$ changes. Display properties such as realized output length (e.g., the current EOS position or populated span), remaining budget, formatting constraints, and citation coverage can be fed back into the thought channel, allowing the model to continuously adjust content rather than enforce such constraints only after generation.

Separating display from thought lets CID revise uservisible text when new evidence arrives without requiring all intermediate cognition to be serialized into that text. It also allows already useful parts of the answer to continue converging while evidence-dependent regions remain uncertain, preserving useful computation during tool latency.

The model performs coupled updates

$$
\begin{array} { r } { T _ { s + 1 } \sim D _ { T } ( T _ { s } \mid P , F _ { s } , Y _ { s } , \mathcal { P } _ { s } , \tau _ { s } ^ { T } ) , } \end{array}\tag{6}
$$

$$
Y _ { s + 1 } \sim D _ { Y } ( Y _ { s } \mid P , F _ { s } , T _ { s + 1 } , \tau _ { s } ^ { Y } ) ,\tag{7}
$$

where $\mathcal { P } _ { s }$ denotes perceptual projections from active bindings and $\tau _ { s } ^ { T } , \tau _ { s } ^ { \bar { Y } }$ are local editability schedules for thought cells and display positions, respectively. For $Y _ { s }$ , increased editability can be realized by remasking or bounded revision of already visible tokens.

## 3.2 Typed Cognitive Tensor

The thought channel is a Typed Cognitive Tensor (TCT). Cognitive cell i is

$$
c _ { s , i } = ( h _ { s , i } , r _ { s , i } , a _ { s , i } , q _ { s , i } , u _ { s , i } , \tau _ { s , i } , \ell _ { s , i } ) ,\tag{8}
$$

with the following fields:

$h _ { s , i } \in \mathbb { R } ^ { d }$ : continuous semantic content;

$r _ { s , i } \colon$ a soft role distribution, such as hypothesis, information need, percept, plan, constraint, or conclusion;

$a _ { s , i } \colon$ sparse symbolic anchors, including tool identifiers, entities, paths, numeric values, schema fields, or output spans;

$q _ { s , i } \colon$ links to fact items, bindings, or other cognitive cells;

$u _ { s , i } \colon$ epistemic or binding uncertainty;

$\tau _ { s , i } \mathrm { : }$ a local difusion level controlling editability;

$\ell _ { s , i } \colon$ lifecycle state, such as active, waiting, stable, or retired.

Roles are soft rather than mutually exclusive. A cell can simultaneously represent a hypothesis and the information need that would verify it. Sparse anchors prevent precise objects from drifting in a continuous latent space, while the continuous component preserves ambiguity and multiple competing interpretations.

These fields are included because diferent parts of tool-augmented reasoning require diferent behavior. Soft roles retain the flexibility of continuous cognition; symbolic anchors preserve exact tool identifiers, entities, paths, numbers, and output locations; source links identify which state should be revised when evidence changes; uncertainty exposes whether an external result is still needed; and per-cell difusion levels allow revision to concentrate on afected regions. The expected benefit is higher grounding accuracy with less destructive or global recomputation after each external event.

![](images/01488ff03cc76067871d7c54dfef058f50d2657d61a4466a79337ffdc3d6ecb4.jpg)  
Figure 3: Structure of a Typed Cognitive Tensor cell.

The local noise vector

$$
\pmb { \tau } _ { s } = ( \tau _ { s , 1 } , \dots , \tau _ { s , N } )\tag{9}
$$

allows diferent regions to evolve asynchronously. A supported conclusion may be nearly stable, an unresolved query may remain highly difused, and a previously stable hypothesis may be reopened after a changing external fact arrives.

## 3.3 Latent Information Needs

Tool use is represented as a role within $T _ { s }$ , not as a separate text region. A model-side intent adapter reads the cognitive field and registered source descriptors $\mathcal { D } =$ $\{ d _ { m } \}$ :

$$
I _ { s } = G _ { \phi } ( T _ { s } , Y _ { s } , { \mathcal { D } } ) .\tag{10}
$$

Each candidate need $i _ { k } \in I _ { s }$ contains

$$
i _ { k } = ( n _ { k } , \pi _ { k } , \alpha _ { k } , \omega _ { k } , \eta _ { k } , \chi _ { k } ) ,\tag{11}
$$

where $n _ { k }$ is a continuous need embedding, $\pi _ { k }$ a distribution over registered sources or tools, $\alpha _ { k }$ partially bound arguments, $\omega _ { k }$ uncertainty, $\eta _ { k }$ freshness or persistence demand, and $\chi _ { k }$ links to afected cognitive or display regions.

The adapter exposes a typed interface over the cognitive field. This choice preserves early access to latent intent while insulating the runtime from model-specific hidden-state geometry. It also separates three stages that need not occur at the same difusion step:

$$
\begin{array} { r l } & { \mathrm { n e e d ~ e m e r g e n c e }  \mathrm { s o u r c e ~ s e l e c t i o n } } \\ & { ~  \mathrm { a r g u m e n t ~ b i n d i n g } . } \end{array}\tag{12}
$$

For read-only tools, a binding may begin before every argument has reached textual certainty, provided the source accepts approximate or progressively refined selectors.

![](images/423a3838e40cd2a99959698cc661128646c029fd939ff6bbd8159d4849b1cf3a.jpg)  
Figure 4: Lifecycle of a persistent perceptual binding.

The main eficiency benefit is that external work can start at need emergence or partial argument binding rather than after a complete call has been decoded. If the need is identified several denoising steps earlier than an executable textual call, those steps can overlap tool latency instead of adding to it. At the same time, retaining uncertainty and partial arguments avoids forcing the model to commit prematurely to details that later denoising may change.

## 3.4 Persistent Perceptual Bindings

A conventional tool call is a transient request–response pair. CID instead creates a persistent perceptual binding

$$
b _ { j } = ( i _ { j } , m _ { j } , \alpha _ { j } , \rho _ { j } , z _ { j } , \chi _ { j } ) ,\tag{13}
$$

where $i _ { j }$ is the originating need, $m _ { j }$ the source or tool, $\alpha _ { j }$ current arguments, $\rho _ { j }$ a refresh policy, $z _ { j }$ runtime cache and version state, and $\chi _ { j }$ the target cognitive regions.

A binding remains active while the corresponding information need remains active. Repeated intent is therefore not automatically treated as a redundant model failure. It can mean that the model wants the percept to remain cognitively salient. The runtime distinguishes two operations:

$$
\mathrm { e x t e r n a l ~ r e f r e s h : } \ v _ { j } ^ { s + 1 }  m _ { j } ( \alpha _ { j } , s + 1 ) ,\tag{14}
$$

$$
\mathrm { c o g n i t i v e \ r e f r e s h : \ } P _ { j } ^ { s + 1 } \gets \mathrm { P r o j e c t } ( v _ { j } , T _ { s } , Y _ { s } ) .\tag{15}
$$

Static or Unchanged Source. If the source version is unchanged, the runtime can reuse $v _ { j }$ but recompute $P _ { j } ^ { s + 1 }$ . The same fact may therefore receive a diferent task-specific interpretation as cognition evolves. This supports copying and sustained reference: the source need not be re-read, yet its influence is renewed at each relevant denoising update.

Changing Source. If the source is time-varying or streamable, $\rho _ { j }$ may request polling, version checks, event subscriptions, or incremental chunks. An arriving value updates a fact item if policy marks it as protected, and always permits a new cognitive projection. Examples include current time, evolving realized display length, a modified ${ \mathrm { f i l e } } ,$ sensor input, or streaming retrieval results.

A simple refresh policy is

$$
\begin{array} { r } { \rho _ { j } ( s ) = \left\{ \begin{array} { l l } { \mathrm { R E P R O J E C T } , } & { \nu _ { j } ( s ) = \nu _ { j } ( s - 1 ) , } \\ { \mathrm { R E F E T C H + R E P R O J E C T } , } & { \nu _ { j } ( s ) \neq \nu _ { j } ( s - 1 ) , } \\ { \mathrm { S L E E P } , } & { \mathrm { P r } ( i _ { j } \mathrm { ~ a c t i v e } ) < \delta . } \end{array} \right. } \end{array}\tag{16}
$$

Here $\nu _ { j } ( s )$ denotes a version already observed through a source event or a lightweight freshness probe; obtaining it is itself external work. If no cheap version signal exists, a due refresh must re-read the source rather than assume that its version is known. More advanced policies can trade information gain, latency, and tool cost.

This distinction is what allows persistent information use without persistent $\mathrm { I } / \mathrm { O }$ . For a static source, CID can repeatedly refresh the result’s influence on evolving cognition while paying the external access cost only once. For a changing source, the same binding provides an explicit place to decide when a new fetch is worth its latency or cost. Persistent bindings therefore reduce duplicate calls while still allowing long or changing reasoning trajectories to remain grounded in external information.

## 3.5 Perceptual Assimilation

The percept encoder converts each tool result into a context-dependent projection

$$
P _ { j } ^ { s } = E _ { \psi } ( v _ { j } , p _ { j } , T _ { s } , Y _ { s } ) ,\tag{17}
$$

whose interpretation depends on the current thought and display states. Projection cells carry source links to $v _ { j }$ when provenance matters. The thought denoiser integrates them with gated cross-attention or residual updates:

$$
\widetilde { T } _ { s } = T _ { s } + \sum _ { j \in \mathcal { B } _ { s } } g _ { j , s } A ( T _ { s } , P _ { j } ^ { s } ) ,\tag{18}
$$

where $g _ { j , s }$ is predicted from need persistence, relevance, and source freshness. This residual form is one realization of the $\mathcal { P } _ { s }$ conditioning in Equation $( 7 ) ;$ an implementation may perform the same fusion inside $D _ { T }$

![](images/2a3cc0e97bf22e3a63e81aee7abb1d64332d0448a48fc3700f3c5fff564fbb62.jpg)  
Figure 5: CID runtime loop for read-only perception.

Recomputing the projection from the current T<sub>s</sub> and Y<sub>s</sub> lets one cached result support diferent parts of the reasoning as the trajectory evolves, without another external call. It also allows a newly arrived result to afect already formed hypotheses and display regions immediately instead of being useful only for tokens generated afterward.

New evidence may change the editability of existing cells. Let $\Delta _ { j , i }$ estimate how strongly percept $j$ changes the posterior of cell i. The runtime or model updates

$$
\tau _ { s + 1 , i } = \mathrm { c l i p } \left( \tau _ { s , i } + \lambda _ { \mathrm { o p e n } } \Delta _ { j , i } ^ { - } - \lambda _ { \mathrm { c l o s e } } \Delta _ { j , i } ^ { + } \right) ,\tag{19}
$$

where $\Delta _ { j , i } ^ { - }$ measures conflict and $\Delta _ { j , i } ^ { + }$ support. Conflicting cognitive cells become more difused and revisable; supported cells stabilize. Evidence-linked display positions are reopened analogously through $\tau _ { s } ^ { Y }$ , implemented by remasking or bounded visible-token revision.

Local reopening is important for eficiency as well as correctness. New evidence should revise the state it invalidates, but it should not discard unrelated work that was already correct. Adjusting editability per cell therefore aims to preserve useful computation while concentrating additional denoising on regions that actually need correction.

## 3.6 Asynchronous Runtime

Figure 5 describes the runtime loop. Model denoising may continue while read-only work is outstanding, but only until a learned current-information equilibrium signal fires. The runtime then pauses if required work remains and resumes on external progress; otherwise it forms a terminal candidate.

The runtime is asynchronous for a direct performance reason: waiting for retrieval, file $\mathrm { I } / \mathrm { O } ,$ or another readonly source should not automatically become idle model time. CID instead allows the model to refine sourceindependent cognition and display regions while the job is in flight, then revises dependent regions after arrival. End-to-end latency improves when this waiting-time computation remains useful, which is why the evaluation measures latency at matched answer quality rather than token-generation speed alone.

Initialization sets facts F, an initial thought state $T ,$ masked display Y , bindings $\boldsymbol { B } = \boldsymbol { \mathcal { O } }$ , and jobs $\mathcal { I } =$ ∅. Each model update exposes information needs; the runtime matches or updates bindings, launches a read when policy requires a fresh value and no equivalent job is active, consumes events, and re-projects cached values when external access is unnecessary. The model adjusts local noise until current-information equilibrium or a budget boundary; quiescent waiting consumes no modelupdate budget. Before termination, a freshness barrier completes required bindings and refreshes due values, with new evidence resuming refinement. External work is keyed by source and canonicalized arguments, and a completion applies only when an active binding still has the same work key, making obsolete work cancellable or discardable. Equivalent jobs are deduplicated without suppressing repeated perception: identical needs may share one cached value while receiving fresh projections. A refined need can update a binding’s selector, while independent verification can establish another binding. The step cap bounds model updates; a separate wallclock budget bounds the trajectory.

## TRAINING OBJECTIVE

CID requires a model trained for channel semantics and asynchronous events. A training trajectory contains a task x, protected facts F, tool/source descriptors D, external event sequence $E = ( e _ { 1 } , \dots , e _ { K } )$ with arrival times, target thought structures $T ^ { * }$ or weak structural supervision, and final display $Y ^ { * }$

Training randomizes event arrival, source freshness, and cache state. Before a needed event arrives, the model should maintain uncertainty and expose a binding need rather than fabricate a value. Waiting and final snapshots supervise the equilibrium signal; runtime state distinguishes quiescence from termination. After arrival, the model should assimilate the percept, revise afected cells, preserve unrelated cognition, and update the display.

A possible objective is

$$
\begin{array} { r l } & { \mathcal { L } = \mathcal { L } _ { T } + \lambda _ { Y } \mathcal { L } _ { Y } + \lambda _ { I } \mathcal { L } _ { \mathrm { i n t e n t } } + \lambda _ { B } \mathcal { L } _ { \mathrm { b i n d } } } \\ & { \qquad + \lambda _ { P } \mathcal { L } _ { \mathrm { a s s i m } } + \lambda _ { R } \mathcal { L } _ { \mathrm { r e f r e s h } } + \lambda _ { G } \mathcal { L } _ { \mathrm { g r o u n d } } + \lambda _ { C } \mathcal { L } _ { \mathrm { c o n v } } . } \end{array}\tag{20}
$$

The first two terms train thought and display difusion; the others train intent exposure, binding, eventconditioned revision, refresh behavior, symbolic grounding, and the equilibrium signal.

Static-copy training examples repeatedly expose an unchanged source while the output is gradually composed. Dynamic-monitoring examples change the source during denoising and require the thought and display states to track the latest externally supplied value. Search and file-reading examples teach need emergence, partial query binding, delayed arrival, and evidence-conditioned revision.

## 4 Evaluation Protocol

This paper does not claim empirical performance results. We specify the minimum evidence required to support or reject the CID hypothesis and define how the complete model–runtime system should be evaluated beyond token-generation speed.

## 4.1 Research Questions

RQ1: Do Useful Information Needs Emerge Before Explicit Calls? Measure the earliest difusion step at which the typed intent interface correctly identifies source type and information target. Compare it with the step at which a valid textual or JSON call becomes executable. CID is useful only if latent need detection provides a meaningful lead without excessive false bindings.

RQ2: Can Persistent Bindings Hide External Latency? Vary source latency while holding model compute fixed. Use a logical event clock for quality comparisons so accelerator speed cannot change pre-arrival refinement, and report wall-clock latency separately on matched hardware. Measure whether waiting-time computation remains useful after arrival. The central comparison is end-to-end latency at matched answer quality.

RQ3: Does Repeated Perception Improve Assimilation? For static sources, compare one-time injection with per-step cognitive re-projection. For dynamic sources, compare no refresh, fixed polling, and needconditioned refresh. Tasks should test exact copying, evidence application, stale-value avoidance, and revision of already formed output.

RQ4: Is the Typed Cognitive Tensor Necessary? Compare (i) a natural-language thought string, (ii) an untyped continuous latent tensor, and (iii) the proposed TCT. Measure accuracy, source-binding stability, symbolic-reference precision, locality of revision, and runtime inspectability.

RQ5: Is Difusion Essential? Compare CID against an asynchronous autoregressive model with the same sources, training data, and approximate compute budget. A convincing result must show a benefit from revising prior cognition or display regions, rather than from asynchronous I/O alone.

## 4.2 Task Families

Static Reference and Copying. The model reads a fixed document, table, or code fragment and must repeatedly preserve exact values while composing a longer response. This directly tests the distinction between repeated external execution and repeated cognitive refresh.

Dynamic State Tracking. A source changes during generation: current time, a key–value entry, a file version, remaining token budget, or realized display length. The final output must reflect the latest value and revise stale intermediate conclusions.

Delayed Retrieval. Question answering tasks provide a retriever whose results arrive after controlled delays. Some answer regions can be prepared without evidence; others must remain unresolved until retrieval arrives.

Streaming Evidence. A source emits increasingly informative chunks. The model should revise hypotheses incrementally instead of waiting for the complete result or restarting generation after every chunk.

Competing Sources. Multiple bindings expose complementary or conflicting facts. This setting tests source links, uncertainty, repeated verification, and local reopening of cognition.

## 4.3 Baselines

A minimal comparison should include:

• autoregressive ReAct with blocking tools;

• an event-driven asynchronous autoregressive agent <sup>3;7</sup>;

• a dLLM using conventional explicit tool-call regions;

• P-ReAct as instantiated by DLLM-Searcher <sup>20</sup>;

• CID with one-time percept injection;

• CID with persistent static re-projection;

• full CID with static and dynamic refresh;

• oracle bindings and arrival-aware local-noise updates as upper bounds.

## 4.4 Metrics

Task Quality. Use exact match, factual accuracy, evidence support, copying fidelity, and task-specific correctness. For dynamic sources, report stale-value rate.

Interaction Latency. Report wall-clock latency, toolwait overlap, and intent lead time

$$
\Delta _ { \mathrm { l e a d } } = t _ { \mathrm { e x p l i c i t ~ c a l l } } - t _ { \mathrm { b i n d i n g } } .\tag{21}
$$

Assimilation. Define observation assimilation lag as

$$
\Delta _ { \mathrm { a s s i m } } = t _ { \mathrm { c o r r e c t ~ s t a t e } } - t _ { \mathrm { a r r i v a l } } .\tag{22}
$$

Also measure the number and fraction of thought and display regions revised after arrival.

Binding Behavior. Report binding precision and recall, persistence duration, external refresh count, cachehit rate, repeated cognitive projection count, and the fraction of duplicate needs handled without duplicate I/O.

Revision Quality. Measure the ratio of corrected errors to newly introduced errors, preservation of unaffected content, and the compute cost of local reopening. These metrics test whether difusion revision is selective rather than destructive.

## 4.5 Key Ablations

A full evaluation should remove role typing, symbolic anchors, source links, per-cell noise levels, persistent bindings, static re-projection, dynamic source refresh, and display-to-thought feedback. Arrival-time randomization during training should also be ablated to determine whether the system learns genuine asynchronous assimilation or overfits a fixed event schedule.

## 4.6 Falsification Criteria

CID would not be supported if latent bindings do not reliably precede explicit calls, if post-arrival revision destroys more correct state than it repairs, if waiting-time computation is mostly discarded, or if an asynchronous autoregressive baseline matches quality and latency with lower complexity. These outcomes constitute substantive negative results for the CID hypothesis.

## 5 Related Work

Difusion Language Models. Early text difusion work established both continuous and discrete formulations. Difusion-LM denoises continuous word-vector representations <sup>11</sup>, while D3PM develops structured corruption processes for discrete state spaces <sup>2</sup>. DifuSeq extends difusion to conditional sequence-to-sequence text generation <sup>4</sup>. Masked difusion language modeling later showed that a comparatively simple objective can approach autoregressive perplexity <sup>14</sup>. More recent systems scale these ideas toward general-purpose language modeling: LLaDA establishes masked difusion at largemodel scale <sup>13</sup>; Block Difusion interpolates between blockwise autoregression and parallel difusion <sup>1</sup>; and Dream demonstrates flexible decoding order <sup>19</sup>. Planned Difusion partitions output into parallel spans <sup>9</sup>, while RemeDi reopens low-confidence tokens during decoding<sup>8</sup>. CID focuses on asynchronous external events as persistent conditions on a jointly revisable cognitive and display state.

Tool-Using Language Models. ReAct alternates explicit reasoning and action s <sup>17</sup>, and Toolformer trains API calls as token sequences<sup>15</sup>. These methods established tool-augmented reasoning while retaining serialized action boundaries. Asynchronous Tool Usage introduces an event-driven architecture for real-time agents <sup>3</sup>; later work combines asynchronous I/O with speculative calls and streaming-input training <sup>7</sup>. CID shares the objective of overlapping cognition and I/O, while moving persistent percept state into the model–runtime interface and permitting external events to revise earlier internal and displayed content.

Difusion Agents. DLLM-Searcher trains a dLLM search agent and proposes P-ReAct, which prioritizes decoding a tool-call region so reasoning can continue while search runs <sup>20</sup>. This is the closest existing system to CID’s latency motivation. CID removes the requirement that every information need first converge into an explicit call region and generalizes a call into a sustained binding with repeated projection or refresh. Evaluations of current dLLMs in agent workflows find weaknesses in strict tool schemas and temporal feedback, while observing stronger performance in non-causal roles such as selection and summarization <sup>12</sup>. CID’s typed latent interface and separation between continuous thought and symbolic anchors are intended to address this mismatch, though empirical validation remains necessary.

Latent and Difusion-Based Reasoning. Coconut feeds continuous reasoning states back into an autoregressive model and argues that latent states can represent multiple possible reasoning continuations <sup>6</sup>. Difusion of Thoughts applies difusion generation to chains of thought <sup>18</sup>. LaDiR constructs structured latent thought blocks and denoises them with latent difusion <sup>10</sup>. Continuous latent difusion language models further separate global semantic organization from textual realization <sup>5</sup>. CID builds on the general case for non-textual intermediate cognition, but introduces typed cells, external source links, persistent perceptual bindings, and a separate display difusion process.

Internal Tool Intent. Recent probing work reports that tool identity can be read and steered from internal activations before a complete call is emitted <sup>16</sup>. This supports the feasibility of latent intent readout. CID does not expose arbitrary hidden states directly; it requires a trained, typed adapter whose output is part of the model contract and whose uncertainty can be used by the runtime.

## 6 Discussion and Limitations

Architectural Complexity. CID adds a continuous thought representation, a typed intent adapter, binding state, event scheduling, and local difusion control. These components are justified only if they yield benefits unavailable to a simpler asynchronous autoregressive agent. The evaluation in Section 4 therefore treats the AR comparison as a central falsification test.

Supervision for Cognition. High-quality targets for typed cognitive cells and binding lifecycles are not naturally available. Possible strategies include distillation from textual reasoning and tool trajectories, weak supervision from source dependencies, synthetic event schedules, and end-to-end learning with auxiliary consistency losses. Each may bias the internal representation toward the teacher or task generator.

Latent-State Stability. Typed interfaces make the thought state more controllable than an arbitrary hidden tensor, but roles and anchors may still drift across checkpoints or domains. Dynamic tool registration also requires source descriptors that are expressive enough for generalization yet constrained enough for reliable binding.

Fact-Channel Policy. The fact channel is protected from model writes, not guaranteed to be true. A user can pin incorrect information, a trusted source can become stale, and two externally controlled facts can conflict. CID must preserve provenance and uncertainty rather than treating protected information as necessarily correct. Deciding which tool results deserve promotion to the fact channel is itself a policy problem.

Compute and Memory Cost. Persistent reprojection may reduce I/O but increase model computation. Local difusion clocks and cached percept encodings must be implemented carefully to avoid repeatedly processing the full thought and fact state. A useful system should expose explicit budgets for model updates, source refreshes, and retained bindings.

Privacy and Unobservable Reasoning. A latent thought channel may contain sensitive source-derived information even when it is not displayed. Runtime logging, debugging, and model inspection must therefore distinguish operational observability from exposing private chain-of-thought content. Typed metadata and source links can support auditing without requiring full decoding of the thought tensor.

Read-Only Scope. The initial CID system focuses on read-only sources such as retrieval, file reads, databases, calculators, clocks, environment state, and streaming feeds. This restriction isolates the central question— whether sustained asynchronous perception can be integrated into a difusion trajectory—from authorization, rollback, and irreversible action safety. Persistent perception is substantially easier to manage than persistent action. Side-efecting tools require commitment, authorization, idempotency, rollback, and user confirmation mechanisms and are left to future work.

Human Cognition Analogy. Sustained perception and repeated cognitive refresh are useful design analogies, not claims that CID models the human brain. CID should be evaluated as a computational architecture on measurable quality, latency, and revision behavior.

Multimodal Extension. Although this paper is limited to language and read-only software tools, the separation between externally controlled facts, continuous typed cognition, and display naturally admits visual, audio, and sensor streams. Continuous latent difusion may provide a shared mechanism for such modalities but multimodal training and alignment are future work.

## 7 Conclusion

We introduced Continuous Interaction Difusion, a model–runtime co-designed architecture that replaces discrete tool-call rounds with persistent asynchronous perception. CID separates externally controlled facts, continuous typed cognition, and user-visible text; represents information needs inside a Typed Cognitive Tensor; and maintains perceptual bindings that can re-project static results or refresh changing sources throughout denoising. CID makes a falsifiable claim: difusion should let new external information revise existing cognition and display state while useful computation continues. We will test this claim against strong asynchronous autoregressive and explicit-call dLLM baselines.

## References

[1] Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block difusion: Interpolating between autoregressive and difusion language models. arXiv preprint arXiv:2503.09573, 2025.

[2] Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising difusion models in discrete statespaces. arXiv preprint arXiv:2107.03006, 2021.

[3] Antonio A. Ginart, Naveen Kodali, Jason Lee, Caiming Xiong, Silvio Savarese, and John Emmons. Asynchronous tool usage for real-time agents. arXiv preprint arXiv:2410.21620, 2024.

[4] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and Lingpeng Kong. DifuSeq: Sequence to sequence text generation with difusion models. arXiv preprint arXiv:2210.08933, 2022.

[5] Hongcan Guo, Qinyu Zhao, Yian Zhao, Shen Nie, Rui Zhu, Qiushan Guo, Feng Wang, Tao Yang, Hengshuang Zhao, Guoqiang Wei, and Yan Zeng. Continuous latent difusion language model. arXiv preprint arXiv:2605.06548, 2026.

[6] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.

[7] Coleman Hooper, Minwoo Kang, Suhong Moon, Nicholas Lee, Eric Wen, John Wawrzynek, Michael W. Mahoney, Yakun Sophia Shao, Amir Gholami, and Kurt Keutzer. Speculative interaction agents: Building real-time agents with asynchronous I/O and speculative tool calling. arXiv preprint arXiv:2605.13360, 2026.

[8] Zemin Huang, Yuhang Wang, Zhiyang Chen, and Guo-Jun Qi. Don’t settle too early: Self-reflective remasking for difusion language models. arXiv preprint arXiv:2509.23653, 2025.

[9] Daniel Israel, Tian Jin, Ellie Cheng, Guy Van den Broeck, Aditya Grover, Suvinay Subramanian, and Michael Carbin. Planned difusion. arXiv preprint arXiv:2510.18087, 2025.

[10] Haoqiang Kang, Yizhe Zhang, Nikki Lijing Kuang, Nicklas Majamaki, Navdeep Jaitly, Yi-An Ma, and Lianhui Qin. LaDiR: Latent difusion enhances LLMs for text reasoning. arXiv preprint arXiv:2510.04573, 2025.

[11] Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori B. Hashimoto. Difusion-LM improves controllable text generation. arXiv preprint arXiv:2205.14217, 2022.

[12] Qingyu Lu, Liang Ding, Kanjian Zhang, Jinxia Zhang, and Dacheng Tao. The bitter lesson of difusion language models for agentic workflows: A comprehensive reality check. arXiv preprint arXiv:2601.12979, 2026.

[13] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language difusion models. arXiv preprint arXiv:2502.09992, 2025.

[14] Subham Sekhar Sahoo, Marianne Arriola, Yair Schif, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander M. Rush, and Volodymyr Kuleshov. Simple and efective masked difusion language models. arXiv preprint arXiv:2406.07524, 2024.

[15] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.

[16] Zekun Wu, Ze Wang, Seonglae Cho, Yufei Yang, Adriano Koshiyama, Sahan Bulathwela, and Maria Perez-Ortiz. Tool calling is linearly readable and steerable in language models. arXiv preprint arXiv:2605.07990, 2026.

[17] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. Re-Act: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

[18] Jiacheng Ye, Shansan Gong, Liheng Chen, Lin Zheng, Jiahui Gao, Han Shi, Chuan Wu, Xin Jiang, Zhenguo Li, Wei Bi, and Lingpeng Kong. Difusion of thoughts: Chain-of-thought reasoning in difusion language models. arXiv preprint arXiv:2402.07754, 2024.

[19] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Difusion large language models. arXiv preprint arXiv:2508.15487, 2025.

[20] Jiahao Zhao, Shaoxuan Xu, Zhongxiang Sun, Fengqi Zhu, Jingyang Ou, Yuling Shi, Chongxuan Li, Xiao Zhang, and Jun Xu. DLLM-Searcher: Adapting difusion large language model for search agents. arXiv preprint arXiv:2602.07035, 2026.

## A Terminology

The following definitions specify how potentially overloaded terms are used throughout this paper.

Autoregressive language model (AR LM). A language model that generates an output by predicting each next token from the input and the already generated prefix. Standard decoding advances left to right and treats earlier emitted tokens as fixed context.

Difusion language model (dLLM). A language model that generates through iterative denoising or refinement of a partially unknown or corrupted sequence. Depending on the architecture, several positions may be updated in one step and uncertain earlier positions may remain revisable.

Denoising step / difusion update. One iterative update of the difusion state. A difusion update can modify multiple positions or cognitive cells and is therefore distinct from generating one token in an AR model.

Reopening / remasking. Increasing the editability of a previously predicted region so that later computation or newly arrived evidence can revise it. Remasking is one concrete token-level mechanism for reopening.

Tool call. A discrete executable request to an external source, normally containing a tool or source identity and suficiently bound arguments such as a query or structured JSON fields.

Observation. Information returned by an external source or runtime event and made available to the model.

Information need. A model state indicating that external information would be useful. In CID, such a need may become available to the runtime before it has converged into a complete serialized tool call.

Perceptual binding. A persistent runtime relation connecting an information need to an external source, its current arguments, refresh policy, cached state, and afected cognitive regions.

External refresh. Re-executing or re-reading an external source to obtain a potentially newer value. External refresh incurs external I/O or source computation.

Cognitive refresh / re-projection. Recomputing how an already available external value should influence the current cognitive state, without repeating external I/O.

Perceptual projection. A context-dependent model representation of an external value, constructed with respect to the current thought and display states and optionally linked to its provenance.

Fact channel. Externally controlled information that the model may read but may not overwrite. Its contents can still change when the user, environment, runtime, or source updates them.

Thought channel. The revisable internal cognitive state containing hypotheses, plans, information needs, percept interpretations, uncertainty, constraints, and intermediate conclusions.

Display channel. The revisable token canvas that progressively converges to the user-visible response.

Typed Cognitive Tensor (TCT). CID’s structured continuous representation of the thought channel. Cognitive cells combine semantic vectors with typed roles, symbolic anchors, source links, uncertainty, local difusion levels, and lifecycle state.

Local difusion level. A per-region value controlling how revisable that region currently is. Higher difusion permits stronger revision; lower difusion represents a more stable region.

Persistent perception. The maintenance of an information source’s cognitive influence while the corresponding need remains active. Persistence may reuse a static cached value or refresh a changing source according to policy.

Static and dynamic sources. A static source is treated as unchanged while a binding is active, allowing repeated cognitive refresh from a cached value. A dynamic source can change over time and may require polling, version checks, subscriptions, or streaming updates.

Binding lifecycle. The runtime state of a perceptual binding as it moves through stages such as candidate, active, waiting, available, refreshing, and retired.

## B Reference Runtime Interface

A read-only source registered with a CID runtime may expose the following conceptual schema:

```yaml
source:
name: document_search
description: >-
retrieve passages from a corpus
input_schema:
query: text
scope: optional identifier
top_k: optional integer
properties:
read_only: true
cacheable: true
streamable: true
cancellable: true
versioned: true
default_refresh:
mode: on_version_change
max_age_ms: 1000
```

The runtime should not require that every source support every property. A calculator may be cacheable but not streamable; a clock is dynamic but inexpensive; a file may expose a version identifier; a search endpoint may return incremental results without a stable version.

## C Binding State Machine

A perceptual binding can use the following lifecycle:

$$
\begin{array} { r l } & { \mathrm { C A N D I D A T E }  \mathrm { A C T I V E }  \mathrm { W A I T I N G } } \\ & { \qquad \mathrm { A V A I L A B L E }  \mathrm { R E F R E S H I N G } } \\ & { \qquad \mathrm { R E T I R E D . } } \end{array}\tag{23}
$$

A binding can move from available directly back to active when the value is unchanged and only cognitive re-projection is required. It moves to refreshing when the need remains active and the source freshness policy requires external work. Quiescence is a trajectory state rather than a binding state: active bindings remain intact while model updates pause, and the next observation resumes refinement without reconstructing the interaction context.

A candidate binding should record:

• source probability and need confidence;

• partially bound selector or arguments;

• links to originating cognitive cells;

• current cache key and source version;

• external and cognitive refresh policies;

• target fact, thought, and display regions.

## D Conceptual Execution Traces

## D.1 Static Copying

A user requests a response that must reproduce a value from a document exactly. The model forms a persistent need for the value and binds to a file reader. After the file is read once, the runtime caches the exact result. During subsequent denoising steps, the unchanged result is reprojected into the thought tensor, preserving salience while the display is reorganized. No repeated disk I/O is necessary, but the percept remains active until the exact value has been anchored in the display.

## D.2 Dynamic Display Length

The current realized display length—for example, the inferred EOS position or populated span within the fixed canvas—is an externally computed value that the model may read but cannot rewrite. A binding to the length monitor remains active during generation. At each update, the runtime supplies a refreshed value, which changes the thought tensor’s plan and density decisions. The display can compress earlier text because the value conditions both current cognition and already generated output regions.

## D.3 Streaming Retrieval

A retrieval source first returns titles, then snippets, then complete passages. The binding remains active throughout the stream. Early chunks narrow hypotheses and refine the query; later chunks reopen unsupported conclusions and supply exact evidence. Unrelated display regions remain stable under local difusion clocks.

## E Suggested Training Data Construction

A practical initial dataset can be synthesized from ordinary question–answer pairs and documents:

1. identify answer spans that depend on external evidence and hide those spans from the model-visible prompt;

2. create source descriptors, binding targets, and delayed or incremental arrival schedules;

3. construct pre-arrival states with unresolved cells and post-arrival revisions grounded in the source;

4. add static-copy cases that preserve one value across many steps and dynamic cases in which the source changes before display convergence.

Teacher-generated textual reasoning may be encoded into initial cognitive cells, but training should randomize cell ordering and event schedules to avoid reducing the thought tensor to a disguised text chain.

## F Open Technical Questions

Several design decisions remain open:

• which iterative process should update the thought tensor, how many cells it needs, and whether cells should be created or retired dynamically;

• whether display and thought difusion should share a backbone and how source descriptors generalize to unseen tools;

• how to estimate support and conflict for local reopening while preserving privacy during debugging;

• how to tune refinement-epoch and wall-clock budgets across deployment settings.

These questions remain open design variables within the CID framework.