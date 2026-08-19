# STRUCTURE-INTERNALIZED RULE LANGUAGE MODEL FOR FAITHFUL KNOWLEDGE GRAPH REASONING

Xingrui Zhuo<sup>1,2</sup>, Jiapu Wang<sup>3</sup>, Manzong Huang<sup>1,2</sup>, Gongqing Wu<sup>1,2∗</sup>, Xindong Wu<sup>1,2∗</sup> <sup>1</sup>Key Laboratory of Knowledge Engineering with Big Data (Hefei University of Technology), Ministry of Education, China

<sup>2</sup>School of Computer Science and Information Engineering, Hefei University of Technology, China   
<sup>3</sup>Nanjing University of Science and Technology, China   
zxr@mail.hfut.edu.cn, jiapu.wang@njust.edu.cn,   
manzonghuang@mail.hfut.edu.cn, wugq@hfut.edu.cn, xwu@hfut.edu.cn

## ABSTRACT

Knowledge Graph Reasoning (KGR) aims to discover latent facts by leveraging the structural evidence available in KGs, posing a challenge to the structural semantic understanding capability of KGR models. Recent studies have demonstrated that Large Language Models (LLMs) can achieve remarkable progress on KGR tasks via flexible in-context learning. However, the inherent representation inconsistency between KG structural context and LLM parametric knowledge remains inadequately addressed. This limitation prevents LLMs from effectively perceiving reasoning evidence that aligns with KG constraints, which undermines both the effectiveness and faithfulness of reasoning. We refer to this problem as reasoning evidence perception drift of LLMs over KGs. To address this problem, we propose a Structure-Internalized Rule Language Model (SIRLM), which centers on structural rule generation to couple the parametric learning of structural knowledge with the faithfulness evaluation of reasoning logic, enabling LLMs to anchor tightly to KG-grounded evidence. Specifically, we first design a Structure-Internalized Rule Generator (SIRG), which incorporates an in-context learning block augmented with a structural relation memory to coordinate structural and parametric knowledge. Furthermore, we equip SIRG with a KG tokenizer based on structural invariance learning and a neuro-symbolic reasoner based on rule-constrained message propagation. These components provide SIRG with learnable structural representations and faithful rule-execution feedback, respectively. Our SIRLM can be seamlessly integrated into standard LLM training paradigms, such as SFT and GRPO. Extensive experiments against 17 state-of-the-art KGR methods on 36 datasets demonstrate the significant superiority of SIRLM<sup>1</sup>.

## 1 INTRODUCTION

Knowledge Graphs (KGs) (Ji et al., 2022) represent real-world facts through structural triplets composed of entities and relations, maintaining a unified framework for storing and querying largescale knowledge. However, KGs are generally incomplete, which leaves many potential facts unobserved. Consequently, Knowledge Graph Reasoning (KGR) (Liang et al., 2024) has been proposed to leverage the available KG context to infer missing facts, which provides more sufficient evidence for knowledge-driven applications (Huang et al., 2026; Luo et al., 2024).

Traditional KGR methods mainly focus on knowledge embedding (Sun et al., 2019) and logical rule learning (Sadeghian et al., 2019), which achieve efficient fact prediction via structural representation learning and interpretable reasoning (Wan & Du, 2021), respectively. However, these methods generally rely on specific contextual instances within a KG. When confronted with an unfamiliar KG, they often require complex rule mining or model retraining, which makes it difficult for these methods to generalize to more challenging KGR scenarios (Zhu et al., 2022; Lee et al., 2023).

![](images/0555538057300671706cf5083e01464602ab72fb5e257931d552695bf7fb9f33.jpg)  
Sub-KG for the anchor “Transformers”  
LLM reasoning using the sub-KG evidence  
Figure 1: Case of reasoning evidence perception drift of LLMs over KGs. LLM select incorrect reasoning evidence based on its semantic preference, resulting in its inability to follow KG-grounded reasoning logic to arrive at correct reasoning conclusions.

Recently, Large Language Models (LLMs) (Naveed et al., 2025), pre-trained on massive natural language corpora, have demonstrated outstanding performance in various reasoning tasks. Latest studies have confirmed that LLMs can achieve breakthrough success on KGR tasks (Kim et al., 2023; Wang et al., 2024a; Zhuo et al., 2025b). With the support of flexible in-context learning capabilities, LLMs integrate inherent parametric knowledge with structural context of KGs and exhibit remarkable knowledge emergence (Pan et al., 2024), thereby discovering new facts in KGs that are not explicitly observed. More importantly, compared with traditional KGR methods, LLM-based approaches show stronger transferability when facing previously unseen KG structures in inductive reasoning settings (Wang et al., 2024b; Guo et al., 2024; Zhuo et al., 2026).

Despite significant accomplishments, existing LLM-based KGR methods still struggle to address the representation inconsistency between structural and parametric knowledge (Jiang et al., 2024b), which constrains the ability of LLMs to grasp structural KG context during reasoning. This limitation is reflected in the fact that most LLMs tend to select reasoning evidence from a KG based on their pre-trained semantic preferences, rather than strictly following the structural constraints induced from the KG. As a result, LLMs are inclined to generate reasoning logic that appears semantically plausible but is difficult to ground in a KG, leading to the reasoning bias and the faithfulness degradation of results. We refer to this issue as the reasoning evidence perception drift problem of LLMs over KGs, meaning that inconsistencies in knowledge representation hinder LLMs from identifying reasonable evidence from KGs, thereby limiting the effectiveness of LLM reasoning.

Figure 1 illustrates the harm that such a perception shift brings to LLMs. In this case, LLMs tend to accept the assertion “films produced or distributed by the same company are likely to be released in the same region” based on their semantic preferences. As a result, the evidence space is anchored to the sub-KG marked in red in Figure 1. However, the structural rule that can be explicitly induced from the KG, namely release region(X, Y) = language(X, Z) ∧ speak in(X, Z), is instead misjudged as a weakly related evidence space (marked in green in Figure 1). In addition, for relation paths represented by anonymized identifiers (marked in purple in Figure 1), the absence of natural language semantics often leads LLMs to deliberately ignore this structural evidence that could potentially support correct reasoning. This preference for pre-trained semantic cues and the neglect of structural context cause the LLM’s reasoning process to deviate from the correct KG-grounded evidence space, ultimately producing erroneous results.

To address the aforementioned limitation, we propose a Structure-Internalized Rule Language Model (SIRLM), which enables LLMs to accurately perceive KG-grounded evidence for faithful reasoning. Overall, SIRLM consists of a KG tokenizer, a Structure-Internalized Rule Generator (SIRG) and a neuro-symbolic reasoner. In this framework, SIRG serves as the core module to generate structurally grounded rules. Therefore, we propose an in-context learning block based on a Structural Relation Memory (SRM) mechanism, which allows SIRG to align pre-trained knowledge with structural context during rule generation. Complementing SIRG, the KG tokenizer is built upon the Structural Invariance Learning (SIL) principle (Galkin et al., 2024; Huang et al., 2025a; Zhang et al., 2024b), which encodes KG elements into universal structure representations, enabling LLMs to grasp them in the form of new tokens without specific textual annotations. Furthermore, we design the neurosymbolic reasoner as a rule executor based on Rule-Constraint Message Propagation (RCMP), which feeds back a rule faithfulness metric composed of reasoning conclusions to SIRG, forming a closed-loop optimization that reinforces structurally consistent reasoning.

In terms of model training, SIRLM can seamlessly integrate with standard Supervised Fine-Tuning (SFT) and post-training frameworks of LLMs. This capability endows SIRLM with significant transferable advantages across a wide range of cross-scenario KGR tasks.

Our main contributions can be summarized as follows:

• We propose a structure-internalized rule language model to mitigate the reasoning evidence perception drift of LLMs over KGs, enabling reliable LLM-based KG reasoning.

• We design a structure-internalized rule generator with a structural relation memory to coordinate LLM knowledge with KG context, together with a SIL-based KG tokenizer and a RCMP-based rule reasoner for universal structure representation learning and faithful rule execution feedback.

• SIRLM can seamlessly integrate with standard LLM training frameworks such as SFT and GRPO, achieving generalizable reasoning across unknown KGs.

• Extensive experimental results conducted on 36 datasets demonstrate that SIRLM exhibits remarkable reasoning capabilities in both transductive and inductive KGR scenarios.

## 2 RELATED WORK

KGR studies have long focused on mining latent facts from KGs, providing interpretable knowledge support for downstream tasks. Traditional KGR methods adopt structural representation learning techniques by projecting entities and relations into appropriate embedding spaces, such as Euclidean (Bordes et al., 2013), complex (Trouillon et al., 2016), or manifold spaces (Xiao et al., 2016), to infer latent associations between entities and relations. To ensure interpretability in reasoning, some studies introduce rule-based approaches that leverage techniques such as rule mining (Qu et al., 2021; Sadeghian et al., 2019), relational path retrieval (Das et al., 2018), or Markov decision processes (Lao & Cohen, 2010) to perform KGR while providing explicit reasoning paths.

As KGs scale up, early embedding-based and rule-based methods struggle to adapt to more complex knowledge structures. Consequently, researchers have incorporated Graph Neural Networks (GNN) into KGR (Galkin et al., 2022; Teru et al., 2020; Zhu et al., 2022). Methods such as NBFNet (Zhu et al., 2021), RED-GNN (Zhang & Yao, 2022), and InGram (Lee et al., 2023) exploit the message-passing mechanism of GNNs to capture fine-grained structural context about entities and relations, providing stronger evidence for comprehensive reasoning. Building on this, some studies (Galkin et al., 2024; Huang et al., 2025a; Zhang et al., 2024b; Cui et al., 2024) focus on improving the generalization of KGR models to out-of-distribution graph structures. These approaches aim to discover reusable minimal subgraph patterns within KGs and learn universal structural representations of entities and relations, i.e., Structural Invariance Learning (SIL) (Galkin et al., 2024). This technique enables unknown KG elements to be projected into learned structural patterns, thereby enhancing model performance in complex reasoning scenarios.

In recent years, LLM-based KGR has emerged as a new research focus (Wang et al., 2024b; Zhuo et al., 2025a; 2026). By leveraging the strong knowledge emergence and in-context learning capabilities of LLMs (Pan et al., 2024), these approaches uncover deeper facts in KGs. For example, methods such as KICGPT (Wei et al., 2023) and ChatRule (Luo et al., 2025) adopt the LLM-planning and KG-retrieval paradigm to prompt LLMs in performing structural reasoning over given sub-KGs. Meanwhile, approaches such as KoPA (Zhang et al., 2024a), MKGL (Guo et al., 2024), and KRLM (Zhuo et al., 2026) rely on fine-tuning techniques to inject structural knowledge into the pre-trained space of LLMs, aligning their parametric knowledge with the structural knowledge of KGs. Despite significant development, existing LLM-based methods still face the challenge in handling the widely observed KG-LLM representation gaps. Under this condition, structural knowledge that is beneficial for reasoning in KGs is often regarded by LLMs as irrelevant evidence, leading to a phenomenon known as evidence perception drift.

The aforementioned challenge have prompted us to propose a more refined mechanism for internalizing structural knowledge, which uses structural rule generation as a feedback to guide LLMs in perceiving grounded structural logic over KGs, thereby improving model reasoning.

## 3 PRELIMINARIES

In this section, we introduce the background and key conceptions of our study.

## 3.1 KNOWLEDGE GRAPH REASONING

We define a KG as $\mathcal { G } = ( \mathcal { E } , \mathcal { R } , \mathcal { T } )$ , where E and R denote the sets of entities and relations, respectively. $\mathcal { T } = \{ < e _ { h } , r _ { q } , e _ { t } > | e _ { h } , e _ { t } \in \mathcal { E } , r _ { q } \in \mathcal { R } \}$ is the set of triples. Each triple represents a factual statement indicating that the head entity $e _ { h }$ and the tail entity $e _ { t }$ are connected via relation $r _ { q }$

KGR aims to infer new facts of the form $\scriptscriptstyle { 1 < e _ { h } , r _ { q } , ? > \notin \mathcal { T } }$ . Given a training KG $\mathcal { G } _ { t r } = ( \mathcal { E } _ { t r } , \mathcal { R } _ { t r } , \mathcal { T } _ { t r } )$ and a inference KG $\mathcal { G } _ { i n f } = ( \mathcal { E } _ { i n f } , \mathcal { R } _ { i n f } , \mathcal { T } _ { i n f } )$ , KGR tasks are typically categorized into transductive $( \mathcal { E } _ { t r } = \mathcal { E } _ { i n f } , \mathcal { R } _ { t r } \stackrel { \cdot } { = } \mathcal { R } _ { i n f }$ , and $\mathcal { T } _ { t r } \neq \mathcal { T } _ { i n f } )$ and inductive setting $\dot { ( \mathcal { E } _ { t r } \neq \mathcal { E } _ { i n f } }$ or $\mathcal { R } _ { t r } \neq \mathcal { R } _ { i n f }$ and $\mathcal { T } _ { t r } \neq \mathcal { T } _ { i n f } )$ . The inductive setting requires KGR models to generalize to previously unseen entities or relations, thereby ensuring strong extrapolation and generalization capabilities.

## 3.2 STRUCTURAL INVARIANCE LEARNING OF KG

SIL of KG derives universal structural representations for unseen entities and relations through a relational graph and a Neural Bellman-Ford Network (NBFNet) (Zhu et al., 2021).

The relational graph is designed to characterize the structural motifs between relations in a KG. Let the relational graph be defined as $\mathcal { G } _ { r } = ( \mathcal { R } , \mathcal { R } ^ { * } , \mathcal { T } ^ { * } )$ , where $\mathcal { R }$ corresponds to the set of relations in the original $\bar { \mathsf { K G } } \bar { \mathcal { G } }$ and serves as the node set of $\mathcal { G } _ { r } . \ \hat { \mathcal { R } } ^ { * }$ denotes the set of motif edges to connect relation nodes. The detailed definition of motif edges is provided in Appendix B.

Under this inductive paradigm, any relation in an arbitrary KG can be mapped into $\mathcal { G } _ { r }$ . By aggregating these motif edges, we can obtain the universal structural representation for any relation.

NBFNet is a GNN that generalizes path formulations over graph nodes. Given a query triple $< e _ { h } , r _ { q } , ? >$ , we first perform $\mathbf { N B F N e t } ( U , R ^ { * } , { \mathcal { G } } _ { r } )$ , an N-layer NBFNet, over $\mathcal { G } _ { r }$

$$
\index { \textbf { N B F N e t } } ( U , R ^ { * } , \mathcal { G } _ { r } ) = \left\{ \begin{array} { l l } { r _ { j | q } ^ { ( 0 ) } = \operatorname { I n l T } ( r _ { j } , U ) , \mathrm { ~ w h e r e ~ } U = \{ < r _ { q } , 1 ^ { d } > \} , r _ { q } , r _ { j } \in \mathcal { R } , } \\ { r _ { j | q } ^ { ( n ) } = \operatorname { U p } \Big ( r _ { j | q } ^ { ( n - 1 ) } , \operatorname { A G G } \big ( \{ \operatorname { M S G } ( r _ { z | q } ^ { ( n - 1 ) } , r ^ { * } ) | r _ { z } \in \mathcal { N } _ { r ^ { * } } ( r _ { j } ) \} | r ^ { * } \in \mathcal { R } ^ { * } , r ^ { * } \in R ^ { * } \big ) \Big ) } \end{array} \right.\tag{1}
$$

where $U$ is a tuple set that stores conditions that satisfy non-zero initialization nodes, $1 ^ { d }$ is a ddimensional full-one vector, $n \in [ 1 , N ]$ is the layer index, and $R ^ { * } \in \mathbb { R } ^ { | \mathcal { R } ^ { * } | \times d }$ is the randomly initialized embeddings of motif edges. $\mathrm { I N I T } ( \cdot ) , \mathrm { M S G } ( \cdot ) , \mathrm { A G G } ( \cdot )$ , and $\mathrm { U P } ( \cdot )$ are an indicator function, the DistMult message propagation (Yang et al., 2015), the summation aggregation operation, and a Multi-Layer Perceptron (MLP), respectively. The details of Eq. (1) is provided in Eq. (17).

According to Eq. (1), we obtain the structural representations of all relations conditioned on $r _ { q } ,$ , denoted as $\tilde { R } _ { | q } = \mathrm { \tilde { N B F N e t } } ( U , R ^ { * } , \mathcal { G } _ { r } )$ . Based on this, we further compute the structural representations of all entities conditioned on $e _ { h }$ as $E _ { | h }$

$$
E _ { | h } = \mathrm { N B F N e t } ( \{ < e _ { h } , r _ { q | q } ^ { ( N ) } > \} , R _ { | q } , \mathcal { G } ) , r _ { q | q } ^ { ( N ) } \in R _ { | q } .\tag{2}
$$

The details of Eq. (2) is provided in Eq. (18).

Finally, we compute scores for the candidate tail entities of the query triple $< e _ { h } , r _ { q } , ? > :$

$$
s _ { \mathrm { K G } } ^ { ( i ) } = S _ { \mathrm { K G } } ( e _ { i | h } ^ { ( N ) } | | r _ { q | q } ^ { ( N ) } ) i \in [ 1 , | \mathcal { E } | ] ,\tag{3}
$$

where $\boldsymbol { e } _ { i \vert h } \in E _ { \vert h }$ and $S _ { \mathrm { K G } } : \mathbb { R } ^ { 2 d }  \mathbb { R } ^ { 1 }$ is a scoring function.

## 3.3 IN-CONTEXT LEARNING AND NEXT-TOKEN PREDICTION OF LLM

In context learning: Let X be an instruction and $\mathrm { T K N } _ { \mathrm { L L M } }$ denote the pre-trained token embedding table of LLM. By looking up X in $\mathrm { T K N } _ { \mathrm { L L M } }$ , the instruction is transformed into a sequence $\mathrm { T K N } _ { \mathrm { L L M } } [ X ] = X \in \bar { \mathbb { R } } ^ { L \times F }$ with L F-dimensional token embeddings. The LLM aggregates each token embedding $\pmb { x } _ { i } \in \pmb { X }$ together with the embeddings of all preceding tokens through the selfattention mechanism to perform in-context learning, thereby producing the hidden state of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ , defined as $\pmb { h } _ { i } = \operatorname { I n C o n } ( \pmb { x } _ { i } , \pmb { X } _ { \le i } ) \colon$

$$
\operatorname { I n C o n } ( \pmb { x } _ { i } , \pmb { X } _ { \le i } ) = \operatorname { F F N } \biggl ( \mathrm { s o f t m a x } \bigl ( \frac { f _ { Q } ( \pmb { x } _ { i } ) [ f _ { K } ( \pmb { X } _ { \le i } ) ] ^ { \operatorname { T } } } { \sqrt { F } } \bigr ) f _ { V } ( \pmb { X } _ { \le i } ) \biggr ) ,\tag{4}
$$

![](images/2ed1356b1c8793525b9e58006fc6988864d4e6c5a250419ab666c885ef2c2e89.jpg)  
Figure 2: Overall architecture and training framework of SILRM. Given a query triplet, we first 1 convert it into an instruction X and input the triplet into the KG tokenizer to obtain structural embeddings of entities and relations. These structural representations are then 2 tokenized along with the LLM pre-trained tokenizer for X. Next, SIGR reads the aforementioned tokens and $\textcircled{3}$ generates structural rules for the RCMP reasoner. Afterwards, $\textcircled{4}$ the reasoner executes rules and performs neuro-symbolic reasoning, ultimately obtaining scores for candidate entities.

where $f _ { \{ Q , K , V \} } ( \cdot ) : \mathbb { R } ^ { F } \to \mathbb { R } ^ { F }$ are pre-trained linear layers and FFN(·) : $\mathbb { R } ^ { F } \to \mathbb { R } ^ { F }$ is a pre-trained Feed Forward Network (FFN).

Next-token prediction: Based on Eq. (4), the new token generated by LLM conditioned on the input instruction X can be expressed as:

$$
\hat { \pmb x } _ { z + 1 } \gets \underset { \hat { \pmb x } \in \mathrm { T K N } _ { \mathrm { L L M } } } { \arg \operatorname* { m a x } } P _ { \Theta } ( \hat { \pmb x } | \mathrm { I n C o n } ( \hat { \pmb x } _ { z } , [ \pmb X : \hat { \pmb X } _ { \le z } ] ) ) ,\tag{5}
$$

where $\hat { \pmb X } _ { \le z } \in \mathbb { R } ^ { z \times F }$ is the embeddings of the first z generated tokens and [:] denotes a row-wise concatenation operation. I $\operatorname { r c o n } ( \hat { \pmb x } _ { z } , [ \pmb X : \hat { \pmb X } _ { \leq z } ] )$ represents the hidden state of the z-th generated token $\hat { \mathbf { x } } _ { z } \in \hat { X } _ { < z }$ , which is projected through a MLP module parameterized by Θ to produce the generation probability of the (z+1)-th token.

## 4 METHODOLOGY

In this section, we elaborate on the proposed SILRM in detail. As illustrated in Figure 2, SILRM consists of a structure-internalized rule generator and a KG toolkit. In the following, we describe our method from four aspects: the construction of the query instruction and the KG tokenizer (Section 4.1), LLM-based in-context learning and rule generation (Section 4.2), and rule reasoning (Section 4.3). Finally, we introduce how our framework seamlessly integrates with LLM SFT and post-training strategies in Section 4.4.

## 4.1 CONSTRUCTION OF QUERY INSTRUCTION AND KG TOKENIZER

Given a query triplet $< e _ { h } , r _ { q } , ? >$ , we design a query instruction X suitable for generating structural rules using LLMs, whose detailed schema is provided in Appendix A.

In the instruction schema, strings of the form < Ent ID > and < Rel ID > are treated as indivisible structural token identifiers, which denote specific entities and relations in a KG, respectively. $< \mathrm { E N D } >$ serves as a rule body termination. These structural tokens are embedded via a KG tokenizer $\mathrm { T K N } _ { \mathrm { K G } }$ and are jointly processed with the pre-trained LLM tokenizer $\mathrm { T K N } _ { \mathrm { L L M } }$ to encode the remaining text in instruction X. To unify the processing of structural and textual tokens in $X$ , the tokenizer of the rule generator can be expressed as the combination of $\mathrm { T K N } _ { \mathrm { K G } }$ and $\mathrm { T K N } _ { \mathrm { L L M } } \mathrm { : }$

$$
\mathrm { T K N } _ { \mathrm { R G } } = [ \mathrm { T K N } _ { \mathrm { L L M } } : \mathrm { T K N } _ { \mathrm { K G } } ] , ~ \mathrm { T K N } _ { \mathrm { K G } } = f _ { \mathrm { u p } } ( [ r _ { \mathrm { e n d } } : E _ { | h } : { \pmb { R } } _ { | q } ] ) ,\tag{6}
$$

where $r _ { \mathrm { e n d } } \in \mathbb { R } ^ { d }$ is a randomly initialized embedding for $< \mathrm { E N D } > , f _ { \mathrm { u p } } ( \cdot ) : \mathbb { R } ^ { d } \to \mathbb { R } ^ { F }$ is a trainable linear layer, and $\pmb { R } _ { | q } \in \mathbb { R } ^ { | \mathcal { R } | \times d }$ and $E _ { | h } \in \mathbb { R } ^ { | \mathcal { E } | \times d }$ are the structural embeddings of entities and

relations obtained by Eqs. (1) and (2), respectively. It is important to note that the pre-trained $\mathrm { T K N } _ { \mathrm { L L M } }$ remain frozen and only $\mathrm { T K N } _ { \mathrm { K G } }$ is trained. This allows LLMs to perceive the structural representations of the KG through newly introduced tokens.

## 4.2 IN-CONTEXT LEARNING AND RULE GENERATION

To enable LLMs to generate executable rules, we design an in-context learning strategy with the SRM mechanism and subsequently perform autoregressive relation token prediction, which ensures that the generated rules can be grounded in the corresponding KG, providing interpretable evidence for subsequent knowledge reasoning.

In-context learning with structural relation memory: To guide the LLM toward structureinternalized rule generation, we adapt the pre-trained attention layer by integrating a structural relation memory mechanism. Given a query triplet $< e _ { h } , r _ { q } , ? >$ , we first compute a relevance score for each relation $r _ { j } \in \mathcal { R }$ and select the top-K relation as the memory $R _ { \mathrm { m e m } } { \mathrm { : } }$

$$
\begin{array} { r } { R _ { \mathrm { m e m } } = \big \{ r _ { k } \in R _ { | q | } | s _ { \mathrm { r e l } } ^ { ( k ) } \in \mathrm { T o p K } ( \big \{ s _ { \mathrm { r e l } } ^ { ( j ) } \big \} _ { j = 1 } ^ { | \mathcal { R } | } ) \big \} _ { k = 1 } ^ { K } , s _ { \mathrm { r e l } } ^ { ( j ) } = S _ { \mathrm { r e l } } ( e _ { h | h } ^ { ( N ) } | | r _ { q | q } ^ { ( N ) } | | r _ { j | q } ^ { ( N ) } ) , } \end{array}\tag{7}
$$

Here, $\begin{array} { r } { \pmb { e } _ { h | h } ^ { ( N ) } \in E _ { | h } ; \pmb { r } _ { q | q } ^ { ( N ) } , \pmb { r } _ { j | q } ^ { ( N ) } \in \pmb { R } _ { | q } ; } \end{array}$ and $S _ { \mathrm { r e l } } ( \cdot ) : \mathbb { R } ^ { 3 d }  \mathbb { R } ^ { d }$ is a MLP scorer.

Let the instruction X be tokenized by $\mathrm { T K N } _ { \mathrm { R G } }$ into the embedding sequence $\pmb { X } = \{ \pmb { x } _ { i } \in \mathbb { R } ^ { d } \} _ { i = 1 } ^ { L }$ , the in-context learning operation in Eq. (4) can be improved to:

$$
\operatorname { I n C o n } ( x _ { i } , R _ { \operatorname { m e m } } , X _ { \leq i } ) = \operatorname { F F N } \Big ( \operatorname { s o f t m a x } \big ( \frac { f _ { Q } ( x _ { i } ) [ f _ { K } ( X _ { \leq i } ) : m _ { K } ( R _ { \operatorname { m e m } } ) ] ^ { \mathrm { T } } } { \sqrt { F } } \big ) [ f _ { V } ( X _ { \leq i } ) : m _ { V } ( R _ { \operatorname { m e m } } ) ] \Big )\tag{8}
$$

where $\mathrm { F F N } ( \cdot )$ incorporates a LoRA fine-tuning block (Hu et al., 2022) in practical modeling and $m _ { \{ K , V \} } ( \cdot ) : \mathbb { R } ^ { d } \to \mathbf { \dot { \mathbb { R } } } ^ { F }$ are trainable linear layers. Eq. (8) allows the hidden state of last token $x _ { L }$ to fully integrate the multi-modal token representations in X and the relation memory, providing semantically complete activation for the subsequent rule generation. Its effectiveness analysis is provided in Appendix D.

Next-relation prediction for rule generation: Let the hidden state of the last token be $h _ { L } =$ $\mathrm { I n C o n } ( { \pmb x } _ { L } , R _ { \mathrm { m e m } } , X )$ , we formulate rule generation as an autoregressive next-relation prediction process according to Eq. (5). First, the distribution $P _ { \Theta }$ in Eq. (5) can be concretized as:

$$
\begin{array} { r } { P _ { \Theta } ( \pmb { r } | h _ { L } ) = f _ { \Theta } ( h _ { L } | | \pmb { r } ) , ~ \pmb { r } \in [ \pmb { r } _ { \mathrm { e n d } } : \pmb { R } _ { | q } ] , } \end{array}\tag{9}
$$

where $f _ { \Theta } ( \cdot ) : \mathbb { R } ^ { F + d }  \mathbb { R } ^ { 1 }$ is a trainable MLP module, which scores each candidate relation embedding to determine its likelihood as the next generated token.

Then, relation tokens are generated sequentially in an autoregressive manner:

$$
\begin{array} { r } { r _ { z + 1 } \longleftarrow \underset { r \in [ r _ { \mathrm { e n d } } : R _ { | q } ] } { \mathrm { a r g } \mathrm { m a x } } P _ { \Theta } ( r | h _ { L + z } ) , h _ { L + z } = \mathrm { I n C o n } \Big ( f _ { \mathrm { u p } } ( r _ { z } ) , R _ { \mathrm { n e m } } , [ X : f _ { \mathrm { u p } } ( R _ { \le z } ) ] \Big ) , } \end{array}\tag{10}
$$

where $z \in [ 0 , \varepsilon - 1 ]$ , ε denotes the maximum rule length, $\scriptstyle R < _ { z }$ consists of the structural embeddings of the previously generated z relations, and $f _ { \mathrm { u p } } ( \cdot ) : \mathbb { R } ^ { d }  \overline { { \mathbb { R } } } ^ { F }$ is a linear layer defined in Eq. (6). The generation process terminates when $r _ { z + 1 } = r _ { \mathrm { e n d } } \mathrm { o r } z + 1 = \varepsilon$

Through Eq. (10), we can obtain the rule body $\rho = { \textstyle \bigwedge } _ { z = 1 } ^ { \varepsilon } r _ { z }$ together with their hidden states $\left\{ h _ { L + z } \right\} _ { z = 1 } ^ { \varepsilon }$ , which are used for subsequent rule reasoning for $< e _ { h } , r _ { q } , ? >$

## 4.3 REASONING WITH RULE-CONSTRAINT MESSAGE PROPAGATION

Unlike conventional SIL methods that initialize relational graph nodes solely with a full-one vector of the query relation, our RCMP mechanism incorporates the hidden states $\left\{ h _ { L + z } \right\} _ { z = 1 } ^ { \varepsilon }$ of the rule path $\textstyle \bigwedge _ { z = 1 } ^ { \varepsilon } r _ { z }$ generated in Eq. (10) as additional initialization signals. Specifically, we modify the initialization step of relation nodes in Eq. (1) as follows:

$$
\begin{array} { r } { \hat { R } _ { | q } = \mathrm { N B F N e t } ( \{ < r _ { q } , \mathbf { 1 } ^ { d } > \} \cup \{ < r _ { z } , f _ { \mathrm { d o w n } } ( h _ { L + z } ) > \} _ { z = 1 } ^ { \varepsilon } , \hat { R } ^ { * } ) , } \end{array}\tag{11}
$$

where $f _ { \mathrm { d o w n } } ( \cdot ) : \mathbb { R } ^ { F } \to \mathbb { R } ^ { d }$ is a trainable linear layer. Similar to Eq. (1), $\hat { R } ^ { * }$ is a randomly initialized embeddings of motif edges. Eq. (11) transforms the implicit semantic signals of the rule sequence into explicit structural constraints, which enhances the contextual representation and structural

discriminability of the query relation. The details of Eq. (11) is provided in Eq. (19) and a theoretical analysis of RCMP is provided in Appendix E.

Based on Eq. (11), we obtain $\hat { R } _ { | q }$ as the relation representations. Then, we compute entity representations $\hat { \pmb { E } } _ { | h }$ and score the candidate tail entities of $< e _ { h } , r _ { q } , ? > :$

$$
\hat { \pmb { E } } _ { | h } = \mathrm { N B F N e t } ( \{ < e _ { h } , \hat { r } _ { q | q } ^ { ( N ) } > \} , \hat { \pmb { R } } _ { | q } , \mathcal { G } ) ,\tag{12}
$$

$$
s _ { \scriptscriptstyle { \mathrm { R C M P } } } ^ { ( i ) } = \mathcal { S } _ { \scriptscriptstyle { \mathrm { R C M P } } } \bigl ( \hat { e } _ { i | h } ^ { ( N ) } | | \hat { \pmb { r } } _ { q | q } ^ { ( N ) } \bigr ) , \hat { \pmb { e } } _ { i | h } \in \hat { \pmb { E } } _ { | h } ^ { ( N ) } , \hat { \pmb { r } } _ { q | q } ^ { ( N ) } \in \hat { \pmb { R } } _ { | q } ,\tag{13}
$$

where $S _ { \mathrm { R C M P } } ( \cdot ) : \mathbb { R } ^ { 2 d } \to \mathbb { R } ^ { 1 }$ is a MLP scorer. The details of Eq. (12) is provided in Eq. (20).

## 4.4 TRAINING FRAMEWORKS

During the pre-training or SFT phase, SIRLM incorporates the KG tokenizer and RCMP reasoner into the standard LLM training framework with supervision signals. Let $\rho ^ { * }$ and $e _ { t }$ represent the ground truth rule and target entity of $< e _ { h } , r _ { q } , ? >$ , respectively. The pre-training/SFT loss for SIRLM can be expressed as:

$$
\mathcal { L } _ { \mathrm { S F T } } = - \Big [ \sum _ { z = 1 } ^ { \rho ^ { * } } \log \big ( p ( \rho _ { z } ^ { * } | X , \rho _ { < z } ^ { * } ) \big ) + \log \big ( s c _ { \mathrm { K G } } ^ { ( t ) } s c _ { \mathrm { R C M P } } ^ { ( t ) } \big ) - \frac { 1 } { | \mathcal { N } | } \sum _ { e _ { i } \in \mathcal { N } } \log \big ( ( 1 - s c _ { \mathrm { K G } } ^ { ( i ) } ) ( 1 - s c _ { \mathrm { R C M P } } ^ { ( i ) } ) \big ) \Big ] ,\tag{14}
$$

where $p ( \rho _ { z } ^ { * } | X , \rho _ { < z } ^ { * } )$ represents the generation probability of the z-th relation token given the instruction X and the first z-1 generated relation tokens. $s c _ { \mathrm { K G } } ^ { ( t ) }$ and $s c _ { \mathrm { R C M P } } ^ { ( t ) }$ are calculated by Eqs. (7) and (13), respectively, and $\bar { \mathcal { N } }$ is the negative target set of $< e _ { h } , r _ { q } , ? >$

SIRLM is compatible with the post-training paradigm of existing LLMs. When transferring the pre-trained SIRLM on a downstream KG, we use the GRPO framework to train SIRLM:

$$
\mathcal { L } _ { \mathrm { G R P O } } = - \frac { 1 } { \underset { m = 1 } { M } \ | \rho ^ { ( m ) } | } \sum _ { z = 1 } ^ { | \rho ^ { ( m ) } | } \left[ p ( \rho _ { z } ^ { ( m ) } | X , \rho _ { < z } ^ { ( m ) } ) A ^ { ( m ) } - \beta \mathrm { K L } \big ( p ( \rho _ { z } ^ { ( m ) } | X , \rho _ { < z } ^ { ( m ) } ) | p _ { \mathrm { r e f } } ( \rho _ { z } ^ { ( m ) } | X , \rho _ { < z } ^ { ( m ) } ) \big ) \right] ,\tag{15}
$$

where $p$ and $p _ { \mathrm { r e f } }$ represent the policy model and the reference model initialized by the pre-trained SIRLM, respectively. $\{ \rho ^ { ( m ) } \} _ { m = 1 } ^ { \bar { M } }$ denotes the M rules generated by SIRLM for $< e _ { h } , r _ { q } , ? >$ through a sampling strategy and $\beta$ is a fixed weight. The advantage $\begin{array} { r } { \mathcal { A } ^ { ( m ) } = \frac { \mathcal { W } ^ { ( m ) } - \mathrm { m e a n } ( \{ \mathcal { W } ^ { ( m ) } \} _ { m = 1 } ^ { M } ) } { \mathrm { s t d } ( \{ \mathcal { W } ^ { ( m ) } \} _ { m = 1 } ^ { M } ) } } \end{array}$ of each rule is represented as the standardization of the corresponding reward ${ \mathcal { W } } ^ { ( m ) }$

$$
\mathcal { W } ^ { ( m ) } = \frac { \mathbb { I } \big ( \phi ( \rho ^ { ( m ) } ) \big ) } { \mathbf { R a n k } \big ( s c _ { \mathrm { R C M P } } ^ { ( t ) } } - \mathbb { I } \big ( \neg \phi ( \rho ^ { ( m ) } ) \big ) , \phi ( \rho ) = \left\{ \begin{array} { l l } { \mathrm { T r u c } , \rho _ { < | \rho | } \in \mathcal { R } \mathrm { ~ a n d ~ } \rho _ { | \rho | } = < \mathrm { E N D } > } \\ { \mathrm { F a l s e } , \mathrm { e l s e } } \end{array} \right.\tag{16}
$$

Here, Rank $\left( s c _ { \mathrm { R C M P } } ^ { \left( t \right) } \right)$ denotes the score ranking of $e _ { t }$ among all entities when $< e _ { h } , r _ { q } , ? >$ and the generated rule $\rho ^ { ( m ) }$ are given, where the score $s c _ { \mathrm { R C M P } } ^ { ( t ) }$ is obtained by Eq. (13). ϕ(·) is a boolean function used to examine the structure compliance of the generated rule. It should be noted that at this time, the RCMP reasoner, as a evaluation model in GRPO, does not participate in the parameter optimization of Eq. (15).

## 5 EXPERIMENTS

In this section, we demonstrate SIRLM from the following research question: RQ1. Can SIRLM achieve significant performance across a wide range of transductive and inductive KGR scenarios? RQ2. Does the core modules of SIRLM play a crucial role in effect enhancement, including the Multi-Modal Query Instrction (MMQI) containing textual and structural tokens, the SRM mechanism in the SIRG module, and the RCMP reasoner? RQ3. Is SIRLM sensitive to hyperparameter settings, including the number of layers N of NBFNet, the size K of structural relation memory, and the number of rule samples M of GRPO? RQ4. Is SIRLM applicable to different LLM backbones?

Table 1: The overall performance of various methods on different datasets, where the MRR and Hit10 in the IndE and FullInd scenarios are summarized as average values. The colored cells represent the best , second-best , and third-best values, respectively. “–” indicates that the experimental results are unavailable, and “NA” indicates that a model is not applicable to a KGR task.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Methods</td><td colspan="2">FB15k237</td><td colspan="2">CoDEx-M</td><td colspan="2">WN18RR</td><td colspan="2">NELL995</td><td colspan="2">12 IndE Datasets</td><td colspan="2">20 FullInd Datasets</td></tr><tr><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td></tr><tr><td rowspan="3">Embedding methods</td><td>TransE</td><td>0.313</td><td>0.495</td><td>0.320</td><td>0.481</td><td>0.226</td><td>0.501</td><td>0.401</td><td>0.501</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr><tr><td>RotatE</td><td>0.338</td><td>0.533</td><td>0.325</td><td>0.466</td><td>0.476</td><td>0.571</td><td>0.483</td><td>0.565</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr><tr><td>TuckER</td><td>0.358</td><td>0.544</td><td>0.328</td><td>0.458</td><td>0.470</td><td>0.526</td><td>0.520</td><td>0.624</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr><tr><td rowspan="3">Rule-based methods</td><td>NeuralLP</td><td>0.237</td><td>0.362</td><td>0.334</td><td>0.444</td><td>0.435</td><td>0.566</td><td>0.394</td><td>0.482</td><td>0.449</td><td>0.623</td><td>NA</td><td>NA</td></tr><tr><td>DRUM</td><td>0.343</td><td>0.516</td><td>0.312</td><td>0.438</td><td>0.486</td><td>0.586</td><td>0.532</td><td>0.662</td><td>0.458</td><td>0.621</td><td>NA</td><td>NA</td></tr><tr><td>RNNLogic</td><td>0.344</td><td>0.530</td><td>0.310</td><td>0.445</td><td>0.483</td><td>0.558</td><td>0.416</td><td>0.478</td><td></td><td></td><td>NA</td><td>NA</td></tr><tr><td rowspan="4">GNN-based methods</td><td>NBFNet</td><td>0.415</td><td>0.599</td><td>0.343</td><td>0.509</td><td>0.551</td><td>0.666</td><td>0.525</td><td>0.639</td><td>0.527</td><td>0.670</td><td>0.122</td><td>0.259</td></tr><tr><td>RED-GNN</td><td>0.374</td><td>0.558</td><td>0.342</td><td>0.499</td><td>0.533</td><td>0.624</td><td>0.543</td><td>0.651</td><td>0.504</td><td>0.648</td><td>0.151</td><td>0.242</td></tr><tr><td>ULTRA</td><td>0.368</td><td>0.564</td><td>0.372</td><td>0.525</td><td>0.480</td><td>0.614</td><td>0.509</td><td>0.660</td><td>0.565</td><td>0.724</td><td>0.366</td><td>0.529</td></tr><tr><td>MOTIF</td><td>0.357</td><td>0.550</td><td>0.361</td><td>0.517</td><td>0.529</td><td>0.628</td><td>0.514</td><td>0.655</td><td>0.582</td><td>0.739</td><td>0.373</td><td>0.535</td></tr><tr><td rowspan="8">LLM-based methods</td><td>KICGPT</td><td>0.412</td><td>0.554</td><td>-</td><td></td><td>0.549</td><td>0.641</td><td></td><td></td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr><tr><td>LSA</td><td>0.356</td><td>0.538</td><td>-</td><td>-</td><td>0.469</td><td>0.542</td><td>0.469</td><td>0.649</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>KG-FIT</td><td>0.362</td><td>0.572</td><td>一</td><td>-</td><td>0.553</td><td>0.695</td><td>一</td><td>-</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr><tr><td>ChatRule†</td><td>0.383</td><td>0.564</td><td>0.320</td><td>0.481</td><td>0.445</td><td>0.518</td><td>0.511</td><td>0.627</td><td>0.535</td><td>0.660</td><td>0.319</td><td>0.472</td></tr><tr><td>FtG</td><td>0.392</td><td>0.542</td><td>0.395</td><td>0.473</td><td></td><td></td><td>0.538</td><td>0.626</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr><tr><td>MKGL†</td><td>0.415</td><td>0.591</td><td>0.355</td><td>0.518</td><td>0.552</td><td>0.656</td><td>0.530</td><td>0.649</td><td>0.577</td><td>0.721</td><td>NA</td><td>NA</td></tr><tr><td>KRLM†</td><td>0.394</td><td>0.568</td><td>0.367</td><td>0.526</td><td>0.552</td><td>0.659</td><td>0.533</td><td>0.651</td><td>0.578</td><td>0.737</td><td>0.379</td><td>0.540</td></tr><tr><td>SIRLME2E</td><td>0.427</td><td>0.599</td><td>0.368</td><td>0.525</td><td>0.552</td><td>0.644</td><td>0.541</td><td>0.642</td><td>0.588</td><td>0.741</td><td>0.383</td><td>0.545</td></tr><tr><td rowspan="5">Ours</td><td>SIRLMPT</td><td>0.408</td><td>0.593</td><td>0.367</td><td>0.522</td><td>0.461</td><td>0.556</td><td>0.520</td><td>0.638</td><td>0.580</td><td>0.738</td><td>0.374</td><td>0.538</td></tr><tr><td>SIRLMSFT</td><td>0.429</td><td>0.599</td><td>0.379</td><td>0.531</td><td>0.556</td><td>0.648</td><td>0.549</td><td>0.668</td><td>0.599</td><td>0.749</td><td>0.388</td><td>0.550</td></tr><tr><td>SIRLMGRPO</td><td>0.414</td><td>0.595</td><td>0.383</td><td>0.542</td><td>0.523</td><td>0.629</td><td>0.541</td><td>0.644</td><td>0.593</td><td>0.744</td><td>0.381</td><td>0.544</td></tr><tr><td>Avg. gain (%)*</td><td>+6.66</td><td>+5.90</td><td>+4.13</td><td>+5.77</td><td>+6.92</td><td>+5.11</td><td>+5.45</td><td>+6.01</td><td>+6.84</td><td>+6.64</td><td>+10.30</td><td>+12.05</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

• RED-GNN can only obtain experimental results on 12 FullInd datasets partitioned from FB15k237, NELL995, and Wikidata68K.  
† We reproduce ChatRule, MKGL, and KRLM on all datasets.  
⋆ We calculate the average gain of the optimal value in the four training modes of SIRLM compared to all baselines.

## 5.1 DATASETS, BASELINES, AND EXPERIMENTAL SETTINGS

Datasets. To comprehensively evaluate SIRLM, we conduct experiments on 36 KGR datasets including four Transductive datasets (FB15k-237 (Toutanova & Chen, 2015), WN18RR (Dettmers et al., 2018), CoDEx-M (Safavi & Koutra, 2020), and NELL995 (Xiong et al., 2017)), 12 Inductive Entity (IndE) datasets (Teru et al., 2020) created from FB15k237, WN18RR, and NELL995, and 20 Fully Inductive (FullInd) datasets created from FB15k237, NELL995, Wikidata68K (Gesese et al., 2022), and MTDEA (Zhou et al., 2023). Detailed dataset descriptions are provided in Appendix H.

We perform SIRLM on the aforementioned datasets using four paradigms: End-to-End (E2E) training from scratch, Pre-Training (PT), SFT, and GRPO post-training. The training settings for each paradigm are detailed in Appendix H.

Baselines. We compare SIRLM with (1) conventional KG embedding methods (TransE (Bordes et al., 2013), RotatE (Sun et al., 2019), and TuckER (Balazevic et al., 2019)), (2) rule-based methods (NeuralLP (Yang et al., 2017), DRUM (Sadeghian et al., 2019), and RNNLogic (Qu et al., 2021)), (3) GNN-based methods (NBFNet (Zhu et al., 2021), RED-GNN (Zhang & Yao, 2022), ULTRA (Galkin et al., 2024), and MOTIF (Huang et al., 2025a)), and (4) LLM-based methods (KICGPT (Wei et al., 2023), LSA (Li et al., 2025), ChatRule (Luo et al., 2025), MKGL (Guo et al., 2024), KG-FIT (Jiang et al., 2024a), FtG (Liu et al., 2025), and KRLM (Zhuo et al., 2026)).

Experimental settings. Based on previous work (Galkin et al., 2024), we adopt Mean Recurrent Rank (MRR) and top-10 Hit rate (Hit10) as evaluation metrics. In the main experiment, we use Qwen2.5-1.5b as the backbone of the SIRG module. We pretrain and fine-tune SIRLM using 4 A100 (40GB) GPUs. The more detailed settings of model hyperparameters are provided in Appendix I.

## 5.2 MAIN RESULTS (RQ1)

Table 1 reports the overall performance of various methods across 36 KGR datasets under transductive, IndE, and FullInd set-

![](images/dd35fe0f3fdaa4768986f1dd8b7ceb48f26dead1411fd7a1e210eb0fffc1a460.jpg)  
Figure 3: Hit10 of SIRLM variants on different datasets, where the results in the IndE and FullInd scenarios are summarized as average values.

![](images/084d4900af2e6a691b13a13ea894b63dc82ec714ca0192d66178f64f633911f5.jpg)

![](images/ade0004c404c2a38cbbdc4ffaaa6be43f6986bd58d01b7411c16080850e21eaa.jpg)

![](images/e4c3ed43c9dc21acb37dfaea8eb9f5b2a952b2d66fc9ded0082cfc13796f7bf4.jpg)

![](images/2b3435951299387372e327cd3a0bdf840d885d47bab07a34d48a4dcceb208bc3.jpg)

![](images/81129462cdc5ead7dfa8b39aad0ab717c2f28033de776a657907a48c6733b0d8.jpg)  
Figure 4: Hit10 of SIRLM<sub>E2E</sub> with different NBFNet layers N in KG tokenizer and RCMP reasoner.

tings. Overall, our proposed SIRLM achieves consistently strong and competitive results, outperforming prior methods on most datasets and metrics. Compared with embedding and rule-based approaches, LLM-based methods demonstrate clear advantages. This is because LLMs are better at understanding structural context and leveraging it for effective fact inference. GNN-based methods, supported by their ability to learn invariant graph representations, can capture general structural semantics across different KGs and thus achieve meaningful progress on inductive tasks.

In comparison to other baselines, LLM-based methods generally provide a holistic performance. However, most of them primarily focus on the transductive setting. Consequently, we reproduce opensource LLM-based methods (ChatRule, MKGL, and KRLM) for inductive reasoning. These methods are built upon GPT-4o mini or LLaMA2-7B and achieve competitive performance. In contrast, our SIRLM, despite being based on a 1.5B-scale LLM, surpasses these LLM-based approaches. This is primarily attributed to its effective internalization of structural knowledge and stricter adherence to logical semantics. More detailed experimental analysis can be found in Appendixes J.1.

## 5.3 ABLATION EXPERIMENTS (RQ2)

This section discusses the effectiveness of different modules in SIRLM. The experimental results are shown in Figure 3. Overall, the effectiveness of each ablation variant is inferior to that of the full model, especially in structural knowledge learning modules such as “SRM” and “RCMP”. Appendix J.2 provides detailed ablation variant settings and experimental results.

## 5.4 PARAMETER ANALYSIS (RQ3)

Figures 4 and 5 illustrate the performance of SIRLM under different configurations of the NBFNet layers $N ,$ relation memory scale K, and the number of GRPO samples N. When the layers in both the KG tokenizer and the RCMP reasoner is set to $N =$ 6, SIRLM achieves a good balance between reasoning accuracy and memory cost. K is related to the relation scale in the KG. Therefore, we uniformly set $K = 3 0$ in our experiments.

![](images/b4d9ed3d00bc07bcfb7c0265622c3a90894123f61d7f1934fa9b1e3f10e224e5.jpg)

![](images/0ecba1e205cf05f766ba5eeeb537ebb6074835466a1a46ec08cc50049ab8b249.jpg)  
Figure 5: Hit10 of SIRL $\mathbf { M } _ { \mathrm { E 2 E } }$ with different relation memory scale K and SIRL $\mathbf { \delta M _ { G R P O } }$ with different GRPO sample number M.

When $M = 1$ , the group relative advantage in Eq. (15) becomes ineffective, making it difficult for SIRLM to learn high-quality logical patterns, a phenomenon that is particularly pronounced in sparse KGs such as WN18RR v1 and CodeX-M. Therefore, we consistently set $M = \bar { 8 }$ in our experiments.

## 5.5 ADAPTABILITY ANALYSIS OF LLM BACKBONES (RQ4)

This section evaluates the adaptability of $\mathrm { S I R L M _ { E 2 E } }$ across different LLM backbones. As shown in Table 2, SIRLM remains relatively stable across most backbones, with only a slight performance decline on Qwen2.5-0.5B. We attribute this gap to the limited capacity of smaller LLMs to assim ilate structural knowledge. As shown in Fig-

Table 2: The performance of SIRL $\mathbf { M } _ { \mathrm { E 2 E } }$ with different LLM backbones.
<table><tr><td rowspan="2">LLM Backbone</td><td>CoDEx-M</td><td rowspan="2">WN18RR v1</td><td rowspan="2"></td><td rowspan="2">FB15k237-25</td><td rowspan="2"></td><td rowspan="2">NELL995-25</td><td rowspan="2"></td></tr><tr><td>|MRR Hit10</td></tr><tr><td>Qwen2.5-0.5b</td><td>0.351 0.511</td><td>MRR 0.692</td><td>Hit10 0.749</td><td>MRR 0.382</td><td>Hit10 0.637</td><td>MRR 0.389</td><td>Hit10 0.564</td></tr><tr><td>Qwen2.5-1.5b</td><td>0.368 0.525</td><td>0.705</td><td>0.766</td><td>0.386</td><td>0.635</td><td>0.407</td><td>0.601</td></tr><tr><td>Qwen2.5-7b</td><td>0.369 0.531</td><td>0.708</td><td>0.771</td><td>0.387</td><td>0.635</td><td>0.408</td><td>0.606</td></tr><tr><td>Llama-2-7b</td><td>0.3620.528</td><td>0.711</td><td>0.781</td><td>0.387</td><td>0.632</td><td>0.405</td><td>0.602</td></tr></table>

ure 9 of Appendix J.3, Qwen2.5-0.5B exhibits the slowest convergence in rule-token generation accuracy, suggesting greater difficulty in aligning structural representations within its limited parameter space. As model capacity increases, SIRLM can more effectively internalize structural knowledge, leading to more stable reasoning performance.

## 6 CONCLUSION

This paper identifies a pervasive issue in LLM-based KGR, termed reasoning evidence perception drift, caused by the misalignment between KG structural representations and LLM parametric knowledge, which undermines both reasoning effectiveness and faithfulness. To address this issue, we propose the Structure-Internalized Rule Language Model (SIRLM), which integrates a structureinternalized rule generator with a KG tokenizer and a neuro-symbolic reasoner. SIRLM aligns parametric knowledge with KG structure through structural representation learning, rule generation, and faithfulness feedback, enabling evidence-grounded reasoning over KGs. Extensive experiments with 17 baselines on 36 KGR benchmarks show that SIRLM consistently outperforms existing methods across end-to-end training, pre-training, supervised fine-tuning, and post-training settings. Appendix K discusses its limitations and future directions.

## AI USE STATEMENT

In this work, we have not used generative AI tools for any tasks requiring disclosure, and the remaining required disclosure tasks are not applicable to this work. We used generative AI tools only to edit the manuscript for grammar and readability. All AI-assisted edits were reviewed by the authors to ensure that they did not alter the technical content, scientific claims, or intended meaning. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI

## REPRODUCIBILITY STATEMENT

We confirm that our study has reproducibility. Specifically, we have first submitted our desensitized project on anonymous GitHub (https://github.com/lazyloafer/SIRLM). The detailed pseudocode of the algorithm is provided in Appendix F. In addition, we provide specific details of the experimental conclusions in the main text, including dataset partitioning (Appendix H), hyperparameter settings (Appendix I), and ablation variant settings (Appendix J.2).

## REFERENCES

Ivana Balazevic, Carl Allen, and Timothy Hospedales. TuckER: Tensor Factorization for Knowledge Graph Completion. In EMNLP, pp. 5185–5194. ACL, 2019.

Pablo Barcelo, Mikhail Galkin, Christopher Morris, and Miguel A. Romero Orth. Weisfeiler and´ Leman Go Relational. In LoG, pp. 46. PMLR, 2022.

Antoine Bordes, Nicolas Usunier, Alberto Garc´ıa-Duran, Jason Weston, and Oksana Yakhnenko.´ Translating Embeddings for Modeling Multi-relational Data. In NeurIPS, pp. 2787–2795. Curran Associates, Inc., 2013.

Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, and Petar Velickovic. Principal\` Neighbourhood Aggregation for Graph Nets. In NeurIPS, pp. 13260–13271. Curran Associates, Inc., 2020.

Yuanning Cui, Zequn Sun, and Wei Hu. A Prompt-Based Knowledge Graph Foundation Model for Universal In-Context Reasoning. In NeurIPS, pp. 7095–7124. Curran Associates, Inc., 2024.

Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, Luke Vilnis, Ishan Durugkar, Akshay Krishnamurthy, Alex Smola, and Andrew McCallum. Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning. In ICLR. OpenReview.net, 2018.

Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. Convolutional 2D Knowledge Graph Embeddings. In AAAI, pp. 1811–1818. AAAI Press, 2018.

Mikhail Galkin, Etienne G. Denis, Jiapeng Wu, and William L. Hamilton. NodePiece: Compositional and Parameter-Efficient Representations of Large Knowledge Graphs. In ICLR. OpenReview.net, 2022.

Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, and Zhaocheng Zhu. Towards Foundation Models for Knowledge Graph Reasoning. In ICLR, pp. 31598–31619. OpenReview.net, 2024.

Genet Asefa Gesese, Harald Sack, and Mehwish Alam. RAILD: Towards Leveraging Relation Features for Inductive Link Prediction In Knowledge Graphs. In IJCKG, pp. 82–90. ACM, 2022.

Lingbing Guo, Zhongpu Bo, Zhuo Chen, Yichi Zhang, Jiaoyan Chen, Yarong Lan, Mengshu Sun, Zhiqiang Zhang, Yangyifei Luo, Qian Li, Qiang Zhang, Wen Zhang, and Huajun Chen. MKGL: Mastery of a Three-Word Language. In NeurIPS, volume 37, pp. 140509–140534. Curran Associates, Inc., 2024.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In ICLR. OpenReview.net, 2022.

Manzong Huang, Chenyang Bu, Yi He, Xingrui Zhuo, and Xindong Wu. Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG. In AAAI, pp. 31202–31210. AAAI Press, 2026.

Xingyue Huang, Miguel Romero, <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan, and Pablo Barcelo. A Theory of Link´ Prediction via Relational Weisfeiler-Leman on Knowledge Graphs. In NeurIPS, pp. 19714–19748. Curran Associates, Inc., 2023.

Xingyue Huang, Pablo Barcelo, Michael M. Bronstein, ´ <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan, Mikhail Galkin, Juan L. Reutter, and Miguel A. Romero Orth. How Expressive are Knowledge Graph Foundation Models? In ICML, pp. 25021–25058. PMLR, 2025a.

Xingyue Huang, Miguel A. Romero Orth, Pablo Barcelo, Michael M. Bronstein, and´ <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan. Link Prediction with Relational Hypergraphs. Trans. Mach. Learn. Res., 2025(1):1–42, 2025b.

Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and Philip S. Yu. A Survey on Knowledge Graphs: Representation, Acquisition, and Applications. IEEE TNNLS, 33(2):494–514, 2022.

Pengcheng Jiang, Lang Cao, Cao (Danica) Xiao, Parminder Bhatia, Jimeng Sun, and Jiawei Han. KG-FIT: Knowledge Graph Fine-Tuning Upon Open-World Knowledge. In NeurIPS, pp. 136220– 136258. Curran Associates, Inc., 2024a.

Zhouyu Jiang, Ling Zhong, Mengshu Sun, Jun Xu, Rui Sun, Hui Cai, Shuhan Luo, and Zhiqiang Zhang. Efficient Knowledge Infusion via KG-LLM Alignment. In Findings of ACL, pp. 2986–2999. ACL, 2024b.

Jiho Kim, Yeonsu Kwon, Yohan Jo, and Edward Choi. KG-GPT: A General Framework for Reasoning on Knowledge Graphs Using Large Language Models. In Findings of EMNLP, pp. 9410–9421. ACL, 2023.

Ni Lao and William W. Cohen. Relational retrieval using a combination of path-constrained random walks. Mach. Learn., 81(1):53–67, 2010.

Jaejun Lee, Chanyoung Chung, and Joyce Jiyoung Whang. InGram: Inductive Knowledge Graph Embedding via Relation Graphs. In ICML, volume 202, pp. 18796–18809. PMLR, 2023.

Miaomiao Li, Ke Liang, Yuping Lai, and Xinwang Liu. Knowledge Graph Reasoning Based on Information Enhancement and Subgraph Alignment. IEEE TNNLS, pp. 1–13, 2025.

Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, Xinwang Liu, Fuchun Sun, and Kunlun He. A Survey of Knowledge Graph Reasoning on Graph Types: Static, Dynamic, and Multi-Modal. IEEE TPAMI, 46(12):9456–9478, 2024.

Ben Liu, Jihai Zhang, Fangquan Lin, Cheng Yang, and Min Peng. Filter-then-Generate: Large Language Models with Structure-Text Adapter for Knowledge Graph Completion. In COLING, pp. 11181–11195, 2025.

Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning. In ICLR, pp. 14400–14423. OpenReview.net, 2024.

Linhao Luo, Jiaxin Ju, Bo Xiong, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. ChatRule: Mining Logical Rules with Large Language Models for Knowledge Graph Reasoning. In PAKDD, pp. 314–325. Springer-Verlag, 2025.

Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. A Comprehensive Overview of Large Language Models. ACM TIST, 16(5):106:1–106:72, 2025.

Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying Large Language Models and Knowledge Graphs: A Roadmap. IEEE TKDE, 36(7):3580–3599, 2024.

Meng Qu, Junkun Chen, Louis-Pascal Xhonneux, Yoshua Bengio, and Jian Tang. RNNLogic: Learning Logic Rules for Reasoning on Knowledge Graphs. In ICLR, 2021.

Ali Sadeghian, Mohammadreza Armandpour, Patrick Ding, and Daisy Zhe Wang. DRUM: End-To-End Differentiable Rule Mining On Knowledge Graphs. In NeurIPS, pp. 15347–15357. Curran Associates, Inc., 2019.

Tara Safavi and Danai Koutra. CoDEx: A Comprehensive Knowledge Graph Completion Benchmark. In EMNLP, pp. 8328–8350. ACL, 2020.

Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. In ICLR. OpenReview.net, 2019.

Komal K. Teru, Etienne G. Denis, and William L. Hamilton. Inductive Relation Prediction by Subgraph Reasoning. In ICML, volume 119, pp. 9448–9457. PMLR, 2020.

Kristina Toutanova and Danqi Chen. Observed Versus Latent Features for Knowledge Base and Text Inference. In Workshop on CVSC, pp. 57–66. ACL, 2015.

Theo Trouillon, Johannes Welbl, Sebastian Riedel, ´ Eric Gaussier, and Guillaume Bouchard. Complex<sup>´</sup> Embeddings for Simple Link Prediction. In ICML, volume 48 of JMLR Workshop and Conference Proceedings, pp. 2071–2080. JMLR.org, 2016.

Guojia Wan and Bo Du. GaussianPath: A Bayesian Multi-Hop Reasoning Framework for Knowledge Graph Reasoning. In AAAI, pp. 4393–4401. AAAI Press, 2021.

Jiapu Wang, Kai Sun, Linhao Luo, Wei Wei, Yongli Hu, Alan Wee-Chung Liew, Shirui Pan, and Baocai Yin. Large language models-guided dynamic adaptation for temporal knowledge graph reasoning. In NeurIPS, pp. 8384–8410. Curran Associates, Inc., 2024a.

Kai Wang, Yuwei Xu, Zhiyong Wu, and Siqiang Luo. LLM as Prompter: Low-resource Inductive Reasoning on Arbitrary Knowledge Graphs. In Findings of ACL, pp. 3742–3759. ACL, 2024b.

Yanbin Wei, Qiushi Huang, Yu Zhang, and James T. Kwok. KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion. In Findings of EMNLP, pp. 8667–8683. ACL, 2023.

Han Xiao, Minlie Huang, Yu Hao, and Xiaoyan Zhu. From One Point to A Manifold: Orbit Models for Knowledge Graph Embedding. In IJCAI, pp. 1315–1321. ijcai.org, 2016.

Wenhan Xiong, Thien Hoang, and William Yang Wang. DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning. In EMNLP, pp. 564–573. ACL, 2017.

Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding Entities and Relations for Learning and Inference in Knowledge Bases. In ICLR. OpenReview.net, 2015.

Fan Yang, Zhilin Yang, and William W Cohen. Differentiable Learning of Logical Rules for Knowledge Base Reasoning. In NeurIPS, pp. 2319–2328. Curran Associates, Inc., 2017.

Yichi Zhang, Zhuo Chen, Lingbing Guo, Yajing Xu, Wen Zhang, and Huajun Chen. Making Large Language Models Perform Better in Knowledge Graph Completion. In ACM MM, pp. 233–242. ACM, 2024a.

Yongqi Zhang and Quanming Yao. Knowledge Graph Reasoning with Relational Digraph. In ACM WWW, pp. 912–924. ACM, 2022.

Yucheng Zhang, Beatrice Bevilacqua, Mikhail Galkin, and Bruno Ribeiro. TRIX: A More Expressive Model for Zero-shot Domain Transfer in Knowledge Graphs. In LoG Conference. OpenReview.net, 2024b.

Jincheng Zhou, Beatrice Bevilacqua, and Bruno Ribeiro. A Multi-Task Perspective for Link Prediction with New Relation Types and Nodes. In NeurIPS GLFrontiers Workshop, 2023.

Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal A. C. Xhonneux, and Jian Tang. Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction. In NeurIPS, pp. 29476–29490. Curran Associates, Inc., 2021.

Zhaocheng Zhu, Mikhail Galkin, Zuobai Zhang, and Jian Tang. Neural-Symbolic Models for Logical Queries on Knowledge Graphs. In ICML, volume 162 of Proceedings of Machine Learning Research, pp. 27454–27478. PMLR, 2022.

Xingrui Zhuo, Shirui Pan, Jiapu Wang, Gongqing Wu, Zan Zhang, Rui Li, Zizhong Wei, and Xindong Wu. Progressive Prefix-Memory Tuning for Complex Logical Query Answering on Knowledge Graphs. In IJCAI, pp. 3716–3724. ijcai.org, 2025a.

Xingrui Zhuo, Jiapu Wang, Gongqing Wu, Shirui Pan, and Xindong Wu. Effective instruction parsing plugin for complex logical query answering on knowledge graphs. In ACM WWW, pp. 4780–4792. ACM, 2025b.

Xingrui Zhuo, Jiapu Wang, Gongqing Wu, Zhongyuan Wang, Jichen Zhang, Shirui Pan, and Xindong Wu. Knowledge Reasoning Language Model: Unifying Knowledge and Language for Inductive Knowledge Graph Reasoning. In ICLR, pp. 122329–122356, 2026.

## A DESIGN DETAILS OF QUERY INSTRUCTIONS

Given a query triplet $( e _ { h } , r _ { q } , ? )$ , we first provide its schema of a query instruction below:

Schema of the Query Instruction for Rule Generation   
Suppose you are a linguistic expert who is learning a new rule language. Given the following head   
entity and query relation:   
### Head entity: <Ent $e _ { h } >$   
### Query relations: <Rel $r _ { q } >$   
Generate the following rule language: <Rel $r _ { q } > = <$ < Rel r<sub>1</sub> >< Rel r<sub>3</sub> >< Rel r<sub>2</sub> >< END >   
mask for generation

Our query instruction consists of a fixed text prompt along with structural representation placeholders that vary depending on the query triplet. Specifically, < Ent $e _ { h } >$ and < Rel $r _ { q } >$ correspond to the structural representations of $e _ { h }$ and $r _ { q } ,$ , respectively. The masked portion includes the structural representations of atomic relations in the rule body, < Rel $r _ { 1 } > <$ Rel $r _ { 3 } > <$ Rel $r _ { 2 } >$ , as well as the rule termination token $< \mathrm { E N D } >$ , all of which are derived from $\mathrm { T K N } _ { \mathrm { K G } }$ in Eq. (6).

This instruction format integrates textual tokes with the structural representations of a KG, enabling the LLM to further learn the contextual semantics of entities and relations within the structural space. Moreover, since the structural representations provided by NBFNet inherently encode high-order graph information, we can deliver richer structural information to the LLM using fewer tokens. In addition, this instruction format does not rely on explicit textual names of entities and relations, making it suitable for anonymized KGR scenarios.

We compare the instruction lengths designed by several recent LLM-based KGR methods in Table 3, demonstrating that our constructed instructions achieve more efficient utilization of computational resources.

Table 3: Average instruction length of MKG, KRLM, and SIRLM in 36 KGR datasets.
<table><tr><td>Model</td><td>MKGL (Guo et al., 2024)</td><td>KRLM (Zhuo et al., 2026)</td><td>SIRLM</td></tr><tr><td>Avg. Length</td><td> $\overline { { 1 1 5 . 6 6 { \pm 3 . 6 5 } } }$ </td><td>120.50±4.12</td><td> $\overline { { 5 0 . 7 3 { \scriptstyle \pm 0 . 6 7 } } }$ </td></tr></table>

## B RELATIONAL GRAPH CONSTRUCTION

Unlike a typical KG, a relational graph (Galkin et al., 2024; Huang et al., 2025a) is used to describe the relative states between relations. As shown in Figure 6, in our study, the motifs connecting relation nodes in a relational graph are essentially a set of relation-oriented hyperedges that is defined as $\mathcal { R } ^ { * }$ in Eq. (1).

In practical implementation, we use sparse matrices to construct the adjacency matrix of the relational graph. Giving a KG $\mathcal { G } = ( \mathcal { E } , \mathcal { R } , \mathcal { T } )$ defined as a graph with multiple directed edges, and its adjacency matrix can be represented as $\overset { \cdot } { A } \in \mathbb { R } ^ { \mathcal { E } \times \mathcal { R } \times \mathcal { E } }$ . Subsequently, we construct the adjacency matrix by performing the maximum scatter operation on the head and tail node dimensions, resulting in two sparse matrices $\boldsymbol { A } _ { h } \in \mathbb { R } ^ { \mathcal { E } \times \mathcal { R } }$ and $\dot { \boldsymbol { A } } _ { t } \in \mathbb { R } ^ { \mathcal { R } \times \mathcal { E } } . ~ \boldsymbol { A } _ { h }$ represents the out-degree edge of any relation from any node and $\pmb { A } _ { t }$ represents the in-degree edge of any relation pointing to any node. Then, the eight types of motif edges shown in Figure 6 can be obtained through a sparse matrix multiplication

![](images/b05502bfbaa8dfe2c6a29df9cdf4816cf5effb56106d193131edc0e2c9f634e1.jpg)

![](images/1165ebd404aaebfb6ec11eb5cb539dbe45838bc9ced0d6ee1ba1848ad19ecf6f.jpg)

![](images/8981d6a1819d491fec84a7b045c2eb4bea7901e9d325e1346e959b9829daccd2.jpg)

![](images/ed281efa31b1940719fdcc4bbf60fd467d7a280b90bd1e13e153862170eecdb6.jpg)  
(1) tail-to-tail  
(2) head-to-head  
(3) tail-to-head

(4) head-to-tail  
![](images/1cdbe9c7d755239e56e4286de209f1c4c7d9499ff6e9b533409679e4eb60629c.jpg)

![](images/c4f5276e8fa9609a42839b7596b4477d25d537b337c47949cc12983e1287d31a.jpg)

![](images/e2f2e0054c091f939e519e74ccc11c004ed490762470fe8e3930952424d6e370.jpg)  
(5) tail-forward-head  
(6) tail-forward-tail

![](images/9d86964e15f7d62555bf418d711ce7b16e5250e6e21b2e95a5de5e56d3bd5311.jpg)  
(7) head-forward-head  
(8) head-forward-tail  
Figure 6: Motif edges of relations in a KG, whcih can be grouped into three binary edges and four ternary edges. A binary edge represents the state of the entity shared by two ordered relations, including (1) tail-to-tail: a shared tail entity, (2) head-to-head: a shared head entity, (3) tail-to-head: the tail entity of the previous relation being the head entity of the next relation, and (4) head-totail: the head entity of the previous relation being the tail entity of the next relation. A ternary edge represents the state of a forward triplet shared by two ordered relations, which is a hyperedge containing three relation nodes, including (5) tail-forward-head: the forward triplet as the tail of the previous relation and the head of the next relation, (6) tail-forward-tail: the forward triplet as the shared tail of two relations, (7) head-forward-head: the forward triplet as the shared head of two relations, and (8) head-forward-tail: the forward triplet as the head of the previous relation and the tail of the next relation.

(spmm) operator:

$$
\operatorname { T a i l - t o - t a i l } { ( t 2 t ) } \colon A _ { t 2 t } = \operatorname { s p m m } ( A _ { t } , A _ { t } ^ { \operatorname { T } } ) \in \mathbb { R } ^ { \mathcal { R } \times \mathcal { R } }
$$

Binary edges =

$$
\mathbf { H e a d - t o - h e a d } ( h 2 h ) ;
$$

$$
\overset { \vartriangle } { \mathbf { A } _ { h 2 h } } = \mathrm { s p m m } ( \boldsymbol { A } _ { h } ^ { \mathrm { { T } } } , \boldsymbol { A } _ { h } ) \in \mathbb { R } ^ { \mathcal { R } \times \mathcal { R } }
$$

$$
\mathbf { T a i l - t o - h e a d } ( t 2 h ) ;
$$

$$
\pmb { A } _ { t 2 h } = \mathrm { s p m m } ( \pmb { A } _ { t } , \pmb { A } _ { h } ) \in \mathbb { R } ^ { \mathcal { R } \times \mathcal { R } }
$$

Head-to-tail (h2t):

$$
\pmb { A } _ { h 2 t } = \mathrm { s p m m } ( \pmb { A } _ { h } ^ { \mathrm { T } } , \pmb { A } _ { t } ^ { \mathrm { T } } ) \in \mathbb { R } ^ { \mathcal { R } \times \mathcal { R } }
$$

Tail-forward-head (tfh):

$$
A _ { t f h } = \operatorname { s p m m } ( A _ { t } , A , A _ { h } ) \in \mathbb { R } ^ { \mathcal { R } \times \mathcal { R } \times \mathcal { R } }
$$

Ternary edges =

Tail-forward-tail (tft):

$$
A _ { t f t } = \operatorname { s p m m } ( A _ { t } , A , A _ { t } ^ { \mathrm { T } } ) \in \mathbb { R } ^ { \mathcal { R } \times \mathcal { R } \times \mathcal { R } }
$$

Head-forward-heaed

$$
\ O _ { \mathfrak { h f h } } ) \colon { \pmb { A } } _ { t f t } = \operatorname { s p m m } ( { \pmb { A } } _ { h } ^ { \mathrm { T } } , { \pmb { A } } , { \pmb { A } } _ { h } ) \in \mathbb { R } ^ { \mathcal { R } \times \mathcal { R } \times \mathcal { R } }
$$

Head-forward-tail (hft):

$$
A _ { h f t } = \operatorname { s p m m } ( A _ { h } ^ { \mathrm { T } } , A , A _ { t } ^ { \mathrm { T } } ) \in \mathbb { R } ^ { \mathcal { R } \times \mathcal { R } \times \mathcal { R } }
$$

## C CONSTRUCTION DETAILS OF THE KG TOKENIZER AND THE RCMPREASONER

Given that relational graphs contain ternary edges, there are certain differences in the construction details of NBFNet between the original KGs and the relational graphs in the KG tokenizer and the RCMP Reasoner.

For NBFNet on a relational graph, we need to consider the positional information of the relational nodes in the hyperedges. Therefore, Eq. (1) can be concretized as

$$
\begin{array} { l } { r _ { j | q } ^ { ( 0 ) } = \mathbb { I } ( r _ { j } = r _ { q } ) \ast \mathbf { 1 } ^ { d } , } \\ { r _ { j | q } ^ { ( n ) } = \sigma \Big ( W _ { r } ^ { ( n - 1 ) } \Big [ r _ { j | q } ^ { ( n - 1 ) } | | \sum _ { r ^ { * } \in \mathcal { R } ^ { * } } r ^ { * } \odot \big ( \underset { r _ { z } \in \mathcal { N } _ { r ^ { * } } ( r _ { j } ) } { \odot } \big ( r _ { z | q } ^ { ( n - 1 ) } + p _ { z } ) \big ) \Big ] \Big ) , r ^ { * } \in R ^ { * } , } \end{array}\tag{17}
$$

where $\sigma ( \cdot )$ is a ReLU activation function, ⊙ is the Hadamard product operator, and $W _ { r } ^ { ( n - 1 ) } \in \mathbb { R } ^ { d \times d }$ and $\pmb { p } _ { z } \in \mathbb { R } ^ { d }$ is a trainable parameter matrix and a position embedding, respectively.

For the original KG, NBFNet only processes directed binary edges, so position information can be ignored. Therefore, Eq. (2) can be concretized as

$$
\begin{array} { r l } & { \pmb { e } _ { i | h } ^ { ( 0 ) } = \mathbb { I } ( e _ { i } = e _ { h } ) \ast \pmb { r } _ { q | q } ^ { ( N ) } , } \\ & { \pmb { e } _ { i | h } ^ { ( n ) } = \sigma \Big ( \pmb { W } _ { e } ^ { ( n - 1 ) } \Big [ \pmb { e } _ { i | h } ^ { ( n - 1 ) } \Big | \Big | \displaystyle \sum _ { r _ { j } \in \mathcal { R } } \displaystyle \sum _ { e _ { z } \in \mathcal { N } _ { r _ { j } } ( e _ { i } ) } \pmb { r } _ { j | q } ^ { ( N ) } \odot \pmb { e } _ { z | h } ^ { ( n - 1 ) } \Big ] \Big ) , } \end{array}\tag{18}
$$

where $W _ { e } ^ { ( n - 1 ) } \in \mathbb { R } ^ { d \times d }$ is a trainable parameter matrix.

Similarly, Eqs. (11) and (12) can be concretized as Eqs. (19) and (20), respectively:

$$
\begin{array} { r l } & { \hat { r } _ { j | q } = \displaystyle \sum _ { < r , v > \in \mathbb { U } } \mathbb { I } ( r _ { j } = r ) * v , \mathrm { ~ w h e r e ~ } U = \{ < r _ { q } , \mathbf { 1 } ^ { d } > \} \cup \Big \{ < r _ { z } , f _ { \mathrm { d o w n } } ( h _ { L + z } ) > \Big \} _ { z = 1 } ^ { \varepsilon } } \\ & { \hat { r } _ { j | q } ^ { ( n ) } = \sigma \Big ( \hat { W } _ { r } ^ { ( n - 1 ) } \left[ \hat { r } _ { j | q } ^ { ( n - 1 ) } \big | \displaystyle \sum _ { r ^ { * } \in \mathbb { R } ^ { * } } \hat { r } ^ { * } \odot \big ( \displaystyle \sum _ { r _ { z } \in \mathcal { N } _ { r ^ { * } } ( r _ { j } ) } ( \hat { r } _ { z | q } ^ { ( n - 1 ) } + p _ { z } ) \big ) \right] \Big ) , \hat { r } ^ { * } \in \hat { R } ^ { * } , } \\ & { \quad \quad \quad \quad \quad \hat { e } _ { i | h } ^ { ( 0 ) } = \mathbb { I } ( e _ { i } = e _ { h } ) * \hat { r } _ { q | q } ^ { ( N ) } , } \\ & { \quad \quad \quad \quad \quad \hat { e } _ { i | h } ^ { ( n ) } = \sigma \Big ( \hat { W } _ { e } ^ { ( n - 1 ) } \left[ \hat { e } _ { i | h } ^ { ( n - 1 ) } \big | \displaystyle \sum _ { r _ { j } \in \mathcal { R } } \displaystyle \sum _ { e _ { z } \in \mathcal { N } _ { r _ { j } } ( e _ { i } ) } \hat { r } _ { j | q } ^ { ( N ) } \odot \hat { e } _ { z | h } ^ { ( n - 1 ) } \right] \Big ) . } \end{array}\tag{19}
$$

(20)

Here, $\hat { W } _ { r } ^ { ( n - 1 ) } , \hat { W } _ { e } ^ { ( n - 1 ) } \in \mathbb { R } ^ { d \times d }$ are two trainable parameter matrix.

## D DISCUSSION OF THE SRM IN-CONTEXT LAYER

This section discusses the effectiveness of the proposed SRM mechanism from the perspectives of structural representation alignment and semantic drift suppression. First, we provide the following explanation of the assumption and definitions required for subsequent analysis.

Definition 1 (In-context learning for the last token). Given a query triplet $< e _ { h } , r _ { q } , ? >$ and the corresponding instruction representation $\pmb { X } \in \mathbb { R } ^ { L \times F }$ , according to Eq. (4), the hidden state ofthe last token $\pmb { x } _ { L } \in \pmb { X }$ in the vanilla in-context learning module can be expressed as:

$$
\begin{array} { c } { { \displaystyle h _ { L } ^ { \mathrm { b a s e } } = \sum _ { i \leq L } \alpha _ { i } f _ { V } ( { \pmb x } _ { i } ) , } } \\ { { \displaystyle \alpha _ { i } = \frac { \exp { ( a _ { i } ) } } { \sum _ { j \leq L } \exp { ( a _ { j } ) } } , a _ { i } = \frac { f _ { Q } \left( { \pmb x } _ { L } \right) \left[ f _ { K } \left( { \pmb x } _ { i } \right) \right] ^ { T } } { \sqrt { F } } . } } \end{array}
$$

Then, we introduce the structural relation memory $R _ { \mathrm { m e m } } = \left\{ { r _ { k } \in R _ { | q } } \right\} _ { k = 1 } ^ { K }$ from Eq. (7) into the in-context learning module, the hidden state of the last token in Eq. (8) can be expressed as:

$$
\hat { \alpha } _ { i } = \frac { \displaystyle \exp { ( a _ { i } ) } } { \displaystyle \sum _ { j \le L } \exp { ( a _ { j } ) } + \sum _ { z \le K } \exp { ( b _ { z } ) } } , \beta _ { k } = \frac { \displaystyle \exp { ( a _ { k } ) } } { \displaystyle \sum _ { j \le L } \exp { ( a _ { j } ) } + \sum _ { z \le K } \exp { ( b _ { z } ) } } , b _ { k } = \frac { f _ { Q } ( x _ { L } ) [ m _ { K } ( r _ { k } ) ] ^ { T } } { \sqrt { F } } .
$$

Definition 2 (Structural subspace and structural observables). Given a query triplet $< e _ { h } , r _ { q } , ? >$ define the structural subspace constructed by SIRLM as:

$$
\mathcal { S } = \mathrm { s p a n } ( \{ f _ { \mathrm { u p } } ( \boldsymbol { r } ) | \boldsymbol { r } \in \boldsymbol { R } _ { | q } \} \cup \{ m _ { V } ( \boldsymbol { r } _ { k } ) | \boldsymbol { r } _ { k } \in \boldsymbol { R } _ { \operatorname* { m e m } } \} _ { k = 1 } ^ { K } ) .
$$

Let $\Pi _ { \mathcal { S } }$ denote the orthogonal projection operator onto S. Then any $\mathbf { \boldsymbol { x } } \in \mathbb { R } ^ { F }$ can be uniquely decomposed in $s$ as:

$$
\pmb { x } = \pmb { \Pi } _ { S } \pmb { x } + \pmb { \Pi } _ { S } ^ { \bot } \pmb { x } ,
$$

where Π<sub>S</sub>x $\in S$ represents the structural component of x and $\Pi _ { S } ^ { \perp }$ x is the semantic drift component orthogonal $t o \ : S _ { }$

Furthermore, define:

$$
{ \pmb u } _ { q } = \frac { { \pmb \Pi } _ { S } f _ { \mathrm { u p } } ( { \pmb r } _ { q | q } ) } { \big \| { \pmb \Pi } _ { S } f _ { \mathrm { u p } } ( { \pmb r } _ { q | q } ) \big \| _ { 2 } } , r _ { q | q } \in { \pmb R } _ { | q } ,
$$

which represents extracting a unit direction from the structural representation of $r _ { q }$ within the structural subspace S, serving as the structural anchor of $\cdot _ { r _ { q } . }$

Based on $\pmb { u } _ { q }$ and $\Pi _ { S } ,$ , two structural observables can be defined:

• Structural alignment score: $\begin{array} { r } { A _ { q } ( \pmb { x } ) = \pmb { u } _ { q } ^ { \mathrm { T } } \pmb { x } , } \end{array}$ , which measures the relevance ofany x to $r _ { q }$ within the structural subspace S.

• Semantic drift magnitude: $\begin{array} { r } { D _ { \cal S } ( { \pmb x } ) = { \pmb u } _ { q } ^ { \mathrm { T } } { \pmb x } , } \end{array}$ , which measures the relevance of any x to $r _ { q }$ within the structural subspace $S = \| \mathbf { I I } _ { S } ^ { \perp } \pmb { x } \| _ { 2 }$ , which quantifies the component ofany x orthogonal to the structural subspace S.

Assumption A. Given a strictly increasing function $\phi ( \cdot )$ and a strictly decreasing function $\psi ( \cdot )$ , for all Top-K relations selected in Eq. (7), we have:

$$
A _ { q } ( m _ { V } ( { \pmb r } _ { k } ) ) = \phi ( s _ { \mathrm { r e l } } ^ { ( k ) } ) , D s ( m _ { V } ( { \pmb r } _ { k } ) ) = \psi ( s _ { \mathrm { r e l } } ^ { ( k ) } ) .
$$

Assume there exists a threshold $s _ { 0 }$ such that $\left\{ s _ { \mathrm { r e l } } ^ { ( k ) } > s _ { 0 } \right\} _ { k = 1 } ^ { K }$ . Then for $h _ { L } ^ { \mathrm { b a s e } }$ , we have:

$$
A _ { q } ( h _ { L } ^ { \mathrm { b a s e } } ) \leq \phi ( s _ { 0 } ) , D s ( h _ { L } ^ { \mathrm { b a s e } } ) \geq \psi ( s _ { 0 } ) .
$$

That is, the aggregate result of the vanilla in-context learning module has a clear separation boundary from the structural reasoning subspace of the query triplet.

Based on the above definitions and assumptions, we present the following proposition.

Proposition 1. Given a query triplet $< e _ { h } , r _ { q } , ? >$ and its corresponding structural subspace $s ,$ let $h _ { L } ^ { \mathrm { S R M } }$ and $h _ { L } ^ { \mathrm { b a s e } }$ be the outputs obtained from the SRM-based and vanilla in-context learning modules, respectively. Then, theformer exhibits stronger structural representation alignment and better suppression ofsemantic drift than the latter, i.e.,

$$
A _ { q } ( h _ { L } ^ { \mathrm { S R M } } ) > A _ { q } ( h _ { L } ^ { \mathrm { b a s e } } ) , D _ { S } ( h _ { L } ^ { \mathrm { S R M } } ) < D _ { S } ( h _ { L } ^ { \mathrm { b a s e } } ) .
$$

Proof. We prove the two claims separately.

(I) Structural alignment improvement. By Definitions 1 and 2, we have

$$
\left\{ \begin{array} { l l } { \displaystyle { A _ { q } \big ( h _ { L } ^ { \mathrm { b a s e } } \big ) = \frac { \sum _ { i \leq L } \exp \big ( a _ { i } \big ) A _ { q } \big ( f _ { V } ( { \bf x } _ { i } ) \big ) } { \sum _ { j \leq L } \exp \big ( a _ { j } \big ) } } } \\ { \displaystyle { A _ { q } \big ( h _ { L } ^ { \mathrm { S R M } } \big ) = \frac { i \leq L } { \frac { \sum } { j \leq L } \exp \big ( a _ { j } \big ) + \displaystyle \sum _ { z \leq K } \exp \big ( b _ { z } \big ) } + \sum _ { k \leq K } \beta _ { k } A _ { q } \big ( m _ { V } \big ( r _ { k } \big ) \big ) } } \\ { \displaystyle { \sum _ { j \leq L } \exp \big ( a _ { j } \big ) } } \\ { \displaystyle { \Rightarrow A _ { q } \big ( h _ { L } ^ { \mathrm { S R M } } \big ) = \frac { j \leq L } { \sum _ { j \leq L } \exp \big ( a _ { j } \big ) + \displaystyle \sum _ { z \leq K } \exp \big ( b _ { z } \big ) } A _ { q } \big ( h _ { L } ^ { \mathrm { b a s e } } \big ) + \sum _ { k \leq K } \beta _ { k } A _ { q } \big ( m _ { V } \big ( r _ { k } \big ) \big ) . } } \end{array} \right.
$$

Let $c = \frac { \sum _ { j \leq L } \exp { ( a _ { j } ) } } { \sum _ { j \leq L } \exp { ( a _ { j } ) } + \sum _ { z \leq K } \exp { ( b _ { z } ) } }$ , then $\sum _ { k \le K } \beta _ { k } = 1 - c$ . Further, define ${ \overline { { A } } } _ { \mathrm { m e m } } = { \frac { \displaystyle \sum _ { k \leq K } \beta _ { k } A _ { q } \left( m _ { V } ( \pmb { r } _ { k } ) \right) } { \displaystyle \sum _ { k \leq K } \beta _ { k } } } ,$ $A _ { q } ( h _ { L } ^ { \mathrm { S R M } } )$ can be rewritten as:

$$
A _ { q } ( { h } _ { L } ^ { \mathrm { S R M } } ) = A _ { q } ( { h } _ { L } ^ { \mathrm { b a s e } } ) + ( 1 - c ) ( \overline { { A } } _ { \mathrm { m e m } } - A _ { q } ( { h } _ { L } ^ { \mathrm { b a s e } } ) ) .
$$

Therefore, it suffices to prove that $\overline { { A } } _ { \mathrm { m e m } } > A _ { q } ( h _ { L } ^ { \mathrm { b a s e } } )$

According to Assumption A and the definition of ${ \overline { { A } } } _ { \mathrm { m e m } }$ , we have:

$$
\left\{ \begin{array} { l l } { A _ { q } ( m _ { V } ( r _ { k } ) ) = \phi ( s _ { \mathrm { r e l } } ^ { ( k ) } ) > \phi ( s _ { 0 } ) } & { \Rightarrow \overline { { A } } _ { \mathrm { m e m } } > A _ { q } ( h _ { L } ^ { \mathrm { b a s e } } ) . } \\ { A _ { q } ( h _ { L } ^ { \mathrm { b a s e } } ) \leq \phi ( s _ { 0 } ) } & { \Rightarrow \overline { { A } } _ { \mathrm { m e m } } > A _ { q } ( h _ { L } ^ { \mathrm { b a s e } } ) . } \end{array} \right.
$$

Therefore, $A _ { q } ( h _ { L } ^ { \mathrm { S R M } } ) > A _ { q } ( h _ { L } ^ { \mathrm { b a s e } } )$ is proven.

(II) Suppression of semantic drift. Following a similar procedure as in the previous proof, and based on Definitions 1 and 2, we obtain:

$$
\begin{array}{c} \begin{array} { r l } & { \{ { D s ( \boldsymbol { h } _ { L } ^ { \mathrm { b a s e } } ) = \| \frac { \displaystyle { \sum _ { j \leq L } ^ { \sum } \exp { ( a _ { i } ) } \Pi _ { S } ^ { \perp } f _ { V } ( { \bf x } _ { i } ) } } { \displaystyle { \sum _ { j \leq L } ^ { \sum } \exp { ( a _ { j } ) } } } \| _ { 2 } } } \\ & { \qquad \quad \{ { D s ( \boldsymbol { h } _ { L } ^ { \mathrm { S R M } } ) = \| \frac { \displaystyle { \sum _ { i \leq L } ^ { \sum } \exp { ( a _ { i } ) } \Pi _ { S } ^ { \perp } f _ { V } ( { \bf x } _ { i } ) } } { \displaystyle { \sum _ { j \leq L } ^ { \sum } \exp { ( a _ { j } ) } + \sum _ { z \leq K } ^ { \sum } \exp { ( b _ { z } ) } } + \sum _ { k \leq K } \beta _ { k } \Pi _ { S } ^ { \perp } m _ { V } ( { \bf r } _ { k } ) } \| _ { 2 } } } \\ & { \qquad \quad \Rightarrow { D s ( \boldsymbol { h } _ { L } ^ { \mathrm { S R M } } ) \leq c D s ( \boldsymbol { h } _ { L } ^ { \mathrm { b a s e } } ) + \displaystyle { \sum _ { k \leq K } ^ { \sum } \beta _ { k } D s ( m _ { V } ( { \boldsymbol { r } _ { k } } ) ) } } . } \end{array}   \end{array}
$$

Let $\overline { { \cal D } } _ { \mathrm { m e m } } = \frac { \sum _ { k \le K } \beta _ { k } D _ { \mathcal { S } } \Big ( m _ { V } ( { \pmb r } _ { k } ) \Big ) } { \sum _ { k < K } \beta _ { k } }$ , then $D _ { S } ( h _ { L } ^ { \mathrm { S R M } } )$ admits the following upper bound:

$$
D _ { S } ( \boldsymbol { h } _ { L } ^ { \mathrm { S R M } } ) \le D _ { S } ( \boldsymbol { h } _ { L } ^ { \mathrm { b a s e } } ) + ( 1 - c ) ( \overline { { D } } _ { \mathrm { m e m } } - D _ { S } ( \boldsymbol { h } _ { L } ^ { \mathrm { b a s e } } ) ) .
$$

Thus, it suffices to prove that $\overline { { D } } _ { \mathrm { m e m } } < D s ( h _ { L } ^ { \mathrm { b a s e } } )$

According to Assumption A and the definition of $\overline { { D } } _ { \mathrm { m e m } }$ , we have:

$$
\left\{ \begin{array} { l l } { D _ { S } ( m _ { V } ( r _ { k } ) ) = \psi ( s _ { \mathrm { r e l } } ^ { ( k ) } ) < \psi ( s _ { 0 } ) } & { \Rightarrow \overline { { \cal D } } _ { \mathrm { m e m } } < D _ { S } ( h _ { L } ^ { \mathrm { b a s e } } ) . } \\ { D _ { S } ( h _ { L } ^ { \mathrm { b a s e } } ) \geq \psi ( s _ { 0 } ) } & \end{array} \right.
$$

Therefore, $D _ { \mathcal { S } } ( h _ { L } ^ { \mathrm { S R M } } ) < D _ { \mathcal { S } } ( h _ { L } ^ { \mathrm { b a s e } } )$ is proven.

Discussion. The proof above establishes the effectiveness of Eq. (8). The SRM branch does not merely append extra structural representation, it actively reorganizes the hidden state by aligning its structural component and suppressing text-only semantic bias. Because the next-relation predictor in Eqs. (9) and (10) operates directly on this hidden state, the hidden-state improvement transfers into a larger score margin for the correct structural relation, thereby promoting the generation of KG-grounded rules.

## E DISCUSSION OF THE RCMP MECHANISM

This section discusses the effectiveness of the proposed RCMP mechanism. Based on the theoretical foundation of the previous studies (Huang et al., 2023; 2025b), we start with the relational Weisfeiler-Leman (WL) test (Barcelo et al.´ , 2022) to analyze the effectiveness of RCMP in terms of structural invariance learning, lower bound of model expression, and stricter reasoning scenarios.

## E.1 A TWO-STAGE WL TEST FOR THE SIL MECHANISM

Before formally discussing the effectiveness of RCMP, we first provide a theoretical background for the WL test of the primary SIL mechanism (Huang et al., 2025a) without RCMP. Giving a KG $\mathcal { G } = ( \mathcal { E } , \mathcal { R } , \mathcal { T } )$ and a corresponding relational graph $\mathcal { G } _ { r } = ( \mathcal { R } , \mathcal { R } ^ { * } , \mathcal { T } ^ { * } )$ ), the WL test of the SIL mechanism can be defined as a two-stage coloring scheme.

Stage 1: relation coloring on $\mathcal { G } _ { r }$ . According to Eq. (1), giving a query relation $r _ { q } \in \mathcal { R }$ , the color of relation $r _ { j } \in \mathcal { R }$ at iteration n is denoted by co $\mathbb { l } _ { r } ^ { ( n ) } ( r _ { j } | r _ { q } )$ and initialized by

$$
\operatorname { c o l } _ { r } ^ { ( 0 ) } ( r _ { j } | r _ { q } ) = \mathbb { I } ( r _ { j } = r _ { q } ) \cdot { \bf 1 } .
$$

It is then updated as

$$
\begin{array} { r l r } { \mathrm { c o l } _ { r } ^ { ( n + 1 ) } ( r _ { j } | r _ { q } ) } & { = } & \\ & { } & { \mathrm { H A S H } \Big ( \mathrm { c o l } _ { r } ^ { ( n ) } ( r _ { j } | r _ { q } ) , \frac { \mathfrak { g } } { \mathrm { d } } ( \{ ( \mathrm { c o l } _ { r } ^ { ( n ) } ( r _ { s } | r _ { q } ) , p _ { s } ) \ | \ ( r _ { s } , p _ { s } ) \in \mathcal { N } _ { r ^ { * } } ( r _ { j } ) \} , r ^ { * } ) \ | \ r ^ { * } \in \mathcal { R } ^ { * } \mathbb { 1 } \Big ) , } \end{array}
$$

where $\mathcal { N } _ { r ^ { * } } ( r _ { j } )$ represents the set of all neighbors of $r _ { j }$ on $r ^ { * }$ . Since $r ^ { * }$ is a hyperedge, $\mathcal { N } _ { r ^ { * } } ( r _ { j } )$ stores each neighbor $r _ { s }$ and its position $p _ { s }$ on $r ^ { * }$ . HASH(·) is an abstract injective coloring function (Barcelo´ et al., 2022) that compresses old colors and neighborhood information into a new color label. As long as the input is different, the output color will be different.

Stage 2: entity coloring on ${ \mathcal { G } } .$ . Given a fixed number $N$ of first-stage iterations, the color of an entity $e _ { v }$ in a candidate triplet $< e _ { u } , r _ { q } , e _ { v } >$ at iteration ℓ is denoted by co $\mathbb { l } _ { e } ^ { ( \ell ) } ( e _ { v } | e _ { u } , r _ { q } )$ . It is initialized by

$$
\mathrm { c o l } _ { e } ^ { ( 0 ) } ( e _ { v } | e _ { u } , r _ { q } ) = \mathbb { I } ( e _ { v } = e _ { u } ) \cdot \mathrm { c o l } _ { r } ^ { ( N ) } ( r _ { q } | r _ { q } ) ,
$$

and updated as

$$
\begin{array} { r l } & { \mathrm { c o l } _ { c } ^ { ( \ell + 1 ) } ( e _ { v } | e _ { u } , r _ { q } ) = } \\ & { \qquad \mathrm { H A S H } \Bigl ( \mathrm { c o l } _ { e } ^ { ( \ell ) } ( e _ { v } | e _ { u } , r _ { q } ) , \ P ( \mathrm { c o l } _ { e } ^ { ( \ell ) } ( e _ { w } | e _ { u } , r _ { q } ) , \mathrm { c o l } _ { r } ^ { ( N ) } ( r _ { j } | r _ { q } ) ) \mid e _ { w } \in  { \mathcal N } _ { r _ { j } } ( e _ { v } ) , r _ { j } \in  { \mathcal R } \mathbb { J } \Bigr ) . } \end{array}
$$

This two-stage WL test characterizes the separating power of the SIL mechanism (Huang et al., 2025a): it is both an upper bound and a tight characterization under injective message-passing choices.

## E.2 A TWO-STAGE WL TEST FOR THE RCMP MECHANISM

This section provides a detailed elaboration on the effectiveness of the proposed RCMP based on Appendix E.1. First, we outline the overall pipeline of the proposed SIRLM as Definition 3 for subsequent analysis.

Definition 3. Giving a KG $\mathcal { G } = ( \mathcal { E } , \mathcal { R } , \mathcal { T } )$ and a corresponding relational graph $\mathcal { G } _ { r } = ( \mathcal { R } , \mathcal { R } ^ { * } , \mathcal { T } ^ { * } )$ , SIRLM processes a query $< e _ { u } , r _ { q } , ? >$ through the following stages:

Stage 1: structural tokenization. We first run Eqs. (1) and (2) to obtain structural token embeddigns of relation $\boldsymbol { r } _ { j | q } \in R _ { | q }$ and entity $\boldsymbol { e } _ { i \vert u } \in E _ { \vert u }$ . These embeddings are used as structure-only tokens; no entity names, relation names, or textual attributes are used. In particular, only $e _ { u \mid u }$ and $r _ { q \mid q }$ are exposed to the rule generator as query context.

Stage 2: structure-aware rule generation. Let $\mathcal { P } _ { \mathrm { s y s } }$ be a fixed system prompt. For a query $r _ { q } ( e _ { u } , ? )$ define the input prompt as

$$
\Pi ( e _ { u } , r _ { q } ) = [ { \mathcal { P } } _ { \mathrm { s y s } } ; e _ { u | u } ; r _ { q | q } ] .
$$

LLM generates a rule consisting of a sequence of atomic relation

$$
\rho ( e _ { u } , r _ { q } ) = r _ { 1 | \Pi ( e _ { u } , r _ { q } ) } , \ldots , r _ { Z | \Pi ( e _ { u } , r _ { q } ) } \in \mathcal { R }
$$

and corresponding hidden states

$$
h _ { 1 | \Pi ( e _ { u } , r _ { q } ) } , \ldots , h _ { Z | \Pi ( e _ { u } , r _ { q } ) } .
$$

$A t \ z – t h$ generation iteration, the last hidden state $h _ { z - 1 | \Pi ( e _ { u } , r _ { q } ) }$ is computed by a LLM fed with $\Pi ( e _ { u } , r _ { q } )$ and the previously $z - 1$ generated atomic relations. The next atomic relation is then selected by a lookup over the structured relation-token bank $R _ { | q \cdot }$

$$
r _ { z | \Pi ( e _ { u } , r _ { q } ) } = \arg \operatorname* { m a x } _ { r _ { j } \in R } \mathrm { S c o r e } \big ( h _ { z - 1 | \Pi ( e _ { u } , r _ { q } ) } , r _ { j | q } \big ) ,
$$

where $\mathrm { S c o r e } ( \cdot )$ is a relation-token lookup scorer.

Stage 3: RCMP initialization $o f { \mathcal { G } } _ { r }$ . For each relation $r \in \mathcal { R }$ , define the accumulated rule representation as

$$
\pmb { x } ( r | e _ { u } , r _ { q } ) = \sum _ { z : r _ { z | \Pi ( e _ { u } , r _ { q } ) } = r } { \pmb h } _ { z | \Pi ( e _ { u } , r _ { q } ) } ,
$$

with the convention that $\pmb { x } ( r | e _ { u } , r _ { q } ) = \pmb { 0 }$ if r does not occur in $\rho ( e _ { u } , r _ { q } )$ . This additive definition directly handles repeated relations, that $i s ,$ every occurrence contributes its own autoregressive hidden state, and all such occurrence-specific hidden states are accumulated.

Giving the generated rule $\rho ( e _ { u } , r _ { q } )$ and the accumulated rule representation $\pmb { x } ( \cdot | e _ { u } , r _ { q } )$ , the RCMP initialization $o f { \mathcal { G } } _ { r }$ can be concretized as

$$
\widetilde { \mathrm { I N I T } } _ { 1 } ( r _ { j } | e _ { u } , r _ { q } ) = \mathbb { I } ( r _ { j } = r _ { q } ) \cdot { \mathbf { 1 } } + { \mathbf { x } } ( r _ { j } | e _ { u } , r _ { q } ) ,
$$

where 1 $\neq \mathbf { 0 }$ are fixed vectors. Equivalently, if the query relation $r _ { q }$ appears in the generated rule body, then its node in the relational graph is initialized by the anchor vector 1 plus the sum ofall hidden states associated with occurrences $o f r _ { q } i n \rho ( e _ { u } , r _ { q } )$ . Likewise, ifany non-query relation $r _ { j }$ appears multiple times in $\rho ( e _ { u } , r _ { q } )$ , then its initialization is the sum ofthe hidden states associated with all ofits occurrences.

It should be emphasized that RCMP only changes the initialization strategy of the relational graph $\mathcal { G } _ { r }$ in SIL with $\widetilde { \mathrm { I N I T } } _ { 1 } ( r _ { j } | e _ { u } , r _ { q } )$ , while keeping $\mathcal { G } _ { r }$ and all relation and entity message-passing operators unchanged.

Assumption B. We make the following assumptions.

1. Structure-token invariance. The base embeddings $\pmb { r } _ { j | q } \in \pmb { R } _ { | q }$ and $\boldsymbol { e } _ { i \vert u } \in E _ { \vert u }$ are structurally invariants in Definition 3.

2. Equivariance of LLM generation for structural relation tokens. For every isomorphism $h \overset { \cdot } { = } ( \pi , \phi ) : \mathcal { G } \to \mathcal { G } ^ { \prime }$ , the autoregressive decoder and the lookup rule are equivariant with respect to the induced renaming of the structural tokens, i.e.,

$$
\begin{array} { r } { h _ { z | \Pi ( e _ { u } , r _ { q } ) } = h _ { z | \Pi ( \pi ( e _ { u } ) , \phi ( r _ { q } ) ) } , \qquad r _ { z | \Pi ( \pi ( e _ { u } ) , \phi ( r _ { q } ) ) } = \phi \bigl ( r _ { z | \Pi ( e _ { u } , r _ { q } ) } \bigr ) . } \end{array}
$$

3. Anchor separation of relations. Query relations are disjoint from non-query relations, $i . e .$

$$
\mathbf { 1 } + \mathbf { x } ( r _ { j } | e _ { u } , r _ { q } ) \neq \mathbf { x } ( r _ { j } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } ) \qquad \mathrm { w h e n e v e r } r _ { j } ^ { \prime } \neq r _ { q } ^ { \prime } .
$$

Equivalently, $\widetilde { \mathrm { I N I T } _ { 1 } } ( r _ { j } | e _ { u } , r _ { q } ) = \widetilde { \mathrm { I N I T } _ { 1 } } ( r _ { j } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } )$ implies $r _ { j } = r _ { q }$ if and only i $\dot { \boldsymbol { r } } _ { j } ^ { \prime } = \boldsymbol { r } _ { q } ^ { \prime } .$

4. Injective operator in RCMP. The update, aggregation, and message functions in both the relation encoder and the entity encoder are injective, exactly as in the constructive direction of Appendix E.1.

The next analysis shows that the proposed RCMP preserves structural invariance and refines the expressive power of the original SIL model.

Proposition 2. Under Assumptions B1-B4, thefollowing holdfor RCMP.

I. The initialization $\widetilde { \mathrm { I N I T } } _ { 1 } ( u , q , r )$ has relation invariance. Consequently, RCMP computes relation invariants in the relation encoder and link invariants in the entity encoder.

II. Let

$$
\widetilde { \mathrm { c o l } } _ { r } ^ { ( 0 ) } ( r _ { j } | e _ { u } , r _ { q } ) = \chi \left( \widetilde { \mathrm { I N I T } } _ { 1 } ( r _ { j } | e _ { u } , r _ { q } ) \right) ,
$$

where $\chi$ is injective, and let $\widetilde { ( \mathrm { c o l } _ { r } , \mathrm { c o l } _ { e } ) }$ be the two-stage WL test of RCMP, while keeping all update rules unchanged. Then for all $n , \ell \geq 0 ,$ , we have

$$
\widetilde { \mathrm { c o l } } _ { r } ^ { ( n ) } ( r _ { j } | e _ { u } , r _ { q } ) = \widetilde { \mathrm { c o l } } _ { r } ^ { ( n ) } ( r _ { j } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } ) \Longrightarrow \mathrm { c o l } _ { r } ^ { ( n ) } ( r _ { j } | r _ { q } ) = \mathrm { c o l } _ { r } ^ { ( n ) } ( r _ { j } ^ { \prime } | r _ { q } ^ { \prime } )
$$

and

$$
\widetilde { \mathrm { c o l } } _ { e } ^ { ( \ell ) } ( e _ { v } | e _ { u } , r _ { q } ) = \widetilde { \mathrm { c o l } } _ { e } ^ { ( \ell ) } ( e _ { v } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } ) \Longrightarrow \mathrm { c o l } _ { e } ^ { ( \ell ) } ( e _ { v } | e _ { u } , r _ { q } ) = \mathrm { c o l } _ { e } ^ { ( \ell ) } ( e _ { v } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } ) .
$$

Hence RCMP is at least as expressive as the original SIL model.

III. Assume there exist a que $\begin{array} { r } { r \mathrm { y } < e _ { u } , r _ { q } , ? > } \end{array}$ and two relations $r _ { j } , r _ { s } \in \mathcal { R }$ such that

$$
\mathrm { c o l } _ { r } ^ { ( n ) } ( r _ { j } , r _ { q } ) = \mathrm { c o l } _ { r } ^ { ( n ) } ( r _ { s } , r _ { q } ) \quad f o r a l l n \ge 0 ,
$$

but

$$
\widetilde { \mathrm { I N I T 1 } } ( r _ { j } | e _ { u } , r _ { q } ) \neq \widetilde { \mathrm { I N I T } } _ { 1 } ( r _ { s } | e _ { u } , r _ { q } ) .
$$

Then the refinement is strict, that $i s ,$ there exists a KG and two query links that cannot be separated by any instance of SIL, but can be separated by some instance of RCMP.

Proof. We prove the three claims as follows.

$\mathrm { { \bf I } . \widetilde { I N I T _ { 1 } } }$ is a relation invariant. By Assumption B1, $R _ { | q }$ and $\pmb { { E } } _ { | u }$ depend only on KG structure, not on concrete entity or relation names. Since the prompt $\dot { \Pi } ( e _ { u } , r _ { q } )$ is formed only by these structural tokens and a fixed system prompt, Assumption B2 implies that for every graph isomorphism $h = ( \pi , \phi ) : \mathcal { G } \to \mathcal { G } ^ { \prime }$ , we have

$$
\pmb { h } _ { z | \Pi ( e _ { u } , r _ { q } ) } = \pmb { h } _ { z | \Pi ( \pi ( e _ { u } ) , \phi ( r _ { q } ) ) } \qquad \mathrm { a n d } \qquad r _ { z | \Pi ( \pi ( e _ { u } ) , \phi ( r _ { q } ) ) } = \phi \bigl ( r _ { z | \Pi ( e _ { u } , r _ { q } ) } \bigr )
$$

for all decoding steps $z .$ Therefore, for every relation $r _ { j } \in \mathcal { R } _ { \mathbf { \partial } }$ , the accumulated rule signal satisfies

$$
\pmb { x } ( r _ { j } | e _ { u } , r _ { q } ) = \pmb { x } ( \phi ( r _ { j } ) | \pi ( e _ { u } ) , \phi ( r _ { q } ) ) .
$$

Since the query-anchor case $r _ { j } = r _ { q }$ is mapped to $\phi ( r _ { j } ) = \phi ( r _ { q } )$ under the same isomorphism, the initialization satisfies

$$
\widetilde { \mathrm { I N I T } _ { 1 } } ( r _ { j } | e _ { u } , r _ { q } ) = \widetilde { \mathrm { I N I T } _ { 1 } } ( \phi ( r _ { j } ) | \pi ( e _ { u } ) , \phi ( r _ { q } ) ) .
$$

Hence $\widetilde { \mathrm { I N I T } } _ { 1 }$ is a relation invariant.

Now the relation encoder follows exactly the same inductive argument used in the main manuscript. Once the initialization $\widetilde { \mathrm { I N I T } } _ { 1 }$ is a relation invariant, every subsequent layer of the relation encoder remains a relation invariant because message, aggregation, and update only combine invariant quantities over the relational graph $\mathcal { G } _ { r }$ . Likewise, the entity encoder receives invariant relation representations and uses the same message-passing structure as the original SIL model, so it computes link invariants.

II. RCMP is at least as expressive as the original SIL model. We define the first-stage colors of RCMP by

$$
\widetilde { \mathrm { c o l } } _ { r } ^ { ( 0 ) } ( r _ { j } | e _ { u } , r _ { q } ) = \chi \left( \widetilde { \mathrm { I N I T } } _ { 1 } ( r _ { j } | e _ { u } , r _ { q } ) \right) ,
$$

Then, for $n \geq 0$ , we use the same first-stage update as in Appendix E.1, but with the richer initial coloring:

$$
\begin{array} { r l r } {  { \widetilde { \mathrm { c o l } } _ { r } ^ { ( n + 1 ) } ( r _ { j } | e _ { u } , r _ { q } ) = } } \\ & { } & { \mathrm { H A S H } ( \widetilde { \mathrm { c o l } } _ { r } ^ { ( n ) } ( r _ { j } | e _ { u } , r _ { q } ) ,   \{ ( \{ \langle \widetilde { \mathrm { c o l } } _ { r } ^ { ( n ) } ( r _ { s } | e _ { u } , r _ { q } ) , p _ { s } ) \ | \ ( r _ { s } , p _ { s } ) \in \mathcal { N } _ { r ^ { * } } ( r _ { j } ) \} , r ^ { * } ) \ | \ r ^ { * } \in \mathcal { R } ^ { * } \} ) . } \end{array}
$$

Similarly, define the second-stage colors by

$$
\begin{array} { r } { \widetilde { \mathrm { c o l } } _ { e } ^ { ( 0 ) } ( e _ { v } | e _ { u } , r _ { q } ) = \mathbb { I } ( e _ { v } = e _ { u } ) \cdot \widetilde { \mathrm { c o l } } _ { r } ^ { ( N ) } ( r _ { q } | e _ { u } , r _ { q } ) , } \end{array}
$$

and for $\ell \geq 0$

$$
\begin{array} { r l } & { \widetilde { \mathrm { c o l } } _ { e } ^ { ( \ell + 1 ) } ( e _ { v } | e _ { u } , r _ { q } ) = } \\ & { \qquad \mathrm { H A S H } \Bigl ( \widetilde { \mathrm { c o l } } _ { e } ^ { ( \ell ) } ( e _ { v } | e _ { u } , r _ { q } ) , \frac { \ell } { \ell } ( \widetilde { \mathrm { c o l } } _ { e } ^ { ( \ell ) } ( e _ { w } | e _ { u } , r _ { q } ) , \widetilde { \mathrm { c o l } } _ { r } ^ { ( N ) } ( r _ { j } | e _ { u } , r _ { q } ) ) \mid e _ { w } \in \mathcal { N } _ { r _ { j } } ( e _ { v } ) , r _ { j } \in \mathcal { R } \mathbb { 1 } \Bigr ) . } \end{array}
$$

(1) We first prove by induction on n that

$$
\widetilde { \mathrm { c o l } } _ { r } ^ { ( n ) } ( r _ { j } | e _ { u } , r _ { q } ) = \widetilde { \mathrm { c o l } } _ { r } ^ { ( n ) } ( r _ { j } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } ) \Longrightarrow \mathrm { c o l } _ { r } ^ { ( n ) } ( r _ { j } | r _ { q } ) = \mathrm { c o l } _ { r } ^ { ( n ) } ( r _ { j } ^ { \prime } | r _ { q } ^ { \prime } ) .
$$

For $n = 0 ,$ , injectivity of $\chi$ gives $\widetilde { \mathrm { I N I T } _ { 1 } } ( r _ { j } | e _ { u } , r _ { q } ) = \widetilde { \mathrm { I N I T } _ { 1 } } ( r _ { j } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } )$ . According to Assumption B3, such an equation preserves whether the anchor contribution $\mathbb { I } ( r _ { j } = r _ { q } ) \cdot { \bf 1 }$ is present. Therefore, $\mathbb { I } ( r _ { j } = r _ { q } ) = \mathbb { I } ( r _ { j } ^ { \bar { \prime } } = r _ { q } ^ { \prime } )$ , which is exactly

$$
\mathrm { c o l } _ { r } ^ { ( 0 ) } ( r _ { j } | r _ { q } ) = \mathrm { c o l } _ { r } ^ { ( 0 ) } ( r _ { j } ^ { \prime } | r _ { q } ^ { \prime } ) .
$$

Assume the implication holds at layer n and suppose

$$
\widetilde { \mathrm { c o l } } _ { r } ^ { ( n + 1 ) } ( r _ { j } | e _ { u } , r _ { q } ) = \widetilde { \mathrm { c o l } } _ { r } ^ { ( n + 1 ) } ( r _ { j } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } ) .
$$

Since HASH is injective, both the previous colors and the corresponding position-aware neighbor multisets of SIL must agree. Applying the induction hypothesis to all first-stage colors appearing inside those multisets shows that the original colors in Appendix E.1 also agree. Therefore the multisets defining co $1 _ { r } ^ { ( n + 1 ) } ( r _ { j } | r _ { q } )$ and $\mathrm { c o l } _ { r } ^ { ( n + 1 ) } ( r _ { j } ^ { \prime } | r _ { q } ^ { \prime } )$ coincide, thus

$$
\mathrm { c o l } _ { r } ^ { ( n + 1 ) } ( r _ { j } | r _ { q } ) = \mathrm { c o l } _ { r } ^ { ( n + 1 ) } ( r _ { j } ^ { \prime } | r _ { q } ^ { \prime } )
$$

is proven.

(2) We next prove by induction on ℓ that

$$
\widetilde { \mathrm { c o l } } _ { e } ^ { ( \ell ) } ( e _ { v } | e _ { u } , r _ { q } ) = \widetilde { \mathrm { c o l } } _ { e } ^ { ( \ell ) } ( e _ { v } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } ) \Longrightarrow { \mathrm { c o l } } _ { e } ^ { ( \ell ) } ( e _ { v } | e _ { u } , r _ { q } ) = { \mathrm { c o l } } _ { e } ^ { ( \ell ) } ( e _ { v } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } ) .
$$

For $\ell = 0$ , equality of the RCMP colors implies

$$
\mathbb { I } ( e _ { v } = e _ { u } ) = \mathbb { I } ( e _ { v } ^ { \prime } = e _ { u } ^ { \prime } ) \qquad \mathrm { a n d } \qquad \widetilde { \mathrm { c o l } } _ { r } ^ { ( N ) } ( r _ { j } | e _ { u } , r _ { q } ) = \widetilde { \mathrm { c o l } } _ { r } ^ { ( N ) } ( r _ { j } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } ) .
$$

By the first-stage result proved above, we obtain

$$
\mathrm { c o l } _ { r } ^ { ( N ) } ( r _ { q } | r _ { q } ) = \mathrm { c o l } _ { r } ^ { ( N ) } ( r _ { q } ^ { \prime } | r _ { q } ^ { \prime } ) ,
$$

hence

$$
\mathrm { c o l } _ { e } ^ { ( 0 ) } ( e _ { v } | e _ { u } , r _ { q } ) = \mathrm { c o l } _ { e } ^ { ( 0 ) } ( e _ { v } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } ) .
$$

Assume the implication holds at layer ℓ and suppose

$$
\widetilde { \mathrm { c o l } } _ { e } ^ { ( \ell + 1 ) } ( e _ { v } | e _ { u } , r _ { q } ) = \widetilde { \mathrm { c o l } } _ { e } ^ { ( \ell + 1 ) } ( e _ { v } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } ) .
$$

Injectivity of HASH yields equality of the previous second-stage colors of SIL and of the corresponding message multisets. By the induction hypothesis on ℓ and the first-stage implication already

proved, the original second-stage colors and original first-stage relation colors also agree. Therefore, the multisets defining $c o l _ { F , T } ^ { ( \ell + 1 ) } \bar { ( } q ( u , v ) )$ and $c o l _ { F , T } ^ { \overline { { ( \ell + 1 ) } } } ( q ^ { \prime } ( u ^ { \prime } , \check { v ^ { \prime } } ) )$ are identical, and the assumption

$$
\mathrm { c o l } _ { e } ^ { ( \ell + 1 ) } ( e _ { v } | e _ { u } , r _ { q } ) = \mathrm { c o l } _ { e } ^ { ( \ell + 1 ) } ( e _ { v } ^ { \prime } | e _ { u } ^ { \prime } , r _ { q } ^ { \prime } )
$$

is proven.

Therefore, the two-stage WL test of RCMP is a refinement of the original two-stage test of SIL on Appendix E.1. Under Assumption B4, the same constructive argument as in Appendix E.1 applies to the present model, because only the initialization changes while the message-passing architecture over $\mathcal { G } _ { r }$ and over the original KG remains unchanged. Therefore, the refined two-stage test characterizes RCMP, and the refinement of the WL tests implies that RCMP is at least as expressive as SIL.

III. Strictness. Assume there exist a query triplet $< e _ { u } , r _ { q } , ? >$ and two relations $r _ { j } , r _ { s } \in \mathcal { R }$ that satisfy

$$
\begin{array} { r } { \mathrm { c o l } _ { r } ^ { ( n ) } ( r _ { j } | r _ { q } ) = \mathrm { c o l } _ { r } ^ { ( n ) } ( r _ { s } | r _ { q } ) \qquad \mathrm { b u t } \qquad \widetilde { \mathrm { I N T } } _ { 1 } ( r _ { j } | e _ { u } , r _ { q } ) \neq \widetilde { \mathrm { I N T } } _ { 1 } ( r _ { s } | e _ { u } , r _ { q } ) \qquad \mathrm { f o r ~ a l l ~ } n \geq 0 . } \end{array}
$$

We then construct a new $\operatorname { K G } { \mathcal { G } } ^ { \star }$ with fresh entities $\{ a , b , c , d , x , y \}$ and triplets $\{ < x , r _ { j } , b > , <$ $y , r _ { s } , d > \}$ , and no other edges incident to b or d.

Consider two query triplets

$$
< a , r _ { q } , b > \mathrm { a n d } < c , r _ { q } , d > ,
$$

for the original SIL model, the incoming relation colors attached to b and d are the same because $r _ { j }$ and $r _ { s }$ are indistinguishable by the first-stage WL test. Moreover, the local entity neighborhoods of b and d are isomorphic because each target node has exactly one incoming edge from a fresh non-source node. Based on the above conditions, we can obtain

$$
\mathrm { c o l } _ { e } ^ { ( \ell ) } ( b | a , r _ { q } ) = \mathrm { c o l } _ { e } ^ { ( \ell ) } ( d | c , r _ { q } ) \qquad \mathrm { f o r ~ a l l ~ } \ell \ge 0 .
$$

For RCMP, we already have $\widetilde { \mathrm { c o l } } _ { r } ^ { ( 0 ) } ( r _ { j } | e _ { u } , r _ { q } ) \neq \widetilde { \mathrm { c o l } } _ { r } ^ { ( 0 ) } ( r _ { s } | e _ { u } , r _ { q } )$ . Therefore, after a first-stage update, the message multiset received by b contains

$$
\widetilde { \left( \mathrm { c o l } _ { e } ^ { ( 0 ) } ( x | a , r _ { q } ) , \widetilde { \mathrm { c o l } } _ { r } ^ { ( N ) } ( r _ { j } | a , r _ { q } ) \right) } ,
$$

whereas the one received by d contains

$$
\widetilde { \left( \mathrm { c o l } _ { e } ^ { ( 0 ) } ( y | c , r _ { q } ) , \widetilde { \mathrm { c o l } } _ { r } ^ { ( N ) } ( r _ { s } | c , r _ { q } ) \right) } .
$$

Based on $\widetilde { \mathrm { c o l } } _ { e } ^ { ( 0 ) } ( e _ { v } | e _ { u } , r _ { q } ) = \mathbb { I } ( e _ { v } = e _ { u } ) \cdot \widetilde { \mathrm { c o l } } _ { r } ^ { ( N ) } ( r _ { q } | e _ { u } , r _ { q } )$ , the first components are equal, but the second components are different, so the multisets in the second-stage test are different. Finally, we can obtain

$$
\widetilde { \mathrm { c o l } } _ { e } ^ { ( \ell ) } ( b | a , r _ { q } ) \neq \widetilde { \mathrm { c o l } } _ { e } ^ { ( \ell ) } ( d | c , r _ { q } ) \qquad \mathrm { f o r ~ a l l ~ } \ell \ge 0 .
$$

Again using the constructive direction of Assumption B4, there exists an instance where RCMP can separate but SIL cannot. This proves the strict refinement. □

Discussion. The proposition identifies a novel perspective of expressiveness beyond changing the motif set $\mathcal { R } ^ { * }$ itself. The original theory strengthens SIL by enriching the relational hypergraph through richer motifs. Our construction keeps ${ \bar { \mathcal { R } } } ^ { * }$ fixed and instead strengthens the base coloring of the first-stage relation process by injecting a structure-only autoregressive rule signal. Therefore, the gain comes from refining the initial partition of relations rather than from adding new motifs. Moreover, the structure-only prompting design is crucial. The invariance statement depends on the equivariance of the generator with respect to isomorphisms, which would generally fail if one injected relation names, entity names, or arbitrary textual attributes into the prompt.

## F TRAINING ALGORITHM

Algorithm 1 provides a complete training process for $\mathrm { S I R L M _ { P T } }$ . In each training iteration, we first convert a query triplet into a instruction form consisting of textual and structural tokens (Step 6). Next, we obtain the structural representaions of entities and relations to construct a tokenizer for the query instruciton (Steps 7 and 8). We then obtain the structural relation memory and conduct in-context learning for the query instruciton (Steps 9 and 10), which provides the hidden states for rule generation (Step 11) and reasoning (Step 12). Finally, we using the reasoning loss to update the model parameters (Step 16) and return a pre-trained parameter set for downstream KGR tasks.

Algorithm 1 Pre-training framework of SIRLM   
Input: KG $\overline { { \mathcal { G } = ( \mathcal { E } , \mathcal { R } , \mathcal { T } ) } } ;$ relational graph $\mathcal { G } _ { r } = ( \mathcal { R } , \mathcal { R } ^ { * } , \mathcal { T } ^ { * } )$ ; trainable parameter set of SIRLM   
Ω; learning rate η; max training step s; batch size b.   
Output: Optimized parameter set Ω.   
1: step = 0   
2: for step $< s$ do   
3: Randomly select b query triplets from $\tau$ to form $\mathcal { T } _ { q }$   
4: $\mathcal { L } _ { t o t a l } = 0$   
5: for $< e _ { h } , r _ { q } , ? >$ in $\mathcal { T } _ { q }$ do   
6: Construct the query instruction X of $< e _ { h } , r _ { q } , ? >$ according to Appendix A   
7: Obtain structural embeddings $R _ { | q }$ and $E _ { | h }$ according to Eqs (1) and (2), respectively   
8: Obtain the tokenizer $\mathrm { T K N } _ { \mathrm { R G } } ^ { = }$ according to Eq. (6)   
9: Obtain structural relation memory $R _ { \mathrm { m e m } }$ by Eq. (7)   
10: Convert X into token sequence using TKN<sub>RG</sub> and conduct in-context learning with   
$R _ { \mathrm { m e m } }$ according to Eq. (8)   
11: Generate the rule body $\rho = { \textstyle \bigwedge } _ { z = 1 } ^ { \varepsilon } r _ { z }$ with the hidden states $\left\{ h _ { L + z } \right\} _ { z = 1 } ^ { \varepsilon }$ using Eq. (10)   
12: Reason the missing entity in the query triplet according to Eqs. (11)-(13)   
13: Calculate the loss L<sub>SFT</sub> using Eq. (14)   
14: $\mathcal { L } _ { t o t a l }  \mathcal { L } _ { t o t a l } + \mathcal { L } _ { \mathrm { S F T } }$   
15: end for   
16: Optimize trainable parameters according to $\boldsymbol { \Omega }  \Omega - \eta \nabla ( \mathcal { L } _ { t o t a l } )$   
17: step ← step + 1   
18: end for   
19: return Ω

Table 4: Comparison of KRLM and SIRLM in terms of TFLOPs, memory footprint, and wall-clock time under pre-training and fine-tuning settings.
<table><tr><td rowspan="2">Metric</td><td colspan="2">Pre-training (3 transductive datasets)</td><td colspan="2">Fine-tuning (FB15k237 v1)</td><td colspan="2">Fine-tuning (FB15k237-25)</td></tr><tr><td>KRLM</td><td> $\mathbf { S I R L M _ { P T } }$ </td><td>KRLM</td><td> $\mathbf { S I R L M _ { S F T } }$ </td><td>KRLM</td><td> $\mathbf { S I R L M _ { S F T } }$ </td></tr><tr><td>Avg. TFLOPs</td><td>3.3436±0.4540</td><td>2.0397±0.3841</td><td>3.2755±0.5208</td><td>2.0683±0.0175</td><td>3.3312±0.4859</td><td>2.2287±0.0000</td></tr><tr><td>(forward propagation 100 steps) Training memory</td><td>36.12 GB / GPU</td><td>27.58 GB / GPU</td><td>32.57 GB / GPU</td><td>11.38 GB / GPU</td><td>32.67 GB / GPU</td><td>13.12 GB / GPU</td></tr><tr><td>Wall-clock time</td><td>3h10m / epoch</td><td>2h38m / epoch</td><td>7m28s / epoch</td><td>3m39s / epoch</td><td>12m13s / epoch</td><td>10m45s / epoch</td></tr></table>

## G COMPUTATIONAL COMPLEXITY

The computational complexity of SIRLM consists of three parts. For the construction of the relational graph $\mathcal { G } ^ { * }$ , since we need to dynamically construct it based on negative sampling during the training process, the time complexity of this part needs to be considered (Huang et al., 2025a). To construct motif edges in $\mathcal { G } ^ { * }$ , we first need to compress $\pmb { A } \in \mathbb { R } ^ { | \pmb { \mathcal { E } } | \times | \mathcal { R } | \times | \dot { \mathcal { E } } | }$ into two sparse matrices $A _ { h } \in$ $\mathbb { R } ^ { | \mathcal { E } | \times | \mathcal { R } | }$ and $\boldsymbol { A } _ { t } \in \mathbb { R } ^ { | \mathcal { R } | \times | \mathcal { E } | }$ , with a complexity of $O ( | \mathcal { E } | ^ { 2 } | \mathcal { R } | )$ . Next, the construction process of the four types of binary edges theoretically requires a complexity of $\mathcal { O } ( | \mathcal { E } | | \mathcal { R } | ^ { 2 } )$ . In practical situations, we use the sparse operator spmm(·) to reduce the complexity of constructing binary edges to $\mathcal { O } ( \operatorname { n n z } ( X | X \in \{ A _ { h } , \hat { A } _ { t } \} ) \times \operatorname { n n z } ( Y | Y \in \{ A _ { h } ^ { \mathrm { T } } , A _ { t } ^ { \mathrm { T } } \} ) )$ , where nnz(·) is a operator finding the number of non-zero element in X and Y. Similarly, the theoretical complexity and actual sparse computational complexity of constructing ternary edges are $\mathcal { O } ( \vert \mathcal { E } \vert ^ { 2 } \vert \mathcal { R } \vert ^ { 2 } + \vert \mathbf { \bar { \mathcal { E } } } \vert \vert \mathcal { R } \vert ^ { 3 } )$ and $\dot { \mathcal { O } } ( \operatorname { n n z } ( X | \dot { X } \in \{ A _ { h } , A _ { t } \} \rangle \times \operatorname { n n z } ( A ) \times \operatorname { n n z } ( Y | \check { Y } \in \{ A _ { h } ^ { \dagger } , A _ { t } ^ { \tilde { \operatorname { Y } } } \} ) )$ , respectively

From the perspective of the KG tokenizer and RCMP reasoner, the time complexity is upper-bounded by the NBFNet executed on G, as $| \mathcal { R } | \ll | \mathcal { E } |$ For each layer on NBFNet, the reasoning time complexity is calculated as $\mathcal { O } ( \vert \mathcal { T } \vert d + \vert \dot { \mathcal { E } } \vert d ^ { 2 } )$ , where d is the embedding dimension. Therefore, for a N-layer NBFNet executed on ${ \mathcal { G } } ,$ its overall time complexity is $\mathcal { O } ( N ( | \mathcal { T } | d + | \mathcal { E } | d ^ { 2 } ) )$ . Furthermore, by using the efficient relational messaging kernel in the Pytorch-geometric library, the complexity of the NBFNet is optimized to $\mathcal { O } ( N \vert \mathcal { E } \vert d )$ (Galkin et al., 2024).

Table 5: Element statistics of KGR dataset. “Triplets” represents the number of total triplets contained in a training/validation/testing graph. “#Valid” and “#Test” are the number of evaluation triplets in the validation and testing graph, respectively.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Dataset</td><td colspan="3">Training graph</td><td colspan="4">Validation Graph</td><td colspan="4">Testing Graph</td></tr><tr><td>Entities Relations Triplets Entities Relations Triplets #Valid Entities Relations Triplets #Test</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Transductive</td><td>FB15k237 (Toutanova &amp; Chen, 2015)</td><td>14541</td><td>237</td><td>272115</td><td>14541</td><td>237</td><td>272115</td><td>17535</td><td>14541</td><td>237</td><td>272115</td><td>20466</td></tr><tr><td>CoDEx-M (Safavi &amp; Koutra, 2020)</td><td>17050</td><td>51</td><td>185584</td><td>17050</td><td>51</td><td>185584</td><td>10310</td><td>17050</td><td>51</td><td>185584</td><td>10311</td></tr><tr><td>WN18RR (Dettmers et al., 2018)</td><td>40943</td><td>11</td><td>86835</td><td>40943</td><td>11</td><td>86835</td><td>3034</td><td>40943</td><td>11</td><td>86835</td><td>3134 2818</td></tr><tr><td rowspan="9"></td><td>NELL995 (Xiong et al., 2017)</td><td>74536</td><td>200</td><td>149678</td><td>74536</td><td>200</td><td>149678 4245</td><td>543 489</td><td>74536 1093</td><td>200 180</td><td>149678 1993</td><td>411</td></tr><tr><td>FB15k237 v1 (Teru et al., 2020)</td><td>1594</td><td>180</td><td>4245</td><td>1594</td><td>180</td><td>9739</td><td></td><td>1660</td><td>200</td><td>4145</td><td>947</td></tr><tr><td>FB15k237 v2 (Teru et al., 2020)</td><td>2608</td><td>200</td><td>9739</td><td>2608</td><td>200</td><td></td><td>1166 2194</td><td></td><td>215</td><td></td><td>1731</td></tr><tr><td>FB15k237 v3 (Teru et al., 2020)</td><td>3668 4707</td><td>215</td><td>17986 27203</td><td>3668 4707</td><td>215</td><td>17986 27203</td><td>3352</td><td>2501 3051</td><td></td><td>7406</td><td>2840</td></tr><tr><td>FB15k237 v4 (Teru et al., 2020)</td><td>2746</td><td>219 9</td><td>5410</td><td>2746</td><td>219 9</td><td></td><td>630</td><td>922</td><td>219</td><td>11714 1618</td><td>373</td></tr><tr><td>WN18RR v1 (Teru et al., 2020)</td><td>6954</td><td>10</td><td>15262</td><td>6954</td><td>10</td><td>5410 15262</td><td>1838</td><td>2757</td><td>9 10</td><td>4011</td><td>852</td></tr><tr><td>WN18RR v2 (Teru et al., 2020)</td><td>12078</td><td>11</td><td>25901</td><td>12078</td><td>11</td><td>25901</td><td>3097</td><td>5084</td><td>11</td><td></td><td>1143</td></tr><tr><td>WN18RR v3 (Teru et al., 2020) WN18RR v4 (Teru et al., 2020)</td><td>3861</td><td>9</td><td>7940</td><td>3861</td><td>9</td><td>7940</td><td>934</td><td>7084</td><td>9</td><td>6327 12334</td><td>2823</td></tr><tr><td>NELL995 v1 (Teru et al., 2020)</td><td>3103</td><td>14</td><td>4687</td><td>3103</td><td>14</td><td>4687</td><td>414</td><td>225</td><td>14</td><td>833</td><td>201</td></tr><tr><td>NELL995 v2 (Teru et al., 2020)</td><td>2564</td><td>88</td><td>8219</td><td>2564</td><td>88</td><td>8219</td><td>922</td><td>2086</td><td></td><td>88</td><td>935</td></tr><tr><td></td><td>4647</td><td>142</td><td>16393</td><td>4647</td><td>142</td><td>16393</td><td>1851</td><td>3566</td><td></td><td>4586</td><td>1620</td></tr><tr><td>NELL995 v3 (Teru et al., 2020) NELL995 v4 (Teru et al., 2020)</td><td>2092</td><td>76</td><td>7546</td><td>2092</td><td>76</td><td>7546</td><td>876</td><td>2795</td><td>142 76</td><td>8048 7073</td><td>1447</td></tr><tr><td>FB15k237-25 (Lee et al., 2023)</td><td>5190</td><td>163</td><td>91571</td><td>4097</td><td>216</td><td>17147</td><td>5716</td><td>4097</td><td></td><td>17147</td><td>5716</td></tr><tr><td>FB15k237-50 (Lee et al., 2023)</td><td>5190</td><td></td><td>85375</td><td>4445</td><td>205</td><td></td><td>11636</td><td>3879</td><td>4445</td><td>216 205</td><td>11636 3879</td></tr><tr><td rowspan="19"></td><td>FB15k237-75 (Lee et al., 2023)</td><td>4659</td><td>153 134</td><td>62809</td><td>2792</td><td>186</td><td>9316</td><td>3106</td><td>2792</td><td>186</td><td>9316</td><td>3106</td></tr><tr><td>FB15k237-100 (Lee et al., 2023)</td><td>4659</td><td>134</td><td>62809</td><td>2624</td><td>77</td><td>6987</td><td>2329</td><td>2624</td><td>77</td><td>6987</td><td>2329</td></tr><tr><td>NELL995-25 (Lee et al., 2023)</td><td>4396</td><td>106</td><td>17578</td><td>2146</td><td>120</td><td>2230</td><td>743</td><td>2146</td><td>120</td><td>2230</td><td>744</td></tr><tr><td>NELL995-50 (Lee et al., 2023)</td><td>4396</td><td>106</td><td>17578</td><td>2335</td><td>119</td><td>2576</td><td>859</td><td>2335</td><td>119</td><td></td><td>859</td></tr><tr><td></td><td>2607</td><td>96</td><td>11058</td><td>1578</td><td>116</td><td>1818</td><td>606</td><td>1578</td><td>116</td><td>2576 1818</td><td>607</td></tr><tr><td>NELL995-75 (Lee et al., 2023)</td><td>1258</td><td>55</td><td>7832</td><td>1709</td><td>53</td><td>2378</td><td>793</td><td>1709</td><td>53</td><td>2378</td><td>793</td></tr><tr><td>NELL995-100 (Lee et al., 2023)</td><td>12659</td><td>47</td><td>41873</td><td>3228</td><td>74</td><td>3391</td><td>1130</td><td>3228</td><td>74</td><td></td><td>1131</td></tr><tr><td>Wikidata68K-25 (Lee et al., 2023)</td><td>12022</td><td>72</td><td>82481</td><td>9328</td><td>93</td><td>9672</td><td>3224</td><td>9328</td><td>93</td><td>3391</td><td>3225</td></tr><tr><td>Wikidata68K-50 (Lee et al., 2023)</td><td>6853</td><td>52</td><td>28741</td><td>2722</td><td>65</td><td>3430</td><td>1143</td><td>2722</td><td>65</td><td>9672 3430</td><td></td></tr><tr><td>Wikidata68K-75 (Lee et al., 2023)</td><td></td><td></td><td></td><td></td><td></td><td>13487</td><td>4496</td><td>12136</td><td></td><td></td><td>1144</td></tr><tr><td>Wikidata68K-100 (Lee et al., 2023)</td><td>9784</td><td>67</td><td>49875</td><td>12136</td><td>37</td><td></td><td></td><td></td><td>37</td><td>13487</td><td>4496</td></tr><tr><td>MTDEA1-tax (Zhou et al., 2023)</td><td>10000</td><td>10</td><td>17178</td><td>10000</td><td>10</td><td>17178</td><td>1908</td><td>10000</td><td>9</td><td>16526</td><td>1834</td></tr><tr><td>MTDEA1-health (Zhou et al., 2023)</td><td>10000</td><td>7</td><td>14371 23233</td><td>10000</td><td>7</td><td>14371 23233</td><td>1596</td><td>10000 10000</td><td>7</td><td>14110</td><td>1566</td></tr><tr><td>MTDEA2-org (Zhou et al., 2023)</td><td>10000 10000</td><td>10 16</td><td>16471</td><td>10000 10000</td><td>10 16</td><td>16471</td><td>2581 1830</td><td>10000</td><td>11</td><td>21976</td><td>2441</td></tr><tr><td>MTDEA2-sci (Zhou et al., 2023)</td><td>10000</td><td>45</td><td>27262</td><td>10000</td><td>45</td><td>27262</td><td>3026</td><td>10000</td><td>16 45</td><td>14852 28023</td><td>1650 3113</td></tr><tr><td>MTDEA3-art (Zhou et al., 2023) MTDEA3-infra (Zhou et al., 2023)</td><td>10000</td></table>

The complexity of the in-contex learning module with the sturctural relation memory can be divided into the self-attention matrix calculation $( \mathcal { O } ( L ^ { 2 } F ) )$ , the memory key-value calculation $( \mathcal { O } ( L K d ) )$ ), and the final aggregation $( { \mathcal { O } } ( L ( L + K ) F ) )$ , where F is the hidden dimensions of LLM. Because $L \gg K$ , the complexity upper bound of the in-contex learning module can be represented as $\mathcal { O } ( L ( L + K ) F )$

We further provide Table 4, which shows the TFLOPs in forward propagation, memory footprint, and wall-clock time of $\mathrm { S I R L M _ { P T } }$ and $\mathrm { S I R L M _ { S F T } }$ under the condition of batch size = 4 per GPU × 4 GPUs. For the SFT paradigm, we include results on the largest inductive dataset (FB15k237-25) and the smallest inductive dataset (FB15k237 v1) to provide a boundary of the computational cost.

## H DATASETS

In our experiments, we conduct evaluations on 36 datasets. According to the overlap level between train KG $\mathcal { G } _ { t r a i n } = ( \mathcal { E } _ { t r a i n } , \mathcal { R } _ { t r a i n } , \mathcal { T } _ { t r a i n } )$ and test KG $\mathcal { G } _ { t e s t } = ( \mathcal { E } _ { t e s t } , \mathcal { R } _ { t e s t } , \mathcal { T } _ { t e s t } )$ , these datasets can be divided into the following three categories:

• Transductive datasets that $\mathcal { E } _ { t e s t } \ : = \ : \mathcal { E } _ { t r a i n }$ and $\mathcal { R } _ { t e s t } \ : = \ : \mathcal { R } _ { t r a i n }$ : FB15k-237 (Toutanova & Chen, 2015), WN18RR (Dettmers et al., 2018), CoDEx-M (Safavi & Koutra, 2020), and NELL995 (Xiong et al., 2017).

• Inductive Entity (IndE) datasets that $\mathcal { E } _ { t e s t } \neq \mathcal { E } _ { t r a i n }$ and $\mathcal { R } _ { t e s t } = \mathcal { R } _ { t r a i n }$ , including 12 datasets from GraIL (Teru et al., 2020) (FB15k237 V1, FB15k237 V2, FB15k237 V3, FB15k237 V4, WN18RR V1, WN18RR V2, WN18RR V3, WN18RR V4, NELL995 V1, NELL995 V2, NELL995 V3, and NELL995 V4).

• Fully Inductive (FullInd) datasets that $\mathcal { E } _ { t e s t } \neq \mathcal { E } _ { t r a i n }$ and $\mathcal { R } _ { t e s t } \neq \mathcal { R } _ { t r a i n }$ , including 20 datasets from InGram (Lee et al., 2023) (FB15k237-25, FB15k237-50, FB15k237-75, FB15k237-100, NELL995-25, NELL995-50, NELL995-75, NELL995-100, Wikidata68K-25, Wikidata68K-50, Wikidata68K-75, and Wikidata68K-100) and MTDEA (Zhou et al., 2023) (MTDEA1-tax,

![](images/4cd09f18e4c93139b477593dc9bf189d236043d639728fd598e84dbea2513350.jpg)  
Figure 7: The length distribution of rules mined from various datasets.

MTDEA1-health, MTDEA2-org, MTDEA2-sci, MTDEA3-art, MTDEA3-infra, MTDEA4-sci, and MTDEA4-health).

These dataset are used to evaluate the model in the four training paradigms mentioned in Section 5.1.   
Table 5 provides detailed elemental statistics for these datasets.

Next, we discuss the rule mining approach in the data preprocessing stage. For each complete triplet in the training KG, we extract all closed paths within three hops. Due to the sparsity of the KG, some triplets may not have any closed paths of limited length that can be extracted. Figure 7 presents the proportion of triplets without rules (No rule) and the distribution of rules at different lengths across datasets. It can be observed that datasets such as WN18RR and the MTDEA1, MTDEA2, and MTDEA4 series exhibit a relatively high proportion of rule-less triplets. This is one of the reasons that the performance gains of SIRLM on these datasets, as shown in Tables 1, 8, and 9, are generally lower than on other datasets, i.e., the difficulty in obtaining comprehensive rule-based contextual information from the KG.

After extracting the closed paths, we handle triplets without any rules by directly using the query relation of the triplet as the rule body. For the remaining triplets that contain at least one closed path, we further construct candidate rule bodies by sequentially aggregating the directed relation sequences along each closed path. We then count the occurrence frequency of all candidate rule bodies for each triplet and select the most frequent one as the rule to be generated for that triplet, i.e., the masked component of the query instruction described in Appendix A.

Table 6: Hyperperameters of KRLM used in pre-training and end-to-end training from scratch.
<table><tr><td rowspan=1 colspan=1>Module</td><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Parameter</td></tr><tr><td rowspan=4 colspan=1>KG Tokenizer</td><td rowspan=1 colspan=1>NBFNet on Gr</td><td rowspan=1 colspan=1>Layer number N = 6Hidden dim d = 64Message function MsG(·) = DistMultAggregation function AGG(·) = SumUpdating function UP(·) = nn.Linear(128, 64)Motif Edge embeddings R* = nn.Embedding(8, 64)</td></tr><tr><td rowspan=1 colspan=1>NBFNet on G</td><td rowspan=1 colspan=1>Layer number N = 6Hidden dim d = 64Message function MsG(·) = DistMultAggregation function AGG(·) = SumUpdating function UP(·) = nn.Linear(128, 64)</td></tr><tr><td rowspan=1 colspan=1>Up-scale layer $\overline { { f _ { \mathrm { u p } } ( \cdot ) } }$ </td><td rowspan=1 colspan=1>nn.Linear(64, 1536)</td></tr><tr><td rowspan=1 colspan=1>Score function $s _ { \mathrm { K G } } ( \cdot )$ </td><td rowspan=1 colspan=1>nn.Linear(128, 64)nn.ReLU(·)nn.Linear(64, 1)</td></tr><tr><td rowspan=5 colspan=1>In-context Learning Module</td><td rowspan=1 colspan=1>Qwen2.5-1.5b backbone</td><td rowspan=1 colspan=1>Default configuration</td></tr><tr><td rowspan=1 colspan=1>LoRA configuration</td><td rowspan=1 colspan=1>r = 64α = 32droupout = 0.1target module = [gate_proj, up-proj, down_proj]</td></tr><tr><td rowspan=1 colspan=1>Memory key layer m K (·)</td><td rowspan=1 colspan=1>nn.Linear(64, 1536)</td></tr><tr><td rowspan=1 colspan=1>Memory value layer mv (·)</td><td rowspan=1 colspan=1>nn.Linear(64, 1536)</td></tr><tr><td rowspan=1 colspan=1>Scale of structural relation memory</td><td rowspan=1 colspan=1>K = 30</td></tr><tr><td rowspan=1 colspan=1>Next-relation Predictor</td><td rowspan=1 colspan=1>Porject layer fə(·)</td><td rowspan=1 colspan=1>nn.Linear(1600, 1600)nn.ReLU(·)nn.Linear(1600, 1)</td></tr><tr><td rowspan=4 colspan=1>RCMP Reasoner</td><td rowspan=1 colspan=1>NBFNet on $\mathcal { G } _ { r }$ </td><td rowspan=1 colspan=1>Layer number N = 6Hidden dim d = 64Message function MsG(·) = DistMultAggregation function AGG(·) = SumUpdating function UP(·) = nn.Linear(128, 64)Motif Edge embeddings R* = nn.Embedding(8, 64)</td></tr><tr><td rowspan=1 colspan=1>NBFNet on G</td><td rowspan=1 colspan=1>Layer number N = 6Hidden dim d = 64Message function MsG(·) = DistMultAggregation function AGG(·) = SumUpdating function UP(·) = nn.Linear(128, 64)</td></tr><tr><td rowspan=1 colspan=1>Down-scale layer $\overline { { f _ { \mathrm { d o w n } } ( \cdot ) } }$ </td><td rowspan=1 colspan=1>nn.Linear(1536, 64)</td></tr><tr><td rowspan=1 colspan=1>Score function $S _ { \mathrm { R C M P } } ( \cdot )$ </td><td rowspan=1 colspan=1>nn.Linear(128, 64)nn.ReLU(·)nn.Linear(64, 1)</td></tr><tr><td rowspan=2 colspan=1>Training</td><td rowspan=1 colspan=1>Optimizer</td><td rowspan=1 colspan=1>AdamW</td></tr><tr><td rowspan=1 colspan=1>Number of negative samples</td><td rowspan=1 colspan=1>512</td></tr></table>

## I EXPERIMENTAL HYPERPARAMETER SETTINGS

In Section 5.2, we report four training paradigms of SIRLM, including End-to-End (E2E) training from scratch, Pre-Training (PT), SFT, and GRPO post-training. The hyperparameters of the model architecture under the four paradigms are uniformly set to the values in Table 6. In the PT paradigm, we set the learning rate to 1e-4 and use the AdamW optimizer with a 1% warm-up step. The batch size per GPU is 12. Each epoch consists of 5000 iterations and the training process is conducted for a total of 20 epochs. More detailed training hyperparameters for the other training paradigms are provided in Table 7.

## J DETAILS EXPERIMENTAL ANALYSIS

## J.1 ADDITIONAL ANALYSIS OF MAIN EXPERIMENTS

Tables 8 and 9 correspond to the detailed experimental results of each method in Table 1 on the IndE and FullInd datasets, respectively.

Traditional embedding-based methods rely on initializing fixed representations of entities and relations tailored to a specific KGR scenario.

Table 7: Detailed Training parameters, where b, η, and M represent batch size, learning rate, and GRPO sampling number, respectively. “all” means that each epoch needs to iterate through all training queries.
<table><tr><td>Datasets</td><td>SIRLME2E (b, η, epoch, step)</td><td>SIRLMSFT (b, η, epoch, step)</td><td>SIRLMGRPO (b, η, M, epoch, step)</td></tr><tr><td>FB15k237</td><td>(12, 5e-4, 10, all)</td><td>(12, 1e-4, 3, all)</td><td>(8, 1e-5, 8, 1, all)</td></tr><tr><td>CoDEx-M</td><td>(24, 1e-4, 10, all)</td><td>(24, 5e-5, 3, all)</td><td>(8, 1e-5, 8, 1, all)</td></tr><tr><td>WN18RR</td><td>(12, 5e-4, 10, all)</td><td>(12, 1e-4, 3, all)</td><td>(8, 1e-5, 8, 3, all)</td></tr><tr><td>NELL995</td><td>(12, 5e-4, 10, all)</td><td>(12, 1e-4, 3, all)</td><td>(8, 1e-5, 8, 1, all)</td></tr><tr><td>FB15k237 v1</td><td>(24, 1e-4, 10, all)</td><td>(24, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>FB15k237 v2</td><td>(24, 1e-4, 10, all)</td><td>(24, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>FB15k237 v3</td><td>(24, 1e-4, 10, all)</td><td>(24, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>FB15k237 v4</td><td>(24, 1e-4, 5, all)</td><td>(24, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>WN18RR v1</td><td>(48, 1e-4, 10, all)</td><td>(48, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>WN18RR v2</td><td>(48, 1e-4, 10, all)</td><td>(48, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>WN18RR v3</td><td>(48, 1e-4, 20, all)</td><td>(48, 5e-5, 5, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>WN18RR v4</td><td>(48, 1e-4, 10, all)</td><td>(48, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>NELL995 v1</td><td>(32, 1e-4, 5, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>NELL995 v2</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>NELL995 v3 NELL995 v4</td><td>(32, 1e-4, 5, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>FB15k237-25</td><td>(32, 1e-4, 5, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>FB15k237-50</td><td>(32, 1e-4, 10, all) (32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(8, 1e-5, 8, 3, all)</td></tr><tr><td>FB15k237-75</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(8, 1e-5, 8, 3, all)</td></tr><tr><td>FB15k237-100</td><td></td><td>(32, 5e-5, 3, all)</td><td>(8, 1e-5, 8, 3, all)</td></tr><tr><td>NELL995-25</td><td>(32, 1e-4, 10, all) (32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(8, 1e-5, 8, 3, all)</td></tr><tr><td>NELL995-50</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(8, 1e-5, 8, 3, all)</td></tr><tr><td>NELL995-75</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(8, 1e-5, 8, 3, all)</td></tr><tr><td>NELL995-100</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(8, 1e-5, 8, 3, all)</td></tr><tr><td>Wikidata68K-25</td><td></td><td>(32, 5e-5, 3, all)</td><td>(8, 1e-5, 8, 3, all)</td></tr><tr><td>Wikidata68K-50</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>Wikidata68K-75</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td></td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>Wikidata68K-100</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>MTDEA1-tax</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>MTDEA1-health</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>MTDEA2-org</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>MTDEA2-sci</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>MTDEA3-art</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>MTDEA3-infra MTDEA4-sci</td><td>(32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all)</td></tr><tr><td>MTDEA4-health</td><td>(32, 1e-4, 10, all) (32, 1e-4, 10, all)</td><td>(32, 5e-5, 3, all) (32, 5e-5, 3, all)</td><td>(16, 1e-5, 8, 3, all) (16, 1e-5, 8, 3, all)</td></tr></table>

As a result, they cannot accommodate unseen entities and relations in the inference KG, making them unsuitable for inductive KGR tasks. Rule-based approaches, while capable of handling predictions involving unseen entities under a fixed set of relation types, fail to effectively generalize rule bodies when relations are dynamic, which prevents their application to FullInd KGR settings. In contrast, GNN-based methods are applicable across all KGR scenarios considered in our experiments. This advantage stems from their ability to learn structural context, enabling more structural pattern induction over the entire KG and

![](images/2c51982ccbb2f7c0ebbb6070924542e05f958e86afb4a7fae474ca3d606a9b61.jpg)  
Figure 8: The correlation trend between KG sparsity and the MRR gain of SIRLM.

allowing them to identify unfamiliar entities and relations through representations grounded in structural invariance.

LLM-based methods leverage pre-trained knowledge and strong contextual reasoning capabilities to uncover latent facts in incomplete KGs and to recognize unseen entities and relations. For instance, ChatRule utilizes the natural language understanding ability of LLMs to filter candidate textualized KG rules and select those that best support reasoning over a query triplet, while MKGL and KRLM project explicit KG structures into the implicit parametric knowledge of LLMs and ultimately generate potential facts through next-entity prediction. However, these approaches lack faithfulness constraints on the implicit reasoning process of LLMs, making them susceptible to biases arising from misalignment between structural knowledge and parametric representations, which can lead to erroneous inference outcomes.

In contrast, our SIRLM enforces the generation of structurally grounded rules over the KG, mitigating the tendency of LLMs to overlook the underlying structural evidence in favor of pre-trained knowledge when generating answers. Furthermore, we observe a correlation between the performance gains of SIRLM and the sparsity of the KG. Based on the results in Tables 8 and 9, Figure 8 illustrates this trend. Notably, SIRLM performs slightly worse on sparse KGs (e.g., the WN18RR series dataset) than on dense ones, which highlights an inherent limitation of structural rule-based reasoning under sparse conditions and points to an important direction for future research.

Table 8: The overall performance of various methods on IndE datasets. The colored cells represent the best , second-best , and third-best values, respectively.
<table><tr><td rowspan="2">Methods</td><td colspan="2">FB15k237 v1</td><td colspan="2">FB15k237 v2</td><td colspan="2">FB15k237 v3</td><td colspan="2">FB15k237 v4</td></tr><tr><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td></tr><tr><td>NeuralLP</td><td>0.325</td><td>0.468</td><td>0.389</td><td>0.586</td><td>0.400</td><td>0.571</td><td>0.396</td><td>0.593</td></tr><tr><td>DRUM</td><td>0.333</td><td>0.474</td><td>0.395</td><td>0.595</td><td>0.402</td><td>0.571</td><td>0.410</td><td>0.593</td></tr><tr><td>NBFNet</td><td>0.422</td><td>0.574</td><td>0.514</td><td>0.685</td><td>0.476</td><td>0.637</td><td>0.453</td><td>0.627</td></tr><tr><td>RED-GNN</td><td>0.369</td><td>0.483</td><td>0.469</td><td>0.629</td><td>0.445</td><td>0.603</td><td>0.442</td><td>0.621</td></tr><tr><td>ULTRA</td><td>0.509</td><td>0.670</td><td>0.524</td><td>0.710</td><td>0.504</td><td>0.663</td><td>0.496</td><td>0.684</td></tr><tr><td>MOTIF</td><td>0.530</td><td>0.702</td><td>0.557</td><td>0.744</td><td>0.519</td><td>0.684</td><td>0.508</td><td>0.695</td></tr><tr><td>ChatRule†</td><td>0.448</td><td>0.546</td><td>0.484</td><td>0.628</td><td>0.471</td><td>0.653</td><td>0.457</td><td>0.598</td></tr><tr><td>MKGL†</td><td>0.475</td><td>0.595</td><td>0.508</td><td>0.681</td><td>0.486</td><td>0.643</td><td>0.471</td><td>0.645</td></tr><tr><td>KRLM†</td><td>0.511</td><td>0.668</td><td>0.527</td><td>0.712</td><td>0.520</td><td>0.681</td><td>0.505</td><td>0.693</td></tr><tr><td> $\overline { { \mathbf { S I R L M _ { E 2 E } } } }$ </td><td>0.548</td><td>0.710</td><td>0.557</td><td>0.744</td><td>0.527</td><td>0.705</td><td>0.512</td><td>0.708</td></tr><tr><td>SIRLMPT</td><td>0.552</td><td>0.718</td><td>0.543</td><td>0.741</td><td>0.510</td><td>0.702</td><td>0.497</td><td>0.705</td></tr><tr><td>SIRLMSFT</td><td>0.558</td><td>0.718</td><td>0.555</td><td>0.747</td><td>0.529</td><td>0.708</td><td>0.510</td><td>0.702</td></tr><tr><td> $\mathbf { S I R L M _ { G R P O } }$ </td><td>0.561</td><td>0.721</td><td>0.559</td><td>0.752</td><td>0.532</td><td>0.711</td><td>0.515</td><td>0.714</td></tr><tr><td>Avg. gain*</td><td>+12.50%</td><td>+14.50%</td><td>+7.38%</td><td>+8.87%</td><td>+6.28%</td><td>+7.70%</td><td>+5.52%</td><td>+7.52%</td></tr><tr><td rowspan="3">Methods</td><td></td><td>WN18RR v1</td><td>WN18RR v2</td><td></td><td></td><td>WN18RR v3</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>WN18RR v4</td><td></td></tr><tr><td>MRR↑ 0.649</td><td>Hit10↑ 0.772</td><td>MRR↑ 0.635</td><td>Hit10↑ 0.749</td><td>MRR↑ 0.361</td><td>Hit10↑ 0.476</td><td>MRR↑</td><td>Hit10↑</td></tr><tr><td>NeuralLP DRUM</td><td>0.666</td><td>0.777</td><td>0.646</td><td>0.747</td><td>0.380</td><td>0.477</td><td>0.628 0.627</td><td>0.706 0.702</td></tr><tr><td>NBFNet</td><td>0.741</td><td>0.826</td><td>0.704</td><td>0.798</td><td>0.432</td><td>0.568</td><td>0.641</td><td>0.694</td></tr><tr><td>RED-GNN</td><td>0.701</td><td>0.799</td><td>0.690</td><td>0.780</td><td>0.427</td><td>0.524</td><td>0.651</td><td>0.721</td></tr><tr><td>ULTRA</td><td>0.685</td><td>0.793</td><td>0.679</td><td>0.779</td><td>0.411</td><td>0.546</td><td>0.614</td><td>0.720</td></tr><tr><td>MOTIF</td><td>0.703</td><td>0.806</td><td>0.680</td><td>0.781</td><td>0.466</td><td>0.590</td><td>0.659</td><td>0.733</td></tr><tr><td>ChatRule†</td><td>0.621</td><td>0.694</td><td>0.650</td><td>0.686</td><td>0.395</td><td>0.497</td><td>0.568</td><td>0.649</td></tr><tr><td>MKGL†</td><td>0.746</td><td>0.822</td><td>0.712</td><td>0.799</td><td>0.456</td><td>0.559</td><td>0.664</td><td>0.741</td></tr><tr><td>KRLM†</td><td>0.705</td><td>0.797</td><td>0.690</td><td>0.791</td><td>0.457</td><td>0.590</td><td>0.662</td><td>0.737</td></tr><tr><td>SIRLME2E</td><td>0.705</td><td>0.766</td><td>0.699</td><td>0.789</td><td>0.456</td><td>0.562</td><td>0.658</td><td>0.732</td></tr><tr><td>SIRLMPT</td><td>0.668</td><td>0.744</td><td>0.670</td><td>0.766</td><td>0.433</td><td>0.549</td><td>0.627</td><td>0.693</td></tr><tr><td>SIRLMSFT</td><td>0.706</td><td>0.778</td><td>0.714</td><td>0.799</td><td>0.459</td><td>0.568</td><td>0.662</td><td>0.745</td></tr><tr><td>SIRLMGRPO</td><td>0.695</td><td>0.762</td><td>0.708</td><td>0.791</td><td>0.454</td><td>0.566</td><td>0.662</td><td>0.741</td></tr><tr><td>Avg. gain</td><td>+0.38%</td><td>+0.21%</td><td>+3.78%</td><td>+3.12%</td><td>+3.84%</td><td>+3.17%</td><td>+2.71%</td><td>+3.36%</td></tr><tr><td></td><td colspan="2">NELL995 v1</td><td colspan="2">NELL995 v2</td><td colspan="2">NELL995 v3</td><td colspan="2"></td></tr><tr><td rowspan="3">Methods NeuralLP</td><td></td><td>Hit10↑</td><td></td><td></td><td></td><td></td><td>NELL995 v4</td><td></td></tr><tr><td>MRR↑</td><td></td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td></tr><tr><td>0.610 0.628</td><td>0.871 0.873</td><td>0.361 0.365</td><td>0.564 0.540</td><td>0.367 0.375</td><td>0.576 0.577</td><td>0.261 0.270</td><td>0.539</td></tr><tr><td>DRUM NBFNet</td><td>0.648</td><td>0.862</td><td>0.421</td><td>0.599</td><td>0.462</td><td>0.578</td><td>0.404</td><td>0.531 0.588</td></tr><tr><td>RED-GNN</td><td>0.637</td><td>0.866</td><td>0.419</td><td>0.601</td><td>0.436</td><td>0.594</td><td>0.363</td><td>0.556</td></tr><tr><td>ULTRA</td><td>0.757</td><td>0.878</td><td>0.575</td><td>0.761</td><td>0.563</td><td>0.755</td><td>0.469</td><td>0.733</td></tr><tr><td>MOTIF</td><td>0.712</td><td>0.873</td><td>0.566</td><td>0.765</td><td>0.580</td><td>0.764</td><td>0.507</td><td>0.740</td></tr><tr><td>ChatRule†</td><td>0.719</td><td>0.806</td><td>0.529</td><td>0.690</td><td>0.552</td><td>0.732</td><td>0.528</td><td>0.741</td></tr><tr><td>MKGL†</td><td>0.749</td><td>0.886</td><td>0.570</td><td>0.767</td><td>0.571</td><td>0.759</td><td>0.525</td><td>0.749</td></tr><tr><td>KRLM†</td><td>0.679</td><td>0.898</td><td>0.568</td><td>0.761</td><td>0.561</td><td>0.755</td><td>0.554</td><td>0.762</td></tr><tr><td>SIRLME2E</td><td>0.680</td><td>0.871</td><td>0.580</td><td>0.769</td><td>0.585</td><td>0.769</td></table>

† We reproduce the experimental metrics of ChatRule, MKGL, and KRLM.  
⋆ We calculate the average gain of the optimal value in the four training modes of SIRLM compared to all baselines.

Table 9: The overall performance of various methods on FullInd datasets. The colored cells represent the best , second-best , and third-best values, respectively.
<table><tr><td rowspan="2">Methods</td><td colspan="2">FB15k237-25</td><td colspan="2">FB15k237-50</td><td colspan="2">FB15k237-75</td><td colspan="2">FB15k237-100</td></tr><tr><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td></tr><tr><td>NBFNet</td><td>0.224</td><td>0.410</td><td>0.130</td><td>0.259</td><td>0.089</td><td>0.166</td><td>0.072</td><td>0.154</td></tr><tr><td>RED-GNN</td><td>0.145</td><td>0.284</td><td>0.129</td><td>0.251</td><td>0.107</td><td>0.201</td><td>0.121</td><td>0.263</td></tr><tr><td>ULTRA</td><td>0.383</td><td>0.635</td><td>0.334</td><td>0.538</td><td>0.400</td><td>0.598</td><td>0.444</td><td>0.643</td></tr><tr><td>MOTIF</td><td>0.388</td><td>0.635</td><td>0.340</td><td>0.544</td><td>0.399</td><td>0.607</td><td>0.439</td><td>0.642</td></tr><tr><td>ChatRule†</td><td>0.336</td><td>0.602</td><td>0.289</td><td>0.514</td><td>0.250</td><td>0.507</td><td>0.361</td><td>0.554</td></tr><tr><td>KRLM†</td><td>0.398</td><td>0.640</td><td>0.345</td><td>0.552</td><td>0.414</td><td>0.620</td><td>0.455</td><td>0.655</td></tr><tr><td>SIRLME2E</td><td>0.386</td><td>0.635</td><td>0.345</td><td>0.540</td><td>0.411</td><td>0.623</td><td>0.444</td><td>0.646</td></tr><tr><td>SIRLMPT</td><td>0.376</td><td>0.631</td><td>0.315</td><td>0.534</td><td>0.396</td><td>0.613</td><td>0.442</td><td>0.655</td></tr><tr><td>SIRLMSFT</td><td>0.388</td><td>0.647</td><td>0.345</td><td>0.541</td><td>0.400</td><td>0.610</td><td>0.448</td><td>0.648</td></tr><tr><td>SIRLMGRPO</td><td>0.396</td><td>0.649</td><td>0.347</td><td>0.546</td><td>0.420</td><td>0.626</td><td>0.449</td><td>0.656</td></tr><tr><td>Avg. gain</td><td>+8.37%</td><td>+11.47%</td><td>+8.58%</td><td>+10.30%</td><td>+14.35%</td><td>+17.62%</td><td>+13.37%</td><td>+17.08%</td></tr><tr><td></td><td>NELL995-25</td><td></td><td>NELL995-50</td><td></td><td>NELL995-75</td><td></td><td>NELL995-100</td><td></td></tr><tr><td>Methods</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td></td><td>Hit10↑</td></tr><tr><td>NBFNet</td><td>0.283</td><td>0.417</td><td>0.225</td><td>0.346</td><td>0.137</td><td>0.255</td><td>MRR↑</td><td>0.199</td></tr><tr><td>RED-GNN</td><td>0.214</td><td>0.266</td><td>0.179</td><td>0.115</td><td>0.203</td><td>0.353</td><td>0.096</td><td>0.385</td></tr><tr><td>ULTRA</td><td>0.407</td><td>0.596</td><td>0.418</td><td>0.595</td><td>0.374</td><td>0.570</td><td>0.212 0.458</td><td>0.684</td></tr><tr><td>MOTIF</td><td>0.390</td><td>0.580</td><td>0.414</td><td>0.573</td><td>0.360</td><td>0.548</td><td>0.464</td><td>0.682</td></tr><tr><td>ChatRule†</td><td>0.359</td><td>0.564</td><td>0.368</td><td>0.540</td><td>0.302</td><td>0.521</td><td>0.444</td><td>0.608</td></tr><tr><td>KRLM†</td><td>0.401</td><td>0.596</td><td>0.432</td><td>0.598</td><td>0.367</td><td>0.559</td><td>0.489</td><td>0.688</td></tr><tr><td>SIRLME2E</td><td>0.407</td><td>0.601</td><td>0.413</td><td>0.595</td><td>0.371</td><td>0.570</td><td>0.477</td><td>0.666</td></tr><tr><td>SIRLMPT</td><td>0.394</td><td>0.593</td><td>0.394</td><td>0.581</td><td>0.355</td><td>0.550</td><td></td><td>0.668</td></tr><tr><td>SIRLMSFT</td><td>0.412</td><td>0.613</td><td>0.418</td><td>0.598</td><td>0.379</td><td>0.578</td><td>0.472 0.489</td><td>0.667</td></tr><tr><td>SIRLMGRPO</td><td>0.410</td><td>0.607</td><td>0.382</td><td>0.567</td><td>0.368</td><td>0.560</td><td>0.477</td><td>0.668</td></tr><tr><td>Avg. gain</td><td>+6.97%</td><td>+10.98%</td><td>+7.87%</td><td>+13.68%</td><td>+8.85%</td><td>+11.03%</td><td></td><td>+12.70%</td></tr><tr><td rowspan="3"></td><td>Wikidata68K-25</td><td></td><td>Wikidata68K-50</td><td></td><td></td><td></td><td>+12.85%</td><td></td></tr><tr><td>Methods</td><td></td><td></td><td></td><td></td><td>Wikidata68K-75</td><td></td><td>Wikidata68K-100</td></tr><tr><td>MRR↑</td><td>Hit10↑ 0.301</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td></tr><tr><td>NBFNet RED-GNN</td><td>0.154 0.170</td><td>0.263</td><td>0.062 0.058</td><td>0.105 0.093</td><td>0.072 0.172</td><td>0.172 0.290</td><td>0.014</td><td>0.026 0.136</td></tr><tr><td></td><td>0.321</td><td>0.535</td><td>0.140</td><td>0.280</td><td>0.380</td><td>0.530</td><td>0.096 0.168</td><td>0.286</td></tr><tr><td>ULTRA MOTIF</td><td>0.317</td><td>0.505</td><td>0.160</td><td>0.304</td><td>0.371</td><td>0.535</td><td>0.173</td><td>0.284</td></tr><tr><td></td><td>0.256</td><td>0.372</td><td>0.090</td><td>0.126</td><td>0.266</td><td>0.483</td><td>0.107</td><td>0.196</td></tr><tr><td>ChatRule†</td><td>0.332</td><td>0.550</td><td>0.168</td><td>0.328</td><td>0.384</td><td>0.538</td><td></td><td>0.313</td></tr><tr><td>KRLM†</td><td>0.322</td><td>0.536</td><td>0.173</td><td>0.329</td><td>0.406</td><td>0.568</td><td>0.189</td><td></td></tr><tr><td>SIRLME2E</td><td>0.303</td><td>0.497</td><td>0.166</td><td>0.309</td><td>0.399</td><td>0.556</td><td>0.186</td><td>0.304 0.300</td></tr><tr><td>SIRLMPT</td><td>0.328</td><td>0.541</td><td>0.171</td><td>0.330</td><td>0.412</td><td>0.566</td><td>0.189</td><td>0.316</td></tr><tr><td>SIRLMSFT</td><td>0.314</td><td>0.511</td><td>0.170</td><td>0.315</td><td>0.385</td><td>0.550</td><td>0.191 0.185</td><td>0.307</td></tr><tr><td>SIRLMGRPO Avg. gain</td><td colspan="2">+6.97% +12.00%</td><td colspan="2">+5.80% +12.40%</td><td colspan="2">+13.78% +14.33%</td><td colspan="2">+6.65% +10.92%</td></tr><tr><td></td><td>MTDEA1-tax</td><td></td><td>MTDEA1-health</td><td></td><td>MTDEA2-org</td><td></td><td>MTDEA2-sci</td><td></td></tr><tr><td>Methods</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td><td>MRR↑</td><td>Hit10↑</td></tr><tr><td>NBFNet</td><td>0.113</td><td>0.315</td><td>0.122</td><td>0.339</td><td>0.109</td><td>0.292</td><td>0.091</td><td>0.235</td></tr><tr><td>ULTRA</td><td>0.330</td><td>0.459</td><td>0.380</td><td>0.467</td><td>0.104</td><td>0.170</td><td>0.311</td><td>0.451 0.520</td></tr><tr><td>MOTIF ChatRule</td><td>0.416 0.300</td><td>0.522 0.388</td><td>0.385 0.352</td><td>0.473 0.427</td><td>0.106 0.101</td><td>0.170 0.140</td><td>0.326 0.343</td></table>

† We reproduce the experimental metrics of ChatRule and KRLM.  
⋆ We calculate the average gain of the optimal value in the four training modes of SIRLM compared to all baselines.

Table 10: The overall performance of ablation variants on different datasets, where the MRR and Hit10 in the IndE and FullInd scenarios are summarized as average values. The colored cells represent the best , second-best , and third-best values, respectively.
<table><tr><td rowspan="2">Methods</td><td colspan="2">FB15k237</td><td colspan="2">CodeX-M</td><td colspan="2">WN18RR</td><td colspan="2">NELL995</td><td colspan="2">12 IndE Datasets</td><td colspan="2">20 FullInd Datasets</td></tr><tr><td>MRR</td><td>Hit10</td><td>MRR</td><td>Hit10</td><td>MRR</td><td>Hit10</td><td>MRR</td><td>Hit10</td><td>MRR</td><td>Hit10</td><td>MRR</td><td>Hit10</td></tr><tr><td>Full Model</td><td>0.408</td><td>0.593</td><td>0.367</td><td>0.522</td><td>0.461</td><td>0.556</td><td>0.520</td><td>0.638</td><td>0.580</td><td>0.738</td><td>0.374</td><td>0.538</td></tr><tr><td>w/o MMQI</td><td>0.389</td><td>0.579</td><td>0.360</td><td>0.499</td><td>0.458</td><td>0.525</td><td>0.514</td><td>0.627</td><td>0.570</td><td>0.721</td><td>0.364</td><td>0.522</td></tr><tr><td>w/o SRM</td><td>0.358</td><td>0.566</td><td>0.348</td><td>0.478</td><td>0.455</td><td>0.526</td><td>0.488</td><td>0.606</td><td>0.558</td><td>0.709</td><td>0.348</td><td>0.508</td></tr><tr><td>w/o RCMP</td><td>0.352</td><td>0.538</td><td>0.339</td><td>0.422</td><td>0.404</td><td>0.500</td><td>0.501</td><td>0.611</td><td>0.540</td><td>0.685</td><td>0.344</td><td>0.502</td></tr></table>

![](images/aabaece7193a227d9ace4ee1872666b40d10a273ab9d617507190a9bc3464f41.jpg)  
Training step

![](images/d0d3ad611ed5bec97ae02a557aaac3fd4dde019d9aa935e7b8d503a7bb904bef.jpg)  
Training step

![](images/545f2f5803a410de1e7e2248e8a4f2f1dddef158136f6066bcd8daaa37b4df14.jpg)  
Training step

![](images/e81158f77066bb77463e51e2292a0d56837d5680ee969c334aafa731b40a7e64.jpg)  
Figure 9: Convergence of rule generation accuracy for SIRLM with different LLMs on WN18RR v1.

## J.2 DETAILS ABLATION ANALYSIS

Section 5.3 analyzes the ablation components of SIRLM. To verify the universality of each component, we chose to perform ablation variants in SIRLM<sub>PT</sub> training mode. We first provide the design details of each ablation variant.

• -MMQI. This variant only provides the textual instructions of query triplets, without including structural representations of entities and relations. Following MKGL (Guo et al., 2024), such instruction sequences contain textual strings of entities/relations along with their corresponding textual explanations. During tokenization, we use the principal neighborhood aggregation method (Corso et al., 2020) to aggregate the text token sequences of entity/relation into word-level token embeddings, which are then used for subsequent rule generation and reasoning. Since the instructions do not include structural representations, the tokenizer in Eq. (6) degenerates into $\mathrm { T K N } _ { \mathrm { L L M } }$ . The rest of the SIRLM architecture remains unchanged.

• -SRM. This variant removes the structural relation memory in the in-context learning module, along with its corresponding linear layers $( m _ { K } ( \cdot )$ and $m _ { V } ( \cdot ) )$ in Eq. (8). All other components remain consistent with the primary SIRLM.

• -RCMP. This variant removes $\{ < r _ { z } , f _ { \mathrm { d o w n } } ( \pmb { h } _ { L + z } ) > \} _ { z = 1 } ^ { \varepsilon }$ in Eq. (11), reverting it back to the original SIL module.

Table 10 provides the details ablation experimental results of each ablation variant.

## J.3 ADDITIONAL ANALYSIS ON DIFFERENT LLM BACKBONES

Section 5.5 presents the experimental results of SIRLM across four different LLM backbones. Overall, the results show a positive correlation between the parameter scale of the backbone and its reasoning performance. We attribute this to the fact that larger parameter sizes enable the model to more quickly fit the newly injected structured representation space.

To further illustrate this, Figure 9 reports the convergence of rule generation accuracy for the four LLM backbones on WN18RR v1. It is evident that the two 7B backbones (Qwen2.5-7b and Llama2- 7b) approach convergence at around 200 steps, whereas Qwen2.5-0.5b and Qwen2.5-1.5b backbones require approximately 400 steps to reach a similar level of convergence.

## J.4 CASE STUDY AND ERROR ANALYSIS

To better understand the behavior of our SIRLM, we visualize several representative cases of rule generation in Figure 10. Cases 1 and 2 is a successful example where the generated rule exactly matches the ground-truth rule, indicating that the model can accurately identify the correct multi-hop reasoning pattern. As a result, the correct entities “Best Academy Picture Award” and “University of Texas at Austin” appears in the top-ranked candidates with a relatively high confidence score.

Cases 3 and 4 demonstrates that an exact match to the ground-truth rule is not necessary for correct prediction. Although the generated rule deviates from the annotated reasoning path, it still leads to the correct answers “Tokyo” and “Columbia Records”. This suggests that the model can exploit alternative reasoning paths with similar structural semantics. In these case, the generated atomic relations (blue square) differ from the ground-truth atomic relations (orange triangle), but remain semantically similar in the representation space.

<table><tr><td></td><td>Visualization of Relation Representation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>6 4 2</td><td>Query relation Ground truth rule body Generated rule body</td><td></td><td>A</td><td></td><td></td><td></td><td>Case 1: &lt;Howard Hughes, award_nominations, Best Academy Picture Award&gt;</td><td>Top-5 candidates:</td><td colspan="2"></td></tr><tr><td>0</td><td></td><td>7</td><td></td><td></td><td></td><td>Ground truth rule body:</td><td>award_nominations=nationality→inv_country→inv_nominated_for</td><td>Best Academy Actress Award: 0.946 Best Academy Picture Award: 0.860</td><td colspan="2"></td></tr><tr><td>-2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>BAFTA Best Film Award : 0.546 Best Academy Director Award: 0.524</td><td colspan="2"></td></tr><tr><td></td><td></td><td>0</td><td></td><td></td><td></td><td>Generated rule body:</td><td>award_nominations=nationality→inv_country→inv_nominated_for</td><td>Golden Raspberry Award: 0.358</td><td colspan="2"></td></tr><tr><td></td><td></td><td>Visualization of Relation Representation</td><td>2</td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td>7</td><td>Query relation Ground truth rule body Generated rule body</td><td></td><td></td><td></td><td></td><td>Ground truth rule body:</td><td>Case 2: &lt;Mitch Pileggi, gdp_nominal, University of Texas at Austin&gt;</td><td>Top-5 candidates:</td><td colspan="2"></td></tr><tr><td>0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>gdp_nominal=nationality→inv_students_graduates</td><td>University of Texas at Austin: 0.822 Princeton University: 0.801</td><td colspan="2">Cornell University: 0.696</td></tr><tr><td>-1 D -2</td><td>-3</td><td>5</td><td></td><td></td><td></td><td>Generated rule body:</td><td>gdp_nominal=nationality→inv_students_graduates</td><td>New York University: 0.683 Yale University : 0.643</td><td></td><td></td></tr><tr><td></td><td>Visualization of Relation Representation</td><td>-2</td><td>-1</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Top-5 candidates:</td></tr><tr><td>4 2</td><td colspan="6"></td><td colspan="4">Case 3: &lt;Bandai Namco Holdings, headquarters_state_province, Tokyo&gt;</td></tr><tr><td>0</td><td>Query relation Ground truth rule body Generated rule body</td><td></td><td></td><td></td><td></td><td>Ground truth rule body:</td><td colspan="2">headquarters_state_province=webpage_category→inv_webpage_category→headquarters_citytown</td><td>Tokyo: 0.991 California: 0.952</td><td>Minato: 0.851</td></tr><tr><td>-2</td><td></td><td></td><td></td><td></td><td></td><td>Generated rule body:</td><td colspan="2">headquarters_state_province=webpage_category→inv_webpage_category→inv_contains</td><td></td><td>Santa Clara: 0.736 Sapporo: 0.519</td></tr><tr><td></td><td>-6</td><td>-4</td><td>-2</td><td></td><td>2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2.5 0.0</td><td></td><td></td><td>★</td><td>Query relation</td><td>Visualization of Relation Representation Ground truth rule body</td><td></td><td>Case 4: &lt;Nick Jonas, inv_taxonomy, Columbia Records&gt;</td><td>Top-5 candidates:</td><td></td><td></td></tr><tr><td>-2.5</td><td></td><td></td><td></td><td>Generated rule body</td><td></td><td></td><td>Ground truth rule body:</td><td>Columbia Records: 0.961</td><td colspan="2"></td></tr><tr><td>-5.0 -7.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td>inv_taxonomy=profession→inv_profession</td><td>Avex Trax: 0.869 Island Records: 0.628</td><td colspan="2"></td></tr><tr><td>-10.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Generated rule body: inv_taxonomy=inv_artists→artists</td><td>RCA Records: 0.549 Atlantic Records: 0.517</td><td colspan="2"></td></tr><tr><td>-12.5</td><td>0</td><td></td><td>10 15</td><td></td><td>20</td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td><img src="images/2bad616a71e9a4fe423154f3e42f5f4f1a10dca2cafb8b8148c98a7e3d3622c3.jpg"/></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td><img src="images/949264f4d055f3f642280c1d7aedce5e0e5fd006ba0ff85b1b7fc5c2b3fb0d19.jpg"/></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td><td colspan="2"></td></tr></table>

Figure 10: Case studies of rule generation. Red, blue, and orange denote the query relation, the atomic relations in the ground-truth rule, and the atomic relations in the predicted rule, respectively. Green denotes the target entities that need to be predicted.

In contrast, Cases 5 and 6 reveals a limitation of the method. The generated rule introduces irrelevant atomic relations that differ substantially from the ground truth, causing the reasoning process to deviate from the correct semantic direction and resulting in incorrect top-ranked candidates.

These observations indicate that the effectiveness of SIRLM critically depends on the quality of the generated rule body, that is, semantically accurate rules improve both interpretability and predictive performance, whereas semantically inconsistent rules can significantly degrade performance.

## K LIMITATIONS AND FUTURE WORK

SIRLM provides a novel modeling and training framework for LLM-based KGR research, effectively alleviating the problem of reasoning evidence perception drift caused by the knowledge representation gap between LLMs and KGs. However, SIRLM still has several potential limitations. We discuss these limitations as follows and provide potential future research directions.

Computational complexity. As shown in Appendix G, SIRLM needs to make real-time relational graph updates and message passing over the entire KG during training. Although sparse operators are adopted in practice to significantly reduce the computational cost of graph processing, the upper bound remains quadratic in the number of entities, which leads to a notable computational bottleneck for large-scale KGs. Therefore, in future work, we plan to introduce a query-driven evidence graph extraction mechanism (Huang et al., 2026), such that the graph structure processed by SIRLM is closely aligned with the query triplets. This would help eliminate the unnecessary computational overhead incurred by processing irrelevant KG context.

Sparse KG reasoning. The comprehensive analysis of Figures 7 and 8, as well as Tables 8 and 9, shows that there is still significant room for improvement in the reasoning performance of SIRLM on sparse KGs. We consider that this limitation is attributed to the fact that SIRLM is a rule-driven KGR framework. In sparse KGs, the diversity and completeness of structural rules are inherently limited, which in turn hinders the subsequent processes of rule learning and generation by LLMs. To address this issue, we plan to incorporate a soft rule mechanism (Qu et al., 2021) into the rule mining stage. Specifically, instead of relying solely on hard inductive methods based on observable closed-path patterns, we will jointly employ soft rule mining to construct approximate rules and corresponding soft labels for query triplets that lack closed-path evidence. These soft signals can serve as additional supervision for subsequent LLM-based rule generation. In this way, the contextual exploration space of LLMs over sparse KGs can be effectively expanded, thereby alleviating the limitations in rule learning caused by insufficient structural context.