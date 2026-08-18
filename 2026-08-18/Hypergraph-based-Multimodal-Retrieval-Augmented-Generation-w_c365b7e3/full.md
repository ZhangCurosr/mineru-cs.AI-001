# Hypergraph-based Multimodal Retrieval-Augmented Generation with Incremental Refinement

Shenao Chen<sup>∗</sup>   
241080005@hdu.edu.cn   
Hangzhou Dianzi University   
Hangzhou, China

Rundong Xue xuerundong2002@gmail.com Xi’an Jiaotong University Xi’an, China

Chenggang Yan   
cgyan@hdu.edu.cn   
Hangzhou Dianzi University   
Hangzhou, China   
Yidan Xu<sup>∗</sup>   
yidanxu2024@163.com   
Hangzhou Dianzi University   
Hangzhou, China   
Duanpo Wu   
wuduanpo@hdu.edu.cn   
Hangzhou Dianzi University   
Hangzhou, China

Xiangmin Han hanxiangmin@bjut.edu.cn Beijing University of Technology Beijing, China

Yue Gao   
gaoyue@tsinghua.edu.cn   
Tsinghua University   
Beijing, China   
Yuhan Gao<sup>✉</sup>   
yuhangao@hdu.edu.cn   
Hangzhou Dianzi University   
Hangzhou, China

## Abstract

Modern Multimodal Retrieval-Augmented Generation (M-RAG) systems are fundamentally limited by the binary connectivity paradigm of traditional simple graphs, which fails to capture the intricate, high-order correlations among heterogeneous entities—such as the N-ary relationships between a visual chart, its scattered textual descriptions, and underlying numerical data. Furthermore, existing refinement strategies often rely on exhaustive, full-page reconstruction to align cross-modal information, leading to prohibitive computational redundancy and the introduction of contextual noise in long-form document processing. In this paper, we propose Hyper-M2RAG, a novel framework that redefines multimodal document retrieval through High-order Hypergraph Representation Learning. We first formalize the document structure as a Multimodal Hypergraph, utilizing hyperedges as unified seman tic containers to encapsulate multi-way associations across text, images, and tables, thereby transcending point-to-point modeling. To mitigate semantic fragmentation caused by physical pagination, we introduce an Anchor-driven Incremental Refinement mechanism. Rather than performing a global sweep, our approach surgi cally identifies boundary-crossing anchors—nodes and reconstructs their local hyper-topology using one-hop neighborhood contexts. This targeted refinement efectively bridges cross-page knowledge gaps with minimal computational footprints. Extensive evaluations on multimodal benchmarking datasets demonstrate that Hyper-M2RAG significantly outperforms state-of-the-art methods in both retrieval precision and generation coherence. Our code is accessible at https://github.com/ShenAoChen2001/MMHRAG.

CCS Concepts

• Information systems → Document representation; Multimedia information systems; • Computing methodologies → Knowledge representation and reasoning; Natural language generation.

Keywords

Hypergraph, Multimodal RAG, Document Understanding, Crosspage Association, Incremental Refinement

ACM Reference Format: Shenao Chen, Yidan Xu, Xiangmin Han, Rundong Xue, Duanpo Wu, Yuhan Gao, Chenggang Yan, and Yue Gao. 2026. Hypergraph-based Multimodal Retrieval-Augmented Generation with Incremental Refinement. In Proceedings of the 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3767308.3835434

## 1 Introduction

Document Question Answering (Doc-QA) [19, 22] aims to generate accurate answers grounded in document content. With the rapid progress of Large Vision-Language Models (LVLMs), Doc-QA has evolved from text-only understanding to multimodal reasoning over long and structurally complex documents. In real-world scenarios such as academic papers, technical reports, and business documents, critical evidence is often distributed across text, figures, tables, formulas, and page layouts, and may span many pages. This setting requires not only long-context modeling, but also faithful preservation of cross-modal structure and document-level semantic dependencies.

Retrieval-augmented generation (RAG) has become an efective paradigm for document understanding by retrieving relevant evidence before answer generation. Existing RAG frameworks, however, are still largely rooted in text-centric assumptions [9, 23]. Even recent multimodal variants often remain limited in how they represent document structure. A common strategy is to linearize visual content into captions or OCR-derived text, and then perform retrieval over textual surrogates [4, 26]. Although practical, this design inevitably weakens the original spatial organization of the document and discards part of the visual semantics embedded in figures, tables, and layouts. As a result, evidence that depends on visual context, layout proximity, or joint interpretation across modalities can be overlooked during retrieval.

Beyond this general limitation, a more fundamental issue lies in the relational assumption inherited by many graph-based or knowledge-enhanced RAG systems. In text-centric settings, knowledge is commonly modeled through binary relations, typically represented as triples of the form (head, relation, tail). This abstraction is often adequate when semantic dependencies can be decomposed into pairwise links between entities or text spans. However, multi modal documents exhibit a diferent structural nature. Key evidence is frequently formed not by a single pairwise connection, but by the joint interaction among multiple heterogeneous elements, such as a figure region, its caption, a nearby paragraph, a table cell, and their shared layout context. In such cases, the meaning emerges from a group-wise association rather than from isolated binary edges.

This mismatch makes conventional graph modeling inherently limited for multimodal RAG. When high-order multimodal dependencies are decomposed into a set of pairwise edges, the original evidence unit becomes fragmented. The system may preserve local connections, yet lose the higher-level semantic coherence that binds multiple elements into a single interpretable structure. For example, a textual claim may only be fully supported when a chart, its legend, and the surrounding explanatory paragraph are considered together; reducing this evidence to independent pairwise links weakens structural fidelity and makes downstream reasoning less reliable. Therefore, the challenge in multimodal RAG is not merely to add more modalities into existing retrieval pipelines, but to redesign the underlying relational structure from graph-based pairwise modeling to hypergraph-based high-order organization.

Recent advances in document parsing pipelines [13], such as MinerU2.5 [17], DeepSeekOCR [21], and PaddleOCR [2], make this shift increasingly practical. These systems can extract high-fidelity multimodal elements from complex PDFs, including text blocks, figure regions, table structures, and layout coordinates, thereby providing a stronger foundation for structured multimodal indexing. This progress opens the door to a retrieval framework that operates directly on rich document elements and their structural relations, instead of relying on lossy textual abstractions alone.

Motivated by this gap, we ask the following question: how can we extend RAG from graph-based pairwise retrieval to hypergraph-based high-order multimodal reasoning, while keeping refinement eficient for long documents?

To answer this question, we identify two core challenges. Challenge 1: Pairwise graph modeling is insuficient for highorder multimodal evidence. Existing graph-based retrieval structures are designed for binary relations and therefore struggle to preserve group-wise dependencies among text, figures, tables, and layout context. This limitation leads to fragmented evidence representation and weakens the retriever’s ability to discover visually grounded or structurally coupled information. Challenge 2: Efficient refinement in long multimodal documents remains dificult. In long documents, relevant evidence and entity relations often span many pages. Existing refinement strategies frequently revisit entire pages or large contexts to recover missing cross-page connections [10], which introduces substantial computational redundancy, especially when multimodal tokens dominate the cost.

We propose Hyper-M2RAG, a hypergraph-based framework for multimodal retrieval-augmented generation. The central idea is to move the structural foundation of RAG from graphs to hypergraphs. Hyper-M2RAG constructs a multimodal hypergraph in which vertices represent textual, visual, and tabular entities, while hyperedges connect multiple heterogeneous elements that jointly express a semantic unit. This design allows the retriever to preserve and exploit high-order multimodal dependencies in a unified topological space, rather than approximating them through decomposed pairwise links. As a result, retrieval becomes natively aware of textual semantics, visual evidence, and layout structure.

To further support long-document reasoning, Hyper-M2RAG introduces an anchor-driven surgical refinement mechanism. Our key observation is that entities recurring across pages often signal incomplete or ambiguous structural associations. Instead of reprocessing entire pages, the system identifies these boundary-crossing entities as anchors and selectively reconstructs their local one-hop neighborhoods. By refining only the hyperedges around these anchors and merging the updates back into the global hypergraph, our method concentrates computation on the structurally uncertain regions that matter most. This makes refinement substantially more eficient while preserving document-level reasoning capacity.

Our contributions are summarized as follows:

• From graph to hypergraph for multimodal RAG. We show that the pairwise relational assumption underlying conventional graph-based RAG is insuficient for multimodal documents, and introduce a hypergraph formulation that naturally models highorder dependencies among text, figures, tables, and layout elements.

• Native multimodal hypergraph indexing. We leverage visual and layout information during document parsing to extract structured entities and relationships. These extracted textual representations are then organized into a unified hypergraph, supporting retrieval over semantically enriched structural evidence.

• Anchor-driven surgical refinement. We propose a localized refinement strategy that updates only the structurally ambiguous regions around cross-page anchors, avoiding exhaustive pagelevel reprocessing and improving eficiency in long-document settings.

• Efective and eficient multimodal reasoning. Through extensive experiments, we show that Hyper-M2RAG improves multimodal retrieval and downstream answer quality while achieving a better eficiency profile on long and complex documents.

The remainder of this paper details the proposed method. Section 4 formalizes Hyper-M2RAG and presents its multimodal hypergraph construction and anchor-driven refinement mechanism.

## 2 Related Work

Traditional RAG typically relies on flattened chunk retrieval, which struggles to capture long-range dependencies. To bridge this gap, graph-enhanced approaches like GraphRAG [3] and LightRAG [8] extract entity-centric relations to improve multi-hop reasoning. However, these binary graph models are limited to pairwise associations. To model higher-order group semantics, Hypergraph-based

![](images/bd53ba8090a52fb9172fb836218153fc6fb1532288540a8b5cd6944fc8b84da1.jpg)  
Figure 1: Overview of the Hyper-M2RAG framework. The system progresses through three core stages: (1) Multi-modal Data Ingestion via PMUs; (2) Two-stage Hypergraph Construction featuring anchor-driven structural refinement; (3) Cognitiveinspired Hybrid Retrieval. The comparative case (bottom right) highlights our advantage in cross-page factual grounding over vanilla LLMs.

RAG (HyperRAG) [5] introduces hyperedges to connect multiple vertices simultaneously. Recent variants like CogRAG[11] further explore cognitive-inspired hypergraph structures to enhance adaptive reasoning. While these methods excel in text-based high-order modeling, they remain largely "vision-blind" when faced with com plex multi-modal layouts.

Recent studies have begun extending RAG to multi-modal domains, encompassing cross-modal alignment, specialized document parsing, and multi-source retrieval strategies [27]. Systems such as RAG-Anything [7], MegaRAG [10], and MHier-RAG [6] attempt to retrieve interleaved visual and textual components from complex layouts, while complementary approaches address query reformulation [14], heterogeneous source aggregation [16], multi-turn reasoning [12], and evidence fusion [25]. However, existing meth ods sufer from retrieval-modality decoupling: they either treat images as secondary "post-retrieval supplements" or rely on lossy text-only indices (e.g., captions) that collapse the original structural layout. This leads to critical information decay and visual blind spots, especially when evidence spans across multiple pages. Unlike these approaches, our work pursues a native multi-modal indexing paradigm that preserves high-order structural and visual integrity within a unified hypergraph.

## 3 Task Description and Preliminaries

The objective of multi-modal long-context Document-based Question Answering (Doc-QA) is to generate a comprehensive answer

$y _ { a n s }$ based on a query Q and a document D containing heterogeneous modalities. We formalize the generation process as:

$$
Y _ { a n s } = \mathrm { L L M } _ { g e n } ( y _ { i } \mid y _ { < i } , Q , S ) ,\tag{1}
$$

where � represents the Structured Evidence Package retrieved from the document-indexed corpus.

In our Hyper-M2RAG framework, the document � is organized into a multi-modal hypergraph $G = ( V , E )$ , where � denotes a set of normalized semantic entities vertices and $E = \{ E _ { l o w } \cup E _ { h i g h } \cup$ $E _ { r e f i n e } \}$ represents the set of hyperedges capturing multi-order relationships. Specifically, $E _ { l o w }$ and $E _ { h i g h }$ capture intra-page pairwise and group associations, respectively, while $E _ { r e f i n e }$ encapsulates latent cross-page dependencies reconstructed through anchor-driven refinement. The retrieval component is defined as a mapping function:

$$
S = { \mathrm { H y p e r R e t r i e v e r } } ( G , Q ) ,\tag{2}
$$

where HyperRetriever(·) executes a dual-path alignment across the symbolic hypergraph and the vector space. This strategy ensures that � integrates both vertex-anchored entity evidence $C _ { e n t }$ and hyperedge-directed relational context $C _ { r e l } ,$ facilitating complex reasoning over the high-order multi-modal substrate.

## 4 Methodology

The core workflow of Hyper-M2RAG is illustrated in Fig. 1, highlighting the transition from fragmented multi-modal pages to a unified, high-order hypergraph index.

## 4.1 Multi-modal Data Ingestion

To better preserve heterogeneous document structures, we redefine the primary unit of data ingestion for structured knowledge extraction. Unlike conventional RAG systems that treat documents as linear text streams, we adopt the Page-level Multi-modal Unit (PMU) as the fundamental atomic unit. Specifically, we utilize MinerU as a high-fidelity engine to parse raw PDFs into a structured sequence $\mathcal { D } = \{ p _ { 1 } , p _ { 2 } , . . . , p _ { n } \}$ . The �-th unit $\mathcal { P } i$ is defined as:

$$
{ { p } _ { i } } = \{ { T } _ { i } ^ { o c r } , { { I } _ { i } ^ { p a g e } } , \mathcal { F } _ { i , l o c a l } \} ,\tag{3}
$$

where $T _ { i } ^ { o c r }$ is the aggregated textual content, $I _ { i } ^ { p a g e }$ is the global page image, and $\mathcal { F } _ { i , l o c a l } = \{ I _ { i , j } ^ { f i g } \} _ { j = 1 } ^ { n _ { i } }$ represents localized image patches (e.g., figures, tables). This representation assigns distinct semantic roles to visual inputs: $I _ { i } ^ { p a g e }$ provides macro-visual context for layout awareness, while $\mathcal { F } _ { i , l o c a l }$ ofers fine-grained evidence from detected regions. To ensure robust ingestion, vision-dominant pages with sparse text are preserved via semantic placeholders (e.g., “Please see the Figures”). Furthermore, this unified schema treats visual fields as optional (∅), allowing the system to handle both text-only and multi-modal data through a consistent interface.

## 4.2 Multi-modal Hypergraph Index Construction

The construction ofthe HyperIndex follows a hierarchical paradigm: Initial Page-level Extraction followed by Anchor-driven Structural Refinement. This process transforms fragmented page-level units into a cohesive, high-fidelity semantic network.

4.2.1 Initial Page-level Indexing and Aggregation. For each PMU $p _ { i } \in \mathcal { D }$ , we first execute a multi-modal extraction operator LLM<sub>���</sub> to capture the localized entity set $V _ { i }$ and their corresponding high order associations $E _ { i } .$ . Unlike traditional text-based extraction, this process simultaneously digests the textual substrate and hierarchical visual evidence. The extraction is formalized as follows:

$$
\left\{ \begin{array} { r l } { V _ { i } = \mathrm { L L M } _ { e x t } \big ( P _ { e n t } ( \boldsymbol { p } _ { i } ) \big ) } & { } \\ { E _ { i , l o w } = \mathrm { L L M } _ { e x t } \big ( P _ { l o w } ( \boldsymbol { p } _ { i } , V _ { i } ) \big ) } & { \mathrm { ~ f o r ~ } \boldsymbol { p } _ { i } \in \mathcal { D } , } \\ { E _ { i , h i g h } = \mathrm { L L M } _ { e x t } \big ( P _ { h i g h } ( \boldsymbol { p } _ { i } , V _ { i } ) \big ) } & { } \end{array} \right.\tag{4}
$$

where $P _ { e n t } , P _ { l o w } ,$ and $P _ { h i g h }$ are specialized prompts (detailed in Appendix) for identifying vertices, pairwise relations, and highorder correlations, respectively. Each extracted element is tagged with a provenance attribute src = {�} to anchor semantic entities to their original page index �.

To synthesize a document-level structure, a Canonical Merging operator M is applied to aggregate page-level sets. Vertices are normalized via case-insensitive alignment, while hyperedges are collapsed based on their unique, sorted constituent vertex sets:

$$
\mathcal { G } _ { i n i t } = M \left( \bigcup _ { i = 1 } ^ { n } \{ V _ { i } , E _ { i , l o w } , E _ { i , h i g h } \} \right) ,\tag{5}
$$

where $\mathcal { G } _ { i n i t }$ denotes the initial global hypergraph. In this stage, cross-page alignment is achieved through entity unification, where each global node or edge maintains an aggregated provenance index $\mathrm { s r c } _ { g l o b a l } = \bigcup \mathrm { s r c } _ { i }$

4.2.2 Anchor-driven Structural Refinement. While $\mathcal { G } _ { i n i t }$ aggregates local evidence, it remains constrained by the page-level receptive field, often failing to capture long-range semantic dependencies that transcend individual page boundaries. To bridge these disparate fragments, we introduce a refinement stage centered on Crosspage Anchors $V _ { a n c } .$ These anchors are identified as entities with high provenance frequency, representing consistent semantic pivots across the document:

$$
V _ { a n c } = \{ v \in V | | \mathrm { s r c } ( v ) | \geq \tau \} ,\tag{6}
$$

where � is the frequency threshold.

Instead of re-processing entire page sequences with high computational redundancy as in MegaRAG, our approach performs targeted refinement within the topological neighborhoods of $V _ { a n c } .$

For each anchor � ∈ $V _ { a n c } ,$ the system extracts its star-expansion subgraph $G _ { s u b } ( v ) = ( V _ { s u b } , E _ { s u b } )$ to encapsulate the multi-source contexts linked to this pivot through shared hyperedges:

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { E _ { s u b } ( v ) = \{ e \in ( E _ { l o w } \cup E _ { h i g h } ) \mid v \in e \} } \\ { V _ { s u b } ( v ) = \{ u \in V \mid \exists e \in E _ { s u b } ( v ) , u \in e \} } \end{array} \right. } \end{array}\tag{7}
$$

where $E _ { s u b }$ represents the set of all hyperedges (both pairwise and high-order) incident to �, and $V _ { s u b }$ denotes the union of all vertices contained within these incident hyperedges.

This localized sub-hypergraph $G _ { s u b }$ acts as a structural nexus, providing a condensed reasoning substrate for the refinement operator R to bridge disparate page-level fragments and synthesize latent cross-page associations:

$$
E _ { r e f i n e } = \mathrm { L L M } _ { r e f } ( \mathcal { R } ( G _ { s u b } ( v ) , P _ { r e f } ) ) ,\tag{8}
$$

where $P _ { r e f }$ is a specialized prompt directing the model to synthesize novel associations or consolidate existing relations by reconciling multi-source evidence. Finally, the refined relations are integrated into the global multi-modal hypergraph $\ G _ { f i n a l } { \mathrm { : } }$

$$
\mathcal { G } _ { f i n a l } = \{ V , E _ { l o w } , E _ { h i g h } , E _ { r e f i n e } \} .\tag{9}
$$

By shifting the refinement focus from raw page sequences to topological anchor neighborhoods, the system efectively eliminates the redundant processing of stable local information. This ensures a more precise discovery of high-order relational structures while maintaining significant computational eficiency.

## 4.3 Cognitive-inspired Hybrid Retrieval

Guided by the hierarchical structure of the HyperIndex, we design a cognitive-inspired hybrid retrieval strategy to bridge user queries with high-fidelity evidence. As illustrated in our framework, the process initiates with a Query Decomposition stage, followed by a Dual-channel Retrieval mechanism that traverses both the symbolic hypergraph and the dense vector space.

4.3.1 Multi-modal Query Decomposition. For a given user query �, the system first employs an MLLM-based decomposition operator to extract two levels of semantic anchors:

$$
\left\{ K _ { l o w } , K _ { h i g h } \right\} = \mathrm { L L M } _ { d e c } ( P _ { d e c } ( q ) ) ,\tag{10}
$$

where $K _ { l o w }$ represents fine-grained entity keywords (e.g., specific objects or technical terms), and $K _ { h i g h }$ denotes high-level thematic keywords (e.g., abstract concepts or cross-page relations). $P _ { d e c }$ is the decomposition prompt detailed in the Appendix.

4.3.2 Dual-channel Evidence Retrieval. The retrieval process simultaneously executes two complementary paths to collect structured evidence from the HyperIndex:

Path I: Vertex-anchored Entity Retrieval. This path targets finegrained evidence by anchoring the query to specific semantic vertices within the hypergraph. For each keyword $k \in K _ { l o w } ,$ , the system identifies the most relevant vertices $V _ { q }$ via semantic similarity search in the entity vector space:

$$
V _ { q } = \{ v \in V \mid \sin ( \phi ( k ) , \phi ( v ) ) > \gamma \} ,\tag{11}
$$

where $\phi ( \cdot )$ denotes the embedding function and � is the similarity threshold. Using $V _ { q }$ as entry points, the entity-centric context $C _ { e n t }$ is expanded via their incident hyperedges in the HypergraphDB:

$$
C _ { e n t } = \{ e \in ( E _ { l o w } \cup E _ { h i g h } \cup E _ { r e f i n e } ) \mid \exists v \in V _ { q } , v \in e \} .\tag{12}
$$

This operation captures the topological neighbors of the query entities as structural evidence, preserving the immediate relational context and leveraging refined cross-page associations.

Path II: Hyperedge-directed Relational Retrieval. To capture overarching themes and cross-page narratives, this path interfaces directly with the relational vector index using $K _ { h i g h }$ . For each thematic keyword $k \in K _ { h i q h } ,$ the system retrieves a candidate set of highorder hyperedges $E _ { q }$ whose semantic embeddings align with the query intent:

$$
E _ { q } = \{ e \in ( E _ { l o w } \cup E _ { h i g h } \cup E _ { r e f i n e } ) \mid \sin ( \phi ( k ) , \phi ( e ) ) > \gamma \} ,\tag{13}
$$

where $E _ { q }$ specifically targets the refined associations that transcend individual page boundaries. The relation-centric context $C _ { r e l }$ is then synthesized by back-tracing the constituent vertices and their multi-source provenance:

$$
C _ { r e l } = \{ \operatorname { s r c } ( e ) , \operatorname { n o d e s } ( e ) \mid e \in E _ { q } \} ,\tag{14}
$$

where nodes(�) extracts the semantic entities within the hyperedge and src(�) retrieves the original multi-page text chunks. This path ensures the model reconciles broad conceptual queries with the high-order structural evidence consolidated during the refinement stage.

4.3.3 Evidence Fusion and Generation. Finally, the system integrates the multi-source evidence from both retrieval channels into a unified, structured context $S = { \mathrm { F u s e } } ( C _ { e n t } , C _ { r e l } )$ . Unlike conventional RAG systems that return fragmented raw text, � is a Structured Evidence Package comprising: (i) normalized entities, (ii) highorder hyperedges, and (iii) their synchronized multi-page source chunks. The final response � is synthesized as:

$$
R = \operatorname { L L M } _ { g e n } ( P _ { g e n } ( q , S ) ) ,\tag{15}
$$

where $P _ { g e n }$ is a generation prompt that instructs the model to reason over the provided topological structures and cross-page evidence.

The eficacy of this dual-channel paradigm is fundamentally rooted in the structural depth of the ����������. Specifically, the cross-page hyperedges $E _ { r e f i n e }$ generated during the anchor-driven refinement stage enable Path II to retrieve integrated evidence that transcends individual page boundaries. By reconciling these high order associations with local entity-centric details, the framework provides a holistic and factually grounded understanding of complex, long-form documents.

## 5 Experiments

In this section, we provide a comprehensive evaluation of Hyper-M2RAG. We first detail our experimental configurations and benchmark selections, followed by a quantitative analysis comparing our approach against state-of-the-art RAG baselines.

## 5.1 Datasets

To evaluate Hyper-M2RAG’s proficiency in managing multi-modal document structures encompassing both holistic understanding and granular retrieval, we conduct experiments across two task granularities: Global QA and Local QA.

5.1.1 Global QA. To evaluate the system’s ability to synthesize information across entire document collections (book-level), we employ both textual and multi-modal benchmarks:

• Textual Corpus: We utilize the Mixed-Domain (≈0.62M tokens) from the UltraDomain [18] dataset and NeurologyCorp (≈1.97M tokens) from MedRAG [24]. These corpora specifically test the model’s capacity for maintaining long-range semantic consistency within dense, specialized knowledge domains.

• multi-modal Benchmark: To address the scarcity of standardized multi-modal global QA datasets, we utilize a specialized evaluation set characterized by dense, visually-rich content. This benchmark incorporates a World History textbook (788 pages) and a Sustainable Development Report (SDR) slide deck (491 pages), both of which feature a high concentration of interleaved graphical elements, including intricate charts, tables, and thematic illustrations that are deeply integrated with the primary text.

5.1.2 Global Question Generation Protocol. To address the lack of human-labeled queries, we use an automated protocol with two modality-specific pipelines:

• Textual Generation: Using the document outline as a scafold, we prompt an LLM to simulate 5 distinct professional users. For each user, we define 5 strategic tasks, each generating 5 complex questions that require a holistic understanding of the document. This results in 125 global questions per dataset.

• multi-modal Generation: For vision-heavy documents, we establish a generation pipeline. We sample 25 representative pages per dataset (organized into 5 batches of 5 pages). Each batch involves a specific user persona and 5 tasks. Crucially, we enforce a multi-modal-dependency constraint: questions must be unanswerable by text alone, requiring reasoning over visual data such as map coordinates, chart trends, or diagram structures. This ensures 125 high-quality multi-modal questions per dataset.

5.1.3 Local QA. To evaluate retrieval precision at the granular level (page- or slide-level), we utilize the RealMMBench [20] benchmark. RealMMBench is specifically designed to assess multi-modal RAG systems under stressed conditions, including visual-rich layouts, table-heavy content, and sophisticated query rephrasing. While RealMMBench spans multiple domains, we perform a strategic evaluation on the TechReport (1,674 pages) and TechSlides (1,963 pages) sub-datasets. These technical domains are characterized by dense high-order structural dependencies, making them ideal for validating our high-order hypergraph construction and anchor-driven refinement mechanisms. This selection ensures a rigorous stress test of document intelligence while maintaining computational feasibility during the evaluation process.

## 5.2 Baselines and Evaluation Metrics

We compare Hyper-M2RAG against several state-of-the-art baselines: GraphRAG [3], a widely adopted knowledge graph framework; HyperRAG [5], which utilizes text-only hypergraphs; and MegaRAG [10], a multi-modal graph-based RAG system. To ensure a fair assessment, we evaluate all methods across both textual and multi-modal benchmarks to test their core retrieval and reasoning capabilities.

5.2.1 Global QA Metrics. Given the absence of ground-truth answers for book-level queries, prior works often rely on LLM-based evaluation. However, such evaluations are frequently prone to position bias, where the evaluator model favors responses presented earlier in the prompt. To mitigate this and ensure reliability, we implement a Dual-Response Swap & Average strategy:

• Dual-Prompt Evaluation: For every pair of responses (�, �), we generate two distinct evaluation prompts: a forward prompt (A followed by B) and a reversed prompt (B followed by A).

• Result Normalization: To maintain a consistent reference frame, results from the reversed evaluation are flipped (e.g., if B was preferred in the reversed prompt, it is mapped to a preference for the second position in the forward frame).

• Consistency Arbitration: We compare the forward and normal ized reversed results. If the evaluator remains consistent across both prompts, the result is recorded. If the two evaluations conflict, the pair is marked as a Tie to filter out stochastic noise and ensure robust win-rate calculation.

Responses are assessed across four qualitative dimensions following [8]: (1) Comprehensiveness: Coverage of all query facets; (2) Diversity: Richness of perspectives; (3) Empowerment: Support for user understanding; and (4) Overall: An aggregate measure of the preceding criteria.

5.2.2 Local QA Metrics. For granular (page- or slide-level) QA, performance is measured by semantic alignment with ground-truth reference answers. We utilize a powerful LLM as an automated judge to determine semantic consistency between the generated and reference answers, reporting Accuracy as the primary metric.

## 5.3 Implementation Details

To ensure consistency across all evaluated RAG frameworks, we standardize the backbone models and processing pipeline. For response generation, we utilize Qwen3-VL-8B [1], while DeepSeek-v3 [15] is employed for synthetic question generation and evaluation due to its superior reasoning robustness.

All methods share a unified embedding space facilitated by gme Qwen2-VL-2B-Instruct [28], which supports single-, cross-, and fused-modality retrieval tasks. Textual corpora are partitioned into 1,200-token chunks with a 100-token overlap. For multi-modal documents, we leverage the MinerU2.5 toolkit [17] to extract text, figures, and tables. MinerU’s ability to preserve complex layouts and symbols in machine-readable formats is particularly efective for the technical documents used in our study.

## 5.4 Global QA Performance

The primary results in Table 1 compare Hyper-M2RAG against three state-of-the-art baselines across four domains: two purely textual corpora (Mix and Neurology) and two multimodal datasets (World History and Sustainable Report). While our method maintains a consistent lead in textual tasks, it achieves a decisive, overwhelming victory in multimodal scenarios. Our analysis yields the following key insights:

Superiority across Diverse Domains. Hyper-M2RAG consistently outperforms GraphRAG, HyperRAG, and MegaRAG across all qualitative dimensions. Notably, in the Sustainable Report domain characterized by high-density multimodal content and complex layouts, our method achieves a dominant Overall win rate of 88.8% against MegaRAG and 77.6% against GraphRAG. These results demonstrate that our high-order hypergraph structure captures multi-scale semantic dependencies in technical documents more efectively than traditional binary graphs.

Gains in Diversity and Empowerment. Hyper-M2RAG demonstrates a substantial lead in Diversity and Empowerment metrics. For instance, in the World History domain, our method achieves a 75.2% win rate in Diversity against MegaRAG. This performance gain is driven by our hyper-relational modeling mechanism, which transcends the limitations of binary-edge retrieval. Unlike traditional graphs that only link pairwise entities, our high-order hyperedges group multi-modal entities into unified semantic clusters. This allows the retriever to capture non-local, multi-hop connections across disparate document sections in a single traversal.

Robustness against Hypergraph Baselines. Compared to the textonly HyperRAG, Hyper-M2RAG maintains a consistent margin of improvement, particularly in Overall win rates ranging from 22.4% to 44.8%. While HyperRAG introduces basic hyper-structures, our anchor-driven refinement and multimodal integration ensure that the constructed hypergraphs are semantically precise rather than merely dense. The high tie rates in Comprehensiveness (e.g., 91.2% in Mix) indicate that while most hypergraph methods achieve basic content coverage, Hyper-M2RAG excels in the qualitative depth and organizational clarity of the generated responses.

Multimodal Advantage. In multimodal datasets like World History and Sustainable Report, the performance gap between Hyper-M2RAG and baselines widens significantly. This advantage stems from our layout-aware hyperedges, which directly link visual elements to their corresponding textual descriptions within a unified topological space. For instance, the 90.4% Empowerment score in the Sustainable domain demonstrates the model’s superior ability to interpret and synthesize information from complex tables and figures.

## 5.5 Local QA Performance

Table 2 reports the accuracy of diferent methods on the TechReport and TechSlides sub-datasets of RealMMBench. Hyper-M2RAG achieves superior performance, reaching 70.0% and 78.0% accuracy, respectively. Compared with the state-of-the-art baseline MegaRAG, our method achieves a notable 9 percentage point improvement on

Table 1: Win-rate analysis comparing Hyper-M2RAG against GraphRAG, HyperRAG, and MegaRAG across four distinct domains. Results are presented as percentages (%). Bold values indicate the winning method (excluding ties).
<table><tr><td rowspan="2">Metrics</td><td colspan="3">Mix</td><td colspan="3">Neurology</td><td colspan="3">World History</td><td colspan="3">Sustainable Report</td></tr><tr><td>GraphRAG</td><td>Ours</td><td>Tie</td><td>GraphRAG</td><td>Ours</td><td>Tie</td><td>GraphRAG</td><td>Ours</td><td>Tie</td><td>GraphRAG</td><td>Ours</td><td>Tie</td></tr><tr><td>Comprehensiveness</td><td>0.0</td><td>46.4</td><td>53.6</td><td>0.8</td><td>37.6</td><td>61.6</td><td>0.8</td><td>47.2</td><td>52.0</td><td>0.0</td><td>62.4</td><td>37.6</td></tr><tr><td>Diversity</td><td>11.2</td><td>60.8</td><td>28.0</td><td>9.6</td><td>56.0</td><td>34.4</td><td>4.8</td><td>69.6</td><td>25.6</td><td>3.2</td><td>91.2</td><td>5.6</td></tr><tr><td>Empowerment</td><td>4.0</td><td>56.0</td><td>40.0</td><td>3.2</td><td>61.6</td><td>35.2</td><td>5.6</td><td>66.4</td><td>28.0</td><td>1.6</td><td>77.6</td><td>20.8</td></tr><tr><td>Overall</td><td>2.4</td><td>51.2</td><td>46.4</td><td>2.4</td><td>52.0</td><td>45.6</td><td>4.8</td><td>61.6</td><td>33.6</td><td>0.8</td><td>77.6</td><td>21.6</td></tr><tr><td></td><td>HyperRAG</td><td>Ours</td><td>Tie</td><td>HyperRAG</td><td>Ours</td><td>Tie</td><td>HyperRAG</td><td>Ours</td><td>Tie</td><td>HyperRAG</td><td>Ours</td><td>Tie</td></tr><tr><td>Comprehensiveness</td><td>0.0</td><td>8.8</td><td>91.2</td><td>6.4</td><td>17.6</td><td>76.0</td><td>0.0</td><td>44.0</td><td>56.0</td><td>1.6</td><td>29.6</td><td>68.8</td></tr><tr><td>Diversity</td><td>23.2</td><td>35.2</td><td>41.6</td><td>3.2</td><td>18.4</td><td>78.4</td><td>15.2</td><td>54.4</td><td>30.4</td><td>17.6</td><td>43.2</td><td>39.2</td></tr><tr><td>Empowerment</td><td>15.2</td><td>28.8</td><td>56.0</td><td>14.4</td><td>28.8</td><td>56.8</td><td>16.8</td><td>46.4</td><td>36.8</td><td>20.8</td><td>26.4</td><td>52.8</td></tr><tr><td>Overall</td><td>13.6</td><td>27.2</td><td>59.2</td><td>11.2</td><td>22.4</td><td>66.4</td><td>14.4</td><td>44.8</td><td>40.8</td><td>11.2</td><td>25.6</td><td>63.2</td></tr><tr><td></td><td>MegaRAG</td><td>Ours</td><td>Tie</td><td>MegaRAG</td><td>Ours</td><td>Tie</td><td>MegaRAG</td><td>Ours</td><td>Tie</td><td>MegaRAG</td><td>Ours</td><td>Tie</td></tr><tr><td>Comprehensiveness</td><td>4.8</td><td>53.6</td><td>41.6</td><td>0.0</td><td>20.0</td><td>80.0</td><td>0.0</td><td>44.0</td><td>56.0</td><td>0.0</td><td>64.0</td><td>36.0</td></tr><tr><td>Diversity</td><td>8.8</td><td>44.8</td><td>46.4</td><td>6.4</td><td>52.8</td><td>40.8</td><td>2.4</td><td>75.2</td><td>22.4</td><td>1.6</td><td>90.4</td><td>8.0</td></tr><tr><td>Empowerment</td><td>3.2</td><td>63.2</td><td>33.6</td><td>0.8</td><td>44.8</td><td>54.4</td><td>3.2</td><td>62.4</td><td>34.4</td><td>1.6</td><td>90.4</td><td>8.0</td></tr><tr><td>Overall</td><td>5.6</td><td>58.4</td><td>36.0</td><td>0.0</td><td>36.8</td><td>63.2</td><td>1.6</td><td>59.2</td><td>39.2</td><td>1.6</td><td>88.8</td><td>9.6</td></tr></table>

Table 2: Local QA performance (Accuracy %) on RealMM-Bench.
<table><tr><td>Method</td><td>TechReport</td><td>TechSlides</td></tr><tr><td>GraphRAG</td><td>30.0</td><td>57.0</td></tr><tr><td>HyperRAG</td><td>56.0</td><td>67.0</td></tr><tr><td>MegaRAG</td><td>68.0</td><td>69.0</td></tr><tr><td>Hyper-M2RAG (Ours)</td><td>70.0</td><td>78.0</td></tr></table>

TechSlides. This specific gain highlights the eficacy of our framework in visual-heavy contexts. Technical slides often contain fragmented information across diferent visual blocks; while MegaRAG relies on dense graph connectivity, our high-order hypergraph structure more efectively clusters spatially disparate but semantically related elements (e.g., a figure and its corresponding bullet points), leading to more precise local retrieval.

## 5.6 Ablation Study

To investigate the contribution of diferent components in Hyper-M2RAG, we progressively introduce multi-modal extraction and anchor-driven refinement based on the HyperRAG baseline, as shown in Table 3. Our analysis yields two primary insights:

Efect of Multi-modal Hypergraph Construction. Adding multimodal extraction (+ MM Extraction) consistently improves performance over the text-only HyperRAG baseline on both TechReport and TechSlides datasets. This demonstrates that incorporating visual elements and layout information helps preserve cross-modal semantic dependencies, enabling more complete evidence retrieval from complex documents.

Efect of Anchor-driven Refinement. Introducing anchor-driven refinement further improves the performance over the multi-modal hypergraph variant. This verifies that selectively reconstructing local neighborhoods around cross-page anchors efectively captures long-range dependencies while avoiding unnecessary processing of irrelevant regions.

Overall, the ablation results confirm that both multi-modal representation and anchor-driven refinement are essential components of Hyper-M2RAG, with their combination enabling more efective multimodal document understanding.

Table 3: Ablation study of Hyper-M2RAG components.
<table><tr><td>Variant</td><td>TechReport</td><td>TechSlides</td></tr><tr><td>HyperRAG (Baseline)</td><td>56.0</td><td>67.0</td></tr><tr><td>+ MM Extraction</td><td>64.0</td><td>74.0</td></tr><tr><td>+ Anchor Refinement</td><td>60.0</td><td>69.0</td></tr><tr><td>Hyper-M2RAG (Full)</td><td>70.0</td><td>78.0</td></tr></table>

## 5.7 Structural Complexity Analysis

To investigate the structural characteristics of Hyper-M2RAG, Table 4 compares the hypergraph statistics of diferent methods, with particular focus on high-order relations (degree > 2).

High-OrderRelational Modeling. Hyper-M2RAG constructs substantially more high-order hyperedges than existing approaches. On TechReport, Hyper-M2RAG builds 8,257 high-order hyperedges, compared with 2,579 from HyperRAG, representing a 220% increase. Similarly, on TechSlides, the number of high-order hyperedges increases from 1,716 to 6,571. Since high-order hyperedges explicitly represent dependencies among multiple entities, this increase indicates that Hyper-M2RAG provides a richer structural representation beyond conventional pairwise relations.

Efect of Multi-modal Hypergraph Construction. Interestingly, simply incorporating multi-modal information does not necessarily lead to more high-order structures. The HyperRAG+MM variant produces fewer high-order hyperedges than HyperRAG on TechReport (2,078 vs. 2,579). This suggests that modality augmentation alone is insuficient to establish efective high-order relations. In contrast, Hyper-M2RAG jointly models textual, visual, and lay out elements through hypergraph construction and anchor-driven refinement, resulting in substantially richer high-order relational structures.

Table 4: Comparison of structural hyperedge counts across datasets.
<table><tr><td rowspan="2">Variant</td><td colspan="3">TechReport (1,674 pages)</td><td colspan="3">TechSlides (1,963 pages)</td></tr><tr><td>Low</td><td>High</td><td>Total</td><td>Low</td><td>High</td><td>Total</td></tr><tr><td>HyperRAG (Baseline)</td><td>12,957</td><td>2,579</td><td>15,536</td><td>8,543</td><td>1,716</td><td>10,259</td></tr><tr><td>HyperRAG + MM</td><td>10,016</td><td>2,078</td><td>12,094</td><td>9,309</td><td>2,004</td><td>11,313</td></tr><tr><td>Hyper-M2RAG (Ours)</td><td>14,221</td><td>8,257</td><td>22,478</td><td>12,522</td><td>6,571</td><td>19,093</td></tr></table>

Overall Structural Connectivity. Beyond high-order relations, Hyper-M2RAG also achieves the largest number of total hyperedges across both datasets, with 22,478 edges on TechReport and 19,093 edges on TechSlides. This demonstrates that our framework constructs a more densely connected document representation, providing a stronger structural foundation for retrieving evidence distributed across long and complex documents.

Table 5: Multi-Judge Audit (%). H-RAG = HyperRAG.
<table><tr><td></td><td colspan="3">Sustainable Report</td><td colspan="3">World History</td></tr><tr><td>Evaluator</td><td>H-RAG</td><td>Ours</td><td>Tie</td><td>H-RAG</td><td>Ours</td><td>Tie</td></tr><tr><td>Kimi-K2.6</td><td>25.6</td><td>55.2</td><td>19.2</td><td>24.0</td><td>44.0</td><td>32.0</td></tr><tr><td>GLM-5.1</td><td>25.6</td><td>51.2</td><td>23.2</td><td>24.8</td><td>38.4</td><td>36.8</td></tr><tr><td>DeepSeek-V4 Pro</td><td>15.2</td><td>41.6</td><td>43.2</td><td>20.0</td><td>36.0</td><td>44.0</td></tr></table>

## 5.8 Cross-Model Judge Audit

Table 5 presents a multi-judge evaluation on the Sustainable Report and World History corpora, conducted with three recently released, architecturally diverse LLMs: Kimi-K2.6, GLM-5.1, and DeepSeek V4 Pro. We select HyperRAG as the primary baseline for this audit because it is the most structurally aligned counterpart to Hyper-M2RAG: both methods organize knowledge using hypergraphs, while only Hyper-M2RAG introduces multimodal anchors and starexpansion refinement. This design isolates the contribution of our multimodal mechanism from general hypergraph-induced gains.

Across both domains and all three evaluators, Hyper-M2RAG consistently outperforms HyperRAG, with win rates ranging from 36.0% to 55.2%. Crucially, the preference for Hyper-M2RAG is not judge-specific: the same ranking trend holds for all model families, indicating that the observed improvements are not a single-judge artifact but a robust, cross-model phenomenon.

## 5.9 Reduction in Token Cost and Latency

Table 6 provides a head-to-head eficiency comparison on the World History corpus (per 100 pages). Hyper-M2RAG reduces build and refinement time to 12 min and lowers token consumption by 72% relative to MegaRAG and 77% relative to HyperRAG+Page. This gain stems from a fundamental shift in refinement granularity.

Table 6: Eficiency on World History (per 100 pages).
<table><tr><td>Metric</td><td>MegaRAG</td><td>HyperRAG + Page</td><td>Ours</td></tr><tr><td>Build &amp; Refine Time (min)</td><td>16</td><td>18</td><td>12</td></tr><tr><td>Token Consumption (103)</td><td>2,742.2</td><td>3,376.1</td><td>765.9</td></tr></table>

Rather than performing document- or page-level rescans during incremental updates, Hyper-M2RAG organizes multimodal evidence into a hypergraph, where anchors serve as structural pivots linking text, tables, figures, and layout elements. Refinement is then confined to anchor-centered star neighborhoods: only hyperedges incident to updated anchors are recomputed.

Consequently, the update process is transformed from a global sweep into a topology-guided local operation. This design eliminates redundant computation across unrelated content while preserving rich cross-modal linkages. Because the number of anchors requiring updates grows more slowly than document length, the refinement cost scales sub-linearly, allowing Hyper-M2RAG to maintain high eficiency on long, multimodal documents without altering the input scope or representation.

## 6 Conclusion

In this paper, we presented Hyper-M2RAG, a novel multi-modal retrieval-augmented generation framework designed for complex, high-density technical documents. By transitioning from traditional binary graphs to a high-order hypergraph structure, our approach efectively captures the multi-scale semantic dependencies that are often fragmented across disparate document sections and modalities. Our core contribution lies in the integration of layout-aware hyperedge construction and an anchor-driven refinement mechanism. This synergy ensures that visual elements, such as tables and figures, are not merely treated as isolated inputs but are topologically aligned with their textual context within a unified semantic space. Furthermore, our dual-path retrieval strategy—comprising Vertexanchored Entity Retrieval and Hyperedge-directed Relational Retrieval—enables the model to reconcile fine-grained evidence with overarching thematic narratives. Experimental results across four diverse domains demonstrate that Hyper-M2RAG consistently outperforms state-of-the-art baselines, particularly in multi-modal scenarios where it achieves significant gains in Diversity and Empowerment.

## 7 Acknowledgments

This work was supported by Brain Science and Brain-like Intelligence Technology——National Science and Technology Major Project (2025ZD0217300), the National Key Research and Development Program of China under Grant (2023YFB4502803), the Na tional Natural Science Foundation of China (No. U25A20532) and the Beijing Natural Science Foundation under Grant (No. L242167).

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025).

[2] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al. 2025. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595 (2025).

[3] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130 (2024).

[4] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2025. Colpali: Eficient document retrieval with vision language models. In International Conference on Learning Representations, Vol. 2025. 61424–61449.

[5] Yifan Feng, Hao Hu, Xingliang Hou, Shiquan Liu, Shihui Ying, Shaoyi Du, Han Hu, and Yue Gao. 2026. Hyper-rag: Combating LLM hallucinations using hypergraph driven retrieval-augmented generation. Nature Communications 17 (2026), 5778. doi:10.1038/s41467-026-71411-1

[6] Ziyu Gong, Chengcheng Mai, and Yihua Huang. 2025. MHier-RAG: Multi-Modal RAG for Visual-Rich Document Question-Answering via Hierarchical and Multi Granularity Reasoning. arXiv preprint arXiv:2508.00579 (2025).

[7] Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and Chao Huang. 2025. Rag anything: All-in-one rag framework. arXiv preprint arXiv:2510.12323 (2025).

[8] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG: Simple and Fast Retrieval-Augmented Generation. In Findings ofthe Association for Computational Linguistics: EMNLP 2025. Association for Computational Lin guistics, Suzhou, China, 10746–10761. doi:10.18653/v1/2025.findings-emnlp.568

[9] Mahd Hindi, Linda Mohammed, Ommama Maaz, and Abdulmalik Alwarafy. 2025. Enhancing the precision and interpretability of retrieval-augmented generation (rag) in legal technology: A survey. IEEE Access (2025).

[10] Chi-Hsiang Hsiao, Yi-Cheng Wang, Tzung-Sheng Lin, Yi-Ren Yeh, and Chu-Song Chen. 2026. MegaRAG: Multimodal Knowledge Graph-Based Retrieval Aug mented Generation. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 48031–48059.

[11] Hao Hu, Yifan Feng, Ruoxue Li, Rundong Xue, Xingliang Hou, Zhiqiang Tian, Yue Gao, and Shaoyi Du. 2026. Cog-rag: Cognitive-inspired dual-hypergraph with theme alignment retrieval-augmented generation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 40. 31032–31040.

[12] Pankaj Joshi, Aditya Gupta, Pankaj Kumar, and Manas Sisodia. 2024. Robust multi model rag pipeline for documents containing text, table & images. In 2024 3rd International Conference on Applied Artificial Intelligence and Computing (ICAAIC). IEEE, 993–999.

[13] Wenjun Ke, Yifan Zheng, Yining Li, Hengyuan Xu, Dong Nie, Peng Wang, and Yao He. 2025. Large language models in document intelligence: A comprehensive survey, recent advances, challenges, and future trends. ACM Transactions on Information Systems 44, 1 (2025), 1–64.

[14] Zhicong Li, Jiahao Wang, Zhishu Jiang, Hangyu Mao, Zhongxia Chen, Jiazhen Du, Yuanxing Zhang, Fuzheng Zhang, Di Zhang, and Yong Liu. 2024. Dmqr-rag: Diverse multi-query rewriting for rag. arXiv preprint arXiv:2411.13154 (2024).

[15] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).

[16] Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, and Jun Ma. 2025. Hm-rag: Hierarchical multi-agent multimodal retrieval augmented generation. In Proceedings of the 33rd ACM international conference on multimedia. 2781–2790.

[17] Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao He, Fan Wu, Qintong Zhang, et al. 2026. Mineru2.5: A decoupled vision-language model for eficient high-resolution document parsing. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 6: Industry Track). Association for Computational Linguistics, San Diego, California, USA, 13–42. doi:10.18653/v1/2026.acl-industry.3

[18] Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and Tiejun Huang. 2025. MemoRAG: Boosting Long Context Processing with Global Memory-Enhanced Retrieval Augmentation. In Proceedings ofthe ACM Web Conference 2025 (TheWebConf2025). ACM, Sydney, Australia. https: //arxiv.org/abs/2409.05591

[19] Minzheng Wang, Longze Chen, Fu Cheng, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan Xu, Lei Zhang, Run Luo, et al. 2024. Leave no document behind: Benchmarking long-context llms with extended multi-doc qa. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 5627–5646.

[20] Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz Goldfarb, Eli Schwartz, Udi Barzelay, and Leonid Karlinsky. 2025. Real-mm-rag: A real-world multimodal retrieval benchmark. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 31660–31683.

[21] Haoran Wei, Yaofeng Sun, and Yukun Li. 2026. DeepSeek-OCR 2: Visual Causal Flow. arXiv preprint arXiv:2601.20552 (2026).

[22] Junda Wu, Yu Xia, Tong Yu, Xiang Chen, Sai Sree Harsha, Akash V Maharaj, Ruiyi Zhang, Victor Bursztyn, Sungchul Kim, Ryan A Rossi, et al. 2025. Doc-react: Multi-page heterogeneous document question-answering. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 2: Short Papers). 67–78.

[23] Shangyu Wu, Ying Xiong, Yufei Cui, Haolun Wu, Can Chen, Ye Yuan, Lianming Huang, Xue Liu, Tei-Wei Kuo, Nan Guan, et al. 2026. Retrieval-augmented generation for natural language processing: a survey. Artificial Intelligence Review (jun 2026). doi:10.1007/s10462-026-11605-7

[24] Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. 2024. Benchmarking retrieval-augmented generation for medicine. In Findings of the Association for Computational Linguistics: ACL 2024. 6233–6251.

[25] Diji Yang, Jinmeng Rao, Kezhen Chen, Xiaoyuan Guo, Yawen Zhang, Jie Yang, and Yi Zhang. 2024. Im-rag: Multi-round retrieval-augmented generation through learning inner monologues. In Proceedings ofthe 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 730–740.

[26] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. 2025. Visrag: Vision-based retrievalaugmented generation on multi-modality documents. In International Conference on Learning Representations, Vol. 2025. 21074–21098.

[27] Rui Zhang, Chen Liu, Yixin Su, Ruixuan Li, Xuanjing Huang, Xuelong Li, and Philip S Yu. 2025. A comprehensive survey on multimodal RAG: all combinations of modalities as input and output. Authorea Preprints (2025).

[28] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. 2024. GME: improving universal multimodal retrieval by multimodal LLMs. arXiv preprint arXiv:2412.16855 (2024).