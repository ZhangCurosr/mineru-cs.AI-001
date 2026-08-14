# Lines and Ladders: A Context-Aware Multi-Agent Framework for Large-Scale Retail Price Taxonomy

Ravi Teja Chunduri Walmart Global Tech Bentonville, AR, USA raviteja.chunduri@walmart.com

Srikaran Reddy Boya Walmart Global Tech Bentonville, AR, USA Srikaran.Reddy.Boya@samsclub.com

Ajay Kumar B   
Walmart Global Tech   
Seattle, WA, USA   
Ajay.Kumar10@walmart.com

Karthik Kumaran Walmart Global Tech Sunnyvale, CA, USA karthik.kumaran@walmart.com

Deep Narayan Mishra Walmart Global Tech Sunnyvale, CA, USA deep.mishra@walmart.com

Pranay Kona   
Walmart Global Tech   
Sunnyvale, CA, USA   
pranay.kona@walmart.com

Abstract—Maintaining price consistency and executing an Every Day Low Price strategy is critical for global retailers. However, with catalogs spanning millions of active items, manual governance of price relationships is infeasible. Inconsistent pricing across item variants distorts customer value perception and cannibalizes sales. To address this, we present a scalable, context-aware Multi-Agent Framework designed to automate the construction of “Lines and Ladders” pricing taxonomies. Our framework employs specialized LLM agents to construct these coherent pricing structures by identifying key attributes, extracting multi-modal values, and applying hierarchical grouping logic. Evaluated on real-world enterprise data and deployed in production, our 3-Agent system achieves an F1-score of 0.83 for Lines, outperforming single-agent baselines by mitigating cognitive overload. The system achieves > 90% precision and > 75% recall in Food & Consumables, and 80.2% assignment accuracy in the unstructured General Merchandise catalog.

Index Terms—Multi-Agent Systems, Taxonomy Construction, Multi-modal Information Extraction, Product Clustering, Ecommerce

## I. INTRODUCTION

Retail pricing directly dictates the core profitability of a retailer [1]. To maintain consistency, retailers structure products into “lines and ladders”—a logical hierarchy grouping items by features, quality, and business context [2]. Traditionally, these structures are created through a manual process led by experts, where category managers make decisions based on deep domain knowledge [3]. However, this approach is inherently static and does not meet the dynamic nature of modern retail catalogs.

For global retailers, price governance complexity scales nonlinearly. With millions of active items, high assortment velocity renders manual oversight infeasible. The core problem is not merely the volume of items, but the operational bottleneck created by the inconsistent and often non-standardized nature of catalog data. When relationships are ignored, illogical pricing gaps emerge, leading to customer arbitrage [4] [5]. In unmanaged catalogs, we frequently observe two gaps: Variant Inconsistency (e.g., identical bikes priced at \$238 vs \$199 solely due to color) and Data Discrepancy (e.g., a 0.1-inch typo in a mirror’s dimensions creating a \$7.50 gap).

To address this scalability challenge, we propose a contextaware multi-agent framework. Unlike traditional rule-based systems, our approach leverages specialized agents to automatically cluster products into meaningful “Lines & Ladders”. This system automates the identification and extraction of key attributes, customized for each ProductType. We utilize this architecture to simulate and replicate the decision-making logic of a human expert at scale [6].

This framework serves the primary objective of automated price governance, establishing the structural foundation required for next-generation autonomous pricing systems such as algorithmic demand forecasting and dynamic pricing models [7]. The main contributions of this paper are summarized as follows:

• Formalizing Taxonomy Construction: We decompose pricing governance into Similarity & Variance discovery via a 3-agent architecture.

• Scalable Multi-Modal Execution: We detail a distributed pipeline utilizing a multi-model routing strategy to process millions of items efficiently.

• Validating Industrial Efficacy: We provide extensive ablation studies and post-launch metrics demonstrating the system’s scalability and operational impact in a live enterprise environment.

## II. RELATED WORK

The concept of “lines and ladders” has been indirectly established in retail strategy for decades. Dean [8] originally defined “price lining” as the practice of offering a range of products at specific, distinct price points to signal varying levels of quality to the consumer. While there is extensive literature on product line pricing problems, as comprehensively reviewed by Guiltinan [9], our work addresses a fundamentally different challenge. Unlike traditional approaches that focus on the mathematical optimization of setting prices, our objective is to automatically cluster a massive catalog into coherent product lines based on quality and features. This structural organization is a prerequisite for scalable price governance rather than price setting itself.

The term “ladder” is a more recent term and can be interpreted in various ways. Dobbs [10] describes a ladder as a form of wholesale pricing contract where the wholesale price is contingent on the retail price, facilitating tiered price setting by the manufacturer. In a marketing context, price laddering aligns closely with price lining, typically involving a “good-better-best” strategy [11]. Furthermore, the concept of “climbing a ladder” refers to the strategy of incentivizing customers to upgrade to higher-quality products within the offered set, as described by [12] and [13].

Draganska and Jain [14] further distinguish between vertical and horizontal product differentiation. In vertical differentiation, products vary by quality, whereas in horizontal differentiation, firms offer products that vary in characteristics such as scent, color, or flavor rather than quality. Their empirical analysis confirms that product lines function as effective price discrimination tools, justifying the common strategy of pricing lines differently based on quality while pricing variants (e.g., flavors or colors) uniformly. While the literature offers various definitions for ladders, in the context of large-scale retail, we specifically define ladders in terms of volume incentives. Therefore, for the purposes of this paper, we refer to “volume discount ladders” simply as ladders.

Although these concepts explain the strategic “why”, they fail to address the operational “how” for massive, unstructured datasets. Pre-LLM (Large Language Models) approaches relied on structured tabular data and extensive feature engineering [15], a workflow rendered impractical by the heterogeneity of modern retail catalogs. The manual effort required to standardize attributes across millions of items is practically impossible. LLMs offer a paradigm shift, enabling the direct interpretation of unstructured text and images to automate feature extraction at scale.

## III. METHODOLOGY

To overcome the limitations of manual governance, we propose a Context-Aware Multi-Agent Framework. We avoid a single-prompt approach as it yields unstable outputs. Instead, the system leverages specialized LLMs to perform a two-pronged analysis, discovering key attributes from nonstandardized data.

## A. Problem Formulation & Scoping

Let $C = \{ i _ { 1 } , i _ { 2 } , . . . , i _ { N } \}$ be a catalog of N items, each with raw metadata $R _ { i } \ { \mathrm { ( e . g . } }$ , title, description, image). As shown in Figure 1, we map each item to a Line $( L _ { k } ) \mathrm { : }$ : a subset of items sharing functional equivalence (e.g., identical items differing only in flavor or color), where $\forall i _ { a } , i _ { b } \in L _ { k } \implies$ $P r i c e ( i _ { a } ) = P r i c e ( i _ { b } )$ and Lines aggregate into a Ladder $( M _ { j } ) { \mathrm { : } }$ a collection of Lines linked by value-based differentiation (e.g., volume, pack size), enforcing a tiered pricing logic where $V o l u m e ( L _ { a } ) > V o l u m e ( L _ { b } ) \implies U n i t P r i c e ( L _ { a } ) <$ $U n i t P r i c e ( L _ { b } )$

![](images/36df2dde029a4d0e09b254e9006c1654d7c60a533bf4dd5c01a724dcc70a657f.jpg)  
Fig. 1. Definition of Lines & Ladders

Analyzing millions of items via a global, top-down approach is computationally expensive and lacks context. Therefore, we decompose the catalog using ProductType (e.g., “Candles”) as our granular anchor. Scoping the analysis to items with shared structural characteristics ensures the extracted attributes are highly relevant, allowing the framework to generate precise, tailored taxonomies rather than relying on generic, one-sizefits-all rules.

## B. Dual-Pronged Attribute Discovery

Constructing effective taxonomies requires distinguishing between Similarity Attributes (features defining functional equivalence at the same price point) and Variance Attributes (features driving price differentiation). We achieve this using two specialized LLM agents. To ensure the generated taxonomy is strictly based on physical attributes, these agents operate without few-shot examples of historical taxonomies. Because legacy merchant structures often reflect ad-hoc operational strategies (e.g., grouping all items from a single supplier) rather than strict attribute equivalence, injecting them into the prompts would introduce strategic noise and confuse the algorithmic extraction.

1) Identifying Price Similarity Attributes (Agent 1): This agent isolates base features common to identically priced items, defining the core value proposition. Figure 2 outlines the prompt strategy.

• Density-based Sampling: From ProductType data set $D _ { p t }$ , we identify the K most frequent tuples $T _ { s i m } ~ = ~ \{ ( B r a n d , M S R P ) _ { 1 } , . . . , ( B r a n d , M S R P ) _ { K } \}$ where MSRP represents manufacturer’s suggested retail price. We strictly utilize base MSRP to avoid promotional noise.

• Corpus Generation: For each tuple, we sample a corpus $C _ { t }$ of 15 item descriptions, yielding K distinct corpora representing clusters of identically priced items.

• Iterative Extraction: Agent 1 processes each $C _ { t }$ to extract consistent core attributes, utilizing a persistent memory $P _ { s i m }$ to iteratively refine its logic. For a “Candles” ProductType, outputs might include {weight oz, height inches, burn time hours}.

2) Identifying Price Variance Attributes (Agent 2): This agent identifies premium features that justify price stratification.

![](images/702174778758ec05ed07c826ee7a89a1e72a8794287d18d0c5282a4047233947.jpg)  
Fig. 2. Prompt strategy to identify similarity attributes

Figure 3 outlines the prompt strategy, instructing the agent to act as a Pricing Strategist.

• Stratified Sampling: To capture price diversity, we select $B _ { t o p } .$ , the top 5 frequent brands in $D _ { p t }$ . For each brand $b \in B _ { t o p } ,$ we identify its top 5 frequent MSRP points, creating a stratified set $T _ { v a r }$ of up to 25 groups, $T _ { v a r } = \{ ( B r a n d _ { 1 } , M S R P _ { 1 } ) , \ldots , ( B r a n d _ { 5 } , M S R P _ { 5 } ) \}$ Empirical trials indicated that expanding beyond 5 brands introduces excessive nomenclature variance that obscures core ProductType signal.

• Corpus Generation: We sample 5 items for each group in $T _ { v a r }$ . Unlike Agent 1, Agent 2 ingests the entire stratified dataset $D _ { v a r }$ as a single input to analyze a wide quality and price spectrum.

• Comparative Analysis: Agent 2 compares low- and high-priced groups across $D _ { v a r }$ to identify differentiating features (e.g., {number of wicks, wax type, has decorative vessel}). The agent is also equipped with a memory module, allowing it to incorporate feedback and refine attributes based on specific business requirements.

Figure 7 illustrates this end-to-end workflow, detailing the parallel execution and subsequent synthesis of Agents outputs.

3) Parameter Selection & Heuristics: The sampling constants used by Agent 1 (30 groups / 15 items) and Agent 2 (25 groups / 5 items) were derived via parameter tuning to optimize the Pareto frontier between signal-to-noise ratio and LLM token costs. As shown in Table I, providing more data to the LLM degrades performance. Expanding sample breadth (e.g., 80 groups) introduces long-tail catalog noise and niche brands that confuse schema generation. Expanding depth (e.g., 30 items) causes context bloat, leading the LLM to hallucinate variance attributes based on minor, irrelevant differences. Furthermore, average pipeline runtimes were scaled up to 155 minutes for the 80/30 configuration, compared to ∼65 minutes for our proposed thresholds.

![](images/0f0c403716c73553beca154daf487b9ba1b41f5ac447b4d51dfd861743c68673.jpg)  
Fig. 3. Prompt strategy to identify variance attributes

TABLE I  
IMPACT OF SAMPLE SIZE (GROUPS / ITEMS PER GROUP)
<table><tr><td>Metric (Avg 14 Categories)</td><td>50/15</td><td>80/15</td><td>50/30</td><td>80/30</td><td>Proposed*</td></tr><tr><td>Lines (F1 Score)</td><td>0.77</td><td>0.75</td><td>0.78</td><td>0.77</td><td>0.83</td></tr><tr><td>Ladders (F1 Score)</td><td>0.80</td><td>0.76</td><td>0.78</td><td>0.75</td><td>0.82</td></tr></table>

<sup>∗</sup>30 groups / 15 items for Agent 1, and 25 groups / 5 items for Agent 2.

## C. Synthesis Agent

The similarity and variance agents yield distinct attributes sets, $A _ { s i m }$ and $A _ { v a r } ,$ which often exhibit semantic overlap. To resolve these dualities, a Synthesis Agent performs a mapping function Φ to produce a canonical schema $S _ { p t } \colon S _ { p t } = \Phi ( { \cal A } _ { s i m } \cup$ $A _ { v a r } )$

Figure 4 provides the implemented prompt logic. The execution of Φ ensures schema integrity through three sequential operations.

• Attribute Consolidation & Semantic Resolution: Raw outputs from Agents 1 and 2 are aggregated to eliminate redundancy, resolve synonymy (e.g., merging “flavor” and “taste”), and enforce standardized naming conventions.

• Attribute-Unit Decoupling: To resolve UOM inconsistencies (e.g., oz vs g) and enable comparative analysis, the framework explicitly decouples numerical values from units, defining a scalar value field and a UOM field for every quantitative attribute. This yields a streamlined, high-fidelity schema blueprint for downstream extraction.

![](images/b43989c52e922d4e48103ebd40274a99495057d893efed7c24a8077b061138f2.jpg)  
Fig. 4. Prompt strategy for Schema Synthesis

## D. Multi-Modal Attribute Extraction

Following schema synthesis, the framework populates the schema for every item to transform unstructured data into structured feature vectors.

• Attribute Name: wax type   
Attribute Type: String   
Default value: paraffin   
Prompt: Identify the type of wax used in the candle (e.g., soy,   
paraffin, beeswax, coconut) from the image or description.   
• Attribute Name: is decorative   
Attribute Type: Boolean   
Default value: False   
Prompt: Indicate True if the candle features specific artwork,   
themes, or a sculpted shape.  
Fig. 5. Sample Schema Definition for Extraction

1) Schema Definition:: The canonical schema $S _ { p t }$ generated in the previous step consists of k attributes $\{ a _ { 1 } , a _ { 2 } , . . . , a _ { k } \}$ Each attribute $a _ { j }$ is formally defined as a tuple $a _ { j } = \langle n a m e$ data type, default, prompt⟩. Figure 5 shows examples of the generated attribute schema for the “Candles” ProductType.

2) Multi-Modal Execution:: We employ a multi-modal function $f _ { L L M }$ that maps an item’s raw data $R _ { i }$ (text + images) to a feature vector $V _ { i }$ based on the schema $S _ { p t } \mathrm { : }$ $V _ { i } = f _ { L L M } ( R _ { i } , S _ { p t } )$

This multi-modal capability is critical for retail catalogs, where key specifications (e.g., “4-pack” or “organic”) often ap pear only on packaging images. Figure 6 depicts this workflow, where the model ingests item context and schema prompts to generate structured values. Unlike traditional methods requiring thousands of specific NER models [16], this approach utilizes a single generalized agent to extract arbitrary attributes defined by the dynamic schema, resolving the scalability bottleneck.

Because traditional pre-sanitization (e.g., regex) fails on missing or unstructured catalog data, this Multi-Modal Extractor acts as the sanitization engine itself. Initial experiments with smaller open-weight models (e.g., Nemotron VLM 2B/8B/12B) failed to achieve production accuracy without large-scale labeled data tailored to our complex schema, necessitating our reliance on frontier models.

## E. Data Standardization and Normalization

Raw extraction yields structured data, but inconsistencies impede grouping. We employ a post-processing pipeline Ψ to normalize the extracted feature vector V .

• Numerical Standardization (UOM): For attributes suf fixed with uom (e.g., weight), a conversion function

![](images/ef66e7b1ab72ea2dbf39e5ca5726431b4b78ac3194d2e4f4a7d2184836664ee8.jpg)  
Fig. 6. Implemented methodology for attribute extraction

$\phi _ { u o m }$ transforms the raw value $v _ { r a w }$ and unit $u _ { r a w }$ into a normalized value $v _ { n o r m }$ in a standard base unit $u _ { b a s e } .$

$$
v _ { n o r m } = \phi _ { u o m } ( v _ { r a w } , u _ { r a w } ) \to u _ { b a s e }
$$

For example, $( 1 2 , 1 6 \mathrm { s } ) \  \ 1 9 2 \ ( \mathrm { o z } )$ . We enforce a strict 5% numerical tolerance to absorb floating-point inaccuracies during UOM conversions, preventing the artificial separation of identical items.

• Categorical Semantic Clustering: To resolve semantic fragmentation (e.g., “Soy” vs. “Soy Blend”), an LLMbased clustering function $C _ { s e m }$ maps a raw value $u _ { j }$ from the set of unique values U to a canonical term u<sub>canon</sub>, preventing artificial fragmentation: $u _ { c a n o n } = C _ { s e m } ( u _ { j } | U )$

• Brand Normalization: To strictly govern brand identity, a hybrid function $R _ { b r a n d }$ reconciles the extracted brand $\boldsymbol { B _ { e x t } }$ with the catalog master $B _ { i n t }$ via sub-string mapping and a safety rail based on Jaccard Similarity (J) and brand frequency $( b _ { f } )$

$$
B _ { f i n a l } = \left\{ \begin{array} { l l } { B _ { i n t } } & { \mathrm { i f ~ } J ( B _ { e x t } , B _ { i n t } ) = 0 \land b _ { f } ( B _ { i n t } ) \geq 5 } \\ { B _ { e x t } } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

Here, $J ( B _ { e x t } , B _ { i n t } )$ denotes the Jaccard Similarity between the extracted and internal brand strings, and $b _ { f } ( B _ { i n t } )$ represents the frequency count of the internal brand within the catalog. This prioritizes the system of record $B _ { i n t }$ when the extracted brand is disjoint from a well-established entity, preventing hallucinated overrides.

## F. Adaptive Taxonomy Construction

Grouping is scoped at the brand level to respect attribute applicability, as brands within a ProductType often employ distinct conventions (e.g., weight vs. burn time).

1) Feature Selection & Pruning: We apply heuristic filters on the normalized vector $V _ { i }$ to retain discriminative features:

• Sparsity & Imputation: Through iterative deployment and experimentation across diverse catalog samples, we established operational heuristics to drop attributes with > 65% nullity $\mathrm { o r } > 7 0 \%$ cardinality; retaining sparser features consistently caused artificial fragmentation of product lines. Gaps in retained attributes are imputed using $S _ { p t }$ defaults.

• Numerical Binning: To handle numerical variance, we apply ϵ neighborhood clustering. Dictated by strict business requirements, values within a 5% tolerance (e.g., 12.0 oz vs 12.3 oz) are binned into a single discrete value $v _ { b i n }$

2) Hierarchical Grouping Logic: Let $A _ { c a t }$ be the set of categorical attributes and $A _ { n u m }$ be the set of numerical/volume attributes.

• Line Generation: Items are grouped by the complete set $A _ { t o t a l } = A _ { c a t } \cup A _ { n u m }$ . Each unique combination forms a Line $( L _ { k } )$ assigned a persistent UUID:

$$
L _ { k } = \{ i \in C \ | \ V _ { i } ( A _ { t o t a l } ) = \mathrm { c o n s t } \}
$$

![](images/9406d656f8c49ce2f407204e763ea8c9cd01ff228d336fd6d47434e09b006d52.jpg)  
Fig. 7. Implemented methodology for each ProductType

• Ladder Generation: Relaxing constraints by excluding $A _ { n u m }$ , we group by $A _ { c a t }$ to cluster Lines into Ladders $( M _ { j } )$ , linking items differing only by size or pack quantity:

$$
M _ { j } = \{ i \in C \mid V _ { i } ( A _ { c a t } ) = \mathrm { c o n s t } \}
$$

Consequently, a ladder is defined as the union of its constituent Lines: $\textstyle M _ { j } = \bigcup _ { k } L _ { k }$ . This logic automatically structures the catalog, placing functionally equivalent items into Lines and connecting them via volume-based Ladders.

## G. Human-in-the-Loop Feedback Mechanism

While the multi-agent framework provides a robust baseline, the inherent stochasticity of LLMs and the specificity of business strategy (e.g., pricing soft drinks at the parent brand level) necessitate human intervention. We incorporate a Human-in-the-Loop (HITL) mechanism to capture merchant corrections as high-quality labeled signals. This feedback updates a ProductType specific contextual memory, $P _ { c o n t e x t } ,$ designed to drive future pipeline refinements.

• Strategic Alignment: Agents 1 and 2 will learn to prioritize or ignore attributes based on merchant preferences.

• Schema Refinement: Agent 3 will update its merging logic to respect business-specific naming conventions and optimize extraction prompts.

As $P _ { c o n t e x t }$ accumulates, agents recall these corrections to ensure subsequent runs align with established business logic, allowing the system to progressively evolve into a domainadapted expert. While this HITL architecture successfully captures immediate strategic intent, quantifying the system’s long-term convergence toward expert-level behavior remains an ongoing area of empirical validation within our deployed environment.

## IV. DEPLOYMENT AT SCALE

Deployed on a distributed compute cluster for horizontal scalability and fault tolerance (Figure 8), the framework is managed by a refresh orchestrator supporting two execution modes: full refresh (recomputing attribute schema) and delta refresh (reusing persisted schema for steady-state updates).

## A. Distributed Pipeline Architecture

Decomposing the pipeline by Department → ProductType enables independent failure domains. To balance reasoning and cost at scale, we employ a multimodel routing strategy using secure, enterprise-managed deployments of frontier commercial multi-modal LLMs:

• Multi-Agent Attribute Identifier: Coordinates the three agents. Reasoning-heavy tasks (schema generation, conflict resolution) are routed to a proprietary, enterprisemanaged frontier LLM (comparable in scale and capability $\mathrm { t o } > 1 0 0 B$ parameter instruction-tuned “thinking” models). A low-latency RDBMS implements persistent memory $( P _ { c o n t e x t } )$ , using deterministic SQL queries to retrieve and inject historical merchant rules into prompts. To prevent context overflow, this injected history is strictly truncated to feedback collected since the last full refresh. Due to enterprise confidentiality, exact commercial model names cannot be disclosed. However, all agents utilize state-of-the-art frontier models with the temperature strictly set at 0 to maximize determinism.

• Multi-Modal Attribute Extractor: Executes extraction function $f _ { L L M }$ . To process millions of items, extraction is routed to a highly efficient, cost-optimized multi-modal model from the same family (comparable in scale to 7B-13B parameter Vision-Language Models), designed for high-throughput, low-latency execution.

• Attribute Cleaner & Standardizer: Normalizes values via pipeline Ψ. A SQL-based semantic cache stores synonym mappings $( \mathrm { e . g . , ^ { * } S o y ^ { 9 }  ^ { * } S o y \ W a x ^ { 9 } } )$ to optimize latency and reduce redundant LLM calls during delta refreshes.

• Lines & Ladders Clustering: Transforms standardized vectors into Lines $( L _ { k } )$ and Ladders (M ) using hierarchical logic (Section 3.6.2).

## B. Asynchronous Feedback Loop

Human validation utilizes an event-driven architecture. Merchant UI corrections emit events to a distributed message bus,

![](images/7f6e441fb69f740debb58ac21526b7987aa96ec235b0f01f9ac11e4dfc94119c.jpg)  
Fig. 8. Lines & Ladder System Design

where an Intelligent Feedback Classifier routes them by error category:

• Missing Attributes → Triggers agent refinement.

• Extraction Errors → Updates Extractor prompts.

• Standardization Errors → Updates SQL Cache.

• Strategic Decisions → Updates memory P<sub>context</sub>.

## C. Performance and Scalability

We processed ∼1,700 ProductTypes (∼1 million items) in 30 minutes using 10 worker nodes.

• Throughput: Distributed implementation enables linear worker node scaling within available LLM quotas.

• Cost Optimization: LLM inference dominates spend (∼4K tokens/item). Micro-batching and sampling during identification achieve significant cost reductions.

• Observability: Structured logging enables partial reruns and ProductType-level regression analysis.

This architecture maintains strict catalog freshness SLAs within a sustainable cost envelope.

## V. EXPERIMENTAL EVALUATION

To validate the framework, we evaluate architectural ablation, structural accuracy, and post-launch business impact.

## A. Architecture Ablation

To justify the multi-agent design, we conducted an ablation study across 14 categories. We specifically evaluate zeroshot LLM architectures rather than traditional supervised Named Entity Recognition (NER) models (e.g., fine-tuned BERT). Retail catalogs require dynamic schema generation (e.g., extracting “wax type” for candles but “motor power” for blenders); therefore, training and maintaining static models for thousands of evolving ProductTypes is operationally infeasible at enterprise scale. Furthermore, traditional dense embedding-based clustering approaches are inadequate for this task; they rely on semantic language similarity rather than the strict physical attribute equivalence and multi-modal business context required for pricing taxonomies. Consequently, our baseline comparison focuses on a Single-Agent LLM, a 2- Agent system (Similarity + Variance), and our proposed 3- Agent system.

TABLE II  
COMPARISON ACROSS AGENT ARCHITECTURES
<table><tr><td rowspan="2">ProductType</td><td colspan="3">Lines (F1)</td><td colspan="3">Ladders (F1)</td></tr><tr><td>1-Agt</td><td>2-Agt</td><td>3-Agt</td><td>1-Agt</td><td>2-Agt</td><td>3-Agt</td></tr><tr><td>Deodorants &amp; Antiperspirants</td><td>0.63</td><td>0.90</td><td>1.00</td><td>0.63</td><td>0.90</td><td>1.00</td></tr><tr><td>Prepared &amp; Packaged Soups</td><td>0.60</td><td>0.92</td><td>0.96</td><td>0.60</td><td>0.79</td><td>0.92</td></tr><tr><td>Juices</td><td>0.66</td><td>0.69</td><td>0.90</td><td>0.69</td><td>0.69</td><td>0.92</td></tr><tr><td>Ice Cream Bars, Cones</td><td>0.50</td><td>0.70</td><td>0.88</td><td>0.73</td><td>0.70</td><td>0.95</td></tr><tr><td>Over-the-Counter Medicine</td><td>0.70</td><td>0.75</td><td>0.87</td><td>0.72</td><td>0.70</td><td>0.75</td></tr><tr><td>Bath Soaps</td><td>0.69</td><td>0.88</td><td>0.87</td><td>0.77</td><td>0.90</td><td>0.93</td></tr><tr><td>Packaged Meals</td><td>0.56</td><td>0.75</td><td>0.87</td><td>0.58</td><td>0.85</td><td>0.91</td></tr><tr><td>Period Panties</td><td>0.93</td><td>0.84</td><td>0.84</td><td>0.74</td><td>0.78</td><td>0.78</td></tr><tr><td>Sandwiches, Filled Rolls</td><td>0.73</td><td>0.80</td><td>0.82</td><td>0.82</td><td>0.84</td><td>0.95</td></tr><tr><td>Toilet Paper</td><td>0.54</td><td>0.52</td><td>0.79</td><td>0.67</td><td>0.31</td><td>0.76</td></tr><tr><td>Laundry Detergents</td><td>0.54</td><td>0.76</td><td>0.75</td><td>0.66</td><td>0.68</td><td>0.63</td></tr><tr><td>Flushable Cleansing Cloths</td><td>0.74</td><td>0.93</td><td>0.71</td><td>0.91</td><td>0.53</td><td>0.53</td></tr><tr><td>Snack Crackers</td><td>0.59</td><td>0.59</td><td>0.68</td><td>0.64</td><td>0.59</td><td>0.77</td></tr><tr><td>Cat Food</td><td>0.53</td><td>0.74</td><td>0.64</td><td>0.55</td><td>0.61</td><td>0.66</td></tr><tr><td>Average</td><td>0.64</td><td>0.77</td><td>0.83</td><td>0.69</td><td>0.70</td><td>0.82</td></tr></table>

As Table II shows, a single LLM tasked with simultaneous similarity and variance discovery suffers severe cognitive overload (0.64 Lines F1). While a 2-Agent system improves performance (0.77 F1), utilizing a Synthesis Agent (3-Agt) to resolve semantic overlaps was strictly necessary to achieve the > 80% accuracy required for production.

## B. Quantitative Manual Evaluation

To measure “Assignment Accuracy” (the percentage of items correctly grouped without requiring manual edits) in unstructured General Merchandise, domain experts evaluated 1000 randomly sampled items across 11 diverse ProductTypes. Because evaluating placement requires reviewing multi-modal data and historical pricing strategies, this represents tens of hours of specialized annotation. As shown in Table III, the framework achieved an 80.0% weighted average accuracy.

Qualitative feedback reveals three distinct categories of error:

• Technical Granularity: The model occasionally missed specific technical variants, such as “Ultrasonic” vs. “Evaporative” (Humidifiers), “Down Rod” vs. “Hugger” (Ceiling Fans), or “Weed-Control” vs. “Feeding” (Fertilizers).

• Aesthetic & Subjective Nuance: Differentiation for Lamps or Duffel Bags often relies on intangible qualities (e.g., “silhouette elegance” or “material quality”) requiring tacit domain knowledge, underscoring the need for HITL merchant intuition.

• Data Completeness & Complexity: Errors in Sewing Machines correlated with sparse descriptions, while Office Boards struggled with complex bundle combinations (board + markers), confirming performance is bounded by catalog data quality.

Synthesizing these observations, in most failure cases, the grouping logic was directionally correct but missed a single differentiating attribute. This indicates the system does not hallucinate random groups, but rather is just one feature away from perfect alignment.

## C. Comparative Evaluation

In Food & Consumables, we utilized merchant-generated pricing structures across 14 ProductTypes as a directional benchmark, noting that merchants often construct lines based on ad-hoc strategies rather than strict attribute equivalence. We pre-processed reference data to remove statistical outliers.

• Matching Logic & Metrics: Because our algorithm generates novel UUIDs, direct mapping to legacy IDs is infeasible. We utilized a maximum F1 score optimization approach (similar to bipartite matching) to align generated lines with merchant lines, calculating precision and recall.

• Results & System Guardrails: Table IV shows an average Line Precision of 98% and Line Recall of 75% (Ladders: 92% Precision, 81% Recall). This highprecision/low-recall profile is an intentional guardrail. High-recall groupings risk applying price changes to unrelated items (e.g., incorrectly grouping 11 distinct “Prego Sauces”). Our system conservatively groups only the 4 identical Alfredo sauces, safely protecting the customer experience.

As detailed in Table IV, the framework consistently delivers near-perfect precision alongside a lower, category-dependent recall. This dynamic highlights a fundamental divergence between algorithmic strictness and human strategic intent:

TABLE III  
MANUAL AUDIT RESULTS AND QUALITATIVE FEEDBACK
<table><tr><td>ProductType</td><td>Sample Size</td><td>Accuracy</td></tr><tr><td>Bean Bags</td><td>100</td><td>100%</td></tr><tr><td>Face Mirrors</td><td>38</td><td>100%</td></tr><tr><td>Air Conditioners</td><td>100</td><td>88%</td></tr><tr><td>Surge Protectors</td><td>100</td><td>86%</td></tr><tr><td>Humidifiers</td><td>100</td><td>84%</td></tr><tr><td>Fertilizers</td><td>100</td><td>82%</td></tr><tr><td>Sewing Machines</td><td>100</td><td>80%</td></tr><tr><td>Ceiling Fans</td><td>100</td><td>72%</td></tr><tr><td>Lamps</td><td>100</td><td>69%</td></tr><tr><td>Office Boards</td><td>62</td><td>68%</td></tr><tr><td>Duffel Bags</td><td>100</td><td>61%</td></tr><tr><td>Weighted Avg</td><td>1000</td><td>80%</td></tr></table>

TABLE IV  
COMPARATIVE PERFORMANCE AGAINST MERCHANTS REFERENCE STRUCTURES
<table><tr><td rowspan="2">ProductType</td><td colspan="3">Lines</td><td colspan="3">Ladders</td></tr><tr><td>F1</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Prec.</td><td>Rec.</td></tr><tr><td>Deodorants &amp; Antiperspirants</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>Prepared &amp; Packaged Śoups</td><td>0.96</td><td>1.00</td><td>0.93</td><td>0.92</td><td>0.93</td><td>0.93</td></tr><tr><td>Juiċes</td><td>0.90</td><td>1.00</td><td>0.85</td><td>0.92</td><td>1.00</td><td>0.88</td></tr><tr><td>Ice Cream Bars, Cones</td><td>0.88</td><td>1.00</td><td>0.83</td><td>0.95</td><td>1.00</td><td>0.92</td></tr><tr><td>Over-the-Counter Medicines</td><td>0.87</td><td>0.99</td><td>0.81</td><td>0.75</td><td>0.73</td><td>0.91</td></tr><tr><td>Bath Soaps</td><td>0.87</td><td>1.00</td><td>0.78</td><td>0.93</td><td>1.00</td><td>0.89</td></tr><tr><td>Packaged Meals</td><td>0.87</td><td>1.00</td><td>0.79</td><td>0.91</td><td>0.88</td><td>0.97</td></tr><tr><td>Period Panties</td><td>0.84</td><td>1.00</td><td>0.75</td><td>0.78</td><td>1.00</td><td>0.69</td></tr><tr><td>Sandwiches, Filled Rolls</td><td>0.82</td><td>0.95</td><td>0.76</td><td>0.95</td><td>1.00</td><td>0.91</td></tr><tr><td>Toilet Paper</td><td>0.79</td><td>0.97</td><td>0.71</td><td>0.76</td><td>0.82</td><td>0.78</td></tr><tr><td>Laundry Detergents</td><td>0.75</td><td>1.00</td><td>0.63</td><td>0.63</td><td>0.87</td><td>0.59</td></tr><tr><td>Flushable Cleansing Cloths</td><td>0.71</td><td>0.89</td><td>0.67</td><td>0.53</td><td>0.72</td><td>0.65</td></tr><tr><td>Snack Crackers</td><td>0.68</td><td>1.00</td><td>0.54</td><td>0.77</td><td>1.00</td><td>0.65</td></tr><tr><td>Cat Food</td><td>0.64</td><td>0.99</td><td>0.50</td><td>0.66</td><td>0.99</td><td>0.54</td></tr><tr><td>Average</td><td>0.83</td><td>0.98</td><td>0.75</td><td>0.82</td><td>0.92</td><td>0.81</td></tr></table>

• High Precision: The agent achieves near-perfect precision, confirming it groups physically identical items without hallucinating relationships, ensuring safe deployment.

• The Recall Gap: Lower recall indicates the agent frequently over-splits lines compared to the merchant reference due to lacking strategic context. This explains the F1 variance: highly standardized categories (e.g., Deodorants, F1=1.00) align perfectly, whereas categories with subjective, marketing-driven descriptions (e.g., Cat Food, F1=0.64) rely on nuanced terms like “Ocean Whitefish Pate” vs. “Flaked Tuna in Sauce”. Merchants frequently over-group these items based on promotional strategies or broad “flavor families” rather than strict attribute equivalence. Our algorithm’s strict physical grouping naturally splits these broad merchant groups, resulting in lower recall.

This divergence highlights the necessity of the HITL workflow (Section 3.7). While the agent provides a safe, highprecision cold-start state, continuous merchant feedback enables the persistent memory to learn these strategic nuances, allowing future iterations to automatically align with the merchant’s preferred granularity.

## D. Post-Launch Business Impact

This fully launched production system processes tens of thousands of active items, generating structured Lines and Ladders for > 90% of the previously un-managed catalog for the first time. Beyond creating this entirely new data foundation, end-to-end telemetry tracking merchant UI workflows over a 13-week period indicates an 86.1% reduction in time-ontask for existing workflows, calculated via: New Hours = Old Hours × Remaining Workload × Remaining Time.

• Workload Reduction: Automated extraction reduces manual catalog cleanup by 35 − 40% (Remaining Workload ≈ 0.625).

• Velocity Increase: Automated grouping and UI review is 4–5x faster than manual creation (Remaining Time ≈ 0.222).

Applying this mid-case scenario $( 0 . 6 2 5 \times 0 . 2 2 2 = 0 . 1 3 8 )$ the workflow requires only 13.9% of original hours, enabling merchants to focus entirely on high-level pricing strategy.

## VI. LIMITATIONS AND FUTURE WORK

• Limitations: Despite a temperature of 0, LLM nondeterminism complicates strict reproducibility. The system is sensitive to sampling bias, where poor catalog quality can skew attribute discovery. Furthermore, compute costs currently restrict the framework to batch processing. In addition, reliance on specific LLM families necessitates re-validation upon model updates to mitigate hallucination risks. Finally, while we argue that traditional embeddingbased clustering lacks the contextual reasoning required for this task, future work will include empirical benchmarking against these non-LLM baselines to rigorously quantify the multi-agent system’s added value.

• Future Directions: We plan to refine anchoring from ProductType to the brand level to capture niche features and utilize HITL feedback to stabilize nondeterministic outputs. We also aim to conduct longitudinal studies measuring the system’s convergence to merchant strategy over multiple HITL feedback cycles, alongside statistical significance testing and sensitivity analysis on heuristic thresholds. To resolve data sparsity in niche categories (∼5%), we will implement cross-category transfer learning. Finally, to enable real-time inference, we will adopt a teacher-student architecture, distilling large models into Small Language Models (SLMs) for low-latency execution.

## VII. CONCLUSION

This paper presented “Lines and Ladders”, a multi-agent framework that automates retail price governance by decomposing taxonomy construction into Similarity and Variance discovery. Synergizing LLM-based extraction with hierarchical grouping, the system structures heterogeneous data into coherent pricing tiers.

Evaluations confirm > 80% accuracy in General Merchandise and > 90% precision in Food categories, with 75% recall targeted for HITL optimization. Deployed in production, the system drastically reduces manual effort, enabling merchants to focus on high-level strategy. Beyond efficiency, this taxonomy establishes the structural foundation for autonomous pricing and anomaly detection. Ultimately, this work demonstrates that multi-agent architectures are highly viable for industrial-scale governance, offering a framework broadly applicable to other heterogeneous domains like industrial supply chains and online marketplaces.

## REFERENCES

[1] T. T. Nagle and G. Muller, ¨ The Strategy and Tactics of Pricing: A Guide to Growing More Profitably, 6th ed. Routledge, 2016. [Online]. Available: https://www.academia.edu/39005285/THE STRATEGY AND TACTICS OF PRICING

[2] K. B. Monroe, Pricing: Making Profitable Decisions, 3rd ed. McGraw-Hill/Irwin, 2003, ch. 15, product-line pricing.

[3] S. Basroy, M. K. Mantrala, and R. G. Walters, “The impact of category management on retailer prices and performance,” Journal of Retailing, pp. 17–18, 2001. [Online]. Available: https: //journals.sagepub.com/doi/10.1509/jmkg.65.4.16.18382?utm source=re searchgate.net&utm medium=article

[4] L. Xia, K. B. Monroe, and J. L. Cox, “The price is unfair! a conceptual framework of price fairness perceptions,” Journal of Marketing, p. 9, 2004. [Online]. Available: https://www.researchgate.net/publication/228 590264 The Price Is Unfair A Conceptual Framework of Price Fai rness Perceptions

[5] D. R. Lehmann, H. Yuan, A. Krishna, and R. Briesch, “A meta-analysis of the impact of price presentation on perceived savings,” Journal of Retailing, vol. 78, no. 2, pp. 101–118, 2002.

[6] J. S. Park, J. C. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, “Generative agents: Interactive simulacra of human behavior,” in Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, 2023, pp. 1–22. [Online]. Available: https://arxiv.org/pdf/2304.03442

[7] P. S. Fader and B. G. Hardie, “Modeling consumer choice among skus,” Journal of Marketing Research, vol. 33, pp. 442–452, 1996. [Online]. Available: https://doi.org/10.2307/3152215

[8] J. Dean, “Problems of product-line pricing,” Journal of Marketing, vol. 14, no. 4, pp. 518–528, 1950.

[9] J. Guiltinan, “Progress and challenges in product line pricing,” Journal of Product Innovation Management, vol. 28, no. 5, pp. 744–756, 2011.

[10] I. M. Dobbs, “When does tiered wholesale pricing create an incentive to reduce retail prices?” Applied Economics Letters, vol. 23, no. 11, pp. 777–780, 2016.

[11] R. Mohammed, The art of pricing. New York: Crown Business, 2005.

[12] T. Data, “Pricing ladders: 5 pointers for outstanding performance,” TGN Data, accessed on 2025-11-18. [Online]. Available: https: //tgndata.com/pricing-ladders-5-pointers-for-outstanding-performance/

[13] PriceBeam, “Price ladders in emerging markets: 4 steps to higher margins,” PriceBeam Blog, Oct. 2017. [Online]. Available: https://blog.pricebeam .com/price-ladders-in-emerging-markets-4-steps-to-higher-margins

[14] M. Draganska and D. C. Jain, “Consumer preferences and product-line pricing strategies: An empirical analysis,” Marketing science, vol. 25, no. 2, pp. 164–174, 2006.

[15] S. Mudgal, H. Li, T. Rekatsinas, A. Doan, Y. Park, G. Krishnan, R. Deep, E. Arcaute, and V. Raghavendra, “Deep learning for entity matching: A design space exploration,” in Proceedings of the 2018 international conference on management of data, 2018, pp. 19–34.

[16] W.-T. Chen, K. Shinzato, N. Yoshinaga, and Y. Xia, “Does named entity recognition truly not scale up to real-world product attribute extraction?” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track. Singapore: Association for Computational Linguistics, Dec. 2023. [Online]. Available: https://aclanthology.org/2023.emnlp-industry.16/