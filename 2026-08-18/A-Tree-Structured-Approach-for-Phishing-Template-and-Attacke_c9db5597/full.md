# A Tree-Structured Approach for Phishing Template and Attacker Attribution Analysis

Unai Agirre\*, Imanol Jerico\*, Felipe Castano\*†, Andrea Venturi\*, Francesco Zola\*

Digital Security Department, Vicomtech (BRTA), Donostia / San Sebastian, Spain

†University of León, León, Spain

{uagirre, ijerico, fcastano, aventuri, fzola} @vicomtech.org

Abstract—Phishing remains a persistent and evolving cybersecurity threat, with attack volumes reaching record levels. This growth is driven by the industrialization of phishing through widely available phishing kits and reusable templates, which enable cybercriminals to rapidly generate and deploy large numbers of fraudulent webpages. Although surface-level attributes may differ across these websites, their underlying structures often exhibit significant similarities. However, most existing defenses rely on reactive blocklists or supervised classification models that focus on individual phishing instances, limiting their ability to identify structural reuse and detect coordinated phishing campaigns. To address this limitation, this study investigates whether HTML structure can serve as a robust fingerprint for identifying phishing template reuse. We model webpages as Document Object Model (DOM) trees and extract structural features, optionally enriched with HTML tag-based content information. These representations are then clustered using unsupervised learning methods to group structurally similar webpages. Three clustering algorithms are evaluated and compared, while also analyzing how the depth of the extracted DOM-tree affects cluster formation and overall clustering performance. Finally, cluster quality is also evaluated both quantitatively and qualitatively, including a novel level-wise Jaccard Distance Score and manual inspection supported by visualization tools. Results demonstrate that structural representations of webpages can effectively reveal hidden similarities across phishing sites, enabling the detection of emerging and zero-day templates and supporting the analysis of coordinated phishing threats.

Index Terms—Phishing Template, Structural Analysis, Attacker Attribution, Clustering, DOM analysis, Tree extraction

## I. INTRODUCTION

Phishing remains a persistent and evolving threat within the cybersecurity landscape, maintaining its efficacy despite user awareness. According to the Anti-Phishing Working Group (APWG) [1], attack volumes reached record levels in early 2025, with over 1 million unique phishing attacks detected in the first quarter alone, the highest volume observed since late 2023. This resurgence is characterized by a high degree of sector-specific targeting; the Software as a Service (SaaS) and Webmail category remains the most frequently attacked sector (17.6%), while the financial and payment industries combined account for over 30.9% of all observed incidents. These numbers underscore a broader trend: phishing is no longer a collection of isolated fraudulent events but a massive, industrialized operation that continues to scale across global infrastructures [2].

Traditional defense mechanisms primarily focus on identifying individual malicious instances through reactive blocklists or supervised machine learning models [3]. While these approaches achieve high performance in binary classification [4], [5], they remain limited in their ability to provide adversarial insight, as they fail to capture the broader context of attack infrastructures or the structural signatures left by organized actors. This limitation is increasingly relevant, as the proliferation of phishing kits, reusable templates, and web components [6], [7], [8] enables adversaries to deploy numerous fraudulent pages rapidly with minimal effort [2]. In these cases, phishing instances may vary in their domain names or superficial visual elements [9], but they frequently retain a congruent internal architecture.

For this reason, the primary objective of this study is to evaluate whether the HTML structure of a webpage serves as a sufficiently robust fingerprint for identifying similarities among malicious sites, thereby providing a stable indicator of phishing structural logic and enabling the detection of systematic template reuse across malicious websites.

Specifically, to do this, the proposed methodology uses Document Object Model (DOM) information to build a parent-child tree. Then, tree properties (structural information) are used exclusively and combined with content-based data (HTML tags within the DOM) to feed unsupervised learning techniques that cluster similar samples without the need for prior labeling. Finally, once groups of similar trees are generated, the methodology proposes a quantitative and qualitative validation. The first is achieved by introducing a new metric based on level-wise Jaccard Distance Score, while the second is performed manually using dedicated graph/tree visualization tools.

By focusing on revealing structures rather than isolated events, this framework offers the adaptability necessary to identify emergent, zero-day phishing templates that evolve continuously to evade static or supervised detection rules. Once identified, these templates form the foundation for the comprehensive characterization and attribution of coordinated phishing campaigns.

Summarizing, the main contributions of this study are:

• Validate if tree-based representation of phishing webpages derived from DOM can be used for detecting structural templates;

• Compare the performance of three different clustering methodologies when tree properties (structural) are used solely or combined with content-based information (HTML tags);

• Validate how the depth of the tree, i.e., the number of nested levels, affects clustering performance;

• Define and validate a new level-wise Jaccard Distance Score to evaluate clustering performance (quantitative validation).

The remainder of this article is structured as follows. Section II provides the theoretical background and reviews the state-of-the-art relevant to the proposed approach. Section III outlines the methodology, while Section IV details the dataset, experiments, and model configurations. Section V reports the results, while Section VI presents qualitative insights, limitations, and ethical concerns. Finally, Section VII concludes the study and suggests potential avenues for future research.

## II. BACKGROUND

## A. Clustering

The goal of this work is to detect and group phishing templates that exhibit similar structural patterns. To achieve this, clustering algorithms are used as the main analytical tool, since they enable the identification of latent relationships between webpages without requiring prior labels.

Among the different clustering paradigms, this study focuses on density-based and hierarchical approaches. Densitybased methods are suitable for phishing analysis because they can identify groups with arbitrary shapes while also separating isolated samples that may correspond to structural outliers. Hierarchical methods provide a complementary perspective by organizing samples according to nested similarity relationships. Together, these two families of methods make it possible to capture diverse similarity patterns, which is particularly important in a phishing ecosystem characterized by template reuse, campaign variation, and heterogeneous construction strategies [10], [11].

Among density-based methods, DBSCAN (Density-Based Spatial Clustering of Applications with Noise) [12] forms clusters by connecting neighbouring samples that satisfy two conditions: a maximum neighbourhood distance € and a minimum number of points min\_samples to form a cluster. Its main advantages are that it does not require the number of clusters to be specified in advance and that it can identify noise or isolated samples. However, its performance is sensitive to the choice of ε and may decrease when clusters have different densities. OPTICS (Ordering Points To Identify the Clustering Structure) [13] addresses this limitation by producing an ordering of the samples that reflects their density-based connectivity. This makes it more suitable for data with varying density levels, although the resulting ordering must still be interpreted or converted into final clusters.

Agglomerative Hierarchical Clustering (AHC) [14] follows a bottom-up strategy, where each sample initially forms its own cluster and the most similar clusters are progressively merged according to a linkage criterion. This process produces a hierarchy of nested groups, allowing the data to be examined at different levels of granularity. AHC does not necessarily require a predefined number of clusters, depending on the chosen stopping criterion. However, once two clusters are merged, the decision cannot be reversed, and the final structure can be sensitive to the selected distance measure and linkage criterion.

## B. Related Work

Phishing Analysis is divided into two complementary groups [15]. On the one hand, the first kind of approaches treats phishing detection as a binary classification problem, prioritizing the immediate identification and mitigation of malicious instances to protect the end-user. On the other hand, the second approach focuses on post-detection analysis, using confirmed phishing samples to extract structural patterns, common infrastructures, and forensic markers. This second perspective shifts the objective from simple filtering to the characterization of phishing templates and the potential attribution of adversarial groups. The following sections detail the technical evolution of both methodologies.

1) Phishing Detection as Binary Classification: This research group focuses on the development of computationally efficient and robust architectures to distinguish between legitimate and malicious entities. One of the highlight approaches is that proposed by Otieno et al. [16]. In this approach, the authors use semantic analysis to eliminate manual feature engineering, demonstrating that bidirectional Transformers (BERT) capture contextual representations of URL strings, achieving 96% accuracy. To address the trade-off between efficiency and robustness, hybrid ensemble systems have been proposed to integrate BERT-derived vectors with stacked classifiers and technical parameters like SSL validity, achieving accuracies up to 98.6% [17], [18].

Beyond URL analysis, Asiri et al. [4] introduced Phish-Transformer, an approach that examines the HTML web page source as complementary information to identify sophisticated threats. The authors use CNNs and Transformers to analyze embedded URL links within the HTML content, addressing the limitations of traditional systems that rely solely on the primary URL. Experimental results on a balanced dataset of 50,000 samples demonstrated that PhishTransformer achieves an average accuracy of 98.9% and an F1-score of 99%. Later, Jiang et al. [5] developed D-PhishNet, employing Graph Attention Networks (GATs) to fuse URL and HTML features dynamically. The authors introduced the Dual-Branch Alignment Mixer (DBA-CMixer), which uses gated fusion and channel attention to integrate structural URL data with latent semantic features extracted via a pre-trained BERT module to address the “semantic gap"inherent in multi-feature fusion. The authors report an accuracy of up to 99.30%.

While binary classification models achieve high performance in isolating malicious samples, they remain limited in their ability to provide adversarial insight. By focusing strictly on the categorization of isolated events, these methods fail to capture the broader context of the attack infrastructure or the underlying relationships between samples. Given that our research focus is the analysis of patterns within attacks to detect authorship and phishing templates, traditional binary models are insufficient for identifying the systemic signatures left by organized actors.

2) Phishing Template Analysis: Another group focuses on analyzing confirmed phishing samples to identify templates and shared patterns. Nguyen et al. [19] proposed identifying structural clones of legitimate pages by calculating similarity between DOM trees using genetic algorithms, achieving a 0% false-negative rate at specific thresholds. While achieving high accuracy at a threshold of δ = 0.6, the method remains sensitive to obfuscation and exhibits a significant trade-off between false positives and negatives depending on threshold calibration. Unlike this approach, which seeks to identify a phishing site by its proximity to a legitimate target, our research shifts the focus from victim-mimicry detection toward uncovering latent structural links and shared indicators across diverse phishing attacks to characterize broader patterns.

To bridge the gap between active attacks and their generative sources, Castaño et al. [7] introduced PhiKitA, a dataset specifically designed to link phishing kits directly to their deployed website counterparts. Their analysis employed graph representation of the HTML DOM, MD5 hashing, and structural fingerprints to identify familiarity among samples, successfully uncovering a coordinated campaign involving 172 distinct attacks against a single financial institution. The graphbased approach achieved high performance in binary detection tasks (92.5% accuracy). While the findings demonstrated that structural HTML data is critical for tracing the lineage of phishing operations, the reliance on known kit repositories highlights a dependency that limits detection to documented sources. In contrast, the methodology proposed in this work uses unsupervised clustering of web pages, which does not require a prior sample of detected phishing templates. By operating independently of kit-specific metadata, this approach enables the identification of zero-day samples and emergent templates that remain undocumented in existing repositories.

As mentioned earlier, the nature of templates detection involves extracting patterns across heterogeneous file types and different attack vectors, which interferes with the use of supervised learning methods due to the lack of annotated datasets. Consequently, unsupervised methodologies are required to discover latent relationships without prior labeling. Althobaiti et al. [20] addressed this by evaluating Mean Shift and DBSCAN algorithms to group corporate email threats, demonstrating that Mean Shift achieves 82% precision in assigning samples to representative clusters based on URL and origin metadata. Although this approach uses email corpora rather than raw HTML, the underlying methodology aligns with current research objectives; the unsupervised nature of the framework provides critical adaptability to emergent templates that constantly evolve to evade static detection rules.

## III. METHODOLOGY

The main goal of this methodology is to analyse and validate structural similarity among phishing webpages. In this sense, the methodology leverages DOM information to build a treebased representation and then extracts structural and contentbased features to assess similarities among phishing webpages. In this way, the process enables the identification of pages that are likely (i) deployed by the same attacker or (ii) created using the same template or phishing kit.

The proposed methodology comprises four main steps: data preprocessing, tree construction, feature extraction & clustering, and validation.

## A. Phase 1: Data Preprocessing

The experimental evaluation of the proposed methodology uses the dataset developed by Aljofey et al. [21], specifically the subset identified as DS-2. This dataset was selected for its inclusion of full HTML source code, which is an essential requirement for the tree-based representation described in Phase 2. Collected during 2023 and published in 2025, the corpus captures current evasion tactics and modern web design patterns, thereby serving as a benchmark for structural analysis. Phishing samples were collected from PhishStats and validated between July 15 and August 9, 2023. Legitimate instances were sourced from Alexa top-ranked websites as of November 2023. From the complete dataset, this study focuses exclusively on phishing instances, discarding legitimate samples as noise for the specific scope of this research. During preprocessing, the remaining samples are thoroughly cleaned by removing duplicate records and incomplete data.

![](images/978727a8f88453d99be17c5fd57051e591ba32bd88268d7925131542a2c773c3.jpg)  
Fig. 1: Example of a tree extracted from an HTML DOM.

## B. Phase 2: Tree Construction

Starting from the HTML content of phishing pages, the first phase of the proposed methodology consists of constructing a tree-based representation of their DOM. The objective is to preserve the hierarchical structure of the original document by representing each HTML element (or tag) as a node in the tree. In this representation, nodes correspond to individual HTML tag, while edges capture the parent-child relationships between elements. More specifically, an edge connects a parent node to a child node when the corresponding HTML tag is nested within another tag in the original document, as shown in Figure 1. This representation allows the construction of a tree with varying depths, according to the nesting structure of the DOM.

Each depth in the tree defines a level. Thus, webpages share a common top-level DOM structure, where the root node (0- level) corresponds to the <html> element and its immediate children (1-level) correspond to the <head> and <body> elements.

To prevent node collision when identical tags appear multiple times at the same hierarchical level, each node receives a unique identifier comprising the tag name, depth level, local child index, and a cumulative traversal counter.

## C. Phase 3: Feature Extraction & Clustering

Once the trees have been generated, this phase focuses on extracting structural and content-based parameters to characterize each individual tree. More specifically, the study considers two complementary groups of features. The first group relies on 5 structural properties of the tree, reported in Table I. These features capture the overall shape and complexity of the tree. On the other hand, the second group extends this structural description by incorporating contentbased information, that are the occurrence of the HTML tags observed at each level of the DOM tree (their distribution across the hierarchy). Thus, the following 22 HTML tags are used: form, input, button, a, iframe, script, img, link, meta, div, span, label, style, title, svg, section, picture, ul, li, nav, g, p. These elements are selected for their relevance in phishing threat since their are commonly associated with credential collection interfaces, embedded external resources, page layout, and client-side behavior. By integrating this tag-level information, the resulting representation preserves not only the structural arrangement of nodes but also the semantic role of the elements composing the tree.

Thanks to this feature extraction process, trees are converted into numerical vectors that provide a compact yet informative representation of webpages, enabling objective mathematical comparisons and aggregation. Thus, clustering algorithms are applied to group webpages exhibiting similar structural and content-based characteristics, allowing for the identification of latent patterns and templates that may not be immediately apparent through manual inspection.

## D. Phase 4: Validation

In this fourth phase, the groups generated by the clustering algorithms are evaluated both quantitatively and qualitatively. Specifically, in the first case, a dedicated metric based on Jaccard distance is used, while in the second case, representative samples are visually inspected for the final evaluation.

Quantitative Validation. In this step, to compare the performance of the clustering algorithms, a metric based on the level-wise Jaccard comparison of HTML tag-frequency distributions is used. We refer to this metric as the Level-wise Jaccard Distance Score $( \mathcal { L } \mathcal { I } \mathcal { D } )$ . The proposed metric evaluates cluster quality by measuring whether samples within the same cluster exhibit similar distributions of HTML tags across DOM levels, while samples assigned to different clusters remain sufficiently dissimilar.

First, for each pair of samples $( x , y )$ and for each DOM level l, both samples are represented by tag-frequency counters. Specifically, $c _ { x , l } ( t )$ and $c _ { y , l } ( t )$ denote the number of occurrences of tag t at level l in samples x and y, respectively. The level-wise Jaccard similarity is computed as:

$$
\mathcal { T } _ { l } ( x , y ) = \frac { \sum _ { t \in \mathcal { T } _ { l } } \operatorname* { m i n } ( c _ { x , l } ( t ) , c _ { y , l } ( t ) ) } { \sum _ { t \in \mathcal { T } _ { l } } \operatorname* { m a x } ( c _ { x , l } ( t ) , c _ { y , l } ( t ) ) }\tag{1}
$$

where the numerator represents the shared tag occurrences between the two samples at level l, while the denominator represents the total tag occurrences after combining both samples, computed through the maximum tag counts.

The level-wise similarities are then combined through a weighted average:

$$
\mathcal { L I } ( x , y ) = \sum _ { l \in \mathcal { L } } w _ { l } \cdot \mathcal { I } _ { l } ( x , y ) , \quad \mathrm { w i t h } \quad w _ { l } = \frac { \frac { 1 } { l } } { \sum _ { k \in \mathcal { L } } \frac { 1 } { k } }\tag{2}
$$

where $\mathcal { L }$ denotes the set of DOM levels considered in the comparison. In this work, the first two DOM levels are not considered, since they usually correspond to the common toplevel structure of HTML documents. The remaining levels are weighted according to their depth, assigning larger weights to upper levels and lower weights to deeper levels. This weighting scheme reflects the intuition that differences in higher DOM levels are more relevant for identifying changes in the global structure of a webpage, whereas differences in deeper levels are more likely to correspond to local variations, nested layout details, or minor implementation differences.

Since $ { \mathcal { L I } } ( x , y )$ defines a similarity score, it is converted into a distance as follows:

$$
\mathcal { L T D } ( x , y ) = 1 - \mathcal { L T } ( x , y )\tag{3}
$$

This transformation preserves the usual interpretation of distance-based cluster validation: small distances indicate similar samples, whereas large distances indicate dissimilar samples.

Let $\mathcal { P } ^ { A } = \{ C _ { 1 } , C _ { 2 } , \ldots , C _ { K } \}$ be the partition generated by a clustering algorithm A, where $K$ is the number of clusters produced by the algorithm. For each cluster $C _ { i }$ , it is possible to compute its internal compactness (the intra metric) using Equation 4, i.e., the average distance between unordered pairs of distinct samples within $C _ { i }$ . Likewise, it is possible to compute the degree of separation between a cluster and all the others (the inter metric), i.e., the average distance between samples belonging to different clusters, as shown in Equation 5. This latter metric yields an array of distances for each cluster. Therefore, for each $C _ { i }$ they can be aggregated into a single value, denoted by ${ \overline { { \mathcal { L T D } } } } _ { \mathrm { i n t e r } } .$ as shown in Equation 6.

$$
\mathcal { L I D } _ { \mathrm { i n t r a } } ( C _ { i } ) = \frac { 1 } { \binom { | C _ { i } | } { 2 } } \sum _ { \stackrel { x , y \in C _ { i } } { x < y } } \mathcal { L I D } ( x , y )\tag{4}
$$

4. Select C samples within each cluster to validate internal consistency

<table><tr><td></td><td>Features Description</td><td></td></tr><tr><td>1</td><td>Number of nodes</td><td>It represents the total count of HTML elements within the pruned DOM tree, indicating the overall structural complexity of the web page.</td></tr><tr><td>2</td><td>Number of edges</td><td>It quantifies the total hierarchical and relational links between elements. A high edge-to-node ratio often suggests a more complex layout.</td></tr><tr><td>3</td><td>Maximum centrality</td><td>It identifies the most interconnected node in the graph. In phishing, this often corresponds to a central container or a critical form element [22].</td></tr><tr><td>4</td><td>Minimum centrality</td><td>It measures the lowest degree of connectivity, helping to identify peripheral elements or isolated scripts often used for obfuscation [22].</td></tr><tr><td>5</td><td>Centrality on average</td><td>This feature provides a global measure of structural importance across all elements, capturing the general density of the page&#x27;s architecture [23].</td></tr></table>

TABLE I: List of structural properties extracted from the tree.

$$
\mathcal { L I D } _ { \mathrm { i n t e r } } ( C _ { i } , C _ { j } ) = \frac { 1 } { | C _ { i } | | C _ { j } | } \sum _ { x \in C _ { i } } \sum _ { y \in C _ { j } } \mathcal { L I D } ( x , y ) , \quad i \neq j\tag{5}
$$

$$
\overline { { \mathcal { L T D } } } _ { \mathrm { i n t e r } } ( C _ { i } ) = \frac { 1 } { K - 1 } \sum _ { \stackrel { j = 1 } { j \neq i } } ^ { K } \mathcal { L T D } _ { \mathrm { i n t e r } } ( C _ { i } , C _ { j } )\tag{6}
$$

Using equation 7, intra and inter metrics can be finally combined to generate an unique value for each cluster $( \mathcal { L I D } ( C _ { i } ) )$ where small values indicate that cluster $C _ { i }$ is internally compact and well separated from the remaining clusters. Conversely, high values indicate that the internal dispersion of the cluster is large relative to its separation from the rest of the partition.

$$
\mathcal { L I D } ( C _ { i } ) = \frac { \mathcal { L I D } _ { \mathrm { i n t r a } } ( C _ { i } ) } { \overline { { \mathcal { L I D } } } _ { \mathrm { i n t e r } } ( C _ { i } ) }\tag{7}
$$

The $\mathcal { L } \mathcal { I } \mathcal { D } ^ { A }$ (Level-wise Jaccard Distance Score associated with algorithm A) is defined as a vector containing each $\mathcal { L I D } ( C _ { i } )$ for all the $C _ { i } \in \mathcal { P } ^ { A }$ , as detailed in Equation 8.

$$
\mathcal { L I D } ^ { A } = [ \mathcal { L I D } ( C _ { 1 } ) , \mathcal { L I D } ( C _ { 2 } ) , \ldots , \mathcal { L I D } ( C _ { K } ) ]\tag{8}
$$

Finally, to summarize the behaviour of each clustering algorithm, three descriptive statistics are computed over the elements of the vector $\mathcal { L } \mathcal { I } \mathcal { D } ^ { A }$ : the mean $( \mathcal { L } \mathcal { I } \mathcal { D } _ { m e a n } ^ { A } )$ the standard deviation $( \mathcal { L } \mathcal { I } \mathcal { D } _ { s t d } ^ { A } )$ , and the maximum value $( \mathcal { L } \mathcal { I } \mathcal { D } _ { m a x } ^ { A } )$ . Yet, $\mathcal { L I D } _ { m e a n } ^ { A }$ provides an overall estimate of cluster quality, the $\mathcal { L I D } _ { s t d } ^ { A }$ measures the variability of the cluster-level scores, and the $\hat { L } \mathcal { I } \mathcal { D } _ { m a x } ^ { A }$ identifies the most problematic cluster generated by the algorithm. As $\mathcal { L I D } _ { i n t r a } ( C _ { i } )$ is not defined for singletons (i.e., clusters with just one element), we do not consider them for the $\mathcal { L } \mathcal { I } \mathcal { D } _ { m e a n } ^ { \bar { A } }$ and $\mathcal { L I D } _ { s t d } ^ { A }$ computations.

It should be noted that $\mathcal { L } \mathcal { I } \mathcal { D } ^ { A }$ differs from standard featurespace clustering metrics. Rather than evaluating the clustering partition directly on the normalized feature vectors used by the clustering algorithms, it is computed from the extracted DOMtree representations by comparing the level-wise distributions of HTML tags. Therefore, its interpretation plays a key role in the cluster evaluation, providing a DOM-level validation criterion complementary to standard feature-space clustering metrics.

Qualitative Validation. In this case, due to the high number of groups generated by the clustering algorithms, only N representative clusters are considered, as shown in step 2 in Figure 2. Specifically, the N most compact are selected, i.e., those with the lowest ${ \mathcal { L I D } } _ { \mathrm { i n t r a } }$ values. Then, they are visually inspected through their cluster medoids.

Medoid Evaluation. For each of the selected N compact cluster, the sample with the minimum average $\mathcal { L T D }$ to the other elements of the same cluster is identified as the cluster medoid (step 3 in Figure 2). These medoids are then visually compared across the selected clusters to assess whether they provide representative and distinguishable structural patterns. Specifically, the graphs are analysed and visualised using the Gephi tool [24]. Then, to obtain a more comprehensive view of the selected groups, C additional samples within each considered cluster are selected and compared with their medoid to assess consistency within each cluster (step 4 in Figure 2).

![](images/1137f1661a139b0e33f92e1e69e7d8c534817563411e7503ebf6e2af13e3ade5.jpg)  
Fig. 2: Schema of the qualitative validation steps followed during the analysis.

## IV. EXPERIMENTAL STUDY

## A. Dataset

The experimental evaluation of the proposed methodology uses the dataset developed by Aljofey et al. [21], specifically the subset identified as DS-2. This dataset was selected for its inclusion of full HTML source code, which is an essential requirement for the tree-based representation described in Phase 1. Collected during 2023 and published in 2025, the corpus captures current evasion tactics and modern web design patterns, thereby serving as a benchmark for structural analysis. The dataset contains 23,366 samples, including 8,366 phishing instances and 15,000 legitimate instances. Phishing samples were collected from PhishStats¹ and validated between July 15 and August 9, 2023. Only the sub-corpus of 8,366 phishing pages is used in this study, as the primary objective is the structural analysis and clustering of phishing threats to identify recurrent attack patterns.

## B. Experiment setup

As mentioned earlier, each webpage can generate a tree with different depth, depending on the number of nested elements. In order to provide a more comprehensive view of similarity, this study investigates how the number of considered levels can affect the clustering task. In particular, varying the depth of the extracted DOM representation may influence both the structural detail captured and the resulting similarity between webpages. For this reason, eight different depths are considered and compared, ranging from 8 to 15 levels. This range is chosen to balance representational completeness and computational efficiency, ensuring that both shallow and deep structural information are evaluated. Specifically, for each maximum level selected, if a webpage has a deeper tree, it is pruned to the maximum fixed depth. If a sample has fewer levels, the missing-level (tags) information is set to 0.

For the qualitative analysis, the parameters N and C must be fixed. In this study, the number of representative compact clusters is set to $N \ = \ 3 .$ while the number of additional samples per cluster is set to C = 2. Furthermore, to ensure a meaningful and reliable validation, only clusters containing more than 4 elements are considered in the qualitative analysis. This filtering step excludes clusters that, although potentially relevant, do not contain a sufficient number of samples to support a representative visual inspection and robust qualitative assessment of compact consistency. As a result, the analysis focuses on clusters with adequate cardinality, reducing the risk of drawing conclusions from sparsely populated or statistically unstable groups.

## C. Model configurations

To obtain a broader view of the clustering performance, the results of three different algorithms are compared. Specifically, two density-based methods (DBSCAN and OPTICS) and one hierarchical method (Agglomerative-AHC) are used. Nevertheless, as introduced in Section II-A, these algorithms require the tuning of specific parameters. Thus, for both density-based methods, a grid-search mechanisms is implemented varying the € (or xi) and min\_samples parameters, while different distance thresholds are used for hierarchically-based method, as reported in Table II.

To evaluate the performance of the clustering algorithms and perform the grid search analysis, state-of-the-art metrics are considered [25], such as the Silhouette Coefficient $( S _ { s c o r e } )$ , the Calinski-Harabasz Index $( \mathcal { C } \mathcal { H } _ { i n d e x } )$ and the Davies-Bouldin Index $( D B _ { i n d e x } )$ . These metrics are mainly used during grid search to identify the best configuration for each algorithm at each considered level of the analysis. Yet, the search approach is initially applied using only the structural properties of the tree, and then combining also content-based features, as detailed in Section III-C.

Notably, during the search task, the introduced metric $( \mathcal { L } \mathcal { I } \mathcal { D } )$ is not used, as the analysis prioritizes established and well-validated metrics. Thus, once best configurations are detected, they are further validated using quantitative and qualitative approaches.

<table><tr><td>Algorithms</td><td>Parameter</td><td>Values</td></tr><tr><td>DBSCAN</td><td>€ min_samples</td><td>{0.1, 0.2,..., 1.0}, step=0.1 2, 3, 5, 8, 10</td></tr><tr><td>OPTICS</td><td>xi min_samples</td><td>{0.1, 0.2,..., 1.0}, step=0.1 2, 3, 5, 8, 10</td></tr><tr><td>AHC</td><td>t</td><td>5, 25, 50, 75, 100</td></tr></table>

TABLE II: Grid search parameters for clustering algorithms.

## V. RESULTS

## A. Grid-search results

The results of the clustering algorithms during the search phase are reported in Figures 3, 4, and 5. Specifically, for simplicity, the analysis reports only the $S _ { s c o r e } .$ used as the performance-driven metric to detect the best configuration. Then, for the best configuration obtained at each level, the corresponding $\mathcal { C } \mathcal { H } _ { i n d e x }$ and $\mathcal { D } \boldsymbol { B } _ { i n d e x }$ values are further detailed in the next section. However, this section shows only the results obtained when structural features are used. The same grid-search analysis is also repeated using both structural and content-based features, and the results of this second case are reported in Appendix A.

Figure 3 reports the DBSCAN performance, demonstrating a consistent behavior of the algorithm across most tree depths. In fact, the highest $S _ { s c o r e }$ is obtained for a fixed value of $\epsilon = 0 . 6$ at almost all levels (8, 9, 11, 12, and 13). Deviations from this pattern occur only at levels 10 and 14, where the peak shifts toward lower-density configurations despite maintaining $\epsilon \ = \ 0 . 6$ At level 15, however, the maximum $S _ { s c o r e }$ is achieved when the neighborhood radius decreases to $\epsilon = 0 . 5$ Considering the min\_samples parameter, the highest scores are generally obtained with larger values (min\_samples = 8 or 10) across all levels except level 10, where the optimal min\_samples decreases to 3.

![](images/e39a23c723ff24731d67e8284f72e725082b48dfa68df3de998013ad98993bc0.jpg)  
(a) Level 8.

![](images/0302fa623e4d724d933d5ea656d0f8f9d0ab1df66b5fde5d06638dacb61d4b60.jpg)

![](images/a9d3e7bf9a304a296817700e41e0de2d7e745d47b6cf9feb4e762e1d51b9aee3.jpg)  
(b) Level 9.

![](images/8e6d9eccc0c05d0e1378514493042f690644aa2ab717664db63274659ff2eb05.jpg)

![](images/62392c57c9fd8834ab50918332d7a0ac100ed8c9cdcbe2cc35c0262d92a0850a.jpg)  
(e) Level 12.

(d) Level 11.  
(c) Level 10.  
![](images/6066e9ae706446f3f20e0c3f7d75bf32a930e48760d12ad623c986699fdf9137.jpg)  
(f) Level 13.

![](images/0895e4add7b95b8ae53f270d0ba29164b71ce393026e5aae353fe15e46738816.jpg)  
(g) Level 14.

![](images/21f5498e2d99469efc8499491af6db5a157c59711747f8ffb63e0d9c734853bb.jpg)  
(h) Level 15.

Fig. 3: DBSCAN $S _ { s c o r e }$ during the grid-search using only structural features.  
![](images/d6ec39817826861145127547784ad3cba5f3eb3ea36eb402d27ef5eab77eac13.jpg)

![](images/9d2ad04ebe6183b2733d2da77a96aa5bfe149c9107756ca6f76b5cd41f502b1f.jpg)

![](images/754d59256c9cfdf0375f6060b8503086f5110b6434b2f0dbe86293b8b8ab6319.jpg)

![](images/d9ba58cded8bb3bffa7a8923415a216e718a40087e2e89ea38cef1033e3909be.jpg)

(a) Level 8.  
![](images/8ddf7ae1f3815aa944fcc82f240f6b8e917aa5b59fbe6885604dd4924881cadb.jpg)  
(e) Level 12.

(b) Level 9.  
(d) Level 11.  
![](images/ad2fce16c44cbc6399a4ffd9fe4221abd43741a0b4cd3aee46d7b1933a976dd7.jpg)  
(f) Level 13.

(c) Level 10.  
![](images/88310c3328ffac942f1a0da18ab2740b3c6d30049103d244b003073c3371a5c0.jpg)  
(g) Level 14.

![](images/5ea955f5446aefebdd1a19dced954a2dea6afd1588afdcf87813bbe5d266e73d.jpg)  
(h) Level 15.  
Fig. 4: OPTICS $S _ { s c o r e }$ during the grid-search using only structural features.

Figure 4 shows the OPTICS performance, highlighting that, across all pruning stages, the highest value is invariably achieved with the lowest density value (min\_samples= 2). Conversely, looking at the xi parameter, it concentrates at 0.1 in all stages excluding level 10, whereas the xi value increases to 0.2.

AHC grid-search results are reported in Figure 5. Specifically, the figure shows consistent behavior in the algorithm performance, indicating that across all levels, the highest $S _ { s c o r e }$ is achieved with the same threshold t = 75.

## B. Structural and content-based features contribution analysis

Once the best configurations are identified through the gridsearch analysis conducted in Section V-A (and Appendix A), the clustering metrics presented in Section IV-C for each algorithm across all structural depths are summarized in Table III. Specifically, the first table reports the results when only structural properties are used, while the latter reports the results when content-based information is also included.

As shown in Table IIIa, OPTICS achieves the highest performance according to $\mathcal { S } _ { s c o r e } ,$ obtaining values greater than 0.75 across all eight evaluated levels. However, it performs worst with respect to $\mathcal { D } \boldsymbol { B } _ { i n d e x }$ and $\mathcal { C } \mathcal { H } _ { i n d e x }$ In contrast, DBSCAN and AHC achieve lower $\mathcal { D } \boldsymbol { B } _ { i n d e x }$ values, while AHC also attains higher $\mathcal { C } \mathcal { H } _ { i n d e x }$ values. Yet, OPTICS yields a relatively large number of clusters and a higher proportion of outliers $( N _ { o \% } \in [ 6 . 6 6 \% , 7 . 3 4 \% ] )$ . On the other hand, DBSCAN and AHC produce only a few clusters: the DBSCAN with very low proportions of outliers (≤ 1.04%); and the AHC with no singletons (clusters with just one element).

<table><tr><td></td><td colspan="5">DBSCAN</td><td colspan="6">OPTICS</td><td colspan="5">AHC</td></tr><tr><td>Level</td><td> $N _ { c }$ </td><td> $N _ { o \% }$ </td><td> $S _ { s c o r e }$ </td><td> $\mathcal { D B } _ { i n d e x }$ </td><td> $\mathcal { C H } _ { i n d e x }$ </td><td> $N _ { c }$ </td><td> $N _ { o \% }$ </td><td> $S _ { s c o r e }$ </td><td> $\mathcal { D B } _ { i n d e x }$ </td><td> $\mathcal { C H } _ { i n d e x }$ </td><td> $N _ { c }$ </td><td> $S _ { i \% }$ </td><td> $S _ { s c o r e }$ </td><td> $\mathcal { D } \boldsymbol { B } _ { i n d e x }$ </td><td></td><td> $\mathcal { C H } _ { i n d e x }$ </td></tr><tr><td>8</td><td>7</td><td>1.04%</td><td>0.545</td><td>0.484</td><td>732</td><td>551</td><td>7.34%</td><td>0.782</td><td>1.696</td><td>29</td><td>333</td><td>0.0%</td><td></td><td>0.625</td><td>0.575</td><td>3,487</td></tr><tr><td>9</td><td>7</td><td>0.91%</td><td>0.556</td><td>0.464</td><td>778</td><td>566</td><td>7.34%</td><td>0.773</td><td>1.605</td><td></td><td>45</td><td></td><td>0.0%</td><td>0.628</td><td>0.569</td><td>3,376</td></tr><tr><td>10</td><td>10</td><td>0.20%</td><td>0.546</td><td>0.924</td><td>650</td><td>567</td><td>7.11%</td><td>0.765</td><td>1.448</td><td>181</td><td></td><td>0.0%</td><td></td><td>0.656</td><td>0.566</td><td>3,435</td></tr><tr><td>11</td><td>7</td><td>0.68%</td><td>0.556</td><td>0.492</td><td>787</td><td>574</td><td>6.89%</td><td>0.766</td><td>1.502</td><td>191</td><td>3</td><td>0.0%</td><td></td><td>0.638</td><td>0.568</td><td>3,558</td></tr><tr><td>12</td><td>7</td><td>0.68%</td><td>0.552</td><td>0.461</td><td>786</td><td>583</td><td>6.86%</td><td>0.765</td><td>1.545</td><td>172</td><td>3 3</td><td>0.0%</td><td></td><td>0.606</td><td>0.622</td><td>3,457</td></tr><tr><td>13</td><td>7</td><td>0.63%</td><td>0.553</td><td>0.444</td><td>792</td><td>578</td><td>7.50%</td><td>0.755</td><td>1.525</td><td>114</td><td></td><td>0.0%</td><td></td><td>0.650</td><td>0.600</td><td>3,828</td></tr><tr><td>14</td><td>7</td><td>0.52%</td><td>0.555</td><td>0.475</td><td>794</td><td>591</td><td>7.07%</td><td>0.762</td><td>1.576</td><td>96</td><td></td><td>0.0%</td><td></td><td>0.663</td><td>0.432</td><td>4,450</td></tr><tr><td>15</td><td>7</td><td>1.04%</td><td>0.565</td><td>0.559</td><td>753</td><td>593</td><td>6.66%</td><td>0.767</td><td>1.547</td><td>132</td><td></td><td>0.0%</td><td></td><td>0.670</td><td>0.521</td><td>3,696</td></tr></table>

(a) Only structural properties results.
<table><tr><td></td><td colspan="5">DBSCAN</td><td colspan="5">OPTICS</td><td colspan="5">AHC</td></tr><tr><td>Level</td><td> $N _ { c }$ </td><td> $N _ { o \% }$ </td><td> $S _ { s c o r e }$ </td><td> $\mathcal { D } \boldsymbol { B } _ { i n d e x }$ </td><td> $\mathcal { C } \mathcal { H } _ { i n d e x }$ </td><td> $N _ { c }$ </td><td> $N _ { o \% }$ </td><td> $S _ { s c o r e }$ </td><td> $\mathcal { D B } _ { i n d e x }$ </td><td> $\mathcal { C H } _ { i n d e x }$ </td><td> $N _ { c }$ </td><td> $S _ { i \% }$ </td><td> $S _ { s c o r e }$ </td><td> $\mathcal { D B } _ { i n d e x }$ </td><td> $\mathcal { C H } _ { i n d e x }$ </td></tr><tr><td>8</td><td>287</td><td>20.64%</td><td>0.591</td><td>1.908</td><td>8.63</td><td>434</td><td>20.53%</td><td>0.639</td><td>1.981</td><td>7.21</td><td>900</td><td>65.22%</td><td>0.715</td><td>0.316</td><td>1,455</td></tr><tr><td>9</td><td>293</td><td>21.82%</td><td>0.583</td><td>1.944</td><td>7.77</td><td>463</td><td>23.77%</td><td>0.557</td><td>1.820</td><td>6.23</td><td>944</td><td>66.84%</td><td>0.711</td><td>0.292</td><td>1,661</td></tr><tr><td>10</td><td>295</td><td>22.00%</td><td>0.581</td><td>1.996</td><td>7.22</td><td>468</td><td>24.15%</td><td>0.550</td><td>2.015</td><td>5.80</td><td>966</td><td>68.12%</td><td>0.707</td><td>0.285</td><td>1,879</td></tr><tr><td>11</td><td>299</td><td>22.32%</td><td>0.581</td><td>2.056</td><td>6.67</td><td>486</td><td>23.77%</td><td>0.544</td><td>2.066</td><td>5.58</td><td>992</td><td>69.25%</td><td>0.707</td><td>0.265</td><td>2,173</td></tr><tr><td>12</td><td>298</td><td>22.72%</td><td>0.572</td><td>2.056</td><td>6.70</td><td>423</td><td>29.63%</td><td>0.488</td><td>2.000</td><td>5.17</td><td>1,009</td><td>69.87%</td><td>0.703</td><td>0.255</td><td>2,387</td></tr><tr><td>13</td><td>302</td><td>23.24%</td><td>0.568</td><td>2.074</td><td>6.21</td><td>389</td><td>26.26%</td><td>0.571</td><td>2.074</td><td>5.32</td><td>1,028</td><td>69.46%</td><td>0.707</td><td>0.247</td><td>2,532</td></tr><tr><td>14</td><td>309</td><td>23.58%</td><td>0.561</td><td>2.096</td><td>6.10</td><td>424</td><td>23.95%</td><td>0.598</td><td>2.158</td><td>5.26</td><td>1,047</td><td>69.34%</td><td>0.707</td><td>0.247</td><td>2,708</td></tr><tr><td>15</td><td>307</td><td>23.81%</td><td>0.559</td><td>2.138</td><td>6.03</td><td>489</td><td>27.57%</td><td>0.488</td><td>2.090</td><td>4.80</td><td>1,062</td><td>69.77%</td><td>0.706</td><td>0.243</td><td>2,956</td></tr></table>

(b) Structural and content-based properties results.  
TABLE III: Metrics of the best configuration of each algorithms for each level. $N _ { c }$ indicates the number of generated clusters; $N _ { o \% }$ the number of samples in percentage clustered as noisy points; $S _ { i \% }$ the number of cluster in percentage that have only one sample within. For each metric and each algorithm, the best metric value is shown in bold, while the model selected as the best performer is highlighted in yellow.

![](images/a4417b5a1e34ad83a0a4ffede343a4a6ab9b377f124f89d905743b2b42f3a353.jpg)  
Fig. 5: $\operatorname { A H C } S _ { s c o r e }$ during the grid-search using only structural features.

When content-based features are incorporated into the clustering process, two distinct behaviors are observed (Table IIIb). On the one hand, OPTICS exhibits a marked deterioration in clustering performance, reflected by higher $\mathcal { D } \boldsymbol { B } _ { i n d e x }$ values and lower $\boldsymbol { S _ { s c o r e } }$ and $\mathcal { C } \mathcal { H } _ { i n d e x }$ values. In addition, the proportion of outliers increases considerably, ranging from approximately 20% to 32%. Similarly, DBSCAN exhibits poorer performance in terms of $\mathcal { D } \boldsymbol { B } _ { i n d e x }$ and $\mathcal { C } \mathcal { H } _ { i n d e x }$ , while its $S _ { s c o r e }$ values slightly increased compared to those observed in the previous case. (Table IIIa). Furthermore, DBSCAN is now able to generate a considerable number of clusters $( N _ { c } ~ > ~ 2 8 6 )$ , albeit with a higher proportion of outliers (approximately 20-24%). On the other hand, AHC shows improvements in in $\boldsymbol { S _ { s c o r e } }$ and $\mathcal { D } \boldsymbol { B } _ { i n d e x }$ , with an increase of 0.046 and a decrease of 0.189 in their best values, respectively. Yet, AHC now produces approximately 65–70% of clusters containing a single element.

A comprehensive analysis of Table III reveals that the algorithms do not consistently achieve their best performance at a single configuration level. In fact, only in four cases do they achieve at least two best metrics at the same level, namely structural AHC at level 14, content-based DBSCAN at level 8, content-based OPTICS at level 8, and contentbased AHC at level 15. In the remaining cases, the final level selection was based on a holistic assessment of all three metrics, prioritizing the overall trade-off among them rather than the best performance in any single metric.

Looking at these insights and the extracted level information, the results show that, when considering only structural properties, all the models perform better with a higher number of levels within the trees (14, 11, and 14). However, when content-based information is incorporated, DBSCAN and OPTICS tend to perform better with fewer levels (just 8). These results are somewhat expected since the content-based information acts as noisy information during the clustering process, increasing uncertainty in terms of clustering metrics.

<table><tr><td>Algorithms</td><td>Features</td><td>Parameters</td><td>Level</td><td> $\mathcal { L I D } _ { m e a n }$ </td><td> $\mathcal { L I D } _ { s t d }$ </td><td> $\mathcal { L I D } _ { m a x }$ </td><td># Cluster with  $\mathcal { L T D } ( \mathcal { C } ) = 0$ </td></tr><tr><td rowspan="2">DBSCAN</td><td>structural</td><td> $\overline { { \epsilon = 0 . 6 , \mathrm { { m i n } } _ { - } { s a m p } 1 \mathrm { { e s } } \ = \ \ 1 0 } }$ </td><td>14</td><td>0.903</td><td>0.052</td><td>0.963</td><td>0</td></tr><tr><td>structural + content</td><td> $\epsilon = 0 . 9 , \mathrm { m i n \_ s a m p l e s } \ = \ 2$ </td><td>8</td><td>0.158</td><td>0.254</td><td>0.979</td><td>133</td></tr><tr><td rowspan="2">OPTICS</td><td>structural</td><td> $\overline { { x i } } = 0 . 1 , \mathrm { { m i n } } _ { - } \mathrm { { s a m p l e s } } \ = \ 2$ </td><td>11</td><td>0.592</td><td>0.368</td><td>1.093</td><td>87</td></tr><tr><td>structural + content</td><td> $x i = 0 . 3 , \mathrm { \ m i n \_ s a m p l e s } \ = \ 2$ </td><td>8</td><td>0.242</td><td>0.305</td><td>1.088</td><td>170</td></tr><tr><td rowspan="2">AHC</td><td>structural</td><td> $t = 7 5 . 0$ </td><td>14</td><td>0.901</td><td>0.047</td><td>0.959</td><td>0</td></tr><tr><td>structural + content</td><td> $t = 5 . 0$ </td><td>15</td><td>0.351</td><td>0.335</td><td>1.036</td><td>55</td></tr></table>

TABLE IV: Best configuration for each algorithm using structural and content-based information, and the corresponding results obtained during quantitative validation. In bold the best values in each case.

## VI. CLUSTER VALIDATION

## A. Quantitative Validation

Table IV reports the level-wise Jaccard Distance Score computed for the best configuration of each algorithm, as introduced in Section III-D. The table shows that all three algorithms produce the lowest $\mathcal { L I D } _ { m e a n }$ values when content features are used, with DBSCAN achieving the overall lowest value of 0.158. These lower values indicate that, on average, the ratio between cluster compactness and dispersion decreases when this additional information is incorporated. Thus, this suggests improved cluster separation among samples. However, considering $\mathcal { L T D } _ { s t d } ,$ high variability is generally observed across all algorithms, except for DBSCAN and AHC using only structural features. In fact, in both cases, $\mathcal { L I D } _ { m e a n }$ is similar to $\mathcal { L I D } _ { m a x }$ . Looking at these latter values $( \mathcal { L } \mathcal { I } \mathcal { D } _ { m a x } )$ , the algorithms remain consistent, showing that in the worst case at least one cluster exhibits a ratio close to 1, meaning that samples within the cluster are approximately as distant from each other as they are from samples in different clusters (or even more so). Finally, the last column of Table IV reports the number of clusters for which $\mathcal { L T D } ( C ) = 0 , \mathrm { i . e . }$ , fully compact clusters where $\mathcal { L T D } _ { \mathrm { { i n t r a } } } = 0$ , meaning that all samples within the cluster are structurally identical. Surprisingly, even when only structural features are considered during clustering, OPTICS is already able to identify 87 fully compact clusters. This suggests that structural information alone can be sufficient to group together subsets of webpages with identical level-wise DOM-tag distributions. When content-based features are incorporated, the number of fully compact clusters further increases, reaching 133 clusters for DBSCAN, 170 for OPTICS, and 55 for AHC. This result indicates that adding tag-level information strengthens the formation of highly homogeneous groups, although it also contributes to a more fragmented clustering structure.

## B. Qualitative Validation

Figures 6 and 7 report the tree structures belonging to the three most compact clusters, i.e., the clusters with the lowest $\mathcal { L T D } ( C )$ values, when considering structural features and structural-content features, respectively. To avoid presenting trivial cases composed of identical samples and to provide a more informative visual comparison, clusters with $\mathcal { L T D } ( C ) =$ 0 were excluded from the selection.

Figure 6 shows the medoids and samples of the clusters obtained using only structural features, labeled as S1, S2, and S3. By inspecting the medoids, it can be observed that they share a similar structure in their <head> branch. Specifically, S2 and S3 exhibit the same tag structure. Moreover, S2 and S3 also share a similar structure within the <body> branch. However, they differ significantly in their last two levels, where S3 contains more nodes than S2. Yet, looking at the samples extracted from each cluster, it possible to see that samples from S1 have the same structure of the medoid, due to the very low LJD (0.0022). On the other hand, for higher values, such as 0.0102 in S3, the samples show small difference with the medoids.

Figure 7 reports the qualitative results obtained when content-based features are also included, labeled as SC1, SC2, and SC3. In this case, considering only the medoids, their structures differ substantially from one another, with SC2 exhibiting a particularly complex configuration. Nevertheless, in SC1, the medoid and the two selected samples are visually almost identical, differing only by a single change in the <head> branch. In contrast, SC2 is characterized by much larger and deeper trees. Despite their size, the medoid and the selected samples share a highly similar overall structure, mainly consisting of an extended horizontal organization with several repeated small branches. Finally, the samples belonging to SC3 differ in the number of nodes present in their deepest levels, as shown in Figure 7.

Finally, to provide a more comprehensive view of the generated clusters, a qualitative analysis of the medoids extracted from the most dispersed clusters was conducted and is reported in Appendix B.

## C. Discussion

The results presented in this work show that DOM-tree representations provide a useful basis for identifying structural similarities among phishing webpages. Across the evaluated clustering algorithms, the grid-search analysis indicates that meaningful groupings can be obtained from the extracted tree features, although each algorithm reacts differently to the type of information used. When only structural properties are considered, the best configurations tend to appear at deeper DOM levels, suggesting that global tree complexity and nested organization provide relevant information for distinguishing phishing templates. In contrast, when content-based features are incorporated, the best configurations for density-based methods are generally obtained at shallower levels. This indicates that tag-level information introduces a more specific but also more variable description of the webpages, making deeper levels more sensitive to local implementation differences.

![](images/d95183e9d9e3df187baed60af1193c74ab611a04e8ec8e5ba5b88522024c58b9.jpg)  
Fig. 6: Qualitative analysis: medoids of N compacted clusters and two random samples from each cluster, using structural based features. Changes are marked with a black circle.

The comparison between structural-only and structuralcontent representations highlights a central trade-off. Structural features provide a general characterization of the DOM, capturing aspects such as size, connectivity, and global topology. As a result, they are useful for detecting broad structural reuse patterns, but they may group together webpages that share a similar shape while differing in the specific HTML elements used. Conversely, adding content-based features substantially reduces $\mathcal { L I D } _ { m e a n }$ for all algorithms, indicating more compact clusters according to the level-wise distribution of HTML tags. This suggests that content information refines the notion of similarity and helps identify more specific template variants. However, this refinement also increases fragmentation: density-based methods produce a higher proportion of noisy samples, while AHC generates many singleton clusters. Therefore, content-based features improve specificity but reduce coverage, whereas structural-only features provide broader but less fine-grained groupings.

![](images/d477d730fec495d62d6ef207092d2e20b95da531e603a17baa71b10c9dafbba7.jpg)

![](images/43032c908038172c861b53c6afc4cfb547c294b070006fd1f4aec07277db7000.jpg)

![](images/41648e53044240d6d80e0ea5a037eef2e467d17b71bd895429d3a1f8c9493fc0.jpg)  
Fig. 7: Qualitative analysis: medoids of N compacted clusters and two random samples from each cluster, using structural and content-based features. Changes are marked with a black circle.

The qualitative validation further supports this interpretation. The selected compact clusters correspond to visually coherent groups of DOM trees, which is consistent with the idea that phishing kits, reusable templates, and shared web components can generate multiple webpages with highly similar internal structures. In these cases, even if pages differ in domains, visual details, or deployment context, their underlying DOM organization may preserve evidence of template reuse or common construction logic. Conversely, the dispersed clusters contain heterogeneous structures, including simplified or almost flattened DOMs. These cases suggest that not all phishing webpages expose the same level of structural regularity: some may rely on minimal HTML, redirection mechanisms, dynamically generated content, or incomplete/obfuscated structures, making them more difficult to compare through tree-based representations.

## D. Limitation & Future Work

Despite the promising results achieved by the proposed approach, several (potential) issues need to be highlighted. First, although the use of content-based features yielded the best overall performance according to the introduced metric, clustering algorithms based on these features were characterized by a high number of unclustered elements (DBSCAN, OPTICS) and singleton groups (AHC). In this regard, strategies for incorporating these instances into the analysis, such as nearest-cluster assignment or two-stage clustering, should be further investigated; otherwise, a significant portion of the dataset remains unanalyzed. Furthermore, although the proposed metric proved useful for assessing cluster quality, it is important to note that it relies exclusively on tag-based and level information. As these same features are also employed during clustering, the evaluation may be partially biased toward clustering solutions generated from similar information. Consequently, approaches based on content features may naturally yield more compact clusters according to this metric. To mitigate this potential bias, alternative and more featureagnostic distance measures, such as graph edit distance, should be explored.

## E. Ethics Concerns

From an ethical research perspective, three key considerations must be considered. Firstly, although the dataset primarily consists of publicly accessible phishing webpages collected for security research purposes, it may still contain sensitive information, including personally identifiable information (PII) such as email addresses, names, telephone numbers, and other identifiers inadvertently exposed within page content.

Secondly, the proposed methodology raises dual-use concerns. While the approach is intended to support cybersecurity objectives by characterising phishing infrastructure and improving detection and defensive mechanisms, the inferred structural and behavioural insights could potentially be exploited by malicious actors to adapt their tactics, techniques, and procedures (TTPs), thereby increasing the sophistication and evasiveness of phishing campaigns.

Thirdly, although clustering and structural similarity analysis may suggest commonalities between phishing webpages, such inferences should not be interpreted as definitive evidence of shared authorship or attacker identity. Any attributionrelated conclusions must be treated as probabilistic and used exclusively for defensive and threat intelligence purposes.

## VII. CONCLUSION

This work investigated whether the HTML structure of phishing webpages can be used as a reliable fingerprint for identifying structural similarities and potential template reuse. By representing webpages as DOM trees and clustering them using structural and content-based features, the results show that recurrent patterns can be detected without relying on prior labels. Structural features provide a broader view of common DOM organization, while the inclusion of tag-level content information enables a more fine-grained identification of highly homogeneous template variants. The proposed $\mathcal { L T D }$ metric and the qualitative validation further confirm that compact clusters correspond to visually coherent tree structures. Overall, the findings support the use of DOM-based clustering as a promising approach for phishing template analysis, while also highlighting the need to better handle noisy, flattened, or highly fragmented samples in future work.

## ACKNOWLEDGMENTS

This work has been partially supported by the European Union through the Horizon Europe Programme under the projects SAFEHORIZON (Grant Agreement No. 101168562), ENSEMBLE (Grant Agreement No. 101168360), and by Vicomtech under the project ATHENA-CV (Grant Number E22503/2024). The content of this article does not reflect the official opinion of the European Union. Responsibility for the information and views expressed therein lies entirely with the authors.

## REFERENCES

[1] Anti-Phishing Working Group (APWG), "Phishing activity trends report, 1st quarter 2025," APWG, Tech. Rep., July 2025, published July 2, 2025. [Online]. Available: https://www.apwg.org

[2] Europol, “Internet organised crime threat assessment - iocta 2026," accessed on 08/06/2026. [Online]. Available: https://www.europol. europa.eu/cms/sites/default/files/documents/IOCTA-2026.pdf

[3] W. Li, S. U. A. Laghari, S. Manickam, Y.-W. Chong, and B. Li, "Machine learning-enabled attacks on anti-phishing blacklists," IEEE Access, vol. 12, pp. 191 586–191 602, 2024.

[4] S. Asiri, Y. Xiao, and T. Li, "Phishtransformer: A novel approach to detect phishing attacks using url collection and transformer," Journal Name (Reemplazar por el nombre de la revista si lo tienes), 2024, published online Dec 2023.

[5] H. Jiang, Y. Chen, Y. Zhu, X. Xu, Y. Song, and Q. Chen, "D-phishnet: A dual-branch network for url and html feature fusion in phishing webpage detection," Nombre de la Revista (ej. Future Generation Computer Systems o similar), 2025.

[6] T. Song, P. Casas, and M. Meo, “"Phishing the phishers with specularnet: Hierarchical graph autoencoding for reference-free web phishing detection," arXiv preprint arXiv:2603.01874, 2026.

[7] F. Castaño, E. F. Fernañdez, R. Alaiz-Rodríguez, and E. Alegre, "Phikita: Phishing kit attacks dataset for phishing websites identification,"IEEE Access, vol. 11, pp. 40 779–40 789, 2023.

[8] H. Bijmans, T. Booij, A. Schwedersky, A. Nedgabat, and R. van Wegberg, "Catching phishers by their bait: Investigating the dutch phishing landscape through phishing kit detection," in 30th USENIX security symposium (USENIX security 21), 2021, pp. 3757–3774.

[9] A. K. Jain and B. B. Gupta, "Phishing detection: analysis of visual similarity based approaches," Security and Communication Networks, vol. 2017, no. 1, p. 5421046, 2017.

[10] A. Oest, Y. Safei, A. Doupé, G.-J. Ahn, B. Wardman, and G. Warner, "Inside a phisher's mind: Understanding the anti-phishing ecosystem through phishing kit analysis," in 2018 APWG Symposium on Electronic Crime Research (eCrime). IEEE, 2018, pp. 1–12.

[11] E. Merlo, M. Margier, G.-V. Jourdan, and I.-V. Onut, "Phishing kits source code similarity distribution: A case study," in 2022 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER). IEEE, 2022, pp. 983–994.

[12] M. Ester, H.-P. Kriegel, J. Sander, X. Xu et al., "A density-based algorithm for discovering clusters in large spatial databases with noise," in kdd, vol. 96, no. 34, 1996, pp. 226–231.

[13] M. Ankerst, M. M. Breunig, H.-P. Kriegel, and J. Sander, "Optics: Ordering points to identify the clustering structure," ACM Sigmod record, vol. 28, no. 2, pp. 49–60, 1999.

[14] D. Müllner, “Modern hierarchical, agglomerative clustering algorithms," arXiv preprint arXiv:1109.2378, 2011.

[15] O. K. Sahingoz, E. Buber, O. Demir, and B. Diri, "Machine learning based phishing detection from urls," Expert Systems with Applications, vol. 117, pp. 345–357, 2019.

[16] D. O. Otieno, F. Abri, A. S. Namin, and K. S. Jones, “Detecting phishing urls using the bert transformer model," in 2023 IEEE International Conference on Big Data (BigData). IEEE, 2023, pp. 2483–2492.

[17] K. S. Mandapati, S. Meesala, D. Maddela, K. Ponnada, H. Neyyala, and E. A. Shaik, “A hybrid transformer ensemble approach for phishing website detection," in 2023 International Conference on Self Sustainable Artificial Intelligence Systems (ICSSAS). IEEE, 2023, pp. 1–8.

[18] K. V. Ajay Kumar, P. S. Balasubramaniyamoorthy, R. Deepalakshmi, and K. R. SenthilMurugan, “A proactive method using machine learning models to detect phishing attacks in thread sharing network," Nombre de la Revista o Conferencia, 2024.

[19] L. D. Nguyen, D.-N. Le, and L. T. Vinh, "Detecting phishing web pages based on dom-tree structure and graph matching algorithm." Association for Computing Machinery, 2014, pioneering structural analysis using Genetic Algorithms.

[20] K. Althobaiti, M. K. Wolters, N. Alsufyani, and K. Vaniea, "Using clustering algorithms to automatically identify phishing campaigns,' Nombre de la Revista o Conferencia (ej. ACM Digital Library), 2023.

[21] A. Aljofey, S. A. Bello, J. Lu, and C. Xu, "Comprehensive phishing detection: A multi-channel approach with variants tcn fusion leveraging url and html features," Journal of Network and Computer Applications, vol. 238, p. 104170, 2025.

[22] D. R. White and S. P. Borgatti, "Betweenness centrality measures for directed graphs," Social networks, vol. 16, no. 4, pp. 335–346, 1994.

[23] L. C. Freeman, “Centrality in social networks conceptual clarification," Social Networks, vol. 1, pp. 215–236, 1978.

[24] M. Bastian, S. Heymann, and M. Jacomy, "Gephi: an open source software for exploring and manipulating networks," in Proceedings of the international AAAI conference on web and social media, vol. 3, 2009, pp. 361–362.

[25] I. F. Ashari, E. D. Nugroho, R. Baraku, I. N. Yanda, and R. Liwardana, "Analysis of elbow, silhouette, davies-bouldin, calinski-harabasz, and rand-index evaluation on k-means algorithm for classifying floodaffected areas in jakarta," Journal of Applied Informatics and Computing, vol. 7, no. 1, pp. 95–103, 2023.

## APPENDIX A GRID-SEARCH RESULTS (PART II)

As mentioned, while Section V-A reports grid-search results obtained using clustering algorithms based solely on structural information, this section presents the results when structural information is combined with content-based features. In these figures (9,10,8), the best-performing configurations are highlighted with a red box. These configurations are then used in Section V-B.

## APPENDIX B

## QUALITATIVE VALIDATION - PART II

In terms of dispersed clusters, Figure 11 reports the medoids of the clusters with the highest $\mathcal { L T D } ( C )$ values. Specifically, Figure 11a shows the three selected clusters obtained using only structural features, while Figure 11b shows the three selected clusters obtained when structural and content-based features are combined. Compared with the compact clusters reported in Figures 6 and 7, these medoids exhibit more heterogeneous configurations, including trees with reduced depth, simplified structures, or clearly different branching organizations. In some cases, the medoids consist of only three levels; that is, after removing the first two levels, corresponding to <html> and <head>/<body>, the remaining DOM structure is almost completely flattened.

![](images/f4a108c6d8c41e98ab769f32123529ace45a73da4f0d2600f7a23b5e3f458e20.jpg)  
Fig. 8: AHC $S _ { s c o r e }$ during the grid-search using structural and content-based features.

![](images/84a18c1a3564bf0f2428dad9baae57a7c5e1c6463b8566f5113597d3c3d905df.jpg)  
(a) Level 8.

![](images/6c4e8f7f940d86496e3ee61f78c6f71528d3909199727e3e7450cdf1df29f869.jpg)  
(b) Level 9.

![](images/0c667cfa2c719782792bfbe31e505a7f07e83e4542745c5750efc23581529570.jpg)  
(c) Level 10.

![](images/c6ec1b17d05cb3466090fea88cebcc8c3ce39030c80ef73dcdbb9489093d9de3.jpg)  
(d) Level 11.

![](images/8de18e4826f8fc014522efc5c86319589986fb89b8de4f08d3b92c194b3a08fd.jpg)  
(e) Level 12.

![](images/e4f5549a46664a38d980979553a08ad705d9b44dcc77b8ed96f85c91b1c3820b.jpg)  
(f) Level 13.

![](images/d0eea9f4ef32ba624738cecb83f64dabcffd3d0c055b6fdc9eec3e4a7d8738dd.jpg)  
(g) Level 14.

![](images/47fd5b310dcdb2a7e5833ac3d780ed1a932f9660dc766f22eee70d33964cf0ae.jpg)  
(h) Level 15.  
Fig. 9: DBSCAN $S _ { s c o r e }$ during the grid-search using structural and content-based features.

![](images/cd34d186fbb8878b0e7af2f256339d561f65d5f7d187722dc9d187160a4b51d3.jpg)  
(a) Level 8.

![](images/e3e0fae103fc1e9511b8d538cbdc95117040e8c81044b5ba0ef36af0a84f5f6a.jpg)  
(b) Level 9.

![](images/f618401ea0e1e2d80eb07ee1b0a56973b455624a539b2f6bdfa9effd6aac0672.jpg)  
(c) Level 10.

![](images/a1002f136e219992e9f423d90e853fa611432028c4af75603a962332e51fbe4a.jpg)  
(d) Level 11.

![](images/da59cc0617e41973a8fac2fd72415b03bc7fed7a07d37cb2950f27cdb09e917d.jpg)  
(e) Level 12.

![](images/482785609f8b987cf65edd962f131054999335f52bb3021c211d0bbb1525ee99.jpg)  
(f) Level 13.

![](images/6af47306fb8693463068628a9dff14a584c81fd7dc0b687151e33b9ff517ad92.jpg)  
(g) Level 14.

![](images/615fe83b21bca0f200d45f638eacac0c48a982f26ed8bfd1f294bbd47c91323f.jpg)  
(h) Level 15.  
Fig. 10: OPTICS $S _ { s c o r e }$ during the grid-search using structural and content-based features.

![](images/6a4228eff61792e2b530bf374ae1b583993d214418f609b04794d424d5fafa68.jpg)  
(a) Structural features

![](images/b12f907e52c9ab6f8fb3e27fd8bf0d41ee3d154e4aa2f65b24bbe90fc5e9ea6a.jpg)  
(b) Structural and content-based features  
Fig. 11: Qualitative analysis: - medoids of N dispersed clusters.