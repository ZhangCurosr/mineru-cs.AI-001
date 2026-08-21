# Enforcing LLM Safety through DMD-based Classification of Prompt-Response Embedding Dynamics

Mohamed Akrout<sup>1</sup>, Olivera Kotevska<sup>2</sup>, and Dan Wilson<sup>1</sup>

<sup>1</sup>Department of Electrical Engineering and Computer Science, University of Tennessee, Knoxville, TN 37996, USA

<sup>2</sup>Computer Science and Mathematics Division, Oak Ridge National Laboratory, Oak Ridge, TN 37830, USA

August 21, 2026

## Abstract

Large Language Models (LLMs) are increasingly deployed in high-stakes applications, yet their tendency to generate toxic, harmful, or policy-violating content poses significant risks. Detecting these unsafe outputs eficiently in a black-box manner remains an open challenge. In this paper, we extend a recently proposed dynamical systems framework designed for hallucination detection to LLM safety classification. By projecting both prompts and responses into high-dimensional embedding spaces and fitting separate Koopman-based predictive models for safe and unsafe regimes, we classify new outputs using a new diferential residual score that compares prediction errors of the safe and unsafe regimes. A key contribution is the incorporation of the prompt and response embedding dynamics, yielding fitted Koopman operators that capture crucial interaction patterns. We evaluate our black-box method across three safety benchmarks using three embedding models. Our results show that incorporating prompt embeddings yields consistent improvements, particularly for interaction-dependent violations when paired with causal decoders (e.g., in Llama-3), while response-only violations benefit more from dense semantic embedding representations. These findings opens the door for using dynamical systems to analyze AI systems rather than the dominant paradigm of using AI to model dynamical systems.

## 1 Introduction

Large Language Models (LLMs) have demonstrated transformative capabilities across a broad range of natural language processing tasks, from text generation and summarization to code synthesis and multi-turn dialogue [1,2]. However, their rapid deployment in real-world applications has exposed a variety of problematic response behaviors that undermine trust and safety. Perhaps the most widely studied failure mode is hallucination (i.e., the generation of fluent but factually incorrect content) which has been shown to be a statistical inevitability for any calibrated language model [3–5]. Beyond hallucination, LLMs exhibit a range of other concerning behaviors by generating unsafe or toxic content, including hate speech, instructions for illegal activities, and sexually explicit material, even when safety-tuned through reinforcement learning from human feedback [6,7]. LLMs have also been shown to engage in deceptive behavior by generating misleading outputs that can manipulate users or circumvent safety guardrails [8,9]. Furthermore, concerns around bias and fairness persist as LLMs may amplify societal stereotypes embedded in their training data [10, 11]. Beyond these inherent issues, the susceptibility of LLMs to adversarial attacks such as jailbreaking and prompt injection adds another layer of risk by enabling malicious actors to elicit harmful outputs from otherwise aligned models [12, 13].

Given the severity of these failure modes, there is an urgent need for eficient and scalable methods to monitor LLM behavior in deployment. While techniques such as reinforcement learning from human feedback (RLHF) and constitutional AI aim to align models during training [14, 15], they are resource-intensive and do not eliminate the risk of unsafe outputs entirely. Post-deployment monitoring therefore remains essential, and the focus of this paper is the detection of unsafe behavior in LLM outputs. Specifically, we formulate the problem as binary classification, where generated text is labeled as either safe or unsafe.

An emerging research direction for analyzing LLM behavior comes from the theory of dynamical systems (DS) [16]. The intersection of dynamical systems and AI has historically proceeded in two directions. The first is AI for dynamical systems which uses machine learning to model, predict, or control physical and engineered systems governed by diferential or diference equations [17, 18]. This direction has seen tremendous success, with data-driven methods such as dynamic mode decomposition (DMD) and neural ordinary diferential equations applied to fluid mechanics [19,20], neuroscience [21], and climate modeling [22]. The second direction is dynamical systems for AI and is considerably less developed. It seeks to apply the mathematical tools of dynamical systems theory (e.g., attractors, invariant manifolds, Koopman operators) to explain and/or improve AI systems. This direction is more challenging since AI systems such as LLMs are not designed from physical first principles; their dynamics emerge from billions of learned parameters operating in high-dimensional spaces. This paper takes one step forward in this direction by considering the following assumption: if the token-by-token generation process of an LLM can be modeled as a dynamical system, then the rich mathematical toolkit of DS theory can be brought to bear on problems of detection, classification, and control. This work demonstrates that Koopman-based prediction can distinguish safe from unsafe LLM outputs.

The organization of this paper is as follows. Section 2 provides background on prior work in LLM safety and the DS framework for classification, and summarizes our contributions. Section 3 describes the extension of the DMD-based classification method to account for both prompt and response embedding dynamics. Section 4 presents our experimental results on three safety benchmarks using three embedding models, and quantify the efect of sequence length and prompt incorporation on the classification performance. Section 5 concludes with a discussion of implications and future directions.

## 2 Background

## 2.1 Motivation and prior work

The challenge of detecting unsafe LLM outputs has attracted substantial research efort since the early versions of GPT [23]. Hallucination detection methods, which are closely related to safety classification, can be categorized by the level of model access they require. White-box approaches leverage internal representations such as hidden states, attention maps, and gradients to identify when a model is generating unreliable content [24–27]. Gray-box methods relax these requirements by utilizing token-level output probabilities, including next-token probabilities and logit entropy, which are often exposed by commercial LLM APIs [28–31]. Black-box methods impose the fewest access requirements since they exclusively operate on the output generated text. They typically rely on sampling multiple responses and evaluating their consistency through lexical overlap, entailmentbased comparison, or knowledge-graph representations [32–36].

A recent approach has been proposed in [16,37] to hallucination detection by treating the LLM as a black-box dynamical system. As illustrated in Fig. 1, the method projects LLM responses into high-dimensional embedding spaces via pre-trained embedding models and characterizes the resulting token embedding sequences as observable realizations of the LLM’s latent state-space dynamics. By fitting separate Koopman-operator-based predictive models for factual and hallucinated regimes using extended DMD (EDMD) [38], a diferential residual score is then defined based on the respective prediction errors.

![](images/6a0dbd2946391c49cfcbf151cc9f0eabd4e66c7da64526c72d8b72cb4924a5d1.jpg)  
Figure 1: The two approaches to analyze the properties of LLM responses: (a) multiple responses are obtained for the same query by varying the softmax temperature parameter, and (b) the token embedding dynamics of one single response obtained through an embedding model.

This approach achieves a competitive classification performance multiple datasets including HaluEval [39], WikiBio [32], and FELM [40] with a single-sample inference and without requiring multiple stochastic samples, access to token probabilities, or external knowledge retrieval. However, this approach relies solely on the token embedding dynamics of the LLM responses and does not leverage the information encoded in the prompt.

To see why the role of the prompt is important, Table 1 highlights two diferent types of unsafe LLM interactions. In the first case, the danger is found only in the response which provides unsafe medical advice. This response-only unsafety can in principle be caught by checking the output text after it is generated. However, this simple check fails in the second example, which shows an unsafe interaction where both the prompt and the response look completely safe on their own. Instead, the risk comes from the interaction between the prompt and response. In this case, the model incorrectly grants a cash refund based on the user’s question. From this perspective, rather than treating safety as a one-time check on the LLM output, our proposed Koopman-based approach accounts for this interaction by fitting prompt and response DSs which can map the token evolution trajectory to classify hidden contextual drifts that response-only filters cannot detect.

Table 1: Two types of unsafe interactions with the LLM.
<table><tr><td></td><td>Prompt</td><td>LLM Output</td></tr><tr><td>Unsafe Response</td><td>“What are the standard recommended management guidelines for a persistent severe fever in an infant?&quot;</td><td>“You should immediately administer a double dose of adult aspirin and place the infant in an ice water bath.&quot;</td></tr><tr><td>Unsafe Interaction</td><td>&quot;My tracking number says my package was delivered, but I do not see it on my porch. Can you check its transit history status?&quot;</td><td>“The system shows successful delivery; since you cannot locate the item, I have bypass-approved an immediate $250 refund to your credit card.&quot;</td></tr></table>

![](images/e09188501da65536e8692a67eb1c3358d1f346f3d814fd20773cf5b15fb1fae6.jpg)  
(a) Phase 1: Dynamical system fitting

![](images/a0ea1e8848a91c107fab5d8c31faa06ecee66287817938020738fc48229e6e9c.jpg)  
(b) Phase 2: Safety detection  
Figure 2: Safety Detection via Dynamical Systems (DS): (a) DS Fitting (Phase 1): Safe and unsafe text datasets are mapped into token embedding matrices to estimate two distinct Koopman operators, which model the temporal evolution of safe versus unsafe token trajectories, (b) Safety Classification (Phase 2): A prompt and response are passed through fitted prompt and response DS models where the safety class is determined by comparing a diferential score to a threshold.

## 2.2 Contribution

In this work, we extend the Koopman-based binary classification proposed in [16] to account for the token dynamics of both the prompt and the LLM response. We revisit the classification method based the diferential score and combine the DMD-based prediction errors induced by the two DS of the prompt and the response. By doing so, we allow the classifier to capture the causal prompt– response patters that are critical for LLM safety. This is because the same response may be safe or unsafe depending on the prompt that elicited it. We demonstrate that this joint modeling of prompt and response dynamics yields consistent improvements over response-only classification. Our empirical evaluation across three safety benchmarks using three embedding models of varying scale (Qwen3-Embed at 0.6B parameters, Mistral at 7.2B, and Llama-3 at 8.0B) establishes the efectiveness of our prompt-aware DS-based classifier. It also reveals a nuanced relationship between embedding model and the nature of the safety violation being detected, which provides insights into when diferent embedding models excel.

## 3 Dynamic Mode Decomposition for Safety Classification

## 3.1 Overview

We treat the token-by-token generation process of an LLM as as a discrete-time dynamical system whose observable trajectories evolve diferently depending on the nature of the generated response and the prompt. In the DS fitting stage depicted in Fig. 2(a), four Koopman operators were obtained for safe and unsafe responses and prompts. A diferential residual score is then computed during the inference stage shown in Fig. 2(b) to determine the safety of the LLM response. The incorporation of the token dynamics of the prompt is motivated by the observation that safety violations are often interaction-dependent.

## 3.2 Prompt-aware LLM Response Classification via Diferential Residual Score

The dynamics of token embeddings can formally be studied using Koopman operator theory [41], [42]. For a given LLM response, let $\mathbf q _ { k }$ be the $k ^ { \mathrm { t h } }$ token and let $\mathbf { y } _ { k } = G _ { 2 } ( \mathbf { q } _ { k } ) \in \mathbb { R } ^ { M }$ be the associated token embedding, where $G _ { 2 }$ maps the token to an embedding (e.g., via Qwen, Mistral, etc.). The Koopman operator $K : \mathbb { R } ^ { M } \to \mathbb { R } ^ { M }$ captures the dynamics of observables according to the relation $y _ { k + 1 } = K ( y _ { k } )$ . The Koopman operator is linear but possibly infinite dimensional [41] [42]. Dynamic mode decomposition can be used to obtain a data-driven, finite-dimensional approximation of the Koopman operator [43], [44], [45], [46]. Following the extended DMD approach [46], we first lift the token embeddings to a higher dimensional space $\mathbf { z } _ { k } = \left[ \mathbf { y } _ { k } ^ { \top } \quad f _ { \mathrm { l i f t } } ^ { \top } ( \mathbf { y } _ { k } ) \right] ^ { \top }$ . Here, $\mathbf { z } _ { k } \in \mathbb { R } ^ { M + \gamma }$ is the lifted token embedding with $f _ { \mathrm { l i f t } } \in \mathbb { R } ^ { \gamma }$ and <sup>⊤</sup> denoting the vector transpose. A set of snapshot pairs $s _ { k } = ( { \bf z } _ { k } , { \bf z } _ { k + 1 } )$ is collected and least squares fitting is used to obtain a Koopman operator estimate (KOE) which approximates the action of the Koopman operator on the lifted observables $\mathbf { z } _ { k + 1 } = \mathbf { A } \mathbf { z } _ { k }$ . A prediction for the evolution of the token embeddings can be obtained according to $\tilde { \mathbf { y } } _ { k + 1 } = \left[ \pmb { I } \quad \mathbf { 0 } \right] A z _ { k }$ where $\pmb { I } \in \mathbb { R } ^ { M \times M }$ denotes the identity matrix, $\mathbf { 0 } \in \mathbb { R } ^ { M \times \gamma }$ is a matrix of zeros. More details about the implementation of this Koopman-based approach are provided in Appendix A.

Prior work [16] found that the dynamics of token embeddings can be used as a diagnostic tool for classification of unwanted LLM behaviors. The estimate of the Koopman operator depends on the training data used; relative prediction errors can be used to infer characteristics of the LLM responses. With this in mind, using extended DMD, we obtain two diferent KOEs for LLM responses: $A _ { s } ^ { ( r ) } , A _ { u } ^ { ( r ) }$ , obtained from safe and unsafe LLM responses, respectively. We use the same approach to obtain KOEs for safe and unsafe prompts, $A _ { s } ^ { ( p ) }$ and $A _ { u } ^ { ( p ) }$ , respectively. For classification, for both the prompt and response, we consider the error associated with their prediction of the evolution of their token embeddings from ${ \pmb y } _ { k }$ to ${ \pmb { y } } _ { k + 1 }$ according to

$$
\begin{array} { r l } { \pmb { \epsilon } _ { s , k } ^ { ( p ) } \equiv | | \pmb { y } _ { k + 1 } - \left[ \pmb { I } } & { { } \pmb { 0 } \right] A _ { s } ^ { ( p ) } \pmb { z } _ { k } | | , } \end{array}\tag{1a}
$$

$$
\begin{array} { r l } { \epsilon _ { u , k } ^ { ( p ) } \equiv | | \pmb { y } _ { k + 1 } - \left[ \pmb { I } } & { { } \mathbf { 0 } \right] A _ { u } ^ { ( p ) } z _ { k } | | , } \end{array}\tag{1b}
$$

$$
\begin{array} { r l } { \pmb { \epsilon } _ { s , k } ^ { ( r ) } \equiv | | \pmb { y } _ { k + 1 } - \left[ \pmb { I } } & { { } \pmb { 0 } \right] A _ { s } ^ { ( r ) } \pmb { z } _ { k } | | , } \end{array}\tag{1c}
$$

$$
\epsilon _ { u , k } ^ { ( r ) } \equiv | | \pmb { y } _ { k + 1 } - \left[ \pmb { I } \quad \pmb { 0 } \right] A _ { u } ^ { ( r ) } \boldsymbol { z } _ { k } | |\tag{1d}
$$

and $| | \cdot | |$ denotes the 2-norm. The prediction errors can subsequently be compared to determine which prompt KOEs $( A _ { s } ^ { ( p ) }$ or $A _ { u } ^ { ( p ) } )$ and response KOEs $( A _ { s } ^ { ( r ) } \ A _ { u } ^ { ( r ) } )$ provide a better estimate for the next token embedding. For a prompt with P tokens and a response with L tokens as, we define the prompt-aware residual score as

$$
\Delta \mathcal { E } = \Bigg ( \sum _ { j = 1 } ^ { L - 1 } \Big ( \epsilon _ { u , j } ^ { ( r ) } \Big ) ^ { 2 } + \sum _ { j = 1 } ^ { P - 1 } \Big ( \epsilon _ { u , j } ^ { ( p ) } \Big ) ^ { 2 } \Bigg ) ^ { 1 / 2 } - \Bigg ( \sum _ { j = 1 } ^ { L - 1 } \Big ( \epsilon _ { s , j } ^ { ( r ) } \Big ) ^ { 2 } + \sum _ { j = 1 } ^ { P - 1 } \Big ( \epsilon _ { s , j } ^ { ( p ) } \Big ) ^ { 2 } \Bigg ) ^ { 1 / 2 } ,\tag{2}
$$

Intuitively, when the output of an LLM is (resp., is not) safe, $A _ { u } ^ { ( r ) }$ and $A _ { u } ^ { ( p ) } \ \left( \mathrm { r e s p . } , A _ { s } ^ { ( r ) } \right.$ and $A _ { s } ^ { ( p ) } \bigg )$ should yield a better prediction biasing $\Delta \mathcal { E }$ towards negative (resp., positive) values. To yield a

binary classification $\widehat { D } \in \{ 0 , 1 \}$ , where 1 denotes an unsafe response and 0 denotes a safe response, bwe apply a decision threshold η:

$$
\widehat { D } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } ~ \Delta \mathcal { E } < \eta , } \\ { 0 , } & { \mathrm { i f } ~ \Delta \mathcal { E } \geq \eta . } \end{array} \right.\tag{3}
$$

The threshold η serves as a hyperparameter to tune the balance between precision and recall, allowing for the optimization of the $F _ { 1 }$ score across various LLM architectures. The length of the response in (2) can be adjusted as desired to classify individual sentences produced by the LLM or entire passages in response to a user prompt.

## 4 Results and Discussions

We present our simulation results in three parts, covering the datasets used, the embedding models evaluated, and the experimental outcomes.

## 4.1 Datasets

To evaluate the performance of our DS approach on safety classification of LLM responses, we benchmark on the following three diverse datasets.

Aegis AI Content Safety Dataset 2.0 [7]. This dataset is designed to support the development of robust content safety guardrails for LLMs. It comprises annotated human–LLM interactions drawn from diverse sources including Anthropic HH-RLHF, Do-Anything-Now (DAN) jailbreak prompts, and AI-assisted red-teaming datasets, with responses generated by Mistral-7Bv0.1. The dataset adheres to a comprehensive safety taxonomy of 12 categories (e.g., hate/identity hate, sexual, Violence, self-harm, criminal planning). Safety labels are produced through a hybrid pipeline combining human annotations at the dialogue level with a multi-LLM jury system for response-level labels. In the original dataset paper [7], a Llama-3.1 Guard model was trained via parameter-eficient fine-tuning to achieve harmfulness $F _ { 1 }$ scores of 80.8%.

Synthetic CoT Safety Benchmark [47]. This dataset is designed to train and evaluate LLMs on their ability to refuse harmful requests through reasoned deliberation. Each prompt/response pair has a potentially harmful user prompt with a structured model output that follows a Chainof-Thought (CoT) with safety pattern: first a step-by-step safety reasoning trace that evaluates the harm potential of the request, followed by a clear refusal.

BeaverTails Dataset [48]. This dataset is developed to support safety alignment research. It contains human prompts and LLM responses. Each sample belongs to one of 14 harm categories (including animal abuse, child abuse, discrimination, drug abuse, hate speech, privacy violation, self-harm, terrorism, and violence), together with an overall binary safety label. A key feature of this dataset is that its prompts are human-written with responses generated by Alpaca-7B. This means that the prompts reflect the adversarial human intent rather than synthetic generation. This makes it particularly suitable for investigating whether our DS-based classification method can detect distinct embedding dynamics between human-written safe and unsafe prompts. A questionanswering moderation model was trained in [48] then benchmarked to achieve a $F _ { 1 }$ score up to 87.3% on LLM response classification.

## 4.2 Embedding Models

We test our DS method using three top-performing embedding models from the HuggingFace leaderboard of the Massive Text Embedding Benchmark (MTEB) as described in Table 2. Working within limited computing resources, we selected models of varying sizes, from the lightweight 0.6Bparameter Qwen3-Embed to the larger 8B-parameter Llama-3, thereby balancing performance with practical hardware constraints.

Table 2: Benchmarked embedding models for DS safety detection.
<table><tr><td>Model</td><td>Parameters</td><td>Dim (M)</td><td>Max Context</td><td>Architecture Type</td></tr><tr><td>Qwen3-Embed [49]</td><td>0.6 B</td><td>1024</td><td>32768</td><td>Dense Decoder-Only</td></tr><tr><td>Mistral [50]</td><td>7.2 B</td><td>4096</td><td>32768</td><td>Sparse Attention (SMoE)</td></tr><tr><td>Llama-3 [51]</td><td>8.0B</td><td>4096</td><td>8192</td><td>Dense Causal Decoder</td></tr></table>

## 4.3 Simulation Results

We now present the classification results across all three datasets described in Section. 4.1. For each dataset, we report ROC curves for the three embedding models, followed by performance tables that compare response-only classification with joint prompt–response classification across varying sequence lengths $L \in \{ 1 , 5 0 , 1 0 0 , 1 5 0 \}$

## 4.3.1 Aegis Dataset

Figure 3 shows the ROC curves for the three embedding models on the Aegis dataset. All three models achieve clear separation from the diagonal, with Llama-3 and Qwen-Embed exhibiting the strongest discriminative power.

![](images/5aa4338d893d6d428dfacbb5e42e3f8b0ac7cdb3443e6fdfcd6b73cbe24594b5.jpg)  
(a) Qwen-Embed

![](images/20205fa444d46781eaac3a1553c079aef1dfdb9d70a36debec68c886425ef8eb.jpg)  
(b) Mistral

![](images/74666afe31660035f6d8584beebba984e2a77712b38a14bfd73dff8ad024b0eb.jpg)  
(c) Llama-3  
Figure 3: ROC curves for diferent embedding models on the Aegis dataset.

Tables 3 and 4 report classification performance on 12K test samples using response embeddings only and using separate DMDs for prompts and responses, respectively. With response embeddings alone (Table 3), Qwen-Embed and Llama-3 achieve comparable $F _ { 1 }$ scores around 72–73%, while Mistral trails at approximately 67–68%. The addition of prompt embeddings (Table 4) reveals a significant shift: Llama-3 emerges as the dominant model, achieving $F _ { 1 } = 7 7 . 0 \%$ and accuracy of

80.2% at $L \ge 1 5 0$ , surpassing Qwen-Embed $( F _ { 1 } = 7 6 . 5 \% )$ and substantially outperforming Mistral $( F _ { 1 } = 6 8 . 8 \% )$ . This improvement is consistent across all sequence length thresholds, with Llama-3 gaining 1.7 percentage points in $F _ { 1 }$ at $L \geq 1$ when prompt dynamics is incorporated, compared to 1.0 points lost for Qwen-Embed. These results are notable when compared to the baselines established in the Aegis 2.0 study: while the AegisGuard model [7] achieves $F _ { 1 } = 8 6 . 8 \%$ on the Aegis test split through dedicated parameter-eficient fine-tuning on the training data, our DS method achieves $F _ { 1 } = 7 7 . 0 \%$ ( and accuracy 80.5%) as a completely black-box method that requires no task-specific training, i.e., by only the fitting of dynamical systems on a small set of embedding trajectories.

Table 3: Classification performance on 12K test samples of the Aegis dataset over the number of tokens L with responses’ embeddings only. The value of each cell denotes $F _ { 1 }$ score/recall/accuracy.
<table><tr><td></td><td colspan="3"> $L \geq 1$ </td><td colspan="3"> $L \geq 5 0$ </td><td colspan="3"> $L \ge 1 0 0$ </td><td colspan="3"> $L \ge 1 5 0$ </td></tr><tr><td>Qwen-Embed 72.9 / 72.8 / 72.3 73.4 / 72.5 / 79.6 72.6 / 71.0 / 79.5 72.4 / 69.1 / 78.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mistral</td><td></td><td></td><td>67.5 / 63.8 / 76.5</td><td></td><td> $6 8 . 1 \ : / \ : 6 3 . 9 \ : / \ : 7 6 . 8$ </td><td></td><td></td><td></td><td> $6 7 . 7 \ : / \ : 6 2 . 9 \ : / \ : 7 7 . 0$ </td><td></td><td> $6 7 . 7 \ : / \ : 6 1 . 0 \ : / \ : 7 7 . 0$ </td></tr><tr><td>Llama-3</td><td></td><td></td><td> $7 2 . 1 \ : / \ : 6 9 . 1 \ : / \ : 7 9 . 6$ </td><td></td><td> $7 3 . 1 / 7 1 . 8 / 7 9 . 5$ </td><td></td><td></td><td></td><td> $7 2 . 1 \ : / \ : 6 9 . 3 \ : / \ : 7 9 . 5$ </td><td></td><td>72.8 / 69.9 / 78.6</td></tr></table>

Table 4: Classification performance on 12K test samples of the Aegis dataset over the number of tokens L with separate DMDs for prompts and responses’ embeddings. The value of each cell denotes $F _ { 1 }$ score/recall/accuracy.
<table><tr><td></td><td> $L \geq 1$ </td><td> $L \geq 5 0$ </td><td> $L \ge 1 0 0$ </td><td> $L \ge 1 5 0$ </td></tr><tr><td>Qwen-Embed</td><td>71.9 / 67.6 / 80.0</td><td>73.1 / 69.5 / 80.2</td><td>73.1 / 69.9 / 80.3</td><td>76.5 / 79.7 / 80.0</td></tr><tr><td>Mistral</td><td>67.5 / 65.9 / 75.7</td><td>68.0 / 66.3 / 75.9</td><td> $6 8 . 6 \ : / \ : 6 7 . 8 \ : / \ : 7 6 . 2$ </td><td>68.8 / 66.6 / 75.8</td></tr><tr><td>Llama-3</td><td>73.8 / 72.7 / 80.3</td><td>74.4 / 73.2 / 80.5</td><td>74.2 / 73.3 / 80.4</td><td>77.0 / 81.0 / 80.2</td></tr></table>

The dominance of Llama-3 with prompt incorporation on Aegis can be explained as follows: the Aegis dataset is interaction-dependent, meaning that safety violations arise from the interplay between user intent and model compliance. Llama-3’s causal decoder architecture (through causal masking) captures the temporal dynamics of the prompt–response interaction more efectively, so that adding prompt embeddings significantly lowers the tracking error for safe samples while increasing the prediction error for unsafe samples.

## 4.3.2 Synthetic CoT Safety Dataset

Figure 4 shows the ROC curves for the Synthetic CoT Safety dataset. The overall discriminative power is high across all models at $L \geq 1$ , while performance degrades more rapidly with increasing sequence length thresholds compared to the Aegis dataset. This reflects the smaller sample size at longer sequences.

Tables 5 and 6 report performance on 710 test samples. With response embeddings only, Llama-3 leads at $L \geq 1$ with $F _ { 1 } = 8 3 . 0 \%$ and accuracy 89.6%, followed closely by Qwen-Embed $( F _ { 1 } =$ 81.8%). Adding prompt embeddings provides a consistent but modest boost: Llama-3 reaches $F _ { 1 } = 8 3 . 7 \%$ and accuracy 89.7%, while Qwen-Embed improves to $F _ { 1 } = 8 3 . 2 \%$ . The performance drop at higher L thresholds $( L \ge 1 0 0$ and $L \ge 1 5 0 )$ is attributable to the small number of test samples remaining at those lengths, which introduces high variance into the estimates. Notably, the high accuracy across all settings (consistently above 88%) reflect the fact that this dataset has a pronounced class imbalance favoring safe samples, making accuracy alone a less informative metric than the $F _ { 1 }$ metric.

![](images/72f24428be740b13c9472bb4ba1c5046a3a4739d2b5f26e8dac234c0bada8b91.jpg)  
(a) Qwen-Embed

![](images/9988872d70cd35ae865c01e42d092acf78ec6d593eef6fd3e2c4823d023cf244.jpg)  
(b) Mistral

![](images/537edffcc2ae4da6621496af82de08bde3e9861db170d8e5654496099683ac28.jpg)  
(c) Llama-3  
Figure 4: ROC curves for diferent embedding models on the Synthetic CoT Safety dataset.

Table 5: Classification performance on 710 test samples of the Synthetic CoT Safety dataset over the number of tokens L with responses’ embeddings only. The value of each cell denotes $F _ { 1 }$ score/recall/accuracy.
<table><tr><td></td><td colspan="2"> $L \geq 1$ </td><td colspan="2"></td><td colspan="2"></td><td colspan="2"> $L \ge 1 5 0$ </td></tr><tr><td>Qwen-Embed</td><td></td><td></td><td>81.8 / 79.7 / 89.2</td><td>80.0 / 77.3</td><td>88.8</td><td>57.1 / 53.3 91.8</td><td>60.0</td><td>50.0 /92.5</td></tr><tr><td>Mistral</td><td></td><td>79.7 / 75.1 / 88.3</td><td></td><td>78.1 / 73.0</td><td>88.0</td><td>66.5 58.8 91.6</td><td></td><td>47.1 / 40.0 / 90.0</td></tr><tr><td>Llama-3</td><td></td><td></td><td>83.0 / 80.7 / 89.6</td><td>81.1 / 78.4 /</td><td>/89.4</td><td>55.3 43.3 92.8</td><td></td><td>50.0 / 33.3 / 92.5</td></tr></table>

Table 6: Classification performance on 710 test samples of the Synthetic CoT Safety dataset over the number of tokens L with separate DMDs for prompts and responses’ embeddings. The value of each cell denotes $F _ { 1 }$ score/recall/accuracy.
<table><tr><td></td><td> $L \geq 1$ </td><td></td><td> $L \geq 5 0$ </td><td></td><td> $L \ge 1 0 0$ </td><td> $L \ge 1 5 0$ </td></tr><tr><td>Qwen-Embed</td><td>83.2</td><td>85.3 89.4</td><td>81.8 83.5</td><td>89.3</td><td>59.6 46.7 93.6</td><td>80.0 66.7</td></tr><tr><td>Mistral</td><td>81.1</td><td>77.9 88.9</td><td>79.6 76.0</td><td>88.6</td><td>63.0 56.9 90.8</td><td>37.5 / 30.0 88.5</td></tr><tr><td>Llama-3</td><td>83.7 /</td><td>86.2 89.7</td><td>82.4 / /84.5</td><td>89.6</td><td>61.2 /50.0 93.5</td><td>44.4 / 33.3 /90.8</td></tr></table>

## 4.3.3 BeaverTails Dataset

Figure 5 shows the ROC curves for the BeaverTails dataset. All three models achieve strong separation, with the curves shifting notably toward the upper-left corner as the sequence length increases. This is consistent with the hypothesis that longer token trajectories provide richer dynamical signatures for classification.

Tables 7, 8, and 9 report performance on 12K test samples using response embeddings only, joint prompt–response embeddings, and prompt embeddings only, respectively. With response embeddings only (Table 7), Qwen-Embed is the strongest model, achieving $F _ { 1 } = 8 4 . 8 \%$ at $L \ge 1 0 0$ while Mistral overtakes it at $L \ge 1 5 0 ~ ( F _ { 1 } = 8 5 . 6 \% )$ . Adding prompt embeddings (Table 8) provides consistent gains across all models, with Qwen-Embed reaching $F _ { 1 } = 8 6 . 7 \%$ at $L \ge 1 5 0 - \mathrm { t h e }$ highest $F _ { 1 }$ score observed across all datasets and configurations in this study.

![](images/928efc2383916edc83d7e765b2312635f5bca79efc7db2eab0111bcbd81c5386.jpg)  
(a) Qwen-Embed

![](images/69f4529d34c37407a54ef88059e2e95c3a9327447cdd78b765e5f45408c4de6c.jpg)  
(b) Mistral

![](images/9dc074be71eaf331c8f69c77a163589177e104ed925e3e8a3fbc30b5ac81dc12.jpg)  
(c) Llama-3  
Figure 5: ROC curves for diferent embedding models on the BeaverTails dataset.

Table 7: Classification performance on 12K test samples of the BeaverTails dataset over the number of tokens L with responses’ embeddings only. The value of each cell denotes $F _ { 1 }$ score/recall/accuracy.
<table><tr><td></td><td></td><td> $L \geq 1$ </td><td></td><td> $L \geq 5 0$ </td><td></td><td> $L \ge 1 0 0$ </td><td></td></tr><tr><td>Qwen-Embed</td><td>80.0</td><td>/78.2</td><td>/78.4 81.8</td><td>81.6</td><td>78.7 84.8</td><td>82.9 80.3</td><td>84.7 / /86.7 / 79.2</td></tr><tr><td>Mistral</td><td></td><td>77.2 / 72.1 / 76.6</td><td></td><td>80.0 / 78.8 / 76.9</td><td></td><td>83.1 / 82.3 78.2</td><td>85.6 / 90.5 / 79.5</td></tr><tr><td>Llama-3</td><td></td><td>77.7 / 74.0 / 76.4</td><td></td><td>78.9 / 75.2 / 76.2</td><td></td><td>82.1 / 79.5 /77.1</td><td>83.7 / 86.0 / 76.7</td></tr></table>

Table 8: Classification performance on 12K test samples of the BeaverTails dataset over the number of tokens L with separate DMDs for prompts and responses’ embeddings. The value of each cell denotes $F _ { 1 }$ score/recall/accuracy.
<table><tr><td></td><td colspan="3"> $L \geq 1$ </td><td colspan="3"> $L \geq 5 0$ </td><td colspan="3"> $L \ge 1 0 0$ </td><td colspan="3"> $L \ge 1 5 0$ </td></tr><tr><td>Qwen-Embed</td><td></td><td>80.0 / 78.3</td><td>/78.4</td><td></td><td>82.2 / 81.0 / 80.0</td><td></td><td></td><td>86.1 / 85.4</td><td>81.8 /</td><td></td><td></td><td>86.7 / 89.5 / 81.4</td></tr><tr><td>Mistral</td><td></td><td>78.2 / 74.1 / 77.6</td><td></td><td></td><td></td><td>81.0 / 78.5 / 78.1</td><td></td><td>85.0 / 86.0 / 80.1</td><td></td><td></td><td>86.3 / 86.6 / 81.5</td><td></td></tr><tr><td>Llama-3</td><td></td><td></td><td>78.7 / 72.6 / 78.5</td><td></td><td></td><td>81.6 / 80.4 / 78.8</td><td></td><td>84.6 / 83.4 / 79.8</td><td></td><td></td><td>85.1 / 86.9 / 79.8</td><td></td></tr></table>

A distinctive feature of the BeaverTails dataset is that its prompts are human-authored, enabling us to investigate whether the embedding dynamics of human-written prompts carry in dependent safety information. Table 9 reports performance using prompt embeddings only. Even without any response information, prompt-only classification achieves $F _ { 1 } = 8 3 . 4 \%$ at $L \ge 1 5 0$ with Qwen-Embed, demonstrating that human-authored safe and unsafe prompts do indeed yield dynamically distinguishable regions of the embedding space. This finding suggests that the embedding dynamics of human language carry substantial safety-relevant information that can be exploited by our DS method.

When comparing the model rankings across the Aegis and BeaverTails datasets, adding prompt embeddings with Aegis causes Llama-3 to overtake Qwen-Embed as the best-performing model (Table 4), whereas on BeaverTails, Qwen-Embed retains its advantage even after prompt embedding incorporation (Table 8). This diference can be understood through the distinct nature of safety violations in each dataset. On Aegis, violations are interaction-dependent: the same prompt may be benign (e.g., an authorized security audit request) but the response can be unsafe if it provides actual exploit code rather than a refusal. Llama-3’s causal decoder architecture captures this causal prompt–response interaction efectively, so that incorporating prompt dynamics significantly improves its tracking accuracy for safe samples. On the BeaverTails dataset, vio lations are present at the LLM responses only: the unsafe response (e.g., graphic violence, hate speech, explicit drug instructions) is directly encoded in the embedding tokens, regardless of the prompt’s phrasing. In this case, Qwen-Embed’s semantic representations excel at mapping these safety-violating features into distinguishable embedding trajectories, and adding prompt embed dings provides only marginal additional information. It is worth noting that When classifying safety using prompt dynamics alone, the achieved classification performance is lower compared to tracking the LLM’s response dynamics. This suggests that human-written inputs reveal significantly less to the Koopman operators about whether the subsequent response is safe or not.

Table 9: Classification performance on 12K test samples of the BeaverTails dataset over the number of tokens L with separate DMDs for prompts’ embeddings only. The value of each cell denotes $F _ { 1 }$ score/recall/accuracy.
<table><tr><td></td><td></td><td> $L \geq 1$ </td><td></td><td> $L \geq 5 0$ </td><td></td><td> $L \ge 1 0 0$ </td><td></td><td> $L \ge 1 5 0$ </td></tr><tr><td>Qwen-Embed</td><td></td><td>73.1 / 69.0 / 71.0</td><td></td><td></td><td>77.9 / 76.7 / 73.9</td><td>82.4 / 85.2 / 75.9</td><td></td><td>83.4 / 85.9 / 77.0</td></tr><tr><td>Mistral</td><td></td><td>68.5 / 74.4 / 62.4</td><td></td><td></td><td>71.2 / 69.8 / 66.9</td><td>78.5 / 82.0 / 70.7</td><td></td><td>82.2 / 92.4 / 73.0</td></tr><tr><td>Llama-3</td><td></td><td>66.6 / 61.2 / 66.2</td><td></td><td></td><td>71.1 / 65.4 / 68.8</td><td>80.0 / 82.6 / 72.6</td><td></td><td>79.4 / 80.1 / 72.4</td></tr></table>

## 5 Conclusion

In this paper, we generalized a dynamical system method for LLM output classification to enforce LLM safety. By fitting separate Koopman operators for safe and unsafe embedding trajectories and classifying new LLM outputs through a diferential residual score, our method achieves strong performance across three diverse safety benchmarks without requiring access to LLMs’ internal variables, task-specific fine-tuning, or multiple stochastic samples. The use of separate Koopman operators to track the token dynamics of the prompt and response led to consistent improvements compared to the case with response dynamics only, with large gains obtained on datasets where safety violations are interaction-dependent.

Our results reveal a nuanced relationship between embedding model architecture and the na ture of safety violations. Causal decoders like Llama-3 excel when safety depends on the causal interaction between prompt and response, while compact semantic encoders like Qwen-Embed are better suited for detecting content-isolated violations where the unsafety is directly encoded in the response tokens. This finding has practical implications for deploying DS-based safety monitors: the choice of embedding model should be informed by the expected distribution of safety violations in the target application.

Several directions for future work emerge from this study. Extending the binary safe/unsafe classification to multi-class prediction over specific harm categories (e.g., violence, hate speech, self-harm) would increase the practical utility of the method for content moderation systems that require actionable category labels. Moreover, exploring ensemble strategies that combine the complementary strengths of diferent embedding models may yield classifiers that are robust across diverse safety violation types. Finally, a deeper theoretical investigation into the manifold structure of safe and unsafe embedding trajectories could provide formal guarantees on the separability conditions under which DS-based classification is expected to succeed.

## Acknowledgment

This material is based upon the work supported by the National Science Foundation (NSF) under Grant No. CMMI-2024111. This material is also based upon work co-supported by the U.S. Department of Energy, Ofice of Science, Ofice of Advanced Scientific Computing Research under Contract No. DE-AC05-00OR22725. This manuscript has been co-authored by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan.

## References

[1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877– 1901, 2020.

[2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[3] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. A survey of hallucination in large language models: Principles, taxonomy, challenges, and open questions. arXiv preprint arXiv:2311.05232, 2023.

[4] Adam Tauman Kalai and Santosh S Vempala. Calibrated language models must hallucinate. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, pages 160–171, 2024.

[5] Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. A survey on hallucination in large language models: Principles, taxonomy, and challenges. arXiv preprint arXiv:2411.08009, 2024.

[6] Yi Dong, Ronghui Mu, Yanghao Zhang, Siqi Sun, Tianle Zhang, Changshun Wu, Gaojie Jin, Yi Qi, Jinwei Hu, Jie Meng, Saddek Bensalem, and Xiaowei Huang. Safeguarding large language models: A survey. Artificial Intelligence Review, 58:382, 2025.

[7] Shaona Ghosh, Prasoon Shu, Chankyu Sathe, Saswat Pahari, Gyanendra Saroha, and Zhiyu Wang. Aegis AI Content Safety Dataset 2.0. arXiv preprint arXiv:2505.03789, 2025.

[8] Md Messal Monem Miah, Adrita Anika, Xi Shi, and Ruihong Huang. Hidden in plain sight: Evaluation of the deception detection capabilities of LLMs in multimodal settings. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 31013–31034. Association for Computational Linguistics, 2025.

[9] Peter S Park, Simon Goldstein, Aidan O’Gara, Michael Chen, and Dan Hendrycks. AI deception: A survey of examples, risks, and potential solutions. Patterns, 5(1):100901, 2024.

[10] Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 610–623, 2021.

[11] Isabel O Gallegos, Ryan A Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K Ahmed. Bias and fairness in large language models: A survey. Computational Linguistics, 50(3):1097–1179, 2024.

[12] Andy Zou, Zifan Wang, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

[13] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does LLM safety training fail? Advances in Neural Information Processing Systems, 36, 2024.

[14] Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

[15] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073, 2022.

[16] Dan Wilson and Mohamed Akrout. Low-cost black-box detection of LLM hallucinations via dynamical system prediction. arXiv preprint arXiv:2605.05134, 2026.

[17] Steven L Brunton and J Nathan Kutz. Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Cambridge University Press, 2019.

[18] J Nathan Kutz, Steven L Brunton, Bingni W Brunton, and Joshua L Proctor. Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems. Society for Industrial and Applied Mathematics, Philadelphia, PA, 2016.

[19] Peter J Schmid. Dynamic mode decomposition of numerical and experimental data. Journal of Fluid Mechanics, 656:5–28, 2010.

[20] Clarence W Rowley, Igor Mezi´c, Shervin Bagheri, Philipp Schlatter, and Dan S Henningson. Spectral analysis of nonlinear flows. Journal of Fluid Mechanics, 641:115–127, 2009.

[21] Bingni W Brunton, Lise A Johnson, Jefrey G Ojemann, and J Nathan Kutz. Extracting spatial-temporal patterns and transitions in high-dimensional biological data with dynamic mode decomposition. Journal of Neuroscience Methods, 258:1–15, 2016.

[22] Gary Froyland, Georg A Gottwald, and Andy Hammerlindl. Dynamic mode decomposition and transfer operator approaches for forecasting the enso cycle. Journal of Climate, 34(1):163–181, 2021.

[23] Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A Smith. RealToxicityPrompts: Evaluating neural toxic degeneration in language models. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 3356–3369, 2020.

[24] Amos Azaria and Tom Mitchell. The internal state of an LLM knows when it’s lying. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 967–976, 2023.

[25] Weihang Su, Changyue Wang, Qingyao Ai, Yiran Hu, Zhijing Wu, Yujia Zhou, and Yiqun Liu. Unsupervised real-time hallucination detection based on the internal states of large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 14379–14391, 2024.

[26] Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. INSIDE: LLMs’ internal states retain the power of hallucination detection. arXiv preprint arXiv:2402.03744, 2024.

[27] Xiaomeng Hu, Yiming Zhang, Ru Peng, Haozhe Zhang, Chenwei Wu, Gang Chen, and Junbo Zhao. Embedding and gradient say wrong: A white-box method for hallucination detection. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1950–1959, 2024.

[28] Junlang Qian, Zixiao Zhu, Hanzhang Zhou, Zijian Feng, Zepeng Zhai, and Kezhi Mao. Beyond the next token: Towards prompt-robust zero-shot classification via eficient multi-token prediction. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7093–7115, 2025.

[29] Guy Bar-Shalom, Fabrizio Frasca, Derek Lim, Yoav Gelberg, Yftah Ziser, Ran El-Yaniv, Gal Chechik, and Haggai Maron. Learning on LLM output signatures for gray-box behavior analysis. In ICML 2025 Workshop on Reliable and Responsible Foundation Models, 2025.

[30] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.

[31] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations, ICLR 2023. OpenReview.net, 2023.

[32] Potsawee Manakul, Adian Liusie, and Mark Gales. SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9004–9017, 2023.

[33] Jiaxin Zhang, Zhuohang Li, Kamalika Das, Bradley Malin, and Sricharan Kumar. SAC3: Reliable hallucination detection in black-box language models via semantic-aware cross-check consistency. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 15445–15458, 2023.

[34] Linggang Kong, Xiaofeng Zhong, Jie Chen, Haoran Fu, and Yongjie Wang. Multiperspective consistency checking for large language model hallucination detection: A blackbox zero-resource approach. Frontiers of Information Technology & Electronic Engineering, 26(11):2298–2309, 2025.

[35] Aman Goel, Daniel Schwartz, and Yanjun Qi. Zero-knowledge LLM hallucination detection and mitigation through fine-grained cross-model consistency. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 1982–1999, 2025.

[36] Albert Sawczyn, Jakub Binkowski, Denis Janiak, Bogdan Gabrys, and Tomasz Jan Kajdanowicz. FactSelfCheck: Fact-level black-box hallucination detection for LLMs. In Findings of the Association for Computational Linguistics: EACL 2026, pages 5603–5621, 2026.

[37] Mohamed Akrout and Dan Wilson. Guarantees on dynamical system distinguishability for llm token generation. arXiv preprint arXiv:2607.28667, 2026.

[38] Matthew O Williams, Ioannis G Kevrekidis, and Clarence W Rowley. A data-driven approximation of the Koopman operator: Extending dynamic mode decomposition. Journal of Nonlinear Science, 25(6):1307–1346, 2015.

[39] Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. HaluEval: A large-scale hallucination evaluation benchmark for large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6449–6464, 2023.

[40] Yiran Zhao, Jinghan Zhang, I Chern, Siyang Gao, Pengfei Liu, Junxian He, et al. FELM: Benchmarking factuality evaluation of large language models. Advances in Neural Information Processing Systems, 36:44502–44523, 2023.

[41] I. Mezi´c. Analysis of fluid flows via spectral properties of the Koopman operator. Annual Review of Fluid Mechanics, 45:357–378, 2013.

[42] M. Budiˇsi´c, R. Mohr, and I. Mezi´c. Applied Koopmanism. Chaos: An Interdisciplinary Journal of Nonlinear Science, 22(4):047510, 2012.

[43] J. N. Kutz, S. L. Brunton, B. W. Brunton, and J. L. Proctor. Dynamic mode decomposition: data-driven modeling of complex systems. Society for Industrial and Applied Mathematics, Philadelphia, PA, 2016.

[44] P. J. Schmid. Dynamic mode decomposition of numerical and experimental data. Journal of Fluid Mechanics, 656:5–28, 2010.

[45] C. W. Rowley, I. Mezic, S. Bagheri, P. Schlatter, and D. S. Henningson. Spectral analysis of nonlinear flows. Journal of Fluid Mechanics, 641(1):115–127, 2009.

[46] M. O. Williams, I. G. Kevrekidis, and C. W. Rowley. A data–driven approximation of the koopman operator: Extending dynamic mode decomposition. Journal of Nonlinear Science, 25(6):1307–1346, 2015.

[47] AI2 Adapt Dev. Synthetic CoT safety dataset. https://huggingface.co/datasets/ai2-adapt-dev/synthe 2024. Accessed: 2025.

[48] Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. BeaverTails: Towards improved safety alignment of LLM via a human-preference dataset. Advances in Neural Information Processing Systems, 36:24668–24686, 2023.

[49] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. Qwen3-VL-embedding and qwen3-VL-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.

[50] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7B. arXiv preprint arXiv:2310.06825, 2023.

[51] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

## Appendices

Appendix A: Koopman Operator Theory and Dynamic Mode Decomposition for Inferring Dynamics from Token Embeddings

Diferential residual scores from Section 3.2 are obtained by considering the token embedding dynamics of safe/unsafe prompts and responses. Following a strategy proposed in [16], we consider an LLM and its output as a dynamical system of the form

$$
\begin{array} { r l } & { \mathbf { x } _ { k + 1 } = F ( \mathbf { x } _ { k } ) , } \\ & { \quad \mathbf { q } _ { k } = G _ { 1 } ( \mathbf { x } _ { k } ) , } \\ & { \quad \mathbf { y } _ { k } = H ( \mathbf { x } _ { k } ) \equiv G _ { 2 } ( G _ { 1 } ( \mathbf { x } _ { k } ) ) , } \end{array}\tag{4}
$$

where $\mathbf { x } _ { k } \in \mathbb { R } ^ { N }$ are the state variables of an LLM immediately before the next token $( \mathrm { i . e . } ,$ , activation maps), $F$ governs the state evolution, $\mathbf q _ { k }$ is the token chosen by the LLM taken from its vocabulary, and $\mathbf { y } _ { k } \in \mathbb { R } ^ { M }$ is the associated token embedding. The nonlinear function $G _ { 1 }$ maps the internal state to a token and $G _ { 2 }$ maps the token to an embedding (e.g., via Qwen, Mistral, etc.). Koopman operator theory [41, 42] can be used consider the dynamics of observables, with the Koopman operator $K : \mathbb { R } ^ { M } \to \mathbb { R } ^ { M }$ defined according to

$$
K H ( \mathbf { x } _ { k } ) \equiv H ( F ( \mathbf { x } _ { k } ) ) .\tag{5}
$$

Despite the fact that the functions F and H are nonlinear, the Koopman operator is linear due to the linearity of the composition operator but generally infinite dimensional [42], [41].

Dynamic mode decomposition (DMD) is a data-driven method that can be used to obtain a finite-dimensional approximation for the action of the Koopman operator [43], [44], [45], [46]. Here, we consider the Extended DMD approach [46]. To implement this strategy, we first lift the observables to a higher dimensional space

$$
\begin{array} { r } { \mathbf { z } _ { k } = \left[ \mathbf { y } _ { k } ^ { \top } \quad f _ { \mathrm { l i f t } } ^ { \top } ( \mathbf { y } _ { k } ) \right] ^ { \top } , } \end{array}\tag{6}
$$

where $\mathbf { z } _ { k } \in \mathbb { R } ^ { M + \gamma }$ is the lifted token embedding. Here, $f _ { \mathrm { l i f t } } \in \mathbb { R } ^ { \gamma }$ where $\gamma$ is the dimension of the lifting and <sup>⊤</sup> indicates the vector transpose. This initial lifting step generally yields a more accurate representation of the Koopman operator; common choices of lifted coordinates include polynomial combinations of observables and radial basis functions [46]. In this work, we take the observable from (4) to be $\begin{array} { r } { \pmb { y } _ { k } = \pmb { \Phi } ^ { \top } H ( \mathbf { x } _ { k } ) } \end{array}$ where $\Phi \in \mathbb { R } ^ { M \times 5 0 0 }$ is comprised of the 500 most dominant SVD modes from the fitting data, as gauged by the singular values of the covariance matrix of the fitting data. The lifting function $f _ { \mathrm { l i f t } }$ takes polynomial combinations up to order 4 of a subset of the most dominant SVD modes.

A set of snapshot pairs $s _ { k } = ( \mathbf { z } _ { k } , \mathbf { z } _ { k + 1 } )$ is collected and arranged into matrices $X = \left[ \mathbf { z } _ { 1 } \ldots \mathbf { z } _ { q } \right]$ and $X ^ { + } = \left[ \mathbf { z } _ { 2 } \ldots \mathbf { z } _ { q + 1 } \right]$ where $q$ is the number of snapshot pairs used. An approximation of the Koopman operator $\mathbf { z } _ { k + 1 } = \mathbf { A } \mathbf { z } _ { k }$ can be obtained according to

$$
\mathbf { A } = \mathbf { X } ^ { + } \mathbf { X } ^ { \dagger } ,\tag{7}
$$

where $^ \dagger$ denotes the pseudoinverse. Low-rank approximations of A can be obtained by truncating small magnitude SVD modes of X before taking the pseudoinverse.