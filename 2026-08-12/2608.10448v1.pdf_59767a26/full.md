# RATIONALE-GUIDED LEARNING FOR MULTIMODAL EMOTION RECOGNITION

Sujung Oh<sup>1</sup>, Jung Uk Kim<sup>2∗</sup>, Sangmin Lee<sup>3∗</sup>

<sup>1</sup>Pixel Lab, Sungkyunkwan University, South Korea <sup>2</sup>Visual AI Lab, Kyung Hee University, South Korea <sup>3</sup>Pixel Lab, Korea University, South Korea

## ABSTRACT

Multimodal emotion recognition in conversation (MERC) requires understanding complex interactions between verbal and non-verbal cues. However, most existing approaches fundamentally treat this as a direct input-output (multimodal cues-emotion labels) mapping problem, overlooking the causal reasoning that humans use when interpreting emotions. We propose rationale-guided learning (RGL), a novel framework that transforms MERC into a cognitively-inspired reasoning task. Based on dual-process theory, we decompose emotional reasoning into three facets: Intuitive (immediate perception, System 1), Contextual (situational analysis, System 2), and Integrative (synthesis of both). We leverage a Multimodal Large Language Model (MLLM) to generate structured rationales, which are encoded as rationale banks to guide model training via aligning internal representations with human-like reasoning patterns. Our final model operates without any MLLM overheads at inference time. Experimental results show that RGL achieves state-of-the-art performance on the IEMOCAP and MELD benchmarks. Further, for interpretation, we demon strate that the model’s internal features effectively retrieve semantically correct rationales for unseen test samples, validating its rationale reasoning capabilities.

Index Terms— Multimodal emotion recognition, rationaleguided learning, reasoning patterns, representation learning, multimodal large language model

## 1. INTRODUCTION

Multimodal emotion recognition in conversation (MERC) aims to identify the emotional state of speakers within dialogues by leveraging text, audio, and video streams [1]. Unlike analyzing isolated clips, MERC requires a deep understanding of conversational context where emotions evolve through complex multimodal interplay. This contextual understanding is crucial for building truly empathetic interactive systems [2].

The field has progressed through several architectural shifts to better capture conversational dynamics. Initial approaches based on Recurrent Neural Networks (RNNs) [3, 4] modeled dialogue sequentially, but struggled with long-range dependencies. Transformer-based models [5, 6, 7] addressed this by leveraging self-attention to capture distant contextual cues. Subsequently, Graph Neural Networks (GNNs) [8, 9, 10] were introduced to explicitly represent speakers and utterances as nodes, allowing more nuanced propagation of context that reflects the multi-party nature of the conversation. Alongside these architectural shifts, recent work has also emphasized robust multimodal fusion and generalization, including cross-modal knowledge distillation [11], dynamic attention mechanisms [12], and context-aware contrastive learning [13].

However, despite this progress, current approaches still suffer from a fundamental limitation. They treat emotion recognition as a direct mapping problem from raw inputs to emotion labels, focusing on predicting ‘what’ the final emotion is. This overlooks the causal reasoning humans use to understand ‘how’ verbal and non-verbal cues interact to convey an emotional state. For instance, wide eyes can signify fear in one context but joyful surprise in another, a nuance that rationale-free models often miss. Without modeling such rationale-driven processes, they are prone to learning superficial shortcuts from spurious correlations between input cues and emotions.

To address this issue, we propose Rationale-Guided Learning (RGL) for Multimodal Emotion Recognition, a novel framework that injects human-like, rationale-guided reasoning into the process using a Multimodal Large Language Model (MLLM). Crucially, the MLLM is leveraged only once during the offline training preparation step to generate rationales, allowing our final model to remain efficient without requiring any MLLM overheads at inference time. Our approach is inspired by the dual-process theory of human cognition [14], which distinguishes between fast, automatic System 1 and slow, deliberate System 2. Based on this theory, we decompose this reasoning into three explicit facets: Intuitive rationales mirror the rapid perception of facial cues (System 1), Contextual rationales reflect the deliberate analysis of the situation (System 2), and Integrative rationales synthesize both (System 1 & 2) to form a coherent conclusion. These pre-generated rationales are transformed into intermediate supervision signals that guide our model to internalize these reasoning pathways, prioritizing causal understanding over direct input-output mappings.

![](images/755bea9f6d556f7fea0b0660c17d70bd055ee86a58b62288cf1de53ed099e50f.jpg)  
Fig. 1: An overview of the RGL architecture’s two main stages. (Top) An MLLM generates structured rationales (Intuitive, Contextual, Integrative) to construct rationale banks offline. (Bottom) A compact model is trained with contrastive losses to align its internal representations with the rationale vectors from the banks, fostering a human-like reasoning process.

Our main contributions are summarized as follows:

• We propose RGL, a novel rationale-aware framework for MERC that leverages MLLM to inject human-like, rationale-guided reasoning. It enables models to learn reasoning patterns rather than superficial predictions.

• We propose a three-facet rationale decomposition: Intuitive, Contextual, and Integrative rationales. We utilize rationale features for intermediate guidance, enhancing the robustness without any inference overhead.

• Through comprehensive experiments on IEMOCAP and MELD benchmarks, we demonstrate that RGL outperforms existing state-of-the-art methods, validating the effectiveness of our rationale-aware approach.

## 2. PROPOSED METHOD

Our proposed framework RGL, is designed to explicitly guide the reasoning process using rationales. The overall architecture is illustrated in Fig. 1. The process consists of two primary stages: (1) Offline phase for generating three types of rationale banks (Intuitive, Contextual, Integrative) by leveraging the reasoning of an MLLM, and (2) training phase for an emotion recognition model to learn rationale-guided reasoning patterns while predicting the target emotion.

## 2.1. Rationale generation

The cornerstone of RGL lies in structured rationale banks, generated offline by an MLLM (GPT-4o [15]) to emulate the cognitive system from dual-process theory [14]. This onetime offline process ensures our model operates without the overhead of running the MLLM at training and inference time.

We prompt the MLLM with the multimodal inputs (video frames and dialogue text) and the ground-truth emotion label for leveraging the reasoning power. The prompt is meticulously designed to guide the MLLM through a three-step analytical process, forcing it to deconstruct its reasoning into distinct, cognitively-motivated facets:

• Intuitive rationale (r<sub>I</sub>): This facet is designed to capture System 1 processing, which involves the immediate, automatic perception of evidence. The MLLM is instructed to describe only the objective facial muscle configurations (e.g., “eyebrows are lowered and drawn together”) without using any emotional terminology.

• Contextual rationale (r ): This facet models System 2 reasoning, specifically the slower, more deliberate analysis of the surrounding situation. The MLLM identifies the specific conversational event (e.g., “the speaker is informed their work has been shut down”) that likely triggered the emotion, which requires a deeper understanding of the dialogue’s narrative.

• Integrative rationale (r<sub>G</sub>): This represents the final synthesis where the outputs of System 1 (Intuitive) and System 2 (Contextual) are logically connected. The MLLM formulates an explanation that justifies emotion label by combining the observed cues with the situational trigger.

This three-step process yields a dataset of textual descriptions for each training sample. These texts are then encoded using a pre-trained text embedder (BGE-large-en-v1.5[16]) to create dense vector representations, denoted as the rationales $\{ r _ { \mathrm { I } } , r _ { \mathrm { C } } , r _ { \mathrm { G } } \}$ . These rationale vectors are organized into three distinct banks, $B _ { \mathrm { I } } , B _ { \mathrm { C } }$ , and $\boldsymbol { B } _ { \mathrm { G } }$ , corresponding to the Intuitive, Contextual, and Integrative facets, respectively. They serve as supervisory targets for our emotion recognition model. We refer to the combination of $\displaystyle B _ { \mathrm { I } } , \ : B _ { \mathrm { C } } ,$ , and $\boldsymbol { B } _ { \mathrm { G } }$ as the rationale banks B.

## 2.2. Model architecture

The trainable part of RGL is a compact, end-to-end network consisting of unimodal encoders and a multimodal fusion module.

Unimodal encoders. To extract modality-specific features, our model processes visual (V), textual (T), and audio (A) modalities using standard pre-trained backbones: ViT-base [17], RoBERTa-large [18], and HuBERT-base [19], respectively. The visual and textual encoders are designed with a dual-head architecture to output two distinct representations: (1) the main feature $f _ { \mathrm { m a i n , V } }$ and $f _ { \mathrm { m a i n , T } }$ for the primary emotion prediction, (2) rationale feature $f _ { \mathrm { r a t , V } }$ and $f _ { \mathrm { r a t , T } }$ specifically for aligning with rationale banks. The audio encoder outputs a single main feature, denoted as $f _ { \mathrm { A } }$ . This dual-head design decouples the tasks, enabling targeted rationale alignment without interfering with the main classification objective.

Multimodal fusion. The main features from all encoders $\{ f _ { \mathrm { m a i n , V } } , f _ { \mathrm { m a i n , T } } , f _ { \mathrm { A } } \}$ are first concatenated and then processed by a stack of Transformer encoder layers [20] to model crossmodal interactions. This captures complex, cross-modal interactions through self-attention, yielding a sequence of contextually enriched hidden states $\dot { \mathbf { H } } \in \mathbb { R } ^ { L \times D }$ , where L is the input sequence length and D is the hidden dimension. To aggregate these sequential states into a single vector, $f _ { \mathrm { f u s e d } }$ , we employ attention pooling [21], which dynamically weighs the importance of each token. Finally, this vector $f _ { \mathrm { f u s e d } }$ is projected through two task-specific heads: an MLP classifier for emotion prediction, and a rationale head for the rationale-guided reasoning objective.

## 2.3. Rationale-guided representation learning

The core of our training is to align the model’s rationale features $( f _ { \mathrm { r a t , V } } , \ f _ { \mathrm { r a t , T } }$ , and $f _ { \mathrm { r a t , F } } )$ with their corresponding rationales from the pre-computed banks, B. This alignment is achieved through a contrastive learning objective. The objec tive pulls each model representation $( f ^ { ( i ) } )$ , referred to as the anchor, towards its corresponding target rationale $( r ^ { ( i ) } )$ from the bank, which forms a positive pair. Simultaneously, the objective pushes the anchor away from rationales of different emotions, which form negative pairs.

To make this process more effective, we employ a hard negative mining strategy. For each anchor $f ^ { ( i ) }$ , we construct a set of hard negatives $\mathcal { N } _ { K } ^ { ( i ) }$ by sampling from the rationale banks B. Specifically, we first form a candidate pool by excluding all rationales that share the same emotion label as the positive pair $r ^ { ( i ) }$ . Then, from this pool, we retrieve the top-K $( K = 1 2 8 )$ most similar negative samples that have the highest cosine similarity to the anchor $f ^ { ( i ) }$ . This approach forces the model to learn finer-grained distinctions between semantically close yet emotionally distinct concepts, moving beyond simple class separation. For brevity, let $\bar { s _ { i } ^ { + } } = \mathrm { s i m } \bar { ( } f ^ { ( i ) } , r ^ { ( i ) } )$ ) denote the similarity score for the positive pair, and $s _ { i k } ^ { - } = \sin ( f ^ { ( i ) } , r _ { k } )$ for a negative pair $r _ { k } \in \mathcal { N } _ { K } ^ { \left( i \right) }$ . The rationale loss for a sample i is then defined as:

$$
\mathcal { L } _ { \mathrm { r a t } } ^ { ( i ) } = - \log \frac { \exp ( s _ { i } ^ { + } / \tau ) } { \exp ( s _ { i } ^ { + } / \tau ) + \sum _ { k = 1 } ^ { K } \exp ( s _ { i k } ^ { - } / \tau ) } ,\tag{1}
$$

where sim $( \cdot , \cdot )$ is the cosine similarity between two vectors, $\mathcal { N } _ { K } ^ { ( i ) }$ is the set of K hard negatives for sample i, and τ is a temperature hyperparameter.

This alignment process is designed to mirror a cognitive reasoning pipeline inspired by the human dual-process theory.

1. Aligning visual features with Intuitive rationale: We ground the model’s understanding in fast, intuitive rationale. We align the visual rationale representation $f _ { \mathrm { r a t , V } }$ with the Intuitive rationale vector $r _ { \mathrm { I } }$ from its corresponding bank $\boldsymbol { B } _ { \mathrm { I } }$ using the loss $\mathcal { L } _ { \mathrm { r a t , I } }$

2. Aligning textual features with Contextual rationale: We train the model to be aware of slow, analytical reasoning of contexts from the dialogue. We align the textual rationale representation $f _ { \mathrm { r a t , T } }$ with the Contextual rationale $r _ { \mathrm { C } }$ from $\boldsymbol { B _ { \mathrm { C } } }$ via ${ \mathcal { L } } _ { \mathrm { r a t , C } }$

3. Aligning fused features with Integrative rationale: Finally, we align the fused rationale representation $f _ { \mathrm { r a t , F } }$ with the Integrative rationale vector $r _ { \mathrm { G } }$ from $\boldsymbol { B } _ { \mathrm { G } }$ based on $\mathcal { L } _ { \mathrm { r a t , G } }$ This step guides the model to synthesize both Intuitive and Contextual insights, forming a coherent inference. This staged alignment ensures meaningful unimodal representations are learned first, providing a robust foundation for the final synthesis.

As a result, a final training objective can be formulated as:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \underbrace { { \mathcal { L } } _ { \mathrm { C E } } } _ { \mathrm { E m o t i o n } \mathrm { C l a s s i f i c a t i o n } } + \lambda \underbrace { { \left( { \mathcal { L } } _ { \mathrm { r a t , I } } + { \mathcal { L } } _ { \mathrm { r a t , C } } + { \mathcal { L } } _ { \mathrm { r a t , G } } \right) } } _ { \mathrm { R a t i o n a l e - G u i d e d ~ A l i g n m e n t } } ,\tag{2}
$$

where $\mathcal { L } _ { \mathrm { C E } }$ represents a cross-entropy loss, each rationale loss term $\mathcal { L } _ { \mathrm { r a t } , \mathrm { X } }$ (for $X \in \{ I , C , G \} )$ is computed as defined in Eq. (1), and λ is a hyperparameter that balances the rationaleguided objectives, thereby training RGL to structure its embedding space in a way that mirrors a logical, human-like reasoning process.

## 3. EXPERIMENTS

## 3.1. Datasets and implementation details

Datasets. We conduct experiments on two widely adopted datasets: IEMOCAP [29] and MELD [30]. IEMOCAP is a dyadic dataset for which we use six standard emotion categories: ‘neutral’, ‘sad’, ‘angry’, ‘happy’, ‘excited’, and ‘frustrated’. MELD is a multi-party dataset extracted from the TV show “Friends”, containing seven emotion labels: ‘anger’, ‘disgust’, ‘fear’, ‘joy’, ‘neutral’, ‘sadness’, and ‘surprise’.

Table 1: Performance comparison with existing methods on IEMOCAP and MELD datasets.
<table><tr><td rowspan="2">Method</td><td colspan="2">IEMOCAP</td><td colspan="2">MELD</td></tr><tr><td>W-F1</td><td>Acc</td><td>W-F1</td><td>Acc</td></tr><tr><td>DialogueRNN[3] (AAAI&#x27;19)</td><td>62.75</td><td>63.40</td><td>-</td><td>-</td></tr><tr><td>DialogueTRM[5] (EMNLP&#x27;21)</td><td>69.7</td><td>69.5</td><td>63.50</td><td>65.70</td></tr><tr><td>MM-DFN[22] (ICASSP&#x27;22)</td><td>68.18</td><td>68.21</td><td>59.46</td><td>62.49</td></tr><tr><td>SCFA[7] (INTERSPEECH&#x27;23)</td><td>66.42</td><td>67.91</td><td>63.69</td><td>64.86</td></tr><tr><td>FacialMMT[23] (ACL&#x27;23)</td><td></td><td></td><td>66.58</td><td></td></tr><tr><td>EASUM[24] (WACV&#x27;24)</td><td>69.75</td><td>70.10</td><td>65.93</td><td>66.70</td></tr><tr><td>TelME[11] (NAACL&#x27;24)</td><td>70.48</td><td>一</td><td>67.37</td><td>一</td></tr><tr><td>HAUCL[25] (ACM MM&#x27;24)</td><td>70.27</td><td>70.30</td><td>66.72</td><td>68.05</td></tr><tr><td>BIG-FUSION[26] (AAAI&#x27;25)</td><td>72.91</td><td>72.64</td><td>67.17</td><td>68.24</td></tr><tr><td>DIB-HGCN[27] (AAAI&#x27;25)</td><td>72.46</td><td></td><td>66.61</td><td></td></tr><tr><td></td><td></td><td>72.58</td><td></td><td>68.01</td></tr><tr><td>MAGTKD[28] (IJCAI’25)</td><td>69.59</td><td>69.38</td><td>65.32</td><td>66.36</td></tr><tr><td>RGL (Ours)</td><td>73.68</td><td>73.51</td><td>67.43</td><td>68.31</td></tr></table>

Table 2: Ablation study of RGL’s components on the IEMO-CAP test set. The full model’s performance is in bold.
<table><tr><td>Model Configuration</td><td>W-F1</td><td>Acc</td></tr><tr><td>RGL (Full Model)</td><td>73.68</td><td>73.51</td></tr><tr><td>w/o Intuitive loss  $( \mathcal { L } _ { \mathrm { r a t , I } } )$ </td><td>72.70</td><td>72.52</td></tr><tr><td>w/o Contextual loss  $( \mathcal { L } _ { \mathrm { r a t , C } } )$ </td><td>68.78</td><td>68.70</td></tr><tr><td>wlo Integrative loss  $( \mathcal { L } _ { \mathrm { r a t , G } } )$ </td><td>72.44</td><td>72.34</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { r a t , I } } , { \mathcal { L } } _ { \mathrm { r a t , C } } , { \mathcal { L } } _ { \mathrm { r a t , G } }$ </td><td>68.01</td><td>67.71</td></tr></table>

Implementation details. We train using the AdamW optimizer with a learning rate of 1e−5 and a batch size of 4. The temperature parameter in Eq. (1) is τ = 0.07 , and the hyperparameter in Eq. (2) is set to λ = 0.3. For video streams, we follow FacialMMT [23] and apply TalkNet-ASD [31] to detect the face of the active speakers based on vocal activity.

## 3.2. Performance evaluation

We evaluate performance using two standard metrics, Weighted F1 (W-F1) and accuracy (Acc), following [8, 25].

Performance comparison. As shown in Table 1, our RGL achieves state-of-the-art results on both datasets. The consistent improvements across two different settings, dyadic interactions on IEMOCAP and multi-party conversations in MELD, provide strong evidence for our hypothesis that explicitly supervising internal representations with structured cognitive rationales is effective.

Ablation studies. We also conduct ablation studies to verify the contribution of our proposed rationale designs. As shown in Table 2, removing all losses simultaneously causes the most significant drop in performance, confirming their overall

![](images/0b56bcf5265bc702fbe0e622313199d101aa96eedef8ce6d45253d988ea4a395.jpg)  
Fig. 2: Rationale retrieval example on an unseen test sample. importance. The results reveal that $\mathcal { L } _ { \mathrm { { r a t } , \mathrm { { C } } } }$ is the most critical component, while $\mathcal { L } _ { \mathrm { r a t , G } }$ and $\mathcal { L } _ { \mathrm { r a t , I } }$ are also effective.

## 3.3. Reasoning interpretation

To validate that RGL genuinely learns reasoning patterns, we analyze how it leverages its learned rationale banks for unseen test samples. For a given test case, we extract the model’s internal rationale representations $( f _ { \mathrm { r a t , V } } , f _ { \mathrm { r a t , T } } , f _ { \mathrm { r a t , F } } )$ and use them as queries to retrieve the most similar rationales from the training rationale banks. Figure 2 shows this capability on a challenging sample, where successful retrieval requires a deep semantic understanding beyond superficial cues. The model accesses an Intuitive rationale from rationale bank that aligns with the facial expression and a Contextual rationale relevant to the “loss of people.” Furthermore, the retrieved Integrative rationale correctly synthesizes both aspects. This confirms that RGL learns a robust mapping from raw multimodal signals to a structured, semantically meaningful rationale space, proving its internal features are grounded in human-like reasoning.

## 4. CONCLUSION

We introduce Rationale-Guided Learning (RGL) for Multimodal Emotion Recognition, a novel framework that trains a model by aligning its representations with cognitive rationales generated offline by an MLLM. Achieving state-of-the-art results on IEMOCAP and MELD, our work demonstrates that emulating cognitive reasoning is an effective approach for advancing multimodal emotion recognition.

Acknowledgement. This work was supported in part by the NRF grant funded by the Korea government(MSIT) (RS-2025-00563942), the IITP grant funded by the Korea government(MSIT)(IITP-2026-RS-2020-II201819, 20%), and the IITP-ITRC grant funded by the Korea government(MSIT) (IITP-2026-RS-2023-00258649, 30%).

## 5. REFERENCES

[1] S. Poria et al., “A review of affective computing: From unimodal analysis to multimodal fusion,” Information Fusion, vol. 37, pp. 98–125, 2017.

[2] R. W. Picard, Affective Computing. MIT Press, 1997.

[3] N. Majumder et al., “Dialoguernn: An attentive rnn for emotion detection in conversations,” in AAAI, 2019, pp. 6818–6825.

[4] D. Hazarika et al., “Icon: Interactive conversational memory network for multimodal emotion detection,” in EMNLP, 2018, pp. 2594–2604.

[5] Y. Mao et al., “DialogueTRM: Exploring multi-modal emotional dynamics in a conversation,” in Findings of EMNLP, 2021, pp. 2694–2704.

[6] X. Zhang et al., “A cross-modality context fusion and semantic refinement network for emotion recognition in conversation,” in ACL-Long, 2023, pp. 13 099–13 110.

[7] H. Zhao et al., “Speaker-aware cross-modal fusion for conversational emotion recognition,” in INTERSPEECH, 2023, pp. 2718–2722.

[8] Y. Shou et al., “Dynamic graph neural ODE network for multi-modal emotion recognition in conversation,” in COLING, 2025, pp. 256–268.

[9] D. Ghosal et al., “Dialoguegcn: A graph convolutional neural network for emotion recognition in conversation,” in EMNLP, 2019, pp. 154–164.

[10] J. Hu et al., “Mmgcn: Multimodal fusion via deep graph convolution network for emotion recognition in conversation,” in ACL-Long, 2021, pp. 5666–5675.

[11] T. Yun et al., “Telme: Teacher-leading multimodal fusion network for emotion recognition in conversation,” in NAACL-Long, 2024, pp. 82–95.

[12] Y. Jing et al., “Dq-former: Querying transformer with dynamic modality priority for cognitive-aligned multimodal emotion recognition in conversation,” in ACM MM, 2024, pp. 4795–4804.

[13] Y. Xie et al., “A dual contrastive learning framework for enhanced multimodal conversational emotion recognition,” in COLING, 2025, pp. 4055–4065.

[14] J. S. B. T. Evans et al., “Dual-process theories of higher cognition: Advancing the debate,” Perspect. Psychol. Sci., vol. 8, no. 3, pp. 223–241, 2013.

[15] OpenAI et al., “Gpt-4o system card,” arXiv, 2024.

[16] S. Xiao et al., “C-pack: Packed resources for general chinese embeddings,” in ACM SIGIR, 2024, pp. 641– 649.

[17] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in ICLR, 2021.

[18] Y. Liu et al., “Roberta: A robustly optimized BERT pretraining approach,” arXiv, 2019.

[19] W.-N. Hsu et al., “Hubert: Self-supervised speech representation learning by masked prediction of hidden units,” in ACM-TASLP, 2021, pp. 3451–3460.

[20] A. Vaswani et al., “Attention is all you need,” in NeurIPS, 2017, p. 6000–6010.

[21] Z. Lin et al., “A structured self-attentive sentence embedding,” in ICLR, 2017.

[22] D. Hu et al., “Mm-dfn: Multimodal dynamic fusion network for emotion recognition in conversations,” in ICASSP, 2022, pp. 7037–7041.

[23] W. Zheng et al., “A facial expression-aware multimodal multi-task learning framework for emotion recognition in multi-party conversations,” in ACL-Long, 2023, pp. 15 445–15 459.

[24] Y. Hwang et al., “Easum: Enhancing affective state understanding through joint sentiment and emotion modeling for multimodal tasks,” in WACV, 2024, pp. 5668– 5678.

[25] Z. Yi et al., “Multimodal fusion via hypergraph autoencoder and contrastive learning for emotion recognition in conversation,” in ACM MM, 2024, p. 4341–4348.

[26] Y. Wang et al., “Big-fusion: Brain-inspired global-local context fusion framework for multimodal emotion recognition in conversations,” in AAAI, 2025, pp. 1574–1582.

[27] X. Chen et al., “Dynamic interactive bimodal hypergraph networks for emotion recognition in conversations,” in AAAI, 2025, pp. 1256–1264.

[28] J. Li et al., “Multi-modal anchor gated transformer with knowledge distillation for emotion recognition in conversation,” in IJCAI, 2025.

[29] C. Busso et al., “Iemocap: Interactive emotional dyadic motion capture database,” in LREC, 2008, pp. 335–339.

[30] S. Poria et al., “Meld: A multimodal multi-party dataset for emotion recognition in conversations,” in ACL, 2019, pp. 527–536.

[31] R. Tao et al., “Is someone speaking? exploring longterm temporal features for audio-visual active speaker detection,” in ACM MM, 2021, p. 3927–3935.