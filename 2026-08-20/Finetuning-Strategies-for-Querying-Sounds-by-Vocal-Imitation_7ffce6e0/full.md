# Finetuning Strategies for Querying Sounds by Vocal Imitation

Aditya Bhattacharjee, Christos Plachouras, Sungkyun Chang, Emmanouil Benetos

School of Electronic Engineering and Computer Science, Queen Mary University of London, UK

Abstract—This technical report describes our winning submission to the AES AIMLA 2025 Challenge on querying sound effects by vocal imitation. We investigate two complementary finetuning strategies: contrastive learning with a frozen, pretrained CED encoder, and joint contrastive-triplet learning with semihard negatives using a MobileNetV3 encoder. This report has been updated for posterity to include details released after the challenge.

## I. INTRODUCTION

Query by Vocal Imitation (QbVI) involves retrieving audio samples from a reference collection based on a vocal imitation provided by the user. This problem is particularly challenging due to the variability in human vocalizations and the acoustic diversity of sound classes. Effective systems must learn representations that bridge the modality gap between vocal and non-vocal audio, while also being robust to the diverse ways humans imitate sounds.

Early approaches to QbVI focused on latent bottleneck features from autoencoders [1] or semi-Siamese convolutional architectures trained with contrastive loss [2]. These methods typically relied on hand-crafted or moderately learned features, combined with traditional classifiers or similarity measures such as dynamic time warping or cosine distance.

More recently, Greif et al. [3] demonstrated the effectiveness of contrastive learning with neural audio embeddings pretrained on large-scale datasets like AudioSet. Their system employed a dual-tower MobileNetV3 architecture, fine-tuned using SimCLR-style contrastive objectives and an extensive augmentation pipeline. This approach set a strong baseline for neural QbVI systems.

In this work, we focus specifically on neural network-based representations for QbVI and explore two complementary submission strategies for the AES-AIMLA 2025 Challenge. Our first submission involves contrastive fine-tuning of a frozen Consistent Ensemble Distillation (CED) encoder [4], which was originally trained for audio tagging via knowledge distillation. Our second submission builds upon the MobileNetV3 architecture and the setup of Greif et al., but integrates a triplet-based regularization objective. The motivation is to improve generalization by leveraging hard or semi-hard negatives sampled from a curated unpaired set of vocal imitations.<sup>1</sup>

## II. DATASETS

We use two primary datasets for training:

• VimSketch Dataset [5]: This dataset provides supervised (reference, imitation) pairs with known correspondence. These pairs are used directly in our contrastive learning objective.

• VocalSketch Dataset [6]: This dataset contains vocal imitations that were excluded from the curated VimSketch set. Although they do not have matched reference samples, they are labeled by sound class. We use these as structured negatives in a triplet learning framework, sampling same-class negatives when available and falling back to random sampling otherwise.

In our first submission, only the VimSketch dataset is used. The second submission incorporates both datasets to enable a hybrid training regime that combines contrastive and triplet learning.

We use an online augmentation methodology on both query and reference data. This strategy is used in both our submissions to improve robustness to variation in vocal imitations and reference recordings. At each training step, a random subset of transformations is applied to each audio sample.

The augmentations include time-domain operations such as time shifting, gain adjustment, pitch shifting, and time stretching. In addition, we incorporate light additive Gaussian noise and frame-level corruptions to simulate temporal dropouts and distortions common in human vocalizations. Specifically, each sample is passed through up to one transformation randomly selected from a pool that includes:

• Shift: Random temporal offset of the waveform within ±30% of its length.

• Gain: Random amplification or attenuation between −10 and +10 dB.

• Pitch Shift: Semitone shift within a range of ±3 semitones.

• Time Stretch: Temporal scaling by a factor between 0.8 and 1.2.

• Additive Gaussian Noise: Injects low-amplitude noise with amplitude sampled between 0.001 and 0.015.

• Frame-Level Corruption: Applies stochastic frame-wise duplication, silencing, or removal to simulate human inconsistency.

• Frequency and Time Masking: Randomly masks contiguous frequency bins or time frames in the spectrogram to increase robustness to partial input corruption.

This online augmentation is applied independently to both the query and reference branches of the model during training.

It plays a crucial role in improving generalization across different vocalization styles, recording conditions, and intraclass variation.

For our validation dataset, we use the official qvim-dev set provided as part of the QVIM 2025 Challenge. Evaluation is performed using the Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG) metrics.

## III. SUBMISSION #1

## A. Architecture and Training Setup

Our first submission is based on the Consistent Ensemble Distillation (CED) framework, originally introduced by Dinkel et al. [4]. CED is a ViT-based audio encoder trained for AudioSet audio tagging using consistent ensemble distillation. It performed strongly on the HEAR 2021 benchmark, particularly for vocal imitation classification tasks, a motivation for its inclusion here.

In this submission, we freeze the pretrained CED-base encoder (768-dimensional output) and train a lightweight MLP projection head to map embeddings to a 256-dimensional space suitable for retrieval. The encoder is shared across both the query and reference branches. Audio is resampled to 16 kHz and processed using the pretrained CED feature extractor. We extract the final encoder hidden representation, average-pool it over the sequence dimension to obtain a 768- dimensional embedding, and keep the CED encoder frozen during training.

The training objective is a supervised contrastive loss applied to known (reference, imitation) pairs from the VimSketch dataset. In-batch negatives are used to structure the embedding space.

We apply online data augmentation symmetrically to both the query and reference audio inputs. Each training example receives a randomly selected transformation from a curated pool of time-domain effects; namely, the ones discussed in Section II.

## B. Implementation

The system uses the AdamW optimizer with a cosine learning rate schedule (starting at $1 \times 1 0 ^ { - 3 } )$ , weight decay of $1 \times 1 0 ^ { - 5 }$ , and a batch size of 128. The model is trained for 50 epochs, with evaluation performed after every epoch. The temperature τ for the contrastive loss is fixed at 0.07. The output projection is a linear layer mapping the encoder’s 768- dimensional output to a 256-dimensional embedding, which is then used to compute cosine similarities for retrieval.

## IV. SUBMISSION #2

## A. Architecture and Training Setup

Our model architecture builds on the QbVI framework proposed by Greif et al. [3], using a MobileNetV3 encoder pretrained on AudioSet [7]. In contrast to the dual-tower encoder architecture of [3], we use a single shared encoder for both reference and imitation audio, promoting tighter alignment in the learned embedding space and reducing parameter overhead. The encoder outputs are L2-normalized, enabling cosine similarity as a metric for both the loss computation and retrieval.

We adopt a hybrid learning strategy where supervised contrastive learning is regularized by a triplet loss. Contrastive loss is applied over positive pairs sampled from the VimSketch dataset, while the triplet loss uses negative examples drawn from the excluded subset of the VocalSketch dataset. These excluded examples consist of practice recordings and human-rejected imitations and do not participate in standard contrastive supervision. For triplet construction, negatives are sampled based on shared AudioSet ontology labels where available, falling back to random negatives otherwise. A semihard mining strategy is used, selecting negatives that are closer to the anchor than easy negatives but still harder than positives.

To adaptively balance the contribution of both losses during training, we implement a dynamic weighting strategy where the weight assigned to the triplet loss is scaled based on the number of active semi-hard triplets in the batch. If a few valid triplets are found, the triplet loss is down-weighted to avoid destabilizing training.

## B. Implementation

The input features closely follow AudioSet’s preprocessing pipeline, where all audio is resampled to 32 kHz and converted to 128-band log-Mel spectrograms using a 10-second duration, a window size of 800 samples, a hop size of 320, and an FFT size of 1024. Online data augmentation, as described earlier, is applied independently to both the query and reference inputs during training to increase robustness to vocal variation and noise. 960-dimensional embeddings are output from the penultimate layer of the encoder; the classifier head of the encoder is discarded.

We train using the AdamW optimizer and a cosine learning rate schedule with one warmup epoch. The maximum learning rate is set to $2 \times 1 0 ^ { - 4 }$ and the minimum to $5 \times 1 0 ^ { - 5 }$ . Training runs for 30 epochs with a batch size of 64. We fix the contrastive temperature to $\tau = 0 . 0 7$ and the triplet margin to 0.6.

## V. RESULTS

The evaluation follows the official QVIM 2025 validation protocol, using the ‘qvim-dev‘ set provided by the organizers. The validation set includes both query-reference pairings and class metadata, enabling both pair-wise and class-wise evaluation.

The primary evaluation metric is the Mean Reciprocal Rank (MRR) computed over known query-reference pairs. Secondary metrics include Normalized Discounted Cumulative Gain (NDCG) and class-wise MRR, which assess ranking quality and class-level retrieval consistency, respectively.

Table I summarizes the validation performance for both of our submissions, in comparison to the baselines. For posterity, we report that Submission #2 is the winning submission in the AES AIMLA 2025 Challenge on querying sound effects by vocal imitation. Further details about the objective and subjective evaluation on test datasets can be found in the challenge website <sup>2</sup>.

TABLE I  
PERFORMANCE METRICS
<table><tr><td>Model Name</td><td> $\mathbf { M R R } _ { e x }$ </td><td> $\mathbf { N D C G } _ { c a t }$ </td></tr><tr><td>random</td><td>0.0444</td><td>0.337</td></tr><tr><td>2DFT</td><td>0.1262</td><td>0.4793</td></tr><tr><td>Greif et al.</td><td>0.2726</td><td>0.6463</td></tr><tr><td>Submission #1</td><td>0.2876</td><td>0.6600</td></tr><tr><td>Submission #2</td><td>0.2932</td><td>0.6468</td></tr></table>

## VI. CONCLUSION

The goal of the Query by Vocal Imitation (QbVI) task is to retrieve audio samples from a reference database based on a vocal imitation provided by the user. As part of our participation in the AES-AIMLA 2025 Challenge, we explored two complementary approaches aimed at improving neuralnetwork-based retrieval systems for QbVI. The submission differ in encoder architecture, training strategy, and supervision design.

In Submission #1, we leverage a frozen CED-base encoder pretrained via knowledge distillation on AudioSet. A supervised contrastive loss is applied over known referenceimitation pairs, using in-batch negatives to structure the embedding space. This lightweight approach achieves strong retrieval performance with minimal fine-tuning. In Submission #2, we extend the training pipeline by fine-tuning a shared MobileNetV3 encoder using a hybrid loss combining contrastive and triplet objectives. Semi-hard negative mining is performed using excluded vocal imitations, which are sampled based on class labels. A dynamic weighting strategy balances the two losses based on the availability of active triplets.

Together, these submissions highlight the value of both transfer learning and curriculum design in QbVI systems. Whether through careful pretraining or structured negative sampling, both strategies yield significant gains in retrieval performance over baseline systems.

## REFERENCES

[1] Y. Zhang and Z. Duan, “Supervised and unsupervised sound retrieval by vocal imitation,” Journal of the Audio Engineering Society, vol. 64, no. 7/8, pp. 533–543, 2016.

[2] Y. Zhang and Z. Duan, “IMINET: Convolutional semi-Siamese networks for sound search by vocal imitation,” in IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). New Paltz, USA: IEEE, 2017, pp. 304–308.

[3] J. Greif, F. Schmid, P. Primus, and G. Widmer, “Improving queryby-vocal imitation with contrastive learning and audio pretraining,” in Detection and Classification of Acoustic Scenes and Events (DCASE), Tokyo, Japan, 2024.

[4] H. Dinkel, Y. Wang, Z. Yan, J. Zhang, and Y. Wang, “CED: Consistent ensemble distillation for audio tagging,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). Seoul, Republic of Korea: IEEE, 2024, pp. 291–295.

[5] F. Pishdadian, P. Seetharaman, B. Kim, and B. Pardo, “Classifying nonspeech vocals: Deep vs signal processing representations,” 2019.

[6] M. Cartwright and B. Pardo, “Vocalsketch: Vocally imitating audio concepts,” in Proceedings ofthe 33rd Annual ACM Conference on Human Factors in Computing Systems, Seoul, Republic of Korea, 2015, pp. 43– 46.

[7] J. F. Gemmeke, D. P. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, “Audio set: An ontology and humanlabeled dataset for audio events,” in IEEE international conference on acoustics, speech and signal processing (ICASSP). New Orleans, USA: IEEE, 2017, pp. 776–780.