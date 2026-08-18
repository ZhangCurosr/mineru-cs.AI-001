# Foresight-England: Development of a National-Scale Generative AI Model of Electronic Health Records for Medical Event Prediction across the COVID-19 Pandemic

Simon Ellershaw<sup>∗1,</sup> <sup>2</sup> Christopher Tomlinson\*<sup>†1,</sup> <sup>2,</sup> <sup>3,</sup> <sup>4</sup> Zeljko Kraljevic <sup>2</sup>

Spiros Denaxas <sup>1,</sup> <sup>4,</sup> <sup>5,</sup> <sup>6,</sup> <sup>7</sup> Harry Hemingway <sup>1,</sup> <sup>4,</sup> <sup>7</sup> Cathie Sudlow <sup>8</sup>

Angela M. Wood <sup>6,</sup> <sup>9,</sup> <sup>10,</sup> <sup>11,</sup> <sup>12,</sup> <sup>13,</sup> <sup>14</sup> Anoop D. Shah <sup>1,</sup> <sup>4,</sup> <sup>15</sup> Richard Dobson <sup>1,</sup> <sup>2,</sup> <sup>4,</sup> <sup>7,</sup> <sup>16</sup>

## on behalf of the CVD-COVID-UK/COVID-IMPACT Consortium

<sup>1</sup> Institute of Health Informatics, University College London, London, UK

<sup>2</sup> Department of Biostatistics and Health Informatics, Institute of Psychiatry, Psychology and Neuroscience, King’s College London, London, UK

<sup>3</sup> King’s Institute for Artificial Intelligence, King’s College London, London, UK

<sup>4</sup> University College London Hospitals National Institute for Health Research Biomedical Research Centre, London, UK

<sup>5</sup> Interdisciplinary Transformation University, Linz, Austria

<sup>6</sup> British Heart Foundation Data Science Centre, Health Data Research UK, London, UK

<sup>7</sup> Health Data Research UK, London, UK

<sup>8</sup> Usher Institute, School of Population Health Sciences, The University of Edinburgh

<sup>9</sup> British Heart Foundation Cardiovascular Epidemiology Unit, Department of Public Health and Primary Care, University of Cambridge, Cambridge, UK

<sup>10</sup> Victor Phillip Dahdaleh Heart and Lung Research Institute, University of Cambridge, Cambridge, UK

<sup>11</sup> British Heart Foundation Centre of Research Excellence, University of Cambridge, Cambridge, UK

<sup>12</sup> National Institute for Health and Care Research Blood and Transplant Research Unit in Donor Health and Behaviour, University of Cambridge, Cambridge, UK

<sup>13</sup> Health Data Research UK Cambridge, Wellcome Genome Campus and University of Cambridge, Cambridge, UK

<sup>14</sup> Cambridge Centre of Artificial Intelligence in Medicine, University of Cambridge, Cambridge, UK

<sup>15</sup> Department of Clinical Pharmacology, University College London Hospitals NHS Foundation Trust, London, UK

<sup>16</sup> National Institute for Health Research Biomedical Research Centre at South London and Maudsley NHS Foundation Trust and King’s College London, London, UK

## Abstract

Foresight-England (Foresight-E) is the first national-scale generative foundation model of electronic health records (EHRs), developed as a research pilot strictly for COVID-19-related research. We evaluated its ability to model the direct and indirect effects of the COVID-19 pandemic.

Trained from scratch entirely within the NHS England Secure Data Environment, Foresight-E is a 243-million-parameter transformer decoder. It is trained and evaluated on a national-scale, de-identified, longitudinal EHR dataset of approximately 61 million individuals, integrating primary and secondary care, death registrations, and COVID-19 testing/vaccination datasets. Training and validation were conducted on a random subset of 90% of individuals (54.9 million) for events recorded between 1st November 2018 and 31st December 2022. The remaining 10% of individuals (6.1 million) were held out for evaluation.

Foresight-E models patient timelines autoregressively, predicting the next medical event given an individual’s prior history. At inference, it operates zero-shot, generating predictions for any concept in its approximately 40,000-code medical vocabulary without additional task-specific training. Our custom tokenisation scheme retains the recorded clinical granularity of ICD-10, OPCS-4, and SNOMED CT codes, and jointly represents both absolute (e.g., calendar dates) and relative timing (e.g., chronological age).

We designed and implemented an evaluation framework spanning 30-day COVID-19 hospitalisation and mortality using Brier scores and area under the receiver operating characteristic (AUROC) and precision-recall (AUPRC) curves. Subgroup analysis of these results by age, ethnicity, sex and COVID vaccination status was also conducted. We also tested Foresight-E on medical events from 2023, extending beyond its 2018–2022 training period, to assess how well it captured the enduring, system-wide indirect effects of the pandemic on future unseen data, simulating a prospective deployment. We benchmarked model performance against logistic regression and XGBoost.

As detailed in the Project Status section, NHS England has paused access to data for the Foresight-E project, meaning quantitative results are not currently available. Instead, we share our strategy for tokenisation, model architecture, training, inference, and evaluation, as a methodological template and a case study in the challenges of building population-scale EHR foundation models.

## 1 Lay summary

Predicting who gets ill, when, and with which diseases are vital questions for individuals, doctors, and healthcare systems, such as the National Health Service (NHS). The COVID-19 pandemic urgently highlighted this need. The NHS had to quickly identify people who might become very sick if they caught the virus, so they could be prioritised for vaccinations or treatments. While many predictive tools, including some using artificial intelligence (AI), were developed during the pandemic, they often focused on single, narrow questions, like predicting the risk of death only after a patient was admitted to hospital. Because they couldn’t look at the bigger picture, they struggled to capture the wider, long-term impacts of the pandemic on the healthcare system.

To address this, we developed a new AI system called Foresight-England (Foresight-E) to predict a patient’s future medical events. Our tool learns from past patient data to estimate what might happen to a patient’s health in the future, and when these events might occur. It works a bit like the predictive text on a mobile phone, which tries to predict the next word in a sentence, but it uses medical codes instead of words.

Our goal was to test Foresight-E’s ability to model changes in people’s health during the pandemic. We looked at two main areas:

• Direct Effects of COVID-19: We tested how accurately the AI could tell us which patients would be hospitalised or die within 30 days of a positive COVID-19 test. We also checked if the tool could learn and adapt to the shifting nature of the pandemic, such as new viral variants and the vaccine rollout. Importantly, we checked if the AI worked fairly across different demographic groups, looking for evidence of bias in the underlying data that might lead to unequal predictions.

• Indirect Effects of the Pandemic: The pandemic caused severe disruptions to routine healthcare, leaving lasting effects. To see if the tool could predict these broader impacts, we trained it on data up to the end of 2022, and then tested it on "unseen" data from 2023. We looked at emergency hospital admissions, overall deaths, and the new onset of over 1,400 different diseases whose diagnosis or treatment might have been delayed or triggered by the pandemic.

AI models are only as good as the data they learn from. To make sure the model was fair and represented people from all backgrounds, we trained Foresight-E using de-identified electronic health records from 54.9 million people—representing almost the entire population of England. We then tested the tool on a separate group of 6.1 million people. This data was securely accessed via the British Heart Foundation Data Science Centre’s CVD-COVID-UK/COVID-IMPACT consortium. The data looked a bit like a computer spreadsheet of medical codes and dates, not written text notes from doctors. It included GP records, hospital visits, COVID-19 testing, vaccinations, and national death registrations.

The project was designed so that the data, the AI model, and its predictions were kept entirely within a highly secure digital system called the NHS England Secure Data Environment (SDE). This environment uses strict rules (the "Five Safes" framework) meaning no patient data ever left the NHS. The AI was built from scratch inside this secure system. Our industry partners, Amazon Web Services (AWS) and Databricks, provided computer power and technical help, but they had absolutely no access to the data, the AI model, or control over the research. It is important to note that Foresight-E was developed strictly as a research project for COVID-19 and is not used by the NHS for patient care.

NHS England has paused access to the data for this project. Because we cannot access the secure environment, we cannot retrieve the initial results of our study. Therefore, we have provided a transparent overview of how we designed and tested the system, alongside "placeholder" mock-ups of our results tables to demonstrate the work we have already completed, and how we were intending to report it.

## 2 Project Status

In May 2025, the British Medical Association and Royal College of General Practitioners’ Joint GP IT Committee (JGPITC) raised concerns that they were unaware that primary care data collected for COVID-19 research was being used to train an AI model, and queried whether the correct processes, including GDPR principles, had been followed [1]. Following these concerns, NHS England paused access to data for the Foresight-E project whilst a governance review was carried out. This review considered whether the correct approval processes and privacy protections were in place for a project of this nature. The JGPITC additionally wrote to the Information Commissioner’s Office (ICO), the UK’s data protection regulator, asking them to investigate.

In April 2026, following a thorough review, the ICO closed its review of the project and concluded that the project’s data use was compatible with the purpose for which it was originally shared therefore there was no breach of GDPR. The project’s access to the NHS England SDE remains paused.

At the point at which data access was paused, the researchers had trained several iterations of the Foresight-E model (including on the full training dataset), generated multiple sets of predictions for direct and indirect COVID-19 outcomes in the test set, and run initial quantitative evaluations on these predictions, producing aggregated performance metrics, as detailed in this manuscript. However these aggregate predictions had not yet been requested for export from the NHS England Secure Data Environment (SDE) via the approved Safe Output Service, subject to statistical disclosure control and review.

This means quantitative results are not available, and this paper instead seeks to provide a transparent overview of the underpinning methodology, evaluation strategy and work undertaken to date on Foresight-E. Placeholder results are presented to provide full transparency over the nature of the evaluation and aggregate data (e.g. tables, figures) that were intended to be exported from the SDE following standard procedure. Whilst the researchers describe the work to the best of their ability, it is important to note that since data access was paused, they have been unable to access not only the underlying data, but their codebase, models, experiment tracking, documentation and results, which are stored inside the SDE in platforms such as Databricks, Gitlab and MLflow.

## 3 Introduction

Accurate risk stratification is an important component of clinical decision-making, enabling individu als, healthcare professionals and health systems to anticipate adverse outcomes, tailor interventions, and inform resource allocation. Traditional clinical risk prediction models typically combine established expert knowledge with rule-based or statistical approaches that harness a limited number of static features, such as demographics, pre-existing conditions, or test results, at a single point in time to predict a single outcome [2, 3]. While effective for narrow, well-understood diseases, these methods face fundamental limitations during a global pandemic caused by a novel pathogen. When COVID-19 emerged, healthcare systems lacked a priori knowledge of interacting risk factors, and the epidemiology shifted rapidly with viral variants and changing public health interventions, such as vaccination rollouts. Traditional approaches struggle to model this complexity and cannot easily scale to capture the multitude of ways a systemic shock like COVID-19 impacts the healthcare system across a vast array of possible clinical outcomes.

Neural network transformer models, initially developed for natural language processing (NLP) [4], offer a way to harness the temporal and longitudinal information in electronic health record (EHRs), often underutilised by traditional modelling approaches. The move toward generative pretrained transformers (GPTs) using autoregressive next-token prediction has resulted in large language models (LLMs) capable of zero- and few-shot performance across diverse tasks without task-specific retraining [5], inspiring analogous efforts in healthcare [6–11].

Generative EHR models hold the promise of learning directly from rich, longitudinal data without relying on predefined rules, adapting to shifting epidemiology and capturing complex temporal associations that prove challenging for traditional models - properties which make them uniquely suited to modelling the complexity of COVID-19 pandemic. In applied use they offer the potential to support early detection, risk stratification, and simulation of clinical scenarios in a single model, without the need to retrain for each outcome of interest, enabling faster insights in a rapidly evolving public health emergency.

In England, the Control of Patient Information (COPI) Regulations [12] provided a legal framework to make de-identified, routinely collected, national-scale NHS data available for COVID-19-related research [13]. These data include primary care, secondary care, COVID-19 testing, vaccination, and mortality data from the population of England (approximately 61 million people [14]) and are securely stored within the NHS England Secure Data Environment (NHSE SDE) [15]. Despite the potential of such a resource for the development and evaluation of AI models for COVID-19 research, computational restrictions within the NHSE SDE have limited previous projects to substantially smaller cohorts [16, 17].

In this work, we developed Foresight-England (Foresight-E), a 243-million-parameter transformer trained from scratch entirely within the NHSE SDE on linked primary and secondary care data from 54.9 million people (90%; total dataset size of 61 million). Foresight-E was designed for the zero-shot prediction of both the direct effects of COVID-19 (e.g., hospitalisation, mortality) and its indirect systemic impacts across the population.

Access to this scale and demographic diversity is important to accurately model the COVID-19 pandemic, particularly to capture the outcomes of ethnic minority groups and patients with rare diseases or COVID-related complications who are only represented in statistically significant numbers at a population level [14, 18, 19]. Recent methodological research also shows that EHR foundation models exhibit scaling laws similar to those seen with LLMs, suggesting that larger scale training data improves performance [20]. Linking primary care records with secondary care, COVID-19 testing, vaccination data, and death registries provides a mechanism to capture the entire spectrum of the disease—from mild, community-managed presentations to critical hospital care. Crucially, this national-scale baseline allows for rigorous evaluation of algorithmic fairness across different ethnicities, age groups, and socioeconomic backgrounds.

Indeed, most prior generative EHR models have been restricted to single healthcare institutions [9], specific EHR providers [11], or limited population subsets [6]. This limits their applicability to the English general population and risks algorithmic bias, a failure to maintain consistent performance across diverse demographic and clinical groups [21].

Furthermore, the true burden of COVID-19 extends far beyond acute viral infection. The pandemic caused profound, systemic disruptions to routine healthcare delivery, resulting in delayed diagnoses, altered treatment pathways, and the emergence of post-acute sequelae (such as Long COVID) which continue to impact both individuals and healthcare systems today. Whilst transformer models offer the potential to capture these changes during their training, the extent to which this generalises to future, unseen data remains unquantified. Therefore to rigorously evaluate a model’s capacity to capture these critical indirect pandemic effects, and its ability to generalise to future data, it is necessary to test its predictive performance across the wider spectrum of human disease and beyond its training data, here encompassing over 1,400 clinical phenotypes in the year of 2023.

As outlined in Project Status 2, quantitative results are unavailable, therefore we report the data pipeline, tokenisation, architecture, training, inference, and evaluation framework underpinning the model, assessed against the TRIPOD+AI [22] and PROBAST-AI [23] reporting guidelines (see Appendix C and D).

## We contribute the following:

1. Development of Foresight-E, the first national-scale generative foundation model of EHRs for COVID-19 research.

2. A reproducible methodology for longitudinal EHR tokenisation and model development.

3. A comprehensive evaluation framework to assess zero-shot prediction of COVID-19’s direct and indirect effects on a held-out disjoint test set. Comprising 6.1 million unique patients whose data was not used during model training, and temporally held out data.

4. First demonstration of a multi graphics processing unit (GPU)-accelerated foundation model pretrained on national-scale NHS data within the NHSE SDE.

Together, these provide both a methodological blueprint for future EHR foundation models and a case study in the challenges of developing national-scale generative AI within secure health data environments.

## 4 Methods

We developed Foresight-E for COVID-19 research using linked, de-identified, routinely collected national datasets [24, 13]. A key principle of Foresight-E is that the model is treated with the same security as the underlying data. Therefore, all processing, model training, inference, and evaluation occurred entirely within the ‘Five Safes’ framework of the NHSE SDE [15]. Neither the model weights nor any generated patient timelines can be exported outside this environment; only aggregated, non-disclosive evaluation metrics are eligible for release, via the approved Safe Output Service, subject to statistical disclosure control and review.

Here we outline the datasets, patient timeline construction, tokenisation, model architecture, training regime, and our inference and evaluation strategy, including uncertainty estimation and comparative baselines. Each step was designed to support robust, generalisable, zero-shot medical event prediction for the direct and indirect effects of COVID-19.

## 4.1 Data

To train and evaluate Foresight-E, longitudinal patient data were accessed from eight pseudonymised, linked, routinely collected national datasets covering 1 November 2018 to 31 December 2023 [24, 13]. These encompassed primary care (General Practice Extraction Service (GPES) Data for Pandemic Planning and Research (GDPPR) [25]), secondary care (Hospital Episode Statistics (HES): Outpatients (OP), Accident & Emergency (A&E), Admitted Patient Care (APC) and Critical Care (CC) [26]), mortality (Office for National Statistics (ONS) Civil Registration of Deaths [27]), COVID-19 testing (UK Health Security Agency (UKHSA), formerly Public Health England (PHE), COVID-19 Second Generation Surveillance System (SGSS) [28]) and COVID-19 vaccination (NHS England COVID-19 Vaccination Status [29]) data (see Fig. 1a).

The base cohort comprised individuals alive on or after 1 November 2019 (the GDPPR dataset inclusion criterion) with known age and sex, resident in England, GP-registered, and without conflicting death dates. To ensure a minimum one year of past medical history before predicting outcomes, we included data from 1 November 2018 onwards. Because primary care registration dates were unavailable to confirm exact prior follow-up lengths, this fixed calendar gap served as a populationlevel proxy for baseline history. To accommodate a dynamic cohort, individuals born after the study start date could enter the cohort at birth, however to enforce the minimum one year history for these newborns, we required them to reach at least one year of age prior to death or the end of the training period (31 December 2022). Ultimately, this yielded a nationally representative cohort of 61 million patients [14].

Table 1: PLACEHOLDER Selected characteristics of the training, validation, and test sets (used in Sections 5.1 and 5.2). Ages are computed at the time of the last input event in each patient’s timeline. Counts are rounded to the nearest five to comply with the NHSE SDE’s disclosure control requirements. Total sample sizes for each split are derived from previously reported counts [14]. The rationale for placeholders is provided in Section 5.
<table><tr><td></td><td>Train</td><td>Validation Test: 30-day COVID-19</td><td>Test: 1 Year</td></tr><tr><td>Sex</td><td></td><td></td><td></td></tr><tr><td>Female</td><td>-(_%)</td><td>-(_%) -(_%)</td><td>_(_%)</td></tr><tr><td>Male</td><td>-(%) _(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>Ethnicity</td><td></td><td></td><td></td></tr><tr><td>Asian</td><td>_(_%) -(_%)</td><td>_(_%)</td><td>_(_%)</td></tr><tr><td>Black</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>Mixed</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>Other</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>Unknown</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>White</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>Age</td><td></td><td></td><td></td></tr><tr><td>0-9</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>10-19</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>20-29</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>30-39</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>40-49</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>50-59</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>60-69</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>70-79</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>80-89</td><td>-(_%) -(%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>90+</td><td>-(_%) -(_%)</td><td>-(_%)</td><td>-(_%)</td></tr><tr><td>Total</td><td>48.8M 6.1M</td><td>6.1M</td><td></td></tr></table>

We created training (48.8M patients), validation (6.1M patients), and test sets (6.1M patients) via disjoint 10% patient samples. All 2023 events were reserved to test temporal generalisation, mimicking a prospective deployment on future, unseen data. Fig. 1 shows temporal coverage and partitioning of the datasets.

![](images/0e6d23f45123abef453f32b8ee52dcbc71914923ccf7065142e98337e8186ba3.jpg)  
(a) Linked primary/secondary care, deaths, testing, and vaccination datasets (1 Nov 2018–31 Dec 2023).

![](images/9f9f222b628ab63f026253ad5995dfd05a4b16729986d6a4ee46db79d136808f.jpg)  
(b) Disjoint 10% validation/test cohorts; 2023 held out for temporal generalisation evaluation.  
Figure 1: Datasets and cohort splits used in the training and evaluation of Foresight-E. Total sample sizes for each split are derived from previously reported counts [14].

## 4.2 Patient Timelines

Each patient’s history was encoded as a chronological sequence of dated, coded events, including diagnoses, procedures, medications, and other healthcare interactions. Clinical codes followed standard terminologies: ICD-10 [30] for HES diagnoses and ONS causes of death, OPCS-4 [31] for HES procedures, and SNOMED CT [32] for GDPPR records and COVID-19 vaccinations. We supplemented these with custom tokens for events not coded in standard terminologies, such as positive SARS-CoV-2 tests, hospital admission indicators, and primary or secondary diagnosis flags. No free text terms were available in the underlying data, or included in the model training.

Because event timestamps were available only at day-level precision, we applied a consistent withinday ordering: COVID-19 vaccination; SARS-CoV-2 test; GDPPR events; HES OP; HES A&E; HES APC; HES CC; and, lastly, death. Where relevant, within each dataset events were further ordered by admission date, then by primary diagnoses, secondary diagnoses, procedures, and finally alphanumerically.

Geographic (region) and deprivation (Indices of Multiple Deprivation) data were excluded from the training data to avoid explicitly encoding these existing biases and enable held-out subgroup analysis [21]. Codes deemed sensitive (e.g. relating to sexually transmitted infections) were removed, using a clinically-informed codelist supplied by NHS England [33].

## 4.3 Tokenisation

We converted each patient’s event timeline into a sequence of discrete tokens representing clinical codes and temporal intervals. Each sequence began with static demographic tokens for sex (e.g., SEX\_FEMALE) and ethnicity (e.g., ETHNICITY\_ASIAN). We then added the sequence of clinical codes, ordered as described in Section 4.2, from each patient’s timeline as tokens. If two consecutive events occurred on different days, we inserted a time-difference token (e.g., TIME\_DIFFERENCE\_1) to represent the interval in days; no time-difference token was added between consecutive same-day events. To encode absolute calendar time, we inserted a YEAR\_START token at the beginning of each calendar year, followed by an AGE\_N token for the patient’s integer age (or AGE\_UNBORN if not yet born). These yearly markers ensured that time gaps never exceeded 366 days and allowed the representation of age-related and absolute temporal patterns to generalise temporally.

A lookup-based tokeniser (acting as a simple 1:1 dictionary) mapped all unique tokens in the training data to integer IDs, including clinical codes, age (integer years and unborn), 1–366 day gaps, and special tokens for padding (<PAD>) and sequence end (<EOS>). The resulting fixed tokeniser vocabulary contained approximately 40,000 tokens. At inference time, if a code was encountered that was not in the trained tokeniser’s vocabulary, it was dropped from the sequence without substitution, rather than using an unknown token (e.g., <UNK>).

During training, we truncated each individual patient’s timeline to a maximum of 1,024 tokens using left truncation (discarding the earliest events) to retain the most recent clinical context. Truncation was applied at the event level followed by re-tokenisation, rather than truncating token sequences directly; this preserved demographic, year-start and age tokens and correctly recomputed time-difference tokens. For batching, we right-padded sequences to a uniform length with padding tokens, which were masked from attention and loss calculations. At inference, we pre-truncated inputs to 1,024 − L<sub>forecast</sub>, where $L _ { \mathrm { f o r e c a s t } }$ denotes the tokens allocated for forecast generation, instead of implementing dynamic on-GPU truncation. The maximum sequence length of 1,024 tokens was selected to balance capturing the maximum possible clinical context while maintaining adequate training and inference batch sizes on the available NVIDIA A10 GPUs.

## 4.4 Training

Foresight-E is a 243-million-parameter transformer-decoder model trained to predict the next code in a tokenised EHR sequence, see Section A.1. This is analogous to how large language models (LLMs) are trained and can be considered as predicting what will happen next to a patient, based on their past medical history [5, 8, 9]. We adapted the open source Llama 2 architecture [34, 35] and trained the model from scratch with randomly initialised weights due to the inability to import pretrained models into the NHS SDE and the use of a custom vocabulary, with clinical codes and custom tokens, rather than natural language. Training comprised a single pass through the dataset with periodic checkpointing; the checkpoint with the lowest validation loss was selected as the final model. Hyperparameters followed established defaults [8, 36]. Complete architectural and training details are provided in Section A.

![](images/aab6a4b454943dfc0ce217cb96ae03786c3c86d54e5ba99de744f9eba255afda.jpg)  
Figure 2: Schematic of next token prediction training of Foresight-E on a synthetic patient timeline. Tokens at position IDs 2–7 are omitted for clarity. Given the sequence of tokens up to position i, the model predicts the probability distribution over the vocabulary for the token at position i + 1. These probabilities are compared with the true next token to compute the cross-entropy loss for self-supervised training. Note that the initial static demographic tokens are masked ([Ignore]) during the loss calculation to prevent the model from learning to predict sequence progression solely from demographic features. The final and padding tokens are also ignored because they have no label to compare against.

## 4.5 Zero-Shot Inference

As a generative model, once trained, Foresight-E can predict future clinical events across the breadth of its vocabulary from patient histories, without requiring task-specific supervised fine-tuning (‘zeroshot’ inference). This can be viewed as analogous to LLMs ‘autocompleting’ a sentence.

At inference, the model is prompted with a tokenised patient timeline (see Section 4.3) and produces a probability distribution over the next token. We sample one token from this distribution and append it to the input. This autoregressive procedure is repeated until a prespecified stopping criterion is met: a sentinel event token (e.g., mortality), a maximum forecast horizon (e.g., 30 days or 1 year), or a limit on newly generated tokens (e.g., 300). The generated tokens are then decoded to form the predicted event timeline, see Figure 3

Because greedy decoding yielded repetitive sequences, we used multinomial sampling from the model’s output distribution to generate diverse and clinically plausible trajectories. We did not use parameter-dependent sampling techniques (e.g., top-k, temperature), which would have introduced additional task-specific hyperparameters. A key-value (KV) cache was used to increase generation throughput by avoiding the recomputation of past steps.

To quantify predictive uncertainty at the individual patient level, an essential consideration for safe and reliable clinical decision-making [37], we performed S = 48 independent stochastic rollouts per patient. This was the maximum batch size possible on a single A10 GPU [38] given Foresight-E’s size and context length. Inference was parallelised across eight GPUs for throughput [39].

For event i and horizon t (in days), the predicted probability was estimated as [9]

$$
P ( { \mathrm { E v e n t } } _ { i , t } ) = { \frac { s _ { i , t } } { S } } ,\tag{1}
$$

where $s _ { i , t }$ denotes the number of rollouts in which the event occurred within the horizon. This formulation provides probability estimates across all medical events observed during training, enabling probabilistic zero-shot predictions across diverse clinical outcomes.

![](images/2620b36720915557fc89aab54f1624d7f4eb9a1f7583d24dc7f3cebbf539469f.jpg)  
Figure 3: Schematic of idealised zero-shot inference on a synthetic patient timeline. Given a history ending with a positive SARS-CoV-2 test, Foresight-E predicts a hospital admission within the next 30 days. Tokens 2–7 are omitted for clarity.

![](images/92b2dbc842c58f392156af315f5522a3cb061cbd978db24f0527f9487c5d353e.jpg)  
Figure 4: Visualisation of an idealised Foresight-E zero-shot inference for the task of 30-day hospitalisation following a positive SARS-CoV-2 test. A tokenised input timeline, shortened for visualisation, is used to prompt three independent predicted timelines in this example. ‘Time Diff‘ tokens are in units of days. In two of the predicted timelines, hospitalisation is predicted within the next 30 days (at 20 and 25 days, respectively). The remaining timeline predicts the next event at 100 days, falling outside the target window. Therefore, the predicted probability is 2/3.

## 4.6 Evaluation

We evaluated Foresight-E on two acute COVID outcomes: hospitalisation and death within 30 days following a positive SARS-CoV-2 test. Eligible patients in the test set had at least one positive test between 23 January 2020, the date of the first recorded COVID-19 case in the UK, and 1 December 2023 [40]. While mass community testing in England ended on 1 April 2022, we deliberately included tests beyond this date [41]. Although testing post-April 2022 is inherently selective, capturing predominantly high-risk, hospitalised, or healthcare worker populations, evaluating our model across this policy shift assesses its robustness to changing clinical testing practices and population risk profiles across the pandemic. Timelines were truncated after the first positive test, and predictions generated for the next 30 days. Hospitalisation was defined as an admission in HES APC (APC token), and death as an entry in the Office for National Statistics (ONS) death registry (DEATH token). Inference was completed in 2 days of wall-clock compute time.

Secondly, we assessed the ‘indirect’ effects of COVID-19 in the held-out year of 2023, evaluating emergency hospitalisation, all-cause mortality, and the onset of over 1,400 Phecode-defined diseases. We evaluated these outcomes across the entire held-out cohort, irrespective of a formally recorded positive SARS-CoV-2 test or COVID-19 diagnosis. This population-wide evaluation accounts for the high levels of prior SARS-CoV-2 infection by 2023, and captures the systemic healthcare disruptions that affected patients across all care pathways [42–45].

Here, timelines were truncated after the 2023 year-start token and subsequent events predicted over a one-year horizon. Emergency admissions were identified via admission method codes in HES APC, and mortality as above. Phenome-wide disease onset was defined by the first occurrence of each Phecode, a previously validated collection of over 1,400 groups of ICD-10 codes (example shown in Table 2) [46], recorded in HES APC. Inference was completed in 15 days of wall-clock compute time.

## 4.6.1 Metrics

For evaluation, each outcome was treated as a binary classification task: given a patient’s history, the model provided an estimate of the probability of event i occurring within t days, and this was compared against the observed outcome. This allowed for the evaluation of single medical event prediction as well as bespoke phenotypes composed of multiple clinical codes or more complex definitions. Under this binary classification formulation, false positives and false negatives are quantified and correspond to the concepts of hallucinations and omissions in LLMs, respectively. Competing risks (such as death prior to the onset of an outcome) were not explicitly modelled as separate states; instead, the prediction of a death token terminated the sequence generation.

Discriminative performance was assessed using the area under the receiver operating characteristic curve (AUROC) [47] and the area under the precision-recall curve (AUPRC) [48], the latter being commonly used under class imbalance [49, 50]. Overall performance and calibration were jointly measured using the Brier score [51]. We also generated ROC, precision-recall, and calibration curves for visual inspection. To reflect potential use in a clinical screening workflow, we also evaluated recall at a fixed 10% false-positive rate (FPR10), consistent with prior studies [52] and quantifying performance at a hypothetical operational threshold.

Confidence intervals (95%) were obtained via non-parametric patient-level bootstrapping with the percentile method, using 1,000 resamples for all analyses except the Phecode tasks (where 100 were used due to computational cost).

## 4.6.2 Subgroup Analyses

We conducted subgroup analyses of mortality predictions at both the 30-day post–SARS-CoV-2–positive test and one-year horizons, stratifying test-set patients independently by age, sex, ethnicity, and vaccination status. All of these stratification variables were tokenised and included in the model’s training data; age and vaccination status were embedded longitudinally within the patient timelines, while sex and ethnicity were included as static demographic tokens at the start of each timeline. For each subgroup, we calculated the AUROC and AUPRC in comparison with the overall cohort. Additionally, we investigated how performance on these metrics varied with the number of historical patient events provided as model input. Finally, we assessed changes in AUROC and AUPRC relative to the timing of the positive SARS-CoV-2 test to determine whether the Foresight-E model could model the changing epidemiology of COVID across the pandemic.

## 4.7 Baseline Methods

We benchmarked Foresight-E against supervised classifiers trained separately for all-cause mortality and hospitalisation at 30 days after a positive SARS-CoV-2 test, and for the same outcomes over a one-year horizon in 2023.

These comparators used the same training split as Foresight-E but were task-specific, in contrast to Foresight-E’s zero-shot capability. The first was a simple logistic regression model [53] using age, sex, and ethnicity (one-hot encoded, meaning each categorical variable was represented as a binary vector where only the position corresponding to the observed category is set to one), with vaccination status added for acute SARS-CoV-2 outcomes. The second was an XGBoost model [54] using count vectors of all medical codes from the Foresight-E vocabulary, providing equivalent structured clinical data but without temporal information.

To allow training outcomes to be observed, input cut-offs were set to 1 December 2022 for the 30-day tasks and 1 January 2022 for the one-year tasks. This requirement highlights a key advantage of Foresight-E’s self-supervised learning: it is trained without requiring explicitly defined outcome labels. This eliminates the need to withhold a temporally separated subset of data for outcome labelling. Performance for all models was evaluated using AUROC, AUPRC, and Brier score, as described in Section 4.6.1.

## 4.8 Governance

The North East - Newcastle and North Tyneside 2 research ethics committee provided ethical approval for the CVD-COVID-UK/COVID-IMPACT research program (REC No 20/NE/0161) for approved research projects to access, within secure trusted research environments, whole-population, de-identified data from EHRs collected as part of patients’ routine healthcare.

The CVD-COVID-UK/COVID-IMPACT programme, led by the BHF Data Science Centre [55], received approval to access data in the NHSE SDE service for England from the Independent Group Advising on the Release of Data (IGARD) [56] via an application made in the Data Access Request Service (DARS) Online system (ref. DARS-NIC-381078-Y9C5K) [57].

The CVD-COVID-UK/COVID-IMPACT Approvals & Oversight Board that includes patient and public advisors [58] subsequently granted approval to this project (CCU078: Foresight: a generative AI model of patient trajectories across the COVID-19 pandemic) in December 2023 to access the data within the NHSE SDE service for England.

Patient and public involvement was included in the approvals process and has continued to shape the research and communications through Patient and Public Involvement and Engagement sessions organised via British Heart Foundation (BHF) Data Science Centre [58].

## 4.8.1 Data Availability

The data used in this study are available in the NHSE SDE service for England, but as restrictions apply, they are not publicly available [15].

The de-identified data used in this study were made available to accredited and approved researchers only. Those wishing to gain access to the data should contact bhfdsc@hdruk.ac.uk in the first instance, noting that as detailed in Project Status 2 data access is currently paused.

## 4.8.2 Code Availability

All data preparation, model training, and evaluation code was intended to be released, following export and review via the SDE’s Safe Output Service, on GitHub at: https://github.com/ BHFDSC/CCU078\_Foresight-England, including a requirements.txt specifying all package versions. However as detailed in Project Status 2 data access is currently paused meaning code cannot be exported and shared.

The code used to produce the placeholder results figures was developed outside the SDE, without access to the underlying data, and is available at https://github.com/simonEllershaw/ foresight\_placeholder\_graphs

Due to data restrictions, trained model weights and artefacts are only accessible to a subset of approved CVD-COVID-UK/COVID-IMPACT consortium researchers on a dedicated Foresight-E cluster within the NHSE SDE. Those wishing to gain access to the data should contact bhfdsc@hdruk.ac.uk in the first instance, noting that as detailed in Project Status 2 data access (including model weights, artefacts and code) is currently paused.

All analyses were executed within the NHSE SDE [15] using Databricks Runtime 14.3 LTS for ML [59]; training/evaluation used an AWS g5.48xlarge instance with eight NVIDIA A10 GPUs [39, 38]. AWS and Databricks had no access to the underlying datasets or trained AI model and no control over the research or its findings.

## 5 Results

As detailed in Project Status 2, an initial quantitative evaluation has been completed, but the ongoing pause in data access prevents export of the results from the SDE. To be transparent about what was evaluated and what outputs were intended for publication, we present the relevant tables and figures with placeholder values, marked with ’\_’, and placeholder figures where data cannot be shown.

## 5.1 30-day COVID-19 Mortality and Hospitalisation Prediction

To demonstrate the utility of Foresight-E in predicting the direct effects of COVID-19 without finetuning, we applied it to the important clinical prediction task of forecasting 30-day mortality and hospitalisation following a positive SARS-CoV-2 test.

Foresight-E achieved an AUROC of \_ (\_-\_, 95% CI) and \_ (\_-\_, 95% CI) for 30-day prediction of mortality and hospitalisation, respectively. For comparison, the baseline supervised logistic regression and XGBoost models yielded AUROCs of \_ and \_, and \_ and \_, respectively. Calibration and discrimination was quantified using Brier scores, yielding \_ and \_ for the mortality and hospitalisation endpoints, respectively. At a 10% false positive rate (FPR), detection rate (DR10) was \_ and \_, respectively. ROC, precision-recall, and calibration curves for each task are shown in Fig. 5, with 95% confidence intervals estimated via 1,000 sampled bootstrap iterations.

![](images/170b8a1e966825f112ad3916b50e0940a368295d10dd1ad07405a1c4e7b7650f.jpg)

![](images/2b2b5e6fac7e2cbfff6b81a8244c9e8bf1558640c44a1713aa7b481b292dad10.jpg)  
(a) Mortality

![](images/97c21ef43d6c6639edad5b624086b825e3ac2ea75e9387e12d1df2da16b47542.jpg)

![](images/a7bd41dce0d3485391db6bd421ba133498906ed855d933f32b1bd9f3c7828033.jpg)

![](images/770659dd2409930a8856e4eb32aef8c438cf8707c24a7ee9180fd28e887b514a.jpg)  
(b) Hospitalisation

![](images/beb771d62b8ceee12b38aff31df27df57bea01d04bd266d2c7728e06aa0010e3.jpg)  
Figure 5: PLACEHOLDER. ROC, PR, and calibration curves showing Foresight-E’s prediction performance for 30-day mortality and hospitalisation following a positive SARS-CoV-2 test. Shaded regions represent 95% confidence intervals estimated via 1,000 bootstrap iterations. Counts are rounded to the nearest five to comply with the NHSE SDE’s disclosure control requirements.

## 5.1.1 COVID-19 Outcomes Across the Pandemic

To evaluate Foresight’s ability to model the changing epidemiology of COVID-19 across the duration of the pandemic (including varying case rates, mortality, and circulating viral variants), we performed a subgroup analysis stratified by the calendar year and month of the patient’s first positive SARS-CoV-2 test. Across these temporal strata, we observed AUROC values ranging between \_ and \_. On the 2023 data, a temporal holdout set excluded from training or validation, the computed AUROC was \_.

![](images/b36be0855df5621992f30807af02e4e1dcebd0780dac2a2030e9b8f2e96321a1.jpg)  
Figure 6: PLACEHOLDER. Temporal evaluation of 30-day mortality prediction across the COVID-19 pandemic. Monthly performance of Foresight-E in predicting death within 30 days of a positive SARS-CoV-2 test, measured by AUROC (top panel) and AUPRC (second panel). Error bars represent 95% confidence intervals estimated via 1,000 bootstrap iterations. Also shown are monthly mortality occurrence rates (third panel) and the number of positive SARS-CoV-2 tests (bottom panel), illustrating changes in disease dynamics over time. Shaded background regions denote the predominant SARS-CoV-2 variants [60] with labels shown along the upper x-axis. Counts are rounded to the nearest five to comply with the NHSE SDE’s disclosure control requirements.

## 5.1.2 COVID-19 Outcomes by Subgroup

Fig. 7 shows the performance of the Foresight-E model in predicting 30-day mortality following a positive SARS-CoV-2 test, stratified by age, ethnicity, sex, and COVID-19 vaccination status.

To assess for performance disparities across demographic groups, we computed AUROC scores of for White patients, \_ for Asian patients, \_ for Black patients, and \_ for Mixed/Other ethnicities. Between sexes, AUROC scores were \_ for male patients and \_ for female patients. Across age brackets, AUROC values measured \_ to \_, mapped against a crude mortality rate of $- \% 1 0 \mathrm { ~ ‰ ~ }$ in those respective demographics.

![](images/6b6cd65269f22bd666cbfa5032e735023169ac7611a919bbdd43268c3114da47.jpg)  
Figure 7: PLACEHOLDER. Variation in AUROC and AUPRC metrics for the Foresight-E model predicting 30-day mortality after a positive SARS-CoV-2 test, stratified by age, ethnicity, sex, and COVID-19 vaccination status. Error bars show 95% confidence intervals calculated by 1,000 bootstrap iterations. Also shown is the mortality occurrence rate, as well as the count of patients for each group. Performance is consistent across ethnicity and sex, but varies notably by age group and vaccination status. Counts are rounded to the nearest five to comply with the NHSE SDE’s disclosure control requirements.

## 5.1.3 COVID-19 Outcomes By Event History

Fig. 8 plots model performance predicting 30-day mortality after a positive SARS-CoV-2 test against the number of historical patient events provided as input to Foresight-E. We measured a correlation of \_ between the number of past medical events and AUROC, contextualised by a correlation of \_ between the number of past medical events and the outcome. The distribution of the number of past events across the cohort exhibited a skewness of \_.

![](images/a8a48a8f1992bcd5e8a7968628d20fa85fbc61bd1863c063c7966eefdb6b4828.jpg)  
Figure 8: PLACEHOLDER. Variation in AUROC and AUPRC metrics for the Foresight-E model predicting 30-day mortality after a positive SARS-CoV-2 test with the number of past medical events recorded in a patient’s timeline. Also shown are the binned frequency of past events and corresponding mortality occurrence rates. A bin size of 16 events was used, and the final bin includes all patients with more than 498 events. Counts are rounded to the nearest five to comply with the NHSE SDE’s disclosure control requirements.

## 5.2 Predicting Indirect Effects of COVID-19 in 2023

To evaluate Foresight-E’s ability to predict the indirect effects of the COVID-19 pandemic, we simulated a prospective deployment. All patient timelines were truncated at 1 January 2023, using the YEAR\_START token. Foresight-E was not exposed to any event data from 2023 during training, thereby testing generalisation to unseen future events.

## 5.2.1 Emergency Hospitalisation and Mortality Prediction

The ROC, precision-recall, and calibration curves are presented in Fig. 9.

Foresight-E achieved an AUROC of \_ (\_-\_, 95% CI) and \_ (\_–\_, 95% CI) for emergency hospitalisation and mortality, respectively. For comparison, the baseline supervised logistic regression and XGBoost models yielded AUROCs of \_ and \_, and \_ and \_, respectively. Calibration and discrimination was quantified using Brier scores, yielding \_ and \_ for emergency hospitalisation and mortality endpoints, respectively.

![](images/dd102204cbf311c04895441b27cc44d47f1d7150310d08f54c253ec8835137cf.jpg)

![](images/8dcda483c1f55ac9c273a1252795d4bf2c3321b55b42d01115a39d62e0a66f51.jpg)

![](images/55422642b005a03f3322161410b60097ca26f92973f187ec4f170af2bc339ec4.jpg)

(a) One-year mortality prediction: ROC, PR, and Calibration Curves.  
![](images/7474967dd9e1d0024adf3a3c4f5f49bf2e9e4a5ae00a41cd6e8a1d1a166e93a9.jpg)

![](images/64696a2415514a45622d980137f51f8e6a656181f1bef78689cdbc036e6c7d8c.jpg)

![](images/b581ddf0e30ad8fbd7a06481c60d54c2def875c74c4dbd5bdbe6a95cb013578a.jpg)  
(b) One-year emergency hospitalisation prediction: ROC, PR, and Calibration Curves.

Figure 9: PLACEHOLDER. Prediction performance of Foresight-E in 2023 using ROC, PR, and calibration curves. Shaded regions represent 95% confidence intervals estimated via 1,000 bootstrap iterations. Counts are rounded to the nearest five to comply with the NHSE SDE’s disclosure control requirements.

## 5.2.2 Phecode Predictions

To assess the breadth of Foresight-E’s capacity to predict the indirect effects of COVID, we evaluated performance on over 1,400 distinct Phecodes, previously validated groupings of ICD codes representing clinically meaningful phenotypes [46]. AUROC values varied \_-\_ across Phecodes, with an \_ association between event prevalence and predictive performance (Fig. 10).

![](images/bb970a854063f3f97aac78521bbdc31f9d20b162bd1fe022daa821eda386dac9.jpg)  
Figure 10: PLACEHOLDER. Relationship between AUROC and event occurrence rate for Foresight-E’s 1-year prediction in 2023. \_ AUROCs were observed for more \_ Phecodes, such as \_

## 6 Discussion

We developed Foresight-E, a 243-million-parameter generative foundation model trained de novo on longitudinal electronic health records from 54.9 million patients, entirely within the NHS England Secure Data Environment. Designed as a research pilot for COVID-19, the model was evaluated on its zero-shot capability to predict both the direct acute outcomes of SARS-CoV-2 infection and the pandemic’s enduring, system-wide indirect effects. By successfully engineering this pipeline within existing national infrastructure, this work establishes a methodological template for future pandemic response and broader applications, from clinical risk prediction to population-level healthcare planning. As quantitative results on our 6.1-million-patient test set are currently withheld pending an ongoing governance review, we focus our discussion below on the core methodological strengths and limitations of this work, including data integration, modelling and evaluation, before discussing future directions for the field.

## 6.1 Data

Foresight-E’s key strength is its first use of national-scale, routinely collected EHRs spanning primary care, secondary care, COVID-19, and death registrations for the development of a generative AI model. This ensures Foresight-E is trained on diverse data representative of the general population [14], helping to mitigate algorithmic bias [61], enabling prediction of rare events [18, 62], and generating evidence for the methodology’s translational potential through aligning training data with intended use populations. Given that primary care accounts for most healthcare delivery, integrating this data provides a more complete representation of an individual’s health, enables earlier risk stratification, and may ultimately guide preventive interventions to avert disease onset, complications and costly secondary care.

Despite its breadth, the datasets reflect typical challenges of routinely collected health data: incomplete or imprecise coding [63], historical biases in care [64, 61], and shifts in recording practices [65], especially during the COVID-19 pandemic [66]. Notably, the GDPPR primary care dataset contains a subset of codes, specifically, those already available from previous GPES extracts that were deemed relevant for COVID-related research. As a result, it omits examples of both common and rare diseases, as well as signs and symptoms data that is crucial for understanding disease presentation and evolution - for example, COVID presenting with anosmia, or non-specific symptoms of long COVID, such as fatigue.

Furthermore, GDPPR only contains those alive on or after 1 November 2019. This means that individuals who died before this date are excluded, restricting the ability to model or evaluate longterm disease progression or events in the pre-COVID era. While this is not a limitation for models focused on COVID-19 infection, since such cases did not exist before early 2020, it does reduce the available historical context. Expanding the temporal window to include earlier records would be expected to increase predictive performance and improve the capacity to model long-term outcomes, but would require methods capable of efficiently handling longer sequences [67] and risk issues such as immortal time-bias without changes in dataset inclusion criteria.

While Foresight-E is strictly confined to the structured data approved for COVID-19 research, future applications of this methodological approach could be enhanced by incorporating additional modalities such as clinical notes or medical imaging. The development of foundational multi-modal transformer models in the general domain shows how the methods presented here could be extended [68, 69]. However, such data is not currently available at a national scale, and provisioning new data streams for future models would require appropriate governance, infrastructure, and technical methods, including de-identification and pre-processing.

## 6.2 Model

Foresight-E uses a 243-million-parameter Llama 2-style transformer decoder. While larger than most prior EHR models [10] and trained on national scale data, the scale of data, model size, and compute is still orders of magnitude smaller than the general domain [70].

Foresight-E is designed to be data-driven, rather than relying on predefined parametric models such as exponential hazard–style structures [10, 71]. Although we trained from scratch due to a custom vocabulary and NHSE SDE import restrictions, smaller-scale studies indicate that starting with an LLM pre-trained on vast general data and then fine-tuning for medical-event prediction is beneficial to performance [72, 73]. If future NHSE SDE policy permits importing pretrained models, initialising from a general model, and adapting in-domain is a promising direction.

Our tokenisation strategy preserves the recorded granularity of clinical codes and day-level timing of the EHR data , rather than aggregating codes into higher level categories [10] or phenotypes [6], filtering low-occurrence events [8] or binning time [9]. This maximised diagnostic specificity and allowed prediction of rare events often excluded in other models [62]. However, this enlarged the vocabulary and prediction space and reduced per-token training frequency.

Furthermore, all event codes present in the training set were added to the tokeniser’s vocabulary. Therefore, during inference, codes absent from the training data could not be represented because the vocabulary did not include an unknown token (e.g., <UNK>), so we excluded them, discarding potentially clinically relevant information. The tokeniser could be further improved by exploiting the hierarchical structure of clinical ontologies [9], for which there is evidence of performance gains on EHR prediction tasks [74].

Building on previous work that models chronological age and relative time between events, we additionally encoded absolute calendar time to model the changing healthcare system during the COVID pandemic. Because Foresight-E has a maximum context length of 1,024 tokens, long histories required truncation. Naïve token-level truncation (e.g., dropping the first n tokens) corrupts timelines by removing demographic, year-start, or age tokens or invalidating time-difference tokens. We therefore applied event-level truncation followed by re-tokenisation as a pre-processing step for training. During autoregressive inference, however, all operations must run on the GPU, making event-level truncation infeasible. We therefore used fixed maximum input and generation lengths, which occasionally discarded more tokens than necessary and, in some cases, prevented forecasted trajectories from reaching the final time horizon (e.g. 1 year).

Previous work [9] constructs training examples by concatenating all patient EHRs into a single sequence and then randomly sampling subsequences of a fixed length. Although this maximises computational efficiency by eliminating the need for padding tokens, it allows individual training examples to span across multiple patients. This risks the model learning spurious cross-patient relationships. To avoid this, we adopt a patient boundary-preserving strategy [20], forming sequences strictly at the patient level, which comes at the expense of computational efficiency. Future work could incorporate intra-document causal masking [75] to enable efficient concatenation without information leakage.

## 6.3 Inference

We extended prior probabilistic patient-trajectory approaches [9], applying Foresight-E in a zeroshot setting to forecast over 1,400 medical events, across 30-day and 1-year time horizons relevant to direct and indirect COVID-19 outcomes. This offers the broad ability to predict differential diagnoses and clinical trajectories, rather than single outcomes, but is computationally intensive, particularly at a population scale. Comparative studies with task-specific fine-tuning are needed to clarify efficiency–flexibility trade-offs.

Our current setup generates 48 trajectories per patient, constrained practically by the NVIDIA A10 GPU’s memory. Exploring how probability estimates stabilise with the number of samples and alternative decoding strategies (beam search, top-k, temperature) may yield gains in efficiency, accuracy, and calibration, but would require tuning. Furthermore, quantifying the temporal horizon over which Foresight-E can reliably forecast patient outcomes, both in absolute time and in terms of tokens, requires further study.

## 6.4 Evaluation

We evaluated Foresight-E’s ability to predict COVID-19 outcomes in two settings designed to reflect real-world deployment challenges.

First, we assessed its ability to predict direct COVID-19 outcomes during the pandemic, a period marked by shifting conditions such as emerging viral variants, changing testing protocols, public health interventions and population immunity. Unlike earlier models [9], Foresight-E explicitly encodes both absolute calendar time and patient age, enabling it to adapt to these temporal shifts. Forecasting future novel threats, outside the current training data and vocabulary, could be supported by continued pretraining with vocabulary expansion on the latest batches of newly collected data.

Second, we sought to predict the indirect effects of COVID-19 and simulate a prospective deployment by predicting events in 2023, one year beyond the training period, on over 1,400 Phecodes, emergency hospitalisation, and all-cause mortality. COVID-19 highlights the challenges of temporal data shifts. Foresight-E was trained during the height of the COVID-19 pandemic, a period of profound healthcare system disruption and excess all-cause mortality, which left an enduring and evolving legacy in the evaluation period of 2023, encompassing the indirect effects the pandemic exerted on individuals, healthcare systems, and society at large [43].

This represents, to our knowledge, the broadest zero-shot evaluation of an EHR foundation model to date. Working at the scale of the English population meant facing a low occurrence rate for many acute outcomes, in contrast to models trained solely on high-acuity inpatient cohorts (e.g., MIMIC-IV [76]). While this sparsity poses challenges, it also demonstrates Foresight-E’s potential utility for COVID-specific population screening as well as high-risk patient monitoring, such as identifying vulnerable individuals to prioritise targeted interventions (e.g. vaccination, antivirals), by training on both general ‘healthy’ and acutely ill patient timelines. In order to critically evaluate Foresight-E’s potential for population screening, we additionally calculated recall at a fixed 10% false-positive rate, consistent with prior studies [52] and representing a hypothetical operational threshold.

Whilst evaluating each patient trajectory as multiple binary prediction tasks allowed use of established metrics (AUROC, AUPRC, Brier score) to quantify performance on clinically-relevant tasks, these fail to capture the overall fidelity of generated trajectories, an important direction for future work.

A key limitation of this work is the inability to benchmark against commonly used clinical risk prediction tools, due to a lack of required data (e.g. 4C Mortality Score for COVID requires physiological measurements and test results [77]), limited data duration (e.g. QRisk measures 10- year cardiovascular disease risk [2]) and the fact that where risk scores were recorded the resulting outcomes were then conditioned on resultant clinical decision making [78]. For methodological comparators NHSE SDE constraints meant external pretrained models could not be imported and resource limitations restricted the number of comparator models trained.

External validation, to assess model generalisability across different populations, was not possible as Foresight-E was developed entirely inside the NHSE SDE and the trained model could not be transferred out of the environment (including to another SDE), nor could other cohorts be imported into the NHSE SDE.

Explainability techniques, such as attention-weighted visualisation, gradient-based saliency, or counterfactual generation, are left as directions for future work but could help clinicians understand the model’s forecasts by scrutinising learned associations for known or plausible patterns, as well as the presence of spurious correlations, such as shortcut learning. Such methods could support safe adoption in practice by explaining why the model anticipates particular outcomes, as well as identifying potentially modifiable risk factors as targets for interventions to optimise health.

## 6.5 Future Directions and Considerations

Foresight-E was developed as a research pilot strictly for COVID-19-related research and is not a validated clinical tool. The model is therefore confined to this scope, and any future directions for the methodology are entirely contingent on navigating the significant challenges outlined below.

The generative, zero-shot forecasting methodology demonstrated by models like Foresight-E offers broad potential, including forecasting population health, stratifying groups at increased risk of adverse outcomes, and enabling personalised risk prediction to guide preventive interventions. Beyond direct clinical care, potential applications include improving clinical trial efficiency through prognostic enrichment or advancing drug discovery by better modelling disease trajectories. Models like Foresight are by design associative, rather than causal, and therefore a priority area for research is the robust evaluation of the extent to which counterfactual questions can be answered, a crucial step toward creating robust digital twins and enabling trustworthy in-silico trials.

Moving from research to application would require secure, real-time model deployment, a capability beyond the current NHSE SDE infrastructure. This gap is particularly critical for large generative models, which can inadvertently memorise and expose sensitive training data [79]. A potential mitigation strategy is to deploy such models within a secure environment behind narrowly scoped APIs. These would provide only predefined, validated outcomes, such as calibrated risk scores, rather than open-ended generative trajectories, thereby constraining vectors for data extraction and mitigating privacy risks [80].

However, these ambitions are secondary to the fundamental legal and governance challenges. The most significant barrier to any extension of this methodology is that there is currently no lawful basis to use this national dataset beyond emergency directions issued specifically for COVID-19 pandemic research [12]. Therefore, advancing this work requires new legal permissions through transparent public consultation, paired with a clear public benefit. A lawful basis and a social licence must go hand in hand; this requires deep and sustained engagement with patients, the public, and professional bodies. Furthermore, the path from a research model to a trustworthy clinical tool would necessitate additional rigorous evaluation and a clear route to regulatory approval as a medical device. Addressing these socio-technical challenges is the central prerequisite for future progress in this domain.

## 7 Conclusion

We have presented Foresight-E, a 243-million-parameter transformer trained on national-scale EHR data from 54.9 million NHS patients and evaluated on its ability to perform zero-shot prediction across ∼1.4k COVID-related outcomes for a 6.1-million-patient test set. We outline the data integration pipeline, tokenisation strategy, model architecture, training procedure, and inference and evaluation framework.

Although quantitative results are currently withheld pending ongoing discussions, this work demonstrates that it is technically feasible to develop a foundation model for healthcare entirely within existing NHS infrastructure. By combining routinely collected population-scale EHRs with modern generative modelling, Foresight-E offers a blueprint for zero-shot healthcare AI systems.

Rebuilding beyond COVID-restricted datasets, expanding access to broader clinical modalities, and developing safe deployment pathways could enable models like Foresight-E to support both population-level planning and individualised care. Realising this potential will require not only technical advances but also transparent governance, sustained public and professional engagement, and rigorous evaluation in real-world clinical settings to generate evidence for regulatory approval.

## 8 Acknowledgements

The British Heart Foundation Data Science Centre (grant No SP/19/3/34678, awarded to Health Data Research UK) funded co-development (with NHS England) of the SDE service for England, provision of linked datasets, data access, user software licences, computational usage, and data management and wrangling support, with additional contributions from the HDR UK Data and Connectivity component of the UK Government Chief Scientific Adviser’s National Core Studies program to coordinate national COVID-19 priority research. Consortium partner organisations funded the time of contributing data analysts, biostatisticians, epidemiologists, and clinicians.

AWS provided the compute credits which made this work possible. Databricks provided technical support. AWS and Databricks had no access to the underlying datasets or trained AI model and no control over the research or its findings.

SE and CT are funded by the UKRI Centre for Doctoral Training in AI-enabled healthcare (EP/S021612/1). CT also receives support from a King’s College London AI+ Senior Academic Fellowship, a MRC Clinical Top-Up, NIHR Biomedical Research Centre at UCL Hospital NHS Trust, and Health Data Research UK.

AMW is supported by the BHF Data Science Centre (HDRUK2023.0239), Health Data Research UK (Big Data for Complex Disease-HDR-23012), and as an NIHR Research Professor (NIHR303137). Her research is also supported by core funding from the British Heart Foundation (RG/F/23/110103), NIHR Cambridge Biomedical Research Centre (NIHR203312) [\*], BHF Chair Award (CH/12/2/29428), Cambridge BHF Centre of Research Excellence (RE/24/130011), and by Health Data Research UK (HDRUK2023.0028), which is funded by the UK Medical Research Council, Engineering and Physical Sciences Research Council, Economic and Social Research Council, Department of Health and Social Care (England), Chief Scientist Office of the Scottish Government Health and Social Care Directorates, Health and Social Care Research and Development Division (Welsh Government), Public Health Agency (Northern Ireland), British Heart Foundation and the Wellcome Trust.

RD’s work is supported by (1) National Institute for Health Research (NIHR) Biomedical Research Centre at South London and Maudsley NHS Foundation Trust and King’s College London. (2) Health Data Research UK, which is funded by the UK Medical Research Council, Engineering and Physical Sciences Research Council, Economic and Social Research Council, Department of Health and Social Care (England), Chief Scientist Office of the Scottish Government Health and Social Care Directorates, Health and Social Care Research and Development Division (Welsh Government), Public Health Agency (Northern Ireland), British Heart Foundation and Wellcome Trust. (3) The National Institute for Health Research University College London Hospitals Biomedical Research Centre.

This work was carried out with the support of the BHF Data Science Centre led by HDR UK (BHF Grant no. SP/19/3/34678). This study made use of de-identified data held in the NHSE SDE service for England and made available via the BHF Data Science Centre’s CVD-COVID-UK/COVID-IMPACT consortium. This work used data provided by patients and collected by the NHS as part of their care and support. We would also like to acknowledge all data providers who make health-relevant data available for research.

We thank partners at NHS England, AWS, Databricks, and the BHF Data Science Centre for their support, without which this project would not have been possible. We extend our sincere gratitude to the public contributors of the British Heart Foundation Data Science Centre for providing critical review, stimulating discussion, and constructive feedback throughout the project’s lifecycle, including their invaluable guidance in shaping the lay summary.

## 9 Contributor Statement

S.E. and C.T. contributed equally to this work and share joint first authorship.

SE: Conceptualisation, Data curation, Formal analysis, Investigation, Methodology, Software, Validation, Visualisation, Writing – original draft, Writing – review & editing.

CT: Conceptualisation, Data curation, Formal analysis, Funding acquisition, Investigation, Methodology, Project administration, Resources, Software, Supervision, Validation, Visualisation, Writing – original draft, Writing – review & editing.

ZK: Methodology, Writing – review & editing.

SD: Methodology, Writing – review & editing.

HH: Writing – review & editing.

CS: Funding acquisition, Resources, Writing – review & editing.

AMW: Writing – review & editing.

AS: Writing – review & editing.

RD: Conceptualisation, Funding acquisition, Methodology, Resources, Supervision, Writing – review & editing.

In strict accordance with the information governance and data security protocols of the NHS England Secure Data Environment (SDE), access to the underlying datasets and dedicated Foresight cluster was restricted to only CT and SE. Consequently, all data curation, model training, formal data analysis, software implementation, and quantitative evaluation were conducted exclusively by CT and SE within the secure environment. CT is the guarantor for this work.

CS was previously the Director of the British Heart Foundation (BHF) Data Science Centre and coordinated approvals for and access to data within the NHS Digital Trusted Research Environment for England for CVD-COVID-UK/COVID-IMPACT.

## 10 Disclosures

SE contracted part-time for Parexel International during the period this work was conducted.

CT was previously employed by LifeArc, and has received research funding via the UCL-GSK Phenomics Hub from GSK.

ZK is a co-founder of Nuraxi.

RD is a co-founder of CogStack and Onsentia.

ADS receives research funding from BMJ Publishing Group.

The remaining authors declare no competing interests.

None of these commercial organisations had any involvement in the funding, study design, data access, model training, evaluation, or execution of the Foresight-England project, nor do they have access to the underlying data, code, or model weights.

## References

[1] Stephen Armstrong. Nhs england faces investigation over granting foresight access to gp patient data, 2025.

[2] Julia Hippisley-Cox, Carol Coupland, and Peter Brindle. Development and validation of QRISK3 risk prediction algorithms to estimate future risk of cardiovascular disease: prospective cohort study. BMJ, 357, 2017.

[3] Gregory YH Lip, Robby Nieuwlaat, Ron Pisters, Deirdre A Lane, and Harry JGM Crijns. Refining clinical risk stratification for predicting stroke and thromboembolism in atrial fibrillation using a novel risk factor-based approach: the euro heart survey on atrial fibrillation. Chest, 137 (2):263–272, 2010.

[4] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[6] Yikuan Li, Shishir Rao, José Roberto Ayala Solares, Abdelaali Hassaine, Rema Ramakrishnan, Dexter Canoy, Yajie Zhu, Kazem Rahimi, and Gholamreza Salimi-Khorshidi. BEHRT: transformer for electronic health records. Scientific reports, 10(1):7155, 2020.

[7] Laila Rasmy, Yang Xiang, Ziqian Xie, Cui Tao, and Degui Zhi. Med-bert: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. NPJ digital medicine, 4(1):86, 2021.

[8] Zeljko Kraljevic, Dan Bean, Anthony Shek, Rebecca Bendayan, Harry Hemingway, Joshua Au Yeung, Alexander Deng, Alfred Baston, Jack Ross, Esther Idowu, et al. Foresight—a generative pretrained transformer for modelling of patient timelines using electronic health records: a retrospective modelling study. The Lancet Digital Health, 6(4):e281–e290, 2024.

[9] Pawel Renc, Yugang Jia, Anthony E Samir, Jaroslaw Was, Quanzheng Li, David W Bates, and Arkadiusz Sitek. Zero shot health trajectory prediction using transformer. NPJ Digital Medicine, 7(1):256, 2024.

[10] Artem Shmatko, Alexander Wolfgang Jung, Kumar Gaurav, Søren Brunak, Laust Mortensen, Ewan Birney, Tom Fitzgerald, and Moritz Gerstung. Learning the natural history of human disease with generative transformers. medRxiv, 2024.

[11] Shane Waxler, Paul Blazek, Davis White, Daniel Sneider, Kevin Chung, Mani Nagarathnam, Patrick Williams, Hank Voeller, Karen Wong, Matthew Swanhorst, et al. Generative medical event models improve with scale. arXiv preprint arXiv:2508.12104, 2025.

[12] NHS Digital. Control of patient information (COPI) notice. https://digital.nhs.uk/ coronavirus/coronavirus-covid-19-response-information-governance-hub/ control-of-patient-information-copi-notice, 2022. Accessed: 2025-07-14.

[13] Angela Wood, Rachel Denholm, Sam Hollings, Jennifer Cooper, Samantha Ip, Venexia Walker, Spiros Denaxas, Ashley Akbari, Amitava Banerjee, William Whiteley, et al. Linked electronic health records for research on a nationwide cohort of more than 54 million people in england: data resource. BMJ, 373, 2021.

[14] Marta Pineda-Moncusí, Freya Allery, Antonella Delmestri, Thomas Bolton, John Nolan, Johan H Thygesen, Alex Handy, Amitava Banerjee, Spiros Denaxas, Christopher Tomlinson, et al. Ethnicity data resource in population-wide health records: completeness, coverage and granularity of diversity. Scientific Data, 11(1):221, 2024.

[15] NHS England. Secure Data Environment. https://digital.nhs.uk/services/ secure-data-environment-service, 2025. Accessed: 2025-08-13.

[16] Alex Handy, Angela Wood, Cathie Sudlow, Christopher Tomlinson, Frank Kee, Johan H Thygesen, Mohammad Mamouei, Reecha Sofat, Richard Dobson, Samantha Ip, et al. A nationwide deep learning pipeline to predict stroke and COVID-19 death in atrial fibrillation. Medrxiv, 2021.

[17] Freya Allery, Marta Pineda-Moncusí, Christopher Tomlinson, Nikolas Pontikos, Johan H Thygesen, Sara Khalid, and CVD-COVID-UK/COVID-IMPACT Consortium. Towards mitigating health inequity via machine learning: a nationwide cohort study to develop and validate ethnicity-specific models for prediction of cardiovascular disease risk in COVID-19 patients. medRxiv, 2023.

[18] Johan H Thygesen, Huayu Zhang, Hanane Issa, Jinge Wu, Tuankasfee Hama, Ana-Caterina Phiho-Gomes, Tudor Groza, Sara Khalid, Thomas R Lumbers, Mevhibe Hocaoglu, et al. Prevalence and demographics of 331 rare diseases and associated COVID-19-related mortality among 58 million individuals: a nationwide retrospective observational study. The Lancet Digital Health, 7(2):e145–e156, 2025.

[19] Rochelle Knight, Venexia Walker, Samantha Ip, Jennifer A Cooper, Thomas Bolton, Spencer Keene, Rachel Denholm, Ashley Akbari, Hoda Abbasizanjani, Fatemeh Torabi, et al. Association of COVID-19 with major arterial and venous thrombotic diseases: a population-wide cohort study of 48 million adults in england and wales. Circulation, 146(12):892–906, 2022.

[20] Sheng Zhang, Qin Liu, Naoto Usuyama, Cliff Wong, Tristan Naumann, and Hoifung Poon. Exploring scaling laws for EHR foundation models. arXiv preprint arXiv:2505.22964, 2025.

[21] Joseph E Alderman, Joanne Palmer, Elinor Laws, Melissa D McCradden, Johan Ordish, Marzyeh Ghassemi, Stephen R Pfohl, Negar Rostamzadeh, Heather Cole-Lewis, Ben Glocker, et al. Tackling algorithmic bias and promoting transparency in health datasets: the standing together consensus recommendations. The Lancet Digital Health, 7(1):e64–e88, 2025.

[22] Gary S Collins, Karel GM Moons, Paula Dhiman, Richard D Riley, Andrew L Beam, Ben Van Calster, Marzyeh Ghassemi, Xiaoxuan Liu, Johannes B Reitsma, Maarten Van Smeden, et al. Tripod+ ai statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ, 385, 2024.

[23] Karel GM Moons, Johanna AA Damen, Tabea Kaul, Lotty Hooft, Constanza Andaur Navarro, Paula Dhiman, Andrew L Beam, Ben Van Calster, Leo Anthony Celi, Spiros Denaxas, et al. Probast+ ai: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. BMJ, 388, 2025.

[24] Johan H Thygesen, Christopher Tomlinson, Sam Hollings, Mehrdad A Mizani, Alex Handy, Ashley Akbari, Amitava Banerjee, Jennifer Cooper, Alvina G Lai, Kezhi Li, et al. COVID-19 trajectories among 57 million adults in england: a cohort study using electronic health records. The Lancet Digital Health, 4(7):e542–e557, 2022.

[25] NHS Digital. COVID-19 general practice extraction service (gpes) data for pandemic planning and research (GDPPR). https://digital.nhs.uk/ services/data-access-request-service-dars/dars-products-and-services/ data-set-catalogue/gpes-data-for-pandemic-planning-and-research-gdppr, 2025. Accessed: 2025-08-13.

[26] NHS Digital. Hospital Episode Statistics (HES). https://digital.nhs. uk/data-and-information/data-tools-and-services/data-services/ hospital-episode-statistics, 2025. Accessed: 2025-08-13.

[27] Health Data Research Gateway. Civil Registration - Deaths. https://healthdatagateway. org/en/dataset/877, 2024. Accessed: 2025-08-13.

[28] NHS Digital. COVID-19 Second Generation Surveillance System. https: //digital.nhs.uk/services/data-services-for-commissioners/datasets/ covid-19-second-generation-surveillance-system, 2023. Accessed: 2025-01-10.

[29] NHS Digital. COVID-19 Vaccination Status. https://digital.nhs.uk/services/ data-services-for-commissioners/datasets/covid-19-vaccination-status, 2023. Accessed: 2025-01-10.

[30] World Health Organization. International Statistical Classification of Diseases and Related Health Problems (ICD). https://www.who.int/standards/classifications/ classification-of-diseases, 2025. Accessed: 2025-04-14.

[31] NHS Digital. Clinical Classifications. https://digital.nhs.uk/services/ terminology-and-classifications/clinical-classifications, 2025. Accessed: 2025-04-14.

[32] NHS Digital. SNOMED CT. https://digital.nhs.uk/services/ terminology-and-classifications/snomed-ct, 2025. Accessed: 2025-04-14.

[33] NHS Digital. Data Provision Notice General Practice Data for Planning and Research. https://www.spinneybrookmedcentre.co.uk/mf.ashx?ID= 36ab53fe-7e31-4383-b4aa-fd525cf8fa50, 2021. Accessed: 2026-03-06.

[34] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[35] Hugging Face. Llama 2. https://huggingface.co/docs/transformers/en/model\_ doc/llama2, 2023. Accessed: 2025-04-28.

[36] Andrej Karpathy. nanoGPT. https://github.com/karpathy/nanoGPT, 2025. Accessed: 2025-04-15.

[37] Richard D Riley, Gary S Collins, Laura Kirton, Kym IE Snell, Joie Ensor, Rebecca Whittle, Paula Dhiman, Maarten van Smeden, Xiaoxuan Liu, Joseph Alderman, et al. Uncertainty of risk estimates from clinical prediction models: rationale, challenges, and approaches. BMJ, 388, 2025.

[38] NVIDIA. NVIDIA A10 Tensor Core GPU. https://www.nvidia.com/en-gb/ data-center/products/a10-gpu/, 2025. Accessed: 2025-04-14.

[39] AWS. Amazon EC2 G5 Instances. https://aws.amazon.com/ec2/instance-types/g5/, 2025. Accessed: 2025-04-14.

[40] Patrick J Lillie, Alison Samson, Angharad Li, Kate Adams, Richard Capstick, Gavin D Barlow, Nicholas Easom, Eve Hamilton, Peter J Moss, Ashley Gow, et al. Novel coronavirus disease (Covid-19): The first two patients in the UK with person to person transmission. Journal of Infection, 80(5):578–579, 2020. doi: 10.1016/j.jinf.2020.02.020.

[41] Cabinet Office. COVID-19 response: Living with COVID-19. Technical report, UK Government, London, 2022. URL https://www.gov.uk/government/publications/ covid-19-response-living-with-covid-19. Accessed: 2026-05-26.

[42] Amitava Banerjee, Laura Pasea, Steve Harris, Arturo Gonzalez-Izquierdo, Ana Torralbo, Laura Shallcross, Mahdad Noursadeghi, Deenan Pillay, Neil Sebire, Chris Holmes, et al. Estimating excess 1-year mortality associated with the COVID-19 pandemic according to underlying conditions and age: a population-based cohort study. The lancet, 395(10238):1715–1725, 2020.

[43] Simon Ball, Amitava Banerjee, Colin Berry, Jonathan R Boyle, Benjamin Bray, William Bradlow, Afzal Chaudhry, Rikki Crawley, John Danesh, Alastair Denniston, et al. Monitoring indirect impact of COVID-19 pandemic on services for cardiovascular diseases in the uk. Heart, 106(24):1890–1897, 2020.

[44] National Audit Office. Managing NHS backlogs and waiting times in England. https://www. nao.org.uk/reports/managing-nhs-backlogs-and-waiting-times-in-england/, 2022. Accessed: 2026-05-25.

[45] Office for National Statistics. Coronavirus (COVID-19) Infection Survey, antibody data, UK: 18 May 2022. https://www.ons.gov.uk/peoplepopulationandcommunity/ healthandsocialcare/conditionsanddiseases/bulletins/ coronaviruscovid19infectionsurveyantibodyandvaccinationdatafortheuk/ 18may2022, 2022. Accessed: 2026-05-25.

[46] Patrick Wu, Aliya Gifford, Xiangrui Meng, Xue Li, Harry Campbell, Tim Varley, Juan Zhao, Robert Carroll, Lisa Bastarache, Joshua C Denny, et al. Mapping icd-10 and icd-10-cm codes to phecodes: workflow development and initial evaluation. JMIR medical informatics, 7(4): e14325, 2019.

[47] Mark RJ Junge and Joseph R Dettori. ROC solid: Receiver operator characteristic (ROC) curves as a foundation for better diagnostic tests. Global spine journal, 8(4):424–429, 2018.

[48] Mu Zhu. Recall, precision and average precision. Department of Statistics and Actuarial Science, University ofWaterloo, Waterloo, 2(30):6, 2004.

[49] Brice Ozenne, Fabien Subtil, and Delphine Maucort-Boulch. The precision–recall curve overcame the optimism of the receiver operating characteristic curve in rare diseases. Journal ofclinical epidemiology, 68(8):855–859, 2015.

[50] Takaya Saito and Marc Rehmsmeier. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PloS one, 10(3):e0118432, 2015.

[51] Glenn W. Brier et al. Verification of forecasts expressed in terms of probability. Monthly weather review, 78(1):1–3, 1950.

[52] Julia Carrasco-Zanini, Maik Pietzner, Jonathan Davitte, Praveen Surendran, Damien C Croteau-Chonka, Chloe Robins, Ana Torralbo, Christopher Tomlinson, Florian Grünschläger, Natalie Fitzpatrick, et al. Proteomic signatures improve risk prediction for common and rare diseases. Nature medicine, 30(9):2489–2498, 2024.

[53] scikit-learn. LogisticRegression. https://scikit-learn.org/stable/modules/ generated/sklearn.linear\_model.LogisticRegression.html, 2025. Accessed: 2025-08-13.

[54] DMLC XGBoost. XGBoost Documentation. https://xgboost.readthedocs.io/en/ stable/index.html, 2022. Accessed: 2025-07-14.

[55] British Heart Foundation. Home- British Heart Foundation. https:// bhfdatasciencecentre.org/, 2025. Accessed: 2025-07-20.

[56] NHS Digital. Advisory Group for Data (AGD). https://digital. nhs.uk/about-nhs-digital/corporate-information-and-documents/ advisory-group-for-data, 2025. Accessed: 2025-07-20.

[57] NHS Digital. Data Access Request Service (DARS) products and services. https://digital.nhs.uk/services/data-access-request-service-dars/ dars-products-and-services, 2024. Accessed: 2025-07-20.

[58] British Heart Foundation. CVD-COVID-UK / COVID-IMPACT. https:// bhfdatasciencecentre.org/areas/cvd-covid-uk-covid-impact/, 2025. Accessed: 2025-01-10.

[59] Databricks. Databricks Runtime 14.3 LTS. https://docs.databricks.com/aws/en/ release-notes/runtime/14.3lts, 2024. Accessed: 2025-04-29.

[60] Harrison Wilde, Christopher Tomlinson, Bilal A Mateen, David Selby, Hari Krishnan Kanthimathinathan, Padmanabhan Ramnarayan, Pascale Du Pre, Mae Johnson, Nazima Pathan, Arturo Gonzalez-Izquierdo, et al. Hospital admissions linked to sars-cov-2 infection in children and adolescents: cohort study of 3.2 million first ascertained infections in england. BMJ, 382, 2023.

[61] Anmol Arora, Joseph E Alderman, Joanne Palmer, Shaswath Ganapathi, Elinor Laws, Melissa D Mccradden, Lauren Oakden-Rayner, Stephen R Pfohl, Marzyeh Ghassemi, Francis Mckay, et al. The value of standards for health datasets in artificial intelligence-based applications. Nature medicine, 29(11):2929–2938, 2023.

[62] The Lancet Rheumatology. Translating ai innovation into clinical practice. The Lancet Rheumatology, 7(7):e451, 2025. ISSN 2665-9913. doi: 10.1016/S2665-9913(25)00161-4. URL https://doi.org/10.1016/S2665-9913(25)00161-4. Published July 1, 2025.

[63] Olga Kostopoulou, Christopher Tracey, and Brendan C Delaney. Can decision support combat incompleteness and bias in routine primary care data? Journal of the American Medical Informatics Association, 28(7):1461–1467, 2021.

[64] Michael W Sjoding, Robert P Dickson, Theodore J Iwashyna, Steven E Gay, and Thomas S Valley. Racial bias in pulse oximetry measurement. New England Journal of Medicine, 383 (25):2477–2478, 2020.

[65] Salwa S Zghebi, David Reeves, Christos Grigoroglou, Brian McMillan, Darren M Ashcroft, Rosa Parisi, and Evangelos Kontopantelis. Clinical code usage in uk general practice: a cohort study exploring 18 conditions over 14 years. BMJ Open, 12(7):e051456, 2022.

[66] Marta Pineda-Moncusí, Freya Allery, Hoda Abbasizanjani, David Powell, Albert Prats-Uribe, Johan H Thygesen, Angela Wood, Christopher Tomlinson, Amitava Banerjee, Ashley Akbari, et al. Ethnic disparities in COVID-19 mortality and cardiovascular disease in England and Wales between 2020–2022. Nature Communications, 16(1):6059, 2025.

[67] Michael Wornow, Suhana Bedi, Miguel Angel Fuentes Hernandez, Ethan Steinberg, Jason Alan Fries, Christopher Ré, Sanmi Koyejo, and Nigam H Shah. Context clues: Evaluating long context models for clinical prediction tasks on ehrs. arXiv preprint arXiv:2412.16178, 2024.

[68] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[69] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651–4664. PMLR, 2021.

[70] Meta. The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation. https://ai.meta.com/blog/llama-4-multimodal-intelligence/, 2025. Accessed: 2025-07-14.

[71] Ethan Steinberg, Jason Fries, Yizhe Xu, and Nigam Shah. Motor: a time-to-event foundation model for structured medical records. arXiv preprint arXiv:2301.03150, 2023.

[72] Zeljko Kraljevic, Joshua Au Yeung, Daniel Bean, James Teo, and Richard J Dobson. Large language models for medical forecasting–foresight 2. arXiv preprint arXiv:2412.10848, 2024.

[73] Stefan Hegselmann, Georg von Arnim, Tillmann Rheude, Noel Kronenberg, David Sontag, Gerhard Hindricks, Roland Eils, and Benjamin Wild. Large language models are powerful electronic health record encoders. arXiv preprint arXiv:2502.17403, 2025.

[74] Xiaorui Su, Shvat Messica, Yepeng Huang, Ruth Johnson, Lukas Fesser, Shanghua Gao, Faryad Sahneh, and Marinka Zitnik. Multimodal medical code tokenizer. arXiv preprint arXiv:2502.04397, 2025.

[75] Yu Zhao, Yuanbin Qu, Konrad Staniszewski, Szymon Tworkowski, Wei Liu, Piotr Miłos,´ Yuxiang Wu, and Pasquale Minervini. Analysing the impact of sequence composition on language model pre-training. arXiv preprint arXiv:2402.13991, 2024.

[76] Alistair EW Johnson, Lucas Bulgarelli, Lu Shen, Alvin Gayles, Ayad Shammout, Steven Horng, Tom J Pollard, Sicheng Hao, Benjamin Moody, Brian Gow, et al. Mimic-iv, a freely accessible electronic health record dataset. Scientific data, 10(1):1, 2023.

[77] Stephen R Knight, Antonia Ho, Riinu Pius, Iain Buchan, Gail Carson, Thomas M Drake, Jake Dunning, Cameron J Fairfield, Carrol Gamble, Christopher A Green, et al. Risk stratification of patients admitted to hospital with covid-19 using the isaric who clinical characterisation protocol: development and validation of the 4c mortality score. BMJ, 370, 2020.

[78] Hugh Logan Ellis, Edward Palmer, James T Teo, Martin Whyte, Kenneth Rockwood, and Zina Ibrahim. The early warning paradox. npj Digital Medicine, 8(1):81, 2025.

[79] Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In 30th USENIX security symposium (USENIX Security 21), pages 2633–2650, 2021.

[80] Emily Jefferson, James Liley, Maeve Malone, Smarti Reel, Alba Crespi-Boixader, Xaroula Kerasidou, Francesco Tava, Andrew McCarthy, Richard Preen, Alberto Blanco-Justicia, et al. Graimatter green paper: Recommendations for disclosure control of trained machine learning (ml) models from trusted research environments (tres). arXiv preprint arXiv:2211.01656, 2022.

[81] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[82] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.

## A Training

## A.1 Objective

Foresight-E was trained with an autoregressive next-token prediction objective, analogous to the paradigm used for LLMs [5, 8, 9]. Let N be the batch size, T the maximum sequence length, and V the vocabulary size. At each position t, given preceding tokens $y _ { n , < t } ,$ the model outputs a distribution over V. A binary mask $m _ { n , t }$ excludes padding and the initial demographic tokens from loss calculation. Excluding the initial tokens aims to mitigate the risk of biasing the model by learning to predict based solely on demographic features. The loss, L, given the model weights, θ and the true token, $y _ { n , t } ,$ is:

$$
{ \mathcal { L } } = - { \frac { 1 } { \sum _ { n , t } m _ { n , t } } } \sum _ { n = 1 } ^ { N } \sum _ { t = 1 } ^ { T } m _ { n , t } \log P ( y _ { n , t } \mid y _ { n , < t } , \theta )\tag{2}
$$

## A.2 Architecture

We adapted the Llama 2 transformer-decoder [34, 35] architecture with Rotary Positional Embeddings (ROPE) [81] and FlashAttention-2 [82]. Pretrained weights were not imported into the SDE due to governance restrictions and the custom vocabulary; therefore, training was conducted from scratch. We scaled the Llama decoder architecture down to 243 million parameters to facilitate efficient training on NVIDIA A10 GPUs. While we maintained the foundational hyperparameter ratios used across the Llama family [34], we adapted the final configuration to a 12-layer architecture with a hidden size of 1024, 8 attention heads, a feed-forward dimension of 4096, and a 1024-token context window. Additionally, input and output embeddings were tied to optimise parameter efficiency.

## A.3 Training Protocol

We used bfloat16 mixed precision, attention dropout 0.1, gradient clipping (max norm 1.0), and weight decay 0.1. The Adam optimizer had $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ , with a linear warm-up over 3% of steps to a peak learning rate of $\dot { 5 } \times 1 0 ^ { - 4 }$ , then cosine decay. The global batch size was 128, achieved via 8-way data parallelism and gradient accumulation (factor 2). Sequences were right-padded to 1024 tokens. The model was trained for one epoch, with validation loss evaluated every 1,000 steps on 32k sampled sequences. Training was completed in 4 days of wall-clock compute time distributed across 8 NVIDIA A10 GPUs. We did not conduct hyperparameter tuning due to resource constraints; instead, training parameters were selected based on standard defaults from prior literature [8, 36].

## B Endpoint to Token Mapping

Table 2: Mapping of evaluation endpoints and associated figures to timeline tokens. An example phecode definition is shown.
<table><tr><td>Endpoint</td><td colspan="2">Tokens</td></tr><tr><td>Mortality</td><td>DEATH</td><td></td></tr><tr><td>Hospitalisation</td><td>APC</td><td></td></tr><tr><td>Emergency Hospitalisation</td><td>APC_ADMIMETH_21,</td><td>APC_ADMIMETH_22,</td></tr><tr><td rowspan="7">Phecode 8.0 - Intestinal Infection</td><td>APC_ADMIMETH_23, APC_ADMIMETH_24,</td></tr><tr><td>APC_ADMIMETH_25, APC_ADMIMETH_28,</td></tr><tr><td>APC_ADMIMETH_2A, APC_ADMIMETH_2B,</td></tr><tr><td>APC_ADMIMETH_2C, APC_ADMIMETH_2D ICD10_A000, ICD10_A009, ICD10_A011, ICD10_A012,</td></tr><tr><td></td></tr><tr><td>ICD10_A013, ICD10_A014, ICD10_A059, ICD10_A06O,</td></tr><tr><td>ICD10_A062, ICD10_A063, ICD10_A064, ICD10_A065, ICD10_A067, ICD10_A068, ICD10_A069, ICD10_A079</td></tr></table>

## C TRIPOD+AI Checklist

Table 3: Assessment of Foresight-E against the Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD)+AI Checklist [22]. The results section is excluded as not included in this paper.
<table><tr><td>Item</td><td>Checklist item</td><td>Section</td></tr><tr><td>Title</td><td></td><td></td></tr><tr><td>Title</td><td>Identify the study as developing or evaluating the performance of a multivariable prediction model, the target population, and the outcome</td><td>Title</td></tr><tr><td>Abstract</td><td>to be predicted</td><td></td></tr><tr><td>Title</td><td>Identify the study as developing or evaluating the performance of a multivariable prediction model, the target population, and the outcome</td><td>Title</td></tr><tr><td>Background</td><td>to be predicted Provide a brief explanation of the healthcare context and rationale for developing or evalu- ating the performance of all models</td><td>Abstract</td></tr><tr><td>Objectives</td><td>Specify the study objectives, including whether the study describes model develop- ment, evaluation, or both</td><td>Abstract</td></tr><tr><td rowspan="4">Methods</td><td>Describe the sources of data Describe the eligibility criteria and setting</td><td>Abstract Abstract</td></tr><tr><td>where the data were collected Specify the outcome to be predicted by the model, including time horizon of predictions</td><td>Abstract</td></tr><tr><td>in case of prognostic models Specify the type of model, a summary of the model-building steps, and the method for in-</td><td>Abstract</td></tr><tr><td>ternal validation Specify the measures used to assess model performance (eg, discrimination, calibration</td><td>Abstract</td></tr><tr><td rowspan="2">Results</td><td>clinical utility) Report the number of participants and out-</td><td>Pending</td></tr><tr><td>come events Summarise the predictors in the final model Report model performance estimates (with</td><td>Pending</td></tr><tr><td>Discussion</td><td>confidence intervals) Give an overall interpretation of the main re-</td><td>Pending Pending</td></tr><tr><td>Registration</td><td>sults Give the registration number and name of the</td><td>None</td></tr><tr><td></td><td>registry or repository</td><td></td></tr><tr><td>Introduction Background</td><td>Explain the healthcare context (including whether diagnostic or prognostic) and ratio-</td><td>3</td></tr><tr><td rowspan="2"></td><td>nale for developing or evaluating the predic- tion model, including references to existing models Describe the target population and the in-</td><td></td></tr><tr><td>tended purpose of the prediction model in the context of the care pathway, including its in- tended users (eg, healthcare professionals, pa- tients, public)</td><td>3</td></tr><tr><td>Objectives</td><td>Describe any known health inequalities be- tween sociodemographic groups Specify the study objectives, including whether the study describes the development</td><td>3 3</td></tr><tr><td>Methods</td><td>or validation of a prediction model (or both)</td><td></td></tr><tr><td>Data</td><td>Describe the sources of data separately for the development and evaluation datasets (eg, ran- domised trial, cohort, routine care or registry data), the rationale for using these data, and</td><td>4.1</td></tr><tr><td rowspan="3">Participants</td><td>representativeness of the data Specify the dates of the collected participant data, including start and end of participant</td><td>4.1</td></tr><tr><td>accrual; and, if applicable, end of follow-up Specify key elements of the study setting (eg, primary care, secondary care, general popu- lation) including the number and location of</td><td>4.1</td></tr><tr><td>centres Describe the eligibility criteria for study par- ticipants</td><td>4.1</td></tr><tr><td></td><td>Give details of any treatments received, and how they were handled during model develop- ment or evaluation, if relevant</td><td>4.2</td></tr><tr><td>Data prepara- tion Outcome</td><td>Describe any data pre-processing and quality checking, including whether this was similar across relevant sociodemographic groups Clearly define the outcome that is being pre-</td><td>4.1, 4.2</td></tr><tr><td rowspan="2"></td><td>dicted and the time horizon, including how and when assessed, the rationale for choos- ing this outcome, and whether the method of outcome assessment is consistent across so- ciodemographic groups</td><td>4.6</td></tr><tr><td>If outcome assessment requires subjective in- terpretation, describe the qualifications and demographic characteristics of the outcome assessors</td><td>N/A</td></tr><tr><td rowspan="3">Predictors</td><td>Report any actions to blind assessment of the outcome to be predicted Describe the choice of initial predictors (eg,</td><td>None</td></tr><tr><td>literature, previous models, all available pre- dictors) and any pre-selection of predictors before model building</td><td>4.1, 4.2</td></tr><tr><td>Clearly define all predictors, including how and when they were measured (and any ac- tions to blind assessment of predictors for the outcome and other predictors) If predictor measurement requires subjective</td><td>4.1, 4.3</td></tr><tr><td>Sample size</td><td>interpretation, describe the qualifications and demographic characteristics of the predictor assessors Explain how the study size was arrived at (sep- arately for development and evaluation), and</td><td>4.1</td></tr><tr><td>Missing data</td><td>justify that the study size was sufficient to an- swer the research question. Include details of any sample size calculation Describe how missing data were handled. Pro- vide reasons for omitting any data</td><td>4.1, 4.3</td></tr><tr><td rowspan="5">Analytical methods</td><td>Describe how the data were used (eg, for de- velopment and evaluation of model perfor- mance) in the analysis, including whether the data were partitioned, considering any sample size requirements</td><td>4.1</td></tr><tr><td>Depending on the type of model, describe how predictors were handled in the analyses (func- tional form, rescaling, transformation, or any standardisation) Specify the type of model, rationale†, all</td><td>4.1, 4.2, 4.3</td></tr><tr><td>model-building steps, including any hyperpa- rameter tuning, and method for internal vali- dation Describe if and how any heterogeneity in</td><td>4.4</td></tr><tr><td>estimates of model parameter values and model performance was handled and quan- tified across clusters (eg, hospitals, countries). See TRIPOD-Cluster for additional consider- ations</td><td>None</td></tr><tr><td>Specify all measures and plots used (and their rationale) to evaluate model performance (eg, discrimination, calibration, clinical utility) and, if relevant, to compare multiple models</td><td>4.6.1,4.7</td></tr><tr><td rowspan="4"></td><td>Specify all measures and plots used (and their rationale) to evaluate model performance (eg, discrimination, calibration, clinical utility) and, if relevant, to compare multiple models</td><td>4.6.1</td></tr><tr><td>Describe any model updating (eg, recalibra- tion) arising from the model evaluation, either overall or for particular socio-demographic groups or settings</td><td>None</td></tr><tr><td>For model evaluation, describe how the model predictions were calculated (eg, formula, code, object, application programming interface) If class imbalance methods were used, state</td><td>4.5</td></tr><tr><td>why and how this was done, and any subse- quent methods to recalibrate the model or the</td><td>None</td></tr><tr><td>Fairness</td><td>model predictions Describe any approaches that were used to address model fairness and their rationale</td><td>4.1, A.1</td></tr><tr><td>Model output</td><td>Specify the output of the prediction model (eg, probabilities, classification). Provide details and rationale for any classification and how the thresholds were identified</td><td>4.5, 4.6.1</td></tr><tr><td>Training ver- sus evaluation</td><td>Identify any differences between the develop- ment and evaluation data in healthcare setting, eligibility criteria, outcome, and predictors</td><td>4.1, 4.3,4.6.1</td></tr><tr><td>Ethical ap- proval</td><td>Name the institutional research board or ethics committee that approved the study and de- scribe the participant informed consent or the</td><td>4.8</td></tr><tr><td>Open Science</td><td>ethics committee waiver of informed consent</td><td></td></tr><tr><td>Funding</td><td>Give the source of funding and the role of the</td><td>Blinded for review</td></tr><tr><td>Conflicts of in- terest</td><td>funders for the present study Declare any conflicts of interest and financial disclosures for all authors</td><td>Blinded for review</td></tr><tr><td>Protocol</td><td>Indicate where the study protocol can be ac- cessed or state that a protocol was not pre- pared</td><td>Currently not publicly released</td></tr><tr><td>Registration</td><td>Provide registration information for the study, including register name and registration num- ber, or state that the study was not registered</td><td>Not registered</td></tr><tr><td>Data sharing</td><td>Provide details of the availability of the study data</td><td>4.8.1</td></tr><tr><td>Code sharing</td><td>Provide details of the availability of the ana- lytical code</td><td>4.8.2</td></tr><tr><td colspan="3">Patient and public involvement</td></tr><tr><td>Patient and public involve- ment</td><td>Provide details of any patient and public in- volvement during the design, conduct, report- ing, interpretation, or dissemination of the study or state no involvement</td><td>4.8</td></tr><tr><td>Discussion</td><td>Give an overall interpretation of the main re-</td><td></td></tr><tr><td>Interpretation</td><td>sults, including issues of fairness in the con- text of the objectives and previous studies Discuss any limitations of the study (such as a</td><td>Pending</td></tr><tr><td>Limitations</td><td>non-representative sample, sample size, over- fitting, missing data) and their effects on any biases, statistical uncertainty, and generalis-</td><td>6.1</td></tr><tr><td rowspan="2">Usability of the model in the context of current care</td><td>ability Describe how poor quality or unavailable in- put data (eg, predictor values) should be as- sessed and handled when implementing the prediction model</td><td>N/A</td></tr><tr><td>Specify whether users will be required to in- teract in the handling of the input data or use of the model, and what level of expertise is required of users Discuss any next steps for future research,</td><td>6.5</td></tr></table>

## D PROBAST+AI Assessment

Table 4: Assessment of Foresight-E against Risk Of Bias ASsessment Tool (PROBAST) + AI tool [23]
<table><tr><td>Item</td><td>Description</td><td>Section/ Comment</td></tr><tr><td colspan="3">Step 1: PICOTS guidance</td></tr><tr><td>Population</td><td>Define the target population (e.g., patients) in whom the assessed prediction models are to be applied. The target population not only directs search strings and in/exclusion criteria of prediction models or prediction model studies in case of a systematic literature review, but also directs the</td><td>4.1</td></tr><tr><td>Index Model</td><td>applicability assessment. Define the targeted prediction models to be assessed, which may be a single prediction model (the index model) of which the predictive accuracy is meta-analysed across multiple external evaluation studies of that index model but may also address multiple prediction models (developed or evaluated) for the targeted population, outcome or setting, depending</td><td>4.4</td></tr><tr><td>Comparator model(s)</td><td>on the assessor&#x27;s or prediction model review focus. Define the other prediction models whose predictive ability is compared to that of the index model.</td><td>4.7</td></tr><tr><td>Outcome(s)</td><td>Define the outcomes or endpoints that are predicted by the index (and possibly comparator) prediction models in the target population.</td><td>4.6</td></tr><tr><td>Timing</td><td>Define the moment or time-point (e.g., in the patient work- up) at which the prediction with the prediction models is made (i.e., the start point or T0 of the use of the models). Define the time or follow-up period in which the outcomes</td><td>4.6</td></tr><tr><td>Setting and intended</td><td>are being predicted by the prediction models in the targeted population (prediction horizon). Define the healthcare setting or context to which the index</td><td>4.6</td></tr><tr><td>use of the prediction model</td><td>prediction models apply. The prediction ability of models may change across healthcare settings or contexts.</td><td>3</td></tr><tr><td colspan="3">Step 2: Classify the type of prediction model assessment</td></tr><tr><td>Development only</td><td>Prediction model development only, i.e., without evaluation of its performance.</td><td></td></tr><tr><td>Evaluation only Combination</td><td>External validation of one or more existing models in new data Prediction model development combined in the same</td><td></td></tr><tr><td></td><td>study(publication) with the evaluation of its apparent per- formance, internal validation performance, or external vali- dation performance.</td><td></td></tr><tr><td colspan="3">Step 3: Assess quality and applicability or risk of bias and applicability</td></tr><tr><td rowspan="6">Participants and data sources</td><td>Describe the sources of data and criteria for participant</td><td></td></tr><tr><td>selection</td><td>4.1</td></tr><tr><td>Were appropriate data sources used? Was an appropriate study design used?</td><td>Yes Yes</td></tr><tr><td>Did the in- and exclusions of study participants result in a representative dataset?</td><td>Yes</td></tr><tr><td>Concern regarding quality of selection of participants and</td><td>Low</td></tr><tr><td>data sources Concern that the (data of the) included participants do not</td><td>Low</td></tr><tr><td>the prediction model</td><td>match the review question or the assessor&#x27;s intended use of</td><td></td></tr></table>

Overall concern re- Low concern regarding quality- If all four domains were ✓ garding quality of the rated low concern regarding quality.   
prediction model de  
velopment

Step 4: Assess the overall concerns regarding quality, risk of bias and applicability of the prediction model
<table><tr><td>Predictors</td><td>List and describe predictors included in the final prediction 4.1, 4.3 model, how they were defined and assessed, and their timing of assessment</td><td></td></tr><tr><td></td><td>Were predictors defined and assessed in a similar way for Yes all participants?</td><td></td></tr><tr><td>pants?</td><td>Was any pre-processing of predictors similar for all partici- Yes</td><td></td></tr><tr><td></td><td>Were predictor assessments made without knowledge of Yes outcome data?</td><td></td></tr><tr><td></td><td>Were the predictors included in the model available at theYes time the model was intended to be used?</td><td></td></tr><tr><td>assessment</td><td>Concern regarding the quality of the predictors or their Low</td><td></td></tr><tr><td>Outcome</td><td>Concern that the definition, pre-processing, assessment, or Low timing of assessment of the predictors in the model do not</td><td></td></tr><tr><td></td><td>match the review question or the assessor&#x27;s intended use Describe the outcome, how it was defined and determined, 4.6 and the time interval between predictor assessment and</td><td></td></tr><tr><td></td><td>outcome determination At what time point was the outcome determined? If a 4.6 composite outcome was used, describe the relative fre-</td><td></td></tr><tr><td></td><td>quency/distribution of each contributing outcome? Were outcomes defined and assessed appropriately?</td><td>Yes</td></tr><tr><td></td><td>Were outcomes defined and assessed in a similar way for Yes all participants?</td><td></td></tr><tr><td>of predictor data?</td><td>Were outcome assessments made without use or knowledge Yes</td><td></td></tr><tr><td>tion</td><td>Was the time interval between predictor assessment and Yes outcome assessment appropriate?</td><td></td></tr><tr><td></td><td>Concern regarding quality of the outcome or its determina- Low</td><td></td></tr><tr><td>Analysis</td><td>Concern that the outcome, its definition, assessment, or Low timing of assessment do not match the review question or</td><td></td></tr><tr><td></td><td>the assessor&#x27;s intended use Describe the numbers of participants, number of candidate 4.1, 4.6</td><td></td></tr><tr><td></td><td>predictors, number of outcome events Describe how the prediction model was developed (e.g., 4.4 with respect to modelling technique, predictor selection,</td><td></td></tr><tr><td></td><td>and classification or risk group definition) Describe the performance measures of the prediction model, 4.6.1 e.g., (re)calibration, discrimination, (re)classification, net</td><td></td></tr><tr><td></td><td>benefit, and whether they were adjusted for optimism Describe missing data on predictors and outcomes as well 4.3</td><td></td></tr><tr><td></td><td>as methods used for handling these missing data Was there evidence that the sample size was reasonable?</td><td>Yes</td></tr><tr><td>priately?</td><td>Were continuous and categorical predictors handled appro- Yes</td><td></td></tr><tr><td></td><td>Were participants with missing or censored data handled N/A</td><td></td></tr><tr><td></td><td>appropriately in the analysis? If methods to address class imbalance were used, was the N/A</td><td></td></tr><tr><td></td><td>model or the model predictions recalibrated?</td><td>Yes</td></tr><tr><td></td><td>Were methods used to address potential model overfitting? Concern regarding quality of the analysis</td><td>Low</td></tr></table>

<table><tr><td></td><td>High concern regarding quality- If at least one domain was rated high concern regarding quality. Unclear concern regarding quality- If at least one domain was rated unclear concern regarding quality and no domains were rated high concern.</td></tr><tr><td>garding applicability of the prediction model development</td><td>Overall concern re- Low concern for applicability- If all three domains were y rated low concern for applicability</td></tr><tr><td>rated high concern for applicability. was rated unclear concern for applicability and no domains</td><td>High concern for applicability- If at least one domain was Unclear concern for applicability- If at least one domain</td></tr></table>