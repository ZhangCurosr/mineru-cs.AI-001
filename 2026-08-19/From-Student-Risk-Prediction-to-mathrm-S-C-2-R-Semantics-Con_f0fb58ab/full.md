# From Student Risk Prediction to $\mathrm { S C ^ { 2 } R }$ Semantics-Constrained Counterfactual Recourse for Educational Decision Support

Ngoc Luyen Le<sup>∗†</sup> ngoc-luyen.le@hds.utc.fr

Marie-Hel´ ene Abel\` <sup>†</sup> marie-helene.abel@hds.utc.fr

Bertrand Laforge<sup>‡</sup> laforge@lpnhe.in2p3.fr

<sup>∗</sup>Gamaizer, 93340 Le Raincy, France.

<sup>†</sup>Universite de Technologie de Compi´ egne, CNRS, Heudiasyc (Heuristics and Diagnosis of Complex Systems),\` CS 60319 - 60203 Compiegne Cedex, France.\`

<sup>‡</sup>Sorbonne Universite, CNRS UMR 7585, LPMHE (Laboratoire de Physique Nucl ´ eaire et des Hautes ´ Energies), <sup>´</sup>

75252 Paris cedex 05, France

Abstract—Learning analytics models can identify students at risk of poor performance, but they do not directly indicate which interventions are feasible, actionable, and compatible with educational constraints. This paper introduces $\mathbf { s } \mathbf { \hat { C } } ^ { 2 } \mathbf { R } .$ , a semanticsconstrained counterfactual recourse framework for educational decision support. $\mathbf { S C ^ { 2 } R }$ combines a calibrated predictive model, integer-programming-based recourse generation over discrete action variables, a lightweight RDF vocabulary for interventionplan representation, and SHACL validation for enforcing timing, budget, immutability, and availability constraints. The framework is evaluated offline on the OULAD dataset using snapshots constructed relative to each assessment at two decision horizons. Results show that the predictive component provides strong performance, that compact intervention plans can be generated at scale, and that semantic validation reveals infeasible plans that lighter optimization-only settings would otherwise accept. Rather than claiming causal improvement in student outcomes, this work shows that counterfactual recourse becomes more operationally meaningful in education when recommendations are not only model-valid, but also semantically feasible and machine-checkable.

Index Terms—learning analytics, counterfactual recourse, educational decision support, SHACL, semantic validation, RDF

## I. INTRODUCTION

Learning analytics has made progress in predicting student failure, disengagement, and dropout risk from educational traces, assessment records, and learner profiles. However, predictive performance alone is insufficient for effective educational decision support. In practice, instructors, advisors, and student support services need recommendations that are actionable, contextually appropriate, and operationally feasible, rather than risk scores in isolation. Recent work in learning analytics and AI in education has highlighted this need to move beyond prediction toward systems that better support intervention, human oversight, and trustworthiness [1]–[3].

A promising direction for bridging prediction and action lies in counterfactual explanations and algorithmic recourse. These approaches aim to identify what should change in order to obtain a more desirable model outcome, with recourse methods emphasizing actionability for the affected individual [4], [5]. In educational settings, however, application of recourse remains challenging. Many methods operate primarily in feature space and may therefore generate recommendations that are mathematically valid but difficult to enact in practice. Educational recommendations must often respect limited time before an assessment, bounded intervention effort, immutable learner attributes, and the availability of pedagogical resources. In addition, explainable learning analytics has shown that usefulness depends not only on interpretability, but also on stability, practical validity, and alignment with stakeholder needs [6].

This paper addresses these limitations through $\mathbf { S C ^ { 2 } R _ { \delta \alpha } } -$ Semantics-Constrained Counterfactual Recourse – a framework for educational decision support that combines predictive modeling, integer-programming-based recourse generation over discrete action variables, an OWL/SKOS vocabulary for intervention actions, and SHACL validation for enforcing timing, budget, immutability, and optional availability con straints [7], [8]. The contribution of the paper is intentionally focused: rather than claiming causal educational impact, we investigate whether counterfactual recourse becomes more reliable and operationally meaningful when its outputs are represented as machine-checkable intervention plans subject to explicit semantic constraints. The results show that constrained optimization can generate compact plans at scale and, more importantly, that richer semantic constraints expose infeasible recommendations that lighter optimization-only settings would otherwise accept. These findings suggest that, for educational decision support, counterfactual recourse becomes substantially more defensible when recommendations are not only model-valid, but also semantically validated, interpretable, and operationally feasible.

The remainder of this paper is organized as follows. Section II reviews the relevant literature. Section III formalizes the problem. Section IV presents $\mathbf { S C ^ { 2 } R } ,$ including its predictive, recourse, and semantic validation components. Section V describes the experimental setup and evaluation protocol. Section VI reports the empirical results. Section VII discusses the findings and limitations. Finally, Section VIII concludes the paper and outlines directions for future work.

## II. RELATED WORK

The present work lies at the intersection of learning analytics for student support, counterfactual explanations and algorithmic recourse, and semantic technologies for validation and feasibility checking. Although each of these areas has advanced substantially, their integration into a unified framework for educational decision support remains limited.

Learning analytics has produced a large body of work on the prediction of student failure, disengagement, and dropout from educational traces, assessment histories, and learner profiles. These studies have demonstrated the value of predictive models for identifying students who may require support. However, several studies have also emphasized that predictive performance alone is not sufficient for educational usefulness. In practice, instructors, advisors, and support services need outputs that can inform timely, actionable, and contextually meaningful intervention, rather than risk scores in isolation [3]. This limitation is important in educational settings, where recommendations are typically interpreted by human stakeholders and must remain compatible with pedagogical and institutional constraints. Recent work has improved the reliability of educational early-warning models through leakage-excluded evaluation [9], but has not addressed actionable recourse.

Counterfactual explanations and algorithmic recourse help bridge this gap by identifying changes that could lead to a more desirable model outcome [4], [10], [11]. Prior work has studied desirable properties such as sparsity, diversity, plausibility, robustness, and feasibility [5], [12], [13], and has increasingly framed recourse as an intervention-oriented problem under actionability constraints [14]. However, most existing approaches remain defined mainly in feature space, which can produce recommendations that are model-valid but difficult to justify or implement in practice.

This limitation is particularly important in education, where recommendations must satisfy temporal, organizational, and pedagogical constraints. Suggested actions may need to occur before an assessment, remain within a bounded intervention effort, avoid changing immutable learner attributes, and match the actual availability of pedagogical resources. Recent work in explainable learning analytics also emphasizes that usefulness depends not only on interpretability, but also on practical validity, stability, and alignment with stakeholder expectations [6].

Semantic Web technologies offer a natural way to formalize such requirements. SHACL provides a W3C-standard mechanism for validating RDF graphs against declarative constraints and is therefore well suited to expressing conditions related to timing, budget, immutability, and resource availability [8]. Although widely used in knowledge graph validation, such semantic technologies have rarely been integrated into counterfactual recourse pipelines.

Against this background, this paper combines predictive modeling, constrained counterfactual recourse, and semantic validation within a unified framework for educational decision support. The objective is not only to generate recommendations that are model-valid, but also to represent them as machine-checkable, interpretable, and semantically feasible intervention plans. In this respect, the paper addresses a gap left by prior work, which has generally treated prediction, recourse, and semantic validation as separate concerns rather than as components of the same decision-support pipeline.

## III. PROBLEM FORMULATION

We consider the problem of generating feasible intervention plans for students in a digital learning environment. Let $x _ { t } \in \mathbb { R } ^ { p }$ denote the state of a student observed at snapshot time t, with snapshots constructed at predefined times relative to the due date d of an upcoming assessment or learning milestone (e.g., $t = d - 1 4$ or $t = d - 7 )$ . This state is constructed from information typically available in learning management systems, including learner profile attributes, prior assessment history, interaction traces with learning resources and activities, engagement indicators, and temporal variables derived from the course schedule. Based on this representation, a predictive model $f _ { h } ( x _ { t } )$ estimates the probability of success on the next assessment, where $h = d - t$ denotes the remaining time-to-deadline horizon (e.g., $h = 1 4$ or $h = 7$ days).

The objective is not only to predict the likelihood of success or failure, but also to identify candidate actions that may support a more desirable outcome. To this end, we define a counterfactual intervention plan as $\pi = \{ a _ { 1 } , \ldots , a _ { m } \}$ , where each $a _ { i }$ denotes a discrete, human-readable intervention action. Such actions are intended to modify aspects of learner engagement or study behavior within the remaining time window before the next assessment. Applying the plan π transforms the student state into a post-intervention state denoted by $x _ { t } \oplus \pi$

The problem is then to identify an intervention plan that minimizes intervention burden while satisfying a target decision threshold:

min c(π) s.t. $f _ { h } ( x _ { t } \oplus \pi ) \geq \tau , \pi \in { \cal A } , { \cal S } ( G _ { \pi } ) = 1 .$ . (1)

Here, τ denotes a generic target decision threshold for recourse generation. The numeric value $\tau = 0 . 6 0$ shown in Fig. 1 is purely illustrative and is included only to clarify how candidate plans are accepted. This formulation is consistent with the broader recourse literature, where the objective is to identify low-cost changes that achieve a desired model outcome while remaining actionable [4], [10], [14]. Here, $c ( \pi )$ denotes a weighted intervention cost, A is the allowable action space, $G _ { \pi }$ is the RDF graph representing the intervention plan, and $S ( G _ { \pi } )$ is a binary SHACL conformance function. Under this formulation, a plan is acceptable only if it both achieves the desired predictive objective and satisfies the semantic feasibility constraints encoded in the validation layer.

The constraints considered in this work are intentionally explicit and operational. First, timing constraints require that all actions be schedulable before the due date of the next assessment and within the available snapshot-to-deadline window. Second, budget constraints ensure that the total intervention burden remains below a configurable upper bound. Third, immutability constraints prevent the recourse mechanism from modifying fixed learner characteristics or non-actionable assessment metadata. Fourth, an optional availability constraint requires that actions involving learning resources or activities be compatible with their availability conditions in the platform.

As illustrated in Fig. 1, educational recourse is therefore not treated as a purely geometric perturbation problem in feature space, but as a constrained decision-support problem. In this setting, a recommendation is useful only if it is not only model-valid, but also feasible to enact and understandable to human stakeholders. The next section describes the framework used to operationalize this formulation.

![](images/dbe3376673be6186caf7cf884083a0f9345f6267634514c1af4846d40efa3b2d.jpg)  
Fig. 1: Illustration of the problem formulation. Candidate plans are retained only if they satisfy both the prediction target and the semantic constraints.

## IV. $\mathsf { S C } ^ { 2 } \mathsf { R } \colon$ SEMANTICS-CONSTRAINED COUNTERFACTUALRECOURSE FRAMEWORK

SC<sup>2</sup>R consists of three main components: a predictive component for estimating next-assessment success, a recourse component for generating candidate intervention plans, and a semantic validation layer for checking feasibility. Together, these components turn counterfactual recourse into a machinecheckable decision-support process.

Formally, SC<sup>2</sup>R proceeds as follows. Starting from a student snapshot $x _ { t } ,$ the predictive component computes $f _ { h } ( x _ { t } )$ , the probability of success at horizon h. The recourse component searches for an intervention plan $\pi \in { \mathcal { A } }$ that minimizes the intervention cost $c ( \pi )$ while producing a post-intervention state $x _ { t }$ ⊕ π such that $f _ { h } ( x _ { t } \oplus \pi ) \ge \tau$ . Each candidate plan is represented as an RDF graph $G _ { \pi }$ and retained only if it satisfies the semantic feasibility condition $S ( G _ { \pi } ) = 1$

Fig. 2 presents the overall $\mathbf { S C ^ { 2 } R }$ framework. Starting from data available in a learning management system or digital learning environment, the pipeline first constructs snapshot representations at predefined decision horizons. A calibrated predictive model is then applied to estimate the probability of success on the next assessment. Based on this prediction, an optimization module generates candidate intervention plans over a discrete action space. These plans are represented as RDF graphs and passed through a semantic validation layer based on SHACL. Only plans that satisfy both the predictive objective and the explicit semantic constraints are retained for downstream evaluation. The resulting outputs are finally analyzed through predictive and recourse evaluation metrics.

## A. Predictive Component

The predictive component maps each student snapshot $x _ { t }$ to a probability of success $f _ { h } ( x _ { t } )$ for the next assessment at horizon h. In this work, the primary predictor is a calibrated logistic-regression pipeline. This choice is deliberate. A linear model is transparent, computationally efficient, and compatible with actionable recourse formulations for linear decision boundaries [4]. Numeric features are imputed and standardized before classification. Model calibration is assessed using the Brier score, which is especially relevant because the downstream recourse mechanism operates on predicted probabilities rather than on hard class labels [15], [16].

![](images/2c507da164768bff6c34df3bf26099c552b4599159179e69b32d74515de4677a.jpg)  
Fig. 2: Overview of $\mathbf { S C ^ { 2 } R } .$ . From student snapshots, the pipeline predicts next-assessment success, generates intervention plans, applies semantic validation, and retains only conformant plans for evaluation.

Although the broader implementation also includes stronger nonlinear models, such as gradient-boosted trees and neural architectures, the main paper does not rely on them. Their role is supplementary, whereas the calibrated linear baseline provides a more interpretable foundation for the $\mathbf { S C ^ { 2 } R }$ framework developed here.

## B. Recourse Generation

Given a student state $x _ { t }$ and predictor $f _ { h } ,$ , the recourse component searches over the allowable action space A for an intervention plan π that minimizes the weighted cost $c ( \pi )$ while ensuring that the post-intervention state $x _ { t } \oplus \pi$ satisfies the target condition $f _ { h } ( x _ { t } \oplus \pi ) \ \geq \ \tau$ . Rather than recommending arbitrary perturbations in feature space, the optimizer selects among interpretable intervention increments derived from learner engagement and study-related variables, such as increasing interaction with particular categories of learning resources or activities during the remaining study window. The total intervention burden is modeled through a weighted cost function, optionally combined with a sparsity preference so that shorter plans are favored.

The main solver is formulated as an integer program. This choice is natural because intervention variables are discrete, several feasibility conditions are combinatorial, and the optimization objective explicitly minimizes intervention burden. A generated plan is considered model-valid if the corresponding post-intervention state crosses the target decision threshold under the predictive model, that is, if $f _ { h } ( x _ { t } \oplus \pi ) \ge \tau$ . However, model validity alone is not sufficient. In implementation, the optimization module first generates candidate plans over the discrete action space, after which semantic feasibility is enforced through SHACL validation. Thus, the formal condition ${ \cal S } ( G _ { \pi } ) = 1$ is realized as a validation step between plan generation and downstream evaluation.

## C. Semantic Validation Layer

The semantic validation layer provides the representation and checking mechanisms that make candidate intervention plans operationally meaningful. It combines a lightweight ontology for intervention-plan representation with SHACLbased validation constraints.

We define a lightweight RDF vocabulary for interventionplan representation, using the prefix cbe: for terms introduced within a competency-based education ontology extended here for recourse representation [17]. The vocabulary is centered on two main classes, cbe:ActionPlan and cbe:Action. An intervention plan is modeled as an instance of cbe:ActionPlan, linked to one or more actions through cbe:hasAction, and associated with the target student and target assessment through cbe:forStudent and cbe:forAssessment, as shown in Fig. 3. At the data level, a plan records the snapshot day, due day, and budget, while each action records its activity type, intended engagement change, action cost, and scheduled time interval. We combine a lightweight ontology with SKOS because these two layers serve complementary purposes: the ontology captures the formal structure of plans, actions, students, and assessments, whereas SKOS provides a controlled and extensible classification of activity types. For example, an intervention step can be represented structurally as an instance of cbe:Action linked to a cbe:ActionPlan, while its pedagogical category is assigned through a SKOS concept such as quiz practice, forum participation, or resource review. This design makes the action space explicit, reusable across plans, and easy to refine or align without introducing unnecessary ontological commitments.

![](images/1ebb3411ab05c664dba567b330174e956a2cc9c2443a20219ec7992979905994.jpg)  
Fig. 3: Ontology fragment for representing intervention plans, centered on cbe:ActionPlan and cbe:Action.

## Listing 1: Compact SHACL summary.

ActionPlanShape:   
targetClass ActionPlan   
requires {snapshotDay:int, dueDay:int, budget:decimal,   
hasAction -> ActionShape}   
ActionShape:   
targetClass Action   
requires {activityType, deltaClicks:int>=0,   
actionCost:decimal>=0,   
scheduledFromDay:int, scheduledToDay:int}   
SPARQL constraints:   
C1 (timing): snapshotDay <= from <= to <= dueDay   
C2 (budget): sum(actionCost) <= budget   
C3 (availability): availableFromDay(activityType) <= from   
and to <= availableToDay(activityType)

Each candidate intervention plan π generated by the recourse module is encoded as an RDF graph $G _ { \pi }$ using this vocabulary. This representation makes the plan machine-readable and structured, while preserving the distinction between the plan as a whole and the individual actions it contains.

SHACL shapes are used to determine whether $S ( G _ { \pi } ) = 1$ that is, whether the RDF graph of the candidate plan satisfies the semantic feasibility conditions of $\mathbf { S C ^ { 2 } R }$ [8]. At the structural level, the shapes require the presence of the core plan and action properties. At the constraint level, they enforce timing and budget consistency, and optionally availability compatibility for activity types whose valid time windows are known. Listing 1 summarizes the main shapes and constraints used in the current implementation.

TABLE I: Compact intervention vocabulary used in $\mathbf { S C ^ { 2 } R } .$
<table><tr><td>Intervention</td><td>Feature</td><td>Increments Cost</td><td>Constraints</td><td></td></tr><tr><td>Review quizzes</td><td>Quiz clicks</td><td> $^ { + 1 , + 2 , + 3 }$ </td><td></td><td>1/inc. Before due day; available</td></tr><tr><td>Revisit pages</td><td>Page clicks</td><td>+1 to +5</td><td> $1 / 2 \mathsf { p }$ </td><td>Before due day</td></tr><tr><td>Attend support</td><td>Support count</td><td>+1</td><td>3</td><td>Session available; before due day</td></tr><tr><td>Prep activity</td><td>Prep feature</td><td>+1</td><td>2</td><td>Schedule-compatible</td></tr></table>

Table I summarizes the intervention vocabulary used in $\mathbf { S C ^ { 2 } R } ,$ including the affected feature(s), admissible increments, intervention costs, and semantic feasibility conditions. It clarifies how candidate recourse actions are grounded in interpretable educational interventions and subsequently validated through the semantic layer.

In addition to these checks, the semantic layer supports the immutability principle adopted in this work by ensuring that candidate plans act only on allowable intervention variables and not on fixed learner characteristics or non-actionable assessment metadata. This layer is essential because feasibility cannot be reduced to geometric proximity in feature space. Two plans may have similar numerical cost while differing substantially in whether they can be completed before the assessment deadline or whether the required pedagogical resources are actually available. The semantic layer therefore provides an explicit feasibility-checking layer around the recourse optimizer, ensuring that retained recommendations are not only model-valid, but also feasible and interpretable.

Taken together, $\mathbf { S C ^ { 2 } R }$ provides a framework in which predictive modeling, constrained recourse generation, and semantic validation are treated as parts of the same decision-support pipeline. The result is not merely a set of counterfactual feature changes, but a structured intervention artifact that can be checked and interpreted before any downstream use.

## V. EXPERIMENTAL SETUP

This section presents the dataset, snapshot protocol, comparison methods, and evaluation metrics used in the study<sup>1</sup>.

## A. Dataset and Snapshot Protocol

We evaluate $\mathbf { S C ^ { 2 } R }$ on the OULAD dataset, which provides student demographics, assessment schedules and scores, virtual learning environment (VLE) metadata, and daily interaction summaries across multiple module presentations [18]. We use the tables studentInfo, assessments, studentAssessment, vle, and studentVle.

For each assessment with due day d, we construct two snapshots per eligible student, at $t = d { - } 1 4$ and $t = d { - } 7 .$ . Only information available up to the snapshot time is used; post-t interactions and assessments are excluded. Early assessments that would otherwise induce negative snapshot times are handled during preprocessing.

TABLE II: Dataset and protocol summary.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Snapshot horizons</td><td>d—14, d—7 before next assessment</td></tr><tr><td>Train / val / test presentations</td><td>15/2/5</td></tr><tr><td>Instances per horizon (train / val / test)</td><td> $1 9 0 , 7 5 2 / 2 9 , 0 5 4 / 8 4 , 7 9 1$ </td></tr><tr><td>Primary label</td><td>Next-assessment pass/fail (score  $\geq 4 0 )$ </td></tr><tr><td>Primary recourse method Full-scale IP plans scored of- fline</td><td>Integer programming on action variables 127,972</td></tr></table>

The feature set includes aggregated click counts by activity type over rolling 7-, 14-, and 28-day windows, availabilityalignment features, prior assessment history, time-to-deadline variables, and immutable learner descriptors. The prediction target is next-assessment pass/fail, where pass is defined as score $\geq 4 0 .$ To avoid leakage across closely related offerings, data are split at the module-presentation level. The frozen split contains 15 training presentations, 2 validation presentations, and 5 test presentations. Unless otherwise stated, results are reported on the held-out test set, with d−14 as the main horizon and d−7 used as a secondary horizon.

## B. Comparison Methods

The primary predictive component is a calibrated logisticregression model, used as the reference predictor for $\mathbf { S C ^ { 2 } R } .$ Although the implementation also includes stronger nonlinear models, including XGBoost [19], a TabTransformer-style architecture [20], and a BiLSTM-based model [21], [22], these are treated as supplementary analyses.

For recourse generation, the main method is an integerprogramming (IP) formulation over discrete action variables. As a baseline, we use a Wachter-style counterfactual search [10] on a controlled 200-case subset. A stricter IP variant with availability constraints is also included to assess the added value of richer semantic validation.

## C. Evaluation Metrics

We evaluate the framework along two dimensions: (i) predictive quality is measured using AUC, F1, accuracy, average precision, and Brier score; and (ii) recourse quality is measured using model validity, SHACL conformance, intervention cost, number of actions, predicted probability gain, k-nearestneighbor plausibility, and stability under noise and retraining.

## VI. EXPERIMENTAL RESULTS

## A. Predictive Quality

Fig. 4 summarizes the predictive results on the test set. The calibrated logistic-regression baseline provides strong performance, with AUC values of 0.884 at d−14 and 0.889 at d−7. F1 and average precision follow the same pattern, indicating that the later snapshot benefits from more recent learner evidence. The Brier scores are 0.128 and 0.122, which indicate usable calibration for downstream recourse generation.

Supplementary checks with stronger nonlinear models do not materially change the central picture of the paper. On the d−14 split, XGBoost and the TabTransformer-style model reach an AUC of 0.898, while the MLP reaches 0.895 and the BiLSTM 0.888. These improvements remain modest relative to the calibrated logistic baseline. For this reason, logistic regression remains the reference predictor for $\mathbf { S C ^ { 2 } R }$

![](images/7f9304061d40132f02f1d599272a1a41514a019aec342da59bef8236a9a825a5.jpg)  
Fig. 4: Predictive performance overview. Left: logistic regression performs strongly at both decision horizons, with slightly better results at $d - 7 .$ Right: supplementary nonlinear models yield only modest AUC gains on the d−14 split.

TABLE III: Full-scale IP recourse summary.
<table><tr><td>Metric</td><td>Value</td><td>Metric</td><td>Value</td></tr><tr><td>Plans scored</td><td></td><td>127,972 Mean number of actions</td><td>1.075</td></tr><tr><td>SHACL conformance rate</td><td>1.000</td><td>Mean predicted gain</td><td>0.383</td></tr><tr><td>Model validity rate</td><td>1.000</td><td>Mean 5-NN plausibility distance</td><td>6.950</td></tr><tr><td>Mean total cost</td><td>10.357</td><td>Mean noise stability</td><td>0.656</td></tr><tr><td>Median total cost</td><td>11.000</td><td>Mean retrain stability</td><td>0.228</td></tr></table>

## B. Recourse Quality at Scale

The main full-scale integer-programming evaluation produces 127,972 scored plans. Table III summarizes the results. Both model validity and SHACL conformance reach 1.0. The generated plans are also compact, with an average of 1.075 actions and a mean total cost of 10.357. The mean predicted probability gain is 0.383. These results show that $\mathbf { S C ^ { 2 } R }$ can generate low-cardinality intervention plans at scale while satisfying both the predictive objective and the semantic feasibility constraints considered in the main evaluation setting.

Plausibility provides a more nuanced picture. The mean 5- nearest-neighbor distance of the recourse-adjusted states is 6.950, suggesting that the generated plans remain close to observed learner profiles without collapsing to trivial copies of successful historical cases. Stability under noise and retraining is discussed together with the semantic ablation in Fig. 5.

## C. Effect of Semantic Validation

The contribution of the semantic layer is most visible in the controlled 200-case ablation shown in Fig. 5. The IP solver and the Wachter-style baseline have nearly identical average cost (24.127 vs. 24.175), and both achieve perfect model validity and SHACL conformance under the baseline constraint configuration. However, when availability constraints are activated in the IP pipeline, the conformance rate decreases to 0.875 without any reduction in optimization cost. This is an informative result rather than a negative one: it shows that richer semantic constraints reveal plans that would otherwise appear acceptable under a lighter constraint regime.

Fig. 5 also reports the robustness indicators associated with the generated plans. Noise stability is moderate (0.656), whereas retrain stability is lower (0.228). This suggests that the recourse outputs are more stable under small perturbations of the input than under changes in the learned decision boundary.

![](images/290f9f4a945d2d27ce40c70e42384361c1d2897c5bd2528f090273a9a2147ff0.jpg)  
Fig. 5: Effect of semantic validation and robustness. Left: 200- case ablation comparing IP recourse, Wachter-style recourse, and IP recourse with availability constraints. Right: robustness indicators under input noise and retraining.

## VII. DISCUSSION

The results support a focused interpretation of $\mathbf { S C ^ { 2 } R } .$ First, the predictive component provides a strong and wellcalibrated basis for downstream recourse generation. Second, the integer-programming formulation produces compact intervention plans at scale, suggesting that discrete recourse can be operationalized efficiently in this setting. Third, and most importantly, the semantic layer changes how recourse outputs should be interpreted: its contribution is not to reduce optimization cost, but to distinguish between plans that are merely model-valid and plans that remain feasible under explicit timing, budget, immutability, and availability constraints.

At the same time, the scope of the study remains intentionally limited. The evaluation is offline, model-based, and observational, and therefore does not establish that following a generated plan would improve student outcomes. In addition, the current action vocabulary is necessarily simplified and derived from structured learner activity signals rather than richer pedagogical interaction data. The lower retrain stability also suggests that some recommendations may remain sensitive to changes in the learned predictive boundary.

These observations indicate that $\mathbf { S C ^ { 2 } R }$ should be understood as a framework for machine-checkable and semantically feasible educational recourse, rather than as evidence of causal intervention effectiveness. Its value lies in structuring and validating candidate intervention plans before any practical use by instructors, advisors, or learning-support services.

## VIII. CONCLUSION AND PERSPECTIVE

This paper introduced $\mathbf { S C ^ { 2 } R } ,$ , a semantics-constrained counterfactual recourse framework for educational decision support. Using the OULAD dataset, the results showed that a calibrated logistic-regression model provides a strong reference predictor, that integer-programming-based recourse can generate compact plans at scale, and that semantic validation plays a central role in detecting hidden infeasibility beyond optimization alone. The main perspective of this work is to move from model-consistent recourse toward more educationally grounded intervention support settings. Future work will examine dynamic replanning, institution-specific constraints, and recourse robustness across retrained models.

## ACKNOWLEDGMENTS

We warmly thank the Ikigai consortium led by the association Games for Citizens, the company Gamaizer, as well as the FORTEIM project (winner of the AMI CMA France 2030 call for projects), for their support and collaboration. Their contributions have provided significant added value to the completion of this research.

## REFERENCES

[1] G. Akc¸apınar, A. Altun, et al., “Using learning analytics to develop early-warning system for at-risk students,” International Journal of Educational Technology in Higher Education, vol. 16, no. 1, p. 40, 2019.

[2] A. Y. Huang, J. W. Chang, A. C. Yang, H. Ogata, S. T. Li, R. X. Yen, and S. J. Yang, “Personalized intervention based on the early prediction of at-risk students to improve their learning performance,” Educational Technology & Society, vol. 26, no. 4, pp. 69–89, 2023.

[3] K. Alalawi, R. Athauda, R. Chiong, and I. Renner, “Evaluating the student performance prediction and action framework through a learning analytics intervention study,” Education and Information Technologies, vol. 30, no. 3, pp. 2887–2916, 2025.

[4] B. Ustun, A. Spangher, and Y. Liu, “Actionable recourse in linear classification,” in Proceedings of the conference on fairness, accountability, and transparency, pp. 10–19, 2019.

[5] R. K. Mothilal, A. Sharma, and C. Tan, “Explaining machine learning classifiers through diverse counterfactual explanations,” in Proceedings of the 2020 conference on fairness, accountability, and transparency, pp. 607–617, 2020.

[6] E. Tiukhova, P. Vemuri, N. L. Flores, A. S. Islind, M. Oskarsd<sup>´</sup> ottir,´ S. Poelmans, et al., “Explainable learning analytics: Assessing the stability of student success prediction models by means of explainable ai,” Decision Support Systems, vol. 182, p. 114229, 2024.

[7] A. Miles and S. Bechhofer, “Skos simple knowledge organization system reference.” W3C Recommendation, 2009. Online. Available: http://www. w3.org/TR/skos-reference.

[8] H. Knublauch and D. Kontokostas, “Shapes constraint language (shacl).” W3C Recommendation, 2017. Available: https://www.w3.org/TR/shacl/.

[9] N. L. Le, M.-H. Abel, and B. Laforge, “When can we trust early warnings? leakage-excluded early outcome prediction from lms interaction logs,” arXiv preprint arXiv:2605.25794, 2026.

[10] S. Wachter, B. Mittelstadt, and C. Russell, “Counterfactual explanations without opening the black box: Automated decisions and the gdpr,” Harv. JL & Tech., vol. 31, p. 841, 2017.

[11] D. Slack, A. Hilgard, H. Lakkaraju, and S. Singh, “Counterfactual explanations can be manipulated,” Advances in neural information processing systems, vol. 34, pp. 62–75, 2021.

[12] A. Ferrario and M. Loi, “The robustness of counterfactual explanations over time,” IEEE access, vol. 10, pp. 82736–82750, 2022.

[13] S. Sharma, A. Gee, J. Henderson, and J. Ghosh, “Faster-ce: Fast, sparse, transparent, and robust counterfactual explanations,” in IFIP International Conference on Artificial Intelligence Applications and Innovations, pp. 183–196, Springer, 2024.

[14] A.-H. Karimi et al., “Algorithmic recourse: from counterfactual explanations to interventions,” in Proceedings of the ACM conference on fairness, accountability, and transparency, pp. 353–362, 2021.

[15] W. B. Glenn et al., “Verification of forecasts expressed in terms of probability,” Monthly weather review, vol. 78, no. 1, pp. 1–3, 1950.

[16] W. Yang, J. Jiang, E. M. Schnellinger, et al., “Modified brier score for evaluating prediction accuracy for binary outcomes,” Statistical methods in medical research, vol. 31, no. 12, pp. 2287–2296, 2022.

[17] N. L. Le, M.-H. Abel, and B. Laforge, “Vers un cadre ontologique pour la gestion des competences: ´ a des fins de formation, de recrutement, de\` metier, ou de recherches associ ´ ees,” ´ preprint arXiv:2507.05767, 2025.

[18] J. Kuzilek, M. Hlosta, and Z. Zdrahal, “Open university learning analytics dataset,” Scientific data, vol. 4, no. 1, p. 170171, 2017.

[19] T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pp. 785–794, 2016.

[20] X. Huang, A. Khetan, M. Cvitkovic, and Z. Karnin, “Tabtransformer: Tabular data modeling using contextual embeddings,” arXiv preprint arXiv:2012.06678, 2020.

[21] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[22] M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE transactions on Signal Processing, vol. 45, no. 11, pp. 2673–2681, 1997.