# Correct Is Not Governed: Provenance Integrity in Agentic Workflows

Jesus Salas<sup>∗</sup> Independent Researcher jesus.salas@gmail.com

August 2026

## Abstract

Agentic workflows are commonly evaluated by whether they reach the correct outcome. That is insuficient in institutional settings, where a correct action may rely on the wrong authority, an unsupported completion claim, or work made stale by a later change. We define governed execution as work whose decisions, completion, and response to change are supported by inspectable provenance. We present Matrix, a deterministic causal-state layer that records authority and fact dependencies, verifies completion evidence, and selectively invalidates afected work. Across controlled comparisons, governed and direct workflows often reached the same outcomes, but only the governed path consistently preserved governing evidence, refused unsupported closure, and limited recovery to dependent tasks. A role-separated transfer challenge then failed: a deterministically enforced completeness contract severely over-blocked synthetic packets produced outside its authoring context. These results do not establish Matrix as a general accuracy enhancer; they support its primary role as an institutional integrity layer for making agentic work auditable and independently verifiable.

## 1 Introduction

Correct execution is not necessarily governed execution. An agent can choose the expected action while relying on a draft, expired waiver, or adjacent-domain exception. It can announce that a task is complete without producing the evidence required by the task’s definition of done. It can also reach a current business outcome after a policy change while leaving no account of which prior work became stale, which evidence remained valid, or why only some tasks were repeated. Ordinary success metrics can score each workflow as correct.

These failures concern the institutional account of the work rather than the surface answer alone. An enterprise reviewer may need to establish what authority and facts produced a decision, whether execution actually satisfied its declared obligations, and how later changes propagated through prior work. A transcript, retrieved document set, or agent assertion can provide evidence for that account, but none is automatically the governing account itself. This concern is also reflected in institutional guidance. Article 12 of the European Union AI Act requires logging capabilities for high-risk AI systems to support traceability, while the NIST AI Risk Management Framework emphasizes documentation across governance and risk-management activities (European Union, 2024; Tabassi, 2023). These sources do not prescribe the architecture studied here, and Matrix is not claimed to establish regulatory compliance. They motivate treating reconstructable evidence as a system requirement rather than a presentation feature.

We organize this problem around three forms of provenance integrity:

Decision provenance identifies the current authority and accepted facts that materially governed an action, while distinguishing them from merely retrieved or semantically related evidence.

Execution provenance connects each obligation to independently checkable completion evidence and prevents an agent’s own completion claim from serving as suficient proof.

Change provenance records which decisions, tasks, receipts, and prior verdicts depend on a superseded authority or fact so that recovery can invalidate stale work without discarding unafected proof.

We investigate whether these properties can be externalized into a deterministic layer around probabilistic agents. Matrix records typed operations over versioned authority, facts, tasks, completion receipts, de pendencies, invalidations, replacements, and supervisory verdicts. Models continue to interpret documents, propose plans, and perform open-ended work. The framework owns identity, transition validity, mechanically declared preconditions, receipt matching, bounded rework, and replay. This separation is intended to make the lifecycle inspectable without pretending that governance makes semantic judgment infallible.

The evaluation deliberately includes cases where business outcomes tie. In an authored retrieval corpus, direct RAG and governed paths selected every expected action, but difered in whether non-governing records were cited. In matched completion fixtures, the same Phi-4 plan and a generic self-review preceded both conditions; the direct control nevertheless self-closed on unsupported execution claims, while Matrix preserved residual work and required valid receipts. In a versioned-change fixture, both conditions stayed current, but dependency lineage reduced task reexecution from 18 to 3 across three seeds. A separate metadata-decidable invariant forced escalation when a required authority class was absent, even though Ph never proposed escalation itself.

These results motivate the paper’s central claim:

Task correctness and governed execution are diferent properties.

A workflow is governed only when the authority behind its decisions, the evidence behind its completion, and the efects of later change are explicit and independently checkable.

The distinction changes what counts as a useful result. A tied business outcome can still reveal a provenance or assurance diference. Conversely, a deterministic invariant can perfectly implement an authored contract while the contract itself fails to transfer. We therefore report a failed synthetic contract-transfer test alongside the positive mechanism results.

Contributions. This paper makes three contributions:

1. It defines governed execution as a conjunction of decision, execution, and change provenance, and makes each property measurable separately from task correctness.

2. It reports a controlling negative result: a deterministically enforced completeness contract achieved perfect sensitivity but zero specificity on the initial role-separated transfer set, showing that mechanism fidelity does not establish contract validity.

3. It presents an implemented lifecycle joining versioned authority and facts to decision admission, task issuance, receipt-verified closure, supervisory verdicts, and dependency-scoped recovery, with controlled same-outcome demonstrations across those surfaces.

The contribution is this lifecycle-wide governed-work formulation and its matched evaluation, not the novelty of append-only logs, runtime contracts, citation checking, or dependency replay in isolation. Section 8 compares those established and concurrent mechanisms directly.

Scope. The principal studies use small, self-authored synthetic fixtures and repeated model seeds. They establish implemented mechanisms, not enterprise prevalence, human-review cost, semantic completeness, or general accuracy gains. The paper does not claim superiority over citation-verifying RAG, objective truth from ledger acceptance, natural detection of every relevant change, or validated process forecasting. The next claimbearing gate is an independently authored, human-reviewed governed-work task with operational-overhead and false-block measurement; additional self-authored rungs cannot establish that validity.

## 2 Operational Model of Governed Execution

## 2.1 Lifecycle state

At time t, the workflow state contains six first-class sets: versioned authority $A _ { t } ;$ accepted facts $F _ { t } ;$ admitted decisions $Q _ { t } ;$ issued tasks and their definitions of done $T _ { t } ;$ completion receipts $R _ { t } ;$ and supervisory verdicts $V _ { t } .$ Typed dependency edges record which versions were read or produced by each later object.

This state is an institutional claim, not an objective-truth oracle. Authority can be incomplete, facts can be wrong, and models can misinterpret evidence. The purpose of the state is to make those accepted claims explicit, versioned, and revisable.

## 2.2 Decision provenance

Let K be the available institutional corpus, $D _ { K } \subseteq { \mathcal { K } }$ the top-K records retrieved for a decision, and $G \subseteq \kappa$ the records that are efective, authoritative, and applicable. Retrieval recall asks whether every required member of G appears in $D _ { K }$ . It does not ask whether the system avoids promoting records in $D _ { K } \backslash G$ into governing state.

If C is the set cited as materially governing the action, citation soundness requires $C \subseteq G$ . When every member of G is required to justify the decision, completeness also requires $G \subseteq C$ . A decision may therefore select the expected action while remaining provenance-incorrect.

## 2.3 Execution provenance

For a task $\tau \in T _ { t }$ , let $E _ { \tau }$ be the current evidence attached to its completion receipt and DoD<sub>τ</sub> its declared definition of done. The task is complete only when an independent verifier accepts $E _ { \tau }$ against DoD<sub>τ</sub> under the current authority and fact state. The executing agent’s statement that it completed τ is an input claim, not suficient evidence. A failed receipt remains in history; a successful retry creates a later receipt rather than overwriting the failed attempt.

## 2.4 Change provenance

When authority or fact version $x ^ { v }$ is superseded, the afected set contains the current decisions, tasks, receipts, and verdicts whose dependency paths reach that version. A governed recovery invalidates those dependent objects, preserves current objects outside the afected set, issues replacement obligations, and records how new objects supersede stale ones. Broad reruns can also restore currentness, but they do not demonstrate dependency-scoped recovery.

## 2.5 Correctness and governance are measured separately

We use business correctness for whether a workflow reaches its expected business outcome. We use governed execution for the conjunction of sound decision provenance, verified execution provenance, and current change provenance. This is an operational definition, not a theorem.

The empirical separation criterion is a matched pair in which both workflows reach the expected business outcome but difer on at least one provenance property. The studies test whether such pairs occur under controlled conditions. Deployed work requires both correctness and governance; the claim is that one cannot be used as a measurement proxy for the other.

## 2.6 Research questions

RQ1: When direct RAG and governed retrieval select the same expected actions, do they difer in which evidence is cited or promoted as governing?

RQ2: Can deterministic receipt verification distinguish model-claimed completion from evidence-backed completion after a generic self-review?

<table><tr><td>Surface</td><td>Recorded responsibility</td></tr><tr><td>AuthoritySet</td><td>Policies, delegations, status, applicability metadata, versions, and supersession.</td></tr><tr><td>FactSet</td><td>Accepted case facts with source references, versions, explicit unknowns, and conflicts.</td></tr><tr><td>DecisionSet</td><td>Admitted rulings, selected authority and facts, dependencies, revisions, and status.</td></tr><tr><td>TaskSet</td><td>Obligations with owner, dependencies, preconditions, permitted actions, re- quired outputs, and definition of done.</td></tr><tr><td>CompletionReceiptSet</td><td>Immutable attempts, tool or external evidence, timestamps, task references, and verification status.</td></tr><tr><td>SupervisorVerdict</td><td>Completed, rework-required, blocked, rejected, or escalated status with residual obligations.</td></tr><tr><td>Event and dependency state </td><td>Ordered operations, read/produce edges, invalidations, replacements, and replay.</td></tr></table>

Table 1: Logical governed-work surfaces in the current prototype.

RQ3: When both conditions observe an authority change and remain current, can declared dependency lineage reduce recovery churn without stale residue?

RQ4: Can a metadata-decidable completeness invariant force appropriate refusal without degenerating into refusal on every complete packet?

RQ5: Can the selected decision, execution, change, and refusal controls compose in one bounded workflow while preserving their component checks?

## 3 The Matrix Substrate

## 3.1 Design boundary

Matrix separates deterministic institutional mechanics from model-dependent cognition. Models interpret documents, judge applicability, propose plans, perform work, and suggest repairs. The framework owns stable identity, direct references, transition validity, materialization, declared preconditions, receipt matching, dependency traversal, bounded retry counters, and replay.

This boundary does not make semantic judgment deterministic. If a model admits the wrong policy as applicable, the ledger can preserve that wrong judgment faithfully. The deterministic claim is narrower: once authority, facts, obligations, and evidence contracts are declared, the runtime can enforce the associated state-transition rules independently of a model’s willingness to self-enforce them.

## 3.2 State model

The implemented prototype exposes the six logical sets from Section 2 through an append-only event log and deterministic materialization (Table 1).

## 3.3 Governed-work lifecycle

A complete run follows the same ordered lifecycle across the product fixtures:

1. record applicable authority and case facts, including explicit unknowns;

2. accept or reject a model-proposed decision and obligation plan;

3. issue each task only after its declared preconditions are satisfied;

4. preserve every execution attempt and attach evidence to a receipt;

5. verify each receipt against the task’s definition of done;

![](images/7972760603cebd79a74e34f86bf3b0cc9b3343d90580d20ce0867bde3d602931.jpg)  
Fact correction changes the governed state; dependency-directed obligations return only affected work to the roles.

Figure 1: Implemented causal-state architecture. Model and human roles submit typed proposals; the governance kernel validates state transitions and records them in an append-only ledger. Materialized facts, lineage, stale work, and rework obligations feed an outcome supervisor, while the inspector exposes both current state and replayable history. Tasks, receipts, and verdicts use the same event and dependency path.

6. emit a supervisor verdict with any residual obligations;

7. request bounded, targeted rework or escalate when the retry budget is exhausted; and

8. on authority or fact change, invalidate dependent state and repeat only the required portion of the lifecycle.

The event log is append-only. A retry does not erase a failed receipt, and a replacement task does not erase the authority version under which its predecessor ran. Current materialized state and historical reconstruction are therefore separate views over the same operations.

## 3.4 Decision admission

For corpus-backed decisions, a model adjudicator receives the task and the retrieved records with their public identity, authority, status, and content. It returns a resolution and the record ids it judges applicable. The kernel checks that selected ids came from the recorded search and that later fact acceptance follows valid transitions. Accepted records become explicit facts and are bound to downstream work.

The kernel does not compare the model’s selection with private applicability labels. Any retrieved record selected by the adjudicator can be promoted, and an incorrect promotion is scored only after execution. The mandatory-governance errors in Experiment 5 confirm that this semantic step remains fallible.

For plan admission, a contract can require specific authority or fact references without mutating mode output. A failed proposal receives an exact residual obligation and a bounded opportunity to return a replacement. Only a replacement that independently passes the same contract is admitted.

## 3.5 Receipt verification and bounded rework

An execution adapter emits a claim and evidence. The supervisor evaluates that evidence against the task-specific definition of done. Invalid evidence creates a rework\_required verdict and preserves the task as residual. A later attempt creates a new receipt and is independently checked. On every initial or revised receipt set, the current kernel recomputes the efective verdict from the contract checks and rejects a proposed verdict that contradicts those checks or their residual obligations. Model completion language cannot authorize closure. Exhausting the declared retry allowance produces escalated; it does not silently convert the last model claim into completion.

Definitions of done are domain contracts supplied to the framework. Matrix can deterministically implement a contract, but it cannot establish that the contract captures every semantic requirement of the real process.

## 3.6 Version change and selective recovery

When a new authority or fact version is admitted, the runtime records supersession, traverses declared read dependencies, marks afected tasks, receipts, and prior verdicts stale, and creates replacement obligations. State outside the transitive dependent set remains current. A conservative full rerun is a safe alternative when dependency lineage is unavailable; the C6 comparison measures the work avoided by the declared graph, not a correctness advantage over that control.

## 3.7 Structurally decidable refusal

A public completeness contract may declare mechanically required authority classes, domains, multiplicity, or applicability tags. If the visible candidate set lacks a required class, the kernel records the model proposal but blocks a confident ruling and makes escalation efective. Private answer labels are not consulted.

This guarantee is intentionally bounded. It can decide that a declared source class is absent; it cannot decide whether an available source semantically covers an edge case, whether the contract omitted a requirement, or whether an unknown open-world policy exists.

## 4 Experimental Design

## 4.1 Evidence selection and reporting

The experiments preceded the present three-provenance framing. We therefore freeze a claim-evidence map before this rewrite and apply three reporting rules. First, studies are selected because they isolate decision, execution, or change provenance, not because Matrix wins on task accuracy. Second, original scenarios, seeds, controls, repairs, and claim boundaries remain unchanged. Third, counts from diferent studies are not pooled: scenarios, repeated seeds, and deterministic action assertions represent diferent units.

Table 2 maps every evidence source reported in the paper. Studies 1–5 address the five research questions. Study 6 is a controlling transfer failure rather than favorable downstream evidence. L4 is retained as a matched decision-admission walkthrough and Experiment 3 as a supporting state-surface stress test. The table maps paper labels to internal artifact labels so the claims remain traceable to frozen reports.

Artifact scope. The accompanying reproducibility artifact contains the prototype source and tests, scenario and contract inputs, frozen machine-readable run reports, analysis scripts, the claim-evidence map used for this rewrite, and build instructions. Repeated and repaired diagnostics remain labeled separately from claim-bearing artifacts. A stable archive identifier is a release requirement, not evidence supplied by the current manuscript.

## 4.2 Study 1: decision provenance after retrieval

Experiment 5 contains five public scenarios and 13 authored institutional records. Three scenarios have one efective governing policy. The remaining two have no governing record: a harmless internal theme change and an unresolved high-impact biometric proposal. Distractors include drafts, expired waivers, research notes, adjacent-domain policies, and narrow exceptions.

<table><tr><td>Paper label</td><td>Property</td><td>Comparator</td><td>Primary unit</td></tr><tr><td>5)</td><td>Study 1 (Exp. Decision provenance</td><td>Competent direct RAG</td><td>Five scenarios × three seeds</td></tr><tr><td>L4 demo</td><td>Decision admission</td><td>Shared initial Phi proposal; di- Three matched seeds rect control</td><td></td></tr><tr><td>C5)</td><td>Study 2 (C4– Execution provenance</td><td>review</td><td>Same Phi plan plus generic self- Three matched seeds per rung</td></tr><tr><td></td><td>Study 3 (C6) Change provenance</td><td>Conservative full rerun</td><td>Three matched seeds</td></tr><tr><td>6)</td><td>Study 4 (Exp. Structural refusal</td><td>plete contracts</td><td>Raw model proposal and com- 12 incomplete and 120 complete executions</td></tr><tr><td>Study 5 (C7) Composition</td><td></td><td>Same self-review control plus Three matched seeds Phi compaction</td><td></td></tr><tr><td>3)</td><td>Support (Exp. State-surface stress</td><td>Phi-compacted summary</td><td>Three cells × 40 paired seeds</td></tr><tr><td>7)</td><td>Study 6 (Exp. Contract transfer</td><td>Reviewed packet completeness</td><td>Thirty synthetic packets</td></tr></table>

Table 2: Evaluation map. Repeated seeds are stability checks, not independent task samples; action assertions are implementation checks.

Expected actions, resolutions, and governing ids reside in a private control file used only for scoring and scripted qualification. Automated payload scans reject private control keys. Four conditions are compared:

1. fixed scope, which retains the initial proposal without retrieval;

2. direct RAG, which uses the shared query and top-K records to emit an action and governing citations;

3. model-signaled governance, which retrieves and adjudicates only after a typed scope interrupt and then binds accepted facts before revision; and

4. mandatory governance, which runs the same adjudication and binding path for every scenario.

All retrieval conditions use local Phi-4 Q4, three seeds, temperature 0.2, a 4,096-token role budget, BM25-style retrieval, and K = 4. The query planner is physically shared. Direct RAG and the governed reviser receive the same action contract and authority/status metadata. Both are explicitly told to reject drafts, expired waivers, research notes, unrelated jurisdictions, and narrow exceptions. Governed paths add an applicability call, so inference count is not matched. The condition-call totals exclude the shared 15-call query planner: direct RAG uses 15 action calls, model-signaled governance uses 12 applicability and nine action-revision calls, and mandatory governance uses 15 applicability and 11 action-revision calls.

Primary clean success requires the expected action with no unsafe completion, unsupported citation or promotion, false halt, unnecessary rework, or missing required grounding. Action correctness and provenance integrity are also reported separately.

## 4.3 Study 2: claimed versus verified completion

C4 and C5 are cumulative vendor-onboarding fixtures. Both conditions use the same local Phi-4 Q4 checkpoint, temperature zero, and seeds 1–3. Both receive the same initial plan and one active, oracle-free generic self-review. The checklist asks the model to represent each applicable obligation exactly once, remove inapplicable work, cite governing authority and applicability facts, name independently checkable evidence, and mark the plan ready only when all work is represented and verifiable. It contains no case-specific answer.

After self-review, the direct control executes its revised plan and accepts its own completion claims. Matrix independently compiles obligations, checks preconditions, issues tasks, validates receipts against obligation-specific definitions of done, and closes only from a supervisor verdict.

C4 gives both arms the same exact six-obligation plan and six execution claims. One DPA claim lacks its contract-required evidence. Correct governed behavior is rework\_required with the DPA obligation preserved as residual; C4 does not include a successful recovery.

C5 adds one receipt-targeted retry. The first DPA attempt must remain in history, the correction must name the missing evidence, the retry must be independently validated, and closure may follow only from a complete current receipt set. A deterministic regression repeats invalid evidence through the bounded retry allowance and expects escalation. That branch tests implementation behavior, not model failure frequency.

Primary measures are premature closure, evidence-backed completion, residual obligations, bounded rework, supervisor verdict, and exact lifecycle-action assertions.

## 4.4 Study 3: change provenance and recovery scope

C6 starts from an exact six-obligation plan and completed initial execution. Before closure, privacy authority v2 supersedes v1 and changes only the DPA evidence contract. Both conditions receive the amendment. Matrix follows declared dependencies and reissues only DPA. The strong direct control lacks dependency lineage and conservatively reruns all six tasks, ensuring that it is current rather than stale.

Primary measures are current lifecycle exactness, stale residue, afected task reissues, preserved unafected receipts, and action assertions. Because both conditions are designed to be safe, task accuracy is expected to tie; the measured diference is recovery scope.

## 4.5 Study 4: metadata-decidable refusal

Experiment 6 contains 30 public policy sources, 12 standard scenarios across six conflict families, and two missing-authority controls. Each control removes a source class required by a public completeness contract. The contract is visible to the framework but never enters the model prompt; private expected actions and source labels are not consulted by enforcement.

On the incomplete controls, the framework records the raw adjudicator proposal, checks whether the required authority class exists, blocks a confident ruling when it does not, and makes escalation efective. The three-seed qualification contains 12 such executions across two controls and two primary packet surfaces.

After observing sensitivity, complete contracts were added for all 12 standard scenarios to test the degenerate strategy of escalating everything. The frozen five-seed primary pilot supplies 120 complete executions across two packet surfaces. This specificity amendment is post hoc and uses the same authored fixture. We report both repeated executions and unique contract counts.

Ordinary policy-conflict reasoning remains model-dependent and is not a primary result here. The five-seed graph condition achieved 27/60 exact adjudications versus 21/60 for a matched flat packet, with descriptive paired McNemar $p = 0 . 2 3 8$ . We treat that diference as directional and unpowered.

## 4.6 Study 5: composition under bounded history

C7 composes conditional selection, an unknown-gated dependency, bounded receipt rework, selective authority recovery, and supervisor closure. The estimated raw history is 6,160 tokens against a frozen 4,096-token treatment budget. Phi performs both the fair control compaction and all closure reviews. Three probes receive fresh truth, mechanically scoped Matrix state, or the Phi-compacted control. The composition gate requires the governed lifecycle, every component action, and all budget guards to pass; it does not require the compactor to fail.

## 4.7 Supporting lifecycle and state-surface tests

L4 is the fourth rung of a synthetic vendor-onboarding lifecycle ladder and supplies an interpretable decisionadmission walkthrough. It uses a matched governed-versus-direct design across three Phi seeds. Both arms share the initial Phi proposal and diverge only afterward: the direct arm executes without governed admission, while Matrix may issue bounded decision rework before execution. The initial proposal omitted a required preferred-vendor fact reference in all three seeds. Matrix rejected the proposal, issued the exact residual, allowed one model revision, and revalidated the replacement before any task execution. The direct path retained the omission, while both paths ultimately reached the expected business state. We use L4 to illustrate the lifecycle rather than as a separate population estimate.

<table><tr><td>Metric</td><td>Fixed</td><td>Direct RAG</td><td>Signaled gov.</td><td>Mandatory gov.</td></tr><tr><td>Expected action</td><td>3/15</td><td>15/15</td><td>15/15</td><td>15/15</td></tr><tr><td>Primary clean success</td><td>3/15</td><td>9/15</td><td>15/15</td><td>13/15</td></tr><tr><td>Unsafe completion</td><td>12/15</td><td>0/15</td><td>0/15</td><td>0/15</td></tr><tr><td>Unsupported citation</td><td>0/15</td><td>6/15</td><td>0/15</td><td>2/15</td></tr><tr><td>Unsupported promotion</td><td>0/15</td><td>0/15</td><td>0/15</td><td>2/15</td></tr><tr><td>Applicability + action calls</td><td> $0 + 0$ </td><td> $0 + 1 5$ </td><td>12 + 9</td><td> $1 5 + 1 1$ </td></tr></table>

Table 3: Study 1 (Experiment 5) over five authored scenarios and three seeds. Primary clean success combines action, safety, and provenance requirements. Call counts exclude the shared 15-call query planner.

Experiment 3 remains a supporting test of whether provenance state can become outcome-relevant under forced context loss. A 25,254-token two-correction history exceeds a 6,144-token budget. The same frontier reviser receives either a model-generated transcript summary or an approximately 5,579-token mechanical scope query containing answer and distractor lineage. Three preselected compaction cells use 40 paired seeds. A faithful prose rendering of the same scope query tests whether record syntax is the active ingredient. This study is reported as a controlled state-surface stress test, not a general product accuracy claim.

## 4.8 Study 6: transfer challenge

Experiment 7 asks whether an authored structural completeness contract transfers to independently con textualized synthetic packets. Role-separated agent contexts curated 30 packets; two fresh domain-reviewer contexts and a later adjudicator assessed completeness without access to the machine contract. This is procedural information separation on a shared filesystem, not independent human expertise or a security boundary.

The controlling gate requires both incomplete sensitivity and complete specificity. If the gate fails, downstream workflow comparisons remain diagnostic. A later semantic-normalization pass uses a separately expanded, balanced 40-packet diagnostic set (20 reviewer-complete and 20 reviewer-incomplete) assembled after the original 30-packet corpus was exposed. It is therefore labeled a burned-corpus diagnostic rather than a new transfer test.

## 5 Results

The studies answer diferent questions and use diferent units, so we do not report a pooled Matrix success rate. The recurring pattern is instead whether task outcome and provenance integrity agree or diverge.

## 5.1 Study 1: equal actions, diferent governing evidence

Table 3 reports Experiment 5. Every retrieval condition selected the expected action in all 15 repeated runs. Direct RAG was therefore perfect on action accuracy, yet it cited non-governing records in six runs. The model-signaled governed path cited or promoted none. Mandatory governance promoted and cited an irrelevant exception in two runs.

Relative to direct RAG’s 15 condition-specific calls, model-signaled governance used 21 calls (1.40×) and mandatory governance used 26 (1.73×). Including the shared 15-call query planner, the corresponding end-to-end totals were 30, 36 (1.20×), and 41 (1.37×). These are model-call counts, not measurements of tokens, latency, or monetary cost.

The six direct-RAG errors were stable by scenario: in every seed it cited an internal-tool exception and public-brand standard for the harmless theme task, and a non-authoritative research note for the unresolved biometric task. The shared planner interrupted all 12 consequential cases and none of three harmless controls; retrieval returned every required governing record in the nine applicable-policy runs. The observed errors were therefore not failures to make the relevant policy available.

Study 1: expected action ties 15/15 Governing citation integrity diverges

![](images/48d3fedf5e30bab0da893639191579b12f84b9ab5b0b70506e062a88d9ae2d98.jpg)

![](images/9ce794cfbe5eea9f4c33b298b0244983d083e41e1b3a15213febd4591473b8ff.jpg)

Figure 2: Headline matched comparisons. Left: expected-action accuracy ties at 15/15, while runs without unsupported governing citations are 9/15 for direct RAG and 15/15 for model-signaled governance. Right: both Study 3 arms reach current state in 3/3 repeated seeds, while the full-rerun control reissues 18 tasks and dependency-scoped recovery reissues 3. Seeds are stability checks, not independent task samples.
<table><tr><td>Rung</td><td>Direct control</td><td>MATRIX</td><td>Checks</td></tr><tr><td>C4</td><td>Valid completion 0/3 after self-closing 3/3</td><td>Returned rework_required and preserved the DPA residual 3/3</td><td>39/39</td></tr><tr><td>C5</td><td>Valid completion 0/3 after self-closing 3/3</td><td>Retained the failed attempt, directed one retry, independently verified it, and closed 3/3</td><td>45/45</td></tr></table>

Table 4: Study 2: claimed versus verified completion. Checks are deterministic lifecycle-action assertions, not independent samples.

Clean governed grounding was not answer-key-constrained. The adjudicator could select any retrieved id, and the framework would bind that choice. The two mandatory-condition promotions show that semantic adjudication remained fallible. The result supports staged adjudication plus governed binding versus the measured direct-RAG pipeline, not governance by construction.

Figure 2 makes the two principal same-outcome comparisons explicit. Study 1 ties on expected action while citation integrity diverges; Study 3 ties on current lifecycle while reexecution scope diverges.

## 5.2 Decision admission: the same outcome after explicit grounding repair

L4 illustrates how the distinction enters a workflow. Phi omitted the required preferred-vendor fact reference in the initial proposal on all three seeds. Both initial proposals therefore failed the same contract. Matrix issued the exact residual obligation, accepted one model-authored replacement only after it passed revalidation, and admitted the corrected proposal 3/3. No task ran before admission. The direct path retained the omission 3/3.

Both paths later reached the expected business state 3/3. The contribution is not improved business accuracy: only the governed path repaired and recorded the grounding before execution. A tested persistentomission branch escalates after the bounded rework allowance with zero task execution; the live Phi runs exercised correction rather than escalation.

## 5.3 Study 2: self-closure is not verified closure

Table 4 reports C4 and C5. The shared Phi plan was exact after generic self-review in every run. The direct condition nevertheless accepted an unsupported DPA completion claim and self-closed 3/3 in both rungs.

C4 deliberately stops at the residual obligation; evidence-backed completion is zero in both conditions because no valid replacement is provided. C5 adds that replacement. Matrix retains the invalid first receipt, issues the specific evidence correction, validates the countersigned retry, and closes from the new current receipt set. The direct condition reaches the same claimed business outcome but never obtains valid DPA evidence.

<table><tr><td>Metric</td><td>MATRIX</td><td>Full-rerun control</td></tr><tr><td>Current lifecycle exact</td><td>3/3</td><td>3/3</td></tr><tr><td>Task reissues, total</td><td>3</td><td>18</td></tr><tr><td>Unaffected receipts preserved</td><td>5 per seed</td><td>0 per seed</td></tr><tr><td>Governed stale residue</td><td>0</td><td></td></tr><tr><td>Independent supervisor closure</td><td>3/3</td><td>0/3</td></tr></table>

Table 5: Study 3: change recovery. The direct control is intentionally conservative and current; the measured diference is dependency-scoped work.
<table><tr><td>Packet set</td><td>Unique contracts Executions</td><td></td><td>Result</td></tr><tr><td>Structurally incomplete</td><td>2</td><td>12</td><td>Phi escalation 0/12; effective Ma- trix escalation 12/12</td></tr><tr><td>Structurally complete</td><td>12</td><td>120</td><td>False escalations 0/120; specificity 1.00</td></tr><tr><td>Resolvable complete subset</td><td>10</td><td>100</td><td>False escalations 0/100</td></tr></table>

Table 6: Study 4: metadata-completeness invariant. Complete-contract specificity was added post hoc on the same authored fixture.

The generic self-review control matters: the observed diference is not against a one-shot model that was never asked to check its plan. The self-review produced a complete plan, but it did not create an independent execution verifier. A deterministic persistent-failure regression consumes the same retry allowance, retains both invalid receipts, and escalates with the DPA obligation unresolved.

## 5.4 Study 3: currentness ties while recovery scope diverges

C6 gives both conditions the same privacy amendment and makes both recovery strategies safe. Table 5 shows that both arms reached the current lifecycle 3/3. Matrix traversed declared dependencies and reissued DPA once per seed. The direct control conservatively repeated all six tasks per seed.

All 48 governed lifecycle assertions passed. The result demonstrates a sixfold diference in task reexecution on one declared amendment, not a general cost ratio. Change detection itself was supplied by the fixture and is not measured.

## 5.5 Study 4: contract fidelity with a bounded refusal guarantee

Phi proposed escalation on none of the 12 structurally incomplete Experiment 6 executions. The public completeness invariant blocked each confident ruling and made escalation efective in all 12. The same enforcement path later stayed silent on every complete pilot execution.

This result rules out the degenerate implementation that escalates every packet, but only within the declared contract family. It demonstrates that the code faithfully implements metadata-decidable refusal; it does not show that the contract captures semantic completeness. The ordinary policy-adjudication comparison was weak and remains outside the claim: graph 27/60 versus flat 21/60, descriptive p = 0.238.

## 5.6 Study 5: assurance survives while context accuracy ties

C7’s raw history exceeded the treatment budget on all three seeds (6,160 > 4,096 estimated tokens). Matrix composed dependency gating, receipt rework, selective invalidation, and current supervisor closure 3/3; all 60 lifecycle assertions passed. The direct strict lifecycle remained 0/3 because its registry completion claim was never independently verified.

<table><tr><td>Compaction cell</td><td>Ledger exact</td><td>Transcript exact</td><td>Ledger F1</td><td>Transcript F1</td></tr><tr><td>Frontier, 800</td><td>40/40</td><td>1/40</td><td>1.000</td><td>0.025</td></tr><tr><td>Mini, 800</td><td>40/40</td><td>7/40</td><td>1.000</td><td>0.407</td></tr><tr><td>Mini, 400</td><td>40/40</td><td>0/40</td><td>1.000</td><td>0.217</td></tr></table>

Table 7: Supporting state-surface stress test: second-correction recovery. The final reviser is unchanged across cells.

Crucially, fresh truth, Matrix scoped state, and the fair Phi-compacted control all selected current closure 3/3. C7 therefore supports composition of the assurance lifecycle under a bounded state surface, not a context-accuracy advantage.

## 5.7 Supporting state-surface stress test

The earlier short-context frontier diagnostic reached 40/40 exact recovery in both governed and transcript conditions. In Experiment 3, the same frontier reviser was then held fixed while a two-correction history was forced through compaction. Table 7 reports the result.

The compacted failures retained no complete answer-lineage items. A mechanical scope query exposed six answer-lineage and ten non-answer-lineage references, so the ledger was not filtered to the answer set. Faithful prose rendered from the same query also achieved 40/40, identifying complete targeted provenance retrieval rather than JSON syntax as the primary mechanism in this fixture. The non-monotonic F1 values are set-overlap behavior rather than an exact-score contradiction: the frontier summaries usually omitted the target lineage almost entirely, while the mini 400-token summaries retained scattered target items but never the complete required set.

This is supporting evidence that a governed state surface can become outcome-relevant when provenance is dropped. It is not promoted to a robust product accuracy claim: L6’s later diference reduced to a top-K recall cutof, and C7’s compactor preserved current truth.

## 5.8 Study 6: the completeness contract failed to transfer

Experiment 7’s reviewers agreed on structural completeness for 26/30 synthetic packets (Cohen’s κ = 0.659). The frozen contract detected all 23 adjudicated-incomplete packets but blocked all seven reviewer-complete packets: sensitivity 1.00, specificity 0.00, and accuracy 0.767. The validity gate failed, so no downstream Experiment 7 workflow comparison is claim-bearing.

A final semantic-normalization diagnostic used a separately expanded, balanced 40-packet set assembled after the original 30-packet corpus was exposed. It blocked all 20 reviewer-incomplete packets but admitted only 3/20 reviewer-complete packets, for specificity 0.15 and accuracy 0.575. Because this was a burned-corpus repair diagnostic rather than a fresh transfer test, it does not replace the controlling zero-specificity result.

The negative result separates mechanism correctness from contract validity. The authored ladders show that Matrix can enforce a declared contract; Experiment 7 shows that the contract’s notion of completeness may not transfer to independently produced synthetic packets.

## 6 Discussion

## 6.1 The tied outcomes are the central result

The most title-aligned comparisons are not accuracy wins. Direct RAG and governed retrieval selected the same actions in Experiment 5. Both C6 recovery strategies reached current state. L4 reached the same final business state in both conditions. If evaluation stops at the action or final state, these cases are ties.

The workflows difer on institutionally material questions: which evidence was allowed to govern, whether completion was independently proven, and which prior work a later change invalidated. The results therefore support measuring task success and governance as separate axes. Governance is not a substitute for a correct outcome; it is additional structure needed to justify, close, and revise that outcome.

## 6.2 One provenance concept across three lifecycle surfaces

Decision, execution, and change provenance are not unrelated features. Each binds a consequential state transition to the prior objects that justify it:

• a decision binds to accepted authority and facts;

• a completion verdict binds to a task, definition of done, and current receipt evidence; and

• an invalidation or preservation decision binds to the versioned object and dependency path afected by a change.

This common representation enables one append-only lifecycle rather than separate citation, workflow, and audit subsystems. It also supports residual obligations: a failed decision, receipt, or recovery step remains explicit instead of disappearing behind a later green status.

The synthesis, rather than any individual primitive, is the architectural claim. Event logs provide replay, runtime monitors enforce declared rules, citation systems bind claims to evidence, tool contracts gate efects, and dependency graphs support selective recomputation. Matrix uses those ideas to maintain one institutional lifecycle from versioned authority and facts through admitted decisions, obligations, receipts, closure, and later invalidation. The transfer failure is equally part of the contribution: a faithful mechanism does not validate the institutional contract it executes.

## 6.3 Why not just RAG?

Experiment 5 answers a narrow version of this question. Retrieval made relevant text available, and its recall was complete on the applicable-policy cases. It did not by itself establish which retrieved records were institutionally efective. Direct RAG could therefore choose the expected action while attaching plausible but non-governing evidence.

The baseline is not a cite-everything strawman. It shares the model, physical query, retriever, top K, records, authority metadata, action space, and context budget, and it is explicitly instructed to reject the same distractor classes. It is still a direct action-and-citation pipeline. A RAG system with an independent applicability verifier may close the measured gap. If it also records accepted authority, binds that state to downstream tasks, checks receipts, and propagates later changes, it is implementing much of the governance pattern studied here.

The governed path uses more inference and a separate adjudication call. The current evidence cannot distinguish how much of Experiment 5 comes from decomposition, prompt specialization, or persistent binding. It establishes the system comparison, not an isolated ledger efect.

## 6.4 Why not ask the model to check its work?

C4–C6 use a stronger control than one-shot generation. The same Phi plan and one explicit generic self-review are shared before the conditions diverge. The self-review successfully produces exact obligation plans, yet the direct condition still accepts its own unsupported completion claim. Planning review and completion verification are diferent functions. This is consistent with prior findings that intrinsic self-correction and self-verification can fail without a sound external signal (Huang et al., 2024; Stechly et al., 2025); the present result instantiates that distinction at an evidence-backed workflow closure boundary.

Another model call could serve as a verifier, but independence then becomes a system property: the verifier requires a frozen evidence contract, access to external or tool-produced proof, an immutable record of failed attempts, and authority to prevent closure. Those properties do not arise from reflective language alone. The verifier may itself be a model where semantic judgment is necessary; deterministic infrastructure can still own the surrounding identity, retry, and transition rules.

## 6.5 Hard mechanics around soft judgment

Experiment 6 illustrates the useful boundary. Phi proposed escalation in 0/12 executions where a mechanically required authority class was missing. Once that absence was represented as a public completeness contract, the framework could guarantee refusal for the declared condition and remain silent on authored complete contracts. No larger model is required to enforce a missing-key invariant.

The framework cannot make the contract semantically complete. Experiment 7 shows why this distinction matters: perfect authored-fixture contract fidelity coexisted with severe over-blocking after synthetic packet authorship and review were separated from the contract. Determinism makes a rule reproducible; it does not make the rule valid.

This suggests an engineering split. Metadata-decidable invariants can be hard, model-independent gates. Applicability, completeness, and conflict adjudication that require open-world interpretation remain soft judgments and may require models or humans. The ordinary Experiment 6 reasoning comparison was weak, which reinforces rather than undermines that boundary.

## 6.6 Capability changes the reasoner, not the institutional requirement

A stronger model may avoid some citation errors, remember a correction through compaction, or catch an unsupported claim during self-review. The frequency of intervention can therefore fall as capability improves. The institutional requirements do not disappear: a reviewer still needs to know what authority governed, what proof closed the task, and what changed after an amendment.

The supporting context experiments sharpen this point without supporting a universal accuracy claim. A frontier model reconstructed short reliable history perfectly, while mechanical provenance became load-bearing under one forced compaction treatment. Later product fixtures did not reproduce a robust context accuracy gap. The durable claim is therefore the inspectable state and lifecycle, not permanent superiority over model memory.

## 6.7 Governance is not objective truth

Matrix records accepted institutional state under explicit authority and evidence policies. It does not equate acceptance, consensus, or ledger presence with objective truth. The mandatory-governance errors demonstrate that a bad semantic choice can be promoted. The Experiment 7 transfer failure demonstrates that an authored completeness contract can be wrong for new packets.

The value of the trace is that these judgments and dependencies are visible, versioned, and revisable. When an executable or independently checkable oracle exists, the supervisor can block false closure. Where no complete oracle exists, governance improves accountability and controlled escalation but cannot guarantee correctness.

## 6.8 Product implication

The product implied by these results is an institutional-integrity layer, not a general accuracy enhancer. Its stable responsibilities are to distinguish candidate evidence from admitted authority, compile accepted state into obligations, require evidence for closure, preserve failures and residual work, and target recovery after change. Models, prompts, retrievers, and execution adapters can be replaced around that layer.

The next validation milestone is correspondingly concrete: an independently authored enterprise task whose authority, facts, definition of done, execution evidence, and review judgments are supplied by people who did not design the contract. More self-authored rungs would expand coverage but not external validity.

## 7 Evidence Boundaries

The results should be read as mechanism evidence, not as a deployment-rate estimate. The experiments show that external governance can preserve provenance, direct recovery, and make evidence grounding inspectable under controlled conditions. They do not show that every enterprise workflow can be specified correctly, that all dependency declarations will be complete, or that governed agents will outperform ungoverned agents on every correctness metric.

Several boundaries are especially important. The main fixtures are authored and synthetic, with seeds measuring output stability rather than independent task designs. Some lifecycle counts are deterministic contract assertions rather than population samples. Directed rework tests the governance loop after a failure is detected; it does not prove that the model would autonomously discover the missing fact or correction. The retrieval study compares governed applicability adjudication against direct RAG, but does not establish superiority over all possible citation-verifying RAG systems.

<table><tr><td>Research line</td><td>Primary object</td><td>Control boundary</td><td>Distinction studied here</td></tr><tr><td>Citation grounding</td><td>Claim or proposed ac- tion</td><td>Generation or pre-tool check</td><td>Institutional applicability persists into tasks, closure, and later change.</td></tr><tr><td>Runtime enforcement</td><td>Path, tool call, state, or durable effect</td><td>Before action or at com- mit</td><td>One state model links authority and facts to decisions, obligations, re- ceipts, verdicts, and rework.</td></tr><tr><td>Durable workflows</td><td>Checkpointed workflow state or artifact DAG</td><td>Resume, replay, or up- date</td><td>Authority applicability and evidence- sufficient closure govern what may remain current.</td></tr><tr><td>MATRIX</td><td>Institutional work ob- Admission jects</td><td>through post-change recovery</td><td>Matched outcome/provenance eval- uation plus an explicit contract- transfer failure.</td></tr></table>

Table 8: Nearest-neighbor scope comparison. Rows describe each line’s stated center of gravity, not an assertion that other implementations cannot be extended to the Matrix lifecycle

The paper also separates assurance from semantic capability. Matrix can make state lineage, residual obligations, and closure criteria explicit, but it does not make a weak model semantically equivalent to a stronger one. Later Phi diagnostics support this boundary: verifier feedback can rescue localized repair cases, while difuse multi-constraint repairs and policy-modality judgments remain model-capability limits.

Appendix A gives the detailed evidence-boundary record, including post-hoc framing controls, transfer-gate failures, post-freeze verifier hardening, model/runtime coverage, and claims not permitted by the current evidence.

## 8 Related Work and Novelty Boundary

The individual mechanisms in Matrix have substantial prior art. The paper’s claimed contribution is their integration into one governed-work lifecycle and an evaluation that separates expected outcome from decision, execution, and change provenance. Table 8 states that boundary against the closest research lines.

Citation, attribution, and decision provenance. RAG couples generation to retrieved non-parametric memory (Lewis et al., 2020). ALCE evaluates citation correctness and completeness in retrieval-grounded generation (Gao et al., 2023), while GAVEL binds atomic subclaims to evidence units and performs deterministic citation validation (Xu et al., 2026). Controlled authorization audits show that source authority can change agent action even when proposition content is held fixed (Liao, 2026); ProvenanceGuard checks whether a proposed tool call is supported by traceable evidence before execution (She et al., 2026). These are direct neighbors, not merely adjacent work. Study 1 is narrower than state-of-the-art citation-verifying pipelines: its direct-RAG arm has no independent applicability verifier. Its role is to demonstrate, on the authored corpus, that expected action and governing citation can diverge. Matrix then persists the admitted authority choice as input to downstream work.

Runtime governance, policy as code, and verified efects. Policy-as-code engines such as Open Policy Agent separate policy evaluation from application enforcement over structured input (Open Policy Agent, 2026). AgentSpec provides a language for runtime constraints with triggers, predicates, and enforcement actions (Wang et al., 2025); path-based governance treats the partial execution path and proposed next action as the policy object (Kaptein et al., 2026). ToolGate maintains typed symbolic state and uses preconditions and postconditions to gate tool invocation and state commit (Liu et al., 2026). Commit-time authorization is especially close to the paper’s title claim: visible endpoint success may remain high after the witness authorizing a durable efect has become invalid (Santos-Grueiro, 2026). Matrix does not claim to originate deterministic runtime enforcement, contract-gated tools, or the distinction between visible and authorized success. Its diferent unit is an institutional workflow whose decision, obligation, receipt, supervisor closure, and later rework share versioned authority and fact dependencies.

Durable orchestration and change propagation. Current workflow systems already provide typed graphs, checkpointing, durable resumption, replay, and forking. Microsoft Agent Framework supersedes the older AutoGen-centered orchestration reference and supports graph workflows, checkpointing, human interaction, and durable execution (Microsoft, 2026). LangGraph similarly persists per-step checkpoints for replay, fault recovery, and time travel (LangChain, 2026). Execution Lineage represents AI-native work as an artifact DAG with explicit dependencies and identity-based replay, and demonstrates selective propagation with unafected-artifact preservation (Rosen and Rosen, 2026). This directly overlaps Study 3. The appendonly log, dependency graph, and selective replay are therefore substrate rather than a standalone novelty claim. The distinction tested here is their use with versioned institutional authority, evidence contracts, supervisory closure, and residual obligations in the same lifecycle.

Truth maintenance, data provenance, and process evidence. Truth maintenance systems record justifications and revise dependent beliefs when assumptions change (Doyle, 1979; de Kleer, 1986); formal belief revision studies rational knowledge-base change (Alchourrón et al., 1985). Database provenance semirings explain how query outputs depend on input records (Green et al., 2007). W3C PROV represents entities, activities, agents, derivations, and responsibility (Moreau and Missier, 2013), while process mining uses event logs for discovery and conformance checking (IEEE Task Force on Process Mining, 2012). Matrix applies related dependency ideas to a live agent-control surface. It does not propose a new general provenance algebra or interoperability standard.

Auditability and specification validity. Runtime verification can faithfully enforce observed executions against a specification (Leucker and Schallhart, 2009). Recent work frames agent auditability across lifecycle coverage, policy checkability, recoverability, responsibility, and evidence integrity (Nian et al., 2026). Matrix operationalizes a subset of that agenda in one prototype. Study 6 adds a separate warning: mechanism fidelity does not establish that an authored institutional contract is semantically valid on newly produced packets. That negative transfer result, and its role as a gate on downstream claims, is part of the paper’s novelty boundary rather than an implementation footnote.

## 9 Future Validation and System Scope

## 9.1 Independently authored governed-work study

The next claim-bearing study should begin with one real or realistic enterprise task whose policy owners, packet authors, execution adapters, definitions of done, and audit reviewers did not design Matrix or its contract. The study should freeze:

• the authority and fact inventory before workflow execution;

• task preconditions and definitions of done before evidence is observed;

• a direct self-review baseline, a citation-verifying retrieval baseline, and the governed condition under matched models and evidence;

• independent judgments of applicability, completion, and afected-work scope; and

• stop/go thresholds for false blocks, missed residual work, stale residue, and human review burden.

The target outcome is not another decision benchmark. It is one task that the system can ground, issue, supervise, and close from independently verifiable proof, with unresolved work preserved when closure is not justified.

## 9.2 Contract validity and invariant families

Experiment 7 shows that contract transfer is a first-class research problem. Future work should separate policy authorship, schema translation, packet production, and completeness review across independent humans or organizations. Calibration and test packets must remain disjoint, and contract changes after label access must trigger a new held-out evaluation.

The missing-authority rule is one member of a possible invariant family: expired authority, superseded policy, delegation bounds, jurisdiction, and required approval multiplicity may also be metadata-decidable. Each candidate invariant requires both incomplete and complete held-out cases. A broad family is valuable only if it avoids turning uncertainty into systematic over-blocking.

## 9.3 Stronger matched baselines and operational measures

Experiment 5 should be repeated against an equal-call citation-verifying RAG pipeline. Completion studies should compare deterministic verification with a separate model verifier that receives the same evidence contract and tool results. Measurements should include latency, tokens, tool calls, human intervention time, false escalation, rework attempts, and the cost of broad versus selective recovery.

## 9.4 Interoperable governed state

A generalized runtime requires stable schemas for authority, facts, tasks, receipts, verdicts, invalidations, and replacement lineage; identity and role assignment; allowed tools; and pluggable outcome verifiers. Mapping these records to W3C PROV and process-event standards would make traces portable to existing audit and process-mining tools. The current prototype is a local implementation, not a completed interoperability standard.

## 9.5 Causal process intelligence

Branch-native replay could eventually compare actual execution with a branch in which a governing variable changes. Aggregated traces could expose process variants, bottlenecks, and intervention candidates. Futurevariable models might support projective scenarios across cost, time, capacity, and risk. These are research hypotheses, not capabilities established by this paper. A modeled counterfactual is not proof of causality, and a projected branch is not a guaranteed future.

## 10 Conclusion

Correct actions are necessary, but they do not establish governed execution. An institutional workflow must also identify what authority and facts governed the decision, what independent evidence proves completion, and how later changes afected prior work.

The experiments make those diferences observable. Direct RAG and governed retrieval selected the same expected actions, yet difered in whether non-governing evidence was cited. A generic model self-review produced complete plans but did not prevent self-certified closure on invalid evidence; receipt verification and bounded rework did. A conservative full rerun and dependency-directed recovery both restored current state after an amendment, but only the latter preserved unafected proof and limited reexecution to dependent work. A deterministic invariant forced refusal for a declared missing authority class where the model never self-refused.

The negative boundary is equally important. Perfect implementation of an authored completeness contract did not make that contract valid on independently contextualized synthetic packets. Matrix can make declared institutional rules inspectable and deterministic; it cannot manufacture semantic truth or external validity.

The resulting claim is deliberately bounded: a deterministic causal-state layer can preserve decision, execution, and change provenance around probabilistic agents, allowing correct outcomes to be distinguished from outcomes that are also grounded, verified, current, and replayable.

## References

Carlos E. Alchourrón, Peter Gärdenfors, and David Makinson. On the logic of theory change: Partial meet contraction and revision functions. The Journal of Symbolic Logic, 50(2):510–530, 1985. doi: 10.2307/2274239.

Johan de Kleer. An assumption-based TMS. Artificial Intelligence, 28(2):127–162, 1986. doi: 10.1016/ 0004-3702(86)90080-9.

Jon Doyle. A truth maintenance system. Artificial Intelligence, 12(3):231–272, 1979. doi: 10.1016/ 0004-3702(79)90008-0.

European Union. Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence, 2024. URL https://eur-lex.europa.eu/eli/reg/2024/1689/oj. Article 12.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate text with citations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6465–6488. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main.398. URL https://aclanthology.org/2023.emnlp-main.398/.

Todd J. Green, Grigoris Karvounarakis, and Val Tannen. Provenance semirings. In Proceedings of the Twenty-Sixth ACM SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems, pages 31–40. ACM, 2007. doi: 10.1145/1265530.1265535.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=IkmD3fKBPQ.

IEEE Task Force on Process Mining. Process mining manifesto. In Business Process Management Workshops, volume 99 of Lecture Notes in Business Information Processing, pages 169–194. Springer, 2012. doi: 10.1007/978-3-642-28108-2\_19.

Maurits Kaptein, Vassilis-Javed Khan, and Andriy Podstavnychy. Runtime governance for AI agents: Policies on paths, 2026. URL https://arxiv.org/abs/2603.16586.

LangChain. LangGraph persistence, 2026. URL https://docs.langchain.com/oss/python/langgraph/ persistence. Accessed 2026-08-11.

Martin Leucker and Christian Schallhart. A brief account of runtime verification. The Journal of Logic and Algebraic Programming, 78(5):293–303, 2009. doi: 10.1016/j.jlap.2008.08.004.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrievalaugmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 6b493230205f780e1bc26945df7481e5-Abstract.html.

Junchi Liao. Auditing provenance sensitivity in LLM agent action selection, 2026. URL https://arxiv. org/abs/2607.20827.

Yanming Liu, Xinyue Peng, Jiannan Cao, Xinyi Wang, Songhang Deng, Jintao Chen, Jianwei Yin, and Xuhong Zhang. ToolGate: Contract-grounded and verified tool execution for LLMs. In Findings of the Association for Computational Linguistics: ACL 2026, pages 9653–9684. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.findings-acl.470. URL https://aclanthology.org/2026.findings-acl.470/.

Microsoft. Microsoft agent framework workflows, 2026. URL https://learn.microsoft.com/en-us/ agent-framework/workflows/. Accessed 2026-08-11.

Luc Moreau and Paolo Missier. PROV-DM: The PROV data model. W3c recommendation, World Wide Web Consortium, 2013. URL https://www.w3.org/TR/prov-dm/.

Yi Nian, Aojie Yuan, Haiyue Zhang, Jiate Li, and Yue Zhao. Auditable agents, 2026. URL https: //arxiv.org/abs/2604.05485.

Open Policy Agent. Open policy agent documentation, 2026. URL https://www.openpolicyagent.org/docs. Accessed 2026-08-11.

Josh Rosen and Seth Rosen. From agent loops to deterministic graphs: Execution lineage for reproducible AI-native work, 2026. URL https://arxiv.org/abs/2605.06365.

Igor Santos-Grueiro. Temporary authority, permanent efects: Commit-time authorization for LLM agents, 2026. URL https://arxiv.org/abs/2607.10487.

Yining She, Yiliang Liang, and Eunsuk Kang. Safeguarding LLM agents from misalignment through provenance analysis, 2026. URL https://arxiv.org/abs/2607.01236.

Kaya Stechly, Karthik Valmeekam, and Subbarao Kambhampati. On the self-verification limitations of large language models on reasoning and planning tasks. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ f3c5e56274140e0420baa3916c529210-Paper-Conference.pdf.

Elham Tabassi. Artificial intelligence risk management framework (AI RMF 1.0). Technical Report NIST AI 100-1, National Institute of Standards and Technology, 2023. URL https://doi.org/10.6028/NIST.AI. 100-1.

Haoyu Wang, Christopher M. Poskitt, and Jun Sun. AgentSpec: Customizable runtime enforcement for safe and reliable LLM agents, 2025. URL https://arxiv.org/abs/2503.18666.

Ruoyu Xu, Gaoxiang Li, and Victor S. Sheng. GAVEL: Evidence-contract debate with mechanized scrutiny for provenance-grounded fact-checking. In Findings of the Association for Computational Linguistics: ACL 2026, pages 35907–35920. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026. findings-acl.1789. URL https://aclanthology.org/2026.findings-acl.1789/.

## A Detailed Evidence Boundaries

Post-hoc paper framing. The experiments were not preregistered as one three-provenance study. The paper recomposes existing frozen artifacts after their results were known. To reduce selection drift, the claim-evidence map was frozen before the rewrite, original conditions and negative results are retained, and no cross-study aggregate is reported. This discipline improves transparency but does not remove the risk of post-hoc narrative selection.

Self-authored fixtures and small independent sample. Experiment 5 has five authored scenarios; C4–C7 and L4 each have one authored case repeated over three seeds; Experiment 6 has 12 authored standard contracts and two incomplete control types. Seeds measure output stability, not new task designs. Counts such as 39/39 or 60/60 are deterministic lifecycle assertions, not independent observations and not confidence about deployment rates.

Contract fidelity is partly by construction. Receipt gates, dependency traversal, and metadatacompleteness checks are designed to implement declared contracts. Perfect action assertions show that the code follows those specifications. They do not independently establish that the definitions of done, dependencies, or completeness requirements are semantically correct. This is the intended mechanism claim, but it is weaker than enterprise validity.

The transfer gate failed. Experiment 7 directly challenges contract validity. Its frozen completeness contract had specificity 0.00 on seven reviewer-complete synthetic packets, and the final burned-corpus normalization diagnostic reached only 0.15. The reviewers were fresh role contexts, not independent humans, but the result is still suficient to reject a general semantic-completeness claim from the authored ladders. Downstream Experiment 7 comparisons are diagnostic only.

Direct RAG uses less inference. Experiment 5 matches model, query, retriever, top K, records, metadata, action contract, and context budget, but governed paths add applicability adjudication. Direct RAG has no independent citation verifier. The study cannot attribute the full diference to persistent state rather than decomposition or additional inference, and it does not establish superiority over citation-verifying RAG.

The self-review control is strong but not exhaustive. C4–C7 give the direct condition one generic review pass and the same revised plan. Other prompting, verifier models, external workflow engines, or tool-enforced completion contracts could reproduce parts of the behavior. Such systems would be alternative implementations of the assurance functions, not evidence that those functions are unnecessary.

Directed rework tests the loop. L4 and C5 issue exact residual obligations. They test detection, directed correction, independent revalidation, and bounded escalation; they do not test whether the model autonomously discovers the missing fact or evidence. The model authors the replacement, but the framework specifies what the failed contract requires.

Post-freeze verifier hardening. A later non-claim-bearing diagnostic on a separate experimental closure surface false-closed two of three deceptive executions because a revised model response bypassed final deterministic admission. The frozen C4–C7 fixtures use a separate completion path that derives closure directly from the current receipt set; they do not route a revised model verdict through the defective admission branch, and current regression tests exercise that receipt-derived path. The defect therefore does not alter their frozen counts, but it exposes an implementation risk beyond those fixtures. The current kernel recomputes revised receipt sets from check state and rejects contradictory verdicts; a bounded follow-on diagnostic produced no verifier false close on two unsafe designs and no false block on two complete endpoint designs. These are regression checks, not population safety estimates or a live-model replication of C4–C7.

Synthetic execution and no human-load measurement. The completion ladders use controlled execution adapters and declared evidence artifacts rather than production enterprise systems. Latency, human review time, escalation burden, and monetary cost are not measured. Experiment 7’s review-time values are model-role workload proxies and are not used here.

Change detection is supplied. C6 measures invalidation and recovery after an observed amendment. It does not measure whether Matrix, a retriever, or a human detects the change in the first place, nor whether real dependency declarations are complete. Missing edges can leave stale residue; overly broad edges can erase the measured churn benefit.

Safety specificity is post hoc and authored. Experiment 6 added complete contracts after observing the 12/12 sensitivity result. The 120 complete executions repeat 12 authored contracts across seeds and two surfaces. Zero false fires rule out escalate-all on that fixture, not a population false-positive rate. The invariant covers metadata absence, not semantic or open-world incompleteness.

Model and runtime coverage. The lifecycle ladders primarily use one quantized local Phi-4 checkpoint. The retrieval and context studies use additional local and frontier models under diferent runtimes and sampling controls. The paper does not estimate model- family interactions or a scaling curve.

Context-accuracy non-replication. Experiment 3 demonstrates a controlled compaction mechanism, but later product diagnostics did not establish a robust accuracy advantage. L6 reduced to a top-K recall cutof, and C7’s Phi compactor preserved current truth 3/3. The paper therefore uses Experiment 3 only as supporting state-surface evidence.

No truth, enterprise, or forecasting guarantee. Ledger acceptance does not establish objective truth, and the current studies do not estimate real enterprise prevalence or efectiveness. Counterfactual forks, process optimization, and forecasting remain unvalidated research directions.