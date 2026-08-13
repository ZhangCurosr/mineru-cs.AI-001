# Beyond Memory: A Transactional Continuity Kernel for Long-Lived AI Agents

Jun He OpenKedge.io

Deying Yu OpenKedge.io

## Abstract

Persistent AI agents accumulate versioned state across long horizons, but storage retention alone does not identify authoritative state. Without an explicit control plane, unmediated updates by models, tools, and background workers risk stale overwrites, un-audited exposures, and self-authorizing privi lege escalation. We argue that agent state governance is an infrastructural activation problem, defining continuity as an unbroken, authorized lineage of accepted branch heads. We present the Continuity Kernel (CK), an activation contract that decouples off-commit candidate evaluation from atomic state activation. Untrusted components propose typed changes against an exact predecessor head or typed absence. A short activation transaction revalidates ownership, pre-state authority, freshness, and effect uniqueness, recording one stable disposition (Commit, Reject, Quarantine, or Defer). Only Commit atomically advances the branch head and installs the complete accepted unit (state, authority, lineage, effects, outcome, and receipt). A bounded executable model verifies the protocol across 2,808,230 reachable states and 5,526,474 state-changing transitions with zero invariant violations.

## 1 Introduction

Long-lived AI agents increasingly retain memories, profiles, plans, tool state, and policies beyond a single context window [1,2]. However, high retrieval quality does not determine which state version is authoritative when models, tools, compactors, and background recovery workers submit concurrent or conflicting updates. Without a strict mutation boundary, a storage layer may preserve every version yet accept a stale retry, expose uncommitted state without audit trails, or allow a proposal to authorize itself by injecting the required privileges into its own proposed state.

We define continuity as an unbroken, authorized lineage of accepted heads on one branch. This is infrastructural continuity—a guarantee about state activation and lineal provenance—not a philosophical claim about agent consciousness or behavioral identity. Continuity fundamentally differs from retention: rejected and quarantined objects may remain stored for diagnostic or policy reasons, but they remain unreachable from the authoritative branch head. Cryptographic commitments can enforce this reachability boundary; they cannot make an inferred memory factually true or a policy normatively legitimate.

The core thesis of this paper is that agent state governance is an infrastructural activation contract problem. To operationalize this principle, we define the Continuity Kernel (CK) as an activation contract that separates off-commit candidate proposal from atomic activation. Models, tools, and operators act as untrusted proposers. They may perform probabilistic reasoning, remote network fetches, and candidate state construction, but only the deterministic CK control plane may advance an authoritative branch head. Each proposal targets an exact predecessor head or typed target absence. Slow evaluation and evidence acquisition precede activation; a short transaction then revalidates all mutable facts governing admission and records exactly one durable disposition. Figure 1 illustrates this activation boundary.

CK introduces no new low-level transaction engine. Instead, standard substrates—such as relational database transactions, conditional object manifests, or consensus statemachine logs—execute its atomic step. The primary contribution is the agent-state activation contract enforced within that step: it atomically binds proposal identity, exact predecessor state, pre-state authority, acquired evidence, lifecycle status, effect manifests, lineage, and outcome receipts. Specifically, this paper makes three contributions:

1. System Model and Contract: A formal model separating stored retention from authoritative reachability, defining the boundary between untrusted proposers and trusted activation (Section 2);

2. Activation Protocol: An owner-bound, exact-head activation protocol with pre-state authorization, four terminal dispositions (Commit, Reject, Quarantine, Defer), at-most-once effect execution, and explicit receipt evidence levels (Section 3); and

![](images/f0d03f1a85a6ad6825a145aab57ab96e5802970fde84b8e11cd0fba286230262.jpg)  
Figure 1: Probabilistic proposal generation and remote evaluation precede the short activation transaction. Only activation advances an authoritative head.

Table 1: Storage does not by itself confer authority.
<table><tr><td>Namespace</td><td>Purpose</td><td>In head?</td></tr><tr><td>Accepted</td><td>Current typed state</td><td>Yes</td></tr><tr><td>Candidate</td><td>Prepared successor</td><td>No</td></tr><tr><td>Attempt</td><td>Outcome and diagnostics</td><td>No</td></tr><tr><td>Quarantine</td><td>Isolated candidate</td><td>No</td></tr><tr><td>Projection</td><td>Read-only runtime view</td><td>No</td></tr></table>

3. Lifecycle Rules and Model Verification: Typed lifecycle semantics for branch creation, writer handoff, schema migration, and forward restoration, verified through bounded state-space exploration (Sections 4– 5).

Sections 2–5 develop the core architecture and evaluation; Appendices A–D detail normative specifications, lineage proofs, and model parameters.

## 2 System Model and Contract

## 2.1 Authority Is Reachability

As illustrated in Figure 1, CK establishes a strict boundary between candidate object preparation and authoritative state activation. Agent state is partitioned into isolated branches identified by a subject/system identifier sid and a branch identifier bid, denoted $\boldsymbol { k } = ( s i d , b i d )$ . The branch-indexed state X[k] holds the complete collection of typed agent components for branch $k ,$ including memory records, user profiles, tool configurations, and policy-bearing objects.

As summarized in Table 1, the storage layer maintains multiple functional namespaces. Candidate proposals, execution attempt logs, quarantined candidates, and read-only runtime projections coexist in storage alongside accepted state. However, storage presence alone does not confer authority: an object is authoritative if and only if it is reachable from the current branch head committed by CK.

For branch $\boldsymbol { k } = ( s i d , b i d )$ , the branch directory maintains

the complete-head type h and branch-row type $B [ k ]$ :

$$
\begin{array} { r } { \begin{array} { r } { h = \langle s i d , b i d , s e q , r o o t , v , p a r e n t R e f , l i n e a g e R e f \rangle , } \\ { B [ k ] = \langle h , s t a t u s , w r i t e r , e p o c h , d i r S e q , h a n d o f f , c r e a t e d F r o m \rangle . } \end{array} } \end{array}\tag{1}
$$

Here root commits the typed component state; v = (schemaV, policyV, evalV, authV ) names the pinned versions; parentRef is the parent head reference; lineageRef binds the causal lineage; $d i r S e q$ is directory sequence; handoff is the open revision pointer; and createdFrom records genesis provenance. Heads are equal only when every canonically encoded field agrees.

The abstract kernel configuration is

$$
C = \langle X , B , \Gamma , S , t \rangle .\tag{2}
$$

Γ is the pre-state authority context; S represents persistent metadata stores (outcomes, receipts, lineage, effects, and quarantine); and t is trusted transaction time. An activation serializes the branch row and every admission, allocation, and effect row that its proposal names.

## 2.2 Proposals and Typed Transitions

Models, tools, memory services, and operators are untrusted proposers. Evaluators and approvers issue signed evidence but cannot advance a head. A trusted gateway authenticates proposal ownership; a preparation service acquires the required evidence and derives a candidate; the activation engine alone changes authority.

A signed proposal is summarized by

$$
\tau = \langle p i d , t a r g e t , e x p e c t e d , o p s , a , r e q \rangle .\tag{3}
$$

Here pid is the unique proposal identifier; target is the target branch key $k = ( s i d , b i d ) ;$ ; expected is the expected predecessor head; ops is an ordered sequence of state operations; a is a sequence of authority changes; and req declares the evidence required for admission.

Candidate derivation occurs off-commit, producing candidate state $X ^ { \prime } = F ( X , o p s , W )$ using evidence W. Crucially, authorization evaluates against the pre-state context A:

$$
X ^ { \prime } = F ( X , o p s , W ) , \qquad A = \left\{ \Gamma , \quad \mathrm { e x i s t i n g ~ b r a n c h } , \right.\tag{4}
$$

Authorize $_ A ( \tau , W )$ runs before computing the authority update $\Gamma ^ { \prime } = F _ { A } ( A , a ) ;$ ; a proposal cannot authorize itself. The candidate seal binds the proposal, evidence, interpreter, and component roots. Activation recomputes these bindings against the serialized branch state.

Schema evolution is an explicit Migration transition with a pinned, deterministic migrator. Deletion creates a typed tombstone; physical erasure remains subject to retention policy. Hashes establish cryptographic integrity, not confidentiality, semantic truth, or policy legitimacy.

## 2.3 Threat Model and Conditional Contract

Proposers may be buggy or adversarial: they may replay requests, reuse identifiers, reorder operations, bind a stale head, omit evidence, forge scope, or request self-authorizing changes. Infrastructure may crash, lose replies, duplicate messages, expose stale replicas, or partition. Evaluators may be wrong; their outputs are policy inputs rather than semantic oracles.

The safety claims depend on nine explicit assumptions (A1– A9, stated individually in Appendix D): (1) Boundary & Cryptography (A1–A3): all authoritative writes cross an authenticated kernel boundary, typed cryptography is sound, and signing keys are protected; (2) Atomic Serialization (A4): the storage substrate atomically serializes the complete activation key set; (3) Complete Context & Time (A5–A6): policy, authority, revocation, lifecycle, dependency, and trusted-time values are available for commit-time validation; (4) Lifecycle Order & Non-Recycled Scopes (A7–A8): lifecycle operations share one order and proposal/effect scopes and writer epochs are not recycled; and (5) Verification Objects (A9): stronger receipt claims are made only when their required verification objects are available.

Under those assumptions, the contract has four consequences:

1. Exact succession. One complete predecessor has at most one accepted successor; one serialized absence has at most one genesis. Each is installed with state, lineage, effects, outcome, and receipt.

2. Pre-state admission. Authorization and freshness are checked against the serialized predecessor or creation context, never the proposed context.

3. Stable execution identity. A proposal identifier has one terminal disposition and an effect identifier has at most one accepted binding, including after safe reclamation.

4. Lifecycle isolation. Branch creation, fencing, handoff, migration, and restoration preserve branch scope and current authority.

These are safety properties, not availability guarantees. The kernel may be unavailable or may return Defer when evidence cannot be obtained. It governs only state behind its mutation boundary; prompts, caches, model weights, toollocal data, prior disclosures, and remote side effects remain outside unless separately mediated.

## 3 Activation Protocol

The protocol separates slow, fallible preparation from one short serialized decision. Preparation authenticates the proposer, acquires exactly the evidence declared by the proposal, runs the pinned transition function, and seals the candidate. Activation admits that sealed package only if the relevant state is still current. Appendix A fixes the complete stage order and receipt variants; this section states the activation rule.

## 3.1 One activation predicate

Let $P = \langle \tau , W , X ^ { \prime } , M _ { E } , s \rangle$ be a prepared package containing the signed proposal, exact evidence set, candidate state, ordered effect manifest, and trusted candidate seal. Inside the activation transaction, the kernel evaluates the ordered vector

$$
\mathcal { G } _ { k i n d } ( C , P ) = \langle G _ { a } ^ { k i n d } ( C , P ) \rangle _ { a \in \mathsf { A c t } \mathsf { C h e c k } } .\tag{5}
$$

The kernel evaluates this vector strictly in the Appendix A order and accepts only an all-Pass vector. Table 2 gives one predicate per stage; no aggregate guard can move a failure across stages.

The trusted acquisition service returns an exact one-to-one mapping for the declared requirements: no declared requirement is omitted, and no unrequested evidence is included. Any unavailable evidence is returned as an explicit, authenticated Missing status, preventing callers from manufacturing Defer by suppressing inconvenient facts or injecting extraneous context. Mutable requirements specify explicit serialization keys and versions, which VersionBinding revalidates at commit time. Freshness is validated prior to policy evaluation and re-checked immediately before prospective state construction. Ordinary transitions revalidate the exact branch head snapshot, while BranchCreate revalidates target absence, creation context $\xi _ { k } ,$ , allocator versions, and genesis inputs.

The candidate seal certifies that the candidate state was derived by the pinned deterministic interpreter from the signed inputs and acquired evidence. Authorization is a separate commit-time predicate evaluated over the pre-state authority context $A _ { \chi }$ . The subsequent Decision stage records the policy result, and an authority transition Γ<sup>′</sup> is computed only after both authorization and decision stages succeed.

Table 2: One commit-time predicate per activation stage; read down the left pair, then down the right pair.
<table><tr><td>Stage</td><td>Predicate</td><td>Stage</td><td>Predicate</td></tr><tr><td>TargetLookup</td><td>Existing kind:  $B [ k ]$  exists. BranchCreate:  $B [ k ]$  is absent and serialized E[k] exists.</td><td>CandidateBinding</td><td>Seal binds proposal, evidence, interpreter, candidate root, and applicable creation-context or open-revision digest.</td></tr><tr><td>Allocation</td><td>Owner allocation is current; pid/eid names are live, un- EffectBinding used, and unretired.</td><td></td><td>Operations and manifest are bijective; effect scopes are live and unused.</td></tr><tr><td>ExpectedHead</td><td>Existing kind: Present(dh) is exact; BranchCreate: ex- Freshnesslnitial pectation is Absent.</td><td></td><td>Revalidate time and the kind-indexed snapshot: head/epoch/dirSeq/authority, or absence  $\prime d _ { \xi } /$  genesis</td></tr><tr><td>Lifecycle</td><td>Status, writer, epoch, and kind-specific lifecycle row are admissible.</td><td>Authorization</td><td>inputs.  $A _ { x } -$  predecessor Γ or serialized creation context  $\xi _ { k } -$ </td></tr><tr><td>HandoffRevision</td><td>Applicable target, open SourceFenced revision, reserved epoch, and dirSeq are exact.</td><td>Decision</td><td>authorizes the operations. Policy records Proceed, Reject, or Quarantine indepen- dently of authorization.</td></tr><tr><td>AcquisitionAuth</td><td>Acquisition result set and signer are authentic.</td><td>AuthorityTransition</td><td>A declared authority change is valid under the authorized</td></tr><tr><td>RequirementCoverage</td><td>Signed requirements and results form an exact ordered cover.</td><td>FreshnessFinal</td><td>kind context  $A _ { x } .$  Repeat the complete kind-indexed snapshot immediately</td></tr><tr><td>VersionBinding</td><td>Policy, evaluator, authority, revocation, issuer/allocation,WellFormedness dependency, and processing versions are current.</td><td></td><td>before prospective construction. Receipt-independent prospective unit  $P _ { x }$  is complete, de- </td></tr><tr><td>WitnessBinding</td><td>Every required witness or authenticated Missing is bound exactly.</td><td></td><td>terministic, and well typed.</td></tr></table>

Table 3: Only Commit changes authoritative state.
<table><tr><td>Result</td><td>Meaning</td></tr><tr><td>Commit</td><td>Install the exact successor and all acceptance metadata atomically.</td></tr><tr><td>Reject</td><td>A permanent validation or policy failure; retain no successor.</td></tr><tr><td>Quarantine</td><td>Retain sealed material outside the authoritative head.</td></tr><tr><td>Defer</td><td>Trusted acquisition reports required evidence un- available.</td></tr></table>

## 3.2 Four stable dispositions

As illustrated in the proposal transition lifecycle (Figure 2), for an authenticated, owner-bound identifier, the kernel records exactly one of four dispositions:

$$
q \in \{ { \mathrm { C o m m i t } } , { \mathrm { R e j e c t } } ( r ) , { \mathrm { Q u a r a n t i n e } } ( r ) , { \mathrm { D e f e r } } ( r ) \} .\tag{6}
$$

Every disposition is terminal for its proposal identifier. $\mathrm { R e } \mathrm { - }$ submission after Defer or Quarantine uses a new identifier linked to the old attempt; a lost response is resolved by looking up the original identifier. Malformed outer framing, failed outer authentication, or an invalid allocation capability are protocol errors rather than dispositions and must not consume the identifier. After ownership succeeds, a fixed stage order chooses the result: any permanent failure that can already be evaluated precedes a trusted Missing; later predicates remain unevaluated. This prevents a missing witness from masking a known invalid proposal.

Algorithm 1 abstracts concrete wire encodings and storage driver APIs. After all validation stages pass, the kernel constructs the receipt-independent prospective unit ${ \widehat { P } } \mathbf { : }$

$$
\widehat { P } = { \tt B u i l d P r o s p e c t i v e } ( X ^ { \prime } , h , B [ k ] ) .\tag{7}
$$

Algorithm 1 Prepare and activate a proposal (abstract)   
Require: signed proposal τ and authenticated allocation   
1: authenticate the principal and ownership of pid   
2: return a prior outcome or identifier conflict, if present   
3: run PrepStage to Ready; acquire and derive only at named   
stages   
4: on terminal preparation Fail/Missing, durably return q(r<sub>P</sub>)   
5: begin a transaction over the branch and every named mutable   
key   
6: return the prior outcome if another retry installed it   
7: read every row named by ActCheck   
8: for a in the normative ActCheck order do   
9: if a = WellFormedness then   
10: $\widehat { P } _ { \chi } \gets \mathsf { B u i l d P } _ { \mathsf { I } }$ rospective $_ { \stackrel { . } { x } } ( P , C )$   
11: end if   
12: evaluate only $G _ { a } ^ { k i n d }$ and append its stage result   
13: if $\cdot _ { a } =$ Decision and result is Quarantine then   
14: durably return Quarantine $( r _ { Q } )$   
15: else if result is Fail or Missing then   
16: durably return $q ( r _ { A } )$   
17: end if   
18: end for   
19: $r _ { C }  r b _ { C } ( \widehat { P } _ { \chi } ) ;$ compute $d _ { r _ { C } }$   
20: $\mathcal { U } _ { a c c e p t } ( \chi ) $ Finalize $_ { x } ( \widehat { P } _ { x } , d _ { r _ { C } } )$   
21: atomically install the complete unit; return Commit(r<sub>C</sub>)

Construction follows a strictly cycle-free order:

$$
\widehat { P } \longrightarrow { \sf W e l l F o r m e d n e s s } ( \widehat { P } ) \longrightarrow r _ { C } \longrightarrow h ^ { \prime } \longrightarrow \mathcal { U } _ { \mathrm { a c c e p t } } .\tag{8}
$$

First, the Commit receipt $r _ { C }$ is derived directly over ${ \widehat { P } } _ { : }$ , binding candidate state $X ^ { \prime }$ , evidence, and pinned versions without circular dependency on the receipt itself. Second, the complete head $\bar { h } ^ { \prime }$ is finalized by binding $r _ { C }$ and lineage. Finally, the kernel installs the complete accepted unit $\mathcal { U } _ { \mathrm { a c c e p t } } \mathrm { : }$

$$
\mathcal { U } _ { \mathrm { a c c e p t } } = \{ X ^ { \prime } , \Gamma ^ { \prime } , h ^ { \prime } , B ^ { \prime } [ k ] , r _ { C } , O [ p i d ] \} .\tag{9}
$$

![](images/ab22bbeb5141444c796d0ce20e1df495483aed2df084b50986b6c47b895f8b2b.jpg)  
Figure 2: Proposal transition lifecycle and four terminal dispositions, expanding chronologically left-to-right.

Table 4: A signature is not evidence that its transaction committed.
<table><tr><td>Level</td><td>Established claim</td></tr><tr><td>Structural</td><td>The receipt has canonical syntax, typed bindings, and a valid signature under its declared key.</td></tr><tr><td>Attested</td><td>A trusted kernel key attests the first terminal stage and reason; this does not independently establish correct evaluation.</td></tr><tr><td>Replay</td><td>Retained inputs and pinned versions reproduce the stated decision or transition.</td></tr><tr><td>Inclusion</td><td>A certified snapshot or log proof places the re- ceipt and outcome, and for Commit the head and lineage, in durable state.</td></tr></table>

The set $\mathcal { U } _ { \mathrm { a c c e p t } }$ denotes one all-or-nothing atomic transaction, installing candidate state $X ^ { \prime }$ , updated authority context $\Gamma ^ { \prime }$ finalized branch row $B ^ { \prime } [ k ]$ pointing to head $h ^ { \prime } ,$ execution outcome $O [ p i d ]$ , and receipt record $r _ { C }$ . Any non-Commit disposition leaves X, B, and Γ unchanged.

Proposal and effect records need not remain online forever. Reclamation first advances a durable, monotonically increasing watermark for a non-recycled allocation scope and only then removes covered rows. An identifier below that watermark returns RetiredIdentifier rather than re-entering execution. This preserves stable identity while allowing bounded online metadata.

## 3.3 Receipts state what they prove

A receipt binds the proposal, disposition, reached protocol stage, decisive reason, and the evidence available at that stage. Commit receipts additionally bind the applicable typed parent and context, successor core, lineage, authority versions, and effect manifest. Preparation failures need not pretend that a canonical proposal or candidate existed.

Receipt verification has four increasing evidence levels:

The distinction matters because a kernel may sign a receipt before its storage transaction aborts. Replay strength also depends on retained objects: deleting an old proposal may preserve replay exclusion through its watermark while removing the evidence needed to reproduce its original decision.

Appendix A specifies these verification obligations.

## 3.4 Safety properties under the stated assumptions

Under the conditional assumptions A1–A9 introduced in Section 2 and detailed in Appendix D, the protocol satisfies three primary safety properties:

Proposition 1 (Owner-bound stable outcome). Under A1–A4 and A8, only the principal named by an active allocation may create O[pid], and at most one disposition becomes durable for that identifier.

Proposition 2 (Single exact continuation). Under A1–A4, at most one proposal commits from one complete predecessor or serialized target absence, and the installed state is its sealed candidate.

Proposition 3 (At-most-once accepted effect). Under A1, A4, and A8, each effect identifier has at most one accepted binding, including after online effect records are reclaimed.

The proofs are short serialization arguments and appear in Appendix B. The propositions do not claim semantic correctness: a policy may approve a false memory, and an idempotent intent may still cause a non-idempotent remote action if its connector is faulty.

## 3.5 Realization boundary

The transaction contains no model call, network fetch, or hu man interaction; those finish during preparation. A relational database may lock the branch and named context rows. An object store may conditionally install one immutable commit manifest, and a replicated service may serialize one activation command. In every case readers reject a head whose accepted unit is incomplete.

CK atomicity ends at its state boundary. External actions should be committed as intents and delivered through an idempotent outbox when possible. No local protocol can make an irreversible action atomic with an unrelated remote system. Similarly, strict revocation requires the revocation row to share the activation serialization order; a remote revocation service provides only a declared bounded-staleness guarantee.

![](images/2a4b35bca0e568165e0555cb860ab636c2aa7dc3ba01bf18a9e8990da39b8cb7.jpg)  
Figure 3: Handoff first removes the source writer, then activates the target. Retargeting advances the epoch while the branch remains fenced.

## 4 Branch Lifecycle and Restoration

Copying state, transferring write authority, and restoring old content are different operations. Treating all three as “load checkpoint” can silently fork a branch or revive an obsolete credential. CK gives each operation a typed forward transi tion.

## 4.1 Branch creation

A new branch has no predecessor. Its genesis record binds the authenticated creation request, $d _ { \xi } ,$ , fresh identifier, initial roots/writer, and absence proof. Candidate derivation uses declared genesis inputs; authorization uses serialized $\xi _ { k } ,$ never nonexistent predecessor state or authority. Both freshness checks protect continued absence, $\xi _ { k }$ versions/dirSeq, and all bound genesis inputs. An optional source head is provenance, not a parent.

Branch creation runs Algorithm 1 with kind BranchCreate and records the ordinary Commit disposition. One atomic unique insert installs the initial state, admission context, genesis lineage, receipt, complete head, and branch-directory row. Two concurrent creators cannot both win the absent-key comparison. Every later transition has an ordinary accepted parent. Reconciliation between branches is a later typed proposal over exact source and target heads; the kernel does not choose a semantic merge policy.

## 4.2 Writer handoff

A handoff uses a monotonically versioned directory record. Figure 3 shows the small state machine; Table 5 explains its actions. Every step compares the exact current directory revision and appends a signed lifecycle result, yielding main tenance receipts distinct from proposal dispositions.

The gap between Fence and Activate is intentional. After fencing, no writer is authorized; therefore a crash cannot expose two active writers. A retry either uses the same handoff revision or fails after retargeting. Target activation is the intentional exception to writer-bearing admission: it requires the exact Frozen row, writer ⊥, open SourceFenced revision, target, canonical epoch, separately reserved successor epoch, and current admission context. Its ordinary Commit receipt is indexed by both O and $H _ { R }$ , so there is one authoritative activation receipt.

Table 5: Directory-serialized handoff actions.
<table><tr><td>Action</td><td>Atomic directory effect</td></tr><tr><td>Prepare</td><td>Append Prepared and open its pointer; source may still advance.</td></tr><tr><td>Abort</td><td>Append Aborted, clear the pointer, and keep the source active.</td></tr><tr><td>Fence</td><td>Append SourceFenced, advance the open pointer, clear writer, advance epoch.</td></tr><tr><td>Retarget</td><td>Append SourceFenced and advance the open pointer, target, and epoch.</td></tr><tr><td>Activate</td><td>Run Algorithm 1 against the fenced revision and append TargetActive, clear the pointer, and make the target writer active.</td></tr></table>

Prepared recovery may retry, fence, or abort. SourceFenced recovery may retry target activation or append a retarget, but it never returns to Prepared. If directory and branch storage cannot share atomic ordering, a consensus or coupled protocol is required; polling alone cannot establish writer isolation.

## 4.3 Migration and forward restoration

Migration names source and target schemas plus a pinned deterministic migrator. The migrator must be allowed by current policy, terminate with a well-formed target, and obey an explicit loss policy. A rollback is another forward migration, never a rewrite of an accepted head.

Restoration also creates a new successor. Let $X _ { c }$ be current state, $X _ { k }$ an authenticated historical checkpoint, $M _ { \mu }$ an optional migration, and M a typed path mask. The restored candidate is

$$
X ^ { \prime } = \mathrm { R e s t o r e } _ { M } ( X _ { c } , M _ { \mu } ( X _ { k } ) ) , \qquad \Gamma ^ { \prime } = \Gamma _ { c } .\tag{10}
$$

Paths in M take checkpoint values after migration; all other paths take current values. Authority, revocation, writer epoch, and lifecycle fields are protected and therefore remain current. The successor’s parent is the current head, while the checkpoint is an auxiliary provenance reference. Restoration cannot truncate lineage, revive an old writer or credential, or undo an external action already performed. Appendix C gives the total postconditions and stale-request behavior for every lifecycle action.

## 5 Evaluation

We evaluate the Continuity Kernel (CK) design by addressing four key research questions:

• RQ1 (Activation Integrity & Succession): Does the 17-stage activation predicate enforce exact predecessor succession, pre-state authorization, and non-selfauthorizing state transitions across all five transition kinds?

• RQ2 (Concurrency & Replay Isolation): Does the protocol maintain at-most-once execution identity, prevent replay attacks, and safely reclaim online proposal/effect records under concurrent races and watermark advances?

• RQ3 (Lifecycle Isolation & Writer Fencing): Do lifecycle rules for branch creation, writer handoff, retargeting, aborts, and forward restoration prevent split-brain access and stale writer state transitions?

• RQ4 (Self-Critical Scope & Realization Limits): What are the exact bounds of the formal state-space exploration, and what system-level guarantees require physical storage-engine verification beyond the abstract model?

## 5.1 Formal State-Space Exploration (RQ1–RQ3)

To evaluate the protocol’s state-transition logic, we construct an executable bounded model in Python $( \mathsf { a r t i f a c t s } /$ bounded\_model.py). The model performs exhaustive breadth-first search (BFS) state exploration over a finite abstraction of the kernel specification using only the Python standard library.

The model state space incorporates: (1) one active branch directory $B [ k ]$ and state store $X [ k ]$ ; (2) a shared admission context Γ and creation context $\Xi [ k ] ;$ ; (3) an honest proposal owner and an adversarial principal; (4) proposal identifiers pid $\in \{ 0 \ldots 1 2 \}$ (13 IDs) and effect identifiers eid $\in \{ 0 \ldots 3 \}$ (4 IDs); (5) source writer $w _ { 0 }$ and target writers w<sub>1</sub>, w<sub>2</sub>; and (6) integer versioning for schema, policy, and authority contexts.

Transitions simulate all five closed transition kinds (Ordinary, Migration, Restoration, HandoffActivate, BranchCreate), exact-head concurrency races, pre-state authority changes, missing/malformed evidence, two-stage freshness expiration, resubmissions, watermark reclamation, writer fencing, retargeting, and masked restoration.

Under CPython 3.13.9, the depth-seven search evaluates 8,880,248 total transition attempts (including idempotency checks and retries). Of these, exactly

$$
N _ { \mathrm { s t a t e s } } = 2 , 8 0 8 , 2 3 0 , \qquad N _ { \mathrm { t r a n s i t i o n s } } = 5 , 5 2 6 , 4 7 4\tag{11}
$$

are unique state-changing transitions where the successor state differs from the predecessor state. As shown in Table $^ { 6 , }$ state space exploration scales exponentially up to depth 7. Depth 6 timing (124.62s) includes evaluating all 7.4M outgoing transition attempts to generate depth 7 states, whereas depth 7 timing (27.45s) reflects terminal invariant validation on the 2.27M boundary states without further successor expansion. On a standard single-threaded execution harness, kernel transition evaluation achieves a throughput of 31,132 transitions/sec with an average evaluation latency of 32.12 µs per transition.

Table 6: State-space expansion and BFS exploration metrics by search depth d.
<table><tr><td>Depth</td><td>Unique States</td><td>Cum. States</td><td>Cum. Attempts</td><td>Time (s)</td></tr><tr><td>0</td><td>1</td><td>1</td><td>0</td><td>&lt;0.01</td></tr><tr><td>1</td><td>15</td><td>16</td><td>17</td><td>&lt;0.01</td></tr><tr><td>2</td><td>160</td><td>176</td><td>270</td><td>0.04</td></tr><tr><td>3</td><td>1,455</td><td>1,631</td><td>2,959</td><td>0.33</td></tr><tr><td>4</td><td>11,309</td><td>12,940</td><td>27,353</td><td>2.69</td></tr><tr><td>5</td><td>76,028</td><td>88,968</td><td>216,413</td><td>18.88</td></tr><tr><td>6</td><td>445,649</td><td>534,617</td><td>1,483,284</td><td>124.62</td></tr><tr><td>7</td><td>2,273,613</td><td>2,808,230</td><td>8,880,248</td><td>27.45</td></tr></table>

Table 7: Distribution of transition outcomes across full state space (8.88M transition attempts).
<table><tr><td>Transition Outcome</td><td>Full Count</td><td>Share (%)</td></tr><tr><td>IdReuseConflict</td><td>1,673,300</td><td>18.84%</td></tr><tr><td>Commit (Head Advance)</td><td>916,956</td><td>10.33%</td></tr><tr><td>ReclaimedEffects</td><td>842,195</td><td>9.48%</td></tr><tr><td>RejectStaleHead</td><td>770,555</td><td>8.68%</td></tr><tr><td>PrefixNotFinal</td><td>731,466</td><td>8.24%</td></tr><tr><td>Defer (Missing Evidence)</td><td>418,375</td><td>4.71%</td></tr><tr><td>RejectDuplicateInProposal</td><td>418,375</td><td>4.71%</td></tr><tr><td>RejectUnauthorized</td><td>395,905</td><td>4.46%</td></tr><tr><td>RejectExpiredAtActivation</td><td>395,905</td><td>4.46%</td></tr><tr><td>RejectCandidateBinding</td><td>395,905</td><td>4.46%</td></tr><tr><td>RejectPolicy</td><td>395,905</td><td>4.46%</td></tr><tr><td>Prepared (Handoff)</td><td>306,121</td><td>3.45%</td></tr><tr><td>Other Dispositions (11</td><td>1,219,285</td><td>13.73%</td></tr><tr><td>kinds) Total Attempts</td><td>8,880,248</td><td>100.00%</td></tr></table>

Table 7 details the empirical distribution of transition dispositions across all 8.88M evaluated transition attempts. Successful state transitions (Commit) represent 916,956 committed heads (10.33%), while concurrency and identifier reuse protections (RejectStaleHead, IdReuseConflict, ReclaimedEffects) account for 37.00% (3.28M transitions), proving that the protocol actively isolates concurrent races and stale proposals.

As summarized in Table 8, the exploration finds zero encoded invariant violations across all 2.8 million reachable states and reaches 100% of the named protocol coverage witnesses.

Table 8: Reached coverage witnesses and invariant assertions (2, 808, 230 reachable states, depth 7).
<table><tr><td>Target Category</td><td>Evaluated Invariant / Wit- ness</td><td>Status</td></tr><tr><td>Dispositions</td><td>Reached Commit, Reject, Quarantine, and Defer</td><td>Pass</td></tr><tr><td>Activation Guard</td><td>17 stages evaluated in norma- tive sequential order</td><td>Pass</td></tr><tr><td>Authorization</td><td>Self-authorizing proposal (τ.a) rejected at pre-state</td><td>Pass</td></tr><tr><td>Freshness</td><td>Revalidated initially and im- mediately prior to commit</td><td>Pass</td></tr><tr><td>Reclamation</td><td>Safe watermark advances for pid and eid scopes</td><td>Pass</td></tr><tr><td>Exact Succession</td><td>At-most-one accepted succes-</td><td>Pass</td></tr><tr><td>Writer Fencing</td><td>sor per complete head Stale/abandoned writer blocked from branch muta-</td><td>Pass</td></tr><tr><td>Handoff Lifecycle</td><td>tion SourceFenced, TargetActive, Retarget, and Abort</td><td>Pass</td></tr><tr><td>Restoration</td><td>Masked restored root with protected current fields</td><td>Pass</td></tr></table>

## 5.2 Invariant & Safety Analysis

The model validates three core structural properties across every reachable state:

1. Pre-State Authorization Safety (RQ1). For every committed transition, authorization is checked against the predecessor authority context Γ (or genesis context $\xi _ { k }$ for BranchCreate). In the model, when an adversarial proposer submits a self-authorizing proposal that attempts to inject its own required permissions into $^ { a , }$ the Authorization stage evaluates Authorize<sub>Γ</sub>(τ, W) and rejects 395,905 invalid proposals (4.46% of transitions) with RejectUnauthorized.

2. Execution Identity & Replay Safety (RQ2). Each proposal identifier pid has at most one durable disposition O[pid], and each effect identifier eid is bound at most once in E[eid]. Watermark advances correctly transition covered online identifiers to RetiredIdentifier, preventing re-execution or identifier recycling attacks after online metadata is pruned.

3. Writer Fencing & Lifecycle Isolation (RQ3). During writer handoff, once the source writer is fenced (SourceFenced), any subsequent write attempt by the old writer w fails with StaleRevision or WriterEpoch. Retargeting to a new target writer w<sub>2</sub> advances the writer epoch, ensuring that an abandoned target writer $w _ { 1 }$ cannot activate the branch.

## 5.3 Self-Critical Thesis Analysis & Realization Limits (RQ4)

While bounded model exploration provides strong evidence for the logical consistency of the finite protocol, a self-critical assessment highlights critical gaps between the formal model and physical system implementations:

1. Abstraction vs. Physical Storage Realization. The model treats preparation and atomic activation as single logical state steps. In a production system, atomic activation relies on the underlying storage engine (e.g., PostgreSQL conditional updates, FoundationDB OCC, or Raft consensus). Physical storage crashes during write-ahead logging (WAL), network partitions, or storage-engine bugs fall outside the model’s abstract state space.

2. Signature vs. Durable Inclusion. As established in Section 3.3, a signed candidate seal or preparation receipt proves cryptographic origin (Attested level) but not transactional durability (Inclusion level). If a kernel gateway signs a receipt in memory but the backing transaction fails to commit, the signed receipt is uncommitted. Production deployments must issue Level 4 inclusion proofs backed by durable log anchors.

3. External Side-Effect Atomicity. As discussed in Section 3, CK atomicity governs state transitions inside its mutation boundary. It cannot make irreversible remote side effects (e.g., external API calls or physical robot actuation) atomic with internal state updates. Remote actions must be staged as idempotent outbox intents.

4. Performance & Latency Trade-offs. The protocol adds overhead during commit: 17 sequential stage checks, double freshness validation, and prospective unit construction (32.12 µs in single-threaded Python). In high-throughput environments, this overhead requires storage-level optimizations, such as single-pass SQL transaction procedures or batched multi-key compare-and-swap operations.

## 6 Related Work

Agent memory. Generative Agents and MemGPT organize reflection, retrieval, and long-term context [1, 2]; LoCoMo, A-MEM, Mem0, and MemOS study long-horizon evaluation, consolidation, scalable memory, and versioned management [3–6]. These systems motivate governed memory. As part of the broader Persistent Cognitive Identity (PCI) framework [7], CK addresses the narrower question of how any typed component version becomes the authoritative branch head.

MemTX is the closest agent-memory transaction system: it stages evidence-bearing beliefs under a snapshot, gates actions, and propagates repairs after retraction [8]. CK ap plies an activation boundary to typed persistent state and specifies pre-state authority, complete-head equality, stable four-way outcomes, writer epochs, and forward restoration. MemTxn supplies source-supported memory admission, visible-version selection, snapshots, and journal-based recovery [9]. MemTX’s unit is a belief commit and MemTxn’s is a source-supported memory update; CK specifies a typed branch-head transition and its pre-state authority boundary.

Transactions and retries. Optimistic concurrency control validates read assumptions at commit, database transactions provide atomic durability, and replicated terms fence stale leaders [10–12]. For linearizable retries, RIFL (Reusable Infrastructure for Linearizability) combines unique request identifiers, atomic completion records, and safe reclamation [13]. CK composes these mechanisms into an agent-state contract: the accepted unit also binds typed state, authority, evidence, lineage, effects, and a calibrated receipt. Commit-time authorization for LLM agents similarly exposes the gap between temporary authority and durable effects [14]; CK places that check inside a persistent-state transition.

Effects and evidence. Cordon defines task-level semantic transactions with staged effects, delegated authorization, compensation, and audit evidence [15]. Its task boundary complements CK’s branch-head boundary: an outbox intent may be a CK component, but CK cannot make an unrelated remote action atomic. PROV-DM, JSON-LD, RDFC-1.0, and Data Integrity provide provenance, canonicalization, and cryptographic proof formats [16–19]. Those standards can show derivation and integrity; the activation protocol is still needed to determine which validly encoded proposal became authoritative.

## 7 Conclusion

For persistent agents, state retention is not authority. Infrastructural continuity requires an authorized lineage of accepted branch heads. CK makes the transition to authority explicit: untrusted components propose typed candidates off-commit; a deterministic control plane validates an exact predecessor head and pre-state authority (or typed absence and genesis context); a short activation transaction records one stable disposition; and only Commit advances the branch head. The protocol contract addresses linearizable retries, effect uniqueness, writer epoch fencing, schema migration, and forward restoration without expanding the kernel’s trusted semantic scope.

Our safety propositions are conditional on explicit serialization, access-control, cryptographic, and durability assumptions. In a depth-seven finite abstraction, an executable bounded model explores 2,808,230 reachable states and 5,526,474 state-changing transitions, finding zero encoded invariant violations and reaching 100% of named coverage witnesses. The model validates logical protocol consistency within its finite bounds; it does not establish unbounded mathematical correctness, physical storage driver conformance, or throughput performance under hardware faults. The core contribution of this work is the activation contract itself—providing a principled systems foundation that separates stored retention from authoritative reachability and distinguishes cryptographic receipt signatures from durable transactional inclusion.

## References

[1] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology, pages 1–22, 2023.

[2] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560, 2023.

[3] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 13851–13870. Association for Computational Linguistics, 2024.

[4] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-MEM: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems 38, pages 17577–17604. Curran Associates, Inc., 2025.

[5] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready AI agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

[6] Zhiyu Li, Chenyang Xi, Chunyu Li, Ding Chen, Boyu Chen, Shichao Song, Simin Niu, Hanyu Wang, Jiawei Yang, Chen Tang, Qingchen Yu, Jihao Zhao, Yezhaohui Wang, Peng Liu, Zehao Lin, Pengyuan Wang, Jiahao Huo, Tianyi Chen, Kai Chen, Kehang Li, Zhen Tao, Huayi Lai, Hao Wu, Bo Tang, Zhengren Wang, Zhaoxin Fan, Ningyu Zhang, Linfeng Zhang, Junchi Yan, Mingchuan Yang, Tong Xu, Wei Xu, Huajun Chen, Haofen Wang, Hongkang Yang, Wentao Zhang, Zhi-Qin John Xu, Siheng Chen, and Feiyu Xiong. MemOS: A memory OS for AI system, 2025. Preprint, arXiv:2507.03724v4, version 4, 3 December 2025. https://arxiv.org/abs/2507.03724v4.

[7] Jun He and Deying Yu. Persistent cognitive identity: A systems architecture for continuity across AI substrates and embodiments. Position and architecture manuscript, OpenKedge.io, 2026.

[8] Xiaoyang Li, Yiqi Wang, Haohui Lu, Zhi Chen, Mo Li, Pingan Song, Mingkai Zheng, and Taotao Cai. MemTX: Transactional belief commit for stateful agent memory, 2026. Preprint, arXiv:2607.23929v2, version 2, 28 July 2026. https://arxiv.org/abs/ 2607.23929v2.

[9] Hanshuai Cui, Zhiqing Tang, Zhi Yao, Fanshuai Meng, Qianli Ma, and Weijia Jia. MemTxn: A transaction boundary for source-supported updates and complete-state recovery in agent memory. arXiv preprint arXiv:2607.27834, 2026. Version 1, submitted July 30, 2026.

[10] H-T Kung and John T Robinson. On optimistic methods for concurrency control. ACM Transactions on Database Systems (TODS), 6(2):213–226, 1981.

[11] Jim Gray and Andreas Reuter. Transaction Processing: Concepts and Techniques. Morgan Kaufmann, 1992.

[12] Diego Ongaro and John Ousterhout. In search of an understandable consensus algorithm. In 2014 USENIX Annual Technical Conference (USENIX ATC 14), pages 305–319, 2014.

[13] Collin Lee, Seo Jin Park, Ankita Kejriwal, Satoshi Matsushita, and John Ousterhout. Imple menting linearizability at large scale and low latency. In Proceedings ofthe 25th Symposium on Operating Systems Principles, pages 71–86. ACM, 2015.

[14] Igor Santos-Grueiro. Temporary authority, permanent effects: Commit-time authorization for LLM agents, 2026. Preprint, arXiv:2607.10487v1, version 1, 11 July 2026. https: //arxiv.org/abs/2607.10487v1.

[15] Zheng Chen, Hanqing Liu, Duling Xu, Dong Dong, Jialin Li, Bangzheng Pu, and Jidong Zhai. Cordon: Semantic transactions for tool-using LLM agents, 2026. Preprint, arXiv:2606.17573v1, version 1, 16 June 2026. https://arxiv.org/abs/2606. 17573v1.

[16] Luc Moreau and Paolo Missier. PROV-DM: The PROV data model. W3C Recommendation, 2013.

[17] W3C JSON-LD Working Group. JSON-LD 1.1: A JSON-based serialization for linked data. W3C Recommendation, 2020. 16 July 2020.

[18] Dave Longley, Gregg Kellogg, and Dan Yamamoto. RDF dataset canonicalization. W3C Recommendation, 2024. 21 May 2024, Sections 4.4.3 and 7.1.

[19] W3C Verifiable Credentials Working Group. Verifiable credential data integrity 1.0: Securing the integrity of verifiable credential data. W3C Recommendation, 2025. 15 May 2025.

[20] W3C JSON-LD Working Group. JSON-LD 1.1 processing algorithms and API. W3C Recommendation, 2020. 16 July 2020

## A Normative Receipt Semantics

## A.1 Stage Sequences and Disposition Mapping

This appendix fixes the stage order and minimum contents left implicit in the main-text receipt abstractions. Preparation and activation checks evaluate two strictly ordered, closed sequences: Preparation Stage Order (PrepStage):

$$
\mathsf { R a w S c r e e n \prec D e c o d e \prec C a n o n i c a l i z e }
$$

$$
\prec P r o p o s a l A u t h \prec \mathsf { S t a t i c V a l i d a t i o n }
$$

$$
\prec \mathsf { T a r g e t L o o k u p \prec A c q u i s i t i o n A u t h }
$$

$$
\prec \mathsf { R e q u i r e m e n t C o v e r a g e \prec W i t n e s s A u t h }
$$

$$
\prec A c q u i s i t i o n \mathsf { T i m e } \prec C a \mathsf { n d i d a t e D e r i v a t i o n }
$$

$$
\prec P 0 \vert \mathsf { i c y A u t h o r i z a t i o n \prec R e a d y . }
$$

## Activation Check Order (ActCheck):

$$
\mathsf  I a r g e t L o o k u p \prec A l l o c a t i o n \prec E x p e c t e d H e a d
$$

$$
\prec \mathsf { L i f e c y c l e \prec H a n d o f f R e v i s i o n }
$$

$$
\prec A c q u i s i t i o n A u t h \prec \mathsf { R e q u i r e m e n t C o v e r a g e }
$$

$$
\prec \mathsf { V e r s i o n B i n d i n g \prec W i t n e s s B i n d i n g }
$$

$$
\prec \mathsf { C a n d i d a t e B i n d i n g \prec E H e c t B i n d i n g }
$$

$$
\preccurlyeq { \mathsf { F r e s h n e s s l n i t i a l } } \preccurlyeq { \mathsf { A u t h o r i z a t i o n } }
$$

$$
\prec \mathsf { D e c i s i o n } \prec \mathsf { A u t h o r i t y T r a n s i t i o n }
$$

$$
\prec \mathsf { F r e s h n e s s F i n a l \prec W e l l F o r m e d n e s s . }
$$

In both pipelines, TargetLookup requires existence for existingbranch kinds and absence plus serialized $\xi _ { k }$ for BranchCreate. The latter’s preparation derives from genesis inputs and runs PolicyAuthorization against $\xi _ { k } ;$ activation ExpectedHead accepts only Absent. Existing kinds use the current predecessor $X , \Gamma$ and Presen $: ( d _ { h } )$ .

An outcome-eligible failure at position j records

$$
v _ { i } = \left\{ \begin{array} { l l } { \mathsf { P a s s } , } & { i < j , } \\ { \mathsf { F a i l } ( r ) \mathrm { o r } \mathsf { M i s s i n g } ( r ) , } & { i = j , } \\ { \mathsf { N E } , } & { i > j , } \end{array} \right.\tag{12}
$$

where NE means not evaluated. A verifier never asks a failing predicate to pass. Known permanent failures precede a trusted Missing, so unavailability cannot mask a rejection. Malformed outer framing, failed outer authentication, and invalid allocation are protocol errors: they create neither $O [ p k e y ]$ nor a receipt.

For $T ~ \in ~ \{ P , A , Q , C \}$ , rb<sub>T</sub> canonically encodes and signs exactly r<sub>T</sub>, computes $d _ { r _ { T } }$ , and uses the sole receipt-store rule $\mathcal { R } [ d _ { r _ { T } } ] = r _ { T }$ . No other proposal-receipt builder or receipt key exists. Table 10 is the closed stage–reason map.

At activation TargetLookup, BranchCreate evaluates target presence before creation context: $B [ k ] \neq \mathsf { A }$ bsent yields TargetAlreadyExistsAtActivation; $B [ k ] = { \mathsf { A b s e n t } } \wedge { \Xi } [ k ]$ missing yields CreationContextMissingAtActivation. Freshness checks evaluate in deterministic intra-stage order. For existing branches: $( 1 ) t >$ deadline yields ExpiredAtActivation; (2) head mismatch yields StaleHead; (3) epoch mismatch yields StaleEpoch; (4) dirSeq mismatch yields StaleDirSeq; (5) authority/revocation/policy mismatch yields StaleAuthority; (6) pinned version mismatch yields

StaleVersion. For BranchCreate: (1) t > deadline yields ExpiredAtActivation; (2) post-lookup target appearance $( B [ k ] \neq \mathsf { A }$ bsent) yields TargetAppearedAtActivation; (3) creation context $( \Xi [ k ]$ missing/altered) yields StaleCreationContext; (4) genesis profile/allocator version mismatch yields StaleVersion. Every reason selects Reject $( r b _ { A } )$ , requires preparation and prefix $\Pi _ { j - 1 }$ , forbids later fields, and leaves $X , B , \Gamma , \Xi , E$ unchanged. Earlier stages dominate later stages.

At preparation, a Fail selects Reject/rb<sub>P</sub> and only Evidence-Unavailable is Missing, selecting Defer/rb<sub>P</sub>. At activation, every reason selects $\mathbf { R e j e c t } / r b _ { A }$ except PolicyQuarantine, which selects Quarantine/rb<sub>Q</sub>; all-pass through WellFormedness selects Commit/rb<sub>C</sub>. Thus Algorithm 1 has no other terminal path. Adding a stage or reason requires a new receipt schema version. An exact retry returns the indexed result; any changed attempt under the same pid conflicts.

Lifecycle maintenance is typed separately. Its closed actions are Prepare, Fence, Retarget, and Abort. Its closed results are Applied, StaleHead, StaleRevision, StaleDirSeq, WriterEpoch, TargetConflict, InvalidState, and Unauthorized. $r b _ { L } =$ Build(LifecycleReceiptV1) binds the action identifier, pre/post directory tuples, optional input/output revision digests, result, trusted time, and signature. It is stored in R and indexed by the typed $H _ { R }$ key defined in Appendix C; it never creates $O [ p k e y ]$ and is not a fifth proposal disposition. Its digest is $d _ { r _ { L } } = d _ { \mathsf { L i f e c y c l e R e c e i p t V 1 } } ( r _ { L } )$ under Equation 14. Replay recomputes the action and its pre/post tuples; inclusion requires a certified state containing $\mathcal { R } , H _ { R }$ , and, for Applied, the assigned F and B records.

Lifecycle field presence is action-indexed. Every $r _ { L }$ binds its common fields, authenticated request, requested pre-directory tuple, result, and exact post tuple. For Applied, the input/output revision states are Prepare $\perp $ Prepared, Fence Prepared→SourceFenced, Retarget SourceFenced→SourceFenced, and Abort Prepared→Aborted. For an allowed non-Applied result, post equals pre, output is forbidden, and neither $F$ nor B changes; Prepare forbids input, while the other actions require the requested input. Let

$$
\begin{array} { r l } & { S = \{ \mathsf { S t a l e R e v i s i o n } , \mathsf { S t a l e D i r S e q } , \mathsf { W r i t e r E p o c h } , } \\ & { \qquad | \mathsf { n v a l i d S t a t e } , \mathsf { U n a u t h o r i z e d } \} . } \end{array}
$$

The exact failure sets are $\textit { S } \backslash$ {StaleRevision} ∪ {StaleHead, TargetConflict} for Prepare, S ∪ {StaleHead} for Fence, S ∪ {TargetConflict} for Retarget, and S for Abort. No other action/result pair is schema-valid.

## A.2 Field presence and early input binding

After owner authentication, the gateway computes

$$
d _ { r a w } = d _ { \mathsf { W i r e l n p u t V 1 } } ( r a w B y t e s ) .\tag{13}
$$

An early RawScreen, Decode, or Canonicalize receipt binds $d _ { r a w }$ but omits $d _ { \tau } ;$ there is not yet a canonical proposal to hash. Let j be the terminal stage. A receipt requires the common fields, the terminal Fail/Missing observation, and exactly the fields whose producing stages successfully completed before j. It forbids fields first produced at $j$ or later. Thus later NE stages impose no field obligation. The only profile choice is raw retention: $d _ { r a w }$ is required when rawRetention=true and forbidden otherwise. Absence is encoded by the typed variant, never a zero digest.

Table 9: The four proposal-receipt variants and their terminal semantics.
<table><tr><td>Tag</td><td>Terminal location</td><td>Stage/reason rule</td><td>Builder</td><td>Durable effect</td></tr><tr><td> $r _ { P }$ </td><td>Preparation</td><td>First failed or trusted-Missing preparation stage; later stages are NE</td><td> $r b _ { P }$ </td><td>Insert one owner-bound O[pkey] and  $\mathcal { R } [ d _ { r _ { P } } ] = r _ { P } ;$  no candidate becomes authoritative.</td></tr><tr><td>rA</td><td>Serialized activation</td><td>First failed activation check; earlier checks pass and later checks are NE</td><td> $r b _ { A }$ </td><td>Insert one O[pkey] and  $\mathcal { R } [ d _ { r _ { A } } ] = r _ { A } ; X , B , \Gamma ,$  E are unchanged.</td></tr><tr><td>rQ</td><td>Decision = Quarantine</td><td>Earlier checks pass, Decision records Quarantine, later checks are NE</td><td>rbQ</td><td>Insert  $O [ p k e y ] , \mathcal { R } [ d _ { r _ { Q } } ] = r _ { Q }$  , and unreachable Q[pkey] only.</td></tr><tr><td>rC</td><td>All activation checks Pass</td><td> $P _ { \mathcal { X } }$  passes WellFormedness; the terminal result is Commit</td><td>rbc</td><td>Store  $\mathcal { R } [ d _ { r _ { C } } ] = r _ { C }$  and install the complete accepted unit of Equation 9.</td></tr></table>

Table 10: Closed V1 stage–reason map. “Limit” abbreviates CanonicalizationLimit; a stage vector disambiguates repeated reasons.
<table><tr><td>Preparation stage → allowed reason Activation check → allowed reason</td></tr><tr><td>RawScreen→ {Limit}; Decode→ {DecodeFailure, Limit}; TargetLookup→{TargetMissingAtActivation, TargetAlreadyExistsAtActivation, Canonicalize→ {CanonicalizationFailure, Limit}; CreationContextMissingAtActivation}; Allocation→{AllocationRevoked}; ProposalAuth→ {ProposalAuthentication}; ExpectedHead→ {StaleHead}; Lifecycle→ {OrdinaryLifecycle, StaticValidation→{MalformedProposal, EffectManifest, HandoffLifecycle}; HandoffRevision→ {StaleHandoffRevision, StaleDirSeq, DuplicateEffectInProposal}: TargetLookup→{UnknownTargetAtPreparation TargetWriter, StaleEpoch}; AcquisitionAuth→{AcquisitionAuthentication}; CreationContextMissing, BranchAlreadyExists}; RequirementCoverage→{RequirementCoverage}; AcquisitionAuth→ {UntrustedRequirementResult, AcquisitionAuthentication}; VersionBinding→ {StaleVersion}; WitnessBinding→ {WitnessBinding}; RequirementCoveràge→ {RequirementCoverage}; CandidateBinding→ {CandidateBinding}; EffectBinding→ {EffectManifest, WitnessAuth {WitnessAuthentication}; EffectScope, RetiredEffect, DuplicateEffect}; FreshnessInitial, AcquisitionTime→ {ExpiredAtPreparation, EvidenceUnavailable}; FreshnessFinal→{ExpiredAtActivation, StaleHead, StaleEpoch, StaleDirSeq, CandidateDerivation→ {CandidateDerivation}; StaleAuthority, StaleVersion, StaleCreationContext,</td></tr></table>

Table 11: Stage-indexed field production. A field is required exactly after its producer passes and forbidden beforehand.
<table><tr><td>Field group</td><td>Producer that must Pass</td><td>Contents</td></tr><tr><td>Common</td><td>Outcome-eligible owner authentication</td><td>Type/version, subject/branch, principal, allocation scope, pid, tag, disposition, stage vector, reason, issue time, kernel key and signature</td></tr><tr><td>Raw input</td><td>Request capture under the signed</td><td> $d _ { r a w } ;$  profile makes it required or forbidden, never optional</td></tr><tr><td>Canonical proposal</td><td>profile Canonicalize</td><td>dτ, typed Present(dh)/Absent expectation, kind reference, and unverified signature/credential bytes</td></tr><tr><td>Authenticated proposal</td><td>ProposalAuth</td><td>Verified signer, owner, principal, allocation scope, and proposal-authentication result</td></tr><tr><td>Acquisition</td><td>RequirementCoverage (preparation)</td><td>Ordered requirements/results, witness and dependency-version commitments</td></tr><tr><td>Candidate package</td><td>CandidateDerivation</td><td>Candidate root, pinned interpreter, candidate-seal key/signature, effect manifest, and direct applicable  $d _ { \xi }$  open SourceFenced-revision digest</td></tr><tr><td>Activation prefix II</td><td>Corresponding ActCheck stage</td><td>Existing row or absence plus  $d _ { \xi } ;$  allocation; exact Present/Absent expectation; lifecycle; open revision; acquisition signer; exact cover; versions; witness binding; candidate binding; effect binding; initial freshness; authorization; Decision; authority-after; final freshness—each introduced only by its</td></tr><tr><td>Prospective unit</td><td>BuildProspective after FreshnessFinal</td><td>same-named successful stage Produced successor/core/lineage inputs, receipt-independent branch-row plan  $( \widehat { B } _ { \chi } ) .$  receipt-independent</td></tr><tr><td>Finalized Commit</td><td>WellFormedness Pass</td><td>assignment plan  $( \widehat { \Delta } _ { x } )$  , and deterministic parameters for receipt-dependent finalization Commit receipt/digest, complete head, outcome,  $H _ { R } ,$  effects, receipt store, and directory binding</td></tr></table>

Consequently, a CandidateDerivation failure forbids candidate fields. Every activation receipt requires the completed preparation package, but an early TargetLookup or ExpectedHead failure contains only its successful activation prefix and terminal observation— never later freshness, authorization, Decision, authority-after, or successor fields. A Decision Reject or Quarantine binds that terminal policy result; it does not manufacture an AuthorityTransition prefix. A BranchCreate prefix binds absence and $d _ { \xi } ,$ never a predecessor head, state, or authority. An InvalidSuccessor receipt may bind the produced prospective fields but forbids every finalized Commit field.

Exact evidence coverage is a bijection between the signed ordered requirement sequence and acquired results: no omission, duplicate, substitution, injection, or forbidden reordering is accepted. Each result is either a signed witness or an authenticated Missing. A candidate seal authenticates these bindings; it is neither pre-state authorization nor proof of durable commit.

## A.3 Verification levels

Verification is monotone but not automatic:

1. Structural authenticity checks the receipt variant, canonical syntax, typed digests, signature, key scope, and internal bindings.

2. Kernel-attested reason additionally establishes that the authenticated kernel reported the declared first terminal stage. It does not independently establish that the predicate was evaluated correctly.

3. Independent replay recomputes each predicate through the terminal stage from retained inputs and pinned versions. For Commit it also recomputes candidate derivation, the authority transition, manifest, successor core, and lineage. Replay is unavailable when any required object or interpreter is absent.

4. Durable inclusion verifies a certified snapshot, commit manifest, or replicated-log proof containing $O [ p k e y ]$ and the receipt; Commit also requires the installed head and lineage. An archive proof terminates at a retirement root in such a certified state. A signature or locator alone is not inclusion.

A conforming verifier first completes level 1, reports level 2 only for a trusted kernel key, attempts level 3 only with a complete replay package, and reports level 4 only with an inclusion proof. Thus a signed receipt from an aborted storage transaction cannot be upgraded to a committed fact.

## B Typed Lineage and Cycle-Free Construction

## B.1 Acyclic Hash Graph and Construction Order

Here $\begin{array} { r c l } { { K _ { H } } } & { { = } } & { { \mathsf { H a n d o f f R e s u l t l e y V 1 } ( p k e y , h i d ) } } \end{array}$ and $\begin{array} { r l } { K G } & { { } = } \end{array}$ DirectoryK $\mathsf { e y V 1 } ( k , d i r S e q ^ { \prime } )$ ; these are definitions, not aliases for differently keyed records.

Every content reference uses a type- and version-separated digest

$$
\begin{array} { r } { d _ { T } ( y ) = H \big ( \mathrm { L P } ( \mathbb { C } \mathbb { K } ) \parallel \mathrm { L P } ( T ) \parallel \mathstrut } \\ { \mathrm { u } 3 2 ( v ) \parallel \mathrm { L P } ( \mathrm { C a n o n } _ { T } ( y ) ) \big ) , } \end{array}\tag{14}
$$

where LP is an unambiguous length prefix. Logical proposal and effect identifiers are allocated names, not content digests.

Equation 1 is the sole definition of immutable head core, complete head, and branch row; this appendix introduces no shortened head. The closed parent type is

$$
\begin{array} { r } { \mathsf { L i n e a g e P a r e n t } = \mathsf { A c c e p t e d P a r e n t } ( d _ { h } ) } \\ { \mid \mathsf { G e n e s i s P a r e n t } ( d _ { g } ) . } \end{array}\tag{15}
$$

Exactly one branch-genesis edge uses the second variant. Every later edge uses the complete current head. A checkpoint, migration input, or handoff source is an auxiliary typed reference and can never occupy the parent field.

A LineageEdgeV1 binds sid, bid, transition kind, lineage parent, $d _ { h _ { c } } , d _ { \tau }$ , the proposal allocation key, policy and authority versions, epoch and sequence, plus the transition-kind-specific source, checkpoint, migration, or handoff references. A BranchGenesisV1 binds the authenticated creation request, absent-directory proof, initial state and admission roots, writer, epoch 0, sequence 0, and optional source provenance. Source provenance does not create cross-branch ancestry. In Table 12, every V1 name has version 1, that exact name as its domain tag, and its type-specific canonical encoder under Equation 14. Component roots carry their declared type/schema domain. Rows 3–10 and 12 are content-addressed by their displayed typed digest; rows 1–2, 11, and 13–17 use exactly the displayed map key. Map keys are allocated typed names, not hidden hash inputs.

For HandoffActivate, TransitionV1 directly binds $d _ { f _ { i } ^ { S o u r c e F e n c e d } } ;$ AcquisitionV1 binds it transitively through $d _ { \tau } ,$ , and the seal binds it directly. The revision must be the current open SourceFenced revision or HandoffRevision fails stale. $h _ { c } . a u x R e f$ also names this input but never the output or seal. The head core has no direct seal dependency: $h ^ { \prime }$ binds $d _ { s e a l }$ transitively through $d _ { r _ { C } }$ . The same rule applies to BranchCreate with $d _ { \xi } \colon$ direct in the transition and seal, transitive through d<sub>τ</sub> in the acquisition, and revalidated from Ξ[k] at both freshness checks. The output revision binds the open input and $d _ { h _ { c } } ;$ successful activation then clears the pointer while retaining both revisions in F. The order is

$$
\begin{array} { r l } {  { \big ( d _ { \xi } ? , d _ { f _ { j } ^ { S o u r c e F e n c e d } } ? \big ) \prec d _ { \tau } \prec d _ { W } \prec ( X ^ { \prime } , \Gamma ^ { \prime } , M _ { E } ) } } \\ & { \prec d _ { g } ? \prec d _ { s e a l } \prec h _ { c } \prec d _ { f _ { j + 1 } ^ { T a r g e t A c t i v e } } ? } \\ & { \prec \ell \prec r _ { C } \prec h ^ { \prime } \prec d _ { h ^ { \prime } } } \\ & { \prec ( O , E , H _ { R } ? , B ^ { \prime } , G ^ { \prime } ) \prec \mathsf { P u b l i s h } \prec i . } \end{array}\tag{16}
$$

The receipt binds the prospective unit ${ \widehat { P } } _ { \chi } ,$ , not the complete head; the complete head $h ^ { \prime }$ then binds $d _ { r _ { C } } ,$ , producing $d _ { h ^ { \prime } }$ . The lineage edge likewise binds $h _ { c }$ and typed predecessor, not the complete successor. Therefore every digest edge points left in Equation 16, and the audited graph is acyclic. Outcome, effect, handoff-result, branch, and directory records point only to named earlier digests; no earlier object points back. Inclusion evidence is constructed only after publication. Atomic publication changes visibility, not this hash order.

## B.2 Proof sketches

Proposition 1. Every outcome-writing path first authenticates the proposal allocation and revalidates it in the proposal-scope serialization domain. Invalid outer requests never write an outcome. The first authorized write uniquely inserts $O [ p k e y ] ;$ ; a compatible retry returns it, an incompatible retry conflicts, and a reclaimed identifier is rejected by the monotone watermark. A second durable disposition is therefore impossible under A1–A4 and A8. □

Proposition 2. The seal binds the proposal, evidence, interpreter, typed Present/Absent expectation, candidate root, and effect manifest. Activation serializes the branch key and compares either the complete head or continued absence plus $d _ { \xi }$ . The first Commit changes that serialization point atomically; every contender then fails exact-head or absence admission. Thus at most one sealed candidate becomes the exact successor or genesis under A1–A4. □

Proposition 3. The operation–manifest bijection rejects an effect identifier repeated within a proposal. Across proposals, Commit uniquely inserts $E [ e i d ]$ while holding its scope. Reclamation advances the scope watermark before deleting online rows, and scopes are never recycled. Hence deletion cannot reopen an accepted identifier under $\mathbf { A } 1 , \mathbf { A } 4 ,$ and A8. □

These are conditional serialization arguments. They establish neither content truth nor correctness of policy, canonicalization, cryptography, or storage implementations.

Table 12: Exact typed construction dependencies. Every input appears earlier.
<table><tr><td></td><td>Object and domain</td><td>Immediate inputs</td><td>Downstream references</td></tr><tr><td>1</td><td> $d _ { \xi } / { \mathsf { C r e a t i o n C o n t e x t V 1 } } ;$   $\boldsymbol { \Xi } [ \boldsymbol { k } ]$ </td><td>Namespace, allocator and policy versions, dirSeq, authority root, genesis profile</td><td>BranchCreate proposal, seal, authorization, freshness</td></tr><tr><td>2</td><td> $d _ { f _ { j } ^ { S o u r c e F e n c e d } } ^ { \cdot } \prime$  existing HandoffRevisionV1;</td><td>Prior digest, SourceFenced state, branch/head, target, epochs, directory tuple, context versions</td><td>HandoffActivate proposal, seal, admission</td></tr><tr><td>3</td><td> $F [ h i \bar { d } , j ]$  dτ/TransitionV1</td><td>pkey, k, Present/Absent expectation, χ, operations, authority changes, dependencies, requirements, validity, kind reference, and direct</td><td>Acquisition and candidate derivation</td></tr><tr><td>4</td><td> $d _ { W } / \mathsf { A c q u i s i t i o n V } 1$ </td><td>applicable  $d _ { \xi } \ \mathrm { o r } \ \bar { d } _ { f _ { i } ^ { S } }$  ourceFenced  $d _ { \tau } ,$  ordered requirements/results, signer/key identifiers and pinned versions; detached authentication is verified</td><td>Candidate seal</td></tr><tr><td>5</td><td> $d _ { X ^ { \prime } } , d _ { \Gamma ^ { \prime } } , d _ { M _ { E } } $   $\mathsf { S t a t e R o o t V 1 , }$   $\mathsf { A u t h o r i t y S t a t e V 1 , }$ </td><td>Ordered typed successor component digests; ordered authority entries; ordered (opIndex, eid, opDigest), derived from initial inputs with  $d _ { \tau } , d _ { W }$  and the interpreter</td><td>Seal, head core</td></tr><tr><td>6</td><td> $E H _ { e c t M a n i t e s t V 1 }$   $g , d _ { g } / \mathsf { B r a n c h G e n e s i s V 1 }$ </td><td> $d _ { \tau } , d _ { \xi }$  , absence proof, initial roots/writer/zero counters, optional source</td><td>Genesis parent</td></tr><tr><td>7</td><td> $_ { d _ { s e a l } / \mathsf { C a n d i d a t e S e a l V 1 } }$ </td><td>provenance  $\bar { \boldsymbol { d } } _ { \mathcal { T } } , \boldsymbol { d } _ { \boldsymbol { W } }$  , interpreter, component/effect roots, and direct applicable  $d _ { \xi }$ </td><td>Receipt</td></tr><tr><td>8</td><td> $h _ { c } , d _ { h _ { c } } / { \mathsf { H e a d C o r e V } } 1$ </td><td>or open-revision digest Exact Equation 1 fields: root, typed parent, versions, counters, time,</td><td>Output revision, lineage</td></tr><tr><td>9</td><td> $d _ { f _ { j + 1 } ^ { T a r g e t A c t i v e } } / \mathrm { n e w }$ </td><td>kind, and closed aux Re f dfSourceFenced  $, d _ { h _ { c } }$  , exact target, epochs, directory tuple and Ji</td><td>Lineage, receipt, HResult</td></tr><tr><td>10</td><td> $\mathsf { H a n d o f f R e v i s i o n V 1 }$   $\ell , d _ { \ell } / \lfloor \mathsf { i n e a g e E d g e V 1 } ;$ </td><td>context versions Parent,  $d _ { h _ { c } } , d _ { \tau } , p k e y ,$  versions, and applicable input/output handoff</td><td>Commit receipt</td></tr><tr><td>11</td><td> $L [ \lambda ]$   $r _ { C } , d _ { r _ { C } } /$ </td><td>digests pkey,  $d _ { \tau } , d _ { W } , d _ { s e a l } .$  parent,  $d _ { h _ { c } } , d _ { \ell }$  , all-Pass vector,  ${ \widehat { P } } _ { \chi }$ </td><td>Complete head, accepted records</td></tr><tr><td>12</td><td> $\mathtt { C o m m i t R e c e i p t V 1 } ; \mathcal { R } [ d _ { r _ { C } } ]$ </td><td>prospective fields, effect/authority and handoff bindings</td><td>Outcome, branch row, effect, handoff result, directory binding</td></tr><tr><td>13</td><td> $h ^ { \prime } , d _ { h ^ { \prime } } / \mathsf { C o m p l e t e H e a d V } 1$ </td><td> $h _ { c } , d _ { \ell } , d _ { r _ { C } }$  pkey, Commit tag,  $d _ { r _ { C } } , h ^ { \prime }$ </td><td>Retry, inclusion</td></tr><tr><td>14</td><td> $\mathsf { O u t c o m e V 1 ; } \mathsf { O } [ p k e y ]$  EffectRecordV1; E[eid]</td><td>eid, pkey, operation index/digest,  $d _ { r _ { C } } , d _ { h ^ { \prime } }$ </td><td>Inclusion, duplicate exclusion</td></tr><tr><td>15</td><td>HandoffResultV1;</td><td>pkey, hid, input/output revision digests,  $d _ { r _ { C } } ^ { } , h ^ { \prime }$ </td><td>Retry, inclusion</td></tr><tr><td>16</td><td> $H _ { R } [ K _ { H } ]$  BranchRowV1; B[k]</td><td> $k , h ^ { \prime } ,$  status, writer, epoch, dirSeq, open pointer, immutable</td><td>Authoritative lookup</td></tr><tr><td>17</td><td>DirectoryBindingV1; G[KG]</td><td>createdFrom k, pkey, χ, pre/post dirSeq, allocator/context versions,  $d _ { r _ { C } } , d _ { h ^ { \prime } }$ </td><td>Freshness, inclusion</td></tr><tr><td>18</td><td>i / InclusionProofV1, later</td><td>Certified state containing rows 11–17</td><td>External verifier; never an input to</td></tr></table>

## C Total Lifecycle, Handoff, and Restoration Semantics

## C.1 Handoff Union Types and Kind-Specific Admission

Equation 1 is normative for B. In particular, dirSeq lives only in B; a revision records dirSeqIn, dirSeqOut as snapshots. createdFrom ∈ {None, SourceProvenance $( d _ { h } ) ]$ } is assigned at genesis and copied byte-for-byte thereafter. Pointer activity and revision state are distinct closed types:

$$
\begin{array} { r l } & { \mathsf { H P t r } = \perp \mid \mathsf { O p e n } ( h i d , d _ { f } ) , } \\ & { \mathsf { F S t a t e } = \mathsf { P r e p a r e d } \mid \mathsf { S o u r c e F e n c e d } \mid } \\ & { \qquad \mathsf { A b o r t e d } \mid \mathsf { T a r g e t A c t i v e } . } \end{array}\tag{17}
$$

An ordinary Commit during Prepared is nonconflicting but makes that revision’s source-head binding stale. Fence then returns Stale-Head; Abort followed by a fresh Prepare is required before fencing. Any change to target, epoch, dirSeq, input revision, $d _ { G } ,$ , or the current admission versions similarly invalidates an older target package.

## C.2 Authoritative State Postconditions

Let $\boldsymbol { b } = \langle h , s , w , e , d , z , c \rangle$ follow the field order of Equation 1; $b [ a ~  ~ x ]$ copies b and explicitly replaces field a. The HResult map is $H _ { R }$ . Let ${ \cal K } _ { L } ~ = ~ \mathsf { L i f e c y c l e K e y V 1 } ( h i d , l i d )$ and $K _ { H } =$ HandofResultKeyV1(pkey, hid) be disjoint typed keys; lid and pid are owner-allocated, non-recycled names. A unique insert makes each $H _ { R }$ key immutable: an exact retry returns the stored value and a variant conflicts. “Append $r _ { L } \ "$ below means $\mathcal { R } [ d _ { r _ { L } } ] ~ = ~ r _ { L }$ in the same atomic unit. For compactness, let $s = {  { \left. X , \Gamma \right. } }$ , rev, issuer, policy, $\Xi , h , L , O , E , G \rangle ; \ : { \cal S } ^ { \prime } = { \cal S } \ : \mathrm { e x } .$ plicitly copies every listed family, including $\Xi ^ { \prime } = \Xi$

Every HandoffRevisionV1 binds its predecessor revision, action/state, branch key, source head/writer, target writer, old/new and reserved epochs, di ${ \cdot } S e q I n , d i r S e q O u t .$ , pinned policy/authority/revocation versions, digests of Γ, G, optional $d _ { h _ { c } }$ , reason, and trusted time. $F [ h i d , j ]$ is append-only. Maintenance changes only the fields and indexes assigned in Table 14; it never creates an $O$ entry.

The consumed target package is

$$
\begin{array} { r l } & { P _ { H } = \langle p k e y , d _ { \tau } , d _ { W } , d _ { s e a l } , h i d , d _ { f _ { j } ^ { S o u r c e F e n c e d } } , d _ { h s } , } \\ & { \qquad w _ { t } , e _ { c } , e _ { r } , d i r S e q , p o l i c y V , a u t h V , r e v V , d _ { \Gamma } , d _ { G } \rangle , } \end{array}\tag{18}
$$

where $e _ { c }$ is the pre-activation canonical epoch and $e _ { r }$ is the separately reserved successor epoch. Admission requires both to remain exact and equal. The accepted unit creates the distinct output $f _ { i + 1 } ^ { T a r g e t A c t i v e } ,$ ; Appendix B fixes every digest placement. Precisely, $\Breve { \Gamma _ { T } } = \Gamma _ { c u r r e n }$ <sub>t</sub> when no separately authorized authority transition exists, and $\Gamma _ { T } = \Gamma ^ { \prime }$ only when the same accepted proposal authorizes and binds $\Gamma ^ { \prime }$ . The identical rule applies to revocation, issuer, and policy; the fenced historical head is never their source.

## C.3 Restoration Path Mask and Protected Semantics

Restoration binds

$$
\begin{array} { c } { { a u x _ { R } = \langle d _ { h _ { c } } , d _ { r o o t _ { c } } , d _ { h _ { k } } , d _ { r o o t _ { k } } , } } \\ { { p r o o f _ { k } , d _ { \mu } , M , c o m p o n e n t R o o t s \rangle } } \end{array}\tag{19}
$$

Table 13: Transition-kind-specific head admission within the single ordered ActCheck vector.
<table><tr><td>Kind</td><td>Serialized  $G _ { h e a d } ^ { k i n d }$  precondition</td><td>Parent and disposition</td></tr><tr><td>Ordinary, Migration, Restoration HandoffActivate</td><td> $B = \langle h ,$  Active, w, e, d, z, c); exact expected = h, writer w, epoch e, and  $d i r S e q = d ;$  an Open z must reference Prepared, never SourceFenced.  $B = \langle \hat { h } _ { s } , \mathsf { F r o z e n } , \overset { \cdot } { \bot } , e _ { c } , d , \mathsf { O p e n } ( h i d , \dot { d } _ { f } S$  ourceFenced  $) , c \rangle ;$  exact</td><td> $\mathsf { A c c e p t e d P a r e n t } ( d _ { h } ) ;$  one of the four proposal dispositions. AcceptedParent  $( d _ { h _ { s } } ) ;$  an ordinary Commit/Reject/Quarantine/Defer outcome</td></tr><tr><td></td><td>j fenced head, target writer,  $d i r S e q = d ,$  input revision and current context; the</td><td>under  $O [ p k e y ] .$ </td></tr><tr><td></td><td>revision separately binds reserved Epoch = er, and admission requires the</td><td></td></tr><tr><td></td><td>proposal to bind both  $e _ { c } , e _ { r }$  with  $\boldsymbol { e } _ { c } \overset { \bar { } } { = } e _ { r } .$ </td><td></td></tr><tr><td>BranchCreate</td><td>B[k] absent, expected = Absent, and exact serialized  $\Xi [ k ] = \xi _ { k } ;$ </td><td>GenesisParent  $( d _ { g } ) ;$  ordinary Commit with kind BranchCreate</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>authenticated allocation/absence proof and zero initial counters.</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

Table 14: Total lifecycle postconditions. Every authoritative family is assigned explicitly or through the complete accepted unit; $\Xi ^ { \prime } = \Xi$ holds for all rows.

Action Exact precondition Complete authoritative poststate Receipt/result records   
Genesis B[k] absent; BranchCreate row Install $\mathcal { U } _ { a c c e p t } ($ (BranchCreate): O[pkey] = OutcomeV1(Commit, $d _ { r _ { C } } , h _ { 0 } )$ and   
of Table 13 $X _ { 0 } , \Gamma _ { 0 } , g , \dot { h _ { 0 } } , L , E [ M _ { E } ] , G _ { \chi } ^ { \prime }$ and $\mathcal { R } [ d _ { r _ { C } } ] = r _ { C } .$   
$B ^ { \prime } [ k ] = \langle h _ { 0 } , \mathsf { A c t i v e } , w _ { t } , 0 , 0 , \bot , c _ { 0 } \rangle ; \Xi ^ { \prime } = \Xi ; F , H _ { R }$ not   
written.   
Prepare b.status = Active, $\begin{array} { r } { S ^ { \prime } = S ; B ^ { \prime } = b [ d i r S e q  d + 1 , } \end{array}$ hando $f f \gets$ Append $r _ { \bot } = r b _ { L } ( \mathsf { A p p l i e d } )$ and   
b.writer = w , exact Open(hid, d <sub>Prepared</sub> )], preserving createdF rom; $\begin{array} { r } { \overrightarrow { H _ { R } } [ \overrightarrow { K _ { L } } ] = \mathsf { L i f e c y c l e } ( d _ { r _ { L } } , \bot , d _ { f _ { j + 1 } ^ { P r e p a r e d } } ) . } \end{array}$   
head/epoch/dirSeq, and $f _ { j + 1 } ^ { * }$   
b.handof $: f = \perp$ append $f _ { j + 1 } ^ { P r e p a r e d }$ to F and reserve $e _ { r } = e + 1 . $   
Abort Exact $f _ { j } ^ { P r e p a r e d } ;$ Active, $S ^ { \prime } = \bar { S ; B ^ { \prime } } = b [ d i r { S e q }  d + 1 ,$ hando $f f \gets \bot ] ,$ Append $r _ { L }$ and   
writer w<sub>s</sub>, epoch e, preserving createdFrom; append the Aborted revision to F and $\begin{array} { r } { \dot { \bar { H _ { R } } } [ K _ { L } ] ^ { \sim } = \mathsf { L i f e c y c l e } ( d _ { r _ { L } } , d _ { f _ { j } ^ { P r e p a r e d } } , d _ { f _ { j + 1 } ^ { A b o r t e d } } ) . } \end{array}$   
$d i r S e q = \mathsf { \bar { d } } ,$ , and matching cancel $e _ { r } .$   
Open pointer   
Fence Exact $\bar { \boldsymbol { f } } _ { j } ^ { P r e p a r e d } ,$ its final h, $s ^ { \prime } = s { \mathrm { ; } }$ ; preserve createdFrom and set Append $r _ { L }$ and $H _ { R } [ K _ { L } ] =$   
and active source tuple $B ^ { \prime } . s t a t u s =$ Frozen, writer $: = \bot , e p o c h = e _ { r } ,$ $\mathsf { L i f e c y c l e } ( \breve { d } _ { r _ { L } } , d _ { f _ { j } ^ { P r e p a r e d } } , d _ { f _ { j + 1 } ^ { S o u r c e F e n c e d } } ) .$   
$d i r S e q = d + 1 , \mathrm { a n d }$   
handoff = Open(hid, d ); append   
j+   
SourceFenced revision with old epoch e and new/reserved epoch e<sub>r</sub>.   
Retarget Exact $f _ { j } ^ { S o u r c e F e n c e d } ;$ $s ^ { \prime } = s { \mathrm { ; } }$ preserve status, writer, head, and createdF rom; set Append $r _ { L }$ and $H _ { R } [ K _ { L } ] =$   
Froze $\mathsf { 1 , w r i t e r \perp , e p o c h } e ,$ epoch $+ 1 , d i r S e q = d + 1$ 1, and $\mathsf { L i f e c y c l e } \big ( \overset { \sim } { d } _ { r _ { L } } , \overset { \mathtt { \tiny ~ . ~ . ~ . ~ . ~ . ~ . ~ . ~ . ~ . ~ . ~ . ~ } } { d } _ { f _ { j } ^ { S o u r c e F e n c e d } } , \overset { \mathtt { \tiny ~ . ~ . ~ . ~ . ~ } } { d } _ { f _ { j + 1 } ^ { S o u r c e F e n c e d } } \big ) .$   
$\operatorname* { d } i r S e q = d ,$ and matching hando $f f = 0 \mathsf { p e n } ( \mathsf { \dot { h } } i d , d _ { f _ { \therefore \times } ^ { S o u r c e F e n c e d } } ) ;$ append   
Open pointer j+1   
SourceFenced revision naming the new target and reserved epoch   
e + 1.   
Activate HandoffActivate row of Install $\mathcal { U } _ { a c c e p t } ($ (HandofActivate): $X _ { T } ^ { \prime } , h ^ { \prime } , L , E [ M _ { E } ] , G _ { \chi } ^ { \prime } ,$ $\mathcal { R } [ d _ { r _ { C } } ] = r _ { C } , O [ p k e y ] = 0 \mathsf { u t c o m e V 1 } ( \mathsf { C o m m i t } , d _ { r _ { C } } , h ^ { \prime } ) ,$   
Table 13 append $f _ { j + 1 } ^ { T a r g e t A c t i v e } , \mathrm { s e t } \Xi ^ { \prime } = \Xi \mathrm { a n }$ nd and ${ \tilde { H _ { R } } } [ K _ { H } ] =$ Activation( ${ i } _ { f _ { j } ^ { S } }$ <sub>ourceFenced</sub> ,   
$B ^ { \prime } = \langle h ^ { \prime } , \rangle$ , Active, $w _ { t } , e _ { c } , d + 1 , \bot , c \rangle ;$ ; install the $d _ { \mathbf { \nabla } _ { \boldsymbol { f } } } T a$ <sub>rgetActive</sub> , $d _ { T _ { C } } , h ^ { \prime } ) \colon$ the receipt digest is identical.   
current-or-authorized $\Gamma _ { T } , \ L _ { z } r e v _ { T , \ L _ { z } } i .$ ssuer<sub>T</sub> , policy<sub>T</sub> . <sup>f</sup>j+1   
Restore Ordinary-kind row; Install $\mathcal { U } _ { a c c e p t } ( \mathfrak { f }$ Restoration) with Ordinary Commit in O and R; no $H _ { R }$ or F write; checkpoint is   
authenticated $X ^ { \prime } = \mathbf { I }$ Restore ${ \ L } _ { M } ( X _ { c } , M _ { \mu } ( X _ { k } ) ) ,$ new $h ^ { \prime } , L , E [ M _ { E } ] , G _ { \chi } ^ { \prime } ,$ auxiliary lineage, never a parent.   
checkpoint/proof/mask and $\Xi ^ { \prime } = \Xi$ and $B ^ { \prime } = b [ h e a d  h ^ { \prime } ] ;$ ; status, writer, epoch,   
optional pinned migrator   
dirSeq, handoff, createdFrom, and protected authority   
families stay current.   
Failure Any authenticated maintenance $\begin{array} { r } { { \cal S } ^ { \prime } = { \cal S } , \dot { \cal B } ^ { \prime } = { \cal B } , } \end{array}$ , and $F ^ { \prime } = F { \mathrm { ; } }$ no accepted state, effect, Append $r _ { L } = r b _ { L } ( q _ { L } )$ and the action-indexed $H _ { R } [ K _ { L } ]$ from   
action whose exact precondition directory, or lifecycle-revision write. Appendix A; the output revision is forbidden.   
fails

and computes $X ^ { \prime } = \operatorname { R e s t o r e } _ { M } ( X _ { c } , M _ { \mu } ( X _ { k } ) )$ . For protected paths

P = {authority, policy, revocation, issuer,

writer, epoch, dirSeq, lifecycle, createdF rom},

$$
M \cap P = \varnothing , \qquad X ^ { \prime } | _ { P } = X _ { c } | _ { P } .\tag{20}
$$

Malformed or unauthenticated outer lifecycle requests create no receipt.

## D Deterministic Resource Profile and Artifact

## D.1 Canonicalization resource profile

Canonicalization is governed by the signed, versioned profile

κ = ⟨jsonldV, rdfcV, contextDigests, maxContextUrls, maxContextDepth, maxBytes, maxDecodeDepth, maxDecodedItems, maxQuads, maxExpandedTerms, maxExpansionDepth, maxBlankNodes, maxNDegreeCalls, maxPermutationBranches, maxIssuerEntries, maxPathUnits, maxIntermediateUnits⟩. (21)

jsonldV pins the 16 July 2020 JSON-LD 1.1 Processing Algorithms and API Recommendation: processingMode=json-ld-1.1, its Context Processing and Expansion algorithms, ordered=true, rdfDirection=null, a signed absolute base IRI, and produceGeneralizedRdf=false. Input is UTF-8 application/ld+json containing one JSON object or array; HTML extraction and ambient bases are forbidden [17, 20]. Remote contexts are digest-pinned package objects, never network-fetched. Each reference resolves against the signed base to one exact URL–digest manifest entry; redirects, negotiation, and unlisted URLs are rejected. Primary and all supplied context bytes are charged at RawScreen and decoded at Decode. Duplicate supplied objects are charged separately; repeated references and cache hits do not recharge bytes but do count logical context-processing attempts. rdfcV pins RDFC-1.0 [18]. The path is JSON decode, Context Processing/Expansion, RDF emission, then RDFC canonicalization. Table 15 defines mathematical nonnegative counters, not implementation allocations. All cumulative counters reset once per proposal and sum occurrences across its datasets; they never reset per recursive invocation. Depth counters are active recursion gauges. inputOctets belongs to RawScreen, decodeDepth/decodedItems to Decode, and the remaining counters to Canonicalize.

Table 15: Exact counter increments. An occurrence is counted even if later deduplicated; tests happen before the event that would exceed its bound.
<table><tr><td>Counter</td><td>Increment or depth event</td><td>Counter</td><td>Increment or depth event</td></tr><tr><td>inputOctets</td><td>Before consuming each primary or supplied-context octet; duplicate packaged objects count again.</td><td>decodeDepth</td><td>Entering a JSON array/object; root depth is 1 and exit decrements it.</td></tr><tr><td>decodedltems</td><td>Decoder emits each member, array element, or scalar occurrence in the primary or each supplied context object.</td><td>expandedTerms</td><td>JSON-LD expansion emits each node, property, or value occurrence.</td></tr><tr><td>contextUrls</td><td>Before each JSON-LD Context Processing URL attempt after base resolution; repeated URLs, repeated references, and cache hits count.</td><td>contextDepth</td><td>Entering each logical nested Context Processing call; root is 1, cache use changes neither entry nor exit.</td></tr><tr><td>expansionDepth</td><td>Entering a recursive JSON-LD expansion call; root is 1 and exit decrements it.</td><td>quads</td><td>RDF conversion emits each quad occurrence, before set</td></tr><tr><td>blankNodes</td><td>First allocation or capture of each distinct blank-node identifier.</td><td>nDegreeCalls</td><td>deduplication. Entry to each RDFC Hash N-Degree Quads invocation. Before a logical issuer state first becomes reachable, charge</td></tr><tr><td>permutationBranches</td><td>Before exploring each RDFC permutation candidate</td><td>issuerEntries</td><td>every mapping in that state: a fresh insertion charges one and a clone charges its full inherited mapping cardinality. Equal mappings in distinct issuers or clones count again; structural</td></tr><tr><td>pathUnits</td><td>Each identifier, position, or digest token appended to an N-degree path.</td><td>intermediateUnits</td><td>sharing gives no discount. Simultaneously with each increment of decodedItems, expandedTerms, quads, permutationBranches, issuerEntries, or pathUnits; hence it is their running sum.</td></tr></table>

Every bound is tested before its event; context URL/depth exhaustion terminates at Canonicalize before lookup or recursive entry, with CanonicalizationLimit and no partial candidate. For a clone, the full inherited charge is tested before the clone becomes reachable; for insertion, the unit charge is tested before insertion. Either failure terminates at Canonicalize with the same CanonicalizationLimit receipt and no partial issuer state. Crossing any other limit terminates at its governing RawScreen, Decode, or Canonicalize stage with CanonicalizationLimit, produces no partial candidate, and binds $d _ { r a w }$ rather than a nonexistent d . RDFC-1.0 separately identifies adversarially expensive graph structure as a denial-of-service risk [18]; this is canonicalization complexity poisoning, not a semantic claim that the resulting graph is false. A local wall-clock, memory, or process watchdog may stop work earlier, but that event is a transient operational failure with no durable disposition unless the implementation can translate it into the deterministic counters above.

## D.2 Conditional assumptions

All safety claims are conditional on:

A1 Mediation. No bypass credential or API can advance an authoritative head or protocol index.

A2 Encoding and cryptography. Canonical encodings, typed hash domains, and signature schemes have their stated security and interoperability properties.

A3 Key custody. Proposal, allocator, evaluator, preparation-service, candidate-seal, and kernel keys are protected and restricted to their declared principals and scopes.

A4 Atomic serialization. The complete accepted unit, lifecycle changes, and reclamation changes serialize over every named mutable key and are durably all-or-none.

A5 Complete context. Policy, evaluator, authority, revocation, dependency, allocation, and lifecycle versions are locally lockable or transactionally revalidated.

A6 Trusted time. Activation obtains a conservative commit-time interval or an equivalent storage-enforced deadline.

A7 Lifecycle order. Branch creation, epochs, handoff, and recovery share one order observed by writers.

A8 Non-recycled scopes. Authenticated allocators issue monotone proposal/effect identifiers; retirement watermarks advance before covered rows are deleted.

A9 Verification objects. Replay and inclusion are claimed only when their required objects, interpreters, keys, and certified-state proofs are available.

Availability is not assumed. Receipt structural authenticity and kernel attestation use A1–A3; independent replay additionally uses the relevant A5–A6 objects and A9; durable inclusion additionally uses A4, A8, and A9.

## D.3 Executable Model and Artifact Specification

The formal state-space exploration of Section 5 is provided as an executable Python artifact (artifacts/bounded\_model.py). The artifact imports only standard library modules (collections, dataclasses, typing) and executes an exhaustive breadth-first search (BFS) state exploration over the finite protocol state space.

The executable model verifies that across all 2,808,230 reachable states and 5,526,474 state-changing transitions: (1) no encoded invariant is violated; (2) every named protocol coverage witness is reached; and (3) all terminal dispositions, 17-stage check sequences, pre-state authority rules, writer fencing, and forward restoration postconditions hold. Reproduction instructions and SHA-256 integrity digests are documented in artifacts/README.md.