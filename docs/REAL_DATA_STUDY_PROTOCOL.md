# Real-data eligibility study, version 1

**Status: prospective analysis plan after source inspection, not a blinded or externally registered preregistration.** Written 2026-11-07 after inspecting the published paper, source, record schemas, counts, grouping fields, and some individual saved-score fields. No case-level eligibility outcomes or comparative study results have been computed for this plan. This document specifies an evaluation; it does not report a successful result or authorize additional model execution.

The first question is deliberately narrow: **on published real records, does the frozen comparison policy avoid misleading dataset-level likelihood evidence better than cheap mismatch gates, at comparable coverage, while retaining evidence on official member comparisons?** The study can falsify the policy's practical usefulness. Its size and observational labels cannot establish a low false-positive rate, individual membership detection, causal exposure effects, or a new research contribution.

## 1. Source and claim boundary

Use the released [OLMo-Detect repository](https://github.com/LuckyDaydreamer/OLMo-Detect/tree/6eb88de34a09dea85c4b41deff4048e5a203e1b0), revision `6eb88de34a09dea85c4b41deff4048e5a203e1b0`. The local receipt is `.cache/research/olmo_detect/source_receipt.json`; the downloaded archive SHA-256 is `d7a157423bf537a343975db13bfddd9b271a7c24a9cb7ecc8f6b2342ca8d30ce`. Upstream programs must not be executed to obtain this study's results. A separate reader must join the released records and score vectors and audit score orientation and raw AUCs.

The primary target is the released `OLMo2-1B-Instruct` score artifact, using pretraining-domain records. Other released OLMo sizes are secondary correlated views of substantially the same documents, not independent model families or new test samples. Do not pool them to inflate the effective sample size. Later MIMIR/CheckMIABench or another family's results require a separately frozen protocol and provenance audit.

[OLMo-Detect](https://arxiv.org/html/2610.02986v1) constructs members from published training corpora and nonmembers from held-out/source-remainder pools, with 13-gram filtering against the union of training stages. Those are useful externally published labels. They are not randomized inclusion assignments, an independently traced batch-consumption history, or proof of absence of every semantic duplicate. Filtering can also alter the nonmember population. Use the terms **official member/nonmember** and **published-label comparison**, rather than certain real-world training history.

Saved samples contain `record_index`, `sample_id`, a shortened `text_sha1`, `token_count`, and method scores. They do not provide exact retained token identities, immutable model revision, dtype, or complete run configuration. `sample_0` is not a globally unique identifier. Join by source file, row index, and verified text hash, and add a full SHA-256 of the raw text. An inconsistent/missing join is a failed record, never an invitation to take the next convenient score.

The inspected 1B nonmember vectors reach 4,095 scored tokens. Counts above 1,023 are DCLM 53/134, OpenWebMath 47/136, peS2o 16/241, Java 2/50, and Python 40/90. Consequently, **do not describe their analysis as a 1,024-token replication**. A saved scored-token footprint is observable; the exact upstream context and execution cost are not. Engine parity, a controlled context-budget comparison, and this saved-vector study are separate tasks.

## 2. Available data and independence

These are counts from the pinned raw files, before the final connected-component quarantine. Each member condition has the listed count; matched and shifted members and shared nonmembers must not be added as independent samples.

| Pretraining stratum | Dev records per arm/condition | Test records per arm/condition |
| --- | ---: | ---: |
| DCLM | 16 | 134 |
| OpenWebMath | 15 | 136 |
| peS2o | 27 | 241 |
| StarCoder Java | 6 | 50 |
| StarCoder Python | 10 | 90 |
| Total | 74 | 651 |

The nonmember test records have 134 DCLM URL groups, 136 OpenWebMath URL groups, 241 peS2o paper-ID groups, and 95 known StarCoder repository groups. Thirty-seven Python records have no usable repository provenance. Java has 50 records in 43 repositories; Python has 53 provenance-bearing records in 52 repositories; these repository sets must be unioned, not added. Dev has 16, 15, 27, and 13 known groups respectively, with three code records lacking repository provenance.

Group all arms, conditions, and splits together before allocating cases:

1. Union rows with identical normalized text (`" ".join(text.casefold().split())`) or identical domain-specific source identity. DCLM/OpenWebMath use document URL, peS2o uses paper `id`, and code uses `max_stars_repo_name`. Strip URL fragments and normalize scheme/host case; preserve path/query content. Record the normalization implementation and hash. A code file path is not a repository ID; `py_nm_topup_*` is not repository provenance.
2. Quarantine every component touching dev from held-out test, including across member conditions. An already observed exact normalized OpenWebMath shifted-member overlap occurs at dev row 1 and test row 52 despite different arXiv URLs. These are zero-based row indices. The automated audit must find and quarantine it; do not hard-code only that pair.
3. Exclude unknown code provenance from the primary bank. Report its descriptive sensitivity analysis separately. If an exact text has contradictory official membership labels, quarantine and report the conflict. Shared source identity with different texts is a dependence concern, not by itself a contradictory label.
4. Set each component ID to its lexicographically smallest stable row ID (source-relative path, zero-based row index, and raw-text SHA-256). Select at most one representative per component, arm, and member condition by smallest raw-text SHA-256, breaking ties by source path and row index. Do not select representatives using a score, length, eligibility outcome, or classifier prediction.

Known source groups do not capture all web-host, author, topic, repository-fork, or semantic-clone dependence. Report these remaining limits. A host-grouped sensitivity analysis may reduce the bank further; it cannot increase the sample count.

With eight suspect and eight reference groups per case, current source-group counts give an **upper bound of 8 + 8 + 15 + 5 = 36 disjoint nonmember cases**, before connected-component exclusions. The final manifest must give actual counts and exclusions. Do not advertise 651 individual texts as 651 independent dataset-level audits.

## 3. Freeze and case construction

Use namespace `real-data-selectivity-v1/20261107`. Before computing held-out case outcomes, save a manifest containing all raw/vector file hashes, joins, component IDs, exclusions, group representatives, case assignments, dev-fitted parameters, implementation hashes for `validity.py`, `calibration.py`, `stats.py` and the reader/evaluator, dependency versions, score signs, policy version, and the protocol hashes. Check the source hashes again after evaluation. Changes create version 2 and a fresh held-out evaluation; they must not silently replace version 1.

For each primary domain (DCLM, OpenWebMath, peS2o, StarCoder), sort eligible nonmember test components by SHA-256 of namespace, domain, and component ID. Form consecutive complete blocks of 16 groups. The first eight form the suspect side and the next eight the reference side. Keep their order fixed for the interleaved control self-check. Record unused remainder groups. A record appears in only one primary null case. Do not rotate partitions until something rejects.

The official member views reuse a null case's reference and replace its suspect side with eight unused, score-independent official member groups from the same domain. Construct matched-member and shifted-member views separately, in the same deterministic group order. Exclude groups overlapping the reference or a previous member case within that condition. For code, match the reference's Java/Python counts where possible without reusing a repository; otherwise record the case as unavailable rather than changing its reference. Report actual member-case counts: neither 36 positive cases nor equal counts across conditions are assumed. Published collection-level matching does not guarantee matching within each new eight-record case.

Also create one **dependent mismatch stress view** per null suspect block by taking a reference block from the next domain in the fixed cycle DCLM → OpenWebMath → peS2o → StarCoder → DCLM, using index modulo the next domain's block count. If a domain has no block, mark the view unavailable; do not alter the cycle. These views reuse records and may reuse references. Report them separately, never as additional independent null trials. Both sides remain official nonmembers; their score distributions need not be equal. A rejection here can correctly detect a distribution difference while being misleading as membership evidence.

No new text, paraphrase, corruption, synthetic leakage, fine-tuning, or additional target-model calls enter phase A. No bootstrap or repartition creates new records or independent evidence.

## 4. Fixed endpoint and policy

The unit is a **dataset-level comparison**, not a document-level accusation. Primary score is the upstream `min_k_pp_score`, unchanged in orientation: higher is more member-like. The primary diagnostic is the current one-sided Mann–Whitney comparison, suspect above reference, computed by the frozen implementation, with its exact/asymptotic/tie method recorded.

For the current three-method family, guided and order are unobserved and assigned p=1. If the eligibility policy accepts the likelihood comparison, the report diagnostic is `min(1, 3 * p_MW)`. If it blocks, the diagnostic is withheld; encode the action as abstention, not p=0 or a favorable missing value. A reported diagnostic rejection means accepted and adjusted p<0.05. Do not reduce the family multiplier to one because only Min-K++ was run. This conditional score-distribution test is not a calibrated test of contamination. Bonferroni does not establish exchangeability or repair selection bias in the accepted subset; an accepted-case error rate need not equal the nominal level.

Freeze `comparison-v1` with `min_calibration_examples=4`, unchanged numeric/length/self-check thresholds, no score canonicalization, and no dev-derived policy repair. At eight reference records, the interleaved self-check has four observations per half. The available-vector variant uses raw full-text overlap/duplicate checks, the current finite-decimal per-text numeric fingerprint, released `token_count` for the length check, and Min-K++ reference scores for the self-check. It must explicitly report effective scoring-token identity as **unavailable**. Passing raw-text checks cannot rule out identical truncated inputs. This is an observable policy variant, not verification of every local-model safeguard.

Report upstream `loss_score` as the fixed primary simple likelihood comparator. The pinned code stores **mean log-likelihood**, so higher is member-like; do not negate it because its name contains “loss.” Evaluate LOSS with its own fixed, identically adjusted one-sided MW rule. Do not combine or select LOSS and Min-K++ by the smaller p-value, higher AUC, or favorable domain. Other methods are descriptive and must be labeled exploratory, with their full list disclosed.

**Unreachable individual-screen operating point:** with eight controls, conformal p-values have minimum `1/9`, which exceeds the default BH level 0.10. No individual item can be flagged, even under complete score separation. Report this structural zero power. Zero per-item flags here provides no empirical evidence of individual-screen safety. Raising q or silently pooling controls changes the study and is prohibited.

## 5. Comparators, development, and coverage

All comparators share the same score-independent provenance, identity, join, and finite-score requirements. Keep unavailable/failed cases in attempted-case accounting. Compare these gates applied to the same fixed Min-K++ diagnostic:

| Gate | Fixed rule |
| --- | --- |
| No eligibility screen | Accept every common-quality complete case. |
| Length only | Current length blocker: two-sided MW p<0.05 and median scored-token ratio outside [2/3, 1.5]. |
| Model-free origin classifier | The dev-fitted text-only rule below. It sees which set a text belongs to, never its hidden membership label. |
| Full observable policy | The exact frozen available-vector `comparison-v1` checks above. |
| Random at matched coverage | Analytical uniform selection of the same number of cases, without inspecting scores or labels. |

The text-origin gate uses raw text only. For each 8+8 case, assign each origin's groups, ordered by a fixed fold hash, cyclically to four folds. Fit TF–IDF character 3–5-grams (`lowercase=True`, `min_df=1`, `max_features=20000`, `sublinear_tf=True`, L2 normalization) and logistic regression (`C=1`, `solver=liblinear`, `max_iter=1000`, seed 20261107) **inside each training fold**. Score the held-out origin labels. Its mismatch statistic is the mean of the four within-fold AUCs, higher meaning more origin distinguishability. Empty-vocabulary fits predict 0.5 and are reported; optimizer failures make the gate unavailable. Paths, URLs, IDs, overlap filters, source metadata, and membership-label-specific schema fields are not features.

Calibrate its threshold before test analysis using **dev nonmember text only**, not unavailable dev target-model scores. Apply the same provenance rules and form complete disjoint 16-group dev blocks per domain. Current counts allow at most two such blocks, DCLM and peS2o. For each available block, make 32 unique balanced origin partitions ordered by the fixed hash namespace and partition index; use the same within-fold fitting procedure. Set the threshold to the empirical 95th percentile using the “higher” quantile rule. Accept when the test statistic is at or below that threshold. Freeze the actual threshold, partition IDs and calibration-block count. With fewer than two blocks, mark the operational origin gate uncalibrated and withhold superiority claims against it.

The resulting threshold is a **development-fitted heuristic**, not a 95%-valid gate. Sixty-four partition views from two blocks remain two underlying source pools. Do not count them as 64 independent calibration cases or infer a distribution-free guarantee. This scarcity is a study limitation, not a reason to use held-out test outcomes for threshold selection. Dev target-model scoring, if later obtained, may support a new protocol but must not retroactively tune this policy.

At matched coverage, also rank complete test cases by origin mismatch and separately by absolute log median token-length ratio; accept the lowest k, where k is the full policy's accepted count. Resolve ties by case hash. This is a **transductive descriptive comparator**, not a frozen deployable threshold: k uses the policy's observed coverage, but neither outcome flags nor true membership labels select cases. Report null and positive strata separately so equal pooled coverage cannot conceal different positive retention. The operational dev-threshold results must remain visible.

For a random gate accepting k of N cases with F ungated diagnostic rejections, expected retained rejections are `k*F/N`; report the finite hypergeometric distribution where desired. Random seeds or repeated draws are not replications. If N or k is zero, display the corresponding quantity as unavailable, not a successful error rate.

A separate blind membership baseline fits the same fixed text feature family and logistic configuration once on official **matched dev member versus dev nonmember** representatives, changing only `class_weight="balanced"`, and evaluates held-out matched and shifted records. It uses published membership labels only during dev training and is explicitly a supervised, target-model-free comparator. Reconcile its grouping/exclusions before fit; do not use a label-bearing metadata field. It tests whether apparent membership separability is already carried by text. Do not select features/hyperparameters on test AUC or use this classifier's test errors to choose eligibility thresholds.

## 6. Outputs and uncertainty

Publish every case, including failures and abstentions. Required fields are source/model artifact, domain/condition, record/group IDs, scored-token counts, common-quality status, missing-check status, each blocker, each gate/action, raw AUC, unadjusted p, family-adjusted diagnostic p, and the separately identified official-label outcome. Report at least:

- Attempted, complete, accepted, rejected, and unavailable case counts by domain and condition.
- For entirely nonmember cases: diagnostic rejections/all attempted cases; diagnostic rejections/accepted complete cases; and null acceptance coverage. The latter denominator is undefined when acceptance is zero.
- For official member cases: rejections/all attempted positives; rejections/accepted complete positives; and positive acceptance coverage, separately for matched and shifted views. These are empirical dataset-level sensitivities under published labels, not individual membership power.
- Paired changes against no gate, length, origin, and random matched coverage; blocker ablations as descriptive sensitivity analyses. Show the individual cases behind each claimed improvement.
- Actual preprocessing wall time, classifier fits, number of target-model calls (zero in phase A), retained scored-token footprint, exclusions, and unknown provenance. Do not convert stored token counts into a measured dollar cost or a controlled context budget.

Use exact finite-bank counts as the primary evidence. Model sizes, member conditions, mismatch views, overlapping texts, and alternative partitions remain dependent. Do not attach binomial confidence intervals to their pooled total. Even the disjoint primary bank spans heterogeneous domains and shared source ecosystems, so a binomial population model requires an additional sampling assumption. If shown, label such intervals as assumption-dependent; a source-group/domain sensitivity resample is not new evidence.

For scale only, zero events in 36 independent identically distributed trials would still give the one-sided 95% binomial upper bound `1 - 0.05**(1/36)`, about 8%. That idealized calculation does not validate this bank's sampling assumptions. At a fixed threshold, zero events need at least 299 such independent negatives to put that bound below 1%, and 2,995 below 0.1%. These concern the specified trial unit: individual nonmember records cannot be substituted for independent dataset-level cases. Larger released benchmarks can improve a later individual-error study only after their label and grouping audits.

## 7. Decision rules and research bar

Phase A is a bounded falsification/feasibility study, not a confirmatory success test. No favorable effect-size margin was independently validated before seeing this benchmark. Do not turn a post-hoc Pareto improvement into a formal acceptance criterion or a paper-readiness claim.

Stop expansion of the current policy if its apparent benefit comes from near-total abstention, if it loses the positive cases that produce useful evidence, if length/origin/random matched coverage explains the same benefit, if no gate produces no errors to avoid, or if the favorable result depends on a selected domain, model size, threshold, or duplicated view. Identify the specific blocker and publish the negative result; do not immediately tune it and reuse the same held-out bank as validation.

Proceed to designing a genuinely confirmatory study only if the frozen policy has a visible, nontrivial error/coverage/positive-retention advantage that survives the cheap comparators and provenance audit, with explicit failure cases. Before running that study, declare its deployment loss tradeoff, minimum acceptable null and positive coverage, meaningful improvement margin, independently held-out source/model family, and achievable uncertainty from actual sample counts. Phase A alone cannot supply a low-FPR guarantee.

Novelty must clear the following existing primary work:

- [OLMo-Detect](https://arxiv.org/html/2610.02986v1) already studies matched/shifted real membership, blind text separability, many attacks, and cross-domain robustness. Recomputing its AUCs or showing mismatch matters is replication, not a new contribution.
- [The Reliability Gap](https://arxiv.org/html/2606.03305v2) already examines practical breakdowns across models and evaluation conditions. Another small failure catalogue needs an explanatory mechanism or useful corrective result.
- [Shift-aware score transport](https://arxiv.org/html/2610.31549v1) already constructs nonmember-only calibration tasks to correct observational auditing under shift. Its target is a conservative transported privacy estimate under conditional-invariance/support/error-envelope assumptions, not this policy's dataset rejection rate. A fixed-reference-budget/selectivity claim still needs a direct comparison in a compatible access/target setting; “nonmember-only calibration” is not itself a gap.
- [OLMo-Twin](https://arxiv.org/html/2610.41732v1) already separates document inference from collection evidence with randomized matched-pair exposure and a close reference. This observational reanalysis cannot claim its causal strength or merely repeat the collection-versus-document distinction.
- [KTD](https://proceedings.iclr.cc/paper_files/paper/2025/file/c859b99b5d717c9035e79d43dfd69435-Paper-Conference.pdf) and [FTD](https://aclanthology.org/2026.acl-long.1390/) already pursue false-discovery control through different constructions and assumptions. KTD needs target gradients and generated knockoffs; target fine-tuning constructs its experimental members and is not necessarily an attack-time requirement. FTD uses a member-score reference to select clean items, reversing our suspicious-item discovery direction. Reconcile these access costs, references, targets, and assumptions before asserting a common guarantee or superior cost.
- [CheckMIABench](https://aclanthology.org/2026.acl-short.30/) already provides checkpoint-based matched-population evaluation, blind baselines, and common likelihood attacks on Pythia and OLMo. Its nonmember labels concern chunks after the specified intermediate checkpoint, not the final trained model. Repeating that evaluation or using a larger clean bank is necessary validation, not a new benchmark contribution.
- [Membership Inference Attacks with FDR Control](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhao_Membership_Inference_Attacks_with_False_Discovery_Rate_Control_ICCV_2025_paper.pdf) already proposes a post-hoc conformal-style MIA wrapper using auxiliary/surrogate information. [Null-Calibrated Conformal Selection](https://arxiv.org/abs/2606.22336) studies confirmed non-target calibration, finite-sample null validity, BH/BY, and power costs. Its “target membership” is the selected statistical event, not specifically LLM training membership. A clean-control rank plus multiplicity correction is not a new mechanism.
- [Verdict Instability under Reference Resampling](https://arxiv.org/html/2609.00691v2) studies finite-reference uncertainty and abstention versus random in OOD detection; [Excess Separability](https://arxiv.org/html/2608.12652v2) studies control-set/nuisance correction uncertainty and withheld contamination verdicts for activation probes. These use different statistics, but rule out an unsupported generic novelty claim about finite controls and abstention.
- [Weighted conformal under covariate shift](https://arxiv.org/abs/1904.06019) and [conformal beyond exchangeability](https://arxiv.org/abs/2202.13415) already develop robustness beyond naive exchangeability. The present heuristic gates inherit no such theorem. Abstention, fixed-family adjustment, and a finite clean-reference budget are not established novelty claims.

CheckMIABench's larger released banks are a candidate later validation resource, not yet a demonstrated gap in that benchmark or a substitute for checking what its published baselines already establish. Pin its data revision, exact checkpoint, packed-token semantics, selection rules, and family splits before evaluating it.

A potentially publishable result would need something materially stronger: for example, a new correctly scoped bound/diagnostic on accepted-case risk under explicit, empirically testable assumptions, or a reproducible failure mechanism and repair that improves the error/coverage/power frontier on independently held-out real families against these existing approaches. A truthful and reproducible negative result can still improve the tool. Neither an attractive report, agreement among reviewers, nor this protocol establishes a strong-paper contribution.
