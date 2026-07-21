# Methods

contamcheckr reports evidence with uncertainty. It does not turn any method into a yes or no contamination claim.

## Guided overlap gap

Access requirement: greedy text generation from a causal language model.

Each example is split on a token boundary. The model receives the same prefix twice. One prompt names the dataset and split, while the other describes only an example from a dataset. Both outputs are compared with the reference suffix using exact match, normalized near match, and ROUGE-L F1.

The primary score is the mean guided ROUGE-L minus the mean general ROUGE-L. Its index is the positive part of that gap, clamped to the range from zero to one. A paired bootstrap gives an interval when at least two examples can be scored.

Local generations are grouped by the same requested budget so that batchmates do not change the retained prompt context. Fresh and cached completions use the same Unicode-safe truncation. A split is rejected if its decoded pieces cannot reconstruct the source without corruption. Greedy decoding can still vary with backend numerical behavior. The `--chat-template` flag wraps prompts in the tokenizer's chat template for instruction-tuned models; the default is plain prompts.

The guided prompt also runs a **mismatched-prefix negative control**: each target suffix is requested from a different example's prefix (cyclic donor). Reproduction under the wrong prefix is compatible with format or prior effects. High guided overlap with a near-zero mismatch gap now withholds the guided index and fused contribution; the raw results remain visible. The `--guided-stability` flag repeats the probe at 25% and 75% splits and reports the gap at each split; the combined index still uses the configured split only.

The diagnostic sign-flip p-value requires within-pair exchangeability or symmetric gap distributions. Fixed guided/general prompts do not randomize treatment labels, and a zero mean alone does not justify sign flips. This is not an exact test of “no contamination.”

The method can mislead when a model is poor at instruction following, when the suffix is easy to infer without memorization, when formatting dominates the match, or when the benchmark name changes the expected answer style. High guided overlap with a near-zero gap is weak evidence.

It is skipped when generation is unavailable or when no example contains enough tokens to split.

## Min-K%++ and Min-K

Access requirement: a full next-token probability distribution for every scored token.

For each target token, Min-K%++ compares the token log probability with the probability-weighted mean and standard deviation of the full vocabulary distribution. Tokens with zero or non-finite standard deviation are skipped. The lowest `k` percent of valid normalized scores are averaged. The Min-K baseline applies the same subset rule to raw target log probabilities.

The raw distribution is descriptive. It becomes a contamination index only when a matched control set is supplied. ContamCheckr computes AUC between benchmark and control scores, then maps AUC above chance into the range from zero to one. Random text and synthetic gibberish are not accepted as a published calibration baseline.

Calibration also requires at least `min_calibration_examples` scored examples per side (default 4). Below that, the index is withheld. Diagnostics include a DeLong normal-approximation interval, Mann–Whitney p-value, and Cohen's d. Rank-test tails enumerate all label assignments when there are at most 100,000, including ties; larger cases use a tie-corrected normal approximation with stable `erfc` tails. The calculation method is recorded. The legacy add-half smoothed AUC is a heuristic, **not a Bayesian posterior**: its pairwise comparisons are dependent. DeLong/bootstrap intervals can degenerate on a perfectly separated finite sample and do not prove perfect population discrimination.

## Null calibration

Access requirement: a matched control scored into at least 8 examples.

The control scores are split into interleaved halves and compared. A detected difference warns about heterogeneity or ordering; a nonsignificant result establishes neither homogeneity nor benchmark/control exchangeability. The check is reported as `control_null`.

A companion `domain_shift` check compares scored token counts. `domain_fingerprint` examines symbol fractions and finite decimal numbers, using one median numeric value per text rather than treating correlated numbers within a text as independent. Its bounded rank effect detects even constant but separated identifiers.

The versioned `comparison-v1` policy blocks likelihood inference for overlapping/repeated normalized text, identical effective scoring token sequences where the local scorer exposes them, strong numeric shifts (two-sided p<0.05 and absolute rank effect>0.5), strong length shifts (p<0.05 and median ratio outside 1/1.5–1.5), and detected control split-half heterogeneity. Missing controls/insufficient scores also withhold inference. **These are conservative heuristic thresholds, not a validated matching algorithm.** A clear result is labeled `conditional`; exchangeability and control nonmembership remain unverified. A custom adapter without effective-token identity support is labeled as unchecked for that diagnostic.

One eligibility decision controls the likelihood index, fused p-value contribution, and score-outlier screen. A rejected comparison retains raw score distributions, descriptive AUCs and item vectors. Duplicate example IDs fail early instead of overwriting distinct records.

Alongside Min-K%++, the report calibrates zlib, entropy, mean-rank, top-1-probability, and leave-one-out word 13-gram overlap (`ngram_auc`) against the same control as `score_baselines` (reported only, never aggregated). Compare them before treating Min-K%++ as distinctive: on the members-only validation split, mean log-probability separates exactly as well as Min-K%++.

## Membership screen

Access requirement: calibrated Min-K%++ scores for the benchmark and a matched control.

Each benchmark example gets a conformal p-value against the control distribution (higher scores are more suspicious), followed by Benjamini–Hochberg at the requested FDR level (default 0.10). This is a **conditional score-outlier screen**. It requires clean calibration records and exchangeable null scores, together with the applicable BH dependence conditions. The tool does not verify these assumptions or establish a contamination FDR. Flags are for follow-up and do not enter the combined index. A blocked comparison produces a machine-readable withheld reason, not flags. The report gives calibration count and the minimum attainable p-value, `1/(n_control+1)`.

It is skipped when full distributions are unavailable or when no text has at least one finite token with non-zero distribution variance. Without a matched control, raw scores are reported but the method receives no aggregation weight.

## Canonical-order permutation test

Access requirement: sequence log-likelihood from the local model.

This is the sharded likelihood comparison test from Oren, Meister, Chatterji, Ladhak, and Hashimoto, "Proving Test Set Contamination in Black-Box Language Models" (ICLR 2024), Algorithm 1. Examples are divided into contiguous shards in published order. Each shard is scored in canonical order and against seeded within-shard permutations. The p-value is a one-sided t-test on the shard advantages, canonical log-likelihood minus the mean permuted log-likelihood. Sequence log-likelihoods are accumulated in float64.

The p-value is always reported. It enters the combined index only when all three reporting gates pass:

- at least 10 shards, because Oren et al. found the asymptotic t-test unreliable below that
- mean advantage of at least 0.001 nats, because smaller gaps are consistent with accumulation error
- one-sided p-value below 0.05

When the gates pass, the order index is one minus the t-test p-value. When they fail, the index is omitted and the reason is included in the caveats. A 25-draw permutation rank is not used as the p-value. That rank cannot go below 1/26, including when the canonical ordering wins by numerical noise.

The method can mislead when published order has ordinary topical or narrative coherence, when examples are nearly independent, or when model context truncation removes much of a shard.

It is skipped when sequence likelihood is unavailable or when fewer than four examples remain after loading.

## Combined index

The base weights are 0.5 for guided overlap, 0.3 for calibrated Min-K%++, and 0.2 for canonical order. A method with no usable evidence receives no weight, and active weights are renormalized. Guided overlap is omitted when neither prompt produces a near match and the absolute ROUGE-L gap is below 0.05, because the model is not reproducing the suffix. Canonical order is omitted unless the gates above pass. If no method remains, the index is not estimated and the band is insufficient evidence.

The weights and bands are heuristic summaries; they are not learned or calibrated contamination probabilities. Alongside them is a **combined diagnostic p-value** (`combined_p_value`). Its family is always guided, Min-K%++, and order, and its correction is `min(1, 3 * min(component_p))`. Unavailable or gated components contribute p=1 rather than shrinking the denominator after results are seen. With no admitted components, the aggregate is not estimated. Raw and admitted p-values, fixed family, effective values and component assumptions are recorded separately.

Bonferroni's union bound tolerates arbitrary dependence **when the component p-values are valid for their own nulls**. Those nulls are prompt effects, score-distribution differences and order effects under the assumptions above. They are not interchangeable with “no training contamination.” No uniform-null or contamination-error guarantee is claimed. Historical artifacts using other combiners are not validation of this implementation.

Mean token log-probability is scored against the same control as Min-K%++ and reported as `likelihood_baseline`. It is not part of the index. On an easy memorization split the two AUCs should be compared before treating Min-K%++ as additional evidence.

A bootstrap interval is reported only for a single admitted method and is conditional on that selection and sample. Multiple methods share examples and shards; their combined interval is withheld until an aligned joint resampling design is implemented. The interval is not uncertainty about a contamination probability. The `planning` field is a two-sample normal-theory effect-size planning aid, not measured detector power.

## Engine: batching, caching, and provenance

Token statistics, sequence likelihoods, and generations use a batched engine and optional SQLite cache. Windowed execution propagates its stride through batching, fallback and cache identity; a failed windowed batch cannot silently become truncated scoring. Generation-cache semantics are versioned. Cache-open failures produce a caveat.

The v2 manifest records source-file content digests, resolved Hub commits when available, tokenizer processing identity, order-independent and ordered dataset digests, configuration and effective execution settings. Local model metadata signatures are explicitly distinguished from cryptographic weight hashes; a runtime manifest alone does not prove exact reproduction. The randomized experiment runner separately seals full checkpoint hashes and literal source copies.

Remote `/completions` access provides generation only, with a Unicode-codepoint proxy for splitting/truncation and heuristic provider budgets. Provider token counts are unknown, requests are sequential, and temperature zero does not establish remote determinism. Metadata and caveats are exported; no live endpoint reliability has been established by offline transport tests.
