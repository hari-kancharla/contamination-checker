# Methods

contamcheck reports evidence with uncertainty. It does not turn any method into a yes or no contamination claim.

## Guided overlap gap

Access requirement: greedy text generation from a causal language model.

Each example is split on a token boundary. The model receives the same prefix twice. One prompt names the dataset and split, while the other describes only an example from a dataset. Both outputs are compared with the reference suffix using exact match, normalized near match, and ROUGE-L F1.

The primary score is the mean guided ROUGE-L minus the mean general ROUGE-L. Its index is the positive part of that gap, clamped to the range from zero to one. A paired bootstrap gives an interval when at least two examples can be scored.

The method can mislead when a model is poor at instruction following, when the suffix is easy to infer without memorization, when formatting dominates the match, or when the benchmark name changes the expected answer style. High guided overlap with a near-zero gap is weak evidence.

It is skipped when generation is unavailable or when no example contains enough tokens to split.

## Min-K%++ and Min-K

Access requirement: a full next-token probability distribution for every scored token.

For each target token, Min-K%++ compares the token log probability with the probability-weighted mean and standard deviation of the full vocabulary distribution. Tokens with zero or non-finite standard deviation are skipped. The lowest `k` percent of valid normalized scores are averaged. The Min-K baseline applies the same subset rule to raw target log probabilities.

The raw distribution is descriptive. It becomes a contamination index only when a matched control set is supplied. ContamCheckr computes AUC between benchmark and control scores, then maps AUC above chance into the range from zero to one. Random text and synthetic gibberish are not accepted as a published calibration baseline.

The method can mislead when the control differs in domain, length, style, or difficulty. Scores also vary across tokenizers and model families.

It is skipped when full distributions are unavailable or when no text has at least one finite token with non-zero distribution variance. Without a matched control, raw scores are reported but the method receives no aggregation weight.

## Canonical-order permutation test

Access requirement: sequence log-likelihood from the local model.

Examples are divided into deterministic, non-empty shards in published order. Each shard is scored in canonical order and in seeded random permutations. The reported advantage is the mean canonical likelihood minus the mean permuted likelihood. The p-value compares the observed advantage with permutation-derived null advantages, and the index is one minus that p-value.

The method can mislead when published order has ordinary topical or narrative coherence, when examples are nearly independent, or when model context truncation removes much of a shard.

It is skipped when sequence likelihood is unavailable or when fewer than four examples remain after loading.

## Combined index

The base weights are 0.5 for guided overlap, 0.3 for calibrated Min-K%++, and 0.2 for canonical order. Unsupported or uncalibrated methods receive no weight, and active weights are renormalized.

The interval bootstraps per-example guided gaps, matched Min-K%++ groups when calibrated, and shard-level order advantages. If the available sample sizes cannot support that calculation, the interval is omitted with a caveat.
