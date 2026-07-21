# Real-checkpoint results

The fixed study scored 3,000 CheckMIABench records with Pythia-1.4B-deduped at step97000: 1,000 development nonmembers, 1,000 test members and 1,000 test nonmembers. Each original 2,049-ID sequence supplies 2,048 causal targets. Model forwards used FP16 on MPS; the primary score statistics used FP32.

| Method | Test AUC | Member flags / 1,000 | Nonmember flags / 1,000 |
| --- | ---: | ---: | ---: |
| Likelihood | 0.497832 | 17 | 16 |
| Min-K | 0.500376 | 15 | 18 |
| Weighted Min-K++, centered FP32 | 0.499672 | 20 | 20 |
| Uniform-vocabulary FP32 comparator | 0.501376 | 21 | 17 |
| Text-only classifier | 0.486586 | not assigned | not assigned |

The primary AUC difference, weighted Min-K++ minus likelihood, is +0.001840, with paired conditional 95% bootstrap interval [-0.022092, +0.026618]. There is no demonstrated advantage in this setting. The 3,000 bootstrap draws resample within labels and share indices across methods. These intervals do not cover unknown document dependence, label error or new model families.

The tail rule uses the 1,000 development nonmembers and p <= 0.01; its observed counts are not a population FPR or FDR guarantee. The text-only classifier uses development text only. Score direction and primary comparisons were fixed before held-out scoring.

All required FP32 targets are supported. The strict native variant is unavailable for 516 records: 165 development nonmembers, 174 test nonmembers and 177 test members. All remain accounted for. Its full-cohort AUC is withheld with bounds [0.3335465, 0.6537485], rather than replaced by a complete-case result. Strict support failure is not proof of a published pipeline crash.

The official before/after-checkpoint labels were not independently traced to original training batches. Malformed row-level step metadata is not used as a training receipt. No benchmark-task inflation, detector novelty or contamination certification follows from these results.

[All method summaries](../research/results/checkmia-methods.json) and [declared contrasts](../research/results/checkmia-contrasts.json) are byte-identical copies of the retained aggregate artifacts. Raw text, tokens, fitted parameters, vectors and local execution logs are not distributed. The scripts retain their historical evidence requirements; the public summaries alone are not a complete independent replication bundle. See [reproduction scope](REPRODUCE_REAL_RESEARCH.md).
