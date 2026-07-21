# Retained-logit precision diagnostic

Run this separate diagnostic only after the full study and CPU-forward diagnostic exit. It performs one new MPS-FP16 forward on the previously observed development record, preserving the 64 by 50,304 logit block and exact targets. Aligned target 353 is original token position 354 and row 33 of block [320:384].

Compare retained native log-probabilities, probability results, products, reductions and guard arithmetic with explicitly named CPU-FP64 substitutions. Keep nonfinite values and failed support visible. The first log-softmax comparison varies backend and dtype; later fixed-input substitutions have narrower interpretations. Descriptive mass summaries are CPU-FP64 calculations.

Run `python experiments/checkmia_precision_mechanism.py --help` for required paths. Both completed prerequisite manifests and the private historical capsule are required; see [reproduction scope](REPRODUCE_REAL_RESEARCH.md). A selected-target explanation is not a failure-frequency, detector-accuracy or historical-program-crash claim.
