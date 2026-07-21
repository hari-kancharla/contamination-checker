# Evidence and limitations

The real [Pythia checkpoint study](CHECKMIA_REAL_RESULTS.md) found no demonstrated weighted Min-K++ advantage over ordinary likelihood. Published aggregate summaries are in [research/results](../research/results/).

An analysis of released OLMo-Detect scores tested the observable comparison policy. For the primary 1B view, ungated versus accepted-policy results were: same-domain nonmembers 1/36 versus 0/30 flags; matched members 2/32 formed versus 2/30 (four unavailable templates); shifted members 0/36 versus 0/15; cross-domain nonmember stress 4/36 versus 3/25. The policy did not establish a useful general advantage. These are dependent fixed-bank comparison counts, not population error rates. The [primary summary](../research/results/selectivity-primary.csv) records attempted, formed and accepted denominators.

Historical small-model and synthetic experiments are engineering probes. They are not evidence of realistic pretraining-membership detection, calibrated contamination probabilities or top-venue novelty. Synthetic fixtures remain solely to keep the tested software examples and regression suite reproducible.

Passing unit tests verifies software behavior. It does not establish trustworthy membership labels, exchangeable references, generalization, demand or production readiness. In particular, the ideal reference-budget planner discloses a remaining floating-point BH equality-boundary limitation; see [reference planning](REFERENCE_RESOLUTION.md).
