# Reference resolution for the individual screen

`contamcheckr resolution` shows whether reference counts make an **individual
conformal score-outlier screen followed by Benjamini–Hochberg (BH)**
reachable under **ideal exact BH arithmetic**. It loads no model, dataset, or remote endpoint and
does not change an audit run or its eligibility rules.

```sh
contamcheckr resolution --candidates 8 --controls 8 --q 0.10
contamcheckr resolution --candidates 40 --controls 40 --q 0.10
contamcheckr resolution --candidates 1000 --controls 1000 --q 0.10
```

The equivalent module invocation uses `python -m contamcheckr.cli resolution`.
For this workspace's relocated virtual environment, use:

```sh
.venv/bin/python -m contamcheckr.cli resolution --candidates 8 --controls 8 --q 0.10
```

The existing installed console wrapper retains its old absolute interpreter
path. This module invocation avoids that wrapper; no environment changes are
needed while an experiment is running. A fresh installation creates a wrapper
for its own environment.

The command writes JSON. Counts mean nominal **usable examples after quality
validation**, rather than merely raw rows available for download. Matching,
control nonmembership, effective token identity and score exchangeability
remain unverified by this arithmetic.

For N candidates, M controls, and requested BH level q, the smallest attainable
upper-tail conformal p-value is:

```text
p_min = 1 / (M + 1)
```

It requires a candidate score strictly above every control score; ties increase
the p-value. BH compares sorted p-values at rank k with `q × k / N`. Thus the
first rank which could cross a BH threshold is:

```text
k_required = ceil(N / (q × (M + 1)))
```

If `k_required > N`, even making every candidate maximally extreme cannot
produce a discovery in this individual screen. This says nothing about the
resolution of dataset-level MW comparisons or guided/order evidence, which
have separate requirements.

When reachable, `k_required` is the smallest **jointly extreme candidate group**
that could be flagged if those candidates attain p_min and all remaining
candidates have p=1. It is **not** a required number of leaked or memorized
items. Extreme scores can have other explanations, and actual scores may never
attain the theoretical floor.

The planner does not promise exact parity with the existing floating-point
runtime. At equality boundaries that runtime can round an adjusted p-value
upward. For example, with N=7, M=9, q=0.7 and one p=0.1 candidate, exact BH
permits the first-rank discovery, while the current float expression produces
0.7000000000000001 and withholds it. With N=6, M=9, q=0.10, six minimum p-values
similarly produce float adjusted values of 0.10000000000000002. These are
pre-existing runtime boundaries. This command reports ideal resolution and
does not change the frozen statistical implementation.

For one extreme candidate to cross the first BH threshold, the reference-count
requirement is:

```text
M_needed_for_one = ceil(N / q) - 1
```

The output also reports the difference between that count and the supplied M,
floored at zero. That is conditional arithmetic, not a recommendation to gather
arbitrary unmatched controls or change q to force a flag.

| Candidates N | Controls M | q | Minimum p-value | Minimum extreme group | Controls for one extreme candidate |
|---:|---:|---:|---|---|---:|
| 8 | 8 | 0.10 | 1/9 | Unreachable: required rank 9 exceeds N | 79 |
| 40 | 40 | 0.10 | 1/41 | 10 | 399 |
| 1000 | 1000 | 0.10 | 1/1001 | 10 | 9999 |

In particular, eight references make the individual screen unreachable at
q=0.10 even under complete score separation. A zero flag count in that design
therefore provides no empirical evidence of safety.

## Output and count eligibility

Important JSON fields are:

- `minimum_attainable_p_value_exact` and `q_exact`: rational strings, keeping
  nonzero values exact even when a float would round or underflow.
- `required_bh_rank_at_minimum_p`: the ceiling above, including unreachable
  values larger than N.
- `minimum_jointly_extreme_group_size`: that rank when reachable, otherwise
  `null`; `best_case_discovery_possible` states reachability explicitly.
- `scope` and `runtime_parity`: identify ideal exact arithmetic and explicitly
  withhold floating-point runtime parity.
- `controls_needed_for_one_extreme_candidate`: the first-threshold requirement.
- `minimum_calibration`: separate candidate/control minimum-count checks. The
  default is four usable examples **per group**, matching the current audit
  configuration's count rule. Passing these count checks does not establish
  any of the other eligibility assumptions.

An optional planning count can be declared explicitly:

```sh
contamcheckr resolution --candidates 40 --controls 40 --q 0.10 \
  --min-calibration-examples 4
```

The planner accepts N≥1, M≥0, a finite decimal 0<q<1, and a declared minimum of
at least two examples per group. Invalid values return an error without
falling through to model or dataset loading. Calculations use exact integer
and Fraction arithmetic; the CLI preserves decimal q input instead of parsing
it through a binary float. Scientific decimal notation is also accepted.

The supported input representation is bounded: integer counts and reduced q
numerator/denominator components may have at most **1,000 decimal digits**.
Decimal q spellings also have at most 1,000 coefficient digits and absolute
exponent at most 1,000; the reduced-component bound still applies. Larger
inputs, including `q=1e-5000`, return a clear bounded-representation error before
formatting or constructing an enormous exponent. Supported derived values fit
within 2,000 decimal digits (the p-value denominator can have 1,001). No global
integer-conversion setting is changed and no nonzero p-value is rounded to zero.

This command proves only the stated best-case threshold arithmetic. It does
not guarantee exact current-runtime flags, estimate actual statistical power, validate conformal/BH assumptions,
guarantee FDR, establish membership, or mark the whole contamination audit
unreachable. The tests enumerate exact BH thresholds independently and check
decimal boundaries, very large integers, and CLI loader tripwires. Those are
engineering checks, not machine-learning performance evidence.
