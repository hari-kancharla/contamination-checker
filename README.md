# ContamCheckr

ContamCheckr is a local toolkit for investigating benchmark contamination in causal language models. It runs guided reproduction, Min-K%++ likelihood analysis and canonical-order tests, and exports JSON, per-example CSV and HTML reports. These are established methods; their sources and assumptions are documented below.

## Usage

Requires Python 3.10 or later. From this repository:

```sh
python -m pip install .
contamcheckr run --model ./checkpoints/model --local-files-only \
  --benchmark benchmark.jsonl --control reference.jsonl \
  --output report.json --csv per_example.csv --html report.html
```

Supply your local checkpoint and JSONL files with `id` and `text` fields, for example `{"id":"1","text":"Example text."}`. Reference records should be known nonmembers with comparable content and sampling. Without suitable controls, likelihood scores remain descriptive or are withheld from inference.

## Interpretation

The index is heuristic, not a contamination probability or certification of membership, nonmembership or benchmark validity. Results depend on model access, tokenization, context, reference quality and statistical assumptions. Passing diagnostic checks does not verify those assumptions.

In a fixed 3,000-record Pythia-1.4B study, Min-K%++ showed no demonstrated advantage over ordinary likelihood. The published summaries document that negative result; full raw runs remain local.

- [Methods and sources](docs/METHODS.md)
- [Reference-count planning](docs/REFERENCE_RESOLUTION.md)
- [Exact-token research scoring](docs/REAL_SCORING.md)
- [Real-checkpoint results](docs/CHECKMIA_REAL_RESULTS.md)
- [Reproducing the research](docs/REPRODUCE_REAL_RESEARCH.md)

MIT License. Author: Hari Krishna Kancharla.
