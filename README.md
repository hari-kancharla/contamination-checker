# ContamCheckr

ContamCheckr is a local, CPU-friendly auditor for benchmark contamination evidence in causal language models. It combines guided reproduction, Min-K%++ with a matched control, and a canonical-order permutation test. The output is an evidence score with caveats, not a membership verdict.

The command-line tool and Python package are installed as `contamcheckr`.

[![License: MIT](https://img.shields.io/badge/license-MIT-3b4a54.svg)](LICENSE) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3b4a54.svg)

## Quick start

```bash
python -m pip install .
contamcheckr run --model sshleifer/tiny-gpt2 --benchmark sample --limit 10 --output report.json --csv per_example.csv --html report.html
```

The command writes a JSON report, a per-example CSV, and a standalone HTML report.

## Validation

The approach was checked with a controlled memorization experiment. A small model was fine-tuned on a synthetic leak set while a matched clean set was held out of training. The adapted model scored higher on the leak set than on the clean set and higher than the untouched base model did, which is the separation the index is meant to surface. This is a controlled experiment, not a claim about contamination in any real model.

See [docs/VALIDATION.md](docs/VALIDATION.md) for the setup, exact commands, and saved reports, and [docs/EXAMPLE_REPORT.md](docs/EXAMPLE_REPORT.md) for a run against a public benchmark.

## Local interface

Run the same checker in a browser:

```bash
contamcheckr serve --port 8000
```

Open `http://127.0.0.1:8000`, configure a model and benchmark, and run the same audit pipeline used by the CLI. The web result uses the same bands, caveats, skipped-method reasons, and per-method values as the JSON and CLI output. It also provides downloadable JSON, CSV, and standalone HTML reports. The server rejects non-loopback bind addresses.

See [docs/WEB_UI.md](docs/WEB_UI.md) for field mappings and operating details.

## How it works

Guided overlap asks the model to reproduce the second half of each example with and without the dataset name. The paired overlap difference is the signal. High overlap with little guided-over-general gain is treated as weak evidence.

Min-K%++ uses the full next-token distribution. It contributes to the index only when a matched control set is supplied; otherwise, the raw distribution remains visible.

Canonical order compares sequence likelihood in published order with seeded within-shard permutations.

## What the score means

The index combines only methods that produced calibrated or internally comparative evidence. Guided overlap, calibrated Min-K%++, and canonical order each contribute with a fixed base weight, and the active weights are renormalized over whichever methods produced usable evidence. The result maps to low, moderate, and high evidence bands. See [docs/METHODS.md](docs/METHODS.md) for the exact method weights.

Scores are signals with uncertainty, not proof that an example was or was not in a training corpus.

## Limitations

Results depend on model capability, prompts, tokenization, sample size, and control quality. A weak model can fail to reproduce memorized text. A capable model can produce overlap without memorization. Min-K scores shift across domains and model families, while order tests can respond to ordinary document coherence. Paraphrasing, deduplication, fine-tuning, and deliberate evasion can all reduce detection.

## License

MIT License, copyright 2026 Hari Krishna Kancharla.
