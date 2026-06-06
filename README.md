# ContamCheckr

ContamCheckr is a local, CPU-friendly auditor for benchmark contamination evidence in causal language models. It combines guided reproduction, Min-K%++ with a matched control, and a canonical-order permutation test. The output is an evidence score with caveats, not a membership verdict.

The command-line tool and Python package are installed as `contamcheck`.

[![License: MIT](https://img.shields.io/badge/license-MIT-3b4a54.svg)](LICENSE) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3b4a54.svg) ![Tests: 33 passed](https://img.shields.io/badge/tests-33%20passed-2f6f4e.svg)

## Quick start

```bash
python -m pip install .
contamcheck run --model sshleifer/tiny-gpt2 --benchmark sample --limit 10 --output report.json --csv per_example.csv --html report.html
```

The command writes a JSON report, a per-example CSV, and a standalone HTML report.

## What it found

The controlled validation fine-tuned `sshleifer/tiny-gpt2` on 40 synthetic leak examples and kept 40 matched examples out of training. The adapted model scored **0.492** on the leak set, compared with **0.008** on the clean set and **0.131** for the untouched base model on the leak set.

| Run | Index | Band |
| --- | ---: | --- |
| Adapted model on leak set | 0.492 | moderate evidence |
| Adapted model on clean set | 0.008 | low evidence |
| Base model on leak set | 0.131 | low evidence |

The trained run was clearly above both controls, but it did not reach the high-evidence band. Min-K%++ and canonical order supplied the separation; the guided-over-general gap stayed at zero. This is a controlled memorization experiment, not a claim about contamination in a real model. See [docs/VALIDATION.md](docs/VALIDATION.md) for the method, exact command, full table, and saved reports.

The public-benchmark example used `distilgpt2` on 10 GLUE SST-2 validation rows. It returned **0.275**, in the **moderate evidence** band, with an interval of **0.000 to 0.275**. That result is weak and uncertain: guided overlap contributed no evidence, Min-K%++ had no matched control, and the score came from a small canonical-order test. See [docs/EXAMPLE_REPORT.md](docs/EXAMPLE_REPORT.md).

## Local interface

Run the same checker in a browser:

```bash
contamcheck serve --port 8000
```

Open `http://127.0.0.1:8000`, configure a model and benchmark, and run the same audit pipeline used by the CLI. The web result uses the same bands, caveats, skipped-method reasons, and per-method values as the JSON and CLI output. It also provides downloadable JSON, CSV, and standalone HTML reports. The server rejects non-loopback bind addresses.

See [docs/WEB_UI.md](docs/WEB_UI.md) for field mappings and operating details.

## How it works

Guided overlap asks the model to reproduce the second half of each example with and without the dataset name. The paired overlap difference is the signal. High overlap with little guided-over-general gain is treated as weak evidence.

Min-K%++ uses the full next-token distribution. It contributes to the index only when a matched control set is supplied; otherwise, the raw distribution remains visible.

Canonical order compares sequence likelihood in published order with seeded within-shard permutations.

## What the score means

The index combines only methods that produced calibrated or internally comparative evidence. Guided, calibrated Min-K%++, and canonical order receive base weights of 0.5, 0.3, and 0.2, with active weights renormalized. Scores below 0.2 are low evidence, scores from 0.2 to below 0.5 are moderate evidence, and scores of 0.5 or higher are high evidence.

Scores are signals with uncertainty, not proof that an example was or was not in a training corpus.

## Limitations

Results depend on model capability, prompts, tokenization, sample size, and control quality. A weak model can fail to reproduce memorized text. A capable model can produce overlap without memorization. Min-K scores shift across domains and model families, while order tests can respond to ordinary document coherence. Paraphrasing, deduplication, fine-tuning, and deliberate evasion can all reduce detection.

## License

MIT License, copyright 2026 Hari Krishna Kancharla.
