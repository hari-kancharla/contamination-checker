# Public benchmark example

This audit was run on June 6, 2026 with:

- Model: `distilgpt2`
- Benchmark: `nyu-mll/glue`, configuration `sst2`
- Split: `validation`
- Examples: first 10 non-empty rows
- Device: CPU
- Seed: 0

`distilgpt2` was the largest recognizable causal model already cached in this environment, and the full audit completed on CPU.

## Result

| Field | Value |
| --- | ---: |
| Contamination index | 0.275 |
| Band | moderate evidence |
| Bootstrap interval | 0.000 to 0.275 |
| Index basis | guided_plus_order |
| Guided ROUGE-L | 0.023 |
| General ROUGE-L | 0.034 |
| Guided-over-general gap | -0.011 |
| Guided index | 0.000 |
| Canonical-order index | 0.962 |
| Canonical-order p-value | 0.038 |
| Calibrated Min-K%++ index | unavailable |

This is weak, uncertain evidence rather than a contamination finding. The interval includes zero, guided prompting performed slightly worse than general prompting, and Min-K%++ had no matched control. The moderate score came entirely from the canonical-order test over five two-example shards. With 25 permutations, 0.038 is the smallest possible p-value, so this small sample should not carry much weight.

## Reproduce

The recorded command used the already cached model and dataset:

```bash
HF_HUB_OFFLINE=1 .venv/bin/contamcheck run --model distilgpt2 --benchmark nyu-mll/glue --dataset-config sst2 --split validation --text-field sentence --limit 10 --device cpu --output docs/example_report.json --csv docs/example_per_example.csv --html docs/example_report.html
```

Remove `HF_HUB_OFFLINE=1` when the model or dataset is not cached.

Captured artifacts:

- [JSON report](example_report.json)
- [Per-example CSV](example_per_example.csv)
- [Standalone HTML report](example_report.html)

For a larger GPU run with a less fragile order test:

```bash
contamcheck run --model EleutherAI/pythia-1.4b --benchmark nyu-mll/glue --dataset-config sst2 --split validation --text-field sentence --limit 100 --device cuda --output pythia_glue_report.json --csv pythia_glue_per_example.csv --html pythia_glue_report.html
```

A larger model and sample improve the audit conditions, but they do not guarantee a higher contamination score.

The recorded environment used Python 3.12.13, torch 2.12.0, transformers 5.10.2, datasets 5.0.0, and numpy 2.4.6.
