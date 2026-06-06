# Controlled validation

## Purpose

This experiment checks whether ContamCheckr assigns a higher score after a model is trained on a known set of examples. It is a controlled memorization experiment, not a claim that any real model or public benchmark is contaminated.

## Method

`experiments/validation.py` creates two deterministic datasets with 40 examples each. Both use the same synthetic archive-card style, but their identifiers and sentences do not overlap. The script writes them to `experiments/data/leak_set.jsonl` and `experiments/data/clean_set.jsonl`.

The recorded run used seed 17 and `sshleifer/tiny-gpt2`. A copy of the model was trained only on the leak set for 60 epochs with AdamW, a learning rate of 0.005, and a batch size of 4. Training used 40 individual examples plus five ordered eight-example records. Loss fell from 10.818 on the first update to 3.452 on the last update. The untouched base model served as the model control.

All three audits went through the normal `build_report` pipeline. The clean set was supplied as the matched control so that Min-K%++ calibration was active:

1. Adapted model on the leak set
2. Adapted model on the clean set
3. Base model on the leak set

The script falls back to `distilgpt2` when the positive index is below 0.4 or is less than 0.2 above the strongest control. The tiny model cleared both thresholds, so the fallback was not used.

## Reproduce

From the repository after installing its existing dependencies:

```bash
.venv/bin/python experiments/validation.py
```

The captured run used the cached model:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python experiments/validation.py
```

The script saves the adapted model under `experiments/models/` and overwrites the reports under `experiments/results/`.

## Results

| Run | Index | Band | 95% interval | Guided index | Guided gap | Min-K%++ index | Min-K%++ AUC | Order index | Order p-value |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adapted model on leak set | 0.492 | moderate evidence | 0.354 to 0.492 | 0.000 | 0.000 | 1.000 | 1.000 | 0.962 | 0.038 |
| Adapted model on clean set | 0.008 | low evidence | 0.000 to 0.103 | 0.000 | 0.000 | 0.000 | 0.500 | 0.038 | 0.962 |
| Base model on leak set | 0.131 | low evidence | 0.008 to 0.170 | 0.000 | 0.000 | 0.000 | 0.397 | 0.654 | 0.346 |

The adapted model on the leak set scored 0.362 above the strongest control. Min-K%++ perfectly separated the trained examples from the matched clean set, and the order test also favored the trained leak set.

The result does not show that every method detected memorization. Guided overlap contributed zero in all three runs, and the positive result remained just below the high-evidence boundary of 0.5. The data are synthetic, the model is unusually small, and this is one seeded training run. The experiment establishes an end-to-end control response at this scale; it does not estimate performance on natural benchmarks or larger models.

## Artifacts

- [Comparison CSV](../experiments/results/comparison.csv)
- [Adapted model on leak set](../experiments/results/contaminated_on_leak.json)
- [Adapted model on clean set](../experiments/results/contaminated_on_clean.json)
- [Base model on leak set](../experiments/results/base_on_leak.json)
- [Run metadata](../experiments/results/metadata.json)
