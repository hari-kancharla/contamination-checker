# Reproduction scope

Install the toolkit and optional research dependencies:

```sh
python -m pip install -e '.[dev,research]'
python -m pytest
python experiments/research_sources.py --list
```

The source helper defaults to offline verification. Missing files fail verification; existing mismatched files are not replaced. An explicit download retrieves the pinned public sources and checkpoints into the ignored local cache:

```sh
python experiments/research_sources.py --download --asset olmo-detect
python experiments/research_sources.py --asset olmo-detect
```

The selected OLMo source archive does not require model weights. Other optional assets include multi-gigabyte checkpoints. Source acquisition does not run models, execute downloaded research scripts or handle gated dataset access. Keep third-party models and datasets under their own terms; this project's MIT license does not relicense them.

The OLMo release audit and source-based selectivity study can be regenerated from those pinned public sources. Use fresh output directories:

```sh
OLMO=.cache/research/olmo_detect/OLMo-Detect-6eb88de34a09dea85c4b41deff4048e5a203e1b0
python experiments/olmo_release_audit.py --snapshot "$OLMO" \
  --receipt .cache/research/olmo_detect/source_receipt.json \
  --output experiments/results/olmo-new
python experiments/real_selectivity_study.py seal --snapshot "$OLMO" \
  --audit-dir experiments/results/olmo-new \
  --protocol experiments/protocols/real-data-selectivity-v1.json \
  --output experiments/results/selectivity-new
python experiments/real_selectivity_study.py evaluate \
  --output experiments/results/selectivity-new
```

These steps analyze released scores; they are not new inference over every model. Output is kept local and ignored by Git.

## Historical CheckMIA execution

The archived full scoring runner, device checks and reconstruction tools require the original sealed inputs, failed-development execution and amendment evidence. That private capsule is not included: it contains third-party corpus material and machine-specific runtime records. The original amendment is deliberately not edited or redistributed as a fabricated portable original. Consequently, this repository alone does not reproduce the complete historical 3,000-forward run. Published [method summaries](../research/results/checkmia-methods.json), [contrasts](../research/results/checkmia-contrasts.json) and their [provenance](../research/PROVENANCE.json) expose the retained aggregates and their origin hashes without raw data.

The recorded model run used Python 3.12.14, PyTorch 2.12.0, transformers 5.10.2, NumPy 2.4.6 and Apple MPS/FP16/SDPA. A fresh CPU/CUDA or precision variant is a different execution; matching dependency versions does not imply bitwise parity. These commands do not rerun the historical CheckMIA study; optional cached-model tests may run local inference.
