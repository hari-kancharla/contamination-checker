"""Hardened null-behavior check: does the audit separate hardened sets?

Scores hardened leak-vs-clean (and, for comparison, the legacy
confounded generator's leak-vs-clean) with Min-K%++ on two base models
with no training:

* ``sshleifer/tiny-gpt2`` (loaded offline from the HF cache)
* ``experiments/models/distilgpt2`` (local copy)

Under no training, leak and clean are exchangeable up to the template
draw rule, so the Min-K%++ AUC should be near 0.5 if the frequency
confound is gone.  The legacy comparison is expected to stay far from
0.5 for distilgpt2 (0.84 in the frozen docs).

The module only defines constants and functions on import; loading a
model happens inside :func:`main`, so an import check stays fast and
offline.

Usage (offline):
  HF_HUB_OFFLINE=1 PYTHONPATH=.:experiments \\
      .venv/bin/python experiments/hardened_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before any Hugging Face / transformers import so the
# runner never touches the network.  ``local_files_only=True`` is also
# passed to ``load_model`` for the same reason.
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent  # experiments/
PROJECT_ROOT = ROOT.parent

for _candidate in (str(PROJECT_ROOT), str(ROOT)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

try:  # PYTHONPATH=experiments or script-directory import
    from generators_hardened import build_examples_hardened
except ImportError:  # package import from the project root
    from experiments.generators_hardened import build_examples_hardened

try:
    from validation import build_examples
except ImportError:
    from experiments.validation import build_examples

from contamcheckr.calibration import calibrate_minkpp
from contamcheckr.model import load_model
from contamcheckr.methods.minkpp import score_text_batch

TINY_MODEL_ID = "sshleifer/tiny-gpt2"
DISTILGPT2_PATH = str(ROOT / "models" / "distilgpt2")
MODEL_IDS: list[str] = [TINY_MODEL_ID, DISTILGPT2_PATH]
LEAK_PREFIX = "LQ"
CLEAN_PREFIX = "CQ"
LEAK_OFFSET = 100
CLEAN_OFFSET = 600
SEED = 17
MAX_TOKENS = 256
BATCH_SIZE = 4


def minkpp_auc(model, benchmark_texts: list[str], control_texts: list[str]) -> float | None:
    """Score two text sets with Min-K%++ and return the AUC (benchmark high)."""
    benchmark_scores = [
        score.minkpp
        for score in score_text_batch(
            model, benchmark_texts, max_tokens=MAX_TOKENS, batch_size=BATCH_SIZE
        )
    ]
    control_scores = [
        score.minkpp
        for score in score_text_batch(
            model, control_texts, max_tokens=MAX_TOKENS, batch_size=BATCH_SIZE
        )
    ]
    _, auc_value = calibrate_minkpp(benchmark_scores, control_scores)
    return auc_value


def _free_model(model) -> None:
    import gc

    import torch

    try:
        model.model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> int:
    hardened_leak = build_examples_hardened(LEAK_PREFIX, LEAK_OFFSET)
    hardened_clean = build_examples_hardened(CLEAN_PREFIX, CLEAN_OFFSET)
    old_leak = build_examples(LEAK_PREFIX, LEAK_OFFSET)
    old_clean = build_examples(CLEAN_PREFIX, CLEAN_OFFSET)

    hardened_leak_texts = [example["text"] for example in hardened_leak]
    hardened_clean_texts = [example["text"] for example in hardened_clean]
    old_leak_texts = [example["text"] for example in old_leak]
    old_clean_texts = [example["text"] for example in old_clean]

    print(
        f"{'model':<42} {'hardened_AUC':>12} {'old_AUC':>10}"
    )
    for model_id in MODEL_IDS:
        print(f"Loading {model_id} (offline)...", flush=True)
        model = load_model(
            model_id,
            device="auto",
            seed=SEED,
            local_files_only=True,
        )
        try:
            hardened_auc = minkpp_auc(
                model, hardened_leak_texts, hardened_clean_texts
            )
            old_auc = minkpp_auc(model, old_leak_texts, old_clean_texts)
        finally:
            _free_model(model)
        hardened_text = (
            f"{float(hardened_auc):.3f}" if hardened_auc is not None else "n/a"
        )
        old_text = f"{float(old_auc):.3f}" if old_auc is not None else "n/a"
        print(f"{model_id:<42} {hardened_text:>12} {old_text:>10}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
