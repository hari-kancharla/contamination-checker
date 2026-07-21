# Scoring real externally labeled checkpoints

`experiments/real_score_engine.py` scores released data against an existing local
checkpoint. It never assigns exposure labels, fine-tunes a model, starts a
server, executes upstream scripts, or downloads weights. Mathematical checks
are engineering checks; they are not evidence of membership detection.

The intended targets are the official OLMo 2 1B Instruct checkpoint and the
precisely corresponding Pythia checkpoint for an independently released
CheckMIABench split. Benchmark labels and checkpoint mapping must be verified
separately. The earlier tiny-model/GSM8K experiments remain incremental-exposure
pilots and do not become pretraining-membership evidence through this engine.

## Frozen author convention

The implementation follows inspected source at OLMo-Detect commit
`6eb88de34a09dea85c4b41deff4048e5a203e1b0`, specifically
[the plain scorer](https://github.com/LuckyDaydreamer/OLMo-Detect/blob/6eb88de34a09dea85c4b41deff4048e5a203e1b0/model_utils.py)
and [Min-K / Min-K%++](https://github.com/LuckyDaydreamer/OLMo-Detect/blob/6eb88de34a09dea85c4b41deff4048e5a203e1b0/methods/minkprob_method.py).
The current source's plain scoring path actually crops to **4,096 input tokens**.
A method constructor's nominal 1,024-token setting does not govern this path.
Using a smaller explicitly recorded cap is a prefix experiment, not reproduction
of the full author input convention.

For OLMo plaintext the engine:

1. Applies `str.strip()` as the upstream `data_utils.extract_text` does, then
   tokenizes with `add_special_tokens=True`, right truncation and
   the declared cap (4,096 by default), without applying a chat template. It
   records the original text hash separately from the processed text hash.
2. Aligns input positions `0…T−2` with causal targets `1…T−1`.
3. Removes a final EOS target, retaining the original model input sequence.
4. Rejects any other special target token and empty target support.

For CheckMIABench the engine instead consumes the released integer token IDs
directly. It never decodes and re-encodes them. Original EOS and other training
sequence tokens remain targets. With original sequence `s`, the forwarded
input is `s[:-1]` and the targets are `s[1:]`: all original causal targets are
retained, including the last token, which is an external next-token target.
Thus a released 2,049-ID sequence uses 2,048 inputs and scores 2,048 targets,
within the configured Pythia context. The unused final query is omitted.
This is equivalent under causal attention in exact arithmetic; bitwise
full-input hardware-kernel agreement is not assumed.

The full original IDs, their hash and count remain in the output. No prefix
is dropped unless an explicit cap is supplied; `max_tokens` now counts
**forwarded input tokens**, retaining up to `max_tokens + 1` source IDs and
`max_tokens` targets. Exceeding the model context otherwise fails.
Corpus/model/split identity, packing semantics and membership labels remain
external evidence. This v3 input path is a declared causal-target equivalent
path, rather than the historical Pandora full-sequence invocation.
`target_sequence_positions` index the original selected source sequence;
`logit_positions` index the aligned model output. The final target position
2,048 therefore uses logit position 2,047 for a 2,049-ID original sequence.

## Four same-forward variants

Every record uses one batch-one forward pass with caching disabled. All variants
use the same retained inputs, target IDs, and model-returned logits.

| Variant | Log-softmax / probabilities | Moments and variance | Normalized score |
| --- | --- | --- | --- |
| `author_native` | Incoming logit dtype | Native second moment minus squared mean | Native dtype |
| `fp32_second_moment` | FP32 | FP32 second moment minus squared mean | FP32 |
| `fp32_centered` | Same FP32 values as above | FP32 weighted squared centered deviations | FP32 |
| `fixed_fp32_logp_bf16_reductions` | Same FP32 values, moment products and FP32 sums as above | Explicit BF16 casts after the sums, then BF16 variance, clamp and square root | FP32, using the unchanged FP32 target log-probabilities |

For each target, the mean is `Σ p log(p)`. The author variance is
`Σ p log(p)² − mean²`. Both variance forms use the same `1e-9` floor and the
same additional `1e-9` in the normalization denominator. Thus the two FP32
variants change only the variance calculation. The fixed-logp diagnostic holds
FP32 log-probabilities, probabilities, products, and target scores fixed while
changing moment-result quantization and subsequent arithmetic; it is deliberately
**not** the author's native pipeline. The engine first computes each sum in
FP32, then explicitly casts its result to BF16; it does not use
`torch.sum(dtype=bfloat16)`, which would cast operands before reduction. This
experiment does not observe internal kernel accumulation. Native-versus-FP32 also
changes log-softmax precision and cannot by itself isolate a moment mechanism.

Engine version `contamcheckr.real-score.v2` corrects the fourth diagnostic in
v1, which used `sum(dtype=bfloat16)` and therefore quantized the moment-product
operands before reduction as well. That v1 variant is superseded and must not
be described as isolating result quantization. Frozen v1 runs remain historical
artifacts; changing this source does not update their scores. V2 also adds
upstream text stripping to the public plaintext preparation helper. A v1 run
whose orchestrator already stripped every text used the same inputs.

Version `contamcheckr.real-score.v3` adds the external causal-target input
layout for original token sequences and the optional Pandora comparators
below. It preserves the four weighted variants' formulas and the OLMo
plaintext/EOS convention. Earlier frozen runs are not changed.

Every variant reports:

- LOSS: mean target log-probability, with higher values more member-like.
- Min-K: the mean of the smallest `max(1, floor(0.2 × target_count))` target
  log-probabilities, with the ratio explicitly recorded.
- Min-K%++: the same floor selection over normalized target log-probabilities.
- Zlib: the sum of scored target log-probabilities divided by the compressed
  byte count of the full stripped source text, using Python's default zlib level 6.
  This follows the pinned OLMo baseline; it is withheld when source text is
  unavailable. The denominator remains the full text even when model inputs
  are explicitly cropped, and that convention must stay visible.

The production checker's older ceil selection, centered variance and Zlib
definition are preserved. They are not silently substituted into this study.
Reference-model likelihood differences require a separately verified reference
checkpoint and matching exact inputs. They are not the same as a before/after
fine-tuning delta, and are not synthesized by this engine.

## Optional Pandora same-forward comparators

The pinned [Pandora LOSS implementation](https://github.com/safr-ai-lab/pandora_llm/blob/189a1f43fed580f002ebba3c894c8509f9da14b0/src/pandora_llm/features/LOSS.py)
computes `torch.std_mean(log_probs, dim=-1)` with the default `correction=1`.
Its [MinKXX reduction](https://github.com/safr-ai-lab/pandora_llm/blob/189a1f43fed580f002ebba3c894c8509f9da14b0/src/pandora_llm/features/MinKXX.py)
normalizes by `(target_logp - mean) / (std + 1e-8)`. The mean and standard
deviation are **unweighted across vocabulary entries**. This differs from
the probability-weighted normalization of the four variants above. This
source observation does not establish the exact historical paper pipeline.

With `include_pandora=True`, the engine computes `native` and `fp32`
comparators from the same aligned forward logits and respective log-softmax
values. There is no variance clamp. The pinned
[MinK reduction](https://github.com/safr-ai-lab/pandora_llm/blob/189a1f43fed580f002ebba3c894c8509f9da14b0/src/pandora_llm/features/MinK.py)
sorts z-scores, selects `floor(ratio * target_count)` and computes
`selected.sum() / selected_count`, preserving that arithmetic rather than
substituting `mean()`. The comparator ratio is separately recorded and
defaults to 10%; the weighted variants retain their separate 20% default.
A zero selected count is invalid, without a minimum-one adjustment.

Outputs are under `pandora_comparators.native` and `.fp32`, with per-token
log-probability, unweighted mean, standard deviation, z-score and support.
`scores.pandora_minkxx` is the mean tail z-score, with higher values treated
as more member-like. `scores.pandora_minkxx_feature` is its negative, preserving
the original feature's lower-is-member orientation. Neither is named `minkpp`.
Invalid comparator support, selection or aggregates fail the overall record.

These compare normalization and reduction on a shared forward pass. They do
not reproduce every historical loading or masking choice. The engine forwards
an all-ones attention mask. Pandora's helper defaults to `input_ids > 0` when
callers provide no mask; whether a particular upstream dataloader supplied a
mask must be established separately. No full historical replication is claimed.

## Evidence and resource bounds

Per-token log-probability, mean, raw variance, clamped sigma, normalized score,
clamp mask and support mask are saved for every variant. Original/retained input
hashes, exact input/target IDs, selected indices, context convention, component
dtypes, device and model identity are retained. Nonfinite observations are
explicit nulls plus a support mask. A variant with incomplete support receives
no aggregate scores; the CLI retains its record and marks the run failed.

Vocabulary statistics run in bounded token chunks (64 by default). Model
forward logits remain resident: chunking statistics does not make that tensor
disappear. At vocabulary size 100,352, batch-one 4,096-token logits require about
1.53 GiB in FP32 or 0.77 GiB in BF16, before weights, attention and workspaces.
All model/tokenizer loading is local-only and custom model code is disabled.
The engine does not change MPS safety caps or enable silent CPU fallback.

An optional `cpu_float64_centered` diagnostic copies each bounded logit chunk
to CPU and computes higher-precision statistics there. It never allocates an
FP64 tensor on MPS: transfer to CPU completes before its separate FP64 cast.
This diagnoses statistics conditional on returned logits;
it cannot undo model-forward rounding.

Record forward and statistics timings, process peak RSS and available MPS
allocation metrics on **real development records** before freezing the full
study's resource budget. GPU timings synchronize the selected device. Process
peak RSS includes earlier allocations and is not a model-only memory estimate.
Vector artifacts can be large; gzip is optional. No throughput or full-study
memory claim is supported until those actual measurements exist.

## API and command line

The orchestration API is `load_local_checkpoint`, `prepare_olmo_plaintext`,
`prepare_checkmiabench_tokens`, and `score_prepared`. `score_logits_variants`
also accepts already-aligned logits for checks against inspected formulas.
Both scoring functions accept `include_pandora=True, pandora_k_percent=10.0`;
the weighted ratio remains the independent `k_percent` argument. The CLI uses
`--include-pandora --pandora-k-percent 10`. After the score handle closes,
the generic CLI stores its final bytes and SHA256 in `execution.json`, including
partial artifacts from failed runs.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/real_score_engine.py \
  --model-path /absolute/path/to/pinned/local/checkpoint \
  --input /absolute/path/to/frozen-real-development-records.jsonl \
  --input-kind olmo_plaintext --device mps --model-dtype bfloat16 \
  --output experiments/results/real-development-parity-new \
  --chunk-tokens 64 --k-percent 20 --gzip-vectors
```

CheckMIABench input uses `--input-kind checkmiabench_tokens` and a `tokens` field.
Field names can be declared explicitly. Any `--limit` and prefix cap appear in
the execution manifest. Existing output directories are rejected. Source and
input hashes, environment versions, declared settings and failures remain
visible. The loader's metadata signature is not a cryptographic weight hash;
attach the separately verified official download receipt.

Before interpreting actual test scores, verify exact input/target agreement,
full support, native/BF16 versus FP32 differences, CPU/MPS forward parity, and
the correspondence between released author vectors and the selected source
revision. Prespecify thresholds and sample selection using development data.
Agreement of implementations does not validate membership labels, scientific
novelty, false-positive calibration, or deployment usefulness.
