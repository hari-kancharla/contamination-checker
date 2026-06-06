from argparse import ArgumentParser
import csv
import gc
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from contamcheck.benchmarks import BenchmarkData, load_benchmark
from contamcheck.config import RunConfig
from contamcheck.model import (
    LocalCausalLM,
    load_model,
    resolve_device,
    set_seed,
)
from contamcheck.report import build_report, write_json


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
SEPARATOR = "\n\n---\n\n"


def build_examples(prefix: str, offset: int) -> list[dict[str, str]]:
    colors = ["amber", "blue", "copper", "green", "ivory"]
    objects = ["dial", "lever", "marker", "panel"]
    directions = ["north", "east", "south", "west"]
    times = ["dawn", "noon", "dusk", "midnight"]
    verbs = ["faces", "meets", "passes", "reaches"]
    examples: list[dict[str, str]] = []
    for index in range(40):
        number = offset + index
        color = colors[(index * 3 + offset) % len(colors)]
        object_name = objects[(index + offset) % len(objects)]
        direction = directions[(index * 3 + offset) % len(directions)]
        time = times[(index * 5 + offset) % len(times)]
        verb = verbs[(index * 7 + offset) % len(verbs)]
        code = f"{prefix}-{number:03d}"
        text = (
            f"Archive card {code} records that the {color} {object_name} "
            f"{verb} the {direction} mark before the {time} inspection."
        )
        examples.append({"id": code.lower(), "text": text})
    random.Random(offset).shuffle(examples)
    return examples


def write_jsonl(path: Path, examples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")


def prepare_data() -> tuple[Path, Path]:
    leak_path = DATA_DIR / "leak_set.jsonl"
    clean_path = DATA_DIR / "clean_set.jsonl"
    write_jsonl(leak_path, build_examples("LQ", 100))
    write_jsonl(clean_path, build_examples("CQ", 600))
    return leak_path, clean_path


def encoded_record(
    tokenizer: Any,
    prefix: str,
    target: str,
    *,
    max_length: int,
) -> tuple[list[int], list[int]]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        target_ids.append(eos_id)
    input_ids = (prefix_ids + target_ids)[:max_length]
    prefix_length = min(len(prefix_ids), len(input_ids))
    labels = [-100] * prefix_length + input_ids[prefix_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("A training record was truncated before its target text.")
    return input_ids, labels


def training_records(
    tokenizer: Any,
    examples: list[dict[str, str]],
    *,
    max_length: int,
) -> list[tuple[list[int], list[int]]]:
    records = [
        encoded_record(
            tokenizer,
            "",
            example["text"],
            max_length=max_length,
        )
        for example in examples
    ]
    for start in range(0, len(examples), 8):
        shard = SEPARATOR.join(
            example["text"] for example in examples[start : start + 8]
        )
        records.append(
            encoded_record(
                tokenizer,
                "",
                shard,
                max_length=max_length,
            )
        )
    return records


def collate(
    batch: list[tuple[list[int], list[int]]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(input_ids) for input_ids, _ in batch)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    for input_ids, labels in batch:
        padding = width - len(input_ids)
        input_rows.append(input_ids + [pad_token_id] * padding)
        label_rows.append(labels + [-100] * padding)
        attention_rows.append([1] * len(input_ids) + [0] * padding)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(attention_rows, dtype=torch.long, device=device),
        torch.tensor(label_rows, dtype=torch.long, device=device),
    )


def fine_tune(
    model_id: str,
    output_dir: Path,
    examples: list[dict[str, str]],
    *,
    device_name: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
) -> dict[str, float | int | str]:
    set_seed(seed)
    device = resolve_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The validation tokenizer has no pad or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.to(device)
    model.train()
    model.config.use_cache = False
    records = training_records(tokenizer, examples, max_length=max_length)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    randomizer = random.Random(seed)
    losses: list[float] = []
    progress = tqdm(
        total=epochs * ((len(records) + batch_size - 1) // batch_size),
        desc=f"Training {model_id}",
    )
    for _ in range(epochs):
        order = list(range(len(records)))
        randomizer.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [records[index] for index in order[start : start + batch_size]]
            input_ids, attention_mask, labels = collate(
                batch,
                tokenizer.pad_token_id,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError("Validation training produced a non-finite loss.")
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            progress.set_postfix(loss=f"{losses[-1]:.3f}")
            progress.update()
    progress.close()
    model.config.use_cache = True
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.to("cpu")
    del optimizer
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {
        "model_id": model_id,
        "device": str(device),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "record_count": len(records),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_last_ten_losses": float(np.mean(losses[-10:])),
    }


def report_config(
    model_id: str,
    benchmark: BenchmarkData,
    control: BenchmarkData,
    *,
    device: str,
    seed: int,
) -> RunConfig:
    return RunConfig(
        model_id=model_id,
        benchmark_source=benchmark.source,
        control_source=control.source,
        split=benchmark.split,
        limit=len(benchmark.examples),
        device=device,
        seed=seed,
        shard_count=5,
        permutation_count=25,
        max_tokens=256,
    )


def run_report(
    model: LocalCausalLM,
    benchmark: BenchmarkData,
    control: BenchmarkData,
    *,
    model_id: str,
    device: str,
    seed: int,
) -> dict[str, Any]:
    return build_report(
        model,
        benchmark,
        control,
        report_config(
            model_id,
            benchmark,
            control,
            device=device,
            seed=seed,
        ),
    )


def move_model_to_cpu(model: LocalCausalLM) -> None:
    model.model.to("cpu")
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def evaluate_candidate(
    model_id: str,
    leak: BenchmarkData,
    clean: BenchmarkData,
    *,
    device: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    safe_name = model_id.replace("/", "--")
    adapted_path = MODELS_DIR / safe_name
    training = fine_tune(
        model_id,
        adapted_path,
        leak.examples,
        device_name=device,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length,
    )

    adapted = load_model(str(adapted_path), device=device, seed=seed)
    contaminated_on_leak = run_report(
        adapted,
        leak,
        clean,
        model_id=str(adapted_path),
        device=device,
        seed=seed,
    )
    contaminated_on_clean = run_report(
        adapted,
        clean,
        clean,
        model_id=str(adapted_path),
        device=device,
        seed=seed,
    )
    move_model_to_cpu(adapted)
    del adapted
    gc.collect()

    base = load_model(model_id, device=device, seed=seed)
    base_on_leak = run_report(
        base,
        leak,
        clean,
        model_id=model_id,
        device=device,
        seed=seed,
    )
    move_model_to_cpu(base)
    del base
    gc.collect()

    reports = {
        "contaminated_on_leak": contaminated_on_leak,
        "contaminated_on_clean": contaminated_on_clean,
        "base_on_leak": base_on_leak,
    }
    positive = contaminated_on_leak["contamination_index"]
    strongest_negative = max(
        contaminated_on_clean["contamination_index"],
        base_on_leak["contamination_index"],
    )
    minimum_positive_index = 0.4
    minimum_control_separation = 0.2
    separation = positive - strongest_negative
    metadata = {
        "training": training,
        "adapted_model": str(adapted_path),
        "positive_index": positive,
        "strongest_negative_index": strongest_negative,
        "separation": separation,
        "validation_thresholds": {
            "minimum_positive_index": minimum_positive_index,
            "minimum_control_separation": minimum_control_separation,
        },
        "clear_separation": (
            positive >= minimum_positive_index
            and separation >= minimum_control_separation
        ),
    }
    return reports, metadata


def comparison_rows(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    labels = {
        "contaminated_on_leak": "adapted model on leak set",
        "contaminated_on_clean": "adapted model on clean set",
        "base_on_leak": "base model on leak set",
    }
    rows: list[dict[str, Any]] = []
    for name, report in reports.items():
        rows.append(
            {
                "run": name,
                "description": labels[name],
                "contamination_index": report["contamination_index"],
                "band": report["band"],
                "index_guided": report["index_guided"],
                "guided_general_gap": report["guided_general_gap"],
                "index_minkpp": report["index_minkpp"],
                "minkpp_auc": report["minkpp_auc"],
                "index_order": report["index_order"],
                "order_p_value": report["order_p_value"],
            }
        )
    return rows


def write_results(
    reports: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        write_json(report, RESULTS_DIR / f"{name}.json")
    rows = comparison_rows(reports)
    with (RESULTS_DIR / "comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_comparison(reports: dict[str, dict[str, Any]]) -> None:
    print(
        "run                         index   band               "
        "guided  minkpp  order"
    )
    for row in comparison_rows(reports):
        print(
            f"{row['run']:<28} "
            f"{row['contamination_index']:>5.3f}   "
            f"{row['band']:<18} "
            f"{row['index_guided']:>6.3f}  "
            f"{row['index_minkpp']:>6.3f}  "
            f"{row['index_order']:>5.3f}"
        )


def parser() -> ArgumentParser:
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    argument_parser.add_argument("--fallback-model", default="distilgpt2")
    argument_parser.add_argument("--device", default="auto")
    argument_parser.add_argument("--seed", type=int, default=17)
    argument_parser.add_argument("--epochs", type=int, default=60)
    argument_parser.add_argument("--fallback-epochs", type=int, default=12)
    argument_parser.add_argument("--learning-rate", type=float, default=0.005)
    argument_parser.add_argument(
        "--fallback-learning-rate",
        type=float,
        default=0.0002,
    )
    argument_parser.add_argument("--batch-size", type=int, default=4)
    argument_parser.add_argument("--max-length", type=int, default=256)
    argument_parser.add_argument("--no-fallback", action="store_true")
    return argument_parser


def main() -> int:
    args = parser().parse_args()
    leak_path, clean_path = prepare_data()
    leak = load_benchmark(str(leak_path), split="validation", limit=40)
    clean = load_benchmark(str(clean_path), split="validation", limit=40)
    candidates = [
        (
            args.model,
            args.epochs,
            args.learning_rate,
        )
    ]
    if not args.no_fallback and args.fallback_model != args.model:
        candidates.append(
            (
                args.fallback_model,
                args.fallback_epochs,
                args.fallback_learning_rate,
            )
        )

    selected_reports: dict[str, dict[str, Any]] | None = None
    selected_metadata: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    for model_id, epochs, learning_rate in candidates:
        print(f"\nValidating with {model_id}")
        try:
            reports, metadata = evaluate_candidate(
                model_id,
                leak,
                clean,
                device=args.device,
                seed=args.seed,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
        except (OSError, RuntimeError, ValueError) as error:
            attempts.append({"model_id": model_id, "error": str(error)})
            print(f"{model_id} could not complete: {error}", file=sys.stderr)
            continue
        attempts.append(
            {
                "model_id": model_id,
                "separation": metadata["separation"],
                "positive_index": metadata["positive_index"],
                "clear_separation": metadata["clear_separation"],
            }
        )
        selected_reports = reports
        selected_metadata = metadata
        print_comparison(reports)
        if metadata["clear_separation"]:
            break
        if model_id != candidates[-1][0]:
            print(
                "The separation was below the validation threshold. "
                f"Trying {candidates[-1][0]}."
            )

    if selected_reports is None or selected_metadata is None:
        raise RuntimeError("No validation model completed successfully.")
    selected_metadata["attempts"] = attempts
    selected_metadata["seed"] = args.seed
    selected_metadata["leak_set"] = str(leak_path)
    selected_metadata["clean_set"] = str(clean_path)
    write_results(selected_reports, selected_metadata)
    print("\nSelected validation result")
    print_comparison(selected_reports)
    if not selected_metadata["clear_separation"]:
        print(
            "The controlled effect is small at this model scale. "
            "The saved reports contain the measured values."
        )
    print(f"Results written to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
