from dataclasses import dataclass
import csv
import json
from pathlib import Path
import sys
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


@dataclass(frozen=True)
class BenchmarkData:
    examples: list[dict[str, str]]
    source: str
    dataset_name: str
    split: str


def _sample_path(filename: str) -> Path:
    source_path = Path(__file__).resolve().parent.parent / "data" / filename
    if source_path.exists():
        return source_path
    installed_path = (
        Path(sys.prefix) / "share" / "contamcheckr" / "data" / filename
    )
    if installed_path.exists():
        return installed_path
    raise ValueError(
        "Bundled sample data is missing. Reinstall contamcheckr to restore it."
    )


def _resolve_text(record: dict[str, Any], text_field: str, row: int) -> str | None:
    if text_field in record:
        value = record[text_field]
    elif text_field == "text":
        aliases = ("prompt", "question", "answer")
        values = [
            str(record[name]).strip()
            for name in aliases
            if record.get(name) is not None and str(record[name]).strip()
        ]
        if not values:
            available = ", ".join(sorted(record)) or "none"
            raise ValueError(
                f"Row {row} has no 'text' field or usable text alias. "
                f"Available fields: {available}."
            )
        value = "\n".join(values)
    else:
        available = ", ".join(sorted(record)) or "none"
        raise ValueError(
            f"Row {row} has no text field named '{text_field}'. "
            f"Available fields: {available}."
        )
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_records(
    records: Any,
    *,
    text_field: str,
    limit: int,
    allow_generated_id: bool = False,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for row, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Row {row} is not an object.")
        text = _resolve_text(record, text_field, row)
        if text is None:
            continue
        if "id" in record:
            identifier = record["id"]
        elif "idx" in record:
            identifier = record["idx"]
        elif allow_generated_id:
            identifier = row
        else:
            raise ValueError(f"Row {row} has no required 'id' field.")
        identifier_text = str(identifier).strip()
        if not identifier_text:
            raise ValueError(f"Row {row} has an empty id.")
        examples.append({"id": identifier_text, "text": text})
        if len(examples) >= limit:
            break
    if not examples:
        raise ValueError("The benchmark contains no non-empty text examples.")
    return examples


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for row, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {row} of {path}: {error.msg}."
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Line {row} of {path} must contain a JSON object."
                    )
                records.append(value)
    except OSError as error:
        raise ValueError(f"Could not read benchmark file {path}: {error}.") from error
    return records


def _load_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise ValueError(f"Could not read benchmark file {path}: {error}.") from error


def _select_hf_split(dataset: Dataset | DatasetDict, requested: str) -> tuple[Dataset, str]:
    if isinstance(dataset, Dataset):
        return dataset, requested
    if requested in dataset:
        return dataset[requested], requested
    if requested == "test":
        for candidate in ("validation", "train"):
            if candidate in dataset:
                return dataset[candidate], candidate
    available = ", ".join(dataset.keys())
    raise ValueError(
        f"Dataset split '{requested}' was not found. Available splits: {available}."
    )


def load_benchmark(
    source: str,
    *,
    text_field: str = "text",
    dataset_config: str | None = None,
    split: str = "test",
    limit: int = 200,
    sample_control: bool = False,
) -> BenchmarkData:
    if limit < 1:
        raise ValueError("Limit must be at least 1.")
    if source == "sample":
        filename = "sample_control.jsonl" if sample_control else "sample_benchmark.jsonl"
        path = _sample_path(filename)
        records = _load_jsonl(path)
        return BenchmarkData(
            examples=_normalize_records(records, text_field=text_field, limit=limit),
            source="sample",
            dataset_name="contamcheckr sample control" if sample_control else "contamcheckr sample",
            split="sample",
        )

    path = Path(source).expanduser()
    if path.exists():
        suffix = path.suffix.casefold()
        if suffix == ".jsonl":
            records = _load_jsonl(path)
        elif suffix == ".csv":
            records = _load_csv(path)
        else:
            raise ValueError("Local benchmarks must use .jsonl or .csv.")
        return BenchmarkData(
            examples=_normalize_records(records, text_field=text_field, limit=limit),
            source=str(path.resolve()),
            dataset_name=path.stem,
            split=split,
        )
    if path.suffix.casefold() in {".jsonl", ".csv"}:
        raise ValueError(f"Benchmark file does not exist: {path}.")

    try:
        loaded = load_dataset(source, dataset_config)
    except (OSError, ValueError, RuntimeError) as error:
        raise ValueError(f"Could not load HuggingFace dataset '{source}': {error}.") from error
    selected, resolved_split = _select_hf_split(loaded, split)
    return BenchmarkData(
        examples=_normalize_records(
            selected,
            text_field=text_field,
            limit=limit,
            allow_generated_id=True,
        ),
        source=source,
        dataset_name=source,
        split=resolved_split,
    )
