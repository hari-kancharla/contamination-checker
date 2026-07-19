#!/usr/bin/env python3
"""Offline audit of the released OLMo-Detect pretraining texts and saved scores.

This program imports no project or upstream code, loads no model, and makes no
network requests. It reconstructs published-score AUCs, not fresh model scores.
Run with --help for inputs. Output directories must not already exist.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import platform
import re
import sys
import tarfile
from typing import Any


REVISION = "6eb88de34a09dea85c4b41deff4048e5a203e1b0"
ARCHIVE_SHA256 = "d7a157423bf537a343975db13bfddd9b271a7c24a9cb7ecc8f6b2342ca8d30ce"
PAPER_URL = "https://arxiv.org/html/2610.02986v1"
SIZES = ("1B", "7B", "13B", "32B")
METHODS = {
    "LOSS": ("loss_zlib_lowercase", "loss_score"),
    "MinK": ("minkprob", "min_k_prob_mean_logprob"),
    "MinK++": ("minkprob", "min_k_pp_score"),
}
DOMAINS = {"dclm": ("",), "pes2o": ("",), "openwebmath": ("",),
           "starcoder": ("java", "python")}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def auc_pairs(positive: list[float], negative: list[float]) -> float | None:
    """P(S_member > S_nonmember) + .5 P(tie), using binary searches."""
    if not positive or not negative:
        return None
    ordered = sorted(negative)
    twice_wins = sum(bisect_left(ordered, p) + bisect_right(ordered, p)
                     for p in positive)
    return twice_wins / (2 * len(positive) * len(negative))


def auc_ranks(positive: list[float], negative: list[float]) -> float | None:
    """Independent Mann-Whitney rank-sum calculation with midranks."""
    if not positive or not negative:
        return None
    combined = sorted([(s, 1) for s in positive] + [(s, 0) for s in negative])
    rank_sum = 0.0
    start = 0
    while start < len(combined):
        stop = start + 1
        while stop < len(combined) and combined[stop][0] == combined[start][0]:
            stop += 1
        midrank = (start + 1 + stop) / 2
        rank_sum += midrank * sum(label for _, label in combined[start:stop])
        start = stop
    n = len(positive)
    return (rank_sum - n * (n + 1) / 2) / (n * len(negative))


def finite_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def count_summary(counts: list[int]) -> dict:
    return {
        "n": len(counts), "min": min(counts) if counts else None,
        "max": max(counts) if counts else None,
        "above_1023": sum(n > 1023 for n in counts),
        "above_1024": sum(n > 1024 for n in counts),
        "equal_4095": counts.count(4095),
        "above_4095": sum(n > 4095 for n in counts),
    }


class PaperTables(HTMLParser):
    """Read table cells as text; never evaluate scripts or imported code."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: dict[str, list[list[str]]] = {}
        self.table_id: str | None = None
        self.row: list[str] = []
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple]) -> None:
        values = dict(attrs)
        if tag == "figure" and "ltx_table" in values.get("class", ""):
            self.table_id = values.get("id")
            self.tables[self.table_id] = []
        if self.table_id and tag == "tr":
            self.row = []
        if self.table_id and tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.table_id:
            self.tables[self.table_id].append(self.row)
        if tag == "figure":
            self.table_id = None


def compare_paper(html: str, metrics: list[dict], aggregates: list[dict]) -> dict:
    """Compare exact reconstructed AUCs to the paper's two-decimal cells.

    Comparison tolerance is half a displayed unit (.005), not an assertion of
    bitwise or model-scoring replication. Table 85 has significance markers;
    only its point differences are checked here, not those significance claims.
    """
    parser = PaperTables()
    parser.feed(html)
    comparisons = []
    method_column = {"LOSS": 1, "MinK": 4, "MinK++": 5}
    domain_tables = {"dclm": "A9.T37", "pes2o": "A9.T38",
                     "starcoder": "A9.T39", "openwebmath": "A9.T40"}

    def add(table: str, descriptor: dict, printed: str, actual: float | None) -> None:
        # arXiv MathML duplicates the sign (e.g. '−-0.06‡'); take magnitude
        # and inspect the prefix. All target cells contain exactly one number.
        number = re.search(r"\d+\.\d+", printed)
        if not number:
            raise ValueError(f"No decimal in paper cell {table}: {printed!r}")
        value = float(number.group())
        if "-" in printed[:number.start()] or "−" in printed[:number.start()]:
            value = -value
        difference = None if actual is None else actual - value
        comparisons.append({"table": table, **descriptor, "paper_cell": printed,
                            "paper_value": value, "reconstructed": actual,
                            "difference": difference,
                            "matches_display_precision": difference is not None and
                            abs(difference) <= .005000000001})

    for row in aggregates:
        if row["kind"] != "domain_macro" or row["condition"] != "matched" or row["split"] != "test":
            continue
        table = domain_tables[row["domain"]]
        size = row["model"].split("-")[1]
        paper_row = next(r for r in parser.tables[table] if r and r[0] == size)
        add(table, {k: row[k] for k in ("method", "model", "domain", "condition", "split")},
            paper_row[method_column[row["method"]]], row["auc"])

    subset_rows: dict[tuple[str, str], list[str]] = {}
    subset = None
    for cells in parser.tables["A9.T46"]:
        if cells and cells[0] in ("Java", "Python"):
            subset = cells[0].lower()
            cells = cells[1:]
        if subset and cells and cells[0] in SIZES:
            subset_rows[(subset, cells[0])] = cells
    for row in metrics:
        if row["domain"] != "starcoder" or row["condition"] != "matched" or row["split"] != "test":
            continue
        size = row["model"].split("-")[1]
        cells = subset_rows[row["subset"], size]
        add("A9.T46", {k: row[k] for k in ("method", "model", "domain", "subset", "condition", "split")},
            cells[method_column[row["method"]]], row["auc"])

    table4 = {r[0].removesuffix("L"): r for r in parser.tables["S4.T4"] if r}
    for row in aggregates:
        if row["kind"] != "pretraining_macro" or row["condition"] != "matched" or row["split"] != "test":
            continue
        size = row["model"].split("-")[1]
        name = "PPL" if row["method"] == "LOSS" else row["method"]
        add("S4.T4", {k: row[k] for k in ("method", "model", "condition", "split")},
            table4[name][1 + 4 * SIZES.index(size)], row["auc"])

    # Table 85 reports the model-averaged domain change, not pooled-score AUC.
    delta_rows = {r[0]: r for r in parser.tables["A15.T85"] if r}
    for method in METHODS:
        paper_method = "PPL" if method == "LOSS" else method
        for domain, column in (("dclm", 1), ("pes2o", 2), ("openwebmath", 3), ("starcoder", 4)):
            selected = [r for r in aggregates if r["kind"] == "domain_macro" and
                        r["domain"] == domain and r["method"] == method and r["split"] == "test"]
            values = {(r["model"], r["condition"]): r["auc"] for r in selected}
            changes = [values[(f"OLMo2-{size}-Instruct", "shifted")] -
                       values[(f"OLMo2-{size}-Instruct", "matched")] for size in SIZES
                       if values.get((f"OLMo2-{size}-Instruct", "shifted")) is not None and
                       values.get((f"OLMo2-{size}-Instruct", "matched")) is not None]
            actual = sum(changes) / len(changes) if len(changes) == len(SIZES) else None
            add("A15.T85", {"method": method, "domain": domain,
                               "statistic": "mean_over_models_of_shifted_minus_matched_auc"},
                delta_rows[paper_method][column], actual)
    return {"source_url": PAPER_URL, "comparisons": comparisons,
            "n": len(comparisons),
            "matches": sum(c["matches_display_precision"] for c in comparisons),
            "interpretation": "Agreement to published display precision, not a fresh model-score replication; significance not verified."}


class Audit:
    def __init__(self, snapshot: Path, output: Path):
        self.snapshot = snapshot.resolve()
        self.output = output.resolve()
        self.sources: dict[str, dict] = {}
        self.issues: list[dict] = []
        self.benchmark: list[dict] = []
        self.benchmark_files: list[dict] = []
        self.bench_by_leaf: dict[tuple, list[dict]] = defaultdict(list)
        self.rows: list[dict] = []
        self.score_files: list[dict] = []

    def issue(self, severity: str, kind: str, **details: Any) -> None:
        self.issues.append({"severity": severity, "kind": kind, **details})

    def read(self, path: Path) -> str:
        data = path.read_bytes()
        try:
            name = str(path.resolve().relative_to(self.snapshot))
        except ValueError:
            name = str(path.resolve())
        self.sources[name] = {"path": str(path.resolve()), "sha256": sha256_bytes(data),
                              "bytes": len(data)}
        return data.decode("utf-8")

    def load_benchmark(self) -> None:
        for path in sorted((self.snapshot / "benchmark/pretraining").rglob("*.jsonl")):
            relative = str(path.relative_to(self.snapshot))
            parts = path.relative_to(self.snapshot / "benchmark/pretraining").parts
            domain = parts[0]
            label = 1 if parts[1] == "member" else 0
            condition = parts[2] if label else "shared"
            split = parts[-2]
            subset = next((s for s in ("java", "python") if f"_{s}_" in path.name), "") if domain == "starcoder" else ""
            if domain not in DOMAINS or subset not in DOMAINS[domain] or split not in ("dev", "test"):
                raise ValueError(f"Unexpected benchmark layout: {relative}")
            records = [json.loads(line) for line in self.read(path).splitlines() if line.strip()]
            hashes = []
            ids = []
            for index, record in enumerate(records):
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing text in {relative}:{index}")
                stripped = text.strip()
                text_hash = sha256_bytes(stripped.encode("utf-8"))
                short = hashlib.sha1(stripped.encode("utf-8")).hexdigest()[:20]
                source_id = record.get("id")
                expected_id = source_id.strip() if isinstance(source_id, str) and source_id.strip() else f"sample_{index}"
                row = {"benchmark_path": relative, "benchmark_row_index": index,
                       "domain": domain, "subset": subset, "condition": condition,
                       "split": split, "label": label, "sample_id": expected_id,
                       "text_hash": text_hash, "text_sha1": short,
                       "raw_text_sha256": sha256_bytes(text.encode("utf-8")),
                       "characters": len(stripped), "utf8_bytes": len(stripped.encode("utf-8")),
                       "publisher_reported_13gram_overlap": {k: record[k] for k in record if k.endswith("13-gram_overlap_score")},
                       "metadata_keys": sorted(k for k in record if k != "text")}
                self.benchmark.append(row)
                self.bench_by_leaf[(domain, subset, condition, split)].append(row)
                hashes.append(text_hash)
                ids.append(expected_id)
            duplicate_hashes = [k for k, v in Counter(hashes).items() if v > 1]
            duplicate_ids = [k for k, v in Counter(ids).items() if v > 1]
            if duplicate_hashes:
                self.issue("error", "duplicate_benchmark_text_within_file", path=relative, hashes=duplicate_hashes)
            if duplicate_ids:
                self.issue("warning", "duplicate_benchmark_sample_id", path=relative, ids=duplicate_ids)
            self.benchmark_files.append({"path": relative, "domain": domain, "subset": subset,
                                         "condition": condition, "split": split, "label": label,
                                         "rows": len(records), "unique_texts": len(set(hashes)),
                                         "duplicate_ids": duplicate_ids})
        expected = 5 * 3 * 2
        if len(self.benchmark_files) != expected:
            self.issue("error", "unexpected_benchmark_file_count", expected=expected, actual=len(self.benchmark_files))

        by_text = defaultdict(list)
        for row in self.benchmark:
            by_text[row["text_hash"]].append(row)
        for text_hash, rows in by_text.items():
            if len(rows) < 2:
                continue
            labels = {r["label"] for r in rows}
            splits = {r["split"] for r in rows}
            if len(labels) > 1:
                kind, severity = "exact_text_label_conflict", "error"
            elif len(splits) > 1:
                kind, severity = "exact_text_cross_dev_test", "warning"
            else:
                kind, severity = "exact_text_shared_benchmark_sets", "warning"
            self.issue(severity, kind, text_hash=text_hash,
                       locations=[{k: r[k] for k in ("benchmark_path", "benchmark_row_index", "label")} for r in rows])

    def load_scores(self) -> None:
        # Enumerate all relevant physical files, rather than assuming test names
        # also represent dev. Hash matching determines the benchmark split.
        allowed_models = {f"OLMo2-{s}-Instruct" for s in SIZES}
        stems = {stem for stem, _ in METHODS.values()}
        pattern = re.compile(r"^(member|non-member)_pretraining_(dclm|pes2o|openwebmath|starcoder)(?:_(java|python))?(?:_(matched|shifted))?(?:_(dev|test))?$")
        for leaf in sorted((self.snapshot / "results").iterdir()):
            if not leaf.is_dir() or "_pretraining_" not in leaf.name:
                continue
            match = pattern.fullmatch(leaf.name)
            if match is None:
                self.issue("warning", "unrecognized_result_leaf", path=str(leaf.relative_to(self.snapshot)))
                continue
            membership, domain, subset, condition, named_split = match.groups()
            label = int(membership == "member")
            subset = subset or ""
            condition = condition or "shared"
            if (label and condition not in ("matched", "shifted")) or (not label and condition != "shared"):
                raise ValueError(f"Unexpected member condition: {leaf.name}")
            candidates = [r for r in self.benchmark if r["domain"] == domain and r["subset"] == subset and
                          r["label"] == label and r["condition"] == condition]
            hash_index = defaultdict(list)
            for row in candidates:
                hash_index[row["text_sha1"]].append(row)
            for model_dir in sorted(leaf.iterdir()):
                if not model_dir.is_dir() or model_dir.name not in allowed_models:
                    continue
                for path in sorted(model_dir.glob("*.json")):
                    if path.stem not in stems:
                        continue
                    relative = str(path.relative_to(self.snapshot))
                    data = json.loads(self.read(path))
                    samples = data.get("result", {}).get("samples")
                    if not isinstance(samples, list):
                        raise ValueError(f"Missing result.samples list: {relative}")
                    manifest = path.with_name(f"run_manifest_{path.stem}.json")
                    manifest_data = json.loads(self.read(manifest)) if manifest.exists() else None
                    info = {"path": relative, "model": model_dir.name, "domain": domain, "subset": subset,
                            "condition": condition, "label": label, "rows": len(samples),
                            "top_level_keys": sorted(data), "result_metadata_keys": sorted(k for k in data.get("result", {}) if k != "samples"),
                            "run_manifest_present": manifest.exists(),
                            "run_manifest": manifest_data, "joined_rows": 0, "unjoined_rows": 0,
                            "record_index_mismatches": 0, "sample_id_mismatches": 0,
                            "duplicate_text_hashes": [], "split_counts": {},
                            "score_columns": [column for stem, column in METHODS.values() if stem == path.stem]}
                    physical_hashes = []
                    split_counts = Counter()
                    counts = []
                    for result_index, sample in enumerate(samples):
                        text_sha1 = sample.get("text_sha1")
                        inline_text = sample.get("text")
                        if isinstance(inline_text, str):
                            inline_hash = hashlib.sha1(inline_text.strip().encode("utf-8")).hexdigest()[:20]
                            if text_sha1 and text_sha1 != inline_hash:
                                self.issue("error", "inline_text_hash_conflict", path=relative, result_row_index=result_index)
                            text_sha1 = inline_hash
                        all_matches = hash_index.get(text_sha1, [])
                        # Upstream data_utils drops _test from result names but
                        # retains _dev. This is also corroborated by all other
                        # uniquely matched rows in these files. Restrict to the
                        # corresponding dataset split BEFORE resolving identity:
                        # one released OWM text occurs in both dev and test.
                        file_split = named_split or "test"
                        matches = [r for r in all_matches if r["split"] == file_split]
                        bench = matches[0] if len(matches) == 1 else None
                        if len(all_matches) > 1 and bench is not None:
                            self.issue("warning", "hash_present_in_multiple_benchmark_splits", path=relative,
                                       result_row_index=result_index, text_sha1=text_sha1,
                                       selected_split=file_split, rule="publisher_filename_contract_then_exact_text_hash")
                        if bench is None:
                            self.issue("error", "score_benchmark_join_not_unique", path=relative,
                                       result_row_index=result_index, text_sha1=text_sha1, matches=len(matches))
                            info["unjoined_rows"] += 1
                        else:
                            info["joined_rows"] += 1
                            split_counts[bench["split"]] += 1
                            if named_split and named_split != bench["split"]:
                                self.issue("error", "named_split_hash_conflict", path=relative, result_row_index=result_index)
                            if sample.get("record_index") != bench["benchmark_row_index"]:
                                info["record_index_mismatches"] += 1
                            if sample.get("sample_id") != bench["sample_id"]:
                                info["sample_id_mismatches"] += 1
                            physical_hashes.append(bench["text_hash"])
                        token_count = sample.get("token_count")
                        token_valid = type(token_count) is int and token_count > 0
                        if not token_valid:
                            self.issue("error", "invalid_token_count", path=relative, result_row_index=result_index)
                        else:
                            counts.append(token_count)
                        for method, (stem, column) in METHODS.items():
                            if path.stem != stem:
                                continue
                            score = finite_score(sample.get(column))
                            if score is None:
                                self.issue("error", "missing_or_nonfinite_score", path=relative,
                                           result_row_index=result_index, method=method, raw_value=str(sample.get(column)))
                            self.rows.append({"path": relative, "method": method, "model": model_dir.name,
                                              "domain": domain, "subset": subset, "condition": condition,
                                              "split": bench["split"] if bench else None, "label": label,
                                              "text_hash": bench["text_hash"] if bench else None,
                                              "score": score, "token_count": token_count if token_valid else None,
                                              "result_row_index": result_index,
                                              "record_index": sample.get("record_index"), "sample_id": sample.get("sample_id"),
                                              "released_text_sha1": text_sha1,
                                              "benchmark_path": bench["benchmark_path"] if bench else None,
                                              "benchmark_row_index": bench["benchmark_row_index"] if bench else None,
                                              "join_status": "unique_exact_stripped_text_hash" if bench else "unresolved",
                                              "split_assignment_rule": "publisher_filename_contract_validated_by_hash",
                                              "score_column": column, "score_orientation": "higher_is_member"})
                    info["duplicate_text_hashes"] = [h for h, n in Counter(physical_hashes).items() if n > 1]
                    info["split_counts"] = dict(split_counts)
                    info["token_counts"] = count_summary(counts)
                    if info["duplicate_text_hashes"]:
                        self.issue("error", "duplicate_scored_text_within_file", path=relative, hashes=info["duplicate_text_hashes"])
                    if info["record_index_mismatches"] or info["sample_id_mismatches"]:
                        self.issue("warning", "hash_join_metadata_mismatch", path=relative,
                                   record_index_mismatches=info["record_index_mismatches"], sample_id_mismatches=info["sample_id_mismatches"],
                                   action="Use hash join; do not join by local record_index or sample_id alone.")
                    self.score_files.append(info)

        keys = [(r["method"], r["model"], r["domain"], r["subset"], r["condition"], r["split"], r["text_hash"]) for r in self.rows]
        duplicates = [k for k, n in Counter(keys).items() if n > 1]
        if duplicates:
            self.issue("error", "duplicate_normalized_row_key", keys=duplicates)

        # Alignment across methods and model sizes is observable in saved counts;
        # alignment of actual token IDs is NOT observable (vectors were not released).
        by_input = defaultdict(list)
        for row in self.rows:
            by_input[(row["domain"], row["subset"], row["condition"], row["split"], row["text_hash"])].append(row)
        for key, rows in by_input.items():
            if len({r["token_count"] for r in rows}) != 1:
                self.issue("warning", "token_count_disagreement_across_methods_or_models", key=key,
                           observations=[{k: r[k] for k in ("method", "model", "token_count", "path")} for r in rows])

    def metrics(self) -> tuple[list[dict], list[dict]]:
        grouped = defaultdict(list)
        for row in self.rows:
            if row["join_status"] == "unique_exact_stripped_text_hash" and row["score"] is not None:
                grouped[(row["method"], row["model"], row["domain"], row["subset"], row["condition"], row["split"])].append(row)
        metrics = []
        aggregates = []
        for method in METHODS:
            for size in SIZES:
                model = f"OLMo2-{size}-Instruct"
                for condition in ("matched", "shifted"):
                    for split in ("dev", "test"):
                        domain_aucs = {}
                        pooled_positive, pooled_negative = [], []
                        for domain, subsets in DOMAINS.items():
                            subset_aucs = []
                            for subset in subsets:
                                positive = grouped[method, model, domain, subset, condition, split]
                                negative = grouped[method, model, domain, subset, "shared", split]
                                p = [r["score"] for r in positive]
                                n = [r["score"] for r in negative]
                                expected_p = len(self.bench_by_leaf[domain, subset, condition, split])
                                expected_n = len(self.bench_by_leaf[domain, subset, "shared", split])
                                score = auc_pairs(p, n)
                                ranks = auc_ranks(p, n)
                                if score is not None and abs(score - ranks) > 1e-12:
                                    self.issue("error", "independent_auc_disagreement", method=method, model=model, domain=domain, subset=subset)
                                metrics.append({"method": method, "model": model, "domain": domain, "subset": subset,
                                                "condition": condition, "split": split, "auc": score,
                                                "auc_rank_check": ranks, "n_member": len(p), "n_nonmember": len(n),
                                                "expected_member": expected_p, "expected_nonmember": expected_n,
                                                "complete": len(p) == expected_p and len(n) == expected_n,
                                                "status": "available" if score is not None else "no_released_scores_for_both_labels",
                                                "member_token_counts": count_summary([r["token_count"] for r in positive if r["token_count"] is not None]),
                                                "nonmember_token_counts": count_summary([r["token_count"] for r in negative if r["token_count"] is not None])})
                                if score is not None:
                                    subset_aucs.append(score)
                                pooled_positive.extend(p)
                                pooled_negative.extend(n)
                            value = sum(subset_aucs) / len(subset_aucs) if len(subset_aucs) == len(subsets) else None
                            domain_aucs[domain] = value
                            aggregates.append({"kind": "domain_macro", "method": method, "model": model, "domain": domain,
                                               "condition": condition, "split": split, "auc": value,
                                               "formula": "unweighted_mean_of_all_constituent_subset_AUCs",
                                               "available_subsets": len(subset_aucs), "expected_subsets": len(subsets)})
                        complete = all(v is not None for v in domain_aucs.values())
                        aggregates.append({"kind": "pretraining_macro", "method": method, "model": model,
                                           "condition": condition, "split": split,
                                           "auc": sum(domain_aucs.values()) / len(DOMAINS) if complete else None,
                                           "formula": "unweighted_mean_of_four_domain_macro_AUCs"})
                        aggregates.append({"kind": "pretraining_pooled", "method": method, "model": model,
                                           "condition": condition, "split": split,
                                           "auc": auc_pairs(pooled_positive, pooled_negative),
                                           "n_member": len(pooled_positive), "n_nonmember": len(pooled_negative),
                                           "formula": "single_AUC_over_all_five_subsets_without_domain_normalization"})
        return metrics, aggregates

    def source_evidence(self) -> dict:
        paths = ("README.md", "data_utils.py", "model_utils.py", "evaluate.py",
                 "analysis/paper_tables.py", "analysis/consolidate_all.py",
                 "methods/minkprob_method.py", "methods/loss_zlib_lowercase_method.py")
        evidence = {}
        patterns = {
            "model_utils.py": ("max_length=4096", "add_special_tokens=True", "special_mask", "eos_token_id"),
            "methods/minkprob_method.py": ("k = max(1, int(", "config_dict =", "PAPER-DEFAULT", "clamp(min=1e-9)", "sigma + 1e-9"),
            "methods/loss_zlib_lowercase_method.py": ("loss_score =", "ll_sum_orig / z_bytes"),
            "data_utils.py": ("return text.strip()", "return raw_id.strip()"),
            "analysis/paper_tables.py": ("MACRO-averaging", "hexdigest()[:20]", '"sc_python","sc_java"'),
            "evaluate.py": ('"StarCoder","Assembly"', "POOLING"),
            "README.md": ("TedSSS/OLMo-Detect", "olmo-mix-1124"),
            "analysis/consolidate_all.py": ("stem_samp", "mt>cur[hh][0]", 'json.dump({"result"'),
        }
        for relative in paths:
            source = self.read(self.snapshot / relative)
            evidence[relative] = [{"line": i, "text": line.strip()} for i, line in enumerate(source.splitlines(), 1)
                                  if any(p in line for p in patterns.get(relative, ()))]
        return evidence

    def verify_archive_sources(self, archive: Path) -> int:
        """Bind consumed snapshot bytes to the verified tar, without extraction."""
        remaining = {name: value["sha256"] for name, value in self.sources.items()
                     if not Path(name).is_absolute()}
        verified = 0
        with tarfile.open(archive, "r:gz") as stream:
            prefix = self.snapshot.name + "/"
            for member in stream:
                if not member.isfile() or not member.name.startswith(prefix):
                    continue
                name = member.name[len(prefix):]
                if name not in remaining:
                    continue
                content = stream.extractfile(member)
                if content is None or sha256_bytes(content.read()) != remaining.pop(name):
                    raise ValueError(f"Consumed snapshot source differs from pinned archive: {name}")
                verified += 1
        if remaining:
            raise ValueError(f"Consumed snapshot sources absent from archive: {sorted(remaining)}")
        return verified

    def reported_overlap_checks(self) -> list[dict]:
        checks = []
        for (domain, subset, condition, split), rows in sorted(self.bench_by_leaf.items()):
            if condition != "shared":
                continue
            for size in SIZES:
                field = f"{size.lower()}_13-gram_overlap_score"
                values = [finite_score(r["publisher_reported_13gram_overlap"].get(field)) for r in rows]
                observed = [v for v in values if v is not None]
                checks.append({"domain": domain, "subset": subset, "split": split, "label": 0,
                               "model_size": size, "field": field, "n_rows": len(rows),
                               "n_present_finite": len(observed), "missing_or_invalid": len(rows) - len(observed),
                               "minimum": min(observed) if observed else None,
                               "maximum": max(observed) if observed else None,
                               "mean": sum(observed) / len(observed) if observed else None,
                               "at_or_above_0_2": sum(v >= .2 for v in observed),
                               "outside_0_1": sum(not 0 <= v <= 1 for v in observed),
                               "status": "publisher_reported_values_only_not_recomputed_corpus_overlap"})
        return checks
