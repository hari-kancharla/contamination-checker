"""Generate the paper's markdown tables from the suite results JSONL.

Every number in the draft tables comes from this script's output, which
is printed to stdout.  Run:

  PYTHONPATH=. .venv/bin/python experiments/paper_tables.py \
      experiments/results/paper-suite.jsonl
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import evaluate_scores  # noqa: E402

# detector -> (benchmark field, control field) in the compact audits
DETECTORS = {
    "minkpp": ("minkpp_score", "minkpp"),
    "loss": ("loss", "loss"),
    "zlib": ("zlib", "zlib"),
    "entropy": ("entropy", "entropy"),
    "mean_rank": ("mean_rank", "mean_rank"),
    "top1_prob": ("top1_prob", "top1_prob"),
}


def vectors(compact_audit: dict[str, object], detector: str) -> tuple[list[float], list[float], list[str]]:
    bench_field, ctrl_field = DETECTORS[detector]
    per_example = compact_audit.get("per_example") or []
    control_rows = compact_audit.get("control_per_example") or []
    bench = [float(row[bench_field]) for row in per_example if row.get(bench_field) is not None]
    ids = [str(row["id"]) for row in per_example if row.get(bench_field) is not None]
    ctrl = [float(row[ctrl_field]) for row in control_rows if row.get(ctrl_field) is not None]
    return bench, ctrl, ids


def fmt(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/results/paper-suite.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cells = [row for row in rows if "adapted_on_leak" in row]

    print("## Table 1: per-cell discrimination (adapted model, leak vs clean)")
    print("\n| model | epochs | seed | minkpp AUC | loss AUC | zlib AUC | TPR@5% FPR |")
    print("|---|---|---|---|---|---|---|")
    for cell in sorted(cells, key=lambda r: (r["model_id"], r["epochs"], r["seed"])):
        audit = cell["adapted_on_leak"]
        out = {}
        for name in DETECTORS:
            bench, ctrl, ids = vectors(audit, name)
            out[name] = evaluate_scores(bench, ctrl, ids) if bench and ctrl else None
        mink = out["minkpp"] or {}
        tpr = (mink.get("tpr_at_fpr") or {}).get("tpr_at_fpr_0.05")
        short = Path(cell["model_id"]).name
        print(
            f"| {short} | {cell['epochs']} | {cell['seed']} | "
            f"{fmt(mink.get('auc'))} | {fmt((out['loss'] or {}).get('auc'))} | "
            f"{fmt((out['zlib'] or {}).get('auc'))} | {fmt(tpr)} |"
        )

    mismatch = [row for row in rows if "mismatch_easy_control" in row]
    if mismatch:
        easy = mismatch[0]["mismatch_easy_control"]
        matched = mismatch[0]["matched_control"]
        print("\n## Table 2: domain-mismatch condition (base model, no training)")
        print("\n| control | minkpp AUC | AUC 95% CI | control-null AUC | median tokens (bench/control) | flagged at FDR 0.1 |")
        print("|---|---|---|---|---|---|")
        for label, audit in (("matched (cards clean)", matched), ("mismatched (easy one-liners)", easy)):
            bench, ctrl, ids = vectors(audit, "minkpp")
            ev = evaluate_scores(bench, ctrl, ids)
            null = ev["control_null"] or {}
            screen = ev["screen"] or {}
            shift = audit.get("domain_shift") or {}
            meds = f"{shift.get('benchmark_median_tokens')}/{shift.get('control_median_tokens')}"
            print(
                f"| {label} | {fmt(ev['auc'])} | {ev['auc_ci']} | "
                f"{fmt(null.get('auc'))} | {meds} | {screen.get('n_flagged')}/{screen.get('n_scored')} |"
            )

    print("\n## Table 3: seed stability (tiny-gpt2, 60 epochs)")
    print("\n| detector | AUC mean | AUC values |")
    print("|---|---|---|")
    selected = [c for c in cells if "tiny" in c["model_id"] and c["epochs"] == 60]
    for name in DETECTORS:
        aucs = []
        for cell in selected:
            bench, ctrl, ids = vectors(cell["adapted_on_leak"], name)
            if bench and ctrl and len(bench) >= 4 and len(ctrl) >= 4:
                aucs.append(float(evaluate_scores(bench, ctrl, ids)["auc"]))
        if aucs:
            mean = sum(aucs) / len(aucs)
            print(f"| {name} | {mean:.3f} | {', '.join(f'{v:.3f}' for v in aucs)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
