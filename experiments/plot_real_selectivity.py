#!/usr/bin/env python3
"""Static figures from the independently verified, frozen real-selectivity run.

No metric inference, classifier fitting, model calls or network requests.
Existing output directories are rejected. PNG/SVG exports share one figure;
when available, rsvg-convert produces separate SVG inspection renders.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


VERSION = "real-selectivity-figures.v3"
MODEL = "OLMo2-1B-Instruct"
GATES = ("no_gate", "length_only", "origin_threshold", "full_policy")
GATE_LABELS = {"no_gate": "No gate", "length_only": "Length only", "origin_threshold": "Origin gate",
               "full_policy": "Observable policy"}
VIEWS = ("nonmember_null", "matched_member", "shifted_member", "cross_domain_nonmember_stress")
VIEW_LABELS = {"nonmember_null": "Nonmember null", "matched_member": "Matched members",
               "shifted_member": "Shifted members", "cross_domain_nonmember_stress": "Cross-domain nonmembers"}
DOMAIN_LABELS = {"ALL": "Pooled", "dclm": "DCLM", "openwebmath": "OpenWebMath",
                 "pes2o": "peS2o", "starcoder": "StarCoder (Java + Python)"}
DOMAINS = tuple(DOMAIN_LABELS)
COLORS = {"accepted": "#285B87", "abstained": "#DCE1E6", "unavailable": "#F5F6F7",
          "hatch": "#78838D", "flag": "#A75A14", "ink": "#202A33", "muted": "#52616D", "grid": "#E7EBEE"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_data(run, verification):
    hashes = read_json(run / "final_artifact_hashes.json")
    independent = read_json(verification)
    if (independent["status"] != "pass" or independent["failures"] or
            independent["sealed_manifest_sha256"] != sha(run / "sealed_manifest.json") or
            independent["final_artifact_hashes_sha256"] != sha(run / "final_artifact_hashes.json") or
            independent["verifier_sha256"] != sha(verification.with_suffix(".py"))):
        raise ValueError("Independent verification is absent, failed or does not bind this run.")
    sources = {}
    for name in ("summary_counts.json", "blind_baseline_results.json", "sealed_manifest.json"):
        path = run / name
        actual = sha(path)
        if actual != hashes[name]:
            raise ValueError(f"Frozen plotting input changed: {name}")
        sources[str(path.resolve())] = actual
    sources[str((run / "final_artifact_hashes.json").resolve())] = sha(run / "final_artifact_hashes.json")
    sources[str(verification.resolve())] = sha(verification)
    sources[str(verification.with_suffix(".py").resolve())] = sha(verification.with_suffix(".py"))
    summary = read_json(run / "summary_counts.json")
    selected = [r for r in summary if r["model"] == MODEL and r["bank"] == "primary" and
                r["domain"] == "ALL" and r["gate"] in GATES]
    lookup = {(r["condition"], r["gate"]): r for r in selected}
    if len(selected) != 16 or len(lookup) != 16 or set(lookup) != {(v, g) for v in VIEWS for g in GATES}:
        raise ValueError("Expected one row for every four-gate by four-view combination.")
    coverage = []
    for view in VIEWS:
        for gate in GATES:
            r = lookup[view, gate]
            attempted, accepted, flags = r["attempted_templates"], r["accepted"], r["diagnostic_rejections"]
            if (attempted != 36 or accepted + r["abstained"] + r["gate_unavailable"] != attempted or
                    not 0 <= flags <= accepted or r["coverage_per_attempted"] != accepted / attempted or
                    r["gate_unavailable"] != (4 if view == "matched_member" else 0)):
                raise ValueError("Coverage/flag denominators do not reconcile.")
            coverage.append({"model": MODEL, "bank": "primary", "view": view, "view_label": VIEW_LABELS[view],
                             "gate": gate, "gate_label": GATE_LABELS[gate], "attempted_templates": attempted,
                             "formed_cases": r["formed_cases"], "complete_cases": r["common_quality_complete"],
                             "accepted_cases": accepted, "abstained_cases": r["abstained"],
                             "unavailable_cases": r["gate_unavailable"], "diagnostic_flags": flags,
                             "accepted_coverage_per_attempted": accepted / attempted,
                             "diagnostic_flags_per_accepted": flags / accepted if accepted else None,
                             "diagnostic_flags_per_attempted": flags / attempted})
    blind = read_json(run / "blind_baseline_results.json")
    bmap = {(r["condition"], r["domain"]): r for r in blind}
    if len(blind) != 10 or set(bmap) != {(c, d) for c in ("matched", "shifted") for d in DOMAINS}:
        raise ValueError("Expected the ten verified text-only AUC cells.")
    baseline = []
    for domain in DOMAINS:
        for condition in ("matched", "shifted"):
            r = bmap[condition, domain]
            if r["auc"] is None or not 0 <= r["auc"] <= 1 or min(r["member_records"], r["nonmember_records"]) < 1:
                raise ValueError("Unavailable or invalid baseline cell.")
            baseline.append({"domain": domain, "domain_label": DOMAIN_LABELS[domain], "condition": condition,
                             "auc": r["auc"], "official_member_representatives": r["member_records"],
                             "official_nonmember_representatives": r["nonmember_records"],
                             "score_direction": "higher_is_member_fixed_without_test_direction_flip"})
    return coverage, baseline, sources


def style(matplotlib):
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"], "xtick.color": COLORS["muted"], "ytick.color": COLORS["ink"],
        "axes.edgecolor": COLORS["hatch"], "axes.linewidth": .65, "xtick.major.width": .65,
        "ytick.major.width": .65, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "path", "svg.hashsalt": VERSION, "pdf.fonttype": 42,
    })


def coverage_figure(plt, rows, summary_hash):
    from matplotlib.patches import Patch
    lookup = {(r["view"], r["gate"]): r for r in rows}
    fig = plt.figure(figsize=(9.2, 9.0))
    grid = fig.add_gridspec(4, 2, left=.19, right=.965, top=.84, bottom=.22,
                           width_ratios=(1.3, 1.), wspace=.38, hspace=.67)
    fig.text(.19, .978, "Coverage and diagnostic flags on published-label cases", fontsize=14, weight="bold", va="top")
    fig.text(.19, .946, "OLMo2-1B-Instruct · observable saved-vector policy · same fixed diagnostic", fontsize=10, color=COLORS["muted"], va="top")
    fig.text(.19, .899, "Accepted / attempted (%)", fontsize=10.5, weight="bold")
    right_start = grid[0, 1].get_position(fig).x0
    fig.text(right_start, .899, "Diagnostic flags / accepted", fontsize=10.5, weight="bold")
    for i, view in enumerate(VIEWS):
        left = fig.add_subplot(grid[i, 0])
        right = fig.add_subplot(grid[i, 1], sharey=left)
        suffix = "32 complete / 36 attempted; 4 unavailable" if view == "matched_member" else "36 complete / 36 attempted"
        if view == "cross_domain_nonmember_stress":
            suffix += "; dependent stress views"
        left.set_title(f"({chr(97 + i)}) {VIEW_LABELS[view]}  ·  {suffix}", loc="left", fontsize=10, weight="bold", pad=10)
        for y, gate in enumerate(GATES):
            r = lookup[view, gate]
            percent = 100 * r["accepted_coverage_per_attempted"]
            left.barh(y, 100, height=.60, color=COLORS["abstained"], edgecolor="none", zorder=1)
            unavailable = 100 * r["unavailable_cases"] / r["attempted_templates"]
            if unavailable:
                left.barh(y, unavailable, left=100 - unavailable, height=.60, facecolor=COLORS["unavailable"],
                          edgecolor=COLORS["hatch"], linewidth=.45, hatch="////", zorder=2)
            left.barh(y, percent, height=.60, color=COLORS["accepted"], edgecolor="none", zorder=3)
            left.text(percent - 2.1, y, f"{r['accepted_cases']}/{r['attempted_templates']}", ha="right", va="center",
                      fontsize=9, color="white", weight="bold", zorder=4)
            flag = r["diagnostic_flags"]
            right.scatter([flag], [y], s=35, marker="o", color=COLORS["flag"], zorder=3)
            right.text(flag + .19, y, f"{flag} / {r['accepted_cases']}", ha="left", va="center", fontsize=9.4)
        left.set_yticks(range(4), [GATE_LABELS[g] for g in GATES], fontsize=9.3)
        left.set_ylim(3.55, -.55)
        left.set_xlim(0, 100)
        left.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"], fontsize=8)
        left.tick_params(axis="y", length=0, pad=7)
        left.tick_params(axis="x", length=3)
        left.spines["left"].set_visible(False)
        right.set_xlim(-.12, 5.65)
        right.set_xticks([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"], fontsize=8)
        right.spines["bottom"].set_bounds(0, 4)
        right.spines["left"].set_visible(False)
        right.tick_params(axis="y", left=False, labelleft=False)
        right.tick_params(axis="x", length=3)
        right.set_axisbelow(True)
        right.grid(axis="x", color=COLORS["grid"], linewidth=.65)
        if i == len(VIEWS) - 1:
            right.set_xlabel("Diagnostic flag count", fontsize=8.5, labelpad=5)
    legend = [Patch(facecolor=COLORS["accepted"], edgecolor="none", label="Accepted"),
              Patch(facecolor=COLORS["abstained"], edgecolor="none", label="Abstained"),
              Patch(facecolor=COLORS["unavailable"], edgecolor=COLORS["hatch"], hatch="////", label="Unavailable")]
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(.183, .148), ncol=3, frameon=False, fontsize=9,
               handlelength=1.8, columnspacing=2.0)
    note = ("Flags: fixed-family adjusted p < 0.05 score-distribution diagnostics; not verified training membership.\n"
            "Unavailable matched templates remain in the attempted denominator. Views reuse records/references and are dependent.\n"
            "Finite-bank counts only; no confidence intervals or population-FPR guarantee.\n"
            f"Source: summary_counts.json · SHA256 {summary_hash[:16]}… · full hashes and rows in manifest.json / plotted_data.json.")
    fig.text(.19, .13, note, fontsize=7.6, color=COLORS["muted"], va="top", linespacing=1.5)
    return fig


def baseline_figure(plt, rows, baseline_hash):
    from matplotlib.lines import Line2D
    lookup = {(r["domain"], r["condition"]): r for r in rows}
    fig = plt.figure(figsize=(9.5, 5.4))
    ax = fig.add_axes([.245, .28, .425, .50])
    counts = fig.add_axes([.725, .28, .25, .50], sharey=ax)
    fig.text(.245, .975, "Text-only baseline AUCs on held-out published-label records", fontsize=13.5, weight="bold", va="top")
    fig.text(.245, .927, "Matched-dev fit · higher = member · score direction fixed", fontsize=10, color=COLORS["muted"], va="top")
    ax.axvline(.5, color=COLORS["hatch"], linewidth=.85, linestyle=(0, (4, 3)), zorder=1)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=.65)
    for i, domain in enumerate(DOMAINS):
        for condition, offset, color, marker in (("matched", -.15, COLORS["accepted"], "o"),
                                                  ("shifted", .15, COLORS["flag"], "D")):
            row = lookup[domain, condition]
            ax.scatter(row["auc"], i + offset, s=35, marker=marker, edgecolor=color,
                       facecolor=color if condition == "matched" else "white", linewidth=1., zorder=3)
            ax.text(row["auc"] + .022, i + offset, f"{row['auc']:.3f}", va="center", fontsize=9, color=color)
            x = .25 if condition == "matched" else .77
            counts.text(x, i, f"{row['official_member_representatives']} / {row['official_nonmember_representatives']}",
                        ha="center", va="center", fontsize=9.2, color=COLORS["ink"])
    ax.set_xlim(0, 1)
    ax.set_ylim(4.55, -.55)
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1.00"], fontsize=8.5)
    ax.set_yticks(range(5), [DOMAIN_LABELS[d] for d in DOMAINS], fontsize=9.4)
    ax.set_xlabel("AUC (fixed higher-is-member direction)", fontsize=9.3, labelpad=8)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.spines["left"].set_visible(False)
    counts.set_xlim(0, 1)
    counts.axis("off")
    counts.text(.25, 1.035, "Matched\nmember /\nnonmember", transform=counts.transAxes,
                ha="center", va="bottom", fontsize=8.4, color=COLORS["accepted"], linespacing=1.5)
    counts.text(.77, 1.035, "Shifted\nmember /\nnonmember", transform=counts.transAxes,
                ha="center", va="bottom", fontsize=8.4, color=COLORS["flag"], linespacing=1.5)
    legend = [Line2D([], [], marker="o", linestyle="none", color=COLORS["accepted"], markersize=6, label="Matched members"),
              Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", color=COLORS["flag"], markersize=5.5, label="Shifted members")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.235, .87), frameon=False, ncol=2, fontsize=9,
               handlelength=1., columnspacing=2.)
    note = ("Frozen character-TFIDF / logistic baseline; dev labels only, no target-model inference. Full eligible record bank, not case blocks.\n"
            "Pooled AUC combines records rather than averaging domains. OpenWebMath 0.200 is not direction-flipped.\n"
            "Conditions share nonmembers; counts are representatives. No confidence intervals or population-validity claim.\n"
            f"Source: blind_baseline_results.json · SHA256 {baseline_hash[:16]}… · full hashes and plotted rows retained.")
    fig.text(.245, .155, note, fontsize=7.4, color=COLORS["muted"], va="top", linespacing=1.5)
    return fig


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent / "results/real-selectivity-v1"
    parser.add_argument("--run", type=Path, default=base / "run-2026-11-07")
    parser.add_argument("--verification", type=Path, default=base / "independent_technical_verification.json")
    parser.add_argument("--output", type=Path, default=base / "figures")
    args = parser.parse_args()
    coverage, baseline, sources = load_data(args.run, args.verification)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.output / "plot_real_selectivity.py")
    source_sha = sha(Path(__file__))
    data = {"schema_version": 1, "model": MODEL, "coverage_and_flags": coverage, "text_only_baseline": baseline,
            "coverage_denominator": "All 36 attempted templates per view, including 4 unavailable matched templates",
            "flag_denominator": "Accepted complete cases within each gate and view",
            "auc_direction": "Higher is official member; no direction flip", "source_files_sha256": sources,
            "transformations": "Filter frozen primary1B/all-domain rows; convert accepted/attempted fraction to percent for geometry. No re-estimation, smoothing or inference."}
    write_json(args.output / "plotted_data.json", data)
    csv_write(args.output / "coverage_and_flags.csv", coverage)
    csv_write(args.output / "text_only_baseline.csv", baseline)
    summary_hash = sha(args.run / "summary_counts.json")
    baseline_hash = sha(args.run / "blind_baseline_results.json")
    exported = []
    with tempfile.TemporaryDirectory(prefix="contamcheckr-mpl-") as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        style(matplotlib)
        figures = [("coverage_and_diagnostic_flags", coverage_figure(plt, coverage, summary_hash)),
                   ("text_only_baseline_auc", baseline_figure(plt, baseline, baseline_hash))]
        for name, fig in figures:
            description = json.dumps({"plot_version": VERSION, "source_sha256": source_sha, "input_sha256": sources}, sort_keys=True)
            fig.savefig(args.output / f"{name}.png", dpi=300,
                        metadata={"Title": name, "Description": description, "Software": VERSION})
            fig.savefig(args.output / f"{name}.svg",
                        metadata={"Title": name, "Description": description, "Creator": VERSION, "Date": None})
            exported.extend([f"{name}.png", f"{name}.svg"])
            plt.close(fig)
        matplotlib_version = matplotlib.__version__
    converter = shutil.which("rsvg-convert")
    svg_renders = []
    if converter:
        qa = args.output / "visual_qa"
        qa.mkdir()
        for name in ("coverage_and_diagnostic_flags", "text_only_baseline_auc"):
            target = qa / f"{name}_from_svg.png"
            subprocess.run([converter, "--width", "2400", "--output", str(target), str(args.output / f"{name}.svg")], check=True)
            svg_renders.append(str(target.relative_to(args.output)))
    caption = """# Real-selectivity figures

The two figures show the already independently verified frozen run. They add no inference.

`coverage_and_diagnostic_flags`: Primary OLMo2-1B-Instruct saved-score case views. Blue bars show accepted / all36 attempted templates; gray is abstention and hatching is unavailable construction. Four unavailable matched-member templates stay in the denominator. The separate points show diagnostic-flag counts with accepted denominators, not contamination determinations. The member and mismatch views reuse records/references and are dependent.

`text_only_baseline_auc`: The fixed matched-dev text-only classifier evaluated on the full eligible held-out representative bank, separately from dataset-case blocks. AUC uses the unchanged higher-is-member orientation. The pooled cells combine records rather than average domain AUCs; OpenWebMath0.200 is retained. Member conditions share nonmembers. Counts next to each domain are official-member / official-nonmember representatives.

No confidence intervals, smoothing, fitted trend or new significance claim is added. PNG and SVG are exported from the same Matplotlib figure; SVG inspection renders, when available, are in `visual_qa/`. `plotted_data.json` and the two CSV files retain every displayed value and denominator. `manifest.json` binds all plotting inputs, the independent verifier, plotting source and exports by full SHA256.

Reproduce from the project root with `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/plot_real_selectivity.py --output NEW_DIRECTORY`. Existing output is never overwritten. The copied plotting source is also executable with explicit `--run` and `--verification` paths.
"""
    (args.output / "README.md").write_text(caption)
    output_hashes = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    manifest = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "run_directory": str(args.run.resolve()), "source_files_sha256": sources,
                "plot_script_sha256": source_sha, "matplotlib": matplotlib_version,
                "exported_figures": exported, "svg_inspection_renders": svg_renders,
                "output_files_sha256": output_hashes, "visual_review": "Pending inspection of exported PNG and SVG inspection renders",
                "new_inference": False, "intervals_added": False, "model_calls": 0, "network_requests": 0}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"output": str(args.output), "figures": exported, "svg_inspection_renders": svg_renders}, indent=2))


if __name__ == "__main__":
    main()
