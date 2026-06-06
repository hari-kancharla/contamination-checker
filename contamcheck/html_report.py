from html import escape
from pathlib import Path
from typing import Any


def _value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return escape(str(value))


def build_html(report: dict[str, Any]) -> str:
    interval = report["confidence_interval"]
    interval_text = (
        f"{interval[0]:.3f} to {interval[1]:.3f}"
        if interval is not None
        else "Not estimated"
    )
    guided_interval = report["guided_gap_ci"]
    guided_interval_text = (
        f"{guided_interval[0]:.3f} to {guided_interval[1]:.3f}"
        if guided_interval is not None
        else "not estimated"
    )
    capability_rows = "".join(
        f"<tr><th>{escape(name.replace('_', ' ').title())}</th>"
        f"<td>{_value(value)}</td></tr>"
        for name, value in report["capabilities"].items()
    )
    method_rows = [
        (
            "Guided overlap gap",
            report["index_guided"],
            (
                f"ROUGE-L gap {_value(report['guided_general_gap'])}; "
                f"gap interval {guided_interval_text}"
            ),
        ),
        (
            "Min-K%++",
            report["index_minkpp"],
            (
                f"AUC {_value(report['minkpp_auc'])}; "
                + (
                    "matched-control calibration"
                    if report["index_minkpp"] is not None
                    else "raw distribution only"
                )
            ),
        ),
        (
            "Canonical order",
            report["index_order"],
            f"Permutation p-value {_value(report['order_p_value'])}",
        ),
    ]
    method_html = "".join(
        f"<tr><th>{escape(name)}</th><td>{_value(index)}</td>"
        f"<td>{escape(detail)}</td></tr>"
        for name, index, detail in method_rows
    )
    caveats = "".join(f"<li>{escape(item)}</li>" for item in report["caveats"])
    skipped = (
        "".join(
            f"<li><strong>{escape(item['method'])}:</strong> "
            f"{escape(item['reason'])}</li>"
            for item in report["skipped_methods"]
        )
        or "<li>None</li>"
    )
    preview_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{_value(item.get(field))}</td>"
            for field in (
                "id",
                "guided_rouge",
                "general_rouge",
                "guided_exact",
                "guided_near",
                "minkpp_score",
            )
        )
        + "</tr>"
        for item in report["per_example"][:20]
    )
    versions = ", ".join(
        f"{escape(name)} {escape(version)}"
        for name, version in report["library_versions"].items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ContamCheckr report</title>
<style>
:root {{ color-scheme: light; --ink: #17202a; --muted: #5d6873; --line: #d9dee3; --paper: #fff; --wash: #f4f6f7; --accent: #174a5c; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--wash); color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }}
main {{ max-width: 1040px; margin: 40px auto; padding: 0 20px 48px; }}
header, section {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 24px; margin-bottom: 18px; }}
h1, h2 {{ margin: 0 0 12px; line-height: 1.2; }}
h1 {{ font-size: 24px; }}
h2 {{ font-size: 18px; }}
.eyebrow {{ color: var(--muted); font-size: 13px; letter-spacing: .06em; text-transform: uppercase; }}
.score {{ color: var(--accent); font-size: 56px; font-weight: 700; letter-spacing: -.04em; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 12px 24px; color: var(--muted); }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; vertical-align: top; }}
th {{ font-weight: 600; }}
thead th {{ color: var(--muted); font-size: 13px; }}
.scroll {{ overflow-x: auto; }}
ul {{ margin: 8px 0 0; padding-left: 20px; }}
code {{ background: var(--wash); padding: 2px 5px; border-radius: 3px; }}
footer {{ color: var(--muted); font-size: 13px; }}
@media (max-width: 640px) {{ .score {{ font-size: 44px; }} header, section {{ padding: 18px; }} }}
</style>
</head>
<body>
<main>
<header>
  <div class="eyebrow">Benchmark contamination evidence</div>
  <h1>{escape(report['dataset_name'])}</h1>
  <div class="score">{report['contamination_index']:.3f}</div>
  <div class="meta">
    <span>{escape(report['band'].title())}</span>
    <span>95% interval: {escape(interval_text)}</span>
    <span>Basis: {escape(report['index_basis'])}</span>
  </div>
  <p>This score combines available comparative evidence. It is not a yes or no claim that an example appeared in training.</p>
</header>
<section>
  <h2>Capabilities</h2>
  <table>{capability_rows}</table>
</section>
<section>
  <h2>Method summary</h2>
  <table><thead><tr><th>Method</th><th>Index</th><th>Evidence</th></tr></thead><tbody>{method_html}</tbody></table>
</section>
<section>
  <h2>Caveats</h2>
  <ul>{caveats}</ul>
</section>
<section>
  <h2>Skipped methods</h2>
  <ul>{skipped}</ul>
</section>
<section>
  <h2>Per-example preview</h2>
  <div class="scroll">
  <table>
    <thead><tr><th>ID</th><th>Guided</th><th>General</th><th>Exact</th><th>Near</th><th>Min-K%++</th></tr></thead>
    <tbody>{preview_rows}</tbody>
  </table>
  </div>
</section>
<section>
  <h2>Reproducibility</h2>
  <p><strong>Model:</strong> <code>{escape(report['model_id'])}</code></p>
  <p><strong>Benchmark:</strong> <code>{escape(report['benchmark_source'])}</code>, split <code>{escape(report['split'])}</code>, {report['example_count']} examples</p>
  <p><strong>Seed:</strong> {report['seed']}</p>
  <footer>{versions}</footer>
</section>
</main>
</body>
</html>
"""


def write_html(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_html(report), encoding="utf-8")
