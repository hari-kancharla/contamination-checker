from dataclasses import dataclass, field
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .benchmarks import load_benchmark
from .config import RunConfig
from .html_report import build_html, write_html
from .model import load_model, set_seed
from .report import build_report, write_csv, write_json


DEFAULT_FORM = {
    "model": "sshleifer/tiny-gpt2",
    "benchmark": "sample",
    "control": "",
    "text_field": "text",
    "dataset_config": "",
    "split": "test",
    "limit": "10",
    "k": "20",
    "split_ratio": "0.5",
    "repro_threshold": "0.6",
    "device": "auto",
    "seed": "0",
    "trust_remote_code": "",
}


class _IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6


@dataclass
class WebState:
    output_dir: Path
    model_cache: dict[tuple[str, str, bool], Any] = field(default_factory=dict)
    audit_lock: threading.Lock = field(default_factory=threading.Lock)
    latest_report: dict[str, Any] | None = None
    latest_form: dict[str, str] = field(default_factory=lambda: DEFAULT_FORM.copy())

    @property
    def json_path(self) -> Path:
        return self.output_dir / "latest.json"

    @property
    def csv_path(self) -> Path:
        return self.output_dir / "latest.csv"

    @property
    def html_path(self) -> Path:
        return self.output_dir / "latest.html"


def _field(fields: dict[str, str], name: str) -> str:
    return fields.get(name, "").strip()


def _integer(fields: dict[str, str], name: str) -> int:
    value = _field(fields, name)
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name.replace('_', ' ').title()} must be an integer.") from error


def _number(fields: dict[str, str], name: str) -> float:
    value = _field(fields, name)
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name.replace('_', ' ').title()} must be a number.") from error


def form_to_config(fields: dict[str, str]) -> RunConfig:
    control = _field(fields, "control")
    dataset_config = _field(fields, "dataset_config")
    return RunConfig(
        model_id=_field(fields, "model"),
        benchmark_source=_field(fields, "benchmark"),
        control_source=control or None,
        text_field=_field(fields, "text_field") or "text",
        dataset_config=dataset_config or None,
        split=_field(fields, "split") or "test",
        limit=_integer(fields, "limit"),
        k=_number(fields, "k"),
        split_ratio=_number(fields, "split_ratio"),
        repro_threshold=_number(fields, "repro_threshold"),
        device=_field(fields, "device") or "auto",
        seed=_integer(fields, "seed"),
        trust_remote_code=_field(fields, "trust_remote_code") == "on",
    )


def execute_audit(
    state: WebState,
    fields: dict[str, str],
    *,
    model_loader: Callable[..., Any] = load_model,
) -> dict[str, Any]:
    config = form_to_config(fields)
    with state.audit_lock:
        cache_key = (
            config.model_id,
            config.device,
            config.trust_remote_code,
        )
        model = state.model_cache.get(cache_key)
        if model is None:
            model = model_loader(
                config.model_id,
                device=config.device,
                trust_remote_code=config.trust_remote_code,
                seed=config.seed,
            )
            state.model_cache[cache_key] = model
        else:
            set_seed(config.seed)

        benchmark = load_benchmark(
            config.benchmark_source,
            text_field=config.text_field,
            dataset_config=config.dataset_config,
            split=config.split,
            limit=config.limit,
        )
        control = None
        if config.control_source:
            control = load_benchmark(
                config.control_source,
                text_field=config.text_field,
                dataset_config=config.dataset_config,
                split=config.split,
                limit=config.limit,
                sample_control=config.control_source == "sample",
            )
        report = build_report(model, benchmark, control, config)
        state.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(report, state.json_path)
        write_csv(report, state.csv_path)
        write_html(report, state.html_path)
        state.latest_report = report
        state.latest_form = fields.copy()
        return report


def _input(
    label: str,
    name: str,
    value: str,
    *,
    help_text: str,
    input_type: str = "text",
    required: bool = False,
) -> str:
    required_text = " required" if required else ""
    return (
        '<label class="field">'
        f"<span>{escape(label)}</span>"
        f'<input type="{input_type}" name="{escape(name)}" '
        f'value="{escape(value)}"{required_text}>'
        f"<small>{escape(help_text)}</small>"
        "</label>"
    )


def _select(label: str, name: str, value: str, options: list[str]) -> str:
    choices = "".join(
        f'<option value="{escape(option)}"'
        + (" selected" if option == value else "")
        + f">{escape(option)}</option>"
        for option in options
    )
    return (
        '<label class="field">'
        f"<span>{escape(label)}</span>"
        f'<select name="{escape(name)}">{choices}</select>'
        "<small>Automatic uses CUDA, then MPS, then CPU.</small>"
        "</label>"
    )


def _format(value: Any) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.3f}"
    return escape(str(value))


def _result_html(report: dict[str, Any]) -> str:
    interval = report["confidence_interval"]
    interval_text = (
        f"{interval[0]:.3f} to {interval[1]:.3f}"
        if interval is not None
        else "Not estimated"
    )
    method_rows = [
        (
            "Guided overlap gap",
            report["index_guided"],
            f"ROUGE-L gap {_format(report['guided_general_gap'])}",
        ),
        (
            "Min-K%++",
            report["index_minkpp"],
            (
                f"AUC {_format(report['minkpp_auc'])}"
                if report["minkpp_auc"] is not None
                else "Raw distribution only"
            ),
        ),
        (
            "Canonical order",
            report["index_order"],
            f"p-value {_format(report['order_p_value'])}",
        ),
    ]
    methods = "".join(
        "<tr>"
        f"<th>{escape(name)}</th>"
        f"<td>{_format(index)}</td>"
        f"<td>{escape(detail)}</td>"
        "</tr>"
        for name, index, detail in method_rows
    )
    capabilities = "".join(
        '<div class="capability">'
        f"<span>{escape(name.replace('_', ' ').title())}</span>"
        f"<strong>{'Yes' if supported else 'No'}</strong>"
        "</div>"
        for name, supported in report["capabilities"].items()
    )
    caveats = "".join(f"<li>{escape(item)}</li>" for item in report["caveats"])
    skipped = "".join(
        f"<li><strong>{escape(item['method'])}:</strong> {escape(item['reason'])}</li>"
        for item in report["skipped_methods"]
    )
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(item['id']))}</td>"
        f"<td>{_format(item['guided_rouge'])}</td>"
        f"<td>{_format(item['general_rouge'])}</td>"
        f"<td>{_format(item['minkpp_score'])}</td>"
        "</tr>"
        for item in report["per_example"][:20]
    )
    return f"""
<section class="result" id="result">
  <div class="result-head">
    <div>
      <p class="kicker">Latest audit</p>
      <h2>{escape(report['dataset_name'])}</h2>
      <p class="reading">Evidence with uncertainty, not proof of training inclusion.</p>
    </div>
    <div class="score-block">
      <strong>{report['contamination_index']:.3f}</strong>
      <span>{escape(report['band'].title())}</span>
    </div>
  </div>
  <div class="result-meta">
    <span>95% interval: {escape(interval_text)}</span>
    <span>Basis: {escape(report['index_basis'])}</span>
    <span>{report['example_count']} examples</span>
  </div>
  <div class="downloads">
    <a href="/artifacts/latest.json">Download JSON</a>
    <a href="/artifacts/latest.csv">Download CSV</a>
    <a href="/report" target="_blank">Open standalone report</a>
  </div>
</section>
<section>
  <div class="section-heading">
    <div><p class="kicker">Access probe</p><h2>Model capabilities</h2></div>
  </div>
  <div class="capabilities">{capabilities}</div>
</section>
<section>
  <div class="section-heading">
    <div><p class="kicker">Evidence</p><h2>Method results</h2></div>
  </div>
  <div class="table-wrap">
    <table><thead><tr><th>Method</th><th>Index</th><th>Reading</th></tr></thead>
    <tbody>{methods}</tbody></table>
  </div>
</section>
<section class="two-column">
  <div>
    <p class="kicker">Interpretation</p>
    <h2>Caveats</h2>
    <ul>{caveats}</ul>
  </div>
  <div>
    <p class="kicker">Availability</p>
    <h2>Skipped methods</h2>
    <ul>{skipped or '<li>None</li>'}</ul>
  </div>
</section>
<section>
  <div class="section-heading">
    <div><p class="kicker">Inspection</p><h2>Per-example preview</h2></div>
    <span class="muted">First 20 rows</span>
  </div>
  <div class="table-wrap">
    <table><thead><tr><th>ID</th><th>Guided</th><th>General</th><th>Min-K%++</th></tr></thead>
    <tbody>{rows}</tbody></table>
  </div>
</section>
"""


def render_dashboard(
    *,
    form: dict[str, str] | None = None,
    report: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    values = DEFAULT_FORM.copy()
    if form:
        values.update(form)
    checked = " checked" if values.get("trust_remote_code") == "on" else ""
    error_html = (
        f'<div class="error" role="alert"><strong>Audit failed.</strong> '
        f"{escape(error)}</div>"
        if error
        else ""
    )
    result_html = _result_html(report) if report else """
<section class="empty-state">
  <p class="kicker">Ready</p>
  <h2>No audit has run in this session</h2>
  <p>Use the bundled sample for a quick check, or enter a local JSONL/CSV path or HuggingFace dataset ID.</p>
</section>
"""
    form_html = (
        _input(
            "Model",
            "model",
            values["model"],
            help_text="Local HuggingFace causal language model ID or path.",
            required=True,
        )
        + _input(
            "Benchmark",
            "benchmark",
            values["benchmark"],
            help_text="Use sample, a local .jsonl/.csv path, or a HuggingFace dataset ID.",
            required=True,
        )
        + _input(
            "Matched control",
            "control",
            values["control"],
            help_text="Optional. Use sample for the bundled control set.",
        )
        + _input(
            "Text field",
            "text_field",
            values["text_field"],
            help_text="For GLUE SST-2, use sentence.",
            required=True,
        )
        + _input(
            "Dataset config",
            "dataset_config",
            values["dataset_config"],
            help_text="Optional HuggingFace configuration, such as sst2.",
        )
        + _input(
            "Split",
            "split",
            values["split"],
            help_text="Falls back from test to validation to train when possible.",
            required=True,
        )
        + _input(
            "Example limit",
            "limit",
            values["limit"],
            help_text="Start with 10 for a quick local run.",
            input_type="number",
            required=True,
        )
        + _select("Device", "device", values["device"], ["auto", "cpu", "mps", "cuda"])
        + _input(
            "Min-K percent",
            "k",
            values["k"],
            help_text="Lowest token-score percentage used by Min-K methods.",
            input_type="number",
            required=True,
        )
        + _input(
            "Guided split ratio",
            "split_ratio",
            values["split_ratio"],
            help_text="Fraction of tokens kept as the visible prefix.",
            input_type="number",
            required=True,
        )
        + _input(
            "Reproduction threshold",
            "repro_threshold",
            values["repro_threshold"],
            help_text="Guided ROUGE-L threshold for the reproduction rate.",
            input_type="number",
            required=True,
        )
        + _input(
            "Seed",
            "seed",
            values["seed"],
            help_text="Controls permutations and bootstrap resampling.",
            input_type="number",
            required=True,
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ContamCheckr</title>
<style>
:root {{ color-scheme: light; --ink: #13242b; --muted: #637078; --line: #d8e0e3; --paper: #fff; --wash: #f1f5f5; --accent: #15596b; --accent-dark: #0d3f4d; --soft: #e8f1f2; --danger: #8a2c2c; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--wash); color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }}
.topbar {{ background: var(--ink); color: white; }}
.topbar-inner {{ max-width: 1180px; margin: auto; padding: 18px 24px; display: flex; align-items: center; justify-content: space-between; }}
.brand {{ font-size: 18px; font-weight: 750; letter-spacing: -.02em; }}
.local {{ color: #bfd0d5; font-size: 13px; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 36px 24px 64px; }}
.hero {{ display: grid; grid-template-columns: 1.5fr 1fr; gap: 24px; align-items: end; margin-bottom: 28px; }}
.hero h1 {{ font-size: clamp(34px, 5vw, 58px); line-height: 1.02; letter-spacing: -.045em; margin: 8px 0 16px; max-width: 780px; }}
.hero p {{ color: var(--muted); max-width: 720px; font-size: 17px; }}
.hero-aside {{ border-left: 3px solid var(--accent); padding: 4px 0 4px 18px; color: var(--muted); }}
section, .panel {{ background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: 24px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(20, 45, 55, .035); }}
.panel h2, section h2 {{ margin: 0; font-size: 21px; letter-spacing: -.02em; }}
.section-heading, .result-head {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; margin-bottom: 18px; }}
.kicker {{ color: var(--accent); text-transform: uppercase; letter-spacing: .1em; font-size: 11px; font-weight: 750; margin: 0 0 5px; }}
.form-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin-top: 20px; }}
.field {{ display: flex; flex-direction: column; gap: 6px; }}
.field span {{ font-weight: 650; }}
.field small {{ color: var(--muted); min-height: 2.8em; }}
input, select {{ width: 100%; border: 1px solid #bfcbd0; border-radius: 7px; background: white; color: var(--ink); padding: 11px 12px; font: inherit; }}
input:focus, select:focus {{ outline: 3px solid #c9e1e6; border-color: var(--accent); }}
.form-actions {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-top: 22px; padding-top: 20px; border-top: 1px solid var(--line); }}
.checkbox {{ display: flex; align-items: center; gap: 9px; color: var(--muted); }}
.checkbox input {{ width: auto; }}
button {{ border: 0; border-radius: 7px; padding: 12px 20px; background: var(--accent); color: white; font: inherit; font-weight: 700; cursor: pointer; }}
button:hover {{ background: var(--accent-dark); }}
button:disabled {{ opacity: .65; cursor: wait; }}
.run-note {{ color: var(--muted); font-size: 13px; }}
.error {{ background: #fff0f0; border: 1px solid #e4b8b8; color: var(--danger); border-radius: 8px; padding: 14px 16px; margin: 0 0 18px; }}
.result {{ border-top: 5px solid var(--accent); }}
.score-block {{ text-align: right; }}
.score-block strong {{ display: block; color: var(--accent); font-size: 52px; line-height: 1; letter-spacing: -.05em; }}
.score-block span {{ color: var(--muted); font-weight: 650; }}
.reading, .muted {{ color: var(--muted); }}
.result-meta, .downloads {{ display: flex; flex-wrap: wrap; gap: 10px 22px; }}
.result-meta {{ color: var(--muted); padding-top: 15px; border-top: 1px solid var(--line); }}
.downloads {{ margin-top: 18px; }}
.downloads a {{ color: var(--accent); font-weight: 700; text-decoration: none; }}
.capabilities {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
.capability {{ background: var(--soft); border-radius: 8px; padding: 16px; display: flex; justify-content: space-between; gap: 10px; }}
.capability span {{ color: var(--muted); }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 11px 9px; text-align: left; vertical-align: top; }}
thead th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
.table-wrap {{ overflow-x: auto; }}
.two-column {{ display: grid; grid-template-columns: 1fr 1fr; gap: 34px; }}
ul {{ margin: 12px 0 0; padding-left: 20px; }}
.empty-state {{ text-align: center; padding: 50px 24px; }}
.empty-state p:last-child {{ color: var(--muted); }}
@media (max-width: 760px) {{
  .hero, .form-grid, .two-column {{ grid-template-columns: 1fr; }}
  .hero-aside {{ border-left: 0; border-top: 3px solid var(--accent); padding: 16px 0 0; }}
  .capabilities {{ grid-template-columns: 1fr; }}
  .form-actions, .result-head {{ align-items: stretch; flex-direction: column; }}
  .score-block {{ text-align: left; }}
}}
</style>
</head>
<body>
<div class="topbar"><div class="topbar-inner">
  <div class="brand">ContamCheckr</div>
  <div class="local">Local session at 127.0.0.1</div>
</div></div>
<main>
  <div class="hero">
    <div>
      <p class="kicker">Benchmark contamination auditor</p>
      <h1>Check whether a benchmark may have appeared in model training.</h1>
      <p>Run guided overlap, Min-K%++, and canonical-order evidence against a local HuggingFace causal language model.</p>
    </div>
    <div class="hero-aside">Your benchmark text stays on this machine. Network access is used only when HuggingFace needs to download a model or dataset.</div>
  </div>
  <div class="panel">
    <div class="section-heading">
      <div><p class="kicker">New audit</p><h2>Configure the check</h2></div>
      <span class="muted">Start with the bundled sample</span>
    </div>
    {error_html}
    <form method="post" action="/run" id="audit-form">
      <div class="form-grid">{form_html}</div>
      <div class="form-actions">
        <label class="checkbox"><input type="checkbox" name="trust_remote_code"{checked}> Trust model remote code</label>
        <div>
          <span class="run-note" id="run-note">Runs locally and may take a moment.</span>
          <button type="submit" id="run-button">Run contamination check</button>
        </div>
      </div>
    </form>
  </div>
  {result_html}
</main>
<script>
const form = document.getElementById("audit-form");
form.addEventListener("submit", () => {{
  const button = document.getElementById("run-button");
  const note = document.getElementById("run-note");
  button.disabled = true;
  button.textContent = "Running audit...";
  note.textContent = "Loading the model and scoring examples. ";
}});
</script>
</body>
</html>
"""


def _parse_form(body: bytes) -> dict[str, str]:
    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    fields = DEFAULT_FORM.copy()
    for name in DEFAULT_FORM:
        values = parsed.get(name)
        fields[name] = values[-1] if values else ""
    return fields


def make_handler(state: WebState) -> type[BaseHTTPRequestHandler]:
    class ContamCheckHandler(BaseHTTPRequestHandler):
        server_version = "contamcheck"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def _send(
            self,
            content: bytes,
            *,
            status: HTTPStatus = HTTPStatus.OK,
            content_type: str = "text/html; charset=utf-8",
            disposition: str | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; connect-src 'self'",
            )
            if disposition:
                self.send_header("Content-Disposition", disposition)
            self.end_headers()
            self.wfile.write(content)

        def _redirect(self, location: str) -> None:
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                page = render_dashboard(
                    form=state.latest_form,
                    report=state.latest_report,
                )
                self._send(page.encode("utf-8"))
                return
            if path == "/health":
                self._send(
                    json.dumps({"status": "ok"}).encode("utf-8"),
                    content_type="application/json",
                )
                return
            if path == "/report" and state.latest_report is not None:
                self._send(build_html(state.latest_report).encode("utf-8"))
                return
            artifacts = {
                "/artifacts/latest.json": (
                    state.json_path,
                    "application/json",
                    "contamcheck-report.json",
                ),
                "/artifacts/latest.csv": (
                    state.csv_path,
                    "text/csv; charset=utf-8",
                    "contamcheck-per-example.csv",
                ),
            }
            artifact = artifacts.get(path)
            if artifact and artifact[0].exists():
                file_path, content_type, filename = artifact
                self._send(
                    file_path.read_bytes(),
                    content_type=content_type,
                    disposition=f'attachment; filename="{filename}"',
                )
                return
            if path == "/favicon.ico":
                self._send(b"", status=HTTPStatus.NO_CONTENT)
                return
            self._send(
                b"Not found",
                status=HTTPStatus.NOT_FOUND,
                content_type="text/plain; charset=utf-8",
            )

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/run":
                self._send(
                    b"Not found",
                    status=HTTPStatus.NOT_FOUND,
                    content_type="text/plain; charset=utf-8",
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > 64 * 1024:
                self._send(
                    b"Invalid form body.",
                    status=HTTPStatus.BAD_REQUEST,
                    content_type="text/plain; charset=utf-8",
                )
                return
            fields = _parse_form(self.rfile.read(length))
            try:
                report = execute_audit(state, fields)
            except (OSError, RuntimeError, ValueError) as error:
                page = render_dashboard(form=fields, error=str(error))
                self._send(
                    page.encode("utf-8"),
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            state.latest_report = report
            self._redirect("/#result")

    return ContamCheckHandler


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    output_dir: str | Path = ".contamcheck-web",
) -> None:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError(
            "The web interface is localhost-only. "
            "Use 127.0.0.1, ::1, or localhost."
        )
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    state = WebState(Path(output_dir))
    server_type = _IPv6ThreadingHTTPServer if host == "::1" else ThreadingHTTPServer
    server = server_type((host, port), make_handler(state))
    print(f"ContamCheckr is running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
