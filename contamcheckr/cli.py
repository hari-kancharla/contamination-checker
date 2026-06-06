import argparse
import json
import sys
from typing import Sequence

from .benchmarks import load_benchmark
from .config import RunConfig
from .html_report import write_html
from .model import load_model
from .report import build_report, format_summary, write_csv, write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contamcheckr",
        description="Estimate benchmark contamination evidence for a local causal LM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capabilities = subparsers.add_parser(
        "capabilities",
        help="Load a model and report supported evidence methods.",
    )
    capabilities.add_argument("--model", required=True)
    capabilities.add_argument("--device", default="auto")
    capabilities.add_argument("--trust-remote-code", action="store_true")

    run = subparsers.add_parser("run", help="Run a contamination evidence audit.")
    run.add_argument("--model", required=True)
    run.add_argument("--benchmark", required=True)
    run.add_argument("--control")
    run.add_argument("--text-field", default="text")
    run.add_argument("--dataset-config")
    run.add_argument("--split", default="test")
    run.add_argument("--limit", type=int, default=200)
    run.add_argument("--k", type=float, default=20.0)
    run.add_argument("--split-ratio", type=float, default=0.5)
    run.add_argument("--repro-threshold", type=float, default=0.6)
    run.add_argument("--device", default="auto")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output")
    run.add_argument("--csv")
    run.add_argument("--html")
    run.add_argument("--trust-remote-code", action="store_true")

    serve = subparsers.add_parser(
        "serve",
        help="Run the interactive local contamination checker.",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--output-dir", default=".contamcheckr-web")
    return parser


def _capabilities(args: argparse.Namespace) -> int:
    model = load_model(
        args.model,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    payload = {
        "model_id": args.model,
        "device": str(model.device),
        "capabilities": model.capabilities.as_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    config = RunConfig(
        model_id=args.model,
        benchmark_source=args.benchmark,
        control_source=args.control,
        text_field=args.text_field,
        dataset_config=args.dataset_config,
        split=args.split,
        limit=args.limit,
        k=args.k,
        split_ratio=args.split_ratio,
        repro_threshold=args.repro_threshold,
        device=args.device,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )
    model = load_model(
        config.model_id,
        device=config.device,
        trust_remote_code=config.trust_remote_code,
        seed=config.seed,
    )
    capability_line = ", ".join(
        f"{name}={'yes' if supported else 'no'}"
        for name, supported in model.capabilities.as_dict().items()
    )
    print(f"Capabilities: {capability_line}", file=sys.stderr)
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
    if args.output:
        write_json(report, args.output)
    if args.csv:
        write_csv(report, args.csv)
    if args.html:
        write_html(report, args.html)
    print(format_summary(report))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "capabilities":
            return _capabilities(args)
        if args.command == "serve":
            from .web import run_server

            run_server(args.host, args.port, args.output_dir)
            return 0
        return _run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"contamcheckr: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
