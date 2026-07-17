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
    parser.add_argument("--version", action="version", version="contamcheckr 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolution = subparsers.add_parser(
        "resolution",
        help="Plan ideal exact-BH reference resolution for the individual screen.",
    )
    resolution.add_argument("--candidates", required=True, type=int)
    resolution.add_argument("--controls", required=True, type=int)
    resolution.add_argument("--q", default="0.10", help="Exact decimal BH level strictly between 0 and 1.")
    resolution.add_argument("--min-calibration-examples", type=int, default=4)

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
    run.add_argument("--dtype", default="auto")
    run.add_argument("--revision")
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--chat-template", action="store_true")
    run.add_argument("--guided-stability", action="store_true")
    run.add_argument("--scoring-mode", choices=["truncated", "windowed"], default="truncated")
    run.add_argument("--canonicalize", action="store_true")
    run.add_argument("--remote-base-url", help="Black-box mode: audit against an OpenAI-style completions endpoint instead of loading weights.")
    run.add_argument("--remote-model", help="API model name for black-box mode (defaults to --model).")
    run.add_argument("--remote-api-key-env", default="CONTAMCHECKR_API_KEY")
    run.add_argument("--window-tokens", type=int, default=256)
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--shard-count", type=int, default=10)
    run.add_argument("--permutation-count", type=int, default=25)
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--cache-path")
    run.add_argument("--generation-batch-size", type=int, default=4)
    run.add_argument("--scoring-batch-size", type=int, default=4)
    run.add_argument("--bootstrap-samples", type=int, default=2000)
    run.add_argument("--min-calibration-examples", type=int, default=4)
    run.add_argument("--conformal-fdr", type=float, default=0.10)

    serve = subparsers.add_parser(
        "serve",
        help="Run the interactive local contamination checker.",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--output-dir", default=".contamcheckr-web")
    return parser


def _progress(label: str, done: int, total: int) -> None:
    print(f"{label}: {done}/{total}", file=sys.stderr)


def _resolution(args: argparse.Namespace) -> int:
    from .resolution import reference_resolution

    payload = reference_resolution(
        args.candidates, args.controls, q=args.q,
        min_calibration_examples=args.min_calibration_examples,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _capabilities(args: argparse.Namespace) -> int:
    model = load_model(
        args.model,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        dtype=getattr(args, "dtype", "auto"),
        revision=getattr(args, "revision", None),
        local_files_only=getattr(args, "local_files_only", False),
        progress=_progress,
    )
    payload = {
        "model_id": args.model,
        "device": str(model.device),
        "capabilities": model.capabilities.as_dict(),
        "scoring_engine": {
            "batched_token_stats": hasattr(model, "token_stats_batch"),
            "batched_sequence_ll": hasattr(model, "sequence_loglikelihood_batch"),
            "batched_generation": hasattr(model, "generate_batch"),
        },
        "model_fingerprint": model.fingerprint,
        "context_limit": model.context_limit,
        "parameter_count": model.parameter_count,
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
        chat_template=args.chat_template,
        guided_stability=args.guided_stability,
        scoring_mode=args.scoring_mode,
        canonicalize_scores=args.canonicalize,
        window_tokens=args.window_tokens,
        cache_enabled=not args.no_cache,
        cache_path=args.cache_path,
        dtype=args.dtype,
        revision=args.revision,
        local_files_only=args.local_files_only,
        max_tokens=args.max_tokens,
        shard_count=args.shard_count,
        permutation_count=args.permutation_count,
        generation_batch_size=args.generation_batch_size,
        scoring_batch_size=args.scoring_batch_size,
        bootstrap_samples=args.bootstrap_samples,
        min_calibration_examples=args.min_calibration_examples,
        conformal_fdr=args.conformal_fdr,
    )
    if args.remote_base_url:
        from .api_adapter import load_remote_model

        model = load_remote_model(
            config.model_id,
            base_url=args.remote_base_url,
            api_key_env=args.remote_api_key_env,
            model_name=args.remote_model,
        )
    else:
        model = load_model(
            config.model_id,
            device=config.device,
            dtype=config.dtype,
            trust_remote_code=config.trust_remote_code,
            seed=config.seed,
            revision=config.revision,
            local_files_only=config.local_files_only,
            progress=_progress,
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
        if args.command == "resolution":
            return _resolution(args)
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
