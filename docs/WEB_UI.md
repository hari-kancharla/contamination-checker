# Local web interface

The web interface runs the same audit code as `contamcheckr run`. It does not have a separate scoring path.

Start it with:

```bash
contamcheckr serve --port 8000
```

Open `http://127.0.0.1:8000`. The server accepts only `127.0.0.1`, `::1`, or `localhost` as bind hosts. It rejects `0.0.0.0` and other network-facing addresses.

## Form fields

The form maps directly to CLI arguments:

| Web field | CLI argument |
| --- | --- |
| Model | `--model` |
| Benchmark | `--benchmark` |
| Matched control | `--control` |
| Text field | `--text-field` |
| Dataset config | `--dataset-config` |
| Split | `--split` |
| Example limit | `--limit` |
| Device | `--device` |
| Min-K percent | `--k` |
| Guided split ratio | `--split-ratio` |
| Reproduction threshold | `--repro-threshold` |
| Seed | `--seed` |
| Trust model remote code | `--trust-remote-code` |

The model is cached in the web process after its first load. Each audit resets the configured random seed before scoring.

## Result parity

The dashboard reads the result dictionary returned by `build_report`. The score, band, confidence interval, basis, caveats, skipped-method reasons, capability flags, and per-method values therefore match the CLI and JSON report for the same inputs.

The page links to:

- `latest.json`, the complete machine-readable report
- `latest.csv`, the per-example table
- the standalone HTML report

Files are written under `.contamcheckr-web/` by default. Use `--output-dir` to choose another local directory.

## Network behavior

Benchmark text is scored in the local process. The tool contacts HuggingFace only when a requested model or dataset is not already cached. There is no telemetry or hosted inference call.
