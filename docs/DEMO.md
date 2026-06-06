# Demo

Use Python 3.10 or newer from a fresh checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Check the local model backend:

```bash
contamcheckr capabilities --model sshleifer/tiny-gpt2
```

Run the bundled sample benchmark:

```bash
contamcheckr run \
  --model sshleifer/tiny-gpt2 \
  --benchmark sample \
  --limit 10 \
  --output report.json \
  --csv per_example.csv \
  --html report.html
```

`report.json` is the machine-readable result, `per_example.csv` contains one row per benchmark example, and `report.html` is a standalone static report.

Run the interactive local interface:

```bash
contamcheckr serve --port 8000
```

Then open `http://127.0.0.1:8000`.

The server binds to loopback only. The web page displays the same report band, caveats, method values, and skipped-method reasons as the CLI and JSON output.

Add the bundled matched control to calibrate Min-K%++:

```bash
contamcheckr run \
  --model sshleifer/tiny-gpt2 \
  --benchmark sample \
  --control sample \
  --limit 10 \
  --output controlled_report.json \
  --csv controlled_per_example.csv \
  --html controlled_report.html
```

Model and HuggingFace dataset downloads require network access on first use. Once cached, local model scoring does not send benchmark text to a hosted inference service.
