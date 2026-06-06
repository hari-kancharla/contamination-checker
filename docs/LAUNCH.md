# Launch notes

- Positioning: a local, CPU-friendly contamination evidence auditor with explicit method gates and caveats.
- Controlled result: the adapted tiny model scored 0.492 on its leak set, versus 0.008 on the clean set and 0.131 for the base model on the leak set.
- Public example: `distilgpt2` on 10 GLUE SST-2 validation examples scored 0.275, with an interval from 0.000 to 0.275. Treat it as weak and uncertain.
- Reproduce the controlled run: `.venv/bin/python experiments/validation.py`
- Reproduce the public run: `contamcheck run --model distilgpt2 --benchmark nyu-mll/glue --dataset-config sst2 --split validation --text-field sentence --limit 10 --device cpu --output docs/example_report.json --csv docs/example_per_example.csv --html docs/example_report.html`
- Channels: Show HN, r/MachineLearning, ML evaluation community Slack or Discord groups, and a short maintainer post on LinkedIn or X.
- Keep both results as measured. Do not turn either one into a claim about training-set membership.
- Never buy stars or fake engagement.
