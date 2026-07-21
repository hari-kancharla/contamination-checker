# CPU/MPS forward diagnostic

This diagnostic uses one previously observed development record after the complete 3,000-record study exits. It verifies the checkpoint, original causal IDs and retained MPS evidence, then performs one CPU-FP16 forward.

CPU-FP64 scoring on both forwards holds the statistic device, dtype and function fixed. Direct FP32 comparisons additionally include device-specific scoring-kernel differences. Report complete target support, per-token differences, aggregate differences and selected-tail overlap. There is no predefined parity threshold and no extrapolation to membership AUC.

Run `python experiments/checkmia_forward_fidelity.py --help` for required paths. The historical evidence capsule is required and is not included in the public tree; see [reproduction scope](REPRODUCE_REAL_RESEARCH.md). Use a fresh output directory and preserve incomplete support.
