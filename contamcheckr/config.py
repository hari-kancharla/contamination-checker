from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    model_id: str
    benchmark_source: str
    control_source: str | None = None
    text_field: str = "text"
    dataset_config: str | None = None
    split: str = "test"
    limit: int = 200
    k: float = 20.0
    split_ratio: float = 0.5
    repro_threshold: float = 0.6
    device: str = "auto"
    seed: int = 0
    trust_remote_code: bool = False
    shard_count: int = 10
    permutation_count: int = 25
    max_tokens: int = 512
    chat_template: bool = False
    cache_enabled: bool = True
    cache_path: str | None = None
    dtype: str = "auto"
    revision: str | None = None
    local_files_only: bool = False
    generation_batch_size: int = 4
    scoring_batch_size: int = 4
    bootstrap_samples: int = 2000
    min_calibration_examples: int = 4
    conformal_fdr: float = 0.10
    guided_stability: bool = False
    scoring_mode: str = "truncated"
    window_tokens: int = 256
    window_stride_tokens: int | None = None
    canonicalize_scores: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("Model identifier cannot be empty.")
        if not self.benchmark_source.strip():
            raise ValueError("Benchmark source cannot be empty.")
        if self.limit < 1:
            raise ValueError("Limit must be at least 1.")
        if not 0.0 < self.k <= 100.0:
            raise ValueError("k must be greater than 0 and at most 100.")
        if not 0.0 < self.split_ratio < 1.0:
            raise ValueError("Split ratio must be between 0 and 1.")
        if not 0.0 <= self.repro_threshold <= 1.0:
            raise ValueError("Reproduction threshold must be between 0 and 1.")
        if self.shard_count < 2:
            raise ValueError("Shard count must be at least 2.")
        if self.permutation_count < 1:
            raise ValueError("Permutation count must be at least 1.")
        if self.max_tokens < 2:
            raise ValueError("Max tokens must be at least 2.")
        if self.generation_batch_size < 1:
            raise ValueError("Generation batch size must be at least 1.")
        if self.scoring_batch_size < 1:
            raise ValueError("Scoring batch size must be at least 1.")
        if self.bootstrap_samples < 1:
            raise ValueError("Bootstrap samples must be at least 1.")
        if self.min_calibration_examples < 2:
            raise ValueError("Minimum calibration examples must be at least 2.")
        if not 0.0 < self.conformal_fdr < 1.0:
            raise ValueError("Conformal FDR level must be between 0 and 1.")
        if self.scoring_mode not in {"truncated", "windowed"}:
            raise ValueError("Scoring mode must be 'truncated' or 'windowed'.")
        if self.window_tokens < 2:
            raise ValueError("Window tokens must be at least 2.")
        if self.window_stride_tokens is not None and not (
            1 <= self.window_stride_tokens <= self.window_tokens
        ):
            raise ValueError(
                "Window stride must be between 1 and window_tokens."
            )

