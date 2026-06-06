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

