from pathlib import Path

import numpy as np
import pytest
import torch

from contamcheckr.cache import NullCache, ScoreCache, cache_key, open_cache
from contamcheckr.hashing import hash_examples, hash_text, stable_hash
from contamcheckr.methods.token_stats import (
    TokenStats,
    score_token_stats,
    token_stats_from_logits,
    zlib_ratio,
)
from contamcheckr.model import load_model


ROOT = Path(__file__).resolve().parent.parent
TINY_MODEL = ROOT / "experiments" / "models" / "sshleifer--tiny-gpt2"


def test_stable_hash_is_deterministic_and_bounded() -> None:
    assert stable_hash("abc") == stable_hash(b"abc")
    assert len(stable_hash("abc", length=16)) == 16
    with pytest.raises(ValueError):
        stable_hash("abc", length=4)


def test_hash_examples_is_order_independent_but_content_sensitive() -> None:
    first = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]
    second = list(reversed(first))
    assert hash_examples(first) == hash_examples(second)
    changed = [{"id": "a", "text": "alpha!"}, {"id": "b", "text": "beta"}]
    assert hash_examples(first) != hash_examples(changed)
    assert hash_text("x") == hash_text("x")


def test_cache_roundtrip_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite3"
    key = cache_key("model", "text", 512)
    with ScoreCache(path) as cache:
        cache.put_token_stats(key, {"target_logprobs": [1.0, 2.0]})
        cache.put_sequence_ll(key, -12.5)
        cache.put_generation(key, "hello")
        assert cache.get_token_stats(key) == {"target_logprobs": [1.0, 2.0]}
        assert cache.get_sequence_ll(key) == -12.5
        assert cache.get_generation(key) == "hello"
        assert cache.stats() == {
            "token_stats": 1,
            "sequence_ll": 1,
            "generations": 1,
        }
    with ScoreCache(path) as reopened:
        assert reopened.get_sequence_ll(key) == -12.5


def test_cache_ignores_corrupt_payload(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite3"
    cache = ScoreCache(path)
    cache._connection.execute(
        "INSERT INTO token_stats (key, payload, created_at) VALUES (?, ?, ?)",
        ("bad", "not-json", 0.0),
    )
    cache._connection.commit()
    assert cache.get_token_stats("bad") is None
    cache.close()


def test_null_cache_and_disabled_open() -> None:
    cache = open_cache(enabled=False)
    assert isinstance(cache, NullCache)
    cache.put_sequence_ll("x", 1.0)
    assert cache.get_sequence_ll("x") is None


def test_token_stats_roundtrip() -> None:
    stats = TokenStats(
        target_logprobs=(-1.0, -2.0),
        means=(-3.0, -4.0),
        stds=(1.0, 2.0),
        entropies=(3.0, 4.0),
        top1_probs=(0.5, 0.25),
        ranks=(1, 2),
        token_count=2,
        truncated=True,
    )
    restored = TokenStats.from_dict(stats.to_dict())
    assert restored == stats


def test_token_stats_from_logits_matches_manual() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0, -1.0],
            [0.0, 1.0, 2.0],
            [1.5, -0.5, 0.0],
        ]
    )
    targets = torch.tensor([0, 2, 0])
    stats = token_stats_from_logits(logits, targets)
    scores = score_token_stats(stats, "some text", k=20.0)
    assert scores.mink == pytest.approx(-0.40760595, abs=1e-4)
    assert scores.minkpp == pytest.approx(0.420817, abs=1e-4)
    assert scores.token_count == 3
    assert scores.skipped_token_count == 0
    assert 0.0 < scores.zlib_ratio < 2.0


def test_token_stats_chunking_does_not_change_scores() -> None:
    torch.manual_seed(2)
    logits = torch.randn(17, 11)
    targets = torch.randint(0, 11, (17,))
    whole = token_stats_from_logits(logits, targets, token_chunk_size=64)
    chunked = token_stats_from_logits(logits, targets, token_chunk_size=3)
    assert whole.target_logprobs == pytest.approx(chunked.target_logprobs)
    assert whole.stds == pytest.approx(chunked.stds)


def test_zlib_ratio_is_positive() -> None:
    assert zlib_ratio("") == 1.0
    assert zlib_ratio("aaaaaaaaaaaaaaaa") < 1.0
    assert zlib_ratio("the quick brown fox") > 0.0


@pytest.fixture(scope="module")
def tiny_model():
    if not TINY_MODEL.exists():
        pytest.skip("Local tiny model fixture is missing.")
    return load_model(str(TINY_MODEL), device="cpu", local_files_only=True)


def test_batched_token_stats_match_single(tiny_model) -> None:
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Archive card LQ-100 records that the amber dial faces north.",
        "A triangle has three sides and three interior angles.",
    ]
    batched = tiny_model.token_stats_batch(texts, max_tokens=64, batch_size=3)
    singles = [tiny_model.token_stats(text, max_tokens=64) for text in texts]
    for left, right in zip(batched, singles, strict=True):
        assert left.target_logprobs == pytest.approx(right.target_logprobs, abs=1e-5)
        assert left.means == pytest.approx(right.means, abs=1e-5)


def test_batched_sequence_likelihood_matches_single(tiny_model) -> None:
    texts = ["alpha beta gamma", "one two three four five"]
    batched = tiny_model.sequence_loglikelihood_batch(
        texts, max_tokens=32, batch_size=2
    )
    singles = [tiny_model.sequence_loglikelihood(text, max_tokens=32) for text in texts]
    assert batched == pytest.approx(singles, abs=1e-4)


def test_generate_batch_returns_one_string_per_prompt(tiny_model) -> None:
    outputs = tiny_model.generate_batch(["hello", "world"], 2, batch_size=2)
    assert len(outputs) == 2
    assert all(isinstance(value, str) for value in outputs)


def test_effective_batch_size_is_bounded(tiny_model) -> None:
    assert tiny_model._effective_batch_size(1, 512) == 1
    assert tiny_model._effective_batch_size(64, 512) >= 1
