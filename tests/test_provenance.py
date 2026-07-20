"""Offline checks that provenance identifies score-changing inputs."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from contamcheckr.hashing import (
    hash_examples,
    hash_ordered_examples,
    local_model_signature,
    model_fingerprint,
    source_snapshot,
    tokenizer_fingerprint,
)
from contamcheckr.model import load_model


class FakeTokenizer:
    name_or_path = "vendor/tokenizer"
    vocab_size = 3
    model_max_length = 128
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 1
    unk_token_id = 2
    padding_side = "right"
    truncation_side = "right"
    chat_template = "{{ messages }}"
    special_tokens_map = {"pad_token": "<pad>"}

    def __init__(self) -> None:
        self.init_kwargs = {"_commit_hash": "tokenizer-commit", "add_prefix_space": False}

    def get_vocab(self):
        return {"<pad>": 0, "a": 1, "b": 2}


class FakeBackend:
    def __init__(self) -> None:
        self.settings = {
            "model": {"type": "BPE", "merges": [["a", "b"]]},
            "normalizer": {"type": "Lowercase"},
            "padding": None,
            "truncation": None,
        }

    def to_str(self):
        return json.dumps(self.settings)


def test_ordered_hash_records_permutation_and_avoids_delimiter_ambiguity() -> None:
    records = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]
    assert hash_examples(records) == hash_examples(reversed(records))
    assert hash_ordered_examples(records) != hash_ordered_examples(reversed(records))
    assert hash_ordered_examples(records) == hash_ordered_examples(iter(records))
    assert hash_ordered_examples([{"id": "a\0b", "text": "c"}]) != hash_ordered_examples(
        [{"id": "a", "text": "b\0c"}]
    )


def test_source_digest_uses_content_and_relative_paths(tmp_path: Path) -> None:
    package = tmp_path / "contamcheckr"
    package.mkdir()
    source = package / "engine.py"
    source.write_text("value = 1\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "example"\n')
    initial = source_snapshot(tmp_path)
    assert initial["complete"] is True
    assert initial["file_count"] == 2
    assert "contamcheckr/engine.py" in initial["files"]
    os.utime(source, (1, 1))
    assert source_snapshot(tmp_path) == initial
    (tmp_path / "report.json").write_text('{"result": 1}\n')
    assert source_snapshot(tmp_path) == initial
    source.write_text("value = 2\n")  # Same file length; content still changes.
    assert source_snapshot(tmp_path)["sha256"] != initial["sha256"]
    source.write_text("value = 1\n")
    assert source_snapshot(tmp_path) == initial
    (package / "uncommitted.py").write_text("new = True\n")
    assert source_snapshot(tmp_path)["sha256"] != initial["sha256"]


def test_source_digest_is_explicitly_unavailable_without_sources(tmp_path: Path) -> None:
    result = source_snapshot(tmp_path)
    assert result["sha256"] is None
    assert result["complete"] is False


def test_saved_experiment_sources_do_not_change_live_source_identity(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "runner.py").write_text("x = 1\n")
    original = source_snapshot(tmp_path)
    archived = experiments / "results" / "pilot" / "source_snapshot"
    archived.mkdir(parents=True)
    (archived / "runner.py").write_text("x = 2\n")
    assert source_snapshot(tmp_path) == original


def test_tokenizer_fingerprint_includes_processing_not_temporary_limits() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.backend_tokenizer = FakeBackend()
    original = tokenizer_fingerprint(tokenizer)
    tokenizer.backend_tokenizer.settings["padding"] = {"length": 128}
    tokenizer.backend_tokenizer.settings["truncation"] = {"max_length": 64}
    assert tokenizer_fingerprint(tokenizer) == original
    tokenizer.backend_tokenizer.settings["normalizer"] = None
    assert tokenizer_fingerprint(tokenizer) != original
    tokenizer.backend_tokenizer.settings["normalizer"] = {"type": "Lowercase"}
    tokenizer.backend_tokenizer.settings["model"]["merges"] = [["b", "a"]]
    assert tokenizer_fingerprint(tokenizer) != original


def test_tokenizer_fingerprint_includes_slow_bpe_and_settings() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.bpe_ranks = {("a", "b"): 0, ("b", "a"): 1}
    original = tokenizer_fingerprint(tokenizer)
    tokenizer.bpe_ranks = {("a", "b"): 1, ("b", "a"): 0}
    assert tokenizer_fingerprint(tokenizer) != original
    for name, value in (("chat_template", "different"), ("padding_side", "left")):
        changed = FakeTokenizer()
        setattr(changed, name, value)
        assert tokenizer_fingerprint(changed) != tokenizer_fingerprint(FakeTokenizer())
    changed = FakeTokenizer()
    changed.init_kwargs["add_prefix_space"] = True
    assert tokenizer_fingerprint(changed) != tokenizer_fingerprint(FakeTokenizer())


def test_model_fingerprint_changes_with_resolved_commit() -> None:
    kwargs = dict(model_id="vendor/model", revision="main", dtype="float32", tokenizer=FakeTokenizer())
    assert model_fingerprint(**kwargs, resolved_revision="first") != model_fingerprint(
        **kwargs, resolved_revision="second"
    )
    assert model_fingerprint(**kwargs, tokenizer_revision="first") != model_fingerprint(
        **kwargs, tokenizer_revision="second"
    )


class FakeModel(torch.nn.Module):
    def __init__(self, commit: str | None = "model-commit") -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.config = SimpleNamespace(vocab_size=3, max_position_embeddings=128, _commit_hash=commit)

    def generate(self, *args, **kwargs):
        raise AssertionError("No inference should run in provenance tests.")


def _stub_loaders(monkeypatch, model: FakeModel, tokenizer: FakeTokenizer) -> None:
    monkeypatch.setattr("contamcheckr.model.AutoTokenizer.from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr("contamcheckr.model.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: model)


def test_load_model_exposes_resolved_revisions_without_claiming_weight_digest(monkeypatch) -> None:
    _stub_loaders(monkeypatch, FakeModel(), FakeTokenizer())
    loaded = load_model("vendor/model", device="cpu", local_files_only=True, revision="main")
    assert loaded.revision == "main"
    assert loaded.resolved_revision == "model-commit"
    assert loaded.tokenizer_revision == "tokenizer-commit"
    assert loaded.provenance["weight_identity"] == "hub_commit_reference"
    assert loaded.provenance["weight_sha256"] is None


def test_local_model_signature_and_loaded_identity_are_metadata_only(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"fake weights")
    _stub_loaders(monkeypatch, FakeModel(None), FakeTokenizer())
    loaded = load_model(str(tmp_path), device="cpu", local_files_only=True)
    assert loaded.provenance["weight_identity"] == "local_metadata_signature"
    assert loaded.provenance["weight_sha256"] is None
    signature = local_model_signature(str(tmp_path))
    assert signature["cryptographic_weights"] is False
    assert signature["identity_kind"] == "file_size_and_mtime_signature"
    assert signature["files"][0]["size"] == 12


def test_load_model_keeps_unexposed_revision_unknown(monkeypatch) -> None:
    tokenizer = FakeTokenizer()
    tokenizer.init_kwargs.clear()
    _stub_loaders(monkeypatch, FakeModel(None), tokenizer)
    loaded = load_model("vendor/model", device="cpu", local_files_only=True, revision="main")
    assert loaded.resolved_revision is None
    assert loaded.tokenizer_revision is None
    assert loaded.provenance["weight_identity"] == "unresolved"
