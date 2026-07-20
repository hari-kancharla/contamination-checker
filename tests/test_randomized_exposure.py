"""Scientific-design and artifact integrity tests; no weights or training."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments import randomized_exposure as experiment


def pool_rows(count: int = 12) -> list[dict[str, str]]:
    return [{"id": f"item-{i:03}", "text": f"Observation {i} has an independently written target."} for i in range(count)]


def write_pool(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    pool = write_pool(tmp_path / "input.jsonl", pool_rows())
    checkpoint = tmp_path / "parent"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"fake"}')
    (checkpoint / "model.safetensors").write_bytes(b"not real weights: never loaded")
    return pool, checkpoint


def settings(**overrides) -> experiment.Settings:
    return experiment.Settings(bootstrap_draws=40, **overrides)


def test_assignment_is_content_independent_and_order_invariant() -> None:
    records = [{**row, "group_id": row["id"]} for row in pool_rows()]
    assigned = experiment.assign_exposure(records, settings())
    modified = [{**row, "text": "Completely changed content " + row["id"]} for row in reversed(records)]
    repeated = experiment.assign_exposure(modified, settings())
    membership = lambda rows: {row["id"]: row["exposed"] for row in rows}
    assert membership(assigned) == membership(repeated)
    assert membership(assigned) != membership(experiment.assign_exposure(records, settings(assignment_seed=42)))
    assert sum(row["exposed"] for row in assigned) == 6


def test_groups_never_cross_arms_and_change_sample_unit(tmp_path: Path) -> None:
    rows = [{**row, "family": f"family-{i // 2}"} for i, row in enumerate(pool_rows())]
    records = experiment.read_pool(write_pool(tmp_path / "groups.jsonl", rows), group_field="family")
    assignment = experiment.assign_exposure(records, settings(group_field="family"))
    exposed = {row["group_id"] for row in assignment if row["exposed"]}
    held_out = {row["group_id"] for row in assignment if not row["exposed"]}
    assert len(exposed) == len(held_out) == 3
    assert exposed.isdisjoint(held_out)
    with pytest.raises(ValueError, match="at least two groups"):
        experiment.assign_exposure(records[:4], settings(group_field="family"))


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows + [rows[0]], "Duplicate id"),
    (lambda rows: rows + [{"id": "other", "text": "  " + rows[0]["text"] + "  "}], "Duplicate text"),
    (lambda rows: rows + [{"id": "other", "text": "   "}], "nonempty text"),
    (lambda rows: rows + [{"id": 123, "text": "Different text"}], "nonempty string id"),
])
def test_invalid_pool_rejected_before_any_output(tmp_path: Path, inputs, mutation, match) -> None:
    pool, model = inputs
    write_pool(pool, mutation(pool_rows()))
    output = tmp_path / "run"
    with pytest.raises(ValueError, match=match):
        experiment.create_plan(pool, output, model, settings())
    assert not output.exists()


def test_plan_is_stdlib_only_and_writes_no_model_outputs(tmp_path: Path, inputs) -> None:
    pool, checkpoint = inputs
    output = tmp_path / "plan"
    result = subprocess.run([
        sys.executable, "-S", str(Path(experiment.__file__)), "plan",
        "--input", str(pool), "--model", str(checkpoint), "--output", str(output),
    ], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert set(path.name for path in output.iterdir()) == {
        "pool.jsonl", "assignment.jsonl", "settings.json", "parent_checkpoint.json",
        "source_manifest.json", "source_snapshot", "plan.json", "plan_seal.json",
    }
    plan = json.loads((output / "plan.json").read_text())
    assert plan["input_file_sha256"] == experiment._file_sha(pool)
    assert plan["n_items"] == 12 and plan["n_exposed"] == 6
    assert not (output / "execution.json").exists()
    with pytest.raises(FileExistsError):
        experiment.create_plan(pool, output, checkpoint, settings())


@pytest.mark.parametrize("filename", ["pool.jsonl", "assignment.jsonl", "settings.json", "plan.json"])
def test_tampered_plan_fails_before_model_loading(tmp_path: Path, inputs, filename: str) -> None:
    pool, checkpoint = inputs
    output = experiment.create_plan(pool, tmp_path / "run", checkpoint, settings())
    with (output / filename).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="modified"):
        experiment.execute_plan(output)
    assert not (output / "execution.json").exists()


def test_parent_checkpoint_mutation_rejected_before_model_loading(tmp_path: Path, inputs) -> None:
    pool, checkpoint = inputs
    output = experiment.create_plan(pool, tmp_path / "run", checkpoint, settings())
    (checkpoint / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Parent checkpoint changed"):
        experiment.execute_plan(output)
    assert not (output / "execution.json").exists()


def test_source_snapshot_is_retained_and_tamper_detected(tmp_path: Path, inputs) -> None:
    pool, checkpoint = inputs
    output = experiment.create_plan(pool, tmp_path / "run", checkpoint, settings())
    snapshot = output / "source_snapshot" / "experiments" / "randomized_exposure.py"
    assert snapshot.read_bytes() == Path(experiment.__file__).read_bytes()
    snapshot.write_text("# edited snapshot\n")
    with pytest.raises(ValueError, match="source snapshot was modified"):
        experiment.execute_plan(output)
    assert not (output / "execution.json").exists()


def test_existing_sealed_plan_is_executable_from_cli_without_overrides(tmp_path: Path, inputs, monkeypatch) -> None:
    pool, checkpoint = inputs
    output = experiment.create_plan(pool, tmp_path / "run", checkpoint, settings())
    runtime, calls = fake_runtime(output)
    monkeypatch.setattr(experiment, "_get_runtime", lambda: runtime)
    assert experiment.main(["run", "--from-plan", str(output)]) == 0
    assert "train" in calls
    assert (output / "summary.json").exists()


def test_cli_cannot_override_sealed_training_seed(tmp_path: Path, inputs) -> None:
    pool, checkpoint = inputs
    output = experiment.create_plan(pool, tmp_path / "run", checkpoint, settings())
    with pytest.raises(SystemExit) as error:
        experiment.main(["run", "--from-plan", str(output), "--training-seed", "42"])
    assert error.value.code == 2
    assert not (output / "execution.json").exists()


def fake_runtime(output: Path, *, fail_training: bool = False, tokenization_drift: bool = False,
                 skipped_parent: bool = False, skipped_adapted: bool = False,
                 count_mismatch: bool = False, default_special_token: bool = False) -> tuple[experiment.Runtime, list[str]]:
    calls: list[str] = []
    members: set[str] = set()

    def load(path, **kwargs):
        assert kwargs["local_files_only"] is True
        adapted = str(path).endswith("adapted_model")
        calls.append("load_adapted" if adapted else "load_parent")
        class Tokenizer:
            def encode(self, text, *, add_special_tokens=True):
                tokens = [ord(char) + int(adapted and tokenization_drift) for char in text]
                return ([999] if add_special_tokens and default_special_token else []) + tokens

            def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
                token_rows = [self.encode(text)[:max_length] for text in texts]
                width = max(map(len, token_rows))
                return {
                    "input_ids": [tokens + [0] * (width - len(tokens)) for tokens in token_rows],
                    "attention_mask": [[1] * len(tokens) + [0] * (width - len(tokens)) for tokens in token_rows],
                }

        tokenizer = Tokenizer()
        return SimpleNamespace(adapted=adapted, tokenizer=tokenizer, model_id=str(path))

    def score(model, texts, **kwargs):
        calls.append("score_adapted" if model.adapted else "score_parent")
        encoded = model.tokenizer(texts, padding=True, truncation=True, max_length=kwargs["max_tokens"], return_tensors="pt")
        scored = []
        for index, text in enumerate(texts):
            boost = float(model.adapted and text in members)
            skipped = int(skipped_adapted if model.adapted else skipped_parent)
            count = sum(encoded["attention_mask"][index]) - 1 - skipped + int(count_mismatch)
            scored.append(SimpleNamespace(loss=-10. + boost, mink=-11. + boost, minkpp=-12. + boost,
                token_count=count, skipped_token_count=skipped, truncated=False))
        return scored

    def train(parent, adapted, examples, **kwargs):
        calls.append("train")
        assert (output / "plan_seal.json").is_file()
        assert (output / "parent_scores.jsonl").is_file()
        assert not (output / "adapted_model").exists()
        assert kwargs["include_order_canaries"] is False
        members.update(row["text"] for row in examples)
        assignment = experiment._read_rows(output / "assignment.jsonl")
        assert {row["id"] for row in examples} == {row["id"] for row in assignment if row["exposed"]}
        if fail_training:
            raise RuntimeError("Injected training failure")
        adapted.mkdir()
        (adapted / "config.json").write_text("{}")
        (adapted / "model.safetensors").write_bytes(b"fake adapted weights")
        return {"seed": kwargs["seed"], "epochs": kwargs["epochs"], "record_count": len(examples)}

    return experiment.Runtime(load, train, score, lambda model: calls.append("release")), calls


def test_end_to_end_saves_aligned_vectors_and_paired_comparisons(tmp_path: Path, inputs) -> None:
    pool, checkpoint = inputs
    output = experiment.create_plan(pool, tmp_path / "run", checkpoint, settings())
    runtime, calls = fake_runtime(output)
    summary = experiment.execute_plan(output, runtime=runtime)
    assert calls == ["load_parent", "score_parent", "release", "train", "load_adapted", "score_adapted", "release"]
    rows = experiment._read_rows(output / "per_item.jsonl")
    assert len(rows) == 12
    assert all(row["delta"]["loss"] == int(row["exposed"]) for row in rows)
    assert summary["metrics"]["parent/loss"]["auc"] == .5
    assert summary["metrics"]["adapted/loss"]["auc"] == 1.
    assert summary["metrics"]["delta/loss"]["auc"] == 1.
    assert summary["metrics"]["adapted/minkpp"]["auc_difference_vs_loss"] == 0.
    assert summary["metrics"]["adapted/minkpp"]["difference_ci_percentile_95"] == [0., 0.]
    assert summary["metrics"]["adapted/loss"]["auc_bootstrap_degenerate"] is True
    assert json.loads((output / "execution.json").read_text())["status"] == "complete"
    assert experiment._file_sha(output / "summary.json") == json.loads((output / "execution.json").read_text())["summary_sha256"]
    assert json.loads((output / "training_metadata.json").read_text())["record_count"] == 6
    with pytest.raises(FileExistsError, match="already started"):
        experiment.execute_plan(output, runtime=runtime)


def test_failed_training_preserves_parent_vectors_and_blocks_overwrite(tmp_path: Path, inputs) -> None:
    pool, checkpoint = inputs
    output = experiment.create_plan(pool, tmp_path / "run", checkpoint, settings())
    runtime, _ = fake_runtime(output, fail_training=True)
    with pytest.raises(RuntimeError, match="Injected training failure"):
        experiment.execute_plan(output, runtime=runtime)
    assert (output / "parent_scores.jsonl").is_file()
    assert not (output / "summary.json").exists()
    assert json.loads((output / "execution.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError, match="already started"):
        experiment.execute_plan(output, runtime=runtime)


def test_tokenization_drift_cannot_silently_change_paired_delta(tmp_path: Path, inputs) -> None:
    pool, checkpoint = inputs
    output = experiment.create_plan(pool, tmp_path / "run", checkpoint, settings())
    runtime, _ = fake_runtime(output, tokenization_drift=True)
    with pytest.raises(ValueError, match="tokenization changed"):
        experiment.execute_plan(output, runtime=runtime)
    assert json.loads((output / "execution.json").read_text())["status"] == "failed"
    assert not (output / "per_item.jsonl").exists()


def test_actual_scorer_token_ids_are_retained_not_a_raw_encode_proxy(tmp_path: Path, inputs) -> None:
    pool, checkpoint = inputs
    output = experiment.create_plan(pool, tmp_path / "run", checkpoint, settings())
    runtime, calls = fake_runtime(output, default_special_token=True)
    with pytest.raises(ValueError, match="default special tokens"):
        experiment.execute_plan(output, runtime=runtime)
    parent_rows = experiment._read_rows(output / "parent_scores.jsonl")
    assert all(row["token_ids"][0] == 999 for row in parent_rows)
    assert all(row["token_ids_sha256"] == experiment._sha(experiment._bytes(row["token_ids"])) for row in parent_rows)
    assert all(row["training_text_prefix_matches"] is False for row in parent_rows)
    assert "train" not in calls


def test_duplicate_retained_prefixes_rejected_before_training_with_artifact(tmp_path: Path, inputs) -> None:
    pool, checkpoint = inputs
    output = experiment.create_plan(pool, tmp_path / "run", checkpoint, settings(max_tokens=4))
    runtime, calls = fake_runtime(output)
    with pytest.raises(ValueError, match="Duplicate effective scored input"):
        experiment.execute_plan(output, runtime=runtime)
    parent_rows = experiment._read_rows(output / "parent_scores.jsonl")
    assert len({row["token_ids_sha256"] for row in parent_rows}) == 1
    assert "train" not in calls
    assert json.loads((output / "execution.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("stage", ["parent", "adapted"])
def test_skipped_targets_fail_conservatively_and_preserve_stage_vectors(tmp_path: Path, inputs, stage: str) -> None:
    pool, checkpoint = inputs
    output = experiment.create_plan(pool, tmp_path / "run", checkpoint, settings())
    runtime, calls = fake_runtime(output, **{f"skipped_{stage}": True})
    with pytest.raises(ValueError, match="support mismatch or skipped"):
        experiment.execute_plan(output, runtime=runtime)
    rows = experiment._read_rows(output / f"{stage}_scores.jsonl")
    assert all(row["skipped_token_count"] == 1 for row in rows)
    assert ("train" in calls) == (stage == "adapted")
    assert not (output / "summary.json").exists()


def test_target_count_mismatch_cannot_be_silently_accepted(tmp_path: Path, inputs) -> None:
    pool, checkpoint = inputs
    output = experiment.create_plan(pool, tmp_path / "run", checkpoint, settings())
    runtime, calls = fake_runtime(output, count_mismatch=True)
    with pytest.raises(ValueError, match="support mismatch"):
        experiment.execute_plan(output, runtime=runtime)
    assert "train" not in calls
    assert (output / "parent_scores.jsonl").is_file()


@pytest.mark.parametrize("positive,negative,expected", [([1., 2.], [0., 1.], .875), ([1., 1.], [1., 1.], .5), ([0., 0.], [1., 1.], 0.)])
def test_tie_aware_auc_against_hand_counted_cases(positive, negative, expected) -> None:
    assert experiment.auc(positive, negative) == expected


def test_score_direction_is_never_selected_after_seeing_labels() -> None:
    rows = []
    for index in range(8):
        exposed = index < 4
        values = {"loss": float(exposed), "mink": float(not exposed), "minkpp": float(exposed)}
        rows.append({"id": str(index), "group_id": str(index), "exposed": exposed, "parent": values, "adapted": values, "delta": values})
    summary = experiment.summarize(rows, settings())
    assert summary["metrics"]["adapted/mink"]["auc"] == 0.
    assert summary["metrics"]["adapted/mink"]["auc_difference_vs_loss"] == -1.
    assert summary["metrics"]["adapted/mink"]["difference_ci_percentile_95"] == [-1., -1.]
