#!/usr/bin/env python3
"""Portable pinned research sources; default verification is offline/read-only.

Only --download fetches missing public files/materializes a missing archive.
Existing assets and receipts are never overwritten. No model/upstream program
is loaded or executed, and no credentials/access terms are handled.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tarfile
import tempfile
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT = Path(__file__).resolve().parent.parent
OLMO_REVISION = "6eb88de34a09dea85c4b41deff4048e5a203e1b0"
OLMO_ARCHIVE = {"bytes": 35215846, "sha256": "d7a157423bf537a343975db13bfddd9b271a7c24a9cb7ecc8f6b2342ca8d30ce",
                "url": f"https://codeload.github.com/LuckyDaydreamer/OLMo-Detect/tar.gz/{OLMO_REVISION}"}
HF_ASSETS = [
    {
        "asset": "checkmia",
        "repo_id": "jeffreygwang/pythia_dedupe_mia_0-97000_97000-98500",
        "repo_type": "dataset",
        "revision": "236fd818e7c877518d588eacc44658bdbf6b3a1b",
        "files": {
            "README.md": {
                "bytes": 438,
                "sha256": "4cf54dc2794e5f73407580d6b9f8dbaf9122acdc076d7a46fe74fdc6879fa043"
            },
            "data/member-00000-of-00001.parquet": {
                "bytes": 125021411,
                "sha256": "430b438ca2d03d2c14df6dff8db461d60eae53700bef707e9da611c6ec43a7f4"
            },
            "data/nonmember-00000-of-00001.parquet": {
                "bytes": 124995539,
                "sha256": "9972cd5c8e373af3d65476752d69b3b7e709ae7515ca4a7314a8f0c193bf3a01"
            },
            "member/metadata.json": {
                "bytes": 328290,
                "sha256": "6a3dbd708eb463550b66317fb9f26de1996f6e96f755f69fe1b0eb4493d42c7a"
            },
            "nonmember/metadata.json": {
                "bytes": 320536,
                "sha256": "29a256f18ea075cd7fea2841bae1c4c711d7712d864440eb677dd98945b6460b"
            }
        }
    },
    {
        "asset": "olmo-1b",
        "repo_id": "allenai/OLMo-2-0425-1B-Instruct",
        "repo_type": "model",
        "revision": "48d788eca847d4d7548f375ad03d3c9312f6139e",
        "files": {
            "README.md": {
                "bytes": 6161,
                "sha256": "5f5b891d66a46079920172f8d5957ec386825f76cfc14ae2e79c7208d4dca4d8"
            },
            "config.json": {
                "bytes": 625,
                "sha256": "0d15ebb6cb8d998513b46ef337214176a6fd59fe5f16b30387c70d5f87795a9c"
            },
            "generation_config.json": {
                "bytes": 121,
                "sha256": "437b97826bb4430c205083a0986fbe31f4959cffbe445e47fb7c5192fdcb8d58"
            },
            "merges.txt": {
                "bytes": 916646,
                "sha256": "b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502"
            },
            "model.safetensors": {
                "bytes": 2969854224,
                "sha256": "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
            },
            "special_tokens_map.json": {
                "bytes": 581,
                "sha256": "78afb564e81264029b25f9caf24bda2521d5bdaeff5cd3fdbc01d3da2e8ce2f2"
            },
            "tokenizer.json": {
                "bytes": 7137177,
                "sha256": "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca"
            },
            "tokenizer_config.json": {
                "bytes": 4884,
                "sha256": "50c412c57d832057a3d5db42064c741f751e570f7c8788f037bfb0d2dd6e5f49"
            },
            "vocab.json": {
                "bytes": 1611056,
                "sha256": "9e14712c91b37c7aab74b1306baa46ac342d620637a4b44523cdc3aec7d24195"
            }
        }
    },
    {
        "asset": "pythia-1.4b-step97000",
        "repo_id": "EleutherAI/pythia-1.4b-deduped",
        "repo_type": "model",
        "revision": "29376c0a610279f1d79507fc887c6170082f8839",
        "files": {
            "README.md": {
                "bytes": 13686,
                "sha256": "059fe2521508d76dbab6948a1b1a8b3d42a88735e07eda5b5a13f98460432c7e"
            },
            "config.json": {
                "bytes": 678,
                "sha256": "7da078ed62b3b68defcebede969b46a598136cba4fd33456f8c6051bedf47007"
            },
            "generation_config.json": {
                "bytes": 111,
                "sha256": "0c6190f4b464ae6d6062068bb719baf8fff2bedc261c09852b86a3c85630b8b3"
            },
            "model.safetensors": {
                "bytes": 2829329920,
                "sha256": "852d5bc740615a542c02c72a049d369a089e64c0709275498f9c3e0d6d29054a"
            },
            "special_tokens_map.json": {
                "bytes": 3,
                "sha256": "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
            },
            "tokenizer.json": {
                "bytes": 2113738,
                "sha256": "3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8"
            },
            "tokenizer_config.json": {
                "bytes": 4643,
                "sha256": "84f28e3ab5abb48724a3e678ad5de5f6f920b4948dc01814503bd34b00823e3c"
            }
        }
    }
]  # exact metadata from original downloaded receipts
OBSERVED_ENVIRONMENT = {
    "python": "3.12.14", "platform": "macOS-27.2-arm64-arm-64bit",
    "packages": {"torch": "2.12.0", "transformers": "5.10.2", "numpy": "2.4.6", "matplotlib": "3.11.2",
                 "scikit-learn": "1.9.1", "pyarrow": "24.0.0", "scipy": "1.18.1", "threadpoolctl": "3.8.0"},
    "provenance": ["experiments/results/checkmia-real-v1/scoring-amendment-1/execution.json",
                   "experiments/results/real-selectivity-v1/run-2026-11-07/sealed_manifest.json"],
    "limit": "Observed direct package versions, not a complete transitive lock or a numerical bitwise-identity guarantee",
}
HISTORICAL_AMENDMENT_SHA = "79e92d55b800401d90888250cd06461aaf8a47a56c41f98efff7747db5028c22"
HISTORICAL_FAILURE_SHA = "6f90df27b01349a095ef25084deea5b92f953f3eced31747adbe8b07fe160cbd"


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def hf_path(root, asset):
    kind = "datasets" if asset["repo_type"] == "dataset" else "models"
    return root / "hf_cache" / f"{kind}--{asset['repo_id'].replace('/', '--')}" / "snapshots" / asset["revision"]


def verify_file(path, expected):
    if not path.exists() and not path.is_symlink():
        return {"status": "missing", "path": str(path)}
    if not path.is_file():
        return {"status": "invalid_type", "path": str(path)}
    size = path.stat().st_size
    actual = file_hash(path)
    return {"status": "verified" if size == expected["bytes"] and actual == expected["sha256"] else "content_mismatch",
            "path": str(path), "bytes": size, "sha256": actual}


def download_missing(path, expected, url, *, root, timeout):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace {path}")
    if not path.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError("Download parent resolves outside the selected source root")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".research-download-", suffix=".part", dir=path.parent)
    partial = Path(temporary)
    try:
        h, size = hashlib.sha256(), 0
        with os.fdopen(descriptor, "wb") as target:
            with urlopen(Request(url, headers={"User-Agent": "ContamCheckr-ResearchSources/1"}), timeout=timeout) as response:
                if not response.geturl().startswith("https://"):
                    raise ValueError("Public download redirected outside HTTPS")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > expected["bytes"]:
                        raise ValueError("Download exceeds its pinned byte count")
                    h.update(chunk)
                    target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if size != expected["bytes"] or h.hexdigest() != expected["sha256"]:
            raise ValueError(f"Downloaded bytes do not match pinned source identity: {path}")
        # Atomic exclusive publication on the same filesystem; no replace().
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def validate_members(members, expected_root):
    if len(members) > 10000 or sum(m.size for m in members if m.isfile()) > 512 * 1024 * 1024:
        raise ValueError("Archive exceeds this source's declared extraction limits")
    paths, casefolded = {}, set()
    for member in members:
        name = member.name.rstrip("/")
        if (not name or "\\" in name or "\x00" in name or ":" in name or PurePosixPath(name).is_absolute()
                or PureWindowsPath(name).drive):
            raise ValueError(f"Unsafe archive path: {member.name}")
        parts = name.split("/")
        if parts[0] != expected_root or any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"Archive path escapes expected source: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Archive links/special files are not accepted: {member.name}")
        if name.casefold() in casefolded:
            raise ValueError(f"Repeated/case-colliding archive path: {member.name}")
        casefolded.add(name.casefold())
        paths[name] = member
    folded_paths = {name.casefold(): member for name, member in paths.items()}
    for name in paths:
        for parent in PurePosixPath(name).parents:
            if str(parent).casefold() in folded_paths and folded_paths[str(parent).casefold()].isfile():
                raise ValueError(f"Archive file is also a parent path: {parent}")
    return paths


def extract_new_archive(archive, destination, expected_root):
    """Validate every member first, then write only into a newly reserved dir."""
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        paths = validate_members(members, expected_root)
        # Exist_ok=False also rejects a concurrent destination, links or files.
        destination.mkdir(parents=False, exist_ok=False)
        for name, member in paths.items():
            relative = PurePosixPath(name).relative_to(expected_root)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            contents = source.extractfile(member)
            if contents is None:
                raise ValueError(f"Missing archive content: {name}")
            with target.open("xb") as output:
                for chunk in iter(lambda: contents.read(1024 * 1024), b""):
                    output.write(chunk)
    # Partial new directories survive failures, so they cannot be mistaken for
    # a verified tree or silently overwritten by a later invocation.


def verify_archive_tree(archive, destination, expected_root):
    with tarfile.open(archive, "r:gz") as source:
        paths = validate_members(source.getmembers(), expected_root)
        expected_files = {str(PurePosixPath(name).relative_to(expected_root)): member
                          for name, member in paths.items() if member.isfile()}
        if not destination.is_dir() or destination.is_symlink():
            return {"status": "missing" if not destination.exists() else "invalid_type", "path": str(destination)}
        observed_files = set()
        for path in destination.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink in extracted source tree: {path}")
            if path.is_file():
                observed_files.add(path.relative_to(destination).as_posix())
        if observed_files != set(expected_files):
            return {"status": "tree_file_set_mismatch", "missing": sorted(set(expected_files) - observed_files),
                    "unexpected": sorted(observed_files - set(expected_files)), "path": str(destination)}
        for relative, member in expected_files.items():
            path = destination / relative
            contents = source.extractfile(member)
            h = hashlib.sha256()
            for chunk in iter(lambda: contents.read(1024 * 1024), b""):
                h.update(chunk)
            if path.stat().st_size != member.size or file_hash(path) != h.hexdigest():
                return {"status": "tree_content_mismatch", "path": str(path)}
    return {"status": "verified", "path": str(destination), "files": len(expected_files)}
