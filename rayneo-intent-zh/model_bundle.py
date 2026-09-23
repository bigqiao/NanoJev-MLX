#!/usr/bin/env python3
"""Package/restore the accepted NanoJev checkpoint as verified Git LFS parts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
ACCEPTED = ROOT / "accepted-model.json"
BUNDLE = ROOT / "bundle"
PART_BYTES = 1024 * 1024 * 1024  # Below GitHub Free/Pro's 2 GB LFS file limit.
RUNTIME_FILES = (
    "config.json",
    "backbone_config/config.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accepted() -> tuple[Path, str]:
    info = json.loads(ACCEPTED.read_text(encoding="utf-8"))
    target = (ROOT / info["checkpoint"]).resolve()
    if not target.is_relative_to((ROOT / "models").resolve()):
        raise ValueError("Accepted checkpoint must be under this project's models directory")
    return target, info["weightsSha256"]


def package_model(source: Path, bundle: Path, expected_hash: str, part_bytes: int = PART_BYTES) -> dict:
    weights = source / "best.safetensors"
    if not weights.is_file() or digest_file(weights) != expected_hash:
        raise ValueError("Accepted best.safetensors is absent or has the wrong SHA-256")
    if part_bytes < 1 or part_bytes >= 2 * 1024 * 1024 * 1024:
        raise ValueError("Each LFS part must be smaller than 2 GiB")
    bundle.mkdir(parents=True, exist_ok=True)
    files = []
    for relative in RUNTIME_FILES:
        item = source / relative
        if not item.is_file():
            raise ValueError(f"Required checkpoint file missing: {relative}")
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, destination)
        files.append({"name": relative, "size": item.stat().st_size, "sha256": digest_file(item)})
    parts = []
    with weights.open("rb") as original:
        index = 0
        while True:
            block = original.read(part_bytes)
            if not block:
                break
            name = f"weights.part-{index:03d}"
            destination = bundle / name
            with tempfile.NamedTemporaryFile(dir=bundle, prefix=".weights-", delete=False) as temporary:
                temporary.write(block)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            temporary_path.replace(destination)
            parts.append({"name": name, "size": len(block), "sha256": hashlib.sha256(block).hexdigest()})
            index += 1
    manifest = {"schema": 1, "weights": {"size": weights.stat().st_size, "sha256": expected_hash},
                "parts": parts, "files": files}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_entry(bundle: Path, item: dict) -> Path:
    name = item["name"]
    if Path(name).is_absolute() or ".." in Path(name).parts or not isinstance(item["size"], int):
        raise ValueError("Unsafe or invalid checkpoint bundle entry")
    path = bundle / name
    if not path.is_file() or path.is_symlink() or path.stat().st_size != item["size"] or digest_file(path) != item["sha256"]:
        raise ValueError(f"Checkpoint bundle entry absent or corrupt: {name}; run git lfs pull")
    return path


def verify_bundle(bundle: Path, expected_hash: str) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("weights", {}).get("sha256") != expected_hash:
        raise ValueError("Bundle does not match accepted checkpoint")
    for item in manifest["parts"] + manifest["files"]:
        verify_entry(bundle, item)
    return manifest


def restore_model(bundle: Path, target: Path, expected_hash: str) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("weights", {}).get("sha256") != expected_hash:
        raise ValueError("Bundle does not match accepted-model.json")
    if not manifest.get("parts") or {part["name"] for part in manifest["parts"]} != {
            f"weights.part-{i:03d}" for i in range(len(manifest["parts"]))}:
        raise ValueError("Invalid checkpoint part sequence")
    if sum(part["size"] for part in manifest["parts"]) != manifest["weights"]["size"]:
        raise ValueError("Checkpoint part sizes do not match the expected weight size")
    if {item["name"] for item in manifest.get("files", [])} != set(RUNTIME_FILES):
        raise ValueError("Checkpoint runtime file set is incomplete")
    parts = [verify_entry(bundle, item) for item in manifest["parts"]]
    files = [(item, verify_entry(bundle, item)) for item in manifest["files"]]
    target.mkdir(parents=True, exist_ok=True)
    destination = target / "best.safetensors"
    if not destination.is_file() or destination.stat().st_size != manifest["weights"]["size"] or digest_file(destination) != expected_hash:
        with tempfile.NamedTemporaryFile(dir=target, prefix=".best-", delete=False) as output:
            temporary_path = Path(output.name)
            combined = hashlib.sha256()
            try:
                for part in parts:
                    with part.open("rb") as stream:
                        while block := stream.read(4 * 1024 * 1024):
                            output.write(block)
                            combined.update(block)
                output.flush()
                os.fsync(output.fileno())
                if combined.hexdigest() != expected_hash:
                    raise ValueError("Reassembled checkpoint SHA-256 mismatch")
                temporary_path.replace(destination)
            finally:
                temporary_path.unlink(missing_ok=True)
    for item, source in files:
        dest = target / item["name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file() or dest.stat().st_size != item["size"] or digest_file(dest) != item["sha256"]:
            shutil.copyfile(source, dest)
    return {"checkpoint": str(target), "weightsSha256": expected_hash, "bytes": manifest["weights"]["size"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "restore", "verify"))
    args = parser.parse_args()
    target, expected_hash = accepted()
    if args.command == "pack":
        result = package_model(target, BUNDLE, expected_hash)
        print(json.dumps({"bundle": str(BUNDLE), "parts": result["parts"], "weights": result["weights"]}, indent=2))
    elif args.command == "restore":
        print(json.dumps(restore_model(BUNDLE, target, expected_hash), indent=2))
    else:
        manifest = verify_bundle(BUNDLE, expected_hash)
        print(json.dumps({"bundle": str(BUNDLE), "parts": len(manifest["parts"]), "status": "verified"}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"model_bundle: {error}", file=sys.stderr)
        raise SystemExit(1)
