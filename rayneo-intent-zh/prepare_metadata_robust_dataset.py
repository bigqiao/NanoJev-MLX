#!/usr/bin/env python3
"""Augment a recording-split intent set with production-shaped metadata states.

The label and transcript stay fixed. The four state variants cover the two
speaker-ID and timestamp forms sent by the Android/backend pipeline; natural
previous turns retain their TextGrid speaker IDs only in stable-ID variants.
"""

import argparse
import hashlib
import json
from pathlib import Path


AUGMENTATION = "full-factorial-null-or-stable-speaker-null-or-ms-time-v1"
VARIANTS = (
    ("null_null", False, False),
    ("stable_null", True, False),
    ("null_ms", False, True),
    ("stable_ms", True, True),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(case, milliseconds):
    original_speaker = case.get("speakerId", "person_001")
    for name, use_speaker, use_time in VARIANTS:
        speaker = original_speaker if use_speaker else None
        context = []
        for turn in case.get("context", []):
            if isinstance(turn, dict):
                context.append({"text": turn["text"],
                                "speakerId": turn["speakerId"] if use_speaker else None})
            else:
                context.append({"text": turn, "speakerId": speaker})
        yield {**case, "id": f"{case['id']}__{name}", "canonicalId": case["id"],
               "metadataVariant": name, "speakerId": speaker,
               "recordedAt": milliseconds if use_time else None,
               "context": context}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Recording-level split dataset from prepare_public_context_dataset.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    source = {split: json.loads((args.source / f"{split}.json").read_text())
              for split in ("train", "dev")}
    if any(suite.get("version") != "zh-intent-v4-natural-context-recording-split"
           for suite in source.values()):
        raise ValueError("Expected v4 recording-level train/dev partitions")
    output = {}
    for split, suite in source.items():
        milliseconds = suite["recordedAt"]
        if type(milliseconds) is not int or milliseconds < 1_000_000_000_000:
            raise ValueError("Recorded time must be a Unix millisecond timestamp")
        cases = [variant for case in suite["cases"] for variant in expand(case, milliseconds)]
        output[split] = {**suite, "version": "zh-intent-v5-metadata-robust-recording-split",
                         "metadataAugmentation": AUGMENTATION, "cases": cases}
    args.output.mkdir(parents=True)
    for split, suite in output.items():
        (args.output / f"{split}.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n")
    provenance = {
        "sourceSha256": {split: digest(args.source / f"{split}.json") for split in ("train", "dev")},
        "sourceCounts": {split: len(suite["cases"]) for split, suite in source.items()},
        "outputCounts": {split: len(suite["cases"]) for split, suite in output.items()},
        "augmentation": AUGMENTATION,
        "timestampMs": source["train"]["recordedAt"],
        "speakerId": "null or stable original TextGrid tier / person_001 for synthetic cases",
        "context": "Null speaker IDs in null variants; original stable tier IDs in stable variants",
        "selection": "No held-out, fresh meeting or contrast example was included or used to select variants",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == "__main__":
    main()
