#!/usr/bin/env python3
"""Build an evaluation-only intent set with preceding natural meeting turns.

The two previously reviewed public negative pools are split by *recording*.
The first recording supplies training negatives; the second supplies dev
negatives. This keeps an utterance from one split out of the other split's
context. Synthetic examples stay in their original train/dev partitions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

from evaluate_public_meeting import TAG


ITEM = re.compile(r'^    item \[\d+\]:\s*$', re.MULTILINE)
NAME = re.compile(r'^\s*name = "(.*)"\s*$', re.MULTILINE)
INTERVAL = re.compile(
    r'        intervals \[\d+\]:\s+'
    r'            xmin = (\S+)\s+'
    r'            xmax = (\S+)\s+'
    r'            text = "(.*)"', re.MULTILINE)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intervals(path):
    content = path.read_text()
    headings = list(ITEM.finditer(content))
    if not headings:
        raise ValueError(f"Missing TextGrid tiers in {path}")
    rows = []
    for i, heading in enumerate(headings):
        block = content[heading.end():headings[i + 1].start() if i + 1 < len(headings) else len(content)]
        name = NAME.search(block)
        if name is None:
            raise ValueError(f"Missing speaker tier in {path}")
        for interval in INTERVAL.finditer(block):
            text = TAG.sub("", interval.group(3)).strip()
            rows.append({"id": f"public-{path.stem}-{len(rows)}", "start": float(interval.group(1)),
                         "end": float(interval.group(2)), "text": text, "speakerId": name.group(1)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Previously reviewed v3 natural-negative dataset directory")
    parser.add_argument("--textgrid", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-turns", type=int, default=3)
    args = parser.parse_args()
    if args.context_turns != 3:
        parser.error("This v4 experiment fixes the context window at three preceding turns")
    if len(args.textgrid) != 2:
        parser.error("Provide exactly two meeting TextGrids: train, then dev")
    if args.output.exists():
        parser.error("Output must be a new directory")
    original = {split: json.loads((args.source / f"{split}.json").read_text())
                for split in ("train", "dev")}
    provenance = json.loads((args.source / "provenance.json").read_text())
    known_files = {item["file"]: item["sha256"] for item in provenance["source"]}
    selected = {case["id"]: case for split in ("train", "dev")
                for case in original[split]["cases"] if case["group"] == "public_meeting_negative"}
    if len(selected) != 200:
        raise ValueError("Expected exactly 200 previously reviewed natural negatives")
    source_metadata, by_file = [], {}
    for path in args.textgrid:
        if path.name not in known_files or sha256(path) != known_files[path.name]:
            raise ValueError(f"Unreviewed or modified source: {path}")
        rows = intervals(path)
        ordered = sorted((r for r in rows if r["text"]),
                         key=lambda row: (row["start"], row["end"], row["id"]))
        for index, row in enumerate(ordered):
            if row["id"] not in selected:
                continue
            if selected[row["id"]]["text"] != row["text"]:
                raise ValueError(f"Selected text changed: {row['id']}")
            row["context"] = [{"text": older["text"], "speakerId": older["speakerId"]}
                              for older in ordered[max(0, index - args.context_turns):index]]
            by_file[row["id"]] = row
        source_metadata.append({"file": path.name, "sha256": known_files[path.name],
                                "selectedCount": sum(key.startswith(f"public-{path.stem}-") for key in by_file)})
    if set(by_file) != set(selected):
        raise ValueError("Selected rows were not fully recovered from source TextGrids")
    # File-level split prevents any train utterance from appearing in the
    # dev recording or its previous-turn context, and vice versa.
    output = {}
    for split, path in zip(("train", "dev"), args.textgrid):
        suite = {k: v for k, v in original[split].items() if k != "cases"}
        synthetic = [case for case in original[split]["cases"]
                     if case["group"] != "public_meeting_negative"]
        natural = []
        for case_id, case in selected.items():
            if not case_id.startswith(f"public-{path.stem}-"):
                continue
            row = by_file[case_id]
            natural.append({**case, "speakerId": row["speakerId"], "context": row["context"]})
        natural.sort(key=lambda case: case["id"])
        if len(natural) != 100:
            raise ValueError(f"Expected 100 natural rows in {path}")
        suite["version"] = "zh-intent-v4-natural-context-recording-split"
        suite["cases"] = synthetic + natural
        output[split] = suite
    train_ids = {c["id"] for c in output["train"]["cases"]}
    dev_ids = {c["id"] for c in output["dev"]["cases"]}
    if train_ids & dev_ids:
        raise ValueError("Duplicate case IDs across train/dev")
    args.output.mkdir(parents=True)
    for split, suite in output.items():
        (args.output / f"{split}.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n")
    metadata = {"sourceDatasetSha256": {split: sha256(args.source / f"{split}.json")
                                       for split in ("train", "dev")},
                "sourceTextGrids": source_metadata,
                "partition": "First recording train, second recording dev; synthetic partitions preserved",
                "context": "Chronological three preceding nonempty TextGrid intervals from all speaker tiers, with source speaker IDs",
                "naturalLabels": "200 previously reviewed negatives; no holdout recording used",
                "limitations": "Only two meeting recordings and synthetic positives; dev is one natural meeting domain."}
    (args.output / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train": len(output["train"]["cases"]),
                      "dev": len(output["dev"]["cases"]),
                      "source": source_metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
