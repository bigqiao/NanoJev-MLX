#!/usr/bin/env python3
"""Append public meeting negative controls to the existing synthetic train/dev.

TextGrid inputs and the generated cases stay under ignored .runtime; this
script does not distribute the source transcripts. Human review is required
before treating any selected utterance as a negative intent label.
"""

import argparse
import json
from pathlib import Path
import re

from evaluate_public_meeting import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textgrid", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-file", type=int, default=60)
    parser.add_argument("--dev-per-file", type=int, default=40)
    args = parser.parse_args()
    if not 1 <= args.train_per_file <= 1000 or not 1 <= args.dev_per_file <= 1000:
        parser.error("Per-file sample counts must be 1..1000")
    source = Path(__file__).resolve().parent / "datasets/zh-intent-v1"
    train = json.loads((source / "train.json").read_text())
    dev = json.loads((source / "dev.json").read_text())
    sources = []
    for path in args.textgrid:
        metadata, rows = select(path, args.train_per_file + args.dev_per_file)
        if len(rows) != args.train_per_file + args.dev_per_file:
            raise RuntimeError(f"Too few eligible utterances in {path}")
        sources.append(metadata)
        for split, chosen in ((train, rows[:args.train_per_file]),
                              (dev, rows[args.train_per_file:])):
            split["cases"].extend({"id": "public-" + row["id"], "group": "public_meeting_negative",
                                   "text": row["text"], "context": [],
                                   "assist": False, "schedule": False} for row in chosen)
    normalize = lambda value: re.sub(r"[\W_]", "", value)
    train_text = [normalize(row["text"]) for row in train["cases"]]
    dev_text = [normalize(row["text"]) for row in dev["cases"]]
    if (len(set(train_text)) != len(train_text) or len(set(dev_text)) != len(dev_text)
            or set(train_text) & set(dev_text)):
        raise RuntimeError("Duplicate text or cross-split leakage")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, suite in (("train", train), ("dev", dev)):
        suite["version"] = "zh-intent-v3-public-meeting-negative"
        (args.output / f"{name}.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n")
    (args.output / "provenance.json").write_text(json.dumps({
        "source": sources,
        "selection": "SHA256(file name, TextGrid text index, cleaned text); 8..160 characters; first 60 train, next 40 dev per file",
        "label": "Candidate negatives from public human meetings, not assistant-directed speech or personally committed calendar events",
        "review": "The selected 200 utterances require human review; the two calendar-like training examples discuss schoolwork/work plans, and three dev examples discuss school pickup/food/traffic, not future personal appointments.",
        "limitations": "Two meetings, one source domain; this is not an independent blind test or a guarantee that every negative label is correct."
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train": len(train["cases"]), "dev": len(dev["cases"]),
                      "sources": sources}, ensure_ascii=False))


if __name__ == "__main__":
    main()
