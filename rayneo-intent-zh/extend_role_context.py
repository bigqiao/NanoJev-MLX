#!/usr/bin/env python3
"""Deterministically lengthen v6 speaker-role contexts and drop the writer's 甲/乙 names.

The generator's LLM wrote about three context turns per scene, while the backend sends
the last two minutes of a recording (median nine turns in real speech). This prepends an
earlier stretch borrowed from another scene of the same split so each case carries 2-10
turns, oldest first. Only older turns are added, so who the current sentence is said to
and its label are unchanged. Every variant of one scene borrows the same stretch, so a
scene's variants still differ only in speaker roles. Borrowed turns take roles already in
the scene (all 未知 in unrecognised variants). The letters 甲/乙 that the writer used as
speaker names in the text are replaced by ordinary names so they cannot become a cue.
"""
import argparse
import json
from pathlib import Path
import random

from role_state import ROLE_STATE, UNKNOWN, relabel_others

NAMES = ["小王", "老张", "小李", "阿明", "小陈", "老刘", "小赵", "阿杰", "小林", "老周", "小何", "阿芳"]


def rename(text, names):
    return text.replace("甲", names[0]).replace("乙", names[1])


def extend(suite, rng):
    by_scene = {}
    for case in suite["cases"]:
        by_scene.setdefault(case["canonicalId"], []).append(case)
    scenes = sorted(by_scene)
    # Donor stretches: each scene's original-variant context, text only.
    donors = {sid: [t["text"] for t in next(c for c in rows if c["variant"] == "original")["context"]]
              for sid, rows in by_scene.items()}
    out = []
    for sid in scenes:
        rows = by_scene[sid]
        names = rng.sample(NAMES, 2)
        target = rng.randint(2, 10)
        own = len(rows[0]["context"])
        borrowed = []
        while own + len(borrowed) < target:
            donor = rng.choice(scenes)
            if donor != sid:
                borrowed = donors[donor] + borrowed
        borrowed = borrowed[len(borrowed) - max(0, target - own):] if target > own else []
        pattern = [rng.random() for _ in borrowed]
        for case in rows:
            present = sorted({t["speaker"] for t in case["context"]} | {case["speaker"]})
            roles = [UNKNOWN if case["variant"] == "unrecognised" else present[int(p * len(present))] for p in pattern]
            context = [{"speaker": r, "text": rename(t, names)} for r, t in zip(roles, borrowed)] + \
                      [{**t, "text": rename(t["text"], names)} for t in case["context"]]
            speaker, context = relabel_others(case["speaker"], context)
            out.append({**case, "speaker": speaker, "text": rename(case["text"], names), "context": context,
                        "borrowedTurns": len(borrowed)})
    rng.shuffle(out)
    return {**suite, "version": "zh-intent-v6-roles-ctx", "contextExtension": "borrowed-older-turns-2-10-v1", "cases": out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=here / "datasets/zh-intent-v6-roles")
    parser.add_argument("--output", type=Path, default=here / "datasets/zh-intent-v6-roles-ctx")
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output must be a new directory.")
    args.output.mkdir(parents=True)
    rng = random.Random(args.seed)
    for split in ("train", "dev"):
        suite = json.loads((args.source / f"{split}.json").read_text())
        if suite.get("stateShape") != ROLE_STATE:
            raise ValueError("Expected a speaker-role suite")
        result = extend(suite, rng)
        (args.output / f"{split}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
        lengths = [len(c["context"]) for c in result["cases"]]
        print(json.dumps({"split": split, "cases": len(lengths), "meanContext": sum(lengths) / len(lengths),
                          "minContext": min(lengths), "maxContext": max(lengths)}))
    (args.output / "provenance.json").write_text(json.dumps({"source": str(args.source.name), "seed": args.seed,
        "method": __doc__.strip()}, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
