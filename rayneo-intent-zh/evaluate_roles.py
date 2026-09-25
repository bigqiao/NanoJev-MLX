#!/usr/bin/env python3
"""Score speaker-role suites against a running NanoJev server, with invariance checks.

--shape roles   sends the speaker-role-v1 state (role_state.py), for v6 candidates.
--shape legacy  sends what the RayNeoRemaster backend sent to v5: an opaque id per speaker
                (fresh UUIDs), null for an unrecognised voice, a real millisecond timestamp.

Invariance: the same case under changes that must not alter intent (others relettered, a
fresh set of opaque ids, an older filler turn, ASCII question marks). A robust model keeps
every score within --tolerance of the original.
"""
import argparse
import json
from pathlib import Path
import uuid
from urllib.request import ProxyHandler, Request, build_opener

from prompts import PROFILES
from role_state import OTHERS, SELF, UNKNOWN, relabel_others, role_state

QUESTIONS = PROFILES["baseline"]
RECORDED_AT = 1790263351709


def legacy_state(case, ids):
    speaker_id = lambda role: None if role == UNKNOWN else ids.setdefault(role, str(uuid.uuid4()))
    return json.dumps({"transcript": case["text"], "speakerId": speaker_id(case["speaker"]),
                       "context": [{"text": t["text"], "speakerId": speaker_id(t["speaker"])} for t in case["context"]],
                       "timeZone": "Asia/Singapore", "recordedAt": RECORDED_AT}, ensure_ascii=False, separators=(",", ":"))


def render(case, shape, ids):
    return role_state(case["text"], case["speaker"], case["context"]) if shape == "roles" else legacy_state(case, ids)


def variants(case):
    """Intent-preserving rewrites of [case], each (name, case)."""
    out = []
    others = sorted({r for r in [case["speaker"], *(t["speaker"] for t in case["context"])] if r in OTHERS})
    if len(others) >= 2:
        swap = {others[0]: others[1], others[1]: others[0]}
        speaker, context = relabel_others(swap.get(case["speaker"], case["speaker"]),
                                          [{**t, "speaker": swap.get(t["speaker"], t["speaker"])} for t in case["context"]])
        out.append(("swap_others", {**case, "speaker": speaker, "context": context}))
    out.append(("fresh_ids", case))
    out.append(("older_filler", {**case, "context": [{"speaker": UNKNOWN, "text": "嗯。"}, *case["context"]][-3:]
                                  if len(case["context"]) < 3 else case["context"]}))
    if "？" in case["text"]:
        out.append(("ascii_question", {**case, "text": case["text"].replace("？", "?")}))
    return out


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="+", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--shape", choices=("roles", "legacy"), required=True)
    parser.add_argument("--thresholds", default="0.3,0.5,0.85")
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    opener = build_opener(ProxyHandler({}))

    def score(states):
        body = json.dumps({"states": [{"id": str(i), "state": s, "questions": QUESTIONS} for i, s in enumerate(states)]},
                          ensure_ascii=False).encode()
        with opener.open(Request(args.url + "/api/evaluate", body, {"Content-Type": "application/json"}), timeout=300) as r:
            answers = {row["id"]: row["answers"] for row in json.load(r)["states"]}
        return [{q: answers[str(i)][q]["p_true"] for q in QUESTIONS} for i in range(len(states))]

    report = {"shape": args.shape, "url": args.url, "suites": {}}
    for path in args.suites:
        suite = json.loads(path.read_text())
        cases = [c for c in suite["cases"] if not c.get("exclude")]
        ids = {}
        base = []
        for first in range(0, len(cases), 8):
            base += score([render(c, args.shape, ids) for c in cases[first:first + 8]])
        drift = {}
        for name in ("swap_others", "fresh_ids", "older_filler", "ascii_question"):
            pairs = [(i, v) for i, c in enumerate(cases) for n, v in variants(c) if n == name]
            if not pairs:
                continue
            fresh = {} if name == "fresh_ids" else ids
            got = []
            for first in range(0, len(pairs), 8):
                got += score([render(v, args.shape, fresh) for _, v in pairs[first:first + 8]])
            deltas = [abs(g["assist"] - base[i]["assist"]) for (i, _), g in zip(pairs, got)]
            drift[name] = {"cases": len(deltas), "meanDelta": sum(deltas) / len(deltas), "maxDelta": max(deltas),
                           "overTolerance": sum(d > args.tolerance for d in deltas)}
        metrics = {}
        for q in QUESTIONS:
            gold = [c[q] for c in cases]
            s = [b[q] for b in base]
            if not any(gold):
                metrics[q] = {"positives": 0, "negatives": len(gold),
                              "falsePositives": {t: sum(x >= float(t) for x in s) for t in args.thresholds.split(",")}}
                continue
            metrics[q] = {"positives": sum(gold), "negatives": len(gold) - sum(gold),
                          "auc": auc([x for g, x in zip(gold, s) if g], [x for g, x in zip(gold, s) if not g]),
                          "at": {t: {"recall": sum(g and x >= float(t) for g, x in zip(gold, s)),
                                     "falsePositives": sum((not g) and x >= float(t) for g, x in zip(gold, s))}
                                 for t in args.thresholds.split(",")}}
        report["suites"][path.name] = {"cases": len(cases), "metrics": metrics, "invariance": drift,
                                       "scores": [{"id": c["id"], **b} for c, b in zip(cases, base)]}
        print(json.dumps({"suite": path.name, "cases": len(cases), "metrics": metrics, "invariance": drift},
                         ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
