#!/usr/bin/env python3
"""Score a frozen public meeting input without reading its gold labels."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from prompts import PROFILES


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--context-turns", type=int, default=3)
    parser.add_argument("--prompt-profile", choices=PROFILES, default="baseline")
    args = parser.parse_args()
    if not 0 <= args.context_turns <= 8:
        parser.error("--context-turns must be 0..8")
    raw = args.input.read_bytes()
    suite = json.loads(raw)
    segments = suite["segments"]
    states = []
    for index, segment in enumerate(segments):
        if not segment["score"]:
            continue
        context = [{"text": older["text"], "speakerId": older["speaker"]}
                   for older in segments[max(0, index - args.context_turns):index]]
        states.append({"id": segment["id"], "state": json.dumps({
            "transcript": segment["text"], "speakerId": segment["speaker"],
            "context": context, "timeZone": "Asia/Shanghai",
            "recordedAt": "2026-09-23T10:00:00+08:00"}, ensure_ascii=False),
            "questions": PROFILES[args.prompt_profile]})
    if not states:
        raise RuntimeError("Frozen input has no scored segments")
    opener = build_opener(ProxyHandler({}))
    endpoint = f"http://127.0.0.1:{args.port}/api/evaluate"
    checkpoint = None
    answers = {}
    for first in range(0, len(states), 8):
        batch = states[first:first + 8]
        request = Request(endpoint, method="POST", headers={"Content-Type": "application/json"},
                          data=json.dumps({"states": batch}, ensure_ascii=False).encode())
        with opener.open(request, timeout=120) as response:
            result = json.loads(response.read(2_000_001))
        execution = result.get("execution", {})
        if execution.get("backend") != "mlx" or execution.get("network_model_calls") != 0:
            raise RuntimeError("Expected native MLX inference with no remote model calls")
        if checkpoint is not None and checkpoint != result.get("checkpoint"):
            raise RuntimeError("Checkpoint changed during inference")
        checkpoint = result.get("checkpoint")
        for row in result["states"]:
            answers[row["id"]] = {key: row["answers"][key]["p_true"]
                                  for key in ("assist", "schedule")}
    if set(answers) != {row["id"] for row in states}:
        raise RuntimeError("Response did not cover all scored segments")
    directory = Path(checkpoint["directory"])
    output = {"createdAt": datetime.now(timezone.utc).isoformat(),
              "frozenInputSha256": digest(raw), "source": suite["source"],
              "promptProfile": args.prompt_profile,
              "contextTurns": args.context_turns,
              "checkpoint": {"name": directory.name,
                             "weightsSha256": digest((directory / "best.safetensors").read_bytes())},
              "threshold": 0.85, "scoredCount": len(states),
              "triggerCount": {key: sum(value[key] >= 0.85 for value in answers.values())
                               for key in ("assist", "schedule")},
              "scores": answers,
              "note": "No gold labels were read; context uses the requested number of prior annotated turns."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("checkpoint", "scoredCount", "triggerCount")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
