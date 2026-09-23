#!/usr/bin/env python3
"""Send the fixed synthetic Chinese intent suite to NanoJev's real local API."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from urllib.request import ProxyHandler, Request, build_opener

from prompts import PROFILES
from common import HERE, URL, health, settings

# This is the backend's initial prompt profile, fixed before seeing results.
# Gold labels and group names are deliberately excluded from model inputs.
QUESTIONS = PROFILES["baseline"]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def payload_for(suite, cases, questions=QUESTIONS):
    def context_turn(turn, fallback_speaker):
        if isinstance(turn, dict):
            return {"text": turn["text"], "speakerId": turn["speakerId"]}
        return {"text": turn, "speakerId": fallback_speaker}

    return {"states": [{"id": row["id"], "state": json.dumps({"transcript": row["text"], "speakerId": row.get("speakerId", "synthetic_speaker"),
        "context": [context_turn(turn, row.get("speakerId", "synthetic_speaker")) for turn in row.get("context", [])],
        "timeZone": row.get("timeZone", suite["timeZone"]), "recordedAt": row.get("recordedAt", suite["recordedAt"])}, ensure_ascii=False, separators=(",", ":")),
        "questions": questions} for row in cases]}


def metrics(rows, threshold):
    output = {}
    for name in QUESTIONS:
        counts = Counter()
        for row in rows:
            expected = row["expected"][name]
            predicted = row["pTrue"][name] >= threshold
            counts["tp" if predicted and expected else "fp" if predicted else "fn" if expected else "tn"] += 1
        tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
        output[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / len(rows),
                        "precision": tp / (tp + fp) if tp + fp else None,
                        "recall": tp / (tp + fn) if tp + fn else None,
                        "falsePositiveRate": fp / (fp + tn) if fp + tn else None}
    return output


def evaluate(suite, repeats, batch_size, port=8765, prompt_profile="baseline"):
    questions = PROFILES[prompt_profile]
    endpoint = URL if port == 8765 else f"http://127.0.0.1:{port}"
    if port == 8765:
        ready = health()
    else:
        try:
            with build_opener(ProxyHandler({})).open(endpoint + "/api/health", timeout=2) as response:
                body = response.read(8193)
            value = json.loads(body) if len(body) <= 8192 else None
            ready = value if isinstance(value, dict) and value.get("ready") is True and value.get("model_loaded_once") is True and value.get("provider_calls") == 0 else None
        except (OSError, ValueError):
            ready = None
    if not ready:
        raise RuntimeError(f"NanoJev is not ready at 127.0.0.1:{port}.")
    cases = suite["cases"]
    if not cases or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Case IDs must be unique.")
    for row in cases:
        if any(type(row[name]) is not bool for name in QUESTIONS):
            raise ValueError("Gold labels must be booleans.")
    runs, raw_responses, latencies, executions, reported_checkpoint = [], [], [], [], None
    for iteration in range(repeats):
        rows = []
        for first in range(0, len(cases), batch_size):
            batch = cases[first:first + batch_size]
            request = Request(endpoint + "/api/evaluate", method="POST", headers={"Content-Type": "application/json"},
                              data=json.dumps(payload_for(suite, batch, questions), ensure_ascii=False).encode())
            started = time.perf_counter()
            with build_opener(ProxyHandler({})).open(request, timeout=120) as response:
                data = response.read(2_000_001)
                if len(data) > 2_000_000:
                    raise ValueError("API response exceeds limit.")
                result = json.loads(data)
            latencies.append((time.perf_counter() - started) * 1000)
            execution = result.get("execution", {})
            if execution.get("backend") != "mlx" or execution.get("network_model_calls") != 0:
                raise ValueError("Expected native MLX with zero network model calls.")
            checkpoint = result.get("checkpoint", {})
            if reported_checkpoint is not None and checkpoint != reported_checkpoint:
                raise ValueError("Checkpoint metadata changed during the evaluation.")
            reported_checkpoint = checkpoint
            executions.append(execution)
            answers = {row["id"]: row.get("answers", {}) for row in result.get("states", [])}
            if set(answers) != {row["id"] for row in batch}:
                raise ValueError("API states do not match the requested cases.")
            for case in batch:
                scores = {name: answers[case["id"]].get(name, {}).get("p_true") for name in QUESTIONS}
                if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in scores.values()):
                    raise ValueError("Invalid p_true returned by the model.")
                rows.append({"id": case["id"], "group": case["group"], "text": case["text"],
                             "context": case.get("context", []), "expected": {name: case[name] for name in QUESTIONS}, "pTrue": scores})
            # Local model path is represented separately relative to the configured root.
            result["checkpoint"] = {key: val for key, val in checkpoint.items() if key != "directory"}
            raw_responses.append({"repeat": iteration + 1, "firstCase": first, "response": result})
        runs.append(rows)
    root, checkpoint, _ = settings()
    actual = Path(reported_checkpoint.get("directory", "")).resolve()
    if actual != checkpoint:
        raise ValueError("The running server reports a different checkpoint. Set NANOJEV_CHECKPOINT explicitly and rerun.")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    source_hashes = {name: digest(root / "scripts" / name) for name in ("serve_decisions.py", "predict_toy_decisions.py", "mlx_decisions.py")}
    consistency = all(rows == runs[0] for rows in runs[1:]) if len(runs) > 1 else None
    checkpoint_config = json.loads((checkpoint / "config.json").read_text())
    synthetic = suite.get("synthetic", True)
    return {"createdAt": datetime.now(timezone.utc).isoformat(), "suiteVersion": suite["version"], "synthetic": synthetic,
            "caseCount": len(cases), "repeats": repeats, "batchSize": batch_size, "endpoint": endpoint + "/api/evaluate",
            "thresholdPolicy": "0.85 is the initial backend threshold; 0.5 is diagnostic only. No threshold is fitted on this suite.",
            "promptProfile": prompt_profile, "promptTemplatesSha256": digest(HERE / "prompts.py"),
            "questions": questions, "backend": "mlx", "checkpoint": {**{k: v for k, v in reported_checkpoint.items() if k != "directory"},
                "name": checkpoint.name, "weightsSha256": digest(checkpoint / "best.safetensors"),
                "configSha256": digest(checkpoint / "config.json"), "trainingSchema": checkpoint_config.get("schema_version"),
                "adaptation": checkpoint_config.get("adaptation")},
            "source": {"commit": commit.stdout.strip() if commit.returncode == 0 else None, "scriptSha256": source_hashes},
            "repeatScoresIdentical": consistency,
            "requestLatencyMs": {"samples": latencies, "median": statistics.median(latencies), "max": max(latencies),
                                 "note": "Batch HTTP latency; not per-case or end-to-end glasses latency. First request may be cold."},
            "metrics": {str(threshold): metrics(runs[0], threshold) for threshold in (0.85, 0.5)},
            "groups": {group: metrics([r for r in runs[0] if r["group"] == group], 0.85) for group in sorted({r["group"] for r in runs[0]})},
            "cases": runs[0], "executions": executions, "rawResponses": raw_responses,
            "limitations": ["Gate-conditioned public meeting negatives; this measures only conditional false triggers, not production prevalence or recall." if not synthetic else
                            "Hand-authored small synthetic suite; no held-out production distribution or calibration.",
                            "Adaptation metadata is recorded; synthetic held-out performance does not establish production calibration." if checkpoint_config.get("adaptation") else "Published unified-games-v1 weights were not fine-tuned for these intents.",
                            "This evaluates classification only, not LLM extraction, calendar writes or glasses operations."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=HERE / "cases.zh.json")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "intent-zh-smoke.json")
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=2)
    parser.add_argument("--batch-size", type=int, choices=range(1, 33), default=1,
                        help="Submit one utterance per request by default, matching the backend; larger batches are throughput experiments only")
    parser.add_argument("--port", type=int, choices=range(1024, 65536), default=8765,
                        help="Loopback NanoJev API port; allows isolated candidate evaluation without replacing the managed service")
    parser.add_argument("--prompt-profile", choices=PROFILES, default="baseline",
                        help="Versioned local question template; candidate profiles are experimental until separately wired into the backend")
    args = parser.parse_args()
    suite = json.loads(args.cases.read_text())
    result = evaluate(suite, args.repeats, args.batch_size, args.port, args.prompt_profile)
    result["suiteSha256"] = digest(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"], "repeatScoresIdentical": result["repeatScoresIdentical"],
                      "requestLatencyMs": result["requestLatencyMs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
