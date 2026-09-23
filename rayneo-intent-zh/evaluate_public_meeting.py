#!/usr/bin/env python3
"""Probe NanoJev on fixed samples from local public meeting TextGrid files.

This is a negative-control diagnostic, not a labeled intent benchmark. The
source transcripts and per-utterance output stay in the ignored .runtime tree.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.request import ProxyHandler, Request, build_opener

from evaluate import QUESTIONS, payload_for
from prompts import PROFILES
from common import URL


TEXT = re.compile(r'^\s*text = "(.*)"\s*$', re.MULTILINE)
TAG = re.compile(r'<[^>]*>')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def select(path, count):
    content = path.read_bytes()
    candidates = []
    for index, match in enumerate(TEXT.finditer(content.decode("utf-8"))):
        value = TAG.sub("", match.group(1)).strip()
        if 8 <= len(value) <= 160:
            rank = sha256(f"{path.name}:{index}:{value}".encode("utf-8"))
            candidates.append({"id": f"{path.stem}-{index}", "text": value,
                               "group": path.stem, "rank": rank})
    candidates.sort(key=lambda row: row["rank"])
    rows = [{key: value for key, value in row.items() if key != "rank"}
            for row in candidates[:count]]
    return {"file": path.name, "sha256": sha256(content),
            "eligible": len(candidates), "selected": len(rows)}, rows


def probe(rows, port, prompt_profile):
    endpoint = URL if port == 8765 else f"http://127.0.0.1:{port}"
    opener = build_opener(ProxyHandler({}))
    suite = {"timeZone": "Asia/Shanghai", "recordedAt": "2026-09-23T10:00:00+08:00"}
    answers = {}
    checkpoint = None
    for first in range(0, len(rows), 8):
        batch = rows[first:first + 8]
        request = Request(endpoint + "/api/evaluate", method="POST",
                          headers={"Content-Type": "application/json"},
                          data=json.dumps(payload_for(suite, batch, PROFILES[prompt_profile]),
                                          ensure_ascii=False).encode("utf-8"))
        with opener.open(request, timeout=120) as response:
            result = json.loads(response.read(2_000_001))
        execution = result.get("execution", {})
        if execution.get("backend") != "mlx" or execution.get("network_model_calls") != 0:
            raise RuntimeError("Expected local MLX inference with no network model calls")
        reported = result.get("checkpoint", {})
        if checkpoint is not None and reported != checkpoint:
            raise RuntimeError("Checkpoint changed during probe")
        checkpoint = reported
        for row in result.get("states", []):
            answers[row["id"]] = {name: row["answers"][name]["p_true"] for name in QUESTIONS}
    if set(answers) != {row["id"] for row in rows}:
        raise RuntimeError("Model response did not cover every selected utterance")
    directory = Path(checkpoint.get("directory", "")).resolve()
    if not directory.is_dir():
        raise RuntimeError("Model did not report a local checkpoint directory")
    return {"name": directory.name,
            "configSha256": sha256((directory / "config.json").read_bytes()),
            "weightsSha256": sha256((directory / "best.safetensors").read_bytes())}, answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textgrid", type=Path, action="append", required=True)
    parser.add_argument("--sample-per-file", type=int, default=60)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--prompt-profile", choices=PROFILES, default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.sample_per_file <= 1000:
        parser.error("--sample-per-file must be 1..1000")
    sources, rows = [], []
    for path in args.textgrid:
        source, selected = select(path, args.sample_per_file)
        sources.append(source)
        rows.extend(selected)
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate source file stem or utterance ID")
    checkpoint, answers = probe(rows, args.port, args.prompt_profile)
    from subprocess import run
    from common import RAYNEO_ROOT
    js = "import { explicitIntent } from './services/backend/src/intent-policy.js';" + \
         "import{readFileSync}from'node:fs';const rows=JSON.parse(readFileSync(0,'utf8'));" + \
         "process.stdout.write(JSON.stringify(rows.map(r=>explicitIntent(r.text))));"
    routed = run(["node", "--input-type=module", "-e", js],
                 input=json.dumps(rows, ensure_ascii=False), text=True,
                 capture_output=True, cwd=RAYNEO_ROOT, check=True)
    direct = json.loads(routed.stdout)
    result_rows = [{**row, "pTrue": answers[row["id"]], "direct": direct[index]}
                   for index, row in enumerate(rows)]
    output = {"createdAt": datetime.now(timezone.utc).isoformat(), "source": sources,
              "selection": "SHA256(file name, TextGrid text index, cleaned text); 8..160 characters; first N ranks per file",
              "modelInput": "Current transcript only, no previous-turn context; fixed China time zone and processing date",
              "goldLabels": None, "checkpoint": checkpoint,
              "promptProfile": args.prompt_profile,
              "policySha256": sha256(policy.read_bytes()),
              "selectedCount": len(result_rows),
              "directMatches": sum(row["direct"] is not None for row in result_rows),
              "modelAbove085": {name: sum(row["pTrue"][name] >= 0.85 for row in result_rows)
                                for name in QUESTIONS},
              "rows": result_rows,
              "limitations": "No human intent labels. Model counts are trigger rates, not false-positive rates; meeting transcripts are not glasses ASR."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("selectedCount", "directMatches", "modelAbove085", "checkpoint")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
