#!/usr/bin/env python3
"""Decision latency of the MLX backend on recorded game states.

Times `DecisionPredictor.predict` end to end (tokenization, tensor assembly, backbone,
heads, softmax) for single decisions and small batches, per task, on real Maze, Snake
and ViZDoom observations from the released evaluation episodes. Reports p50/p95 so the
numbers are directly comparable with a service's per-request latency.
"""
import argparse
import json
import random
import statistics
import time

from unified_game_pipeline import policy_request


def load_states(episodes_path, per_task, seed):
    rows = [json.loads(line) for line in open(episodes_path, encoding="utf-8") if line.strip()]
    by_task = {}
    for row in rows:
        for j, step in enumerate(row["steps"]):
            obs = step["observation"]
            if obs.get("candidates") and len(obs["candidates"]) >= 2:
                task = row["case"]["spec"]["task"]
                if task == "shooting":
                    task = "shooting/" + row["case"]["spec"].get("scenario", "")
                by_task.setdefault(task, []).append(policy_request(obs, f"{row['case']['id']}:{j}"))
    rng = random.Random(seed)
    for task, items in by_task.items():
        rng.shuffle(items)
        by_task[task] = items[:per_task]
    return by_task


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--episodes", default="data/NanoJev-unified/evaluation/experiment/selected_test.jsonl")
    parser.add_argument("--per-task", type=int, default=24)
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--quantize", type=int, choices=[4, 8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    from predict_toy_decisions import DecisionPredictor, prepare_examples
    import mlx.core as mx
    states = load_states(args.episodes, args.per_task, args.seed)
    engine = DecisionPredictor(args.checkpoint_dir, device_name="mlx", precision=args.precision, quantize=args.quantize)
    engine.predict({"states": next(iter(states.values()))[:2]})  # warm-up: kernels, caches
    report = {"checkpoint": args.checkpoint_dir, "precision": args.precision, "quantize": args.quantize,
              "device": engine.device, "scope": "predict() end to end incl. tokenization; excludes HTTP", "results": {}}
    for task, items in sorted(states.items()):
        examples = prepare_examples({"states": items}, engine.tokenizer, engine.limit)
        tokens = [max(map(len, ex["leaf_tokens"])) for ex in examples]
        paths = [len(ex["leaf_tokens"]) for ex in examples]
        task_report = {"decisions": len(items), "mean_candidates": statistics.mean(paths),
                       "mean_path_tokens": statistics.mean(tokens), "max_path_tokens": max(tokens), "by_batch": {}}
        for size in [int(s) for s in args.batch_sizes.split(",")]:
            timings = []
            for start in range(0, len(items) - size + 1, size):
                began = time.perf_counter()
                engine.predict({"states": items[start:start + size]})
                timings.append((time.perf_counter() - began) * 1000)
            if not timings:
                continue
            task_report["by_batch"][str(size)] = {
                "requests": len(timings), "p50_ms": percentile(timings, .5), "p95_ms": percentile(timings, .95),
                "mean_ms": statistics.mean(timings), "per_decision_ms": statistics.mean(timings) / size,
                "decisions_per_second": 1000 * size / statistics.mean(timings)}
        report["results"][task] = task_report
        line = " | ".join(f"batch {b}: p50 {v['p50_ms']:.0f} ms, p95 {v['p95_ms']:.0f} ms, {v['per_decision_ms']:.0f} ms/decision"
                          for b, v in task_report["by_batch"].items())
        print(f"{task:28s} {task_report['mean_candidates']:.1f} cands, {task_report['mean_path_tokens']:.0f} tok/path | {line}", flush=True)
    report["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    print(f"peak MLX memory {report['peak_memory_gb']:.2f} GB")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
