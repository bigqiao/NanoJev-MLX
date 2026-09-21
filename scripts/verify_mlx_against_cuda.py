#!/usr/bin/env python3
"""Re-score recorded CUDA decisions with the MLX backend and report agreement.

The released dataset stores, for every step of the selected checkpoint's test
episodes, the exact observation and the probabilities the CUDA service returned
(`steps[i].scores`). Rebuilding the same policy request and scoring it locally
gives an end-to-end check of the MLX port on real Maze, Snake and ViZDoom states,
for each precision / quantization mode.
"""
import argparse
import json
import random
import time

from unified_game_pipeline import policy_request


def load_decisions(episodes_path, per_task, seed):
    rows = [json.loads(line) for line in open(episodes_path, encoding="utf-8") if line.strip()]
    by_task = {}
    for row in rows:
        for j, step in enumerate(row["steps"]):
            obs = step["observation"]
            if obs.get("candidates") and len(obs["candidates"]) >= 2:
                task = row["case"]["spec"]["task"]
                if task == "shooting":
                    task = "shooting/" + row["case"]["spec"].get("scenario", "")
                by_task.setdefault(task, []).append((f"{row['case']['id']}:{j}", obs, step["scores"]))
    rng = random.Random(seed)
    picked = []
    for task in sorted(by_task):
        items = by_task[task]
        rng.shuffle(items)
        picked += [(task,) + item for item in items[:per_task]]
    return picked, {task: len(items) for task, items in by_task.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--episodes", default="data/NanoJev-unified/evaluation/experiment/selected_test.jsonl")
    parser.add_argument("--per-task", type=int, default=24)
    parser.add_argument("--modes", default="bf16,q8,q4", help="Comma-separated: fp32, bf16, q8, q4")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    import mlx.core as mx
    from predict_toy_decisions import DecisionPredictor
    picked, available = load_decisions(args.episodes, args.per_task, args.seed)
    states = [policy_request(obs, identity) for _, identity, obs, _ in picked]
    report = {"checkpoint": args.checkpoint_dir, "episodes": args.episodes, "decisions_available": available,
              "decisions_compared_per_task": args.per_task, "modes": {}}
    for mode in args.modes.split(","):
        precision = "fp32" if mode == "fp32" else "bf16"
        quantize = int(mode[1:]) if mode.startswith("q") else None
        mx.reset_peak_memory()
        engine = DecisionPredictor(args.checkpoint_dir, device_name="mlx", precision=precision, quantize=quantize)
        resident = mx.get_active_memory() / 1e9
        per_task, started = {}, time.perf_counter()
        for start in range(0, len(states), args.batch):
            result = engine.predict({"states": states[start:start + args.batch]})
            for answered, (task, _, _, cuda) in zip(result["states"], picked[start:start + args.batch]):
                mlx_probs = answered["answers"]["action"]["probabilities"]
                if set(mlx_probs) != set(cuda):
                    raise ValueError("Candidate sets differ between the recording and the rebuilt request")
                bucket = per_task.setdefault(task, {"decisions": 0, "argmax_agree": 0, "max_abs_prob_diff": 0.0,
                                                    "mean_abs_prob_diff": 0.0})
                diffs = [abs(mlx_probs[k] - cuda[k]) for k in cuda]
                bucket["decisions"] += 1
                bucket["argmax_agree"] += max(mlx_probs, key=mlx_probs.get) == max(cuda, key=cuda.get)
                bucket["max_abs_prob_diff"] = max(bucket["max_abs_prob_diff"], max(diffs))
                bucket["mean_abs_prob_diff"] += sum(diffs) / len(diffs)
        for bucket in per_task.values():
            bucket["mean_abs_prob_diff"] /= bucket["decisions"]
        total = sum(b["decisions"] for b in per_task.values())
        agree = sum(b["argmax_agree"] for b in per_task.values())
        summary = {"precision": precision, "quantize": quantize, "resident_gb": resident,
                   "peak_gb": mx.get_peak_memory() / 1e9, "seconds": time.perf_counter() - started,
                   "decisions": total, "argmax_agree": agree,
                   "max_abs_prob_diff": max(b["max_abs_prob_diff"] for b in per_task.values()),
                   "by_task": per_task}
        report["modes"][mode] = summary
        print(f"{mode:5s} resident {resident:.2f} GB, peak {summary['peak_gb']:.2f} GB | max|Δp| {summary['max_abs_prob_diff']:.4f} | "
              f"argmax {agree}/{total} | " + ", ".join(f"{t}: {b['max_abs_prob_diff']:.3f}/{b['argmax_agree']}/{b['decisions']}"
                                                       for t, b in sorted(per_task.items())), flush=True)
        del engine
        mx.clear_cache()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
