#!/usr/bin/env python3
"""Bounded native MLX Chinese intent QLoRA; only last two q/v projections + scalar.

Uses original NanoJev helpers but an intent-only sampler. Test data is never read.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace
from common import HERE, PROJECT, settings
from evaluate import QUESTIONS, digest, metrics, payload_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=HERE / "datasets/zh-intent-v1")
    p.add_argument("--output", type=Path, default=PROJECT / "models/nanojev-intent-zh-lora-v1")
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()
    if not 1 <= args.steps <= 400:
        raise ValueError("Bounded experiment permits only 1..400 steps.")
    if not args.probe_only and args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty model directory.")
    root, checkpoint, _ = settings()
    sys.path.insert(0, str(root / "scripts"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_decisions import MLXDecisionPredictor
    from predict_toy_decisions import prepare_examples
    from train_unified_games_mlx import attach_lora, set_backbone_trainable, dequantized_fused_state
    generator = random.Random(args.seed); mx.random.seed(args.seed)
    train = json.loads((args.dataset / "train.json").read_text())
    dev = json.loads((args.dataset / "dev.json").read_text())
    clock = time.perf_counter()
    runtime = MLXDecisionPredictor(checkpoint, max_length=2048, precision="bf16", quantize=8)
    model = runtime.model
    model.freeze()
    options = SimpleNamespace(lora_layers=2, lora_rank=4, lora_scale=8.0, lora_keys="self_attn.q_proj,self_attn.v_proj")
    attach_lora(model, options); set_backbone_trainable(model, True, True)
    model.scalar.unfreeze()
    model.shared_prefix = False
    trainable = dict(tree_flatten(model.trainable_parameters()))
    if not trainable or any(not (key.startswith("scalar.") or ".lora_a" in key or ".lora_b" in key) for key in trainable):
        raise RuntimeError("Unexpected trainable parameter; refusing to update unrelated model weights.")
    params = sum(value.size for value in trainable.values())
    all_examples = prepare_examples(payload_for(train, train["cases"]), runtime.tokenizer, 2048)
    labels = {case["id"] + ":" + name: int(case[name]) for case in train["cases"] for name in QUESTIONS}
    indexed = {case["id"]: [ex for ex in all_examples if ex["state_id"] == case["id"]] for case in train["cases"]}
    pad = runtime.tokenizer.pad_token_id

    def loss(group):
        logits, _ = model(group, pad)
        z = logits[:, 1]
        y = mx.array([labels[ex["id"]] for ex in group], dtype=mx.float32)
        return (mx.maximum(z, 0) - z * y + mx.log1p(mx.exp(-mx.abs(z)))).mean()

    value_and_grad = nn.value_and_grad(model, loss)
    body_optimizer = optim.AdamW(2e-4, weight_decay=0.01)
    head_optimizer = optim.AdamW(1e-3, weight_decay=0.01)

    def step(group):
        model.train(); model.shared_prefix = False
        value, gradients = value_and_grad(group)
        mx.eval(value, gradients)
        gradients, norm = optim.clip_grad_norm(gradients, 1.0)
        if not math.isfinite(float(value)) or not math.isfinite(float(norm)):
            raise ValueError("Nonfinite LoRA loss/gradient.")
        body_gradient = gradients.pop("backbone", None)
        if not body_gradient or not tree_flatten(body_gradient) or "scalar" not in gradients:
            raise RuntimeError("Missing LoRA or scalar gradient.")
        body_optimizer.update(model.backbone, body_gradient)
        head_optimizer.update(model.scalar, gradients["scalar"])
        mx.eval(model.parameters(), body_optimizer.state, head_optimizer.state)
        peak = mx.get_peak_memory() / 1e9
        if peak > 8.0:
            raise RuntimeError("LoRA exceeded this experiment's 8 GB MLX peak memory budget.")
        return float(value), float(norm), peak

    # Probe the widest eight Boolean paths; no files in the source model are changed.
    widest = sorted(all_examples, key=lambda ex: len(ex["leaf_tokens"][0]), reverse=True)[:8]
    before_probe = {key: np.asarray(value).copy() for key, value in trainable.items()}
    begin = time.perf_counter(); probe_loss, probe_gradient, probe_peak = step(widest)
    after_probe = dict(tree_flatten(model.trainable_parameters()))
    changed = [key for key, value in after_probe.items() if not np.array_equal(np.asarray(value), before_probe[key])]
    if not any(".lora_b" in key for key in changed) or not any(key.startswith("scalar.") for key in changed):
        raise RuntimeError("Probe did not actually update LoRA and scalar parameters.")
    probe = {"parameters": params, "trainableKeys": list(trainable), "changedKeys": changed,
             "loss": probe_loss, "gradientNorm": probe_gradient, "stepSeconds": time.perf_counter() - begin,
             "peakMlxMemoryGb": probe_peak, "loadAndProbeSeconds": time.perf_counter() - clock,
             "rank": 4, "layers": 2, "projections": ["q_proj", "v_proj"]}
    preflight_path = HERE / "results" / f"{args.output.name}-preflight.json"
    preflight_path.write_text(json.dumps(probe, indent=2) + "\n")
    print(json.dumps({"preflight": probe}), flush=True)
    if args.probe_only:
        return
    # Restore pristine adapter/head state so the probe cannot train on a biased widest batch.
    model.load_weights([(key, mx.array(value)) for key, value in before_probe.items()], strict=False)
    body_optimizer = optim.AdamW(2e-4, weight_decay=0.01)
    head_optimizer = optim.AdamW(1e-3, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)

    def evaluate_dev():
        model.eval(); model.shared_prefix = True
        rows = []; ce = []
        for first in range(0, len(dev["cases"]), 8):
            cases = dev["cases"][first:first + 8]
            result = runtime.predict(payload_for(dev, cases))
            answers = {row["id"]: row["answers"] for row in result["states"]}
            for case in cases:
                probs = {name: answers[case["id"]][name]["p_true"] for name in QUESTIONS}
                for name in QUESTIONS:
                    prob = min(1 - 1e-7, max(1e-7, probs[name])); truth = case[name]
                    ce.append(-math.log(prob if truth else 1 - prob))
                rows.append({"id": case["id"], "group": case["group"], "expected": {name: case[name] for name in QUESTIONS}, "pTrue": probs})
        return float(np.mean(ce)), rows

    best, initial_rows = evaluate_dev(); initial_loss = best; best_step = 0
    def save_adapter():
        mx.save_safetensors(str(args.output / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    save_adapter()
    history = []; start = time.perf_counter()
    for iteration in range(1, args.steps + 1):
        sample = generator.sample(train["cases"], 4)
        group = [ex for case in sample for ex in indexed[case["id"]]]
        value, gradient, peak = step(group)
        if iteration % 10 == 0 or iteration == args.steps:
            record = {"step": iteration, "loss": value, "gradientNorm": gradient, "peakMlxMemoryGb": peak,
                      "elapsedSeconds": time.perf_counter() - start}
            if iteration % 40 == 0 or iteration == args.steps:
                dev_loss, rows = evaluate_dev(); record["devLoss"] = dev_loss
                record["devMetrics"] = metrics(rows, 0.85)
                if dev_loss < best:
                    best = dev_loss; best_step = iteration; save_adapter()
            history.append(record); print(json.dumps(record), flush=True)
            (args.output / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
    model.load_weights(list(mx.load(str(args.output / "adapters.safetensors")).items()), strict=False)
    final_dev, final_rows = evaluate_dev()
    mx.save_safetensors(str(args.output / "best.safetensors"), dequantized_fused_state(model), metadata={"format": "pt"})
    shutil.copytree(checkpoint / "tokenizer", args.output / "tokenizer")
    shutil.copytree(checkpoint / "backbone_config", args.output / "backbone_config")
    config = dict(runtime.run_config)
    config.update(schema_version="rayneo-intent-zh-lora-v1", max_length=2048,
                  adaptation={"kind": "qlora", "rank": 4, "layers": 2, "scale": 8, "projections": ["q_proj", "v_proj"],
                              "selected_on": "dev_bce", "best_step": best_step, "seed": args.seed,
                              "source_weights_sha256": digest(checkpoint / "best.safetensors")})
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {"method": config["adaptation"], "preflight": probe, "completedSteps": args.steps, "bestStep": best_step,
              "trainCases": len(train["cases"]), "devCases": len(dev["cases"]), "testUsed": False,
              "initialDevLoss": initial_loss, "selectedDevLoss": final_dev, "trainingSeconds": time.perf_counter() - start,
              "dataSha256": {split: digest(args.dataset / f"{split}.json") for split in ("train", "dev")},
              "scriptSha256": digest(__file__), "weightsSha256": digest(args.output / "best.safetensors"),
              "devMetrics": {str(t): metrics(final_rows, t) for t in (0.85, 0.5)},
              "devPredictions": final_rows, "history": history,
              "notes": "Synthetic supervised prototype. The old 64-case set is regression only; fresh held-out evaluation follows selection."}
    for destination in (args.output / "training.json", HERE / "results" / f"{args.output.name}-training.json"):
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    shutil.copyfile(__file__, args.output / "train_source.py")
    print(json.dumps({"complete": args.output.name, "bestStep": best_step, "devMetrics": report["devMetrics"]}), flush=True)


if __name__ == "__main__":
    main()
