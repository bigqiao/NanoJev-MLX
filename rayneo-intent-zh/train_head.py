#!/usr/bin/env python3
"""Supervise NanoJev's existing scalar head on frozen native MLX encoder features.

Train/dev only. Held-out test is evaluated later through the exported server API.
No game-task sampler, replacement encoder, or LLM decision fallback is used.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

from common import HERE, PROJECT, settings
from evaluate import QUESTIONS, digest, metrics, payload_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE / "datasets/zh-intent-v1")
    parser.add_argument("--output", type=Path, default=PROJECT / "models/nanojev-intent-zh-head-v1")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--standardize", action="store_true", help="Train-only feature preconditioning, folded into exported scalar weights")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new empty output directory; existing weights are never overwritten.")
    if not 1 <= args.epochs <= 2000 or not 0 < args.learning_rate <= 0.1:
        raise ValueError("Invalid training hyperparameters.")
    root, checkpoint, _ = settings()
    sys.path.insert(0, str(root / "scripts"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx_decisions import MLXDecisionPredictor
    from predict_toy_decisions import prepare_examples
    from safetensors import safe_open
    from safetensors.numpy import save_file
    random.seed(args.seed); np.random.seed(args.seed); mx.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    train_suite = json.loads((args.dataset / "train.json").read_text())
    dev_suite = json.loads((args.dataset / "dev.json").read_text())
    # Reject exact cross-split duplicates before any model is trained. A
    # declared metadata augmentation may repeat a transcript within a split,
    # but only as four explicitly identified, label-identical state variants.
    import re
    norm = lambda text: re.sub(r"[\W_]", "", text)
    train_text = {norm(c["text"]) for c in train_suite["cases"]}
    dev_text = {norm(c["text"]) for c in dev_suite["cases"]}
    augmented = (train_suite.get("metadataAugmentation") == "full-factorial-null-or-stable-speaker-null-or-ms-time-v1"
                 and dev_suite.get("metadataAugmentation") == train_suite.get("metadataAugmentation"))
    if train_text & dev_text:
        raise ValueError("Cross-split text leakage.")
    if augmented:
        variants = {"null_null", "stable_null", "null_ms", "stable_ms"}
        for split, suite in (("train", train_suite), ("dev", dev_suite)):
            grouped = {}
            for case in suite["cases"]:
                key = case["canonicalId"]
                grouped.setdefault(key, []).append(case)
            if len(grouped) != len({norm(rows[0]["text"]) for rows in grouped.values()}):
                raise ValueError(f"Repeated canonical transcripts in {split}.")
            for key, rows in grouped.items():
                if ({row["metadataVariant"] for row in rows} != variants or len(rows) != 4
                        or len({(row["text"], row["assist"], row["schedule"]) for row in rows}) != 1):
                    raise ValueError(f"Invalid metadata variants for {key}.")
    elif len(train_text) != len(train_suite["cases"]) or len(dev_text) != len(dev_suite["cases"]):
        raise ValueError("Duplicate supervision within a split.")
    before = time.perf_counter()
    runtime = MLXDecisionPredictor(checkpoint, max_length=2048, precision="bf16", quantize=8)
    runtime.model.backbone.freeze(); runtime.model.norm.freeze()
    head = runtime.model.scalar

    def encode(suite, split):
        arrays, targets, qids = [], [], []
        cases = suite["cases"]
        for first in range(0, len(cases), 8):
            batch = cases[first:first + 8]
            examples = prepare_examples(payload_for(suite, batch), runtime.tokenizer, 2048)
            leaves = runtime.model.leaf_states(examples, runtime.tokenizer.pad_token_id)
            features = runtime.model.norm(leaves.astype(mx.float32))
            mx.eval(features)
            arrays.append(np.asarray(features, dtype=np.float32))
            for case in batch:
                for name in QUESTIONS:
                    if type(case[name]) is not bool:
                        raise ValueError("Expected boolean supervision.")
                    targets.append(float(case[name])); qids.append(name)
            if first % 40 == 0:
                print(json.dumps({"encoding": split, "done": first + len(batch), "total": len(cases),
                                  "elapsedSeconds": time.perf_counter() - before}), flush=True)
        features = np.concatenate(arrays)
        # Cache is private ignored model state, tied to exact dataset/base hashes below.
        np.savez(args.output / f"features_{split}.npz", x=features, y=np.array(targets, dtype=np.float32))
        return mx.array(features), mx.array(targets, dtype=mx.float32), qids

    x_train, y_train, train_qids = encode(train_suite, "train")
    x_dev, y_dev, dev_qids = encode(dev_suite, "dev")
    encoded_seconds = time.perf_counter() - before
    feature_mean = mx.zeros((x_train.shape[1],), dtype=mx.float32)
    feature_scale = mx.ones_like(feature_mean)
    if args.standardize:
        feature_mean = mx.mean(x_train, axis=0)
        feature_scale = mx.maximum(mx.std(x_train, axis=0), 0.005)
        # Same initial logits; parameterize the optimizer in centered/scaled units.
        old_weight = head.weight
        head.bias = head.bias + old_weight @ feature_mean
        head.weight = old_weight * feature_scale
        x_train = (x_train - feature_mean) / feature_scale
        x_dev = (x_dev - feature_mean) / feature_scale
        mx.eval(head.parameters(), x_train, x_dev, feature_mean, feature_scale)
        np.savez(args.output / "preconditioner.npz", mean=np.asarray(feature_mean), scale=np.asarray(feature_scale))
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)

    def loss(x, y):
        z = head(x).squeeze(-1)
        return (mx.maximum(z, 0) - z * y + mx.log1p(mx.exp(-mx.abs(z)))).mean()

    loss_and_grad = nn.value_and_grad(head, loss)
    best_loss = float(loss(x_dev, y_dev)); best_epoch = 0
    best = {"weight": np.asarray(head.weight).copy(), "bias": np.asarray(head.bias).copy()}
    initial_dev = best_loss
    history = []
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = np.random.permutation(len(train_qids))
        losses = []
        for first in range(0, len(order), 64):
            indices = mx.array(order[first:first + 64])
            value, gradients = loss_and_grad(x_train[indices], y_train[indices])
            optimizer.update(head, gradients)
            mx.eval(value, head.parameters(), optimizer.state)
            if not np.isfinite(float(value)):
                raise ValueError("Nonfinite training loss.")
            losses.append(float(value))
        if epoch % 5 == 0 or epoch == args.epochs:
            dev_loss = float(loss(x_dev, y_dev))
            if dev_loss < best_loss:
                best_loss = dev_loss; best_epoch = epoch
                best = {"weight": np.asarray(head.weight).copy(), "bias": np.asarray(head.bias).copy()}
            record = {"epoch": epoch, "trainLoss": float(np.mean(losses)), "devLoss": dev_loss,
                      "bestEpoch": best_epoch, "elapsedSeconds": time.perf_counter() - training_start}
            history.append(record)
            if epoch % 20 == 0 or epoch == args.epochs:
                print(json.dumps(record), flush=True)
            if epoch - best_epoch >= 40:
                break
    head.load_weights([(name, mx.array(value)) for name, value in best.items()])
    mx.eval(head.parameters())

    def prediction_rows(suite, features):
        values = np.asarray(mx.sigmoid(head(features).squeeze(-1))).tolist()
        return [{"id": case["id"], "group": case["group"], "expected": {name: case[name] for name in QUESTIONS},
                 "pTrue": {name: values[i * 2 + n] for n, name in enumerate(QUESTIONS)}} for i, case in enumerate(suite["cases"])]

    dev_rows = prediction_rows(dev_suite, x_dev)
    report = {"method": "Supplied NanoJev encoder and LayerNorm frozen; existing scalar linear head supervised with BCE.",
              "parentAdaptation": runtime.run_config.get("adaptation"),
              "encoderQuantization": 8, "encoderPrecision": "bf16", "headPrecision": "float32",
              "trainOnlyStandardization": args.standardize,
              "trainCases": len(train_suite["cases"]), "devCases": len(dev_suite["cases"]), "testUsedForSelection": False,
              "selection": "Minimum dev BCE including the unchanged initial head; no threshold fitting.",
              "seed": args.seed, "learningRate": args.learning_rate, "maxEpochs": args.epochs,
              "bestEpoch": best_epoch, "initialDevLoss": initial_dev, "selectedDevLoss": best_loss,
              "encodingSeconds": encoded_seconds, "trainingSeconds": time.perf_counter() - training_start,
              "peakMlxMemoryGb": mx.get_peak_memory() / 1e9, "headParameters": sum(value.size for value in best.values()),
              "baseWeightsSha256": digest(checkpoint / "best.safetensors"),
              "dataSha256": {split: digest(args.dataset / f"{split}.json") for split in ("train", "dev")},
              "scriptSha256": digest(__file__), "questions": QUESTIONS,
              "devMetrics": {str(threshold): metrics(dev_rows, threshold) for threshold in (0.85, 0.5)},
              "history": history, "devPredictions": dev_rows,
              "limitations": ["Synthetic supervision with category-conditioned LLM labels; limited manual audit only.",
                              "Only a linear decision head is adapted; no encoder/LoRA training in this run.",
                              "Test remains unseen until the exported checkpoint is selected and served."]}
    # Export exact original backbone/norm/set-head tensors, replacing only scalar.*.
    # This preserves the reference checkpoint/API contract without modifying NanoJev.
    config = dict(runtime.run_config)
    config.update(schema_version="rayneo-intent-zh-head-v1", max_length=2048,
                  adaptation={"kind": "frozen_encoder_scalar_head", "selected_on": "dev_bce", "best_epoch": best_epoch,
                              "parent_adaptation": runtime.run_config.get("adaptation"),
                              "train_only_preconditioning_folded_into_head": args.standardize,
                              "base_weights_sha256": report["baseWeightsSha256"], "seed": args.seed})
    export_weight = best["weight"] / np.asarray(feature_scale)
    export_best = {"weight": export_weight,
                   "bias": best["bias"] - export_weight @ np.asarray(feature_mean)}
    del runtime, loss_and_grad, optimizer, x_train, x_dev, head
    gc.collect(); mx.clear_cache()
    with safe_open(str(checkpoint / "best.safetensors"), framework="np") as source:
        tensors = {name: (export_best[name.removeprefix("scalar.")] if name in ("scalar.weight", "scalar.bias") else source.get_tensor(name))
                   for name in source.keys()}
    save_file(tensors, str(args.output / "best.safetensors"), metadata={"format": "pt"})
    del tensors
    shutil.copytree(checkpoint / "tokenizer", args.output / "tokenizer")
    shutil.copytree(checkpoint / "backbone_config", args.output / "backbone_config")
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    report["weightsSha256"] = digest(args.output / "best.safetensors")
    report["outputName"] = args.output.name
    (args.output / "training.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    shutil.copyfile(__file__, args.output / "train_source.py")
    destination = HERE / "results" / f"{args.output.name}-training.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": args.output.name, "bestEpoch": best_epoch, "devMetrics": report["devMetrics"],
                      "trainingSeconds": report["trainingSeconds"], "encodingSeconds": encoded_seconds}), flush=True)


if __name__ == "__main__":
    main()
