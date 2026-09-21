#!/usr/bin/env python3
"""MLX port of the unified SFT stage for Apple Silicon: QLoRA on a quantized, frozen backbone.

Same dataset contract, sampler, population weights, CE objective, dev selection and
output bundle as `train_unified_games.py --stage sft --loss ce`. Memory-wise it differs
from the CUDA trainer on purpose: the backbone is quantized (8-bit by default) and frozen,
LoRA adapters on the attention projections plus the decision heads are trained, every
transformer block is gradient-checkpointed, and each forward has a padded-token budget.
`--full-backbone` restores full fine-tuning for machines with enough unified memory.

The selected checkpoint is exported as a dequantized, LoRA-fused float32 `best.safetensors`
in the torch key layout, so it loads in both the CUDA and the MLX backend. The critic stage
(Brier / paired_brier_pg / TD) is not ported; use the CUDA trainer for it.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

from predict_toy_decisions import read_json
from train_pipeline_decisions import SPLITS, dump, pack_complete_questions, target_for
from train_unified_games import (
    BalancedQuestionSampler, file_sha256, objective_for, population_weights, prepare_unified_examples,
    read_unified_dataset, sampling_cell, summarize_predictions, target_transform, validate_sampling_pools,
)

DEFAULT_LORA_KEYS = "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"
TORCH_KEY = {"set_attention.in_proj.weight": "set_attention.in_proj_weight",
             "set_attention.in_proj.bias": "set_attention.in_proj_bias"}


def question_targets(examples, kmax):
    """Dense [B, kmax] targets and per-question population weights; padded slots stay zero."""
    import numpy as np
    targets = np.zeros((len(examples), kmax), dtype=np.float32)
    weights = np.zeros(len(examples), dtype=np.float32)
    for i, ex in enumerate(examples):
        target = target_for(ex, objective_for(ex))
        if target is None:
            raise ValueError(f"Missing eligible target entered training: {ex['id']}")
        if ex["record_role"] != "policy":
            raise ValueError("Outcome rows cannot enter SFT optimization")
        weight = ex.get("loss_weight")
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Every sampled question needs a finite positive population weight")
        targets[i, :len(target)] = target
        weights[i] = weight
    return targets, weights


def weighted_ce(model, group, pad_token, targets, weights):
    """Sum over questions of weight * CE(target, softmax over that question's real candidates)."""
    import mlx.core as mx
    logits, _ = model(group, pad_token)
    # -1e9 fill on padded candidates vanishes under log_softmax; targets there are zero.
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -(targets * logp).sum(-1) @ weights


def mlx_prediction_record(ex, values):
    import mlx.core as mx
    k = len(ex["candidate_ids"])
    z = values[:k].astype(mx.float32)
    keep = ("id", "state_id", "family_id", "split", "qid", "type", "candidate_ids", "gold_index",
            "teacher_raw_probs", "teacher_probs", "teacher_rounding")
    record = {**{key: ex.get(key) for key in keep},
              "teacher_target_kind": "rounded_proxy_distribution" if ex.get("teacher_probs") is not None else None,
              "student_logits": z.tolist(), "student_probs": mx.softmax(z, axis=-1).tolist()}
    if "gold_distribution_probs" in ex:
        record.update(gold_probs=ex.get("gold_probs"), gold_distribution_probs=ex["gold_distribution_probs"],
                      gold_probs_kind=ex.get("gold_probs_kind"), gold_label_kind=ex.get("gold_label_kind"),
                      teacher_target_error=ex.get("teacher_target_error"))
    return record


def evaluate_split(model, examples, pad_token, args, weights, path=None, require_all=False):
    import mlx.core as mx
    model.eval()
    rows = []
    for group in pack_complete_questions(examples, args.microbatch_questions, args.max_microbatch_tokens):
        logits, _ = model(group, pad_token)
        mx.eval(logits)
        for ex, values in zip(group, logits):
            if not bool(mx.isfinite(values[:len(ex["candidate_ids"])]).all()):
                raise RuntimeError("Nonfinite evaluation logits")
            row = mlx_prediction_record(ex, values)
            row.update(task=ex["task"], record_role=ex["record_role"],
                       continuation_policy_id=ex["continuation_policy_id"],
                       training_target=target_for(ex, objective_for(ex)), target_objective=objective_for(ex),
                       policy_target_kind=ex.get("policy_target_kind"), scenario=ex.get("scenario"),
                       target_transform=target_transform(ex))
            rows.append(row)
    if path:
        Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    return summarize_predictions(rows, weights, require_all)


def is_adapter_key(key):
    return not key.startswith("backbone.") or ".lora_a" in key or ".lora_b" in key


def adapter_state(model):
    """Every trained tensor (LoRA factors and decision heads) regardless of the current freeze state."""
    from mlx.utils import tree_flatten
    return {key: value for key, value in tree_flatten(model.parameters()) if is_adapter_key(key)}


def trainable_state(model):
    from mlx.utils import tree_flatten
    return dict(tree_flatten(model.trainable_parameters()))


def attach_lora(model, args):
    """Freeze the backbone and add LoRA factors to the configured projections of the last N blocks."""
    from mlx_lm.tuner.utils import linear_to_lora_layers
    model.backbone.freeze()
    layers = len(model.backbone.layers) if args.lora_layers < 0 else args.lora_layers
    linear_to_lora_layers(model.backbone, layers, {"rank": args.lora_rank, "scale": args.lora_scale,
                                                    "dropout": 0.0, "keys": set(args.lora_keys.split(","))})
    return layers


def set_backbone_trainable(model, flag, lora):
    """Head warmup freezes everything in the backbone; otherwise LoRA factors (or all weights) train."""
    from mlx_lm.tuner.lora import LoRALinear
    if not lora:
        (model.backbone.unfreeze if flag else model.backbone.freeze)()
        return
    model.backbone.freeze()
    if flag:
        for _, module in model.backbone.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(recurse=False, keys=["lora_a", "lora_b"])


def dequantized_fused_state(model):
    """float32 tensors in the torch layout: LoRA fused into dequantized base weights, heads as-is."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear
    tensors, handled = {}, []
    for name, module in model.named_modules():
        if any(name.startswith(prefix + ".") for prefix in handled):
            continue
        if isinstance(module, LoRALinear):
            fused = module.fuse(dequantize=True)
            tensors[name + ".weight"] = fused.weight.astype(mx.float32)
            if "bias" in fused:
                tensors[name + ".bias"] = fused.bias.astype(mx.float32)
            handled.append(name)
        elif isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            tensors[name + ".weight"] = mx.dequantize(module.weight, module.scales, module.get("biases"),
                                                     module.group_size, module.bits, module.mode).astype(mx.float32)
            if "bias" in module:
                tensors[name + ".bias"] = module.bias.astype(mx.float32)
            handled.append(name)
    for key, value in tree_flatten(model.parameters()):
        if key not in tensors and not any(key.startswith(prefix + ".") for prefix in handled):
            tensors[key] = value.astype(mx.float32)
    mx.eval(tensors)
    return {TORCH_KEY.get(key, key): value for key, value in tensors.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Dataset directory with manifest.json and five split JSONL files")
    parser.add_argument("--init-checkpoint", help="Complete local DecisionModel bundle; fresh optimizer")
    parser.add_argument("--output-dir")
    parser.add_argument("--balance", choices=["task", "task_role"], default="task")
    parser.add_argument("--policy-pool-weights", help="JSON file of absolute SFT pool weights")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--head-steps", type=int, default=0)
    parser.add_argument("--batch-questions", type=int, default=12)
    parser.add_argument("--microbatch-questions", type=int, default=4)
    parser.add_argument("--max-microbatch-tokens", type=int, default=4096,
                        help="Padded token budget per forward; the main activation-memory knob")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--quantize", type=int, choices=[0, 4, 8], default=8,
                        help="Backbone quantization bits; 8 is near-lossless, 4 is smallest, 0 keeps bf16/fp32")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16",
                        help="Backbone dtype before quantization and for --full-backbone")
    parser.add_argument("--full-backbone", action="store_true",
                        help="Full fine-tuning without LoRA (needs --quantize 0 and far more unified memory)")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--lora-layers", type=int, default=-1, help="Blocks (from the top) that receive LoRA; -1 = all")
    parser.add_argument("--lora-keys", default=DEFAULT_LORA_KEYS, help="Comma-separated module keys inside each block")
    parser.add_argument("--lora-lr", type=float, default=1e-4, help="LoRA factor learning rate")
    parser.add_argument("--backbone-lr", type=float, default=2e-5, help="Only with --full-backbone")
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--head-warmup-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--memory-fraction", type=float, default=.6,
                        help="Abort when MLX peak memory exceeds this fraction of physical memory")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--validate-only", action="store_true", help="Stdlib schema/provenance audit; no model")
    args = parser.parse_args(argv)
    args.stage, args.loss = "sft", "ce"
    try:
        args.resolved_policy_pool_weights = read_json(args.policy_pool_weights) if args.policy_pool_weights else None
        weights = population_weights("sft", args.balance, .25, args.resolved_policy_pool_weights)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if min(args.steps, args.microbatch_questions, args.max_length, args.eval_every) <= 0 or args.head_steps < 0 or args.max_microbatch_tokens < 0:
        parser.error("Steps, batch and token limits must be valid positive sizes")
    if args.batch_questions < len(weights):
        parser.error(f"--batch-questions must be >= {len(weights)}")
    if any(not math.isfinite(v) or v <= 0 for v in (args.lora_lr, args.backbone_lr, args.head_lr, args.head_warmup_lr)):
        parser.error("Learning rates must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Weight decay must be finite and nonnegative")
    if args.full_backbone and args.quantize:
        parser.error("--full-backbone requires --quantize 0; quantized weights cannot receive dense updates")
    if args.lora_rank <= 0 or args.lora_scale <= 0 or not 0 < args.memory_fraction <= 1:
        parser.error("LoRA rank/scale must be positive and --memory-fraction in (0,1]")
    if not args.validate_only and (not args.init_checkpoint or not args.output_dir):
        parser.error("Training requires --init-checkpoint and --output-dir")
    return args


def main(argv=None):
    args = parse_args(argv)
    records, manifest, files, schema_audit = read_unified_dataset(args.input)
    weights = population_weights("sft", args.balance, .25, args.resolved_policy_pool_weights)
    sampling_audit = validate_sampling_pools(records, weights)
    if args.validate_only:
        print(json.dumps({**schema_audit, "stage": "sft", "loss": "ce",
                          "population_weights": {"/".join(k): v for k, v in weights.items()},
                          "eligible_by_split_sampling_pool": sampling_audit}, ensure_ascii=False))
        return
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory; existing run artifacts are not overwritten")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_decisions import MLXDecisionPredictor, limit_mlx_memory
    random.seed(args.seed)
    mx.random.seed(args.seed)
    memory_limit = limit_mlx_memory(args.memory_fraction)
    runtime = MLXDecisionPredictor(args.init_checkpoint, max_length=args.max_length, precision=args.precision,
                                   quantize=args.quantize or None)
    model, tokenizer = runtime.model, runtime.tokenizer
    pad = tokenizer.pad_token_id
    lora = not args.full_backbone
    lora_layers = attach_lora(model, args) if lora else 0
    if args.gradient_checkpointing:
        # Recompute each transformer block in the backward pass; unified memory is the binding limit.
        from mlx_lm.tuner.trainer import grad_checkpoint
        grad_checkpoint(model.backbone.layers[0])
    set_backbone_trainable(model, True, lora)
    trainable = trainable_state(model)
    parameter_counts = {"stored_elements": sum(v.size for _, v in tree_flatten(model.parameters())),
                        "trainable": sum(v.size for v in trainable.values()),
                        "lora": sum(v.size for k, v in trainable.items() if ".lora_" in k),
                        "heads": sum(v.size for k, v in trainable.items() if not k.startswith("backbone."))}
    examples, target_audit = prepare_unified_examples(records, tokenizer, args.max_length)
    splits = {split: [ex for ex in examples if ex["split"] == split] for split in SPLITS}
    sampler = BalancedQuestionSampler(examples, "sft", args.balance, .25, args.seed, args.resolved_policy_pool_weights)
    for split_examples in splits.values():
        pack_complete_questions(split_examples, args.microbatch_questions, args.max_microbatch_tokens)

    def loss_fn(group, targets, weights_):
        return weighted_ce(model, group, pad, mx.array(targets), mx.array(weights_))

    value_and_grad = nn.value_and_grad(model, loss_fn)

    def check_memory(stage):
        peak = mx.get_peak_memory()
        if peak > memory_limit:
            raise RuntimeError(f"MLX peak memory {peak/1e9:.2f} GB exceeded the {memory_limit/1e9:.2f} GB limit during "
                               f"{stage}; lower --max-microbatch-tokens / --microbatch-questions / --max-length, "
                               "or raise --memory-fraction on a machine with more unified memory")
        return peak

    # Preflight: the widest training microbatch, forward and backward, before anything is written.
    widest = max(pack_complete_questions(splits["train"], args.microbatch_questions, args.max_microbatch_tokens),
                 key=lambda g: sum(len(ex["leaf_tokens"]) for ex in g) * max(len(t) for ex in g for t in ex["leaf_tokens"]))
    model.train()
    mx.reset_peak_memory()
    probe = [{**ex, "loss_weight": 1.0} for ex in widest]
    loss, grads = value_and_grad(probe, *question_targets(probe, max(len(ex["candidate_ids"]) for ex in probe)))
    mx.eval(loss, grads)
    del loss, grads
    mx.clear_cache()
    preflight = {"questions": len(widest), "leaf_paths": sum(len(ex["leaf_tokens"]) for ex in widest),
                 "padded_tokens": sum(len(ex["leaf_tokens"]) for ex in widest) * max(len(t) for ex in widest for t in ex["leaf_tokens"]),
                 "peak_memory_gb": check_memory("preflight") / 1e9, "memory_limit_gb": memory_limit / 1e9}
    print(json.dumps({"preflight": preflight, "parameters": parameter_counts}), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "schema_version": "nanojev-unified-games-v1", "backend": "mlx",
              "model": runtime.run_config.get("model"), "set_head": runtime.run_config["set_head"],
              "resolved_model_revision": runtime.run_config.get("resolved_model_revision"),
              "initialization": "local DecisionModel warm start; fresh optimizer",
              "init_weights_sha256": file_sha256(runtime.root / "best.safetensors"),
              "data_sha256": {str(path): file_sha256(path) for path in files},
              "implementation_sha256": file_sha256(__file__),
              "continuation_policy_id": schema_audit["continuation_policy_id"],
              "population_weights": {"/".join(k): v for k, v in weights.items()},
              "policy_pool_weights_sha256": file_sha256(args.policy_pool_weights) if args.policy_pool_weights else None,
              "eligible_by_split_sampling_pool": sampling_audit,
              "sampling": "stratified with replacement; per-cell exact population weights; train rows only",
              "adaptation": ({"kind": "qlora", "quantization": runtime.quantization, "lora_layers": lora_layers,
                              "lora_keys": args.lora_keys, "rank": args.lora_rank, "scale": args.lora_scale}
                             if lora else {"kind": "full", "quantization": None}),
              "parameter_counts": parameter_counts, "preflight": preflight,
              "parameter_storage": "bfloat16" if args.precision == "bf16" else "float32",
              "forward_autocast": "none (native MLX dtype)",
              "objective": "Choice CE with explicit API or expert targets",
              "selection": "minimum fixed population-weighted dev CE including initial checkpoint; test only after selection",
              "export": "dequantized, LoRA-fused float32 best.safetensors in the torch key layout",
              "temperature": 1.0, "temperature_fitted": False,
              "deps": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "transformers", "safetensors")},
              "gpu": f"mlx:{mx.default_device().type.name} ({mx.device_info()['device_name']})",
              "schema_counts": schema_audit, "td_enabled": False}
    dump(out / "config.json", config)
    dump(out / "target_audit.json", target_audit)
    tokenizer.save_pretrained(out / "tokenizer")
    (out / "backbone_config").mkdir(exist_ok=True)
    (out / "backbone_config" / "config.json").write_text(
        (runtime.root / "backbone_config" / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
    initial = evaluate_split(model, splits["dev"], pad, args, weights, out / "initial_dev.jsonl", require_all=True)
    dump(out / "initial_dev_metrics.json", initial)
    best, best_step = initial["selection_ce"], 0
    if not math.isfinite(best):
        raise RuntimeError("Initial dev metric is nonfinite")

    def save_best():
        # Small: LoRA factors + heads. The fused float32 bundle is exported once at the end.
        mx.save_safetensors(str(out / "adapters.safetensors"), adapter_state(model))

    save_best()
    body_opt = optim.AdamW(args.lora_lr if lora else args.backbone_lr, weight_decay=args.weight_decay, bias_correction=True)
    head_opt = optim.AdamW(args.head_lr, weight_decay=args.weight_decay, bias_correction=True)

    def apply_update(grads, warm):
        # Split by top-level key. A frozen backbone still yields an empty subtree from
        # trainable_parameters(), so body_opt must only ever see complete backbone gradients.
        body_grads = grads.pop("backbone", None)
        if not warm:
            if not body_grads or not tree_flatten(body_grads):
                raise RuntimeError("Backbone gradients are missing outside head warmup")
            body_opt.update(model.backbone, body_grads)
        head_opt.update(model, grads)

    started = time.perf_counter()
    logs = []
    online_compute = dict.fromkeys(("forward_calls", "question_instances", "leaf_paths", "padded_tokens"), 0)
    for step in range(args.head_steps + args.steps):
        warm = step < args.head_steps
        set_backbone_trainable(model, not warm, lora)
        head_opt.learning_rate = args.head_warmup_lr if warm else args.head_lr
        batch = sampler.sample(args.batch_questions)
        groups = pack_complete_questions(batch, args.microbatch_questions, args.max_microbatch_tokens)
        online_step = {"forward_calls": len(groups), "question_instances": len(batch),
                       "leaf_paths": sum(len(ex["leaf_tokens"]) for ex in batch),
                       "padded_tokens": sum(sum(len(ex["leaf_tokens"]) for ex in group) *
                                            max(len(tokens) for ex in group for tokens in ex["leaf_tokens"])
                                            for group in groups)}
        for key, value in online_step.items():
            online_compute[key] += value
        model.train()
        loss_sum, grads = 0.0, None
        for group in groups:
            kmax = max(len(ex["candidate_ids"]) for ex in group)
            targets, qweights = question_targets(group, kmax)
            loss, group_grads = value_and_grad(group, targets, qweights)
            mx.eval(loss, group_grads)
            if not math.isfinite(float(loss)):
                raise RuntimeError("Nonfinite training loss")
            loss_sum += float(loss)
            grads = group_grads if grads is None else tree_map(lambda a, b: a + b, grads, group_grads)
        grads, grad_norm = optim.clip_grad_norm(grads, 1.0)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError("Nonfinite gradient norm")
        apply_update(grads, warm)
        mx.eval(model.parameters(), body_opt.state, head_opt.state)
        del grads
        cells = Counter((ex["task"], ex["record_role"]) for ex in batch)
        item = {"step": step + 1, "phase": "head" if warm else "full", "loss": loss_sum,
                "gradient_norm_before_clip": float(grad_norm), "microbatches": len(groups),
                "sample_counts": {"/".join(k): v for k, v in sorted(cells.items())},
                "sample_pool_counts": {"/".join(k): v for k, v in sorted(Counter(sampling_cell(ex, weights) for ex in batch).items())},
                "batch_question_ids_sha256": hashlib.sha256("\n".join(ex["id"] for ex in batch).encode()).hexdigest(),
                "elapsed_seconds": time.perf_counter() - started, "online_compute": online_step,
                "peak_memory_gb": check_memory(f"step {step + 1}") / 1e9}
        if not warm and ((step + 1 - args.head_steps) % args.eval_every == 0 or step + 1 == args.steps + args.head_steps):
            metrics = evaluate_split(model, splits["dev"], pad, args, weights, require_all=True)
            item["dev"] = metrics
            if metrics["selection_ce"] < best:
                best, best_step = metrics["selection_ce"], step + 1
                save_best()
        logs.append(item)
        if step % 12 == 0 or "dev" in item:
            dump(out / "train_log.json", logs)
            print(json.dumps(item, allow_nan=False), flush=True)
    training_seconds = time.perf_counter() - started
    peak_training = mx.get_peak_memory() / 1e9

    # Export the selected adapters as one dense float32 bundle, then evaluate that bundle from disk
    # exactly as inference would load it.
    model.load_weights(list(mx.load(str(out / "adapters.safetensors")).items()), strict=False)
    mx.save_safetensors(str(out / "best.safetensors"), dequantized_fused_state(model), metadata={"format": "pt"})
    del model, runtime, value_and_grad, body_opt, head_opt
    mx.clear_cache()
    selected = MLXDecisionPredictor(out, max_length=args.max_length, precision="bf16")
    final = {}
    for split in ("dev", "calibration", "test", "ood"):
        if splits[split]:
            final[split] = evaluate_split(selected.model, splits[split], pad, args, weights,
                                          out / f"predictions_{split}.jsonl", require_all=split == "dev")
    summary = {"best_step": best_step, "best_dev_selection_ce": best, "selected_on": "dev only",
               "stage": "sft", "loss": "ce", "backend": "mlx", "adaptation": config["adaptation"],
               "completed_steps": args.steps + args.head_steps, "metrics_by_split": final,
               "final_eval_note": "metrics_by_split come from the exported float32 bundle loaded in bf16 without quantization",
               "training_seconds": training_seconds, "max_gpu_allocated_gb": peak_training,
               "weights_sha256": file_sha256(out / "best.safetensors"),
               "continuation_policy_id": schema_audit["continuation_policy_id"],
               "temperature": 1.0, "temperature_fitted": False, "online_compute": online_compute}
    dump(out / "train_log.json", logs)
    dump(out / "summary.json", summary)
    print(json.dumps({"done": str(out), **summary}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
