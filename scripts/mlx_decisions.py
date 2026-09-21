#!/usr/bin/env python3
"""MLX backend for NanoJev decision inference on Apple Silicon.

Mirrors `train_toy_decisions.DecisionModel` (Qwen3 backbone + LayerNorm + scalar
head + set-attention head) with mlx-lm's Qwen3 blocks, and loads the exact
`best.safetensors` produced by the CUDA trainer. Same request/response contract
as `predict_toy_decisions.DecisionPredictor`; no torch, no CUDA, no generation.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from predict_toy_decisions import (answer_from_probabilities, complete_question_batches,
                                   local_checkpoint_files, prepare_examples, read_json,
                                   validate_request)

SET_DIM, SET_HEADS = 128, 4  # nn.MultiheadAttention(128, 4) in the torch reference
QUANT_GROUP_SIZE = {8: 64, 4: 32}  # 4-bit needs finer groups to keep decision probabilities close to bf16
MEMORY_FRACTION = 0.6  # default cap on MLX memory relative to physical memory; refuse rather than swap
CACHE_LIMIT_BYTES = 512 * 1024 * 1024  # freed buffers MLX may keep for reuse instead of returning to the OS


def limit_mlx_memory(fraction=MEMORY_FRACTION, cache_bytes=CACHE_LIMIT_BYTES):
    """Cap MLX's memory guideline and its buffer cache; return the memory cap in bytes."""
    mx, _ = _mx()
    physical = mx.device_info().get("memory_size") or os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    limit = int(physical * fraction)
    mx.set_memory_limit(limit)
    mx.set_cache_limit(cache_bytes)
    return limit


def quantize_backbone(model, bits):
    """4/8-bit affine quantization of every Linear and Embedding in the backbone; heads stay float32."""
    mx, nn = _mx()
    if bits not in {4, 8}:
        raise ValueError("quantize 必须为 4 或 8")
    group = QUANT_GROUP_SIZE[bits]
    nn.quantize(model.backbone, group_size=group, bits=bits)
    mx.eval(model.parameters())
    mx.clear_cache()
    return {"bits": bits, "group_size": group, "mode": "affine", "scope": "backbone Linear+Embedding"}


def _backbone_args(body_config):
    """Build mlx-lm Qwen3 ModelArgs from a transformers config.json (4.x or 5.x layout)."""
    from mlx_lm.models.qwen3 import ModelArgs
    cfg = dict(body_config)
    if cfg.get("model_type") != "qwen3":
        raise ValueError(f"MLX backend supports qwen3 backbones only, got {cfg.get('model_type')!r}")
    rope = cfg.get("rope_parameters") or {}
    cfg.setdefault("rope_theta", rope.get("rope_theta", 10000.0))
    if rope.get("rope_type", "default") != "default" and "rope_scaling" not in cfg:
        cfg["rope_scaling"] = rope
    cfg.setdefault("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    cfg.setdefault("tie_word_embeddings", True)
    cfg.setdefault("num_key_value_heads", cfg["num_attention_heads"])
    return ModelArgs.from_dict(cfg)


def _mx():
    import mlx.core as mx
    import mlx.nn as nn
    return mx, nn


def build_decision_model(body_config, set_head):
    """Instantiate the MLX DecisionModel; parameters are uninitialised until load_weights."""
    mx, nn = _mx()
    from mlx_lm.models.qwen3 import Qwen3Model

    class SetAttention(nn.Module):
        # torch nn.MultiheadAttention(embed_dim, heads, batch_first=True) with key_padding_mask.
        def __init__(self, dims, heads):
            super().__init__()
            self.heads = heads
            self.in_proj = nn.Linear(dims, 3 * dims)
            self.out_proj = nn.Linear(dims, dims)

        def __call__(self, u, valid):
            B, K, D = u.shape
            q, k, v = mx.split(self.in_proj(u), 3, axis=-1)
            shape = (B, K, self.heads, D // self.heads)
            q = q.reshape(shape).transpose(0, 2, 1, 3)
            k = k.reshape(shape).transpose(0, 2, 1, 3)
            v = v.reshape(shape).transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) * (D // self.heads) ** -0.5
            scores = mx.where(valid[:, None, None, :], scores, mx.array(-mx.inf, dtype=scores.dtype))
            mixed = mx.softmax(scores, axis=-1) @ v
            return self.out_proj(mixed.transpose(0, 2, 1, 3).reshape(B, K, D))

    class DecisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Qwen3Model(_backbone_args(body_config))
            self.shared_prefix = False  # set by the predictor; training keeps the flat reference path
            hidden = body_config["hidden_size"]
            self.norm = nn.LayerNorm(hidden, eps=1e-5)
            self.scalar = nn.Linear(hidden, 1)
            self.set_head = set_head
            if set_head == "attention":
                self.set_project = nn.Linear(hidden + 1, SET_DIM)
                self.set_attention = SetAttention(SET_DIM, SET_HEADS)
                self.set_output = nn.Linear(SET_DIM, 1)

        # --- backbone: flat reference path and shared-prefix path -------------------------
        def _flat_leaves(self, paths, pad_token):
            """One padded forward over complete paths; returns the last-real-token state per path."""
            lengths = np.array([len(ids) for ids in paths], dtype=np.int32)
            width = int(lengths.max())
            tokens = np.full((len(paths), width), pad_token, dtype=np.int32)
            for i, ids in enumerate(paths):
                tokens[i, :len(ids)] = ids
            # Right padding + causal mask: a real position never attends to padding,
            # so the last-real-token state equals the torch attention_mask result.
            hidden = self.backbone(mx.array(tokens))
            return hidden[mx.array(np.arange(len(paths))), mx.array(lengths - 1)]

        def _attention_qkv(self, attn, x, offset):
            B, L, _ = x.shape
            q = attn.q_norm(attn.q_proj(x).reshape(B, L, attn.n_heads, -1)).transpose(0, 2, 1, 3)
            k = attn.k_norm(attn.k_proj(x).reshape(B, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = attn.v_proj(x).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            return attn.rope(q, offset=offset), attn.rope(k, offset=offset), v

        def _shared_leaves(self, paths, prefix_length, pad_token):
            """Encode the common prefix once, then all suffixes as one batch against its K/V.

            Causal attention makes prefix states independent of what follows, so this is
            exactly the flat computation with the prefix work done once instead of per path.
            The suffix batch uses SDPA's lower-right-aligned "causal" mask: query i sees the
            whole prefix plus suffix positions <= i, i.e. absolute positions <= P + i."""
            backbone = self.backbone
            prefix = mx.array(np.array([paths[0][:prefix_length]], dtype=np.int32))
            h = backbone.embed_tokens(prefix)
            cache = []
            for layer in backbone.layers:
                attn = layer.self_attn
                q, k, v = self._attention_qkv(attn, layer.input_layernorm(h), 0)
                cache.append((k, v))
                o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask="causal")
                h = h + attn.o_proj(o.transpose(0, 2, 1, 3).reshape(1, prefix_length, -1))
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            suffix_lengths = np.array([len(ids) - prefix_length for ids in paths], dtype=np.int32)
            width = int(suffix_lengths.max())
            tokens = np.full((len(paths), width), pad_token, dtype=np.int32)
            for i, ids in enumerate(paths):
                tokens[i, :len(ids) - prefix_length] = ids[prefix_length:]
            h = backbone.embed_tokens(mx.array(tokens))
            n = len(paths)
            for layer, (pk, pv) in zip(backbone.layers, cache):
                attn = layer.self_attn
                q, k, v = self._attention_qkv(attn, layer.input_layernorm(h), prefix_length)
                keys = mx.concatenate([mx.broadcast_to(pk, (n,) + pk.shape[1:]), k], axis=2)
                values = mx.concatenate([mx.broadcast_to(pv, (n,) + pv.shape[1:]), v], axis=2)
                o = mx.fast.scaled_dot_product_attention(q, keys, values, scale=attn.scale, mask="causal")
                h = h + attn.o_proj(o.transpose(0, 2, 1, 3).reshape(n, width, -1))
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            hidden = backbone.norm(h)
            return hidden[mx.array(np.arange(n)), mx.array(suffix_lengths - 1)]

        def leaf_states(self, examples, pad_token):
            """Last-token backbone states for every candidate path, in request order."""
            if not self.shared_prefix:
                return self._flat_leaves([ids for ex in examples for ids in ex["leaf_tokens"]], pad_token)
            leaves = []
            start = 0
            while start < len(examples):  # paths of one state are contiguous
                end = start
                while end < len(examples) and examples[end]["state_id"] == examples[start]["state_id"]:
                    end += 1
                paths = [ids for ex in examples[start:end] for ids in ex["leaf_tokens"]]
                shared = shared_prefix_length(paths)
                if len(paths) == 1 or shared == 0:
                    leaves.append(self._flat_leaves(paths, pad_token))
                else:
                    leaves.append(self._shared_leaves(paths, shared, pad_token))
                start = end
            return leaves[0] if len(leaves) == 1 else mx.concatenate(leaves, axis=0)

        def __call__(self, examples, pad_token):
            paths = [ids for ex in examples for ids in ex["leaf_tokens"]]
            leaves = self.leaf_states(examples, pad_token)
            leaves = mx.concatenate([leaves, mx.zeros((1, leaves.shape[-1]), dtype=leaves.dtype)])
            kmax = max(len(ex["candidate_ids"]) for ex in examples)
            gather = np.full((len(examples), kmax), len(paths), dtype=np.int32)  # -> zero row
            valid = np.zeros((len(examples), kmax), dtype=bool)
            offset = 0
            for i, ex in enumerate(examples):
                n = len(ex["leaf_tokens"])
                gather[i, :n] = np.arange(offset, offset + n)
                valid[i, :len(ex["candidate_ids"])] = True
                offset += n
            valid = mx.array(valid)
            h = self.norm(leaves[mx.array(gather)].astype(mx.float32))
            z = self.scalar(h).squeeze(-1)
            choice = [i for i, ex in enumerate(examples) if ex["type"] == "choice"]
            if self.set_head == "attention" and choice:
                choice = mx.array(np.array(choice, dtype=np.int32))
                hc, vc = h[choice], valid[choice]
                log_k = mx.log(vc.sum(-1).astype(mx.float32))[:, None, None]
                log_k = mx.broadcast_to(log_k, (hc.shape[0], kmax, 1))
                u = self.set_project(mx.concatenate([hc, log_k], axis=-1))
                delta = self.set_output(mx.tanh(u + self.set_attention(u, vc))).squeeze(-1)
                z = z.at[choice].add(delta)
            # Boolean: one semantic path, logits [0, z].
            is_bool = mx.array(np.array([ex["type"] == "boolean" for ex in examples]))
            boolean = mx.concatenate([mx.zeros((z.shape[0], 1)), z[:, :1],
                                      mx.zeros((z.shape[0], kmax - 2))], axis=1)
            out = mx.where(is_bool[:, None], boolean, z)
            return mx.where(valid, out, mx.array(-1e9)), valid

    return DecisionModel()


def length_bucketed_batches(examples, max_tokens):
    """Sort complete questions by path length and pack them under a padded-token budget.

    A request mixes short and long states; one padded forward would pay the longest
    length for every path. Bucketing keeps padding waste bounded, which is faster per
    decision than either one big batch or one forward per question."""
    order = sorted(examples, key=lambda ex: max(map(len, ex["leaf_tokens"])))
    batches, group, paths, width = [], [], 0, 0
    for ex in order:
        n, size = len(ex["leaf_tokens"]), max(map(len, ex["leaf_tokens"]))
        if group and (paths + n) * max(width, size) > max_tokens:
            batches.append(group)
            group, paths, width = [], 0, 0
        group.append(ex)
        paths += n
        width = max(width, size)
    if group:
        batches.append(group)
    return batches


def shared_prefix_length(paths):
    """Longest common token prefix over a group of paths, leaving every path >= 1 suffix token."""
    shortest = min(map(len, paths))
    first = paths[0]
    common = 0
    while common < shortest - 1 and all(ids[common] == first[common] for ids in paths):
        common += 1
    return common


def torch_key_to_mlx(key):
    if key == "set_attention.in_proj_weight":
        return "set_attention.in_proj.weight"
    if key == "set_attention.in_proj_bias":
        return "set_attention.in_proj.bias"
    return key


def load_decision_weights(model, weights_path, dtype):
    """Load the torch state dict layout into the MLX model; every key must match exactly."""
    mx, _ = _mx()
    from mlx.utils import tree_flatten
    from safetensors import safe_open
    expected = {k for k, _ in tree_flatten(model.parameters())}
    converted = {}
    # One tensor at a time: the float32 file is never fully resident, only the converted copy is.
    with safe_open(str(weights_path), framework="np") as handle:
        for key in handle.keys():
            target = torch_key_to_mlx(key)
            if target not in expected:
                raise ValueError(f"checkpoint tensor has no MLX parameter: {key}")
            value = mx.array(handle.get_tensor(key))
            converted[target] = value.astype(dtype) if target.startswith("backbone.") else value.astype(mx.float32)
            mx.eval(converted[target])
            del value
    missing = expected - set(converted)
    if missing:
        raise ValueError(f"checkpoint is missing parameters: {sorted(missing)[:5]}")
    model.load_weights(list(converted.items()), strict=True)
    mx.eval(model.parameters())
    mx.clear_cache()
    return len(converted)


class MLXDecisionPredictor:
    """Persistent MLX inference object: weights load once, each predict scores complete questions."""

    def __init__(self, checkpoint_dir, max_length=None, precision="bf16", quantize=None, max_batch_tokens=8192,
                 shared_prefix=True):
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision 必须为 fp32 或 bf16")
        if quantize not in {None, 0, 4, 8}:
            raise ValueError("quantize 必须为 None/0/4/8")
        root, paths = local_checkpoint_files(checkpoint_dir)
        run_config = read_json(paths["run_config"])
        if not isinstance(run_config, dict) or run_config.get("set_head") not in {"none", "attention"}:
            raise ValueError("checkpoint config 缺少合法 set_head")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        mx, _ = _mx()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(paths["tokenizer"]), local_files_only=True,
                                                 trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        body_config = read_json(paths["body_config"] / "config.json")
        limit = run_config.get("max_length", 512) if max_length is None else max_length
        if type(limit) is not int or limit <= 0:
            raise ValueError("max-length 必须为正整数")
        context_limit = body_config.get("max_position_embeddings")
        if isinstance(context_limit, int) and limit > context_limit:
            raise ValueError("max-length 超过backbone配置声明的上下文长度")
        dtype = mx.bfloat16 if precision == "bf16" else mx.float32
        limit_mlx_memory()
        model = build_decision_model(body_config, run_config["set_head"])
        self.tensor_count = load_decision_weights(model, paths["weights"], dtype)
        self.quantization = quantize_backbone(model, quantize) if quantize else None
        model.shared_prefix = bool(shared_prefix)
        model.eval()
        self.model, self.tokenizer, self.root = model, tokenizer, root
        self.run_config, self.limit, self.precision = run_config, limit, precision
        self.device = f"mlx:{mx.default_device().type.name}"
        self.max_batch_tokens = max_batch_tokens
        self.inference_calls = 0
        self._mx = mx

    def predict(self, payload, batch_questions=0, temperature=1.0):
        states = validate_request(payload)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature 必须为有限正数")
        mx = self._mx
        examples = prepare_examples(payload, self.tokenizer, self.limit)
        batches = (length_bucketed_batches(examples, self.max_batch_tokens) if batch_questions == 0
                   else complete_question_batches(examples, batch_questions))
        self.inference_calls += 1
        outputs = {state["id"]: {"id": state["id"], "answers": {}} for state in states}
        for batch in batches:
            logits, _ = self.model(batch, self.tokenizer.pad_token_id)
            mx.eval(logits)
            for example, values in zip(batch, logits):
                k = len(example["candidate_ids"])
                scores = values[:k].astype(mx.float32)
                if not bool(mx.isfinite(scores).all()):
                    raise ValueError("模型产生非有限logits，未返回部分预测")
                probabilities = mx.softmax(scores / temperature, axis=-1).tolist()
                outputs[example["state_id"]]["answers"][example["qid"]] = answer_from_probabilities(example, probabilities)
        return {
            "schema_version": "openjev-toy-inference-v1",
            "checkpoint": {"directory": str(self.root), "base_model": self.run_config.get("model"),
                           "base_revision": self.run_config.get("resolved_model_revision"),
                           "set_head": self.run_config["set_head"]},
            "temperature": {"value": float(temperature), "fitted_by_this_command": False,
                            "note": "显式应用给定标量；默认1不表示模型已校准。"},
            "execution": {"device": self.device, "backend": "mlx",
                          "parameter_storage": "bfloat16" if self.precision == "bf16" else "float32",
                          "quantization": self.quantization,
                          "peak_memory_gb": self._mx.get_peak_memory() / 1e9,
                          "precision": self.precision,
                          "forward_autocast": "native bfloat16 backbone, float32 heads" if self.precision == "bf16" else "disabled",
                          "states": len(states), "questions": len(examples),
                          "candidate_paths": sum(len(ex["leaf_tokens"]) for ex in examples),
                          "forward_passes": len(batches),
                          "batch_questions_limit": batch_questions or f"length-bucketed, {self.max_batch_tokens} padded tokens",
                          "autoregressive_decode_steps": 0, "prefix_sharing": self.model.shared_prefix,
                          "max_length": self.limit, "disable_native_triton": None,
                          "network_model_calls": 0, "persistent_model_load_count": 1,
                          "inference_call_index": self.inference_calls},
            "states": list(outputs.values()),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input", required=True, help="含states数组的JSON文件")
    parser.add_argument("--output", help="不设置时将完整结果输出到stdout")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-questions", type=int, default=0)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--quantize", type=int, choices=[4, 8], help="Quantize the backbone to 4 or 8 bits (group size 64)")
    parser.add_argument("--no-shared-prefix", dest="shared_prefix", action="store_false",
                        help="Encode every candidate path in full (reference behaviour) instead of sharing the state prefix")
    args = parser.parse_args()
    try:
        payload = read_json(args.input)
        validate_request(payload)
        started = time.perf_counter()
        engine = MLXDecisionPredictor(args.checkpoint_dir, max_length=args.max_length, precision=args.precision,
                                      quantize=args.quantize, shared_prefix=args.shared_prefix)
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result = engine.predict(payload, batch_questions=args.batch_questions, temperature=args.temperature)
        result["execution"]["model_load_seconds"] = load_seconds
        result["execution"]["evaluation_seconds"] = time.perf_counter() - started
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            print(json.dumps({"output": str(destination), "execution": result["execution"]}, ensure_ascii=False))
        else:
            print(text, end="")
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
