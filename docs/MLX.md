# Running NanoJev on Apple Silicon with MLX

NanoJev's released checkpoint runs natively on Apple Silicon through
[MLX](https://github.com/ml-explore/mlx). The MLX backend loads the same
`best.safetensors`, keeps the same request/response contract, and serves the same
browser demos; the CUDA path is untouched. Verified on an Apple M5 with 16 GB of
unified memory, macOS 26, Python 3.12, `mlx 0.32.2`, `mlx-lm 0.31.3`.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mlx.txt
```

`requirements-mlx.txt` has no torch. The tokenizer comes from `transformers`,
weights are read with `safetensors`, and the Qwen3 blocks come from `mlx-lm`.

## Inference and the local service

```bash
python scripts/serve_decisions.py \
  --checkpoint-dir checkpoints/NanoJev-unified \
  --web-root web --port 8765 --quantize 8
```

`--device auto` (the default everywhere) picks `cuda:0` when a CUDA device exists and
`mlx` otherwise, so `serve_decisions.py`, `predict_toy_decisions.py`,
`evaluate_game_policy.py` and the other `DecisionPredictor` callers need no changes.
`--device mlx` forces the backend; `--disable-native-triton` is accepted and ignored.

Memory and agreement with the CUDA service, measured on the released checkpoint over
96 recorded test decisions (24 each: Maze, Snake, ViZDoom Basic, Predict Position) that
`scripts/verify_mlx_against_cuda.py` rebuilds from `evaluation/experiment/selected_test.jsonl`
and re-scores locally (`results/mlx/cuda_agreement.json`):

| Mode | Resident | Inference peak | Max ∣Δp∣ vs. the CUDA run | Argmax agreement |
|---|---:|---:|---:|---:|
| `--precision bf16` (default) | 1.19 GB | 1.91 GB | 0.011 | 95/96 |
| `--quantize 8` | 0.63 GB | 1.69 GB | 0.012 | 96/96 |
| `--quantize 4` | 0.37 GB | 1.45 GB | 0.092 | 95/96 |

`--quantize 8` (affine, group size 64) is effectively lossless and is the recommended
low-memory setting. `--quantize 4` uses group size 32 and still shifts probabilities by
up to 0.09; use it only when memory matters more than calibration. The bf16 and 4-bit
disagreements are the same Predict Position decision, which the CUDA service scored as an exact tie (left 0.313 vs. right 0.313).

The backend caps MLX's buffer cache at 512 MB and its memory guideline at 60 % of
physical memory, so a long-running service does not accumulate freed buffers.

### Numerical fidelity

With `--precision fp32` on the CPU device the MLX backbone matches the torch
reference to a relative 1.6e-7. On the GPU device Metal kernels accumulate in a
different order, giving ~1e-4 relative deviation in hidden states and ≤0.01 in
logits; the bf16 path deviates about as much as bf16 autocast does on CUDA.
Right padding plus a causal mask is used instead of an explicit attention mask:
a real token never attends to later padding, so the last-real-token state is
identical to the masked torch computation.

### Decision latency

`scripts/benchmark_mlx_latency.py` times `predict()` end to end (tokenization, tensor
assembly, backbone, heads, softmax) on recorded test decisions, 24 per task. Apple M5,
released checkpoint, bf16 (`results/mlx/latency_bf16.json`; 8-bit in `latency_q8.json`
is 4–9 % slower and peaks at 1.7 GB instead of 1.9 GB):

| Task | Candidates | Tokens per path | 1 decision p50 / p95 | 8 decisions per request |
|---|---:|---:|---:|---:|
| ViZDoom Basic | 4.0 | 334 | 123 / 132 ms | 125 ms per decision |
| Snake | 3.0 | 521 | 156 / 159 ms | 164 ms per decision |
| ViZDoom Predict Position | 4.0 | 776 | 254 / 487 ms | 313 ms per decision |
| Maze (8×8 test set) | 2.8 | 1364 | 283 / 684 ms | 388 ms per decision |

Latency is proportional to padded tokens: the backbone prefills about 13k tokens/s, and
every candidate repeats the full state prefix (no prefix sharing yet, as in the CUDA
reference). Within a request `MLXDecisionPredictor` sorts questions by path length and
packs them under an 8192 padded-token budget (`max_batch_tokens`), so mixed-length
batches no longer pay the longest state for every path; before this, 8 Maze decisions
per request cost 720 ms each instead of 388 ms. Shared-prefix scoring (one prefix pass,
KV cache reused per candidate) would cut Maze and Predict Position cost by roughly the
candidate count and is the natural next step.

## Fine-tuning with QLoRA

`scripts/train_unified_games_mlx.py` ports the SFT stage
(`train_unified_games.py --stage sft --loss ce`): same dataset validation, stratified
sampler, population weights, CE objective, dev-based selection and output bundle.
To fit in 16 GB it changes how the model is adapted, not what is optimized:

- the backbone is quantized to 8 bits (`--quantize 8`, or 4 / 0) and frozen;
- LoRA factors (rank 8, scale 20) on `q/k/v/o_proj` of every block and all decision
  heads are trained: 2.5 M trainable parameters instead of 600 M;
- every transformer block is gradient-checkpointed;
- each forward has a padded-token budget (`--max-microbatch-tokens 4096`);
- a preflight forward/backward on the widest training microbatch runs before any
  file is written, and the run aborts if peak memory exceeds `--memory-fraction`
  (60 %) of physical memory instead of swapping.

```bash
python scripts/train_unified_games_mlx.py \
  --input data/NanoJev-unified/unified/hard \
  --init-checkpoint checkpoints/NanoJev-unified \
  --policy-pool-weights configs/sonic_policy_pool_weights.json \
  --output-dir runs/mlx_qlora \
  --steps 300 --eval-every 50 --batch-questions 12 --microbatch-questions 4
```

Measured on the M5 with the defaults: 2.0 GB peak, about 10 s per 8-question
update. Full fine-tuning (`--full-backbone --quantize 0`) is still available; on this
machine it peaks above 14 GB and thrashes, so it is only sensible with 32 GB or more.

The best dev checkpoint is kept as `adapters.safetensors` (LoRA factors plus heads,
~10 MB). When training ends the selected adapters are fused into dequantized weights
and exported as a dense float32 `best.safetensors` in the torch key layout, and the
final dev/calibration/test/OOD metrics are computed by loading that exported bundle
from disk, so the reported numbers describe exactly what inference will load. Note
that the exported base weights carry the 8-bit quantization error (~0.7 % relative);
train with `--quantize 0` if the base must stay exact.

Not ported: the critic stage (`brier`, `paired_brier_pg`, TD targets). Use the CUDA
trainer for it; its output loads in the MLX backend unchanged.

## Files

| File | Role |
|---|---|
| `scripts/mlx_decisions.py` | MLX `DecisionModel`, weight loading, quantization, `MLXDecisionPredictor`, CLI |
| `scripts/predict_toy_decisions.py` | `DecisionPredictor` gains `device_name="auto" \| "mlx"` and `quantize` |
| `scripts/serve_decisions.py` | `--device`, `--quantize` |
| `scripts/train_unified_games_mlx.py` | QLoRA SFT trainer |
| `scripts/test_mlx_decisions.py` | Structure tests on a tiny random backbone |
| `scripts/benchmark_mlx_latency.py` | Per-task decision latency (p50/p95, batch scaling) |
| `scripts/verify_mlx_against_cuda.py` | Re-scores recorded CUDA decisions per mode; agreement and memory |
| `results/mlx/latency_*.json`, `cuda_agreement.json` | Recorded latency and agreement runs on the M5 |
| `requirements-mlx.txt` | Pinned MLX stack |
