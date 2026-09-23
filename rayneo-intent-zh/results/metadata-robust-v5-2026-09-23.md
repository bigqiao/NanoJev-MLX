# NanoJev production-metadata robustness experiment (2026-09-23)

The v3 intent head was sensitive to the metadata shape sent by the real backend. Its earlier evaluations supplied a fixed speaker string and an ISO timestamp, while the backend sends `speakerId` as null or a stable ID and `recordedAt` as null or a Unix millisecond number. This bounded experiment trained a new scalar head against those four production-shaped combinations. It did not change the backend gate, thresholds, NanoJev source, or the managed intent service.

The dataset starts from the recording-level train/dev split documented in `natural-context-v4-2026-09-23.md`. `prepare_metadata_robust_dataset.py` expands every case into `null/null`, stable-speaker/null, null/millisecond, and stable-speaker/millisecond states, carrying the same shape through the previous-turn speaker IDs. It yields 1,360 train and 720 dev states while keeping the underlying 340/180 utterances and entire-meeting split. The native MLX encoder comes from `nanojev-intent-zh-lora-head-v2` and remains frozen. Only the scalar head was trained, selected by dev BCE at epoch 145. Exported checkpoint: `models/nanojev-intent-zh-metadata-robust-head-v5`; `best.safetensors` SHA-256 `87909e384605bbced586af58dc3a198886aebc30c712c2e47a0ea62079e706fd`. The full report is `nanojev-intent-zh-metadata-robust-head-v5-training.json` here.

All figures below are from the exported NanoJev `/api/evaluate` MLX service, then the production JavaScript `proactiveEvidence` gate. The assist threshold is 0.30 and the schedule threshold is 0.85. Explicit commands have their separate backend route and are excluded from these classifier-and-gate counts. No threshold, prompt, dataset label, or gate rule was changed based on the holdouts.

| Frozen set and metadata shape | v5 assist TP, FP | v5 schedule TP, FP |
| --- | ---: | ---: |
| 48 independent synthetic contrasts, null/null | 9/12, 2/36 | 8/12, 0/36 |
| 48 contrasts, null/millisecond | 8/12, 2/36 | 6/12, 0/36 |
| 48 contrasts, stable/millisecond | 8/12, 2/36 | 8/12, 0/36 |
| Prior 52 synthetic cases, null/null | 8/12, 0/40 | 7/14, 0/38 |
| Prior 52, null/millisecond | 10/12, 1/40 | 7/14, 0/38 |

On the same 48-case contrast set, the previously observed v3 null/null gate result was assist 0/12 with no false positives and schedule 3/12 with no false positives. V5 recovers substantial recall, though two near-miss human-conversation sentences still pass the assist gate: `assist-07-near-miss` (people discussing the outside temperature) and `assist-08-near-miss` (a teacher explaining a camera term). Their v5 null/null assist scores are 0.653 and 0.853. The stable-ID and timestamp variants remain sensitive, but less catastrophically than v3.

An independent 40-utterance AISHELL-4 holdout was selected because the frozen text gate passes the utterances; all were manually labeled negative before scoring. V5 produced **zero** assist or schedule candidates in the stable-speaker/millisecond and null/null shapes. In the null/millisecond shape, v5 produced one assist candidate, `gate-natural-26`, score 0.387 (`或者是主要就还是它是以哪个年龄段儿的，咱这个有统计吗，是`), and no schedule candidate. V3 produced three assist candidates after the gate in the same null/millisecond shape. V3 also gave one schedule model score above 0.85 (0.888), but the **production JS schedule gate rejected it**, so it is **not** a final schedule false positive.

Two other frozen AISHELL-4 bundles contain 43 and 39 natural negative utterances. Replayed with eight preceding annotated turns and either null/null or null/millisecond metadata, v5 yielded **zero final assist and zero final schedule triggers** in each 82-state shape. Some raw model scores crossed a threshold, but the unchanged JS gate rejected them. This is a background negative-control replay, not a natural-positive recall test.

The saved ignored evidence is `.runtime/intent-holdout/v5-synthetic-contrast-v1-production-shapes-{raw,gated}.json`, `v5-old52-production-shapes-{raw,gated}.json`, the `v5-` and `v3-gate-conditioned-hard40-*` outputs, and `v5-natural82-production-null-shapes-raw.json`. The frozen hard40 input SHA-256 is `6669abb5bee4f51c319067d03a341d0ff626e61d055b9d73ca429b851daf5a55`. The managed v3 service was stopped by the root integration task while a background v3 comparison was running, so no complete v3 comparison for the 82-state replay is claimed.

These tests show v5 materially improves production-shaped synthetic positive recall while keeping the observed natural gate-positive false-trigger count low. They do **not** establish production calibration: the natural holdouts contain no positive assistant requests or confirmed personal events, ASR errors can change the gate input, and the 40 hard negatives span a limited public meeting domain. Those limitations should remain visible if v5 is enabled as an optional proactive feature.
