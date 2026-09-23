# NanoJev natural-context head experiment (2026-09-23)

This evaluation-only experiment tested whether training on three preceding meeting turns reduces unsolicited assist or calendar triggers. It does not change the production launcher or backend policy.

The generator `prepare_public_context_dataset.py` reused the 200 AISHELL-4 utterances previously selected and manually reviewed as non-assistant, non-calendar speech. It retained the original synthetic train/dev examples. Unlike the earlier random utterance split, the natural samples are separated by entire recording: `L_R004S03C01` contributes 100 training negatives; `L_R003S02C02` contributes 100 development negatives. Each natural sample includes the three immediately preceding nonempty TextGrid turns and their speaker IDs. The resulting ignored dataset is `.runtime/intent/zh-intent-v4-natural-context-recording-split/` (340 train, 180 dev). The independent `L_R003S01C02` clip and three fresh meetings were excluded from both splits.

`train_head.py` started from `nanojev-intent-zh-lora-head-v2`, froze the native MLX encoder, and selected the scalar head at epoch 60 by development BCE. The exported v4 weights are `models/nanojev-intent-zh-natural-context-head-v4/best.safetensors`, SHA-256 `404f362aa8e7018fcd0b9d4e47a0fc857fdcff31f1e503a23979905f332886a1`. At the unchanged 0.85 threshold, the 180-case development split gave assist TP 14/24 with 1/156 false positives and schedule TP 22/24 with 2/156 false positives. This dev set differs from v3's, so its aggregate scores are not a direct comparison.

| Frozen evaluation | v4 assist | v4 schedule | v3 assist | v3 schedule |
| --- | ---: | ---: | ---: | ---: |
| Previous 52 synthetic cases: true positives / expected, false positives | 5/12, 0 | 8/14, 0 | 5/12, 0 | 9/14, 0 |
| Prior 9 natural negatives: false triggers | 1/9 | 3/9 | 1/9 | 3/9 |
| Three new meetings, 43 natural negatives: false triggers | 4/43 | 11/43 | 5/43 | 14/43 |

Each model was scored once on the frozen natural inputs with three preceding turns; the 52-case v4 evaluation was repeated and gave identical scores. Predictions were written before consulting the corresponding frozen labels. The 43-case bundle contains no positive intent examples, so it cannot measure natural positive recall. The separate frozen 48-case synthetic contrast suite exposed a severe v3 domain gap: assist TP 2/12 and FP 18/36; schedule TP 12/12 and FP 21/36. V4 was not rerun on that suite after the model choice.

The v4 head did not materially fix natural false triggers and slightly reduced recall on the old 52-case suite. It was **not selected for production**. Any enablement decision needs an independently evaluated execution gate and clear UI wording that suggestions may be missed or incorrect.

Ignored prediction files are `.runtime/intent/v4-old52.json`, `.runtime/intent-holdout/v4-natural-context-old9-predictions.json`, the `v3-` and `v4-` prediction files for `L_R003S03C02`, `M_R003S04C01`, and `S_R003S02C01`, and `.runtime/intent-holdout/v3-synthetic-contrast-v1-predictions.json`. The full training report is `nanojev-intent-zh-natural-context-head-v4-training.json` in this directory. Source transcripts and model weights remain local and ignored by Git.
