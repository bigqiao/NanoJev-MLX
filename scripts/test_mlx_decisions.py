#!/usr/bin/env python3
"""MLX backend structure tests on a tiny random backbone; no checkpoint, no torch."""
import unittest

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - CUDA hosts skip the MLX suite
    mx = None

TINY = {"model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6,
        "vocab_size": 128, "max_position_embeddings": 512, "tie_word_embeddings": True,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"}}


def fake_examples():
    return [
        {"id": "s:b", "state_id": "s", "qid": "b", "type": "boolean", "candidate_ids": ["false", "true"],
         "candidate_texts": ["The proposition is true."], "leaf_tokens": [[5, 6, 7, 1]]},
        {"id": "s:c", "state_id": "s", "qid": "c", "type": "choice", "candidate_ids": ["x", "y", "z"],
         "candidate_texts": ["x", "y", "z"], "leaf_tokens": [[5, 6, 8, 1], [5, 6, 9, 10, 1], [5, 6, 11, 1]]},
        {"id": "s:r", "state_id": "s", "qid": "r", "type": "score", "candidate_ids": ["0", "1"],
         "candidate_texts": ["low", "high"], "leaf_tokens": [[5, 12, 1], [5, 13, 14, 15, 16, 1]]},
    ]


@unittest.skipIf(mx is None, "mlx is not installed")
class MLXDecisionModelTest(unittest.TestCase):
    def setUp(self):
        from mlx_decisions import build_decision_model
        mx.random.seed(0)
        self.model = build_decision_model(TINY, "attention")

    def test_backbone_args_accept_transformers_5_rope_layout(self):
        from mlx_decisions import _backbone_args
        args = _backbone_args(TINY)
        self.assertEqual(args.rope_theta, 1000000)
        self.assertEqual(args.head_dim, 16)
        self.assertIsNone(args.rope_scaling)

    def test_torch_key_mapping_round_trip(self):
        from mlx.utils import tree_flatten
        from mlx_decisions import torch_key_to_mlx
        from train_unified_games_mlx import TORCH_KEY
        mlx_keys = {k for k, _ in tree_flatten(self.model.parameters())}
        torch_keys = {TORCH_KEY.get(k, k) for k in mlx_keys}
        self.assertEqual({torch_key_to_mlx(k) for k in torch_keys}, mlx_keys)
        self.assertIn("set_attention.in_proj_weight", torch_keys)
        self.assertNotIn("set_attention.in_proj.weight", torch_keys)

    def test_forward_layout_and_masking(self):
        logits, valid = self.model(fake_examples(), pad_token=0)
        mx.eval(logits, valid)
        self.assertEqual(logits.shape, (3, 3))
        self.assertEqual(valid.tolist(), [[True, True, False], [True, True, True], [True, True, False]])
        rows = logits.tolist()
        self.assertEqual(rows[0][0], 0.0)  # Boolean logits are [0, z]
        self.assertEqual(rows[0][2], -1e9)
        self.assertEqual(rows[2][2], -1e9)
        self.assertTrue(all(abs(v) < 1e9 for row in rows for v in row[:2]))
        probabilities = mx.softmax(logits[1], axis=-1).tolist()
        self.assertAlmostEqual(sum(probabilities), 1.0, places=5)

    def test_right_padding_does_not_change_scores(self):
        examples = fake_examples()
        base, _ = self.model([examples[1]], pad_token=0)
        # Adding a much longer path to the batch widens padding for the others but must not change them.
        longer = {**examples[2], "leaf_tokens": [[5, 12, 1], [5] + [13] * 40 + [1]]}
        both, _ = self.model([examples[1], longer], pad_token=0)
        mx.eval(base, both)
        self.assertTrue(mx.allclose(base[0], both[0], atol=1e-4, rtol=1e-4).item())

    def test_shared_prefix_matches_flat_reference(self):
        from mlx_decisions import shared_prefix_length
        examples = fake_examples()
        # Two states: the first has three questions sharing one state prefix, the second one question.
        second = [{**ex, "id": "t:" + ex["qid"], "state_id": "t",
                   "leaf_tokens": [[20, 21] + ids[2:] for ids in ex["leaf_tokens"]]} for ex in examples[1:]]
        batch = examples + second
        self.assertEqual(shared_prefix_length([[5, 6, 7, 1], [5, 6, 8, 1]]), 2)
        self.assertEqual(shared_prefix_length([[5, 6, 1], [5, 6, 1, 1]]), 2)  # keeps one suffix token
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)  # deterministic accumulation order: the two paths must agree exactly
        try:
            self.model.shared_prefix = False
            flat, valid_flat = self.model(batch, pad_token=0)
            self.model.shared_prefix = True
            shared, valid_shared = self.model(batch, pad_token=0)
            mx.eval(flat, shared)
        finally:
            mx.set_default_device(previous)
        self.assertEqual(valid_flat.tolist(), valid_shared.tolist())
        self.assertTrue(mx.allclose(flat, shared, atol=1e-5, rtol=1e-5).item())

    def test_quantized_backbone_still_scores(self):
        from mlx_decisions import quantize_backbone
        report = quantize_backbone(self.model, 8)
        self.assertEqual(report["bits"], 8)
        logits, _ = self.model(fake_examples(), pad_token=0)
        mx.eval(logits)
        self.assertTrue(mx.isfinite(logits[:, :2]).all().item())


if __name__ == "__main__":
    unittest.main()
