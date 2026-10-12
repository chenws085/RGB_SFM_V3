"""Verify intervention fidelity and hook cleanup on an actual small MergingViT."""
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.MergingViT import MergingViT
from scripts.concept_intervention import capture, intervene, parse_group


class InterventionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.model = MergingViT(img_size=16, patch_size=4, num_classes=3,
                                embed_dims=[8, 16], num_heads=[2, 2], depths=[1, 1],
                                drop_rate=0, drop_path_rate=0).eval()
        self.images = torch.randn(2, 3, 16, 16)

    def test_empty_and_identity_interventions_preserve_logits(self):
        expected, features = capture(self.model, self.images, 0)
        mask = torch.zeros(features.shape[:2], dtype=torch.bool)
        torch.testing.assert_close(intervene(self.model, self.images, 0, mask, torch.zeros_like(features)), expected)
        mask[:] = True
        torch.testing.assert_close(intervene(self.model, self.images, 0, mask, features), expected)
        self.assertFalse(self.model.stages[0][-1]._forward_hooks)

    def test_masked_intervention_matches_independent_forward(self):
        expected, features = capture(self.model, self.images, 0)
        mask = torch.zeros(features.shape[:2], dtype=torch.bool)
        mask[0, 2] = True
        mask[1, [1, 4]] = True
        result = intervene(self.model, self.images, 0, mask, torch.zeros_like(features))
        x = features.clone()
        x[mask] = 0
        x, height, width = self.model.merges[0](x, 4, 4)
        x = x + self.model.pos_embeds[1]
        for block in self.model.stages[1]:
            x = block(x)
        manual = self.model.head(self.model.head_drop(self.model.norm_final(x.mean(1))))
        torch.testing.assert_close(result, manual)
        self.assertGreater(float((result - expected).abs().max()), 1e-6)
        torch.testing.assert_close(self.model(self.images), expected)

    def test_hooks_removed_after_failure(self):
        with self.assertRaises(ValueError):
            intervene(self.model, self.images, 0, torch.zeros(2, 3, dtype=torch.bool), torch.zeros(8))
        self.assertFalse(self.model.stages[0][-1]._forward_hooks)

    def test_group_parsing(self):
        self.assertEqual(parse_group('2:0+4:1'), ((2, 0), (4, 1)))
        with self.assertRaises(Exception):
            parse_group('2:0+2:1')


if __name__ == '__main__':
    unittest.main()
