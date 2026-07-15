from __future__ import annotations

import unittest

import torch

from src.court.courtkeynet_model import COURTKEYNET_MODEL_CONFIG, CourtKeyNet


class CourtKeyNetArchitectureTest(unittest.TestCase):
    def test_forward_returns_official_inference_contract(self) -> None:
        model = CourtKeyNet(COURTKEYNET_MODEL_CONFIG)
        model.eval()

        with torch.no_grad():
            outputs = model(torch.zeros((1, 3, 64, 64)))

        self.assertEqual(
            set(outputs),
            {"heatmaps", "kpts_init", "kpts_refined", "offsets"},
        )
        self.assertEqual(tuple(outputs["heatmaps"].shape), (1, 4, 32, 32))
        self.assertEqual(tuple(outputs["kpts_init"].shape), (1, 4, 2))
        self.assertEqual(tuple(outputs["kpts_refined"].shape), (1, 4, 2))
        self.assertEqual(tuple(outputs["offsets"].shape), (1, 4, 2))
        for tensor in outputs.values():
            self.assertTrue(torch.isfinite(tensor).all().item())


if __name__ == "__main__":
    unittest.main()
