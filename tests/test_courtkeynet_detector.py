from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch

from src.court import courtkeynet_detector as courtkeynet_detector_module
from src.court.courtkeynet_detector import (
    CourtKeyNetConfig,
    CourtKeyNetLineDetector,
    combined_courtkeynet_confidence,
    geometric_courtkeynet_confidence,
    resolve_courtkeynet_weights,
)
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


class CourtKeyNetConfidenceTest(unittest.TestCase):
    def test_combined_confidence_uses_official_component_weights(self) -> None:
        heatmaps = torch.tensor(
            [[
                [[8.0, 0.0], [0.0, 0.0]],
                [[0.0, 8.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 8.0]],
                [[0.0, 0.0], [8.0, 0.0]],
            ]],
            dtype=torch.float32,
        )
        keypoints = torch.tensor(
            [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.9], [0.2, 0.9]]],
            dtype=torch.float32,
        )

        combined, components = combined_courtkeynet_confidence(heatmaps, keypoints)

        expected = (
            0.4 * components["heatmap"]
            + 0.4 * components["geometry"]
            + 0.2 * components["entropy"]
        )
        torch.testing.assert_close(combined, expected)
        self.assertEqual(tuple(combined.shape), (1,))
        for component in components.values():
            self.assertTrue(torch.isfinite(component).all().item())

    def test_crossed_quad_has_lower_geometric_confidence(self) -> None:
        valid = torch.tensor(
            [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.9], [0.2, 0.9]]],
            dtype=torch.float32,
        )
        crossed = valid[:, [0, 2, 1, 3], :]

        self.assertGreater(
            geometric_courtkeynet_confidence(valid).item(),
            geometric_courtkeynet_confidence(crossed).item(),
        )


class _FakeCourtKeyNet(torch.nn.Module):
    def __init__(self, keypoints: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("keypoints", keypoints)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = inputs.shape[0]
        heatmaps = torch.zeros((batch_size, 4, 4, 4), device=inputs.device)
        heatmaps[:, 0, 0, 0] = 16.0
        heatmaps[:, 1, 0, 3] = 16.0
        heatmaps[:, 2, 3, 3] = 16.0
        heatmaps[:, 3, 3, 0] = 16.0
        return {
            "heatmaps": heatmaps,
            "kpts_refined": self.keypoints.to(inputs.device).expand(batch_size, -1, -1),
        }


class _SequenceCourtKeyNet(_FakeCourtKeyNet):
    def __init__(self, keypoints: list[torch.Tensor]) -> None:
        super().__init__(keypoints[-1])
        self.keypoint_sequence = keypoints
        self.forward_count = 0

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        index = min(self.forward_count, len(self.keypoint_sequence) - 1)
        self.keypoints = self.keypoint_sequence[index]
        self.forward_count += 1
        return super().forward(inputs)


class CourtKeyNetDetectorTest(unittest.TestCase):
    def test_maps_normalized_corners_to_non_square_source_geometry(self) -> None:
        normalized = torch.tensor(
            [[[0.2, 0.25], [0.8, 0.25], [0.85, 0.9], [0.15, 0.9]]],
            dtype=torch.float32,
        )
        detector = CourtKeyNetLineDetector(
            CourtKeyNetConfig(confirmation_frames=1),
            model=_FakeCourtKeyNet(normalized),
        )
        frame = np.zeros((400, 1000, 3), dtype=np.uint8)

        prediction = detector.predict(frame, frame_id=7, timestamp_ms=280, force=True)

        self.assertTrue(prediction.valid, prediction.to_dict())
        self.assertEqual(prediction.source_size, (1000, 400))
        np.testing.assert_allclose(
            prediction.corners,
            [[200.0, 100.0], [800.0, 100.0], [850.0, 360.0], [150.0, 360.0]],
            atol=1e-4,
        )
        self.assertEqual(prediction.scheme, "courtkeynet")
        self.assertIn("doubles_outer", prediction.projected_lines)
        self.assertEqual(
            prediction.metrics["components"]["courtkeynet_confirmation_complete"],
            1.0,
        )

    def test_relative_weight_path_is_resolved_only_from_project_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            expected = project_root / "~" / "weights.safetensors"
            expected.parent.mkdir()
            expected.touch()
            with mock.patch.object(courtkeynet_detector_module, "PROJECT_ROOT", project_root):
                resolved = resolve_courtkeynet_weights("~/weights.safetensors")

        self.assertEqual(resolved, expected.resolve())

    def test_relative_weight_path_cannot_escape_project_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            project_root = temp_root / "project"
            project_root.mkdir()
            outside = temp_root / "outside.safetensors"
            outside.touch()
            with mock.patch.object(courtkeynet_detector_module, "PROJECT_ROOT", project_root):
                with self.assertRaises(FileNotFoundError) as context:
                    resolve_courtkeynet_weights("../outside.safetensors")

        self.assertIn(str(outside.resolve()), str(context.exception))

    def test_malformed_model_output_returns_invalid_prediction(self) -> None:
        class MalformedCourtKeyNet(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
                return {
                    "heatmaps": torch.zeros((1, 4, 4, 4), dtype=torch.int64, device=inputs.device),
                    "kpts_refined": torch.zeros((1, 4, 2), device=inputs.device),
                }

        detector = CourtKeyNetLineDetector(
            CourtKeyNetConfig(confirmation_frames=1),
            model=MalformedCourtKeyNet(),
        )

        prediction = detector.predict(np.zeros((40, 100, 3), dtype=np.uint8), 0, 0, force=True)

        self.assertFalse(prediction.valid)
        self.assertTrue(prediction.attempted)
        self.assertEqual(prediction.rejected_count, 1)

    def test_three_fresh_consistent_frames_lock_trusted_geometry(self) -> None:
        normalized = torch.tensor(
            [[[0.20, 0.25], [0.80, 0.25], [0.85, 0.90], [0.15, 0.90]]],
            dtype=torch.float32,
        )
        model = _SequenceCourtKeyNet([normalized, normalized, normalized])
        detector = CourtKeyNetLineDetector(CourtKeyNetConfig(), model=model)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        first = detector.predict(frame, 0, 0, force=True)
        second = detector.predict(frame, 1, 40, force=True)
        trusted = detector.predict(frame, 2, 80, force=True)
        locked = detector.predict(frame, 3, 120, force=True)

        self.assertEqual(first.status, "courtkeynet confirmation 1/3")
        self.assertEqual(second.status, "courtkeynet confirmation 2/3")
        self.assertEqual(first.update_type, "provisional")
        self.assertEqual(second.update_type, "provisional")
        self.assertFalse(first.valid)
        self.assertFalse(second.valid)
        self.assertTrue(trusted.valid)
        self.assertEqual(
            trusted.metrics["components"]["courtkeynet_confirmation_complete"],
            1.0,
        )
        self.assertEqual(locked.status, "locked trusted calibration")
        self.assertEqual(locked.update_type, "locked trusted calibration")
        self.assertFalse(locked.attempted)
        self.assertFalse(locked.updated)
        self.assertEqual(locked.corners, trusted.corners)
        self.assertEqual(locked.projected_lines, trusted.projected_lines)
        self.assertEqual(model.forward_count, 3)

    def test_confirmation_compares_each_fresh_frame_to_first_anchor(self) -> None:
        base = torch.tensor(
            [[[0.20, 0.25], [0.80, 0.25], [0.85, 0.90], [0.15, 0.90]]],
            dtype=torch.float32,
        )
        shift_three = torch.tensor([3.0 / 160.0, 0.0], dtype=torch.float32)
        shift_six = torch.tensor([6.0 / 160.0, 0.0], dtype=torch.float32)
        detector = CourtKeyNetLineDetector(
            CourtKeyNetConfig(max_corner_shift_ratio=0.035),
            model=_SequenceCourtKeyNet([base, base + shift_three, base + shift_six]),
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        statuses = [
            detector.predict(frame, index, index * 40, force=True).status
            for index in range(3)
        ]

        self.assertEqual(
            statuses,
            [
                "courtkeynet confirmation 1/3",
                "courtkeynet confirmation 2/3",
                "courtkeynet confirmation 1/3",
            ],
        )

    def test_reset_and_rejected_candidates_clear_unfinished_confirmation(self) -> None:
        normalized = torch.tensor(
            [[[0.20, 0.25], [0.80, 0.25], [0.85, 0.90], [0.15, 0.90]]],
            dtype=torch.float32,
        )
        detector = CourtKeyNetLineDetector(
            CourtKeyNetConfig(),
            model=_SequenceCourtKeyNet([normalized, normalized, normalized]),
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        first = detector.predict(frame, 0, 0, force=True)
        detector.reset()
        after_reset = detector.predict(frame, 1, 40, force=True)

        self.assertEqual(first.status, "courtkeynet confirmation 1/3")
        self.assertEqual(after_reset.status, "courtkeynet confirmation 1/3")

        detector.config.confidence_threshold = 1.1
        rejected = detector.predict(frame, 2, 80, force=True)
        detector.config.confidence_threshold = 0.5
        after_rejection = detector.predict(frame, 3, 120, force=True)

        self.assertEqual(rejected.status, "courtkeynet confidence below threshold")
        self.assertEqual(after_rejection.status, "courtkeynet confirmation 1/3")

    def test_malformed_candidate_clears_unfinished_confirmation(self) -> None:
        normalized = torch.tensor(
            [[[0.20, 0.25], [0.80, 0.25], [0.85, 0.90], [0.15, 0.90]]],
            dtype=torch.float32,
        )

        class ValidMalformedValidCourtKeyNet(_FakeCourtKeyNet):
            def __init__(self) -> None:
                super().__init__(normalized)
                self.forward_count = 0

            def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
                self.forward_count += 1
                if self.forward_count == 2:
                    return {
                        "heatmaps": torch.zeros(
                            (1, 4, 4, 4),
                            dtype=torch.int64,
                            device=inputs.device,
                        ),
                        "kpts_refined": normalized.to(inputs.device),
                    }
                return super().forward(inputs)

        detector = CourtKeyNetLineDetector(
            CourtKeyNetConfig(),
            model=ValidMalformedValidCourtKeyNet(),
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        first = detector.predict(frame, 0, 0, force=True)
        malformed = detector.predict(frame, 1, 40, force=True)
        after_malformed = detector.predict(frame, 2, 80, force=True)

        self.assertEqual(first.status, "courtkeynet confirmation 1/3")
        self.assertEqual(malformed.status, "courtkeynet candidate rejected")
        self.assertEqual(after_malformed.status, "courtkeynet confirmation 1/3")


if __name__ == "__main__":
    unittest.main()
