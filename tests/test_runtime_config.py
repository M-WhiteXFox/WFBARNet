from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from main import build_runtime_config_from_args, runtime_config_from_mapping


class RuntimeConfigCliTest(unittest.TestCase):
    def test_unknown_config_fields_are_kept_in_extra(self) -> None:
        config = runtime_config_from_mapping(
            {
                "source": "input.mp4",
                "pipeline": "track_only",
                "pose_input_size": [192, 256],
                "pose_stride": 3,
            }
        )

        self.assertEqual(config.source, "input.mp4")
        self.assertEqual(config.pose_input_size, (192, 256))
        self.assertEqual(config.extra["pose_stride"], 3)

    def test_cli_config_is_loaded_and_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "infer.json"
            config_path.write_text(
                json.dumps(
                    {
                        "source": "from_config.mp4",
                        "pipeline": "track_only",
                        "output_dir": "outputs/from_config",
                        "save_vis": True,
                    }
                ),
                encoding="utf-8",
            )

            config = build_runtime_config_from_args(
                [
                    "--config",
                    str(config_path),
                    "--source",
                    "from_cli.mp4",
                    "--pipeline",
                    "pose_only",
                    "--output-dir",
                    "outputs/from_cli",
                    "--no-vis",
                ]
            )

        self.assertEqual(config.source, "from_cli.mp4")
        self.assertEqual(config.pipeline, "pose_only")
        self.assertEqual(config.output_dir, "outputs/from_cli")
        self.assertFalse(config.save_vis)


if __name__ == "__main__":
    unittest.main()
