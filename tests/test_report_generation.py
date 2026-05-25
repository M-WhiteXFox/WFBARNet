from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from typing import Any, Mapping

from src.utils.report_generation import (
    OpenAICompatibleReportApiClient,
    REPORT_INTERFACE_VERSION,
    ReportApiClient,
    ReportGenerationRequest,
    ReportGenerationService,
    generate_report_from_rally_record,
)


def _rally_record() -> dict[str, Any]:
    return {
        "rally_id": "clip-001",
        "rally_name": "clip.mp4",
        "summary": {
            "duration_s": 30.0,
            "rally_hit_count": 10,
            "avg_hit_interval_ms": 1800.0,
            "hit_confidence_avg": 0.8,
            "out_of_frame_count": 0,
            "stroke_distribution": {"高远球": 4, "杀球": 2, "搓放": 1},
            "data_reliability": {
                "ball_visible_rate": 0.9,
                "pose_valid_rate": 0.86,
                "court_valid_rate": 0.82,
                "avg_ball_confidence": 0.78,
            },
            "players": {
                "bottom": {
                    "label": "测试用户",
                    "distance_m": 40.0,
                    "avg_speed_mps": 1.1,
                    "max_speed_mps": 3.8,
                    "stop_count": 8,
                    "start_count": 9,
                    "hit_count": 10,
                    "zone_hits": {"front": 1, "mid": 3, "back": 6},
                    "passive_hit_count": 4,
                    "high_intensity_count": 5,
                    "max_continuous_m": 5.0,
                },
                "top": {
                    "label": "对手",
                    "distance_m": 36.0,
                    "avg_speed_mps": 1.0,
                    "max_speed_mps": 3.5,
                    "stop_count": 7,
                    "start_count": 8,
                },
            },
        },
        "details": {
            "hits": [
                {
                    "timestamp_ms": 1200,
                    "player": "bottom",
                    "zone": "back",
                    "stroke": "高远球",
                    "confidence": 0.8,
                }
            ]
        },
    }


class ReportGenerationTest(unittest.TestCase):
    def test_request_payload_uses_structured_interface(self) -> None:
        generation_request = ReportGenerationRequest.from_rally_record(
            _rally_record(),
            athlete_name="测试用户",
            generated_at="2026-05-19 12:00",
            options={"style": "coach"},
        )

        payload = generation_request.to_payload()

        self.assertEqual(payload["interface_version"], REPORT_INTERFACE_VERSION)
        self.assertEqual(payload["task"], "badminton_report_generation")
        self.assertEqual(payload["data"]["athlete_name"], "测试用户")
        self.assertEqual(payload["data"]["rally_record"]["rally_id"], "clip-001")
        self.assertEqual(payload["options"]["style"], "coach")

    def test_local_service_generates_html_report(self) -> None:
        generation_request = ReportGenerationRequest.from_rally_record(
            _rally_record(),
            athlete_name="测试用户",
            generated_at="2026-05-19 12:00",
        )
        response = ReportGenerationService().generate(generation_request)

        self.assertEqual(response.provider, "local")
        self.assertIn("羽毛球水平分析报告", response.html)
        self.assertEqual(response.report_data["meta"]["athlete_name"], "测试用户")

    def test_api_client_sends_json_payload_and_service_uses_response(self) -> None:
        captured: dict[str, Any] = {}

        def fake_transport(
            url: str,
            payload: Mapping[str, Any],
            headers: Mapping[str, str],
            timeout_s: float,
        ) -> Mapping[str, Any]:
            captured["url"] = url
            captured["payload"] = dict(payload)
            captured["headers"] = dict(headers)
            captured["timeout_s"] = timeout_s
            return {
                "status": "ok",
                "provider": "fake-api",
                "report_text": "API 自动生成的训练复盘",
            }

        api_client = ReportApiClient(
            "https://example.test/report",
            api_key="secret-token",
            transport=fake_transport,
        )
        generation_request = ReportGenerationRequest.from_rally_record(_rally_record())
        response = ReportGenerationService(api_client).generate(generation_request, use_api=True)

        self.assertEqual(captured["url"], "https://example.test/report")
        self.assertEqual(captured["payload"]["interface_version"], REPORT_INTERFACE_VERSION)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret-token")
        self.assertEqual(response.provider, "fake-api")
        self.assertEqual(response.report_text, "API 自动生成的训练复盘")
        self.assertIn("羽毛球水平分析报告", response.html)

    def test_openai_compatible_client_parses_chat_completion_text(self) -> None:
        captured: dict[str, Any] = {}

        def fake_transport(
            url: str,
            payload: Mapping[str, Any],
            headers: Mapping[str, str],
            timeout_s: float,
        ) -> Mapping[str, Any]:
            captured["url"] = url
            captured["payload"] = dict(payload)
            captured["headers"] = dict(headers)
            return {
                "choices": [
                    {
                        "message": {
                            "content": "本次回合节奏偏短，建议等待完整视频分析后再导出报告。"
                        }
                    }
                ]
            }

        api_client = OpenAICompatibleReportApiClient(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            model="qwen-plus",
            api_key="secret-token",
            transport=fake_transport,
        )
        generation_request = ReportGenerationRequest.from_rally_record(_rally_record())
        response = ReportGenerationService(api_client).generate(generation_request, use_api=True)

        self.assertEqual(captured["payload"]["model"], "qwen-plus")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret-token")
        self.assertEqual(response.provider, "openai-compatible")
        self.assertIn("节奏偏短", response.report_text)
        self.assertIn("节奏偏短", response.html)

    def test_api_report_data_cannot_clear_local_metrics(self) -> None:
        def fake_transport(
            url: str,
            payload: Mapping[str, Any],
            headers: Mapping[str, str],
            timeout_s: float,
        ) -> Mapping[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"report_text":"API 复盘文本","report_data":{}}'
                        }
                    }
                ]
            }

        api_client = OpenAICompatibleReportApiClient(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            model="qwen-plus",
            api_key="secret-token",
            transport=fake_transport,
        )
        generation_request = ReportGenerationRequest.from_rally_record(_rally_record())
        response = ReportGenerationService(api_client).generate(generation_request, use_api=True)

        self.assertEqual(response.report_data["summary"]["metrics"][0]["value"], "30.0秒")
        self.assertEqual(response.report_data["summary"]["metrics"][1]["value"], "10 次")
        self.assertIn("30.0秒", response.html)
        self.assertIn("10 次", response.html)

    def test_generate_report_can_export_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "report.html"
            response = generate_report_from_rally_record(
                _rally_record(),
                athlete_name="测试用户",
                output_html_path=output_path,
            )

            self.assertTrue(output_path.is_file())
            self.assertIn("羽毛球水平分析报告", output_path.read_text(encoding="utf-8"))
            self.assertEqual(response.status, "ok")


if __name__ == "__main__":
    unittest.main()
