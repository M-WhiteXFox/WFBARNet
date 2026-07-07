from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib import error, request

from src.utils.user_report import build_user_report_data_from_rally_record, render_user_report_html as render_standalone_user_report_html


REPORT_INTERFACE_VERSION = "wfbar-report.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLER_REPORT_TEMPLATE_DIR = PROJECT_ROOT / "assets" / "report_template"
TABLER_REPORT_TEMPLATE_PATH = TABLER_REPORT_TEMPLATE_DIR / "report-template.html"
TABLER_REPORT_ASSETS_DIR = TABLER_REPORT_TEMPLATE_DIR / "static"
TABLER_REPORT_LEGACY_SOURCE_DIR = PROJECT_ROOT / "assets" / "temp"
TABLER_REPORT_DIST_DIR = PROJECT_ROOT / "assets" / "dist"
TABLER_REPORT_LEGACY_TEMPLATE_PATH = TABLER_REPORT_LEGACY_SOURCE_DIR / "report-template.html"
TABLER_REPORT_FALLBACK_TEMPLATE_PATH = TABLER_REPORT_DIST_DIR / "report-template.html"
TABLER_REPORT_OUTPUT_PATH = TABLER_REPORT_DIST_DIR / "index.html"


class ReportGenerationError(RuntimeError):
    """Raised when a report cannot be generated locally or through an API."""


class ReportApiTransport(Protocol):
    def __call__(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        ...


@dataclass(slots=True)
class ReportGenerationRequest:
    """Structured payload sent to a report generation API."""

    report_id: str
    rally_record: Mapping[str, Any]
    athlete_name: str = "训练用户"
    user_player: str = "bottom"
    video_name: str | None = None
    locale: str = "zh-CN"
    generated_at: str | None = None
    prompt: str = "请基于羽毛球视频分析数据生成训练复盘报告。"
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rally_record(
        cls,
        rally_record: Mapping[str, Any],
        *,
        report_id: str | None = None,
        athlete_name: str = "训练用户",
        user_player: str = "bottom",
        video_name: str | None = None,
        locale: str = "zh-CN",
        generated_at: str | None = None,
        prompt: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> "ReportGenerationRequest":
        summary = _mapping(rally_record.get("summary"))
        inferred_id = str(
            report_id
            or rally_record.get("rally_id")
            or summary.get("rally_id")
            or rally_record.get("rally_name")
            or "wfbar-report"
        )
        return cls(
            report_id=inferred_id,
            rally_record=rally_record,
            athlete_name=athlete_name,
            user_player=user_player,
            video_name=video_name,
            locale=locale,
            generated_at=generated_at,
            prompt=prompt or "请基于羽毛球视频分析数据生成训练复盘报告。",
            options=dict(options or {}),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-serializable interface payload for API transmission."""
        return {
            "interface_version": REPORT_INTERFACE_VERSION,
            "task": "badminton_report_generation",
            "report_id": self.report_id,
            "locale": self.locale,
            "generated_at": self.generated_at,
            "prompt": self.prompt,
            "options": dict(self.options),
            "data": {
                "athlete_name": self.athlete_name,
                "user_player": self.user_player,
                "video_name": self.video_name,
                "rally_record": self.rally_record,
            },
        }


@dataclass(slots=True)
class ReportGenerationResponse:
    report_id: str
    status: str
    provider: str
    report_data: dict[str, Any]
    html: str
    report_text: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_version": REPORT_INTERFACE_VERSION,
            "report_id": self.report_id,
            "status": self.status,
            "provider": self.provider,
            "report_text": self.report_text,
            "report_data": self.report_data,
            "html": self.html,
            "raw_response": self.raw_response,
        }


class ReportApiClient:
    """Minimal JSON API client for external report generation services."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        headers: Mapping[str, str] | None = None,
        transport: ReportApiTransport | None = None,
    ) -> None:
        self.endpoint_url = str(endpoint_url).strip()
        self.api_key = str(api_key).strip() if api_key else None
        self.timeout_s = max(1.0, float(timeout_s))
        self.headers = dict(headers or {})
        self.transport = transport or _urllib_post_json
        if not self.endpoint_url:
            raise ValueError("endpoint_url is required for ReportApiClient.")

    def generate(self, generation_request: ReportGenerationRequest) -> Mapping[str, Any]:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            **self.headers,
        }
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        return self.transport(self.endpoint_url, generation_request.to_payload(), headers, self.timeout_s)


class OpenAICompatibleReportApiClient(ReportApiClient):
    """Report client for OpenAI-compatible chat/completions APIs."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        headers: Mapping[str, str] | None = None,
        transport: ReportApiTransport | None = None,
    ) -> None:
        super().__init__(
            endpoint_url,
            api_key=api_key,
            timeout_s=timeout_s,
            headers=headers,
            transport=transport,
        )
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("model is required for OpenAICompatibleReportApiClient.")

    def generate(self, generation_request: ReportGenerationRequest) -> Mapping[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是羽毛球训练报告生成模型。请基于结构化数据输出中文训练复盘，"
                        "优先返回 JSON 对象，字段包含 report_text；如能生成完整可视化数据，"
                        "可附带 report_data 或 html。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(generation_request.to_payload(), ensure_ascii=False, indent=2),
                },
            ],
            "temperature": 0.2,
        }
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            **self.headers,
        }
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        raw = dict(self.transport(self.endpoint_url, payload, headers, self.timeout_s))
        content = _openai_message_content(raw)
        parsed = _json_object_from_text(content)
        if parsed is not None:
            parsed.setdefault("provider", "openai-compatible")
            parsed.setdefault("raw_response", raw)
            return parsed
        return {
            "status": "ok",
            "provider": "openai-compatible",
            "report_text": content,
            "raw_response": raw,
        }


class ReportGenerationService:
    """Generate badminton reports from current WFBARNet rally records."""

    def __init__(self, api_client: ReportApiClient | None = None) -> None:
        self.api_client = api_client

    def build_local_report(
        self,
        generation_request: ReportGenerationRequest,
        *,
        report_text: str = "",
    ) -> ReportGenerationResponse:
        generated_at = generation_request.generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")
        report_data = build_user_report_data_from_rally_record(
            generation_request.rally_record,
            user_player=generation_request.user_player,
            athlete_name=generation_request.athlete_name,
            video_name=generation_request.video_name,
            generated_at=generated_at,
        )
        if report_text.strip():
            report_data = _inject_report_text(report_data, report_text)
        html = render_tabler_report_html(report_data)
        return ReportGenerationResponse(
            report_id=generation_request.report_id,
            status="ok",
            provider="local",
            report_text=report_text,
            report_data=report_data,
            html=html,
            raw_response={},
        )

    def generate(
        self,
        generation_request: ReportGenerationRequest,
        *,
        use_api: bool = False,
    ) -> ReportGenerationResponse:
        if not use_api:
            return self.build_local_report(generation_request)
        if self.api_client is None:
            raise ReportGenerationError("API report generation requested but no ReportApiClient was configured.")

        raw = dict(self.api_client.generate(generation_request))
        report_text = str(raw.get("report_text") or "")
        local = self.build_local_report(generation_request, report_text=report_text)
        report_data = _merge_api_report_data(local.report_data, raw.get("report_data"), report_text)
        html = render_tabler_report_html(report_data)
        return ReportGenerationResponse(
            report_id=str(raw.get("report_id") or generation_request.report_id),
            status=str(raw.get("status") or "ok"),
            provider=str(raw.get("provider") or "api"),
            report_text=str(raw.get("report_text") or ""),
            report_data=report_data,
            html=html,
            raw_response=raw,
        )

    def export_html(self, response: ReportGenerationResponse, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.html, encoding="utf-8")
        copy_tabler_report_assets(path.parent)

    def export_text(self, response: ReportGenerationResponse, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.report_text, encoding="utf-8")


def generate_report_from_rally_record(
    rally_record: Mapping[str, Any],
    *,
    athlete_name: str = "训练用户",
    user_player: str = "bottom",
    video_name: str | None = None,
    api_client: ReportApiClient | None = None,
    use_api: bool = False,
    output_html_path: Path | None = None,
) -> ReportGenerationResponse:
    generation_request = ReportGenerationRequest.from_rally_record(
        rally_record,
        athlete_name=athlete_name,
        user_player=user_player,
        video_name=video_name,
    )
    service = ReportGenerationService(api_client)
    response = service.generate(generation_request, use_api=use_api)
    if output_html_path is not None:
        service.export_html(response, output_html_path)
    return response


def render_tabler_report_html(report_data: Mapping[str, Any]) -> str:
    """Render the user report with the stable local report template."""
    template_path = _tabler_report_template_path()
    if template_path is None:
        return render_standalone_user_report_html(report_data)

    template = template_path.read_text(encoding="utf-8")
    template_data = _tabler_template_data(report_data)
    meta = _mapping(report_data.get("meta"))
    summary = _mapping(report_data.get("summary"))
    players = _mapping(report_data.get("players"))
    user = _mapping(players.get("user"))
    opponent = _mapping(players.get("opponent"))
    reliability = list(_sequence(report_data.get("reliability")))
    zone_distribution = list(_sequence(report_data.get("zone_distribution")))
    metrics = list(_sequence(summary.get("metrics")))

    template = _inject_tabler_report_data(template, template_data)

    title = str(template_data.get("title") or meta.get("title") or "羽毛球水平分析报告")
    subtitle = str(template_data.get("subtitle") or meta.get("subtitle") or "上传视频后由 WFBARNet 自动生成的训练复盘")
    template = re.sub(
        r'(<h1 class="report-title mb-3">)(.*?)(</h1>)',
        lambda match: f"{match.group(1)}{escape(title)}{match.group(3)}",
        template,
        count=1,
        flags=re.DOTALL,
    )

    zones = _zone_percentages(zone_distribution)
    fields = {
        "title": title,
        "subtitle": subtitle,
        "athlete_name": str(meta.get("athlete_name") or "训练用户"),
        "generated_at": str(meta.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M")),
        "video_name": str(meta.get("video_name") or "training_clip.mp4"),
        "duration_s": _metric_value(metrics, "时长", fallback="0.00秒"),
        "duration": _fmt_seconds_short(template_data.get("duration")),
        "hit_count": _metric_value(metrics, "击球", fallback=_fmt_number(user.get("hit_count") or opponent.get("hit_count") or 0)),
        "rally_count": _fmt_number(template_data.get("rally_count")),
        "avg_hit_interval": _metric_value(metrics, "间隔", fallback="0.00秒"),
        "reliability_score": _fmt_percent(_avg_reliability(reliability)),
        "reliability": f"可信度 {_fmt_percent(template_data.get('reliability'))}",
        "distance": _fmt_meters(template_data.get("distance")),
        "front_zone": _fmt_percent(zones.get("front", 0.0)),
        "mid_zone": _fmt_percent(zones.get("mid", 0.0)),
        "back_zone": _fmt_percent(zones.get("back", 0.0)),
        "ai_summary": _summary_html(summary),
        "bottom_distance": _fmt_meters(user.get("distance_m")),
        "top_distance": _fmt_meters(opponent.get("distance_m")),
        "track_visible_rate": _fmt_percent(_find_reliability(reliability, ("球", "track", "visible"))),
        "pose_valid_rate": _fmt_percent(_find_reliability(reliability, ("姿态", "pose"))),
        "court_status": "manual",
        "bst_coverage": _fmt_percent(_avg_reliability(reliability)),
        "notes": _notes_text(report_data),
        "rally_rows": _render_temp_rally_rows(template_data.get("rallies")),
        "event_rows": _render_temp_event_rows(template_data.get("events")),
    }
    for field, value in fields.items():
        template = _replace_report_field(template, field, value)

    return template


def copy_tabler_report_assets(target_dir: Path) -> None:
    """Copy the Tabler dist assets next to exported reports for local HTML preview."""
    source = _tabler_report_assets_source()
    if not source.exists():
        return
    destination_name = "static" if source == TABLER_REPORT_ASSETS_DIR else "dist"
    destination = target_dir / destination_name
    if source.resolve() == destination.resolve():
        return
    if destination.exists() and not destination.is_dir():
        if destination.stat().st_size == 0:
            destination.unlink()
        else:
            raise FileExistsError(
                f"报告静态资源目标路径已存在且不是目录，请手动检查：{destination}"
            )
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _tabler_report_template_path() -> Path | None:
    if TABLER_REPORT_TEMPLATE_PATH.exists():
        return TABLER_REPORT_TEMPLATE_PATH
    if TABLER_REPORT_LEGACY_TEMPLATE_PATH.exists():
        return TABLER_REPORT_LEGACY_TEMPLATE_PATH
    if TABLER_REPORT_FALLBACK_TEMPLATE_PATH.exists():
        return TABLER_REPORT_FALLBACK_TEMPLATE_PATH
    if TABLER_REPORT_OUTPUT_PATH.exists():
        return TABLER_REPORT_OUTPUT_PATH
    return None


def _tabler_report_assets_source() -> Path:
    if TABLER_REPORT_ASSETS_DIR.exists():
        return TABLER_REPORT_ASSETS_DIR
    legacy = TABLER_REPORT_LEGACY_SOURCE_DIR / "dist"
    return legacy if legacy.exists() else TABLER_REPORT_DIST_DIR / "dist"


def _inject_tabler_report_data(template: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if "__REPORT_DATA_PLACEHOLDER__" in template:
        return template.replace("__REPORT_DATA_PLACEHOLDER__", encoded, 1)
    pattern = re.compile(
        r'(<script\b[^>]*\bid="report-data"[^>]*>\s*)(.*?)(\s*</script>)',
        re.DOTALL | re.IGNORECASE,
    )
    return pattern.sub(lambda match: f"{match.group(1)}{encoded}{match.group(3)}", template, count=1)


def _tabler_template_data(report_data: Mapping[str, Any]) -> dict[str, Any]:
    meta = _mapping(report_data.get("meta"))
    summary = _mapping(report_data.get("summary"))
    players = _mapping(report_data.get("players"))
    user = _mapping(players.get("user"))
    reliability = list(_sequence(report_data.get("reliability")))
    metrics = list(_sequence(summary.get("metrics")))
    duration = _metric_number(metrics, "时长")
    hit_count = _metric_number(metrics, "击球")
    distance = _float(user.get("distance_m"))
    score = _float(summary.get("overall_score")) / 100.0
    reliability_score = _avg_reliability(reliability) or score
    events = _tabler_events(report_data.get("events"))
    return {
        "title": str(meta.get("title") or "羽毛球水平分析报告"),
        "subtitle": _tabler_subtitle(meta, summary),
        "rally_count": 1 if duration > 0 or hit_count > 0 else 0,
        "hit_count": int(hit_count),
        "duration": duration,
        "distance": distance,
        "reliability": reliability_score,
        "notes": _notes_text(report_data),
        "rallies": _tabler_rallies(duration, hit_count, reliability_score),
        "events": events,
        "metrics": metrics,
    }


def _tabler_subtitle(meta: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    parts = [
        str(meta.get("video_name") or "training_clip.mp4"),
        str(meta.get("athlete_name") or "训练用户"),
        str(meta.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]
    level = str(summary.get("level_label") or "")
    if level:
        parts.append(level)
    return " · ".join(part for part in parts if part)


def _notes_text(report_data: Mapping[str, Any]) -> str:
    summary = _mapping(report_data.get("summary"))
    takeaways = [str(item) for item in _sequence(summary.get("takeaways")) if str(item).strip()]
    if takeaways:
        return "\n".join(takeaways[:5])
    note = str(summary.get("level_note") or "").strip()
    if note:
        return note
    return str(report_data.get("disclaimer") or "本报告基于视频分析自动生成，适合训练复盘和趋势比较。")


def _tabler_rallies(duration: float, hit_count: float, reliability: float) -> list[dict[str, Any]]:
    if duration <= 0 and hit_count <= 0:
        return []
    return [
        {
            "start_time": 0.0,
            "end_time": max(0.0, duration),
            "hit_count": int(hit_count),
            "reliability": reliability,
        }
    ]


def _tabler_events(value: object) -> list[dict[str, Any]]:
    events = []
    for item in _sequence(value)[:12]:
        event = _mapping(item)
        stroke = str(event.get("stroke") or event.get("type") or "击球事件")
        player = str(event.get("player_label") or "")
        if player.lower() == "none":
            player = ""
        zone = str(event.get("zone_label") or "")
        time_value = _event_seconds(event.get("time"))
        description = " · ".join(part for part in (player, zone, stroke) if part)
        events.append(
            {
                "type": stroke,
                "time": time_value,
                "description": description or "WFBARNet 自动识别事件",
                "confidence": _float(event.get("confidence")),
            }
        )
    return events


def _render_temp_rally_rows(value: object) -> str:
    rows = []
    for index, item in enumerate(_sequence(value)):
        rally = _mapping(item)
        rows.append(
            "<tr>"
            f"<td>{index + 1}</td>"
            f"<td>{escape(_fmt_seconds_short(rally.get('start_time')))}</td>"
            f"<td>{escape(_fmt_seconds_short(rally.get('end_time')))}</td>"
            f"<td>{int(_float(rally.get('hit_count')))}</td>"
            f'<td><span class="badge bg-green-lt">{escape(_fmt_percent(rally.get("reliability")))}</span></td>'
            "</tr>"
        )
    if not rows:
        return '<tr><td colspan="5" class="text-secondary">暂无回合数据。</td></tr>'
    return "".join(rows)


def _render_temp_event_rows(value: object) -> str:
    rows = []
    for item in _sequence(value):
        event = _mapping(item)
        rows.append(
            '<div class="list-group-item">'
            '<div class="row align-items-center">'
            '<div class="col">'
            f'<div class="font-weight-medium">{escape(str(event.get("type") or "击球事件"))}</div>'
            f'<div class="text-secondary">{escape(_fmt_seconds_short(event.get("time")))} · {escape(str(event.get("description") or "WFBARNet 自动识别事件"))}</div>'
            "</div>"
            '<div class="col-auto">'
            f'<span class="badge bg-blue-lt">{escape(_fmt_percent(event.get("confidence")))}</span>'
            "</div>"
            "</div>"
            "</div>"
        )
    if not rows:
        return '<div class="list-group-item text-secondary">暂无事件数据。</div>'
    return "".join(rows)


def _metric_number(metrics: list[Any], keyword: str) -> float:
    value = _metric_value(metrics, keyword, fallback="0")
    return _number_from_text(value)


def _number_from_text(value: object) -> float:
    text = str(value)
    minute_match = re.search(r"(-?\d+(?:\.\d+)?)\s*分", text)
    second_match = re.search(r"(-?\d+(?:\.\d+)?)\s*秒", text)
    if minute_match and second_match:
        return _float(minute_match.group(1)) * 60.0 + _float(second_match.group(1))
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return _float(match.group(0)) if match else 0.0


def _event_seconds(value: object) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    seconds_match = re.search(r"(-?\d+(?:\.\d+)?)\s*s$", text, re.IGNORECASE)
    if seconds_match:
        return _float(seconds_match.group(1))
    parts = text.split(":")
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        return float(text)
    except ValueError:
        return 0.0


def _fmt_seconds_short(value: object) -> str:
    return f"{_float(value):.2f}秒"


def _replace_report_field(template: str, field: str, value: str) -> str:
    pattern = re.compile(
        rf'(<(?P<tag>[a-zA-Z0-9]+)(?P<attrs>[^>]*\bdata-report-field="{re.escape(field)}"[^>]*)>)(?P<body>.*?)(</(?P=tag)>)',
        re.DOTALL,
    )
    return pattern.sub(lambda match: f"{match.group(1)}{value}{match.group(5)}", template, count=1)


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _metric_value(metrics: list[Any], keyword: str, *, fallback: str) -> str:
    for item in metrics:
        metric = _mapping(item)
        label = str(metric.get("label", ""))
        if keyword in label:
            return escape(str(metric.get("value", fallback)))
    return escape(str(fallback))


def _zone_percentages(items: list[Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    total = 0.0
    for item in items:
        zone = _mapping(item)
        key = str(zone.get("key", ""))
        value = _float(zone.get("value"))
        if key:
            values[key] = value
            total += value
    if total <= 0:
        return {"front": 0.0, "mid": 0.0, "back": 0.0}
    return {key: value / total for key, value in values.items()}


def _summary_html(summary: Mapping[str, Any]) -> str:
    takeaways = _sequence(summary.get("takeaways"))
    if not takeaways:
        return escape(str(summary.get("level_note") or "暂无自动复盘摘要。"))
    return "<br>".join(escape(str(item)) for item in takeaways[:3])


def _find_reliability(items: list[Any], keywords: tuple[str, ...]) -> float:
    for item in items:
        metric = _mapping(item)
        label = str(metric.get("label", "")).lower()
        if any(keyword.lower() in label for keyword in keywords):
            return _float(metric.get("value"))
    return _avg_reliability(items)


def _avg_reliability(items: list[Any]) -> float:
    values = [_float(_mapping(item).get("value")) for item in items]
    values = [value for value in values if value > 0]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _fmt_percent(value: object) -> str:
    return f"{_float(value) * 100:.0f}%"


def _fmt_meters(value: object) -> str:
    return f"{_float(value):.1f}米"


def _fmt_number(value: object) -> str:
    number = _float(value)
    return str(int(number)) if number.is_integer() else f"{number:.1f}"


def _float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return number


def _inject_report_text(report_data: Mapping[str, Any], report_text: str) -> dict[str, Any]:
    payload = dict(report_data)
    summary = dict(_mapping(payload.get("summary")))
    lines = [
        line.strip(" -\t")
        for line in str(report_text).replace("\r", "\n").split("\n")
        if line.strip(" -\t")
    ]
    if lines:
        summary["takeaways"] = lines[:5]
        summary["level_note"] = lines[0]
    payload["summary"] = summary
    return payload


def _merge_api_report_data(
    local_report_data: Mapping[str, Any],
    api_report_data: object,
    report_text: str,
) -> dict[str, Any]:
    payload = dict(local_report_data)
    api_data = _mapping(api_report_data)
    if report_text.strip():
        payload = _inject_report_text(payload, report_text)
    if not api_data:
        return payload

    api_summary = _mapping(api_data.get("summary"))
    if api_summary:
        summary = dict(_mapping(payload.get("summary")))
        takeaways = api_summary.get("takeaways")
        if isinstance(takeaways, list) and takeaways:
            summary["takeaways"] = takeaways
        level_note = api_summary.get("level_note")
        if level_note:
            summary["level_note"] = str(level_note)
        payload["summary"] = summary

    for key in ("improvements", "training_plan", "events"):
        value = api_data.get(key)
        if isinstance(value, list) and value:
            payload[key] = value
    return payload


def _openai_message_content(raw: Mapping[str, Any]) -> str:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ReportGenerationError("OpenAI-compatible API response missing choices.")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ReportGenerationError("OpenAI-compatible API choice must be an object.")
    message = first.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
    else:
        content = first.get("text")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        decoded = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _urllib_post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_s: float,
) -> Mapping[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    http_request = request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with request.urlopen(http_request, timeout=timeout_s) as response:
            response_body = response.read().decode("utf-8")
    except error.URLError as exc:
        raise ReportGenerationError(f"Report API request failed: {exc}") from exc
    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise ReportGenerationError("Report API response is not valid JSON.") from exc
    if not isinstance(decoded, dict):
        raise ReportGenerationError("Report API response must be a JSON object.")
    return decoded


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
