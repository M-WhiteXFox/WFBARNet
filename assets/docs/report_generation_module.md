# 报告生成模块

## 1. 模块定位

报告生成模块位于 `src/utils/report_generation.py`，用于把当前项目已经产出的 `rally_record` 转换为训练报告。

模块支持两种运行方式：

1. 本地生成：复用 `src/utils/user_report.py` 中已有的报告数据构建和 HTML 渲染能力。
2. API 生成：通过统一 JSON 接口把结构化数据发送给外部报告生成服务，再接收报告文本、报告数据或 HTML。

该模块不依赖 PyQt 界面，也不直接读取视频文件。它只接收当前分析链路已经输出的结构化数据。

## 2. 输入数据

核心输入是 `RallyStatsAccumulator.export_record()` 或批量模式中的单条回合记录，形状如下：

```json
{
  "rally_id": "clip-001",
  "rally_name": "clip.mp4",
  "summary": {
    "duration_s": 30.0,
    "rally_hit_count": 10,
    "avg_hit_interval_ms": 1800.0,
    "stroke_distribution": {},
    "data_reliability": {},
    "players": {}
  },
  "details": {
    "hits": []
  }
}
```

其中 `summary` 和 `details` 来自现有统计模块，不需要额外从视频中重新计算。

## 3. 接口请求

报告生成请求使用 `ReportGenerationRequest` 表示：

```python
from src.utils.report_generation import ReportGenerationRequest

request = ReportGenerationRequest.from_rally_record(
    rally_record,
    athlete_name="训练用户",
    user_player="bottom",
    video_name="match_clip.mp4",
)
payload = request.to_payload()
```

发送给 API 的 JSON payload 固定包含 `interface_version`，当前版本为：

```text
wfbar-report.v1
```

接口 payload 结构：

```json
{
  "interface_version": "wfbar-report.v1",
  "task": "badminton_report_generation",
  "report_id": "clip-001",
  "locale": "zh-CN",
  "generated_at": "2026-05-19 12:00",
  "prompt": "请基于羽毛球视频分析数据生成训练复盘报告。",
  "options": {},
  "data": {
    "athlete_name": "训练用户",
    "user_player": "bottom",
    "video_name": "match_clip.mp4",
    "rally_record": {}
  }
}
```

## 4. 本地报告生成

```python
from pathlib import Path
from src.utils.report_generation import generate_report_from_rally_record

response = generate_report_from_rally_record(
    rally_record,
    athlete_name="训练用户",
    user_player="bottom",
    output_html_path=Path("outputs/user_reports/report.html"),
)
```

本地生成结果会返回 `ReportGenerationResponse`，其中包含：

| 字段 | 说明 |
| --- | --- |
| `report_id` | 报告 ID。 |
| `status` | 生成状态。 |
| `provider` | `local` 或 API 服务名。 |
| `report_data` | 可渲染的结构化报告数据。 |
| `html` | 独立 HTML 报告内容。 |
| `report_text` | API 生成的自然语言报告文本，本地模式可为空。 |
| `raw_response` | API 原始 JSON 响应，本地模式为空。 |

## 5. 主页导出

PyQt 主页已经接入 `导出报告` 按钮。按钮位于视频/批量控制栏中，当前回合数据可用后自动启用。

当前支持的数据来源：

- 视频分析过程中最新的 `rally_record`。
- 摄像头实时推理过程中最新的 `rally_record`。
- 批量模式中当前选中的单条回合记录。

导出时控制器只读取当前缓存的 `rally_record`，然后调用 `generate_report_from_rally_record(...)` 生成 HTML 文件。导出流程不从表格 UI 反向拼接数据，避免显示格式影响接口数据。

默认导出目录为 `outputs/user_reports`，文件名格式为 `{视频名}_{时间戳}_report.html`。

## 6. API 接入

API 接入使用 `ReportApiClient`。模块使用 HTTP POST 发送 JSON，不传文件路径，不依赖界面状态。

```python
from src.utils.report_generation import (
    ReportApiClient,
    ReportGenerationRequest,
    ReportGenerationService,
)

api_client = ReportApiClient(
    "https://example.com/report",
    api_key="your-api-key",
)
service = ReportGenerationService(api_client)
request = ReportGenerationRequest.from_rally_record(rally_record)
response = service.generate(request, use_api=True)
```

推荐 API 响应也是 JSON 对象：

```json
{
  "status": "ok",
  "provider": "report-api",
  "report_id": "clip-001",
  "report_text": "本次训练节奏较稳定...",
  "report_data": {},
  "html": "<!doctype html>..."
}
```

若 API 只返回 `report_text`，本地仍会根据 `rally_record` 生成默认 `report_data` 和 HTML，保证输出报告可用。

## 7. 测试

新增测试文件：

```text
tests/test_report_generation.py
```

覆盖内容：

- 请求 payload 是否使用统一接口版本。
- 本地报告是否能生成 HTML。
- API 客户端是否通过 JSON payload 发送数据。
- `generate_report_from_rally_record(...)` 是否能导出 HTML 文件。
