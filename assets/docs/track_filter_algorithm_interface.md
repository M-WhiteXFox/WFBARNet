# 轨迹滤波算法接口

更新时间：2026-05-04

本文档描述轨迹滤波可插拔接口。当前生产运行入口统一通过 `create_tracknet_v3_ball_track_filter(...)` 接入 TrackNetV3 风格轨迹滤波；原 `BallTrackFilter` 状态机仍保留为显式兼容实现。

## 1. 设计目标

- 生产路径默认调用 `create_tracknet_v3_ball_track_filter(...)`，统一使用 TrackNetV3 风格轨迹修复算法。
- 新算法仍通过 `BallTrackFilter(algorithm=...)` 接入，外层暴露 `update(...)`、`update_candidates(...)`、`reset()`、`debug_records` 和 `last_debug_record()`。
- 下游继续接收标准 `TrackResult`，因此 `FrameResult`、轨迹尾迹、击球点检测、BST 输入和日志输出不需要理解具体滤波算法。
- 原算法也有显式名称 `LegacyBallTrackFilterAlgorithm`，便于在配置或测试中明确选择“旧算法”。

## 2. 相关代码位置

- `src/postprocess/track_filter.py`
  - `TrackFilterAlgorithm`：新算法需要满足的 Protocol。
  - `FrameShape`：当前帧尺寸类型别名。
  - `PersonBBoxes`：人体框输入类型别名。
  - `BallTrackFilter(..., algorithm=None)`：显式使用原算法；传入 `algorithm` 时作为委托入口。
  - `LegacyBallTrackFilterAlgorithm`：原轨迹滤波实现的显式类名。
- `src/postprocess/track_correction.py`
  - `RealtimeKalmanTrackCorrector`：实时轨迹纠偏算法实现。
  - `RealtimeKalmanTrackCorrectorConfig`：实时纠偏算法参数。
- `src/postprocess/tracknet_v3_filter.py`
  - `TrackNetV3TrajectoryFilter`：当前默认的 TrackNetV3 风格轨迹修复算法。
  - `TrackNetV3TrajectoryFilterConfig`：TrackNetV3 风格修复算法参数。
  - `create_tracknet_v3_ball_track_filter(...)`：PyQt6、CLI runner 和通用 `filter_track_results(...)` 使用的默认工厂。
- `src/utils/structures.py`
  - `TrackResult`：轨迹滤波输入和输出的标准数据结构。
- `src/utils/exporters.py`
  - `TRACK_DEBUG_FIELDS`：`track_debug.csv` 使用的标准调试字段。

## 3. 调用关系

正常运行时，轨迹候选点先由 TrackNet 解码，然后进入滤波接口：

```text
TrackBranch.infer_candidate_results(...)
  -> list[TrackResult] candidates
  -> BallTrackFilter.update_candidates(...)
     -> 原算法状态机
        或
     -> custom_algorithm.update_candidates(...)
  -> TrackResult
  -> FrameResult(frame_id, pose, track)
  -> TrackTrailRenderer / BSTStrokeRecognizer / 日志输出
```

如果传入自定义算法，`BallTrackFilter` 不会预先过滤候选点，也不会替自定义算法标准化 `dt`、`frame_shape`、`court_prediction` 或 `person_bboxes`。这些参数会原样传给新算法。

## 4. 接口定义

新算法不强制继承某个基类，只要结构上满足 `TrackFilterAlgorithm` 即可。

```python
from typing import Any, Protocol, Sequence

from src.utils.structures import TrackResult

FrameShape = tuple[int, ...] | list[int] | None
PersonBBoxes = Sequence[Sequence[float]] | None


class TrackFilterAlgorithm(Protocol):
    debug_records: list[dict[str, object]]

    def reset(self) -> None:
        ...

    def update(
        self,
        track: TrackResult,
        *,
        dt: float | None = None,
        frame_shape: FrameShape = None,
        court_prediction: Any | None = None,
        person_bboxes: PersonBBoxes = None,
    ) -> TrackResult:
        ...

    def update_candidates(
        self,
        tracks: Sequence[TrackResult],
        *,
        dt: float | None = None,
        frame_shape: FrameShape = None,
        court_prediction: Any | None = None,
        person_bboxes: PersonBBoxes = None,
    ) -> TrackResult:
        ...

    def last_debug_record(self) -> dict[str, object] | None:
        ...
```

## 5. 成员契约

| 成员 | 类型 | 作用 |
|---|---|---|
| `debug_records` | `list[dict[str, object]]` | 累积调试记录。PyQt6 实时流程写完一帧后可能调用 `track_filter.debug_records.clear()`，因此新算法必须允许这个列表被外部清空。 |
| `reset()` | `None -> None` | 清空内部状态、轨迹锁定、历史缓存和调试记录。视频重新开始、摄像头重启或测试复用实例时会用到。 |
| `update(...)` | 单个 `TrackResult -> TrackResult` | 处理单个候选点或缺失点。离线 runner 和旧调用路径可能直接使用。 |
| `update_candidates(...)` | 候选列表 `-> TrackResult` | 主运行路径。新算法应在这里完成候选选择、门控、预测、relock 或直接调用自己的 `update(...)`。 |
| `last_debug_record()` | `None` 或 `dict[str, object]` | 返回最近一帧调试记录。建议返回副本，避免调用方意外改动内部状态。 |

## 6. 输入参数

### `track`

`update(...)` 的单点输入，类型为 `TrackResult`：

```python
TrackResult(
    ball_xy=[x, y],
    visible=0 or 1,
    score=float,
    heatmap_shape=[h, w],
)
```

约定：

- `visible=1` 表示该点来自模型候选或算法认为可用。
- `visible=0` 通常表示当前帧没有可用检测，坐标使用 `[-1.0, -1.0]`。
- `score` 是模型置信度或算法生成点的置信度。预测补点可以使用衰减后的低分。
- `heatmap_shape` 建议从被选中的候选点保留，方便排查热力图来源。

### `tracks`

`update_candidates(...)` 的候选列表。正常来自 `TrackBranch.infer_candidate_results(...)`，按热力图候选排序保留多个 `TrackResult`。

约定：

- 列表可能为空。
- 列表可能包含 `score` 低于正式阈值的弱候选。
- 新算法需要自行决定是否使用最高分候选、运动路径附近候选、场地内候选或人体遮挡区域外候选。
- 如果没有可用候选，仍应返回不可见 `TrackResult`，而不是返回 `None`。

### `dt`

当前帧相对上一帧的时间间隔，单位为秒。

约定：

- 可能为 `None`。
- 可能来自视频 FPS，也可能来自摄像头实时计时。
- 自定义算法应自行决定默认值，例如 `1.0 / fps`。
- 传入自定义算法时，外层 `BallTrackFilter` 不会根据 `fps` 自动补全 `dt`。

### `frame_shape`

当前帧尺寸，通常直接传 `numpy.ndarray.shape`：

```text
(height, width, channels)
```

约定：

- 可能为 `None`。
- 只保证至少前两个维度在正常图像帧中分别表示 `height` 和 `width`。
- 如果算法需要做出画判断，应注意不要把它当成 `(width, height)`。

### `court_prediction`

场地检测输出，可能是对象，也可能是字典。常见字段：

| 字段 | 含义 |
|---|---|
| `valid` | 场地检测是否有效。 |
| `image_to_court_h` | 图像坐标到场地平面坐标的 3x3 单应矩阵。 |
| `corners` | 图像中的场地四角点。 |
| `projected_lines["doubles_outer"]` | 可作为场地外框角点的 fallback。 |

自定义算法可以忽略该参数，也可以复用它做场地范围过滤。为了兼容当前数据来源，读取字段时建议同时支持 `dict.get(...)` 和 `getattr(...)`。

### `person_bboxes`

当前稳定球员姿态的 bbox 列表：

```text
[(x1, y1, x2, y2), ...]
```

约定：

- 可能为 `None`。
- 每个 bbox 至少应按前四个值解释为左上和右下坐标。
- 常用于人体遮挡判断、人体区域假点抑制或候选排序惩罚。

## 7. 输出契约

所有接口必须返回 `TrackResult`。

可见输出：

```python
TrackResult(ball_xy=[float(x), float(y)], visible=1, score=float(score), heatmap_shape=[...])
```

不可见输出：

```python
TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=float(score), heatmap_shape=[...])
```

下游会直接使用这个输出：

- `FrameResult.track` 保存该结果。
- `TrackTrailRenderer` 只根据滤波后的可见点更新轨迹尾迹和击球点检测。
- `frame_log.jsonl` 记录滤波后的球点，而不是原始 TrackNet 候选点。
- BST 击球动作识别使用 hit event 附近的滤波轨迹。

因此新算法不应返回 `None`，也不应返回非 `TrackResult` 对象。

## 8. BallTrackFilter 委托规则

### 当前默认路径

```python
from src.postprocess.tracknet_v3_filter import create_tracknet_v3_ball_track_filter

track_filter = create_tracknet_v3_ball_track_filter(fps=fps, debug_enabled=True)
```

行为：

- 使用 `TrackNetV3TrajectoryFilter`。
- `fps` 会写入 `TrackNetV3TrajectoryFilterConfig.fps`。
- `debug_enabled=False` 时不写 `debug_records`。

### 原状态机兼容路径

```python
track_filter = BallTrackFilter(fps=fps, debug_enabled=True)
```

行为：

- 使用原 `BallTrackFilter` 状态机。
- 仅在显式兼容、对比测试或调试旧行为时使用。

### 新算法路径

```python
track_filter = BallTrackFilter(
    fps=fps,
    debug_enabled=True,
    algorithm=my_algorithm,
)
```

行为：

- `update(...)` 直接调用 `my_algorithm.update(...)`。
- `update_candidates(...)` 直接调用 `my_algorithm.update_candidates(...)`。
- `reset()` 直接调用 `my_algorithm.reset()`。
- `last_debug_record()` 直接调用 `my_algorithm.last_debug_record()`。
- `debug_records` 返回 `my_algorithm.debug_records` 这个列表本身。

注意：

- 外层的 `fps`、`config`、`debug_enabled` 不会自动传入 `my_algorithm`。
- 如果新算法需要 FPS 或调试开关，应在新算法自己的构造函数中显式接收。
- 外层不会先做场地过滤、静态热点过滤或候选排序；新算法需要自己完成这些逻辑，或者显式复用原算法。

### 原算法显式名称

```python
track_filter = LegacyBallTrackFilterAlgorithm(fps=fps, debug_enabled=True)
```

这与显式 `BallTrackFilter(...)` 使用同一套旧算法逻辑。它的作用是让配置、测试或文档可以明确表达“当前选择的是原轨迹滤波算法”。

## 9. 内置 TrackNetV3 风格轨迹修复模块

`TrackNetV3TrajectoryFilter` 是当前默认接入的轨迹滤波模块，位于 `src/postprocess/tracknet_v3_filter.py`。它迁移自 `D:\Github\TrackNet-V3-based-Badminton` 项目的轨迹修复思路：

```text
TrackNetV3 heatmap
  -> Top-K TrackResult 候选点
  -> 选择最高分有效候选
  -> 保留可见候选点原始坐标
  -> 可选 fixed-lag 下生成 Inpaint_Mask
  -> 对中间缺失段做 TrackNetV3 线性 inpaint
```

接入方式：

```python
from src.postprocess.tracknet_v3_filter import create_tracknet_v3_ball_track_filter

track_filter = create_tracknet_v3_ball_track_filter(fps=fps, debug_enabled=True)
```

`fixed_lag_frames = 0` 是实时显示默认值，用于避免输出坐标和当前帧时间戳错位。若离线流程希望复用 TrackNetV3 的缺失段修复，可显式设置 fixed-lag，并观察 CSV 中的 `inpaint_mask` 与 `source_frame_offset`。

核心参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `candidate_min_confidence` | `0.35` | 候选进入 TrackNetV3 风格选择阶段的最低分数。 |
| `inpaint_top_threshold_ratio` | `0.05` | 判断缺失段是否属于顶部出画的高度比例；顶部出画不做 inpaint。 |
| `inpaint_top_threshold_px` | `30.0` | 无帧尺寸时使用的顶部阈值。 |
| `inpaint_score` | `0.35` | 线性 inpaint 输出点的最高分数。 |
| `fixed_lag_frames` | `0` | fixed-lag 修复延迟帧数。 |
| `buffer_frames` | `64` | 内部滚动缓存上限。 |

这个模块的关键行为是：不会沿旧速度 coast，也不会用 Kalman 预测点拉扯真实候选点。击球后的急转向候选只要分数和画面范围有效，就会直接输出。

## 10. 内置实时纠偏模块

`RealtimeKalmanTrackCorrector` 是当前工程内置的新轨迹纠偏模块。它实现了：

```text
TrackNetV3 heatmap
  -> Top-K TrackResult 候选点
  -> 多候选 Mahalanobis gating
  -> 自适应 Kalman 预测/更新
  -> 人体遮挡 coast 状态
  -> 出画噪点抑制状态
  -> fixed-lag 小延迟平滑输出
```

接入方式：

```python
from src.postprocess.track_correction import RealtimeKalmanTrackCorrector
from src.postprocess.track_filter import BallTrackFilter

track_filter = BallTrackFilter(
    algorithm=RealtimeKalmanTrackCorrector(
        fps=fps,
        debug_enabled=True,
        fixed_lag_frames=0,
    )
)
```

如果需要调参，可以显式创建配置：

```python
from src.postprocess.track_correction import (
    RealtimeKalmanTrackCorrector,
    RealtimeKalmanTrackCorrectorConfig,
)

config = RealtimeKalmanTrackCorrectorConfig(
    fps=fps,
    gate_chi2=10.0,
    gate_missed_growth=3.0,
    relock_confidence=0.65,
    max_coast_frames=8,
    occlusion_coast_frames=10,
    fixed_lag_frames=0,
)
track_filter = BallTrackFilter(
    algorithm=RealtimeKalmanTrackCorrector(config, debug_enabled=True)
)
```

这个模块不会替换当前默认 TrackNetV3 滤波入口。只有调用方显式传入 `algorithm=RealtimeKalmanTrackCorrector(...)` 时才会启用。

核心参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `candidate_min_confidence` | `0.28` | Top-K 候选进入关联阶段的最低分数。 |
| `bootstrap_confidence` | `0.55` | 初始化轨迹所需的最低候选分数。 |
| `relock_confidence` | `0.65` | 出画或丢失后，单候选重新锁定旧轨迹所需的最低分数。 |
| `strong_relock_confidence` | `0.78` | 靠近画面边缘时更严格的初始化/重锁分数。 |
| `gate_chi2` | `10.0` | Mahalanobis gating 基础门限。 |
| `gate_missed_growth` | `3.0` | 连续漏检时放宽 gating 的幅度。 |
| `score_weight` | `7.5` | 候选排序中检测分数的奖励权重。 |
| `high_confidence_measurement_std_px` | `10.0` | 高分观测的测量噪声。 |
| `low_confidence_measurement_std_px` | `80.0` | 低分观测的测量噪声。 |
| `stable_accel_noise_px_per_sec2` | `3600.0` | 稳定飞行时的过程噪声。 |
| `maneuver_accel_noise_px_per_sec2` | `22000.0` | 疑似突变/击球后短时使用的过程噪声。 |
| `maneuver_snap_innovation_px` | `45.0` | 已通过门控的候选点明显偏离旧预测时，触发短时贴近观测，减少击球后的显示滞后。 |
| `maneuver_snap_confidence` | `0.55` | 触发贴近观测所需的最低候选分数。 |
| `maneuver_snap_weight` | `0.85` | 触发后位置向候选点靠拢的比例。 |
| `max_coast_frames` | `8` | 普通漏检最多预测补点帧数。 |
| `occlusion_coast_frames` | `10` | 人体遮挡时最多预测补点帧数。 |
| `max_missed_frames` | `14` | 连续丢失超过该值后释放旧锁定状态。出画状态也会使用该上限，避免旧预测无限漂移。 |
| `out_of_frame_suppression_frames` | `10` | 出画后抑制边缘噪点的帧数。 |
| `edge_suppression_band_px` | `36.0` | 画面边缘保护带宽度，保护带内候选需要更高分数才允许初始化。 |
| `fixed_lag_frames` | `0` | fixed-lag 平滑延迟帧数。实时显示默认关闭；设为 `1` 可换取轻微平滑但会引入一帧左右延迟。 |

当前实现使用轻量手写 4 维 Kalman 运算，状态为：

```text
[x, y, vx, vy]
```

预测模型为匀速模型；当观测创新较大时，会短时切换到更高过程噪声，相当于自适应机动模型。候选关联使用预测协方差和自适应测量噪声计算 Mahalanobis 距离，高分但远离预测路径的噪点不会仅凭分数直接抢占轨迹。

## 11. 调试记录

调试记录不是强制字段集合，但为了兼容当前 CSV 和 UI 日志，建议尽量使用 `src/utils/exporters.py` 中的 `TRACK_DEBUG_FIELDS`。

当前标准字段：

```text
frame_index
action
reason
raw_candidate_count
candidate_count
selected_candidate_index
selected_candidate_rank
static_filtered_count
static_hotspot_count
input_visible
input_x
input_y
input_score
output_visible
output_x
output_y
output_score
locked_before
locked_after
missed_before
missed_after
coast_before
coast_after
last_x_before
last_y_before
pred_x
pred_y
velocity_x_before
velocity_y_before
velocity_x_after
velocity_y_after
top_exit_remaining
frame_w
frame_h
dt
candidates
```

CSV 写入规则：

- `export_track_debug_csv(...)` 和 PyQt6 的调试 CSV 使用固定字段。
- 额外字段会被忽略。
- 缺失字段会写为空值。

UI 日志目前重点读取：

- `frame_index`
- `action`
- `reason`
- `candidate_count`
- `selected_candidate_index`
- `input_score`
- `pred_x`
- `pred_y`
- `output_visible`

建议新算法至少写入这些字段，方便实时排查。

常用 `action/reason` 可以沿用原算法命名，例如：

- `accept/passes_motion_gate`
- `reject/candidate_failed_motion_gate`
- `coast/velocity_prediction`
- `coast/parabola_prediction`
- `relock_accept/stable_new_candidate`
- `bootstrap_wait/waiting_for_candidate_confirmation`

新算法也可以新增自己的 `action/reason`，但应保持短字符串，方便 CSV 过滤和 UI 日志阅读。

## 12. 最小实现示例

下面是一个只选择最高分候选的极简算法。它不等价于生产滤波逻辑，只用于展示接口形状。

```python
from collections.abc import Sequence
from typing import Any

from src.utils.structures import TrackResult


class HighestScoreTrackFilter:
    def __init__(self, *, fps: float = 25.0, debug_enabled: bool = False) -> None:
        self.fps = fps
        self.debug_enabled = debug_enabled
        self.debug_records: list[dict[str, object]] = []
        self._frame_index = -1
        self._last_debug_record: dict[str, object] | None = None

    def reset(self) -> None:
        self.debug_records.clear()
        self._frame_index = -1
        self._last_debug_record = None

    def update(
        self,
        track: TrackResult,
        *,
        dt: float | None = None,
        frame_shape: tuple[int, ...] | list[int] | None = None,
        court_prediction: Any | None = None,
        person_bboxes: Sequence[Sequence[float]] | None = None,
    ) -> TrackResult:
        self._frame_index += 1
        result = track if track.visible else TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=float(track.score))
        self._record_debug(track, result, dt, frame_shape, candidate_count=1)
        return result

    def update_candidates(
        self,
        tracks: Sequence[TrackResult],
        *,
        dt: float | None = None,
        frame_shape: tuple[int, ...] | list[int] | None = None,
        court_prediction: Any | None = None,
        person_bboxes: Sequence[Sequence[float]] | None = None,
    ) -> TrackResult:
        selected = max(tracks, key=lambda item: float(item.score), default=None)
        if selected is None:
            selected = TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.0)
        return self.update(
            selected,
            dt=dt,
            frame_shape=frame_shape,
            court_prediction=court_prediction,
            person_bboxes=person_bboxes,
        )

    def last_debug_record(self) -> dict[str, object] | None:
        if self._last_debug_record is None:
            return None
        return dict(self._last_debug_record)

    def _record_debug(
        self,
        input_track: TrackResult,
        output_track: TrackResult,
        dt: float | None,
        frame_shape: tuple[int, ...] | list[int] | None,
        *,
        candidate_count: int,
    ) -> None:
        if not self.debug_enabled:
            return
        height = float(frame_shape[0]) if frame_shape and len(frame_shape) >= 1 else 0.0
        width = float(frame_shape[1]) if frame_shape and len(frame_shape) >= 2 else 0.0
        record = {
            "frame_index": self._frame_index,
            "action": "accept" if output_track.visible else "reject",
            "reason": "highest_score",
            "candidate_count": candidate_count,
            "selected_candidate_index": 0 if input_track.visible else -1,
            "input_score": float(input_track.score),
            "output_visible": int(bool(output_track.visible)),
            "output_x": output_track.ball_xy[0] if len(output_track.ball_xy) > 0 else -1.0,
            "output_y": output_track.ball_xy[1] if len(output_track.ball_xy) > 1 else -1.0,
            "pred_x": -1.0,
            "pred_y": -1.0,
            "frame_w": width,
            "frame_h": height,
            "dt": float(dt) if dt is not None else 1.0 / self.fps,
        }
        self._last_debug_record = record
        self.debug_records.append(record)
```

接入：

```python
track_filter = BallTrackFilter(
    algorithm=HighestScoreTrackFilter(fps=fps, debug_enabled=True),
)
```

## 13. 实现检查清单

接入新算法前，建议确认：

1. 空候选列表会返回不可见 `TrackResult`。
2. `dt=None` 时有稳定 fallback。
3. `frame_shape=None` 时不会崩溃。
4. `court_prediction` 为对象或字典时都能处理，或者明确忽略。
5. `person_bboxes=None` 或空列表时不会崩溃。
6. `debug_records` 是可变列表，且允许外部调用 `.clear()`。
7. `last_debug_record()` 与最近一次输出对应。
8. 预测点、补点和 relock 输出都使用标准 `TrackResult`。
9. `update_candidates(...)` 是主路径，测试应覆盖它。
10. 与 `TrackTrailRenderer` 联调时，确认击球点检测看到的是滤波后的轨迹，而不是原始候选。

## 14. 测试入口

TrackNetV3 风格修复模块测试位于 `tests/test_tracknet_v3_filter.py`：

- `test_generates_inpaint_mask_for_middle_disappearance`：确认移植的 `Inpaint_Mask` 规则会修复中间缺失段。
- `test_does_not_inpaint_top_exit_disappearance`：确认顶部出画不被线性补点。
- `test_keeps_post_hit_direction_change_without_kalman_coast`：确认击球后急转向候选直接保留，不发生旧速度 coast。
- `test_can_be_plugged_into_ball_track_filter_interface`：TrackNetV3 修复模块可以通过 `BallTrackFilter(algorithm=...)` 接入。
- `test_factory_builds_tracknet_v3_runtime_filter`：确认生产默认工厂会创建 TrackNetV3 委托滤波器。

建议新增算法时至少补充：

- 候选选择测试。
- 丢帧或空候选测试。
- 状态重置测试。
- 调试记录字段测试。
- 与场地过滤或人体遮挡相关的边界测试。
