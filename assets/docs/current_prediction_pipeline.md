# 当前轨迹推理流程

更新时间：2026-05-04

本文档记录当前 PyQt6 应用中的羽毛球轨迹推理、轨迹滤波、击球点检测、姿态辅助和调试日志流程。文档描述的是当前代码实现，不是理想设计。

主要相关文件：

- `apps/pyqt6/controllers/analysis_controller_runtime.py`
- `src/models/track_branch.py`
- `src/preprocess/track.py`
- `src/postprocess/track.py`
- `src/postprocess/track_filter.py`
- `src/postprocess/trajectory_events.py`
- `src/postprocess/pose.py`
- `src/postprocess/player_distance.py`
- `src/utils/visualize.py`
- `src/utils/exporters.py`
- `assets/docs/track_filter_algorithm_interface.md`

## 1. 总体流程

```text
视频/摄像头帧
  -> 组织 TrackNet 三帧窗口 [prev, current, next]
  -> 场地检测服务输出 court_prediction
  -> TrackBranch 输出候选羽毛球点列表
  -> PoseBranch 输出人体姿态候选
  -> CourtPoseTargetTracker 选择并稳定双方球员姿态
  -> BallTrackFilter 过滤候选点、预测丢失帧、relock 新轨迹
  -> FrameResult(frame_id, pose, track)
  -> RealtimeTrajectoryEventDetector 生成 hit / landing / out_of_frame 候选事件
  -> TrackTrailRenderer 更新轨迹尾迹并绘制事件 marker
  -> BSTStrokeRecognizer 根据 hit_event 做击球动作分类
  -> PlayerDistanceAccumulator 统计球员场地投影位移
  -> UI 显示，并可写入 track_debug.csv / frame_log.jsonl
```

视频回放和摄像头实时流使用相同的核心组件。差异主要在时间戳来源：

- 视频回放使用 `cv2.VideoCapture` 的帧时间和视频 FPS。
- 摄像头实时流使用 `QElapsedTimer` 作为当前时间。

无论当前帧是否实际渲染到 UI，`RealtimeTrajectoryEventDetector.update(...)` 都会被调用，因此 `hit_event`、`trajectory_event` 和 JSONL 日志不会因为降低显示帧率而中断。`TrackTrailRenderer` 在当前主路径中只更新轨迹历史并绘制事件 marker，不生成主 `hit_event`。

## 2. 运行时主循环

运行时在 `analysis_controller_runtime.py` 中创建以下状态对象：

- `BallTrackFilter(algorithm=TrackNetV3TrajectoryFilter(...))`
- `CourtPoseTargetTracker(...)`
- `PlayerDistanceAccumulator()`
- `TrackTrailRenderer(fps=fps, history_seconds=0.5)`
- `RealtimeTrajectoryEventDetector(fps=fps)`
- 可选的 `BSTStrokeRecognizer`

每一帧的核心步骤：

1. 计算当前 `frame_id` 和 `timestamp_ms`。
2. 将当前帧提交给场地检测服务，读取最新 `court_prediction`。
3. 用 `[prev_frame, current_frame, next_frame]` 调用 `TrackBranch.infer_candidate_results(...)`。
4. 每 `POSE_INFERENCE_STRIDE = 2` 帧运行一次姿态推理。
5. 用 `CourtPoseTargetTracker` 更新稳定后的球员姿态。
6. 用 `BallTrackFilter.update_candidates(...)` 从候选球点中输出最终 `TrackResult`。
7. 组装 `FrameResult(frame_id, pose, track)`。
8. 用 TrackNetV3 `bounce_detection` 风格规则更新 `trajectory_event`。
9. 当 `trajectory_event.event_type == "hit"` 时，将该事件作为主 `hit_event`。
10. 写入可选调试日志。
11. 更新轨迹尾迹，并用不同颜色绘制 `hit`、`landing`、`out_of_frame` 事件 marker。
12. 更新 UI payload，包括球点、姿态、场地、球员位移、击球分类和轨迹事件等。

当 TrackNet 使用 TensorRT 后端且姿态推理刚好到期时，轨迹和姿态会通过 `ThreadPoolExecutor(max_workers=2)` 并行推理。

## 3. TrackBranch 推理

`TrackBranch` 支持两种后端：

- `.engine`：TensorRT 后端，通常用于 INT8 量化版本。
- 其他权重文件：PyTorch `TrackNetV3`。

当前 PyQt6 默认创建参数：

```text
input_size = (512, 288)
score_thr = 0.35
max_candidates = 5
candidate_score_thr_ratio = 0.6
```

因此候选点解码阈值为：

```text
candidate_score_thr = score_thr * candidate_score_thr_ratio = 0.21
```

注意：候选列表可以包含 `score < 0.35` 的弱候选。当前 PyQt6 默认接入的 `TrackNetV3TrajectoryFilter` 只接收 `candidate_min_confidence = 0.35` 以上的候选点。

## 4. TrackNet 输入预处理

`preprocess_track_window(...)` 要求刚好 3 帧：

```text
[prev_frame, current_frame, next_frame]
```

预处理步骤：

1. 将三帧都 resize 到 `512 x 288`。
2. BGR 转 RGB。
3. 三帧按通道拼接为 `9 x 288 x 512`。
4. 转为 float32 tensor，并除以 `255.0`。
5. 记录缩放比例 `scale_x`、`scale_y`，用于把热力图坐标还原到原始视频尺寸。

TrackNet 输出取第 1 个前景平面作为羽毛球热力图。

## 5. 热力图候选点解码

候选点解码在 `src/postprocess/track.py` 中完成。

流程：

1. 对热力图按阈值生成二值 mask。
2. 使用 `cv2.connectedComponentsWithStats(mask, 8)` 找连通域。
3. 每个连通域计算：
   - `peak`：区域内最大热力图分数。
   - `mean`：区域平均分数。
   - `area`：连通域面积。
   - `compactness`：面积 / 外接矩形面积。
4. 使用热力图值加权计算连通域中心。
5. 按 `(peak, mean, min(area, 24), compactness)` 排序。
6. 最多保留 `max_candidates = 5` 个候选。
7. 通过 `scale_x/scale_y` 还原到原始帧坐标。

若没有候选点，会返回一个不可见 `TrackResult`：

```python
TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=max_heatmap_score)
```

## 6. BallTrackFilter 输入输出

`BallTrackFilter.update_candidates(...)` 输入：

- TrackNet 候选点列表。
- 当前帧尺寸。
- `court_prediction`。
- 当前稳定姿态的 `person_bboxes`。

输出：

```python
TrackResult(
    ball_xy=[x, y],
    visible=0 or 1,
    score=float,
    heatmap_shape=[h, w],
)
```

`visible=1` 表示当前滤波器认为该帧有可用球点。当前 PyQt6 默认的 `TrackNetV3TrajectoryFilter` 会保留 TrackNetV3 可见候选点，不做 Kalman coast；可选 fixed-lag 模式下，中间缺失段可能来自 TrackNetV3 风格的线性 inpaint。

### 可插拔算法接口

`BallTrackFilter` 现在同时承担默认滤波器和算法入口两种角色：

- `BallTrackFilter(...)`：不传 `algorithm` 时继续使用原状态机算法。
- `BallTrackFilter(algorithm=...)`：传入新算法时，`update(...)`、`update_candidates(...)`、`reset()`、`debug_records` 和 `last_debug_record()` 都委托给新算法。
- `LegacyBallTrackFilterAlgorithm(...)`：原算法的显式类名，用于配置、测试或文档中明确选择旧算法。

新算法需要满足 `TrackFilterAlgorithm` 接口，输入仍是 `TrackResult` 或候选 `list[TrackResult]`，输出必须是标准 `TrackResult`。接口签名、调试字段、参数语义和最小实现示例见 `assets/docs/track_filter_algorithm_interface.md`。

当前 PyQt6 运行时已启用 TrackNetV3 风格轨迹修复模块 `TrackNetV3TrajectoryFilter`，位于 `src/postprocess/tracknet_v3_filter.py`。它从 `D:\Github\TrackNet-V3-based-Badminton` 项目的 `generate_inpaint_mask(...)` 和 `linear_interp(...)` 迁移核心规则：可见候选点原样保留；如果 fixed-lag 模式允许看到未来端点，则对非顶部出画的中间缺失段做线性修复。实时显示默认 `fixed_lag_frames = 0`，避免输出点时间戳落后一帧。

```python
from src.postprocess.track_filter import BallTrackFilter
from src.postprocess.tracknet_v3_filter import TrackNetV3TrajectoryFilter

track_filter = BallTrackFilter(
    algorithm=TrackNetV3TrajectoryFilter(
        fps=fps,
        debug_enabled=True,
        fixed_lag_frames=0,
    )
)
```

旧的 `RealtimeKalmanTrackCorrector` 仍保留在 `src/postprocess/track_correction.py`，可通过 `BallTrackFilter(algorithm=...)` 显式接入。它不再是 PyQt6 默认运行路径。

## 7. 候选点预过滤

当前 PyQt6 默认 `TrackNetV3TrajectoryFilter.update_candidates(...)` 只做轻量候选选择：过滤不可见、低于 `candidate_min_confidence = 0.35`、或超出画面的候选点，然后选择最高分候选。它不做 Kalman 预测、人体遮挡 coast 或场地范围预过滤，因此击球后的高分急转向候选会直接保留，不会被旧速度模型拖拽。

以下规则描述原 `BallTrackFilter` 状态机和可选 Kalman 纠偏路径中的预过滤逻辑，不是当前 PyQt6 默认路径：

`update_candidates(...)` 先做三件事：

1. **场地范围过滤**
   - 如果 `court_prediction` 有 `corners`，使用图像中的场地投影和向上空气区域过滤。
   - 如果没有 `corners` 但有 `image_to_court_h`，将图像点投影到场地平面后过滤。
   - 场地尺寸常量为宽 `610 cm`、长 `1340 cm`。

2. **静态热点过滤**
   - 连续多帧停在很小范围内的高分点会被记为 static hotspot。
   - 常用于抑制 INT8 量化后固定位置反复冒出的假点。
   - 相关日志字段：`static_filtered_count`、`static_hotspot_count`。

3. **候选排序**
   - 未锁定时直接选择最高分候选。
   - 已锁定时根据预测点、候选分数、运动门控、热力图排序、人体遮挡惩罚综合打分。
   - `candidates` 调试字段会记录每个候选的坐标、分数、到预测点距离、gate 结果、遮挡标记和 rank。

## 8. BallTrackFilter 状态机

滤波器内部维护：

- `_last_point`：上一帧确认点。
- `_velocity`：当前速度估计。
- `_missed_frames`：连续未接收真实点数量。
- `_coast_frames`：连续预测补点数量。
- `_locked`：是否锁定轨迹。
- `_candidate`：等待确认的新轨迹候选。
- `_history`：用于抛物线拟合的真实检测历史。
- `_real_detections_since_relock`：relock 后连续真实检测数。
- `_top_exit_frames_remaining`：顶部出画抑制倒计时。

单帧决策顺序：

1. 如果正在顶部出画抑制，直接输出不可见。
2. 将候选转换为 measurement。正式 measurement 要求：
   - `visible=1`
   - `score >= min_confidence = 0.35`
   - 点在画面内
   - 点在场地过滤范围内
3. 如果正式 measurement 不可用，且满足软阈值条件，尝试使用 soft measurement。
4. 如果没有可用 measurement，执行 reject，并在允许时 coast。
5. 如果未锁定，进入 bootstrap：
   - `score >= strong_relock_confidence = 0.72` 且不在顶部边缘区域时直接锁定。
   - 否则需要候选连续确认，且候选簇最高分不低于 `bootstrap_confirm_min_confidence = 0.55`。
   - 如果确认窗口内候选 `x` 几乎固定且 `y` 波动很小，会判为 `static_bootstrap_candidate`，继续等待而不锁定。
6. 如果预测点已经出画，reject/coast。
7. 如果检测到向上出界后的反向假点，进入 top-exit 抑制。
8. 如果当前 measurement 是顶部边缘幻觉点，reject/coast。
9. 如果人体遮挡活跃且候选落在人体 bbox 内，reject；低分候选可用 `person_occlusion_prediction` coast，高分候选会直接输出不可见。
10. 如果通过主运动门控，accept。
11. 如果通过 close gate，accept，并在 Y 方向冲突时废弃抛物线历史。
12. 否则更新 relock candidate：
    - `impact_direction_change`
    - `high_score_fast_relock`
    - `stable_new_candidate`
13. 如果 relock 不成立，reject，并视情况用抛物线或人体遮挡预测补点。

## 9. 主运动门控

`_passes_gate(...)` 依次检查：

1. 预测点是否出画。
2. 从上一点到候选点的速度是否超过 `max_speed_px_per_sec = 12000`。
3. 候选点到预测点距离是否在动态门限内。
4. 是否通过抛物线门控。
5. 如果候选分数位于 `inertia_relax_confidence = 0.65` 到 `inertia_relax_max_confidence = 0.75` 之间，且距离预测点不超过 `base_gate_px * inertia_relax_prediction_gate_ratio`，会跳过惯性门控，避免减速/转向时把贴近预测点的中高分真实候选误拒。
6. 其他情况继续检查惯性门控。

动态门限由以下因素组成：

- `base_gate_px = 80`
- 当前速度对应的像素步长
- `missed_frames * missed_gate_growth_px`
- 候选分数 bonus
- 最大不超过 `max_gate_px = 360`

惯性门控会检查：

- 加速度是否过大。
- 是否明显反向。
- 横向偏差是否过大。

## 10. 抛物线预测

抛物线预测用于短时丢球和 outlier 填补。

启用条件：

- `parabola_enabled = True`
- 至少 `parabola_min_points = 4` 个历史点。
- relock 后至少 `parabola_min_real_detections = 3` 个连续真实检测。
- 历史跨度足够，运动距离不低于 `parabola_min_motion_px = 42`。
- 拟合 RMSE 不超过 `parabola_max_fit_rmse_px = 48`。
- 预测 gap 不超过 `parabola_max_gap_frames = 8`。

预测策略：

- 如果抛物线可用，`_predict(...)` 优先返回抛物线预测点。
- 否则使用线性速度外推。
- `close_prediction_motion_break` 时，如果抛物线 Y 方向和真实检测 Y 方向冲突，会清空历史并改用当前实测 Y 速度。

这样可以减少 relock 后旧抛物线继续拖拽轨迹的问题。

## 11. Coast 预测

当当前帧没有可靠真实点时，滤波器可能输出预测点。

优先级：

1. 有可用抛物线预测时，用抛物线预测。
2. 否则用线性速度外推。

普通 coast 默认最多 `max_coast_frames = 10` 帧，用来覆盖常见的漏检窗口。之后仍有 `max_missed_frames = 12` 的丢失缓冲，避免刚结束 coast 就立即 drop lock。人体遮挡活跃时：

- `person_occlusion_coast_frames = 7`
- `person_occlusion_min_speed_px_per_sec = 250`
- 如果人体框内存在分数 `> person_occlusion_suppress_coast_candidate_score = 0.55` 的候选，说明球可能已经在击球瞬间改变方向，此帧不再沿旧速度外推，直接输出不可见。

预测点仍必须满足：

- 不出画。
- 不超出场地允许区域。

coast 点的分数会衰减，日志中常见：

- `coast/parabola_prediction`
- `coast/person_occlusion_prediction`

## 12. Relock 机制

当候选点无法通过当前轨迹门控时，滤波器不会立即切换轨迹，而是先积累 `_candidate`。

候选确认逻辑：

- 新候选离旧候选超过 relock 距离时，重置候选。
- 否则平滑更新候选位置和最高分。
- 记录候选出现次数、方向一致性和最近确认窗口内的 measurement。

bootstrap 确认额外过滤：

- `bootstrap_static_max_x_span_px = 3.0`
- `bootstrap_static_max_y_std_px = 15.0`
- 当候选在确认窗口内 `x` 几乎固定且 `y` 标准差很小，会认为是网柱、边线等静态热点，不触发 `candidate_confirmed` 锁定。

触发 relock 的路径：

- `stable_new_candidate`：候选连续达到 `relock_confirm_frames = 3` 且分数足够。
- `high_score_fast_relock`：高分候选 `score > 0.72` 且连续 2 次方向一致。
- `impact_direction_change`：短时漏检后出现方向突变，并满足角度、分数和预测误差要求。

relock 时会先 `_drop_lock()` 清空旧锁定状态，再 accept 新 measurement。relock 后抛物线必须重新积累足够真实检测才会启用。

## 13. 顶部出画和顶部假点

顶部出画逻辑用于处理球从画面上方飞出后，热力图在顶部边缘产生假点的问题。

主要规则：

- 如果历史运动显示球高速向上并接近顶部，会进入 `top_exit_enter/likely_top_exit`。
- 进入顶部出画后，会清空锁定并抑制若干帧。
- 如果当前候选在顶部边缘，但与预测/上一点存在大距离落差，会判为 `top_edge_hallucination`。

相关字段：

- `top_exit_remaining`
- `reason=likely_top_exit`
- `reason=top_edge_hallucination`
- `reason=measurement_reverses_after_top_exit`

## 14. 人体遮挡处理

人体遮挡来自姿态跟踪后的 `person_bboxes`。

滤波器使用人体 bbox 做两类判断：

1. 轨迹线段是否穿过人体 bbox，用于判断遮挡是否可能发生。
2. 当前候选是否落在人体 bbox 内，用于拒绝人体区域内的假球点。

人体遮挡活跃时，会倾向于：

- 给落在人体 bbox 内的候选增加惩罚。
- 拒绝 `person_occlusion_candidate`。
- 延长 coast 帧数，用 `person_occlusion_prediction` 保持轨迹连续。
- 但如果人体 bbox 内被拒绝的候选分数 `> 0.55`，会拒绝旧速度 coast，避免击球瞬间沿击球前方向漂移出假轨迹。

## 15. 轨迹渲染

`TrackTrailRenderer` 不参与 BallTrackFilter 的数据滤波，但参与两件事：

- 绘制球点和尾迹。
- 绘制 `trajectory_event` 产生的击球、落地、出画 marker。

尾迹规则：

- 只绘制最近 `history_seconds = 0.5` 秒。
- 不同 segment 之间不连线。
- 相邻点距离超过 `trail_break_threshold_px = 80` 时不连线。

因此 relock 时即使数据存在跳变，视觉尾迹也会自然断开，避免画出穿越半个画面的长线。

## 16. 击球点检测

当前 PyQt6 主路径的 `hit_event` 只来自 `RealtimeTrajectoryEventDetector`：当 `trajectory_event.event_type == "hit"` 时，该事件会写入 `frame_log.jsonl` 的 `hit_event`，并继续供 `BSTStrokeRecognizer` 使用。

旧的 `TrackTrailRenderer` 内部击球检测入口、姿态辅助击球评分和可视化层内部 hit marker 队列已经清理。`TrackTrailRenderer` 现在只维护轨迹历史并绘制外部传入的 `trajectory_event` marker，不再生成击球事件。

## 17. TrackNetV3 轨迹事件检测

`src/postprocess/trajectory_events.py` 从 `D:\Github\TrackNet-V3-based-Badminton\bounce_detection` 迁移了规则式事件候选生成逻辑。它现在是主击球事件来源，并同时输出：

- `hit`：击球候选，规则包括 `vy_reversal`、`vx_reversal`、`acceleration_peak`、`y_local_max`、`speed_local_max`。
- `landing`：落地点候选，规则包括 `speed_step`、`low_speed_start`、`speed_drop`、`visibility_drop`、`trajectory_end`。
- `out_of_frame`：出画/丢失候选，规则包括 `visibility_drop_edge`、`visibility_drop_upward`、`visibility_drop_high_altitude`。

当前主 `hit_event` 会额外做收紧过滤：

- 只有 `vy_reversal` / `vx_reversal` 可以成为主击球点。
- `acceleration_peak` 和 `speed_local_max` 只作为辅助证据，不再单独触发红色击球点。
- 候选点和相邻可见点需要满足最低 track score，避免低分离群点制造反转。
- 顶部出画带内的反转不作为击球点。
- 单帧速度过小或过大的反转都会被过滤，过大的情况通常来自离群点跳变。
- 候选如果在历史窗口里滞留太久才被确认，会被丢弃，避免旧事件延迟冒出。

实时接入类是 `RealtimeTrajectoryEventDetector`。它维护最近轨迹窗口，按帧重算候选，并只在候选有足够未来帧确认后输出一次。输出事件写入：

- `trajectory_event`：当前帧新确认的任意轨迹事件。
- `landing_event`：当 `trajectory_event.event_type == "landing"` 时额外复制一份，方便只关心落地点的后处理读取。

当前主路径中，`trajectory_event.event_type == "hit"` 的事件会成为 `hit_event`，继续供 `BSTStrokeRecognizer` 使用。`TrackTrailRenderer` 负责轨迹尾迹和事件 marker：`hit` 使用红色圆点，`landing` 使用绿色菱形，`out_of_frame` 使用紫色叉号。

## 18. 姿态与击球事件关系

姿态结果仍用于 `BSTStrokeRecognizer` 的击球动作分类、球员距离统计和可视化骨架绘制，但不再参与 `TrackTrailRenderer` 内部击球判定。击球事件统一由 `RealtimeTrajectoryEventDetector` 基于滤波后的球轨迹生成。

## 19. FrameResult

每一帧最终形成统一结构：

```python
FrameResult(
    frame_id=frame_id,
    pose=last_pose,
    track=track,
)
```

其中：

- `pose` 是经过 `CourtPoseTargetTracker` 稳定后的姿态列表。
- `track` 是 `BallTrackFilter` 输出的最终球轨迹点。

后续 UI、日志、击球点检测、BST 动作识别都基于 `FrameResult`。

## 20. 球员位移统计

球员位移统计不影响球轨迹，只用于 UI 指标。

流程：

1. 从稳定后的姿态中取球员 bbox / 关键点。
2. 使用 `court_prediction.image_to_court_h` 投影到真实场地平面。
3. `PlayerDistanceAccumulator` 按球员 id 或稳定索引累计场地平面位移。
4. UI 显示上方球员、下方球员的累计米数。

如果当前没有可用场地投影，会保持已有累计值，但重置当前跟踪点，避免下一次重新检测时跨大距离累加。

## 21. 日志输出

在 UI 中打开 `Debug Logs / Write frame analysis logs` 后，会在以下目录输出：

```text
outputs/pyqt_debug/
```

一组日志包含：

- `*_track_debug.csv`
- `*_frame_log.jsonl`

### track_debug.csv

CSV 来自 `BallTrackFilter.last_debug_record()`。

关键字段：

- `frame_index`：滤波器处理帧序号。
- `action`：当前帧滤波动作。
- `reason`：动作原因。
- `raw_candidate_count`：TrackNet 原始候选数量。
- `candidate_count`：预过滤后的候选数量。
- `selected_candidate_index`：最终选择候选索引。
- `selected_candidate_rank`：候选排序分数或 primary。
- `input_x/input_y/input_score`：送入状态机的候选点。
- `output_x/output_y/output_score`：滤波器输出点。
- `locked_before/locked_after`：处理前后是否锁定。
- `missed_before/missed_after`：处理前后连续丢失帧数。
- `coast_before/coast_after`：处理前后连续预测帧数。
- `pred_x/pred_y`：处理前预测位置。
- `velocity_x_before/velocity_y_before`：处理前速度。
- `velocity_x_after/velocity_y_after`：处理后速度。
- `top_exit_remaining`：顶部出画抑制剩余帧数。
- `source_frame_offset`：fixed-lag 输出时，当前输出点来自几帧前；实时默认 `0`。
- `inpaint_mask`：`1` 表示该输出点来自 TrackNetV3 风格线性 inpaint。
- `candidates`：每个候选的坐标、分数、距离、gate、遮挡和 rank。

常见 `action/reason`：

- `accept/tracknet_v3_candidate`
- `inpaint/tracknet_v3_linear_inpaint`
- `reject/missing_or_low_confidence`
- `bootstrap_wait/waiting_for_candidate_confirmation`
- `bootstrap_wait/static_bootstrap_candidate`
- `bootstrap_accept/candidate_confirmed`
- `accept/passes_motion_gate`
- `accept/soft_confidence_motion_gate`
- `accept/close_prediction_motion_break`
- `relock_accept/stable_new_candidate`
- `relock_accept/high_score_fast_relock`
- `relock_accept/impact_direction_change`
- `coast/parabola_prediction`
- `coast/person_occlusion_prediction`
- `reject/candidate_failed_motion_gate`
- `reject/person_occlusion_candidate`
- `reject/person_occlusion_candidate_high_score`
- `reject/top_edge_hallucination`
- `top_exit_enter/likely_top_exit`
- `drop_lock/max_missed_frames`

### frame_log.jsonl

JSONL 来自 `frame_result_log_record(...)`，每一行是一帧：

```json
{
  "frame_id": 123,
  "timestamp_ms": 2050,
  "ball": {
    "xy": [100.0, 200.0],
    "visible": 1,
    "score": 0.72
  },
  "pose": [
    {
      "person_id": 0,
      "bbox": [10.0, 20.0, 100.0, 220.0],
      "person_score": 0.91,
      "keypoints": [[50.0, 80.0]],
      "keypoint_scores": [0.9]
    }
  ],
  "hit_event": {
    "frame_id": 123,
    "timestamp_ms": 2050,
    "ball_xy": [100.0, 200.0]
  },
  "trajectory_event": {
    "event_type": "landing",
    "frame_id": 128,
    "timestamp_ms": 2133,
    "ball_xy": [120.0, 260.0],
    "rule": "speed_step",
    "confidence": 0.9,
    "all_rules": ["speed_step"],
    "auxiliary_rules": [],
    "features": {}
  },
  "landing_event": {
    "event_type": "landing",
    "frame_id": 128,
    "timestamp_ms": 2133,
    "ball_xy": [120.0, 260.0],
    "rule": "speed_step",
    "confidence": 0.9,
    "all_rules": ["speed_step"],
    "auxiliary_rules": [],
    "features": {}
  }
}
```

如果该帧没有击球事件，`hit_event` 为 `null`。如果没有新确认的 TrackNetV3 轨迹事件，`trajectory_event` 和 `landing_event` 为 `null`。

排查轨迹误检时优先看 `track_debug.csv`；排查击球点漏检或误检时优先看 `frame_log.jsonl`，并结合 `track_debug.csv` 中相同时间附近的 `action/reason`。

## 22. 调试建议

### 轨迹乱飘

优先检查：

1. `track_debug.csv` 中是否出现 `relock_accept`。
2. relock 前后 `pred_x/pred_y` 与候选点距离。
3. `candidates` 中是否有多个高分候选。
4. 是否频繁出现 `coast/parabola_prediction`。
5. 是否存在 `top_edge_hallucination` 或 `person_occlusion_candidate`。
6. 人体遮挡帧内是否出现 `reject/person_occlusion_candidate_high_score`，这表示高分候选被人体框压制且没有继续 coast。

### 击球点漏检

优先检查：

1. `frame_log.jsonl` 中真实拐点附近的三点轨迹。
2. 对应 `track_debug.csv` 是否处于 `coast`、`relock_accept` 或跨段状态。
3. 拐点是否被当前点分数、相邻点分数、顶部忽略区、速度上下限或事件延迟过滤。
4. 拐点是否只有 `acceleration_peak` / `speed_local_max` 辅助证据，而没有 `vy_reversal` / `vx_reversal` 主反转。

### 击球后出现第二个假红点

优先检查：

1. 两个 `hit_event` 的时间间隔是否刚好超过 `event_cooldown_seconds = 0.18`。
2. 第二个红点的 `rule` 是否仍为 `vy_reversal` / `vx_reversal`。
3. 第二个红点的 `features.reversal_magnitude`、`speed_after` 和邻居分数是否足够可信。
4. 第二个红点附近是否是 relock 后的离群跳变，而不是真实击球瞬间。

## 23. CLI Runner

除 PyQt6 外，项目中还有多个 CLI runner：

- `src/runners/track_video_runner.py`
- `src/runners/pose_video_runner.py`
- `src/runners/unified_runner.py`
- `src/runners/tracknet_realtime_runner.py`

这些 runner 复用 `TrackBranch`、`PoseBranch`、`BallTrackFilter`、`TrackTrailRenderer` 等核心模块，适合离线批处理或单模块验证。PyQt6 的实时体验逻辑额外包含 UI 刷新节流、调试 JSONL、BST 击球识别和球员位移统计。
