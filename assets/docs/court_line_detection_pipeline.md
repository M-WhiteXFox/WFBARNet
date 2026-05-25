# 球场线识别流程梳理

本文档记录当前代码中的羽毛球球场线识别流程，重点说明统一入口、默认后端、单帧识别步骤、时序状态更新，以及检测结果在 UI、姿态、轨迹和数据导出中的使用方式。

## 1. 当前结论

当前 PyQt 主流程已临时停用自动球场线识别，改为手动四点标定。历史自动检测能力仍保留在 `src/court/court_line_detector.py`，但主页、视频推理和摄像头推理默认不再启动自动检测服务。

临时手动标定路线：

- PyQt 启动时创建 `ManualCourtCalibrationService`。
- 用户点击主页 `手动标定球场`，在当前视频帧上依次点击外框四角 `TL, TR, BR, BL`。
- 服务根据四点计算标准球场到图像的单应性矩阵，并投影完整标准球场线。
- 下游姿态过滤、球轨过滤、统计与热力图继续读取统一 `CourtLinePrediction` 字段。

历史自动球场线识别曾统一收口到 `src/court/court_line_detector.py`：

- 默认后端是 `shuttlecourt_seg`，即基于 ShuttleCourtNet/YOLO 分割结果估计球场外框，再结合白线与标准球场模板计算单应性。
- `opencv` 和 `monotrack` 仍然保留为可选传统 CV 后端。
- 所有后端最终都会输出同一种 `CourtLinePrediction`，下游只依赖 `valid`、`corners`、`projected_lines`、`court_to_image_h` 和 `image_to_court_h` 等统一字段。

整体链路如下：

```text
视频帧 / 摄像头帧
  -> create_court_line_detector(...) 或 CourtDetectionService
  -> ShuttleCourtSegLineDetector / OpenCVCourtLineDetector / MonoTrackCourtLineDetector
  -> CourtLineDetection 内部候选结果
  -> update_tracking_state(...) 时序更新、平滑、拒绝或复用
  -> CourtLinePrediction 统一输出
  -> UI 叠加显示 / 姿态过滤 / 球轨过滤 / 场地坐标投影 / JSONL 导出
```

## 2. 代码位置

| 文件 | 作用 |
| --- | --- |
| `src/court/court_line_detector.py` | 统一工厂与快捷调用入口，定义 `CourtLineBackend`、`create_court_line_detector(...)` 和 `predict_court_lines(...)`。 |
| `src/court/shuttlecourt_seg_detector.py` | 当前默认后端，读取 YOLO 分割 mask，拟合球场四边形并计算单应性。 |
| `src/court/opencv_court_detector.py` | OpenCV 传统白线检测后端，同时定义统一输出结构 `CourtLinePrediction` 和绘制工具。 |
| `src/court/opencv_court_homography_core.py` | 球场模板、白线 mask、Hough 线族、模板投影、单应性细化、候选评分和时序状态更新等公共核心逻辑。 |
| `src/court/monotrack_court_detector.py` | MonoTrack 风格传统 CV 后端，使用亮脊线、结构张量、Hough 和模板枚举。 |
| `apps/pyqt6/services/court_detection_service.py` | PyQt 后台检测服务，将球场检测放入 `QThread`，通过信号返回结果。 |
| `apps/pyqt6/controllers/analysis_controller_runtime.py` | 播放、摄像头和批处理流程中提交帧、读取 `court_prediction`，并把结果传给轨迹、姿态和统计模块。 |
| `apps/pyqt6/views/components/video_player_panel_runtime.py` | 根据 `projected_lines` 在视频画面上绘制球场线覆盖层。 |
| `src/utils/exporters.py` | 将球场识别摘要写入逐帧 JSONL 日志。 |

## 3. 入口流程

### 3.1 统一工厂

`create_court_line_detector()` 当前默认等价于：

```python
create_court_line_detector(backend="shuttlecourt_seg")
```

可选后端包括：

| backend | 检测器 | 说明 |
| --- | --- | --- |
| `shuttlecourt_seg` | `ShuttleCourtSegLineDetector` | 当前默认；先用 YOLO 分割定位球场区域，再做几何和白线约束。 |
| `opencv` | `OpenCVCourtLineDetector` | 传统 OpenCV 白线检测；不依赖深度分割模型。 |
| `monotrack` | `MonoTrackCourtLineDetector` | MonoTrack 风格传统 CV 方案；偏重亮线脊线与模板枚举。 |

`predict_court_lines(...)` 是单帧快捷入口。若没有传入已有 detector，它会先创建检测器再调用 `predict(...)`。快捷调用默认 `force=True`，因此更适合独立单帧验证；连续视频建议复用同一个 detector。

### 3.2 PyQt 实时播放流程

应用启动时，`apps/pyqt6/main.py` 会创建 `create_court_detection_service()`。该服务默认使用 `shuttlecourt_seg` 后端。

实时播放时的流程是：

1. `_start_playback(...)` 调用 `_reset_court_detection(request_initial_prediction=True)`。
2. 服务重置后台检测器状态，并通过 `request_prediction()` 请求下一帧做一次球场检测。
3. 播放循环中，每一帧都会调用 `submit_frame(current_frame, frame_id, current_ms)`。
4. 后台 worker 只有在已经收到 `request_prediction()`、没有待处理帧、且满足 `submit_interval_s` 时才会接受当前帧。
5. worker 在线程中调用 `detector.predict(..., force=True)`，得到 `CourtLinePrediction`。
6. 结果通过 `resultReady` 信号回到 UI，并以 `prediction.to_dict()` 形式给视图和日志使用。
7. 播放循环通过 `latest_prediction()` 读取最近一次有效或无效结果；若没有再次请求重检，下游会继续使用最近一次结果。

用户点击重新预测球场线时，控制器会再次调用 `request_prediction()`，下一帧会被后台服务接收并强制重检。

### 3.3 批处理 / 导出流程

批处理路径中不会走 PyQt 后台服务，而是在控制器内部直接创建：

```python
court_detector = create_court_line_detector()
```

每帧调用：

```python
court_prediction = court_detector.predict(
    current_frame,
    frame_id,
    current_ms,
    force=processed_frames == 0,
)
```

第一帧强制检测；后续帧由检测器内部的 `should_redetect(...)` 根据 `redetect_interval` 和当前状态决定是否重检。未到重检时间时，检测器会复用已有 `current` 状态构造新的 `CourtLinePrediction`。

## 4. 默认后端：ShuttleCourt 分割检测

`ShuttleCourtSegLineDetector` 的核心目标是把 YOLO 分割出的球场 mask 作为候选 ROI，再在 ROI 内通过 OpenCV 白线检测和标准模板拟合得到真实球场四边形与单应性矩阵。分割结果不再直接等同于最终球场外框。

### 4.1 模型与权重解析

默认配置为 `ShuttleCourtSegConfig(weights="weights/shttlecourtnet")`。权重解析顺序包括：

- 传入的绝对路径或相对路径。
- `weights/shttlecourtnet/`
- `weights/ShuttleCourtNet/`
- `assets/weights/ShuttleCourtNet/`

若路径是目录，会优先使用目录下最近修改的 `.pt` 文件；若存在 `ShuttleCourt.pt`，会优先匹配该文件。`device="auto"` 时优先使用 CUDA，否则使用 CPU。

### 4.2 单帧检测步骤

单帧 `predict(...)` 的流程如下：

1. 校验输入帧，并把 `timestamp_ms` 规范为非负整数。
2. 根据 `force` 或 `should_redetect(...)` 判断本帧是否真正检测。
3. 调用 YOLO `model.predict(...)`，获取 `masks.xy`、`boxes.conf` 和 `boxes.cls`。
4. 调用公共核心逻辑 `create_white_line_mask(...)` 生成白线 mask 和绿色场地区域 mask，用于后续线检测、细化和评分。
5. 遍历每个分割 polygon，过滤点数不足、面积过小或坐标异常的候选。
6. 将 polygon 填充为 ROI mask，并按 `seg_roi_dilate_px` 轻微扩张，避免分割边界刚好切掉白线。
7. 在 ROI 内对白线 mask 做 Hough 线段检测、方向族聚合和两方向/三方向模板枚举，优先得到由真实白线支撑的球场四边形。
8. 若 ROI 白线拟合结果面积相对分割区域过小，会按 `seg_line_min_area_ratio` 拒绝，避免把内部发球线错误当作外框。
9. 通过标准球场外框计算 `court_to_image_h` 和 `image_to_court_h`。
10. 沿标准球场模板线采样，在法线方向搜索附近白线像素，并用 RANSAC 重新拟合单应性。
11. 若细化后的四边形仍然凸、面积有效且角点平均偏移不超过阈值，则接受细化结果。
12. 投影完整标准球场线，生成 `projected_lines`。
13. 计算 `mask_support`、`green_side_support`、`snap_points`、`seg_roi_support` 等质量指标。
14. 若 ROI 白线拟合失败，才回退到旧的 polygon 四边形近似和白线吸附流程。
15. 综合分割置信度、几何合理性、图像边界、面积、画面中心位置、与上一结果的稳定性、白线支撑和绿色边界支撑进行评分。
16. 从多个候选中选择 `rank` 最高的候选作为本帧 candidate。

### 4.3 候选评分

ShuttleCourt 分割后端会把模型框置信度和几何/视觉质量融合：

- `box_confidence`：YOLO 检测置信度。
- `seg_geometry`：四边形几何质量。
- `seg_shape`：球场形状是否符合透视下的合理范围。
- `seg_bounds`：角点是否大幅越界。
- `seg_area_score`：分割面积是否过小或过大。
- `seg_center`：候选是否接近主球场区域。
- `seg_temporal`：与上一帧结果是否稳定。
- `seg_line_support`：投影球场线与白线 mask 的重合度。
- `seg_green_sides`：外框两侧是否有绿色场地支撑。
- `seg_snap_points`：白线细化时的有效吸附点数量。
- `seg_line_fit`：是否使用分割 ROI 内的 OpenCV 白线拟合结果。
- `seg_roi_support`：投影后的标准场线是否仍落在分割 ROI 支持区域内。
- `seg_line_area_ratio`：白线拟合出的球场面积与分割区域面积的比例。

若候选形状不合理、明显越界、四边形质量弱，或首帧候选远离主球场区域，置信度会被降权。

## 5. 可选传统 CV 后端

### 5.1 OpenCV 后端

`OpenCVCourtLineDetector` 不依赖深度模型，主要流程是：

1. 将输入帧按 `detect_max_width` 缩放到检测尺度。
2. 通过 HSV 绿色 ROI、Lab 低色度、局部亮度增强、Top-hat 响应等生成白线 mask。
3. 对白线 mask 做形态学开闭运算和连通域过滤。
4. 使用 Canny + `cv2.HoughLinesP` 提取线段。
5. 根据线段角度直方图选择两个主要方向族。
6. 对同方向线段按法线距离聚合为 `MergedLine`。
7. 枚举两组线族交点形成候选四边形。
8. 若两方向方案不足或置信度低，回退到三方向线族方案。
9. 对候选四边形计算单应性、投影标准球场线、白线吸附细化并评分。
10. 将最佳候选交给统一的时序状态更新逻辑。

### 5.2 MonoTrack 后端

`MonoTrackCourtLineDetector` 使用另一套传统 CV 思路：

1. 在灰度图中寻找相对邻域更亮的局部脊线像素。
2. 用结构张量过滤非线状亮点。
3. 用 `cv2.HoughLinesP` 提取候选线段。
4. 优先尝试三方向线族拟合。
5. 若三方向拟合失败或置信度不足，再选择两方向线族。
6. 枚举标准球场模板中的横纵线对，生成单应性候选。
7. 根据投影模板与白线 mask 的采样重合度选择最佳模型。
8. 使用统一输出结构返回角点、单应性和投影场线。

## 6. 时序状态与重检策略

三个后端最终都会复用 `TrackingState` 和 `update_tracking_state(...)` 逻辑。

### 6.1 何时重检

`should_redetect(...)` 的规则是：

- 第 0 帧必须检测。
- 当前还没有有效 `state.current` 时必须检测。
- 否则只有距离上一次尝试检测的时间超过 `redetect_interval` 才自动重检。

在 PyQt 后台服务中，提交给 worker 的帧当前使用 `force=True`，但 worker 只有收到 `request_prediction()` 后才会接受一帧。因此实时 UI 的球场重检是“按请求触发”的。

### 6.2 如何更新状态

检测候选进入 `update_tracking_state(...)` 后：

- 没有 candidate：记录 `no candidate`，增加 `rejected_count`，保留旧结果。
- `confidence >= reliable_conf`：按 `smooth_alpha_reliable` 与旧角点融合，更新为 `reliable update`。
- `medium_conf <= confidence < reliable_conf`：只有已经存在旧结果时才按 `smooth_alpha_medium` 平滑更新；没有旧结果时不会用中等置信候选初始化。
- `confidence < medium_conf`：标记为 `rejected`，保留旧结果。

平滑时会重新计算角点、关键点、`court_to_image_h`、`image_to_court_h` 和 `projected_lines`，确保输出结构与最新融合结果一致。

## 7. 输出结构

统一输出为 `CourtLinePrediction`。关键字段如下：

| 字段 | 含义 |
| --- | --- |
| `frame_id` / `timestamp_ms` | 当前帧编号和时间戳。 |
| `source_size` | 原始帧尺寸，格式为 `(width, height)`。 |
| `valid` | 当前是否有可用球场结果。 |
| `attempted` | 本帧是否尝试过检测。 |
| `updated` | 本帧 candidate 是否更新了当前状态。 |
| `update_type` / `status` | 时序状态更新类型和 UI 状态文本。 |
| `confidence` | 当前可用结果的置信度。 |
| `candidate_confidence` | 本次候选的置信度；未检测或无候选时可为空。 |
| `scheme` | 结果来源，例如 `shuttlecourt_seg`、`opencv` 或 `monotrack`。 |
| `reason` | 候选评分或拒绝原因。 |
| `corners` | 图像中的球场外框四角，顺序为 TL、TR、BR、BL。 |
| `keypoints` | 投影到图像中的模板关键点。 |
| `court_to_image_h` | 标准球场坐标到图像像素坐标的 3x3 单应性矩阵。 |
| `image_to_court_h` | 图像像素坐标到标准球场坐标的 3x3 单应性矩阵。 |
| `projected_lines` | 投影到图像上的完整标准球场线。 |
| `metrics` | 线段数量、支撑点、白线支撑、绿色支撑、评分组件等调试指标。 |
| `detect_ms` | 本次检测耗时；复用结果时通常为 0。 |
| `rejected_count` | 连续候选被拒绝次数。 |

标准球场坐标来自 `opencv_court_homography_core.py`，宽 `610`、长 `1340`，单位可按厘米理解。因此 `image_to_court_h` 可用于把图像中的球点、脚点或 bbox 底部点投影到标准场地平面。

`projected_lines` 常见键包括：

- `doubles_outer`
- `singles_left_sideline`
- `singles_right_sideline`
- `top_short_service`
- `bottom_short_service`
- `top_doubles_long_service`
- `bottom_doubles_long_service`
- `top_center_service`
- `bottom_center_service`

## 8. 下游使用

### 8.1 视频叠加

`video_player_panel_runtime.py` 从 `court_prediction` 字典读取 `projected_lines`，按 `source_size` 与当前显示区域比例缩放，在视频画面上绘制球场外框、标准线和半透明场地区域。

### 8.2 姿态检测与球员跟踪

姿态分支和 `CourtPoseTargetTracker` 会读取 `image_to_court_h`：

- 将人的脚点、脚踝点或 bbox 底部点投影到标准场地。
- 过滤场外误检。
- 区分上半场和下半场球员。
- 稳定两名球员的身份，减少交换和抖动。

### 8.3 羽毛球轨迹过滤

轨迹过滤模块会读取 `corners`、`projected_lines["doubles_outer"]` 或 `image_to_court_h`：

- 判断球点是否处在合理的球场区域或空气区域。
- 降低背景白点、场外点和不合理跳变对球轨迹的影响。
- 在后续事件检测中提供空间约束。

### 8.4 场地坐标、距离与统计

控制器中的 `project_player_points_to_court(...)` 和 `project_ball_to_court(...)` 使用 `image_to_court_h` 得到场地平面坐标。后续模块基于这些坐标计算：

- 球员移动距离和速度。
- 标准球场视图中的球员位置。
- 热力图。
- 击球区域、回合统计和数据可靠性。

### 8.5 导出日志

`frame_result_log_record(...)` 会把球场结果摘要写入逐帧 JSONL。导出字段包括 `valid`、`attempted`、`updated`、`update_type`、`status`、`confidence`、`candidate_confidence`、`reason`、`scheme`、`corners`、`metrics`、`detect_ms` 和 `rejected_count`。

## 9. 失败与降级行为

常见无效结果包括：

- 未找到 ShuttleCourt 权重。
- 缺少 `ultralytics` 依赖。
- 模型没有输出分割 mask。
- 分割 polygon 面积过小、点数不足或无法形成合理四边形。
- 传统 CV 后端提取到的线段不足。
- 候选四边形形状不合理、越界、白线支撑不足或时间跳变过大。

若已经有旧的有效结果，检测失败或候选被拒绝时通常会继续复用旧结果；若还没有旧结果，则输出 `valid=False`，下游应跳过球场投影相关逻辑。

## 10. 验证入口

推荐的检查方式：

- 单元测试：`python -m pytest tests/test_court_detector.py`
- 单帧/脚本调用：使用 `predict_court_lines(frame, backend="shuttlecourt_seg")`
- 分割结果抽样可视化：`python tools/demo/run_shuttlecourt_video_frames_demo.py`
- OpenCV 后端交互验证：`python tools/opencv_court_homography_demo/run_opencv_court_homography.py`

验证时重点查看：

- `prediction.valid` 是否为 `True`。
- `scheme` 是否符合预期后端。
- `corners` 是否按 TL、TR、BR、BL 覆盖真实场地外框。
- `projected_lines["doubles_outer"]` 与画面外框是否重合。
- `image_to_court_h` 投影后的球员脚点是否落在 `0..610`、`0..1340` 附近。
- `metrics.components` 中的形状、白线支撑和时间稳定性评分是否异常偏低。
