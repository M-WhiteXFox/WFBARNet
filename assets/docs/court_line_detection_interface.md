# 球场线检测模块接口

本文档按模块整理当前球场线检测相关内容，覆盖统一入口、默认后端、可选后端、运行集成、输出结构和排障注意事项。

> 当前 PyQt 主流程临时停用自动球场线检测，改为手动四点标定。自动 `shuttlecourt_seg` / `opencv` / `monotrack` 后端代码仍保留，便于后续恢复或离线调试。

## 临时手动标定接口

PyQt 当前使用 `apps/pyqt6/services/manual_court_calibration_service.py`：

```python
service = create_manual_court_calibration_service()
prediction = service.set_calibration(
    [[x_tl, y_tl], [x_tr, y_tr], [x_br, y_br], [x_bl, y_bl]],
    source_size=(width, height),
)
```

手动标定输出仍是统一的 `CourtLinePrediction`，其中：

- `scheme = "manual"`。
- `corners` 为用户点击的四个外框角点。
- `court_to_image_h` 和 `image_to_court_h` 由四点单应性计算得到。
- `projected_lines` 为标准羽毛球场模板投影到画面后的完整场线。

因此下游姿态过滤、球轨过滤、统计与标准球场视图不需要切换数据结构。

## 文档模块

| 模块 | 内容 |
| --- | --- |
| 模块一：总体结构 | 当前默认行为、代码文件分工、整体数据流。 |
| 模块二：统一接口 | 后端类型、配置类型、检测器协议、工厂函数和快捷调用。 |
| 模块三：检测后端 | `shuttlecourt_seg`、`opencv`、`monotrack` 三个后端的职责和流程。 |
| 模块四：运行集成 | PyQt 实时服务、批处理路径和重检策略。 |
| 模块五：输出数据 | `CourtLinePrediction` 字段、标准场线键和单应性矩阵用途。 |
| 模块六：使用示例 | 默认调用、显式切换后端、单帧验证。 |
| 模块七：部署与排障 | 权重、依赖、实时重检、下游约定和常见问题。 |

## 模块一：总体结构

### 当前默认行为

- 历史自动检测入口 `create_court_line_detector()` 默认创建 `ShuttleCourtSegLineDetector`。
- `predict_court_lines(...)` 默认使用 `backend="shuttlecourt_seg"`，且快捷调用默认 `force=True`。
- PyQt 实时播放和摄像头推理当前通过 `ManualCourtCalibrationService` 读取手动标定结果，不再异步请求自动球场检测。
- 批处理流程当前不再创建自动球场检测器，球场投影相关指标在未提供手动标定时会降级。
- `opencv` 和 `monotrack` 仍是可选传统 CV 后端，可通过 `backend` 显式选择。

### 代码文件分工

| 文件 | 职责 |
| --- | --- |
| `src/court/court_line_detector.py` | 统一接口层，提供后端类型、检测器协议、工厂函数和单帧快捷调用。 |
| `src/court/shuttlecourt_seg_detector.py` | 当前默认后端。使用 ShuttleCourt/YOLO 分割 mask 估计球场外框，并结合白线和标准模板计算单应性。 |
| `src/court/opencv_court_detector.py` | 传统 OpenCV 白线检测后端，同时定义统一输出结构 `CourtLinePrediction` 和 OpenCV 绘制工具。 |
| `src/court/monotrack_court_detector.py` | MonoTrack 风格传统 CV 后端，保留为可选后端。 |
| `src/court/opencv_court_homography_core.py` | 标准球场模板、白线 mask、单应性、模板投影、候选评分和时序状态更新等公共核心逻辑。 |
| `apps/pyqt6/services/court_detection_service.py` | PyQt 异步检测服务，内部通过统一接口创建检测器。 |

### 整体数据流

```text
视频帧 / 摄像头帧
  -> create_court_line_detector(...) 或 CourtDetectionService
  -> ShuttleCourtSegLineDetector / OpenCVCourtLineDetector / MonoTrackCourtLineDetector
  -> CourtLineDetection 内部候选结果
  -> update_tracking_state(...) 时序更新、平滑、拒绝或复用
  -> CourtLinePrediction 统一输出
  -> UI 叠加显示 / 姿态过滤 / 球轨过滤 / 场地坐标投影 / JSONL 导出
```

## 模块二：统一接口

### 后端类型

```python
CourtLineBackend = Literal["shuttlecourt_seg", "monotrack", "opencv"]
```

| 值 | 后端 |
| --- | --- |
| `shuttlecourt_seg` | 当前默认后端。先使用 ShuttleCourt/YOLO 分割得到球场区域，再做四边形拟合、白线吸附、模板投影和单应性估计。 |
| `opencv` | 项目原有传统 OpenCV 白线检测后端，不依赖 YOLO 分割权重。 |
| `monotrack` | MonoTrack 风格传统 CV 检测后端，可显式选择。 |

### 配置类型

```python
CourtLineConfig = ShuttleCourtSegConfig | MonoTrackCourtLineConfig | OpenCVCourtLineConfig
```

`config` 必须和 `backend` 匹配：

| backend | config 类型 |
| --- | --- |
| `shuttlecourt_seg` | `ShuttleCourtSegConfig` |
| `opencv` | `OpenCVCourtLineConfig` |
| `monotrack` | `MonoTrackCourtLineConfig` |

类型不匹配时，`create_court_line_detector(...)` 会抛出 `TypeError`。

### 检测器协议

所有球场线检测后端都需要实现同一个协议：

```python
class CourtLineDetector(Protocol):
    def reset(self) -> None: ...

    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool = False,
    ) -> CourtLinePrediction: ...

    def latest_prediction(self) -> CourtLinePrediction | None: ...
```

| 方法 | 说明 |
| --- | --- |
| `reset()` | 清空内部跟踪状态和最近一次预测。 |
| `predict(...)` | 对当前帧执行检测，或按时序状态复用已有结果。 |
| `latest_prediction()` | 返回最近一次预测结果。 |

`predict(...)` 参数：

| 参数 | 说明 |
| --- | --- |
| `frame` | BGR 图像帧，通常来自 OpenCV。 |
| `frame_id` | 当前帧编号，用于输出记录和重检测节流。 |
| `timestamp_ms` | 当前帧时间戳，单位毫秒。内部会归一化为非负整数。 |
| `force` | 是否强制本帧重新检测。为 `False` 时，检测器会按 `redetect_interval` 和当前状态决定是否复用结果。 |

### 工厂函数

```python
def create_court_line_detector(
    backend: CourtLineBackend = "shuttlecourt_seg",
    *,
    config: CourtLineConfig | None = None,
) -> CourtLineDetector
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `backend` | `"shuttlecourt_seg"` | 选择检测后端。 |
| `config` | `None` | 后端配置。为 `None` 时使用该后端默认配置。 |

### 单帧快捷调用

```python
def predict_court_lines(
    frame: np.ndarray,
    *,
    frame_id: int = 0,
    timestamp_ms: int = 0,
    detector: CourtLineDetector | None = None,
    backend: CourtLineBackend = "shuttlecourt_seg",
    config: CourtLineConfig | None = None,
    force: bool = True,
) -> CourtLinePrediction
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `detector` | `None` | 可传入已有检测器以复用状态；为 `None` 时内部创建新检测器。 |
| `backend` | `"shuttlecourt_seg"` | 当 `detector is None` 时使用的后端。 |
| `config` | `None` | 当 `detector is None` 时使用的配置。 |
| `force` | `True` | 单帧快捷调用默认强制检测。连续视频建议复用同一个 detector，并按需要设置 `force=False`。 |

## 模块三：检测后端

### 3.1 默认后端：ShuttleCourt 分割检测

`ShuttleCourtSegLineDetector` 的核心目标是把 YOLO 分割出的球场 mask 作为候选 ROI，再在 ROI 内使用 OpenCV 白线检测和标准模板拟合得到真实球场四边形与单应性矩阵。分割区域只负责限制搜索范围，不再直接作为最终外框。

默认配置：

```python
@dataclass(slots=True)
class ShuttleCourtSegConfig:
    weights: str = "weights/shttlecourtnet"
    device: str = "auto"
    imgsz: int = 416
    conf: float = 0.25
    iou: float = 0.70
    max_det: int = 3
    retina_masks: bool = True
    redetect_interval: float = 4.0
    reliable_conf: float = 0.75
    medium_conf: float = 0.55
    smooth_alpha_reliable: float = 0.45
    smooth_alpha_medium: float = 0.20
    min_mask_area_ratio: float = 0.025
    small_candidate_area_ratio: float = 0.12
    small_candidate_min_line_support: float = 0.04
    approx_epsilon_ratio: float = 0.02
    seg_roi_dilate_px: int = 18
    seg_line_min_area_ratio: float = 0.45
    white_s_max: int = 130
    white_v_min: int = 120
    white_chroma_max: int = 96
    line_response_percentile: float = 91.0
    line_response_min: int = 72
    line_local_bg_ksize: int = 31
    use_green_roi: bool = True
    green_h_min: int = 30
    green_h_max: int = 100
    green_s_min: int = 70
    green_v_min: int = 35
    white_green_pair_offset_px: int = 8
    keep_all_green_rois: bool = False
    detect_max_width: int = 960
    hough_threshold: int = 45
    min_line_length_ratio: float = 0.055
    max_line_gap_ratio: float = 0.025
    angle_bin_deg: float = 5.0
    angle_tol_deg: float = 16.0
    min_angle_separation_deg: float = 25.0
    merge_rho_px: float = 18.0
    max_lines_per_family: int = 3
    point_scheme: str = "auto"
    refine_homography: bool = True
    snap_search_px: float = 18.0
    snap_response_threshold: float = 0.18
    max_refine_corner_shift_ratio: float = 0.025
    green_side_offset_px: float = 14.0
    min_outer_width_ratio: float = 0.08
    min_outer_depth_ratio: float = 0.08
    min_outer_width_depth_ratio: float = 0.18
    max_outer_width_depth_ratio: float = 5.5
    max_transverse_angle_deg: float = 35.0
    jump_ratio_hard: float = 0.18
```

核心流程：

1. 根据 `force` 或 `redetect_interval` 判断是否本帧重检。
2. 调用 `ultralytics.YOLO.predict(...)` 获取 `masks.xy`、`boxes.conf` 和 `boxes.cls`。
3. 使用公共 OpenCV 核心逻辑生成白线 mask 和绿色场地 mask。
4. 遍历分割 polygon，过滤面积过小、点数不足或坐标异常的候选。
5. 将 polygon 填充成 ROI mask，并按 `seg_roi_dilate_px` 轻微扩张。
6. 在 ROI 内对白线 mask 执行 Hough 线段检测、方向族聚合和两方向/三方向模板枚举。
7. 若白线拟合结果面积相对分割区域过小，会按 `seg_line_min_area_ratio` 拒绝，避免把内部线误当成外框。
8. 根据白线拟合四边形计算 `court_to_image_h` 与 `image_to_court_h`。
9. 沿标准球场模板线采样，在法线方向搜索附近白线像素，并用 RANSAC 细化单应性。
10. 投影完整标准球场线，生成 `projected_lines`。
11. 若 ROI 白线拟合失败，才回退到 polygon 四边形近似和 `cv2.minAreaRect`。
12. 按分割置信度、几何形状、边界、面积、画面中心、时间稳定性、白线支撑和绿色场地支撑综合评分。
13. 选出评分最高的 candidate，并交给统一时序状态更新逻辑。

权重解析顺序：

1. 传入的绝对路径。
2. 项目根目录下的相对路径。
3. `weights/shttlecourtnet/<name>`
4. `weights/ShuttleCourtNet/<name>`
5. `assets/weights/ShuttleCourtNet/<name>`
6. 以上目录中的 `ShuttleCourt.pt` 或最近修改的 `.pt` 文件。

如果找不到权重，首次实际检测时会抛出 `FileNotFoundError`。如果缺少 `ultralytics`，会抛出提示安装依赖的 `RuntimeError`。

### 3.2 OpenCV 白线检测后端

`OpenCVCourtLineDetector` 是传统白线检测后端，可显式传入 `backend="opencv"` 使用。

核心流程：

1. 将输入帧按 `detect_max_width` 缩放到检测尺度。
2. 使用绿色 ROI、Lab 低色度、局部亮度增强和 Top-hat 响应生成白线 mask。
3. 对白线 mask 做形态学开闭运算和连通域过滤。
4. 使用 Canny + `cv2.HoughLinesP` 提取候选直线段。
5. 根据角度直方图选择主要方向族，并按法线距离合并同方向线。
6. 枚举两方向或三方向线族交点生成候选四边形。
7. 计算单应性、投影标准球场模板、白线吸附细化并评分。
8. 将最佳候选交给统一时序状态更新逻辑。

常用配置：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `redetect_interval` | `4.0` | 自动重检间隔，单位秒。 |
| `detect_max_width` | `960` | 检测最大宽度。 |
| `white_s_max` / `white_v_min` | `130` / `120` | HSV 白线饱和度上限和亮度下限。 |
| `white_chroma_max` | `96` | Lab 色度距离上限。 |
| `line_response_percentile` / `line_response_min` | `91.0` / `72` | 白线响应自适应阈值和最低阈值。 |
| `use_green_roi` | `True` | 是否使用绿色场地区域约束白线。 |
| `hough_threshold` | `45` | Hough 投票阈值。 |
| `min_line_length_ratio` / `max_line_gap_ratio` | `0.055` / `0.025` | Hough 线段长度和断裂连接距离比例。 |
| `angle_bin_deg` / `angle_tol_deg` | `5.0` / `16.0` | 方向直方图分箱和方向族容差。 |
| `merge_rho_px` | `18.0` | 同方向线按法线距离合并阈值。 |
| `max_lines_per_family` | `3` | 每个方向族最多参与模板枚举的线数量。 |
| `refine_homography` | `True` | 是否用白线采样点细化单应性。 |
| `reliable_conf` / `medium_conf` | `0.75` / `0.55` | 可靠更新与中等更新阈值。 |

### 3.3 MonoTrack 风格传统 CV 后端

`MonoTrackCourtLineDetector` 是纯 Python/OpenCV 实现，移植的是 MonoTrack 风格传统 CV 思路。

核心流程：

1. 对帧做灰度亮度检测，寻找局部亮脊线像素。
2. 使用结构张量过滤非线状亮点。
3. 使用 `cv2.HoughLinesP` 提取候选直线段。
4. 优先做三方向角度聚类和模板拟合。
5. 若三方向拟合失败或置信度不足，再回退到两方向模板枚举。
6. 用透视变换拟合标准场地模板。
7. 选择与白线二值图重合度最高的模型。
8. 通过统一 `CourtLinePrediction` 输出角点、单应矩阵和投影场线。

常用配置：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `redetect_interval` | `4.0` | 自动重检间隔，单位秒。 |
| `detect_max_width` | `960` | 检测前最大缩放宽度。 |
| `luminance_threshold` / `diff_threshold` | `80` / `20` | 亮脊线像素的亮度和邻域差阈值。 |
| `ridge_offset_px` | `4` | 比较局部亮脊时的邻域采样偏移。 |
| `gradient_kernel_size` / `structure_kernel_size` | `3` / `21` | 结构张量过滤参数。 |
| `hough_threshold` | `50` | `cv2.HoughLinesP` 投票阈值。 |
| `hough_min_line_length` / `hough_max_line_gap` | `50` / `10` | Hough 线段最小长度和最大断裂连接距离。 |
| `model_sample_step_px` / `model_sample_radius_px` | `8.0` / `2` | 模板采样步长和白线命中半径。 |
| `reliable_conf` / `medium_conf` | `0.68` / `0.48` | 可靠更新与中等更新阈值。 |

## 模块四：运行集成

### PyQt 实时服务

`CourtDetectionService` 默认使用 `shuttlecourt_seg`：

```python
service = CourtDetectionService(
    config=None,
    backend="shuttlecourt_seg",
    submit_interval_s=0.75,
)
service.start()
```

服务方法：

| 方法 | 说明 |
| --- | --- |
| `start()` | 启动后台检测线程。 |
| `stop()` | 请求后台线程停止并等待退出。 |
| `reset()` | 重置检测器状态，清空最新预测。 |
| `request_prediction()` | 允许下一次 `submit_frame(...)` 被后台线程接受。 |
| `clear_pending()` | 清空尚未处理的提交帧。 |
| `submit_frame(frame, frame_id, timestamp_ms)` | 向后台线程提交一帧。返回 `True` 表示已接受。 |
| `latest_prediction()` | 返回最近一次预测对象。 |
| `latest_prediction_dict()` | 返回最近一次预测的字典形式。 |

### 实时重检语义

`CourtDetectionService.submit_frame(...)` 只有在先调用 `request_prediction()`、没有待处理帧、并且满足 `submit_interval_s` 时才会接受当前帧。当前 worker 接受帧后会以 `force=True` 调用检测器，因此 PyQt 实时播放和摄像头模式是“按请求重检”：

- 开始播放或开始摄像头推理时，会请求一次初始检测。
- 点击“重新预测球场线”时，会请求下一帧再次检测。
- 没有新请求时，下游继续读取 `latest_prediction()` 的最近结果。

### 批处理路径

批处理路径不走 PyQt 后台服务，而是直接复用同一个 detector：

```python
court_detector = create_court_line_detector()
court_prediction = court_detector.predict(
    current_frame,
    frame_id,
    current_ms,
    force=processed_frames == 0,
)
```

第一帧强制检测；后续帧由检测器内部的 `should_redetect(...)` 根据 `redetect_interval` 和当前状态决定是否重检。未到重检时间时，检测器会复用已有 `current` 状态构造新的 `CourtLinePrediction`。

### 时序状态更新

三个后端最终都会复用 `TrackingState` 和 `update_tracking_state(...)` 逻辑：

| 条件 | 行为 |
| --- | --- |
| 没有 candidate | 记录 `no candidate`，增加 `rejected_count`，保留旧结果。 |
| `confidence >= reliable_conf` | 按 `smooth_alpha_reliable` 与旧角点融合，更新为 `reliable update`。 |
| `medium_conf <= confidence < reliable_conf` | 只有已经存在旧结果时才按 `smooth_alpha_medium` 平滑更新；没有旧结果时拒绝初始化。 |
| `confidence < medium_conf` | 标记为 `rejected`，保留旧结果。 |

## 模块五：输出数据

### `CourtLinePrediction`

所有后端统一输出该对象，并可通过 `to_dict()` 转为 UI/日志使用的字典。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `frame_id` | `int` | 当前帧编号。 |
| `timestamp_ms` | `int` | 当前帧时间戳，单位毫秒。 |
| `source_size` | `tuple[int, int]` | 原始帧尺寸，格式为 `(width, height)`。 |
| `valid` | `bool` | 当前是否有可用球场检测结果。 |
| `attempted` | `bool` | 本帧是否尝试重新检测。 |
| `updated` | `bool` | 本帧是否用候选结果更新了当前状态。 |
| `update_type` | `str` | 跟踪状态更新类型，例如 `reliable update`、`medium smooth`、`rejected`。 |
| `status` | `str` | 面向 UI 的状态文本。 |
| `confidence` | `float` | 当前结果置信度，范围通常为 `0..1`。 |
| `candidate_confidence` | `float | None` | 本次候选置信度；本帧未检测时可能为 `None`。 |
| `reason` | `str` | 当前结果或候选的评分原因。 |
| `scheme` | `str` | 结果来源，例如 `shuttlecourt_seg`、`opencv` 或 `monotrack`。 |
| `corners` | `list[list[float]]` | 图像中的外框四角，顺序为 `top-left, top-right, bottom-right, bottom-left`。 |
| `keypoints` | `list[dict]` | 投影到图像中的模板关键点，每项包含 `name` 和 `point`。 |
| `court_to_image_h` | `list[list[float]]` | 标准场地坐标到图像坐标的 3x3 单应矩阵。 |
| `image_to_court_h` | `list[list[float]]` | 图像坐标到标准场地坐标的 3x3 单应矩阵。 |
| `projected_lines` | `dict[str, list[list[float]]]` | 投影到图像上的标准场线。 |
| `metrics` | `dict[str, Any]` | 检测诊断指标，例如线数量、模板支撑、评分组件。 |
| `detect_ms` | `float` | 本帧检测耗时，单位毫秒；复用结果时为 `0.0`。 |
| `rejected_count` | `int` | 连续候选被拒绝次数。 |

### 标准场线键

| 键 | 含义 |
| --- | --- |
| `doubles_outer` | 双打外框四边形。 |
| `singles_left_sideline` | 左单打边线。 |
| `singles_right_sideline` | 右单打边线。 |
| `top_short_service` | 上半场前发球线。 |
| `bottom_short_service` | 下半场前发球线。 |
| `top_doubles_long_service` | 上半场双打后发球线。 |
| `bottom_doubles_long_service` | 下半场双打后发球线。 |
| `top_center_service` | 上半场中线。 |
| `bottom_center_service` | 下半场中线。 |

`image_to_court_h` 使用标准羽毛球场坐标，宽 `610`、长 `1340`。下游通常把它按厘米使用，用于球点、球员脚点、热力图、移动距离和击球区域统计。

### 下游使用

| 下游模块 | 使用字段 | 用途 |
| --- | --- | --- |
| 视频叠加 | `projected_lines`, `source_size` | 在视频画面上绘制标准球场线和半透明场地区域。 |
| 姿态过滤 | `image_to_court_h` | 将脚点或 bbox 底部点投影到场地坐标，过滤场外误检并稳定上下半场球员。 |
| 球轨过滤 | `corners`, `projected_lines`, `image_to_court_h` | 判断球点是否处于合理区域，降低背景噪声影响。 |
| 统计与热力图 | `image_to_court_h` | 计算球员位置、移动距离、击球区域和热力分布。 |
| 导出日志 | `to_dict()` | 将球场有效性、置信度、角点和诊断指标写入逐帧日志。 |

## 模块六：使用示例

### 默认检测器

```python
from src.court import create_court_line_detector

detector = create_court_line_detector()
prediction = detector.predict(frame, frame_id=0, timestamp_ms=0, force=True)
```

### 单帧快捷调用

```python
from src.court import predict_court_lines

prediction = predict_court_lines(frame, frame_id=0, timestamp_ms=0)
```

### PyQt 默认服务

```python
from apps.pyqt6.services.court_detection_service import create_court_detection_service

court_service = create_court_detection_service()
```

### 显式选择 OpenCV 后端

```python
from apps.pyqt6.services.court_detection_service import create_court_detection_service
from src.court import OpenCVCourtLineConfig

config = OpenCVCourtLineConfig(redetect_interval=4.0)
court_service = create_court_detection_service(config, backend="opencv")
```

### 显式选择 MonoTrack 后端

```python
from apps.pyqt6.services.court_detection_service import create_court_detection_service
from src.court import MonoTrackCourtLineConfig

config = MonoTrackCourtLineConfig(
    redetect_interval=2.0,
    hough_threshold=45,
    reliable_conf=0.65,
)
court_service = create_court_detection_service(config, backend="monotrack")
```

### 单帧 OpenCV 验证

```python
from src.court import predict_court_lines

prediction = predict_court_lines(
    frame,
    frame_id=0,
    timestamp_ms=0,
    backend="opencv",
)
```

### 推荐验证命令

```powershell
python -m pytest tests/test_court_detector.py
```

## 模块七：部署与排障

### 部署检查

- 默认 `shuttlecourt_seg` 后端依赖 `ultralytics` 和 ShuttleCourt `.pt` 权重；仓库可能只保留权重占位说明，实际部署时需要把权重放到解析路径之一。
- `device="auto"` 时会优先使用 CUDA，否则回退到 CPU。
- MonoTrack 后端当前是 Python/OpenCV 移植，不依赖外部 MonoTrack C++ 可执行文件。
- 三个后端输出同一种 `CourtLinePrediction`，下游应依赖统一字段，而不是根据具体后端解析私有结构。

### 常见问题

| 问题 | 可能原因 | 处理方式 |
| --- | --- | --- |
| 启动后首次球场检测失败 | 未找到 ShuttleCourt `.pt` 权重。 | 将权重放到 `weights/shttlecourtnet/`、`weights/ShuttleCourtNet/` 或 `assets/weights/ShuttleCourtNet/`。 |
| 报缺少 `ultralytics` | 默认后端需要 YOLO 推理依赖。 | 安装项目依赖中的 `ultralytics`。 |
| 实时播放没有持续重检 | PyQt 服务是按请求接收帧，不是每帧自动检测。 | 调用 `request_prediction()`，或在 UI 中点击“重新预测球场线”。 |
| 单帧验证每次都很慢 | `predict_court_lines(...)` 不传 detector 时会重新创建检测器。 | 复用 `create_court_line_detector()` 返回的 detector。 |
| 候选被拒绝但旧结果仍显示 | 时序状态会在低置信候选时复用旧结果。 | 查看 `update_type`、`reason`、`candidate_confidence` 和 `metrics.components`。 |
| 需要传统 CV 路线验证 | 默认后端依赖分割模型。 | 显式传入 `backend="opencv"` 或 `backend="monotrack"`。 |

### 调试重点字段

- `valid`：当前是否有可用球场结果。
- `scheme`：结果来自 `shuttlecourt_seg`、`opencv` 还是 `monotrack`。
- `confidence` / `candidate_confidence`：当前结果和候选结果的置信度。
- `reason`：候选评分或拒绝原因。
- `corners`：外框四角是否覆盖真实场地。
- `projected_lines["doubles_outer"]`：外框投影是否与画面球场外框重合。
- `image_to_court_h`：球员脚点投影是否落在 `0..610`、`0..1340` 附近。
- `metrics.components`：形状、白线支撑、绿色支撑和时间稳定性评分是否异常偏低。
