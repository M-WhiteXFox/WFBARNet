# 右侧“数据”选项卡规划

本文档只描述计划新增的数据选项卡，不涉及代码实现。

## 目标

在右侧 `QTabWidget` 中新增“数据”选项卡，用于查看当前回合或当前分析片段的统计数据和逐帧详情。数据选项卡只包含两个子页：

| 子页 | 目标 |
| --- | --- |
| 汇总 | 显示当前回合双方球员的跑动、速度、启动/急停、前中后场击球等核心指标 |
| 详情 | 显示逐帧、轨迹事件、击球识别、球员位置、场地与滤波调试等原始/明细数据 |

## 数据来源

优先复用项目现有数据，不新增模型输出约定。

| 来源 | 当前数据位置 | 用途 |
| --- | --- | --- |
| UI 实时 payload | `frameReady` payload | 当前帧、累计指标、球员投影、场地、轨迹事件、BST 预测 |
| 逐帧 JSONL | `outputs/pyqt_debug/*_frame_log.jsonl` | 球点、姿态、击球/落点/轨迹事件 |
| 轨迹调试 CSV | `outputs/pyqt_debug/*_track_debug.csv` | TrackNet/滤波器选择、拒绝、inpaint、relock、速度等调试字段 |
| 球员场地投影 | `player_projections` / `project_player_points_to_court(...)` | 计算跑动距离、速度、启动、急停、击球区域 |
| BST 击球识别 | `bst_predictions` | 击球类型、置信度、TopK、击球时间 |

## 汇总子页

汇总页建议以“双球员对比表 + 回合击球统计”呈现。上方球员和下方球员并排展示，便于快速比较。

### 回合基础信息

| 字段 | 显示名 | 含义 |
| --- | --- | --- |
| rally_start_ms | 回合开始时间 | 当前回合第一帧或第一条有效轨迹时间 |
| rally_end_ms | 回合结束时间 | 当前回合最后一帧、落地或出画事件时间 |
| rally_duration_s | 回合时长 | `(rally_end_ms - rally_start_ms) / 1000` |
| processed_frames | 处理帧数 | 当前回合参与统计的帧数 |
| rally_hit_count | 该回合击球次数 | 当前回合内 `hit_event` 或 BST 击球结果总数 |
| landing_count | 落点次数 | 当前回合内 `landing_event` 数量 |
| out_of_frame_count | 出画次数 | 当前回合内 `out_of_frame` 数量 |

### 双方球员运动指标

每名球员一行，建议列如下。

| 字段 | 显示名 | 单位 | 含义 |
| --- | --- | --- | --- |
| player_id | 球员 | - | `0` 上方球员，`1` 下方球员 |
| total_distance_m | 累计跑动距离 | m | 当前回合内有效场地投影点的累计位移 |
| avg_speed_mps | 平均速度 | m/s | 当前回合内平均移动速度 |
| max_speed_mps | 最大速度 | m/s | 当前回合内单步/短窗最大移动速度 |
| hard_stop_count | 急停次数 | 次 | 从较高速度快速下降到低速或近似静止的次数 |
| start_count | 启动次数 | 次 | 从静止或低速状态快速进入有效移动的次数 |
| front_court_hit_count | 前场击球次数 | 次 | 该球员在前场完成的击球次数 |
| mid_court_hit_count | 中场击球次数 | 次 | 该球员在中场完成的击球次数 |
| back_court_hit_count | 后场击球次数 | 次 | 该球员在后场完成的击球次数 |
| total_hit_count | 该回合击球次数 | 次 | 该球员本回合击球次数 |

### 指标计算口径

以下为建议口径，后续实现时可以把阈值做成常量或配置项。

#### 累计跑动距离

使用球员足点/脚踝或 bbox 底部点经 `image_to_court_h` 投影后的标准场地坐标，单位为 cm。

```text
step_cm = hypot(curr_x_cm - prev_x_cm, curr_y_cm - prev_y_cm)
total_distance_m = sum(valid_step_cm) / 100
```

建议过滤：

| 规则 | 用途 |
| --- | --- |
| `step_cm < 2` 不累计 | 抑制姿态抖动 |
| `step_cm > 180` 不累计，并重置上一点 | 抑制跟踪跳变 |
| 当前帧缺少球员投影时不累计 | 避免跨丢失段误加距离 |

该口径与当前 `PlayerDistanceAccumulator` 的默认思路一致。

#### 平均速度

推荐使用有效跟踪时长计算：

```text
avg_speed_mps = total_distance_m / valid_tracking_seconds
```

`valid_tracking_seconds` 只统计该球员有连续有效场地投影的时间段。如果有效时间为 0，显示 `--`。

#### 最大速度

逐步计算相邻有效投影点速度：

```text
speed_mps = step_cm / 100 / delta_seconds
max_speed_mps = max(speed_mps)
```

建议对 `step_cm > 180` 或 `delta_seconds <= 0` 的速度样本丢弃，避免异常跳点制造不真实峰值。

#### 启动次数

启动表示从低速或近似静止状态进入明显移动状态。

初始建议规则：

```text
previous_speed_mps <= 0.4
current_speed_mps >= 1.2
```

为避免连续多帧重复计数，建议增加冷却时间：

```text
start_cooldown_ms = 400
```

#### 急停次数

急停表示球员从高速移动快速降到低速或近似静止。

初始建议规则：

```text
previous_speed_mps >= 1.8
current_speed_mps <= 0.6
```

可选增强规则：

```text
deceleration_mps2 = (previous_speed_mps - current_speed_mps) / delta_seconds
deceleration_mps2 >= 4.0
```

同样建议增加冷却时间：

```text
hard_stop_cooldown_ms = 400
```

#### 前中后场击球次数

击球区域以击球时球员的场地 y 坐标划分，场地长度为 `1340 cm`。

对每个 `hit_event` 或 BST 击球结果：

1. 找到击球时间附近最近的球员场地投影点。
2. 根据球员所在半场确定“前/中/后”的方向。
3. 累计到对应球员的前场、中场、后场击球次数。

建议区域划分：

| 球员 | 前场 | 中场 | 后场 |
| --- | --- | --- | --- |
| 上方球员 | `y >= 446.7` 且 `y < 670`，靠近网前 | `223.3 <= y < 446.7` | `0 <= y < 223.3` |
| 下方球员 | `670 < y <= 893.3`，靠近网前 | `893.3 < y <= 1116.7` | `1116.7 < y <= 1340` |

说明：

- 网线位置为 `670 cm`。
- 每个半场长度为 `670 cm`。
- 每个半场按靠近网前到远离网前分为三等份。
- 若击球时找不到可靠球员投影，该次击球可以归入 `unknown_court_hit_count`，不强行分配。

#### 该回合击球次数

总击球次数建议优先使用 `hit_event` 数量；如果 BST 只在 `hit_event` 后输出，可以用 BST 识别结果数量作为展示侧分类统计。

```text
rally_hit_count = count(hit_event)
player_total_hit_count = front + mid + back + unknown
```

如果后续需要区分双方击球，需要先确认击球归属规则。推荐优先用“击球时距离球最近的球员投影点”或 BST 类别中的 `Top_` / `Bottom_` 前缀归属。

## 详情子页

详情页用于承载全部明细数据，不再拆成多个右侧一级选项卡。建议在详情页内部使用筛选控件和表格切换。

### 详情页顶部筛选

| 控件 | 作用 |
| --- | --- |
| 数据类型 | 全部、逐帧、轨迹事件、击球识别、球员位置、场地、滤波调试 |
| 时间范围 | 按 `timestamp_ms` 过滤 |
| 帧范围 | 按 `frame_id` 过滤 |
| 球员 | 上方、下方、全部 |
| 事件类型 | hit、landing、out_of_frame |
| 只看异常 | 低置信度、不可见球点、relock、inpaint、missing |

### 详情页主表

建议主表默认显示轻量字段，复杂数组放入详情面板。

| 数据类型 | 默认列 |
| --- | --- |
| 逐帧 | `frame_id`, `timestamp_ms`, `ball_x`, `ball_y`, `ball_visible`, `ball_score`, `person_count` |
| 轨迹事件 | `event_type`, `frame_id`, `timestamp_ms`, `ball_x`, `ball_y`, `rule`, `confidence` |
| 击球识别 | `event_frame_id`, `timestamp_ms`, `pred_display_name`, `confidence`, `used_homography` |
| 球员位置 | `frame_id`, `timestamp_ms`, `player_id`, `court_x_cm`, `court_y_cm`, `speed_mps` |
| 场地 | `frame_id`, `valid`, `confidence`, `scheme`, `reason` |
| 滤波调试 | `frame_index`, `action`, `reason`, `output_x`, `output_y`, `output_score`, `inpaint_mask` |

### 行详情 JSON

点击任一行后，在详情面板显示完整 JSON。

```json
{
  "frame_id": 123,
  "timestamp_ms": 2050,
  "ball": {},
  "pose": [],
  "court": {},
  "trajectory_event": {},
  "track_debug": {},
  "bst_prediction": {},
  "player_metrics": {}
}
```

建议：

- 大数组默认折叠，例如 `keypoints`、`projected_lines`、`candidates`。
- 空值显示为 `null` 或 `--`，避免误判为 0。
- 支持复制当前行 JSON。
- 支持导出当前筛选结果。

## 需要缓存的新增中间数据

为了让汇总页稳定计算，后续实现时建议在运行时缓存以下结构。

| 数据 | 粒度 | 用途 |
| --- | --- | --- |
| player_position_samples | 每帧/每球员 | 距离、速度、启动、急停、前中后场击球归属 |
| player_speed_samples | 每连续有效步 | 平均速度、最大速度、启动/急停 |
| rally_hit_events | 每次击球 | 回合击球数、前中后场击球数 |
| trajectory_events | 每次轨迹事件 | hit、landing、out_of_frame 详情 |
| bst_predictions | 每次 BST 输出 | 击球类型与置信度 |
| frame_records | 每帧 | 详情页逐帧表与 JSON 详情 |
| track_debug_records | 每帧 | 详情页滤波调试表 |

## 导出建议

| 导出项 | 格式 | 内容 |
| --- | --- | --- |
| 汇总数据 | CSV/JSON/Markdown | 当前回合双方运动指标和击球统计 |
| 详情当前表 | CSV | 当前筛选后的详情表 |
| 逐帧完整数据 | JSONL | 每帧完整记录 |
| 球员轨迹 | CSV | `frame_id,timestamp_ms,player_id,court_x_cm,court_y_cm,speed_mps` |
| 轨迹事件 | CSV/JSON | hit、landing、out_of_frame |
| 滤波调试 | CSV | `track_debug.csv` 字段 |

## 最小可行版本

第一版建议实现：

1. 在右侧新增“数据”选项卡。
2. 数据选项卡内新增“汇总”和“详情”两个子页。
3. 汇总页显示双方：累计跑动距离、平均速度、最大速度、急停次数、启动次数、前中后场击球次数、该回合击球次数。
4. 详情页显示逐帧基础数据、轨迹事件、击球识别、球员位置和滤波调试数据。
5. 点击详情表行显示完整 JSON。

第二版再补充：

1. 汇总页增加小型趋势图或迷你条形图。
2. 详情页增加筛选、排序和导出。
3. 支持按回合切换，查看多个回合的汇总对比。
