# 羽毛球动作分析桌面端需求文档

## 1. 文档目的

本文档用于指导 `Gemini` 开发一个基于：

- `DearPyGui (DPG)`
- `OpenCV`

的桌面端可视化工作台，用于驱动并展示羽毛球视频分析流程。

这份文档将替代此前基于 Web 前端的方案。当前阶段不再要求实现浏览器页面，而是实现一个本地桌面 GUI 应用。

目标是做一个“本地分析工作台”，支持：

- 选择视频
- 启动分析
- 查看任务进度
- 查看日志
- 播放原始视频 / 可视化结果视频
- 叠加显示姿态关键点与羽毛球轨迹
- 查看动作时间轴与摘要信息
- 为后续 BST 输入构建和模型调试服务

---

## 2. 总体定位

这是一个本地桌面工具，不是浏览器系统，不需要：

- React
- TypeScript
- Axios
- 浏览器轮询 UI

而是采用：

- `DearPyGui` 负责 GUI 布局与交互
- `OpenCV` 负责视频解码、帧读取、图像格式转换、视频绘制
- Python 本地线程或任务队列负责耗时推理

应用定位为：

- 面向研发与调试阶段的“视频分析控制台”
- 便于逐步集成 `track only`、`pose only`、`unified`、`BST input build`

---

## 3. 核心目标

桌面 GUI 至少要支持以下功能：

1. 视频输入管理
2. 推理模式选择
3. 参数设置
4. 启动/停止分析
5. 进度与日志展示
6. 原始视频与结果视频显示
7. 单帧姿态与球轨迹可视化
8. 动作时间轴展示
9. 摘要信息展示
10. 导出结果路径展示

---

## 4. 建议目录结构

Gemini 开发时，建议新增一个桌面 GUI 模块目录，例如：

```text
WFBARNet/
├── apps/
│   └── desktop_gui/
│       ├── main.py
│       ├── app.py
│       ├── controllers/
│       │   ├── task_controller.py
│       │   ├── video_controller.py
│       │   └── playback_controller.py
│       ├── panels/
│       │   ├── sidebar_panel.py
│       │   ├── video_panel.py
│       │   ├── control_panel.py
│       │   ├── summary_panel.py
│       │   ├── timeline_panel.py
│       │   └── log_panel.py
│       ├── widgets/
│       │   ├── file_picker.py
│       │   ├── progress_bar.py
│       │   └── status_badge.py
│       ├── state/
│       │   └── app_state.py
│       ├── utils/
│       │   ├── dpg_image.py
│       │   ├── video_cache.py
│       │   └── threading_utils.py
│       └── requirements.txt
```

要求：

- 不要把所有 GUI 逻辑堆在一个文件里
- 视图、状态、任务控制尽量分离
- 保持后续可扩展性

---

## 5. 交互流程

## 5.1 用户主流程

用户操作路径如下：

1. 打开桌面应用
2. 选择视频文件
3. 选择分析模式：
   - `track_only`
   - `pose_only`
   - `unified`
4. 配置参数
5. 点击“开始分析”
6. GUI 显示：
   - 当前状态
   - 日志
   - 进度条
7. 分析结束后：
   - 自动加载结果
   - 自动显示结果视频或叠加画面
   - 展示摘要与时间轴

## 5.2 状态流转

GUI 至少要有这些状态：

- `idle`
- `video_selected`
- `running`
- `completed`
- `failed`
- `stopped`

要求：

- 状态必须在界面中清晰显示
- 分析中按钮状态必须切换
- 失败时要给错误提示

---

## 6. 界面布局要求

建议采用“左侧控制区 + 中央视频区 + 右侧信息区 + 底部日志区”的桌面布局。

### 6.1 左侧控制区

用于放：

- 视频路径
- 模式选择
- 模型路径
- 运行参数
- 启动 / 停止 / 重置按钮

### 6.2 中央视频区

用于显示：

- 原始视频
- 分析结果视频
- 单帧可视化图像

支持：

- 播放
- 暂停
- 拖动进度
- 帧号显示
- 时间显示

### 6.3 右侧信息区

用于显示：

- 摘要信息
- 当前帧姿态信息
- 当前帧球坐标
- 动作列表/时间轴

### 6.4 底部日志区

用于显示：

- 初始化日志
- 推理过程日志
- 结果导出日志
- 错误日志

---

## 7. 功能模块需求

## 7.1 Sidebar / ControlPanel

必须支持：

- 选择输入视频文件
- 选择输出目录
- 选择分析模式
- 设置 `track_weight`
- 设置 `pose_config`
- 设置 `pose_weight`
- 设置设备：
  - `cpu`
  - `cuda:0`
- 设置开关：
  - 保存 JSON
  - 保存 CSV
  - 保存 NPY
  - 保存可视化视频
  - 是否生成 BST 输入

按钮至少包含：

- `Load Video`
- `Start`
- `Stop`
- `Reset`
- `Open Output`

要求：

- 分析运行中禁用重复点击 `Start`
- 没有视频时不能运行
- 参数非法时要提示

---

## 7.2 VideoPanel

这是桌面端最重要的部分之一。

### 必须支持两种显示模式

1. 原始视频显示
2. 分析结果显示

### 显示内容

- 当前帧图像
- 帧号
- 当前时间
- 当前 FPS
- 若有结果，叠加：
  - skeleton
  - bbox
  - ball point
  - ball score

### 技术实现要求

Gemini 需要使用：

- OpenCV 解码视频帧
- 将 BGR 帧转成 RGB
- 将图像转换成 DearPyGui 可显示的 texture 格式

建议封装：

- `numpy.ndarray -> DPG texture` 转换函数
- 视频缓存/预读取机制

### 推荐交互

- 播放/暂停
- 上一帧/下一帧
- 跳到指定帧
- 选择显示：
  - 原始画面
  - 轨迹可视化
  - 姿态可视化
  - 融合可视化

---

## 7.3 ProgressPanel

必须展示：

- 当前状态
- 进度百分比
- 当前阶段说明

例如：

- `Loading model`
- `Decoding video`
- `Running TrackNet`
- `Running Pose`
- `Building BST input`
- `Saving outputs`

要求：

- 使用 DPG progress bar
- 进度值可以平滑更新

---

## 7.4 LogPanel

必须支持：

- 实时追加日志
- 日志滚动查看
- 区分日志级别颜色

级别至少支持：

- `INFO`
- `WARNING`
- `ERROR`

推荐能力：

- 清空日志
- 导出日志

---

## 7.5 SummaryPanel

分析完成后必须展示摘要信息。

至少包括：

- 总帧数
- 有效姿态帧数
- 有效轨迹帧数
- 平均球轨迹置信度
- 平均关键点置信度
- 动作数量

如果当前还没有动作识别结果，也要预留位置。

---

## 7.6 TimelinePanel

需要展示分析结果时间轴。

### 当前阶段最低要求

可以先用列表方式展示动作段：

- 开始时间
- 结束时间
- 动作标签
- 置信度

### 进阶建议

后续可以扩展成真正的时间轴条带。

### 交互要求

- 点击某条动作时，视频跳转到对应时间
- 右侧信息区同步显示对应帧信息

---

## 7.7 TrackPoseInfoPanel

用于显示当前帧详细信息：

- 当前帧号
- 当前时间戳
- 球坐标
- 球是否可见
- 球得分
- 检测到的人数
- 每个人的关键点平均得分
- 当前动作标签

如果没有对应结果，要显示空态。

---

## 8. 模式设计要求

桌面应用必须支持以下四种模式：

### 8.1 `track_only`

只跑球轨迹。

输出：

- `track_results.json`
- `track_results.csv`
- `track_results.npy`
- `track_vis.mp4`

界面中需要：

- 显示羽毛球点
- 显示 ball score

### 8.2 `pose_only`

只跑姿态估计。

输出：

- `pose_results.json`
- `pose_results.csv`
- `pose_results.npy`
- `pose_vis.mp4`

界面中需要：

- 显示 skeleton
- 显示 bbox
- 显示关键点得分

### 8.3 `track_realtime`

实时模式，只跑 TrackNet。

适用于：

- 摄像头
- RTSP
- 低延迟调试视频流

要求：

- 有实时刷新画面
- 有实时 FPS
- 可按 `q` 或按钮停止

### 8.4 `unified`

同时跑：

- `pose`
- `track`

输出：

- `unified_results.json`
- `unified_results.csv`
- `unified_results.npy`
- `unified_vis.mp4`
- `bst_input.npy`

界面中必须：

- 同时显示 skeleton 和 ball point
- 同时显示 pose 与 track 信息

---

## 9. 与现有 Python 后端/推理模块的关系

GUI 不是重新实现推理算法，而是作为桌面端控制层，调用现有 Python 模块。

Gemini 应优先复用现有接口：

- [main.py](/D:/Github/WFBARNet/main.py)
- [src/models/pose_branch.py](/D:/Github/WFBARNet/src/models/pose_branch.py)
- [src/models/track_branch.py](/D:/Github/WFBARNet/src/models/track_branch.py)
- [src/runners/pose_video_runner.py](/D:/Github/WFBARNet/src/runners/pose_video_runner.py)
- [src/runners/track_video_runner.py](/D:/Github/WFBARNet/src/runners/track_video_runner.py)
- [src/runners/tracknet_realtime_runner.py](/D:/Github/WFBARNet/src/runners/tracknet_realtime_runner.py)
- [src/runners/unified_runner.py](/D:/Github/WFBARNet/src/runners/unified_runner.py)
- [src/builders/bst_input_builder.py](/D:/Github/WFBARNet/src/builders/bst_input_builder.py)

要求：

- GUI 作为调用层，不要把大量模型代码重写进 GUI
- GUI 应负责：
  - 参数收集
  - 线程启动
  - 状态更新
  - 结果加载
  - 视频显示

---

## 10. 线程与任务管理要求

由于视频推理是耗时任务，GUI 不能阻塞主线程。

必须采用：

- 后台线程
- 或任务执行器

来运行推理任务。

### 要求

- DPG 主线程只负责界面刷新
- 推理逻辑在后台线程执行
- 线程中不断写入状态
- GUI 定时读取状态并刷新

### 推荐状态共享结构

可以设计一个 `AppState`，包含：

- `current_video_path`
- `current_mode`
- `task_status`
- `task_progress`
- `logs`
- `current_frame_idx`
- `current_frame_image`
- `summary`
- `actions`
- `pose_results`
- `track_results`
- `output_dir`
- `error_message`

---

## 11. OpenCV 集成要求

OpenCV 负责以下事情：

- 视频读取
- 帧提取
- 图像缩放
- BGR/RGB 转换
- 可视化叠加
- 视频写出

Gemini 需要特别处理：

- OpenCV 图像和 DPG texture 格式不一致
- 需要封装稳定转换方法

建议统一封装：

- `load_frame_as_texture(frame: np.ndarray) -> texture_data`
- `draw_overlay(frame, pose, track) -> frame`

---

## 12. DPG 技术要求

必须使用 DearPyGui 原生组件构建，不要混杂其他 GUI 框架。

建议使用：

- `dpg.window`
- `dpg.group`
- `dpg.child_window`
- `dpg.add_button`
- `dpg.add_combo`
- `dpg.add_input_text`
- `dpg.add_progress_bar`
- `dpg.add_image`
- `dpg.add_table`
- `dpg.add_text`

建议风格：

- 深色工作台风格
- 布局整齐
- 信息密度适中

---

## 13. 配置项需求

GUI 中需要有可编辑配置区，至少包含：

- `source`
- `output_dir`
- `pipeline`
- `device`
- `pose_backend`
- `pose_config`
- `pose_weight`
- `track_weight`
- `track_score_thr`
- `pose_conf_thr`
- `save_json`
- `save_csv`
- `save_npy`
- `save_vis`
- `save_bst_input`

要求：

- 配置修改后可直接运行
- 支持默认值回填

---

## 14. 输出展示要求

GUI 中要明确展示输出产物路径，例如：

- `.../track_results.json`
- `.../pose_vis.mp4`
- `.../unified_vis.mp4`
- `.../bst_input.npy`

最好支持：

- 点击打开输出目录
- 点击加载生成的视频

---

## 15. 错误处理要求

必须处理以下错误：

- 视频文件不存在
- 权重文件不存在
- MMPose 未安装
- CUDA 不可用
- 推理中异常
- 视频读取失败

要求：

- GUI 中出现明确错误提示
- 日志中记录错误详情
- 应用不要直接崩溃退出

---

## 16. 最低可交付版本

Gemini 至少要完成以下内容：

1. 一个可启动的 DearPyGui 桌面程序
2. 支持选择视频
3. 支持选择 `track_only / pose_only / unified`
4. 支持点击运行
5. 支持显示进度与日志
6. 支持显示分析后的视频或单帧结果
7. 支持展示摘要信息
8. 支持显示输出目录
9. 支持后台线程执行，不卡 GUI

---

## 17. 推荐增强项

如果 Gemini 能进一步实现，推荐增加：

- 视频播放进度条
- 拖动帧跳转
- 原始/结果双视图对比
- 轨迹历史尾迹显示
- 关键点开关显示
- 时间轴点击跳帧
- 批量视频处理
- 配置保存与加载

---

## 18. 验收标准

### 功能验收

- 可以选择视频并开始运行
- 可以看到进度条变化
- 可以看到日志刷新
- 可以在界面中看到视频画面
- `track_only` 能显示羽毛球点
- `pose_only` 能显示骨架
- `unified` 能同时显示 pose 和 track
- 可以看到输出路径

### 工程验收

- 代码结构清晰
- GUI 与推理解耦
- 线程安全基本可控
- 不把全部逻辑堆在一个文件里

### 体验验收

- 运行时窗口不卡死
- 布局清晰
- 按钮状态明确
- 错误提示易懂

---

## 19. 可以直接给 Gemini 的开发指令

可以直接把下面这段发给 Gemini：

> 请基于本项目现有 Python 推理结构，开发一个 `DearPyGui + OpenCV` 的本地桌面 GUI。  
> 这个 GUI 不再是 Web 前端，而是本地分析工作台。  
> 需要支持视频选择、参数设置、运行模式切换（`track_only`、`pose_only`、`track_realtime`、`unified`）、推理启动/停止、日志展示、进度展示、视频画面显示、结果摘要展示和输出路径展示。  
> 请优先复用现有 `src/` 下的推理模块，不要重写模型逻辑。  
> GUI 需要使用后台线程运行推理，避免阻塞主界面。  
> 在 `unified` 模式下，输出视频中必须同时显示姿态骨架和羽毛球轨迹点。  
> 请按模块化结构开发，不要把所有代码堆在一个文件中。  
> 如果某些功能暂时无法直接接真实结果，请先做结构完整、可运行的占位实现，但整个桌面应用必须能启动并展示工作台界面。

---

## 20. 相关现有文件

Gemini 可参考以下文件：

- [main.py](/D:/Github/WFBARNet/main.py)
- [src/models/pose_branch.py](/D:/Github/WFBARNet/src/models/pose_branch.py)
- [src/models/track_branch.py](/D:/Github/WFBARNet/src/models/track_branch.py)
- [src/runners/pose_video_runner.py](/D:/Github/WFBARNet/src/runners/pose_video_runner.py)
- [src/runners/track_video_runner.py](/D:/Github/WFBARNet/src/runners/track_video_runner.py)
- [src/runners/tracknet_realtime_runner.py](/D:/Github/WFBARNet/src/runners/tracknet_realtime_runner.py)
- [src/runners/unified_runner.py](/D:/Github/WFBARNet/src/runners/unified_runner.py)
- [src/utils/visualize.py](/D:/Github/WFBARNet/src/utils/visualize.py)
- [src/builders/bst_input_builder.py](/D:/Github/WFBARNet/src/builders/bst_input_builder.py)

---

## 21. 当前说明

当前你看到没有 Web 前端页面是正常的，因为项目方向已经不应该再按浏览器前端继续做。

接下来如果采用 `DearPyGui + OpenCV`，那么正确方向应该是：

- 做一个本地桌面 GUI
- 直接调用 Python 推理模块
- 在桌面窗口中显示视频与结果

而不是继续去开发 React 页面。

