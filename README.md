# WFBARNet

WFBARNet 是一个面向羽毛球视频分析的本地智能分析系统，目标是把比赛、训练或摄像头画面转成可复盘、可量化、可导出的结构化数据。项目围绕“羽毛球、球员、球场”三类目标组织分析流程，提供轨迹跟踪、姿态估计、球场映射、击球事件识别、动作分类、热力图统计和 HTML 报告导出能力。

这个仓库更偏向“可调试的工程化分析平台”，而不是单一论文复现。它适合做训练复盘、算法验证、批量视频整理和后续模型实验，也保留了桌面 GUI、调试导出和单元测试等工程组件。

本仓库以 [Apache License 2.0](LICENSE) 协议开源。

## 项目亮点

- 本地运行：支持桌面端 PyQt6 图形界面，不依赖在线服务即可完成核心分析。
- 多模态融合：联合使用球轨迹、人体姿态和球场单应性，避免只看单一视觉分支。
- 可解释后处理：轨迹修复、击球候选、落点事件和回合统计使用可检查的规则式流程。
- 可扩展模型接入：当前集成 TrackNetV3 风格轨迹分支、YOLO pose、可选 MMPose、BST 风格时序动作识别。
- 面向分析输出：支持 JSON、CSV、NPY、调试 CSV、可视化视频和 HTML 报告。

## 核心功能

### 1. 羽毛球轨迹跟踪

系统使用 TrackNetV3 风格的三帧窗口轨迹模型定位羽毛球，并结合候选点解码、轨迹修复和滤波输出逐帧球点结果。

对应实现：

- `src/models/tracknet_v3.py`
- `src/models/track_branch.py`
- `src/postprocess/tracknet_v3_filter.py`

### 2. 球员姿态估计

系统默认使用 YOLO pose 后端检测人体框和关键点，也保留了 MMPose 接入能力。姿态结果用于球员身份稳定、脚下位置估计、移动距离统计和 BST 动作识别输入构建。

对应实现：

- `src/models/pose_branch.py`
- `src/models/yolo_pose_backend.py`
- `src/models/mmpose_backend.py`

### 3. 球场线检测与单应性映射

系统通过球场线检测或手动四点标定恢复标准羽毛球场模板与图像平面的映射关系，使球点与球员位置可以投影到真实球场坐标系。

对应实现：

- `src/court/`
- `apps/pyqt6/services/manual_court_calibration_service.py`

### 4. 轨迹事件识别

系统基于轨迹速度、方向、局部极值和可见性变化识别 `hit`、`landing`、`out_of_frame` 等关键事件，作为回合统计和动作识别的上游输入。

对应实现：

- `src/postprocess/trajectory_events.py`
- `src/postprocess/rally_stats.py`

### 5. BST 动作识别

当本地提供 BST 权重时，系统会围绕击球事件截取时序片段，组合人体姿态、羽毛球位置和球员场地位置进行动作分类。没有 BST 权重时，其余分析流程仍可独立运行。

对应实现：

- `src/models/bst_model.py`
- `src/models/bst_runtime.py`
- `src/models/bst_stroke_runtime.py`
- `src/builders/bst_input_adapter.py`

### 6. 可视化与报告输出

项目包含桌面 GUI、回合统计面板、球场热力图、日志区、HTML 报告生成和结构化结果导出。

对应实现：

- `apps/pyqt6/`
- `src/utils/user_report.py`
- `src/utils/report_generation.py`
- `src/utils/exporters.py`

## 项目结构

```text
WFBARNet/
├─ apps/pyqt6/                # PyQt6 桌面应用
├─ configs/                   # 运行配置示例
├─ src/
│  ├─ builders/               # BST 输入构造
│  ├─ court/                  # 球场检测与单应性
│  ├─ models/                 # 轨迹、姿态、BST 等模型封装
│  ├─ postprocess/            # 轨迹修复、事件识别、统计
│  ├─ preprocess/             # 输入预处理
│  ├─ runners/                # 视频/实时/统一推理入口
│  └─ utils/                  # 导出、报告、可视化、设备工具
├─ tests/                     # 单元测试
├─ tools/                     # demo、导出、实验脚本
├─ assets/weights/            # 本地权重放置目录
└─ main.py                    # 默认入口
```

## 快速开始

### 环境准备

推荐环境：

- Python `3.10`
- Windows + CUDA GPU 更适合桌面实时分析
- CPU 也可运行，但实时性会明显下降

安装依赖：

```powershell
pip install -r requirements.txt
```

如果你需要 MMPose、TensorRT 或其他扩展后端，请按本地 CUDA/驱动环境单独安装对应依赖。

### 模型权重

项目默认从以下目录读取权重：

- `assets/weights/track/`
- `assets/weights/pose/`
- `assets/weights/bst/`
- `assets/weights/ShuttleCourtNet/`

当前仓库中的权重组织方式可参考 [assets/weights/README.md](assets/weights/README.md)。

### 启动图形界面

默认启动桌面应用：

```powershell
python main.py
```

`main.py` 在 `source` 为空时会直接进入 PyQt6 图形界面。

### 运行测试

```powershell
pytest
```

## 典型工作流

### 1. 桌面端视频分析

```powershell
python main.py
```

适合：

- 手动选择视频
- 预览轨迹、姿态、球场叠加
- 查看统计、日志和报告
- 手动标定球场

### 2. 独立 Demo 与排查脚本

常用脚本位于 `tools/demo/`：

- `run_track_only.py`
- `run_pose_only.py`
- `run_unified_infer.py`
- `run_tracknet_realtime.py`
- `run_court_keypoints_yolo.py`

说明文档见 [tools/demo/README.md](tools/demo/README.md)。

### 3. 导出 TrackNet TensorRT Engine

```powershell
python tools/export_tracknetv3_int8_engine.py --help
```

该脚本用于把本地 TrackNet 权重导出为 ONNX / TensorRT 相关产物，适合部署或性能实验。

## 上游参考仓库

WFBARNet 不是以下项目的官方仓库，但当前工程实现明显参考或集成了它们的思路，公开仓库时建议一并说明来源：

### TrackNetV3

- 仓库：`qaz812345/TrackNetV3`
- 链接：https://github.com/qaz812345/TrackNetV3

当前仓库中的对应位置：

- `src/models/tracknet_v3.py`
- `src/postprocess/tracknet_v3_filter.py`
- `tools/export_tracknetv3_int8_engine.py`

说明：

- 本项目实现的是 TrackNetV3 风格轨迹分支与后处理集成版本。
- 当前工程还加入了自己的轨迹修复、候选筛选、导出和 GUI 分析链路。

### BST

- 仓库：`Va6lue/BST-Badminton-Stroke-type-Transformer`
- 链接：https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer

当前仓库中的对应位置：

- `src/models/bst_model.py`
- `src/models/bst_runtime.py`
- `src/models/bst_stroke_runtime.py`
- `src/builders/bst_input_adapter.py`

说明：

- 当前仓库提供 BST 风格动作识别模型封装和推理接入。
- 系统可在击球事件附近构建时序输入，并把分类结果并入回合统计与报告。

## 输出结果

项目常见输出包括：

- `outputs/run/` 下的 JSON、CSV、NPY 和可视化视频
- 调试 CSV 与逐帧事件日志
- `assets/dist/` 或用户报告目录下的 HTML 报告
- BST 输入导出，如 `bst_input.npy`

这些输出适合做复盘、回放、误检排查和后续数据分析。

## 当前状态

当前仓库更接近“持续迭代中的研究工程项目”，而不是已经完全产品化的 SDK。你在使用前最好了解以下边界：

- 分析结果强依赖视频质量、球场角度、遮挡情况和模型权重质量。
- 部分功能默认依赖本地权重，不保证开箱即跑。
- GUI、报告模板和实验脚本目前以 Windows 本地工作流为主。
- 实时性能取决于 GPU、输入分辨率和姿态/轨迹推理频率设置。
- 部分模型和流程是“工程整合版本”，不等同于上游论文仓库的原始训练/评测代码。

## 开发说明

如果你准备把这个仓库继续整理成更标准的开源项目，建议优先补齐以下内容：

- 增加 `LICENSE`
- 增加示例输入和最小可运行配置
- 把模型下载方式脚本化
- 补充中英文 README 与截图
- 为 GUI、CLI 和导出链路提供更明确的使用范例

## 开源协议

本仓库采用 **Apache License 2.0** 协议开源，完整文本见 [LICENSE](LICENSE)。

说明：

- 仓库自身代码按 Apache-2.0 分发。
- README 中列出的 TrackNetV3 与 BST 上游仓库目前均显示为 MIT License，本仓库保留其来源说明以便读者追溯实现背景。
- 如果后续继续引入第三方代码、模型权重或数据集，请在分发前再次核对对应许可证、数据使用条款和署名要求。
