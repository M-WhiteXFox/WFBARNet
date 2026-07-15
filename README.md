<h1>
  <img src="assets/readme/app.ico" alt="WFBARNet" width="44" height="44">
  WFBARNet
</h1>

WFBARNet 是一款面向羽毛球训练与比赛复盘的本地视频分析工具。你可以把比赛录像、训练片段或摄像头画面导入系统，查看羽毛球轨迹、球员移动、击球事件、场地区域分布和训练报告，帮助教练、运动员和研究者更快整理视频中的关键信息。

<p align="center">
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Framework-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green"></a>
  <a href="https://www.bilibili.com/video/BV12VM56uE6o/"><img alt="视频介绍" src="https://img.shields.io/badge/Bilibili-视频介绍-00A1D6?logo=bilibili&logoColor=white"></a>
</p>

## 效果预览

![观看 WFBARNet 演示视频](assets/readme/main.gif)

## 主要能力

- 视频分析：导入本地视频后，自动识别羽毛球、球员和球场。
- 摄像头分析：支持使用摄像头进行实时画面分析。
- 手动球场标定：当自动识别不够稳定时，可以手动点选四个角完成标定。
- 实时预览：在桌面界面中查看轨迹、姿态、球场线和统计结果。
- 数据统计：展示回合状态、击球事件、球速估计、移动距离和区域分布。
- 热力图：把球员活动和落点分布展示在羽毛球场平面上。
- 训练报告：导出可浏览的 HTML 报告，便于保存、分享和复盘。

## 界面与使用方式

启动后会进入桌面图形界面。常见流程如下：

1. 打开视频、选择摄像头，或进入批量分析。
2. 等待系统完成自动球场标定。视频预览会在约 3.75 秒内抽取最多 6 帧，按 CourtKeyNet -> ShuttleCourt -> MonoTrack -> OpenCV 依次尝试四种后端；系统会检查透视形状、模型置信度和连续帧一致性。未通过验证但具备完整几何的结果以黄色虚线显示为“待确认草稿”，通过验证后升级为绿色可信标注。黄色草稿不会用于球速、落点和移动距离等几何统计，可直接拖动角点修正；也可按左上、右上、右下、左下（TL、TR、BR、BL）的顺序重新点选四个角。CourtPose 保留为仅显式选择时使用的兼容后端，不在默认自动链路中。
3. 点击开始分析，等待系统识别轨迹、球员和回合事件。
4. 在右侧面板查看概览、数据、统计、姿态、报告和日志。
5. 分析结束后导出报告或查看输出文件。

如果模型权重齐全，系统会提供完整分析能力；如果缺少部分可选权重，界面仍可启动，并会在开始分析时提示缺失内容。

## 快速开始

推荐使用 Conda 创建独立环境：

```powershell
conda env create -f environments.yml
conda activate WFBARNet
```

如果你已经准备好 Python 3.10 环境，也可以直接安装依赖：

```powershell
pip install -r requirements.txt
```

启动桌面应用：

```powershell
python main.py
```

推荐运行环境：

- Python 3.10
- Windows
- 支持 CUDA 的 NVIDIA 显卡

CPU 也可以启动和运行部分流程，但视频分析速度会明显下降。

## 模型权重

模型权重默认放在 `assets/weights/` 目录下：

```text
assets/weights/
├─ track/
├─ pose/
├─ bst/
├─ courtkeynet/
│  └─ CourtKeyNet.safetensors
└─ court_pose/              # 显式兼容后端
```

权重文件不随仓库一起发布，也不会在启动时自动下载。请根据自己的使用场景，把对应模型文件放到上述目录。模型文件请到Releases下载。

更多说明见 [assets/weights/README.md](assets/weights/README.md)。

## 输出内容

分析后常见输出包括：

- 可视化视频：叠加轨迹、球场线或姿态结果的视频。
- 结构化数据：JSON、CSV、NPY 等文件，便于进一步整理或统计。
- 调试日志：用于排查某段视频为什么识别不稳定。
- HTML 报告：包含关键指标、事件摘要、热力图和训练建议。

默认输出目录通常位于 `outputs/`，报告模板位于 [assets/report_template](assets/report_template)。

## 命令行分析

除了桌面界面，也可以通过命令行执行一次性分析：

```powershell
python main.py --config configs/default_infer.json --source .\demo.mp4 --pipeline unified --output-dir outputs/run --device auto
```

常用参数：

- `--source`：输入视频路径或摄像头编号；为空时进入桌面界面。
- `--output-dir`：输出目录。
- `--device`：运行设备，可使用 `auto` 让程序自动选择。
- `--no-vis`：不导出可视化视频，只保留数据结果。

多数用户建议优先使用桌面界面；命令行更适合批量处理或自动化流程。

## 常见问题

**启动后没有立刻报错，但分析开始失败怎么办？**

通常是缺少某个模型权重。请查看界面日志或状态栏提示，并确认 `assets/weights/` 下的文件路径是否正确。

**一定需要 GPU 吗？**

不一定。CPU 可以运行，但视频分析速度会比较慢。实时分析和较长视频更推荐使用 NVIDIA GPU。

**BST 权重缺失会影响使用吗？**

不会影响基础轨迹、姿态、球场和报告流程。缺少 BST 权重时，只会跳过击球动作类型分类。

**为什么需要手动标定球场？**

不同场馆的机位、光照和遮挡差异很大。手动标定可以帮助系统更准确地把画面位置对应到标准羽毛球场。

## 项目结构

```text
WFBARNet/
├─ apps/pyqt6/              # 桌面应用
├─ assets/                  # 权重、模板和文档资源
├─ configs/                 # 运行配置
├─ src/                     # 核心分析代码
├─ tests/                   # 测试用例
├─ tools/                   # 演示、导出和辅助脚本
├─ outputs/                 # 默认输出目录
└─ main.py                  # 启动入口
```

## 鸣谢

本项目的开发过程中使用、修改或参考了以下优秀的开源项目，在此向相关项目的作者及贡献者表示感谢：

- [TrackNetV3](https://github.com/qaz812345/TrackNetV3)：羽毛球轨迹识别。
- [Ultralytics](https://github.com/ultralytics/ultralytics)：YOLO 姿态识别。
- [CourtKeyNet](https://github.com/adithyanraj03/CourtKeyNet)：原生球场关键点架构与权重格式；本项目固定源提交为 `f852db65b5d435db16f3c624d2d51dc78b903705`，按 MIT License 使用并保留上游归属。
- [TrackNetV3 INT8 Optimization](https://github.com/nickluo/TrackNetV3)：TrackNetV3 INT8 优化。
- [BST Badminton Stroke Type Transformer](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer)：击球动作类型识别。

## 开源协议

本仓库采用 **Apache License 2.0** 协议开源，完整文本见 [LICENSE](LICENSE)。

本项目所使用、引用或修改的第三方软件、模型及代码仍遵循其各自的原始许可证。使用者在使用本项目时，应同时遵守相关第三方项目的许可证要求。
