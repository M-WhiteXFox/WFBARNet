# WFBARNet

一个面向羽毛球视频分析的工程化项目骨架，当前重点是把“人体姿态估计 + 羽毛球轨迹提取”统一到同一个 PyTorch 推理流程里，并为后续动作识别、前后端联调和系统化开发留出清晰的目录边界。

当前仓库已经整理成“主推理工程 + 应用层 + 资源层 + 工具层”的结构，适合继续扩展：

- 统一推理主线：`main.py` + `src/`
- 前后端原型：`apps/badminton-analysis-system/`
- 模型权重与资料：`assets/`
- 配置文件：`configs/`
- 辅助脚本与 demo：`tools/`

## 项目结构

```text
WFBARNet/
├── main.py
├── requirements.txt
├── requirements_mmpose.txt
├── README.md
├── src/
│   ├── models/
│   ├── preprocess/
│   ├── postprocess/
│   ├── runners/
│   └── utils/
├── configs/
│   └── default_infer.json
├── assets/
│   ├── weights/
│   │   ├── bst/
│   │   ├── pose/
│   │   └── track/
│   └── docs/
│       └── papers/
├── tools/
│   ├── demo/
│   └── mmpose/
├── apps/
│   └── badminton-analysis-system/
│       ├── backend/
│       └── frontend/
└── outputs/
```

## 各目录职责

### `src/`

统一推理主工程代码，后续所有核心视觉与时序前处理逻辑都建议放在这里。

- `src/models/`
  - 模型封装层
  - 当前包含：
    - `pose_branch.py`：姿态分支统一接口
    - `mmpose_backend.py`：MMPose 风格后端封装
    - `track_branch.py`：轨迹分支统一接口
    - `tracknet_v3.py`：简化版 TrackNetV3 风格网络
- `src/preprocess/`
  - 输入预处理
  - 当前包含单帧姿态预处理、三帧轨迹窗口预处理
- `src/postprocess/`
  - 输出后处理
  - 当前包含关键点坐标恢复、heatmap 球中心解码
- `src/runners/`
  - 统一调度逻辑
  - 当前 `unified_runner.py` 负责串行执行和可选双 CUDA Stream 执行
- `src/utils/`
  - 通用工具
  - 当前包含：
    - `structures.py`：统一数据结构
    - `video.py`：视频/帧序列读取与三帧窗口生成
    - `exporters.py`：`json/csv/npy` 导出
    - `visualize.py`：可视化视频生成

### `configs/`

放项目配置。

- `default_infer.json`
  - 统一推理默认配置样例
  - 用于记录姿态、轨迹、输出等路径参数

### `assets/`

放模型资源和资料，避免和源码混在一起。

- `assets/weights/`
  - 模型权重目录
  - 建议按任务划分：
    - `bst/`
    - `pose/`
    - `track/`
- `assets/docs/papers/`
  - 文献、论文 PDF、方法说明资料

### `tools/`

放辅助脚本和开发期工具。

- `tools/demo/`
  - 命令行 demo 入口
  - 当前为 `run_unified_infer.py`
- `tools/mmpose/`
  - MMPose 相关辅助内容
  - 当前包含：
    - config 样例
    - 权重下载提示脚本
    - README 说明

### `apps/`

放更上层的应用或系统原型，不和底层推理代码混放。

当前已有：

- `apps/badminton-analysis-system/`
  - 一个面向“任务提交 - 状态轮询 - 结果展示”的系统原型目录
  - 结构分成：
    - `frontend/`
    - `backend/`

其中：

- `frontend/`
  - React + TypeScript + Tailwind 前端
  - 已按组件化结构整理：
    - `src/components/Layout/`
    - `src/components/Dashboard/`
    - `src/hooks/`
    - `src/api/`
    - `src/types/`
- `backend/`
  - 当前只保留占位结构
  - 暂未实现正式后端业务逻辑

### `outputs/`

放推理结果和可视化导出物。

建议输出内容包括：

- `unified_results.json`
- `unified_results.csv`
- `unified_results.npy`
- `unified_vis.mp4`

## 当前主入口

### 1. Python 推理入口

根目录入口：

- [main.py](/D:/Github/WFBARNet/main.py)

这个入口适合直接在文件里改参数后运行。

推荐流程：

1. 打开 `main.py`
2. 修改 `USER_CONFIG`
3. 设置：
   - `source`
   - `pose_weight`
   - `track_weight`
   - 如果用 MMPose，还要确认 `pose_config`
4. 运行：

```powershell
python .\main.py
```

### 2. 命令行 demo 入口

- [tools/demo/run_unified_infer.py](/D:/Github/WFBARNet/tools/demo/run_unified_infer.py)

适合脚本方式运行：

```powershell
python .\tools\demo\run_unified_infer.py --source "your_video.mp4"
```

### 3. 前端入口

- [apps/badminton-analysis-system/frontend/src/App.tsx](/D:/Github/WFBARNet/apps/badminton-analysis-system/frontend/src/App.tsx)

如果后面要启动前端原型，可以进入：

```powershell
cd .\apps\badminton-analysis-system\frontend
npm install
npm run dev
```

## 依赖安装

### 基础推理依赖

```powershell
pip install -r requirements.txt
```

### MMPose 额外依赖

```powershell
pip install -r requirements_mmpose.txt
```

注意：

- `mmpose / mmcv / mmdet / mmengine` 在实际安装时经常需要按 CUDA、PyTorch 版本单独对齐。
- 如果普通 `pip` 安装失败，后续建议改成 `openmim` 方式安装。

## 当前统一推理流程

当前主工程的设计目标是：

1. 读取视频或帧序列
2. 逐帧解码
3. 为轨迹分支构建三帧窗口 `(t-1, t, t+1)`
4. 同时送入：
   - `PoseBranch`
   - `TrackBranch`
5. 按 `frame_id` 对齐结果
6. 导出统一格式数据

统一输出核心结构为：

```json
{
  "frame_id": 12,
  "pose": [
    {
      "person_id": 0,
      "bbox": [x1, y1, x2, y2],
      "keypoints": [[x, y], [x, y]],
      "scores": [0.9, 0.8],
      "person_score": 0.85
    }
  ],
  "track": {
    "ball_xy": [320.5, 114.2],
    "visible": 1,
    "score": 0.93
  }
}
```

这个格式是为后续时序模型准备的，适合继续接：

- ST-GCN
- BST
- 自定义动作识别模型

## 当前状态说明

目前这份仓库属于“可继续开发的整理版骨架”，不是最终完整生产版。你需要注意这些点：

- `src/` 里的统一推理主线已经恢复并可导入。
- `main.py` 当前可以正常 `import`。
- 前端结构已经恢复并按项目风格拆分。
- 后端目前只有占位目录，还没有接回真实业务接口。
- `assets/weights/` 目前需要你自己重新放入真实模型权重。
- `assets/docs/papers/` 目前也需要你自己恢复本地 PDF。

## 建议的后续开发顺序

建议按下面顺序继续推进：

1. 先恢复真实模型权重到 `assets/weights/`
2. 把 `PoseBranch` 替换成你最终要用的 MMPose / RTMPose 配置与权重
3. 把 `TrackBranch` 对齐到你实际的 TrackNetV3 checkpoint
4. 跑通 `main.py` 的完整视频推理
5. 确认 `json/csv/npy` 输出格式满足 BST / ST-GCN 输入需求
6. 再把后端接口和前端工作台接起来

## 你现在最常用的几个路径

- 根入口：[main.py](/D:/Github/WFBARNet/main.py)
- 默认配置：[configs/default_infer.json](/D:/Github/WFBARNet/configs/default_infer.json)
- 推理调度器：[src/runners/unified_runner.py](/D:/Github/WFBARNet/src/runners/unified_runner.py)
- 姿态分支：[src/models/pose_branch.py](/D:/Github/WFBARNet/src/models/pose_branch.py)
- 轨迹分支：[src/models/track_branch.py](/D:/Github/WFBARNet/src/models/track_branch.py)
- 前端首页：[apps/badminton-analysis-system/frontend/src/App.tsx](/D:/Github/WFBARNet/apps/badminton-analysis-system/frontend/src/App.tsx)

