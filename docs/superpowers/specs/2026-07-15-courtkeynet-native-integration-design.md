# CourtKeyNet 原生集成设计

## 状态

- 日期：2026-07-15
- 决策：采用原生源码集成，CourtKeyNet 替换 CourtPose 成为默认球场检测第一后端。
- 用户确认：CourtKeyNet 不做 MonoTrack/OpenCV 白线二次校验；连续 3 个真实推理帧几何一致后，才允许进入轨迹、姿态和统计。

## 背景

WFBARNet 当前自动球场标定按 `court_pose -> shuttlecourt_seg -> monotrack -> opencv` 降级，并把可显示的黄色草稿与可用于测量的绿色可信标定分开。CourtKeyNet 已使用官方微调 SafeTensors 权重在 RTX 4070 Laptop GPU 上完成独立验证，模型能输出 4 个归一化球场角点和 4 张热力图。

直接引用 `D:\Github\CourtKeyNet` 会形成机器路径依赖；原项目推理入口还会通过 `utils/__init__.py` 引入与推理无关的训练依赖。因此本次只移植 MIT 许可下的最小模型架构，使用标准 `safetensors` API 加载权重，不复用原项目 GUI、训练代码或混淆权重加载器。

## 目标

1. WFBARNet 可在没有外部 CourtKeyNet 仓库的机器上独立运行 CourtKeyNet 推理。
2. 默认自动后端顺序改为 `courtkeynet -> shuttlecourt_seg -> monotrack -> opencv`。
3. CourtKeyNet 单帧结果先作为黄色草稿；连续 3 帧相对同一锚点稳定后升级为绿色可信标定。
4. 确认后的几何保持锁定，直到切换输入源、人工覆盖或显式重新检测。
5. 权重、依赖或推理异常不能阻断 GUI，必须继续尝试后续后端。
6. 保留 CourtPose 的公共工厂入口，避免破坏已有脚本，但不再放入默认自动链路。

## 非目标

- 不训练或微调 CourtKeyNet。
- 不把 CourtKeyNet 转为 ONNX、TensorRT 或 TorchScript。
- 不承诺单帧模型置信度等价于白色双打外线准确率。
- 不删除现有 CourtPose、ShuttleCourt、MonoTrack 或 OpenCV 实现。
- 不重构球场检测以外的轨迹、姿态、统计和报告模块。

## 方案比较

### 采用：原生源码与 SafeTensors

将上游最小模型模块放入 WFBARNet，使用标准 PyTorch 和 `safetensors.torch.load_file`。该方案自包含、可测试、错误边界清楚，也能保留与官方权重完全一致的 state dict 名称。

### 不采用：外部仓库适配器

运行时从环境变量或固定目录加载 CourtKeyNet 源码，初始改动较少，但部署到其他机器后容易因路径、版本或依赖差异失效，不适合作为默认后端。

### 不采用：导出模型格式

TorchScript/ONNX 可减少源码移植，但 Polar Attention、Transformer 和 `grid_sample` 需要额外导出一致性验证，也会引入另一份派生权重。本轮没有必要承担该风险。

## 文件与组件

### 模型架构

新增 `src/court/courtkeynet_model/`：

- `model.py`：CourtKeyNet 主网络、soft-argmax 和四角 refinement。
- `octave.py`：Octave Feature Extractor。
- `polar.py`：Polar Transform Attention。
- `qcm.py`：Quadrilateral Constraint Module。
- `__init__.py`：只导出推理所需类型。
- `LICENSE`：保留上游 MIT 许可证和版权归属。

移植代码保持官方 state dict 键名，不引入训练数据集、WandB、Albumentations 或 GUI 依赖。

### 检测适配器

新增 `src/court/courtkeynet_detector.py`：

- `CourtKeyNetConfig`：权重、输入尺寸、设备、置信度阈值和三帧确认参数。
- `resolve_courtkeynet_weights(...)`：只解析显式路径和项目默认权重。
- `CourtKeyNetLineDetector`：实现现有 `CourtLineDetector` 协议。
- 置信度计算函数：按官方 GUI 公式输出热力图峰值、几何有效性、熵和组合分数。
- 状态机：管理确认锚点、确认计数、锁定预测和 reset。

默认配置：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `weights` | `assets/weights/courtkeynet/CourtKeyNet.safetensors` | 官方微调权重 |
| `imgsz` | `640` | 与官方推理一致的方形输入 |
| `confidence_threshold` | `0.50` | 官方 GUI 默认组合置信度阈值 |
| `confirmation_frames` | `3` | 升级为可信标定所需的新鲜观察数 |
| `max_corner_shift_ratio` | `0.035` | 相对原图对角线的最大角点偏移 |
| `device` | 自动 | CUDA 可用时使用 CUDA，否则使用 CPU |

### 统一工厂与自动链路

修改：

- `src/court/court_line_detector.py`：加入 `courtkeynet` backend、配置和工厂分支。
- `src/court/__init__.py`：导出 CourtKeyNet 配置与检测器。
- `src/court/batch_court.py`：默认后端改为 CourtKeyNet 优先；锁定 fallback 只允许配置的权威后端升级。
- `apps/pyqt6/services/automatic_court_calibration_service.py`：默认链路改为 `courtkeynet, shuttlecourt_seg, monotrack, opencv`，权威升级后端设为 `courtkeynet`。

`BatchCourtPredictor` 增加显式 `authoritative_backends` 配置。默认值继续兼容旧 CourtPose 行为；自动标定服务传入 `("courtkeynet",)`，避免任意高优先级候选覆盖已锁定的可信 fallback。

### 权重与依赖

- 权重目标：`assets/weights/courtkeynet/CourtKeyNet.safetensors`。
- `requirements.txt` 增加 `safetensors>=0.5`。
- `assets/weights/README.md` 增加 CourtKeyNet 下载和文件名说明。
- 权重继续遵循仓库现有 Release 分发规则，不提交到普通 Git 历史。

## 推理与数据流

1. 自动服务向 `BatchCourtPredictor` 提交预览样本。
2. CourtKeyNet 首次使用时延迟加载模型和权重，并选择 CUDA 或 CPU。
3. 输入帧缩放为 640 x 640、BGR 转 RGB、归一化到 `[0, 1]`。
4. 模型输出 `heatmaps` 和 `kpts_refined`；归一化角点按原始帧宽高映射回像素坐标。
5. 使用标准球场模板计算单应性、内部场线和现有公共透视几何指标，不执行白线 refinement。
6. 计算官方组合置信度：
   - 热力图峰值置信度：每个热力图 softmax 峰值按 `[0.05, 0.40]` 归一化。
   - 几何有效性：凸性 0.30、面积 0.20、宽高比 0.20、角点相对位置 0.30。
   - 熵置信度：`1 - mean(entropy / log(H * W))`。
   - 组合分数：`0.40 * peak + 0.40 * geometry + 0.20 * entropy`。
7. 分数低于 0.50、角点非有限、非凸或公共透视几何不合理时，不计入确认；若仍有可用四角，则返回黄色 provisional 草稿。
8. 合格的新鲜观察与第一帧确认锚点比较。四个角中任一角偏移超过原图对角线的 3.5%，确认重置为当前帧的 `1/3`，防止逐帧小漂移累积通过。
9. 第三个一致观察输出 `valid=True`、`scheme="courtkeynet"` 和 `courtkeynet_confirmation_complete=1`，并锁定四角及单应性。
10. 锁定后返回同一几何，状态写明 `locked trusted calibration`；不再用后续模型结果漂移标定。
11. CourtKeyNet 尚未确认时，Batch 仍可返回后续后端的可信结果；每 750 ms 重查 CourtKeyNet，第三次确认后由权威后端升级。

## 可信门控

`is_trusted_automatic_court_prediction(...)` 对 `scheme="courtkeynet"` 要求同时满足：

- `prediction.valid`；
- `courtkeynet_confirmation_complete >= 1`；
- 组合置信度不低于配置阈值；
- 现有公共透视几何门控通过。

这里不要求 singles/outer 白线支持度。该行为是用户明确选择的“直接替换 CourtPose”，风险由三帧锚点一致性、组合置信度和公共透视几何共同控制。

## 状态与生命周期

- `reset()` 清除模型以外的所有视频状态：最近预测、锚点、计数和锁定结果。
- 模型实例可以保留以避免切源重复加载；权重或设备配置改变时创建新 detector。
- `attempted=True, updated=True` 只用于本帧真正执行模型的结果。
- 缓存复用和锁定复用使用 `attempted=False, updated=False`，不能增加确认计数。
- 人工四点标定仍立即成为最高优先级可信结果，并使自动服务 generation 失效。

## 错误处理

- 权重不存在：抛出包含期望绝对路径的 `FileNotFoundError`，Batch 记录错误并继续 fallback。
- 缺少 `safetensors`：抛出明确的依赖错误，不尝试 pickle 或不安全反序列化。
- state dict 不匹配：严格加载失败并报告缺失/多余键，不以 `strict=False` 静默运行。
- CUDA 运行错误：本次后端失败并继续 fallback；不自动修改环境或下载不同 PyTorch。
- 无法读取/缩放帧或输出含 NaN/Inf：返回 invalid，清除未完成确认，但不杀死 Qt worker。

## 测试设计

新增 `tests/test_courtkeynet_detector.py`，使用小型 fake model/state 覆盖：

1. 工厂创建与错误配置类型。
2. 640 x 640 预处理和非方形原图坐标映射。
3. 官方三项置信度及组合权重。
4. 低分候选保持 provisional，且不增加确认计数。
5. 三个一致的新鲜观察按 `1/3 -> 2/3 -> valid` 升级。
6. 每帧只相对首个锚点比较，累计漂移不能通过。
7. 缓存复用不增加确认计数。
8. 确认后几何锁定，不继续漂移。
9. `reset()` 清理确认和锁定状态。
10. 缺权重、state dict 不匹配和模型异常可被 Batch fallback 捕获。

修改现有测试覆盖：

- 默认后端顺序和权重存在性判断。
- CourtKeyNet 可信门控。
- 已锁定 fallback 只能被配置的 CourtKeyNet 权威后端升级。
- 自动服务确实创建 CourtKeyNet 第一后端。
- CourtPose 工厂仍可显式使用。

真实验证：

- 使用项目 Conda 环境和官方微调权重执行单帧 GPU 冒烟测试，要求 state dict 严格匹配、输出形状正确、全部数值有限。
- 对仓库自带或现有评估视频按 750 ms 采样至少 6 帧，确认三帧状态能在实际服务链路中升级。
- 运行现有跨场馆样本，保存 CourtKeyNet 候选、最终后端和联系表，人工检查明显错场。
- 运行全量单元测试、`compileall` 和 `git diff --check`。

## 验收标准

1. 默认 GUI 自动标定不再实例化 CourtPose，首个模型后端为 CourtKeyNet。
2. 新环境只需 WFBARNet 源码、规定权重和 requirements 依赖即可运行，不引用外部 CourtKeyNet 路径。
3. 单帧 CourtKeyNet 结果永远不会直接成为绿色可信标定。
4. 三个相对首锚点一致的真实观察可以升级；缓存复用、低分帧和累计漂移不能升级。
5. 锁定后角点不随视频帧变化，切源或人工覆盖后旧结果不会串用。
6. CourtKeyNet 故障时后续球场后端和 GUI 仍能工作。
7. 全量测试通过，并产出可检查的真实运行预览和指标记录。

## 已知风险

- CourtKeyNet 的组合置信度来自模型热力图与一般四边形几何，不证明角点贴合白色双打外线。
- 系统性稳定误识别可能通过三帧一致性，因此跨场馆结果必须保留人工复核能力。
- 640 x 640 非等比缩放与官方实现一致，但可能损失极端宽高比画面的细节；本轮不改变训练时输入协议。
- CPU 可运行但速度可能不足以实时处理；后台样本间隔和 fallback 可避免阻塞主界面。
