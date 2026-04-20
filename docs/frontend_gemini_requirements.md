# 羽毛球动作分析系统前端需求文档

## 1. 文档目的

本文档用于指导 `Gemini` 或其他前端开发代理实现本项目的前端页面与交互逻辑。

目标不是只做一个静态展示页，而是做一个可持续扩展的“羽毛球动作分析工作台”，用于承接后端异步分析任务，并展示以下结果：

- 视频上传与任务提交
- 分析进度与日志轮询
- 人体姿态结果展示
- 羽毛球轨迹结果展示
- 动作识别时间轴展示
- 分析摘要展示
- 后续 BST / 时序模型输入与结果联动能力

当前项目中前端目录已存在于：

- [apps/badminton-analysis-system/frontend](/D:/Github/WFBARNet/apps/badminton-analysis-system/frontend)

但页面尚未真正完成可运行交付，因此需要按本文档重新实现或完善。

## 2. 项目背景

本项目是一个羽毛球视频分析系统，后端会对输入视频执行以下视觉任务：

1. 人体姿态估计
2. 羽毛球轨迹提取
3. 动作识别
4. 结果融合与导出

后端采用异步分析模式，不是“上传后同步等待页面卡住”的方式，而是：

1. 用户上传视频
2. 后端返回 `file_id`
3. 用户发起分析任务
4. 后端返回 `task_id`
5. 前端轮询任务状态
6. 任务完成后拉取最终结果

因此前端必须围绕“任务提交 - 轮询 - 渲染结果”的流程设计。

## 3. 前端开发目标

需要实现一个现代化、工程化、可扩展的前端页面，满足以下要求：

- 基于 `React + TypeScript + Vite + Tailwind CSS`
- 页面结构清晰，组件拆分合理
- 支持异步任务轮询
- 能展示视频、日志、分析摘要、动作时间轴、姿态与轨迹信息
- 后续可以很方便接入真实后端 API
- UI 风格偏“专业分析平台”，不是简单 demo

## 4. 建议技术栈

- 框架：`React 18`
- 语言：`TypeScript`
- 构建工具：`Vite`
- 样式：`Tailwind CSS`
- 图标：`lucide-react`
- HTTP 请求：`axios`
- 状态管理：
  - 初版可以使用 `React Hooks`
  - 如果实现更复杂的数据流，可以用 `Zustand`
- 图表库：
  - 可选 `recharts` 或 `echarts-for-react`
  - 如果时间轴不复杂，也可以先纯 CSS + div 实现

## 5. 目录结构要求

前端目录保持如下结构：

```text
frontend/
├── src/
│   ├── api/
│   │   └── client.ts
│   ├── components/
│   │   ├── Layout/
│   │   │   ├── Sidebar.tsx
│   │   │   └── Header.tsx
│   │   ├── Dashboard/
│   │   │   ├── VideoPanel.tsx
│   │   │   ├── SummaryCards.tsx
│   │   │   ├── ActionTimeline.tsx
│   │   │   ├── TrackPosePanel.tsx
│   │   │   └── LogPanel.tsx
│   ├── hooks/
│   │   └── useAnalysisTask.ts
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── postcss.config.js
└── vite.config.ts
```

要求：

- 页面逻辑不要全部堆在 `App.tsx`
- API 请求与页面展示分层
- 类型定义单独抽离
- 轮询逻辑写在自定义 Hook 中

## 6. 核心用户流程

### 6.1 主流程

用户进入页面后，理想交互流程如下：

1. 进入“视频分析工作台”
2. 选择本地视频文件
3. 页面展示视频基本信息
4. 点击“开始分析”
5. 页面显示任务状态
6. 页面轮询进度并实时刷新日志
7. 任务完成后自动拉取结果
8. 页面展示：
   - 摘要卡片
   - 动作时间轴
   - 轨迹与姿态信息
   - 视频或分析预览面板
9. 用户可切换查看不同时间段、不同帧结果

### 6.2 状态流转

页面至少要支持这些状态：

- `idle`
  - 初始状态，未上传文件
- `file_selected`
  - 已选择本地视频
- `uploading`
  - 上传中
- `uploaded`
  - 上传成功，已获得 `file_id`
- `submitting`
  - 正在创建分析任务
- `pending`
  - 任务已提交，等待进入处理
- `processing`
  - 正在分析中
- `completed`
  - 分析完成
- `failed`
  - 分析失败

要求：

- 不同状态对应不同按钮文案和禁用策略
- 页面不能出现“点击了没反应”的空状态

## 7. 页面布局要求

页面采用“左侧导航 + 顶部标题栏 + 主工作区”的布局。

### 7.1 左侧 Sidebar

必须包含：

- 产品/系统名称
- 当前选中菜单高亮
- 至少两个导航项：
  - `视频分析工作台`
  - `历史记录`

当前阶段可以只实现“视频分析工作台”，但“历史记录”入口要预留。

### 7.2 顶部 Header

必须包含：

- 页面标题
- 系统在线状态
- 可预留：
  - 当前模型版本
  - 当前任务数量
  - 用户菜单

### 7.3 主工作区

建议分为三个层级：

1. 顶部：视频上传区 + 运行日志区
2. 中部：任务进度、错误提示
3. 底部：结果展示区

## 8. 页面模块需求

## 8.1 VideoPanel

功能：

- 选择本地视频文件
- 展示文件名、大小
- 支持点击上传
- 支持重置
- 支持点击“开始分析”
- 展示进度条
- 后续可扩展为视频预览播放器

要求：

- 未选择视频时显示空态上传框
- 选择视频后显示文件卡片
- 分析中禁用重复提交
- 允许用户重新选择文件

建议展示字段：

- 文件名
- 文件大小
- 上传状态
- 分析状态
- 进度百分比

## 8.2 LogPanel

功能：

- 展示任务日志
- 支持按时间顺序滚动
- 自动滚动到最新日志
- 区分日志级别颜色

日志级别至少支持：

- `INFO`
- `WARNING`
- `ERROR`

要求：

- 没有日志时显示“等待任务提交”
- 日志过多时内部滚动，不撑破页面

## 8.3 SummaryCards

功能：

- 展示分析结果摘要

至少展示 4 个指标：

- `检测动作总数`
- `平均置信度`
- `有效轨迹帧数`
- `有效姿态帧数`

要求：

- 卡片样式统一
- 数值突出
- 支持后续扩展更多统计项

## 8.4 ActionTimeline

功能：

- 展示动作识别结果列表或时间轴

每个动作条目至少包含：

- 动作名称
- 开始时间
- 结束时间
- 置信度
- 细节说明

推荐交互：

- 点击某个动作条目时，右侧高亮该时刻相关姿态/轨迹信息
- 后续可以联动视频播放器跳转到对应时间

当前阶段即使先做列表式时间轴也可以，但结构要便于后续升级为真正可拖动时间轴。

## 8.5 TrackPosePanel

功能：

- 展示某一关键帧的姿态与轨迹信息

至少展示：

- 当前高亮帧号
- 球员状态文本
- 关键点平均得分
- 羽毛球坐标
- 羽毛球轨迹置信度

推荐后续扩展：

- 左侧显示某一帧的缩略图
- 在缩略图上叠加 skeleton 和 ball 点
- 可切换上一帧 / 下一帧

## 8.6 视频结果预览模块

这是当前缺失但强烈建议 Gemini 一并开发的模块。

需要新增一个结果视频预览区，支持展示后端输出的视频，例如：

- `pose_vis.mp4`
- `track_vis.mp4`
- `unified_vis.mp4`

至少支持：

- 视频播放器
- 播放 / 暂停
- 当前时间显示
- 切换查看不同类型输出视频

如果后端暂时不能提供可播放 URL，则前端先保留该组件和空态。

## 9. API 对接需求

前端必须使用统一的 API 封装层。

文件建议位置：

- `src/api/client.ts`

### 9.1 健康检查

```http
GET /api/health
```

返回示例：

```json
{
  "status": "ok",
  "service": "Badminton AI Engine"
}
```

### 9.2 上传视频

```http
POST /api/upload
```

请求：

- `multipart/form-data`
- 字段：`file`

返回示例：

```json
{
  "message": "Upload successful",
  "file_id": "vid_12345678",
  "filename": "match.mp4"
}
```

### 9.3 提交分析任务

```http
POST /api/analyze?file_id=vid_xxx
```

返回示例：

```json
{
  "task_id": "task_12345678",
  "message": "Analysis started"
}
```

### 9.4 查询任务状态

```http
GET /api/task/{task_id}/status
```

返回示例：

```json
{
  "task_id": "task_12345678",
  "status": "processing",
  "progress": 65,
  "logs": [
    {
      "time": "12:00:01",
      "level": "INFO",
      "message": "正在逐帧分析..."
    }
  ]
}
```

### 9.5 获取任务结果

```http
GET /api/task/{task_id}/result
```

返回示例：

```json
{
  "summary": {
    "total_actions": 6,
    "avg_confidence": 0.92,
    "valid_track_frames": 1240,
    "valid_pose_frames": 1350
  },
  "actions": [],
  "pose": [],
  "track": []
}
```

## 10. TypeScript 类型要求

类型必须单独定义在：

- `src/types/index.ts`

至少包含：

- `Action`
- `Summary`
- `Log`
- `TaskStatus`
- `PoseSample`
- `TrackSample`
- `AnalysisResult`

要求：

- 不允许页面里到处写 `any`
- 接口返回值需要类型约束
- Hook 和组件 props 都要显式类型化

## 11. Hook 设计要求

建议实现一个核心 Hook：

- `useAnalysisTask`

它至少负责：

- 管理当前文件
- 提交上传
- 提交分析任务
- 轮询任务状态
- 拉取任务结果
- 管理错误状态
- 管理重置逻辑

建议返回：

- `file`
- `setFile`
- `taskId`
- `status`
- `result`
- `error`
- `start`
- `reset`

要求：

- 组件层尽量少写异步业务逻辑
- 轮询定时器必须在组件卸载时清理
- 避免重复启动多个轮询

## 12. UI 风格要求

页面风格建议：

- 深色分析平台风格
- 不要做成花哨的营销官网
- 更像“AI 分析工作台 / 视频审核平台 / 视觉分析控制台”

视觉关键词：

- 稳定
- 专业
- 清晰
- 数据感
- 工具感

颜色建议：

- 背景：深灰、深蓝灰
- 主色：蓝色或青蓝色
- 成功：绿色
- 警告：橙色
- 错误：红色

要求：

- 卡片层次清晰
- 边框和背景有区分
- 重点数据足够突出
- 交互反馈明确

## 13. 交互细节要求

Gemini 在实现时必须关注这些交互细节：

- 上传后按钮状态要即时变化
- 进度条变化要平滑，不突然跳动
- 日志区域要可滚动
- 错误时要有明显提示
- 分析完成后结果区自动显示
- 没有结果时要有清晰空态
- 任务失败时要允许重新发起

推荐增加：

- 分析中按钮 loading 状态
- 分析完成 toast 提示
- 网络异常 toast 或错误卡片

## 14. 响应式要求

页面至少需要支持：

- 1440px 桌面大屏
- 1280px 普通笔记本
- 1024px 横向平板或小屏笔记本

要求：

- 小屏时可以把双列改为单列
- 日志区和结果区不能完全挤坏
- 不要求优先做移动端，但不能彻底不可用

## 15. 性能与工程要求

Gemini 实现时需要遵守以下工程要求：

- 使用函数组件
- 使用 TypeScript 严格类型
- 避免一个组件承担过多职责
- API 封装独立
- Hook 独立
- 样式尽量通过 Tailwind 组织
- 不引入过重的全家桶状态管理

如果需要增加库，请优先轻量：

- `clsx`
- `zustand`
- `recharts`

避免一上来引入复杂框架。

## 16. 当前阶段必须完成的页面交付

Gemini 至少需要完成以下最低可交付版本：

1. 一个可运行的前端工程
2. 一个主工作台页面
3. 视频上传区
4. 日志区
5. 进度条
6. 摘要卡片区
7. 动作时间轴区
8. 姿态/轨迹信息区
9. 基于 mock 数据或真实接口的轮询逻辑
10. 组件化结构，不允许把所有代码堆在一个文件

## 17. 推荐增强项

如果 Gemini 能进一步完成，建议增加：

- 视频结果播放器组件
- 某一帧可视化叠加层
- 任务历史记录页
- 结果导出按钮
- 暗色主题细节优化
- 动作时间轴点击联动
- 结果过滤与搜索

## 18. 验收标准

前端交付至少满足以下验收条件：

### 功能层

- 可以选择视频文件
- 可以调用上传接口
- 可以调用分析接口
- 可以轮询状态接口
- 可以在完成后拉取结果接口
- 可以展示摘要、动作、姿态、轨迹信息

### 工程层

- 项目可以 `npm install` 后启动
- 代码通过 TypeScript 基本检查
- 目录结构符合要求
- 页面组件拆分合理

### 视觉层

- 页面不是空白或原始浏览器样式
- 深色工作台风格统一
- 卡片、进度、日志、时间轴区层次清晰

## 19. 对 Gemini 的明确开发指令

可以直接把下面这段给 Gemini：

> 请基于现有目录 `apps/badminton-analysis-system/frontend` 开发一个可运行的 React + TypeScript + Vite + Tailwind 前端页面。  
> 目标是实现一个“羽毛球动作分析工作台”，包括：视频上传、任务提交、状态轮询、日志展示、摘要卡片、动作时间轴、姿态与轨迹信息面板，以及预留视频结果展示区。  
> 请严格按组件化结构开发，不要把所有逻辑堆在 `App.tsx`。  
> 请优先保证页面可运行、结构清晰、类型完整、交互完整。  
> 如果后端接口尚未完成，请使用 mock 数据或保留接口适配层，但页面结构必须真实可用。  
> UI 风格采用深色专业分析平台风格，不要做成营销官网。  
> 代码需要可扩展，方便后续接入真实后端和统一推理结果。

## 20. 当前后端状态说明

为了避免 Gemini 误解，请明确告知：

- 当前后端尚未完成正式实现
- 前端需要先以 mock 或占位 API 为主开发
- 后续再接真实接口

也就是说，当前前端开发重点是：

- 页面搭建
- 数据结构适配
- 交互流程打通
- 异步轮询机制搭好

不是要求它现在就把整套后端逻辑做完。

## 21. 当前仓库相关入口

可参考的相关文件：

- 前端主入口：
  - [apps/badminton-analysis-system/frontend/src/App.tsx](/D:/Github/WFBARNet/apps/badminton-analysis-system/frontend/src/App.tsx)
- 前端 API 封装：
  - [apps/badminton-analysis-system/frontend/src/api/client.ts](/D:/Github/WFBARNet/apps/badminton-analysis-system/frontend/src/api/client.ts)
- 前端 Hook：
  - [apps/badminton-analysis-system/frontend/src/hooks/useAnalysisTask.ts](/D:/Github/WFBARNet/apps/badminton-analysis-system/frontend/src/hooks/useAnalysisTask.ts)
- 前端类型：
  - [apps/badminton-analysis-system/frontend/src/types/index.ts](/D:/Github/WFBARNet/apps/badminton-analysis-system/frontend/src/types/index.ts)
- 后端占位入口：
  - [apps/badminton-analysis-system/backend/app/main.py](/D:/Github/WFBARNet/apps/badminton-analysis-system/backend/app/main.py)

