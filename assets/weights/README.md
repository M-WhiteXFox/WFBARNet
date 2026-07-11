# 权重文件

原始权重文件应存放在以下目录结构中。

* `assets/weights/bst/`
* `assets/weights/pose/`
* `assets/weights/track/`
* `assets/weights/ShuttleCourtNet/`
* `assets/weights/court_pose/`

自动球场标定使用的 `assets/weights/court_pose/best.pt` 随仓库发布，拉取项目后无需再配置外部球场模型路径。其他大型推理权重仍按部署方式放入对应目录；即使缺少可选权重，GUI 仍然可以正常启动，系统会在分析开始前检查已启用模型所需的文件。BST 权重文件为可选项，仅影响击球类型分类功能。
