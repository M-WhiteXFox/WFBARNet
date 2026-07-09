# 权重文件

原始权重文件应存放在以下目录结构中。

* `assets/weights/bst/`
* `assets/weights/pose/`
* `assets/weights/track/`
* `assets/weights/ShuttleCourtNet/`

在运行完整推理之前，请将对应的模型文件放到上述目录中。即使缺少这些文件，GUI 仍然可以正常启动；系统会在分析开始前检查已启用模型所需的权重文件是否存在。BST 权重文件为可选项，仅影响击球类型分类功能。
