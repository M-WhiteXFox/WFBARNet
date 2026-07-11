# 权重文件

原始权重文件应存放在以下目录结构中。

* `assets/weights/bst/bst_CG_AP_JnB_bone_merged_10.pt`
* `assets/weights/pose/yolo26s-pose.pt`
* `assets/weights/track/model_best.pt`
* `assets/weights/ShuttleCourtNet/ShuttleCourt.pt`
* `assets/weights/court_pose/CourtPose.pt`

模型权重从项目 Releases 下载后，应保持对应目录和文件名放入本目录。即使缺少可选权重，GUI 仍然可以正常启动，系统会在分析开始前检查已启用模型所需的文件。BST 权重文件为可选项，仅影响击球类型分类功能。
