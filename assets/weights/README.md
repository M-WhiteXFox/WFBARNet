# 权重文件

模型权重需要从项目 Releases 单独下载。下载完成后，请保持原始文件名，并按照以下目录结构放置：

- 击球动作分类：`assets/weights/bst/bst_CG_AP_JnB_bone_merged_10.pt`
- 运动员姿态识别：`assets/weights/pose/yolo26s-pose.pt`
- 羽毛球轨迹识别：
  - 默认 PyTorch 模型：`assets/weights/track/model_best.pt`
  - 可选 INT8 ONNX 模型：`assets/weights/track/tracknetv3_int8.onnx`
- 备用球场线识别：`assets/weights/ShuttleCourtNet/ShuttleCourt.pt`
- 默认球场线识别与自动标定：`assets/weights/court_pose/CourtPose.pt`

请勿修改模型文件名或目录结构，否则程序可能无法通过默认配置找到对应权重。
