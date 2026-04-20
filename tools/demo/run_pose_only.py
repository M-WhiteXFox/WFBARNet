from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from main import RuntimeConfig, build_pose_runner


# ========================================
# 在这里配置您的参数
# ========================================
USER_CONFIG = RuntimeConfig(
    pipeline="pose_only",
    source=r"videos/test3.mp4",  # 修改为您的视频路径
    output_dir=str(PROJECT_ROOT / "outputs" / "pose_only"),  # 输出目录
    device="cpu",  # 或 "cuda:0" 使用 GPU
    pose_backend="mmpose",  # "mmpose" | "ultralytics_compat" | "dummy"
    pose_config=str(PROJECT_ROOT / "tools" / "mmpose" / "configs" / "rtmpose-s_8xb256-420e_coco-256x192.py"),
    pose_weight=str(PROJECT_ROOT / "assets" / "weights" / "pose" / "rtmpose-s_8xb256-420e_coco-256x192.pth"),
    pose_bbox_mode="whole_image",  # "whole_image" | "split_two" | "detector"
)
# ========================================


def main() -> None:
    config = USER_CONFIG

    if not config.source:
        raise ValueError("请在文件中设置 video_path 配置！")

    print("=" * 60)
    print("Pose-Only 视频推理")
    print("=" * 60)
    print(f"视频路径: {config.source}")
    print(f"输出目录: {config.output_dir}")
    print(f"设备: {config.device}")
    print(f"姿态后端: {config.pose_backend}")
    print(f"姿态配置: {config.pose_config}")
    print(f"姿态权重: {config.pose_weight}")
    print(f"边界框模式: {config.pose_bbox_mode}")
    print("=" * 60)

    runner = build_pose_runner(config)
    runner.run(
        source=config.source,
        save_json=True,
        save_csv=True,
        save_npy=True,
        save_vis=True,
    )

    print("\n✅ 姿态估计完成！")
    print(f"结果保存在: {config.output_dir}")


if __name__ == "__main__":
    main()
