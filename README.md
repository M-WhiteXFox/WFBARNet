# WFBARNet

WFBARNet is a badminton video analysis project. The current codebase combines ball tracking, pose estimation, result export, visualization, and a local PyQt6 desktop UI.

## Current Layout

```text
WFBARNet/
  main.py                         # CLI-oriented runtime builders
  configs/default_infer.json       # default inference config
  src/                             # core inference package
    models/                        # TrackNet, pose backends, TensorRT backend
    preprocess/                    # frame preprocessing
    postprocess/                   # heatmap decoding and track filtering
    runners/                       # pose, track, realtime, and unified runners
    builders/                      # downstream feature builders
    utils/                         # video, export, visualization, structures
  apps/
    pyqt6/                         # current desktop application
    desktop_gui/                   # legacy DearPyGui desktop application
  tools/
    demo/                          # runnable CLI demos
    benchmarks/                    # local performance benchmark scripts
    mmpose/                        # MMPose helper files
  assets/
    weights/                       # local model weights and exported artifacts
    docs/                          # project reference material
  tests/                           # unit tests
```

## Main Entry Points

- PyQt6 desktop app: `python -m apps.pyqt6.main`
- CLI/default pipeline: edit `USER_CONFIG` in `main.py`, then run `python main.py`
- Unified demo: `python tools/demo/run_unified_infer.py --source path/to/video.mp4`
- Track-only demo: `python tools/demo/run_track_only.py --source path/to/video.mp4`
- Runtime benchmark: `python tools/benchmarks/benchmark_runtime_latency.py --source path/to/video.mp4`

## Models And Data

Expected local model paths:

- `assets/weights/pose/yolo26s-pose.pt`
- `assets/weights/track/model_best.pt`
- optional TensorRT export files under `assets/weights/track/`
- optional BST weights under `assets/weights/bst/`

Large local files are intentionally ignored by git:

- videos under `videos/`
- generated outputs under `outputs/`
- Python caches
- Ultralytics cache files
- TensorRT/ONNX export artifacts

Use Git LFS or external release artifacts if these files need to be shared.

## Tests

```powershell
python -m unittest discover tests
```

The benchmark script is not a unit test and lives in `tools/benchmarks/`.

## Notes

`apps/pyqt6/` is the active UI. `apps/desktop_gui/` is kept as a legacy DearPyGui implementation; remove or archive it only after confirming it is no longer needed.
