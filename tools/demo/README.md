# Demo Scripts

This folder contains small standalone demos for local experiments.

## Court Keypoint Detection

`run_court_keypoints_yolo.py` detects four badminton court corner points with a YOLO keypoint model.

Expected model output:

- one court object
- at least four keypoints
- default keypoint order: `0,1,2,3`

Run on an image:

```powershell
python tools/demo/run_court_keypoints_yolo.py `
  --source path/to/frame.jpg `
  --weights assets/weights/court/court_yolo_keypoints.pt
```

Run on a video and correct the court points every 150 frames:

```powershell
python tools/demo/run_court_keypoints_yolo.py `
  --source path/to/video.mp4 `
  --weights assets/weights/court/court_yolo_keypoints.pt `
  --detect-every 150 `
  --smooth-alpha 0.25
```

Outputs are written to `outputs/court_keypoints_demo/`:

- `*_court_keypoints.json`
- `*_court_keypoints.jpg` for images
- `*_court_keypoints.mp4` for videos unless `--no-video` is used

The saved corner order is:

```text
top-left, top-right, bottom-right, bottom-left
```

If your model uses a different keypoint order, pass:

```powershell
--keypoint-indices 3,2,1,0
```

If your model already returns ordered corners, pass:

```powershell
--order model
```

## Existing Runtime Demos

- `run_pose_only.py`
- `run_track_only.py`
- `run_tracknet_realtime.py`
- `run_unified_infer.py`
- `tracknet_demo.py`
