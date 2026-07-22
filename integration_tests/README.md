# Integration tests

These tests require external videos, local model weights, or GPU inference and
are intentionally excluded from the default test suite.

Set an external court video and run the suite in the `WFBARNet` Conda environment:

```powershell
$env:WFBARNET_COURT_VIDEO = "D:\path\to\court-video.mp4"
python -m unittest discover -s integration_tests -v
```

Set `WFBARNET_RUN_COURT_POSE_ACCURACY=1` to include the CourtPose accuracy gate.
Optional CourtPose paths and settings use the existing `COURT_POSE_*` environment
variables.
