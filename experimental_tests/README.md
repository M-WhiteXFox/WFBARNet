# Experimental tests

This directory keeps reproducibility tests for offline trajectory experiments and
estimate-only APIs that are not called by the application or CLI runners.

Run them explicitly in the `WFBARNet` Conda environment:

```powershell
python -m unittest discover -s experimental_tests -v
```

These tests are intentionally excluded from the default production regression
suite. Move a test back under `tests/` when its behavior is wired into a runtime
entrypoint.
