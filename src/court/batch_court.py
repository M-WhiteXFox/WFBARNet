from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from src.court.court_line_detector import CourtLineBackend, CourtLineDetector, create_court_line_detector
from src.court.courtkeynet_detector import resolve_courtkeynet_weights
from src.court.opencv_court_detector import CourtLinePrediction
from src.court.shuttlecourt_seg_detector import resolve_shuttlecourt_weights


DetectorFactory = Callable[[CourtLineBackend], CourtLineDetector]
PredictionAcceptor = Callable[[CourtLineBackend, CourtLinePrediction], bool]


def default_batch_court_backends() -> tuple[CourtLineBackend, ...]:
    """Return the batch court fallback order available in the current workspace."""
    backends: list[CourtLineBackend] = []
    try:
        resolve_courtkeynet_weights("assets/weights/courtkeynet/CourtKeyNet.safetensors")
    except FileNotFoundError:
        pass
    else:
        backends.append("courtkeynet")
    try:
        resolve_shuttlecourt_weights("weights/shttlecourtnet")
    except FileNotFoundError:
        pass
    else:
        backends.append("shuttlecourt_seg")
    backends.extend(["monotrack", "opencv"])
    return tuple(backends)


def _default_detector_factory(backend: CourtLineBackend) -> CourtLineDetector:
    return create_court_line_detector(backend=backend)


def _default_prediction_acceptor(
    backend: CourtLineBackend,
    prediction: CourtLinePrediction,
) -> bool:
    del backend
    return bool(prediction.valid)


def is_trusted_automatic_court_prediction(prediction: CourtLinePrediction) -> bool:
    """Return whether an automatic result is safe for geometry-dependent analysis."""
    if not prediction.valid:
        return False
    if prediction.scheme == "court_pose_white_line":
        return True
    if not _automatic_prediction_geometry_is_plausible(prediction):
        return False

    metrics = prediction.metrics if isinstance(prediction.metrics, dict) else {}
    components = metrics.get("components") if isinstance(metrics.get("components"), dict) else {}
    if prediction.scheme == "courtkeynet":
        combined = components.get("courtkeynet_combined_confidence")
        threshold = components.get("courtkeynet_confidence_threshold")
        return bool(
            combined is not None
            and threshold is not None
            and _metric_value(components.get("courtkeynet_confirmation_complete")) >= 1.0
            and _metric_value(combined) >= _metric_value(threshold)
        )
    min_singles_support = _metric_value(components.get("singles_min_support"))
    singles_support_ratio = _metric_value(components.get("singles_support_ratio"))
    min_outer_support = _metric_value(components.get("outer_min_support"))
    if prediction.scheme == "shuttlecourt_seg":
        return bool(
            _metric_value(components.get("seg_line_fit")) >= 0.5
            and min_singles_support >= 0.04
            and singles_support_ratio >= 0.18
            and min_outer_support >= 0.15
        )
    if prediction.scheme == "monotrack":
        return bool(
            min_singles_support >= 0.15
            and singles_support_ratio >= 0.40
        )
    if prediction.scheme in {"6", "8"}:
        return bool(
            min_singles_support >= 0.15
            and singles_support_ratio >= 0.40
            and min_outer_support >= 0.15
            and _metric_value(components.get("shape")) >= 0.75
            and _metric_value(components.get("quad")) >= 0.60
        )
    return False


@dataclass
class BatchCourtPredictor:
    """Stateful ordered court fallback for offline and automatic calibration."""

    backends: tuple[CourtLineBackend, ...] = field(default_factory=default_batch_court_backends)
    detector_factory: DetectorFactory = _default_detector_factory
    prediction_acceptor: PredictionAcceptor = _default_prediction_acceptor
    reset_unaccepted_detector_state: bool = False
    lock_confirmed_fallback_geometry: bool = False
    fallback_recheck_interval_ms: int = 750
    fallback_confirm_frames: int = 3
    fallback_max_corner_shift_ratio: float = 0.035
    active_backend: str = ""
    errors: list[str] = field(default_factory=list)
    authoritative_backends: tuple[CourtLineBackend, ...] = ("court_pose",)
    _active_detector: CourtLineDetector | None = field(default=None, init=False, repr=False)
    _latest_prediction: CourtLinePrediction | None = field(default=None, init=False, repr=False)
    _detectors: dict[CourtLineBackend, CourtLineDetector] = field(default_factory=dict, init=False, repr=False)
    _last_backend_selection_timestamp_ms: int | None = field(default=None, init=False, repr=False)
    _fallback_candidate_backend: str = field(default="", init=False, repr=False)
    _fallback_candidate_prediction: CourtLinePrediction | None = field(default=None, init=False, repr=False)
    _fallback_candidate_count: int = field(default=0, init=False, repr=False)
    _locked_fallback_backend: str = field(default="", init=False, repr=False)
    _locked_fallback_prediction: CourtLinePrediction | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        reset_error: Exception | None = None
        reset_ids: set[int] = set()
        for detector in self._detectors.values():
            detector_id = id(detector)
            if detector_id in reset_ids:
                continue
            reset_ids.add(detector_id)
            try:
                detector.reset()
            except Exception as exc:
                if reset_error is None:
                    reset_error = exc
        self._detectors.clear()
        self._active_detector = None
        self._latest_prediction = None
        self._last_backend_selection_timestamp_ms = None
        self._clear_fallback_confirmation()
        self._clear_locked_fallback()
        self.active_backend = ""
        self.errors.clear()
        if reset_error is not None:
            raise reset_error

    def latest_prediction(self) -> CourtLinePrediction | None:
        return self._latest_prediction

    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool = False,
    ) -> CourtLinePrediction | None:
        if self._active_detector is not None:
            failed_backend = self.active_backend
            active_detector = self._active_detector
            if (
                self.lock_confirmed_fallback_geometry
                and self._locked_fallback_backend == failed_backend
                and self._locked_fallback_prediction is not None
            ):
                locked = _locked_prediction_for_frame(
                    self._locked_fallback_prediction,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                )
                recheck_backends = self._locked_upgrade_backends(failed_backend)
                if self._should_recheck_backends(
                    recheck_backends,
                    timestamp_ms=timestamp_ms,
                    force=force,
                ):
                    locked = self._select_backend(
                        frame,
                        frame_id,
                        timestamp_ms,
                        force=force,
                        candidate_backends=recheck_backends,
                        fallback_prediction=locked,
                        fallback_detector=active_detector,
                        fallback_backend=failed_backend,
                    )
                self._latest_prediction = locked
                return locked
            try:
                prediction = active_detector.predict(frame, frame_id, timestamp_ms, force=force)
            except Exception as exc:
                self.errors.append(f"{self.active_backend}: {exc}")
                prediction = None
            else:
                if prediction.valid and self._prediction_is_accepted(failed_backend, prediction):
                    recheck_backends = self._recheck_backends(
                        failed_backend,
                        prediction,
                    )
                    if self._should_recheck_backends(
                        recheck_backends,
                        timestamp_ms=timestamp_ms,
                        force=force,
                    ):
                        prediction = self._select_backend(
                            frame,
                            frame_id,
                            timestamp_ms,
                            force=force,
                            candidate_backends=recheck_backends,
                            fallback_prediction=prediction,
                            fallback_detector=active_detector,
                            fallback_backend=failed_backend,
                        )
                    else:
                        prediction = self._activate_prediction(
                            active_detector,
                            failed_backend,
                            prediction,
                        )
                    self._latest_prediction = prediction
                    return prediction

            if prediction is not None and prediction.valid:
                self._reset_unaccepted_detector(failed_backend, active_detector)
                prediction = _unaccepted_prediction(prediction)
            self._active_detector = None
            if self._fallback_candidate_backend == failed_backend:
                self._clear_fallback_confirmation()
            self.active_backend = ""
            prediction = self._select_backend(
                frame,
                frame_id,
                timestamp_ms,
                force=force,
                candidate_backends=tuple(
                    backend for backend in self.backends if backend != failed_backend
                ),
                fallback_prediction=prediction,
            )
            self._latest_prediction = prediction
            return prediction

        prediction = self._select_backend(frame, frame_id, timestamp_ms, force=force)
        self._latest_prediction = prediction
        return prediction

    def _select_backend(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool,
        candidate_backends: tuple[CourtLineBackend, ...] | None = None,
        fallback_prediction: CourtLinePrediction | None = None,
        fallback_detector: CourtLineDetector | None = None,
        fallback_backend: str = "",
    ) -> CourtLinePrediction | None:
        self._last_backend_selection_timestamp_ms = int(timestamp_ms)
        fallback_provisional = (
            fallback_prediction
            if fallback_prediction is not None
            and not fallback_prediction.valid
            and _prediction_has_geometry(fallback_prediction)
            else None
        )
        best_provisional: CourtLinePrediction | None = None
        best_invalid = (
            fallback_prediction
            if fallback_prediction is not None
            and not fallback_prediction.valid
            and not _prediction_has_geometry(fallback_prediction)
            else None
        )
        coarse_candidate: tuple[CourtLinePrediction, CourtLineDetector, CourtLineBackend] | None = None

        for backend in candidate_backends if candidate_backends is not None else self.backends:
            try:
                detector = self._detector_for_backend(backend)
                prediction = detector.predict(frame, frame_id, timestamp_ms, force=force)
            except Exception as exc:
                self.errors.append(f"{backend}: {exc}")
                continue

            if prediction.valid:
                if backend == "court_pose" and prediction.scheme == "court_pose_coarse":
                    coarse_candidate = (prediction, detector, backend)
                    candidate = _unaccepted_prediction(prediction)
                    if _prefer_provisional_candidate(best_provisional, candidate):
                        best_provisional = candidate
                    continue
                if not self._prediction_is_accepted(backend, prediction):
                    self._reset_unaccepted_detector(backend, detector)
                    prediction = _unaccepted_prediction(prediction)
                    if (
                        _prediction_has_geometry(prediction)
                        and _prefer_provisional_candidate(best_provisional, prediction)
                    ):
                        best_provisional = prediction
                    elif (
                        not _prediction_has_geometry(prediction)
                        and (
                        best_invalid is None
                        or _invalid_prediction_confidence(prediction)
                        > _invalid_prediction_confidence(best_invalid)
                        )
                    ):
                        best_invalid = prediction
                    continue
                return self._activate_prediction(detector, backend, prediction)
            elif _prediction_has_geometry(prediction):
                if _prefer_provisional_candidate(best_provisional, prediction):
                    best_provisional = prediction
            elif (
                best_invalid is None
                or _invalid_prediction_confidence(prediction) > _invalid_prediction_confidence(best_invalid)
            ):
                best_invalid = prediction

        if fallback_prediction is not None and fallback_prediction.valid and fallback_detector is not None:
            return self._activate_prediction(
                fallback_detector,
                fallback_backend,
                fallback_prediction,
            )
        if coarse_candidate is not None:
            prediction, detector, backend = coarse_candidate
            if self._prediction_is_accepted(backend, prediction):
                return self._activate_prediction(detector, backend, prediction)
        if _prefer_provisional_candidate(best_provisional, fallback_provisional):
            best_provisional = fallback_provisional

        self._active_detector = None
        self._clear_fallback_confirmation()
        self.active_backend = ""
        return best_provisional or best_invalid

    def _prediction_is_accepted(
        self,
        backend: CourtLineBackend,
        prediction: CourtLinePrediction,
    ) -> bool:
        return bool(prediction.valid and self.prediction_acceptor(backend, prediction))

    def _activate_prediction(
        self,
        detector: CourtLineDetector,
        backend: str,
        prediction: CourtLinePrediction,
    ) -> CourtLinePrediction:
        self._active_detector = detector
        self.active_backend = backend
        if (
            self.lock_confirmed_fallback_geometry
            and self._locked_fallback_backend == backend
            and self._locked_fallback_prediction is not None
        ):
            return prediction
        if backend in self.authoritative_backends:
            self._clear_fallback_confirmation()
            self._clear_locked_fallback()
            return prediction
        if not self._higher_priority_backends(backend):
            self._clear_fallback_confirmation()
            self._clear_locked_fallback()
            return prediction
        return self._confirm_fallback_prediction(backend, prediction)

    def _confirm_fallback_prediction(
        self,
        backend: str,
        prediction: CourtLinePrediction,
    ) -> CourtLinePrediction:
        required = max(1, int(self.fallback_confirm_frames))
        if required <= 1:
            self._clear_fallback_confirmation()
            if self.lock_confirmed_fallback_geometry:
                self._locked_fallback_backend = backend
                self._locked_fallback_prediction = prediction
            return prediction

        fresh_observation = bool(prediction.attempted and prediction.updated)
        if not fresh_observation and self._fallback_candidate_backend == backend:
            return _pending_fallback_prediction(
                prediction,
                count=max(1, self._fallback_candidate_count),
                required=required,
            )

        consistent = (
            self._fallback_candidate_backend == backend
            and self._fallback_candidate_prediction is not None
            and _predictions_geometry_consistent(
                self._fallback_candidate_prediction,
                prediction,
                max_corner_shift_ratio=float(self.fallback_max_corner_shift_ratio),
            )
        )
        if consistent:
            self._fallback_candidate_count = min(required, self._fallback_candidate_count + 1)
        else:
            self._fallback_candidate_backend = backend
            self._fallback_candidate_count = 1
            self._fallback_candidate_prediction = prediction

        if self._fallback_candidate_count >= required:
            if self.lock_confirmed_fallback_geometry:
                self._locked_fallback_backend = backend
                self._locked_fallback_prediction = prediction
            return prediction
        return _pending_fallback_prediction(
            prediction,
            count=self._fallback_candidate_count,
            required=required,
        )

    def _clear_fallback_confirmation(self) -> None:
        self._fallback_candidate_backend = ""
        self._fallback_candidate_prediction = None
        self._fallback_candidate_count = 0

    def _clear_locked_fallback(self) -> None:
        self._locked_fallback_backend = ""
        self._locked_fallback_prediction = None

    def _higher_priority_backends(self, backend: str) -> tuple[CourtLineBackend, ...]:
        try:
            backend_index = self.backends.index(backend)
        except ValueError:
            return self.backends
        return self.backends[:backend_index]

    def _locked_upgrade_backends(self, backend: str) -> tuple[CourtLineBackend, ...]:
        higher = self._higher_priority_backends(backend)
        return tuple(candidate for candidate in higher if candidate in self.authoritative_backends)

    def _recheck_backends(
        self,
        backend: str,
        prediction: CourtLinePrediction,
    ) -> tuple[CourtLineBackend, ...]:
        if backend == "court_pose" and prediction.scheme == "court_pose_coarse":
            return tuple(candidate for candidate in self.backends if candidate != backend)
        return self._higher_priority_backends(backend)

    def _should_recheck_backends(
        self,
        candidate_backends: tuple[CourtLineBackend, ...],
        *,
        timestamp_ms: int,
        force: bool,
    ) -> bool:
        if not candidate_backends:
            return False
        if force:
            return True
        last_selection = self._last_backend_selection_timestamp_ms
        if last_selection is None:
            return True
        elapsed_ms = int(timestamp_ms) - int(last_selection)
        return elapsed_ms < 0 or elapsed_ms >= max(0, int(self.fallback_recheck_interval_ms))

    def _detector_for_backend(self, backend: CourtLineBackend) -> CourtLineDetector:
        detector = self._detectors.get(backend)
        if detector is None:
            detector = self.detector_factory(backend)
            self._detectors[backend] = detector
        return detector

    def _reset_unaccepted_detector(
        self,
        backend: str,
        detector: CourtLineDetector,
    ) -> None:
        if not self.reset_unaccepted_detector_state or backend == "court_pose":
            return
        try:
            detector.reset()
        except Exception as exc:
            self.errors.append(f"{backend} reset after unaccepted prediction: {exc}")


def _invalid_prediction_confidence(prediction: CourtLinePrediction) -> float:
    candidate_confidence = prediction.candidate_confidence
    if candidate_confidence is None:
        return _metric_value(prediction.confidence)
    return _metric_value(candidate_confidence)


def _prediction_has_geometry(prediction: CourtLinePrediction) -> bool:
    return bool(
        len(prediction.corners) == 4
        and isinstance(prediction.projected_lines, dict)
        and prediction.projected_lines
    )


def _prefer_provisional_candidate(
    current: CourtLinePrediction | None,
    candidate: CourtLinePrediction | None,
) -> bool:
    if candidate is None or not _prediction_has_geometry(candidate):
        return False
    if current is None:
        return True
    return bool(
        _automatic_prediction_geometry_is_plausible(candidate)
        and not _automatic_prediction_geometry_is_plausible(current)
    )


def _automatic_prediction_geometry_is_plausible(
    prediction: CourtLinePrediction,
) -> bool:
    try:
        corners = np.asarray(prediction.corners, dtype=np.float32).reshape(-1, 2)
        width, height = (float(prediction.source_size[0]), float(prediction.source_size[1]))
    except (TypeError, ValueError, IndexError):
        return False
    if corners.shape != (4, 2) or not np.isfinite(corners).all() or width <= 0 or height <= 0:
        return False

    top_y_ratio = float(np.mean(corners[:2, 1]) / height)
    bottom_y_ratio = float(np.mean(corners[2:, 1]) / height)
    top_width = float(np.linalg.norm(corners[1] - corners[0]))
    bottom_width = float(np.linalg.norm(corners[2] - corners[3]))
    left_depth = float(np.linalg.norm(corners[3] - corners[0]))
    right_depth = float(np.linalg.norm(corners[2] - corners[1]))
    width_ratio = bottom_width / max(top_width, 1.0)
    depth_symmetry = min(left_depth, right_depth) / max(left_depth, right_depth, 1.0)
    return bool(
        top_y_ratio >= 0.20
        and bottom_y_ratio >= 0.72
        and 0.75 <= width_ratio <= 4.50
        and depth_symmetry >= 0.80
    )


def _predictions_geometry_consistent(
    previous: CourtLinePrediction,
    current: CourtLinePrediction,
    *,
    max_corner_shift_ratio: float,
) -> bool:
    previous_corners = np.asarray(previous.corners, dtype=np.float32).reshape(-1, 2)
    current_corners = np.asarray(current.corners, dtype=np.float32).reshape(-1, 2)
    if previous_corners.shape != (4, 2) or current_corners.shape != (4, 2):
        return False
    if not np.isfinite(previous_corners).all() or not np.isfinite(current_corners).all():
        return False
    reference_scale = max(
        float(np.linalg.norm(previous_corners[2] - previous_corners[0])),
        float(np.linalg.norm(previous_corners[3] - previous_corners[1])),
        1.0,
    )
    max_shift = float(np.max(np.linalg.norm(current_corners - previous_corners, axis=1)))
    return max_shift / reference_scale <= max(0.0, float(max_corner_shift_ratio))


def _pending_fallback_prediction(
    prediction: CourtLinePrediction,
    *,
    count: int,
    required: int,
) -> CourtLinePrediction:
    status = f"fallback confirmation {int(count)}/{int(required)}"
    metrics = dict(prediction.metrics) if isinstance(prediction.metrics, dict) else {}
    metrics.update(
        {
            "fallback_confirmation_count": int(count),
            "fallback_confirmation_required": int(required),
            "provisional_candidate": 1,
        }
    )
    return replace(
        prediction,
        valid=False,
        updated=False,
        update_type=status,
        status=status,
        confidence=0.0,
        candidate_confidence=float(prediction.confidence),
        reason=f"{status}: {prediction.reason}",
        metrics=metrics,
    )


def _unaccepted_prediction(prediction: CourtLinePrediction) -> CourtLinePrediction:
    status = "provisional candidate; waiting for verified white-line evidence"
    metrics = dict(prediction.metrics) if isinstance(prediction.metrics, dict) else {}
    metrics["provisional_candidate"] = 1
    candidate_confidence = prediction.candidate_confidence
    if candidate_confidence is None:
        candidate_confidence = prediction.confidence
    return replace(
        prediction,
        valid=False,
        updated=False,
        update_type="provisional candidate",
        status=status,
        confidence=0.0,
        candidate_confidence=float(candidate_confidence),
        reason=f"{status}: {prediction.reason}",
        metrics=metrics,
    )


def _locked_prediction_for_frame(
    prediction: CourtLinePrediction,
    *,
    frame_id: int,
    timestamp_ms: int,
) -> CourtLinePrediction:
    return replace(
        prediction,
        frame_id=int(frame_id),
        timestamp_ms=int(timestamp_ms),
        attempted=False,
        updated=False,
        update_type="locked trusted calibration",
        status="locked trusted calibration",
        detect_ms=0.0,
    )


def _metric_value(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(parsed):
        return 0.0
    return parsed
