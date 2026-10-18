from __future__ import annotations

from .config import Config
from .detector import Detection, FaceDetector
from .events import EventType
from .preprocess import frame_issue


def analyze_frame(frame, detector: FaceDetector, cfg: Config) -> tuple[EventType, list[Detection], dict]:
    """One frame -> (raw observation, detections, details). Never raises."""
    issue = frame_issue(frame, cfg)
    if issue:
        return EventType.DETECTION_FAILURE, [], {"reason": issue}
    try:
        dets = detector.detect(frame)
    except Exception as exc:  # detector must never take the session down
        return EventType.DETECTION_FAILURE, [], {"reason": "detector_exception", "error": type(exc).__name__}
    n = len(dets)
    obs = EventType.FACE_ABSENT if n == 0 else EventType.FACE_PRESENT if n == 1 else EventType.MULTIPLE_FACES
    return obs, dets, {"n_faces": n, "max_confidence": round(max((d.confidence for d in dets), default=0.0), 3),
                       "detector": detector.name}
