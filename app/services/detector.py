from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float


class PeopleDetector:
    def __init__(
        self,
        model_path: str,
        person_class_id: int,
        confidence_threshold: float,
    ) -> None:
        self._model_path = model_path
        self._person_class_id = person_class_id
        self._confidence_threshold = confidence_threshold
        self._model: YOLO | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = YOLO(self._model_path)

    def detect_people(self, frame: np.ndarray) -> list[Detection]:
        self._ensure_model()
        assert self._model is not None

        result = self._model(frame, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)

        detections: list[Detection] = []
        for i, class_id in enumerate(cls):
            if class_id != self._person_class_id:
                continue
            confidence = float(conf[i])
            if confidence < self._confidence_threshold:
                continue

            x1, y1, x2, y2 = xyxy[i]
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=confidence,
                )
            )

        return detections
