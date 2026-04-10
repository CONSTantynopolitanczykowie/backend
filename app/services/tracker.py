from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.services.detector import Detection


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


class CentroidTracker:
    def __init__(self, max_distance_px: float, max_missed_frames: int) -> None:
        self._max_distance_px = max_distance_px
        self._max_missed_frames = max_missed_frames

        self._next_track_id = 1
        self._centroids: dict[int, tuple[float, float]] = {}
        self._bboxes: dict[int, tuple[int, int, int, int]] = {}
        self._missed: dict[int, int] = {}

    def reset(self) -> None:
        self._next_track_id = 1
        self._centroids.clear()
        self._bboxes.clear()
        self._missed.clear()

    def _bbox_centroid(self, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _create_track(self, detection: "Detection") -> None:
        track_id = self._next_track_id
        self._next_track_id += 1
        centroid = self._bbox_centroid(detection.bbox)
        self._centroids[track_id] = centroid
        self._bboxes[track_id] = detection.bbox
        self._missed[track_id] = 0

    def update(self, detections: list["Detection"]) -> list[Track]:
        if not detections:
            to_remove: list[int] = []
            for track_id in list(self._missed.keys()):
                self._missed[track_id] += 1
                if self._missed[track_id] > self._max_missed_frames:
                    to_remove.append(track_id)
            for track_id in to_remove:
                self._centroids.pop(track_id, None)
                self._bboxes.pop(track_id, None)
                self._missed.pop(track_id, None)
            return []

        new_centroids = [self._bbox_centroid(d.bbox) for d in detections]
        existing_ids = list(self._centroids.keys())

        if not existing_ids:
            for detection in detections:
                self._create_track(detection)
            return [
                Track(track_id=t_id, bbox=self._bboxes[t_id], centroid=self._centroids[t_id])
                for t_id in self._centroids.keys()
            ]

        unmatched_detections = set(range(len(detections)))
        matched_tracks: set[int] = set()

        for track_id in existing_ids:
            old_centroid = np.array(self._centroids[track_id])
            best_idx = None
            best_distance = float("inf")

            for det_idx in unmatched_detections:
                candidate = np.array(new_centroids[det_idx])
                distance = float(np.linalg.norm(old_centroid - candidate))
                if distance < best_distance:
                    best_distance = distance
                    best_idx = det_idx

            if best_idx is None or best_distance > self._max_distance_px:
                self._missed[track_id] += 1
                continue

            detection = detections[best_idx]
            self._centroids[track_id] = new_centroids[best_idx]
            self._bboxes[track_id] = detection.bbox
            self._missed[track_id] = 0
            unmatched_detections.remove(best_idx)
            matched_tracks.add(track_id)

        to_remove: list[int] = []
        for track_id in list(self._centroids.keys()):
            if track_id in matched_tracks:
                continue
            if track_id not in existing_ids:
                continue
            if self._missed[track_id] > self._max_missed_frames:
                to_remove.append(track_id)

        for track_id in to_remove:
            self._centroids.pop(track_id, None)
            self._bboxes.pop(track_id, None)
            self._missed.pop(track_id, None)

        for det_idx in unmatched_detections:
            self._create_track(detections[det_idx])

        tracks = [
            Track(track_id=t_id, bbox=self._bboxes[t_id], centroid=self._centroids[t_id])
            for t_id in self._centroids.keys()
        ]
        return tracks
