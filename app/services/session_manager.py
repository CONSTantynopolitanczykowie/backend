import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.core.settings import Settings
from app.schemas.session import (
    CounterState,
    CountSummary,
    LineConfig,
    MessageResponse,
    SessionSnapshot,
    StartSessionRequest,
)
from app.services.counter import LineCrossCounter
from app.services.detector import PeopleDetector
from app.services.tracker import CentroidTracker
from app.services.websocket_hub import WebSocketHub

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()

        default_line = LineConfig(
            start={"x": settings.default_line_x1, "y": settings.default_line_y1},
            end={"x": settings.default_line_x2, "y": settings.default_line_y2},
        )

        self._detector = PeopleDetector(
            model_path=str(settings.resolved_model_path()),
            person_class_id=settings.person_class_id,
            confidence_threshold=settings.confidence_threshold,
        )
        self._tracker = CentroidTracker(
            max_distance_px=settings.tracker_max_distance_px,
            max_missed_frames=settings.tracker_max_missed_frames,
        )
        self._counter = LineCrossCounter(
            line=default_line,
            dead_zone_px=settings.counter_dead_zone_px,
            min_crossing_gap_frames=settings.counter_min_crossing_gap_frames,
        )
        self._hub = WebSocketHub()

        self._status: str = "idle"
        self._message: str | None = None
        self._video_path: str | None = None
        self._frame_index: int = 0
        self._timestamp_ms: float = 0.0
        self._active_tracks: int = 0

        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def hub(self) -> WebSocketHub:
        return self._hub

    def _resolve_video_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return self._settings.backend_root() / path

    def _snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            status=self._status,
            video_path=self._video_path,
            frame_index=self._frame_index,
            active_tracks=self._active_tracks,
            timestamp_ms=self._timestamp_ms,
            counts=self._counter.state,
            line=self._counter.line,
            message=self._message,
        )

    def _infer_center_vertical_line(self, video_path: Path) -> LineConfig:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width <= 0 or height <= 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Unable to infer video dimensions")
                height, width = frame.shape[:2]

            center_x = width // 2
            max_y = max(height - 1, 0)
            return LineConfig(
                start={"x": center_x, "y": 0},
                end={"x": center_x, "y": max_y},
            )
        finally:
            cap.release()

    async def start(self, payload: StartSessionRequest) -> SessionSnapshot:
        async with self._lock:
            if self._task is not None and not self._task.done():
                raise ValueError("A session is already running")

            video_path = self._resolve_video_path(payload.video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            if payload.line is not None:
                line = payload.line
            else:
                line = self._infer_center_vertical_line(video_path)

            self._tracker.reset()
            self._counter.reset(line=line)
            self._video_path = str(video_path)
            self._frame_index = 0
            self._timestamp_ms = 0.0
            self._active_tracks = 0
            self._message = "Session started"
            self._status = "running"
            self._stop_event.clear()

            self._task = asyncio.create_task(self._run(video_path=video_path))
            logger.info(
                "Session started video=%s line_start=(%d,%d) line_end=(%d,%d)",
                self._video_path,
                line.start.x,
                line.start.y,
                line.end.x,
                line.end.y,
            )
            return self._snapshot()

    async def stop(self) -> SessionSnapshot:
        async with self._lock:
            if self._task is None:
                self._status = "stopped"
                self._message = "No active session"
                return self._snapshot()

            self._stop_event.set()
            task = self._task

        await task

        async with self._lock:
            self._task = None
            self._status = "stopped"
            self._message = "Session stopped"
            snapshot = self._snapshot()
        return snapshot

    async def reset(self) -> MessageResponse:
        await self.stop()
        async with self._lock:
            self._tracker.reset()
            self._counter.reset(line=self._counter.line)
            self._video_path = None
            self._frame_index = 0
            self._timestamp_ms = 0.0
            self._active_tracks = 0
            self._status = "idle"
            self._message = "Session reset"
        return MessageResponse(message="Session reset")

    async def update_line(self, line: LineConfig) -> SessionSnapshot:
        async with self._lock:
            self._counter.update_line(line)
            self._message = "Line updated"
            return self._snapshot()

    async def get_counts(self) -> CounterState:
        async with self._lock:
            return CounterState(
                up=self._counter.state.up,
                down=self._counter.state.down,
                signed_total=self._counter.state.signed_total,
            )

    async def get_status(self) -> SessionSnapshot:
        async with self._lock:
            return self._snapshot()

    async def get_first_frame_with_detections(self, raw_video_path: str) -> bytes:
        """Load first frame, run YOLO detection, draw bboxes, encode to JPEG."""
        video_path = self._resolve_video_path(raw_video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Unable to read first frame from video")

            detections = self._detector.detect_people(frame)
            frame_with_detections = self._draw_bounding_boxes(frame=frame, detections=detections)
            success, encoded = cv2.imencode(".jpg", frame_with_detections)
            if not success:
                raise RuntimeError("Unable to encode detection frame")
            logger.info("First frame detections: %d people detected", len(detections))
            return encoded.tobytes()
        finally:
            cap.release()

    def _draw_bounding_boxes(self, frame: np.ndarray, detections) -> np.ndarray:
        """Draw YOLO bounding boxes on frame."""
        annotated = frame.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(
                annotated,
                f"P{i} {det.confidence:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated

    async def get_line_preview_frame(self, raw_video_path: str) -> bytes:
        video_path = self._resolve_video_path(raw_video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        preview_line = self._infer_center_vertical_line(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Unable to read first frame from video")

            frame_with_line = self._draw_overlay_line(frame=frame, line=preview_line)
            success, encoded = cv2.imencode(".jpg", frame_with_line)
            if not success:
                raise RuntimeError("Unable to encode preview frame")
            return encoded.tobytes()
        finally:
            cap.release()

    def _draw_overlay_line(self, frame: np.ndarray, line: LineConfig) -> np.ndarray:
        annotated = frame.copy()
        start = (line.start.x, line.start.y)
        end = (line.end.x, line.end.y)

        cv2.line(annotated, start, end, color=(0, 255, 255), thickness=3)
        cv2.circle(annotated, start, radius=6, color=(0, 200, 0), thickness=-1)
        cv2.circle(annotated, end, radius=6, color=(0, 0, 200), thickness=-1)
        cv2.putText(
            annotated,
            "line start",
            (start[0] + 8, start[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "line end",
            (end[0] + 8, end[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 200),
            2,
            cv2.LINE_AA,
        )

        return annotated

    async def _emit_state(self, events: list[dict] | None = None) -> None:
        snapshot = self._snapshot().model_dump()
        await self._hub.broadcast_json(
            {
                "type": "state",
                "snapshot": snapshot,
                "events": events or [],
            }
        )

    async def _run(self, video_path: Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            async with self._lock:
                self._status = "error"
                self._message = f"Unable to open video: {video_path}"
            await self._emit_state()
            return

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                self._frame_index += 1
                if self._settings.process_every_nth_frame > 1:
                    if self._frame_index % self._settings.process_every_nth_frame != 0:
                        continue

                self._timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)

                detections = self._detector.detect_people(frame)
                tracks = self._tracker.update(detections)
                self._active_tracks = len(tracks)

                crossing_events = self._counter.update(tracks, frame_index=self._frame_index)
                event_payload = [e.model_dump() for e in crossing_events]

                if crossing_events:
                    for event in crossing_events:
                        logger.info(
                            "Crossing track_id=%d direction=%s delta=%d frame=%d counters(up=%d,down=%d,signed_total=%d)",
                            event.track_id,
                            event.direction,
                            event.delta,
                            event.frame_index,
                            self._counter.state.up,
                            self._counter.state.down,
                            self._counter.state.signed_total,
                        )

                if self._frame_index % 60 == 0:
                    logger.info(
                        "Progress frame=%d active_tracks=%d counters(up=%d,down=%d,signed_total=%d)",
                        self._frame_index,
                        self._active_tracks,
                        self._counter.state.up,
                        self._counter.state.down,
                        self._counter.state.signed_total,
                    )

                if crossing_events or self._frame_index % 10 == 0:
                    await self._emit_state(events=event_payload)

                if self._settings.max_output_fps > 0:
                    await asyncio.sleep(1.0 / self._settings.max_output_fps)

            async with self._lock:
                if self._status != "error":
                    self._status = "stopped"
                    self._message = "Session finished"
            await self._emit_state()
        except Exception as exc:
            logger.exception("Session processing crashed")
            async with self._lock:
                self._status = "error"
                self._message = str(exc)
            await self._emit_state()
        finally:
            cap.release()

    async def process_and_save_full_video(
        self,
        raw_video_path: str,
        frame_every_n: int = 1,
        output_dir: str | None = None,
    ) -> CountSummary:
        """Process entire video, save annotated frames asynchronously, return summary with count JSON."""
        video_path = self._resolve_video_path(raw_video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_root = self._settings.backend_root() / "dataset" / "camera"
        if output_dir is not None:
            output_path = Path(output_dir)
            output_root = output_path if output_path.is_absolute() else self._settings.backend_root() / output_path

        # Setup output directory
        output_root.mkdir(parents=True, exist_ok=True)
        frames_dir = output_root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Clean up old frames if directory exists
        for old_frame in frames_dir.glob("frame_*.jpg"):
            old_frame.unlink()

        logger.info("Starting full-video processing: %s, saving every %d frames", video_path, frame_every_n)

        # Create fresh tracker and counter for this batch
        tracker = CentroidTracker(
            max_distance_px=self._settings.tracker_max_distance_px,
            max_missed_frames=self._settings.tracker_max_missed_frames,
        )
        counter = LineCrossCounter(
            line=self._counter.line,
            dead_zone_px=self._settings.counter_dead_zone_px,
            min_crossing_gap_frames=self._settings.counter_min_crossing_gap_frames,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        try:
            start_time = datetime.now()
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_idx = 0
            processed_frame_idx = 0
            save_tasks = []
            bboxes_payload: list[dict] = []
            count_timeline_payload: list[dict] = []

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1

                # Process every nth frame
                if (frame_idx - 1) % frame_every_n != 0:
                    continue

                processed_frame_idx += 1

                # Run detection, tracking, counting
                detections = self._detector.detect_people(frame)
                tracks = tracker.update(detections)
                crossing_events = counter.update(tracks, frame_index=processed_frame_idx)

                detection_records = []
                for det_index, det in enumerate(detections):
                    x1, y1, x2, y2 = det.bbox
                    detection_records.append(
                        {
                            "detection_index": det_index,
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "confidence": det.confidence,
                        }
                    )

                track_records = []
                for track in tracks:
                    tx1, ty1, tx2, ty2 = track.bbox
                    cx, cy = track.centroid
                    track_records.append(
                        {
                            "track_id": track.track_id,
                            "bbox": {"x1": tx1, "y1": ty1, "x2": tx2, "y2": ty2},
                            "centroid": {"x": cx, "y": cy},
                        }
                    )

                bboxes_payload.append(
                    {
                        "frame_index": processed_frame_idx,
                        "detections": detection_records,
                        "tracks": track_records,
                    }
                )

                count_timeline_payload.append(
                    {
                        "frame_index": processed_frame_idx,
                        "up": counter.state.up,
                        "down": counter.state.down,
                        "signed_total": counter.state.signed_total,
                        "crossings": [event.model_dump() for event in crossing_events],
                    }
                )

                # Prepare frame with annotations (bboxes)
                annotated_frame = self._draw_bounding_boxes(frame, detections)

                # Schedule async save
                frame_filename = f"frame_{processed_frame_idx:06d}.jpg"
                frame_path = frames_dir / frame_filename
                task = self._save_frame_async(annotated_frame, frame_path)
                save_tasks.append(task)

                if processed_frame_idx % 100 == 0:
                    logger.info(
                        "Batch progress: frame=%d (raw=%d) active_tracks=%d counters(up=%d, down=%d, signed_total=%d)",
                        processed_frame_idx,
                        frame_idx,
                        len(tracks),
                        counter.state.up,
                        counter.state.down,
                        counter.state.signed_total,
                    )

            # Wait for all async saves to complete
            if save_tasks:
                logger.info("Waiting for %d frame writes to complete...", len(save_tasks))
                await asyncio.gather(*save_tasks)
                logger.info("All frames saved")

            # Calculate duration
            end_time = datetime.now()
            duration_seconds = (end_time - start_time).total_seconds()

            # Write verification JSON outputs.
            bboxes_file = output_root / "bboxes.json"
            bboxes_file.write_text(json.dumps(bboxes_payload, indent=2))

            count_timeline_file = output_root / "count_timeline.json"
            count_timeline_file.write_text(json.dumps(count_timeline_payload, indent=2))

            # Write count.json
            count_data = {
                "up": counter.state.up,
                "down": counter.state.down,
                "signed_total": counter.state.signed_total,
                "total_frames_processed": processed_frame_idx,
                "video_path": str(video_path),
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration_seconds,
                "frames_directory": str(frames_dir),
                "bboxes_file": str(bboxes_file),
                "count_timeline_file": str(count_timeline_file),
            }
            count_file = output_root / "count.json"
            count_file.write_text(json.dumps(count_data, indent=2))
            logger.info("Count summary written to %s", count_file)

            summary = CountSummary(
                up=counter.state.up,
                down=counter.state.down,
                signed_total=counter.state.signed_total,
                total_frames_processed=processed_frame_idx,
                video_path=str(video_path),
                timestamp=start_time.isoformat(),
                duration_seconds=duration_seconds,
                frames_directory=str(frames_dir),
                bboxes_file=str(bboxes_file),
                count_timeline_file=str(count_timeline_file),
            )

            logger.info(
                "Batch processing complete: up=%d down=%d signed_total=%d frames=%d duration=%.2fs",
                summary.up,
                summary.down,
                summary.signed_total,
                summary.total_frames_processed,
                summary.duration_seconds,
            )

            return summary

        finally:
            cap.release()

    async def _save_frame_async(self, frame: np.ndarray, path: Path) -> None:
        """Save frame asynchronously using asyncio executor."""
        loop = asyncio.get_event_loop()

        def _save_sync():
            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                raise RuntimeError(f"Failed to encode frame for {path}")
            path.write_bytes(encoded.tobytes())

        await loop.run_in_executor(None, _save_sync)
