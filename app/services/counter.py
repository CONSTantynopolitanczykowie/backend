from collections.abc import Iterable

from app.schemas.session import CounterState, CrossingEvent, LineConfig, Point
from app.services.tracker import Track


class LineCrossCounter:
    def __init__(
        self,
        line: LineConfig,
        dead_zone_px: float,
        min_crossing_gap_frames: int,
    ) -> None:
        self._line = line
        self._dead_zone_px = dead_zone_px
        self._min_crossing_gap_frames = min_crossing_gap_frames

        self._state = CounterState()
        self._track_zone: dict[int, int] = {}
        self._last_counted_zone: dict[int, int] = {}
        self._last_event_frame: dict[int, int] = {}

    @property
    def state(self) -> CounterState:
        return self._state

    @property
    def line(self) -> LineConfig:
        return self._line

    def reset(self, line: LineConfig | None = None) -> None:
        if line is not None:
            self._line = line
        self._state = CounterState()
        self._track_zone.clear()
        self._last_counted_zone.clear()
        self._last_event_frame.clear()

    def update_line(self, line: LineConfig) -> None:
        self._line = line
        self._track_zone.clear()
        self._last_counted_zone.clear()
        self._last_event_frame.clear()

    def _zone_for_point(self, p: Point) -> int:
        center_x = (self._line.start.x + self._line.end.x) / 2.0
        delta_x = p.x - center_x

        if delta_x < -self._dead_zone_px:
            return -1
        if delta_x > self._dead_zone_px:
            return 1
        return 0

    def update(self, tracks: Iterable[Track], frame_index: int) -> list[CrossingEvent]:
        events: list[CrossingEvent] = []
        active_track_ids: set[int] = set()

        for track in tracks:
            track_id = track.track_id
            active_track_ids.add(track_id)

            current_point = Point(x=int(track.centroid[0]), y=int(track.centroid[1]))
            current_zone = self._zone_for_point(current_point)

            self._track_zone[track_id] = current_zone

            # Ignore detections in the middle dead zone.
            if current_zone == 0:
                continue

            if self._last_counted_zone.get(track_id) == current_zone:
                continue

            last_event = self._last_event_frame.get(track_id, -10_000_000)
            if frame_index - last_event < self._min_crossing_gap_frames:
                continue

            # New semantics:
            # - left side of frame  => entering (+1)
            # - right side of frame => exiting (-1)
            delta = 1 if current_zone < 0 else -1

            direction = "up" if delta > 0 else "down"
            if delta > 0:
                self._state.up += 1
            else:
                self._state.down += 1
            self._state.signed_total += delta

            events.append(
                CrossingEvent(
                    track_id=track_id,
                    direction=direction,
                    delta=delta,
                    frame_index=frame_index,
                )
            )
            self._last_event_frame[track_id] = frame_index
            self._last_counted_zone[track_id] = current_zone

        stale_ids = set(self._track_zone.keys()) - active_track_ids
        for stale_id in stale_ids:
            self._track_zone.pop(stale_id, None)
            self._last_counted_zone.pop(stale_id, None)
            self._last_event_frame.pop(stale_id, None)

        return events
