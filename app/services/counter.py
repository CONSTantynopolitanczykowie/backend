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
        self._track_side: dict[int, int] = {}
        self._last_centroid: dict[int, Point] = {}
        self._last_crossing_frame: dict[int, int] = {}

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
        self._track_side.clear()
        self._last_centroid.clear()
        self._last_crossing_frame.clear()

    def update_line(self, line: LineConfig) -> None:
        self._line = line
        self._track_side.clear()
        self._last_centroid.clear()
        self._last_crossing_frame.clear()

    def _signed_distance_proxy(self, p: Point) -> float:
        ax = self._line.start.x
        ay = self._line.start.y
        bx = self._line.end.x
        by = self._line.end.y

        # Cross product sign defines side of point relative to directed line A->B.
        return float((bx - ax) * (p.y - ay) - (by - ay) * (p.x - ax))

    def _sign(self, value: float) -> int:
        if value > self._dead_zone_px:
            return 1
        if value < -self._dead_zone_px:
            return -1
        return 0

    def update(self, tracks: Iterable[Track], frame_index: int) -> list[CrossingEvent]:
        events: list[CrossingEvent] = []
        active_track_ids: set[int] = set()

        for track in tracks:
            track_id = track.track_id
            active_track_ids.add(track_id)

            current_point = Point(x=int(track.centroid[0]), y=int(track.centroid[1]))
            current_sign = self._sign(self._signed_distance_proxy(current_point))

            previous_sign = self._track_side.get(track_id)
            if previous_sign is None:
                self._track_side[track_id] = current_sign
                self._last_centroid[track_id] = current_point
                continue

            if current_sign == 0:
                self._last_centroid[track_id] = current_point
                continue

            if previous_sign == 0:
                self._track_side[track_id] = current_sign
                self._last_centroid[track_id] = current_point
                continue

            if previous_sign == current_sign:
                self._track_side[track_id] = current_sign
                self._last_centroid[track_id] = current_point
                continue

            last_crossing = self._last_crossing_frame.get(track_id, -10_000_000)
            if frame_index - last_crossing < self._min_crossing_gap_frames:
                self._track_side[track_id] = current_sign
                self._last_centroid[track_id] = current_point
                continue

            # Direction is defined by side change across directed line A->B.
            # This works for vertical, horizontal and diagonal lines.
            delta = 1 if previous_sign > current_sign else -1

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
            self._last_crossing_frame[track_id] = frame_index
            self._track_side[track_id] = current_sign
            self._last_centroid[track_id] = current_point

        stale_ids = set(self._track_side.keys()) - active_track_ids
        for stale_id in stale_ids:
            self._track_side.pop(stale_id, None)
            self._last_centroid.pop(stale_id, None)
            self._last_crossing_frame.pop(stale_id, None)

        return events
