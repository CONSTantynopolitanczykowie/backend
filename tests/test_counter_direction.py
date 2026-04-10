import unittest

from app.schemas.session import LineConfig
from app.services.counter import LineCrossCounter
from app.services.tracker import Track


class LineCrossCounterDirectionTests(unittest.TestCase):
    def test_vertical_line_left_to_right_is_positive_delta(self) -> None:
        counter = LineCrossCounter(
            line=LineConfig(start={"x": 100, "y": 0}, end={"x": 100, "y": 200}),
            dead_zone_px=0.0,
            min_crossing_gap_frames=0,
        )

        counter.update(
            [Track(track_id=1, bbox=(0, 0, 0, 0), centroid=(80.0, 80.0))],
            frame_index=1,
        )
        events = counter.update(
            [Track(track_id=1, bbox=(0, 0, 0, 0), centroid=(120.0, 80.0))],
            frame_index=2,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].delta, 1)
        self.assertEqual(counter.state.up, 1)
        self.assertEqual(counter.state.down, 0)

    def test_horizontal_line_top_to_bottom_is_negative_delta_even_if_x_increases(self) -> None:
        counter = LineCrossCounter(
            line=LineConfig(start={"x": 0, "y": 100}, end={"x": 200, "y": 100}),
            dead_zone_px=0.0,
            min_crossing_gap_frames=0,
        )

        counter.update(
            [Track(track_id=7, bbox=(0, 0, 0, 0), centroid=(90.0, 90.0))],
            frame_index=1,
        )
        events = counter.update(
            [Track(track_id=7, bbox=(0, 0, 0, 0), centroid=(110.0, 110.0))],
            frame_index=2,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].delta, -1)
        self.assertEqual(counter.state.up, 0)
        self.assertEqual(counter.state.down, 1)


if __name__ == "__main__":
    unittest.main()

