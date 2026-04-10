from typing import Literal

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)


class LineConfig(BaseModel):
    start: Point
    end: Point


class CounterState(BaseModel):
    up: int = 0
    down: int = 0
    signed_total: int = 0


class CrossingEvent(BaseModel):
    track_id: int
    direction: Literal["up", "down"]
    delta: Literal[-1, 1]
    frame_index: int


class SessionSnapshot(BaseModel):
    status: Literal["idle", "running", "stopped", "error"]
    video_path: str | None = None
    frame_index: int = 0
    active_tracks: int = 0
    timestamp_ms: float = 0.0
    counts: CounterState = Field(default_factory=CounterState)
    line: LineConfig
    message: str | None = None


class StartSessionRequest(BaseModel):
    video_path: str
    line: LineConfig | None = None


class UpdateLineRequest(BaseModel):
    line: LineConfig


class MessageResponse(BaseModel):
    message: str


class CountSummary(BaseModel):
    up: int
    down: int
    signed_total: int
    total_frames_processed: int
    video_path: str
    timestamp: str
    duration_seconds: float
    frames_directory: str
    bboxes_file: str
    count_timeline_file: str
