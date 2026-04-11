from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BEACH_COUNTER_", extra="ignore")

    model_path: str = Field(default="model/yolov8n.pt")
    person_class_id: int = Field(default=0)
    confidence_threshold: float = Field(default=0.35)
    process_every_nth_frame: int = Field(default=2)
    max_output_fps: float = Field(default=12.0)

    tracker_max_distance_px: float = Field(default=80.0)
    tracker_max_missed_frames: int = Field(default=18)

    counter_dead_zone_px: float = Field(default=4.0)
    counter_min_crossing_gap_frames: int = Field(default=8)

    default_line_x1: int = Field(default=200)
    default_line_y1: int = Field(default=260)
    default_line_x2: int = Field(default=1080)
    default_line_y2: int = Field(default=260)

    def backend_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def resolved_model_path(self) -> Path:
        p = Path(self.model_path)
        if p.is_absolute():
            return p
        return self.backend_root() / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
