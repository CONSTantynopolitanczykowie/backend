# Backend

FastAPI backend for people counting on beach footage using YOLOv8n.

The pipeline does the following:

1. Reads a local `.mp4` file.
2. Detects people per frame.
3. Tracks people across frames.
4. Counts entries/exits by track position relative to frame center.

Counting semantics:

- Left side of frame: treated as `entering` (`up += 1`, `signed_total += 1`).
- Right side of frame: treated as `exiting` (`down += 1`, `signed_total -= 1`).
- Middle dead zone: ignored (no counting).
- Counters remain backward-compatible: `up`, `down`, `signed_total`.

## v1 Scope

- Input: local stock-footage files in `dataset/camera/`.
- Output: REST endpoints for status and counts.
- Live stream: WebSocket state/events.
- Utility: first-frame line preview image.

## Directory Layout

```text
backend/
app/
app/main.py
app/api/routes.py
app/core/settings.py
app/schemas/session.py
app/services/counter.py
app/services/detector.py
app/services/session_manager.py
app/services/tracker.py
app/services/websocket_hub.py
dataset/camera/
model/
requirements.txt
```

## Step-by-Step Setup

### 1) Open terminal in backend directory

Run from: `[organisation]/`

```bash
cd backend
```

After this step, your working directory should be: `[organisation]/backend/`.

### 2) Create virtual environment

Run from: `[organisation]/backend/`

```bash
python3 -m venv .venv
```

### 3) Activate virtual environment

Run from: `[organisation]/backend/`

```bash
source .venv/bin/activate
```

### 4) Install dependencies

Run from: `[organisation]/backend/`

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5) Put your video in dataset folder

Place your `.mp4` in:

```text
backend/dataset/camera/
```

Example used in this guide:

```text
backend/dataset/camera/test-crowd.mp4
```

### 6) Run the API server

Run from: `[organisation]/backend/`


REMEMBER TO CHANGE ORGANISATION FOR YOUR FOLDER
```bash
.venv/bin/python -m uvicorn app.main:app --app-dir /home/oliver/hackathon/[organisation]/backend --host 127.0.0.1 --port 8001
```

Keep this terminal open while testing.

### 7) Open docs (optional)

Open in browser:

```text
http://127.0.0.1:8001/docs
```

## Step-by-Step Test Guide

Use a second terminal for API tests.

### 1) Go to backend directory

Run from: `[organisation]/`

```bash
cd backend
```

### 2) Health check

Run from: `[organisation]/backend/`

```bash
curl -sS http://127.0.0.1:8001/health
```

Expected response:

```json
{"status":"ok"}
```

### 3) Generate line preview image on first frame

Run from: `[organisation]/backend/`

```bash
curl -sS "http://127.0.0.1:8001/api/v1/session/line-preview?video_path=dataset/camera/test-crowd.mp4" -o dataset/camera/line-preview.jpg
```

Check that image exists:

```bash
ls -lh dataset/camera/line-preview.jpg
```

### 4) Start counting session

Run from: `[organisation]/backend/`

```bash
curl -sS -X POST http://127.0.0.1:8001/api/v1/session/start -H "Content-Type: application/json" -d '{"video_path":"dataset/camera/test-crowd.mp4"}'
```

### 5) Check running status

Run from: `[organisation]/backend/`

```bash
curl -sS http://127.0.0.1:8001/api/v1/session/status
```

### 6) Check counts

Run from: `[organisation]/backend/`

```bash
curl -sS http://127.0.0.1:8001/api/v1/session/counts
```

### 7) Stop session

Run from: `[organisation]/backend/`

```bash
curl -sS -X POST http://127.0.0.1:8001/api/v1/session/stop
```

### 8) Reset session (optional)

Run from: `[organisation]/backend/`

```bash
curl -sS -X POST http://127.0.0.1:8001/api/v1/session/reset
```

## Offline Verification Runner (tmain.py)

You can run full verification directly without starting FastAPI.

Run from: `[organisation]/backend/`

```bash
.venv/bin/python tmain.py --video-path dataset/camera/test-crowd.mp4 --frame-every-n 1 --output-dir dataset/camera
```

Optional custom line:

```bash
.venv/bin/python tmain.py --video-path dataset/camera/test-crowd.mp4 --line-x1 640 --line-y1 0 --line-x2 640 --line-y2 719
```

Outputs in the output dir:

- `count.json` summary (`up`, `down`, `signed_total`, metadata)
- `bboxes.json` detections/tracks for each processed frame
- `count_timeline.json` cumulative counts per frame
- `frames/frame_XXXXXX.jpg` annotated frame images

## Core API

- `GET /health`
- `POST /api/v1/session/start`
- `POST /api/v1/session/stop`
- `POST /api/v1/session/reset`
- `GET /api/v1/session/counts`
- `GET /api/v1/session/status`
- `PUT /api/v1/session/line`
- `GET /api/v1/session/line-preview?video_path=dataset/camera/test-crowd.mp4`
- `POST /api/v1/session/process-full-video?video_path=...`
- `WS /api/v1/ws`

## Common Run Issues

### Import error: No module named app

Cause: running `python3 main.py` from `backend/app/`.

Fix: run from `backend/` using module mode:

REMEMBER TO CHANGE ORGANISATION FOR YOUR FOLDER
```bash
.venv/bin/python -m uvicorn app.main:app --app-dir /home/oliver/hackathon/[organisation]/backend --host 127.0.0.1 --port 8001
```

### Curl fails with connection refused (exit code 7)

Cause: API server is not running.

Fix: start the server first, then run curl commands from a second terminal.

## Notes

- Model path default: `model/yolov8n.pt`.
- On first inference, Ultralytics may auto-download YOLOv8n weights.
- CPU demo defaults include frame skipping and output FPS cap for stability.
