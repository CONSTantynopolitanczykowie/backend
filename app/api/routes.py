from fastapi import APIRouter, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect

from app.schemas.session import (
    CountSummary,
    MessageResponse,
    SessionSnapshot,
    StartSessionRequest,
    UpdateLineRequest,
)
from app.services.session_manager import SessionManager

router = APIRouter(prefix="/api/v1", tags=["counter"])


def _manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


@router.post("/session/start", response_model=SessionSnapshot)
async def start_session(payload: StartSessionRequest, request: Request) -> SessionSnapshot:
    manager = _manager(request)
    try:
        return await manager.start(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/session/stop", response_model=SessionSnapshot)
async def stop_session(request: Request) -> SessionSnapshot:
    return await _manager(request).stop()


@router.post("/session/reset", response_model=MessageResponse)
async def reset_session(request: Request) -> MessageResponse:
    return await _manager(request).reset()


@router.get("/session/counts")
async def get_counts(request: Request) -> dict:
    counts = await _manager(request).get_counts()
    return counts.model_dump()


@router.get("/session/status", response_model=SessionSnapshot)
async def get_status(request: Request) -> SessionSnapshot:
    return await _manager(request).get_status()


@router.put("/session/line", response_model=SessionSnapshot)
async def update_line(payload: UpdateLineRequest, request: Request) -> SessionSnapshot:
    return await _manager(request).update_line(payload.line)


@router.get("/session/line-preview")
async def get_line_preview(
    request: Request,
    video_path: str = Query(..., description="Path to local mp4 file, absolute or relative to backend root"),
) -> Response:
    manager = _manager(request)
    try:
        image_bytes = await manager.get_line_preview_frame(video_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=image_bytes, media_type="image/jpeg")


@router.get("/session/first-frame-detections")
async def get_first_frame_detections(
    request: Request,
    video_path: str = Query(..., description="Path to local mp4 file"),
) -> Response:
    manager = _manager(request)
    try:
        image_bytes = await manager.get_first_frame_with_detections(video_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=image_bytes, media_type="image/jpeg")


@router.post("/session/process-full-video", response_model=CountSummary)
async def process_full_video(
    request: Request,
    video_path: str = Query(..., description="Path to local mp4 file"),
    frame_every_n: int = Query(1, description="Save every nth frame (default 1 = save all)"),
) -> CountSummary:
    manager = _manager(request)
    try:
        return await manager.process_and_save_full_video(video_path, frame_every_n=frame_every_n)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.websocket("/ws")
async def websocket_events(websocket: WebSocket) -> None:
    manager: SessionManager = websocket.app.state.session_manager
    await manager.hub.connect(websocket)
    try:
        status = await manager.get_status()
        await websocket.send_json({"type": "state", "snapshot": status.model_dump(), "events": []})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.hub.disconnect(websocket)
    except Exception:
        await manager.hub.disconnect(websocket)
