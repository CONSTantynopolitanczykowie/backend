import argparse
import asyncio
import json
import sys

from app.core.settings import get_settings
from app.schemas.session import LineConfig
from app.services.session_manager import SessionManager


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Offline verification runner for full video processing and counting"
	)
	parser.add_argument(
		"--video-path",
		required=True,
		help="Path to local mp4 file, absolute or relative to backend root",
	)
	parser.add_argument(
		"--frame-every-n",
		type=int,
		default=1,
		help="Process and save every nth frame (default: 1)",
	)
	parser.add_argument(
		"--output-dir",
		default="dataset/camera",
		help="Output directory for count.json, bboxes.json, count_timeline.json and frames/",
	)
	parser.add_argument("--line-x1", type=int)
	parser.add_argument("--line-y1", type=int)
	parser.add_argument("--line-x2", type=int)
	parser.add_argument("--line-y2", type=int)
	return parser


def _parse_line(args: argparse.Namespace) -> LineConfig | None:
	line_values = [args.line_x1, args.line_y1, args.line_x2, args.line_y2]
	provided = [value is not None for value in line_values]
	if not any(provided):
		return None

	if not all(provided):
		raise ValueError("Custom line requires all of --line-x1 --line-y1 --line-x2 --line-y2")

	return LineConfig(
		start={"x": args.line_x1, "y": args.line_y1},
		end={"x": args.line_x2, "y": args.line_y2},
	)


async def _run(args: argparse.Namespace) -> int:
	if args.frame_every_n <= 0:
		print("frame_every_n must be >= 1", file=sys.stderr)
		return 2

	settings = get_settings()
	manager = SessionManager(settings=settings)

	line = _parse_line(args)
	if line is not None:
		await manager.update_line(line)

	summary = await manager.process_and_save_full_video(
		raw_video_path=args.video_path,
		frame_every_n=args.frame_every_n,
		output_dir=args.output_dir,
	)

	print("Verification run complete")
	print(f"up: {summary.up}")
	print(f"down: {summary.down}")
	print(f"signed_total: {summary.signed_total}")
	print(f"total_frames_processed: {summary.total_frames_processed}")
	print(f"frames_directory: {summary.frames_directory}")
	print(f"bboxes_file: {summary.bboxes_file}")
	print(f"count_timeline_file: {summary.count_timeline_file}")
	print("count_summary_json:")
	print(json.dumps(summary.model_dump(), indent=2))
	return 0


def main() -> int:
	parser = _build_parser()
	args = parser.parse_args()
	try:
		return asyncio.run(_run(args))
	except FileNotFoundError as exc:
		print(str(exc), file=sys.stderr)
		return 1
	except ValueError as exc:
		print(str(exc), file=sys.stderr)
		return 2
	except RuntimeError as exc:
		print(str(exc), file=sys.stderr)
		return 3
	except KeyboardInterrupt:
		print("Interrupted", file=sys.stderr)
		return 130


if __name__ == "__main__":
	raise SystemExit(main())
