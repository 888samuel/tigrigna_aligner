from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .pipeline import AlignConfig, run_alignment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tigrigna-aligner",
        description=(
            "Forced Tigrinya phrase aligner: dry VO audio + exact script "
            "→ timed-script.json for the Remotion editing engine."
        ),
    )
    p.add_argument(
        "--audio",
        required=True,
        help=(
            "Dry voice-over path. Any ffmpeg-readable format "
            "(mp3, wav, m4a, flac, ogg, mp4, mkv, webm, ...)"
        ),
    )
    p.add_argument(
        "--script",
        required=True,
        help=(
            "Exact Tigrinya script: txt/md/csv/json/srt/vtt/rtf/docx/pdf "
            "(prefer one phrase per line/paragraph)"
        ),
    )
    p.add_argument(
        "--output",
        default="output/timed-script.json",
        help="Output JSON path (default: output/timed-script.json)",
    )
    p.add_argument("--video-id", default="episode", help="video_id field in JSON")
    p.add_argument(
        "--model",
        default="base",
        help="faster-whisper model size (tiny/base/small/medium). Default: base",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type (int8 recommended on CPU/8GB RAM)",
    )
    p.add_argument(
        "--language",
        default="am",
        help=(
            "Whisper language hint. Tigrinya is low-resource; default 'am' "
            "(Amharic) as Ethiopic timing cue. Use '' for auto-detect."
        ),
    )
    p.add_argument(
        "--silence-db",
        type=float,
        default=-40.0,
        help="Silence threshold in dB for boundary snapping (default: -40)",
    )
    p.add_argument(
        "--pad-sec",
        type=float,
        default=0.08,
        help="Padding after silence valleys so text does not cut early",
    )
    p.add_argument(
        "--review-threshold",
        type=float,
        default=65.0,
        help="Match score below this flags needs_review",
    )
    p.add_argument(
        "--vad-only",
        action="store_true",
        help="Skip Whisper; align phrases to silence/VAD regions only (fast)",
    )
    p.add_argument(
        "--no-debug",
        action="store_true",
        help="Emit only Remotion fields (id/start_sec/end_sec/text)",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    language = args.language.strip() or None

    config = AlignConfig(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=language,
        silence_db=args.silence_db,
        pad_sec=args.pad_sec,
        review_threshold=args.review_threshold,
        skip_whisper=args.vad_only,
        video_id=args.video_id,
        include_debug=not args.no_debug,
    )

    audio = Path(args.audio)
    script = Path(args.script)
    output = Path(args.output)

    if not audio.exists():
        print(f"ERROR: audio not found: {audio}", file=sys.stderr)
        return 1
    if not script.exists():
        print(f"ERROR: script not found: {script}", file=sys.stderr)
        return 1

    print(f"Aligning {audio.name} -> {output.as_posix()}")
    print(f"Script phrases from: {script}")
    if config.skip_whisper:
        print("Mode: VAD/silence only")
    else:
        print(f"Mode: Whisper ({config.model_size}) + VAD + silence snap")

    out = run_alignment(audio, script, output, config=config)
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
