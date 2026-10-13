from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .mms_engine import mms_available
from .pipeline import AlignConfig, run_alignment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tigrigna-aligner",
        description=(
            "Forced Tigrinya phrase aligner: dry VO audio + exact script "
            "-> timed-script.json for the Remotion editing engine."
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
        "--engine",
        default="auto",
        choices=["auto", "mms", "hierarchy", "whisper"],
        help=(
            "Alignment engine: auto (MMS CTC if installed, else hierarchy), "
            "mms (true Tigrinya forced align), hierarchy (fast pause-aware), "
            "whisper (optional refine). Default: auto"
        ),
    )
    p.add_argument(
        "--model",
        default="base",
        help="faster-whisper model size if --engine whisper (default: base)",
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
        help="Whisper language hint (only for --engine whisper)",
    )
    p.add_argument(
        "--mms-language",
        default="tir",
        help="MMS ISO-639-3 language code (default: tir for Tigrinya)",
    )
    p.add_argument(
        "--silence-db",
        default="auto",
        help="Silence threshold in dB, or 'auto' (default: auto)",
    )
    p.add_argument(
        "--pad-sec",
        type=float,
        default=0.06,
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
        help="Force fast hierarchy engine (no MMS/Whisper)",
    )
    p.add_argument(
        "--with-whisper",
        action="store_true",
        help="Also try Whisper blending (usually weak for Tigrinya)",
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

    silence_raw = str(args.silence_db).strip().lower()
    silence_db: float | None
    if silence_raw in {"", "auto", "none"}:
        silence_db = None
    else:
        silence_db = float(silence_raw)

    engine = args.engine
    if args.vad_only:
        engine = "hierarchy"

    config = AlignConfig(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=language,
        mms_language=args.mms_language,
        silence_db=silence_db,
        pad_sec=args.pad_sec,
        review_threshold=args.review_threshold,
        engine=engine,
        skip_whisper=not args.with_whisper and engine != "whisper",
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
    print(f"Engine: {engine}")
    if engine in {"auto", "mms"}:
        print(f"MMS available: {mms_available()}")

    out = run_alignment(audio, script, output, config=config)
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
