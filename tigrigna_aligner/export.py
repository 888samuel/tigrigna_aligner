from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .confidence import ScoredSegment


def to_timed_script(
    segments: list[ScoredSegment],
    *,
    video_id: str,
    audio_path: str,
    version: str = "1.0",
    meta: dict[str, Any] | None = None,
    include_debug: bool = True,
) -> dict[str, Any]:
    """Build Remotion-compatible timed-script.json (+ optional QA fields)."""
    out_segments: list[dict[str, Any]] = []
    for s in segments:
        item: dict[str, Any] = {
            "id": s.id,
            "start_sec": s.start_sec,
            "end_sec": s.end_sec,
            "text": s.text,
        }
        if include_debug:
            item["duration_sec"] = s.duration_sec
            item["confidence"] = s.confidence
            item["needs_review"] = s.needs_review
            item["method"] = s.method
            item["match_score"] = s.match_score
        out_segments.append(item)

    payload: dict[str, Any] = {
        "version": version,
        "video_id": video_id,
        "segments": out_segments,
    }
    if include_debug:
        payload["audio"] = audio_path
        payload["meta"] = meta or {}
        payload["stats"] = {
            "phrase_count": len(segments),
            "needs_review_count": sum(1 for s in segments if s.needs_review),
            "mean_confidence": round(
                sum(s.confidence for s in segments) / max(1, len(segments)), 3
            ),
        }
    return payload


def write_timed_script(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def format_srt_timestamp(seconds: float) -> str:
    """SRT timestamp: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list[dict[str, Any]]) -> str:
    """One cue per phrase, using start_sec/end_sec/text."""
    cues: list[str] = []
    index = 1
    prev_end = 0.0
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg["start_sec"])
        end = float(seg["end_sec"])
        start = max(start, prev_end)
        if end <= start:
            end = start + 0.12
        cues.append(
            f"{index}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{text}\n"
        )
        prev_end = end
        index += 1
    return "\n".join(cues) + ("\n" if cues else "")


def write_srt(path: str | Path, segments: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Windows players (VLC, Premiere) read Ge'ez correctly
    path.write_text(segments_to_srt(segments), encoding="utf-8-sig")
    return path


def srt_path_for(json_path: str | Path, override: str | Path | None = None) -> Path:
    if override:
        return Path(override)
    json_path = Path(json_path)
    return json_path.with_suffix(".srt")
