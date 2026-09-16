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
