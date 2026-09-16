from __future__ import annotations

from dataclasses import dataclass

from .match import AlignedPhrase


@dataclass
class ScoredSegment:
    id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    text: str
    confidence: float
    needs_review: bool
    method: str
    match_score: float


def score_segments(
    aligned: list[AlignedPhrase],
    *,
    review_threshold: float = 65.0,
) -> list[ScoredSegment]:
    scored: list[ScoredSegment] = []
    for a in aligned:
        # Convert fuzzy 0-100 match into 0-1 confidence, with method priors
        base = max(0.0, min(100.0, a.match_score)) / 100.0
        if "whisper_force_match" in a.method:
            conf = 0.55 + 0.45 * base
        elif "vad_1to1" in a.method:
            conf = 0.75
        elif "vad_merged" in a.method or "blend" in a.method:
            conf = 0.55 + 0.25 * base
        elif "proportional" in a.method:
            conf = 0.35
        else:
            conf = 0.45 + 0.40 * base

        conf = round(max(0.0, min(1.0, conf)), 3)
        needs_review = conf < (review_threshold / 100.0) or a.match_score < review_threshold
        # Always flag very short or inverted spans
        dur = max(0.0, a.end_sec - a.start_sec)
        if dur < 0.12:
            needs_review = True
            conf = min(conf, 0.4)

        scored.append(
            ScoredSegment(
                id=f"seg_{a.phrase.index + 1:03d}",
                start_sec=round(a.start_sec, 3),
                end_sec=round(a.end_sec, 3),
                duration_sec=round(dur, 3),
                text=a.phrase.text,
                confidence=conf,
                needs_review=needs_review,
                method=a.method,
                match_score=round(a.match_score, 1),
            )
        )
    return scored
