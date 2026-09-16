from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from .script import Phrase, normalize_tigrinya


@dataclass
class WordStamp:
    text: str
    normalized: str
    start: float
    end: float


@dataclass
class AlignedPhrase:
    phrase: Phrase
    start_sec: float
    end_sec: float
    match_score: float
    method: str


def words_from_segments(segments: list[dict]) -> list[WordStamp]:
    words: list[WordStamp] = []
    for seg in segments:
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                text = str(w.get("word") or w.get("text") or "").strip()
                if not text:
                    continue
                start = float(w["start"])
                end = float(w["end"])
                words.append(
                    WordStamp(
                        text=text,
                        normalized=normalize_tigrinya(text),
                        start=start,
                        end=end,
                    )
                )
        else:
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            start = float(seg["start"])
            end = float(seg["end"])
            # Split segment text into pseudo-words for sequential matching
            for token in text.split():
                words.append(
                    WordStamp(
                        text=token,
                        normalized=normalize_tigrinya(token),
                        start=start,
                        end=end,
                    )
                )
    return words


def force_match_phrases(
    phrases: list[Phrase],
    words: list[WordStamp],
    audio_duration: float,
    *,
    min_score: float = 55.0,
) -> list[AlignedPhrase]:
    """Greedily map each script phrase onto contiguous whisper words in order."""
    if not words:
        return proportional_align(phrases, 0.0, audio_duration, method="proportional_empty_asr")

    aligned: list[AlignedPhrase] = []
    cursor = 0
    n_words = len(words)

    for pi, phrase in enumerate(phrases):
        remaining_phrases = len(phrases) - pi
        remaining_words = n_words - cursor
        if remaining_words <= 0:
            # No ASR words left — distribute over remaining audio after last end
            last_end = aligned[-1].end_sec if aligned else 0.0
            rest = proportional_align(
                phrases[pi:],
                last_end,
                audio_duration,
                method="proportional_tail",
            )
            aligned.extend(rest)
            break

        # Prefer leaving at least one word slot per remaining phrase
        max_span = max(1, remaining_words - (remaining_phrases - 1))
        target_tokens = max(1, len(phrase.normalized.split()))
        # Search window around expected token count
        lo = max(1, target_tokens - 2)
        hi = min(max_span, max(target_tokens + 6, target_tokens * 2))

        best_score = -1.0
        best_j = cursor
        raw_score = 0.0

        for span in range(lo, hi + 1):
            j = cursor + span
            window = " ".join(w.normalized for w in words[cursor:j] if w.normalized)
            if not window:
                continue
            score = float(fuzz.token_set_ratio(phrase.normalized, window))
            # Soft length prior: prefer spans near target token count
            length_penalty = abs(span - target_tokens) * 1.5
            score_adj = score - length_penalty
            if score_adj > best_score:
                best_score = score_adj
                best_j = j
                raw_score = score

        if best_score < 0:
            raw_score = 0.0
            best_j = min(cursor + 1, n_words)

        start = words[cursor].start
        end = words[best_j - 1].end
        method = "whisper_force_match" if raw_score >= min_score else "whisper_weak_match"
        aligned.append(
            AlignedPhrase(
                phrase=phrase,
                start_sec=float(start),
                end_sec=float(max(end, start + 0.05)),
                match_score=float(raw_score),
                method=method,
            )
        )
        cursor = best_j

    return _enforce_monotonic(aligned, audio_duration)


def align_phrases_to_regions(
    phrases: list[Phrase],
    regions: list[tuple[float, float]],
    audio_duration: float,
) -> list[AlignedPhrase]:
    """Map phrases onto VAD speech regions (1:1, merge, or split by char length)."""
    if not regions:
        return proportional_align(phrases, 0.0, audio_duration, method="proportional_no_vad")

    if len(regions) == len(phrases):
        return [
            AlignedPhrase(
                phrase=p,
                start_sec=float(s),
                end_sec=float(e),
                match_score=80.0,
                method="vad_1to1",
            )
            for p, (s, e) in zip(phrases, regions)
        ]

    if len(regions) > len(phrases):
        merged = _merge_regions(regions, len(phrases))
        return [
            AlignedPhrase(
                phrase=p,
                start_sec=float(s),
                end_sec=float(e),
                match_score=70.0,
                method="vad_merged",
            )
            for p, (s, e) in zip(phrases, merged)
        ]

    # Fewer regions than phrases: split regions by character weight
    weights = [max(1, len(p.normalized.replace(" ", ""))) for p in phrases]
    aligned: list[AlignedPhrase] = []
    phrase_idx = 0
    for ri, (rs, re) in enumerate(regions):
        remaining_regions = len(regions) - ri
        remaining_phrases = len(phrases) - phrase_idx
        take = remaining_phrases // remaining_regions
        if ri == len(regions) - 1:
            take = remaining_phrases
        take = max(1, take)
        group = phrases[phrase_idx : phrase_idx + take]
        group_w = weights[phrase_idx : phrase_idx + take]
        span = re - rs
        cursor = rs
        gw = sum(group_w) or 1
        for gp, w in zip(group, group_w):
            dur = span * (w / gw)
            end = cursor + dur
            aligned.append(
                AlignedPhrase(
                    phrase=gp,
                    start_sec=float(cursor),
                    end_sec=float(end),
                    match_score=60.0,
                    method="vad_split",
                )
            )
            cursor = end
        phrase_idx += take

    # Safety if math left phrases
    if phrase_idx < len(phrases):
        last_end = aligned[-1].end_sec if aligned else 0.0
        aligned.extend(
            proportional_align(
                phrases[phrase_idx:],
                last_end,
                audio_duration,
                method="proportional_vad_tail",
            )
        )
    return _enforce_monotonic(aligned, audio_duration)


def proportional_align(
    phrases: list[Phrase],
    start_sec: float,
    end_sec: float,
    *,
    method: str,
) -> list[AlignedPhrase]:
    if not phrases:
        return []
    weights = [max(1, len(p.normalized.replace(" ", ""))) for p in phrases]
    total = sum(weights)
    span = max(0.05, end_sec - start_sec)
    cursor = start_sec
    out: list[AlignedPhrase] = []
    for i, (p, w) in enumerate(zip(phrases, weights)):
        dur = span * (w / total)
        end = end_sec if i == len(phrases) - 1 else cursor + dur
        out.append(
            AlignedPhrase(
                phrase=p,
                start_sec=float(cursor),
                end_sec=float(max(end, cursor + 0.05)),
                match_score=40.0,
                method=method,
            )
        )
        cursor = end
    return out


def blend_alignments(
    whisper_aligned: list[AlignedPhrase],
    vad_aligned: list[AlignedPhrase],
) -> list[AlignedPhrase]:
    """Prefer strong whisper matches; fall back to VAD timing when weak."""
    if len(whisper_aligned) != len(vad_aligned):
        return whisper_aligned or vad_aligned
    out: list[AlignedPhrase] = []
    for w, v in zip(whisper_aligned, vad_aligned):
        if w.match_score >= 65.0 and w.method.startswith("whisper"):
            out.append(w)
        else:
            out.append(
                AlignedPhrase(
                    phrase=v.phrase,
                    start_sec=v.start_sec,
                    end_sec=v.end_sec,
                    match_score=max(v.match_score, w.match_score * 0.5),
                    method=f"blend:{v.method}",
                )
            )
    return out


def _merge_regions(
    regions: list[tuple[float, float]], target: int
) -> list[tuple[float, float]]:
    regions = list(regions)
    while len(regions) > target:
        # Merge the pair with the smallest intervening gap
        best_i = 0
        best_gap = float("inf")
        for i in range(len(regions) - 1):
            gap = regions[i + 1][0] - regions[i][1]
            if gap < best_gap:
                best_gap = gap
                best_i = i
        a = regions[best_i]
        b = regions[best_i + 1]
        regions[best_i : best_i + 2] = [(a[0], b[1])]
    return regions


def _enforce_monotonic(
    aligned: list[AlignedPhrase], audio_duration: float
) -> list[AlignedPhrase]:
    if not aligned:
        return aligned
    out: list[AlignedPhrase] = []
    prev_end = 0.0
    for a in aligned:
        start = max(float(a.start_sec), prev_end)
        end = max(float(a.end_sec), start + 0.05)
        end = min(end, audio_duration)
        start = min(start, end - 0.01)
        out.append(
            AlignedPhrase(
                phrase=a.phrase,
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                match_score=a.match_score,
                method=a.method,
            )
        )
        prev_end = end
    # Stretch last end to not exceed duration; keep last phrase end at least its value
    if out:
        last = out[-1]
        out[-1] = AlignedPhrase(
            phrase=last.phrase,
            start_sec=last.start_sec,
            end_sec=min(audio_duration, max(last.end_sec, last.start_sec + 0.05)),
            match_score=last.match_score,
            method=last.method,
        )
    return out
