from __future__ import annotations

from .match import AlignedPhrase, phrase_weight, speech_axis_align, _enforce_monotonic
from .script import Phrase


def hierarchical_speech_align(
    phrases: list[Phrase],
    regions: list[tuple[float, float]],
    audio_duration: float,
    *,
    major_pause_sec: float = 0.28,
) -> list[AlignedPhrase]:
    """Align phrases in chapters split by longer pauses, then by speech-axis weight.

    Global proportional placement drifts when speaking rate changes. Splitting on
    longer pauses and allocating phrases chapter-by-chapter keeps local fit tighter.
    """
    if not phrases:
        return []
    if not regions:
        return speech_axis_align(phrases, [(0.0, audio_duration)], audio_duration)

    chapters = _build_chapters(regions, major_pause_sec=major_pause_sec)
    if len(chapters) <= 1:
        return speech_axis_align(phrases, regions, audio_duration)

    chapter_speech = [max(0.05, c[1] - c[0]) for c in chapters]
    total_speech = sum(chapter_speech)
    weights = [phrase_weight(p) for p in phrases]
    total_w = sum(weights) or 1.0

    # Target cumulative speech fraction for each chapter end
    chapter_frac_ends: list[float] = []
    acc = 0.0
    for dur in chapter_speech:
        acc += dur / total_speech
        chapter_frac_ends.append(acc)

    # Assign phrases so cumulative weight tracks chapter speech fractions
    assignments: list[list[Phrase]] = [[] for _ in chapters]
    phrase_idx = 0
    cum_w = 0.0
    for ci, frac_end in enumerate(chapter_frac_ends):
        if ci == len(chapters) - 1:
            assignments[ci].extend(phrases[phrase_idx:])
            phrase_idx = len(phrases)
            break
        while phrase_idx < len(phrases):
            next_w = weights[phrase_idx]
            # Would adding this phrase overshoot this chapter's speech share too much?
            # Allow slight overshoot; keep at least one phrase opportunity later.
            remaining_chapters = len(chapters) - ci
            remaining_phrases = len(phrases) - phrase_idx
            if remaining_phrases <= remaining_chapters:
                # leave one phrase for each remaining chapter including current
                if not assignments[ci]:
                    assignments[ci].append(phrases[phrase_idx])
                    cum_w += next_w
                    phrase_idx += 1
                break
            projected = (cum_w + next_w) / total_w
            if assignments[ci] and projected > frac_end + 0.02:
                break
            assignments[ci].append(phrases[phrase_idx])
            cum_w += next_w
            phrase_idx += 1
            if projected >= frac_end:
                break

    # Rescue any unassigned phrases into the last chapter
    if phrase_idx < len(phrases):
        assignments[-1].extend(phrases[phrase_idx:])

    aligned: list[AlignedPhrase] = []
    for chapter_phrases, (cs, ce) in zip(assignments, chapters):
        if not chapter_phrases:
            continue
        # Local regions clipped to chapter
        local_regions = [
            (max(cs, rs), min(ce, re))
            for rs, re in regions
            if re > cs and rs < ce
        ]
        local_regions = [(s, e) for s, e in local_regions if e - s > 0.04]
        if not local_regions:
            local_regions = [(cs, ce)]
        part = speech_axis_align(chapter_phrases, local_regions, audio_duration)
        for a in part:
            aligned.append(
                AlignedPhrase(
                    phrase=a.phrase,
                    start_sec=a.start_sec,
                    end_sec=a.end_sec,
                    match_score=max(a.match_score, 76.0),
                    method="hierarchy+speech_axis",
                )
            )

    return _enforce_monotonic(aligned, audio_duration)


def _build_chapters(
    regions: list[tuple[float, float]],
    *,
    major_pause_sec: float,
) -> list[tuple[float, float]]:
    if not regions:
        return []
    chapters: list[tuple[float, float]] = []
    start = regions[0][0]
    end = regions[0][1]
    for i in range(1, len(regions)):
        gap = regions[i][0] - regions[i - 1][1]
        if gap >= major_pause_sec:
            chapters.append((start, end))
            start = regions[i][0]
        end = regions[i][1]
    chapters.append((start, end))
    return chapters
