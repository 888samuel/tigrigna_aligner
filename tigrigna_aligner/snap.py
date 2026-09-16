from __future__ import annotations

from .audio import AudioData, find_silence_valley
from .match import AlignedPhrase


def snap_to_silence(
    aligned: list[AlignedPhrase],
    audio: AudioData,
    *,
    pad_sec: float = 0.08,
    search_radius_sec: float = 0.35,
    silence_db: float = -40.0,
    snap_starts: bool = True,
    snap_ends: bool = True,
) -> list[AlignedPhrase]:
    """Move phrase boundaries toward nearby silence valleys, with a small pad."""
    if not aligned:
        return aligned

    duration = audio.duration_sec
    snapped: list[AlignedPhrase] = []

    for i, a in enumerate(aligned):
        start = a.start_sec
        end = a.end_sec

        if snap_starts and i > 0:
            valley = find_silence_valley(
                audio.samples,
                audio.sample_rate,
                start,
                search_radius_sec=search_radius_sec,
                silence_db=silence_db,
            )
            if valley is not None:
                start = valley + pad_sec

        if snap_ends:
            valley = find_silence_valley(
                audio.samples,
                audio.sample_rate,
                end,
                search_radius_sec=search_radius_sec,
                silence_db=silence_db,
            )
            if valley is not None:
                # Keep a little audio after the valley so text doesn't cut early
                end = min(duration, valley + pad_sec)

        snapped.append(
            AlignedPhrase(
                phrase=a.phrase,
                start_sec=start,
                end_sec=end,
                match_score=a.match_score,
                method=f"{a.method}+silence_snap",
            )
        )

    # Re-enforce order and non-overlap after snapping
    out: list[AlignedPhrase] = []
    prev_end = 0.0
    for i, a in enumerate(snapped):
        start = max(a.start_sec, prev_end)
        # Don't invade next phrase's original start too aggressively
        if i + 1 < len(snapped):
            next_start = snapped[i + 1].start_sec
            end = min(a.end_sec, max(start + 0.05, next_start))
        else:
            end = min(duration, max(a.end_sec, start + 0.05))
        start = min(start, end - 0.01)
        out.append(
            AlignedPhrase(
                phrase=a.phrase,
                start_sec=round(max(0.0, start), 3),
                end_sec=round(min(duration, end), 3),
                match_score=a.match_score,
                method=a.method,
            )
        )
        prev_end = out[-1].end_sec

    return out
