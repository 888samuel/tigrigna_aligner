from __future__ import annotations

from .audio import AudioData, find_silence_valley
from .match import AlignedPhrase


def snap_to_silence(
    aligned: list[AlignedPhrase],
    audio: AudioData,
    *,
    pad_sec: float = 0.05,
    search_radius_sec: float = 0.35,
    silence_db: float = -40.0,
    snap_starts: bool = True,
    snap_ends: bool = True,
    gentle: bool = False,
) -> list[AlignedPhrase]:
    """Move phrase boundaries toward nearby silence valleys.

    When gentle=True (for MMS CTC), only nudge boundaries a little so we do not
    destroy accurate forced-alignment timing.
    """
    if not aligned:
        return aligned

    if gentle:
        search_radius_sec = min(search_radius_sec, 0.14)
        pad_sec = min(pad_sec, 0.04)

    duration = audio.duration_sec
    snapped: list[AlignedPhrase] = []

    for i, a in enumerate(aligned):
        start = a.start_sec
        end = a.end_sec
        original_start, original_end = start, end

        if snap_starts and i > 0:
            valley = find_silence_valley(
                audio.samples,
                audio.sample_rate,
                start,
                search_radius_sec=search_radius_sec,
                silence_db=silence_db,
            )
            if valley is not None:
                candidate = valley + pad_sec
                if gentle:
                    # Only accept small nudges toward silence
                    if abs(candidate - original_start) <= search_radius_sec:
                        start = candidate
                else:
                    start = candidate

        if snap_ends:
            valley = find_silence_valley(
                audio.samples,
                audio.sample_rate,
                end,
                search_radius_sec=search_radius_sec,
                silence_db=silence_db,
            )
            if valley is not None:
                candidate = min(duration, valley + pad_sec)
                if gentle:
                    if abs(candidate - original_end) <= search_radius_sec:
                        end = candidate
                else:
                    end = candidate

        # Never collapse a phrase too aggressively
        min_dur = 0.12 if gentle else 0.08
        if end < start + min_dur:
            end = min(duration, start + min_dur)

        snapped.append(
            AlignedPhrase(
                phrase=a.phrase,
                start_sec=start,
                end_sec=end,
                match_score=a.match_score,
                method=f"{a.method}+silence_snap",
            )
        )

    out: list[AlignedPhrase] = []
    prev_end = 0.0
    for i, a in enumerate(snapped):
        start = max(a.start_sec, prev_end)
        if i + 1 < len(snapped):
            # Keep a tiny gap possibility but don't steal next phrase core
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
