from __future__ import annotations

import numpy as np

from .audio import AudioData, rms_envelope
from .match import AlignedPhrase, phrase_weight


def trim_leading_trailing_silence(
    aligned: list[AlignedPhrase],
    audio: AudioData,
    *,
    silence_db: float,
    keep_pad_sec: float = 0.05,
) -> list[AlignedPhrase]:
    """Tighten each phrase to first→last speech inside its window, then
    cap extreme outliers using speaking-weight expectations.
    """
    if not aligned:
        return aligned

    times, rms = rms_envelope(audio.samples, audio.sample_rate, frame_ms=20.0)
    if times.size == 0:
        return aligned
    eps = 1e-10
    rms_db = 20.0 * np.log10(np.maximum(rms, eps))

    weights = [phrase_weight(a.phrase) for a in aligned]
    total_w = sum(weights) or 1.0
    # Expected durations over the active span of the alignment
    span0 = aligned[0].start_sec
    span1 = aligned[-1].end_sec
    total_span = max(0.5, span1 - span0)

    out: list[AlignedPhrase] = []
    for a, w in zip(aligned, weights):
        start, end = a.start_sec, a.end_sec
        mask = (times >= start) & (times <= end)
        if np.any(mask):
            local_t = times[mask]
            local_db = rms_db[mask]
            speech = local_db >= silence_db
            if np.any(speech):
                speech_idx = np.where(speech)[0]
                first = float(local_t[speech_idx[0]])
                last = float(local_t[speech_idx[-1]])
                start = max(start, first - keep_pad_sec)
                end = min(end, last + keep_pad_sec)

        expected = total_span * (w / total_w)
        min_dur = max(0.22, 0.20 * max(1, len(a.phrase.text.split())))
        max_dur = max(min_dur * 1.5, expected * 2.1)

        dur = end - start
        if dur > max_dur + 0.35:
            # Keep the densest speech mass: center on median speech time
            mask = (times >= a.start_sec) & (times <= a.end_sec)
            if np.any(mask):
                local_t = times[mask]
                local_db = rms_db[mask]
                speech_t = local_t[local_db >= silence_db]
                if speech_t.size:
                    center = float(np.median(speech_t))
                    half = max_dur / 2.0
                    start = max(a.start_sec, center - half)
                    end = min(a.end_sec, center + half)
                    if end - start < min_dur:
                        end = min(a.end_sec, start + min_dur)

        if end < start + 0.12:
            start, end = a.start_sec, a.end_sec

        out.append(
            AlignedPhrase(
                phrase=a.phrase,
                start_sec=start,
                end_sec=end,
                match_score=a.match_score,
                method=f"{a.method}+trim",
            )
        )

    fixed: list[AlignedPhrase] = []
    prev_end = 0.0
    duration = audio.duration_sec
    for i, a in enumerate(out):
        start = max(a.start_sec, prev_end)
        if i + 1 < len(out):
            end = min(a.end_sec, max(start + 0.08, out[i + 1].start_sec))
        else:
            end = min(duration, max(a.end_sec, start + 0.08))
        fixed.append(
            AlignedPhrase(
                phrase=a.phrase,
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                match_score=a.match_score,
                method=a.method,
            )
        )
        prev_end = fixed[-1].end_sec
    return fixed
