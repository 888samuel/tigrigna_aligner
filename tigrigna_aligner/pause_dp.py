from __future__ import annotations

import numpy as np

from .audio import detect_silence_mask, rms_envelope
from .match import AlignedPhrase, Phrase, phrase_weight, _enforce_monotonic


def refine_with_pause_dp(
    seed: list[AlignedPhrase],
    samples: np.ndarray,
    sample_rate: int,
    audio_duration: float,
    *,
    silence_db: float,
) -> list[AlignedPhrase]:
    """Snap phrase boundaries to real pauses using DP around seed timings.

    Seed alignment (speech-axis/hierarchy) gives rough order. This stage locks
    each boundary onto the strongest nearby energy valley so text follows VO
    breaths instead of pure character proportions.
    """
    if len(seed) < 2:
        return seed

    phrases = [a.phrase for a in seed]
    weights = [phrase_weight(p) for p in phrases]
    total_w = sum(weights) or 1.0

    # Expected cumulative end times from seed (soft targets)
    targets = [a.end_sec for a in seed[:-1]]

    candidates, strengths = _pause_candidates(
        samples, sample_rate, audio_duration, silence_db=silence_db
    )
    if len(candidates) < 3:
        return seed

    n_bounds = len(targets)
    m = len(candidates)
    # Keep candidate search local to the seed timing to avoid wild jumps.
    window = max(1.2, audio_duration / max(8, len(phrases)) * 1.8)

    INF = 1e18
    dp = [[INF] * m for _ in range(n_bounds + 1)]
    prev = [[-1] * m for _ in range(n_bounds + 1)]

    start_t = max(0.0, seed[0].start_sec)
    for j, ct in enumerate(candidates):
        if ct <= start_t + 0.12:
            continue
        if abs(ct - targets[0]) > window:
            continue
        dur = ct - start_t
        exp = max(0.25, (seed[0].end_sec - start_t))
        target_pen = abs(ct - targets[0]) * 1.6
        dur_pen = abs(dur - exp) / max(0.35, exp)
        pause_bonus = -1.0 * strengths[j]
        if dur < 0.2 or dur > exp * 2.8 + 1.2:
            continue
        dp[1][j] = target_pen + dur_pen + pause_bonus
        prev[1][j] = -1

    for i in range(1, n_bounds):
        exp = max(0.25, seed[i].end_sec - seed[i].start_sec)
        target = targets[i]
        for j, ct in enumerate(candidates):
            if abs(ct - target) > window:
                continue
            best = INF
            best_k = -1
            # Only look back over a limited previous-candidate range
            k_min = 0
            for k in range(j - 1, -1, -1):
                if candidates[j] - candidates[k] > exp * 3.0 + 2.5:
                    break
                k_min = k
            for k in range(k_min, j):
                if dp[i][k] >= INF / 2:
                    continue
                prev_t = candidates[k]
                dur = ct - prev_t
                if dur < 0.18:
                    continue
                if dur > exp * 2.8 + 1.5:
                    continue
                target_pen = abs(ct - target) * 1.6
                dur_pen = abs(dur - exp) / max(0.35, exp)
                pause_bonus = -1.0 * strengths[j]
                cost = dp[i][k] + target_pen + dur_pen + pause_bonus
                if cost < best:
                    best = cost
                    best_k = k
            if best_k >= 0:
                dp[i + 1][j] = best
                prev[i + 1][j] = best_k

    # Reconstruct
    last_i = n_bounds
    end_j = min(range(m), key=lambda j: dp[last_i][j])
    if dp[last_i][end_j] >= INF / 2:
        return seed

    chosen_idx = [end_j]
    cur = end_j
    for i in range(last_i, 0, -1):
        cur = prev[i][cur]
        if cur < 0:
            break
        chosen_idx.append(cur)
    chosen_idx = list(reversed(chosen_idx))
    if len(chosen_idx) != n_bounds:
        return seed

    ends = [candidates[j] for j in chosen_idx]
    aligned: list[AlignedPhrase] = []
    cursor = start_t
    for i, phrase in enumerate(phrases):
        end = ends[i] if i < len(ends) else audio_duration
        if i == len(phrases) - 1:
            end = max(end, audio_duration - 0.02)
        # Soft minimum duration (don't steal large spans from neighbors)
        min_dur = max(0.20, min(0.55, 0.18 * len(phrase.text.split())))
        if end < cursor + min_dur and i < len(phrases) - 1:
            end = cursor + min_dur
        aligned.append(
            AlignedPhrase(
                phrase=phrase,
                start_sec=float(cursor),
                end_sec=float(max(end, cursor + 0.05)),
                match_score=84.0,
                method="pause_dp",
            )
        )
        cursor = end

    return _enforce_monotonic(aligned, audio_duration)


def _pause_candidates(
    samples: np.ndarray,
    sample_rate: int,
    duration: float,
    *,
    silence_db: float,
) -> tuple[list[float], list[float]]:
    times, rms = rms_envelope(samples, sample_rate, frame_ms=20.0)
    if times.size == 0:
        return [0.0, duration], [0.0, 0.0]

    eps = 1e-10
    rms_db = 20.0 * np.log10(np.maximum(rms, eps))
    # Smooth
    win = 3
    kernel = np.ones(win) / win
    smooth = np.convolve(rms_db, kernel, mode="same")

    _, _, is_silence = detect_silence_mask(
        samples, sample_rate, silence_db=silence_db, min_silence_sec=0.08
    )

    cands: list[float] = [0.0]
    strengths: list[float] = [1.0]

    # Midpoints of silence runs
    n = len(is_silence)
    i = 0
    while i < n:
        if not is_silence[i]:
            i += 1
            continue
        j = i
        while j < n and is_silence[j]:
            j += 1
        mid = float(times[(i + j - 1) // 2])
        depth = float(silence_db - np.min(smooth[i:j]))
        cands.append(mid)
        strengths.append(max(0.1, depth))
        i = j

    # Local minima (breath dips) even if not fully silent
    for k in range(2, len(smooth) - 2):
        if smooth[k] <= smooth[k - 1] and smooth[k] <= smooth[k + 1]:
            if smooth[k] <= silence_db + 6.0:
                t = float(times[k])
                if abs(t - cands[-1]) < 0.12:
                    # keep stronger
                    if (silence_db - smooth[k]) > strengths[-1]:
                        strengths[-1] = float(silence_db - smooth[k])
                    continue
                cands.append(t)
                strengths.append(max(0.05, float(silence_db - smooth[k])))

    cands.append(duration)
    strengths.append(1.0)

    # Deduplicate sorted
    paired = sorted(zip(cands, strengths), key=lambda x: x[0])
    out_t: list[float] = []
    out_s: list[float] = []
    for t, s in paired:
        if out_t and abs(t - out_t[-1]) < 0.1:
            if s > out_s[-1]:
                out_t[-1] = t
                out_s[-1] = s
            continue
        out_t.append(t)
        out_s.append(s)
    return out_t, out_s
