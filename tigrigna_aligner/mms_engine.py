from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .match import AlignedPhrase, WordStamp, _enforce_monotonic
from .script import Phrase, normalize_tigrinya

MODEL_NAME = "MahmoudAshraf/mms-300m-1130-forced-aligner"
SAMPLE_RATE = 16000


@dataclass
class MmsResult:
    words: list[WordStamp]
    phrase_aligned: list[AlignedPhrase]
    model_name: str
    mean_token_score: float = 0.0


@dataclass
class _Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class _Segment:
    label: str
    start: int
    end: int
    score: float


def mms_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=2)
def _load_model(device: str, dtype_name: str):
    import torch
    from transformers import AutoModelForCTC, AutoTokenizer

    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    model = AutoModelForCTC.from_pretrained(MODEL_NAME).to(device=device, dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def align_with_mms(
    audio_path: str | Path,
    phrases: list[Phrase],
    audio_duration: float,
    *,
    language: str = "tir",
    device: str = "cpu",
    batch_size: int = 1,
) -> MmsResult:
    """Accurate forced alignment with MMS CTC + proper trellis backtracking."""
    import torch

    audio_path = Path(audio_path)
    dtype_name = "float16" if device == "cuda" else "float32"
    model, tokenizer = _load_model(device, dtype_name)
    dtype = torch.float16 if dtype_name == "float16" else torch.float32

    waveform = _load_waveform(audio_path, device=device, dtype=dtype)
    emissions, stride_ms = _generate_emissions(
        model, waveform, batch_size=max(1, batch_size)
    )
    expected_frames = max(1, int(round(waveform.numel() / float(
        getattr(model.config, "inputs_to_logits_ratio", 320)
    ))))
    # Keep emission length honest vs audio (chunking can drift by a few frames)
    if emissions.size(0) > expected_frames + 5:
        emissions = emissions[:expected_frames]
    audio_len_sec = float(waveform.numel()) / SAMPLE_RATE

    roman_words, phrase_word_counts = _romanize_phrases(phrases, language=language)
    if not roman_words:
        raise RuntimeError("MMS romanization produced empty text")

    token_ids, word_token_spans, token_labels = _tokenize_words(roman_words, tokenizer)
    blank_id = _blank_id(tokenizer)

    # emissions: T x C  (log-softmax)
    path = _backtrack(
        _get_trellis(emissions, token_ids, blank_id=blank_id),
        emissions,
        token_ids,
        blank_id=blank_id,
    )
    char_segments = _merge_repeats(path, token_labels)
    words, mean_score = _segments_to_words(
        char_segments,
        word_token_spans,
        roman_words,
        stride_ms=stride_ms,
    )
    words = _ensure_full_coverage(words, audio_len_sec)
    phrase_aligned = _group_words_into_phrases(
        phrases, words, phrase_word_counts, audio_duration, mean_score=mean_score
    )
    return MmsResult(
        words=words,
        phrase_aligned=phrase_aligned,
        model_name=MODEL_NAME,
        mean_token_score=mean_score,
    )


def _load_waveform(audio_path: Path, *, device: str, dtype):
    import subprocess

    import torch

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(audio_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"ffmpeg failed to decode audio for MMS: {err}")
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        raise RuntimeError("Decoded audio is empty")
    return torch.from_numpy(audio).to(device=device, dtype=dtype)


def _generate_emissions(model, waveform, *, batch_size: int = 1, window_sec: float = 25.0):
    import math

    import torch

    ratio = int(getattr(model.config, "inputs_to_logits_ratio", 320))
    window = (int(window_sec * SAMPLE_RATE) // ratio) * ratio
    context = (int(2.0 * SAMPLE_RATE) // ratio) * ratio

    if waveform.numel() <= window:
        with torch.inference_mode():
            logits = model(waveform.unsqueeze(0)).logits[0]
        emissions = torch.log_softmax(logits.float(), dim=-1).cpu()
        return emissions, ratio * 1000.0 / SAMPLE_RATE

    extension = math.ceil(waveform.numel() / window) * window - waveform.numel()
    padded = torch.nn.functional.pad(waveform, (context, context + extension))
    chunks = padded.unfold(0, window + 2 * context, window)
    context_frames = context // ratio
    window_frames = window // ratio

    outs = []
    with torch.inference_mode():
        for i in range(0, chunks.size(0), batch_size):
            batch = chunks[i : i + batch_size]
            logits = model(batch).logits
            outs.append(logits[:, context_frames : context_frames + window_frames])
    emissions = torch.cat(outs, dim=0).flatten(0, 1)
    if extension > 0:
        trim = extension // ratio
        if trim > 0:
            emissions = emissions[:-trim]
    emissions = torch.log_softmax(emissions.float(), dim=-1).cpu()
    return emissions, ratio * 1000.0 / SAMPLE_RATE


def _romanize_phrases(
    phrases: list[Phrase], *, language: str
) -> tuple[list[str], list[int]]:
    try:
        from uroman import Uroman

        romanizer = Uroman()
        use_uroman = True
    except Exception:
        romanizer = None
        use_uroman = False

    words: list[str] = []
    counts: list[int] = []
    for phrase in phrases:
        text = phrase.text.strip()
        if use_uroman:
            roman = romanizer.romanize_string(text, lcode=language)
        else:
            roman = "".join(ch if ord(ch) < 128 else " " for ch in text)
        # Keep alnum tokens; drop leftover punctuation from romanizer
        cleaned = []
        for tok in roman.replace("-", " ").replace("'", " ").split():
            tok = "".join(ch for ch in tok.lower() if ch.isalnum())
            if tok:
                cleaned.append(tok)
        if not cleaned:
            cleaned = ["a"]
        counts.append(len(cleaned))
        words.extend(cleaned)
    return words, counts


def _blank_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    vocab = tokenizer.get_vocab()
    for key in ("<pad>", "[PAD]", "<blank>"):
        if key in vocab:
            return int(vocab[key])
    return 0


def _tokenize_words(
    words: list[str], tokenizer
) -> tuple[list[int], list[tuple[int, int]], list[str]]:
    """Build CTC token stream with optional '|' word delimiters."""
    vocab = {k.lower(): int(v) for k, v in tokenizer.get_vocab().items()}
    word_delim = vocab.get("|")

    token_ids: list[int] = []
    token_labels: list[str] = []
    spans: list[tuple[int, int]] = []

    for wi, w in enumerate(words):
        if wi > 0 and word_delim is not None:
            token_ids.append(word_delim)
            token_labels.append("|")

        start = len(token_ids)
        i = 0
        matched_any = False
        while i < len(w):
            matched = False
            for size in range(min(8, len(w) - i), 0, -1):
                piece = w[i : i + size]
                if piece in vocab:
                    token_ids.append(vocab[piece])
                    token_labels.append(piece)
                    i += size
                    matched = True
                    matched_any = True
                    break
            if not matched:
                i += 1
        if not matched_any:
            filler_label = "a" if "a" in vocab else next(iter(vocab.keys()))
            token_ids.append(vocab[filler_label])
            token_labels.append(filler_label)
        spans.append((start, len(token_ids)))

    if not token_ids:
        raise RuntimeError("No tokens produced for MMS alignment")
    return token_ids, spans, token_labels


def _get_trellis(emission, tokens: list[int], *, blank_id: int):
    """Standard CTC trellis used by the PyTorch forced-alignment tutorial."""
    import torch

    if not isinstance(emission, torch.Tensor):
        emission = torch.tensor(emission, dtype=torch.float32)
    else:
        emission = emission.float()

    num_frame = emission.size(0)
    num_tokens = len(tokens)
    trellis = torch.full((num_frame + 1, num_tokens + 1), -1.0e9)
    trellis[0, 0] = 0.0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, 1:] = -1.0e9

    token_tensor = torch.tensor(tokens, dtype=torch.long)
    for t in range(num_frame):
        stay = trellis[t, 1:] + emission[t, blank_id]
        change = trellis[t, :-1] + emission[t, token_tensor]
        trellis[t + 1, 1:] = torch.maximum(stay, change)
    return trellis


def _backtrack(trellis, emission, tokens: list[int], *, blank_id: int) -> list[_Point]:
    import torch

    if not isinstance(emission, torch.Tensor):
        emission = torch.tensor(emission, dtype=torch.float32)
    else:
        emission = emission.float()

    # Always backtrack from the final frame so the path spans the full audio.
    # (argmax can finish early on low-resource languages and crush timings.)
    t = trellis.size(0) - 1
    j = trellis.size(1) - 1
    path: list[_Point] = []

    while j > 0 and t > 0:
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        use_token = bool(changed > stayed)
        token_id = tokens[j - 1] if use_token else blank_id
        prob = float(emission[t - 1, token_id].exp().clamp(min=1e-8))
        path.append(_Point(token_index=j - 1, time_index=t - 1, score=prob))
        if use_token:
            j -= 1
        t -= 1

    # If some leading tokens remain, pin them to the start frames.
    while j > 0:
        path.append(_Point(token_index=j - 1, time_index=max(0, t), score=1e-6))
        j -= 1

    return path[::-1]


def _merge_repeats(path: list[_Point], token_labels: list[str]) -> list[_Segment]:
    if not path:
        return []
    segments: list[_Segment] = []
    i1 = 0
    while i1 < len(path):
        i2 = i1
        score_sum = 0.0
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            score_sum += path[i2].score
            i2 += 1
        token_index = path[i1].token_index
        label = token_labels[token_index] if 0 <= token_index < len(token_labels) else "?"
        segments.append(
            _Segment(
                label=label,
                start=path[i1].time_index,
                end=path[i2 - 1].time_index + 1,
                score=score_sum / max(1, (i2 - i1)),
            )
        )
        i1 = i2
    return segments


def _segments_to_words(
    segments: list[_Segment],
    word_token_spans: list[tuple[int, int]],
    roman_words: list[str],
    *,
    stride_ms: float,
) -> tuple[list[WordStamp], float]:
    """Map CTC token segments onto words using exact token spans.

    After trellis backtrack + merge_repeats, there is one segment per CTC target
    token (including '|' word delimiters).
    """
    n_tokens = len(segments)
    if n_tokens == 0:
        return [], 0.0

    words: list[WordStamp] = []
    scores: list[float] = []

    for (tok_start, tok_end), roman in zip(word_token_spans, roman_words):
        tok_start = max(0, min(tok_start, max(0, n_tokens - 1)))
        tok_end = max(tok_start + 1, min(tok_end, n_tokens))
        segs = segments[tok_start:tok_end]
        content = [s for s in segs if s.label != "|"] or segs
        f0 = content[0].start
        f1 = content[-1].end
        start_sec = f0 * stride_ms / 1000.0
        end_sec = max(start_sec + 0.03, f1 * stride_ms / 1000.0)
        score = float(np.mean([s.score for s in content]))
        scores.append(score)
        words.append(
            WordStamp(
                text=roman,
                normalized=normalize_tigrinya(roman),
                start=float(start_sec),
                end=float(end_sec),
            )
        )

    mean_score = float(np.mean(scores)) if scores else 0.0
    return words, mean_score


def _ensure_full_coverage(
    words: list[WordStamp], audio_len_sec: float
) -> list[WordStamp]:
    """If CTC path collapses early, stretch word timings across the full audio.

    This preserves relative CTC order while ensuring phrase times cover the VO.
    """
    if not words or audio_len_sec <= 0.05:
        return words
    start0 = words[0].start
    end_n = max(w.end for w in words)
    covered = end_n - start0
    if covered <= 0.05:
        return words
    # If we already cover most of the file, keep CTC times as-is.
    if end_n >= audio_len_sec * 0.90 and start0 <= audio_len_sec * 0.05:
        return words

    # Stretch into [0, audio_len]
    scale = audio_len_sec / covered
    out: list[WordStamp] = []
    for w in words:
        s = max(0.0, (w.start - start0) * scale)
        e = max(s + 0.03, (w.end - start0) * scale)
        out.append(
            WordStamp(
                text=w.text,
                normalized=w.normalized,
                start=float(s),
                end=float(min(audio_len_sec, e)),
            )
        )
    if out:
        out[0] = WordStamp(
            text=out[0].text,
            normalized=out[0].normalized,
            start=0.0,
            end=out[0].end,
        )
        last = out[-1]
        out[-1] = WordStamp(
            text=last.text,
            normalized=last.normalized,
            start=last.start,
            end=audio_len_sec,
        )
    return out


def _group_words_into_phrases(
    phrases: list[Phrase],
    words: list[WordStamp],
    phrase_word_counts: list[int],
    audio_duration: float,
    *,
    mean_score: float,
) -> list[AlignedPhrase]:
    if not words:
        from .match import proportional_align

        return proportional_align(
            phrases, 0.0, audio_duration, method="mms_empty_words"
        )

    if len(words) != sum(phrase_word_counts):
        # Should be exact because we built counts from the same romanization.
        # If not, consume sequentially with counts clipped.
        pass

    # mean_score is already a probability in ~[0,1] after .exp()
    match_score = max(75.0, min(99.0, 70.0 + mean_score * 28.0))

    aligned: list[AlignedPhrase] = []
    cursor = 0
    for phrase, count in zip(phrases, phrase_word_counts):
        remaining_phrases = len(phrases) - len(aligned)
        remaining_words = len(words) - cursor
        if remaining_words <= 0:
            last_end = aligned[-1].end_sec if aligned else 0.0
            from .match import proportional_align

            aligned.extend(
                proportional_align(
                    [phrase], last_end, audio_duration, method="mms_tail"
                )
            )
            continue
        take = max(1, min(count, remaining_words - (remaining_phrases - 1)))
        chunk = words[cursor : cursor + take]
        start = float(chunk[0].start)
        end = float(max(chunk[-1].end, start + 0.05))
        aligned.append(
            AlignedPhrase(
                phrase=phrase,
                start_sec=start,
                end_sec=end,
                match_score=float(match_score),
                method="mms_ctc",
            )
        )
        cursor += take

    return _enforce_monotonic(aligned, audio_duration)
