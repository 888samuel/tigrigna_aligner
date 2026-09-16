from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .match import WordStamp, words_from_segments


@dataclass
class WhisperResult:
    language: str | None
    segments: list[dict]
    words: list[WordStamp]
    model_name: str


def transcribe_words(
    audio_path: str | Path,
    *,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    initial_prompt: str | None = None,
) -> WhisperResult:
    """Transcribe with faster-whisper and return word-level stamps when available.

    Tigrinya is low-resource for Whisper. We still use timestamps as timing cues,
    then force-match the exact script phrases afterward.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Prefer Amharic as closest Ethiopic cue if caller did not set a language.
    # Multilingual auto-detect also works; Amharic often helps Ethiopic script VO.
    kwargs = {
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 200},
        "condition_on_previous_text": True,
    }
    if language:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt[:800]

    segments_iter, info = model.transcribe(str(audio_path), **kwargs)

    segments: list[dict] = []
    for seg in segments_iter:
        words = []
        for w in seg.words or []:
            words.append(
                {
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end),
                    "probability": float(getattr(w, "probability", 0.0) or 0.0),
                }
            )
        segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text,
                "words": words,
            }
        )

    words = words_from_segments(segments)
    detected = getattr(info, "language", None)
    return WhisperResult(
        language=detected,
        segments=segments,
        words=words,
        model_name=model_size,
    )
