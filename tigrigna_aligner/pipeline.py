from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import (
    ensure_wav_for_model,
    estimate_silence_db,
    load_audio,
    speech_regions,
)
from .confidence import ScoredSegment, score_segments
from .export import to_timed_script, write_timed_script
from .match import (
    align_phrases_to_regions,
    blend_alignments,
    force_match_phrases,
)
from .script import load_script
from .snap import snap_to_silence
from .whisper_engine import transcribe_words


@dataclass
class AlignConfig:
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = "am"  # closest Ethiopic cue for Whisper
    silence_db: float | None = None  # None = adaptive from audio
    pad_sec: float = 0.08
    review_threshold: float = 65.0
    skip_whisper: bool = False
    video_id: str = "episode"
    include_debug: bool = True


def align_audio_to_script(
    audio_path: str | Path,
    script_path: str | Path,
    *,
    config: AlignConfig | None = None,
) -> tuple[list[ScoredSegment], dict[str, Any]]:
    config = config or AlignConfig()
    audio_path = Path(audio_path)
    script_path = Path(script_path)

    phrases = load_script(script_path)
    audio = load_audio(audio_path)
    model_audio = ensure_wav_for_model(audio)

    silence_db = config.silence_db
    if silence_db is None:
        silence_db = estimate_silence_db(audio.samples, audio.sample_rate)

    regions = speech_regions(
        audio.samples,
        audio.sample_rate,
        silence_db=silence_db,
    )
    # Primary timing for Tigrinya: speech-axis proportional placement
    vad_aligned = align_phrases_to_regions(phrases, regions, audio.duration_sec)

    meta: dict[str, Any] = {
        "engine": "tigrigna_aligner",
        "audio_duration_sec": round(audio.duration_sec, 3),
        "audio_format": audio_path.suffix.lower() or "(none)",
        "script_format": script_path.suffix.lower() or "(none)",
        "phrase_count": len(phrases),
        "vad_region_count": len(regions),
        "silence_db": round(float(silence_db), 2),
        "whisper": None,
    }

    if config.skip_whisper:
        aligned = vad_aligned
        meta["mode"] = "speech_axis+silence_snap"
    else:
        prompt = "\n".join(p.text for p in phrases)
        whisper = transcribe_words(
            model_audio,
            model_size=config.model_size,
            device=config.device,
            compute_type=config.compute_type,
            language=config.language,
            initial_prompt=prompt,
        )
        whisper_aligned = force_match_phrases(
            phrases,
            whisper.words,
            audio.duration_sec,
        )
        # Only trust Whisper when text matching is actually strong; otherwise
        # keep speech-axis timing (Whisper is weak on Tigrinya).
        mean_whisper = (
            sum(a.match_score for a in whisper_aligned) / max(1, len(whisper_aligned))
        )
        if mean_whisper >= 60.0:
            aligned = blend_alignments(whisper_aligned, vad_aligned)
            meta["mode"] = "whisper+speech_axis+silence_snap"
        else:
            aligned = vad_aligned
            meta["mode"] = "speech_axis+silence_snap(whisper_weak)"
        meta["whisper"] = {
            "model": whisper.model_name,
            "detected_language": whisper.language,
            "requested_language": config.language,
            "word_count": len(whisper.words),
            "segment_count": len(whisper.segments),
            "mean_match_score": round(mean_whisper, 1),
        }

    aligned = snap_to_silence(
        aligned,
        audio,
        pad_sec=config.pad_sec,
        silence_db=silence_db,
    )
    scored = score_segments(aligned, review_threshold=config.review_threshold)
    return scored, meta


def run_alignment(
    audio_path: str | Path,
    script_path: str | Path,
    output_path: str | Path,
    *,
    config: AlignConfig | None = None,
) -> Path:
    config = config or AlignConfig()
    scored, meta = align_audio_to_script(audio_path, script_path, config=config)
    payload = to_timed_script(
        scored,
        video_id=config.video_id,
        audio_path=str(Path(audio_path).as_posix()),
        meta=meta,
        include_debug=config.include_debug,
    )
    return write_timed_script(output_path, payload)
