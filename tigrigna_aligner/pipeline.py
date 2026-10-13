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
from .hierarchy import hierarchical_speech_align
from .match import blend_alignments, force_match_phrases
from .mms_engine import align_with_mms, mms_available
from .pause_dp import refine_with_pause_dp
from .script import load_script
from .snap import snap_to_silence
from .whisper_engine import transcribe_words


@dataclass
class AlignConfig:
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = "am"
    mms_language: str = "tir"
    silence_db: float | None = None
    pad_sec: float = 0.06
    review_threshold: float = 65.0
    # auto | mms | hierarchy | whisper
    engine: str = "auto"
    skip_whisper: bool = True  # Whisper is weak for Tigrinya; off by default
    video_id: str = "episode"
    include_debug: bool = True
    mms_batch_size: int = 2


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

    hierarchy_aligned = hierarchical_speech_align(
        phrases, regions, audio.duration_sec, major_pause_sec=0.35
    )
    hierarchy_aligned = refine_with_pause_dp(
        hierarchy_aligned,
        audio.samples,
        audio.sample_rate,
        audio.duration_sec,
        silence_db=float(silence_db),
    )

    meta: dict[str, Any] = {
        "engine": "tigrigna_aligner",
        "audio_duration_sec": round(audio.duration_sec, 3),
        "audio_format": audio_path.suffix.lower() or "(none)",
        "script_format": script_path.suffix.lower() or "(none)",
        "phrase_count": len(phrases),
        "vad_region_count": len(regions),
        "silence_db": round(float(silence_db), 2),
        "requested_engine": config.engine,
        "whisper": None,
        "mms": None,
    }

    engine = (config.engine or "auto").lower().strip()
    aligned = hierarchy_aligned
    meta["mode"] = "hierarchy+pause_dp+silence_snap"

    # Prefer true MMS CTC forced alignment for Tigrinya when available.
    try_mms = engine in {"auto", "mms"}
    if try_mms and mms_available():
        try:
            mms = align_with_mms(
                model_audio,
                phrases,
                audio.duration_sec,
                language=config.mms_language,
                device=config.device if config.device in {"cpu", "cuda"} else "cpu",
                batch_size=config.mms_batch_size,
            )
            aligned = mms.phrase_aligned
            meta["mode"] = "mms_ctc+silence_snap"
            meta["mms"] = {
                "model": mms.model_name,
                "language": config.mms_language,
                "word_count": len(mms.words),
            }
        except Exception as exc:
            aligned = hierarchy_aligned
            meta["mode"] = "hierarchy+pause_dp+silence_snap"
            meta["mms"] = {"error": str(exc)[:400]}
    elif engine == "mms":
        meta["mms"] = {
            "error": "ctc-forced-aligner/torch not installed. "
            "pip install -r requirements-mms.txt"
        }

    # Optional Whisper refine (usually weak for Tigrinya)
    if (not config.skip_whisper) or engine == "whisper":
        try:
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
            mean_whisper = sum(a.match_score for a in whisper_aligned) / max(
                1, len(whisper_aligned)
            )
            meta["whisper"] = {
                "model": whisper.model_name,
                "detected_language": whisper.language,
                "mean_match_score": round(mean_whisper, 1),
                "word_count": len(whisper.words),
            }
            if mean_whisper >= 62.0:
                aligned = blend_alignments(whisper_aligned, aligned)
                meta["mode"] = f"{meta['mode']}+whisper_blend"
        except Exception as exc:
            meta["whisper"] = {"error": str(exc)[:300]}

    aligned = snap_to_silence(
        aligned,
        audio,
        pad_sec=config.pad_sec,
        silence_db=silence_db,
        search_radius_sec=0.55,
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
