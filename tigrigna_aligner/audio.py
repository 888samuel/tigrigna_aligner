from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioData:
    path: Path
    samples: np.ndarray
    sample_rate: int
    duration_sec: float
    source_path: Path
    decoded_wav: Path | None = None


# Common containers ffmpeg can decode; unknown extensions are still attempted.
KNOWN_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".aifc",
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".mpeg",
    ".mpg",
    ".3gp",
    ".amr",
    ".caf",
}


def find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg and restart the terminal "
            "so any audio/video format can be decoded."
        )
    return ffmpeg


def load_audio(path: str | Path, target_sr: int = 16000) -> AudioData:
    """Load any ffmpeg-supported audio/video file as mono float32 PCM."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")

    samples: np.ndarray | None = None
    decoded_wav: Path | None = None

    # Fast path for plain wav/flac via soundfile
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        try:
            data, sr = sf.read(str(path), always_2d=False)
            if getattr(data, "ndim", 1) > 1:
                data = np.mean(data, axis=1)
            samples = np.asarray(data, dtype=np.float32)
            if sr != target_sr:
                samples = librosa.resample(
                    samples, orig_sr=sr, target_sr=target_sr
                )
        except Exception:
            samples = None

    if samples is None:
        decoded_wav = _ffmpeg_to_wav(path, target_sr=target_sr)
        samples, _ = sf.read(str(decoded_wav), dtype="float32", always_2d=False)
        if getattr(samples, "ndim", 1) > 1:
            samples = np.mean(samples, axis=1)
        samples = np.asarray(samples, dtype=np.float32)

    if samples.size == 0:
        raise ValueError(f"Audio is empty: {path}")

    duration = float(len(samples) / target_sr)
    return AudioData(
        path=path,
        samples=samples,
        sample_rate=target_sr,
        duration_sec=duration,
        source_path=path,
        decoded_wav=decoded_wav,
    )


def ensure_wav_for_model(audio: AudioData, target_sr: int = 16000) -> Path:
    """Return a wav path Whisper can read (reuse decode or write temp wav)."""
    if audio.decoded_wav and audio.decoded_wav.exists():
        return audio.decoded_wav
    if audio.source_path.suffix.lower() == ".wav":
        return audio.source_path
    tmp = Path(tempfile.mkdtemp(prefix="tigrigna_align_")) / "audio_16k.wav"
    write_wav(tmp, audio.samples, target_sr)
    return tmp


def _ffmpeg_to_wav(path: Path, target_sr: int = 16000) -> Path:
    ffmpeg = find_ffmpeg()
    out_dir = Path(tempfile.mkdtemp(prefix="tigrigna_align_"))
    out_wav = out_dir / f"{path.stem}_16k_mono.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to run ffmpeg: {exc}") from exc

    if proc.returncode != 0 or not out_wav.exists():
        err = (proc.stderr or proc.stdout or "").strip()
        tail = err[-800:] if err else "unknown ffmpeg error"
        raise RuntimeError(
            f"ffmpeg could not decode audio '{path.name}'. Details:\n{tail}"
        )
    return out_wav


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate)
    return path


def rms_envelope(
    samples: np.ndarray, sample_rate: int, frame_ms: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = frame
    if len(samples) < frame:
        times = np.array([0.0], dtype=np.float64)
        rms = np.array([float(np.sqrt(np.mean(samples**2)))], dtype=np.float64)
        return times, rms

    n = 1 + (len(samples) - frame) // hop
    rms = np.empty(n, dtype=np.float64)
    times = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = i * hop
        chunk = samples[start : start + frame]
        rms[i] = float(np.sqrt(np.mean(chunk**2)))
        times[i] = (start + frame / 2) / sample_rate
    return times, rms


def detect_silence_mask(
    samples: np.ndarray,
    sample_rate: int,
    *,
    silence_db: float = -40.0,
    frame_ms: float = 20.0,
    min_silence_sec: float = 0.18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, rms_db, is_silence) at frame resolution."""
    times, rms = rms_envelope(samples, sample_rate, frame_ms=frame_ms)
    eps = 1e-10
    rms_db = 20.0 * np.log10(np.maximum(rms, eps))
    is_silence = rms_db < silence_db

    min_frames = max(1, int(min_silence_sec * 1000.0 / frame_ms))
    is_silence = _filter_short_runs(is_silence, min_frames, fill_value=False)
    return times, rms_db, is_silence


def speech_regions(
    samples: np.ndarray,
    sample_rate: int,
    *,
    silence_db: float = -40.0,
    min_speech_sec: float = 0.12,
    pad_sec: float = 0.04,
) -> list[tuple[float, float]]:
    times, _, is_silence = detect_silence_mask(
        samples, sample_rate, silence_db=silence_db
    )
    if times.size == 0:
        return [(0.0, len(samples) / sample_rate)]

    regions: list[tuple[float, float]] = []
    in_speech = False
    start = 0.0
    for t, silent in zip(times, is_silence):
        if not silent and not in_speech:
            in_speech = True
            start = float(t)
        elif silent and in_speech:
            in_speech = False
            end = float(t)
            if end - start >= min_speech_sec:
                regions.append((max(0.0, start - pad_sec), end + pad_sec))
    if in_speech:
        end = float(times[-1])
        if end - start >= min_speech_sec:
            regions.append((max(0.0, start - pad_sec), end + pad_sec))

    duration = len(samples) / sample_rate
    regions = [(s, min(e, duration)) for s, e in regions]
    return regions or [(0.0, duration)]


def find_silence_valley(
    samples: np.ndarray,
    sample_rate: int,
    center_sec: float,
    *,
    search_radius_sec: float = 0.35,
    silence_db: float = -40.0,
) -> float | None:
    """Nearest low-energy frame near center_sec, or None if none found."""
    times, rms_db, _ = detect_silence_mask(
        samples, sample_rate, silence_db=silence_db
    )
    if times.size == 0:
        return None
    lo = center_sec - search_radius_sec
    hi = center_sec + search_radius_sec
    mask = (times >= lo) & (times <= hi)
    if not np.any(mask):
        return None
    local_times = times[mask]
    local_db = rms_db[mask]
    quiet = local_db < silence_db
    if np.any(quiet):
        candidates = local_times[quiet]
        idx = int(np.argmin(np.abs(candidates - center_sec)))
        return float(candidates[idx])
    idx = int(np.argmin(local_db))
    return float(local_times[idx])


def _filter_short_runs(mask: np.ndarray, min_len: int, fill_value: bool) -> np.ndarray:
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        if out[i] and (j - i) < min_len:
            out[i:j] = fill_value
        i = j
    return out
