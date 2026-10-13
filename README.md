# Tigrigna Forced Aligner

Python CLI that takes a **dry voice-over** and an **exact Tigrinya script** (one phrase per line) and writes phrase timestamps for your Remotion editing engine.

```text
audio VO + script.txt  →  timed-script.json  →  planner / edit-plan  →  Remotion
```

## What it does

This is a **forced aligner**, not free speech-to-text:

1. Your script is ground truth for *what* was said
2. The audio answers *when* each phrase was spoken
3. Output matches the Remotion `timed-script` contract (`id`, `start_sec`, `end_sec`, `text`)
4. Extra QA fields: `duration_sec`, `confidence`, `needs_review`

### Accepted inputs

| Input | Formats |
|-------|---------|
| Audio / video VO | Any ffmpeg-readable file: `mp3`, `wav`, `m4a`, `aac`, `flac`, `ogg`, `opus`, `wma`, `mp4`, `mkv`, `webm`, `mov`, … |
| Script | `txt`, `md`, `csv`, `json`, `srt`, `vtt`, `rtf`, `docx`, `pdf` |

Pipeline:

1. Decode audio via **ffmpeg** to mono 16 kHz
2. Extract script text from txt/docx/pdf/… (NFC normalized)
3. Detect speech / silence regions
4. Run **faster-whisper** for timing cues (script used as `initial_prompt`)
5. **Force-match** each exact script phrase onto Whisper words in order
6. Blend weak matches with VAD timing
7. **Snap boundaries** to silence valleys (+ small pad)

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on `PATH` (required for most audio/video formats)
- ~2–4 GB RAM for Whisper `base` + `int8` (fits 8 GB machines)

## Install

```bash
cd D:\tigrigna_aliner
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Prepare the script

Best accuracy: one spoken phrase per line/paragraph (breath group / on-screen beat).

Works from Word/PDF too — put one phrase per paragraph in `.docx`, or one phrase per line in `.txt`.

```text
ሰላም ንዓኻትኩም ይኹን
ነቶም ዝደኸሙን ዝጠመዩን
እቲ ናይ እግዚኣብሔር ስቕታ
```

Example commands:

```bash
python align.py --audio vo.m4a --script script.docx --output output\timed-script.json
python align.py --audio interview.mp4 --script narration.pdf --vad-only
```

Rules for accuracy:

- Exact spoken wording (not a cleaned essay)
- Dry VO only (no music/SFX under the align track)
- No ad-libs unless you update the script
- Prefer 3–12 words per line/paragraph
- Legacy `.doc` is not supported — save as `.docx` or `.pdf`

## Run on GitHub Actions

1. Put dry VO + script in `inputs/` (defaults: `inputs/output.mp3`, `inputs/script.docx`)
2. Commit and push
3. GitHub → **Actions** → **Align Tigrinya VO** → **Run workflow**
4. Download the `timed-script-…` artifact

Word scripts work as-is — use `inputs/script.docx` (one phrase per paragraph).

Workflow file: `.github/workflows/align.yml`

## Run

Best accuracy (install MMS first):

```bash
pip install -r requirements-mms.txt
python align.py --audio inputs\output.mp3 --script inputs\script.docx --output output\timed-script.json --engine auto
```

`--engine auto` uses **MMS CTC forced alignment** for Tigrinya (`tir`) when installed.
If MMS is missing, it falls back to pause-aware hierarchy alignment.

Fast mode (no MMS download):

```bash
python align.py --audio inputs\output.mp3 --script inputs\script.docx --engine hierarchy
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--model base` | Whisper size: `tiny` / `base` / `small` (default `base`) |
| `--language am` | Ethiopic timing hint (default). Use `--language ""` for auto |
| `--silence-db -40` | Silence threshold for snapping |
| `--pad-sec 0.08` | Keep a little audio after silence so text doesn't cut early |
| `--no-debug` | Emit only Remotion fields |
| `--vad-only` | Skip Whisper |

## Output shape

```json
{
  "version": "1.0",
  "video_id": "my_video",
  "audio": "path/to/output.mp3",
  "segments": [
    {
      "id": "seg_001",
      "start_sec": 0.12,
      "end_sec": 2.41,
      "duration_sec": 2.29,
      "text": "ሰላም ንዓኻትኩም ይኹን",
      "confidence": 0.87,
      "needs_review": false
    }
  ]
}
```

Remotion only needs `id` / `start_sec` / `end_sec` / `text`. Review any segment with `needs_review: true`.

## Accuracy notes (Tigrinya)

Phrase/sentence timing is the production target (±100–150 ms). Perfect word-level karaoke is harder because Tigrinya is low-resource for Whisper. Accuracy depends most on:

1. Script matching the recording
2. Phrase-sized lines
3. Dry VO
4. Silence snapping + light human QA on flagged lines

## Project layout

```text
align.py                 CLI entry
tigrigna_aligner/
  script.py              load txt/docx/pdf/json/... + normalize
  audio.py               ffmpeg decode any audio/video → mono PCM
  whisper_engine.py      faster-whisper timing
  match.py               force-match script ↔ timestamps
  snap.py                silence boundary snap
  confidence.py          confidence + needs_review
  export.py              timed-script.json writer
  pipeline.py            full align pipeline
  cli.py                 argparse CLI
examples/sample_script.txt
examples/sample_script.docx
```
