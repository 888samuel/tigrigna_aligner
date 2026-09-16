# Aligner inputs (for GitHub Actions)

Put your dry voice-over and exact Tigrinya script here, then commit + push, then run **Actions → Align Tigrinya VO**.

Suggested names (workflow defaults):

```text
inputs/output.mp3   # or .wav / .m4a / .mp4 / any ffmpeg format
inputs/script.txt   # or .docx / .pdf / .json / .srt …
```

Then open **Actions → Align Tigrinya VO → Run workflow**.

Download the `timed-script-…` artifact when the run finishes.

Tips:

- Use dry VO only (no music/SFX under the track you align).
- Prefer one spoken phrase per line/paragraph.
- Large media: GitHub soft-limits apply; keep episodes short when possible.
