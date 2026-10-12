# Aligner inputs (for GitHub Actions)

Put your dry voice-over and exact Tigrinya script here, then commit + push, then run **Actions → Align Tigrinya VO**.

Suggested names (workflow defaults):

```text
inputs/output.mp3     # or .wav / .m4a / .mp4 / any ffmpeg format
inputs/script.docx    # Word file is supported (also .txt / .pdf / …)
```

### Using a Word (.docx) script

1. Save your script as `inputs/script.docx`
2. Prefer **one spoken phrase per paragraph** (same idea as one phrase per line in `.txt`)
3. Commit + push
4. Run the workflow (default `script_file` is already `inputs/script.docx`)

If your file has another name, set `script_file` when running the workflow, e.g. `inputs/my_narration.docx`.

Then open **Actions → Align Tigrinya VO → Run workflow**.

Download the `timed-script-…` artifact when the run finishes.

Tips:

- Use dry VO only (no music/SFX under the track you align).
- Prefer one spoken phrase per paragraph.
- Legacy `.doc` is not supported — save as `.docx` in Word.
- Large media: GitHub soft-limits apply; keep episodes short when possible.
