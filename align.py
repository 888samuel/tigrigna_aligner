#!/usr/bin/env python3
"""CLI entrypoint: python align.py --audio vo.mp3 --script script.txt"""

from tigrigna_aligner.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
