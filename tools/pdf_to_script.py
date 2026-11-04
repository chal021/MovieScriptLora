"""Turn a screenplay PDF into clean text: the official input format.

Every script the LoRA sees, at training time (each example's script.txt)
and at inference time, should come out of this tool. That way the model
trains on exactly the kind of text it will be given.

Steps:
1. `pdftotext -layout`: keeps the screenplay's indentation (character
   names on their own line, dialogue indented), rebuilt as spaces from
   the text's positions on the page. Plain extraction merges a speaker's
   name into their line ("CARMEN I don't understand..."), so it isn't used.
2. Fix characters the PDF's font maps to the wrong Unicode character.
   Copy-paste usually deletes these, which is why pasted scripts look like
   they've "lost punctuation". Found in pans-labyrinth-2006.pdf:
     U+00AD soft hyphen   -> "-" between two letters (eye-glasses),
                             otherwise an em dash ("INT. MAIN CAR — SAME",
                             "stop the car— please—"). 536 times.
     U+037E Greek ";"     -> ";"   ("an old almanac; we're farmers")
     U+00AB "<<"          -> "'"   ("Let's")
     U+2019 curly quote   -> "'"   (normalized so every apostrophe is the same)
     U+2026 ellipsis char -> "..." (normalized to the three-dot form)
   Accented letters (e-acute in Garces, cuple) come through correctly.
3. Page breaks (form feeds) become line breaks. Trailing spaces are
   removed, and runs of 3+ blank lines are cut to 1.

This is deterministic cleanup, so it lives in code, not in what the model
learns. The model never has to guess at broken characters.

Usage:
    python tools/pdf_to_script.py SCRIPT.pdf                 # whole script
    python tools/pdf_to_script.py SCRIPT.pdf -o out.txt
    python tools/pdf_to_script.py SCRIPT.pdf --start "INT. MAIN CAR" --end "The Driver stops."
--start / --end cut an excerpt: from the first line containing --start,
through the first later line containing --end (both inclusive). Matching
ignores runs of spaces, so a phrase split across a line break still won't
match. Pick a phrase that sits on one line.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

EM_DASH = "—"
SOFT_HYPHEN = "­"
# Plain one-to-one replacements (the soft hyphen needs context, see below).
REPLACEMENTS = {
    ";": ";",    # Greek question mark, used by the font as a semicolon
    "«": "'",    # left guillemet, used by the font as an apostrophe
    "’": "'",    # curly apostrophe, normalized
    "…": "...",  # ellipsis character, normalized
}


def extract_layout_bytes(pdf: Path) -> bytes:
    exe = shutil.which("pdftotext")
    if not exe:
        sys.exit("pdftotext not found. It ships with Git for Windows (mingw64/bin) and with poppler-utils.")
    result = subprocess.run([exe, "-layout", "-enc", "UTF-8", str(pdf), "-"], capture_output=True)
    if result.returncode != 0:
        sys.exit(f"pdftotext failed: {result.stderr.decode(errors='replace')}")
    return result.stdout


def fix_characters(raw: bytes) -> str:
    text = raw.decode("utf-8")
    out = []
    for i, ch in enumerate(text):
        if ch == SOFT_HYPHEN:
            before = text[i - 1] if i > 0 else ""
            after = text[i + 1] if i + 1 < len(text) else ""
            out.append("-" if before.isalpha() and after.isalpha() else EM_DASH)
        else:
            out.append(REPLACEMENTS.get(ch, ch))
    return "".join(out)


def tidy(text: str) -> str:
    text = text.replace("\f", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n") + "\n"


def cut(text: str, start: str | None, end: str | None) -> str:
    if not start and not end:
        return text
    squash = lambda s: re.sub(r"\s+", " ", s).strip()
    lines = text.split("\n")
    first = 0
    if start:
        hits = [i for i, l in enumerate(lines) if squash(start) in squash(l)]
        if not hits:
            sys.exit(f"--start text not found: {start!r}")
        first = hits[0]
    last = len(lines) - 1
    if end:
        hits = [i for i, l in enumerate(lines) if i >= first and squash(end) in squash(l)]
        if not hits:
            sys.exit(f"--end text not found after --start: {end!r}")
        last = hits[0]
    return "\n".join(lines[first:last + 1]).strip("\n") + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="write here instead of printing")
    ap.add_argument("--start", help="first line of the excerpt contains this text")
    ap.add_argument("--end", help="last line of the excerpt contains this text")
    args = ap.parse_args()

    text = cut(tidy(fix_characters(extract_layout_bytes(args.pdf))), args.start, args.end)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {len(text.splitlines())} lines to {args.out}", file=sys.stderr)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
