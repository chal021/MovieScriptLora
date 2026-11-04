"""Validate the hand-written examples and pack them into train.jsonl.

Each example lives in its own folder under dataset/examples/:
    script.txt       the raw script excerpt (the model's input)
    breakdown.json   the correct scene/shot breakdown (the model's target)

You write examples as readable files; this script turns them into the
one-JSON-object-per-line "messages" format that TRL's SFTTrainer reads:
    {"messages": [{"role": "system", ...}, {"role": "user", ...},
                  {"role": "assistant", ...}]}

Usage:  python dataset/build_jsonl.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
EXAMPLES = HERE / "examples"
OUT = HERE / "train.jsonl"

MIN_SECONDS, MAX_SECONDS = 4.0, 14.0
D_TAG = re.compile(r"<d>\[([^\]]+)\] (.+?)</d>", re.DOTALL)

# Pacing convention (see PLAN.md): spoken words / 2.5, plus action time,
# plus at least 1.5s of margin after the last word. The script can't count
# action time, so it checks only a floor: every shot must allow at least
# words / 2.5 + 1.5 seconds. A shot below that would cut off its dialogue.
SPOKEN_WORDS_PER_SECOND = 2.5
DIALOGUE_MARGIN_SECONDS = 1.5


def check(breakdown: dict, script: str) -> list[str]:
    problems = []
    scenes = breakdown.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return ["'scenes' must be a non-empty list"]
    for si, scene in enumerate(scenes, 1):
        if not scene.get("scene_label"):
            problems.append(f"scene {si}: missing scene_label")
        shots = scene.get("shots") or []
        if not shots:
            problems.append(f"scene {si}: no shots")
        for hi, shot in enumerate(shots, 1):
            where = f"scene {si} shot {hi}"
            if set(shot) != {"label", "body", "duration_seconds"}:
                problems.append(f"{where}: keys must be exactly label/body/duration_seconds")
                continue
            d = shot["duration_seconds"]
            if not isinstance(d, (int, float)) or not MIN_SECONDS <= d <= MAX_SECONDS:
                problems.append(f"{where}: duration {d} outside {MIN_SECONDS}-{MAX_SECONDS}s")
            body = shot["body"]
            if body.count("<d>") != body.count("</d>"):
                problems.append(f"{where}: unbalanced <d> tags")
            if "<d> [" in body:
                problems.append(f"{where}: no space allowed between <d> and [Language]")
            spoken_words = 0
            for _lang, line in D_TAG.findall(body):
                spoken_words += len(line.split())
                if not line.rstrip().endswith((".", "?", "!")):
                    problems.append(f"{where}: dialogue must end in . ? or !  -> {line!r}")
                if "..." in line or "…" in line:
                    problems.append(f"{where}: avoid ellipses in dialogue -> {line!r}")
            if spoken_words and isinstance(d, (int, float)):
                floor = spoken_words / SPOKEN_WORDS_PER_SECOND + DIALOGUE_MARGIN_SECONDS
                if d < floor:
                    problems.append(
                        f"{where}: {d}s is too short for {spoken_words} spoken words -- "
                        f"needs at least {spoken_words} / {SPOKEN_WORDS_PER_SECOND} + "
                        f"{DIALOGUE_MARGIN_SECONDS} = {floor:.1f}s, before any action time"
                    )
    return problems


def main() -> int:
    system = (HERE / "system_prompt.txt").read_text(encoding="utf-8").strip()
    rows, failed = [], False
    for folder in sorted(p for p in EXAMPLES.iterdir() if p.is_dir()):
        missing = [n for n in ("script.txt", "breakdown.json") if not (folder / n).exists()]
        if missing:
            print(f"[skip] {folder.name}: not finished yet (missing {', '.join(missing)})")
            continue
        script =(folder / "script.txt").read_text(encoding="utf-8").strip()
        try:
            breakdown = json.loads((folder / "breakdown.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[FAIL] {folder.name}: breakdown.json is not valid JSON: {e}")
            failed = True
            continue
        problems = check(breakdown, script)
        if problems:
            failed = True
            print(f"[FAIL] {folder.name}")
            for p in problems:
                print(f"         - {p}")
            continue
        n_shots = sum(len(s["shots"]) for s in breakdown["scenes"])
        total = sum(sh["duration_seconds"] for s in breakdown["scenes"] for sh in s["shots"])
        print(f"[ok]   {folder.name}: {len(breakdown['scenes'])} scenes, {n_shots} shots, {total:.1f}s")
        rows.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": script},
            # Compact JSON (no indentation) -- that's what we want the model to emit.
            {"role": "assistant", "content": json.dumps(breakdown, ensure_ascii=False)},
        ]})

    if failed:
        print("\nFix the failures above; train.jsonl was not written.")
        return 1
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} example(s) to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
