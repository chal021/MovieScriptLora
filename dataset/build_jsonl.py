"""Validate the hand-written targets and pack them into train.jsonl.

Each whole screenplay lives in its own folder under dataset/scripts/:
    script.txt          the whole script, from tools/pdf_to_script.py
    scenes.json         Task 2 target: the scene list, in script order
    characters.json     Task 3 target: the roster (characters + locations)
    shots/Scene01.json  Task 1 target for scene 1 (Scene02.json for scene 2...)

You write only the targets. This script builds every model input itself:
- Task 1 (shots): a "Characters:" / "Locations:" header from scenes.json,
  then that scene's text, cut out of script.txt.
- Task 2 (scenes): the whole script.txt. Only built once scenes.json says
  "complete": true. A partial scene list would teach the model to stop early.
- Task 3 (characters): not built yet.

Each row is written in the chat "messages" format TRL's SFTTrainer reads:
    {"messages": [{"role": "system", ...}, {"role": "user", ...},
                  {"role": "assistant", ...}]}

Usage:  python dataset/build_jsonl.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = HERE / "scripts"
PROMPTS = HERE / "prompts"
OUT = HERE / "train.jsonl"

MIN_SECONDS, MAX_SECONDS = 4.0, 14.0
D_TAG = re.compile(r"<d>\[([^\]]+)\] (.+?)</d>", re.DOTALL)
HEADING = re.compile(r"^\s*(INT\.|EXT\.|INT/EXT|EXT/INT)")
PRONOUNS = {"he", "she", "they", "it", "his", "her", "their", "its", "him", "them"}

# Pacing convention (see ACTION_TIMINGS.md; values from action_timings.csv):
# speech = words / 2.5 + 0.25 per comma + 0.5 per mid-line sentence end,
# plus action time, plus 1.5s of margin after the last line. Code can't count
# action time, so it checks only a floor: speech + margin. Below that, the
# dialogue would be cut off.
SPOKEN_WORDS_PER_SECOND = 2.5
COMMA_PAUSE_SECONDS = 0.25
SENTENCE_PAUSE_SECONDS = 0.5
DIALOGUE_MARGIN_SECONDS = 1.5
MID_SENTENCE_END = re.compile(r"[.?!](?=\s+\S)")  # a . ? ! with more speech after it

SCENE_KEYS = {"label", "starts_with", "characters", "locations"}
SHOT_KEYS = {"label", "location", "body", "duration_seconds"}


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def find_scene_starts(lines: list[str], scenes: list[dict]) -> tuple[list[int], list[str]]:
    """Line index where each scene starts. Each scene's starts_with is searched
    for after the previous scene's start, so repeated headings
    (INT. UPPER BEDROOM — NIGHT appears twice) resolve in script order."""
    starts, problems, after = [], [], -1
    for n, scene in enumerate(scenes, 1):
        needle = squash(scene.get("starts_with", ""))
        hit = next((i for i in range(after + 1, len(lines)) if needle and needle in squash(lines[i])), None)
        if hit is None:
            problems.append(f"Scene{n:02}: starts_with not found after Scene{n - 1:02}: {scene.get('starts_with')!r}")
            starts.append(-1)
        else:
            starts.append(hit)
            after = hit
    return starts, problems


def scene_text(lines: list[str], starts: list[int], n: int) -> tuple[str, str]:
    """Text of scene n (0-based): from its start up to the next listed scene.
    If it's the last listed scene, stop at the next scene heading instead."""
    start = starts[n]
    if n + 1 < len(starts):
        end, how = starts[n + 1], "next listed scene"
    else:
        end = next((i for i in range(start + 1, len(lines)) if HEADING.match(lines[i])), len(lines))
        how = "next heading (last listed scene)"
    return "\n".join(lines[start:end]).strip("\n"), how


def check_scene_entry(n: int, scene: dict, roster: dict) -> list[str]:
    where = f"Scene{n:02}"
    if not isinstance(scene, dict):
        return [f"{where}: each scene must be an object {{\"label\": ..., \"starts_with\": ..., "
                f"\"characters\": [...], \"locations\": [...]}}, not {type(scene).__name__} {scene!r}"]
    if set(scene) != SCENE_KEYS:
        return [f"{where}: keys must be exactly {sorted(SCENE_KEYS)}"]
    problems = []
    if not scene["label"]:
        problems.append(f"{where}: empty label")
    if not scene["locations"]:
        problems.append(f"{where}: needs at least one location")
    for kind, names in (("characters", scene["characters"]), ("locations", scene["locations"])):
        # Roster entries may be plain names ("Ofelia") for now, or objects
        # with a "name" field once the Task 3 format is settled.
        known = {e if isinstance(e, str) else e.get("name") for e in roster.get(kind, [])}
        if known:  # only checked once the roster has entries
            for name in names:
                if name not in known:
                    problems.append(f"{where}: {kind[:-1]} {name!r} is not in characters.json")
    return problems


def check_shots(n: int, target: dict, scene: dict) -> list[str]:
    where_scene = f"Scene{n:02}"
    shots = target.get("shots")
    if set(target) != {"shots"} or not isinstance(shots, list) or not shots:
        return [f"{where_scene}: shots file must be {{\"shots\": [...]}} with at least one shot"]
    problems = []
    for hi, shot in enumerate(shots, 1):
        where = f"{where_scene} Shot{hi:02}"
        if set(shot) != SHOT_KEYS:
            problems.append(f"{where}: keys must be exactly {sorted(SHOT_KEYS)}")
            continue
        if shot["location"] not in scene["locations"]:
            problems.append(f"{where}: location {shot['location']!r} is not in this scene's locations {scene['locations']}")
        d = shot["duration_seconds"]
        if not isinstance(d, (int, float)) or not MIN_SECONDS <= d <= MAX_SECONDS:
            problems.append(f"{where}: duration {d} outside {MIN_SECONDS}-{MAX_SECONDS}s")
        body = shot["body"]
        first = re.sub(r"[^A-Za-z]", "", body.split()[0]).lower() if body.split() else ""
        if first in PRONOUNS:
            problems.append(f"{where}: starts with a pronoun ({body.split()[0]!r}). Use the name.")
        if body.count("<d>") != body.count("</d>"):
            problems.append(f"{where}: unbalanced <d> tags")
        if "<d> [" in body:
            problems.append(f"{where}: no space allowed between <d> and [Language]")
        spoken_words, speech = 0, 0.0
        for _lang, line in D_TAG.findall(body):
            spoken_words += len(line.split())
            speech += (len(line.split()) / SPOKEN_WORDS_PER_SECOND
                       + line.count(",") * COMMA_PAUSE_SECONDS
                       + len(MID_SENTENCE_END.findall(line)) * SENTENCE_PAUSE_SECONDS)
            if not line.rstrip().endswith((".", "?", "!")):
                problems.append(f"{where}: dialogue must end in . ? or !  -> {line!r}")
            if "..." in line or "…" in line or "—" in line:
                problems.append(f"{where}: no ellipses or dashes in dialogue -> {line!r}")
        if spoken_words and isinstance(d, (int, float)):
            floor = speech + DIALOGUE_MARGIN_SECONDS
            if d < floor - 1e-9:
                problems.append(
                    f"{where}: {d}s is too short for its dialogue -- speech {speech:.2f}s "
                    f"({spoken_words} words + pauses) + margin {DIALOGUE_MARGIN_SECONDS} = "
                    f"{floor:.2f}s, before any action time"
                )
    return problems


def unnamed_characters(target: dict, scene: dict) -> list[str]:
    """Characters in the scene header that no shot ever names (a warning:
    maybe they really don't appear, maybe a name is missing)."""
    text = " ".join(s.get("body", "") for s in target.get("shots", []))
    return [c for c in scene["characters"]
            if not any(re.search(rf"\b{re.escape(t)}\b", text) for t in c.split())]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_script(folder: Path, prompts: dict) -> tuple[list[dict], bool]:
    rows, failed = [], False
    name = folder.name
    lines = (folder / "script.txt").read_text(encoding="utf-8").split("\n")
    try:
        scenes_doc = load_json(folder / "scenes.json")
        roster = load_json(folder / "characters.json") if (folder / "characters.json").exists() else {}
    except json.JSONDecodeError as e:
        print(f"[FAIL] {name}: invalid JSON: {e}")
        return [], True
    scenes = scenes_doc.get("scenes", [])

    problems = [p for n, s in enumerate(scenes, 1) for p in check_scene_entry(n, s, roster)]
    if problems:
        print(f"[FAIL] {name}/scenes.json")
        for p in problems:
            print(f"         - {p}")
        return [], True
    starts, problems = find_scene_starts(lines, scenes)
    if problems:
        print(f"[FAIL] {name}/scenes.json")
        for p in problems:
            print(f"         - {p}")
        return [], True
    print(f"[ok]   {name}: {len(scenes)} scene(s) listed"
          + ("" if scenes_doc.get("complete") else " (scene list not complete yet)"))

    # Task 1: one row per scene that has a shots file.
    for n, scene in enumerate(scenes, 1):
        shots_path = folder / "shots" / f"Scene{n:02}.json"
        if not shots_path.exists():
            print(f"[skip]   Scene{n:02} {scene['label']!r}: no shots/Scene{n:02}.json yet")
            continue
        try:
            target = load_json(shots_path)
        except json.JSONDecodeError as e:
            print(f"[FAIL]   Scene{n:02}: invalid JSON: {e}")
            failed = True
            continue
        problems = check_shots(n, target, scene)
        if problems:
            failed = True
            print(f"[FAIL]   Scene{n:02} {scene['label']!r}")
            for p in problems:
                print(f"           - {p}")
            continue
        text, how = scene_text(lines, starts, n - 1)
        header = (f"Characters: {', '.join(scene['characters']) or 'none'}\n"
                  f"Locations: {', '.join(scene['locations'])}")
        total = sum(s["duration_seconds"] for s in target["shots"])
        print(f"[ok]     Scene{n:02} {scene['label']!r}: {len(target['shots'])} shots, {total:.1f}s"
              f" (text cut to {how}, {len(text.splitlines())} lines)")
        for c in unnamed_characters(target, scene):
            print(f"[warn]     {c} is in the header but no shot names them")
        rows.append({"messages": [
            {"role": "system", "content": prompts["shots"]},
            {"role": "user", "content": f"{header}\n\n{text}"},
            # Compact JSON (no indentation): that's what we want the model to emit.
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]})

    # Task 2: the whole scene list, only once it covers the whole script.
    if scenes_doc.get("complete"):
        if "scenes" not in prompts:
            print("[FAIL]   scenes.json is complete but dataset/prompts/scenes.txt doesn't exist yet")
            failed = True
        else:
            rows.append({"messages": [
                {"role": "system", "content": prompts["scenes"]},
                {"role": "user", "content": "\n".join(lines).strip("\n")},
                {"role": "assistant", "content": json.dumps({"scenes": scenes}, ensure_ascii=False)},
            ]})
            print(f"[ok]     Task 2 row: whole scene list ({len(scenes)} scenes)")
    return rows, failed


def main() -> int:
    prompts = {p.stem: p.read_text(encoding="utf-8").strip() for p in PROMPTS.glob("*.txt")}
    rows, failed = [], False
    for folder in sorted(p for p in SCRIPTS.iterdir() if p.is_dir()):
        missing = [n for n in ("script.txt", "scenes.json") if not (folder / n).exists()]
        if missing:
            print(f"[skip] {folder.name}: missing {', '.join(missing)}")
            continue
        script_rows, script_failed = build_script(folder, prompts)
        rows += script_rows
        failed |= script_failed

    if failed:
        print("\nFix the failures above; train.jsonl was not written.")
        return 1
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} row(s) to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
