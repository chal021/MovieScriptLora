# MovieScriptLora

A LoRA fine-tuned LLM that reads a raw movie script and outputs a structured
scene/shot breakdown, which feeds into the existing MiniMax H3 render
pipeline in the sibling project `C:\PythonProjects\Comfyui-ClaudeBuild`.

This is a learning project as much as a building one -- the user is new to
LLM fine-tuning and wants a hands-on, fully-explained walkthrough, writing
training data by hand rather than having it generated for them. Treat this
file as the single source of truth for decisions already made; a fresh
Claude Code session in this directory has no memory of the conversation
that produced it, so don't assume anything beyond what's written here.

## The problem this solves

In the sibling project, `scripts/render/prompt_builder.py`'s `parse_script()`
turns a hand-written script (bracket-delimited shots, e.g. `[Label -- NN
seconds]`) into a flat list of `Shot` objects, which `render_shot_prompt()`
then turns into real MiniMax H3 prompts (subject refs, retention markers,
keyframe continuity, `<d>` dialogue tags, etc.) -- all of that downstream
machinery is deterministic, already built, and already correct. What's
still manual and error-prone is the step before it: reading a real script
and (a) finding true scene boundaries and (b) splitting each scene into
shots that respect MiniMax H3's native 4-15 second per-generation ceiling,
without losing or cutting off dialogue. That's the step this LoRA is for.

Relevant hard-won context from the sibling project, since it shapes what
"correct" output looks like:
- A shot's `duration_seconds` has to leave real margin for its dialogue to
  finish -- a shot clamped to the bare `MIN_SHOT_SECONDS` floor (4.0s) with
  a full spoken line in it was empirically confirmed to cut off the last
  word on a real render.
  **Update 2026-11-03:** that cutoff was probably caused, at least partly, by a
  pipeline bug since fixed in the sibling project (image `v86`): every
  shot came out ~11% short, and `-shortest` trimmed the audio to match. See
  `Comfyui-ClaudeBuild/docs/investigation_shot_duration_shortfall.md`. The
  1.5s dialogue margin was chosen before the fix. Re-measure it on fixed
  renders (silence remaining after each shot's last word) before keeping or
  shrinking it, and change it in every example at once.
- A shot's label/description text (anything outside a `<d>...</d>` tag) must
  never be phrased as something a character could plausibly say in first
  person -- a label like "Im Done Talking" got mistakenly vocalized by the
  model, in the wrong language, eating into the shot's frame budget and
  contributing to the cutoff above. Labels/descriptions must always read as
  third-person visual/action description.
- MiniMax H3's native single-generation ceiling is 4-15 seconds, no matter
  what the prompt text says -- this pipeline keeps every shot under that by
  construction (one shot = one independent native generation call, chained
  via last-frame keyframe continuity). No video-reference continuation is
  used anymore (dropped -- made the next scene redundant); continuity is
  images only.

## Scope decision (already made, don't revisit without good reason)

The LoRA does **not** generate final MiniMax prompt text. It only produces
the structured scene/shot breakdown -- the hard, genuinely-needs-language-
understanding part (scene boundaries, shot decomposition, pacing). The
deterministic rendering (subject refs, retention markers, dialogue tags,
keyframes) stays in the sibling project's existing code. This keeps the
LoRA's task narrow and learnable, and keeps everything previously verified
correct still in control of a human/deterministic code, not the model.

### Output schema

Mirrors the sibling project's existing `Shot`/`ScriptDoc` dataclasses
(`prompt_builder.py:196-207`) with one addition -- scene grouping, which
doesn't exist yet in that data model at all (today it's just a flat list of
shots). Target shape:

```json
{
  "scenes": [
    {
      "scene_label": "The Confrontation",
      "shots": [
        {"label": "", "body": "...", "duration_seconds": 6.25},
        {"label": "", "body": "...", "duration_seconds": 9.4}
      ]
    }
  ]
}
```

- `label` fields follow the "never speakable first-person text" rule above
  -- blank (`""`) is always safe and often preferred, matching the `[ ]`
  empty-bracket convention the sibling project now uses for a label-less
  shot separator.
- `body` is the shot's action + `<d>[Language] ...</d>` dialogue content,
  same shape `Shot.body` already holds.
- `duration_seconds` is the model's own pacing estimate; still needs to be
  checked against `validate_shot_durations()` (4.0-14.0s) downstream.
- Scene grouping has no effect on rendering today (ScriptDoc is a flat shot
  list with no scene concept) -- scenes get flattened into one `shots` list
  when handed to the sibling pipeline. The value of scene detection is
  giving the model a reasoning scaffold (shots within one scene share
  continuity/location/characters) that should make shot-splitting more
  accurate, and giving the human a readable breakdown to review before
  rendering.

## Base model (already decided)

**Qwen2.5-7B-Instruct** (Apache 2.0). Chosen for: 128k native context
(needs to hold a script chunk + JSON output), strong structured-output/JSON
reliability, permissive license (matches the open-source preference already
established in the sibling project -- Qwen-Image over FLUX-Kontext,
Chatterbox MIT, etc.), and modest GPU requirements for LoRA training.
**Qwen2.5-14B-Instruct** is the planned scale-up path if 7B's scene-boundary
reasoning proves insufficient -- same tokenizer/prompt format, so nothing
built around the 7B needs to be redone.

## LoRA hyperparameters (starting point, not final)

Rank 16-32 to start, given the training set will be a few hundred to
low-thousands of hand-written examples -- high enough capacity to learn the
pattern, low enough to resist just memorizing individual examples. Alpha
typically ~2x rank as a starting convention. Target modules: attention
projections (standard default, revisit if underfitting).

## Curriculum / roadmap

Working through these in order; each builds on the last.

1. **LoRA mechanics** -- DONE (conceptual). Frozen base model + small
   low-rank adapter matrices on attention projections; only the adapter
   trains. QLoRA = quantized base + LoRA, for VRAM savings, not yet decided
   if needed.
2. **Base model choice** -- DONE. Qwen2.5-7B-Instruct, see above.
3. **Schema design** -- DONE. See "Output schema" above.
4. **Building the dataset** -- NEXT. User wants to write training examples
   by hand (feeding partial scripts, writing the correct target
   breakdown), with one example built together first as a template. Need
   to settle: how many examples, how to vary scripts enough to generalize,
   exact JSONL file format for training.
5. **Tooling setup** -- `transformers` + `peft` + `trl` (SFTTrainer), on a
   RunPod GPU pod (sibling project's existing RunPod infra/skills apply
   here too).
6. **The training script** -- walk through it line by line, not just run
   it.
7. **Running the training job** -- watch the loss curve, know what healthy
   vs. overfitting looks like.
8. **Evaluation** -- held-out scripts, check output parses + respects the
   14s ceiling, compare against hand-written "gold" breakdowns.
9. **Integration** -- wire the trained LoRA in as a pre-processing step
   feeding the sibling project's existing `Shot`/`ScriptDoc` ->
   `render_shot_prompt()` path.

## Dataset layout (module 4, in progress)

- `dataset/system_prompt.txt` -- the fixed instruction given with every
  example (and later at inference time, so it must stay identical).
- `dataset/examples/NNN_name/script.txt` + `breakdown.json` -- one
  hand-written example per folder, kept as readable files.
- `dataset/build_jsonl.py` -- validates every example (durations 4-14s,
  `<d>[Language] ...</d>` format, schema keys) and packs them into
  `dataset/train.jsonl` in the chat `messages` format SFTTrainer reads.
  The assistant target is compact single-line JSON.
- `001_lighthouse` is the template example, written together.
- `dataset/action_timings.csv` -- **the master timing cheat sheet**: one
  number per action (category, action, seconds, status, notes). Status is
  `measured` once a real render confirms a value, otherwise `working`.
- `dataset/ACTION_TIMINGS.md` -- the duration formula and timing rules,
  plus a history of every action timed so far. It points to the CSV for
  the actual times and doesn't repeat them.

### Official input format (decided 2026-11-03)

Every script the LoRA sees, both each example's `script.txt` and real
scripts at inference time, comes out of **`tools/pdf_to_script.py`**:
1. `pdftotext -layout` (ships with Git for Windows). This keeps screenplay
   indentation: character names on their own line, dialogue indented. Plain
   extraction was rejected because it merges the speaker's name into the
   line ("CARMEN I don't understand..."). Copy-paste was rejected because
   it drops indentation and silently deletes characters the font maps
   wrongly.
2. Deterministic character fixes (code, not learned): soft hyphen U+00AD ->
   em dash, or "-" between letters; Greek question mark U+037E -> ";";
   U+00AB -> "'"; curly apostrophe -> "'"; ellipsis character -> "...".
   (Found in the Pan's Labyrinth PDF: 536 dashes alone. A pasted
   `Water just a bit of water` was really `Water— just a bit of water—`.)
3. Form feeds -> newlines, trailing spaces stripped, 3+ blank lines -> 1.

`--start`/`--end` cut an excerpt by text on its first and last lines.
002 and 003 were re-extracted with it. Their breakdowns didn't change.
New dialogue rule: **dashes in dialogue become commas or periods** inside
`<d>`, the same as ellipses (dashes mark pauses).
A real pipeline will need this run once on the whole script before
chunking; other PDFs may have different bad characters, so scan each new
PDF's non-ASCII characters before trusting it.

Pacing convention used in the template: roughly 2.5 spoken words/second,
plus time for the action in the shot, plus at least 1.5s of margin after the
last word. Each dialogue line is followed by a short physical action so the
character visibly stops talking.

The base model is still Qwen. Before Module 5, decide between
Qwen2.5-7B-Instruct and a newer Qwen release. The dataset format above is
the same for any Qwen chat model, so that choice doesn't block module 4.

## Planned later: character description task

User idea (2026-10-31): a third pipeline function that reads a script's
character descriptions and outputs a one-line description per character, to
feed an image generator for character consistency (the sibling pipeline
already supports character reference images).

Leaning toward: a second task in the **same** LoRA, chosen by a different
fixed system prompt (multi-task fine-tuning). The breakdown schema stays
unchanged. Each example folder later gets a `characters.json` next to
`breakdown.json`, reusing the same `script.txt`, and `build_jsonl.py` emits
one training row per task. Still to decide: should the model use only what
the script states, or fill in plausible details (period, clothing) from
context? Whichever is chosen must be applied consistently in every example.
Not started; the breakdown task comes first.

Clarified goal (2026-11-02): the model reads the character descriptions in
the script, and the pipeline uses them to generate character (and location)
reference images, so characters look the same across shots. Key design
point: a screenplay describes a character once, at first appearance, often
far from the scene being broken down (e.g. Ofelia's age, 11, is given
earlier in the script than the woods excerpt). So the character task must
read the **whole script** (it fits in Qwen's 128k context; a screenplay is
roughly 25-40k tokens), not chunks. Planned order:
1. Character task, run once on the whole script: one-line image prompt per
   character and location.
2. Generate the images (sibling client app's Character Generation tab), and
   save them as `Characters/<Name>.png` / `Locations/<Name>.png`. The file
   name becomes the subject name, so it must match the script's name exactly.
3. Breakdown task, chunk by chunk (the current work).
4. Render with shots plus reference images.
Facts in a description can come from anywhere in the script. Plausible
details (era, clothing) may be filled in only where the script supports
them. Hand-written prompts made for test renders (starting with 003's Part
A) double as first training examples for this task.
Lessons from the first image generation: framing (full-body, plain
background) is a fixed rule appended by code, not learned by the model.
Use positive wording only, because "no vehicles" put a car in the image.
Every reference image goes into every shot, so background objects leak.

## Next step

Module 4: the user writes the next examples by hand, using
`001_lighthouse` as the template, and runs `python dataset/build_jsonl.py`
after each one. Still to settle: target count and a plan for variety.
