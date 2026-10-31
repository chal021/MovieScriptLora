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

## Next step

Module 4: building the training dataset. Start by writing one real example
together end to end (a partial script -> its correct scene/shot JSON
breakdown) as a template, before the user writes the rest by hand.
