# Style guide for writing targets

The rules every hand-written target follows. **Consistency matters more
than quantity:** the model is unreliable exactly where examples disagree.
If a rule changes, change it in every example at once.

Timing numbers live in `action_timings.csv` (the master), and the formula and
timing rules in `ACTION_TIMINGS.md`. The older worksheets in `archive/` hold
the worked history that produced these rules.

---

## Task 2: scenes (`scenes.json`)

- **Scenes follow the script's own scene headings** (`INT.` / `EXT.`).
- **A location change with no new heading is not a new scene.** It's a
  camera move to a new background inside the same scene. A scene can have
  several locations.
- Each entry: `label` (short, third-person, visual), `starts_with` (text on
  the scene's first line, usually the heading itself), `characters` and
  `locations` (exact roster names, in the order they first appear).
- `starts_with` must be unique enough to find. The builder searches for it
  after the previous scene's start, so a repeated heading is fine.
- Set `"complete": true` only once the list covers the whole script. A
  partial list would teach the model to stop early.

## Task 1: shots (`shots/SceneNN.json`)

### Shots
- One shot = one distinct, describable beat (usually one line plus its
  reaction). Cover the whole scene, in order.
- **Continuity check:** read the last sentence of each shot, then the first
  sentence of the next. They should join up, because the last frame of shot
  N is the first frame of shot N+1.
- Don't describe the same action twice across a shot boundary.
- Within a shot, describe camera **movement** ("the camera moves in close
  on..."), never a "cut". Cuts happen between shots.
- **A shot must never contradict the frame it starts from.** If the next
  speaker has their back to the camera at the end of shot N, shot N+1 opens
  with a camera move around to their face. Otherwise MiniMax forces a cut.
- Don't skip story beats, including actions as well as dialogue.
- **Account for every entrance:** where each character comes from (which
  car, which door).

### Locations
- `location` is exactly one name from the scene's `locations`: the place
  the shot is in. Code attaches that location's image and no other.
- **A location change happens on screen, inside a shot.** End the shot on a
  frame that works in both places (black, white, fog, an extreme close-up),
  and give the next shot the new location. Never jump to a new place at a
  shot boundary.

### Characters and names
- **Name every visible character in every shot**, using the exact roster
  name. The pipeline attaches a character's reference image only to shots
  that name them, so an unnamed visible character may morph or vanish.
- **Never start a shot with a pronoun.** Each shot is a separate generation.
  Pronouns are fine later in the same shot.
- **Watch for names inside dialogue.** A name spoken in a line also counts
  as a mention, and pulls that character's image into the shot.
- **Deliberately unnamed figures** (a faceless silhouette, a figure whose
  identity the script hides) stay out of the roster and are described
  visually, never by name. Keep them out of shots that name a character
  they could be mistaken for, or MiniMax may borrow that character's face.
- Describe a character's appearance **once per scene**, where they first
  appear in it, name first ("Mara, a weathered woman in her sixties, ...").
  Then use only the name.
- Pronouns must match the character. A wrong "he" can put the action on the
  wrong person, or invent someone.

### Body text (everything outside `<d>`)
- Third-person, visible description only. If a camera can't show it,
  rewrite it so it can, or drop it.
  - "napping, then speaks": rests with eyes closed, then opens them and says
  - "pauses": show the cause (her face tightens with nausea)
  - writer's commentary ("not very dainty"): use visuals (plain, frayed clothes)
  - an exact age: "a young woman in her late twenties"
- `label` is always `""`.
- **Proofread.** Typos in a target get learned.

### Dialogue (`<d>` tags)
- Format: `<d>[English] text.</d>`, with no space between `<d>` and `[`, and
  one space after `]`.
- Copy the script's words and capitalization exactly. Never drop or shorten
  a line.
- **No ellipses or dashes.** `...` and `—` mark pauses. Write them as commas
  or periods.
- No doubled punctuation: `!!` becomes `!`, and `??` becomes `?`.
- Every line ends in `.` `?` or `!` right before `</d>`.
- **Punctuation is acting direction.** `!` means loud or urgent. Don't add
  emotion the script didn't ask for, but `!` is right when staging requires
  it (calling to someone far away).
- Every line is followed by a **trailing action**, so the character visibly
  stops talking. Ideally that action also sets up the next shot.
- **Two speakers can share a shot** when their lines are one quick exchange
  and the total fits under 14s with room to spare.

### Durations
```
speech    = words / 2.5  +  0.25 per comma  +  0.5 per mid-line sentence end
duration  = speech  +  action times  +  1.5s margin (if any dialogue)
```
- Count only what's inside `<d>`. Every comma counts the same. A line's last
  `.` `?` `!` doesn't count (the margin covers it).
- **Itemize before adding up.** One time per action, from the CSV. Never a
  lump estimate.
- **Sequential** actions add. **Simultaneous** ones ("as", "while", "at the
  same time") count only the longest. The wording and the arithmetic must agree.
- Must land between 4.0 and 14.0 seconds.
- **Under 4.0s: add described action, never padding.** Padded seconds the
  prompt says nothing about get filled with improvised talking.
- The margin stops the line being **cut off**. The trailing action stops the
  character **talking on**.
- `build_jsonl.py` flags any dialogue shot shorter than its speech time + 1.5.
