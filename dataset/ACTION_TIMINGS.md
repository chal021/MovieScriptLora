# Timing rules

**The action times live in `action_timings.csv` (this folder). That file is
the master.** Look an action up there before timing it. If an action isn't
listed, pick the closest row. If none fits, add a new row to the CSV and use
it from then on. Don't copy the times into this file. Two copies drift apart.

When a render confirms a time, change that row's `status` from `working`
to `measured` and note the source in `notes`.

## The formula

```
duration = speech  +  action times  +  1.5s margin (if any dialogue)
speech   = words / 2.5  +  0.25 per comma  +  0.5 per mid-line sentence end
```

- **Speech:** count only what's inside `<d>` tags: the words (divide by 2.5),
  every comma (0.25s each, all counted the same), and every `.` `?` `!` that
  has more speech after it (0.5s each). The line's last mark doesn't count,
  because the margin covers it.
- **Syllables are a candidate** (syllables / 3.5 instead of words / 2.5).
  Words stay the rule until fixed renders show which predicts real speech
  length better.
- **Margin:** 1.5s after the last line. A shot with no dialogue gets none.
  (Chosen before the sibling pipeline's duration fix, image `v87`.
  Re-check it against fixed renders.)
- **Sequential** actions ("then", or one after another): add their times.
- **Simultaneous** actions ("at the same time", "as", "while"): count only
  the longest one. The wording and the arithmetic must agree.
- Total must be **4.0-14.0s**. Under 4.0s: add described action, never padding.

## Rules for timing

- **One number per action, not a range.** A range lets the same action get
  different times in different examples, and the model learns that
  inconsistency.
- **Itemize first, then add up.** List each action with its own time.
  Never estimate a whole shot as one lump number.
- **Look at the previous shot's last frame before timing a walk.** Distance
  on screen becomes walking time.
- **The CSV wins.** Where an example disagrees with it, re-time the example.

---

## History: every action timed so far

For reference only. The "CSV value" column shows what each action should be
re-timed to. Rows in **bold** don't match the CSV yet.

| Action | Time used | Where | CSV value |
|---|---|---|---|
| Establish place + cars stopping | 3.5s | 003 sh1 | 3.5 ✓ |
| Climbs out of a car (hurried) | 2.0s | 003 sh1 | 2.0 ✓ |
| Moves to front of car (a few steps) | 2.0s | 003 sh1 | **3.5** |
| Bends over, braced, breathing hard | 3.0s | 003 sh2 | 3.0 ✓ |
| Approaches slowly | 3.0s | 003 sh2 | **3.5** |
| Raises a hand | 1.0s | 003 sh3 | 1.0 ✓ |
| Grabs abdomen | 1.0s | 003 sh3 | 1.0 ✓ |
| Nods and moves away | 3.0s | 003 sh4 | **4.5** (nod 1.0 + walk 3.5) |
| Someone walks over at the same time | 0s | 003 sh4 | 0 ✓ (overlaps) |
| Reaches someone (walk-up) | 1.0s (measured ~3.5s) | 003 sh5 | **3.5** ✅ |
| Touches a shoulder | 1.0s | 003 sh5 | 1.0 ✓ |
| Camera moves to a face during the line | 0s | 003 sh6 | 0 ✓ |
| Turns to call out / turns back | 1.0s each | 003 sh6 | 1.0 ✓ |
| Takes a book from someone's lap | 2.0s | 002 sh3 | 2.0 ✓ |
| Glance + nausea + hand to mouth | 2.0s total | 002 sh4 | **3.0** (1.0 x 3) |
| Turns and knocks on a window | 2.0s | 002 sh5 | **3.0** (turn 1.0 + knock 2.0) |
| Camera pull-back + character wakes | 4.0s (lump) | 002 sh2 | **3.5** (2.0 + 1.5) |
| Window, look down at book, camera moves in | 6.0s (lump) | 002 sh1 | needs itemizing |
| 001 (all shots) | lump / derived | 001 | needs itemizing |

**To do before training:** re-time 001, 002 and 003 against the CSV.
