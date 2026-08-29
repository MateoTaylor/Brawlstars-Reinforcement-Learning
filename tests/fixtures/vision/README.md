# Vision test fixtures

Recorded clips and their hand-verified ground truth. Every acceptance criterion in
`Terrain_Perception_Build_Plan.md` is a **replay against a clip**, not a live play session —
live-game iteration is too slow and too uncontrollable to debug against.

## What is tracked and what is not

Clips (`*.mp4`, `*.png`) are **gitignored** — extracted game footage, same Supercell-IP rule as
`brawl_sim/render/assets/`. The `.gitignore` entries exclude by *extension*, so everything in
this table that isn't imagery stays in version control:

| File | Tracked | What it is |
|---|---|---|
| `<clip>.mp4` | no | the recording |
| `<clip>.truth.json` | **yes** | the path you actually walked, in tiles |
| `<clip>.grid.csv` | **yes** | hand-verified terrain, in map-CSV legend — Phase I's acceptance compares its locked cells against this, reporting position and classification error separately |
| `labels/*.json` | **yes** | per-cell training labels from `scripts/vision_label.py`; one string per grid row in the map-CSV legend, so a changed cell is one character in a diff |

The ground truth is the valuable half and it is small, diffable text. The clip is reproducible
by re-recording; a hand-counted walked path is not.

**Tests that need a clip are marked `@pytest.mark.vision` and skip when it is absent**, so
`pytest` is green on a fresh clone. Run the full set with the clips present:
`pytest -m vision`.

## Ground-truth formats

### `<clip>.grid.csv` — the answer key

**A picture of the real map, typed out, one character per tile.** It is what Phase I and Phase L
are graded against: the perception stack's accumulated map is compared to it cell by cell.

```
.,.,#,#,.,.,~,~,.,.
.,.,#,#,.,.,~,~,b,b
.,.,.,.,.,.,.,.,b,b
#,#,.,.,f,f,.,.,.,.
```

`.` floor · `#` wall · `b` bush · `~` water · `f` fence · **`?` = don't know**

`?` is free — those cells are skipped on both sides of the comparison. Leave anything ambiguous
(under a loot box, under a brawler, half off the edge) as `?`. **Do not guess to fill space:** a
guessed cell is worse than a blank one, because it looks like evidence.

**The commas are optional.** The file is generated comma-separated so the extension stays honest
and it opens in a spreadsheet, but the loader also takes one character per tile (`..##..~~`),
which is far easier to type against the reference image because it lines up with it column for
column. Same legend, same file, either spelling.

It does not cover the whole map — one region the clip actually walked through, 24x18 tiles by
default. One file per clip is enough.

**Do not write one by hand from scratch.** `scripts/vision_truth.py` generates the two halves of
the job:

```
python scripts/vision_truth.py tests/fixtures/vision/<clip>.mp4     # generate
python scripts/vision_truth.py tests/fixtures/vision/<clip>.mp4 --progress
python scripts/vision_truth.py tests/fixtures/vision/<clip>.mp4 --check
```

| written | tracked | what |
|---|---|---|
| `<clip>.grid.png` | no | the map as a photograph, tile grid and coordinates drawn on |
| `<clip>.grid.csv` | **yes** | a blank grid of matching size — the file you fill in |
| `<clip>.grid.json` | **yes** | which region it covers, so scoring does not have to guess |

Every square in the PNG is one character in the CSV, and the numbers on its edges are the CSV's
own row and column positions. The PNG is a **median** composite of every frame, so brawlers,
projectiles and damage numbers are outvoted and only the terrain survives — it is regenerable, so
it is not tracked.

The generator refuses to overwrite a CSV that has work in it. `--check` runs the whole pipeline
and reports position error and classification error **separately**, plus WALL-vs-FENCE on its own.

The legend is the one `brawl_sim/maps/README.md` documents and
`brawl_sim.constants.CHAR_TO_TILE` parses for free, so the truth is diffable against a rendered
occupancy grid.

`<clip>.truth.json` records a counted path:

```json
{
  "clip": "walk_6e_4s.mp4",
  "start_tile": [0, 0],
  "legs": [ {"dir": "east", "tiles": 6}, {"dir": "south", "tiles": 4} ],
  "notes": "clean stop between legs"
}
```

Walk in straight lines with clean stops. "6 tiles east, then 4 tiles south" is unambiguous
ground truth; a meandering path you estimate afterward is not, and a fixture whose truth is
itself approximate cannot catch a small systematic error.

## Current inventory

All seven recorded on the same device at 2436x1126, screen-recorded from an iPhone. Measured with
`cv2` via `brawl_vision.clips.detect_usable_range`, and cached in the `.bounds.json` sidecars
beside each clip (tracked text, unlike the footage). See `Terrain_Perception_Build_Plan.md`
Section 3.

| Clip | Source mode | Frames | Usable | Content box | Serves |
|---|---|---|---|---|---|
| `standstill.mp4` | Training cave | 478 | 0–408 | x=[0, 2433] full width | Phase F scale check, camera/hero offset; best Phase C calibration candidate (open floor, static camera) |
| `counted_walking.mp4` | Training cave | 547 | 0–495 | x=[0, 2433] full width | Phase F primary odometry fixture |
| `zone_grows_from_east.mp4` | Solo Showdown | 1351 | 0–1219 | x=[217, 2217] pillarboxed | Phase G gas threshold; Phase B real-match HUD; Phase H second map |
| `showdown_alternate_map.mp4` | Solo Showdown | 4802 | 0–4263 | x=[217, 2217] pillarboxed | Phase G gas on a **second map** — the generalization gap; Phase H map diversity; longest clip, so the natural Phase L end-to-end fixture |
| `showdown_alternate_map2.mp4` | Solo Showdown | 2285 | 0–2071 | x=[217, 2219] pillarboxed | Phase G **specificity** fixture — contains a death but effectively no gas, so any gassed cell here is a false positive; Phase H third map |
| `showdown_has_gadget.mp4` | Solo Showdown | 869 | 0–826 | x=[217, 2217] pillarboxed | Phase B — the gadget button, the one known HUD-mask gap |
| `training_gadget.mp4` | Training cave | 599 | 0–557 | x=[0, 2435] full width | Phase B gadget button against the cave HUD, for comparison |

**All seven normalize to the same 2002x1126 viewport (exactly 16:9).** Solo Showdown constricts the visible width
on purpose, at the same zoom (~77 px per tile on screen in both modes), so the crop makes every
clip geometrically comparable — which is what lets one homography serve every mode.

The crop is centered on the detected **viewport**, and getting the viewport right took two
corrections worth knowing about:

- **The content boundary is not a step from black to game.** There is a ~10 px band of dim HUD
  glow outside the world viewport (the "Brawlers left: N" readout is laid out to the full screen).
  A detector asking "is any pixel in this column lit?" latches onto the glow. Asking "is *most* of
  this column lit?" lands on the real edge: measured, columns 206–216 have a column-mean of 6–10
  with 2–23% of pixels lit, then column 218 jumps to a mean of 122 with 99.9% lit.
- **Aggregate across frames with the median, not the union.** The viewport is a fixed rectangle,
  so per-frame bounds are a constant corrupted by noise in *both* directions — a dark frame
  reports the edge too far in, glow reports it too far out. A union only defends against one.

With both corrections the recovered Showdown viewport is x=[217, 2217] — 2002×1126, **exactly
16:9, exactly centered** in the 2436 px frame. That self-consistency is the check that the
detector is measuring the right thing; the earlier max-based estimate gave a suspicious 1.7815.

### Known quirks, all handled in `clips.py`

- **Contaminated tails.** Every clip ends with the iOS Control Center swiping in. Detected as a
  sustained spike in frame-to-frame difference, searched only in the last 15% of the clip so a
  violent moment of mid-clip gameplay cannot be mistaken for the end.
- **The tail also breaks content-box detection**, which is why the box is unioned only over frames
  inside the usable range. The overlay spans the full capture width, so including it reported the
  Showdown clip as full-width and skipped normalization entirely.
- **Variable frame rate.** Nominal 56.95 / 58.02 / 59.42 fps, but these are screen recordings that
  genuinely drop frames — `standstill.mp4` carries a 166.67 ms gap (10 frames at 60 Hz) mid-clip.
  Timestamps come from `CAP_PROP_POS_MSEC`, not from `index / fps`.
- **Rotation metadata.** Stored portrait (1126x2436) with a rotate flag, displayed landscape.
  `cv2` and `imageio-ffmpeg` agree on these files, but that is version-dependent — a source
  decoding portrait raises rather than feeding later stages a sideways world.
- **Seeking is not the same as counting, within `cv2` itself.** `cap.set(CAP_PROP_POS_FRAMES, n)`
  lands on a keyframe and decodes forward, so it returns a frame NEAR `n`, not the frame
  `ClipReader` yields at count `n` — measured on `showdown_alternate_map2`, mean absolute
  difference of 3–25 grey levels at the same index. Seeking is the right tool when any frame from
  a region will do (it turns a 4200-frame decode into 0.3 s); it is the wrong tool for anything
  indexed against a ground-truth annotation.
- **`showdown_alternate_map2` ends with a death, and the trim detector handled it correctly.**
  Usable ends at 2071; the results screen starts at sequential frame ~2090 and the iPhone Control
  Center at ~2280. Both are outside the usable range, which is the tail detector doing its job on
  content it was never shown.
- **Reader choice matters for frame indices.** `cv2` and `imageio_ffmpeg.count_frames_and_secs`
  agree (478 / 547 / 1351); imageio's streaming `read_frames` over-yields (503 / 565 / 1364). Any
  frame-indexed annotation is only meaningful next to the reader that produced it, which is why
  this package standardized on `cv2`.
- **`counted_walking` has only ONE leg of camera motion, and no `.truth.json` was ever written.**
  Phase F measured it: across the stretch that looks like the east leg the background is
  *pixel-for-pixel identical* — the player walked, the camera did not follow. The training cave
  clamps the camera at the map edge. Only the south leg moves the camera, by 3.52 tiles.

  This is not a recording flaw, and it is why the odometry regression test pins the measurement's
  stability rather than its agreement with a walk count: **a walk count is ground truth for the
  hero, and odometry measures the camera.** Where the camera clamps or leads, no tolerance
  reconciles the two. See the plan's Phase F and Phase K.

**`showdown_alternate_map2.mp4`'s content box is 2 px wider** (x=[217, 2219]) than the other two
Showdown clips. It still normalizes to exactly 2002x1126, because `capture.normalize_viewport`
allows `_WIDTH_SLACK_PX = 4` of measurement wobble before it refuses. That slack is doing real
work, not papering over a bug: box detection is per-frame thresholding on a compressed recording,
and demanding an exact pixel match would reject a perfectly good clip.

## Still missing

| Gap | Blocks | Why it matters |
|---|---|---|
| Gas footage on a **fourth** map | Phase G regression | Phase G's threshold is now measured across three maps and passes its acceptance, including the hard case where bushes share the gas hue. A fourth map is no longer a gap, just the next regression fixture. |
| **Hand-labelled cells** (`labels/*.json`) — ~5 frames per map | Phase H, and it is the ONLY thing blocking it | Every piece of Phase H machinery is built and tested; nothing measures accuracy because nothing is labelled. Breadth beats depth: 20 cells across 6 maps beats 100 across 2. Hunt WATER and FENCE deliberately — FLOOR and WALL over-supply themselves. See `labels/README.md`. |
| A clip where gas covers **most of the frame** | Phase G edge case | Validated up to ~7% coverage; late-game, gas fills the screen and every cell is gassed, which is the regime where a threshold that is slightly too generous stops mattering and one slightly too tight fails completely. |
| A counted walk **in open ground**, with its `.truth.json` | Phase F absolute scale | The one existing counted clip cannot serve: the camera clamps for one leg and no truth file exists. Walk in the middle of a Showdown map, well clear of every edge, in one straight line with a clean stop, and record the tile count. This is the only outstanding check on odometry's absolute scale. |
| A clip containing a **camera cut** — a match-start fly-in, or any mode with respawns | Phase F cut handling | Partly answered: `showdown_alternate_map2` contains a death, and Solo Showdown transitions to a spectator view with NO camera discontinuity, so death is not the case. Cut detection still rests on one real discontinuity (`training_gadget` f508). |

Record at the resolution and aspect ratio you intend to play at, and prefer Solo Showdown over the
training cave: the HUD differs (training cave shows "Edit Controls" / "Exit" / "Damage per
second") and the view is wider, so training-cave footage cannot stand in for Phase B or Phase K.

Worth noting alongside each clip: which map it is from, and the game version. When the classifier
eventually regresses after an update, that log is what distinguishes a reskin from a UI change.
