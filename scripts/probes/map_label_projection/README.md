# Map-label projection: Starfruit Supernova, 2026-11-04

These scripts produced the Starfruit Supernova classifier labels. They take the map label
(`brawl_deployment/data/maps/starfruit_supernova.{png,csv,json}`) and one BlueStacks recording
(`bluestacks-starfruit-supernova-west-upper`), and write:

- 10 training `LabelGrid`s in `tests/fixtures/vision/labels/`;
- 7 held-out `LabelGrid`s in `tests/fixtures/vision/labels_starfruit_test/`.

No hand-labelling was done. Every label is the map's own class, projected onto the frame through
a registration.

Truth (2026-11-05) went through `scripts/map_project_labels.py` directly. Its frame picking,
scoring and localizer checks are in `truth/`, with their own README.

**For a new map, use `scripts/map_project_labels.py`.** It runs this method on any map and
BlueStacks clip, with no code edits:

- it fits S for each clip, where these scripts took it as a constant;
- it registers every odometry segment with enough map matches, where `register2.py` started at
  frame 430;
- it takes the cells to leave `?` on its command line (`--leave`, `--leave-row`, `--leave-why`),
  where `project.py` listed the meeple cells in code.

Its docstring has the usage. `brawl_vision/terrain/label_projection.py` holds the parts that need
no video, and `tests/test_label_projection.py` pins them on synthetic data. The scripts here stay
as the record of the first run: the fixture READMEs link to this file, and the numbers below are
theirs.

## Rebuilding the committed labels with the tool

Run from the repo root on the CPU. The recording has to be in a `CLIP_DIRS` folder, because
`brawl_vision/data/training_videos/` is gitignored. The labels go to a scratch folder, because the
Starfruit fixtures aren't tracked and a bad run must not overwrite them.

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/map_project_labels.py starfruit_supernova \
    bluestacks-starfruit-supernova-west-upper \
    --train 460 550 640 690 820 910 1000 1090 1180 1340 \
    --held-out 1840 1900 1960 2020 2080 2140 2200 \
    --train-dir runs/map_checks/starfruit_supernova/repro/labels \
    --held-out-dir runs/map_checks/starfruit_supernova/repro/labels_starfruit_test \
    --leave-row 26:0-3,8-14,18-21,25-31,35-38,43-44,50-52 33:3-7,13-18,22-24,28-35,39-45,51-54,58-59 \
    --leave-why "meeple cells (map rows 26/33, wall-or-fence unsettled)" --date 2026-11-04
```

Then compare, in Git Bash. It prints nothing when every file matches:

```bash
for d in labels labels_starfruit_test; do
  for f in runs/map_checks/starfruit_supernova/repro/$d/*.json; do
    cmp -s "$f" "tests/fixtures/vision/$d/$(basename "$f")" || echo "DIFFERS: $f"
  done
done
```

**Checked on 2026-11-04.** All 17 JSONs came out byte-identical to the committed ones, twice:

- from scratch, with no cached registration: 2 min 13 s;
- again on the cached registration, which reruns only the labels.

The tool fitted S = (1.022, 1.004) from the same 143 samples `scale_fit.py` used, and registered
the same 180 samples as `register2.py`. It printed training 955 map tiles, held-out 502, shared 0,
as `tests/fixtures/vision/labels_starfruit_test/README.md` says.

**What only this rebuild checks.** The fast test pins the rules on synthetic data. It can't pin
the numbers that come from the footage and the models: the odometry, the detectors, the gas
detector and the interpolation kernels. Rerun the rebuild after changing any of them.

A retrained detector can legitimately move a few cells. Then the committed labels stay the record
of 11-04, and regenerating them is a separate decision. The terrain classifier isn't used, so
swapping `terrain.pt` changes nothing here.

## The probes, step by step

Run from the repo root on the CPU (`CUDA_VISIBLE_DEVICES=-1`). Names and paths are Starfruit's
and live in `common.py`. Intermediate files go to `runs/map_checks/starfruit_supernova/projection/`,
which is gitignored and keeps `reg.npz` and `reg2_1.022_1.004.npz`. With those two files, steps 4
and 5 rerun without steps 1–3.

| # | command | writes | time | in the tool |
|---|---|---|---|---|
| 1 | `register.py [step=10]` | `reg.npz`: odometry on every frame, plus scale-1 masked NCC of every 10th rectified frame against the map image | 79 s for 2221 frames | stage 1 |
| 2 | `scale_fit.py` | prints S per axis, the slope of NCC position against odometry | instant | `fit_scale`, at the start of stage 2 |
| 2b | `scale_check.py MAP CLIP FRAME...` | per-frame uniform-scale NCC sweep from 0.96 to 1.05 (a cross-check) | seconds | not ported |
| 3 | `register2.py 1.022 1.004 10` | `reg2_1.022_1.004.npz`: re-registration at S, with 3x3 block residuals | about 1 min | stage 2 |
| 4 | `make_labels.py OUTDIR FRAME...` | `LabelGrid` JSONs, `lab/aux_<f>.npz`, `lab/rect_<f>.png` | about 3 s per frame | stage 3 |
| 5 | `review.py LABELDIR OUTDIR aux` | `review_f<f>.jpg`: labelled cells tinted with their letter, `?` dimmed, diagonals on refused cells | seconds | stage 3, plus a sheet per set |

S has to be rounded to three decimals, as `register2.py`'s command line has it. The template
height is `round(912 / S_y)`, which is 908 at 1.004 but 909 at the unrounded 1.00381. That one
pixel moves every label.

Checked on 2026-11-04: running `make_labels.py` from this folder rebuilds frames 640 and 1960
byte-identical to the committed JSONs. `scale_fit.py` reproduces the S below.

## Method

**Registration.** Call A the world tile under rectified pixel (0,0) and S the number of
rectified pixels per true tile, divided by 48.

1. Odometry gives a smooth track, but its scale is the phone camera model's.
2. Masked NCC of the rectified frame against the map image gives an absolute position at any
   chosen scale.
3. Regressing the two over the odometry segment with the most good matches (segment 7: 143
   samples scoring above 0.4) gives:
   - x slope 0.9784, so **S_x = 1.0221** (residual sd 0.157 tile);
   - y slope 0.9962, so **S_y = 1.0038** (residual sd 0.117 tile).
4. `register2.py` re-registers at that S.
5. `make_labels.py` uses `A(f) = pos(f) / S + offset`. The offset is a running median, over ±3
   samples, of `A_ncc - pos / S`. One bad NCC peak therefore can't move a label. The offset
   drifts smoothly over the clip: x from 25.41 to 25.96, y from 18.98 to 19.39.

**The emulator rectifies about 2.2% too wide in x and 0.4% in y.** The camera model was
calibrated on the phone, and BlueStacks 16:9 captures don't match it exactly.

- A global NCC search agrees: best (1.02, 1.015) at 0.610, against 0.590 at (1, 1). The score is
  flat over 1.02–1.025 by 1.01–1.025.
- The 1080p `bluestacks-dark-passage-top-right` clip also sits at 1.01–1.03.
- Effect: live odometry over-reads x motion by about 2%.

The labels don't depend on this, because they use the measured S. Recalibrating the emulator
camera is a separate piece of work.

**The label rule.** A cell takes the majority class of the world area under it.

- The area used is the ground footprint. A wall's top face appears about 0.9 tile north, in the
  floor cell above, and that cell stays floor.
- A class is kept only if it holds at least 0.60 of the cell under all 9 shifts of ±0.12 tile,
  which is the registration error.
- Only `. # b ~ f` can be labelled.
- The meeple cells (66 of them, map rows 26 and 33) never label. They're `#` in the map CSV, but
  whether they're walls or fences is unsettled. `project.py` lists them as `MEEPLE_RUNS`, and the
  tool takes the same cells as the `--leave-row` arguments above.

**Cells left `?`.** These follow the refusals in `tests/fixtures/vision/labels/README.md`:

- HUD and unobserved cells;
- gas (`detect_zone` at 5% coverage, grown by one cell);
- cells under a detected brawler, crate, cube or projectile (detector boxes, mapped into the
  rectified frame plus a small margin; the boxes already include the name tag, HP bar and ammo
  bar);
- the brawlers-left text and the kill card, which on these 1440p captures run past the emulator
  mask (`TEXT_BOX`, `KILL_BOX`);
- cells that change against the frames 8 either side, shifted by the odometry delta (splashes,
  damage numbers, shells in flight).

A full-screen super flash fails most of the view. Frame 730 had one (at 722), so frame 690 is
labelled instead.

**Floor art.** The map image is a schematic and leaves out the floor art: the pink X tiles,
yellow "?" tiles, the orange "¿" tile, the white dots and the puzzle pattern. These are flat and
walkable, so they label as floor. They're also exactly what a classifier trained on other maps
has never seen.
