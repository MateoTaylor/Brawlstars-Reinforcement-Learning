# Odometry overlay rule, 2026-11-05

These probes measured the odometry's overlay rule: `brawl_vision/terrain/odometry.py`, with
`odometry.overlay_share` in `configs/vision.yaml`. Their numbers are in the odometry docstring, in
the config comments, and in MAP_ONBOARDING.md's Truth section.

**The problem.** Screen-static content covers some of the odometry's windows and reads zero while
the camera moves: the SHOWDOWN banner, the hero and its name tag, a recording's mouse cursor. On
Truth's banner, 5 of 8 windows did for 3 s. The odometry stood still and fell 7.5 tiles behind
the camera, and the localizer dropped its fix.

| script | does |
|---|---|
| `dump_shifts.py CLIP --hud H [--rate HZ] --out F.npz` | For each frame pair, writes every window's own phase-correlation shift and response, unfiltered |
| `vote.py DUMP.npz CLIP --hud H` | Hands each pair's windows to the shipped `Odometry._split`. Each pair it puts in question goes through the shipped `update` twice, rule on and rule off, and the output shows the vote, the status and the camera motion each way. |
| `replay_overlay.py {off,match} CLIP ...` | Runs `scripts/probes/replay_clip.py` with the rule off (the estimator before 2026-11-05), or on only while the loop is in a match |

`vote.py` and `replay_overlay.py` call the shipped code and re-implement nothing. A change to the
rule shows up in their output unedited.

Run from the repo root on the CPU (`CUDA_VISIBLE_DEVICES=-1`), in Git Bash. Everything they write
goes under `runs/`, which is gitignored.

## The dumps

The dumps live in `runs/odometry_overlay/<name>.npz`. `_12` means `--rate 12`, the frames the
deployed loop sees. `_30` means every frame: Truth and Starfruit are 30 fps, and the phone clips
57–59 fps.

| dumps | clip | `--hud` | windows |
|---|---|---|---|
| `truth_12`, `truth_30` | `bluestacks-truth-west-upper` | emulator | 8 |
| `starfruit_12`, `starfruit_30` | `bluestacks-starfruit-supernova-west-upper` | emulator | 8 |
| `dp_bl_12`, `dp_br_12`, `dp_tr_12` | `bluestacks-dark-passage-bottom-left` / `-bottom-right` / `-top-right` | emulator | 8 |
| `standstill_30`, `counted_walking_30`, `zone_30`, `alt2_30`, `gadget_30` | `standstill`, `counted_walking`, `zone_grows_from_east`, `showdown_alternate_map2`, `training_gadget` | phone | 11 |

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/probes/odometry_overlay/dump_shifts.py \
    bluestacks-truth-west-upper --hud emulator --rate 12 --out runs/odometry_overlay/truth_12.npz
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/probes/odometry_overlay/vote.py \
    runs/odometry_overlay/truth_12.npz bluestacks-truth-west-upper --hud emulator \
    > runs/odometry_overlay/shipped/vote_truth_12.txt
```

`runs/odometry_overlay/shipped/vote_<dump>.txt` holds `vote.py`'s output for every dump, from the
shipped code. The `vote_*.txt` files one level up came from a first version that re-implemented
the rule, and are superseded.

## What the vote measured

The match windows come from the replay logs' gate times:

- Truth: 17.0–27.3 s and from 47.4 s on;
- Dark Passage: 11.9–39.5 s (bottom-left), 8.3–58.0 s (bottom-right), 5.3–42.2 s (top-right);
- Starfruit: from 15.8 s on;
- the phone clips are all gameplay.

| where | the moving group won at | lost at |
|---|---|---|
| in a match, 12 Hz (Truth, Starfruit, Dark Passage) | 0.80–0.95 | 0.04–0.26 |
| in a match, every frame (Truth, Starfruit) | 0.53–0.90; the low end is at the death that ends Truth's match 1, frames 815–819 | 0.05–0.13 |
| phone clips | never; the highest vote was 0.20, and `standstill` and `counted_walking` had no pair in question | |
| outside a match (lobby, loading, results) | 0.05–0.89 either way | |

- **Votes outside a match.** They only move the world origin, which the localizer fixes at the
  gate. `replay_overlay.py match` takes them out.
- **Truth at 12 Hz, in a match.** 133 pairs went `uncertain` → `ok` with the rule on, and 8 went
  `ok` → `uncertain` (frames 625–680 and 2435).
  - On those 8, a window over near-still ground reads just over `still_tiles`, so it counts as a
    disagreeing mover once the movers carry the estimate.
  - Another 60 pairs stayed `ok` but won the vote, which moved their estimate by 0.13 tile at the
    median (up to 0.25). These are the frozen and halfway-median frames that nothing had flagged.
    10 more stayed `ok` and lost the vote, so nothing changed for them.
- **Dark Passage at 12 Hz:** `uncertain` → `ok` on 6, 13 and 5 pairs (bottom-left, bottom-right,
  top-right). **Starfruit at 12 Hz:** 2.

## Replays

The replays are in `runs/map_checks/truth/overlay_fix/`. `MAP_ONBOARDING.md` has the tables.

- `{truth,bottom-left,bottom-right,top-right,starfruit}` are the rule as shipped. They used the
  regression replays' arguments (`scripts/probes/map_label_projection/truth/regress.sh`), and
  Truth used M3's.
- `noise/` is Dark Passage bottom-left with the rule off and on, fed from 0.1 s and 0.2 s.
- `inmatch/` has the rule on in a match only.

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/probes/odometry_overlay/replay_overlay.py match \
    brawl_vision/data/training_videos/bluestacks-dark-passage-bottom-left.mp4 \
    --map dark_passage --terrain default --mask on --device cpu --quiet --start 0 \
    --csv runs/map_checks/truth/overlay_fix/inmatch/bottom-left-0.csv
.venv/Scripts/python.exe scripts/probes/map_localize_report.py \
    runs/map_checks/truth/overlay_fix/inmatch/bottom-left-0.csv --map dark_passage
```

**Checked on 2026-11-05.** On bottom-left, `off` reproduces the replay made before the rule existed
(`runs/map_checks/truth/regression/bottom-left-shipped.csv`) tick for tick. `match` reproduces
`inmatch/bottom-left-0.csv`. Only `tick_ms` differs. So `overlay_share: 1.0` is exactly the old
estimator.
