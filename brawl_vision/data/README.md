# Calibration artifacts

Everything here is a **measurement**, not a setting. These files are written by a calibration
script, read by `brawl_vision/camera.py`, and are meaningless to hand-edit — the numbers only
mean anything as a set, produced together from one reference frame.

Tunable thresholds live in `configs/vision.yaml` instead. If you find yourself wanting to nudge
a value in this directory to see what happens, the value you actually want is almost certainly
over there.

## Files

| File | Written by | Phase | Tracked |
|---|---|---|---|
| `hud_mask.json` | `scripts/vision_calibrate.py --hud` | B | yes |
| `homography.json` | `scripts/vision_calibrate.py` | C | yes |
| `frames/*.png` | you, by hand | A/B/C/G | **no** — see below |

## Why `frames/` is gitignored

Screen captures of a live match are extracted game assets (Supercell IP), the same rule
`brawl_sim/render/assets/README.md` states for the sprite PNGs. `.gitignore` excludes them **by
file extension**, not by directory, so this README and the folder structure stay tracked while
the pixels do not.

That distinction matters more than it looks: the hand-authored annotations that live *alongside*
the footage — `calib_points.csv`, label CSVs, `*.truth.json`, `*.grid.csv` — are your work, not
Supercell's, and they are what the regression fixtures actually assert against. They stay
tracked. Only the imagery is excluded.

Practical consequence: a fresh clone has the calibration *outputs* but not the *inputs*. That is
fine for running the pipeline and wrong for re-deriving it, so if you ever need to re-calibrate,
you need to re-capture first.

## `homography.json` schema

```json
{
  "H":               [[...], [...], [...]],   // 3x3, tile-space -> screen pixels
  "H_inv":           [[...], [...], [...]],   // 3x3, screen pixels -> tile-space
  "pixels_per_tile": 32,                      // chosen output scale for the rectified patch
  "capture_size":    [1920, 1080],            // (w, h) the fit was made at
  "is_affine":       false,                   // H[2, :2] ~ 0 -> orthographic camera
  "source_frame":    "calib_2026-08-26.png"
}
```

`capture_size` is not decoration. A homography is specific to the resolution it was fit at, and
applying a 1080p calibration to a 1440p capture produces a plausible-looking, uniformly wrong
rectification — the exact failure mode that is expensive to notice. `camera.py` compares it
against the live frame and raises rather than silently rescaling.

`is_affine` records the Phase C diagnostic: if `H[2, :2]` came back ≈ 0 the game camera is
orthographic-oblique rather than perspective, which means tile scale is constant across the
screen and the visible footprint is a parallelogram rather than a trapezoid. Phase K needs that
answer to pick the sim's view rectangle.

## `hud_mask.json` schema

```json
{
  "rects": [ {"name": "joystick", "x": 0.02, "y": 0.62, "w": 0.20, "h": 0.34}, ... ],
  "notes": "captured from build 61.xxx"
}
```

Rectangles are in **normalized `[0, 1]` screen coordinates**, so the mask survives a resolution
change without re-annotation.

**This mask goes stale silently if Supercell changes the UI layout.** There is no auto-detection
and it is not worth building one yet; the `notes` field exists so that when the mask does break
after a game update, you can tell how old it is.
