"""Measure a brawler's real ammo timing -- `reload_seconds` and `attack_cooldown` -- from footage
or from a deployment run's telemetry.

    python scripts/measure_reload.py tests/fixtures/vision/bluestacks-example-new.mp4
    python scripts/measure_reload.py run1.csv run2.csv
    python scripts/measure_reload.py <clip> --brawler hero_mortis --write

**Why this exists.** The operator's instruction is explicit: the sim's move speed, attack speed
and reload speed are approximations, and a measurement taken from Nulls Brawl beats one taken from
`configs/brawlers.yaml`. The ammo pip bar is the ONLY kit timer the game draws, so it is the only
one that can be read back off a screen at all -- `attack_cd`, `invuln_t` and the dash timers are
invisible and stay dead-reckoned in `brawl_deployment/perception/shadow.py`.

#### Intervals, never events

The pip bar lags the input by about a second -- 0.50-1.25 s measured tap-to-pip, median 1.00
(BRAWL_DEPLOYMENT_DESIGN.md 6.3). That destroys any measurement of WHEN something happened and
leaves every measurement of HOW LONG BETWEEN two things intact, because a constant lag subtracts
out of a difference. So nothing here is timed against an emitted action. Everything is timed
between two pip transitions, and the lag enters only as jitter on the difference.

#### The two families, and why they are reported apart

`brawl_sim/core/hero.tick_timers` models firing as PAUSING the reload for `attack_cooldown`
seconds ("sustained fire is `attack_cooldown + reload_seconds` per shot"). Under that model the
two observable intervals mean different things:

    A   gain -> gain, with no shot in between        =  reload_seconds
    B   shot from a FULL magazine -> the next gain   =  attack_cooldown + reload_seconds

B is well defined only from full, because below full the reload timer is already running when the
shot lands and the interval measures whatever was left of it.

So B - A is `attack_cooldown` measured on its own -- the constant `configs/brawlers.yaml` defers
as "Step C1, paired with the reload pause that gives it meaning" -- and A == B would falsify the
pause model.

**That test is OUT OF SCOPE and not wanted (operator, 2026-09-11).** The pause is Mortis's attack
animation and stays in the sim. This script is for refitting `reload_seconds` (family A). Family B
is still printed, but do not read it as evidence about the pause, record clips to settle it, or
propose removing the pause. BRAWL_DEPLOYMENT_DESIGN.md 6.15 result 2.

#### Never pooled across sources

`reload_seconds` is per-brawler and `tests/fixtures/vision/README.md` does not record which
brawler each clip shows, so a pooled median is a number about a mixture. That is not hypothetical:
on the fixture set `showdown_has_gadget` yields 0.75 s where `bluestacks-example-new` yields
2.50 s. Pooling those two produces a reload nobody has.

Only the BlueStacks clips have documented provenance -- BRAWL_DEPLOYMENT_DESIGN.md records that
"the BlueStacks fixture turned out to be mostly Mortis wandering alone" -- and a deployment
telemetry CSV is Mortis by construction. Everything else is reported under its own filename and
left for the operator, who recorded them, to attribute. `--brawler` only labels the output and
picks which `configs/brawlers.yaml` row to print alongside; it never merges sources.

#### What it cannot do

It cannot name the brawler in a clip, it cannot see a reload that finishes while the hero box is
undetected (those intervals are dropped, not guessed), and on a telemetry CSV it is limited to the
decision rate -- 4 Hz, so 0.25 s of quantization against a ~2.5 s interval. Footage at 30 Hz is
the better instrument; telemetry is the one that is certainly Mortis. Run both.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brawl_vision.config import load_vision_config                              # noqa: E402
from brawl_vision.object_detection.hp_detection import read_ammo                # noqa: E402

OUT_PATH = REPO / "brawl_deployment" / "data" / "kit_timing.json"

# How long a new pip count must hold before it counts as a transition. The reader misses ~4.6% of
# frames and misreads a few more (see `hero_bars.py`'s own table), and a single-frame blip would
# otherwise manufacture a gain and a spend a frame apart. This is a TIME and not a frame count so
# that a 30 fps clip, a 60 fps clip and a 4 Hz telemetry CSV get the same debounce -- an earlier
# frame-count version accepted 0.03 s "reloads" on the 60 fps clips.
#
# 0.25 s is an order below the ~2.5 s interval being measured and above every misread run observed
# (partial fills never persist past 9 frames). MEASURED: the answer does not move at all across
# 0.1 / 0.2 / 0.3 / 0.5 s, which is what makes it safe to have a default at all.
HOLD_SECONDS = 0.25

# Drop an interval containing an unreadable stretch this long. A missed transition does not corrupt
# an interval a little, it DOUBLES it -- two reloads reported as one -- so the guard has to be on
# the gap and not on the answer. Rejecting "too far from the median" would be fitting the filter to
# the conclusion it is supposed to be testing.
MAX_GAP_SECONDS = 0.4

# Frames per second to sample a clip at. 30 Hz puts at most 33 ms of quantization on each end of a
# ~2.5 s interval, under 3% in total, and halves the detector cost on the 60 fps fixtures. Raise it
# if you are ever measuring something short.
SAMPLE_HZ = 30.0


# ---------------------------------------------------------------------------------------------
# sources -- each returns [(t_seconds, ammo_whole_or_None)] in time order
# ---------------------------------------------------------------------------------------------

def series_from_clip(path: Path, rate: float, max_seconds: float | None) -> list:
    """Run the detector and the ammo reader over a recorded clip.

    Resized to the calibrated viewport exactly as `loop.py` does, so this measures the instrument
    the deployed agent actually reads through rather than a differently-scaled one.
    """
    import cv2

    from brawl_deployment.capture import to_viewport
    from brawl_vision.clips import ClipReader
    from brawl_vision.object_detection import ObjectDetector

    cfg = load_vision_config()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or rate
    cap.release()
    step = max(1, round(fps / rate))

    det = ObjectDetector.from_config(cfg)
    out = []
    for frame in ClipReader(path, cfg, step=step):
        if max_seconds is not None and frame.t > max_seconds:
            break
        image = to_viewport(frame.image)
        players = [d for d in det.predict(image) if d.label == "player"]
        if not players:
            out.append((frame.t, None))
            continue
        r = read_ammo(image, max(players, key=lambda p: p.confidence), cfg)
        out.append((frame.t, None if r is None else round(r.ammo)))
    return out


def series_from_telemetry(path: Path) -> list:
    """`ammo_cv` out of a `deploy_run.py --telemetry` CSV.

    `-1` is that column's no-value sentinel -- no decision this tick, no hero box, or the reader
    returned None -- which is exactly the `None` this function's contract wants. It must never be
    read as an empty magazine, which is a real and different reading.
    """
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if "ammo_cv" not in row:
                raise SystemExit(f"{path}: no `ammo_cv` column -- telemetry from before 6.3?")
            value = float(row["ammo_cv"])
            out.append((float(row["t"]), None if value < 0 else round(value)))
    return out


# ---------------------------------------------------------------------------------------------
# transitions and intervals
# ---------------------------------------------------------------------------------------------

def transitions(series: list, hold: float) -> list:
    """`[(t, level)]` at each accepted change, timestamped at the FIRST read of the new level.

    First rather than last because the widget snaps rather than ramps, so the first frame showing
    the new count is the transition as closely as this instrument can see it. Timing it at the end
    of the hold instead would add `hold` to both endpoints of every interval -- which cancels, but
    only for as long as nobody compares two runs measured at different `hold`s.
    """
    out, current, pending, pending_t = [], None, None, None
    for t, value in series:
        if value is None:
            continue
        if value != pending:
            pending, pending_t = value, t
        if pending != current and t - pending_t >= hold:
            current = pending
            out.append((pending_t, current))
    return out


def intervals(steps: list, readable_ts: list, max_ammo: int, max_gap: float) -> dict:
    """The two families, plus counts of what was thrown away and why."""
    def spans_gap(t0: float, t1: float) -> bool:
        window = [t for t in readable_ts if t0 <= t <= t1]
        return len(window) < 2 or any(b - a > max_gap for a, b in zip(window, window[1:]))

    out = {"reload": [], "shot_to_gain": [], "shot_to_shot": [],
           "dropped_gap": 0, "dropped_multi": 0}
    last_gain = last_full_shot = last_shot = None
    for (t0, a0), (t1, a1) in zip(steps, steps[1:]):
        step = a1 - a0
        if abs(step) > 1:
            # Two pips at once, either way, means a level we never saw -- a reload that finished
            # while the hero box was lost, or two shots closer together than the debounce. Nothing
            # anchored before it survives it, and it anchors nothing after it: the interval ending
            # here would be two cycles reported as one, and the interval starting here would be
            # timed from a transition that is not the one it names.
            last_gain = last_full_shot = last_shot = None
            out["dropped_multi"] += 1
        elif step == 1:
            for previous, key in ((last_gain, "reload"), (last_full_shot, "shot_to_gain")):
                if previous is None:
                    continue
                if spans_gap(previous, t1):
                    out["dropped_gap"] += 1
                else:
                    out[key].append(t1 - previous)
            last_gain, last_full_shot = t1, None
        else:
            if last_shot is not None and not spans_gap(last_shot, t1):
                out["shot_to_shot"].append(t1 - last_shot)
            last_shot = t1
            last_gain = None
            # Only a shot from a FULL magazine starts the reload clock at a known instant.
            last_full_shot = t1 if a0 == max_ammo and step == -1 else None
    return out


def describe(samples: list) -> dict | None:
    if not samples:
        return None
    s = sorted(samples)
    return {"n": len(s), "median": statistics.median(s), "min": s[0], "max": s[-1],
            "p25": s[len(s) // 4], "p75": s[(3 * len(s)) // 4]}


def histogram(samples: list, width: float = 0.25, rows: int = 14) -> str:
    """The shape, not just the middle. A second mode at twice the first is a missed transition and
    no summary statistic shows that -- which is the whole reason this is printed.

    Empty buckets are drawn, because the gap between two modes is the thing being looked for. One
    long tail would otherwise draw a hundred of them, so anything past `rows` is summed into an
    overflow line rather than dropped: a discarded outlier is a hidden one.
    """
    if not samples:
        return ""
    buckets: dict[int, int] = {}
    for value in samples:
        buckets[int(value / width)] = buckets.get(int(value / width), 0) + 1
    low, high = min(buckets), max(buckets)
    lines = [f"      {k * width:5.2f}-{(k + 1) * width:4.2f}  {'#' * buckets.get(k, 0) or '.'}"
             for k in range(low, min(high, low + rows - 1) + 1)]
    over = sum(n for k, n in buckets.items() if k > low + rows - 1)
    if over:
        lines.append(f"      >={(low + rows) * width:5.2f}       {'#' * over}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------

def measure(path: Path, args) -> dict:
    if path.suffix.lower() == ".csv":
        series = series_from_telemetry(path)
        instrument = "telemetry, at the decision rate"
    else:
        series = series_from_clip(path, args.rate, args.max_seconds)
        instrument = f"clip resampled to ~{args.rate:g} Hz"

    readable = [t for t, v in series if v is not None]
    steps = transitions(series, args.hold)
    raw = intervals(steps, readable, args.max_ammo, args.max_gap)
    period = (statistics.median([b - a for a, b in zip(readable, readable[1:])])
              if len(readable) > 2 else float("nan"))
    return {
        "source": path.name, "instrument": instrument,
        "frames": len(series), "readable": len(readable),
        "sample_period_s": period, "transitions": len(steps),
        "levels": [v for _, v in steps],
        "reload": describe(raw["reload"]),
        "shot_to_gain": describe(raw["shot_to_gain"]),
        "shot_to_shot": describe(raw["shot_to_shot"]),
        "dropped_gap": raw["dropped_gap"], "dropped_multi": raw["dropped_multi"],
        "_raw": raw,
    }


FAMILIES = (
    ("reload", "A  gain -> gain           reload_seconds"),
    ("shot_to_gain", "B  full-mag shot -> gain  attack_cooldown + reload_seconds"),
    ("shot_to_shot", "   shot -> shot           a FLOOR on the fire cycle, not an estimate"),
)


def report(r: dict) -> None:
    print(f"\n=== {r['source']} ===")
    print(f"  {r['instrument']}, {r['readable']}/{r['frames']} frames readable "
          f"({r['readable'] / max(1, r['frames']) * 100:.1f}%), "
          f"sample period {r['sample_period_s'] * 1000:.0f} ms")
    print(f"  {r['transitions']} accepted transitions: {r['levels']}")
    if r["dropped_gap"] or r["dropped_multi"]:
        print(f"  dropped {r['dropped_gap']} interval(s) spanning an unreadable stretch and "
              f"{r['dropped_multi']} multi-pip jump(s)")
    for key, label in FAMILIES:
        s = r[key]
        if s is None:
            print(f"  {label:58s}  no samples")
            continue
        print(f"  {label:58s}  n={s['n']:<3d} median {s['median']:.2f} s  "
              f"[{s['p25']:.2f}, {s['p75']:.2f}]  min {s['min']:.2f} max {s['max']:.2f}")
        print(histogram(r["_raw"][key]))
    a, b = r["reload"], r["shot_to_gain"]
    if a and b:
        print(f"  B - A = {b['median'] - a['median']:+.2f} s  <- informational only: whether firing "
              f"pauses the reload is out of scope (design doc 6.15)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources", nargs="+", type=Path,
                    help="clips (*.mp4) and/or deploy_run telemetry (*.csv)")
    ap.add_argument("--brawler", default=None,
                    help="row in configs/brawlers.yaml to print alongside, e.g. hero_mortis. It "
                         "LABELS the output; it never merges or reinterprets a source.")
    ap.add_argument("--hold", type=float, default=HOLD_SECONDS)
    ap.add_argument("--max-gap", type=float, default=MAX_GAP_SECONDS)
    ap.add_argument("--rate", type=float, default=SAMPLE_HZ)
    ap.add_argument("--max-ammo", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop each clip early; for a quick look, not for a measurement")
    ap.add_argument("--write", action="store_true",
                    help=f"store the per-source results in {OUT_PATH.name}")
    args = ap.parse_args()

    results = []
    for path in args.sources:
        if not path.exists():
            print(f"skipping {path}: not found")
            continue
        results.append(measure(path, args))
        report(results[-1])

    if not results:
        return 2

    print("\n--- against the sim ---")
    if args.brawler:
        import yaml
        kit = yaml.safe_load((REPO / "configs" / "brawlers.yaml").read_text())[args.brawler]
        print(f"  configs/brawlers.yaml {args.brawler}: reload_seconds {kit['reload_seconds']}, "
              f"attack_cooldown {kit['attack_cooldown']}, so family A should read "
              f"{kit['reload_seconds']:.2f} s and family B "
              f"{kit['reload_seconds'] + kit['attack_cooldown']:.2f} s")
    else:
        print("  no --brawler given, so there is nothing to compare against")
    print("  Sources are NOT pooled: reload_seconds is per-brawler and the fixture README does "
          "not record\n  which brawler each clip shows. See this file's docstring.")

    if args.write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "brawler": args.brawler,
            "params": {"hold_seconds": args.hold, "max_gap_seconds": args.max_gap,
                       "sample_hz": args.rate, "max_ammo": args.max_ammo},
            "sources": [{k: v for k, v in r.items() if k != "_raw"} for r in results],
        }
        OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {OUT_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
