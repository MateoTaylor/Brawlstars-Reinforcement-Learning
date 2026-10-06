"""K4's report on a known-map run: KNOWN_MAP_LOCALIZATION_PLAN.md step K4, and K5 on a live CSV.

Reads the telemetry `replay_clip.py --csv` and `deploy_run.py --telemetry` write (the `map_*` and
`hero_odo_*` columns) and prints, for each file:

- the gate, the first fix, and how long after the gate it came;
- the spawn the first fix put the hero on: the hero's raw odometry position at the gate plus the
  offset, where the hero is still standing on the spawn;
- with `--spawn`, the TRUE offset (that spawn's centre minus the hero at the gate) and every fix's
  distance from it. Odometry drifts, so late in a long match a right fix can read a little off.
  The truth needs the hero still on its spawn at its first logged reading: the gate opens about
  0.5 s after the HUD shows, so a hero that runs at once has left the spawn by then, and the
  report warns when it was moving (K5);
- for each stretch without a fix, where the search's leader sat at its end, scored against the
  truth like a fix, and for how much of the stretch it was right (the `map_lead_*` columns, K5;
  files written before them print none);
- how many times a fix was dropped after the first and found again;
- time in each state, agreement while fixed, corrections per minute, and the crate residuals:
  their mean (a map's crates can sit a fixed amount off the terrain's tile centres) and their
  spread about it;
- the gates from the plan's K4 table.

A fix counts as right while it is within half a tile of the truth on both axes, a tile off while it
is within 1.5, and wrong past that.

Usage (repo root):

    .venv/Scripts/python.exe scripts/probes/map_localize_report.py run.csv --spawn 6,47
    .venv/Scripts/python.exe scripts/probes/map_localize_report.py a.csv b.csv --spawn 6,47 --spawn 47,53
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brawl_deployment.loop import read_telemetry_csv                   # noqa: E402
from brawl_deployment.perception.known_map import KnownMap             # noqa: E402

# The plan's K4 gates. The crate spread's is 0.25 tile, not the first draft's 0.2 (the user,
# 2026-09-28): check 3 lets the fix wander 0.2 tile before it steps, and in perspective a crate
# box's centre is rarely its tile's centre, by an amount that changes across the screen.
GATES = {"commit_s": 2.0, "resid_rms": 0.25, "unfixed": 0.05}


def _spawn(text: str) -> tuple[int, int]:
    col, row = (int(v) for v in text.split(","))
    return col, row


def _versus(offset, truth) -> str:
    """An offset's distance from the truth, as the timeline prints it."""
    err = offset - truth
    far = np.abs(err).max()
    kind = "right" if far < 0.5 else "a tile off" if far < 1.5 else "WRONG"
    return f"  minus truth ({err[0]:+5.2f}, {err[1]:+5.2f})  {kind}"


def _residuals(rows) -> np.ndarray:
    pairs = [tuple(float(v) for v in p.split()) for r in rows if r.map_crate_resid
             for p in r.map_crate_resid.split(";")]
    return np.array(pairs, float).reshape(-1, 2)


def report(path: Path, known: KnownMap, spawn: tuple[int, int] | None) -> dict:
    rows = read_telemetry_csv(path)
    play = [r for r in rows if r.phase == "playing"]
    print(f"\n{path.name}")
    if not play:
        print("  no match in this file")
        return {}
    t = np.array([r.t for r in play])
    dt = np.diff(t, append=t[-1] + (np.median(np.diff(t)) if len(t) > 1 else 0.0))
    gate, length = t[0], float(dt.sum())
    has_fix = np.array([not math.isnan(r.map_dx) for r in play])
    offsets = np.array([(r.map_dx, r.map_dy) for r in play], float)
    print(f"  match {length:.1f} s from the gate at {gate:.2f} s")

    states = {}
    for r, d in zip(play, dt):
        states[r.map_state or "-"] = states.get(r.map_state or "-", 0.0) + d
    print("  time in each state: " + ", ".join(f"{k} {v:.1f} s ({v / length:.0%})"
                                               for k, v in sorted(states.items())))
    unfixed = float(dt[~has_fix].sum()) / length

    heroes = [(r.t, np.array([r.hero_odo_x, r.hero_odo_y])) for r in play
              if not math.isnan(r.hero_odo_x)]
    hero0 = heroes[0][1] if heroes else None
    truth = None
    if spawn is not None and hero0 is not None:
        truth = np.array([spawn[0] + 0.5 - 30.0, spawn[1] + 0.5 - 30.0]) - hero0
        print(f"  truth: spawn {spawn}, hero at the gate ({heroes[0][0]:.2f} s) at odometry "
              f"({hero0[0]:+.2f}, {hero0[1]:+.2f}), true offset ({truth[0]:+.2f}, {truth[1]:+.2f})")
        if len(heroes) > 1:
            # K5: both of the second held-out pair's heroes ran from the first frame, and the
            # truth came out a tile off along the way they ran while still frames showed the
            # fixes within 0.4 tile. Standing still, odometry jitters well under 1 tile/s.
            (ta, a), (tb, b) = heroes[0], heroes[1]
            speed = float(np.linalg.norm(b - a)) / max(tb - ta, 1e-6)
            if speed > 1.0:
                print(f"  WARNING: the hero was already moving, {speed:.1f} tiles/s, so the truth is "
                      f"off by as far as it had run from the spawn; a held-out match needs a second "
                      f"standing still at the start")
    elif spawn is not None:
        print("  truth: no hero_odo columns in this file, so no truth")

    out = dict(length=length, unfixed=unfixed, commit_s=None, picked=None, resid_rms=None)
    if not has_fix.any():
        print("  never fixed")
    else:
        i = int(np.argmax(has_fix))
        out["commit_s"] = float(t[i] - gate)
        print(f"  first fix {t[i] - gate:.2f} s after the gate, offset "
              f"({offsets[i, 0]:+.2f}, {offsets[i, 1]:+.2f})")
        if hero0 is not None:
            at = hero0 + offsets[i]
            k = int(np.argmin(np.linalg.norm(known.spawns - at, axis=1)))
            col, row = (int(v) for v in np.floor(known.spawns[k] + 30.0))
            out["picked"] = (col, row)
            print(f"  it put the gate's hero {np.linalg.norm(known.spawns[k] - at):.2f} tiles from "
                  f"spawn ({col}, {row})")

    # One line per stretch with one offset (or none), and how far it was from the truth.
    print("  timeline, seconds after the gate:")
    start = 0
    for j in range(1, len(play) + 1):
        same = j < len(play) and has_fix[j] == has_fix[start] and \
            (not has_fix[j] or np.array_equal(offsets[j], offsets[start]))
        if same:
            continue
        line = f"    {t[start] - gate:6.2f} s  {float(dt[start:j].sum()):5.1f} s  "
        if not has_fix[start]:
            line += f"{play[start].map_state or '-'}, no fix"
            leads = np.array([(r.map_lead_dx, r.map_lead_dy) for r in play[start:j]], float)
            if not np.isnan(leads[-1]).any():
                line += f", leader at the end ({leads[-1, 0]:+6.2f}, {leads[-1, 1]:+6.2f})"
                if truth is not None:
                    # A tick with no leader (NaN) counts as not right.
                    right = np.abs(leads - truth).max(axis=1) < 0.5
                    share = float(dt[start:j][right].sum()) / float(dt[start:j].sum())
                    line += f"{_versus(leads[-1], truth)}, right for {share:.0%} of the stretch"
        else:
            line += f"fix ({offsets[start, 0]:+6.2f}, {offsets[start, 1]:+6.2f})"
            if truth is not None:
                line += _versus(offsets[start], truth)
        print(line)
        start = j
    if has_fix.any():
        # After the first fix a commit needs only the guard's agreement, so a stretch that reads
        # near it could drop and re-fix over and over, each time a new epoch. This counts it.
        after = has_fix[int(np.argmax(has_fix)):]
        out["drops"] = int((after[:-1] & ~after[1:]).sum())
        print(f"  after the first fix: {out['drops']} dropped, "
              f"{int((~after[:-1] & after[1:]).sum())} found again")

    if truth is not None:
        err = np.abs(offsets - truth).max(axis=1)
        right = has_fix & (err < 0.5)
        wrong = has_fix & (err >= 1.5)
        out["right"] = float(dt[right].sum()) / length
        out["wrong"] = float(dt[wrong].sum()) / length
        print(f"  with a right fix {out['right']:.0%} of the match, a tile off "
              f"{float(dt[has_fix & ~right & ~wrong].sum()) / length:.0%}, wrong {out['wrong']:.0%}, "
              f"no fix {unfixed:.0%}")

    fixed = [r for r in play if r.map_state == "fixed"]
    agree = np.array([r.map_agree for r in fixed if not math.isnan(r.map_agree)])
    if len(agree):
        print(f"  agreement while fixed: median {np.median(agree):.2f}, 5th percentile "
              f"{np.percentile(agree, 5):.2f}, min {agree.min():.2f} over {len(agree)} ticks")
    steps = np.diff(offsets, axis=0)
    moved = has_fix[1:] & has_fix[:-1] & (np.abs(steps).max(axis=1) > 1e-6)
    whole = moved & (np.abs(steps).max(axis=1) >= 0.5)
    minutes = float(dt[has_fix].sum()) / 60.0
    if minutes > 0:
        print(f"  corrections: {int(whole.sum())} whole-tile, {int((moved & ~whole).sum())} sub-tile, "
              f"{int(moved.sum()) / minutes:.1f} per minute fixed")
    resid = _residuals([r for r in play if not math.isnan(r.map_dx)])
    if len(resid):
        # K4: a map's crates can sit a fixed amount off the terrain's tile centres (Dark
        # Passage's by 0.41 tile), so the gate reads their spread about that offset. Both are
        # circular, as the residuals are: 0.45 and -0.45 are a tenth of a tile apart.
        mean = np.angle(np.exp(2j * math.pi * resid).mean(axis=0)) / (2 * math.pi)
        about = (resid - mean + 0.5) % 1.0 - 0.5
        out["resid_rms"] = float(np.sqrt((about ** 2).sum(axis=1).mean()))
        rms = np.sqrt((resid ** 2).mean(axis=0))
        print(f"  crate residuals after the fix: {len(resid)}, mean ({mean[0]:+.3f}, {mean[1]:+.3f}) "
              f"(the map's crate offset), spread about it {out['resid_rms']:.3f} tile; RMS about 0 "
              f"x {rms[0]:.3f} y {rms[1]:.3f}")

    print("  gates:")
    if spawn is not None and out["picked"] is not None:
        ok = out["picked"] == tuple(spawn)
        print(f"    spawn picked right            {'yes' if ok else 'NO'} {out['picked']}")
    c = out["commit_s"]
    print(f"    commit after the gate < {GATES['commit_s']:g} s   "
          f"{'never' if c is None else f'{c:.2f} s'} {'ok' if c is not None and c < GATES['commit_s'] else 'MISS'}")
    rr = out["resid_rms"]
    print(f"    crate residual spread <= {GATES['resid_rms']:g} "
          f"{'no crates' if rr is None else f'{rr:.3f}'} "
          f"{'' if rr is None else 'ok' if rr <= GATES['resid_rms'] else 'MISS'}")
    print(f"    time without a fix < {GATES['unfixed']:.0%}     {unfixed:.1%} "
          f"{'ok' if unfixed < GATES['unfixed'] else 'MISS'}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--spawn", type=_spawn, action="append", default=[],
                    help="col,row of the spawn each CSV's match started on, in the same order")
    ap.add_argument("--map", default="dark_passage")
    args = ap.parse_args(argv)
    if args.spawn and len(args.spawn) != len(args.csv):
        ap.error(f"{len(args.csv)} CSVs but {len(args.spawn)} --spawn values")
    known = KnownMap.load(args.map)
    for i, path in enumerate(args.csv):
        report(path, known, args.spawn[i] if args.spawn else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
