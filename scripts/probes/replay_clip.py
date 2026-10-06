"""Replay a recorded clip through the REAL `DeployLoop` and read back its decisions.

The live path end to end -- `VisionStack.build` (the shipped detectors, classifier and HUD reader),
`MatchState` on the real calibration, the policy from the run `configs/deployment.yaml` names, the
tracker, grid, shadow and dead-bin mask exactly as `scripts/deploy_run.py` wires them -- with three
substitutions and nothing else: the capture hands the loop the clip's frames (resized to the
calibrated viewport, `brawl_deployment.capture.to_viewport`), the controls drive a `NullBackend`,
and the window guard always answers "fine". The loop's clock is the clip's: the shadow hero is
advanced by the gap between the frames it was fed, not by how long a CPU detector took.

What it answers: does the policy, on the footage where the
deployed agent stood pushing into a wall, pick a legal bin once the dead-bin mask is on? With
`--mask both` the clip is replayed twice, once with `policy.dead_bin_mask` off and once on, and the
two runs' decisions are lined up frame by frame (perception is deterministic on a fixed clip, so
they decide on the same ticks): for each decision the unmasked bin, the masked bin, and the legal
set the mask computed. A decision the mask VETOED is one whose unmasked bin the mask marked dead.

Usage (repo root, the venv's python; `--device cpu` keeps every model off the GPU):

    .venv/Scripts/python.exe scripts/probes/replay_clip.py "C:/Users/mateo/Videos/2026-09-23 20-20-47.mp4" \
        --window 22 32 --mask both --device cpu
    .venv/Scripts/python.exe scripts/probes/replay_clip.py <clip> --run runs/<deploy5 run> --mask on
    .venv/Scripts/python.exe scripts/probes/replay_clip.py <clip> --map dark_passage --mask on \
        --device cpu --csv <out.csv>       # KNOWN_MAP_LOCALIZATION_PLAN.md step K4

`--start` skips the clip's head (the gate still has to open, so leave a few seconds before the
window); `--window` is the stretch reported in detail. Writes nothing; `--csv` dumps every tick's
`TickRow` for the last run, as `deploy_run.py` would.

With a map, each run also prints K4's label check (`LabelCheck`): what the classifier saw at every
map tile while the loop was in the map frame, scored against the label. Its best offset must be
(0, 0), and the cells it is sure of that still disagree are listed for a second look in
`scripts/map_label.py`. `scripts/probes/map_localize_report.py` reads the `--csv`.
"""
import argparse
import dataclasses
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brawl_deployment.capture import to_viewport                        # noqa: E402
from brawl_deployment.config import load_deployment_config              # noqa: E402
from brawl_deployment.control import Buttons, Joystick, NullBackend      # noqa: E402
from brawl_deployment.loop import Controls, DeployLoop, VisionStack      # noqa: E402
from brawl_deployment.match_state import Calibration, MatchState        # noqa: E402
from brawl_deployment.perception.known_map import KnownMap              # noqa: E402
from brawl_deployment.policy import DeployedPolicy                      # noqa: E402
from brawl_sim.constants import TILE_TO_CHAR, Tile                      # noqa: E402
from brawl_vision.clips import ClipReader                               # noqa: E402
from brawl_vision.config import load_vision_config                      # noqa: E402
from brawl_vision.terrain.labeling import CLASSES                       # noqa: E402
from brawl_vision.terrain.occupancy import OccupancyMap, compare_to_truth  # noqa: E402

_EAST = 1      # move bin 1 is angle 0, +x: `geo.dir_from_bin(bin - 1)` in `core/hero.decode_action`
_BIN_NAMES = {0: "idle", 1: "E", 2: "ESE", 3: "SE", 4: "SSE", 5: "S", 6: "SSW", 7: "SW", 8: "WSW",
              9: "W", 10: "WNW", 11: "NW", 12: "NNW", 13: "N", 14: "NNE", 15: "NE", 16: "ENE"}


class ClipCapture:
    """`DeployCapture`'s contract -- `grab()`, cumulative `grab_seconds`, a context manager -- fed
    one frame at a time by the replay."""

    def __init__(self):
        self.grab_seconds = 0.0
        self.pending = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def grab(self):
        self.grab_seconds += 1e-4
        return self.pending


class NoGuard:
    def check(self):
        return None


class LabelCheck:
    """K4's label check (KNOWN_MAP_LOCALIZATION_PLAN.md). A second occupancy map takes every deposit
    the loop's own map takes while the frame handed out is the map's, and never resets, so after a
    match it holds what the classifier saw on each map tile. `report` scores it against the label."""

    SURE = 0.9        # a disputed cell is listed only when this share of its views agreed
    LISTED = 40

    def __init__(self, vision):
        self.vision = vision
        self.map = OccupancyMap.from_config(vision.cfg)
        self.loop = None

    def attach(self, loop):
        self.loop = loop
        occupancy = self.vision.occupancy
        original = occupancy.update

        def update(cells, world, plan, **kw):
            out = original(cells, world, plan, **kw)
            if loop.localizer is not None and loop.localizer.in_map_frame:
                self.map.update(cells, dataclasses.replace(world, segment=self.map.segment), plan,
                                **kw)
            return out
        occupancy.update = update

    def detach(self):
        vars(self.vision.occupancy).pop("update", None)

    def report(self, known: KnownMap) -> None:
        def merged(chars):
            # Fence counts as wall, as in the localizer's score.
            return np.where(chars == TILE_TO_CHAR[Tile.FENCE], TILE_TO_CHAR[Tile.WALL], chars)
        rows, cols = known.tiles.shape
        r0, c0 = self.map.height // 2 - rows // 2, self.map.width // 2 - cols // 2
        seen = merged(self.map.to_chars()[r0:r0 + rows, c0:c0 + cols])
        truth = merged(np.array([TILE_TO_CHAR[CLASSES[i]] for i in known.classes.ravel()])
                       .reshape(rows, cols))
        got = compare_to_truth(seen, truth)
        print(f"    label check: {int((seen != '?').sum())} map tiles seen; best offset "
              f"{got['best_offset']} (row, col) at {got['accuracy_at_best_offset']:.3f} over "
              f"{got['cells_compared']} cells, {got['accuracy_at_zero_offset']:.3f} at (0, 0)")
        differ = (seen != "?") & (seen != truth)
        pairs = Counter(f"label {truth[r, c]} seen {seen[r, c]}" for r, c in np.argwhere(differ))
        print(f"      disagree on {int(differ.sum())}: " + ", ".join(f"{k} x{n}" for k, n in
                                                                  pairs.most_common()))
        conf = self.map.confidence()[r0:r0 + rows, c0:c0 + cols]
        sure = np.argwhere(differ & (conf >= self.SURE))
        if len(sure):
            print(f"      {len(sure)} of them with {self.SURE:.0%} of views agreeing, for a look in "
                  f"map_label.py (col, row = label, seen): " +
                  ", ".join(f"({c}, {r}) = {truth[r, c]}, {seen[r, c]}" for r, c in sure[:self.LISTED]) +
                  (" ..." if len(sure) > self.LISTED else ""))


def build_loop(dcfg, *, vision, policy, log):
    """`scripts/deploy_run.py`'s wiring with a `NullBackend` in place of ADB (`Controls.build`'s
    geometry, field for field) and the clip capture."""
    cal = Calibration.load()
    null = NullBackend()
    w, h = cal.screen
    buttons = Buttons(null,
                      attack=(dcfg.control_attack_tap[0] * w, dcfg.control_attack_tap[1] * h),
                      super_=cal.button("super"), gadget=cal.button("gadget"),
                      aim_radius_px=dcfg.control_aim_radius_px)
    controls = Controls(backend=null,
                        joystick=Joystick(null, cal.joystick_anchor, cal.joystick_radius_px,
                                          n_bins=int(policy.cfg.n_move_bins)),
                        buttons=buttons)
    capture = ClipCapture()
    loop = DeployLoop(capture=capture, guard=NoGuard(), match=MatchState(cal), vision=vision,
                      policy=policy, controls=controls, cfg=dcfg, log=log)
    return loop, capture


def replay(clip, dcfg, *, vision, policy, start: float, stop: float | None, quiet: bool,
           check: LabelCheck | None = None):
    """Feed the clip at the loop's tick rate; return (decision rows, stops, ticks fed)."""
    stops = []
    logs = []
    loop, capture = build_loop(dcfg, vision=vision, policy=policy,
                               log=lambda m: logs.append((capture.pending.t if capture.pending
                                                          else -1.0, m)))
    if check is not None:
        check.attach(loop)
    # Record a stop and keep going: on a clip the point is to see every decision, and the reason
    # is reported with its timestamp.
    loop._stop = lambda reason: stops.append((capture.pending.t, reason))
    clock = {"dt": loop.tick_seconds}
    loop._elapsed_since_last_tick = lambda now: clock["dt"]

    rows = []
    period = loop.tick_seconds
    next_t = None
    last_t = None
    fed = 0
    t0 = time.perf_counter()
    # `trim_tail=False`: the sidecar's usable range is a single-match heuristic (a session file
    # with several matches keeps one); the loop's own gate decides what is gameplay here.
    with ClipReader(clip, trim_tail=False) as reader:
        for frame in reader:
            if frame.t < start:
                continue
            if stop is not None and frame.t > stop:
                break
            if next_t is not None and frame.t < next_t:
                continue
            # Sample the clip on the tick cadence; a dropped stretch resets rather than bursts,
            # like `DeployLoop._pace`.
            next_t = frame.t + period if (next_t is None or frame.t - next_t > period) else next_t + period
            clock["dt"] = period if last_t is None else max(frame.t - last_t, 1e-3)
            last_t = frame.t
            capture.pending = dataclasses.replace(frame, image=to_viewport(frame.image))
            loop.tick()
            fed += 1
            row = loop.telemetry[-1]
            if row.decision:
                odo = loop.vision.odometry
                rows.append(dict(t=frame.t, move_bin=int(row.move_bin), move_legal=int(row.move_legal),
                                 note=row.note or "", attack_legal=int(row.attack_legal),
                                 hero_offset=(float(row.hero_offset_x), float(row.hero_offset_y)),
                                 near_edge=int(row.near_edge),
                                 odo_pos=(float(odo.position[0]), float(odo.position[1])),
                                 odo_seg=int(odo.segment), odo_status=str(row.odometry)))
            if not quiet and fed % 120 == 0:
                print(f"    t={frame.t:6.1f}s  ticks {fed}  decisions {len(rows)}  phase {loop.phase.value}"
                      f"  {time.perf_counter() - t0:4.0f} s", flush=True)
    loop.controls.release_all()
    if check is not None:
        check.detach()
    return rows, stops, logs, fed, loop


def _bits(mask: int, n: int = 17) -> str:
    """Bins 1..16 as one character each, `.` legal / `x` dead, for the per-decision table."""
    if mask < 0:
        return "-" * (n - 1)
    return "".join("." if mask >> b & 1 else "x" for b in range(1, n))


def _hist(rows, window):
    lo, hi = window
    inside = [r for r in rows if lo <= r["t"] <= hi]
    c = Counter(r["move_bin"] for r in inside)
    return inside, c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip")
    ap.add_argument("--run", default=None, help="run dir; default configs/deployment.yaml run.dir")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--map", default=None,
                    help="labelled map to localize on (dark_passage); default "
                         "configs/deployment.yaml map.name")
    ap.add_argument("--mask", choices=("off", "on", "both"), default="both")
    ap.add_argument("--device", default="cpu", help="cpu keeps the detectors and policy off the GPU")
    ap.add_argument("--start", type=float, default=0.0, help="clip seconds to skip before feeding")
    ap.add_argument("--stop", type=float, default=None, help="clip seconds to stop feeding at")
    ap.add_argument("--window", type=float, nargs=2, default=(22.0, 32.0),
                    help="clip seconds reported in detail (the stuck stretch)")
    ap.add_argument("--csv", default=None, help="write the last run's telemetry here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    overrides = {"safety": {"capture_stall_seconds": 1e9, "max_match_seconds": 1e9}}
    if args.run:
        overrides["run"] = {"dir": args.run}
    if args.checkpoint:
        overrides.setdefault("run", {})["checkpoint"] = args.checkpoint
    if args.map:
        overrides["map"] = {"name": args.map}
    base = load_deployment_config(overrides=overrides)
    policy = DeployedPolicy.from_run(ROOT / base.run_dir, checkpoint=base.run_checkpoint,
                                     device=args.device, deterministic=True)
    vcfg = None
    if args.device == "cpu":
        vcfg = load_vision_config(overrides={"detector": {"device": "cpu"},
                                             "projectile": {"device": "cpu"},
                                             "classifier": {"device": "cpu"}})
    vision = VisionStack.build(vision_cfg=vcfg)
    print(f"clip {Path(args.clip).name}  run {base.run_dir}  checkpoint {base.run_checkpoint}  "
          f"map {base.map_name}  device {args.device}  "
          f"feed from {args.start:.1f}s  window {args.window[0]:.0f}-{args.window[1]:.0f}s")

    runs = {}
    for mask_on in ((False, True) if args.mask == "both" else ((args.mask == "on"),)):
        dcfg = dataclasses.replace(base, policy_dead_bin_mask=mask_on)
        label = "on" if mask_on else "off"
        print(f"  replay, dead-bin mask {label}", flush=True)
        t0 = time.perf_counter()
        check = LabelCheck(vision) if base.map_name else None
        rows, stops, logs, fed, loop = replay(args.clip, dcfg, vision=vision, policy=policy,
                                              start=args.start, stop=args.stop, quiet=args.quiet,
                                              check=check)
        runs[label] = rows
        notes = Counter(r["note"] or "decided" for r in rows)
        print(f"    {fed} ticks, {len(rows)} decision ticks in {time.perf_counter() - t0:.0f} s; "
              f"notes {dict(notes)}")
        for t, m in logs:
            if "match" in m or "gate" in m or m.startswith("map:"):
                print(f"    log t={t:6.1f}s: {m}")
        for t, reason in stops:
            print(f"    would have STOPPED at t={t:.1f}s: {reason}")
        if check is not None:
            check.report(KnownMap.load(base.map_name))
        inside, hist = _hist(rows, args.window)
        top = ", ".join(f"{_BIN_NAMES.get(b, b)} x{n}" for b, n in hist.most_common(5))
        east = hist.get(_EAST, 0)
        print(f"    window: {len(inside)} decisions; bins {top}; east (bin 1) on {east}/{len(inside)}")
        if args.csv and label == ("on" if args.mask != "off" else "off"):
            loop.write_csv(args.csv)
            print(f"    telemetry -> {args.csv}")

    if "off" in runs and "on" in runs:
        off = {round(r["t"], 3): r for r in runs["off"]}
        on = {round(r["t"], 3): r for r in runs["on"]}
        common = sorted(set(off) & set(on))
        print(f"\n  aligned decisions: {len(common)} (off {len(off)}, on {len(on)})")
        lo, hi = args.window
        vetoed = changed = dead_on = 0
        print(f"  {'t':>6}  {'off':>4}  {'on':>4}  legal bins 1..16 (. legal, x dead)")
        for t in common:
            a, b = off[t], on[t]
            legal = b["move_legal"]
            was_dead = legal >= 0 and a["move_bin"] > 0 and not (legal >> a["move_bin"] & 1)
            on_dead = legal >= 0 and b["move_bin"] > 0 and not (legal >> b["move_bin"] & 1)
            vetoed += was_dead
            changed += a["move_bin"] != b["move_bin"]
            dead_on += on_dead
            if lo <= t <= hi:
                flag = "  VETOED" if was_dead else ""
                print(f"  {t:6.2f}  {_BIN_NAMES.get(a['move_bin'], a['move_bin']):>4}  "
                      f"{_BIN_NAMES.get(b['move_bin'], b['move_bin']):>4}  {_bits(legal)}{flag}")
        n_win = sum(1 for t in common if lo <= t <= hi)
        v_win = sum(1 for t in common if lo <= t <= hi and off[t]["move_bin"] > 0
                    and on[t]["move_legal"] >= 0 and not (on[t]["move_legal"] >> off[t]["move_bin"] & 1))
        print(f"\n  whole clip: the mask vetoed the unmasked bin on {vetoed}/{len(common)} decisions, "
              f"the bin changed on {changed}; masked run chose a dead bin on {dead_on} (must be 0)")
        print(f"  window {lo:.0f}-{hi:.0f}s: vetoed {v_win}/{n_win}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
