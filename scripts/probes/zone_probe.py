"""Time the poison-gas schedule from footage (OBS_PARITY_TASKS.md Z1). CPU only, no detectors.

Three numbers with provenance: `T0` = seconds from the match gate to the first gas on screen,
`P` = seconds between successive gas advances, `D` = cells the front moves per advance. Each is
backed by the events behind it, and the per-clip tables are what `OBS_PARITY_PLAN.md` section 4
records.

Per frame, the deploy loop's own zone path (`scripts/vision_watch.py`'s per-frame pattern):
`rect = plan.rectify(frame)`, `odo = odometry.update(rect)`, `zone = detect_zone(rect, plan, cfg)`,
deposited into the real `GasMap` (`brawl_deployment/perception/grid.py`) so the probe sits on the
loop's gating and rounding. The bookkeeping is outside it: a cell that turns gassed within
`--sharp-s` seconds of last being seen clear is a SHARP advance (the front moved while the camera
watched); a cell first seen already gassed is a REVEAL (the camera arrived after the front) and
says nothing about timing. Sharp events split into bursts wherever they pause for more than
`--burst-gap`; the gaps between burst starts are `P`, the thin extent of a burst is `D`.

Clock origin: `t_gate = t(first gameplay frame) + 0.5` (the gate's `enter_samples` at 12 Hz), the
first gameplay frame from `gameplay.scan_gameplay` at 10 fps -- or `--t0-frame` when given.
Time is always `frame.t`, never `index / fps` (the clips drop frames).

Usage:

    .venv/Scripts/python.exe scripts/probes/zone_probe.py tests/fixtures/vision/zone_grows_from_east.mp4 --hud phone
    .venv/Scripts/python.exe scripts/probes/zone_probe.py tests/fixtures/vision/bluestacks-example-zone.mp4 --hud emulator
    .venv/Scripts/python.exe scripts/probes/zone_probe.py "C:/Users/mateo/Videos/2026-09-23 20-20-47.mp4" \
        --hud emulator --start 400 --stop 1700 --out OBS_PARITY_PLAN_zone_measurements.md

`--start/--stop` are file-relative frame indices, inclusive. `--out` APPENDS the report as
markdown. Writes nothing else (a `.bounds.json` beside a recording that has none is
`clips.load_bounds`' doing, and fine outside the repo).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brawl_deployment.perception.grid import GasMap                     # noqa: E402
from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask   # noqa: E402
from brawl_vision.config import HUD_MASKS, load_vision_config           # noqa: E402
from brawl_vision.gameplay import scan_gameplay, span_frames            # noqa: E402
from brawl_vision.sources import at_viewport, open_source               # noqa: E402
from brawl_vision.terrain.labeling import default_hud                   # noqa: E402
from brawl_vision.terrain.odometry import Odometry                      # noqa: E402
from brawl_vision.terrain.zone import detect_zone                       # noqa: E402

GATE_OFFSET_S = 0.5      # `match_gate.enter_samples` (6) at `loop.tick_hz` 12


def _placement(gas: GasMap, plan, odo):
    """`GasMap.update`'s window: `(dst, sub)` slice pairs, or None when nothing lands."""
    cols, rows = plan.size_tiles
    px, py = odo.position_tiles
    ox, oy = gas.origin
    col0 = int(round(plan.origin_tile[0] + px)) - ox
    row0 = int(round(plan.origin_tile[1] + py)) - oy
    r_lo, r_hi = max(row0, 0), min(row0 + rows, gas.height)
    c_lo, c_hi = max(col0, 0), min(col0 + cols, gas.width)
    if r_hi <= r_lo or c_hi <= c_lo:
        return None
    return ((slice(r_lo, r_hi), slice(c_lo, c_hi)),
            (slice(r_lo - row0, r_hi - row0), slice(c_lo - col0, c_hi - col0)))


_SIDE_CODES = "EWNS"


def _row_fronts(gas: GasMap, observed: np.ndarray, last_clear: np.ndarray) -> dict:
    """{side: {line: (front, t_clear)}} for the lines (rows for E/W, columns for N/S) that are
    observed in this frame and hold gas assigned to that side. `front` is the innermost gassed
    cell on the line (world cells): min col for E, max col for W, max row for N, min row for S;
    `t_clear` is when that cell was last seen clear (NaN: never). Sides are assigned around the
    seen area's centre. Per-line bookkeeping is what makes the measure robust to the camera: a
    pan that reveals a deep gassed region starts new lines at their own first front instead of
    moving an aggregate, and `t_clear` lets the caller reject a front that moved onto cells
    nobody saw clear just before (a reveal, not an advance)."""
    rows, cols = np.nonzero(gas.gassed)
    if len(rows) == 0:
        return {}
    srows, scols = np.nonzero(gas.seen)
    cr, cc = float(srows.mean()), float(scols.mean())
    dr, dc = rows - cr, cols - cc
    vertical = np.abs(dr) >= np.abs(dc)
    side = np.where(vertical, np.where(dr > 0, 3, 2), np.where(dc > 0, 0, 1))
    live_rows = observed.any(axis=1)
    live_cols = observed.any(axis=0)
    out = {}
    for code, name in enumerate(_SIDE_CODES):
        m = side == code
        if not m.any():
            continue
        per = {}
        if name in "EW":
            for r, c in zip(rows[m], cols[m]):
                if live_rows[r]:
                    per[int(r)] = min(per.get(r, 10 ** 9), c) if name == "E" else max(per.get(r, -1), c)
        else:
            for r, c in zip(rows[m], cols[m]):
                if live_cols[c]:
                    per[int(c)] = min(per.get(c, 10 ** 9), r) if name == "S" else max(per.get(c, -1), r)
        if per:
            if name in "EW":
                out[name] = {k: (float(v), float(last_clear[k, int(v)])) for k, v in per.items()}
            else:
                out[name] = {k: (float(v), float(last_clear[int(v), k])) for k, v in per.items()}
    return out


def _front_stats(samples: list, t_gate, *, band_gap: float = 0.5, min_lines: int = 3,
                 min_span_s: float = 5.0, sharp_line_s: float = 1.0) -> dict:
    """Per side, from (t, {side: {line: (front, t_clear)}}) samples:
    - line steps: a line observed in two consecutive samples whose front moved inward by >= 1
      onto a cell seen clear within `sharp_line_s` (so the step is an advance the camera watched,
      not a reveal); clustered in time (gap <= band_gap) into BAND steps when >= min_lines
      distinct lines step together. P_band = gaps between band starts, D_band = median line
      step inside a band.
    - line rates: for every line seen over >= min_span_s, its sharp advance (sum of its counted
      steps) / its span, and the raw advance (last front - first front, reveals included) / span;
      the medians over lines bracket the advance rate whether the gas moves in bands or not."""
    out = {}
    for name in _SIDE_CODES:
        sign = -1.0 if name in "ES" else 1.0          # E/S fronts shrink inward, W/N grow
        prev = None
        events = []                                    # (t, line, size), sharp only
        raw_events = 0
        sharp_adv = {}                                 # line -> summed sharp step size
        first = {}                                     # line -> (t, front)
        last = {}
        for t, fr in samples:
            cur = fr.get(name)
            if cur is None:
                prev = None
                continue
            for line, (f, t_clear) in cur.items():
                if line not in first:
                    first[line] = (t, f)
                last[line] = (t, f)
                if prev is not None and line in prev:
                    size = sign * (f - prev[line][0])
                    if size >= 1.0:
                        raw_events += 1
                        if np.isfinite(t_clear) and t - t_clear <= sharp_line_s:
                            events.append((t, line, size))
                            sharp_adv[line] = sharp_adv.get(line, 0.0) + size
            prev = cur
        if not first:
            continue
        bands = []
        for e in sorted(events):
            if bands and e[0] - bands[-1][-1][0] <= band_gap:
                bands[-1].append(e)
            else:
                bands.append([e])
        band_steps = []
        for b in bands:
            lines = {e[1] for e in b}
            if len(lines) >= min_lines:
                band_steps.append(dict(t=b[0][0], t_end=b[-1][0], lines=len(lines),
                                       size=float(np.median([e[2] for e in b]))))
        periods = [b2["t"] - b1["t"] for b1, b2 in zip(band_steps, band_steps[1:])]
        rates = []
        raw_rates = []
        for line, (t0, f0) in first.items():
            t1, f1 = last[line]
            if t1 - t0 >= min_span_s:
                rates.append(sharp_adv.get(line, 0.0) / (t1 - t0))
                raw_rates.append(sign * (f1 - f0) / (t1 - t0))
        out[name] = dict(lines=len(first), line_events=len(events), raw_events=raw_events,
                         band_steps=band_steps, periods=periods, rates=rates, raw_rates=raw_rates,
                         t_first=min(t for t, _ in first.values()),
                         t_last=max(t for t, _ in last.values()), t_gate=t_gate)
    return out


def probe(clip, *, hud: str, start: int | None, stop: int | None, t0_frame: int | None,
          sharp_s: float, burst_gap: float, step: int = 1, front_dt: float = 0.25,
          verbose: bool = True) -> dict:
    cfg = load_vision_config()
    plan = build_rectify_plan(load_camera_model(), load_hud_mask(HUD_MASKS[hud]))
    odometry = Odometry(plan, cfg)
    gas = GasMap(cfg.occupancy_grid_h, cfg.occupancy_grid_w)
    last_clear = np.full((gas.height, gas.width), np.nan, np.float64)
    front_samples = []      # (t, {side: {line: front}}) every front_dt seconds, per odometry segment
    next_front_t = None

    # -- the clock origin -----------------------------------------------------------------
    if t0_frame is None:
        scan = scan_gameplay(clip, scan_fps=10.0)
        span = span_frames(scan)
        if not span:
            return {"clip": str(clip), "error": "no gameplay span; pass --t0-frame"}
        t0_frame = span[0]
        span_end = span[-1]
    else:
        span_end = None

    events = []            # (t, row, col, gap, sharp)
    per_sec = {}           # second after gate -> counters
    resets = 0
    first_gas = None
    t_first = None
    frames = 0
    t_wall = time.perf_counter()
    with open_source(clip, cfg) as source:
        for frame in at_viewport(source, plan.viewport):
            if t_first is None and frame.index >= t0_frame:
                t_first = frame.t
                # The gate: forget what the lobby and loading screens deposited (their purple
                # UI reads as gas), so first gas and every count below are gameplay only.
                gas = GasMap(cfg.occupancy_grid_h, cfg.occupancy_grid_w)
                last_clear[:] = np.nan
                front_samples.clear()
                next_front_t = None
            if (start is not None and frame.index < start) or (stop is not None and frame.index > stop):
                continue
            if step > 1 and frame.index % step:
                continue
            frames += 1
            rect = plan.rectify(frame.image)
            odo = odometry.update(rect)
            zone = detect_zone(rect, plan, cfg)

            gassed_before = gas.gassed.copy()
            seg_before = gas._segment
            n_new = gas.update(zone, plan, odo)
            was_reset = seg_before is not None and gas._segment != seg_before
            if was_reset:
                last_clear[:] = np.nan
                resets += 1

            fresh = gas.gassed & ~gassed_before
            n_sharp = 0
            if fresh.any():
                for r, c in zip(*np.nonzero(fresh)):
                    gap = frame.t - last_clear[r, c]          # NaN when never seen clear
                    sharp = bool(np.isfinite(gap) and gap <= sharp_s)
                    n_sharp += sharp
                    events.append((frame.t, int(r), int(c), float(gap), sharp))
            if first_gas is None and gas.gassed.any():
                first_gas = (frame.t, frame.index)

            # Cells seen clear this frame, placed as GasMap places its deposits.
            if odo.status == "ok" and not was_reset:
                place = _placement(gas, plan, odo)
                if place is not None:
                    dst, sub = place
                    clear = (zone.observed & ~zone.cells)[sub] & ~gas.gassed[dst]
                    view = last_clear[dst]
                    view[clear] = frame.t

            key = None if t_first is None else int(np.floor(frame.t - (t_first + GATE_OFFSET_S)))
            s = per_sec.setdefault(key, dict(frames=0, ok=0, fresh=0, sharp=0, resets=0,
                                              gassed=0, on_screen=0))
            s["frames"] += 1
            s["ok"] += odo.status == "ok"
            s["fresh"] += int(fresh.sum())
            s["sharp"] += n_sharp
            s["resets"] += was_reset
            s["gassed"] = int(gas.gassed.sum())
            s["on_screen"] = max(s["on_screen"], int(zone.gassed_cells))
            if was_reset:
                front_samples.clear()
            if next_front_t is None or frame.t >= next_front_t:
                next_front_t = frame.t + front_dt
                observed = np.zeros_like(gas.gassed)
                place = _placement(gas, plan, odo) if odo.status == "ok" else None
                if place is not None:
                    dst, sub = place
                    observed[dst] = zone.observed[sub]
                front_samples.append((frame.t, _row_fronts(gas, observed, last_clear)))
            if verbose and frames % 600 == 0:
                print(f"    frame {frame.index}  t={frame.t:6.1f}s  gassed {int(gas.gassed.sum())}  "
                      f"events {len(events)}  {time.perf_counter() - t_wall:4.0f} s", flush=True)

    t_gate = None if t_first is None else t_first + GATE_OFFSET_S
    # -- bursts ----------------------------------------------------------------------------
    sharp = sorted((e for e in events if e[4] and (t_first is None or e[0] >= t_first)),
                   key=lambda e: e[0])
    bursts = []
    for e in sharp:
        if bursts and e[0] - bursts[-1][-1][0] <= burst_gap:
            bursts[-1].append(e)
        else:
            bursts.append([e])
    seen_rows, seen_cols = np.nonzero(gas.seen)
    centre = (float(seen_rows.mean()), float(seen_cols.mean())) if len(seen_rows) else (64.0, 64.0)
    summary = []
    for b in bursts:
        ts = [e[0] for e in b]
        rows = [e[1] for e in b]
        cols = [e[2] for e in b]
        r_span = max(rows) - min(rows) + 1
        c_span = max(cols) - min(cols) + 1
        dr, dc = np.mean(rows) - centre[0], np.mean(cols) - centre[1]
        side = ("S" if dr > 0 else "N") if abs(dr) >= abs(dc) else ("E" if dc > 0 else "W")
        summary.append(dict(t_start=ts[0], t_end=ts[-1], cells=len(b), rows=r_span, cols=c_span,
                            depth=(c_span if side in "EW" else r_span),
                            depth2=len(b) / max(r_span, c_span),
                            side=side))
    periods = [summary[i + 1]["t_start"] - summary[i]["t_start"] for i in range(len(summary) - 1)]
    fronts = _front_stats(front_samples, t_gate)
    after_gate = [s for k, s in per_sec.items() if k is not None and k >= 0]
    ok_frac = (sum(s["ok"] for s in after_gate) / max(sum(s["frames"] for s in after_gate), 1))
    resets_after = sum(s["resets"] for s in after_gate)
    usable = len(summary) >= 3 and resets_after == 0 and ok_frac >= 0.95
    reasons = []
    if len(summary) < 3:
        reasons.append(f"only {len(summary)} sharp burst(s)")
    if resets_after:
        reasons.append(f"{resets_after} odometry reset(s) after the gate")
    if ok_frac < 0.95:
        reasons.append(f"odometry ok on {ok_frac:.0%} of frames after the gate")
    return dict(clip=str(clip), hud=hud, frames=frames, t_first_gameplay=t_first, t_gate=t_gate,
                t0_frame=t0_frame, span_end=span_end, first_gas=first_gas, events=len(events),
                sharp_events=len(sharp), reveals=len(events) - len(sharp), resets=resets,
                bursts=summary, periods=periods, per_sec=per_sec, usable=usable, fronts=fronts,
                front_dt=front_dt,
                reasons=reasons, sharp_s=sharp_s, burst_gap=burst_gap,
                seconds=time.perf_counter() - t_wall)


def report(res: dict) -> str:
    name = Path(res["clip"]).name
    if "error" in res:
        return f"### {name}\n\n{res['error']}\n"
    out = [f"### {name} ({res['hud']} HUD, {res['frames']} frames, {res['seconds']:.0f} s)", ""]
    tg = res["t_gate"]
    out.append(f"- gate: first gameplay frame {res['t0_frame']} at t_clip {res['t_first_gameplay']:.2f} s, "
               f"t_gate = {tg:.2f} s" if tg is not None else "- gate: no gameplay frame reached")
    if res["first_gas"]:
        t, idx = res["first_gas"]
        ag = f", t_after_gate {t - tg:.2f} s" if tg is not None else ""
        out.append(f"- **first gas**: frame {idx}, t_clip {t:.2f} s{ag}")
    else:
        out.append("- **first gas**: none seen")
    out.append(f"- events: {res['events']} fresh cells = {res['sharp_events']} sharp + "
               f"{res['reveals']} reveals; odometry resets {res['resets']} "
               f"(sharp <= {res['sharp_s']} s, burst gap {res['burst_gap']} s)")
    out.append("")
    out.append("| t_after_gate | frames | odometry ok | gassed cells (cum) | on screen (max) | fresh | sharp | resets |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in sorted(res["per_sec"], key=lambda v: (v is None, v)):
        s = res["per_sec"][k]
        if k is None or (k < 0 and s["fresh"] == 0):
            continue          # lobby seconds with nothing in them
        out.append(f"| {k:d} | {s['frames']} | {s['ok']} | {s['gassed']} | {s['on_screen']} | "
                   f"{s['fresh']} | {s['sharp']} | {s['resets']} |")
    pre = [s for k, s in res["per_sec"].items() if k is not None and k < 0]
    if pre:
        out.append(f"\n(before the gate: {sum(s['frames'] for s in pre)} frames, "
                   f"{sum(s['fresh'] for s in pre)} fresh cells)")
    out.append("")
    if res["bursts"]:
        out.append("| burst | t_start | t_end | after gate | cells | rows | cols | depth (thin span) | cells/long span | side |")
        out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|")
        for i, b in enumerate(res["bursts"], 1):
            ag = f"{b['t_start'] - tg:.2f}" if tg is not None else "-"
            out.append(f"| {i} | {b['t_start']:.2f} | {b['t_end']:.2f} | {ag} | {b['cells']} | {b['rows']} | "
                       f"{b['cols']} | {b['depth']} | {b['depth2']:.1f} | {b['side']} |")
        p = res["periods"]
        if p:
            q1, med, q3 = np.percentile(p, [25, 50, 75])
            out.append(f"\n- P (burst-start gaps): median {med:.2f} s, IQR {q1:.2f}-{q3:.2f} s, n = {len(p)}; "
                       f"raw {', '.join(f'{v:.2f}' for v in p)}")
        depths = [b["depth"] for b in res["bursts"]]
        d2 = [b["depth2"] for b in res["bursts"]]
        out.append(f"- D (thin span): median {np.median(depths):.1f} cells "
                   f"(cells/long span median {np.median(d2):.1f}), n = {len(depths)} bursts")
    else:
        out.append("- no sharp bursts")
    if res.get("fronts"):
        out.append("")
        out.append(f"Front per line (rows for E/W, columns for N/S; innermost gassed cell of each line "
                   f"observed in the frame, sampled every {res['front_dt']} s). A line step is the front "
                   f"moving inward >= 1 cell onto a cell seen clear within 1 s (an advance the camera "
                   f"watched, not a reveal); band step = >= 3 lines stepping within 0.5 s; sharp rate = a "
                   f"line's counted steps over its span, raw rate = its total front change over its span "
                   f"(reveals included), lines seen >= 5 s:")
        out.append("")
        out.append("| side | lines | seen (after gate) | line steps sharp / raw | band steps | "
                   "P_band median / IQR (s) | D_band median (cells) | sharp rate median / IQR (cells/s) | "
                   "raw rate median | n lines >= 5 s | sim rate |")
        out.append("|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, f in res["fronts"].items():
            per = f["periods"]
            if per:
                q1, med, q3 = np.percentile(per, [25, 50, 75])
                p_txt = f"{med:.2f} / {q1:.2f}-{q3:.2f} (n={len(per)})"
            else:
                p_txt = "-"
            sizes = [b["size"] for b in f["band_steps"]]
            d_txt = f"{np.median(sizes):.1f}" if sizes else "-"
            if f["rates"]:
                rq1, rmed, rq3 = np.percentile(f["rates"], [25, 50, 75])
                r_txt = f"{rmed:.3f} / {rq1:.3f}-{rq3:.3f}"
            else:
                r_txt = "-"
            fr = f["t_first"] - tg if tg is not None else f["t_first"]
            to = f["t_last"] - tg if tg is not None else f["t_last"]
            raw_txt = f"{np.median(f['raw_rates']):.3f}" if f["raw_rates"] else "-"
            out.append(f"| {name} | {f['lines']} | {fr:.1f}-{to:.1f} | {f['line_events']} / {f['raw_events']} | "
                       f"{len(f['band_steps'])} | {p_txt} | {d_txt} | {r_txt} | {raw_txt} | "
                       f"{len(f['rates'])} | 0.667 |")
        for name, f in res["fronts"].items():
            if f["band_steps"]:
                st = ", ".join(f"{b['t'] - (tg or 0.0):.2f}s +{b['size']:.1f} x{b['lines']}"
                               for b in f["band_steps"])
                out.append(f"- {name} band steps (after gate; +cells x lines): {st}")
    status = "yes" if res["usable"] else "no"
    why = f" ({'; '.join(res['reasons'])})" if res["reasons"] else f" ({len(res['bursts'])} bursts)"
    out.append(f"- **usable for P: {status}**{why}")
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip")
    ap.add_argument("--hud", choices=tuple(HUD_MASKS), default=None)
    ap.add_argument("--start", type=int, default=None, help="first file frame index (inclusive)")
    ap.add_argument("--stop", type=int, default=None, help="last file frame index (inclusive)")
    ap.add_argument("--t0-frame", type=int, default=None, help="first gameplay frame; skips the scan")
    ap.add_argument("--sharp-s", type=float, default=0.3)
    ap.add_argument("--burst-gap", type=float, default=0.5)
    ap.add_argument("--step", type=int, default=1, help="process every Nth frame (1 = all)")
    ap.add_argument("--front-dt", type=float, default=0.25, help="front sampling interval (s)")
    ap.add_argument("--out", default=None, help="append the markdown report here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    hud = args.hud or default_hud(args.clip)
    print(f"{Path(args.clip).name}: {hud} HUD, frames {args.start}..{args.stop}, sharp <= {args.sharp_s} s, "
          f"burst gap {args.burst_gap} s", flush=True)
    res = probe(args.clip, hud=hud, start=args.start, stop=args.stop, t0_frame=args.t0_frame,
                sharp_s=args.sharp_s, burst_gap=args.burst_gap, step=args.step,
                front_dt=args.front_dt, verbose=not args.quiet)
    text = report(res)
    print(text)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"appended to {args.out}")
    return 1 if "error" in res else 0


if __name__ == "__main__":
    sys.exit(main())
