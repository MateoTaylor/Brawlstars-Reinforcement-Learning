"""Measure the camera clamp onsets from footage (OBS_PARITY_TASKS.md P1), on the replay harness.

The sim's camera follows the hero until the hero comes within `camera.clamp_onset` tiles of a map
edge (`configs/default.yaml`, west 12.1 / east 12.8 / north 8.9 / south 5.5, geometry defaults),
then stops; from there the hero drifts off the screen's hero anchor by the distance the camera
did not travel. Live, that drift is `TrackerResult.hero_offset` (OBS_PARITY_TASKS.md C7): the
player box's tiles minus the nominal anchor, which the loop records on every decision as
`hero_offset_x/y`. So replaying a clip through the real loop (`replay_clip.replay`, detectors on
the CPU) gives the offset at 4 Hz, and a run where it exceeds `--threshold` tiles on an axis for
at least `--min-seconds` is the hero hugging an edge: the run's plateau IS that side's onset, or
a lower bound on it when the hero could not reach the border itself.

Camera motion over the run comes from odometry (`odo_pos`, the loop's own camera track). A clamp
holds the camera still along the offset's axis, so a run is a **clamp** when at least three of its
decisions (0.75 s at 4 Hz) moved the camera less than `--static-step` tiles along that axis while
odometry was tracking; its plateau is the median offset over those static decisions. Anything else
is **moving**: a dash's camera lag (Mortis covers ~4 tiles in a fraction of a second and the camera
catches up over the next decisions), a walk with the tracker lagging, or a lost odometry. Only clamp
runs feed the per-side onset estimate. Runs never straddle an odometry segment reset: the camera
position is not comparable across one.

Sign convention: `hero_offset` is (screen-right, screen-down) in tiles. The hero left of the anchor
(x < -threshold) means the camera stopped at the WEST edge; right, EAST; above (y < -threshold),
NORTH; below, SOUTH.

Only emulator-HUD footage runs here (the deployed loop's HUD mask and calibration are BlueStacks'):
the 2026-09-23 OBS recordings, `brawl_vision/data/training_videos/9-10_new*.mp4`, the
`bluestacks-example*` fixtures. Usage:

    .venv/Scripts/python.exe scripts/probes/edge_probe.py "C:/Users/mateo/Videos/2026-09-23 20-*.mp4" --device cpu
    .venv/Scripts/python.exe scripts/probes/edge_probe.py brawl_vision/data/training_videos/9-10_new*.mp4 --out <report.md>

Writes nothing unless `--out` (appends markdown).
"""
import argparse
import dataclasses
import glob
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_clip import replay                                          # noqa: E402
from brawl_deployment.config import load_deployment_config              # noqa: E402
from brawl_deployment.loop import VisionStack                           # noqa: E402
from brawl_deployment.policy import DeployedPolicy                      # noqa: E402
from brawl_vision.config import load_vision_config                      # noqa: E402

_SIDES = {("x", -1): "west", ("x", 1): "east", ("y", -1): "north", ("y", 1): "south"}


def offset_runs(rows, *, threshold: float, min_seconds: float, static_step: float = 0.2) -> list[dict]:
    """Maximal runs of consecutive decisions with |offset| > threshold on one axis and sign."""
    runs = []
    cur = None
    for r in rows:
        if r["near_edge"] < 0:              # no offset on this decision (no player box yet)
            cur = None
            continue
        ox, oy = r["hero_offset"]
        keys = []
        if abs(ox) > threshold:
            keys.append(("x", int(np.sign(ox))))
        if abs(oy) > threshold:
            keys.append(("y", int(np.sign(oy))))
        key = keys[0] if keys else None
        if key is not None and cur is not None and cur["key"] == key and r["odo_seg"] == cur["seg"]:
            cur["rows"].append(r)
        elif key is not None:
            cur = dict(key=key, seg=r["odo_seg"], rows=[r])
            runs.append(cur)
        else:
            cur = None
    out = []
    for run in runs:
        rs = run["rows"]
        dur = rs[-1]["t"] - rs[0]["t"]
        if dur < min_seconds:
            continue
        axis = 0 if run["key"][0] == "x" else 1
        vals = np.array([r["hero_offset"][axis] for r in rs])
        cam = np.array([r["odo_pos"] for r in rs], dtype=np.float64)
        d_axis = np.abs(np.diff(cam[:, axis]))
        tracking = np.array([r.get("odo_status", "ok") in ("ok", "uncertain") for r in rs])
        static = np.concatenate([[False], d_axis < static_step]) & tracking
        n_static = int(static.sum())
        clamp = n_static >= 3
        plateau = float(np.median(np.abs(vals[static]))) if clamp else float(np.median(np.abs(vals)))
        path = float(np.hypot(*np.diff(cam, axis=0).T).sum()) if len(rs) > 1 else 0.0
        out.append(dict(side=_SIDES[run["key"]], t_start=rs[0]["t"], t_end=rs[-1]["t"], seconds=dur,
                        decisions=len(rs), n_static=n_static, verdict="clamp" if clamp else "moving",
                        plateau=plateau, peak=float(np.abs(vals).max()),
                        cam_axis_net=float(abs(cam[-1, axis] - cam[0, axis])),
                        cam_axis_path=float(d_axis.sum()), cam_path=path,
                        near_edge=sum(r["near_edge"] == 1 for r in rs)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clips", nargs="+", help="clip paths or globs (emulator-HUD footage)")
    ap.add_argument("--run", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--stop", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=1.0, help="tiles of offset that count")
    ap.add_argument("--min-seconds", type=float, default=0.5)
    ap.add_argument("--static-step", type=float, default=0.2,
                    help="camera motion per decision (tiles, along the offset axis) that still counts as still")
    ap.add_argument("--out", default=None, help="append the markdown report here")
    args = ap.parse_args(argv)

    clips = []
    for spec in args.clips:
        hits = sorted(glob.glob(spec))
        clips.extend(hits or [spec])

    overrides = {"safety": {"capture_stall_seconds": 1e9, "max_match_seconds": 1e9}}
    if args.run:
        overrides["run"] = {"dir": args.run}
    # mask on: the shipped configuration (policy.dead_bin_mask, design doc 6.16)
    dcfg = dataclasses.replace(load_deployment_config(overrides=overrides), policy_dead_bin_mask=True)
    policy = DeployedPolicy.from_run(ROOT / dcfg.run_dir, checkpoint=dcfg.run_checkpoint,
                                     device=args.device, deterministic=True)
    vcfg = None
    if args.device == "cpu":
        vcfg = load_vision_config(overrides={"detector": {"device": "cpu"},
                                             "projectile": {"device": "cpu"},
                                             "classifier": {"device": "cpu"}})
    vision = VisionStack.build(vision_cfg=vcfg)
    sim = policy.cfg
    onsets = dict(zip(("west", "east", "north", "south"), sim.camera_clamp_onset)) \
        if hasattr(sim, "camera_clamp_onset") else {}
    print(f"run {dcfg.run_dir}  device {args.device}  threshold {args.threshold} tiles  "
          f"min run {args.min_seconds} s  sim onsets {onsets}  edge flag {getattr(sim, 'camera_edge_flag_tiles', '?')}")

    lines = [f"| clip | side | t_start (s) | seconds | decisions | static | verdict | plateau (tiles) | "
             f"peak | camera along axis: net / path | camera path | near_edge set |",
             "|:--|:--|---:|---:|---:|---:|:--|---:|---:|---:|---:|---:|"]
    starts = []
    per_side = {s: [] for s in _SIDES.values()}
    summary = []
    for clip in clips:
        t0 = time.perf_counter()
        try:
            rows, stops, logs, fed, loop = replay(clip, dcfg, vision=vision, policy=policy,
                                                  start=args.start, stop=args.stop, quiet=True)
        except Exception as exc:                      # a clip the reader cannot open must not end the sweep
            print(f"  {Path(clip).name}: FAILED {type(exc).__name__}: {exc}", flush=True)
            summary.append(f"- `{Path(clip).name}`: failed, {type(exc).__name__}: {exc}")
            continue
        valid = [r for r in rows if r["near_edge"] >= 0]
        runs = offset_runs(rows, threshold=args.threshold, min_seconds=args.min_seconds,
                           static_step=args.static_step)
        match_ts = [t for t, m in logs if m.startswith("match started")]
        starts.append((Path(clip).name, match_ts))
        offs = np.array([r["hero_offset"] for r in valid]) if valid else np.zeros((0, 2))
        mid = (f"|offset| median x {np.median(np.abs(offs[:, 0])):.2f} y {np.median(np.abs(offs[:, 1])):.2f}, "
               f"p95 x {np.percentile(np.abs(offs[:, 0]), 95):.2f} y {np.percentile(np.abs(offs[:, 1]), 95):.2f}"
               if len(offs) else "no offsets")
        matches = len(match_ts)
        starts_txt = ", ".join(f"{t:.2f}" for t in match_ts) or "-"
        n_clamp = sum(r["verdict"] == "clamp" for r in runs)
        print(f"  {Path(clip).name}: {fed} ticks, {len(rows)} decisions ({len(valid)} with an offset), "
              f"{matches} match start(s) at {starts_txt} s, {len(stops)} would-stop(s); {mid}; "
              f"{len(runs)} run(s), {n_clamp} clamp [{time.perf_counter() - t0:.0f} s]", flush=True)
        summary.append(f"- `{Path(clip).name}`: {len(rows)} decisions, {len(valid)} with an offset, "
                       f"{matches} match start(s) at {starts_txt} s; {mid}; {len(runs)} run(s), {n_clamp} clamp")
        for r in runs:
            if r["verdict"] == "clamp":
                per_side[r["side"]].append(r["plateau"])
            lines.append(f"| {Path(clip).name} | {r['side']} | {r['t_start']:.1f} | {r['seconds']:.2f} | "
                         f"{r['decisions']} | {r['n_static']} | {r['verdict']} | {r['plateau']:.2f} | "
                         f"{r['peak']:.2f} | {r['cam_axis_net']:.2f} / {r['cam_axis_path']:.2f} | "
                         f"{r['cam_path']:.2f} | {r['near_edge']}/{r['decisions']} |")
            print(f"      {r['side']:<5} t={r['t_start']:6.1f}s {r['seconds']:5.2f}s {r['verdict']:<6} "
                  f"plateau {r['plateau']:.2f} peak {r['peak']:.2f} camera along axis {r['cam_axis_path']:.2f} "
                  f"path {r['cam_path']:.2f} static {r['n_static']}/{r['decisions']}")

    side_lines = []
    for side, vals in per_side.items():
        default = onsets.get(side)
        if vals:
            side_lines.append(f"- **{side}**: {len(vals)} clamp run(s), plateau median {np.median(vals):.2f} tiles "
                              f"(min {min(vals):.2f}, max {max(vals):.2f}); geometry default {default}")
        else:
            side_lines.append(f"- **{side}**: no clamp run reached it; keeps the geometry default {default}")
    side_lines.append("")
    side_lines.append("Match starts (the loop's gate, clip seconds): " + "; ".join(
        f"`{n}` {', '.join(f'{t:.2f}' for t in ts) if ts else 'none'}" for n, ts in starts))
    text = "\n".join(["### Clamp onsets from footage (edge_probe.py)", "", *summary, "", *lines, "",
                      *side_lines, ""])
    print()
    print(text)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"appended to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
