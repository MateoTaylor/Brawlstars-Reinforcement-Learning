"""Projectile boxes -> projectiles with a velocity. BRAWL_DEPLOYMENT_DESIGN.md 6.1.

`projectile_detection/detect.py` names the gap this fills, in its own words: *"projectiles are the
one thing in this game where a track across frames carries information the single frame does not
-- direction, speed, and therefore where it will be in 250 ms. This class deliberately does not
attempt that."* Two of the four fields `configs/agent_obs_deploy.yaml` asks for --
`projectiles.vel` and `projectiles.time_to_closest` -- are that information and nothing else. No
detector can supply either.

#### Why this is not `EntityTracker` with different constants

They share `_greedy_match` and nothing else, because almost every other decision differs:

  * **Rate.** Entities are tracked at the 4 Hz decision rate. Projectiles are tracked at the
    **perception rate** (12 Hz, §6.13; the measurements below were taken at 20)
    and read at 4 Hz, because at 4 Hz they are barely observable -- 93% of projectiles never
    survive to a second sample at that rate (BRAWL_DEPLOYMENT_DESIGN.md 9.5), and one sample is a
    position with no direction attached.
  * **No index stability.** The `entities` group promises slot 3 means the same enemy all match;
    the `projectiles` group explicitly does not, because `obs_select` re-sorts it by
    `time_to_closest` every tick. So there are no slots to protect, and therefore none of
    `EntityTracker`'s `promote_hits` / `n_slots` / hole-preserving machinery.
  * **Constant velocity is the truth here, not an approximation.** A brawler changes direction
    between samples, which is why `EntityTracker` blends each new finite difference into an EMA. A
    projectile flies straight at a fixed speed until it stops existing. That makes the best
    estimator the whole-span difference `(pos_now - pos_first) / (t_now - t_first)`, which divides
    the same anchor noise by a longer baseline every tick instead of decaying it. See `_advance`.
  * **A coasted track is a phantom threat.** An entity that goes undetected still exists, so
    `EntityTracker` coasts it for three ticks. A projectile that goes undetected has usually hit
    something, and reporting it would tell the policy to dodge a bullet that is no longer there.
    Coasting is kept for ASSOCIATION only and never reaches the observation -- see `live`.

#### The gate, measured rather than argued

Association is in world tiles, for `tracker.py`'s reason: the camera pans, so screen motion is not
object motion. `EntityTracker`'s `GATE_NOISE_TILES = 1.0` does not transfer -- it was sized around
a brawler box that encloses a nameplate, a health bar and an aura, and a 0.30 anchor fitted
against the occupancy map. None of that describes a projectile sprite.

Measured on `showdown_alternate_map.mp4` through the real pipeline (rectify -> odometry -> detect
-> `to_tiles`) at 20 Hz, using only frames with exactly one detection before and after so the
association is unambiguous rather than assumed -- 101 consecutive pairs:

    p50 0.55 tiles   p90 0.77   p95 0.84   p99 0.87   max 0.92
    implied speed:   p50 10.9 tiles/s ... max 18.4 tiles/s
    above 1.0 tiles: ZERO of 101

`MAX_PROJ_TILES_S = 20.0` sits just above that measured maximum. Two things worth knowing about
it. It is **faster than anything in the simulator** -- `configs/brawlers.yaml` tops out at
`super_proj_speed: 12.0` -- so real projectile speeds are outside the range the policy trained on
and `projectiles.vel` will carry values the sim never produced. And the measurement bundles motion
with every error term (projection, odometry drift, the anchor), so it is an upper bound on speed
rather than a speed.

#### The detector finds more than projectiles, and this keeps only those

The same model also boxes power-cube crates and dropped cubes (`projectile_detection/classes.py`).
Both are STATIONARY, and a stationary thing in this group causes a specific, bad failure:
`closest_approach` gives a zero-velocity point t = 0, and `obs_select` sorts ascending and keeps
twelve. So every crate on screen would take a slot ahead of the real incoming shots and could evict
all of them. It would also be wrong before it got that far. `_dedupe` keeps the more confident of
two boxes that project close together, so a crate could "merge away" a projectile flying over it.

So `update` filters on the label FIRST, before projection or dedupe, and counts what it dropped
in `ProjectileResult.n_other`. It filters here rather than in the loop because the tracker is the
thing that knows what a projectile track is. `require_projectile_class` is the other half. A model
exported without a `Projectile` class would make this filter drop every box, and the policy would
see no projectiles at all. That fails at startup instead of silently.

#### A projectile that never moves

Many tracks sit still. Replaying 27 clips through the shipped detector and this tracker at 12 Hz
(18,926 ticks), 175 of the 611 projectile tracks that lived long enough to judge (0.15 s) end with
a whole-span speed under 2 tiles/s. Some are real: the labels box impact flashes and puddle-like
ground effects. Some are false positives that every model so far has fired on, such as a dark
crater patch on one map. The sim has stationary entries too: `bot_sniper`'s lingering hazard sits
at `vel = 0` for `on_hit_area_ticks * on_hit_area_interval` = 4.0 s, and bot_sniper is 22% of the
enemy roster. So a still track is not wrong in itself. Two things about it are, and each gets one
rule.

**Its velocity is noise, so it is reported as zero** (`STATIC_TILES_S`, applied in `snapshot`
only). The whole-span fit divides the anchor noise by the span, so a still object reports a small
velocity in a random direction. `closest_approach` then gives it `time_to_closest` 0 when the
noise points away from the hero, and dist/speed -- often tens of seconds, which no sim entry ever
reports -- when it points toward. The sim's still entry is `vel = 0`, `time_to_closest = 0`, and
that is what these become. The threshold is measured, not derived, because nothing in the sim
bounds speed away from zero: a timed lob thrown at a close target crawls. Power-cube crates,
tracked by this class as a known-static control on the same replay, give the noise:

    span        2 ticks   0.2-0.35 s   0.35-0.75 s   0.75-1.5 s     whole-span speed, tiles/s
    p90          1.60        1.44          1.19          0.67
    p99          2.96        2.51          1.78          1.24

2.0 zeroes 95% of still tracks on their second velocity and ~99% after a third of a second. Every
constant-speed mover in the sim is faster: the slowest is `bot_artillery`'s split shard at
`split_distance / split_seconds` = 2.4. What falls under it is a lob landing within 2.5 tiles,
and it loses only its velocity: its position still updates every tick. Association keeps the real
estimate (`Projectile.vel` is untouched), because zeroing it there would only cost a slow mover
its gate.

The same replay, before and after both rules: `time_to_closest` over 2 s went from 817 of 3,861
reported slot-ticks (worst 761 s) to 75 (worst 7.4 s). 45% of what remains reports `vel = 0`.

**Nothing in the sim's projectile group lives longer than 5.5 s, so a track that does is retired**
(`MAX_TRACK_S`, applied in `live`). The longest-lived entry is bot_sniper's. Its rocket flies for
up to `attack_range / proj_speed` = 1.5 s, and the hazard it becomes then inherits its slot for
4.0 s more. So the bound is the whole 5.5 s, not the hazard's 4.0 s of stillness. The age is
measured from the track's birth, and this tracker links the flight to the hazard when it sees
both: 13 replay tracks were fast first and still after, one of them alive 4.6 s. Retiring at
4.0 s would have cut real hazards short by up to the flight time.

A retired track stays in `tracks`, matching and absorbing its own detections, so it cannot respawn
as a fresh track on the next tick. It just stops reaching the observation and the grid. Two tracks
crossed 5.5 s in the replay, at 5.6 s and 10.7 s. No moving track lived past 2.1 s, so in practice
only still tracks ever reach the limit.

What these rules do not catch:

- A false positive that flickers out for longer than `MAX_COAST_S` comes back as a new track with
  a fresh clock. 3 still sites spanned more than 5.5 s in the replay, and 2 of them were split
  across 2 and 4 tracks. Remembering retired sites would catch them, at the cost of a second map
  to keep coherent through resets. What it would buy is small. The feared failure -- still tracks
  taking all twelve slots, since `obs_select` sorts `time_to_closest` ascending -- did not happen
  once: live tracks peaked at 8 on any tick, and at most 4 of them were still.
- A shot that flies and then stops keeps a whole-span velocity that decays as `distance / age`.
  So it reads as slowly moving, not as still, until that drops under `STATIC_TILES_S`: after 4 s
  for an 8-tile flight. `_advance`'s estimator assumes one velocity per track, and a stop breaks
  that. It is 13 tracks of 611, so it is noted rather than fixed.

#### What is deliberately NOT here

No kind, no owner, no damage, no radius. `agent_obs_deploy.yaml` asks for four fields and calls a
projectile *"a moving dot and nothing more"*; the detector says only "Projectile" and could not
supply more if the spec wanted it. The model's other two classes, crates and dropped cubes, are
`loot.py`'s: the loop hands both consumers the same list and each keeps its own label.
"""
import math
from dataclasses import dataclass

from brawl_vision.object_detection.project import to_tiles
from brawl_vision.object_detection.projectile_detection.classes import PROJECTILE

from .tracker import _greedy_match

__all__ = ["Projectile", "ProjectileResult", "ProjectileTracker", "require_projectile_class",
           "MAX_PROJ_TILES_S", "GATE_NOISE_TILES", "PROJ_ANCHOR_FRAC",
           "MAX_COAST_S", "MIN_SAMPLES", "DUPLICATE_TILES", "STATIC_TILES_S", "MAX_TRACK_S"]

# Measured maximum step at 20 Hz was 18.4 tiles/s over 101 unambiguous pairs; see the module
# docstring. Above the simulator's fastest projectile (12.0 tiles/s), which is a fact about the
# game rather than a slack allowance.
MAX_PROJ_TILES_S = 20.0

# Everything in the step that is not motion: the ground-plane projection of an airborne sprite,
# odometry drift, and the anchor. Smaller than `tracker.GATE_NOISE_TILES` (1.0) because a
# projectile box is the sprite itself -- no nameplate or aura inflating it.
GATE_NOISE_TILES = 0.4

# The CENTRE of the box, not `to_tiles`'s 0.30 default. That default is "30% of the box height up
# from the bottom", measured to find a brawler's FEET inside a box that also encloses their
# nameplate and aura. A projectile box encloses a projectile, so neither the reasoning nor the
# number carries over.
#
# **This is a choice, not a calibration, and the residual is a real error.** A projectile flies
# above the ground and the homography maps the ground plane, so projecting any point of the sprite
# places it further from the camera than it is. The centre is a defensible guess at the sprite's
# midpoint and nothing more. There is a cheap accuracy test available if this ever matters: a
# projectile travels in a STRAIGHT line in world space, so a wrong assumed height makes the
# projected track CURVE as the projectile moves toward or away from the camera, and curvature is
# measurable with no hand labels at all. That is the projectile equivalent of `Detection.anchor`'s
# "a brawler cannot stand in a wall" test, and it has not been run.
PROJ_ANCHOR_FRAC = 0.5

# A track older than this is dropped rather than coasted further. Coasting exists so that one
# missed detection does not split a projectile into two tracks and cost the velocity; it is not a
# belief that the projectile is still there, and `live` will not report a coasted track.
#
# The number to hold is ONE missed tick tolerated and two not, so it belongs strictly between one
# and two tick periods. 0.1 was that at 20 Hz (0.05 / 0.10) with the top end exactly on the bound;
# at the shipped 12 Hz (§6.13) one miss is 0.083 s, which 0.1 admits by only 17 ms -- close enough
# that a single long tick would drop a track that is merely blinking. 0.15 sits mid-band at 12 Hz
# (0.083 in, 0.167 out) and is still under two ticks at 16 Hz. Re-check it if the rate moves again.
MAX_COAST_S = 0.15

# Samples needed before a track reaches the observation. Two, because that is the minimum for a
# velocity -- and reporting a one-sample projectile with `vel = (0, 0)` would be worse than
# dropping it, not merely less useful. `obs_select` sorts this group by `time_to_closest`
# ASCENDING and keeps the nearest 12; `closest_approach` returns t = 0 for a stationary point. So
# every unknown-direction projectile would sort AHEAD of the genuinely incoming ones and push them
# out of the slots entirely.
MIN_SAMPLES = 2

# Two projected points closer than this are treated as one projectile detected twice. In TILES,
# after projection, and measured -- see `_dedupe` for why distance beat the more obvious IoU test.
DUPLICATE_TILES = 0.3

# A track whose whole-span speed is under this is STILL: the observation gets `vel = (0, 0)`, the
# sim's own value for a hazard. Measured on power-cube crates, not derived from the sim; see
# "A projectile that never moves" in the module docstring for the table.
STATIC_TILES_S = 2.0

# A track older than this is retired from the observation. It is the sim's number, not a tuning:
# the longest any entry holds a projectile slot, which is `bot_sniper`'s rocket (8.0 / 5.33 = 1.5 s)
# plus the hazard that inherits its slot (2 x 2.0 s), from configs/brawlers.yaml. A test ties the
# two together in both directions, because each direction is a different bug. Set it longer and
# phantoms outlive anything the policy was trained on. Set it shorter and real hazards vanish
# before they end.
MAX_TRACK_S = 5.5


def require_projectile_class(names) -> None:
    """Raise unless the detector's classes include `Projectile`.

    `names` is `ObjectDetector.names` (id -> label, read from the ONNX metadata) or any iterable of
    labels. It is checked once, when the stack is built. Without this check, a model exported
    without a `Projectile` class would load cleanly and run at full speed, and the tracker's label
    filter would then drop every box. The resulting agent plays as if nobody ever shoots at it,
    and nothing in the telemetry says why.
    """
    labels = set(names.values()) if isinstance(names, dict) else set(names)
    if PROJECTILE not in labels:
        raise ValueError(
            f"the projectile model has no {PROJECTILE!r} class (it has {sorted(labels)}), so the "
            f"projectile tracker would see nothing. Check `projectile.model` in "
            f"configs/vision.yaml, and the label spelling in CVAT against "
            f"brawl_vision/object_detection/projectile_detection/classes.py")


def _dedupe(points, threshold: float):
    """Drop points closer than `threshold` tiles to an already-kept, more confident point.

    `points` are `((tile_x, tile_y), confidence)` in the world frame, so this runs AFTER projection.

    **This should not be necessary, and it is a floor rather than a fix.**
    `detector._decode_end2end` runs no NMS on this path, deliberately, because YOLO26 *"trains its
    head to emit one box per object"* and NMS *"would only merge genuinely distinct objects that
    happen to overlap, which for projectiles (a volley of three shots along one line) is exactly
    the wrong outcome"*. Both halves are borne out by the measurement below -- the head does emit
    one box per object most of the time (17 duplicate pairs in 269), and distinct projectiles
    really do arrive with overlapping boxes, so ordinary NMS really would eat volleys.

    It is here anyway because the failure is bad out of proportion to how often it happens: two
    boxes on one projectile become two tracks, the association splits between them, each gets a
    velocity from a half-length baseline, and the observation reports two bullets where there is
    one. One tick of that is a wrong dodge.

    **Distance rather than IoU, because the measurement says distance separates the two cases and
    IoU does not.** 269 simultaneous pairs across six clips, bucketed by box IoU and scored on the
    tile separation the tracker actually associates in:

        IoU band              n    min     p50     max      (tile separation)
        > 0.9  duplicates    17    0.004   0.020   0.061
        0.7-0.9               5    0.057   0.077   0.238
        0.1-0.7              18    0.172   1.323   1.425
        <= 0.1 distinct     229    0.910   3.894  21.179

    Read down the min/max columns: every unmistakable duplicate is inside 0.061 tiles and every
    unmistakably distinct pair is beyond 0.910 -- a 15x gap with nothing in it. Read across the IoU
    bands instead and there is no such gap; the 0.1-0.7 band alone spans 0.17 to 1.43 tiles,
    because these boxes are large (median 114 px) and perspective turns a few pixels near the top
    of the frame into more than a tile. `DUPLICATE_TILES = 0.3` sits 5x above the widest duplicate
    and 3x below the closest distinct pair.

    The residual risk is the same asymmetry either way: merging two real projectiles hides an
    incoming shot, while missing a duplicate costs one phantom. That is why the threshold sits
    nearer the duplicates than the midpoint of the gap.
    """
    keep = []
    for p, conf in sorted(points, key=lambda pc: -pc[1]):
        if any(math.hypot(p[0] - q[0], p[1] - q[1]) < threshold for q, _ in keep):
            continue
        keep.append((p, conf))
    return keep


@dataclass
class Projectile:
    """One projectile's identity across frames. Positions are WORLD tiles in the current odometry
    segment -- comparable across frames within it, meaningless across a segment change."""
    id: int
    pos: tuple[float, float]
    vel: tuple[float, float] = (0.0, 0.0)
    hits: int = 1
    t: float = 0.0
    confidence: float = 0.0
    pos0: tuple[float, float] = (0.0, 0.0)     # first observation, for the whole-span velocity
    t0: float = 0.0

    def predict(self, t: float) -> tuple[float, float]:
        dt = max(0.0, t - self.t)
        return (self.pos[0] + self.vel[0] * dt, self.pos[1] + self.vel[1] * dt)

    @property
    def speed(self) -> float:
        return math.hypot(*self.vel)


@dataclass(frozen=True)
class ProjectileResult:
    """One 20 Hz tick's outcome. `live` is every track carrying a velocity and seen THIS tick --
    the set the observation will be built from, exposed so logging need not recompute it."""
    live: list
    status: str                # "ok" | "reset" | odometry's own status when unusable
    segment: int
    n_detections: int          # PROJECTILE boxes only; the other classes are in `n_other`
    n_merged: int              # boxes dropped as duplicates of a more confident one this tick
    n_other: int = 0           # boxes of any other class (power cubes), dropped before tracking


class ProjectileTracker:
    """Feed it one perception tick's detections; read tracked projectiles. One per match."""

    def __init__(self, min_samples: int = MIN_SAMPLES, max_coast_s: float = MAX_COAST_S,
                 anchor_frac: float = PROJ_ANCHOR_FRAC,
                 duplicate_tiles: float = DUPLICATE_TILES,
                 static_tiles_s: float = STATIC_TILES_S, max_track_s: float = MAX_TRACK_S):
        self.min_samples = min_samples
        self.max_coast_s = max_coast_s
        self.anchor_frac = anchor_frac
        self.duplicate_tiles = duplicate_tiles
        self.static_tiles_s = static_tiles_s
        self.max_track_s = max_track_s
        self.tracks: list = []
        self._next_id = 0
        self._t = 0.0
        self._fresh = False        # was the most recent update usable?
        # None rather than -1: the first tick ADOPTS odometry's segment instead of reading it as a
        # change, the same reasoning as `EntityTracker._segment`.
        self._segment: int | None = None

    # -- the tick ------------------------------------------------------------

    def update(self, detections, plan, odometry, t: float) -> ProjectileResult:
        """`detections` in viewport pixels, `plan` the `RectifyPlan` they came from, `odometry`
        that frame's `OdometryResult`, `t` its timestamp in seconds.

        Gated on odometry exactly as `EntityTracker.update` and `occupancy.update` are, and for the
        same reason: positions live in odometry's world frame, so a tick admitted on a bad estimate
        moves every track by the odometry error and reads as every projectile changing course at
        once. A refused tick does NOT count against a track -- it is not evidence of absence -- but
        it does leave the tracks stale, so nothing is reported until a good tick arrives.

        Only `Projectile` boxes are tracked; see the module docstring for why a power cube must not
        get this far. The split happens before every early return, so `n_detections` means the
        same thing on every path.
        """
        everything = list(detections)
        detections = [d for d in everything if d.label == PROJECTILE]
        n_other = len(everything) - len(detections)

        self._t = t
        if self._segment is None:
            self._segment = odometry.segment
        elif odometry.segment != self._segment:
            self.reset(odometry.segment)
            return ProjectileResult([], "reset", odometry.segment, len(detections), 0, n_other)
        if odometry.status != "ok":
            self._fresh = False
            self._expire(t)
            return ProjectileResult([], odometry.status, self._segment, len(detections), 0,
                                    n_other)

        px, py = odometry.position_tiles
        placed = to_tiles(detections, plan, anchor_frac=self.anchor_frac)
        raw = [((tx + px, ty + py), d.confidence) for d, (tx, ty) in placed]
        points = _dedupe(raw, self.duplicate_tiles)

        gate = self.gate_for(t)
        matched, fresh = _greedy_match([tr.predict(t) for tr in self.tracks],
                                       [p for p, _ in points], gate)
        for ti, pi in matched.items():
            self._advance(self.tracks[ti], points[pi][0], points[pi][1], t)
        for pi in fresh:
            p, conf = points[pi]
            self.tracks.append(Projectile(id=self._take_id(), pos=p, t=t, confidence=conf,
                                          pos0=p, t0=t))
        self._fresh = True
        self._expire(t)
        return ProjectileResult(self.live(), "ok", self._segment, len(detections),
                                len(raw) - len(points), n_other)

    # -- track bookkeeping ---------------------------------------------------

    def _advance(self, tr: Projectile, pos, conf: float, t: float) -> None:
        """Whole-span velocity: not a finite difference, and not an EMA over them.

        A projectile's path is a straight line at constant speed, so every sample since the first
        is evidence about ONE velocity rather than about a velocity that has since changed. Fitting
        the whole span divides the anchor noise by the full elapsed time, which grows with every
        tick. The last-two-samples difference divides it by one tick forever, and an EMA over those
        differences trades that noise for lag. By the third sample the span estimator is already
        twice as precise as either, at no cost, and it has no tuning constant to get wrong.

        What it gives up: it inherits the first observation's error permanently, and it cannot
        follow a projectile that changes direction. Neither costs anything here -- the field this
        feeds extrapolates 250 ms, and nothing in the deployed roster curves.
        """
        span = t - tr.t0
        if span > 1e-6:
            tr.vel = ((pos[0] - tr.pos0[0]) / span, (pos[1] - tr.pos0[1]) / span)
        tr.pos = pos
        tr.t = t
        tr.hits += 1
        tr.confidence = conf

    def _expire(self, t: float) -> None:
        self.tracks = [tr for tr in self.tracks if t - tr.t <= self.max_coast_s]

    def _take_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    # -- interface -----------------------------------------------------------

    def gate_for(self, t: float) -> float:
        """Association radius in tiles, for the largest gap any live track is spanning.

        Scales with ACTUAL elapsed time rather than a nominal 0.05 s, for `EntityTracker.gate_for`'s
        reason: a dropped frame or a slow tick widens the true step, and a gate sized for the
        nominal rate would then reject every correct match at once -- losing tracks exactly when
        the loop is already struggling.
        """
        dt = max((t - tr.t for tr in self.tracks), default=0.0)
        return GATE_NOISE_TILES + MAX_PROJ_TILES_S * max(dt, 0.0)

    def live(self) -> list:
        """Tracks that carry a velocity AND were detected on the most recent usable tick.

        The second half is what keeps a coasted track out of the observation. Coasting buys
        association continuity across one missed detection; it is not a claim that the projectile
        still exists, and the overwhelmingly common reason one stops being detected is that it hit
        something. Reporting it would be telling the policy to dodge a bullet that is gone.

        A track older than `max_track_s` is left out too, though it stays in `tracks`: see "A
        projectile that never moves" in the module docstring.
        """
        if not self._fresh:
            return []
        return [tr for tr in self.tracks
                if tr.hits >= self.min_samples and tr.t == self._t
                and tr.t - tr.t0 <= self.max_track_s]

    def snapshot(self, hero_pos) -> list:
        """`[(rel_pos, vel, time_to_closest)]` for the observation's `projectiles` group.

        `time_to_closest` comes from `brawl_sim.core.geometry.closest_approach`, imported rather
        than reimplemented, because it has to mean exactly what it meant in training -- including
        two conventions a five-line rewrite would plausibly get wrong. The time is clamped to be
        non-negative, so a projectile already past its closest point reports 0 rather than a
        negative time. And a zero-velocity projectile reports 0 rather than infinity.

        That last convention is exercised on purpose, but only for tracks MEASURED still. A track
        slower than `static_tiles_s` reports `vel = (0, 0)` and so `time_to_closest = 0`, which is
        exactly what the sim reports for a hazard. A track with too few samples to have a velocity
        at all is a different thing, and `MIN_SAMPLES` keeps it out entirely. The stored
        `Projectile.vel` is left alone, because association still needs the real estimate.

        Ordering is NOT this function's job. `obs_select` re-sorts the group by `time_to_closest`
        and keeps the nearest `max_slots`; it has to be the one to do it, so that deployment and
        training select identically.
        """
        live = self.live()
        if not live:
            return []
        import torch                                        # noqa: PLC0415 -- one match, one call
        from brawl_sim.core.geometry import closest_approach

        p = torch.tensor([tr.pos for tr in live], dtype=torch.float32)
        v = torch.tensor([(0.0, 0.0) if tr.speed < self.static_tiles_s else tr.vel
                          for tr in live], dtype=torch.float32)
        q = torch.tensor(hero_pos, dtype=torch.float32).expand_as(p)
        ttc, _ = closest_approach(p, v, q)
        rel = p - q
        return [((float(rel[i, 0]), float(rel[i, 1])),
                 (float(v[i, 0]), float(v[i, 1])),
                 float(ttc[i])) for i in range(len(live))]

    def reset(self, segment: int) -> None:
        """A new odometry segment is a new world frame with no defined offset to the old one, so
        every stored position is now meaningless. Unlike `EntityTracker.reset`, there is nothing
        here worth trying to carry across: a projectile's entire lifetime is shorter than the
        discontinuity that caused the reset."""
        self.tracks.clear()
        self._fresh = False
        self._segment = segment
